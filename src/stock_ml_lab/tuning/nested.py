from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit

from stock_ml_lab.dataset import DatasetBundle
from stock_ml_lab.evaluation.importance import extract_tree_feature_importance
from stock_ml_lab.evaluation.metrics import RegressionMetrics, regression_metrics
from stock_ml_lab.evaluation.validation import CVResult, FoldResult, evaluate_time_series_cv
from stock_ml_lab.models.baseline import ZeroReturnRegressor
from stock_ml_lab.tuning.search_spaces import (
    DISPLAY_NAMES,
    TUNABLE_MODELS,
    build_estimator,
    sample_hyperparameters,
)


@dataclass(frozen=True)
class TuningFoldResult:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int
    best_inner_rmse: float
    best_params: dict[str, Any]
    metrics: RegressionMetrics


@dataclass(frozen=True)
class NestedCVResult:
    model_key: str
    model_name: str
    metrics: RegressionMetrics
    folds: tuple[TuningFoldResult, ...]
    predictions: pd.Series
    actuals: pd.Series
    n_splits: int
    inner_splits: int
    gap: int
    n_trials: int
    fold_feature_importances: pd.DataFrame | None = None

    def fold_frame(self) -> pd.DataFrame:
        rows = []
        for fold in self.folds:
            rows.append(
                {
                    "fold": fold.fold,
                    "train_start": fold.train_start,
                    "train_end": fold.train_end,
                    "test_start": fold.test_start,
                    "test_end": fold.test_end,
                    "n_train": fold.n_train,
                    "n_test": fold.n_test,
                    "best_inner_rmse": fold.best_inner_rmse,
                    **fold.metrics.to_dict(),
                }
            )
        return pd.DataFrame(rows).set_index("fold")

    def params_frame(self) -> pd.DataFrame:
        rows = [{"fold": fold.fold, **fold.best_params} for fold in self.folds]
        return pd.DataFrame(rows).set_index("fold")

    def feature_importance_frame(self) -> pd.DataFrame | None:
        if self.fold_feature_importances is None:
            return None
        frame = pd.DataFrame(
            {
                "mean_importance": self.fold_feature_importances.mean(axis=0),
                "std_importance": self.fold_feature_importances.std(axis=0, ddof=0),
            }
        )
        return frame.sort_values("mean_importance", ascending=False)


@dataclass(frozen=True)
class TuningComparisonResult:
    zero_baseline: CVResult
    tuned_results: tuple[NestedCVResult, ...]

    def summary_frame(self) -> pd.DataFrame:
        rows = [
            {
                "model": self.zero_baseline.model_name,
                **self.zero_baseline.metrics.to_dict(),
                "n_predictions": len(self.zero_baseline.predictions),
            }
        ]
        rows.extend(
            {
                "model": result.model_name,
                **result.metrics.to_dict(),
                "n_predictions": len(result.predictions),
            }
            for result in self.tuned_results
        )

        frame = pd.DataFrame(rows).set_index("model")
        zero_mae = float(frame.loc["Zero Return", "mae"])
        zero_rmse = float(frame.loc["Zero Return", "rmse"])

        frame["mae_improvement_vs_zero_pct"] = (
            (zero_mae - frame["mae"]) / zero_mae * 100.0
        )
        frame["rmse_improvement_vs_zero_pct"] = (
            (zero_rmse - frame["rmse"]) / zero_rmse * 100.0
        )
        return frame.sort_values("rmse")

    def assert_common_outer_protocol(self) -> None:
        reference_index = self.zero_baseline.predictions.index
        reference_boundaries = [
            (f.train_start, f.train_end, f.test_start, f.test_end)
            for f in self.zero_baseline.folds
        ]
        for result in self.tuned_results:
            if not result.predictions.index.equals(reference_index):
                raise RuntimeError(
                    f"{result.model_name} used different outer test timestamps."
                )
            boundaries = [
                (f.train_start, f.train_end, f.test_start, f.test_end)
                for f in result.folds
            ]
            if boundaries != reference_boundaries:
                raise RuntimeError(
                    f"{result.model_name} used different outer fold boundaries."
                )


def _validate_inputs(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    outer_splits: int,
    inner_splits: int,
    gap: int,
    n_trials: int,
) -> None:
    if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
        raise TypeError("X must be a DataFrame and y must be a Series.")
    if not X.index.equals(y.index):
        raise ValueError("X and y must have identical indices.")
    if not X.index.is_monotonic_increasing:
        raise ValueError("X and y must be ordered chronologically.")
    if outer_splits < 2 or inner_splits < 2:
        raise ValueError("outer_splits and inner_splits must be >= 2.")
    if gap < 1:
        raise ValueError("gap must be >= 1 for the next_return target.")
    if n_trials < 1:
        raise ValueError("n_trials must be >= 1.")


def _inner_rmse(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    estimator,
    inner_splits: int,
    gap: int,
) -> float:
    splitter = TimeSeriesSplit(n_splits=inner_splits, gap=gap)
    actual_parts: list[np.ndarray] = []
    prediction_parts: list[np.ndarray] = []

    for train_idx, valid_idx in splitter.split(X):
        model = clone(estimator)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = np.asarray(model.predict(X.iloc[valid_idx]), dtype=float).reshape(-1)
        actual_parts.append(y.iloc[valid_idx].to_numpy(dtype=float))
        prediction_parts.append(pred)

    actual = np.concatenate(actual_parts)
    prediction = np.concatenate(prediction_parts)
    return regression_metrics(actual, prediction).rmse


def run_nested_tuning(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    model_key: str,
    outer_splits: int = 5,
    inner_splits: int = 3,
    gap: int = 1,
    test_size: int | None = None,
    n_trials: int = 20,
    random_state: int = 42,
) -> NestedCVResult:
    """Nested expanding-window tuning with an untouched outer test fold.

    For each outer fold:
    1. Optuna searches only within the outer training set.
    2. Candidate parameters are scored by inner TimeSeriesSplit RMSE.
    3. The best candidate is refit on all outer-training observations.
    4. The outer test fold is evaluated exactly once.
    """
    if model_key not in TUNABLE_MODELS:
        raise ValueError(f"Model must be one of {TUNABLE_MODELS}; got {model_key!r}.")

    _validate_inputs(
        X,
        y,
        outer_splits=outer_splits,
        inner_splits=inner_splits,
        gap=gap,
        n_trials=n_trials,
    )

    outer = TimeSeriesSplit(
        n_splits=outer_splits,
        gap=gap,
        test_size=test_size,
    )

    fold_results: list[TuningFoldResult] = []
    prediction_parts: list[pd.Series] = []
    actual_parts: list[pd.Series] = []
    importance_parts: list[pd.Series] = []

    for fold_number, (train_idx, test_idx) in enumerate(outer.split(X), start=1):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        sampler = optuna.samplers.TPESampler(seed=random_state + fold_number)

        def objective(trial) -> float:
            params = sample_hyperparameters(trial, model_key)
            estimator = build_estimator(
                model_key,
                params,
                random_state=random_state + fold_number,
            )
            return _inner_rmse(
                X=X_train,
                y=y_train,
                estimator=estimator,
                inner_splits=inner_splits,
                gap=gap,
            )

        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=1,
            show_progress_bar=False,
        )

        best_params = dict(study.best_trial.params)
        final_model = build_estimator(
            model_key,
            best_params,
            random_state=random_state + fold_number,
        )
        final_model.fit(X_train, y_train)
        y_pred = np.asarray(final_model.predict(X_test), dtype=float).reshape(-1)
        metrics = regression_metrics(y_test.to_numpy(), y_pred)

        fold_results.append(
            TuningFoldResult(
                fold=fold_number,
                train_start=X_train.index[0],
                train_end=X_train.index[-1],
                test_start=X_test.index[0],
                test_end=X_test.index[-1],
                n_train=len(X_train),
                n_test=len(X_test),
                best_inner_rmse=float(study.best_value),
                best_params=best_params,
                metrics=metrics,
            )
        )
        prediction_parts.append(
            pd.Series(y_pred, index=X_test.index, name="prediction")
        )
        actual_parts.append(y_test.rename("actual"))

        importance = extract_tree_feature_importance(final_model, X.columns)
        if importance is not None:
            importance.name = fold_number
            importance_parts.append(importance)

    predictions = pd.concat(prediction_parts).sort_index()
    actuals = pd.concat(actual_parts).sort_index()

    if predictions.index.has_duplicates:
        raise RuntimeError("Nested CV produced duplicate outer-test timestamps.")

    fold_importances: pd.DataFrame | None = None
    if importance_parts:
        if len(importance_parts) != outer_splits:
            raise RuntimeError("Feature importance was unavailable for some outer folds.")
        fold_importances = pd.DataFrame(importance_parts)
        fold_importances.index.name = "fold"

    return NestedCVResult(
        model_key=model_key,
        model_name=DISPLAY_NAMES[model_key],
        metrics=regression_metrics(actuals.to_numpy(), predictions.to_numpy()),
        folds=tuple(fold_results),
        predictions=predictions,
        actuals=actuals,
        n_splits=outer_splits,
        inner_splits=inner_splits,
        gap=gap,
        n_trials=n_trials,
        fold_feature_importances=fold_importances,
    )


def run_tuning_comparison(
    dataset: DatasetBundle,
    *,
    models: tuple[str, ...] = TUNABLE_MODELS,
    outer_splits: int = 5,
    inner_splits: int = 3,
    gap: int = 1,
    test_size: int | None = None,
    n_trials: int = 20,
    random_state: int = 42,
) -> TuningComparisonResult:
    unknown = set(models).difference(TUNABLE_MODELS)
    if unknown:
        raise ValueError(f"Unknown tunable model keys: {sorted(unknown)}")
    if not models:
        raise ValueError("At least one tunable model must be selected.")

    zero = evaluate_time_series_cv(
        X=dataset.X,
        y=dataset.y,
        estimator=ZeroReturnRegressor(),
        model_name="Zero Return",
        n_splits=outer_splits,
        gap=gap,
        test_size=test_size,
    )

    tuned = tuple(
        run_nested_tuning(
            X=dataset.X,
            y=dataset.y,
            model_key=model_key,
            outer_splits=outer_splits,
            inner_splits=inner_splits,
            gap=gap,
            test_size=test_size,
            n_trials=n_trials,
            random_state=random_state,
        )
        for model_key in models
    )

    result = TuningComparisonResult(zero_baseline=zero, tuned_results=tuned)
    result.assert_common_outer_protocol()
    return result
