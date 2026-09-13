from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit

from stock_ml_lab.classification.dataset import DirectionDatasetBundle
from stock_ml_lab.classification.metrics import ClassificationMetrics, classification_metrics
from stock_ml_lab.classification.models import make_prior_classifier
from stock_ml_lab.classification.search_spaces import (
    CLASSIFICATION_DISPLAY_NAMES,
    CLASSIFICATION_MODELS,
    build_classification_estimator,
    sample_classification_hyperparameters,
)
from stock_ml_lab.evaluation.importance import extract_tree_feature_importance


@dataclass(frozen=True)
class ClassificationFoldResult:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int
    best_inner_log_loss: float
    best_params: dict[str, Any]
    metrics: ClassificationMetrics


@dataclass(frozen=True)
class NestedClassificationResult:
    model_key: str
    model_name: str
    metrics: ClassificationMetrics
    folds: tuple[ClassificationFoldResult, ...]
    probabilities: pd.Series
    actuals: pd.Series
    actual_returns: pd.Series
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
                    "best_inner_log_loss": fold.best_inner_log_loss,
                    **fold.metrics.to_dict(),
                }
            )
        return pd.DataFrame(rows).set_index("fold")

    def params_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"fold": f.fold, **f.best_params} for f in self.folds]
        ).set_index("fold")

    def feature_importance_frame(self) -> pd.DataFrame | None:
        if self.fold_feature_importances is None:
            return None
        return pd.DataFrame(
            {
                "mean_importance": self.fold_feature_importances.mean(axis=0),
                "std_importance": self.fold_feature_importances.std(axis=0, ddof=0),
            }
        ).sort_values("mean_importance", ascending=False)


@dataclass(frozen=True)
class PriorClassificationResult:
    model_name: str
    metrics: ClassificationMetrics
    probabilities: pd.Series
    actuals: pd.Series
    actual_returns: pd.Series
    folds: tuple[ClassificationFoldResult, ...]


@dataclass(frozen=True)
class ClassificationComparison:
    prior: PriorClassificationResult
    tuned_results: tuple[NestedClassificationResult, ...]

    def summary_frame(self) -> pd.DataFrame:
        rows = [
            {
                "model": self.prior.model_name,
                **self.prior.metrics.to_dict(),
                "n_predictions": len(self.prior.probabilities),
            }
        ]
        rows.extend(
            {
                "model": result.model_name,
                **result.metrics.to_dict(),
                "n_predictions": len(result.probabilities),
            }
            for result in self.tuned_results
        )
        frame = pd.DataFrame(rows).set_index("model")
        prior_brier = float(frame.loc["Prior Baseline", "brier"])
        prior_log_loss = float(frame.loc["Prior Baseline", "log_loss"])
        prior_acc = float(frame.loc["Prior Baseline", "accuracy"])
        prior_bal = float(frame.loc["Prior Baseline", "balanced_accuracy"])
        frame["brier_improvement_vs_prior_pct"] = (
            (prior_brier - frame["brier"]) / prior_brier * 100.0
        )
        frame["log_loss_improvement_vs_prior_pct"] = (
            (prior_log_loss - frame["log_loss"]) / prior_log_loss * 100.0
        )
        frame["accuracy_improvement_vs_prior_pp"] = (
            frame["accuracy"] - prior_acc
        ) * 100.0
        frame["balanced_accuracy_improvement_vs_prior_pp"] = (
            frame["balanced_accuracy"] - prior_bal
        ) * 100.0
        return frame.sort_values("log_loss")

    def assert_common_protocol(self) -> None:
        ref = self.prior.probabilities.index
        for result in self.tuned_results:
            if not result.probabilities.index.equals(ref):
                raise RuntimeError(f"{result.model_name} used different OOS timestamps.")


def _positive_probability(model, X: pd.DataFrame) -> np.ndarray:
    proba = np.asarray(model.predict_proba(X), dtype=float)
    classes = np.asarray(model.classes_)
    if 1 not in classes:
        raise RuntimeError("Fitted classifier does not expose positive class 1.")
    return proba[:, int(np.where(classes == 1)[0][0])]


def _inner_log_loss(X, y, estimator, *, n_splits: int, gap: int) -> float:
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    probs = []
    actuals = []
    for train_idx, valid_idx in splitter.split(X):
        model = clone(estimator)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        probs.append(_positive_probability(model, X.iloc[valid_idx]))
        actuals.append(y.iloc[valid_idx].to_numpy(dtype=int))
    y_all = np.concatenate(actuals)
    p_all = np.concatenate(probs)
    return classification_metrics(y_all, p_all).log_loss


def evaluate_prior_baseline(
    dataset: DirectionDatasetBundle,
    *,
    outer_splits: int = 5,
    gap: int = 1,
    test_size: int | None = None,
) -> PriorClassificationResult:
    splitter = TimeSeriesSplit(n_splits=outer_splits, gap=gap, test_size=test_size)
    prob_parts = []
    actual_parts = []
    return_parts = []
    fold_results = []

    for fold_number, (train_idx, test_idx) in enumerate(splitter.split(dataset.X), start=1):
        X_train, X_test = dataset.X.iloc[train_idx], dataset.X.iloc[test_idx]
        y_train, y_test = dataset.y.iloc[train_idx], dataset.y.iloc[test_idx]
        model = make_prior_classifier()
        model.fit(X_train, y_train)
        p = _positive_probability(model, X_test)
        metrics = classification_metrics(y_test, p)
        fold_results.append(
            ClassificationFoldResult(
                fold=fold_number,
                train_start=X_train.index[0],
                train_end=X_train.index[-1],
                test_start=X_test.index[0],
                test_end=X_test.index[-1],
                n_train=len(X_train),
                n_test=len(X_test),
                best_inner_log_loss=float("nan"),
                best_params={},
                metrics=metrics,
            )
        )
        prob_parts.append(pd.Series(p, index=X_test.index, name="prior_probability"))
        actual_parts.append(y_test)
        return_parts.append(dataset.next_returns.iloc[test_idx])

    probabilities = pd.concat(prob_parts).sort_index()
    actuals = pd.concat(actual_parts).sort_index()
    returns = pd.concat(return_parts).sort_index()
    return PriorClassificationResult(
        model_name="Prior Baseline",
        metrics=classification_metrics(actuals, probabilities),
        probabilities=probabilities,
        actuals=actuals,
        actual_returns=returns,
        folds=tuple(fold_results),
    )


def run_nested_classification(
    dataset: DirectionDatasetBundle,
    *,
    model_key: str,
    outer_splits: int = 5,
    inner_splits: int = 3,
    gap: int = 1,
    test_size: int | None = None,
    n_trials: int = 20,
    random_state: int = 42,
) -> NestedClassificationResult:
    if model_key not in CLASSIFICATION_MODELS:
        raise ValueError(f"model_key must be one of {CLASSIFICATION_MODELS}.")
    if gap < 1:
        raise ValueError("gap must be >= 1 for next_direction.")

    outer = TimeSeriesSplit(n_splits=outer_splits, gap=gap, test_size=test_size)
    folds = []
    prob_parts = []
    actual_parts = []
    return_parts = []
    importance_parts = []

    for fold_number, (train_idx, test_idx) in enumerate(outer.split(dataset.X), start=1):
        X_train, X_test = dataset.X.iloc[train_idx], dataset.X.iloc[test_idx]
        y_train, y_test = dataset.y.iloc[train_idx], dataset.y.iloc[test_idx]

        sampler = optuna.samplers.TPESampler(seed=random_state + fold_number)

        def objective(trial) -> float:
            params = sample_classification_hyperparameters(trial, model_key)
            estimator = build_classification_estimator(
                model_key,
                params,
                random_state=random_state + fold_number,
            )
            return _inner_log_loss(
                X_train,
                y_train,
                estimator,
                n_splits=inner_splits,
                gap=gap,
            )

        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)

        best_params = dict(study.best_trial.params)
        final_model = build_classification_estimator(
            model_key,
            best_params,
            random_state=random_state + fold_number,
        )
        final_model.fit(X_train, y_train)
        p = _positive_probability(final_model, X_test)
        metrics = classification_metrics(y_test, p)

        folds.append(
            ClassificationFoldResult(
                fold=fold_number,
                train_start=X_train.index[0],
                train_end=X_train.index[-1],
                test_start=X_test.index[0],
                test_end=X_test.index[-1],
                n_train=len(X_train),
                n_test=len(X_test),
                best_inner_log_loss=float(study.best_value),
                best_params=best_params,
                metrics=metrics,
            )
        )
        prob_parts.append(pd.Series(p, index=X_test.index, name="probability_up"))
        actual_parts.append(y_test)
        return_parts.append(dataset.next_returns.iloc[test_idx])

        importance = extract_tree_feature_importance(final_model, dataset.X.columns)
        if importance is not None:
            importance.name = fold_number
            importance_parts.append(importance)

    probabilities = pd.concat(prob_parts).sort_index()
    actuals = pd.concat(actual_parts).sort_index()
    returns = pd.concat(return_parts).sort_index()

    fold_importance = None
    if importance_parts:
        fold_importance = pd.DataFrame(importance_parts)
        fold_importance.index.name = "fold"

    return NestedClassificationResult(
        model_key=model_key,
        model_name=CLASSIFICATION_DISPLAY_NAMES[model_key],
        metrics=classification_metrics(actuals, probabilities),
        folds=tuple(folds),
        probabilities=probabilities,
        actuals=actuals,
        actual_returns=returns,
        n_splits=outer_splits,
        inner_splits=inner_splits,
        gap=gap,
        n_trials=n_trials,
        fold_feature_importances=fold_importance,
    )


def run_classification_comparison(
    dataset: DirectionDatasetBundle,
    *,
    models: tuple[str, ...] = CLASSIFICATION_MODELS,
    outer_splits: int = 5,
    inner_splits: int = 3,
    gap: int = 1,
    test_size: int | None = None,
    n_trials: int = 20,
    random_state: int = 42,
) -> ClassificationComparison:
    prior = evaluate_prior_baseline(
        dataset,
        outer_splits=outer_splits,
        gap=gap,
        test_size=test_size,
    )
    results = tuple(
        run_nested_classification(
            dataset,
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
    comparison = ClassificationComparison(prior=prior, tuned_results=results)
    comparison.assert_common_protocol()
    return comparison
