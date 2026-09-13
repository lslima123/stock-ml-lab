from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit

from stock_ml_lab.evaluation.importance import extract_tree_feature_importance
from stock_ml_lab.evaluation.metrics import RegressionMetrics, regression_metrics


@dataclass(frozen=True)
class FoldResult:
    fold: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    n_train: int
    n_test: int
    metrics: RegressionMetrics


@dataclass(frozen=True)
class CVResult:
    model_name: str
    metrics: RegressionMetrics
    folds: tuple[FoldResult, ...]
    predictions: pd.Series
    actuals: pd.Series
    n_splits: int
    gap: int
    fold_feature_importances: pd.DataFrame | None = None
    fold_model_details: pd.DataFrame | None = None

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
                    **fold.metrics.to_dict(),
                }
            )
        return pd.DataFrame(rows).set_index("fold")

    def feature_importance_frame(self) -> pd.DataFrame | None:
        """Return mean and fold-to-fold standard deviation of tree importance."""
        if self.fold_feature_importances is None:
            return None

        frame = pd.DataFrame(
            {
                "mean_importance": self.fold_feature_importances.mean(axis=0),
                "std_importance": self.fold_feature_importances.std(axis=0, ddof=0),
            }
        )
        return frame.sort_values("mean_importance", ascending=False)


def evaluate_time_series_cv(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    estimator,
    model_name: str,
    n_splits: int = 5,
    gap: int = 1,
    test_size: int | None = None,
) -> CVResult:
    """Evaluate an estimator with expanding-window temporal cross-validation.

    `gap=1` is the safe default for the one-step-ahead target when each model
    is fitted once at the beginning of a test fold. The final training label
    depends on the following trading day's close, so one row is purged before
    the first test observation.
    """
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")
    if not isinstance(y, pd.Series):
        raise TypeError("y must be a pandas Series.")
    if not X.index.equals(y.index):
        raise ValueError("X and y must have identical indices.")
    if not X.index.is_monotonic_increasing:
        raise ValueError("X and y must be ordered chronologically.")
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2.")
    if gap < 1:
        raise ValueError("gap must be >= 1 for the one-step-ahead next_return target.")

    splitter = TimeSeriesSplit(
        n_splits=n_splits,
        gap=gap,
        test_size=test_size,
    )

    fold_results: list[FoldResult] = []
    prediction_parts: list[pd.Series] = []
    actual_parts: list[pd.Series] = []
    importance_parts: list[pd.Series] = []

    for fold_number, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        if train_idx[-1] + gap >= test_idx[0]:
            raise RuntimeError("Temporal purge invariant was violated.")

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        model = clone(estimator)
        model.fit(X_train, y_train)
        y_pred = np.asarray(model.predict(X_test), dtype=float)

        fold_metrics = regression_metrics(y_test.to_numpy(), y_pred)

        fold_results.append(
            FoldResult(
                fold=fold_number,
                train_start=X_train.index[0],
                train_end=X_train.index[-1],
                test_start=X_test.index[0],
                test_end=X_test.index[-1],
                n_train=len(X_train),
                n_test=len(X_test),
                metrics=fold_metrics,
            )
        )

        prediction_parts.append(pd.Series(y_pred, index=X_test.index, name="prediction"))
        actual_parts.append(y_test.rename("actual"))

        importance = extract_tree_feature_importance(model, X.columns)
        if importance is not None:
            importance.name = fold_number
            importance_parts.append(importance)

    predictions = pd.concat(prediction_parts).sort_index()
    actuals = pd.concat(actual_parts).sort_index()

    if predictions.index.has_duplicates:
        raise RuntimeError("Cross-validation produced duplicate prediction timestamps.")

    overall_metrics = regression_metrics(actuals.to_numpy(), predictions.to_numpy())

    fold_importances: pd.DataFrame | None = None
    if importance_parts:
        if len(importance_parts) != n_splits:
            raise RuntimeError("Feature importance was not available for every fold.")
        fold_importances = pd.DataFrame(importance_parts)
        fold_importances.index.name = "fold"

    return CVResult(
        model_name=model_name,
        metrics=overall_metrics,
        folds=tuple(fold_results),
        predictions=predictions,
        actuals=actuals,
        n_splits=n_splits,
        gap=gap,
        fold_feature_importances=fold_importances,
        fold_model_details=None,
    )
