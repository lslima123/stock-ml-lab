from __future__ import annotations

import numpy as np
import pandas as pd

from stock_ml_lab.evaluation.validation import evaluate_time_series_cv
from stock_ml_lab.models.baseline import PersistenceRegressor
from stock_ml_lab.models.linear import make_ridge_pipeline


def make_xy(n: int = 120) -> tuple[pd.DataFrame, pd.Series]:
    index = pd.bdate_range("2024-01-01", periods=n)
    rng = np.random.default_rng(42)

    current_return = rng.normal(0.0, 0.01, size=n)
    X = pd.DataFrame(
        {
            "return_1d": current_return,
            "feature_2": rng.normal(size=n),
        },
        index=index,
    )
    y = pd.Series(
        0.35 * current_return + rng.normal(0.0, 0.005, size=n),
        index=index,
        name="next_return",
    )
    return X, y


def test_cv_predictions_are_temporally_ordered_and_unique() -> None:
    X, y = make_xy()

    result = evaluate_time_series_cv(
        X=X,
        y=y,
        estimator=PersistenceRegressor(),
        model_name="Persistence",
        n_splits=4,
        gap=1,
    )

    assert result.predictions.index.is_monotonic_increasing
    assert not result.predictions.index.has_duplicates
    assert result.predictions.index.equals(result.actuals.index)


def test_gap_purges_boundary_between_train_and_test() -> None:
    X, y = make_xy()

    result = evaluate_time_series_cv(
        X=X,
        y=y,
        estimator=PersistenceRegressor(),
        model_name="Persistence",
        n_splits=4,
        gap=1,
    )

    position = {timestamp: i for i, timestamp in enumerate(X.index)}

    for fold in result.folds:
        train_end_pos = position[fold.train_end]
        test_start_pos = position[fold.test_start]
        assert test_start_pos - train_end_pos >= 2


def test_ridge_runs_inside_temporal_cv() -> None:
    X, y = make_xy()

    result = evaluate_time_series_cv(
        X=X,
        y=y,
        estimator=make_ridge_pipeline(alpha=1.0),
        model_name="Ridge",
        n_splits=4,
        gap=1,
    )

    assert len(result.folds) == 4
    assert np.isfinite(result.predictions.to_numpy()).all()
