from __future__ import annotations

import numpy as np
import optuna
import pandas as pd

from stock_ml_lab.tuning.nested import run_nested_tuning
from stock_ml_lab.tuning.search_spaces import (
    TUNABLE_MODELS,
    build_estimator,
    sample_hyperparameters,
)


def make_xy(n: int = 180):
    index = pd.bdate_range("2023-01-02", periods=n)
    rng = np.random.default_rng(7)
    X = pd.DataFrame(
        {
            "return_1d": rng.normal(0.0, 0.01, n),
            "return_lag_1": rng.normal(0.0, 0.01, n),
            "ma_10_distance": rng.normal(0.0, 0.03, n),
        },
        index=index,
    )
    y = pd.Series(
        0.15 * X["return_1d"].to_numpy()
        - 0.08 * X["return_lag_1"].to_numpy()
        + rng.normal(0.0, 0.008, n),
        index=index,
        name="next_return",
    )
    return X, y


def test_all_search_spaces_build_estimators() -> None:
    fixed = {
        "ridge": {"alpha": 1.0},
        "random_forest": {
            "n_estimators": 200,
            "max_depth": 4,
            "min_samples_leaf": 5,
            "max_features": 0.7,
        },
        "xgboost": {
            "n_estimators": 200,
            "max_depth": 3,
            "learning_rate": 0.03,
            "min_child_weight": 5.0,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.01,
            "reg_lambda": 1.0,
        },
        "catboost": {
            "iterations": 200,
            "depth": 4,
            "learning_rate": 0.03,
            "l2_leaf_reg": 3.0,
        },
    }

    for key in TUNABLE_MODELS:
        trial = optuna.trial.FixedTrial(fixed[key])
        params = sample_hyperparameters(trial, key)
        estimator = build_estimator(key, params)
        assert estimator is not None


def test_nested_ridge_produces_unique_outer_predictions() -> None:
    X, y = make_xy()

    result = run_nested_tuning(
        X=X,
        y=y,
        model_key="ridge",
        outer_splits=3,
        inner_splits=2,
        gap=1,
        test_size=30,
        n_trials=2,
        random_state=11,
    )

    assert len(result.folds) == 3
    assert len(result.predictions) == 90
    assert result.predictions.index.is_monotonic_increasing
    assert not result.predictions.index.has_duplicates
    assert result.predictions.index.equals(result.actuals.index)
    assert np.isfinite(result.predictions.to_numpy()).all()


def test_outer_fold_is_purged_and_params_are_recorded() -> None:
    X, y = make_xy()

    result = run_nested_tuning(
        X=X,
        y=y,
        model_key="ridge",
        outer_splits=3,
        inner_splits=2,
        gap=1,
        test_size=30,
        n_trials=2,
    )
    positions = {timestamp: i for i, timestamp in enumerate(X.index)}

    for fold in result.folds:
        assert positions[fold.test_start] - positions[fold.train_end] >= 2
        assert "alpha" in fold.best_params
        assert fold.best_inner_rmse > 0
