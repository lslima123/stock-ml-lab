from __future__ import annotations

import numpy as np
import pandas as pd

from stock_ml_lab.evaluation.importance import extract_tree_feature_importance
from stock_ml_lab.models.tree import make_catboost, make_random_forest, make_xgboost


def make_small_xy(n: int = 90):
    rng = np.random.default_rng(7)
    X = pd.DataFrame(
        {
            "return_1d": rng.normal(0.0, 0.01, n),
            "volatility_20": rng.uniform(0.005, 0.04, n),
            "rsi_14": rng.uniform(20.0, 80.0, n),
        }
    )
    y = 0.2 * X["return_1d"] - 0.03 * X["volatility_20"] + rng.normal(0, 0.005, n)
    return X, pd.Series(y)


def test_random_forest_fits_and_exposes_importance() -> None:
    X, y = make_small_xy()
    model = make_random_forest(n_estimators=20, n_jobs=1).fit(X, y)
    importance = extract_tree_feature_importance(model, X.columns)

    assert importance is not None
    assert np.isclose(importance.sum(), 1.0)


def test_xgboost_fits_and_exposes_importance() -> None:
    X, y = make_small_xy()
    model = make_xgboost(n_estimators=20, n_jobs=1).fit(X, y)
    importance = extract_tree_feature_importance(model, X.columns)

    assert importance is not None
    assert np.isclose(importance.sum(), 1.0)


def test_catboost_fits_and_exposes_importance() -> None:
    X, y = make_small_xy()
    model = make_catboost(iterations=20, thread_count=1).fit(X, y)
    importance = extract_tree_feature_importance(model, X.columns)

    assert importance is not None
    assert np.isclose(importance.sum(), 1.0)
