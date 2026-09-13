from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from stock_ml_lab.evaluation.importance import extract_tree_feature_importance
from stock_ml_lab.evaluation.validation import evaluate_time_series_cv


def make_xy(n: int = 120):
    index = pd.bdate_range("2024-01-01", periods=n)
    rng = np.random.default_rng(12)
    strong = rng.normal(size=n)
    weak = rng.normal(size=n)
    X = pd.DataFrame({"strong": strong, "weak": weak}, index=index)
    y = pd.Series(0.8 * strong + 0.05 * weak + rng.normal(0, 0.1, n), index=index)
    return X, y


def test_cv_collects_one_normalized_importance_vector_per_fold() -> None:
    X, y = make_xy()
    result = evaluate_time_series_cv(
        X=X,
        y=y,
        estimator=RandomForestRegressor(n_estimators=20, random_state=1, n_jobs=1),
        model_name="RF",
        n_splits=3,
        gap=1,
    )

    assert result.fold_feature_importances is not None
    assert result.fold_feature_importances.shape == (3, 2)
    assert np.allclose(result.fold_feature_importances.sum(axis=1), 1.0)

    summary = result.feature_importance_frame()
    assert summary is not None
    assert list(summary.columns) == ["mean_importance", "std_importance"]
    assert summary.index[0] == "strong"


def test_non_tree_model_returns_no_importance() -> None:
    class Dummy:
        pass

    assert extract_tree_feature_importance(Dummy(), ["x"]) is None
