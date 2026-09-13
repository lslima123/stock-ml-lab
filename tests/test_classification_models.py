from __future__ import annotations

import numpy as np
import pandas as pd

from stock_ml_lab.classification.models import (
    make_catboost_classifier,
    make_logistic_classifier,
    make_prior_classifier,
    make_random_forest_classifier,
    make_xgboost_classifier,
)


def make_xy(n: int = 80):
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    y = pd.Series((X["a"] + 0.2 * rng.normal(size=n) > 0).astype(int))
    return X, y


def test_all_classifiers_expose_positive_probability() -> None:
    X, y = make_xy()
    models = [
        make_prior_classifier(),
        make_logistic_classifier(),
        make_random_forest_classifier(n_estimators=20, max_depth=3),
        make_xgboost_classifier(n_estimators=20, max_depth=2),
        make_catboost_classifier(iterations=20, depth=3),
    ]
    for model in models:
        model.fit(X, y)
        p = model.predict_proba(X)
        assert p.shape == (len(X), 2)
        assert np.isfinite(p).all()
