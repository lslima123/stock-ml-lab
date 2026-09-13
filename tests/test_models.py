from __future__ import annotations

import numpy as np
import pandas as pd

from stock_ml_lab.models.baseline import PersistenceRegressor, ZeroReturnRegressor
from stock_ml_lab.models.linear import make_ridge_pipeline


def sample_xy() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame(
        {
            "return_1d": [0.01, -0.02, 0.03, 0.015],
            "other": [1.0, 2.0, 3.0, 4.0],
        }
    )
    y = pd.Series([0.005, -0.01, 0.02, 0.001])
    return X, y


def test_zero_return_regressor() -> None:
    X, y = sample_xy()
    model = ZeroReturnRegressor().fit(X, y)

    assert np.array_equal(model.predict(X), np.zeros(len(X)))


def test_persistence_regressor_uses_current_return() -> None:
    X, y = sample_xy()
    model = PersistenceRegressor().fit(X, y)

    assert np.array_equal(model.predict(X), X["return_1d"].to_numpy())


def test_ridge_pipeline_contains_scaler_and_ridge() -> None:
    pipeline = make_ridge_pipeline(alpha=2.5)

    assert list(pipeline.named_steps) == ["scaler", "ridge"]
    assert pipeline.named_steps["ridge"].alpha == 2.5
