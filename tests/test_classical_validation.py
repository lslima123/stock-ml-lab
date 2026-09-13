from __future__ import annotations

import numpy as np
import pandas as pd

from stock_ml_lab.evaluation.classical_validation import evaluate_arima_cv, evaluate_var_cv


def make_xy(n: int = 150) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(8)
    index = pd.bdate_range("2023-01-02", periods=n)
    returns = rng.normal(0.0, 0.01, n + 1)
    volume_change = rng.normal(0.0, 0.12, n).clip(-0.9, None)
    X = pd.DataFrame(
        {
            "return_1d": returns[:-1],
            "volume_change_1d": volume_change,
        },
        index=index,
    )
    y = pd.Series(returns[1:], index=index, name="next_return")
    return X, y


def test_fixed_arima_uses_same_test_index_shape() -> None:
    X, y = make_xy()
    result = evaluate_arima_cv(
        X=X,
        y=y,
        model_name="ARIMA(1,0,1)",
        n_splits=3,
        gap=1,
        test_size=20,
        order=(1, 0, 1),
    )
    assert len(result.predictions) == 60
    assert result.predictions.index.equals(result.actuals.index)
    assert result.fold_model_details is not None


def test_var_uses_same_test_index_shape() -> None:
    X, y = make_xy()
    result = evaluate_var_cv(
        X=X,
        y=y,
        n_splits=3,
        gap=1,
        test_size=20,
        lags=3,
    )
    assert len(result.predictions) == 60
    assert result.predictions.index.equals(result.actuals.index)
    assert result.fold_model_details is not None
