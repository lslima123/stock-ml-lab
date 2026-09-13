from __future__ import annotations

import numpy as np
import pandas as pd

from stock_ml_lab.models.classical import build_var_frame, fit_arima, fit_var


def make_X(n: int = 100) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    index = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame(
        {
            "return_1d": rng.normal(0.0, 0.01, n),
            "volume_change_1d": rng.normal(0.0, 0.15, n).clip(-0.9, None),
        },
        index=index,
    )


def test_arima_can_forecast_and_append_without_refit() -> None:
    X = make_X()
    result = fit_arima(X["return_1d"].iloc[:80], order=(1, 0, 1))
    result = result.append(np.asarray([X["return_1d"].iloc[80]]), refit=False)
    forecast = np.asarray(result.forecast(steps=1), dtype=float)
    assert forecast.shape == (1,)
    assert np.isfinite(forecast).all()


def test_var_frame_is_finite_and_has_expected_variables() -> None:
    X = make_X()
    frame = build_var_frame(X)
    assert list(frame.columns) == ["return_1d", "log_volume_change_1d"]
    assert np.isfinite(frame.to_numpy()).all()


def test_var_can_make_one_step_forecast() -> None:
    X = make_X()
    frame = build_var_frame(X)
    result = fit_var(frame.iloc[:80], lags=3)
    forecast = result.forecast(frame.iloc[:80].to_numpy()[-result.k_ar :], steps=1)
    assert forecast.shape == (1, 2)
    assert np.isfinite(forecast).all()
