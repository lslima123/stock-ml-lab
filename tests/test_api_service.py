from __future__ import annotations

import numpy as np
import pandas as pd

from stock_ml_lab.api.schemas import PredictionRequest
from stock_ml_lab.api import service


def make_ohlcv(n: int = 320) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-02", periods=n)
    x = np.arange(n, dtype=float)
    close = 100.0 + 0.04 * x + 4.0 * np.sin(x / 8.0) + 1.5 * np.sin(x / 2.7)
    open_ = close * (1.0 + 0.001 * np.sin(x / 3.0))
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    volume = 1_000_000.0 + 50_000.0 * (1.0 + np.sin(x / 5.0))
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


def test_local_ridge_returns_future_horizon_prediction(monkeypatch) -> None:
    data = make_ohlcv()
    monkeypatch.setattr(service, "download_ohlcv", lambda **_: data)
    request = PredictionRequest(
        ticker="TEST", task="regression", model="ridge", horizon=20
    )
    result = service.predict_local(request)

    assert result["ticker"] == "TEST"
    assert result["scope"] == "local"
    assert result["training"]["training_rows"] > 200
    assert result["as_of"] == data.index[-1].date()
    assert result["prediction"]["predicted_return"] is not None
    assert result["prediction"]["predicted_price"] > 0


def test_local_logistic_returns_probability(monkeypatch) -> None:
    data = make_ohlcv()
    monkeypatch.setattr(service, "download_ohlcv", lambda **_: data)
    request = PredictionRequest(
        ticker="TEST", task="classification", model="logistic", horizon=10
    )
    result = service.predict_local(request)

    p = result["prediction"]["probability_up"]
    assert 0.0 <= p <= 1.0
    assert result["prediction"]["predicted_direction"] in {"up", "down"}


def test_market_snapshot_uses_latest_row(monkeypatch) -> None:
    data = make_ohlcv()
    monkeypatch.setattr(service, "download_ohlcv", lambda **_: data)
    result = service.market_snapshot(ticker="test", start="2024-01-01", end=None)
    assert result["ticker"] == "TEST"
    assert result["as_of"] == data.index[-1].date()
    assert result["close"] == float(data.iloc[-1]["close"])
