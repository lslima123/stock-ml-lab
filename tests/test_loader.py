from __future__ import annotations

import pandas as pd

from stock_ml_lab.data import loader


def test_loader_normalizes_provider_columns(monkeypatch) -> None:
    index = pd.to_datetime(["2025-01-03", "2025-01-02"])
    provider_frame = pd.DataFrame(
        {
            "Open": [101.0, 100.0],
            "High": [102.0, 101.0],
            "Low": [99.0, 98.0],
            "Close": [101.5, 100.5],
            "Volume": [1_100, 1_000],
        },
        index=index,
    )

    monkeypatch.setattr(
        loader,
        "_download_with_yfinance",
        lambda **kwargs: provider_frame,
    )

    result = loader.download_ohlcv(
        ticker="petr4.sa",
        start="2025-01-01",
        end="2025-02-01",
    )

    assert list(result.columns) == ["open", "high", "low", "close", "volume"]
    assert result.index.is_monotonic_increasing
