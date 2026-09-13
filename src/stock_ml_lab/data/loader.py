from __future__ import annotations

from datetime import date, datetime

import pandas as pd


class MarketDataError(RuntimeError):
    """Raised when market data cannot be downloaded or validated."""


def _download_with_yfinance(
    *,
    ticker: str,
    start: str | date | datetime,
    end: str | date | datetime | None,
    interval: str,
) -> pd.DataFrame:
    import yfinance as yf

    return yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
        actions=False,
        multi_level_index=False,
        threads=False,
    )


def _normalize_ohlcv(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        raise MarketDataError("No market data was returned.")

    data = frame.copy()

    if isinstance(data.columns, pd.MultiIndex):
        if data.columns.nlevels != 2:
            raise MarketDataError("Unexpected MultiIndex format returned by provider.")
        data.columns = data.columns.get_level_values(0)

    data = data.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )

    required = ["open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise MarketDataError(f"Missing required OHLCV columns: {missing}")

    data = data[required].copy()
    data.index = pd.to_datetime(data.index)

    if data.index.tz is not None:
        data.index = data.index.tz_localize(None)

    data = data[~data.index.duplicated(keep="last")].sort_index()

    for column in required:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=required)

    if data.empty:
        raise MarketDataError("All downloaded OHLCV rows became invalid after cleaning.")

    if not data.index.is_monotonic_increasing:
        raise MarketDataError("Market data index is not ordered chronologically.")

    if (data["close"] <= 0).any():
        raise MarketDataError("Close prices must be positive.")

    if (data["volume"] < 0).any():
        raise MarketDataError("Volume cannot be negative.")

    return data


def download_ohlcv(
    ticker: str,
    start: str | date | datetime,
    end: str | date | datetime | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Download adjusted OHLCV data for a single ticker.

    `end` follows yfinance semantics and is exclusive.
    """
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("Ticker cannot be empty.")

    if interval != "1d":
        raise ValueError("Milestone 1 supports only daily data (interval='1d').")

    frame = _download_with_yfinance(
        ticker=ticker,
        start=start,
        end=end,
        interval=interval,
    )
    return _normalize_ohlcv(frame)
