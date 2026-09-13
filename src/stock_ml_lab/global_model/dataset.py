from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Mapping

import numpy as np
import pandas as pd

from stock_ml_lab.data.loader import download_ohlcv
from stock_ml_lab.features.technical import FEATURE_COLUMNS, build_features


SUPPORTED_HORIZONS = (1, 5, 10, 20)


def _validate_horizon(horizon: int) -> int:
    horizon = int(horizon)
    if horizon not in SUPPORTED_HORIZONS:
        raise ValueError(f"horizon must be one of {SUPPORTED_HORIZONS}.")
    return horizon


@dataclass(frozen=True)
class GlobalPanelBundle:
    horizon: int
    frame: pd.DataFrame
    feature_names: tuple[str, ...]
    tickers: tuple[str, ...]

    @property
    def X(self) -> pd.DataFrame:
        return self.frame.loc[:, list(self.feature_names)].copy()

    @property
    def regression_y(self) -> pd.Series:
        return self.frame["forward_return"].astype(float).copy()

    @property
    def classification_y(self) -> pd.Series:
        return self.frame["direction"].astype(int).copy()

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.frame["date"])

    @property
    def target_end_dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(self.frame["target_end_date"])


def _asset_panel(
    ohlcv: pd.DataFrame,
    *,
    ticker: str,
    horizon: int,
) -> pd.DataFrame:
    if ohlcv.empty:
        raise ValueError(f"{ticker}: OHLCV is empty.")
    if not ohlcv.index.is_monotonic_increasing:
        raise ValueError(f"{ticker}: OHLCV must be chronologically ordered.")
    if ohlcv.index.has_duplicates:
        raise ValueError(f"{ticker}: OHLCV index contains duplicates.")

    horizon = _validate_horizon(horizon)
    features = build_features(ohlcv).replace([np.inf, -np.inf], np.nan)
    close = ohlcv["close"].astype(float)

    forward = (close.shift(-horizon) / close - 1.0).rename("forward_return")
    index_series = pd.Series(
        pd.DatetimeIndex(ohlcv.index),
        index=ohlcv.index,
        dtype="datetime64[ns]",
    )
    target_end = index_series.shift(-horizon).rename("target_end_date")

    frame = features.copy()
    frame["forward_return"] = forward
    frame["target_end_date"] = target_end
    frame = frame.dropna(
        subset=list(FEATURE_COLUMNS) + ["forward_return", "target_end_date"]
    )
    if frame.empty:
        raise ValueError(f"{ticker}: no supervised rows remain after feature construction.")

    frame["direction"] = (frame["forward_return"] > 0.0).astype(int)
    frame["ticker"] = ticker.strip().upper()
    frame["date"] = pd.DatetimeIndex(frame.index)
    frame = frame.reset_index(drop=True)

    # Metadata first, numerical features next.
    return frame[
        [
            "ticker",
            "date",
            "target_end_date",
            *FEATURE_COLUMNS,
            "forward_return",
            "direction",
        ]
    ]


def build_global_panel_from_ohlcv(
    data_by_ticker: Mapping[str, pd.DataFrame],
    *,
    horizon: int,
) -> GlobalPanelBundle:
    horizon = _validate_horizon(horizon)
    if len(data_by_ticker) < 2:
        raise ValueError("A global panel requires at least two assets.")

    frames: list[pd.DataFrame] = []
    for ticker, ohlcv in data_by_ticker.items():
        frames.append(_asset_panel(ohlcv, ticker=ticker, horizon=horizon))

    panel = pd.concat(frames, ignore_index=True, sort=False)
    panel = panel.sort_values(["date", "ticker"], kind="stable").reset_index(drop=True)

    if panel[list(FEATURE_COLUMNS)].isna().any().any():
        raise ValueError("Global feature matrix contains NaN values.")
    if not np.isfinite(panel[list(FEATURE_COLUMNS)].to_numpy(dtype=float)).all():
        raise ValueError("Global feature matrix contains non-finite values.")

    tickers = tuple(sorted(panel["ticker"].unique().tolist()))
    return GlobalPanelBundle(
        horizon=horizon,
        frame=panel,
        feature_names=tuple(FEATURE_COLUMNS),
        tickers=tickers,
    )


def build_global_panel(
    *,
    tickers: tuple[str, ...],
    start: str | date | datetime,
    end: str | date | datetime | None,
    horizon: int,
) -> GlobalPanelBundle:
    cleaned = tuple(dict.fromkeys(t.strip().upper() for t in tickers if t.strip()))
    if len(cleaned) < 2:
        raise ValueError("At least two distinct tickers are required.")

    data = {
        ticker: download_ohlcv(
            ticker=ticker,
            start=start,
            end=end,
            interval="1d",
        )
        for ticker in cleaned
    }
    return build_global_panel_from_ohlcv(data, horizon=horizon)
