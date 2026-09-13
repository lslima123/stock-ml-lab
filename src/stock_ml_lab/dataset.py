from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd

from stock_ml_lab.data.loader import download_ohlcv
from stock_ml_lab.features.technical import FEATURE_COLUMNS, build_features


SUPPORTED_TARGETS = {"next_return"}


@dataclass(frozen=True)
class DatasetBundle:
    ticker: str
    X: pd.DataFrame
    y: pd.Series
    frame: pd.DataFrame
    feature_names: tuple[str, ...]
    target_name: str


def build_dataset_from_ohlcv(
    ohlcv: pd.DataFrame,
    *,
    ticker: str = "UNKNOWN",
    target: str = "next_return",
) -> DatasetBundle:
    """Build a supervised dataset from normalized OHLCV data."""
    if target not in SUPPORTED_TARGETS:
        raise ValueError(
            f"Unsupported target '{target}'. Supported targets: {sorted(SUPPORTED_TARGETS)}"
        )

    if ohlcv.empty:
        raise ValueError("OHLCV data cannot be empty.")

    if not ohlcv.index.is_monotonic_increasing:
        raise ValueError("OHLCV data must be ordered chronologically.")

    features = build_features(ohlcv)
    returns = ohlcv["close"].astype(float).pct_change(fill_method=None)
    target_series = returns.shift(-1).rename("next_return")

    supervised = features.join(target_series, how="inner")
    supervised = supervised.replace([np.inf, -np.inf], np.nan).dropna()

    if supervised.empty:
        raise ValueError(
            "Dataset is empty after feature/target construction. "
            "Provide a longer historical period."
        )

    X = supervised[FEATURE_COLUMNS].copy()
    y = supervised["next_return"].copy()

    return DatasetBundle(
        ticker=ticker.strip().upper(),
        X=X,
        y=y,
        frame=supervised,
        feature_names=tuple(FEATURE_COLUMNS),
        target_name="next_return",
    )


def build_dataset(
    *,
    ticker: str,
    start: str | date | datetime,
    end: str | date | datetime | None = None,
    target: str = "next_return",
) -> DatasetBundle:
    ohlcv = download_ohlcv(
        ticker=ticker,
        start=start,
        end=end,
        interval="1d",
    )
    return build_dataset_from_ohlcv(
        ohlcv,
        ticker=ticker,
        target=target,
    )
