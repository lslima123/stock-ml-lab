from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd

from stock_ml_lab.dataset import build_dataset, build_dataset_from_ohlcv


@dataclass(frozen=True)
class DirectionDatasetBundle:
    ticker: str
    X: pd.DataFrame
    y: pd.Series
    next_returns: pd.Series
    frame: pd.DataFrame
    feature_names: tuple[str, ...]
    target_name: str = "next_direction"


def direction_dataset_from_regression(regression_dataset) -> DirectionDatasetBundle:
    """Convert the leakage-safe next-return dataset into a binary direction task.

    The feature timestamp remains t. `next_returns[t]` is the realized return at
    t+1 and `y[t] = 1(next_returns[t] > 0)`. Zero returns belong to class 0.
    """
    next_returns = regression_dataset.y.astype(float).rename("next_return")
    y = (next_returns > 0.0).astype(int).rename("next_direction")
    frame = regression_dataset.X.copy()
    frame["next_return"] = next_returns
    frame["next_direction"] = y

    return DirectionDatasetBundle(
        ticker=regression_dataset.ticker,
        X=regression_dataset.X.copy(),
        y=y,
        next_returns=next_returns,
        frame=frame,
        feature_names=regression_dataset.feature_names,
    )


def build_direction_dataset_from_ohlcv(
    ohlcv: pd.DataFrame,
    *,
    ticker: str = "UNKNOWN",
) -> DirectionDatasetBundle:
    regression = build_dataset_from_ohlcv(
        ohlcv,
        ticker=ticker,
        target="next_return",
    )
    return direction_dataset_from_regression(regression)


def build_direction_dataset(
    *,
    ticker: str,
    start: str | date | datetime,
    end: str | date | datetime | None = None,
) -> DirectionDatasetBundle:
    regression = build_dataset(
        ticker=ticker,
        start=start,
        end=end,
        target="next_return",
    )
    return direction_dataset_from_regression(regression)
