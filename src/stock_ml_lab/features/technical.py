from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "return_1d",
    "return_lag_1",
    "return_lag_2",
    "return_lag_3",
    "ma_5_distance",
    "ma_10_distance",
    "ma_20_distance",
    "volatility_5",
    "volatility_20",
    "rsi_14",
    "volume_change_1d",
]


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Compute Wilder-style RSI using exponentially weighted averages."""
    if window < 2:
        raise ValueError("RSI window must be >= 2.")

    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)

    alpha = 1.0 / window
    avg_gain = gain.ewm(alpha=alpha, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=alpha, adjust=False, min_periods=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))

    rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    rsi = rsi.where(avg_loss != 0, 100.0)
    return rsi


def build_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Create features using only information available through each row's close."""
    required = {"close", "volume"}
    missing = required.difference(ohlcv.columns)
    if missing:
        raise ValueError(f"Missing columns required for features: {sorted(missing)}")

    close = ohlcv["close"].astype(float)
    volume = ohlcv["volume"].astype(float)
    returns = close.pct_change(fill_method=None)

    features = pd.DataFrame(index=ohlcv.index)
    features["return_1d"] = returns
    features["return_lag_1"] = returns.shift(1)
    features["return_lag_2"] = returns.shift(2)
    features["return_lag_3"] = returns.shift(3)

    for window in (5, 10, 20):
        moving_average = close.rolling(window=window, min_periods=window).mean()
        features[f"ma_{window}_distance"] = close / moving_average - 1.0

    features["volatility_5"] = returns.rolling(5, min_periods=5).std()
    features["volatility_20"] = returns.rolling(20, min_periods=20).std()
    features["rsi_14"] = compute_rsi(close, window=14)
    features["volume_change_1d"] = volume.pct_change(fill_method=None)

    features = features.replace([np.inf, -np.inf], np.nan)
    return features[FEATURE_COLUMNS]
