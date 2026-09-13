from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd

from stock_ml_lab.classification.dataset import DirectionDatasetBundle
from stock_ml_lab.data.loader import download_ohlcv
from stock_ml_lab.dataset import DatasetBundle
from stock_ml_lab.features.technical import FEATURE_COLUMNS, build_features


LEGACY_COLUMNS = tuple(FEATURE_COLUMNS)
EXTENDED_ONLY_COLUMNS = (
    "momentum_5",
    "momentum_20",
    "momentum_60",
    "drawdown_20",
    "drawdown_60",
    "range_1d",
    "volume_zscore_20",
)
MARKET_ONLY_COLUMNS = (
    "market_return_1d",
    "market_momentum_5",
    "market_momentum_20",
    "market_momentum_60",
    "market_volatility_5",
    "market_volatility_20",
    "relative_return_1d",
    "relative_momentum_20",
)
REGIME_ONLY_COLUMNS = (
    "asset_vol_ratio_5_20",
    "market_vol_ratio_5_20",
    "rolling_beta_60",
    "rolling_corr_60",
    "asset_trend_20_60",
    "market_trend_20_60",
    "relative_strength_60",
)

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "legacy": LEGACY_COLUMNS,
    "extended": LEGACY_COLUMNS + EXTENDED_ONLY_COLUMNS,
    "market": LEGACY_COLUMNS + EXTENDED_ONLY_COLUMNS + MARKET_ONLY_COLUMNS,
    "regime": (
        LEGACY_COLUMNS
        + EXTENDED_ONLY_COLUMNS
        + MARKET_ONLY_COLUMNS
        + REGIME_ONLY_COLUMNS
    ),
}
ALL_RESEARCH_FEATURES = FEATURE_SETS["regime"]


def default_benchmark_for_ticker(ticker: str) -> str:
    ticker = ticker.strip().upper()
    if not ticker:
        raise ValueError("Ticker cannot be empty.")
    if ticker == "^BVSP":
        return "BOVA11.SA"
    if ticker.endswith(".SA"):
        return "^BVSP"
    if ticker == "SPY":
        return "^GSPC"
    if ticker == "^GSPC":
        return "SPY"
    return "SPY"


def _validate_horizon(horizon: int) -> int:
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("horizon must be >= 1 trading day.")
    return horizon


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0.0, np.nan)
    return numerator / denominator


def _build_research_features(
    asset: pd.DataFrame,
    benchmark: pd.DataFrame,
) -> pd.DataFrame:
    common = asset.index.intersection(benchmark.index).sort_values()
    if len(common) < 100:
        raise ValueError("Asset and benchmark have too little overlapping history.")

    asset = asset.loc[common].copy()
    benchmark = benchmark.loc[common].copy()

    close = asset["close"].astype(float)
    high = asset["high"].astype(float)
    low = asset["low"].astype(float)
    volume = asset["volume"].astype(float)
    market_close = benchmark["close"].astype(float)

    asset_ret = close.pct_change(fill_method=None)
    market_ret = market_close.pct_change(fill_method=None)

    features = build_features(asset)

    for window in (5, 20, 60):
        features[f"momentum_{window}"] = close.pct_change(
            periods=window, fill_method=None
        )
    features["drawdown_20"] = close / close.rolling(20, min_periods=20).max() - 1.0
    features["drawdown_60"] = close / close.rolling(60, min_periods=60).max() - 1.0
    features["range_1d"] = high / low - 1.0
    volume_mean_20 = volume.rolling(20, min_periods=20).mean()
    volume_std_20 = volume.rolling(20, min_periods=20).std()
    features["volume_zscore_20"] = _safe_ratio(
        volume - volume_mean_20, volume_std_20
    )

    features["market_return_1d"] = market_ret
    for window in (5, 20, 60):
        features[f"market_momentum_{window}"] = market_close.pct_change(
            periods=window, fill_method=None
        )
    market_vol_5 = market_ret.rolling(5, min_periods=5).std()
    market_vol_20 = market_ret.rolling(20, min_periods=20).std()
    asset_vol_5 = asset_ret.rolling(5, min_periods=5).std()
    asset_vol_20 = asset_ret.rolling(20, min_periods=20).std()
    features["market_volatility_5"] = market_vol_5
    features["market_volatility_20"] = market_vol_20
    features["relative_return_1d"] = asset_ret - market_ret
    features["relative_momentum_20"] = (
        features["momentum_20"] - features["market_momentum_20"]
    )

    features["asset_vol_ratio_5_20"] = _safe_ratio(asset_vol_5, asset_vol_20)
    features["market_vol_ratio_5_20"] = _safe_ratio(market_vol_5, market_vol_20)

    rolling_cov = asset_ret.rolling(60, min_periods=60).cov(market_ret)
    rolling_market_var = market_ret.rolling(60, min_periods=60).var()
    features["rolling_beta_60"] = _safe_ratio(rolling_cov, rolling_market_var)
    features["rolling_corr_60"] = asset_ret.rolling(60, min_periods=60).corr(
        market_ret
    )

    asset_ma20 = close.rolling(20, min_periods=20).mean()
    asset_ma60 = close.rolling(60, min_periods=60).mean()
    market_ma20 = market_close.rolling(20, min_periods=20).mean()
    market_ma60 = market_close.rolling(60, min_periods=60).mean()
    features["asset_trend_20_60"] = asset_ma20 / asset_ma60 - 1.0
    features["market_trend_20_60"] = market_ma20 / market_ma60 - 1.0
    features["relative_strength_60"] = (
        features["momentum_60"] - features["market_momentum_60"]
    )

    return features[list(ALL_RESEARCH_FEATURES)].replace([np.inf, -np.inf], np.nan)


@dataclass(frozen=True)
class ResearchDatasetBundle:
    ticker: str
    benchmark_ticker: str
    horizon: int
    X_all: pd.DataFrame
    forward_returns: pd.Series
    directions: pd.Series
    frame: pd.DataFrame

    @property
    def gap(self) -> int:
        """Purge required by an h-step forward target."""
        return self.horizon

    def feature_names(self, feature_set: str) -> tuple[str, ...]:
        if feature_set not in FEATURE_SETS:
            raise ValueError(
                f"Unknown feature set {feature_set!r}; choose from {tuple(FEATURE_SETS)}."
            )
        return FEATURE_SETS[feature_set]

    def regression_dataset(self, feature_set: str) -> DatasetBundle:
        columns = self.feature_names(feature_set)
        X = self.X_all.loc[:, columns].copy()
        y = self.forward_returns.copy()
        return DatasetBundle(
            ticker=self.ticker,
            X=X,
            y=y,
            frame=X.assign(**{y.name: y}),
            feature_names=columns,
            target_name=y.name,
        )

    def classification_dataset(self, feature_set: str) -> DirectionDatasetBundle:
        columns = self.feature_names(feature_set)
        X = self.X_all.loc[:, columns].copy()
        y = self.directions.copy()
        returns = self.forward_returns.copy()
        frame = X.copy()
        frame[returns.name] = returns
        frame[y.name] = y
        return DirectionDatasetBundle(
            ticker=self.ticker,
            X=X,
            y=y,
            next_returns=returns,
            frame=frame,
            feature_names=columns,
            target_name=y.name,
        )


def build_research_dataset_from_ohlcv(
    asset_ohlcv: pd.DataFrame,
    benchmark_ohlcv: pd.DataFrame,
    *,
    ticker: str = "UNKNOWN",
    benchmark_ticker: str = "BENCHMARK",
    horizon: int = 1,
) -> ResearchDatasetBundle:
    horizon = _validate_horizon(horizon)
    if asset_ohlcv.empty or benchmark_ohlcv.empty:
        raise ValueError("Asset and benchmark OHLCV data must be non-empty.")
    if not asset_ohlcv.index.is_monotonic_increasing:
        raise ValueError("Asset OHLCV must be ordered chronologically.")
    if not benchmark_ohlcv.index.is_monotonic_increasing:
        raise ValueError("Benchmark OHLCV must be ordered chronologically.")

    common = asset_ohlcv.index.intersection(benchmark_ohlcv.index).sort_values()
    asset = asset_ohlcv.loc[common].copy()
    benchmark = benchmark_ohlcv.loc[common].copy()
    features = _build_research_features(asset, benchmark)

    close = asset["close"].astype(float)
    forward_name = f"forward_return_{horizon}d"
    direction_name = f"forward_direction_{horizon}d"
    forward_returns = (close.shift(-horizon) / close - 1.0).rename(forward_name)
    directions = (forward_returns > 0.0).astype(int).rename(direction_name)

    # One common supervised index for all feature sets makes legacy/extended/
    # market/regime comparisons use exactly the same observations.
    supervised = features.copy()
    supervised[forward_name] = forward_returns
    supervised[direction_name] = directions
    supervised = supervised.replace([np.inf, -np.inf], np.nan).dropna(
        subset=list(ALL_RESEARCH_FEATURES) + [forward_name]
    )

    if supervised.empty:
        raise ValueError(
            "Research dataset is empty after feature/target construction. "
            "Provide a longer historical period."
        )

    X_all = supervised.loc[:, list(ALL_RESEARCH_FEATURES)].copy()
    forward = supervised[forward_name].astype(float).copy()
    direction = (forward > 0.0).astype(int).rename(direction_name)
    frame = X_all.copy()
    frame[forward_name] = forward
    frame[direction_name] = direction

    return ResearchDatasetBundle(
        ticker=ticker.strip().upper(),
        benchmark_ticker=benchmark_ticker.strip().upper(),
        horizon=horizon,
        X_all=X_all,
        forward_returns=forward,
        directions=direction,
        frame=frame,
    )


def build_research_dataset(
    *,
    ticker: str,
    start: str | date | datetime,
    end: str | date | datetime | None = None,
    horizon: int = 1,
    benchmark_ticker: str | None = None,
) -> ResearchDatasetBundle:
    ticker = ticker.strip().upper()
    benchmark = (
        default_benchmark_for_ticker(ticker)
        if benchmark_ticker is None
        else benchmark_ticker.strip().upper()
    )
    if ticker == benchmark:
        raise ValueError("ticker and benchmark_ticker must be different.")

    asset = download_ohlcv(ticker=ticker, start=start, end=end, interval="1d")
    benchmark_data = download_ohlcv(
        ticker=benchmark, start=start, end=end, interval="1d"
    )
    return build_research_dataset_from_ohlcv(
        asset,
        benchmark_data,
        ticker=ticker,
        benchmark_ticker=benchmark,
        horizon=horizon,
    )
