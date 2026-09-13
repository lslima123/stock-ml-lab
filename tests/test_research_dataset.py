from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt

from stock_ml_lab.research.dataset import (
    FEATURE_SETS,
    build_research_dataset_from_ohlcv,
    default_benchmark_for_ticker,
)


def make_pair(n: int = 220):
    index = pd.bdate_range("2024-01-02", periods=n)
    t = np.arange(n, dtype=float)

    asset_close = 100.0 * np.exp(0.0006 * t + 0.02 * np.sin(t / 9.0))
    market_close = 200.0 * np.exp(0.0004 * t + 0.012 * np.sin(t / 11.0))

    def frame(close, volume_scale):
        return pd.DataFrame(
            {
                "open": close * 0.999,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": volume_scale * (1.0 + 0.15 * np.sin(t / 7.0)) + 1000 * t,
            },
            index=index,
        )

    return frame(asset_close, 1_000_000), frame(market_close, 5_000_000)


def test_default_benchmark_mapping() -> None:
    assert default_benchmark_for_ticker("PETR4.SA") == "^BVSP"
    assert default_benchmark_for_ticker("AAPL") == "SPY"
    assert default_benchmark_for_ticker("SPY") == "^GSPC"
    assert default_benchmark_for_ticker("^BVSP") == "BOVA11.SA"


def test_forward_return_matches_horizon() -> None:
    asset, market = make_pair()
    bundle = build_research_dataset_from_ohlcv(
        asset,
        market,
        ticker="TEST",
        benchmark_ticker="MARKET",
        horizon=5,
    )
    expected = (asset["close"].shift(-5) / asset["close"] - 1.0).loc[
        bundle.forward_returns.index
    ]
    pdt.assert_series_equal(
        bundle.forward_returns,
        expected.rename("forward_return_5d"),
    )
    assert bundle.gap == 5


def test_feature_sets_are_cumulative_and_share_index() -> None:
    asset, market = make_pair()
    bundle = build_research_dataset_from_ohlcv(asset, market, horizon=10)

    assert set(FEATURE_SETS["legacy"]).issubset(FEATURE_SETS["extended"])
    assert set(FEATURE_SETS["extended"]).issubset(FEATURE_SETS["market"])
    assert set(FEATURE_SETS["market"]).issubset(FEATURE_SETS["regime"])

    reference = bundle.X_all.index
    for feature_set in FEATURE_SETS:
        assert bundle.regression_dataset(feature_set).X.index.equals(reference)
        assert bundle.classification_dataset(feature_set).X.index.equals(reference)


def test_future_changes_do_not_modify_past_research_features() -> None:
    asset, market = make_pair()
    changed_asset = asset.copy()
    changed_market = market.copy()
    cutoff = asset.index[140]
    future = asset.index > cutoff
    changed_asset.loc[future, "close"] *= 2.5
    changed_asset.loc[future, "volume"] *= 4.0
    changed_market.loc[future, "close"] *= 1.8

    original = build_research_dataset_from_ohlcv(asset, market, horizon=5)
    modified = build_research_dataset_from_ohlcv(changed_asset, changed_market, horizon=5)

    common_past = original.X_all.index.intersection(modified.X_all.index)
    common_past = common_past[common_past <= cutoff]
    pdt.assert_frame_equal(
        original.X_all.loc[common_past],
        modified.X_all.loc[common_past],
    )


def test_direction_is_sign_of_forward_return() -> None:
    asset, market = make_pair()
    bundle = build_research_dataset_from_ohlcv(asset, market, horizon=20)
    expected = (bundle.forward_returns > 0.0).astype(int).rename(
        "forward_direction_20d"
    )
    pdt.assert_series_equal(bundle.directions, expected)
