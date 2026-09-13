from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt

from stock_ml_lab.dataset import build_dataset_from_ohlcv
from stock_ml_lab.features.technical import build_features


def make_ohlcv(n: int = 100) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-01", periods=n)
    t = np.arange(n, dtype=float)

    close = 100.0 * np.exp(0.001 * t + 0.015 * np.sin(t / 5.0))
    open_ = close * (1.0 - 0.001)
    high = close * 1.01
    low = close * 0.99
    volume = 1_000_000.0 + 10_000.0 * t + 40_000.0 * np.cos(t / 7.0)

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


def test_target_is_next_trading_day_return() -> None:
    data = make_ohlcv()
    dataset = build_dataset_from_ohlcv(data, ticker="TEST")

    returns = data["close"].pct_change(fill_method=None)
    expected = returns.shift(-1).loc[dataset.y.index]

    pdt.assert_series_equal(dataset.y, expected.rename("next_return"))


def test_future_changes_do_not_modify_past_features() -> None:
    original = make_ohlcv()
    modified = original.copy()

    cutoff = original.index[60]
    future_mask = modified.index > cutoff

    modified.loc[future_mask, "close"] *= 3.0
    modified.loc[future_mask, "volume"] *= 7.0

    features_original = build_features(original).loc[:cutoff]
    features_modified = build_features(modified).loc[:cutoff]

    pdt.assert_frame_equal(features_original, features_modified)


def test_supervised_dataset_has_only_finite_values() -> None:
    data = make_ohlcv()
    dataset = build_dataset_from_ohlcv(data)

    assert np.isfinite(dataset.X.to_numpy()).all()
    assert np.isfinite(dataset.y.to_numpy()).all()
    assert dataset.X.index.equals(dataset.y.index)


def test_feature_order_is_stable() -> None:
    data = make_ohlcv()
    dataset = build_dataset_from_ohlcv(data)

    assert list(dataset.X.columns) == list(dataset.feature_names)
