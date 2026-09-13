from __future__ import annotations

import numpy as np
import pandas as pd
import pandas.testing as pdt

from stock_ml_lab.classification.dataset import build_direction_dataset_from_ohlcv


def make_ohlcv(n: int = 100) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-01", periods=n)
    t = np.arange(n, dtype=float)
    close = 100.0 * np.exp(0.002 * t + 0.03 * np.sin(t / 4.0))
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 1_000_000 + 5_000 * t,
        },
        index=index,
    )


def test_direction_target_matches_next_return_sign() -> None:
    ds = build_direction_dataset_from_ohlcv(make_ohlcv(), ticker="TEST")
    expected = (ds.next_returns > 0.0).astype(int).rename("next_direction")
    pdt.assert_series_equal(ds.y, expected)


def test_direction_dataset_preserves_alignment() -> None:
    ds = build_direction_dataset_from_ohlcv(make_ohlcv())
    assert ds.X.index.equals(ds.y.index)
    assert ds.X.index.equals(ds.next_returns.index)
    assert set(ds.y.unique()).issubset({0, 1})
