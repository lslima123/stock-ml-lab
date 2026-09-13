from __future__ import annotations

import numpy as np
import pandas as pd

from stock_ml_lab.global_model.dataset import build_global_panel_from_ohlcv


def synthetic_ohlcv(seed: int, n: int = 900) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2020-01-01", periods=n)
    returns = rng.normal(0.0004, 0.015, n)
    close = 100.0 * np.exp(np.cumsum(returns))
    volume = rng.lognormal(15, 0.3, n)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": volume,
        },
        index=index,
    )


def test_global_panel_tracks_target_maturity() -> None:
    bundle = build_global_panel_from_ohlcv(
        {"AAA": synthetic_ohlcv(1), "BBB": synthetic_ohlcv(2)},
        horizon=20,
    )
    assert set(bundle.tickers) == {"AAA", "BBB"}
    assert len(bundle.feature_names) == 11
    assert (pd.to_datetime(bundle.frame["target_end_date"]) > pd.to_datetime(bundle.frame["date"])).all()

    aaa = bundle.frame[bundle.frame["ticker"] == "AAA"]
    positions = aaa["date"].reset_index(drop=True)
    maturities = aaa["target_end_date"].reset_index(drop=True)
    assert (maturities > positions).all()
