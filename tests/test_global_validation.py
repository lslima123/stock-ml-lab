from __future__ import annotations

import numpy as np
import pandas as pd

from stock_ml_lab.global_model.dataset import build_global_panel_from_ohlcv
from stock_ml_lab.global_model.validation import (
    evaluate_unseen_asset_holdout,
    panel_walk_forward_splits,
)


def make(seed: int, n: int = 1500) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2018-01-01", periods=n)
    r = rng.normal(0.0003, 0.012 + seed * 0.0002, n)
    close = 80 * np.exp(np.cumsum(r))
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.lognormal(14, 0.2, n),
        },
        index=index,
    )


def test_panel_split_purges_all_training_label_maturities() -> None:
    bundle = build_global_panel_from_ohlcv(
        {"AAA": make(1), "BBB": make(2), "CCC": make(3)},
        horizon=20,
    )
    folds = panel_walk_forward_splits(bundle, n_splits=2, test_size_dates=252)
    for fold in folds:
        train_maturity = pd.to_datetime(bundle.frame.iloc[fold.train_index]["target_end_date"])
        test_date = pd.to_datetime(bundle.frame.iloc[fold.test_index]["date"])
        assert train_maturity.max() < test_date.min()


def test_unseen_asset_holdout_excludes_ticker_from_training_contract() -> None:
    bundle = build_global_panel_from_ohlcv(
        {"AAA": make(1), "BBB": make(2), "CCC": make(3)},
        horizon=5,
    )
    result = evaluate_unseen_asset_holdout(
        bundle,
        task="regression",
        model="ridge",
        test_size_dates=100,
    )
    assert set(result["ticker"]) == {"AAA", "BBB", "CCC"}
    assert (result["train_assets"] == 2).all()
    assert (result["test_rows"] > 0).all()
