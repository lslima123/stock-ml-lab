from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from stock_ml_lab.scope_confirmation.config import locked_config
from stock_ml_lab.scope_confirmation.runner import run_locked_scope_confirmation


def _ohlcv(seed: int, n: int = 260) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2024-01-02", periods=n)
    returns = rng.normal(0.0004, 0.012, n)
    close = 100 * np.exp(np.cumsum(returns))
    return pd.DataFrame({
        "open": close * 0.999,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": 1_000_000 + rng.integers(0, 100_000, n),
    }, index=index)


def test_locked_config_is_disjoint_and_overlap_aware() -> None:
    config = locked_config()
    assert config.horizon == 5
    assert config.hac_lags >= config.horizon - 1
    assert config.bootstrap_block_length_dates >= config.horizon
    assert set(config.training_universe).isdisjoint(config.confirmation_universe)
    assert "AXIA3.SA" in config.confirmation_universe
    assert "ELET3.SA" not in config.confirmation_universe
    assert config.protocol_amendments


def test_locked_runner_keeps_confirmation_out_of_global_fit(tmp_path) -> None:
    base = locked_config()
    config = replace(
        base,
        n_splits=2,
        test_size_dates=20,
        bootstrap_repetitions=1000,
    )
    training = {ticker: _ohlcv(i) for i, ticker in enumerate(config.training_universe)}
    confirmation = {
        ticker: _ohlcv(100 + i) for i, ticker in enumerate(config.confirmation_universe)
    }
    result = run_locked_scope_confirmation(
        training, confirmation, config=config, output_base=tmp_path
    )
    audit = result["training_audit"]
    global_audit = audit[audit["scope"] == "global"]
    assert (global_audit["confirmation_rows_in_fit"] == 0).all()
    assert bool(audit["purge_valid"].all())
    assert len(result["primary_results"]) == 2
    assert set(result["predictions"]["ticker"]) == set(config.confirmation_universe)
