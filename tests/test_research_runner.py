from __future__ import annotations

import numpy as np
import pandas as pd

from stock_ml_lab.classification.nested import run_classification_comparison
from stock_ml_lab.research.dataset import build_research_dataset_from_ohlcv
from stock_ml_lab.tuning.nested import run_tuning_comparison


def make_pair(n: int = 260):
    index = pd.bdate_range("2023-01-02", periods=n)
    rng = np.random.default_rng(123)
    market_r = rng.normal(0.0003, 0.01, n)
    asset_r = 0.25 * market_r + rng.normal(0.0004, 0.012, n)
    market_close = 100 * np.cumprod(1 + market_r)
    asset_close = 50 * np.cumprod(1 + asset_r)

    def frame(close, base_volume):
        return pd.DataFrame(
            {
                "open": close * 0.999,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": base_volume + rng.integers(0, 100_000, n),
            },
            index=index,
        )

    return frame(asset_close, 1_000_000), frame(market_close, 3_000_000)


def test_multihorizon_classification_uses_horizon_gap() -> None:
    asset, market = make_pair()
    bundle = build_research_dataset_from_ohlcv(asset, market, horizon=5)
    dataset = bundle.classification_dataset("legacy")

    comparison = run_classification_comparison(
        dataset,
        models=("logistic",),
        outer_splits=2,
        inner_splits=2,
        gap=bundle.gap,
        test_size=30,
        n_trials=1,
        random_state=5,
    )
    result = comparison.tuned_results[0]
    positions = {timestamp: i for i, timestamp in enumerate(dataset.X.index)}
    for fold in result.folds:
        assert positions[fold.test_start] - positions[fold.train_end] >= 6


def test_multihorizon_regression_smoke() -> None:
    asset, market = make_pair()
    bundle = build_research_dataset_from_ohlcv(asset, market, horizon=10)
    dataset = bundle.regression_dataset("market")

    comparison = run_tuning_comparison(
        dataset,
        models=("ridge",),
        outer_splits=2,
        inner_splits=2,
        gap=bundle.gap,
        test_size=30,
        n_trials=1,
        random_state=7,
    )
    result = comparison.tuned_results[0]
    assert len(result.predictions) == 60
    assert result.predictions.index.equals(result.actuals.index)
    assert np.isfinite(result.predictions.to_numpy()).all()
