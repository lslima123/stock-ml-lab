from __future__ import annotations

import numpy as np

from stock_ml_lab.validation.bootstrap import (
    circular_block_indices,
    forecast_block_bootstrap,
)


def test_circular_blocks_are_reproducible_and_valid() -> None:
    a = circular_block_indices(50, block_length=7, rng=np.random.default_rng(3))
    b = circular_block_indices(50, block_length=7, rng=np.random.default_rng(3))

    assert np.array_equal(a, b)
    assert len(a) == 50
    assert a.min() >= 0
    assert a.max() < 50


def test_bootstrap_interval_contains_point_estimate_structure() -> None:
    rng = np.random.default_rng(5)
    actual = rng.normal(0.0, 0.02, 240)
    model = 0.2 * np.roll(actual, 1)
    model[0] = 0.0
    benchmark = np.zeros_like(actual)

    result = forecast_block_bootstrap(
        actual,
        model,
        benchmark,
        n_bootstrap=100,
        block_length=10,
        random_state=12,
    )

    assert set(result) == {
        "mae",
        "rmse",
        "directional_accuracy",
        "mae_improvement_vs_benchmark_pct",
        "rmse_improvement_vs_benchmark_pct",
    }
    for interval in result.values():
        assert interval.lower <= interval.upper
        assert np.isfinite(interval.estimate)
