from __future__ import annotations

import numpy as np

from stock_ml_lab.validation.forecast_tests import (
    diebold_mariano_vs_benchmark,
    pesaran_timmermann_test,
)


def test_dm_strongly_favors_near_perfect_model() -> None:
    rng = np.random.default_rng(123)
    actual = rng.normal(0.002, 0.02, 500)
    model = actual + rng.normal(0.0, 0.001, 500)
    zero = np.zeros_like(actual)

    result = diebold_mariano_vs_benchmark(
        actual,
        model,
        zero,
        criterion="mse",
        lags=5,
    )

    assert result.statistic < 0
    assert result.pvalue_model_better < 0.01
    assert result.mean_loss_difference < 0


def test_pesaran_timmermann_detects_directional_signal() -> None:
    rng = np.random.default_rng(99)
    actual = rng.normal(size=600)
    predicted = actual + rng.normal(scale=0.15, size=600)

    result = pesaran_timmermann_test(actual, predicted)

    assert result.directional_accuracy > 0.8
    assert result.pvalue_larger < 0.01
