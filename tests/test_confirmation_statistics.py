from __future__ import annotations

import numpy as np

from stock_ml_lab.confirmation.statistics import (
    binary_brier_losses,
    binary_log_losses,
    hac_loss_differential_test,
)


def test_hac_loss_test_negative_statistic_favors_better_model() -> None:
    benchmark = np.linspace(1.0, 2.0, 200)
    model = benchmark - 0.2 + 0.01 * np.sin(np.arange(200))
    result = hac_loss_differential_test(model, benchmark, lags=5)
    assert result.mean_loss_difference < 0
    assert result.statistic < 0
    assert result.pvalue_model_better < 0.05


def test_binary_losses_are_observation_level() -> None:
    y = np.array([0, 1])
    p = np.array([0.25, 0.75])
    log_losses = binary_log_losses(y, p)
    brier = binary_brier_losses(y, p)
    assert log_losses.shape == (2,)
    assert np.allclose(brier, [0.0625, 0.0625])
