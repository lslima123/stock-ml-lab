from __future__ import annotations

import numpy as np
import pandas as pd

from stock_ml_lab.validation.backtest import (
    backtest_positions,
    backtest_predictions,
    positions_from_predictions,
)


def test_long_short_positions_respect_threshold() -> None:
    index = pd.bdate_range("2026-01-02", periods=5)
    pred = pd.Series([0.01, 0.001, 0.0, -0.002, -0.02], index=index)

    positions = positions_from_predictions(pred, threshold=0.005, mode="long_short")

    assert positions.tolist() == [1.0, 0.0, 0.0, 0.0, -1.0]


def test_transaction_cost_uses_position_turnover() -> None:
    index = pd.bdate_range("2026-01-02", periods=3)
    actual = pd.Series([0.01, 0.01, 0.01], index=index)
    positions = pd.Series([1.0, -1.0, -1.0], index=index)

    result = backtest_positions(actual, positions, cost_bps=10.0)

    # 0 -> +1 costs 1 unit; +1 -> -1 costs 2 units; then no change.
    assert np.allclose(result.costs.to_numpy(), [0.001, 0.002, 0.0])
    assert np.isclose(result.metrics.total_cost_return, 0.003)


def test_perfect_direction_strategy_has_positive_return_without_costs() -> None:
    index = pd.bdate_range("2026-01-02", periods=6)
    actual = pd.Series([0.01, -0.02, 0.015, -0.01, 0.005, -0.004], index=index)
    prediction = actual.copy()

    result = backtest_predictions(
        actual,
        prediction,
        threshold=0.0,
        cost_bps=0.0,
        mode="long_short",
    )

    assert (result.net_returns > 0).all()
    assert result.metrics.total_return > 0
