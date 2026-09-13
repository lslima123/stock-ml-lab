from __future__ import annotations

import numpy as np
import pandas as pd

from stock_ml_lab.classification.backtest import positions_from_probabilities


def test_long_short_probability_margin() -> None:
    idx = pd.bdate_range("2025-01-01", periods=5)
    p = pd.Series([0.70, 0.56, 0.50, 0.44, 0.20], index=idx)
    pos = positions_from_probabilities(p, margin=0.05, mode="long_short")
    assert np.array_equal(pos.to_numpy(), [1, 1, 0, -1, -1])


def test_long_flat_probability_margin() -> None:
    idx = pd.bdate_range("2025-01-01", periods=4)
    p = pd.Series([0.70, 0.55, 0.40, 0.20], index=idx)
    pos = positions_from_probabilities(p, margin=0.05, mode="long_flat")
    assert np.array_equal(pos.to_numpy(), [1, 0, 0, 0])
