from __future__ import annotations

import numpy as np

from stock_ml_lab.robustness.multiple_testing import (
    benjamini_hochberg,
    exact_sign_test_greater,
)


def test_bh_adjustment_known_values() -> None:
    p = np.array([0.01, 0.04, 0.03, 0.20])
    q = benjamini_hochberg(p)
    assert np.allclose(q, [0.04, 0.0533333333333, 0.0533333333333, 0.20])


def test_bh_preserves_nan() -> None:
    q = benjamini_hochberg([0.01, np.nan, 0.20])
    assert np.isclose(q[0], 0.02)
    assert np.isnan(q[1])
    assert np.isclose(q[2], 0.20)


def test_exact_sign_test_all_wins() -> None:
    assert np.isclose(exact_sign_test_greater(10, 10), 1 / 1024)


def test_exact_sign_test_half_wins_is_not_significant() -> None:
    assert exact_sign_test_greater(5, 10) > 0.5
