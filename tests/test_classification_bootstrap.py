from __future__ import annotations

import numpy as np

from stock_ml_lab.classification.bootstrap import classification_block_bootstrap


def test_classification_bootstrap_returns_expected_metrics() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    model = np.clip(0.2 + 0.6 * y + rng.normal(0, 0.08, 200), 0.01, 0.99)
    benchmark = np.full(200, y.mean())
    frame = classification_block_bootstrap(
        y,
        model,
        benchmark,
        n_bootstrap=100,
        block_length=10,
    )
    assert "brier_improvement_pct" in frame.index
    assert frame.loc["brier_improvement_pct", "estimate"] > 0
