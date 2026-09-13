from __future__ import annotations

import pandas as pd

from stock_ml_lab.visualization.story import discovery_confirmation_metrics


def test_story_matches_discovery_to_confirmation() -> None:
    m8 = pd.DataFrame(
        [
            {
                "task": "classification",
                "horizon": 20,
                "feature_set": "legacy",
                "model_key": "logistic",
                "log_loss_improvement_vs_prior_pct": 2.24,
            },
            {
                "task": "regression",
                "horizon": 20,
                "feature_set": "legacy",
                "model_key": "ridge",
                "rmse_improvement_vs_zero_pct": 4.47,
            },
        ]
    )
    confirm = pd.DataFrame(
        [
            {
                "task": "classification",
                "horizon": 20,
                "median_primary_improvement_pct": -0.34,
            },
            {
                "task": "regression",
                "horizon": 20,
                "median_primary_improvement_pct": -0.11,
            },
        ]
    )

    metrics = discovery_confirmation_metrics(m8, confirm)
    assert len(metrics) == 2
    assert metrics[0].discovery == 2.24
    assert metrics[0].confirmation == -0.34
