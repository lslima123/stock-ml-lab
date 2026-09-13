from __future__ import annotations

from pathlib import Path

import pandas as pd

from stock_ml_lab.visualization.normalize import combine_m8_runs
from stock_ml_lab.visualization.sources import LoadedRun


def test_combine_m8_prefers_later_duplicate() -> None:
    first = LoadedRun(
        kind="m8",
        path=Path("/tmp/a"),
        manifest={},
        tables={
            "results": pd.DataFrame(
                [{
                    "task": "classification",
                    "horizon": 20,
                    "feature_set": "legacy",
                    "model_key": "logistic",
                    "roc_auc": 0.55,
                }]
            )
        },
    )
    second = LoadedRun(
        kind="m8",
        path=Path("/tmp/b"),
        manifest={},
        tables={
            "results": pd.DataFrame(
                [{
                    "task": "classification",
                    "horizon": 20,
                    "feature_set": "legacy",
                    "model_key": "logistic",
                    "roc_auc": 0.63,
                }]
            )
        },
    )

    merged = combine_m8_runs([first, second])
    assert len(merged) == 1
    assert merged.iloc[0]["roc_auc"] == 0.63
