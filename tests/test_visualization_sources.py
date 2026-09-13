from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from stock_ml_lab.visualization.sources import discover_runs, load_run


def test_loads_m81_schema(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    pd.DataFrame([{"ticker": "AAPL"}]).to_csv(run / "asset_results.csv", index=False)
    pd.DataFrame([{"task": "classification"}]).to_csv(run / "candidate_summary.csv", index=False)
    pd.DataFrame([{"phase": 0}]).to_csv(run / "phase_results.csv", index=False)
    (run / "manifest.json").write_text(json.dumps({"milestone": "8.1"}))

    loaded = load_run(run)
    assert loaded.kind == "m8_1"
    assert "asset_results" in loaded.tables


def test_discovers_nested_runs(tmp_path: Path) -> None:
    run = tmp_path / "reports" / "m8" / "20260101_run"
    run.mkdir(parents=True)
    pd.DataFrame([{"task": "classification"}]).to_csv(run / "results.csv", index=False)
    pd.DataFrame([{"horizon": 20}]).to_csv(run / "dataset_summary.csv", index=False)
    (run / "manifest.json").write_text("{}")

    discovered = discover_runs([tmp_path])
    assert len(discovered) == 1
    assert discovered[0].kind == "m8"
