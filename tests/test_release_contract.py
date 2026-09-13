from __future__ import annotations

import csv
import json
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = "0.15.0"
M14_PROTOCOL_SHA256 = "27a740273531dd8b9bf1d0b1604a03c4f6ad37a4f56f1e88433a3c402c50bac3"


def test_release_versions_are_synchronized() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    package = json.loads((ROOT / "frontend/package.json").read_text(encoding="utf-8"))
    package_lock = json.loads(
        (ROOT / "frontend/package-lock.json").read_text(encoding="utf-8")
    )
    api_source = (ROOT / "src/stock_ml_lab/api/app.py").read_text(encoding="utf-8")

    assert pyproject["project"]["version"] == VERSION
    assert package["version"] == VERSION
    assert package_lock["version"] == VERSION
    assert package_lock["packages"][""]["version"] == VERSION
    assert f'return "{VERSION}"' in api_source


def test_canonical_m14_protocol_is_preserved() -> None:
    manifest = json.loads(
        (ROOT / "examples/m14_snapshot/manifest.json").read_text(encoding="utf-8")
    )
    locked = json.loads(
        (ROOT / "configs/m14_locked_scope.json").read_text(encoding="utf-8")
    )

    assert manifest["protocol_sha256"] == M14_PROTOCOL_SHA256
    assert manifest["horizon"] == locked["horizon"] == 5
    assert manifest["feature_set"] == locked["feature_set"] == "legacy"
    assert set(manifest["training_universe"]).isdisjoint(
        manifest["confirmation_universe"]
    )
    assert (
        manifest["confirmation_universe"]
        == locked["independent_confirmation_universe"]
    )


def test_canonical_m14_decisions_are_preserved() -> None:
    with (ROOT / "examples/m14_snapshot/primary_results.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = {row["task"]: row for row in csv.DictReader(handle)}

    regression = rows["regression"]
    classification = rows["classification"]

    assert regression["confirmed"] == "True"
    assert float(regression["primary_hac_qvalue"]) < 0.05
    assert float(regression["bootstrap_improvement_ci_low_pct"]) > 0.0

    assert classification["confirmed"] == "False"
    assert float(classification["primary_hac_qvalue"]) >= 0.05
    assert float(classification["bootstrap_improvement_ci_low_pct"]) < 0.0
