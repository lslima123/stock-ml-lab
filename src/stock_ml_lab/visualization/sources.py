from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import pandas as pd


@dataclass(frozen=True)
class LoadedRun:
    kind: str
    path: Path
    manifest: dict
    tables: dict[str, pd.DataFrame]


def _read_manifest(path: Path) -> dict:
    manifest_path = path / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _classify_run(path: Path, manifest: dict) -> str | None:
    names = {p.name for p in path.iterdir() if p.is_file()}

    # Use file contracts first. They are more reliable than historical manifests.
    if {"asset_results.csv", "candidate_summary.csv", "phase_results.csv"}.issubset(names):
        return "m8_1"
    if "results.csv" in names and "dataset_summary.csv" in names:
        return "m8"
    if {"summary.csv", "predictions.csv", "calibration.csv"}.issubset(names):
        return "m7"
    if {"asset_results.csv", "model_summary.csv", "fold_results.csv"}.issubset(names):
        return "m6"

    milestone = str(manifest.get("milestone", "")).strip().lower()
    if milestone in {"6", "m6"}:
        return "m6"
    if milestone in {"7", "m7"}:
        return "m7"
    if milestone in {"8", "m8"}:
        return "m8"
    if milestone in {"8.1", "m8.1", "m8_1"}:
        return "m8_1"
    return None


_TABLES = {
    "m6": (
        "asset_results.csv",
        "model_summary.csv",
        "fold_results.csv",
        "failures.csv",
    ),
    "m7": (
        "summary.csv",
        "fold_results.csv",
        "predictions.csv",
        "bootstrap_intervals.csv",
        "calibration.csv",
        "backtest.csv",
        "fixed_probability_margins.csv",
        "nested_probability_margins.csv",
        "feature_importance.csv",
    ),
    "m8": (
        "results.csv",
        "fold_results.csv",
        "feature_importance.csv",
        "dataset_summary.csv",
        "failures.csv",
    ),
    "m8_1": (
        "asset_results.csv",
        "candidate_summary.csv",
        "phase_results.csv",
        "fold_results.csv",
        "dataset_summary.csv",
        "failures.csv",
    ),
}


def load_run(path: str | Path) -> LoadedRun:
    root = Path(path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Report run directory does not exist: {root}")

    manifest = _read_manifest(root)
    kind = _classify_run(root, manifest)
    if kind is None:
        raise ValueError(
            f"Could not identify report schema in {root}. "
            "Expected an M6, M7, M8 or M8.1 run directory."
        )

    tables: dict[str, pd.DataFrame] = {}
    for filename in _TABLES[kind]:
        file_path = root / filename
        if file_path.exists():
            try:
                tables[filename[:-4]] = pd.read_csv(file_path)
            except pd.errors.EmptyDataError:
                tables[filename[:-4]] = pd.DataFrame()

    return LoadedRun(kind=kind, path=root, manifest=manifest, tables=tables)


def discover_runs(sources: list[str | Path]) -> list[LoadedRun]:
    """Discover report run directories from explicit paths or parent directories.

    A supplied path can be a single run directory or a parent directory. Parent
    directories are searched recursively for manifest.json files and known CSV
    schemas. Results are de-duplicated by resolved directory.
    """
    candidates: dict[Path, None] = {}

    for source in sources:
        root = Path(source).expanduser().resolve()
        if not root.exists():
            raise ValueError(f"Source path does not exist: {root}")

        if root.is_file():
            raise ValueError(f"Source must be a directory: {root}")

        manifest = _read_manifest(root)
        if _classify_run(root, manifest) is not None:
            candidates[root] = None
            continue

        # Prefer manifest-bearing run directories, then also catch older runs
        # that may not have a manifest.
        for manifest_path in root.rglob("manifest.json"):
            run_dir = manifest_path.parent
            if _classify_run(run_dir, _read_manifest(run_dir)) is not None:
                candidates[run_dir] = None

        for filename in ("candidate_summary.csv", "results.csv", "summary.csv", "model_summary.csv"):
            for csv_path in root.rglob(filename):
                run_dir = csv_path.parent
                if _classify_run(run_dir, _read_manifest(run_dir)) is not None:
                    candidates[run_dir] = None

    return [load_run(path) for path in sorted(candidates)]
