from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import os

import numpy as np
import pandas as pd


_NORMALIZED_FILES = {
    "discovery": "normalized_m8_results.csv",
    "confirmation_assets": "normalized_m81_asset_results.csv",
    "confirmation_summary": "normalized_m81_candidate_summary.csv",
    "m7": "normalized_m7_summary.csv",
    "m6": "normalized_m6_asset_results.csv",
    "scope_summary": "normalized_m13_scope_summary.csv",
    "scope_assets": "normalized_m13_scope_assets.csv",
    "scope_confirmation_summary": "normalized_m14_scope_confirmation_summary.csv",
    "scope_confirmation_assets": "normalized_m14_scope_confirmation_assets.csv",
}


def _clean_records(frame: pd.DataFrame) -> list[dict]:
    if frame.empty:
        return []
    cleaned = frame.astype(object).where(pd.notna(frame), None)
    return cleaned.to_dict(orient="records")


@dataclass
class AnalyticsRepository:
    root: Path

    @classmethod
    def from_environment(cls) -> "AnalyticsRepository":
        configured = os.getenv("STOCK_ML_LAB_ANALYTICS_DIR")
        if configured:
            return cls(Path(configured).expanduser().resolve())
        cwd_candidate = Path.cwd() / "examples" / "api_analytics"
        if cwd_candidate.exists():
            return cls(cwd_candidate.resolve())
        package_root = Path(__file__).resolve().parents[3]
        return cls(package_root / "examples" / "api_analytics")

    @property
    def source_label(self) -> str:
        return str(self.root)

    def _read(self, key: str) -> pd.DataFrame:
        filename = _NORMALIZED_FILES[key]
        path = self.root / filename
        if not path.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    def discovery(self, *, task: str | None = None, horizon: int | None = None) -> list[dict]:
        frame = self._read("discovery")
        if task and "task" in frame:
            frame = frame[frame["task"] == task]
        if horizon is not None and "horizon" in frame:
            frame = frame[pd.to_numeric(frame["horizon"], errors="coerce") == horizon]
        return _clean_records(frame)

    def confirmation_assets(
        self, *, task: str | None = None, horizon: int | None = None
    ) -> list[dict]:
        frame = self._read("confirmation_assets")
        if task and "task" in frame:
            frame = frame[frame["task"] == task]
        if horizon is not None and "horizon" in frame:
            frame = frame[pd.to_numeric(frame["horizon"], errors="coerce") == horizon]
        return _clean_records(frame)

    def confirmation_summary(self) -> list[dict]:
        return _clean_records(self._read("confirmation_summary"))

    def scope_benchmark(
        self, *, task: str | None = None, horizon: int | None = None
    ) -> list[dict]:
        frame = self._read("scope_summary")
        if task and "task" in frame:
            frame = frame[frame["task"] == task]
        if horizon is not None and "horizon" in frame:
            frame = frame[pd.to_numeric(frame["horizon"], errors="coerce") == horizon]
        return _clean_records(frame)

    def scope_assets(
        self, *, task: str | None = None, horizon: int | None = None
    ) -> list[dict]:
        frame = self._read("scope_assets")
        if task and "task" in frame:
            frame = frame[frame["task"] == task]
        if horizon is not None and "horizon" in frame:
            frame = frame[pd.to_numeric(frame["horizon"], errors="coerce") == horizon]
        return _clean_records(frame)

    def scope_confirmation(
        self,
        *,
        task: str | None = None,
        horizon: int | None = None,
        level: str = "summary",
    ) -> list[dict]:
        key = "scope_confirmation_summary" if level == "summary" else "scope_confirmation_assets"
        frame = self._read(key)
        if task and "task" in frame:
            frame = frame[frame["task"] == task]
        if horizon is not None and "horizon" in frame:
            frame = frame[pd.to_numeric(frame["horizon"], errors="coerce") == horizon]
        return _clean_records(frame)

    def findings(self) -> list[str]:
        manifest = self.root / "report_manifest.json"
        if manifest.exists():
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
                findings = payload.get("findings", [])
                if isinstance(findings, list):
                    return [str(item) for item in findings]
            except (OSError, json.JSONDecodeError):
                pass

        return [
            "PETR4.SA showed stronger exploratory 10d/20d signal than 1d prediction.",
            "The locked nine-asset confirmation stage did not reproduce a robust generalizable edge.",
            "No locked asset-level primary result survived HAC/FDR significance at 5% in M8.1.",
            "M13 favored the pooled global scope in 71 of 80 task/horizon/asset comparisons; the h=5 result was the most stable post-hoc candidate.",
            "M14 confirmed the h=5 pooled Ridge scope effect: +1.72% OOS RMSE improvement, HAC/FDR q=0.0001, and a fully positive 95% block-bootstrap interval.",
            "M14 did not confirm the Logistic scope effect: its +0.41% Log Loss improvement had q=0.1577 and a bootstrap interval crossing zero.",
        ]
