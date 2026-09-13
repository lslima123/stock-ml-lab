from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

import joblib


GLOBAL_TASK_MODELS = {
    "regression": "ridge",
    "classification": "logistic",
}
GLOBAL_HORIZONS = (1, 5, 10, 20)


@dataclass(frozen=True)
class LoadedGlobalArtifact:
    estimator: Any
    metadata: dict[str, Any]
    path: Path


class GlobalArtifactStore:
    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()

    @classmethod
    def from_environment(cls) -> "GlobalArtifactStore":
        root = os.getenv("STOCK_ML_LAB_GLOBAL_MODEL_DIR", "artifacts/global")
        return cls(root)

    def artifact_path(self, task: str, model: str, horizon: int) -> Path:
        return self.root / f"global_{task}_{model}_h{int(horizon)}.joblib"

    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def exists(self, task: str, model: str, horizon: int) -> bool:
        return self.artifact_path(task, model, horizon).is_file()

    def is_complete(self) -> bool:
        return all(
            self.exists(task, model, horizon)
            for task, model in GLOBAL_TASK_MODELS.items()
            for horizon in GLOBAL_HORIZONS
        )

    def available_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for task, model in GLOBAL_TASK_MODELS.items():
            for horizon in GLOBAL_HORIZONS:
                path = self.artifact_path(task, model, horizon)
                records.append(
                    {
                        "task": task,
                        "model": model,
                        "horizon": horizon,
                        "available": path.is_file(),
                        "path": str(path),
                    }
                )
        return records

    def save(
        self,
        *,
        task: str,
        model: str,
        horizon: int,
        estimator: Any,
        metadata: dict[str, Any],
    ) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.artifact_path(task, model, horizon)
        payload = {
            "estimator": estimator,
            "metadata": {
                **metadata,
                "task": task,
                "model": model,
                "horizon": int(horizon),
                "artifact_version": 1,
                "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        }
        joblib.dump(payload, path)
        return path

    def load(self, task: str, model: str, horizon: int) -> LoadedGlobalArtifact:
        path = self.artifact_path(task, model, horizon)
        if not path.is_file():
            raise FileNotFoundError(
                f"Global artifact not found for {task}/{model}/{horizon}d: {path}"
            )
        payload = joblib.load(path)
        if not isinstance(payload, dict) or "estimator" not in payload or "metadata" not in payload:
            raise ValueError(f"Invalid global artifact payload: {path}")
        metadata = dict(payload["metadata"])
        if metadata.get("task") != task or metadata.get("model") != model:
            raise ValueError(f"Global artifact metadata mismatch: {path}")
        if int(metadata.get("horizon", -1)) != int(horizon):
            raise ValueError(f"Global artifact horizon mismatch: {path}")
        return LoadedGlobalArtifact(
            estimator=payload["estimator"],
            metadata=metadata,
            path=path,
        )

    def write_manifest(self, payload: dict[str, Any]) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.manifest_path()
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        return path
