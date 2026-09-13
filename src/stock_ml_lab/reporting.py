from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return slug.strip("_") or "run"


def unique_run_directory(
    *,
    base: str | Path,
    experiment: str,
    label: str,
    timestamp: datetime | None = None,
) -> Path:
    """Create a unique report directory so experiments never overwrite silently."""
    ts = (timestamp or datetime.now()).strftime("%Y%m%dT%H%M%S")
    root = Path(base) / slugify(experiment)
    stem = f"{ts}_{slugify(label)}"
    candidate = root / stem
    suffix = 2
    while candidate.exists():
        candidate = root / f"{stem}_{suffix}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate
