from __future__ import annotations

from pathlib import Path


def frontend_dist_dir() -> Path:
    """Return the repository front-end build directory."""
    return Path(__file__).resolve().parents[3] / "frontend" / "dist"


def frontend_is_built() -> bool:
    root = frontend_dist_dir()
    return root.is_dir() and (root / "index.html").is_file()
