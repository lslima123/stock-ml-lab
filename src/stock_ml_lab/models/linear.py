from __future__ import annotations

from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_ridge_pipeline(alpha: float = 1.0) -> Pipeline:
    """Create a leakage-safe StandardScaler + Ridge pipeline."""
    if alpha < 0:
        raise ValueError("Ridge alpha must be >= 0.")

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )
