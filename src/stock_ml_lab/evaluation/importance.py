from __future__ import annotations

import numpy as np
import pandas as pd


def extract_tree_feature_importance(
    estimator,
    feature_names: pd.Index | list[str] | tuple[str, ...],
) -> pd.Series | None:
    """Extract normalized impurity/gain-style importance when exposed by a model.

    Random Forest, XGBoost and CatBoost all expose `feature_importances_` through
    their sklearn-compatible regressors. Values are normalized per fitted fold so
    fold averages are directly comparable.
    """
    if not hasattr(estimator, "feature_importances_"):
        return None

    values = np.asarray(estimator.feature_importances_, dtype=float).reshape(-1)
    names = list(feature_names)

    if len(values) != len(names):
        raise ValueError(
            "Feature importance length does not match the number of input features."
        )
    if not np.isfinite(values).all():
        raise ValueError("Feature importances contain non-finite values.")

    values = np.clip(values, a_min=0.0, a_max=None)
    total = float(values.sum())
    if total > 0.0:
        values = values / total

    return pd.Series(values, index=names, dtype=float, name="importance")
