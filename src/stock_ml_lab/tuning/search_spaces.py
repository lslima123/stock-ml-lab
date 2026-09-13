from __future__ import annotations

from typing import Any

from stock_ml_lab.models.linear import make_ridge_pipeline
from stock_ml_lab.models.tree import make_catboost, make_random_forest, make_xgboost


TUNABLE_MODELS = ("ridge", "random_forest", "xgboost", "catboost")

DISPLAY_NAMES = {
    "ridge": "Tuned Ridge",
    "random_forest": "Tuned Random Forest",
    "xgboost": "Tuned XGBoost",
    "catboost": "Tuned CatBoost",
}


def sample_hyperparameters(trial, model_key: str) -> dict[str, Any]:
    """Sample a deliberately bounded search space for Milestone 4."""
    if model_key == "ridge":
        return {
            "alpha": trial.suggest_float("alpha", 1e-4, 1e3, log=True),
        }

    if model_key == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 700, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 3, 30),
            "max_features": trial.suggest_float("max_features", 0.4, 1.0, step=0.1),
        }

    if model_key == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 700, step=100),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 15.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0, step=0.1),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.6, 1.0, step=0.1
            ),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30.0, log=True),
        }

    if model_key == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 200, 700, step=100),
            "depth": trial.suggest_int("depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 30.0, log=True),
        }

    raise ValueError(f"Unknown tunable model: {model_key!r}")


def build_estimator(
    model_key: str,
    params: dict[str, Any],
    *,
    random_state: int = 42,
):
    """Build a model from sampled parameters while freezing reproducibility knobs."""
    if model_key == "ridge":
        return make_ridge_pipeline(alpha=float(params["alpha"]))

    if model_key == "random_forest":
        return make_random_forest(
            **params,
            random_state=random_state,
            n_jobs=-1,
        )

    if model_key == "xgboost":
        return make_xgboost(
            **params,
            random_state=random_state,
            n_jobs=-1,
        )

    if model_key == "catboost":
        return make_catboost(
            **params,
            random_seed=random_state,
            thread_count=-1,
        )

    raise ValueError(f"Unknown tunable model: {model_key!r}")
