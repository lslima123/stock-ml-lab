from __future__ import annotations

from typing import Any

from stock_ml_lab.classification.models import (
    make_catboost_classifier,
    make_logistic_classifier,
    make_random_forest_classifier,
    make_xgboost_classifier,
)


CLASSIFICATION_MODELS = ("logistic", "random_forest", "xgboost", "catboost")
CLASSIFICATION_DISPLAY_NAMES = {
    "logistic": "Tuned Logistic Regression",
    "random_forest": "Tuned Random Forest Classifier",
    "xgboost": "Tuned XGBoost Classifier",
    "catboost": "Tuned CatBoost Classifier",
}


def sample_classification_hyperparameters(trial, model_key: str) -> dict[str, Any]:
    if model_key == "logistic":
        return {"C": trial.suggest_float("C", 1e-4, 1e3, log=True)}

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

    raise ValueError(f"Unknown classification model: {model_key!r}")


def build_classification_estimator(
    model_key: str,
    params: dict[str, Any],
    *,
    random_state: int = 42,
):
    if model_key == "logistic":
        return make_logistic_classifier(C=float(params["C"]), random_state=random_state)
    if model_key == "random_forest":
        return make_random_forest_classifier(
            **params, random_state=random_state, n_jobs=-1
        )
    if model_key == "xgboost":
        return make_xgboost_classifier(**params, random_state=random_state, n_jobs=-1)
    if model_key == "catboost":
        return make_catboost_classifier(
            **params, random_seed=random_state, thread_count=-1
        )
    raise ValueError(f"Unknown classification model: {model_key!r}")
