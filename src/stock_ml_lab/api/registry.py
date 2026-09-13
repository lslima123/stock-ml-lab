from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from stock_ml_lab.classification.models import (
    make_catboost_classifier,
    make_logistic_classifier,
    make_prior_classifier,
    make_random_forest_classifier,
    make_xgboost_classifier,
)
from stock_ml_lab.models.baseline import ZeroReturnRegressor
from stock_ml_lab.models.linear import make_ridge_pipeline
from stock_ml_lab.models.tree import make_catboost, make_random_forest, make_xgboost


@dataclass(frozen=True)
class LocalModelSpec:
    task: str
    key: str
    label: str
    hyperparameters: dict[str, Any]


REGRESSION_SPECS = {
    "zero": LocalModelSpec("regression", "zero", "Zero Return Baseline", {}),
    "ridge": LocalModelSpec("regression", "ridge", "Ridge Regression", {"alpha": 1.0}),
    "random_forest": LocalModelSpec(
        "regression",
        "random_forest",
        "Random Forest Regressor",
        {
            "n_estimators": 400,
            "max_depth": 6,
            "min_samples_leaf": 10,
            "max_features": 0.7,
        },
    ),
    "xgboost": LocalModelSpec(
        "regression",
        "xgboost",
        "XGBoost Regressor",
        {
            "n_estimators": 400,
            "max_depth": 3,
            "learning_rate": 0.03,
            "min_child_weight": 5.0,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        },
    ),
    "catboost": LocalModelSpec(
        "regression",
        "catboost",
        "CatBoost Regressor",
        {"iterations": 400, "depth": 5, "learning_rate": 0.03, "l2_leaf_reg": 3.0},
    ),
}

CLASSIFICATION_SPECS = {
    "prior": LocalModelSpec("classification", "prior", "Training Prior Baseline", {}),
    "logistic": LocalModelSpec(
        "classification", "logistic", "Logistic Regression", {"C": 1.0}
    ),
    "random_forest": LocalModelSpec(
        "classification",
        "random_forest",
        "Random Forest Classifier",
        {
            "n_estimators": 400,
            "max_depth": 5,
            "min_samples_leaf": 10,
            "max_features": 0.7,
        },
    ),
    "xgboost": LocalModelSpec(
        "classification",
        "xgboost",
        "XGBoost Classifier",
        {
            "n_estimators": 400,
            "max_depth": 3,
            "learning_rate": 0.03,
            "min_child_weight": 5.0,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
        },
    ),
    "catboost": LocalModelSpec(
        "classification",
        "catboost",
        "CatBoost Classifier",
        {"iterations": 400, "depth": 5, "learning_rate": 0.03, "l2_leaf_reg": 3.0},
    ),
}


def all_specs() -> tuple[LocalModelSpec, ...]:
    return tuple(REGRESSION_SPECS.values()) + tuple(CLASSIFICATION_SPECS.values())


def get_spec(task: str, key: str) -> LocalModelSpec:
    table = REGRESSION_SPECS if task == "regression" else CLASSIFICATION_SPECS
    try:
        return table[key]
    except KeyError as exc:
        raise ValueError(f"Unknown {task} model: {key}") from exc


def build_local_estimator(task: str, key: str, *, random_state: int = 42):
    spec = get_spec(task, key)
    p = spec.hyperparameters

    if task == "regression":
        if key == "zero":
            return ZeroReturnRegressor()
        if key == "ridge":
            return make_ridge_pipeline(alpha=float(p["alpha"]))
        if key == "random_forest":
            return make_random_forest(**p, random_state=random_state, n_jobs=-1)
        if key == "xgboost":
            return make_xgboost(**p, random_state=random_state, n_jobs=-1)
        if key == "catboost":
            return make_catboost(**p, random_seed=random_state, thread_count=-1)

    if task == "classification":
        if key == "prior":
            return make_prior_classifier()
        if key == "logistic":
            return make_logistic_classifier(C=float(p["C"]), random_state=random_state)
        if key == "random_forest":
            return make_random_forest_classifier(**p, random_state=random_state, n_jobs=-1)
        if key == "xgboost":
            return make_xgboost_classifier(**p, random_state=random_state, n_jobs=-1)
        if key == "catboost":
            return make_catboost_classifier(**p, random_seed=random_state, thread_count=-1)

    raise ValueError(f"Unsupported task/model combination: {task}/{key}")
