from __future__ import annotations

from typing import Any
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_prior_classifier() -> DummyClassifier:
    """Probabilistic baseline that predicts the training class prior."""
    return DummyClassifier(strategy="prior")


def make_logistic_classifier(*, C: float = 1.0, random_state: int = 42) -> Pipeline:
    if C <= 0.0:
        raise ValueError("C must be positive.")
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logistic",
                LogisticRegression(
                    C=C,
                    l1_ratio=0.0,
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def make_random_forest_classifier(
    *,
    n_estimators: int = 400,
    max_depth: int = 5,
    min_samples_leaf: int = 10,
    max_features: float = 0.7,
    random_state: int = 42,
    n_jobs: int = -1,
) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=True,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def make_xgboost_classifier(
    *,
    n_estimators: int = 400,
    max_depth: int = 3,
    learning_rate: float = 0.03,
    min_child_weight: float = 5.0,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Any:
    from xgboost import XGBClassifier

    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_child_weight=min_child_weight,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        tree_method="hist",
        random_state=random_state,
        n_jobs=n_jobs,
        verbosity=0,
    )


def make_catboost_classifier(
    *,
    iterations: int = 400,
    depth: int = 5,
    learning_rate: float = 0.03,
    l2_leaf_reg: float = 3.0,
    random_seed: int = 42,
    thread_count: int = -1,
) -> Any:
    from catboost import CatBoostClassifier

    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        random_seed=random_seed,
        thread_count=thread_count,
        verbose=False,
        allow_writing_files=False,
    )
