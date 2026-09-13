from __future__ import annotations

from typing import Any
from sklearn.ensemble import RandomForestRegressor


def make_random_forest(
    *,
    n_estimators: int = 400,
    max_depth: int = 6,
    min_samples_leaf: int = 10,
    max_features: float = 0.7,
    random_state: int = 42,
    n_jobs: int = -1,
) -> RandomForestRegressor:
    """Create a deliberately regularized Random Forest baseline.

    Hyperparameters are fixed in Milestone 3. They are not the result of tuning;
    the goal is to test whether nonlinear tree ensembles add signal under the
    same temporal protocol used by Ridge and the naive baselines.
    """
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        bootstrap=True,
        random_state=random_state,
        n_jobs=n_jobs,
    )


def make_xgboost(
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
    """Create a controlled XGBoost regressor without hyperparameter tuning."""
    from xgboost import XGBRegressor

    return XGBRegressor(
        objective="reg:squarederror",
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


def make_catboost(
    *,
    iterations: int = 400,
    depth: int = 5,
    learning_rate: float = 0.03,
    l2_leaf_reg: float = 3.0,
    random_seed: int = 42,
    thread_count: int = -1,
) -> Any:
    """Create a controlled CatBoost regressor without hyperparameter tuning."""
    from catboost import CatBoostRegressor

    return CatBoostRegressor(
        loss_function="RMSE",
        iterations=iterations,
        depth=depth,
        learning_rate=learning_rate,
        l2_leaf_reg=l2_leaf_reg,
        random_seed=random_seed,
        thread_count=thread_count,
        verbose=False,
        allow_writing_files=False,
    )
