__all__ = [
    "PersistenceRegressor",
    "ZeroReturnRegressor",
    "AutoARIMAConfig",
    "build_var_frame",
    "fit_arima",
    "fit_var",
    "make_ridge_pipeline",
    "make_random_forest",
    "make_xgboost",
    "make_catboost",
]

_MODULES = {
    "PersistenceRegressor": "baseline", "ZeroReturnRegressor": "baseline",
    "AutoARIMAConfig": "classical", "build_var_frame": "classical",
    "fit_arima": "classical", "fit_var": "classical",
    "make_ridge_pipeline": "linear", "make_random_forest": "tree",
    "make_xgboost": "tree", "make_catboost": "tree",
}


def __getattr__(name: str):
    if name in _MODULES:
        from importlib import import_module
        return getattr(import_module(f"stock_ml_lab.models.{_MODULES[name]}"), name)
    raise AttributeError(name)
