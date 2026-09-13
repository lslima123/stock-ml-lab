__all__ = [
    "RegressionMetrics",
    "regression_metrics",
    "CVResult",
    "FoldResult",
    "evaluate_time_series_cv",
    "evaluate_arima_cv",
    "evaluate_var_cv",
]

_MODULES = {
    "RegressionMetrics": "metrics", "regression_metrics": "metrics",
    "CVResult": "validation", "FoldResult": "validation", "evaluate_time_series_cv": "validation",
    "evaluate_arima_cv": "classical_validation", "evaluate_var_cv": "classical_validation",
}


def __getattr__(name: str):
    if name in _MODULES:
        from importlib import import_module
        return getattr(import_module(f"stock_ml_lab.evaluation.{_MODULES[name]}"), name)
    raise AttributeError(name)
