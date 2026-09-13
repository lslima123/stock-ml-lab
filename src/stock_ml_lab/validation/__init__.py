__all__ = [
    "BacktestMetrics",
    "BacktestResult",
    "backtest_predictions",
    "buy_and_hold_backtest",
    "cost_sensitivity_report",
    "threshold_report",
    "BootstrapInterval",
    "forecast_block_bootstrap",
    "DirectionalTest",
    "ForecastComparisonTest",
    "diebold_mariano_vs_benchmark",
    "pesaran_timmermann_test",
    "NestedThresholdResult",
    "select_thresholds_nested",
]

_MODULES = {
    "BacktestMetrics": "backtest", "BacktestResult": "backtest",
    "backtest_predictions": "backtest", "buy_and_hold_backtest": "backtest",
    "cost_sensitivity_report": "backtest", "threshold_report": "backtest",
    "BootstrapInterval": "bootstrap", "forecast_block_bootstrap": "bootstrap",
    "DirectionalTest": "forecast_tests", "ForecastComparisonTest": "forecast_tests",
    "diebold_mariano_vs_benchmark": "forecast_tests", "pesaran_timmermann_test": "forecast_tests",
    "NestedThresholdResult": "threshold_selection", "select_thresholds_nested": "threshold_selection",
}


def __getattr__(name: str):
    if name in _MODULES:
        from importlib import import_module
        return getattr(import_module(f"stock_ml_lab.validation.{_MODULES[name]}"), name)
    raise AttributeError(name)
