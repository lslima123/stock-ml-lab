__all__ = [
    "DirectionDatasetBundle",
    "build_direction_dataset",
    "build_direction_dataset_from_ohlcv",
    "ClassificationMetrics",
    "classification_metrics",
    "calibration_table",
    "ClassificationComparison",
    "NestedClassificationResult",
    "run_classification_comparison",
    "CLASSIFICATION_MODELS",
    "CLASSIFICATION_DISPLAY_NAMES",
    "positions_from_probabilities",
    "backtest_probabilities",
    "probability_margin_report",
    "select_probability_margins_nested",
    "NestedProbabilityMarginResult",
    "classification_block_bootstrap",
]

_MODULES = {
    "DirectionDatasetBundle": "dataset", "build_direction_dataset": "dataset",
    "build_direction_dataset_from_ohlcv": "dataset",
    "ClassificationMetrics": "metrics", "classification_metrics": "metrics",
    "calibration_table": "metrics", "ClassificationComparison": "nested",
    "NestedClassificationResult": "nested", "run_classification_comparison": "nested",
    "CLASSIFICATION_MODELS": "search_spaces", "CLASSIFICATION_DISPLAY_NAMES": "search_spaces",
    "positions_from_probabilities": "backtest", "backtest_probabilities": "backtest",
    "probability_margin_report": "backtest", "select_probability_margins_nested": "backtest",
    "NestedProbabilityMarginResult": "backtest", "classification_block_bootstrap": "bootstrap",
}


def __getattr__(name: str):
    if name in _MODULES:
        from importlib import import_module
        return getattr(import_module(f"stock_ml_lab.classification.{_MODULES[name]}"), name)
    raise AttributeError(name)
