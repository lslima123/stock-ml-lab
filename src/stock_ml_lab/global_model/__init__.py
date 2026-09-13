__all__ = [
    "GlobalArtifactStore",
    "GlobalPanelBundle",
    "build_global_panel",
    "build_global_panel_from_ohlcv",
    "GLOBAL_CLASSIFICATION_MODEL",
    "GLOBAL_REGRESSION_MODEL",
    "train_global_artifacts",
]

_MODULES = {
    "GlobalArtifactStore": "artifacts",
    "GlobalPanelBundle": "dataset", "build_global_panel": "dataset",
    "build_global_panel_from_ohlcv": "dataset",
    "GLOBAL_CLASSIFICATION_MODEL": "training", "GLOBAL_REGRESSION_MODEL": "training",
    "train_global_artifacts": "training",
}


def __getattr__(name: str):
    if name in _MODULES:
        from importlib import import_module
        return getattr(import_module(f"stock_ml_lab.global_model.{_MODULES[name]}"), name)
    raise AttributeError(name)
