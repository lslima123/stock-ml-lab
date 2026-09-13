__all__ = [
    "AssetSpec",
    "CORE_UNIVERSE",
    "resolve_assets",
    "CrossAssetRun",
    "aggregate_cross_asset_results",
    "run_cross_asset_study",
    "benjamini_hochberg",
    "exact_sign_test_greater",
]


def __getattr__(name: str):
    if name in {"AssetSpec", "CORE_UNIVERSE", "resolve_assets"}:
        from stock_ml_lab.robustness import assets
        return getattr(assets, name)
    if name in {"CrossAssetRun", "aggregate_cross_asset_results", "run_cross_asset_study"}:
        from stock_ml_lab.robustness import cross_asset
        return getattr(cross_asset, name)
    if name in {"benjamini_hochberg", "exact_sign_test_greater"}:
        from stock_ml_lab.robustness import multiple_testing
        return getattr(multiple_testing, name)
    raise AttributeError(name)
