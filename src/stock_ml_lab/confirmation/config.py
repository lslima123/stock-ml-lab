from __future__ import annotations

from stock_ml_lab.robustness.assets import AssetSpec, CORE_UNIVERSE


DISCOVERY_TICKER = "PETR4.SA"
LOCKED_HORIZONS = (10, 20)
LOCKED_FEATURE_SET = "legacy"
LOCKED_CLASSIFICATION_MODEL = "logistic"
LOCKED_REGRESSION_MODEL = "ridge"

# PETR4.SA was used to choose the horizon/feature/model formulation in M8.
# It is therefore excluded from the default confirmation universe.
CONFIRMATION_UNIVERSE: tuple[AssetSpec, ...] = tuple(
    asset for asset in CORE_UNIVERSE if asset.ticker != DISCOVERY_TICKER
)
CONFIRMATION_TICKERS = tuple(asset.ticker for asset in CONFIRMATION_UNIVERSE)


def resolve_confirmation_assets(
    tickers: tuple[str, ...] | None = None,
    *,
    allow_discovery_asset: bool = False,
) -> tuple[AssetSpec, ...]:
    if tickers is None:
        return CONFIRMATION_UNIVERSE

    requested = tuple(dict.fromkeys(t.strip().upper() for t in tickers if t.strip()))
    if not requested:
        raise ValueError("At least one non-empty confirmation ticker is required.")

    known = {asset.ticker: asset for asset in CORE_UNIVERSE}
    unknown = [ticker for ticker in requested if ticker not in known]
    if unknown:
        raise ValueError(
            "M8.1 confirmation only accepts the predeclared M6 universe; "
            f"unknown tickers: {unknown}."
        )

    if DISCOVERY_TICKER in requested and not allow_discovery_asset:
        raise ValueError(
            f"{DISCOVERY_TICKER} is the discovery series and is excluded from "
            "locked confirmation. Pass allow_discovery_asset=True only for "
            "diagnostic/reproduction work, not confirmation."
        )

    return tuple(known[ticker] for ticker in requested)
