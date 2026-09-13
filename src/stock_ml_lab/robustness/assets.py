from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AssetSpec:
    ticker: str
    market: str
    asset_type: str


CORE_UNIVERSE: tuple[AssetSpec, ...] = (
    AssetSpec("PETR4.SA", "Brazil", "Equity"),
    AssetSpec("VALE3.SA", "Brazil", "Equity"),
    AssetSpec("ITUB4.SA", "Brazil", "Equity"),
    AssetSpec("WEGE3.SA", "Brazil", "Equity"),
    AssetSpec("BOVA11.SA", "Brazil", "ETF"),
    AssetSpec("AAPL", "United States", "Equity"),
    AssetSpec("MSFT", "United States", "Equity"),
    AssetSpec("NVDA", "United States", "Equity"),
    AssetSpec("JPM", "United States", "Equity"),
    AssetSpec("SPY", "United States", "ETF"),
)


def resolve_assets(
    *,
    universe: str = "core",
    tickers: tuple[str, ...] | None = None,
) -> tuple[AssetSpec, ...]:
    """Resolve a fixed research universe or explicit custom tickers."""
    if tickers:
        cleaned = tuple(dict.fromkeys(t.strip().upper() for t in tickers if t.strip()))
        if not cleaned:
            raise ValueError("At least one non-empty ticker is required.")
        return tuple(AssetSpec(ticker, "Custom", "Unknown") for ticker in cleaned)

    if universe != "core":
        raise ValueError("Only universe='core' is currently supported.")

    return CORE_UNIVERSE
