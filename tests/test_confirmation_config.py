from __future__ import annotations

import pytest

from stock_ml_lab.confirmation.config import (
    CONFIRMATION_TICKERS,
    DISCOVERY_TICKER,
    resolve_confirmation_assets,
)


def test_discovery_ticker_is_excluded_from_default_confirmation() -> None:
    assert DISCOVERY_TICKER not in CONFIRMATION_TICKERS
    assert len(CONFIRMATION_TICKERS) == 9


def test_confirmation_rejects_discovery_ticker_by_default() -> None:
    with pytest.raises(ValueError, match="discovery"):
        resolve_confirmation_assets((DISCOVERY_TICKER,))


def test_confirmation_subset_preserves_predeclared_metadata() -> None:
    assets = resolve_confirmation_assets(("AAPL", "SPY"))
    assert [asset.ticker for asset in assets] == ["AAPL", "SPY"]
    assert all(asset.market == "United States" for asset in assets)
