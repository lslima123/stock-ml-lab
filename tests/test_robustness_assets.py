from __future__ import annotations

from stock_ml_lab.robustness.assets import CORE_UNIVERSE, resolve_assets


def test_core_universe_has_brazil_and_us_assets() -> None:
    tickers = {item.ticker for item in CORE_UNIVERSE}
    assert "PETR4.SA" in tickers
    assert "SPY" in tickers
    assert len(CORE_UNIVERSE) == 10


def test_custom_tickers_are_normalized_and_deduplicated() -> None:
    assets = resolve_assets(tickers=(" aapl ", "AAPL", "petr4.sa"))
    assert [item.ticker for item in assets] == ["AAPL", "PETR4.SA"]
