from __future__ import annotations

from dataclasses import asdict, dataclass


TRAINING_UNIVERSE = (
    "PETR4.SA", "VALE3.SA", "ITUB4.SA", "WEGE3.SA", "BOVA11.SA",
    "AAPL", "MSFT", "NVDA", "JPM", "SPY",
)

CONFIRMATION_UNIVERSE = (
    "BBAS3.SA", "ABEV3.SA", "AXIA3.SA", "SUZB3.SA", "RENT3.SA",
    "GOOGL", "AMZN", "META", "XOM", "UNH",
)


@dataclass(frozen=True)
class ScopeConfirmationConfig:
    milestone: str = "M14"
    status: str = "locked_independent_scope_confirmation"
    horizon: int = 5
    feature_set: str = "legacy"
    start: str = "2018-01-01"
    end_exclusive: str = "2026-09-03"
    n_splits: int = 5
    test_size_dates: int = 252
    hac_lags: int = 4
    bootstrap_block_length_dates: int = 5
    bootstrap_repetitions: int = 10_000
    random_state: int = 20260903
    training_universe: tuple[str, ...] = TRAINING_UNIVERSE
    confirmation_universe: tuple[str, ...] = CONFIRMATION_UNIVERSE
    protocol_amendments: tuple[str, ...] = (
        "2026-09-08: provider identifier ELET3.SA corrected to AXIA3.SA after the "
        "official Eletrobras/AXIA ticker change effective 2025-11-10. This is the "
        "same ordinary-share issuer, and the locked preflight aborted before any model evaluation.",
    )

    def validate(self) -> None:
        if self.horizon != 5 or self.feature_set != "legacy":
            raise ValueError("M14 is locked to horizon=5 and feature_set='legacy'.")
        if set(self.training_universe).intersection(self.confirmation_universe):
            raise ValueError("Training and confirmation universes must be disjoint.")
        if len(set(self.training_universe)) != len(self.training_universe):
            raise ValueError("Training universe contains duplicates.")
        if len(set(self.confirmation_universe)) != len(self.confirmation_universe):
            raise ValueError("Confirmation universe contains duplicates.")
        if self.hac_lags < self.horizon - 1:
            raise ValueError("HAC lags must be at least horizon-1.")
        if self.bootstrap_block_length_dates < self.horizon:
            raise ValueError("Bootstrap block length must be at least the horizon.")
        if self.bootstrap_repetitions < 1000:
            raise ValueError("Locked confirmation requires at least 1,000 bootstrap repetitions.")

    def as_dict(self) -> dict:
        return asdict(self)


def locked_config() -> ScopeConfirmationConfig:
    config = ScopeConfirmationConfig()
    config.validate()
    return config
