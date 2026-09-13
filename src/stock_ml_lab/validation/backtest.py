from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd


StrategyMode = Literal["long_short", "long_flat"]


@dataclass(frozen=True)
class BacktestMetrics:
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    exposure: float
    annualized_turnover: float
    n_rebalances: int
    total_cost_return: float
    hit_rate_active: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class BacktestResult:
    name: str
    threshold: float
    cost_bps: float
    mode: str
    positions: pd.Series
    gross_returns: pd.Series
    costs: pd.Series
    net_returns: pd.Series
    equity_curve: pd.Series
    metrics: BacktestMetrics


def positions_from_predictions(
    predictions: pd.Series,
    *,
    threshold: float = 0.0,
    mode: StrategyMode = "long_short",
) -> pd.Series:
    if threshold < 0.0:
        raise ValueError("threshold must be non-negative.")
    pred = predictions.astype(float)

    if mode == "long_short":
        values = np.where(
            pred.to_numpy() > threshold,
            1.0,
            np.where(pred.to_numpy() < -threshold, -1.0, 0.0),
        )
    elif mode == "long_flat":
        values = np.where(pred.to_numpy() > threshold, 1.0, 0.0)
    else:
        raise ValueError("mode must be 'long_short' or 'long_flat'.")

    return pd.Series(values, index=pred.index, name="position", dtype=float)


def _performance_metrics(
    net_returns: pd.Series,
    positions: pd.Series,
    turnover: pd.Series,
    costs: pd.Series,
    *,
    annualization: int,
) -> BacktestMetrics:
    r = net_returns.to_numpy(dtype=float)
    if len(r) == 0:
        raise ValueError("Backtest requires at least one return.")
    if np.any(r <= -1.0):
        raise ValueError("Backtest contains a return <= -100%, invalidating compounding.")

    equity = np.cumprod(1.0 + r)
    total_return = float(equity[-1] - 1.0)
    annualized_return = float(equity[-1] ** (annualization / len(r)) - 1.0)

    std = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
    annualized_volatility = float(std * np.sqrt(annualization))
    sharpe = float(np.mean(r) / std * np.sqrt(annualization)) if std > 0 else float("nan")

    downside = np.minimum(r, 0.0)
    downside_deviation = float(np.sqrt(np.mean(downside**2)))
    sortino = (
        float(np.mean(r) / downside_deviation * np.sqrt(annualization))
        if downside_deviation > 0
        else float("nan")
    )

    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1.0
    max_drawdown = float(drawdown.min())

    active = positions.to_numpy(dtype=float) != 0.0
    hit_rate_active = (
        float((r[active] > 0.0).mean()) if np.any(active) else float("nan")
    )

    return BacktestMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        annualized_volatility=annualized_volatility,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_drawdown,
        exposure=float(np.mean(np.abs(positions.to_numpy(dtype=float)) > 0.0)),
        annualized_turnover=float(turnover.mean() * annualization),
        n_rebalances=int((turnover > 0.0).sum()),
        total_cost_return=float(costs.sum()),
        hit_rate_active=hit_rate_active,
    )


def backtest_positions(
    actual_returns: pd.Series,
    positions: pd.Series,
    *,
    cost_bps: float = 10.0,
    name: str = "strategy",
    threshold: float = 0.0,
    mode: str = "custom",
    annualization: int = 252,
) -> BacktestResult:
    if cost_bps < 0.0:
        raise ValueError("cost_bps must be non-negative.")
    if not actual_returns.index.equals(positions.index):
        raise ValueError("actual_returns and positions must have identical indices.")

    actual = actual_returns.astype(float)
    position = positions.astype(float)
    previous = position.shift(1, fill_value=0.0)
    turnover = (position - previous).abs().rename("turnover")
    costs = (turnover * (cost_bps / 10_000.0)).rename("cost")
    gross = (position * actual).rename("gross_return")
    net = (gross - costs).rename("net_return")
    equity = (1.0 + net).cumprod().rename("equity")

    metrics = _performance_metrics(
        net,
        position,
        turnover,
        costs,
        annualization=annualization,
    )

    return BacktestResult(
        name=name,
        threshold=threshold,
        cost_bps=cost_bps,
        mode=mode,
        positions=position,
        gross_returns=gross,
        costs=costs,
        net_returns=net,
        equity_curve=equity,
        metrics=metrics,
    )


def backtest_predictions(
    actual_returns: pd.Series,
    predictions: pd.Series,
    *,
    threshold: float = 0.0,
    cost_bps: float = 10.0,
    mode: StrategyMode = "long_short",
    name: str = "model strategy",
) -> BacktestResult:
    if not actual_returns.index.equals(predictions.index):
        raise ValueError("actual_returns and predictions must have identical indices.")
    positions = positions_from_predictions(
        predictions,
        threshold=threshold,
        mode=mode,
    )
    return backtest_positions(
        actual_returns,
        positions,
        cost_bps=cost_bps,
        name=name,
        threshold=threshold,
        mode=mode,
    )


def buy_and_hold_backtest(
    actual_returns: pd.Series,
    *,
    cost_bps: float = 10.0,
) -> BacktestResult:
    positions = pd.Series(1.0, index=actual_returns.index, name="position")
    return backtest_positions(
        actual_returns,
        positions,
        cost_bps=cost_bps,
        name="Buy & Hold",
        threshold=0.0,
        mode="long_only",
    )


def threshold_report(
    actual_returns: pd.Series,
    predictions: pd.Series,
    *,
    thresholds: tuple[float, ...],
    cost_bps: float,
    mode: StrategyMode,
) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        result = backtest_predictions(
            actual_returns,
            predictions,
            threshold=threshold,
            cost_bps=cost_bps,
            mode=mode,
            name=f"threshold={threshold:g}",
        )
        rows.append({"threshold": threshold, **result.metrics.to_dict()})
    return pd.DataFrame(rows).set_index("threshold")


def cost_sensitivity_report(
    actual_returns: pd.Series,
    predictions: pd.Series,
    *,
    threshold: float,
    costs_bps: tuple[float, ...],
    mode: StrategyMode,
) -> pd.DataFrame:
    rows = []
    for cost_bps in costs_bps:
        result = backtest_predictions(
            actual_returns,
            predictions,
            threshold=threshold,
            cost_bps=cost_bps,
            mode=mode,
            name=f"cost={cost_bps:g}bps",
        )
        rows.append({"cost_bps": cost_bps, **result.metrics.to_dict()})
    return pd.DataFrame(rows).set_index("cost_bps")
