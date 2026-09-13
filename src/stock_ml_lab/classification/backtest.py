from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit

from stock_ml_lab.classification.dataset import DirectionDatasetBundle
from stock_ml_lab.classification.nested import NestedClassificationResult
from stock_ml_lab.classification.search_spaces import build_classification_estimator
from stock_ml_lab.validation.backtest import BacktestResult, backtest_positions


ProbabilityStrategyMode = Literal["long_short", "long_flat"]
ProbabilityThresholdObjective = Literal["sharpe", "annualized_return", "sortino"]


def positions_from_probabilities(
    probabilities: pd.Series,
    *,
    margin: float = 0.0,
    mode: ProbabilityStrategyMode = "long_short",
) -> pd.Series:
    if not 0.0 <= margin < 0.5:
        raise ValueError("margin must satisfy 0 <= margin < 0.5.")
    p = probabilities.astype(float)
    if np.any((p.to_numpy() < 0.0) | (p.to_numpy() > 1.0)):
        raise ValueError("probabilities must lie in [0, 1].")

    upper = 0.5 + margin
    lower = 0.5 - margin
    if mode == "long_short":
        values = np.where(p > upper, 1.0, np.where(p < lower, -1.0, 0.0))
    elif mode == "long_flat":
        values = np.where(p > upper, 1.0, 0.0)
    else:
        raise ValueError("mode must be 'long_short' or 'long_flat'.")
    return pd.Series(values, index=p.index, name="position", dtype=float)


def backtest_probabilities(
    actual_returns: pd.Series,
    probabilities: pd.Series,
    *,
    margin: float = 0.0,
    cost_bps: float = 10.0,
    mode: ProbabilityStrategyMode = "long_short",
    name: str = "probability strategy",
) -> BacktestResult:
    positions = positions_from_probabilities(probabilities, margin=margin, mode=mode)
    return backtest_positions(
        actual_returns.loc[positions.index],
        positions,
        cost_bps=cost_bps,
        name=name,
        threshold=margin,
        mode=f"probability-{mode}",
    )


def probability_margin_report(
    actual_returns: pd.Series,
    probabilities: pd.Series,
    *,
    margins: tuple[float, ...],
    cost_bps: float,
    mode: ProbabilityStrategyMode,
) -> pd.DataFrame:
    rows = []
    for margin in margins:
        result = backtest_probabilities(
            actual_returns,
            probabilities,
            margin=margin,
            cost_bps=cost_bps,
            mode=mode,
            name=f"margin={margin:.3f}",
        )
        rows.append({"margin": margin, **result.metrics.to_dict()})
    return pd.DataFrame(rows).set_index("margin")


@dataclass(frozen=True)
class ProbabilityMarginSelection:
    fold: int
    margin: float
    inner_objective: float


@dataclass(frozen=True)
class NestedProbabilityMarginResult:
    selections: tuple[ProbabilityMarginSelection, ...]
    positions: pd.Series
    backtest: BacktestResult

    def selection_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "fold": s.fold,
                    "margin": s.margin,
                    "inner_objective": s.inner_objective,
                }
                for s in self.selections
            ]
        ).set_index("fold")


def _positive_probability(model, X: pd.DataFrame) -> np.ndarray:
    proba = np.asarray(model.predict_proba(X), dtype=float)
    classes = np.asarray(model.classes_)
    idx = np.where(classes == 1)[0]
    if len(idx) != 1:
        raise RuntimeError("Classifier must expose class 1.")
    return proba[:, int(idx[0])]


def _inner_oof_probability(
    X: pd.DataFrame,
    y: pd.Series,
    returns: pd.Series,
    estimator,
    *,
    n_splits: int,
    gap: int,
) -> tuple[pd.Series, pd.Series]:
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    prob_parts = []
    return_parts = []
    for train_idx, valid_idx in splitter.split(X):
        model = clone(estimator)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        p = _positive_probability(model, X.iloc[valid_idx])
        prob_parts.append(pd.Series(p, index=X.index[valid_idx]))
        return_parts.append(returns.iloc[valid_idx])
    return pd.concat(prob_parts).sort_index(), pd.concat(return_parts).sort_index()


def _objective(result: BacktestResult, objective: ProbabilityThresholdObjective) -> float:
    value = float(getattr(result.metrics, objective))
    return value if np.isfinite(value) else -np.inf


def select_probability_margins_nested(
    dataset: DirectionDatasetBundle,
    tuned_result: NestedClassificationResult,
    *,
    margins: tuple[float, ...] = (0.0, 0.025, 0.05, 0.10),
    cost_bps: float = 10.0,
    mode: ProbabilityStrategyMode = "long_short",
    objective: ProbabilityThresholdObjective = "sharpe",
    random_state: int = 42,
) -> NestedProbabilityMarginResult:
    ordered = tuple(sorted(set(float(m) for m in margins)))
    if not ordered or any(m < 0.0 or m >= 0.5 for m in ordered):
        raise ValueError("Margins must be non-empty and satisfy 0 <= margin < 0.5.")

    selections = []
    position_parts = []

    for fold in tuned_result.folds:
        X_train = dataset.X.loc[fold.train_start : fold.train_end]
        y_train = dataset.y.loc[fold.train_start : fold.train_end]
        r_train = dataset.next_returns.loc[fold.train_start : fold.train_end]
        estimator = build_classification_estimator(
            tuned_result.model_key,
            fold.best_params,
            random_state=random_state + fold.fold,
        )
        inner_prob, inner_returns = _inner_oof_probability(
            X_train,
            y_train,
            r_train,
            estimator,
            n_splits=tuned_result.inner_splits,
            gap=tuned_result.gap,
        )

        best_margin = ordered[0]
        best_value = -np.inf
        for margin in ordered:
            bt = backtest_probabilities(
                inner_returns,
                inner_prob,
                margin=margin,
                cost_bps=cost_bps,
                mode=mode,
                name="inner probability-margin selection",
            )
            value = _objective(bt, objective)
            if value > best_value:
                best_value = value
                best_margin = margin

        outer_prob = tuned_result.probabilities.loc[fold.test_start : fold.test_end]
        position_parts.append(
            positions_from_probabilities(outer_prob, margin=best_margin, mode=mode)
        )
        selections.append(
            ProbabilityMarginSelection(
                fold=fold.fold,
                margin=best_margin,
                inner_objective=float(best_value),
            )
        )

    positions = pd.concat(position_parts).sort_index()
    returns = tuned_result.actual_returns.loc[positions.index]
    bt = backtest_positions(
        returns,
        positions,
        cost_bps=cost_bps,
        name=f"{tuned_result.model_name} nested probability margin",
        threshold=float("nan"),
        mode=f"nested-probability-{mode}",
    )
    return NestedProbabilityMarginResult(
        selections=tuple(selections),
        positions=positions,
        backtest=bt,
    )
