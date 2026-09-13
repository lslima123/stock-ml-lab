from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit

from stock_ml_lab.dataset import DatasetBundle
from stock_ml_lab.tuning.nested import NestedCVResult
from stock_ml_lab.tuning.search_spaces import build_estimator
from stock_ml_lab.validation.backtest import (
    BacktestResult,
    StrategyMode,
    backtest_positions,
    backtest_predictions,
    positions_from_predictions,
)


ThresholdObjective = Literal["sharpe", "annualized_return", "sortino"]


@dataclass(frozen=True)
class ThresholdFoldSelection:
    fold: int
    threshold: float
    inner_objective: float


@dataclass(frozen=True)
class NestedThresholdResult:
    selections: tuple[ThresholdFoldSelection, ...]
    positions: pd.Series
    backtest: BacktestResult

    def selection_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "fold": item.fold,
                    "threshold": item.threshold,
                    "inner_objective": item.inner_objective,
                }
                for item in self.selections
            ]
        ).set_index("fold")


def _inner_oof_predictions(
    X: pd.DataFrame,
    y: pd.Series,
    estimator,
    *,
    n_splits: int,
    gap: int,
) -> tuple[pd.Series, pd.Series]:
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    pred_parts: list[pd.Series] = []
    actual_parts: list[pd.Series] = []

    for train_idx, valid_idx in splitter.split(X):
        model = clone(estimator)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        pred = np.asarray(model.predict(X.iloc[valid_idx]), dtype=float).reshape(-1)
        pred_parts.append(pd.Series(pred, index=X.index[valid_idx]))
        actual_parts.append(y.iloc[valid_idx])

    predictions = pd.concat(pred_parts).sort_index()
    actuals = pd.concat(actual_parts).sort_index()
    return predictions, actuals


def _objective_value(result: BacktestResult, objective: ThresholdObjective) -> float:
    value = getattr(result.metrics, objective)
    value = float(value)
    if not np.isfinite(value):
        return -np.inf
    return value


def select_thresholds_nested(
    dataset: DatasetBundle,
    tuned_result: NestedCVResult,
    *,
    thresholds: tuple[float, ...] = (0.0, 0.001, 0.0025, 0.005),
    cost_bps: float = 10.0,
    mode: StrategyMode = "long_short",
    objective: ThresholdObjective = "sharpe",
    random_state: int = 42,
) -> NestedThresholdResult:
    """Choose a trading threshold using outer-training data only.

    Model hyperparameters have already been selected in the nested tuning stage.
    For every outer fold, we rebuild the selected model and generate inner OOF
    forecasts within the outer-training block. Threshold candidates are scored
    on those inner OOF forecasts. The selected threshold is then applied to the
    untouched outer-test predictions that already exist in `tuned_result`.
    """
    if not thresholds:
        raise ValueError("At least one threshold is required.")
    if any(threshold < 0 for threshold in thresholds):
        raise ValueError("Thresholds must be non-negative.")

    ordered_thresholds = tuple(sorted(set(float(t) for t in thresholds)))
    selections: list[ThresholdFoldSelection] = []
    outer_position_parts: list[pd.Series] = []

    for fold in tuned_result.folds:
        X_train = dataset.X.loc[fold.train_start : fold.train_end]
        y_train = dataset.y.loc[fold.train_start : fold.train_end]

        estimator = build_estimator(
            tuned_result.model_key,
            fold.best_params,
            random_state=random_state + fold.fold,
        )
        inner_pred, inner_actual = _inner_oof_predictions(
            X_train,
            y_train,
            estimator,
            n_splits=tuned_result.inner_splits,
            gap=tuned_result.gap,
        )

        best_threshold = ordered_thresholds[0]
        best_value = -np.inf
        for threshold in ordered_thresholds:
            bt = backtest_predictions(
                inner_actual,
                inner_pred,
                threshold=threshold,
                cost_bps=cost_bps,
                mode=mode,
                name="inner threshold selection",
            )
            value = _objective_value(bt, objective)
            if value > best_value:
                best_value = value
                best_threshold = threshold

        outer_pred = tuned_result.predictions.loc[fold.test_start : fold.test_end]
        outer_positions = positions_from_predictions(
            outer_pred,
            threshold=best_threshold,
            mode=mode,
        )
        outer_position_parts.append(outer_positions)
        selections.append(
            ThresholdFoldSelection(
                fold=fold.fold,
                threshold=best_threshold,
                inner_objective=float(best_value),
            )
        )

    positions = pd.concat(outer_position_parts).sort_index()
    actuals = tuned_result.actuals.loc[positions.index]
    backtest = backtest_positions(
        actuals,
        positions,
        cost_bps=cost_bps,
        name=f"{tuned_result.model_name} nested-threshold",
        threshold=float("nan"),
        mode=f"nested-{mode}",
    )

    return NestedThresholdResult(
        selections=tuple(selections),
        positions=positions,
        backtest=backtest,
    )
