from __future__ import annotations

import numpy as np
import pandas as pd

from stock_ml_lab.dataset import DatasetBundle
from stock_ml_lab.tuning.nested import run_nested_tuning
from stock_ml_lab.validation.threshold_selection import select_thresholds_nested


def make_dataset(n: int = 180) -> DatasetBundle:
    index = pd.bdate_range("2023-01-02", periods=n)
    rng = np.random.default_rng(17)
    X = pd.DataFrame(
        {
            "return_1d": rng.normal(0.0, 0.01, n),
            "return_lag_1": rng.normal(0.0, 0.01, n),
            "ma_10_distance": rng.normal(0.0, 0.03, n),
        },
        index=index,
    )
    y = pd.Series(
        0.25 * X["return_1d"].to_numpy() + rng.normal(0.0, 0.007, n),
        index=index,
        name="next_return",
    )
    frame = X.copy()
    frame["next_return"] = y
    return DatasetBundle(
        ticker="TEST",
        X=X,
        y=y,
        frame=frame,
        feature_names=tuple(X.columns),
        target_name="next_return",
    )


def test_nested_threshold_selection_uses_grid_and_outer_index() -> None:
    dataset = make_dataset()
    tuned = run_nested_tuning(
        X=dataset.X,
        y=dataset.y,
        model_key="ridge",
        outer_splits=3,
        inner_splits=2,
        gap=1,
        test_size=30,
        n_trials=2,
        random_state=8,
    )

    result = select_thresholds_nested(
        dataset,
        tuned,
        thresholds=(0.0, 0.002, 0.005),
        cost_bps=5.0,
        mode="long_short",
        objective="sharpe",
        random_state=8,
    )

    assert len(result.selections) == 3
    assert {item.threshold for item in result.selections}.issubset({0.0, 0.002, 0.005})
    assert result.positions.index.equals(tuned.predictions.index)
    assert result.backtest.net_returns.index.equals(tuned.actuals.index)
