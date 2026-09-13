from __future__ import annotations

import numpy as np
import pandas as pd

from stock_ml_lab.dataset import DatasetBundle
from stock_ml_lab.experiment import ComparisonResult, run_model_comparison


def make_dataset(n: int = 140) -> DatasetBundle:
    index = pd.bdate_range("2024-01-01", periods=n)
    rng = np.random.default_rng(22)
    return_1d = rng.normal(0, 0.01, n)
    X = pd.DataFrame(
        {
            "return_1d": return_1d,
            "return_lag_1": np.roll(return_1d, 1),
            "return_lag_2": np.roll(return_1d, 2),
        },
        index=index,
    )
    y = pd.Series(rng.normal(0, 0.01, n), index=index, name="next_return")
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


def test_comparison_uses_identical_temporal_protocol() -> None:
    dataset = make_dataset()
    comparison = run_model_comparison(
        dataset,
        n_splits=3,
        gap=1,
        models=("zero", "persistence", "ridge"),
    )

    comparison.assert_common_protocol()
    indices = [result.predictions.index for result in comparison.results]
    assert all(index.equals(indices[0]) for index in indices[1:])


def test_summary_reports_improvement_relative_to_zero() -> None:
    dataset = make_dataset()
    comparison = run_model_comparison(
        dataset,
        n_splits=3,
        gap=1,
        models=("zero", "ridge"),
    )
    summary = comparison.summary_frame()

    assert "rmse_improvement_vs_zero_pct" in summary.columns
    assert "mae_improvement_vs_zero_pct" in summary.columns
    assert np.isclose(summary.loc["Zero Return", "rmse_improvement_vs_zero_pct"], 0.0)


def test_classical_and_ml_models_share_temporal_protocol() -> None:
    dataset = make_dataset_with_volume()
    comparison = run_model_comparison(
        dataset,
        n_splits=2,
        gap=1,
        test_size=20,
        models=("zero", "arima", "var", "ridge"),
    )
    comparison.assert_common_protocol()


def make_dataset_with_volume(n: int = 140) -> DatasetBundle:
    index = pd.bdate_range("2024-01-01", periods=n)
    rng = np.random.default_rng(222)
    returns = rng.normal(0, 0.01, n + 1)
    X = pd.DataFrame(
        {
            "return_1d": returns[:-1],
            "return_lag_1": np.roll(returns[:-1], 1),
            "return_lag_2": np.roll(returns[:-1], 2),
            "volume_change_1d": rng.normal(0, 0.12, n).clip(-0.9, None),
        },
        index=index,
    )
    y = pd.Series(returns[1:], index=index, name="next_return")
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
