from __future__ import annotations

import numpy as np
import pandas as pd

from stock_ml_lab.confirmation.phases import (
    classification_phase_table,
    phase_labels,
    regression_phase_table,
    summarize_phase_stability,
)


def test_phase_labels_are_h_rows_apart_within_phase() -> None:
    full = pd.date_range("2020-01-01", periods=100, freq="D")
    oos = full[20:]
    labels = phase_labels(full, oos, horizon=10)
    positions = pd.Series(np.arange(len(full)), index=full)
    for phase in range(10):
        idx = labels.index[labels == phase]
        diffs = np.diff(positions.loc[idx].to_numpy())
        assert np.all(diffs == 10)


def test_phase_tables_cover_every_oos_observation_once() -> None:
    full = pd.date_range("2020-01-01", periods=80, freq="D")
    idx = full[20:]
    actual_class = pd.Series((np.arange(len(idx)) % 3 != 0).astype(int), index=idx)
    model_p = pd.Series(np.linspace(0.35, 0.70, len(idx)), index=idx)
    prior_p = pd.Series(np.full(len(idx), 0.55), index=idx)
    class_table = classification_phase_table(
        full_index=full,
        actual=actual_class,
        model_probability=model_p,
        prior_probability=prior_p,
        horizon=5,
    )
    assert class_table["n"].sum() == len(idx)

    actual_reg = pd.Series(np.linspace(-0.1, 0.2, len(idx)), index=idx)
    pred = actual_reg * 0.7
    zero = pd.Series(0.0, index=idx)
    reg_table = regression_phase_table(
        full_index=full,
        actual=actual_reg,
        model_forecast=pred,
        zero_forecast=zero,
        horizon=5,
    )
    assert reg_table["n"].sum() == len(idx)


def test_phase_stability_summary_counts_positive_phases() -> None:
    frame = pd.DataFrame({"metric": [1.0, 2.0, -1.0, 3.0]})
    summary = summarize_phase_stability(frame, primary_column="metric")
    assert summary["phase_count"] == 4
    assert summary["phase_win_count"] == 3
    assert np.isclose(summary["phase_win_rate"], 0.75)
