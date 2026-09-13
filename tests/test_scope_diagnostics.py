from __future__ import annotations

import numpy as np
import pandas as pd

from stock_ml_lab.comparison.diagnostics import build_scope_diagnostics, validate_predictions


def _predictions() -> pd.DataFrame:
    rows = []
    dates = pd.bdate_range("2024-01-02", periods=80)
    for task in ("regression", "classification"):
        for ticker_index, ticker in enumerate(("AAA", "BBB")):
            for index, date in enumerate(dates):
                if task == "regression":
                    y = np.sin(index / 8) * 0.03
                    local, global_ = y + 0.02, y + 0.01
                else:
                    y = float(index % 2)
                    local = 0.54 if y else 0.46
                    global_ = 0.62 if y else 0.38
                rows.append({
                    "ticker": ticker,
                    "date": date,
                    "y": y,
                    "local_score": local,
                    "global_score": global_,
                    "fold": 1 + index // 40,
                    "task": task,
                    "horizon": 5,
                })
    return pd.DataFrame(rows)


def test_scope_diagnostics_preserve_date_clusters_and_status() -> None:
    predictions = _predictions()
    outputs = build_scope_diagnostics(
        predictions, bootstrap_repetitions=200, random_state=7
    )
    assert len(outputs["date_inference"]) == 2
    assert (outputs["date_inference"]["dates"] == 80).all()
    assert (outputs["date_inference"]["date_hac_lags"] == 4).all()
    assert (outputs["date_inference"]["bootstrap_block_length_dates"] == 5).all()
    assert (outputs["date_inference"]["global_improvement_vs_local_pct"] > 0).all()
    assert set(outputs["phase_stability"]["phase"]) == set(range(5))


def test_scope_diagnostics_reject_duplicate_oos_rows() -> None:
    predictions = _predictions()
    duplicated = pd.concat([predictions, predictions.iloc[[0]]], ignore_index=True)
    try:
        validate_predictions(duplicated)
    except ValueError as exc:
        assert "duplicate" in str(exc).lower()
    else:
        raise AssertionError("Duplicate OOS rows should be rejected.")
