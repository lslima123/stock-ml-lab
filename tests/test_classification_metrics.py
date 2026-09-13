from __future__ import annotations

import numpy as np

from stock_ml_lab.classification.metrics import (
    calibration_table,
    classification_metrics,
)


def test_classification_metrics_perfect_probabilities() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    m = classification_metrics(y, p, calibration_bins=2)
    assert m.accuracy == 1.0
    assert m.balanced_accuracy == 1.0
    assert m.roc_auc == 1.0
    assert m.brier < 0.05


def test_calibration_table_accounts_for_all_rows() -> None:
    y = np.array([0, 1, 0, 1, 1, 0, 1, 0])
    p = np.linspace(0.1, 0.9, len(y))
    table = calibration_table(y, p, n_bins=4)
    assert table["count"].sum() == len(y)
    assert (table["mean_probability"].between(0, 1)).all()
