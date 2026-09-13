from __future__ import annotations

import math

import numpy as np

from stock_ml_lab.evaluation.metrics import directional_metrics, regression_metrics


def test_regression_metrics_known_values() -> None:
    y_true = np.array([0.01, -0.02, 0.03, -0.04])
    y_pred = np.array([0.02, -0.01, -0.01, -0.02])

    metrics = regression_metrics(y_true, y_pred)

    assert np.isclose(metrics.mae, 0.02)
    assert np.isclose(
        metrics.rmse,
        np.sqrt((0.01**2 + 0.01**2 + 0.04**2 + 0.02**2) / 4),
    )
    assert np.isclose(metrics.directional_accuracy, 0.75)
    assert np.isclose(metrics.directional_coverage, 1.0)


def test_zero_predictions_have_zero_directional_coverage() -> None:
    y_true = np.array([0.01, -0.02, 0.03])
    y_pred = np.zeros(3)

    accuracy, coverage = directional_metrics(y_true, y_pred)

    assert math.isnan(accuracy)
    assert coverage == 0.0
