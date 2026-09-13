from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from stock_ml_lab.classification.metrics import classification_metrics


@dataclass(frozen=True)
class ClassificationBootstrapInterval:
    metric: str
    estimate: float
    lower: float
    upper: float


def _circular_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=blocks)
    idx = np.concatenate(
        [(start + np.arange(block_length)) % n for start in starts]
    )[:n]
    return idx


def classification_block_bootstrap(
    actual,
    model_probability,
    benchmark_probability,
    *,
    n_bootstrap: int = 2000,
    block_length: int = 20,
    confidence: float = 0.95,
    random_state: int = 42,
) -> pd.DataFrame:
    y = np.asarray(actual, dtype=int).reshape(-1)
    model = np.asarray(model_probability, dtype=float).reshape(-1)
    benchmark = np.asarray(benchmark_probability, dtype=float).reshape(-1)
    if not (len(y) == len(model) == len(benchmark)):
        raise ValueError("actual and probability arrays must align.")
    if block_length < 1 or block_length > len(y):
        raise ValueError("block_length must satisfy 1 <= block_length <= n.")
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap must be >= 100.")

    model_metrics = classification_metrics(y, model)
    benchmark_metrics = classification_metrics(y, benchmark)

    estimates = {
        "accuracy_improvement_pp": 100.0 * (
            model_metrics.accuracy - benchmark_metrics.accuracy
        ),
        "balanced_accuracy_improvement_pp": 100.0 * (
            model_metrics.balanced_accuracy - benchmark_metrics.balanced_accuracy
        ),
        "brier_improvement_pct": 100.0 * (
            benchmark_metrics.brier - model_metrics.brier
        ) / benchmark_metrics.brier,
        "log_loss_improvement_pct": 100.0 * (
            benchmark_metrics.log_loss - model_metrics.log_loss
        ) / benchmark_metrics.log_loss,
        "roc_auc_excess_pp": 100.0 * (model_metrics.roc_auc - 0.5),
    }

    values = {key: [] for key in estimates}
    rng = np.random.default_rng(random_state)
    for _ in range(n_bootstrap):
        idx = _circular_indices(len(y), block_length, rng)
        m = classification_metrics(y[idx], model[idx])
        b = classification_metrics(y[idx], benchmark[idx])
        values["accuracy_improvement_pp"].append(100.0 * (m.accuracy - b.accuracy))
        values["balanced_accuracy_improvement_pp"].append(
            100.0 * (m.balanced_accuracy - b.balanced_accuracy)
        )
        values["brier_improvement_pct"].append(
            100.0 * (b.brier - m.brier) / b.brier
        )
        values["log_loss_improvement_pct"].append(
            100.0 * (b.log_loss - m.log_loss) / b.log_loss
        )
        values["roc_auc_excess_pp"].append(100.0 * (m.roc_auc - 0.5))

    alpha = 1.0 - confidence
    rows = []
    for metric, estimate in estimates.items():
        arr = np.asarray(values[metric], dtype=float)
        arr = arr[np.isfinite(arr)]
        rows.append(
            {
                "metric": metric,
                "estimate": estimate,
                "lower": float(np.quantile(arr, alpha / 2.0)),
                "upper": float(np.quantile(arr, 1.0 - alpha / 2.0)),
            }
        )
    return pd.DataFrame(rows).set_index("metric")
