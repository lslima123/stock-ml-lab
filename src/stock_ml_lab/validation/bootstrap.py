from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stock_ml_lab.evaluation.metrics import regression_metrics


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float


def circular_block_indices(
    n: int,
    *,
    block_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if n < 2:
        raise ValueError("Bootstrap requires at least two observations.")
    if not 1 <= block_length <= n:
        raise ValueError("block_length must satisfy 1 <= block_length <= n.")

    n_blocks = int(np.ceil(n / block_length))
    starts = rng.integers(0, n, size=n_blocks)
    offsets = np.arange(block_length)
    sampled = (starts[:, None] + offsets[None, :]) % n
    return sampled.reshape(-1)[:n]


def _percentile_interval(
    estimate: float,
    samples: np.ndarray,
    *,
    confidence: float,
) -> BootstrapInterval:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between 0 and 1.")
    alpha = (1.0 - confidence) / 2.0
    lower, upper = np.quantile(samples, [alpha, 1.0 - alpha])
    return BootstrapInterval(
        estimate=float(estimate),
        lower=float(lower),
        upper=float(upper),
        confidence=confidence,
    )


def forecast_block_bootstrap(
    actual,
    model_forecast,
    benchmark_forecast,
    *,
    n_bootstrap: int = 2000,
    block_length: int = 20,
    confidence: float = 0.95,
    random_state: int = 42,
) -> dict[str, BootstrapInterval]:
    """Moving-block bootstrap intervals for forecast metrics and improvements."""
    y = np.asarray(actual, dtype=float).reshape(-1)
    model = np.asarray(model_forecast, dtype=float).reshape(-1)
    benchmark = np.asarray(benchmark_forecast, dtype=float).reshape(-1)

    if not (len(y) == len(model) == len(benchmark)):
        raise ValueError("actual and forecasts must have the same length.")
    if len(y) == 0 or not np.isfinite(np.concatenate([y, model, benchmark])).all():
        raise ValueError("Bootstrap inputs must be non-empty and finite.")
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap must be >= 100.")

    model_metrics = regression_metrics(y, model)
    benchmark_metrics = regression_metrics(y, benchmark)

    estimates = {
        "mae": model_metrics.mae,
        "rmse": model_metrics.rmse,
        "directional_accuracy": model_metrics.directional_accuracy,
        "mae_improvement_vs_benchmark_pct": (
            (benchmark_metrics.mae - model_metrics.mae) / benchmark_metrics.mae * 100.0
        ),
        "rmse_improvement_vs_benchmark_pct": (
            (benchmark_metrics.rmse - model_metrics.rmse) / benchmark_metrics.rmse * 100.0
        ),
    }

    rng = np.random.default_rng(random_state)
    samples = {key: np.empty(n_bootstrap, dtype=float) for key in estimates}

    for i in range(n_bootstrap):
        idx = circular_block_indices(len(y), block_length=block_length, rng=rng)
        sampled_y = y[idx]
        sampled_model = model[idx]
        sampled_benchmark = benchmark[idx]

        m = regression_metrics(sampled_y, sampled_model)
        b = regression_metrics(sampled_y, sampled_benchmark)
        samples["mae"][i] = m.mae
        samples["rmse"][i] = m.rmse
        samples["directional_accuracy"][i] = m.directional_accuracy
        samples["mae_improvement_vs_benchmark_pct"][i] = (
            (b.mae - m.mae) / b.mae * 100.0
        )
        samples["rmse_improvement_vs_benchmark_pct"][i] = (
            (b.rmse - m.rmse) / b.rmse * 100.0
        )

    return {
        key: _percentile_interval(
            estimate,
            samples[key],
            confidence=confidence,
        )
        for key, estimate in estimates.items()
    }
