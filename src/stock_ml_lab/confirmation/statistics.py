from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt

import numpy as np


@dataclass(frozen=True)
class HACLossTest:
    statistic: float
    pvalue_two_sided: float
    pvalue_model_better: float
    mean_loss_difference: float
    lags: int


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def hac_loss_differential_test(
    model_loss,
    benchmark_loss,
    *,
    lags: int,
) -> HACLossTest:
    """HAC test of mean(model loss - benchmark loss) = 0.

    A negative statistic favors the model. The one-sided p-value tests the
    alternative that the model has lower expected loss.
    """
    model = np.asarray(model_loss, dtype=float).reshape(-1)
    benchmark = np.asarray(benchmark_loss, dtype=float).reshape(-1)
    if len(model) == 0 or len(model) != len(benchmark):
        raise ValueError("model_loss and benchmark_loss must be non-empty and aligned.")
    if not np.isfinite(np.concatenate([model, benchmark])).all():
        raise ValueError("Loss arrays must be finite.")
    if lags < 0 or lags >= len(model):
        raise ValueError("lags must satisfy 0 <= lags < number of observations.")

    differential = model - benchmark
    centered = differential - differential.mean()
    n = len(differential)

    gamma0 = float(np.dot(centered, centered) / n)
    long_run_variance = gamma0
    for lag in range(1, lags + 1):
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / n)
        weight = 1.0 - lag / (lags + 1.0)
        long_run_variance += 2.0 * weight * covariance

    if long_run_variance <= 0.0:
        statistic = float("nan")
        p_two = float("nan")
        p_better = float("nan")
    else:
        statistic = float(differential.mean() / sqrt(long_run_variance / n))
        p_two = float(2.0 * (1.0 - _normal_cdf(abs(statistic))))
        p_better = float(_normal_cdf(statistic))

    return HACLossTest(
        statistic=statistic,
        pvalue_two_sided=p_two,
        pvalue_model_better=p_better,
        mean_loss_difference=float(differential.mean()),
        lags=lags,
    )


def binary_log_losses(actual, probability, *, eps: float = 1e-12) -> np.ndarray:
    y = np.asarray(actual, dtype=int).reshape(-1)
    p = np.asarray(probability, dtype=float).reshape(-1)
    if len(y) == 0 or len(y) != len(p):
        raise ValueError("actual and probability must be non-empty and aligned.")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("actual must contain only 0 and 1.")
    if not np.isfinite(p).all() or np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("probability must lie in [0, 1].")
    p = np.clip(p, eps, 1.0 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1.0 - p))


def binary_brier_losses(actual, probability) -> np.ndarray:
    y = np.asarray(actual, dtype=int).reshape(-1)
    p = np.asarray(probability, dtype=float).reshape(-1)
    if len(y) == 0 or len(y) != len(p):
        raise ValueError("actual and probability must be non-empty and aligned.")
    return (y - p) ** 2
