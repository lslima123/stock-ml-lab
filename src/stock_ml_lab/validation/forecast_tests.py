from __future__ import annotations

from dataclasses import dataclass
from math import ceil, erf, sqrt
from typing import Literal

import numpy as np


Criterion = Literal["mse", "mae"]


@dataclass(frozen=True)
class ForecastComparisonTest:
    statistic: float
    pvalue_two_sided: float
    pvalue_model_better: float
    mean_loss_difference: float
    criterion: str
    lags: int
    implementation: str


@dataclass(frozen=True)
class DirectionalTest:
    statistic: float
    pvalue_two_sided: float
    pvalue_larger: float
    directional_accuracy: float
    expected_accuracy_under_independence: float
    implementation: str


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _as_1d_finite(values, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one observation.")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values.")
    return arr


def _loss(actual: np.ndarray, forecast: np.ndarray, criterion: Criterion) -> np.ndarray:
    error = actual - forecast
    if criterion == "mse":
        return error**2
    if criterion == "mae":
        return np.abs(error)
    raise ValueError("criterion must be 'mse' or 'mae'.")


def _newey_west_dm(
    actual: np.ndarray,
    model_forecast: np.ndarray,
    benchmark_forecast: np.ndarray,
    *,
    criterion: Criterion,
    lags: int,
) -> ForecastComparisonTest:
    model_loss = _loss(actual, model_forecast, criterion)
    benchmark_loss = _loss(actual, benchmark_forecast, criterion)
    differential = model_loss - benchmark_loss
    n = len(differential)

    centered = differential - differential.mean()
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
        standard_error = sqrt(long_run_variance / n)
        statistic = float(differential.mean() / standard_error)
        p_two = float(2.0 * (1.0 - _normal_cdf(abs(statistic))))
        # Negative DM statistic means the model has lower loss than the benchmark.
        p_better = float(_normal_cdf(statistic))

    return ForecastComparisonTest(
        statistic=statistic,
        pvalue_two_sided=p_two,
        pvalue_model_better=p_better,
        mean_loss_difference=float(differential.mean()),
        criterion=criterion,
        lags=lags,
        implementation="fallback-newey-west",
    )


def diebold_mariano_vs_benchmark(
    actual,
    model_forecast,
    benchmark_forecast,
    *,
    criterion: Criterion = "mse",
    lags: int | None = None,
) -> ForecastComparisonTest:
    """Compare model and benchmark forecast loss with a HAC DM test.

    The loss differential is defined as model loss minus benchmark loss. Thus a
    negative statistic favors the model, and `pvalue_model_better` is a one-sided
    p-value for the alternative that the model has lower expected loss.

    statsmodels 0.15+ ships an official Diebold-Mariano implementation. This
    project uses it when available and falls back to the same Newey-West/HAC
    construction on older statsmodels releases.
    """
    y = _as_1d_finite(actual, "actual")
    model = _as_1d_finite(model_forecast, "model_forecast")
    benchmark = _as_1d_finite(benchmark_forecast, "benchmark_forecast")
    if not (len(y) == len(model) == len(benchmark)):
        raise ValueError("actual, model_forecast and benchmark_forecast must align.")

    if lags is None:
        lags = max(0, ceil(len(y) ** (1.0 / 3.0)))
    if lags < 0 or lags >= len(y):
        raise ValueError("lags must satisfy 0 <= lags < number of observations.")

    try:
        from statsmodels.tsa.stattools import diebold_mariano_test
    except ImportError:
        return _newey_west_dm(
            y,
            model,
            benchmark,
            criterion=criterion,
            lags=lags,
        )

    statsmodels_criterion = "mse" if criterion == "mse" else "mae"
    result = diebold_mariano_test(
        y,
        model,
        benchmark,
        lags=lags,
        criterion=statsmodels_criterion,
        horizon=1,
        harvey_adj=False,
    )
    statistic = float(result.statistic)
    model_loss = _loss(y, model, criterion)
    benchmark_loss = _loss(y, benchmark, criterion)

    return ForecastComparisonTest(
        statistic=statistic,
        pvalue_two_sided=float(result.pvalue),
        pvalue_model_better=float(_normal_cdf(statistic)),
        mean_loss_difference=float((model_loss - benchmark_loss).mean()),
        criterion=criterion,
        lags=lags,
        implementation="statsmodels",
    )


def _pesaran_timmermann_fallback(actual: np.ndarray, predicted: np.ndarray) -> DirectionalTest:
    actual_positive = actual > 0.0
    predicted_positive = predicted > 0.0
    n = len(actual)

    observed = float((actual_positive == predicted_positive).mean())
    p_actual = float(actual_positive.mean())
    p_pred = float(predicted_positive.mean())
    expected = p_actual * p_pred + (1.0 - p_actual) * (1.0 - p_pred)

    variance = expected * (1.0 - expected) / n
    adjustment = (
        (2.0 * p_actual - 1.0) ** 2 * p_pred * (1.0 - p_pred)
        + (2.0 * p_pred - 1.0) ** 2 * p_actual * (1.0 - p_actual)
    ) / n
    denominator_sq = variance - adjustment

    if denominator_sq <= 0.0:
        statistic = float("nan")
        p_two = float("nan")
        p_larger = float("nan")
    else:
        statistic = float((observed - expected) / sqrt(denominator_sq))
        p_two = float(2.0 * (1.0 - _normal_cdf(abs(statistic))))
        p_larger = float(1.0 - _normal_cdf(statistic))

    return DirectionalTest(
        statistic=statistic,
        pvalue_two_sided=p_two,
        pvalue_larger=p_larger,
        directional_accuracy=observed,
        expected_accuracy_under_independence=expected,
        implementation="fallback-pesaran-timmermann",
    )


def pesaran_timmermann_test(actual, predicted) -> DirectionalTest:
    """Test whether forecast and realized directions contain dependence."""
    y = _as_1d_finite(actual, "actual")
    pred = _as_1d_finite(predicted, "predicted")
    if len(y) != len(pred):
        raise ValueError("actual and predicted must align.")

    observed = float(((y > 0.0) == (pred > 0.0)).mean())
    p_actual = float((y > 0.0).mean())
    p_pred = float((pred > 0.0).mean())
    expected = p_actual * p_pred + (1.0 - p_actual) * (1.0 - p_pred)

    try:
        from statsmodels.stats.diagnostic import pesaran_timmermann
    except ImportError:
        return _pesaran_timmermann_fallback(y, pred)

    larger = pesaran_timmermann(y, pred, alternative="larger")
    two_sided = pesaran_timmermann(y, pred, alternative="two-sided")
    return DirectionalTest(
        statistic=float(larger.statistic),
        pvalue_two_sided=float(two_sided.pvalue),
        pvalue_larger=float(larger.pvalue),
        directional_accuracy=observed,
        expected_accuracy_under_independence=expected,
        implementation="statsmodels",
    )
