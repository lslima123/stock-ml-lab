from __future__ import annotations

from math import erf, sqrt

import numpy as np
import pandas as pd

from stock_ml_lab.robustness.multiple_testing import exact_sign_test_greater


REQUIRED_PREDICTION_COLUMNS = {
    "ticker", "date", "y", "local_score", "global_score", "fold", "task", "horizon"
}


def validate_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(REQUIRED_PREDICTION_COLUMNS.difference(predictions.columns))
    if missing:
        raise ValueError(f"Predictions are missing required columns: {missing}")
    frame = predictions.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    if frame.empty:
        raise ValueError("Predictions cannot be empty.")
    if frame.duplicated(["task", "horizon", "ticker", "date"]).any():
        raise ValueError("Predictions contain duplicate task/horizon/ticker/date rows.")
    numeric = frame[["y", "local_score", "global_score"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Predictions contain non-finite target or score values.")
    return frame.sort_values(["task", "horizon", "date", "ticker"], kind="stable")


def loss_arrays(group: pd.DataFrame, task: str) -> tuple[np.ndarray, np.ndarray]:
    y = group["y"].to_numpy(dtype=float)
    local = group["local_score"].to_numpy(dtype=float)
    global_ = group["global_score"].to_numpy(dtype=float)
    if task == "regression":
        return (y - local) ** 2, (y - global_) ** 2
    if task == "classification":
        local = np.clip(local, 1e-8, 1 - 1e-8)
        global_ = np.clip(global_, 1e-8, 1 - 1e-8)
        return (
            -(y * np.log(local) + (1 - y) * np.log(1 - local)),
            -(y * np.log(global_) + (1 - y) * np.log(1 - global_)),
        )
    raise ValueError(f"Unsupported task: {task}")


def score_metrics(group: pd.DataFrame, task: str) -> tuple[float, float, float]:
    local_loss, global_loss = loss_arrays(group, task)
    if task == "regression":
        local_primary = float(np.sqrt(local_loss.mean()))
        global_primary = float(np.sqrt(global_loss.mean()))
    else:
        local_primary = float(local_loss.mean())
        global_primary = float(global_loss.mean())
    improvement = (
        100.0 * (local_primary - global_primary) / local_primary
        if local_primary != 0 else float("nan")
    )
    return local_primary, global_primary, improvement


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def _hac_mean_test(values: np.ndarray, lags: int) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    n = len(values)
    lags = min(max(int(lags), 0), max(n - 1, 0))
    centered = values - values.mean()
    long_run_variance = float(np.dot(centered, centered) / n)
    for lag in range(1, lags + 1):
        covariance = float(np.dot(centered[lag:], centered[:-lag]) / n)
        long_run_variance += 2.0 * (1.0 - lag / (lags + 1.0)) * covariance
    if long_run_variance <= 0.0:
        statistic = pvalue = float("nan")
    else:
        statistic = float(values.mean() / sqrt(long_run_variance / n))
        pvalue = float(1.0 - _normal_cdf(statistic))
    return {
        "date_hac_statistic": statistic,
        "date_hac_pvalue_global_better": pvalue,
        "date_hac_lags": lags,
        "mean_date_clustered_loss_advantage": float(values.mean()),
    }


def circular_date_block_bootstrap(
    predictions: pd.DataFrame,
    *,
    task: str,
    block_length: int,
    repetitions: int,
    random_state: int,
) -> dict[str, float]:
    if repetitions < 100:
        raise ValueError("Bootstrap repetitions must be at least 100.")
    frame = predictions.sort_values(["date", "ticker"], kind="stable").copy()
    local_loss, global_loss = loss_arrays(frame, task)
    frame["_local_loss"] = local_loss
    frame["_global_loss"] = global_loss
    clustered = frame.groupby("date", sort=True).agg(
        local_loss_sum=("_local_loss", "sum"),
        global_loss_sum=("_global_loss", "sum"),
        rows=("ticker", "size"),
    )
    n_dates = len(clustered)
    block_length = min(max(int(block_length), 1), n_dates)
    n_blocks = int(np.ceil(n_dates / block_length))
    offsets = np.arange(block_length)
    local_sum = clustered["local_loss_sum"].to_numpy(dtype=float)
    global_sum = clustered["global_loss_sum"].to_numpy(dtype=float)
    rows = clustered["rows"].to_numpy(dtype=float)
    rng = np.random.default_rng(random_state)
    estimates = np.empty(repetitions, dtype=float)
    for repetition in range(repetitions):
        starts = rng.integers(0, n_dates, size=n_blocks)
        sample = ((starts[:, None] + offsets) % n_dates).ravel()[:n_dates]
        local_mean = float(local_sum[sample].sum() / rows[sample].sum())
        global_mean = float(global_sum[sample].sum() / rows[sample].sum())
        if task == "regression":
            local_primary, global_primary = sqrt(local_mean), sqrt(global_mean)
        else:
            local_primary, global_primary = local_mean, global_mean
        estimates[repetition] = 100.0 * (local_primary - global_primary) / local_primary
    return {
        "bootstrap_repetitions": repetitions,
        "bootstrap_block_length_dates": block_length,
        "bootstrap_improvement_mean_pct": float(estimates.mean()),
        "bootstrap_improvement_ci_low_pct": float(np.quantile(estimates, 0.025)),
        "bootstrap_improvement_ci_high_pct": float(np.quantile(estimates, 0.975)),
        "bootstrap_probability_global_better": float(np.mean(estimates > 0.0)),
    }


def date_clustered_inference(
    predictions: pd.DataFrame,
    *,
    task: str,
    horizon: int,
    bootstrap_repetitions: int = 10_000,
    random_state: int = 42,
) -> dict[str, float | int | str]:
    frame = validate_predictions(predictions)
    local_loss, global_loss = loss_arrays(frame, task)
    frame["_advantage"] = local_loss - global_loss
    by_date = frame.groupby("date", sort=True)["_advantage"].mean().to_numpy(dtype=float)
    local_primary, global_primary, improvement = score_metrics(frame, task)
    hac = _hac_mean_test(by_date, max(horizon - 1, 0))
    bootstrap = circular_date_block_bootstrap(
        frame,
        task=task,
        block_length=max(horizon, 1),
        repetitions=bootstrap_repetitions,
        random_state=random_state,
    )
    return {
        "task": task,
        "horizon": int(horizon),
        "primary_metric": "rmse" if task == "regression" else "log_loss",
        "rows": len(frame),
        "assets": int(frame["ticker"].nunique()),
        "dates": int(frame["date"].nunique()),
        "local_primary": local_primary,
        "global_primary": global_primary,
        "global_improvement_vs_local_pct": improvement,
        **hac,
        **bootstrap,
    }


def _stability(
    predictions: pd.DataFrame,
    *,
    keys: list[str],
    label: str,
) -> pd.DataFrame:
    rows: list[dict] = []
    for group_key, group in predictions.groupby(keys, sort=True):
        values = group_key if isinstance(group_key, tuple) else (group_key,)
        identity = dict(zip(keys, values))
        task = str(group["task"].iloc[0])
        local, global_, improvement = score_metrics(group, task)
        rows.append({
            **identity,
            "task": task,
            "horizon": int(group["horizon"].iloc[0]),
            "level": label,
            "rows": len(group),
            "assets": int(group["ticker"].nunique()),
            "local_primary": local,
            "global_primary": global_,
            "global_improvement_vs_local_pct": improvement,
            "global_win": bool(improvement > 0),
        })
    return pd.DataFrame(rows)


def fold_stability(predictions: pd.DataFrame) -> pd.DataFrame:
    return _stability(validate_predictions(predictions), keys=["task", "horizon", "fold"], label="fold")


def phase_stability(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = validate_predictions(predictions)
    pieces: list[pd.DataFrame] = []
    for (task, horizon, ticker), group in frame.groupby(["task", "horizon", "ticker"], sort=True):
        group = group.sort_values("date", kind="stable").copy()
        group["phase"] = np.arange(len(group)) % int(horizon)
        pieces.append(group)
    phased = pd.concat(pieces, ignore_index=True)
    return _stability(phased, keys=["task", "horizon", "phase"], label="non_overlapping_phase")


def regional_stability(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = validate_predictions(predictions)
    frame["region"] = np.where(frame["ticker"].str.endswith(".SA"), "Brazil", "United States")
    return _stability(frame, keys=["task", "horizon", "region"], label="region")


def build_scope_diagnostics(
    predictions: pd.DataFrame,
    *,
    bootstrap_repetitions: int = 10_000,
    random_state: int = 42,
    analysis_status: str = "post_hoc_robustness_diagnostic",
) -> dict[str, pd.DataFrame]:
    frame = validate_predictions(predictions)
    inference_rows: list[dict] = []
    summary_rows: list[dict] = []
    for index, ((task, horizon), group) in enumerate(
        frame.groupby(["task", "horizon"], sort=True)
    ):
        inference = date_clustered_inference(
            group,
            task=str(task),
            horizon=int(horizon),
            bootstrap_repetitions=bootstrap_repetitions,
            random_state=random_state + index,
        )
        inference["analysis_status"] = analysis_status
        inference_rows.append(inference)

        asset_improvements = []
        for _, asset in group.groupby("ticker", sort=True):
            asset_improvements.append(score_metrics(asset, str(task))[2])
        fold = fold_stability(group)
        phase = phase_stability(group)
        summary_rows.append({
            "task": task,
            "horizon": int(horizon),
            "analysis_status": analysis_status,
            "assets": len(asset_improvements),
            "asset_win_count": int(np.sum(np.asarray(asset_improvements) > 0)),
            "asset_win_sign_test_p": exact_sign_test_greater(
                int(np.sum(np.asarray(asset_improvements) > 0)), len(asset_improvements)
            ),
            "fold_win_count": int(fold["global_win"].sum()),
            "fold_count": len(fold),
            "phase_win_count": int(phase["global_win"].sum()),
            "phase_count": len(phase),
            "global_improvement_vs_local_pct": inference["global_improvement_vs_local_pct"],
            "bootstrap_improvement_ci_low_pct": inference["bootstrap_improvement_ci_low_pct"],
            "bootstrap_improvement_ci_high_pct": inference["bootstrap_improvement_ci_high_pct"],
            "date_hac_pvalue_global_better": inference["date_hac_pvalue_global_better"],
        })
    return {
        "summary": pd.DataFrame(summary_rows),
        "date_inference": pd.DataFrame(inference_rows),
        "fold_stability": fold_stability(frame),
        "phase_stability": phase_stability(frame),
        "regional_stability": regional_stability(frame),
    }
