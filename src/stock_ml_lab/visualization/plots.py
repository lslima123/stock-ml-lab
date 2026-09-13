from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from stock_ml_lab.visualization.story import StoryMetric


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_discovery_vs_confirmation(
    metrics: list[StoryMetric],
    path: Path,
) -> Path | None:
    if not metrics:
        return None

    labels = [m.name for m in metrics]
    discovery = np.array([m.discovery for m in metrics], dtype=float)
    confirmation = np.array([m.confirmation for m in metrics], dtype=float)

    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    ax.bar(x - width / 2, discovery, width, label="PETR4 discovery")
    ax.bar(x + width / 2, confirmation, width, label="9-asset confirmation median")
    ax.axhline(0.0, linewidth=1)
    ax.set_ylabel("Improvement vs naive baseline (%)")
    ax.set_title("Discovery vs locked confirmation")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right")
    ax.legend()
    return _save(fig, path)


def plot_m8_horizon_classification(frame: pd.DataFrame, path: Path) -> Path | None:
    if frame.empty:
        return None
    data = frame[
        (frame["task"] == "classification")
        & (frame["model_key"] == "logistic")
        & (frame["feature_set"] == "legacy")
    ].copy()
    if data.empty:
        return None
    data = data.sort_values("horizon")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(data["horizon"], data["roc_auc"], marker="o")
    ax.axhline(0.5, linewidth=1)
    ax.set_xlabel("Forecast horizon (trading days)")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("PETR4 discovery: Logistic ROC-AUC by horizon")
    ax.set_xticks(data["horizon"].astype(int))
    return _save(fig, path)


def plot_m8_horizon_regression(frame: pd.DataFrame, path: Path) -> Path | None:
    if frame.empty:
        return None
    data = frame[
        (frame["task"] == "regression")
        & (frame["model_key"] == "ridge")
        & (frame["feature_set"] == "legacy")
    ].copy()
    if data.empty:
        return None
    data = data.sort_values("horizon")
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(
        data["horizon"],
        data["rmse_improvement_vs_zero_pct"],
        marker="o",
    )
    ax.axhline(0.0, linewidth=1)
    ax.set_xlabel("Forecast horizon (trading days)")
    ax.set_ylabel("RMSE improvement vs Zero Return (%)")
    ax.set_title("PETR4 discovery: Ridge improvement by horizon")
    ax.set_xticks(data["horizon"].astype(int))
    return _save(fig, path)


def plot_m8_feature_heatmap(frame: pd.DataFrame, path: Path) -> Path | None:
    if frame.empty:
        return None
    data = frame[
        (frame["task"] == "classification")
        & (frame["model_key"] == "logistic")
        & frame["feature_set"].isin(["legacy", "extended", "market", "regime"])
    ].copy()
    if data.empty:
        return None

    pivot = data.pivot_table(
        index="feature_set",
        columns="horizon",
        values="roc_auc",
        aggfunc="last",
    )
    order = [x for x in ["legacy", "extended", "market", "regime"] if x in pivot.index]
    pivot = pivot.reindex(order).sort_index(axis=1)

    fig, ax = plt.subplots(figsize=(8.3, 4.7))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels([f"{int(h)}d" for h in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("PETR4 exploratory Logistic ROC-AUC: feature set × horizon")
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.iloc[i, j]
            if np.isfinite(value):
                ax.text(j, i, f"{value:.3f}", ha="center", va="center")
    fig.colorbar(image, ax=ax, label="ROC-AUC")
    return _save(fig, path)


def plot_confirmation_assets(
    assets: pd.DataFrame,
    *,
    task: str,
    horizon: int,
    path: Path,
) -> Path | None:
    if assets.empty:
        return None
    data = assets[
        (assets["task"] == task)
        & (pd.to_numeric(assets["horizon"], errors="coerce") == horizon)
    ].copy()
    if data.empty:
        return None

    data = data.sort_values("primary_improvement_pct")
    y = np.arange(len(data))
    estimate = data["primary_improvement_pct"].to_numpy(dtype=float)
    lower = data["primary_bootstrap_lower_pct"].to_numpy(dtype=float)
    upper = data["primary_bootstrap_upper_pct"].to_numpy(dtype=float)
    xerr = np.vstack([estimate - lower, upper - estimate])

    fig, ax = plt.subplots(figsize=(9.0, 5.3))
    ax.barh(y, estimate)
    ax.errorbar(estimate, y, xerr=xerr, fmt="none", capsize=3)
    ax.axvline(0.0, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(data["ticker"])
    primary = "ΔLogLoss" if task == "classification" else "ΔRMSE"
    ax.set_xlabel(f"{primary} vs naive baseline (%)")
    ax.set_title(f"Locked confirmation: {task} {horizon}d with 95% block-bootstrap CI")
    return _save(fig, path)


def plot_phase_win_rates(
    assets: pd.DataFrame,
    path: Path,
) -> Path | None:
    if assets.empty or "phase_win_rate" not in assets:
        return None

    data = assets.copy()
    data["candidate"] = (
        data["task"].str.title()
        + " "
        + data["horizon"].astype(int).astype(str)
        + "d"
    )
    grouped = data.groupby("candidate", sort=False)["phase_win_rate"].mean()
    if grouped.empty:
        return None

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.bar(grouped.index, 100.0 * grouped.to_numpy(dtype=float))
    ax.axhline(50.0, linewidth=1)
    ax.set_ylabel("Mean non-overlapping phase win rate (%)")
    ax.set_title("Locked confirmation: phase stability")
    ax.tick_params(axis="x", rotation=15)
    return _save(fig, path)


def plot_m7_auc(summary: pd.DataFrame, path: Path) -> Path | None:
    if summary.empty or "roc_auc" not in summary:
        return None

    data = summary.copy()
    model_col = "model" if "model" in data else data.columns[0]
    if "Prior Baseline" in data[model_col].astype(str).values:
        data = data[data[model_col] != "Prior Baseline"]
    data = data.sort_values("roc_auc")
    if data.empty:
        return None

    fig, ax = plt.subplots(figsize=(8.7, 4.8))
    ax.barh(data[model_col].astype(str), data["roc_auc"].astype(float))
    ax.axvline(0.5, linewidth=1)
    ax.set_xlabel("ROC-AUC")
    ax.set_title("M7 PETR4 one-day direction classification")
    return _save(fig, path)


def plot_m6_cross_asset(frame: pd.DataFrame, path: Path) -> Path | None:
    if frame.empty or "rmse_improvement_vs_zero_pct" not in frame:
        return None

    data = frame.copy()
    if "model" not in data:
        return None
    pivot = data.pivot_table(
        index="ticker",
        columns="model",
        values="rmse_improvement_vs_zero_pct",
        aggfunc="last",
    )
    if pivot.empty:
        return None

    fig, ax = plt.subplots(figsize=(10.0, 5.2))
    x = np.arange(len(pivot.index))
    width = min(0.35, 0.8 / max(1, len(pivot.columns)))
    offsets = np.linspace(
        -width * (len(pivot.columns) - 1) / 2,
        width * (len(pivot.columns) - 1) / 2,
        len(pivot.columns),
    )
    for offset, col in zip(offsets, pivot.columns):
        ax.bar(x + offset, pivot[col].to_numpy(dtype=float), width=width, label=str(col))
    ax.axhline(0.0, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=40, ha="right")
    ax.set_ylabel("RMSE improvement vs Zero Return (%)")
    ax.set_title("M6 cross-asset daily regression robustness")
    ax.legend()
    return _save(fig, path)
