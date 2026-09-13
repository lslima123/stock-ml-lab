from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StoryMetric:
    name: str
    discovery: float
    confirmation: float
    unit: str


def _find_row(
    frame: pd.DataFrame,
    *,
    task: str,
    horizon: int,
    feature_set: str = "legacy",
    model_key: str,
) -> pd.Series | None:
    if frame.empty:
        return None
    mask = (
        (frame["task"] == task)
        & (pd.to_numeric(frame["horizon"], errors="coerce") == horizon)
        & (frame["feature_set"] == feature_set)
        & (frame["model_key"] == model_key)
    )
    rows = frame.loc[mask]
    return None if rows.empty else rows.iloc[-1]


def _find_confirmation(
    frame: pd.DataFrame,
    *,
    task: str,
    horizon: int,
) -> pd.Series | None:
    if frame.empty:
        return None
    mask = (
        (frame["task"] == task)
        & (pd.to_numeric(frame["horizon"], errors="coerce") == horizon)
    )
    rows = frame.loc[mask]
    return None if rows.empty else rows.iloc[-1]


def discovery_confirmation_metrics(
    m8: pd.DataFrame,
    m81_summary: pd.DataFrame,
) -> list[StoryMetric]:
    metrics: list[StoryMetric] = []

    for horizon in (10, 20):
        discovery = _find_row(
            m8,
            task="classification",
            horizon=horizon,
            model_key="logistic",
        )
        confirm = _find_confirmation(
            m81_summary,
            task="classification",
            horizon=horizon,
        )
        if discovery is not None and confirm is not None:
            metrics.append(
                StoryMetric(
                    name=f"Logistic {horizon}d ΔLogLoss",
                    discovery=float(discovery["log_loss_improvement_vs_prior_pct"]),
                    confirmation=float(confirm["median_primary_improvement_pct"]),
                    unit="%",
                )
            )

        discovery = _find_row(
            m8,
            task="regression",
            horizon=horizon,
            model_key="ridge",
        )
        confirm = _find_confirmation(
            m81_summary,
            task="regression",
            horizon=horizon,
        )
        if discovery is not None and confirm is not None:
            metrics.append(
                StoryMetric(
                    name=f"Ridge {horizon}d ΔRMSE",
                    discovery=float(discovery["rmse_improvement_vs_zero_pct"]),
                    confirmation=float(confirm["median_primary_improvement_pct"]),
                    unit="%",
                )
            )

    return metrics


def narrative_findings(
    m8: pd.DataFrame,
    m81_summary: pd.DataFrame,
    m81_assets: pd.DataFrame,
) -> list[str]:
    findings: list[str] = []

    class20 = _find_row(m8, task="classification", horizon=20, model_key="logistic")
    reg20 = _find_row(m8, task="regression", horizon=20, model_key="ridge")
    c20 = _find_confirmation(m81_summary, task="classification", horizon=20)
    r20 = _find_confirmation(m81_summary, task="regression", horizon=20)

    if class20 is not None:
        findings.append(
            "Discovery: PETR4.SA 20d Logistic reached "
            f"AUC {float(class20['roc_auc']):.3f} and "
            f"{float(class20['log_loss_improvement_vs_prior_pct']):+.2f}% "
            "Log Loss improvement versus the prior."
        )
    if reg20 is not None:
        findings.append(
            "Discovery: PETR4.SA 20d Ridge reached "
            f"{float(reg20['rmse_improvement_vs_zero_pct']):+.2f}% RMSE "
            "improvement versus Zero Return."
        )
    if c20 is not None:
        findings.append(
            "Locked confirmation: 20d Logistic won on "
            f"{int(c20['primary_win_count'])}/{int(c20['n_assets'])} assets; "
            f"median improvement {float(c20['median_primary_improvement_pct']):+.2f}%."
        )
    if r20 is not None:
        findings.append(
            "Locked confirmation: 20d Ridge won on "
            f"{int(r20['primary_win_count'])}/{int(r20['n_assets'])} assets; "
            f"median improvement {float(r20['median_primary_improvement_pct']):+.2f}%."
        )

    if not m81_summary.empty and "primary_hac_fdr_sig_count" in m81_summary:
        total_sig = int(pd.to_numeric(m81_summary["primary_hac_fdr_sig_count"], errors="coerce").fillna(0).sum())
        findings.append(
            f"Confirmatory inference: {total_sig} locked asset-level result(s) "
            "survived primary HAC/FDR significance at 5%."
        )

    if not m81_assets.empty and "primary_bootstrap_positive_95" in m81_assets:
        n_positive = int(m81_assets["primary_bootstrap_positive_95"].astype(bool).sum())
        findings.append(
            f"Bootstrap robustness: {n_positive} asset/candidate result(s) had a "
            "fully positive 95% primary-improvement interval."
        )

    return findings
