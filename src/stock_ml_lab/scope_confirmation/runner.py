from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json
from math import erf, sqrt
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from stock_ml_lab.api.registry import build_local_estimator
from stock_ml_lab.comparison.diagnostics import (
    date_clustered_inference,
    fold_stability,
    phase_stability,
    regional_stability,
    score_metrics,
)
from stock_ml_lab.comparison.report import write_scope_report
from stock_ml_lab.global_model.dataset import build_global_panel_from_ohlcv
from stock_ml_lab.global_model.models import build_global_estimator
from stock_ml_lab.global_model.validation import panel_walk_forward_splits
from stock_ml_lab.reporting import unique_run_directory
from stock_ml_lab.robustness.multiple_testing import benjamini_hochberg, exact_sign_test_greater
from stock_ml_lab.scope_confirmation.config import ScopeConfirmationConfig


def _predict(estimator, X: pd.DataFrame, task: str) -> np.ndarray:
    if task == "regression":
        return np.asarray(estimator.predict(X), dtype=float)
    probabilities = np.asarray(estimator.predict_proba(X), dtype=float)
    classes = list(estimator.classes_)
    if 1 not in classes:
        raise ValueError("Classifier does not expose positive class 1.")
    return probabilities[:, classes.index(1)]


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + erf(value / sqrt(2.0)))


def _hac_advantage(group: pd.DataFrame, task: str, lags: int) -> tuple[float, float]:
    y = group["y"].to_numpy(dtype=float)
    local = group["local_score"].to_numpy(dtype=float)
    global_ = group["global_score"].to_numpy(dtype=float)
    if task == "regression":
        advantage = (y - local) ** 2 - (y - global_) ** 2
    else:
        local = np.clip(local, 1e-8, 1 - 1e-8)
        global_ = np.clip(global_, 1e-8, 1 - 1e-8)
        advantage = (
            -(y * np.log(local) + (1 - y) * np.log(1 - local))
            + (y * np.log(global_) + (1 - y) * np.log(1 - global_))
        )
    n = len(advantage)
    lags = min(lags, n - 1)
    centered = advantage - advantage.mean()
    lr = float(np.dot(centered, centered) / n)
    for lag in range(1, lags + 1):
        lr += 2.0 * (1.0 - lag / (lags + 1.0)) * float(
            np.dot(centered[lag:], centered[:-lag]) / n
        )
    if lr <= 0:
        return float("nan"), float("nan")
    statistic = float(advantage.mean() / sqrt(lr / n))
    return statistic, float(1.0 - _normal_cdf(statistic))


def _asset_results(predictions: pd.DataFrame, config: ScopeConfirmationConfig) -> pd.DataFrame:
    rows: list[dict] = []
    for (task, ticker), group in predictions.groupby(["task", "ticker"], sort=True):
        local, global_, improvement = score_metrics(group, str(task))
        statistic, pvalue = _hac_advantage(group, str(task), config.hac_lags)
        rows.append({
            "task": task,
            "horizon": config.horizon,
            "ticker": ticker,
            "region": "Brazil" if str(ticker).endswith(".SA") else "United States",
            "n": len(group),
            "primary_metric": "rmse" if task == "regression" else "log_loss",
            "local_primary": local,
            "global_primary": global_,
            "global_improvement_vs_local_pct": improvement,
            "global_win": improvement > 0,
            "asset_hac_statistic": statistic,
            "asset_hac_pvalue_global_better": pvalue,
            "hac_lags": config.hac_lags,
        })
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["asset_hac_qvalue_within_task"] = frame.groupby("task")[
            "asset_hac_pvalue_global_better"
        ].transform(lambda values: benjamini_hochberg(values.to_numpy(dtype=float)))
    return frame


def _fit_predictions(
    training_data: Mapping[str, pd.DataFrame],
    confirmation_data: Mapping[str, pd.DataFrame],
    config: ScopeConfirmationConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    training_bundle = build_global_panel_from_ohlcv(training_data, horizon=config.horizon)
    confirmation_bundle = build_global_panel_from_ohlcv(confirmation_data, horizon=config.horizon)
    splits = panel_walk_forward_splits(
        confirmation_bundle,
        n_splits=config.n_splits,
        test_size_dates=config.test_size_dates,
    )
    predictions: list[pd.DataFrame] = []
    audit_rows: list[dict] = []
    train_frame = training_bundle.frame
    confirm_frame = confirmation_bundle.frame

    for task, model in (("regression", "ridge"), ("classification", "logistic")):
        train_y = (
            training_bundle.regression_y if task == "regression"
            else training_bundle.classification_y
        )
        confirm_y = (
            confirmation_bundle.regression_y if task == "regression"
            else confirmation_bundle.classification_y
        )
        for split in splits:
            test_dates = pd.DatetimeIndex(
                sorted(pd.unique(confirm_frame.iloc[split.test_index]["date"]))
            )
            test_start = pd.Timestamp(test_dates[0])
            global_train_mask = pd.to_datetime(train_frame["target_end_date"]) < test_start
            global_train_idx = np.flatnonzero(global_train_mask.to_numpy())
            if len(global_train_idx) == 0:
                raise ValueError(f"Fold {split.fold}: global training set is empty after purge.")
            global_estimator = build_global_estimator(task, model)
            global_estimator.fit(
                training_bundle.X.iloc[global_train_idx], train_y.iloc[global_train_idx]
            )
            audit_rows.append({
                "task": task,
                "fold": split.fold,
                "scope": "global",
                "ticker": "__POOLED__",
                "train_rows": len(global_train_idx),
                "train_assets": int(train_frame.iloc[global_train_idx]["ticker"].nunique()),
                "max_target_end_date": pd.to_datetime(
                    train_frame.iloc[global_train_idx]["target_end_date"]
                ).max(),
                "test_start": test_start,
                "purge_valid": bool(
                    pd.to_datetime(train_frame.iloc[global_train_idx]["target_end_date"]).max()
                    < test_start
                ),
                "confirmation_rows_in_fit": 0,
            })

            for ticker in config.confirmation_universe:
                local_train_mask = (
                    (confirm_frame["ticker"] == ticker)
                    & (pd.to_datetime(confirm_frame["target_end_date"]) < test_start)
                )
                test_mask = (
                    (confirm_frame["ticker"] == ticker)
                    & pd.to_datetime(confirm_frame["date"]).isin(test_dates)
                )
                local_train_idx = np.flatnonzero(local_train_mask.to_numpy())
                test_idx = np.flatnonzero(test_mask.to_numpy())
                if len(local_train_idx) < 60 or len(test_idx) == 0:
                    raise ValueError(
                        f"Fold {split.fold}, {ticker}: insufficient local train/test rows."
                    )
                local_estimator = build_local_estimator(task, model, random_state=42)
                local_estimator.fit(
                    confirmation_bundle.X.iloc[local_train_idx], confirm_y.iloc[local_train_idx]
                )
                local_score = _predict(local_estimator, confirmation_bundle.X.iloc[test_idx], task)
                global_score = _predict(global_estimator, confirmation_bundle.X.iloc[test_idx], task)
                test_meta = confirm_frame.iloc[test_idx]
                predictions.append(pd.DataFrame({
                    "ticker": ticker,
                    "date": pd.to_datetime(test_meta["date"]).to_numpy(),
                    "y": confirm_y.iloc[test_idx].to_numpy(dtype=float),
                    "local_score": local_score,
                    "global_score": global_score,
                    "fold": split.fold,
                    "task": task,
                    "horizon": config.horizon,
                    "local_model": model,
                    "global_model": model,
                }))
                max_target_end = pd.to_datetime(
                    confirm_frame.iloc[local_train_idx]["target_end_date"]
                ).max()
                audit_rows.append({
                    "task": task,
                    "fold": split.fold,
                    "scope": "local",
                    "ticker": ticker,
                    "train_rows": len(local_train_idx),
                    "train_assets": 1,
                    "max_target_end_date": max_target_end,
                    "test_start": test_start,
                    "purge_valid": bool(max_target_end < test_start),
                    "confirmation_rows_in_fit": len(local_train_idx),
                })
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(audit_rows)


def run_locked_scope_confirmation(
    training_data: Mapping[str, pd.DataFrame],
    confirmation_data: Mapping[str, pd.DataFrame],
    *,
    config: ScopeConfirmationConfig,
    output_base: str | Path = "reports",
    failures: pd.DataFrame | None = None,
) -> dict[str, object]:
    config.validate()
    if set(training_data) != set(config.training_universe):
        raise ValueError("Training data keys must exactly match the locked training universe.")
    if set(confirmation_data) != set(config.confirmation_universe):
        raise ValueError("Confirmation data keys must exactly match the locked confirmation universe.")
    predictions, audit = _fit_predictions(training_data, confirmation_data, config)
    if not bool(audit["purge_valid"].all()):
        raise AssertionError("Target-maturity purge audit failed.")

    primary_rows: list[dict] = []
    for index, (task, group) in enumerate(predictions.groupby("task", sort=True)):
        row = date_clustered_inference(
            group,
            task=str(task),
            horizon=config.horizon,
            bootstrap_repetitions=config.bootstrap_repetitions,
            random_state=config.random_state + index,
        )
        row["analysis_status"] = config.status
        primary_rows.append(row)
    primary = pd.DataFrame(primary_rows)
    primary["primary_hac_qvalue"] = benjamini_hochberg(
        primary["date_hac_pvalue_global_better"].to_numpy(dtype=float)
    )
    primary["confirmed"] = (
        (primary["primary_hac_qvalue"] < 0.05)
        & (primary["bootstrap_improvement_ci_low_pct"] > 0.0)
    )

    assets = _asset_results(predictions, config)
    folds = fold_stability(predictions)
    phases = phase_stability(predictions)
    regions = regional_stability(predictions)
    summary_rows: list[dict] = []
    for _, result in primary.iterrows():
        task = str(result["task"])
        task_assets = assets[assets["task"] == task]
        task_folds = folds[folds["task"] == task]
        task_phases = phases[phases["task"] == task]
        wins = int(task_assets["global_win"].sum())
        summary_rows.append({
            **result.to_dict(),
            "asset_win_count": wins,
            "asset_win_rate": wins / len(task_assets),
            "asset_win_sign_test_p": exact_sign_test_greater(wins, len(task_assets)),
            "fold_win_count": int(task_folds["global_win"].sum()),
            "fold_count": len(task_folds),
            "phase_win_count": int(task_phases["global_win"].sum()),
            "phase_count": len(task_phases),
        })
    summary = pd.DataFrame(summary_rows)

    protocol = config.as_dict()
    protocol_hash = sha256(
        json.dumps(protocol, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    run_dir = unique_run_directory(base=output_base, experiment="m14", label="locked-scope-confirmation")
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    primary.to_csv(run_dir / "primary_results.csv", index=False)
    summary.to_csv(run_dir / "summary.csv", index=False)
    assets.to_csv(run_dir / "asset_results.csv", index=False)
    folds.to_csv(run_dir / "fold_results.csv", index=False)
    phases.to_csv(run_dir / "phase_results.csv", index=False)
    regions.to_csv(run_dir / "regional_results.csv", index=False)
    audit.to_csv(run_dir / "training_audit.csv", index=False)
    (failures if failures is not None else pd.DataFrame(columns=["ticker", "stage", "error"])).to_csv(
        run_dir / "failures.csv", index=False
    )
    manifest = {
        **protocol,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": protocol_hash,
        "model_pairs": {"regression": "ridge", "classification": "logistic"},
        "ticker_identity_used": False,
        "primary_multiplicity_family": "two locked task hypotheses",
        "decision_rule": (
            "one-sided date-clustered HAC BH q<0.05 AND 95% circular date-block "
            "bootstrap improvement CI lower bound > 0"
        ),
        "results_available": True,
        "notes": [
            "The pooled global fit uses only the original M13 universe.",
            "Confirmation assets never enter a pooled global fit.",
            "Local fits use only the target confirmation asset and matured labels.",
            "No tuning, phase selection, asset substitution or horizon selection is performed.",
            "Economic performance is outside this scope-loss confirmation contract.",
        ],
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    write_scope_report(
        run_dir / "report.md",
        title="M14 — Locked Independent Scope Confirmation",
        summary=summary,
        inference=primary,
        confirmatory=True,
    )
    return {
        "run_dir": str(run_dir),
        "summary": summary,
        "primary_results": primary,
        "asset_results": assets,
        "fold_results": folds,
        "phase_results": phases,
        "regional_results": regions,
        "training_audit": audit,
        "predictions": predictions,
    }
