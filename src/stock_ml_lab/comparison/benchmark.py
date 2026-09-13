from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from stock_ml_lab.api.registry import build_local_estimator
from stock_ml_lab.global_model.dataset import GlobalPanelBundle, build_global_panel_from_ohlcv
from stock_ml_lab.global_model.models import build_global_estimator
from stock_ml_lab.global_model.validation import panel_walk_forward_splits
from stock_ml_lab.reporting import unique_run_directory
from stock_ml_lab.robustness.multiple_testing import benjamini_hochberg, exact_sign_test_greater
from stock_ml_lab.validation.forecast_tests import diebold_mariano_vs_benchmark
from stock_ml_lab.features.technical import FEATURE_COLUMNS
from stock_ml_lab.comparison.diagnostics import build_scope_diagnostics
from stock_ml_lab.comparison.report import write_scope_report

SUPPORTED_PAIRS = {
    "regression": ("ridge",),
    "classification": ("logistic",),
}


@dataclass(frozen=True)
class ComparisonConfig:
    task: str
    horizon: int
    local_model: str
    global_model: str


def _loss(y: np.ndarray, score: np.ndarray, task: str) -> np.ndarray:
    if task == "regression":
        return (y - score) ** 2
    p = np.clip(score, 1e-8, 1 - 1e-8)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def _metric(y: np.ndarray, score: np.ndarray, task: str) -> dict[str, float]:
    if task == "regression":
        rmse = float(np.sqrt(np.mean((y - score) ** 2)))
        mae = float(np.mean(np.abs(y - score)))
        da = float(np.mean((y > 0) == (score > 0)))
        return {"rmse": rmse, "mae": mae, "directional_accuracy": da}
    p = np.clip(score, 1e-8, 1 - 1e-8)
    pred = (p >= 0.5).astype(int)
    ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
    brier = float(np.mean((y - p) ** 2))
    return {
        "log_loss": ll,
        "brier": brier,
        "accuracy": float(np.mean(pred == y)),
    }


def _improvement(global_metric: float, local_metric: float) -> float:
    if local_metric == 0:
        return float("nan")
    return 100.0 * (local_metric - global_metric) / local_metric


def _predict(estimator, X: pd.DataFrame, task: str) -> np.ndarray:
    if task == "regression":
        return np.asarray(estimator.predict(X), dtype=float)
    probabilities = np.asarray(estimator.predict_proba(X), dtype=float)
    classes = list(estimator.classes_)
    if 1 not in classes:
        raise ValueError("Classifier does not expose positive class 1.")
    return probabilities[:, classes.index(1)]


def _aggregate_asset_rows(predictions: pd.DataFrame, *, task: str, horizon: int) -> pd.DataFrame:
    rows: list[dict] = []
    for ticker, group in predictions.groupby("ticker", sort=True):
        y = group["y"].to_numpy(dtype=float)
        local = group["local_score"].to_numpy(dtype=float)
        global_ = group["global_score"].to_numpy(dtype=float)
        lm = _metric(y, local, task)
        gm = _metric(y, global_, task)
        primary_key = "rmse" if task == "regression" else "log_loss"
        rows.append(
            {
                "ticker": ticker,
                "task": task,
                "horizon": horizon,
                "local_model": group["local_model"].iloc[0],
                "global_model": group["global_model"].iloc[0],
                "n": len(group),
                "local_primary": lm[primary_key],
                "global_primary": gm[primary_key],
                "global_improvement_vs_local_pct": _improvement(gm[primary_key], lm[primary_key]),
                **{f"local_{k}": v for k, v in lm.items()},
                **{f"global_{k}": v for k, v in gm.items()},
            }
        )
    return pd.DataFrame(rows)


def run_comparison(
    bundle: GlobalPanelBundle,
    *,
    task: str,
    local_model: str,
    global_model: str,
    n_splits: int = 5,
    test_size_dates: int = 252,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if task not in SUPPORTED_PAIRS:
        raise ValueError(f"Unsupported task: {task}")
    if local_model not in SUPPORTED_PAIRS[task]:
        raise ValueError(f"M13 comparison currently supports local {SUPPORTED_PAIRS[task]}")
    expected_global = SUPPORTED_PAIRS[task][0]
    if global_model != expected_global:
        raise ValueError(f"M13 requires same-family global model '{expected_global}'.")

    folds = panel_walk_forward_splits(bundle, n_splits=n_splits, test_size_dates=test_size_dates)
    X = bundle.X
    y = bundle.regression_y if task == "regression" else bundle.classification_y
    predictions: list[pd.DataFrame] = []
    fold_rows: list[dict] = []

    for split in folds:
        train = bundle.frame.iloc[split.train_index]
        test = bundle.frame.iloc[split.test_index]
        global_estimator = build_global_estimator(task, global_model)
        global_estimator.fit(X.iloc[split.train_index], y.iloc[split.train_index])
        global_score_all = _predict(global_estimator, X.iloc[split.test_index], task)

        for ticker in bundle.tickers:
            train_mask = (train["ticker"] == ticker).to_numpy()
            test_mask = (test["ticker"] == ticker).to_numpy()
            if not test_mask.any() or train_mask.sum() < 60:
                continue
            train_indices = split.train_index[train_mask]
            test_indices = split.test_index[test_mask]
            local_estimator = build_local_estimator(task, local_model, random_state=42)
            local_estimator.fit(X.iloc[train_indices], y.iloc[train_indices])
            local_score = _predict(local_estimator, X.iloc[test_indices], task)
            global_score = global_score_all[test_mask]
            y_test = y.iloc[test_indices].to_numpy(dtype=float)
            lm = _metric(y_test, local_score, task)
            gm = _metric(y_test, global_score, task)
            primary = "rmse" if task == "regression" else "log_loss"
            fold_rows.append(
                {
                    "task": task,
                    "horizon": bundle.horizon,
                    "local_model": local_model,
                    "global_model": global_model,
                    "fold": split.fold,
                    "ticker": ticker,
                    "n": len(test_indices),
                    "local_primary": lm[primary],
                    "global_primary": gm[primary],
                    "global_improvement_vs_local_pct": _improvement(gm[primary], lm[primary]),
                    "test_start": split.test_start.date(),
                    "test_end": split.test_end.date(),
                }
            )
            predictions.append(
                pd.DataFrame(
                    {
                        "ticker": ticker,
                        "date": test.loc[test_mask, "date"].to_numpy(),
                        "y": y_test,
                        "local_score": local_score,
                        "global_score": global_score,
                        "fold": split.fold,
                        "task": task,
                        "horizon": bundle.horizon,
                        "local_model": local_model,
                        "global_model": global_model,
                    }
                )
            )

    pred = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    assets = _aggregate_asset_rows(pred, task=task, horizon=bundle.horizon) if not pred.empty else pd.DataFrame()
    folds = pd.DataFrame(fold_rows)

    if not assets.empty:
        stats_rows: list[dict] = []
        primary = "mse" if task == "regression" else None
        for ticker, group in pred.groupby("ticker", sort=True):
            yv = group["y"].to_numpy(dtype=float)
            if task == "regression":
                test = diebold_mariano_vs_benchmark(
                    yv,
                    group["global_score"].to_numpy(dtype=float),
                    group["local_score"].to_numpy(dtype=float),
                    criterion=primary,
                    lags=max(bundle.horizon - 1, 1),
                )
                stats_rows.append({
                    "ticker": ticker, "task": task, "horizon": bundle.horizon,
                    "criterion": "mse", "statistic": test.statistic,
                    "pvalue_two_sided": test.pvalue_two_sided,
                    "pvalue_global_better": test.pvalue_model_better,
                    "hac_lags": test.lags,
                })
            else:
                # For log loss, use a HAC mean-loss test with the same convention as DM.
                local_loss = _loss(yv, group["local_score"].to_numpy(dtype=float), task)
                global_loss = _loss(yv, group["global_score"].to_numpy(dtype=float), task)
                d = global_loss - local_loss
                n = len(d)
                lags = max(bundle.horizon - 1, 1)
                centered = d - d.mean()
                lr = float(np.dot(centered, centered) / n)
                for lag in range(1, min(lags, n - 1) + 1):
                    cov = float(np.dot(centered[lag:], centered[:-lag]) / n)
                    lr += 2 * (1 - lag / (lags + 1.0)) * cov
                se = float(np.sqrt(max(lr, 0.0) / n)) if lr > 0 else float("nan")
                stat = float(d.mean() / se) if np.isfinite(se) and se > 0 else float("nan")
                from math import erf, sqrt
                p_two = float(2 * (1 - 0.5 * (1 + erf(abs(stat) / sqrt(2))))) if np.isfinite(stat) else float("nan")
                p_better = float(0.5 * (1 + erf(stat / sqrt(2)))) if np.isfinite(stat) else float("nan")
                stats_rows.append({
                    "ticker": ticker, "task": task, "horizon": bundle.horizon,
                    "criterion": "log_loss", "statistic": stat,
                    "pvalue_two_sided": p_two,
                    "pvalue_global_better": p_better,
                    "hac_lags": lags,
                })
        assets = assets.merge(pd.DataFrame(stats_rows), on=["ticker", "task", "horizon"], how="left")
        assets["qvalue_global_better"] = (
            assets.groupby(["task", "horizon"])["pvalue_global_better"]
            .transform(lambda s: benjamini_hochberg(s.to_numpy()))
        )

    return folds, assets, pred


def run_local_vs_global_benchmark(
    data_by_ticker: Mapping[str, pd.DataFrame],
    *,
    horizons: tuple[int, ...] = (1, 5, 10, 20),
    output_base: str | Path = "reports",
    n_splits: int = 5,
    test_size_dates: int = 252,
    bootstrap_repetitions: int = 2_000,
    random_state: int = 42,
) -> dict[str, object]:
    all_folds: list[pd.DataFrame] = []
    all_assets: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    for horizon in horizons:
        bundle = build_global_panel_from_ohlcv(data_by_ticker, horizon=horizon)
        for task, local_model, global_model in (
            ("regression", "ridge", "ridge"),
            ("classification", "logistic", "logistic"),
        ):
            folds, assets, pred = run_comparison(
                bundle,
                task=task,
                local_model=local_model,
                global_model=global_model,
                n_splits=n_splits,
                test_size_dates=test_size_dates,
            )
            all_folds.append(folds)
            all_assets.append(assets)
            all_predictions.append(pred)

    folds = pd.concat(all_folds, ignore_index=True) if all_folds else pd.DataFrame()
    assets = pd.concat(all_assets, ignore_index=True) if all_assets else pd.DataFrame()
    predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()

    summary_rows = []
    if not assets.empty:
        for (task, horizon), group in assets.groupby(["task", "horizon"], sort=True):
            values = group["global_improvement_vs_local_pct"].dropna()
            summary_rows.append({
                "task": task,
                "horizon": int(horizon),
                "assets": int(group["ticker"].nunique()),
                "global_win_count": int((values > 0).sum()),
                "global_win_rate": float((values > 0).mean()) if len(values) else float("nan"),
                "mean_global_improvement_vs_local_pct": float(values.mean()) if len(values) else float("nan"),
                "median_global_improvement_vs_local_pct": float(values.median()) if len(values) else float("nan"),
                "min_global_improvement_vs_local_pct": float(values.min()) if len(values) else float("nan"),
                "max_global_improvement_vs_local_pct": float(values.max()) if len(values) else float("nan"),
                "asset_level_hac_p_lt_0_05": int((group["pvalue_global_better"] < 0.05).sum()),
                "asset_level_hac_fdr_q_lt_0_05": int((group["qvalue_global_better"] < 0.05).sum()),
                "global_win_sign_test_p": exact_sign_test_greater(int((values > 0).sum()), len(values)) if len(values) else float("nan"),
            })
    summary = pd.DataFrame(summary_rows)
    if not summary.empty:
        summary["global_win_sign_test_q_posthoc_across_configurations"] = (
            benjamini_hochberg(summary["global_win_sign_test_p"].to_numpy(dtype=float))
        )
    run_dir = unique_run_directory(base=output_base, experiment="m13", label="local-vs-global")
    run_dir.mkdir(parents=True, exist_ok=True)
    folds.to_csv(run_dir / "fold_results.csv", index=False)
    assets.to_csv(run_dir / "asset_results.csv", index=False)
    predictions.to_csv(run_dir / "predictions.csv", index=False)
    summary.to_csv(run_dir / "summary.csv", index=False)
    diagnostics = build_scope_diagnostics(
        predictions,
        bootstrap_repetitions=bootstrap_repetitions,
        random_state=random_state,
        analysis_status="post_hoc_robustness_diagnostic",
    )
    diagnostics["summary"].to_csv(run_dir / "posthoc_summary.csv", index=False)
    diagnostics["date_inference"].to_csv(run_dir / "posthoc_date_inference.csv", index=False)
    diagnostics["fold_stability"].to_csv(run_dir / "posthoc_fold_stability.csv", index=False)
    diagnostics["phase_stability"].to_csv(run_dir / "posthoc_phase_stability.csv", index=False)
    diagnostics["regional_stability"].to_csv(run_dir / "posthoc_regional_stability.csv", index=False)
    write_scope_report(
        run_dir / "report.md",
        title="M13 — Local vs Global Benchmark",
        summary=diagnostics["summary"],
        inference=diagnostics["date_inference"],
        confirmatory=False,
    )
    manifest = {
        "milestone": "M13",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope_comparison": "local_vs_global_same_model_family",
        "universe": sorted(data_by_ticker),
        "horizons": list(horizons),
        "pairs": {"regression": "ridge", "classification": "logistic"},
        "features": list(FEATURE_COLUMNS),
        "ticker_identity_used": False,
        "n_splits": n_splits,
        "test_size_dates": test_size_dates,
        "posthoc_diagnostics": {
            "status": "post_hoc_robustness_diagnostic",
            "bootstrap_repetitions": bootstrap_repetitions,
            "date_clustered_hac": True,
            "block_length_rule": "max(horizon, 1)",
        },
        "notes": [
            "Local and global models use identical OOS dates and the same feature contract.",
            "Global training uses all eligible assets; local training uses only the target ticker.",
            "Target maturity purge is enforced before every OOS fold.",
            "For h>1, HAC lags are at least h-1 because forward targets overlap.",
            "M13 compares scope while holding model family fixed; it does not compare Ridge against Logistic as a scope effect.",
            "Date-clustered HAC, block bootstrap, fold, phase and regional diagnostics are post hoc and are not preregistered confirmation.",
        ],
    }
    import json
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return {
        "run_dir": str(run_dir),
        "fold_results": folds,
        "asset_results": assets,
        "predictions": predictions,
        "summary": summary,
        "diagnostics": diagnostics,
    }
