from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd

from stock_ml_lab.classification.bootstrap import classification_block_bootstrap
from stock_ml_lab.classification.nested import run_classification_comparison
from stock_ml_lab.confirmation.config import (
    DISCOVERY_TICKER,
    LOCKED_CLASSIFICATION_MODEL,
    LOCKED_FEATURE_SET,
    LOCKED_HORIZONS,
    LOCKED_REGRESSION_MODEL,
)
from stock_ml_lab.confirmation.phases import (
    classification_phase_table,
    regression_phase_table,
    summarize_phase_stability,
)
from stock_ml_lab.confirmation.statistics import (
    binary_brier_losses,
    binary_log_losses,
    hac_loss_differential_test,
)
from stock_ml_lab.research.dataset import build_research_dataset
from stock_ml_lab.robustness.assets import AssetSpec
from stock_ml_lab.robustness.multiple_testing import (
    benjamini_hochberg,
    exact_sign_test_greater,
)
from stock_ml_lab.tuning.nested import run_tuning_comparison
from stock_ml_lab.validation.bootstrap import forecast_block_bootstrap
from stock_ml_lab.validation.forecast_tests import diebold_mariano_vs_benchmark


@dataclass(frozen=True)
class LockedConfirmationResult:
    asset_results: pd.DataFrame
    candidate_summary: pd.DataFrame
    phase_results: pd.DataFrame
    fold_results: pd.DataFrame
    dataset_summary: pd.DataFrame
    failures: pd.DataFrame
    manifest: dict

    def save(self, output_dir: str | Path) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        self.asset_results.to_csv(path / "asset_results.csv", index=False)
        self.candidate_summary.to_csv(path / "candidate_summary.csv", index=False)
        self.phase_results.to_csv(path / "phase_results.csv", index=False)
        self.fold_results.to_csv(path / "fold_results.csv", index=False)
        self.dataset_summary.to_csv(path / "dataset_summary.csv", index=False)
        self.failures.to_csv(path / "failures.csv", index=False)
        (path / "manifest.json").write_text(
            json.dumps(self.manifest, indent=2, default=str), encoding="utf-8"
        )


def _classification_asset_result(
    *,
    asset: AssetSpec,
    bundle,
    outer_splits: int,
    inner_splits: int,
    test_size: int | None,
    n_trials: int,
    n_bootstrap: int,
    block_length: int,
    random_state: int,
):
    dataset = bundle.classification_dataset(LOCKED_FEATURE_SET)
    comparison = run_classification_comparison(
        dataset,
        models=(LOCKED_CLASSIFICATION_MODEL,),
        outer_splits=outer_splits,
        inner_splits=inner_splits,
        gap=bundle.gap,
        test_size=test_size,
        n_trials=n_trials,
        random_state=random_state,
    )
    result = comparison.tuned_results[0]
    prior = comparison.prior
    summary = comparison.summary_frame().loc[result.model_name]

    effective_block = max(block_length, bundle.horizon)
    bootstrap = classification_block_bootstrap(
        result.actuals,
        result.probabilities,
        prior.probabilities,
        n_bootstrap=n_bootstrap,
        block_length=min(effective_block, len(result.actuals)),
        confidence=0.95,
        random_state=random_state,
    )
    log_ci = bootstrap.loc["log_loss_improvement_pct"]
    brier_ci = bootstrap.loc["brier_improvement_pct"]
    auc_ci = bootstrap.loc["roc_auc_excess_pp"]

    hac_lags = max(0, bundle.horizon - 1)
    log_test = hac_loss_differential_test(
        binary_log_losses(result.actuals, result.probabilities),
        binary_log_losses(result.actuals, prior.probabilities),
        lags=hac_lags,
    )
    brier_test = hac_loss_differential_test(
        binary_brier_losses(result.actuals, result.probabilities),
        binary_brier_losses(result.actuals, prior.probabilities),
        lags=hac_lags,
    )

    phase = classification_phase_table(
        full_index=dataset.X.index,
        actual=result.actuals,
        model_probability=result.probabilities,
        prior_probability=prior.probabilities,
        horizon=bundle.horizon,
    )
    phase_summary = summarize_phase_stability(
        phase,
        primary_column="log_loss_improvement_vs_prior_pct",
    )

    row = {
        "ticker": asset.ticker,
        "market": asset.market,
        "asset_type": asset.asset_type,
        "task": "classification",
        "horizon": bundle.horizon,
        "gap": bundle.gap,
        "feature_set": LOCKED_FEATURE_SET,
        "model_key": LOCKED_CLASSIFICATION_MODEL,
        "model": result.model_name,
        "n_predictions": len(result.probabilities),
        **result.metrics.to_dict(),
        "prior_log_loss": prior.metrics.log_loss,
        "prior_brier": prior.metrics.brier,
        "log_loss_improvement_vs_prior_pct": summary[
            "log_loss_improvement_vs_prior_pct"
        ],
        "brier_improvement_vs_prior_pct": summary[
            "brier_improvement_vs_prior_pct"
        ],
        "primary_improvement_pct": summary["log_loss_improvement_vs_prior_pct"],
        "primary_hac_test": "log_loss",
        "primary_hac_statistic": log_test.statistic,
        "primary_hac_p_model_better": log_test.pvalue_model_better,
        "secondary_hac_test": "brier",
        "secondary_hac_statistic": brier_test.statistic,
        "secondary_hac_p_model_better": brier_test.pvalue_model_better,
        "primary_bootstrap_lower_pct": log_ci["lower"],
        "primary_bootstrap_upper_pct": log_ci["upper"],
        "secondary_bootstrap_lower_pct": brier_ci["lower"],
        "secondary_bootstrap_upper_pct": brier_ci["upper"],
        "roc_auc_excess_bootstrap_lower_pp": auc_ci["lower"],
        "roc_auc_excess_bootstrap_upper_pp": auc_ci["upper"],
        "hac_lags": hac_lags,
        "bootstrap_block_length": effective_block,
        **phase_summary,
    }

    fold = result.fold_frame().reset_index()
    fold.insert(0, "model", result.model_name)
    fold.insert(0, "task", "classification")
    fold.insert(0, "horizon", bundle.horizon)
    fold.insert(0, "ticker", asset.ticker)

    phase.insert(0, "model", result.model_name)
    phase.insert(0, "task", "classification")
    phase.insert(0, "horizon", bundle.horizon)
    phase.insert(0, "ticker", asset.ticker)
    return row, fold, phase


def _regression_asset_result(
    *,
    asset: AssetSpec,
    bundle,
    outer_splits: int,
    inner_splits: int,
    test_size: int | None,
    n_trials: int,
    n_bootstrap: int,
    block_length: int,
    random_state: int,
):
    dataset = bundle.regression_dataset(LOCKED_FEATURE_SET)
    comparison = run_tuning_comparison(
        dataset,
        models=(LOCKED_REGRESSION_MODEL,),
        outer_splits=outer_splits,
        inner_splits=inner_splits,
        gap=bundle.gap,
        test_size=test_size,
        n_trials=n_trials,
        random_state=random_state,
    )
    result = comparison.tuned_results[0]
    zero = comparison.zero_baseline
    summary = comparison.summary_frame().loc[result.model_name]

    effective_block = max(block_length, bundle.horizon)
    bootstrap = forecast_block_bootstrap(
        result.actuals,
        result.predictions,
        zero.predictions,
        n_bootstrap=n_bootstrap,
        block_length=min(effective_block, len(result.actuals)),
        confidence=0.95,
        random_state=random_state,
    )
    rmse_ci = bootstrap["rmse_improvement_vs_benchmark_pct"]
    mae_ci = bootstrap["mae_improvement_vs_benchmark_pct"]

    hac_lags = max(0, bundle.horizon - 1)
    dm_mse = diebold_mariano_vs_benchmark(
        result.actuals,
        result.predictions,
        zero.predictions,
        criterion="mse",
        lags=hac_lags,
    )
    dm_mae = diebold_mariano_vs_benchmark(
        result.actuals,
        result.predictions,
        zero.predictions,
        criterion="mae",
        lags=hac_lags,
    )

    phase = regression_phase_table(
        full_index=dataset.X.index,
        actual=result.actuals,
        model_forecast=result.predictions,
        zero_forecast=zero.predictions,
        horizon=bundle.horizon,
    )
    phase_summary = summarize_phase_stability(
        phase,
        primary_column="rmse_improvement_vs_zero_pct",
    )

    row = {
        "ticker": asset.ticker,
        "market": asset.market,
        "asset_type": asset.asset_type,
        "task": "regression",
        "horizon": bundle.horizon,
        "gap": bundle.gap,
        "feature_set": LOCKED_FEATURE_SET,
        "model_key": LOCKED_REGRESSION_MODEL,
        "model": result.model_name,
        "n_predictions": len(result.predictions),
        **result.metrics.to_dict(),
        "zero_rmse": zero.metrics.rmse,
        "zero_mae": zero.metrics.mae,
        "rmse_improvement_vs_zero_pct": summary["rmse_improvement_vs_zero_pct"],
        "mae_improvement_vs_zero_pct": summary["mae_improvement_vs_zero_pct"],
        "primary_improvement_pct": summary["rmse_improvement_vs_zero_pct"],
        "primary_hac_test": "mse",
        "primary_hac_statistic": dm_mse.statistic,
        "primary_hac_p_model_better": dm_mse.pvalue_model_better,
        "secondary_hac_test": "mae",
        "secondary_hac_statistic": dm_mae.statistic,
        "secondary_hac_p_model_better": dm_mae.pvalue_model_better,
        "primary_bootstrap_lower_pct": rmse_ci.lower,
        "primary_bootstrap_upper_pct": rmse_ci.upper,
        "secondary_bootstrap_lower_pct": mae_ci.lower,
        "secondary_bootstrap_upper_pct": mae_ci.upper,
        "hac_lags": hac_lags,
        "bootstrap_block_length": effective_block,
        **phase_summary,
    }

    fold = result.fold_frame().reset_index()
    fold.insert(0, "model", result.model_name)
    fold.insert(0, "task", "regression")
    fold.insert(0, "horizon", bundle.horizon)
    fold.insert(0, "ticker", asset.ticker)

    phase.insert(0, "model", result.model_name)
    phase.insert(0, "task", "regression")
    phase.insert(0, "horizon", bundle.horizon)
    phase.insert(0, "ticker", asset.ticker)
    return row, fold, phase


def _apply_fdr(asset_results: pd.DataFrame) -> pd.DataFrame:
    out = asset_results.copy()
    out["primary_hac_q_fdr"] = np.nan
    out["secondary_hac_q_fdr"] = np.nan

    for (_, _), idx in out.groupby(["task", "horizon"]).groups.items():
        locs = list(idx)
        out.loc[locs, "primary_hac_q_fdr"] = benjamini_hochberg(
            out.loc[locs, "primary_hac_p_model_better"].to_numpy(dtype=float)
        )
        out.loc[locs, "secondary_hac_q_fdr"] = benjamini_hochberg(
            out.loc[locs, "secondary_hac_p_model_better"].to_numpy(dtype=float)
        )

    out["primary_hac_fdr_significant_05"] = out["primary_hac_q_fdr"] < 0.05
    out["secondary_hac_fdr_significant_05"] = out["secondary_hac_q_fdr"] < 0.05
    out["primary_bootstrap_positive_95"] = out["primary_bootstrap_lower_pct"] > 0.0
    out["primary_win"] = out["primary_improvement_pct"] > 0.0
    return out


def _candidate_summary(asset_results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (task, horizon), group in asset_results.groupby(["task", "horizon"], sort=True):
        n = len(group)
        wins = int(group["primary_win"].sum())
        row = {
            "task": task,
            "horizon": int(horizon),
            "feature_set": LOCKED_FEATURE_SET,
            "model": group["model"].iloc[0],
            "n_assets": n,
            "primary_win_count": wins,
            "primary_win_rate": wins / n,
            "primary_win_sign_test_p": exact_sign_test_greater(wins, n),
            "mean_primary_improvement_pct": group["primary_improvement_pct"].mean(),
            "median_primary_improvement_pct": group["primary_improvement_pct"].median(),
            "primary_hac_fdr_sig_count": int(
                group["primary_hac_fdr_significant_05"].sum()
            ),
            "primary_bootstrap_positive_count": int(
                group["primary_bootstrap_positive_95"].sum()
            ),
            "mean_phase_win_rate": group["phase_win_rate"].mean(),
            "median_phase_primary_improvement_pct": group[
                "phase_primary_median"
            ].median(),
        }
        if task == "classification":
            row["mean_roc_auc"] = group["roc_auc"].mean()
            row["median_roc_auc"] = group["roc_auc"].median()
            row["mean_directional_accuracy"] = np.nan
        else:
            row["mean_roc_auc"] = np.nan
            row["median_roc_auc"] = np.nan
            row["mean_directional_accuracy"] = group["directional_accuracy"].mean()
        rows.append(row)
    return pd.DataFrame(rows)


def run_locked_confirmation(
    assets: tuple[AssetSpec, ...],
    *,
    start: str,
    end: str | None,
    outer_splits: int = 5,
    inner_splits: int = 3,
    test_size: int | None = 252,
    n_trials: int = 10,
    n_bootstrap: int = 1000,
    block_length: int = 20,
    random_state: int = 42,
    allow_discovery_asset: bool = False,
    continue_on_error: bool = True,
    progress: bool = True,
) -> LockedConfirmationResult:
    if not assets:
        raise ValueError("At least one confirmation asset is required.")
    if any(asset.ticker == DISCOVERY_TICKER for asset in assets) and not allow_discovery_asset:
        raise ValueError(
            f"{DISCOVERY_TICKER} is the discovery series and cannot enter the default "
            "M8.1 confirmation sample."
        )
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap must be >= 100.")
    if block_length < 1:
        raise ValueError("block_length must be >= 1.")

    rows: list[dict] = []
    folds: list[pd.DataFrame] = []
    phases: list[pd.DataFrame] = []
    datasets: list[dict] = []
    failures: list[dict] = []

    for asset_idx, asset in enumerate(assets, start=1):
        if progress:
            print(f"\n[{asset_idx}/{len(assets)}] {asset.ticker} ({asset.market}, {asset.asset_type})")

        for horizon in LOCKED_HORIZONS:
            if progress:
                print(f"  Horizon {horizon}d | locked legacy + Logistic/Ridge | gap={horizon}")
            try:
                bundle = build_research_dataset(
                    ticker=asset.ticker,
                    start=start,
                    end=end,
                    horizon=horizon,
                )
                datasets.append(
                    {
                        "ticker": asset.ticker,
                        "market": asset.market,
                        "asset_type": asset.asset_type,
                        "benchmark_ticker": bundle.benchmark_ticker,
                        "horizon": horizon,
                        "gap": bundle.gap,
                        "rows": len(bundle.X_all),
                        "start": bundle.X_all.index.min(),
                        "end": bundle.X_all.index.max(),
                        "positive_rate": bundle.directions.mean(),
                        "mean_forward_return": bundle.forward_returns.mean(),
                        "forward_return_std": bundle.forward_returns.std(),
                    }
                )

                seed = random_state + asset_idx * 10000 + horizon * 100
                class_row, class_fold, class_phase = _classification_asset_result(
                    asset=asset,
                    bundle=bundle,
                    outer_splits=outer_splits,
                    inner_splits=inner_splits,
                    test_size=test_size,
                    n_trials=n_trials,
                    n_bootstrap=n_bootstrap,
                    block_length=block_length,
                    random_state=seed + 1,
                )
                reg_row, reg_fold, reg_phase = _regression_asset_result(
                    asset=asset,
                    bundle=bundle,
                    outer_splits=outer_splits,
                    inner_splits=inner_splits,
                    test_size=test_size,
                    n_trials=n_trials,
                    n_bootstrap=n_bootstrap,
                    block_length=block_length,
                    random_state=seed + 2,
                )

                rows.extend([class_row, reg_row])
                folds.extend([class_fold, reg_fold])
                phases.extend([class_phase, reg_phase])

                if progress:
                    print(
                        f"    Logistic: ΔLogLoss={class_row['primary_improvement_pct']:+.2f}% | "
                        f"AUC={class_row['roc_auc']:.3f} | "
                        f"phase wins={class_row['phase_win_count']}/{class_row['phase_count']}"
                    )
                    print(
                        f"    Ridge:    ΔRMSE={reg_row['primary_improvement_pct']:+.2f}% | "
                        f"DA={100*reg_row['directional_accuracy']:.2f}% | "
                        f"phase wins={reg_row['phase_win_count']}/{reg_row['phase_count']}"
                    )
            except Exception as exc:
                failures.append(
                    {
                        "ticker": asset.ticker,
                        "market": asset.market,
                        "asset_type": asset.asset_type,
                        "horizon": horizon,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
                if progress:
                    print(f"    FAILED: {type(exc).__name__}: {exc}")
                if not continue_on_error:
                    raise

    asset_frame = pd.DataFrame(rows)
    if not asset_frame.empty:
        asset_frame = _apply_fdr(asset_frame)
        asset_frame = asset_frame.sort_values(["task", "horizon", "ticker"]).reset_index(drop=True)
    candidate_summary = _candidate_summary(asset_frame) if not asset_frame.empty else pd.DataFrame()
    phase_frame = pd.concat(phases, ignore_index=True) if phases else pd.DataFrame()
    fold_frame = pd.concat(folds, ignore_index=True) if folds else pd.DataFrame()
    dataset_frame = pd.DataFrame(datasets)
    failure_frame = pd.DataFrame(
        failures,
        columns=["ticker", "market", "asset_type", "horizon", "error_type", "error"],
    )

    manifest = {
        "milestone": "8.1",
        "purpose": "locked cross-asset confirmation and non-overlapping phase robustness",
        "discovery_ticker_excluded": DISCOVERY_TICKER,
        "assets": [asset.ticker for asset in assets],
        "locked_candidates": [
            {"task": "classification", "horizon": 10, "feature_set": "legacy", "model": "logistic"},
            {"task": "classification", "horizon": 20, "feature_set": "legacy", "model": "logistic"},
            {"task": "regression", "horizon": 10, "feature_set": "legacy", "model": "ridge"},
            {"task": "regression", "horizon": 20, "feature_set": "legacy", "model": "ridge"},
        ],
        "selection_rule": "candidates frozen from PETR4.SA M8 before viewing 10d/20d confirmation results on the other assets",
        "start": start,
        "end": end,
        "outer_splits": outer_splits,
        "inner_splits": inner_splits,
        "test_size": test_size,
        "trials": n_trials,
        "bootstrap_replications": n_bootstrap,
        "requested_block_length": block_length,
        "effective_block_length_rule": "max(requested block length, horizon)",
        "hac_lag_rule": "horizon - 1",
        "purge_rule": "gap equals horizon",
        "phase_rule": "phase = supervised row position mod horizon; within-phase target windows do not overlap",
        "multiplicity_rule": "Benjamini-Hochberg FDR separately within each locked task/horizon candidate across assets",
        "cross_asset_sign_test_note": "reported as a descriptive exact sign test; assets are not assumed economically independent",
        "failures": len(failure_frame),
    }

    return LockedConfirmationResult(
        asset_results=asset_frame,
        candidate_summary=candidate_summary,
        phase_results=phase_frame,
        fold_results=fold_frame,
        dataset_summary=dataset_frame,
        failures=failure_frame,
        manifest=manifest,
    )
