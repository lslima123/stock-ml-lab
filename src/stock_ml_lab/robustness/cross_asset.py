from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import json

import numpy as np
import pandas as pd

from stock_ml_lab.dataset import build_dataset
from stock_ml_lab.robustness.assets import AssetSpec
from stock_ml_lab.robustness.multiple_testing import (
    benjamini_hochberg,
    exact_sign_test_greater,
)
from stock_ml_lab.tuning import run_tuning_comparison
from stock_ml_lab.validation import (
    backtest_predictions,
    buy_and_hold_backtest,
    diebold_mariano_vs_benchmark,
    forecast_block_bootstrap,
    pesaran_timmermann_test,
    select_thresholds_nested,
)


@dataclass(frozen=True)
class CrossAssetRun:
    asset_results: pd.DataFrame
    model_summary: pd.DataFrame
    fold_results: pd.DataFrame
    failures: pd.DataFrame
    manifest: dict

    def save(self, output_dir: str | Path) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

        self.asset_results.to_csv(path / "asset_results.csv", index=False)
        self.model_summary.to_csv(path / "model_summary.csv", index=False)
        self.fold_results.to_csv(path / "fold_results.csv", index=False)
        self.failures.to_csv(path / "failures.csv", index=False)
        (path / "manifest.json").write_text(
            json.dumps(self.manifest, indent=2, default=str),
            encoding="utf-8",
        )


def _single_asset_model_row(
    *,
    spec: AssetSpec,
    dataset,
    tuned,
    zero,
    bootstrap: int,
    block_length: int,
    dm_lags: int,
    cost_bps: float,
    thresholds: tuple[float, ...],
    strategy: str,
    threshold_objective: str,
    random_state: int,
    include_nested_threshold: bool,
) -> dict:
    actual = tuned.actuals
    forecast = tuned.predictions
    benchmark = zero.predictions.loc[forecast.index]

    zero_metrics = zero.metrics
    model_metrics = tuned.metrics

    rmse_improvement = (
        (zero_metrics.rmse - model_metrics.rmse) / zero_metrics.rmse * 100.0
    )
    mae_improvement = (
        (zero_metrics.mae - model_metrics.mae) / zero_metrics.mae * 100.0
    )

    dm_mse = diebold_mariano_vs_benchmark(
        actual,
        forecast,
        benchmark,
        criterion="mse",
        lags=dm_lags,
    )
    dm_mae = diebold_mariano_vs_benchmark(
        actual,
        forecast,
        benchmark,
        criterion="mae",
        lags=dm_lags,
    )
    pt = pesaran_timmermann_test(actual, forecast)

    intervals = forecast_block_bootstrap(
        actual,
        forecast,
        benchmark,
        n_bootstrap=bootstrap,
        block_length=min(block_length, len(actual)),
        confidence=0.95,
        random_state=random_state,
    )
    rmse_interval = intervals["rmse_improvement_vs_benchmark_pct"]

    buy_hold = buy_and_hold_backtest(actual, cost_bps=cost_bps)
    raw = backtest_predictions(
        actual,
        forecast,
        threshold=0.0,
        cost_bps=cost_bps,
        mode=strategy,
        name=f"{tuned.model_name} threshold=0",
    )

    nested_total_return = np.nan
    nested_sharpe = np.nan
    nested_max_drawdown = np.nan
    nested_exposure = np.nan
    nested_turnover = np.nan

    if include_nested_threshold:
        nested = select_thresholds_nested(
            dataset,
            tuned,
            thresholds=thresholds,
            cost_bps=cost_bps,
            mode=strategy,
            objective=threshold_objective,
            random_state=random_state,
        )
        nested_total_return = nested.backtest.metrics.total_return
        nested_sharpe = nested.backtest.metrics.sharpe
        nested_max_drawdown = nested.backtest.metrics.max_drawdown
        nested_exposure = nested.backtest.metrics.exposure
        nested_turnover = nested.backtest.metrics.annualized_turnover

    return {
        "ticker": spec.ticker,
        "market": spec.market,
        "asset_type": spec.asset_type,
        "model_key": tuned.model_key,
        "model": tuned.model_name,
        "dataset_rows": len(dataset.X),
        "n_predictions": len(forecast),
        "zero_mae": zero_metrics.mae,
        "zero_rmse": zero_metrics.rmse,
        "model_mae": model_metrics.mae,
        "model_rmse": model_metrics.rmse,
        "mae_improvement_vs_zero_pct": mae_improvement,
        "rmse_improvement_vs_zero_pct": rmse_improvement,
        "rmse_bootstrap_lower_pct": rmse_interval.lower,
        "rmse_bootstrap_upper_pct": rmse_interval.upper,
        "directional_accuracy": model_metrics.directional_accuracy,
        "pt_expected_accuracy": pt.expected_accuracy_under_independence,
        "dm_mse_p_model_better": dm_mse.pvalue_model_better,
        "dm_mae_p_model_better": dm_mae.pvalue_model_better,
        "pt_p_model_better": pt.pvalue_larger,
        "buy_hold_total_return": buy_hold.metrics.total_return,
        "buy_hold_sharpe": buy_hold.metrics.sharpe,
        "buy_hold_max_drawdown": buy_hold.metrics.max_drawdown,
        "strategy_total_return": raw.metrics.total_return,
        "strategy_sharpe": raw.metrics.sharpe,
        "strategy_max_drawdown": raw.metrics.max_drawdown,
        "strategy_annualized_turnover": raw.metrics.annualized_turnover,
        "strategy_total_cost_return": raw.metrics.total_cost_return,
        "nested_strategy_total_return": nested_total_return,
        "nested_strategy_sharpe": nested_sharpe,
        "nested_strategy_max_drawdown": nested_max_drawdown,
        "nested_strategy_exposure": nested_exposure,
        "nested_strategy_annualized_turnover": nested_turnover,
    }


def _fold_rows(spec: AssetSpec, tuned) -> list[dict]:
    rows = []
    for fold in tuned.folds:
        rows.append(
            {
                "ticker": spec.ticker,
                "market": spec.market,
                "asset_type": spec.asset_type,
                "model_key": tuned.model_key,
                "model": tuned.model_name,
                "fold": fold.fold,
                "train_start": fold.train_start,
                "train_end": fold.train_end,
                "test_start": fold.test_start,
                "test_end": fold.test_end,
                "n_train": fold.n_train,
                "n_test": fold.n_test,
                "best_inner_rmse": fold.best_inner_rmse,
                "mae": fold.metrics.mae,
                "rmse": fold.metrics.rmse,
                "directional_accuracy": fold.metrics.directional_accuracy,
                "best_params": json.dumps(fold.best_params, sort_keys=True),
            }
        )
    return rows


def _apply_multiple_testing(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()

    for source, target in (
        ("dm_mse_p_model_better", "dm_mse_q_fdr"),
        ("dm_mae_p_model_better", "dm_mae_q_fdr"),
        ("pt_p_model_better", "pt_q_fdr"),
    ):
        out[target] = np.nan
        for model_key, idx in out.groupby("model_key").groups.items():
            locs = list(idx)
            out.loc[locs, target] = benjamini_hochberg(
                out.loc[locs, source].to_numpy(dtype=float)
            )

    out["dm_mse_fdr_significant_05"] = out["dm_mse_q_fdr"] < 0.05
    out["dm_mae_fdr_significant_05"] = out["dm_mae_q_fdr"] < 0.05
    out["pt_fdr_significant_05"] = out["pt_q_fdr"] < 0.05
    out["rmse_bootstrap_positive_95"] = out["rmse_bootstrap_lower_pct"] > 0.0
    out["strategy_beats_buy_hold_return"] = (
        out["strategy_total_return"] > out["buy_hold_total_return"]
    )
    out["strategy_beats_buy_hold_sharpe"] = (
        out["strategy_sharpe"] > out["buy_hold_sharpe"]
    )
    out["nested_beats_buy_hold_return"] = (
        out["nested_strategy_total_return"] > out["buy_hold_total_return"]
    )
    out["nested_beats_buy_hold_sharpe"] = (
        out["nested_strategy_sharpe"] > out["buy_hold_sharpe"]
    )
    return out


def aggregate_cross_asset_results(asset_results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate robustness evidence separately for each tuned model."""
    if asset_results.empty:
        return pd.DataFrame()

    rows = []
    for model_key, group in asset_results.groupby("model_key", sort=False):
        n = len(group)
        rmse_wins = int((group["rmse_improvement_vs_zero_pct"] > 0.0).sum())
        rows.append(
            {
                "model_key": model_key,
                "model": group["model"].iloc[0],
                "n_assets": n,
                "rmse_win_count": rmse_wins,
                "rmse_win_rate": rmse_wins / n,
                "rmse_win_sign_test_p": exact_sign_test_greater(rmse_wins, n),
                "mean_rmse_improvement_pct": group[
                    "rmse_improvement_vs_zero_pct"
                ].mean(),
                "median_rmse_improvement_pct": group[
                    "rmse_improvement_vs_zero_pct"
                ].median(),
                "mean_directional_accuracy": group["directional_accuracy"].mean(),
                "median_directional_accuracy": group["directional_accuracy"].median(),
                "dm_mse_fdr_sig_count": int(
                    group["dm_mse_fdr_significant_05"].sum()
                ),
                "pt_fdr_sig_count": int(group["pt_fdr_significant_05"].sum()),
                "rmse_bootstrap_positive_count": int(
                    group["rmse_bootstrap_positive_95"].sum()
                ),
                "strategy_return_beats_bh_count": int(
                    group["strategy_beats_buy_hold_return"].sum()
                ),
                "strategy_sharpe_beats_bh_count": int(
                    group["strategy_beats_buy_hold_sharpe"].sum()
                ),
                "nested_return_beats_bh_count": int(
                    group["nested_beats_buy_hold_return"].sum()
                ),
                "nested_sharpe_beats_bh_count": int(
                    group["nested_beats_buy_hold_sharpe"].sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def run_cross_asset_study(
    assets: Iterable[AssetSpec],
    *,
    start: str,
    end: str | None,
    models: tuple[str, ...] = ("catboost",),
    outer_splits: int = 5,
    inner_splits: int = 3,
    gap: int = 1,
    test_size: int | None = 252,
    n_trials: int = 20,
    bootstrap: int = 1000,
    block_length: int = 20,
    dm_lags: int = 5,
    cost_bps: float = 10.0,
    thresholds: tuple[float, ...] = (0.0, 0.001, 0.0025, 0.005),
    strategy: str = "long_short",
    threshold_objective: str = "sharpe",
    random_state: int = 42,
    include_nested_threshold: bool = True,
    continue_on_error: bool = True,
    progress: bool = True,
) -> CrossAssetRun:
    rows: list[dict] = []
    folds: list[dict] = []
    failures: list[dict] = []
    asset_list = tuple(assets)

    for asset_idx, spec in enumerate(asset_list, start=1):
        if progress:
            print(
                f"[{asset_idx}/{len(asset_list)}] {spec.ticker} "
                f"({spec.market}, {spec.asset_type})"
            )

        try:
            dataset = build_dataset(
                ticker=spec.ticker,
                start=start,
                end=end,
                target="next_return",
            )
            comparison = run_tuning_comparison(
                dataset,
                models=models,
                outer_splits=outer_splits,
                inner_splits=inner_splits,
                gap=gap,
                test_size=test_size,
                n_trials=n_trials,
                random_state=random_state + asset_idx * 100,
            )

            for model_idx, tuned in enumerate(comparison.tuned_results):
                row = _single_asset_model_row(
                    spec=spec,
                    dataset=dataset,
                    tuned=tuned,
                    zero=comparison.zero_baseline,
                    bootstrap=bootstrap,
                    block_length=block_length,
                    dm_lags=dm_lags,
                    cost_bps=cost_bps,
                    thresholds=thresholds,
                    strategy=strategy,
                    threshold_objective=threshold_objective,
                    random_state=random_state + asset_idx * 1000 + model_idx,
                    include_nested_threshold=include_nested_threshold,
                )
                rows.append(row)
                folds.extend(_fold_rows(spec, tuned))

                if progress:
                    print(
                        f"    {tuned.model_name}: "
                        f"ΔRMSE={row['rmse_improvement_vs_zero_pct']:+.2f}% | "
                        f"DA={100*row['directional_accuracy']:.2f}% | "
                        f"DM p={row['dm_mse_p_model_better']:.3f}"
                    )

        except Exception as exc:
            failures.append(
                {
                    "ticker": spec.ticker,
                    "market": spec.market,
                    "asset_type": spec.asset_type,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            if progress:
                print(f"    FAILED: {type(exc).__name__}: {exc}")
            if not continue_on_error:
                raise

    asset_results = pd.DataFrame(rows)
    if not asset_results.empty:
        asset_results = _apply_multiple_testing(asset_results)
        asset_results = asset_results.sort_values(
            ["model_key", "rmse_improvement_vs_zero_pct"],
            ascending=[True, False],
        ).reset_index(drop=True)

    fold_results = pd.DataFrame(folds)
    failures_frame = pd.DataFrame(
        failures,
        columns=["ticker", "market", "asset_type", "error_type", "error"],
    )
    model_summary = aggregate_cross_asset_results(asset_results)

    manifest = {
        "milestone": 6,
        "purpose": "cross-asset robustness",
        "assets_requested": [spec.ticker for spec in asset_list],
        "models": list(models),
        "start": start,
        "end": end,
        "outer_splits": outer_splits,
        "inner_splits": inner_splits,
        "gap": gap,
        "test_size": test_size,
        "trials_per_outer_fold": n_trials,
        "bootstrap_replications": bootstrap,
        "block_length": block_length,
        "dm_lags": dm_lags,
        "cost_bps": cost_bps,
        "thresholds": list(thresholds),
        "strategy": strategy,
        "threshold_objective": threshold_objective,
        "nested_threshold_enabled": include_nested_threshold,
        "successful_asset_model_runs": len(asset_results),
        "failures": len(failures_frame),
    }

    return CrossAssetRun(
        asset_results=asset_results,
        model_summary=model_summary,
        fold_results=fold_results,
        failures=failures_frame,
        manifest=manifest,
    )
