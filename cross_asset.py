from __future__ import annotations

import argparse

import optuna
import pandas as pd

from stock_ml_lab.robustness import CORE_UNIVERSE, resolve_assets, run_cross_asset_study
from stock_ml_lab.reporting import unique_run_directory
from stock_ml_lab.tuning.search_spaces import DISPLAY_NAMES, TUNABLE_MODELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Milestone 6 cross-asset robustness study."
    )
    parser.add_argument("--universe", choices=["core"], default="core")
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--gap", type=int, default=1)
    parser.add_argument("--test-size", type=int, default=252)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--block-length", type=int, default=20)
    parser.add_argument("--dm-lags", type=int, default=5)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=TUNABLE_MODELS,
        default=["catboost"],
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.0, 0.001, 0.0025, 0.005],
    )
    parser.add_argument(
        "--strategy",
        choices=["long_short", "long_flat"],
        default="long_short",
    )
    parser.add_argument(
        "--threshold-objective",
        choices=["sharpe", "annualized_return", "sortino"],
        default="sharpe",
    )
    parser.add_argument(
        "--skip-nested-threshold",
        action="store_true",
        help="Skip nested threshold selection to reduce runtime.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop on the first ticker failure instead of recording it.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def _format_asset_table(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    cols = [
        "ticker",
        "market",
        "model",
        "rmse_improvement_vs_zero_pct",
        "directional_accuracy",
        "dm_mse_p_model_better",
        "dm_mse_q_fdr",
        "pt_p_model_better",
        "pt_q_fdr",
        "strategy_sharpe",
        "buy_hold_sharpe",
        "nested_strategy_sharpe",
    ]
    out = frame[cols].copy()
    out["rmse_improvement_vs_zero_pct"] = out[
        "rmse_improvement_vs_zero_pct"
    ].map(lambda x: f"{x:+.2f}%")
    out["directional_accuracy"] = out["directional_accuracy"].map(
        lambda x: f"{100*x:.2f}%"
    )
    for col in [
        "dm_mse_p_model_better",
        "dm_mse_q_fdr",
        "pt_p_model_better",
        "pt_q_fdr",
        "strategy_sharpe",
        "buy_hold_sharpe",
        "nested_strategy_sharpe",
    ]:
        out[col] = out[col].map(
            lambda x: "N/A" if pd.isna(x) else f"{float(x):.3f}"
        )
    return out


def _format_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    for col in ["rmse_win_rate", "mean_directional_accuracy", "median_directional_accuracy"]:
        out[col] = out[col].map(lambda x: f"{100*x:.1f}%")
    for col in ["mean_rmse_improvement_pct", "median_rmse_improvement_pct"]:
        out[col] = out[col].map(lambda x: f"{x:+.2f}%")
    out["rmse_win_sign_test_p"] = out["rmse_win_sign_test_p"].map(
        lambda x: f"{x:.4f}"
    )
    return out


def main() -> None:
    args = parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    assets = resolve_assets(
        universe=args.universe,
        tickers=None if args.tickers is None else tuple(args.tickers),
    )
    thresholds = tuple(sorted(set(float(x) for x in args.thresholds)))

    print("\nMilestone 6 — Cross-Asset Robustness")
    print("=" * 78)
    print("Assets: " + ", ".join(spec.ticker for spec in assets))
    print("Models: " + ", ".join(DISPLAY_NAMES[m] for m in args.models))
    print(
        f"Outer CV={args.outer_splits} | inner CV={args.inner_splits} | "
        f"test_size={args.test_size} | trials={args.trials}"
    )
    print(
        f"FDR correction: Benjamini-Hochberg within each model across assets"
    )

    result = run_cross_asset_study(
        assets,
        start=args.start,
        end=args.end,
        models=tuple(args.models),
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        gap=args.gap,
        test_size=args.test_size,
        n_trials=args.trials,
        bootstrap=args.bootstrap,
        block_length=args.block_length,
        dm_lags=args.dm_lags,
        cost_bps=args.cost_bps,
        thresholds=thresholds,
        strategy=args.strategy,
        threshold_objective=args.threshold_objective,
        random_state=args.seed,
        include_nested_threshold=not args.skip_nested_threshold,
        continue_on_error=not args.fail_fast,
        progress=True,
    )

    output_dir = (
        args.output_dir
        if args.output_dir
        else str(unique_run_directory(
            base="reports",
            experiment="m6",
            label="-".join(args.models),
        ))
    )
    result.save(output_dir)

    print("\n=== Cross-asset asset/model results ===")
    if result.asset_results.empty:
        print("No successful runs.")
    else:
        print(_format_asset_table(result.asset_results).to_string(index=False))

    print("\n=== Aggregate robustness by model ===")
    if result.model_summary.empty:
        print("No aggregate results.")
    else:
        print(_format_summary(result.model_summary).to_string(index=False))

    print("\n=== Failures ===")
    if result.failures.empty:
        print("None.")
    else:
        print(result.failures.to_string(index=False))

    print(f"\nSaved machine-readable reports to: {output_dir}")


if __name__ == "__main__":
    main()
