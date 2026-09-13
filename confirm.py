from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import optuna
import pandas as pd

from stock_ml_lab.confirmation.config import (
    CONFIRMATION_TICKERS,
    LOCKED_HORIZONS,
    resolve_confirmation_assets,
)
from stock_ml_lab.confirmation.runner import run_locked_confirmation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run M8.1 locked cross-asset confirmation on the frozen 10d/20d legacy Logistic/Ridge candidates."
    )
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help=(
            "Optional subset of the predeclared confirmation universe. "
            "Default: all nine non-PETR4 assets."
        ),
    )
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--test-size", type=int, default=252)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--block-length", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument(
        "--output-root",
        default="reports/m8_1",
        help="Root directory; a unique run directory is created underneath.",
    )
    return parser.parse_args()


def _unique_output_dir(root: str, tickers: list[str]) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    label = "all9" if tuple(tickers) == tuple(CONFIRMATION_TICKERS) else "subset"
    return Path(root) / f"{stamp}_{label}_h10-20_locked-confirmation"


def _fmt_pct(x) -> str:
    return "N/A" if pd.isna(x) else f"{100*float(x):.2f}%"


def _print_asset_results(frame: pd.DataFrame) -> None:
    if frame.empty:
        print("No successful asset results.")
        return
    cols = [
        "ticker",
        "task",
        "horizon",
        "primary_improvement_pct",
        "primary_hac_p_model_better",
        "primary_hac_q_fdr",
        "primary_bootstrap_lower_pct",
        "phase_win_rate",
        "phase_primary_median",
    ]
    view = frame[cols].copy()
    view["primary_improvement_pct"] = view["primary_improvement_pct"].map(
        lambda x: f"{x:+.2f}%"
    )
    for col in ["primary_hac_p_model_better", "primary_hac_q_fdr"]:
        view[col] = view[col].map(lambda x: "N/A" if pd.isna(x) else f"{x:.4f}")
    view["primary_bootstrap_lower_pct"] = view["primary_bootstrap_lower_pct"].map(
        lambda x: f"{x:+.2f}%"
    )
    view["phase_win_rate"] = view["phase_win_rate"].map(_fmt_pct)
    view["phase_primary_median"] = view["phase_primary_median"].map(
        lambda x: "N/A" if pd.isna(x) else f"{x:+.2f}%"
    )
    print(view.to_string(index=False))


def _print_candidate_summary(frame: pd.DataFrame) -> None:
    if frame.empty:
        print("No candidate summary.")
        return
    view = frame.copy()
    view["primary_win_rate"] = view["primary_win_rate"].map(_fmt_pct)
    view["mean_phase_win_rate"] = view["mean_phase_win_rate"].map(_fmt_pct)
    for col in ["mean_primary_improvement_pct", "median_primary_improvement_pct", "median_phase_primary_improvement_pct"]:
        view[col] = view[col].map(lambda x: f"{x:+.2f}%")
    view["primary_win_sign_test_p"] = view["primary_win_sign_test_p"].map(
        lambda x: f"{x:.4f}"
    )
    if "mean_roc_auc" in view:
        view["mean_roc_auc"] = view["mean_roc_auc"].map(
            lambda x: "N/A" if pd.isna(x) else f"{x:.3f}"
        )
    if "median_roc_auc" in view:
        view["median_roc_auc"] = view["median_roc_auc"].map(
            lambda x: "N/A" if pd.isna(x) else f"{x:.3f}"
        )
    if "mean_directional_accuracy" in view:
        view["mean_directional_accuracy"] = view["mean_directional_accuracy"].map(_fmt_pct)
    print(view.to_string(index=False))


def main() -> None:
    args = parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    assets = resolve_confirmation_assets(
        None if args.tickers is None else tuple(args.tickers)
    )

    print("\nMilestone 8.1 — Locked Confirmation")
    print("=" * 88)
    print("Discovery asset excluded: PETR4.SA")
    print("Assets: " + ", ".join(asset.ticker for asset in assets))
    print("Locked candidates:")
    print("  10d + legacy + Logistic | 20d + legacy + Logistic")
    print("  10d + legacy + Ridge    | 20d + legacy + Ridge")
    print(
        f"Nested CV: outer={args.outer_splits} | inner={args.inner_splits} | "
        f"test_size={args.test_size} | trials={args.trials}"
    )
    print(
        f"Overlap handling: gap=h | HAC lags=h-1 | bootstrap block>=h | "
        f"{LOCKED_HORIZONS[0]}/{LOCKED_HORIZONS[1]} non-overlapping phases"
    )

    result = run_locked_confirmation(
        assets,
        start=args.start,
        end=args.end,
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        test_size=args.test_size,
        n_trials=args.trials,
        n_bootstrap=args.bootstrap,
        block_length=args.block_length,
        random_state=args.seed,
        continue_on_error=not args.fail_fast,
        progress=True,
    )

    output_dir = _unique_output_dir(args.output_root, [a.ticker for a in assets])
    result.save(output_dir)

    print("\n" + "=" * 88)
    print("M8.1 ASSET-LEVEL CONFIRMATION")
    print("=" * 88)
    _print_asset_results(result.asset_results)

    print("\n=== Locked candidate summary across confirmation assets ===")
    _print_candidate_summary(result.candidate_summary)

    print("\n=== Failures ===")
    print("None." if result.failures.empty else result.failures.to_string(index=False))
    print(f"\nSaved unique M8.1 confirmation report to: {output_dir}")
    print(
        "Interpretation rule: confirmation evidence should be judged across assets, "
        "HAC/FDR inference, bootstrap intervals, and non-overlapping phase stability; "
        "do not promote a candidate from one attractive ticker."
    )


if __name__ == "__main__":
    main()
