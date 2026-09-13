from __future__ import annotations

import argparse

import pandas as pd

from stock_ml_lab.dataset import build_dataset
from stock_ml_lab.experiment import AVAILABLE_MODELS, DISPLAY_NAMES, run_model_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run leakage-aware temporal model comparison experiments."
    )
    parser.add_argument("ticker", help="Yahoo Finance ticker, e.g. PETR4.SA or AAPL")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--gap", type=int, default=1)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=AVAILABLE_MODELS,
        default=list(AVAILABLE_MODELS),
        help="Models to evaluate. Default: all.",
    )
    parser.add_argument(
        "--top-features",
        type=int,
        default=8,
        help="Number of tree feature importances to print per model.",
    )
    return parser.parse_args()


def _format_summary(frame: pd.DataFrame) -> pd.DataFrame:
    formatted = frame.copy()
    formatted["mae"] = formatted["mae"].map(lambda x: f"{x:.6f}")
    formatted["rmse"] = formatted["rmse"].map(lambda x: f"{x:.6f}")
    formatted["directional_accuracy"] = formatted["directional_accuracy"].map(
        lambda x: "N/A" if pd.isna(x) else f"{100 * x:.2f}%"
    )
    formatted["directional_coverage"] = formatted["directional_coverage"].map(
        lambda x: f"{100 * x:.2f}%"
    )

    for column in ("mae_improvement_vs_zero_pct", "rmse_improvement_vs_zero_pct"):
        if column in formatted:
            formatted[column] = formatted[column].map(lambda x: f"{x:+.2f}%")

    return formatted


def main() -> None:
    args = parse_args()

    dataset = build_dataset(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        target="next_return",
    )

    comparison = run_model_comparison(
        dataset,
        ridge_alpha=args.ridge_alpha,
        n_splits=args.splits,
        gap=args.gap,
        test_size=args.test_size,
        models=tuple(args.models),
    )

    print(f"\nTicker: {dataset.ticker}")
    print(f"Dataset rows: {len(dataset.X)}")
    print(f"Target: {dataset.target_name}")
    print(f"Temporal CV: {args.splits} splits | gap={args.gap}")
    print("Models: " + ", ".join(DISPLAY_NAMES[key] for key in args.models))

    print("\n=== Overall out-of-fold metrics ===")
    print(_format_summary(comparison.summary_frame()).to_string())

    print("\n=== Fold diagnostics ===")
    for result in comparison.results:
        if result.model_name in {"Zero Return", "Persistence"}:
            continue
        folds = result.fold_frame().copy()
        folds["directional_accuracy"] *= 100.0
        columns = ["n_train", "n_test", "mae", "rmse", "directional_accuracy"]
        print(f"\n[{result.model_name}]")
        print(folds[columns].to_string())

    print("\n=== Tree feature importance (mean across folds) ===")
    found_importance = False
    for result in comparison.results:
        importance = result.feature_importance_frame()
        if importance is None:
            continue
        found_importance = True
        print(f"\n[{result.model_name}]")
        print(importance.head(args.top_features).to_string(float_format=lambda x: f"{x:.4f}"))

    if not found_importance:
        print("No selected model exposes tree feature importance.")

    details_found = False
    for result in comparison.results:
        if result.fold_model_details is None:
            continue
        if not details_found:
            print("\n=== Classical model specifications by fold ===")
            details_found = True
        print(f"\n[{result.model_name}]")
        print(result.fold_model_details.to_string())


if __name__ == "__main__":
    main()
