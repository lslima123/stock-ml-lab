from __future__ import annotations

import argparse

import optuna
import pandas as pd

from stock_ml_lab.dataset import build_dataset
from stock_ml_lab.tuning import run_tuning_comparison
from stock_ml_lab.tuning.search_spaces import DISPLAY_NAMES, TUNABLE_MODELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Nested temporal hyperparameter optimization with Optuna."
    )
    parser.add_argument("ticker")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--gap", type=int, default=1)
    parser.add_argument("--test-size", type=int, default=None)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=TUNABLE_MODELS,
        default=list(TUNABLE_MODELS),
    )
    parser.add_argument("--top-features", type=int, default=8)
    return parser.parse_args()


def _format_summary(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["mae"] = out["mae"].map(lambda x: f"{x:.6f}")
    out["rmse"] = out["rmse"].map(lambda x: f"{x:.6f}")
    out["directional_accuracy"] = out["directional_accuracy"].map(
        lambda x: "N/A" if pd.isna(x) else f"{100*x:.2f}%"
    )
    out["directional_coverage"] = out["directional_coverage"].map(
        lambda x: f"{100*x:.2f}%"
    )
    for col in ("mae_improvement_vs_zero_pct", "rmse_improvement_vs_zero_pct"):
        out[col] = out[col].map(lambda x: f"{x:+.2f}%")
    return out


def main() -> None:
    args = parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    dataset = build_dataset(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        target="next_return",
    )

    result = run_tuning_comparison(
        dataset,
        models=tuple(args.models),
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        gap=args.gap,
        test_size=args.test_size,
        n_trials=args.trials,
        random_state=args.seed,
    )

    print(f"\nTicker: {dataset.ticker}")
    print(f"Dataset rows: {len(dataset.X)}")
    print(
        "Nested temporal CV: "
        f"{args.outer_splits} outer splits | "
        f"{args.inner_splits} inner splits | gap={args.gap}"
    )
    print(f"Optuna trials per outer fold/model: {args.trials}")
    print("Models: " + ", ".join(DISPLAY_NAMES[m] for m in args.models))

    print("\n=== Untouched outer-fold metrics ===")
    print(_format_summary(result.summary_frame()).to_string())

    print("\n=== Nested fold diagnostics ===")
    for tuned in result.tuned_results:
        frame = tuned.fold_frame().copy()
        frame["directional_accuracy"] *= 100.0
        print(f"\n[{tuned.model_name}]")
        print(
            frame[
                [
                    "n_train",
                    "n_test",
                    "best_inner_rmse",
                    "mae",
                    "rmse",
                    "directional_accuracy",
                ]
            ].to_string()
        )

    print("\n=== Best hyperparameters by outer fold ===")
    for tuned in result.tuned_results:
        print(f"\n[{tuned.model_name}]")
        print(tuned.params_frame().to_string())

    print("\n=== Tuned tree feature importance ===")
    found = False
    for tuned in result.tuned_results:
        importance = tuned.feature_importance_frame()
        if importance is None:
            continue
        found = True
        print(f"\n[{tuned.model_name}]")
        print(
            importance.head(args.top_features).to_string(
                float_format=lambda x: f"{x:.4f}"
            )
        )
    if not found:
        print("No selected tuned model exposes tree feature importance.")


if __name__ == "__main__":
    main()
