from __future__ import annotations

import argparse
from pathlib import Path

import optuna
import pandas as pd

from stock_ml_lab.classification.search_spaces import CLASSIFICATION_MODELS
from stock_ml_lab.reporting import unique_run_directory
from stock_ml_lab.research import FEATURE_SETS, run_research_grid
from stock_ml_lab.research.runner import RESEARCH_TASKS
from stock_ml_lab.tuning.search_spaces import TUNABLE_MODELS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Milestone 8: feature and multi-horizon research grid."
    )
    parser.add_argument("ticker")
    parser.add_argument("--benchmark", default=None)
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 10, 20])
    parser.add_argument(
        "--feature-sets",
        nargs="+",
        choices=list(FEATURE_SETS),
        default=list(FEATURE_SETS),
    )
    parser.add_argument(
        "--tasks", nargs="+", choices=RESEARCH_TASKS, default=["classification"]
    )
    parser.add_argument(
        "--classification-models",
        nargs="+",
        choices=CLASSIFICATION_MODELS,
        default=["logistic", "catboost"],
    )
    parser.add_argument(
        "--regression-models",
        nargs="+",
        choices=TUNABLE_MODELS,
        default=["ridge", "catboost"],
    )
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--test-size", type=int, default=252)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def _format_classification(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    for col in ["log_loss", "brier", "roc_auc"]:
        if col in out:
            out[col] = out[col].map(lambda x: "N/A" if pd.isna(x) else f"{x:.4f}")
    for col in ["balanced_accuracy"]:
        if col in out:
            out[col] = out[col].map(lambda x: "N/A" if pd.isna(x) else f"{100*x:.2f}%")
    for col in ["log_loss_improvement_vs_prior_pct", "brier_improvement_vs_prior_pct"]:
        if col in out:
            out[col] = out[col].map(lambda x: "N/A" if pd.isna(x) else f"{x:+.2f}%")
    return out


def _format_regression(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    out = frame.copy()
    for col in ["rmse", "mae"]:
        if col in out:
            out[col] = out[col].map(lambda x: "N/A" if pd.isna(x) else f"{x:.6f}")
    if "directional_accuracy" in out:
        out["directional_accuracy"] = out["directional_accuracy"].map(
            lambda x: "N/A" if pd.isna(x) else f"{100*x:.2f}%"
        )
    for col in ["rmse_improvement_vs_zero_pct", "mae_improvement_vs_zero_pct"]:
        if col in out:
            out[col] = out[col].map(lambda x: "N/A" if pd.isna(x) else f"{x:+.2f}%")
    return out


def main() -> None:
    args = parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if args.output_dir:
        run_dir = Path(args.output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        label = (
            f"{args.ticker}_h{'-'.join(map(str, args.horizons))}_"
            f"{'-'.join(args.tasks)}"
        )
        run_dir = unique_run_directory(base="reports", experiment="m8", label=label)

    result = run_research_grid(
        ticker=args.ticker,
        benchmark_ticker=args.benchmark,
        start=args.start,
        end=args.end,
        horizons=tuple(args.horizons),
        feature_sets=tuple(args.feature_sets),
        tasks=tuple(args.tasks),
        classification_models=tuple(args.classification_models),
        regression_models=tuple(args.regression_models),
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        test_size=args.test_size,
        n_trials=args.trials,
        random_state=args.seed,
        continue_on_error=not args.fail_fast,
        progress=True,
    )
    result.save(run_dir)

    print("\n" + "=" * 88)
    print("M8 DATASET SUMMARY")
    print("=" * 88)
    if result.dataset_summary.empty:
        print("No successful datasets.")
    else:
        ds = result.dataset_summary.copy()
        ds["positive_rate"] = ds["positive_rate"].map(lambda x: f"{100*x:.2f}%")
        print(ds.to_string(index=False))

    if "classification" in args.tasks:
        print("\n=== Classification research grid (OOS; exploratory ranking) ===")
        view = result.classification_view()
        if view.empty:
            print("No classification results.")
        else:
            print(
                _format_classification(
                    view[view["model"] != "Prior Baseline"].sort_values(
                        ["horizon", "log_loss"]
                    )
                ).to_string(index=False)
            )

    if "regression" in args.tasks:
        print("\n=== Regression research grid (OOS; exploratory ranking) ===")
        view = result.regression_view()
        if view.empty:
            print("No regression results.")
        else:
            print(
                _format_regression(
                    view[view["model"] != "Zero Return"].sort_values(
                        ["horizon", "rmse"]
                    )
                ).to_string(index=False)
            )

    print("\n=== Failures ===")
    print("None." if result.failures.empty else result.failures.to_string(index=False))
    print(f"\nSaved unique M8 research report to: {run_dir}")
    print(
        "Note: feature-set/horizon rankings are exploratory. A later untouched "
        "confirmation stage is required before declaring a configuration superior."
    )


if __name__ == "__main__":
    main()
