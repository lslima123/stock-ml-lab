from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from stock_ml_lab.classification import (
    CLASSIFICATION_DISPLAY_NAMES,
    CLASSIFICATION_MODELS,
    backtest_probabilities,
    build_direction_dataset,
    calibration_table,
    classification_block_bootstrap,
    probability_margin_report,
    run_classification_comparison,
    select_probability_margins_nested,
)
from stock_ml_lab.reporting import unique_run_directory
from stock_ml_lab.validation.backtest import buy_and_hold_backtest
from stock_ml_lab.validation.forecast_tests import pesaran_timmermann_test


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Milestone 7: nested temporal direction classification."
    )
    parser.add_argument("ticker")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default="2026-01-01")
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--gap", type=int, default=1)
    parser.add_argument("--test-size", type=int, default=252)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=CLASSIFICATION_MODELS,
        default=list(CLASSIFICATION_MODELS),
    )
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--block-length", type=int, default=20)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument(
        "--probability-margins",
        nargs="+",
        type=float,
        default=[0.0, 0.025, 0.05, 0.10],
        help="Confidence margins around 0.50, e.g. 0.05 means long >0.55 / short <0.45.",
    )
    parser.add_argument(
        "--strategy", choices=["long_short", "long_flat"], default="long_short"
    )
    parser.add_argument(
        "--margin-objective",
        choices=["sharpe", "annualized_return", "sortino"],
        default="sharpe",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


def _pct(x: float) -> str:
    return "N/A" if pd.isna(x) else f"{100*x:.2f}%"


def _format_summary(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for col in ["accuracy", "balanced_accuracy", "precision", "recall", "f1", "actual_positive_rate", "predicted_positive_rate", "mean_probability", "ece"]:
        out[col] = out[col].map(_pct)
    for col in ["roc_auc", "brier", "log_loss"]:
        out[col] = out[col].map(lambda x: "N/A" if pd.isna(x) else f"{x:.4f}")
    for col in [
        "brier_improvement_vs_prior_pct",
        "log_loss_improvement_vs_prior_pct",
        "accuracy_improvement_vs_prior_pp",
        "balanced_accuracy_improvement_vs_prior_pp",
    ]:
        out[col] = out[col].map(lambda x: f"{x:+.2f}")
    return out


def _save_reports(
    run_dir: Path,
    *,
    args,
    dataset,
    comparison,
    bootstrap_frames,
    calibration_frames,
    backtest_rows,
    margin_frames,
    nested_frames,
    importance_frames,
) -> None:
    summary = comparison.summary_frame().reset_index()
    summary.to_csv(run_dir / "summary.csv", index=False)

    fold_rows = []
    for result in comparison.tuned_results:
        f = result.fold_frame().reset_index()
        f.insert(0, "model", result.model_name)
        f.insert(0, "model_key", result.model_key)
        fold_rows.append(f)
    pd.concat(fold_rows, ignore_index=True).to_csv(run_dir / "fold_results.csv", index=False)

    pred = pd.DataFrame(
        {
            "actual_direction": comparison.prior.actuals,
            "actual_return": comparison.prior.actual_returns,
            "prior_probability_up": comparison.prior.probabilities,
        }
    )
    for result in comparison.tuned_results:
        pred[f"{result.model_key}_probability_up"] = result.probabilities
    pred.to_csv(run_dir / "predictions.csv", index_label="date")

    if bootstrap_frames:
        pd.concat(bootstrap_frames, ignore_index=True).to_csv(
            run_dir / "bootstrap_intervals.csv", index=False
        )
    if calibration_frames:
        pd.concat(calibration_frames, ignore_index=True).to_csv(
            run_dir / "calibration.csv", index=False
        )
    if backtest_rows:
        pd.DataFrame(backtest_rows).to_csv(run_dir / "backtest.csv", index=False)
    if margin_frames:
        pd.concat(margin_frames, ignore_index=True).to_csv(
            run_dir / "fixed_probability_margins.csv", index=False
        )
    if nested_frames:
        pd.concat(nested_frames, ignore_index=True).to_csv(
            run_dir / "nested_probability_margins.csv", index=False
        )
    if importance_frames:
        pd.concat(importance_frames, ignore_index=True).to_csv(
            run_dir / "feature_importance.csv", index=False
        )

    manifest = {
        "milestone": 7,
        "ticker": dataset.ticker,
        "target": "next_direction = 1(next_return > 0)",
        "start": args.start,
        "end": args.end,
        "models": args.models,
        "outer_splits": args.outer_splits,
        "inner_splits": args.inner_splits,
        "gap": args.gap,
        "test_size": args.test_size,
        "trials": args.trials,
        "tuning_objective": "inner OOF log loss",
        "classification_threshold": 0.5,
        "bootstrap": args.bootstrap,
        "block_length": args.block_length,
        "calibration_bins": args.calibration_bins,
        "cost_bps": args.cost_bps,
        "probability_margins": args.probability_margins,
        "strategy": args.strategy,
        "margin_objective": args.margin_objective,
        "seed": args.seed,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    margins = tuple(sorted(set(float(x) for x in args.probability_margins)))

    dataset = build_direction_dataset(ticker=args.ticker, start=args.start, end=args.end)
    comparison = run_classification_comparison(
        dataset,
        models=tuple(args.models),
        outer_splits=args.outer_splits,
        inner_splits=args.inner_splits,
        gap=args.gap,
        test_size=args.test_size,
        n_trials=args.trials,
        random_state=args.seed,
    )

    if args.output_dir:
        run_dir = Path(args.output_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
    else:
        run_dir = unique_run_directory(
            base="reports",
            experiment="m7",
            label=f"{dataset.ticker}_{'-'.join(args.models)}",
        )

    print(f"\nTicker: {dataset.ticker}")
    print(f"Dataset rows: {len(dataset.X)}")
    print(f"Actual up-rate: {100*dataset.y.mean():.2f}%")
    print(
        f"Nested temporal CV: {args.outer_splits} outer | "
        f"{args.inner_splits} inner | gap={args.gap}"
    )
    print(f"Tuning objective: inner OOF log loss | trials={args.trials}")
    print("Models: Prior Baseline, " + ", ".join(CLASSIFICATION_DISPLAY_NAMES[m] for m in args.models))

    print("\n=== OOS probabilistic classification metrics ===")
    print(_format_summary(comparison.summary_frame()).to_string())

    bootstrap_frames = []
    calibration_frames = []
    backtest_rows = []
    margin_frames = []
    nested_frames = []
    importance_frames = []

    buy_hold = buy_and_hold_backtest(
        comparison.prior.actual_returns,
        cost_bps=args.cost_bps,
    )
    backtest_rows.append({"model": "Buy & Hold", "variant": "buy_hold", **buy_hold.metrics.to_dict()})

    print("\n=== Model diagnostics ===")
    for result_idx, result in enumerate(comparison.tuned_results):
        print("\n" + "-" * 78)
        print(result.model_name)
        print("-" * 78)

        fold = result.fold_frame()[
            [
                "n_train", "n_test", "best_inner_log_loss", "accuracy",
                "balanced_accuracy", "roc_auc", "brier", "log_loss",
                "actual_positive_rate", "predicted_positive_rate",
            ]
        ].copy()
        print("\nOuter-fold metrics:")
        print(fold.to_string(float_format=lambda x: f"{x:.4f}"))

        print("\nBest hyperparameters by outer fold:")
        print(result.params_frame().to_string())

        pt = pesaran_timmermann_test(
            result.actual_returns,
            result.probabilities - 0.5,
        )
        print(
            "\nPesaran-Timmermann on p(up)-0.5: "
            f"DA={100*pt.directional_accuracy:.2f}% | "
            f"expected={100*pt.expected_accuracy_under_independence:.2f}% | "
            f"p(larger)={pt.pvalue_larger:.4f}"
        )

        boot = classification_block_bootstrap(
            result.actuals,
            result.probabilities,
            comparison.prior.probabilities,
            n_bootstrap=args.bootstrap,
            block_length=min(args.block_length, len(result.actuals)),
            random_state=args.seed + result_idx,
        )
        print("\nCircular block-bootstrap 95% intervals vs Prior Baseline:")
        print(boot.to_string(float_format=lambda x: f"{x:.4f}"))
        tmp = boot.reset_index()
        tmp.insert(0, "model", result.model_name)
        tmp.insert(0, "model_key", result.model_key)
        bootstrap_frames.append(tmp)

        cal = calibration_table(
            result.actuals,
            result.probabilities,
            n_bins=args.calibration_bins,
        )
        print("\nReliability table:")
        print(cal.to_string(float_format=lambda x: f"{x:.4f}"))
        c = cal.reset_index()
        c.insert(0, "model", result.model_name)
        c.insert(0, "model_key", result.model_key)
        calibration_frames.append(c)

        fixed = probability_margin_report(
            result.actual_returns,
            result.probabilities,
            margins=margins,
            cost_bps=args.cost_bps,
            mode=args.strategy,
        )
        print("\nFixed probability-margin backtest (descriptive only):")
        display_cols = [
            "total_return", "annualized_return", "sharpe", "sortino",
            "max_drawdown", "exposure", "annualized_turnover", "total_cost_return",
        ]
        print(fixed[display_cols].to_string(float_format=lambda x: f"{x:.4f}"))
        f = fixed.reset_index()
        f.insert(0, "model", result.model_name)
        f.insert(0, "model_key", result.model_key)
        margin_frames.append(f)

        raw = backtest_probabilities(
            result.actual_returns,
            result.probabilities,
            margin=0.0,
            cost_bps=args.cost_bps,
            mode=args.strategy,
            name=result.model_name,
        )
        backtest_rows.append(
            {"model": result.model_name, "variant": "margin_0", **raw.metrics.to_dict()}
        )

        nested = select_probability_margins_nested(
            dataset,
            result,
            margins=margins,
            cost_bps=args.cost_bps,
            mode=args.strategy,
            objective=args.margin_objective,
            random_state=args.seed,
        )
        print("\nNested probability-margin selection:")
        selection = nested.selection_frame()
        print(selection.to_string(float_format=lambda x: f"{x:.4f}"))
        print(
            "Nested OOS backtest: "
            f"return={100*nested.backtest.metrics.total_return:.2f}% | "
            f"Sharpe={nested.backtest.metrics.sharpe:.3f} | "
            f"MaxDD={100*nested.backtest.metrics.max_drawdown:.2f}% | "
            f"exposure={100*nested.backtest.metrics.exposure:.2f}%"
        )
        n = selection.reset_index()
        n.insert(0, "model", result.model_name)
        n.insert(0, "model_key", result.model_key)
        nested_frames.append(n)
        backtest_rows.append(
            {
                "model": result.model_name,
                "variant": "nested_probability_margin",
                **nested.backtest.metrics.to_dict(),
            }
        )

        importance = result.feature_importance_frame()
        if importance is not None:
            print("\nTree feature importance:")
            print(importance.head(8).to_string(float_format=lambda x: f"{x:.4f}"))
            imp = importance.reset_index().rename(columns={"index": "feature"})
            imp.insert(0, "model", result.model_name)
            imp.insert(0, "model_key", result.model_key)
            importance_frames.append(imp)

    print("\n=== Economic benchmark ===")
    print(
        f"Buy & Hold: return={100*buy_hold.metrics.total_return:.2f}% | "
        f"Sharpe={buy_hold.metrics.sharpe:.3f} | "
        f"MaxDD={100*buy_hold.metrics.max_drawdown:.2f}%"
    )

    _save_reports(
        run_dir,
        args=args,
        dataset=dataset,
        comparison=comparison,
        bootstrap_frames=bootstrap_frames,
        calibration_frames=calibration_frames,
        backtest_rows=backtest_rows,
        margin_frames=margin_frames,
        nested_frames=nested_frames,
        importance_frames=importance_frames,
    )
    print(f"\nSaved unique M7 report to: {run_dir}")


if __name__ == "__main__":
    main()
