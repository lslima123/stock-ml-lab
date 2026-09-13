from __future__ import annotations

import argparse

import numpy as np
import optuna
import pandas as pd

from stock_ml_lab.dataset import build_dataset
from stock_ml_lab.tuning import run_tuning_comparison
from stock_ml_lab.tuning.search_spaces import DISPLAY_NAMES, TUNABLE_MODELS
from stock_ml_lab.validation import (
    backtest_predictions,
    buy_and_hold_backtest,
    cost_sensitivity_report,
    diebold_mariano_vs_benchmark,
    forecast_block_bootstrap,
    pesaran_timmermann_test,
    select_thresholds_nested,
    threshold_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Statistical validation and economic backtesting of nested OOS forecasts."
    )
    parser.add_argument("ticker")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--gap", type=int, default=1)
    parser.add_argument("--test-size", type=int, default=252)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=TUNABLE_MODELS,
        default=["catboost", "xgboost"],
    )
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--block-length", type=int, default=20)
    parser.add_argument("--dm-lags", type=int, default=5)
    parser.add_argument("--cost-bps", type=float, default=10.0)
    parser.add_argument(
        "--cost-grid-bps",
        nargs="+",
        type=float,
        default=[0.0, 5.0, 10.0, 20.0],
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
    return parser.parse_args()


def _format_backtest_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    pct_columns = [
        "total_return",
        "annualized_return",
        "annualized_volatility",
        "max_drawdown",
        "exposure",
        "hit_rate_active",
    ]
    for col in pct_columns:
        if col in out.columns:
            out[col] = out[col].map(
                lambda x: "N/A" if pd.isna(x) else f"{100 * float(x):.2f}%"
            )
    for col in ["sharpe", "sortino", "annualized_turnover"]:
        if col in out.columns:
            out[col] = out[col].map(
                lambda x: "N/A" if pd.isna(x) else f"{float(x):.3f}"
            )
    if "total_cost_return" in out.columns:
        out["total_cost_return"] = out["total_cost_return"].map(
            lambda x: f"{100 * float(x):.2f}%"
        )
    return out


def _backtest_summary(model_name: str, buy_hold, raw, nested) -> pd.DataFrame:
    rows = [
        {"strategy": "Buy & Hold", **buy_hold.metrics.to_dict()},
        {"strategy": f"{model_name} threshold=0", **raw.metrics.to_dict()},
        {"strategy": f"{model_name} nested threshold", **nested.backtest.metrics.to_dict()},
    ]
    return pd.DataFrame(rows).set_index("strategy")


def main() -> None:
    args = parse_args()
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    thresholds = tuple(sorted(set(float(x) for x in args.thresholds)))
    costs = tuple(sorted(set(float(x) for x in args.cost_grid_bps)))

    dataset = build_dataset(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        target="next_return",
    )

    comparison = run_tuning_comparison(
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
        f"{args.outer_splits} outer | {args.inner_splits} inner | gap={args.gap}"
    )
    print(f"Models: {', '.join(DISPLAY_NAMES[m] for m in args.models)}")
    print(f"Transaction cost assumption: {args.cost_bps:g} bps per unit turnover")

    benchmark = comparison.zero_baseline.predictions

    for model_idx, tuned in enumerate(comparison.tuned_results):
        print("\n" + "=" * 78)
        print(tuned.model_name)
        print("=" * 78)

        actual = tuned.actuals
        forecast = tuned.predictions
        zero = benchmark.loc[forecast.index]

        print("\n=== Forecast metrics ===")
        metrics = tuned.metrics
        print(f"MAE:                  {metrics.mae:.6f}")
        print(f"RMSE:                 {metrics.rmse:.6f}")
        print(f"Directional accuracy:{100 * metrics.directional_accuracy:8.2f}%")

        print("\n=== Forecast significance vs Zero Return ===")
        dm_mse = diebold_mariano_vs_benchmark(
            actual,
            forecast,
            zero,
            criterion="mse",
            lags=args.dm_lags,
        )
        dm_mae = diebold_mariano_vs_benchmark(
            actual,
            forecast,
            zero,
            criterion="mae",
            lags=args.dm_lags,
        )
        pt = pesaran_timmermann_test(actual, forecast)

        tests = pd.DataFrame(
            [
                {
                    "test": "Diebold-Mariano (MSE)",
                    "statistic": dm_mse.statistic,
                    "p_two_sided": dm_mse.pvalue_two_sided,
                    "p_model_better": dm_mse.pvalue_model_better,
                },
                {
                    "test": "Diebold-Mariano (MAE)",
                    "statistic": dm_mae.statistic,
                    "p_two_sided": dm_mae.pvalue_two_sided,
                    "p_model_better": dm_mae.pvalue_model_better,
                },
                {
                    "test": "Pesaran-Timmermann direction",
                    "statistic": pt.statistic,
                    "p_two_sided": pt.pvalue_two_sided,
                    "p_model_better": pt.pvalue_larger,
                },
            ]
        ).set_index("test")
        print(tests.to_string(float_format=lambda x: f"{x:.6f}"))
        print(
            f"PT observed DA={100 * pt.directional_accuracy:.2f}% | "
            f"independence expectation={100 * pt.expected_accuracy_under_independence:.2f}%"
        )

        print("\n=== Circular block-bootstrap 95% intervals ===")
        intervals = forecast_block_bootstrap(
            actual,
            forecast,
            zero,
            n_bootstrap=args.bootstrap,
            block_length=args.block_length,
            confidence=0.95,
            random_state=args.seed + model_idx,
        )
        interval_frame = pd.DataFrame(
            [
                {
                    "metric": key,
                    "estimate": interval.estimate,
                    "lower": interval.lower,
                    "upper": interval.upper,
                }
                for key, interval in intervals.items()
            ]
        ).set_index("metric")
        print(interval_frame.to_string(float_format=lambda x: f"{x:.6f}"))

        buy_hold = buy_and_hold_backtest(actual, cost_bps=args.cost_bps)
        raw = backtest_predictions(
            actual,
            forecast,
            threshold=0.0,
            cost_bps=args.cost_bps,
            mode=args.strategy,
            name=f"{tuned.model_name} threshold=0",
        )
        nested = select_thresholds_nested(
            dataset,
            tuned,
            thresholds=thresholds,
            cost_bps=args.cost_bps,
            mode=args.strategy,
            objective=args.threshold_objective,
            random_state=args.seed,
        )

        print("\n=== Economic backtest ===")
        print(
            _format_backtest_frame(
                _backtest_summary(tuned.model_name, buy_hold, raw, nested)
            ).to_string()
        )

        print("\n=== Nested threshold selection by outer fold ===")
        selection_frame = nested.selection_frame().copy()
        selection_frame["threshold_pct"] = selection_frame["threshold"] * 100.0
        print(selection_frame.to_string(float_format=lambda x: f"{x:.6f}"))

        print("\n=== Fixed-threshold sensitivity (descriptive only; not model selection) ===")
        fixed = threshold_report(
            actual,
            forecast,
            thresholds=thresholds,
            cost_bps=args.cost_bps,
            mode=args.strategy,
        )
        print(_format_backtest_frame(fixed).to_string())

        print("\n=== Transaction-cost sensitivity at threshold=0 ===")
        cost_frame = cost_sensitivity_report(
            actual,
            forecast,
            threshold=0.0,
            costs_bps=costs,
            mode=args.strategy,
        )
        print(_format_backtest_frame(cost_frame).to_string())

    print("\nNotes:")
    print("- DM negative statistics favor the model because loss = model - Zero Return.")
    print("- p_model_better is the one-sided p-value for lower loss / better direction.")
    print("- Thresholds in the fixed table are descriptive; only the nested threshold row")
    print("  chooses thresholds without observing the corresponding outer test fold.")
    print("- Long-short results exclude stock-borrow and financing costs.")


if __name__ == "__main__":
    main()
