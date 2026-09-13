from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from stock_ml_lab.data.loader import download_ohlcv
from stock_ml_lab.robustness.assets import resolve_assets
from stock_ml_lab.comparison.benchmark import run_local_vs_global_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="M13: compare local and global models on identical OOS dates.")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 5, 10, 20])
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--output-base", default="reports")
    parser.add_argument("--test-size", type=int, default=252)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    assets = resolve_assets(universe="core", tickers=tuple(args.tickers) if args.tickers else None)
    data: dict[str, pd.DataFrame] = {}
    failures: list[dict[str, str]] = []
    print("Milestone 13 — Local vs Global Benchmark")
    print("=" * 92)
    print("Pairs: Local Ridge vs Global Ridge | Local Logistic vs Global Logistic")
    print("Evaluation: identical OOS dates + target-maturity purge + HAC comparison")
    print()
    for i, asset in enumerate(assets, start=1):
        print(f"[{i}/{len(assets)}] Downloading {asset.ticker} ...", end=" ", flush=True)
        try:
            data[asset.ticker] = download_ohlcv(ticker=asset.ticker, start=args.start, end=args.end, interval="1d")
            print(f"{len(data[asset.ticker])} rows")
        except Exception as exc:
            failures.append({"ticker": asset.ticker, "error": str(exc)})
            print(f"FAILED: {exc}")
    if len(data) < 2:
        raise SystemExit("Fewer than two assets downloaded successfully.")

    result = run_local_vs_global_benchmark(
        data,
        horizons=tuple(args.horizons),
        output_base=args.output_base,
        n_splits=args.splits,
        test_size_dates=args.test_size,
        bootstrap_repetitions=args.bootstrap_repetitions,
        random_state=args.random_state,
    )
    run_dir = Path(result["run_dir"])
    pd.DataFrame(failures).to_csv(run_dir / "failures.csv", index=False)

    print("\nAggregate comparison")
    print("-" * 92)
    summary = result["summary"]
    if isinstance(summary, pd.DataFrame) and not summary.empty:
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nInference note: date-clustered HAC/bootstrap diagnostics are POST HOC for M13.")
    print(f"\nReport directory: {run_dir}")


if __name__ == "__main__":
    main()
