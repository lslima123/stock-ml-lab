from __future__ import annotations

import argparse

from stock_ml_lab.dataset import build_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the Stock ML Lab dataset.")
    parser.add_argument("ticker", help="Yahoo Finance ticker, e.g. PETR4.SA or AAPL")
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = build_dataset(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        target="next_return",
    )

    print(f"Ticker: {dataset.ticker}")
    print(f"Rows: {len(dataset.X)}")
    print(f"Period: {dataset.X.index.min().date()} -> {dataset.X.index.max().date()}")
    print(f"Target: {dataset.target_name}")

    print("\nFeatures:")
    for feature in dataset.feature_names:
        print(f"  - {feature}")

    print("\nLast observations:")
    preview = dataset.X.tail(5).copy()
    preview[dataset.target_name] = dataset.y.tail(5)
    print(preview.to_string())


if __name__ == "__main__":
    main()
