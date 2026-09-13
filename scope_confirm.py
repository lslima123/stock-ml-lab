from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from stock_ml_lab.data.loader import download_ohlcv
from stock_ml_lab.scope_confirmation.config import locked_config
from stock_ml_lab.scope_confirmation.runner import run_locked_scope_confirmation


def main() -> None:
    parser = argparse.ArgumentParser(
        description="M14: run the locked independent h=5 local-vs-global confirmation."
    )
    parser.add_argument("--output-base", default="reports")
    args = parser.parse_args()
    config = locked_config()
    print("Milestone 14 — Locked Independent Scope Confirmation")
    print("=" * 92)
    print("Locked: h=5 + legacy | Ridge vs Ridge | Logistic vs Logistic")
    print("Primary: date-clustered HAC + BH across two tasks + circular block bootstrap")
    print("Confirmation assets are disjoint from the global training universe.\n")
    print("Protocol amendment: ELET3.SA → AXIA3.SA is an official ticker-identifier change; no asset was substituted.\n")

    training: dict[str, pd.DataFrame] = {}
    confirmation: dict[str, pd.DataFrame] = {}
    failures: list[dict[str, str]] = []
    sequence = [
        *(('training', ticker) for ticker in config.training_universe),
        *(('confirmation', ticker) for ticker in config.confirmation_universe),
    ]
    for index, (stage, ticker) in enumerate(sequence, start=1):
        print(f"[{index}/{len(sequence)}] {stage}: {ticker} ...", end=" ", flush=True)
        try:
            data = download_ohlcv(
                ticker=ticker,
                start=config.start,
                end=config.end_exclusive,
                interval="1d",
            )
            (training if stage == "training" else confirmation)[ticker] = data
            print(f"{len(data)} rows")
        except Exception as exc:
            failures.append({"ticker": ticker, "stage": stage, "error": str(exc)})
            print(f"FAILED: {exc}")

    if failures:
        failure_dir = Path(args.output_base) / "m14" / "preflight_failure"
        failure_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(failures).to_csv(failure_dir / "failures.csv", index=False)
        raise SystemExit(
            "Locked preflight failed. No substitutions are allowed and no model was evaluated. "
            f"See {failure_dir / 'failures.csv'}"
        )

    result = run_locked_scope_confirmation(
        training,
        confirmation,
        config=config,
        output_base=args.output_base,
        failures=pd.DataFrame(columns=["ticker", "stage", "error"]),
    )
    summary = result["summary"]
    print("\nLocked primary results")
    print("-" * 92)
    print(summary[[
        "task", "horizon", "global_improvement_vs_local_pct", "primary_hac_qvalue",
        "bootstrap_improvement_ci_low_pct", "bootstrap_improvement_ci_high_pct",
        "asset_win_count", "assets", "confirmed",
    ]].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nReport directory: {result['run_dir']}")


if __name__ == "__main__":
    main()
