from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from stock_ml_lab.data.loader import download_ohlcv
from stock_ml_lab.reporting import unique_run_directory
from stock_ml_lab.robustness.assets import resolve_assets
from stock_ml_lab.global_model.training import train_global_artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="M12: train and validate pooled multi-asset global models."
    )
    parser.add_argument("--start", default="2018-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help="Optional custom universe. Default: the 10-asset core universe.",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=[1, 5, 10, 20],
    )
    parser.add_argument("--output-dir", default="artifacts/global")
    parser.add_argument("--test-size", type=int, default=252)
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--skip-validation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    assets = resolve_assets(
        universe="core",
        tickers=tuple(args.tickers) if args.tickers else None,
    )
    if len(assets) < 2:
        raise SystemExit("Global training requires at least two assets.")

    print("Milestone 12 — Global Multi-Asset Model")
    print("=" * 88)
    print("Universe:", ", ".join(asset.ticker for asset in assets))
    print("Models: pooled Ridge (regression) | pooled Logistic (classification)")
    print("Features: legacy only | ticker identity: NOT used")
    print("Validation: calendar walk-forward + target maturity purge + unseen-asset holdout")
    print()

    data: dict[str, pd.DataFrame] = {}
    failures: list[dict[str, str]] = []
    for i, asset in enumerate(assets, start=1):
        print(f"[{i}/{len(assets)}] Downloading {asset.ticker} ...", end=" ", flush=True)
        try:
            data[asset.ticker] = download_ohlcv(
                ticker=asset.ticker,
                start=args.start,
                end=args.end,
                interval="1d",
            )
            print(f"{len(data[asset.ticker])} rows")
        except Exception as exc:
            failures.append({"ticker": asset.ticker, "error": str(exc)})
            print(f"FAILED: {exc}")

    if len(data) < 2:
        raise SystemExit("Fewer than two assets downloaded successfully.")

    result = train_global_artifacts(
        data,
        horizons=tuple(args.horizons),
        output_dir=args.output_dir,
        validate=not args.skip_validation,
        n_splits=args.splits,
        test_size_dates=args.test_size,
    )

    report_dir = unique_run_directory(
        base="reports",
        experiment="m12",
        label="global-multi-asset",
    )
    report_dir.mkdir(parents=True, exist_ok=True)

    for key in ("temporal_metrics", "per_asset_metrics", "unseen_asset_metrics"):
        frame = result[key]
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            frame.to_csv(report_dir / f"{key}.csv", index=False)

    pd.DataFrame(failures).to_csv(report_dir / "failures.csv", index=False)

    manifest = {
        "milestone": "M12",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "start": args.start,
        "end": args.end,
        "tickers": sorted(data),
        "horizons": args.horizons,
        "artifact_dir": str(Path(args.output_dir).resolve()),
        "validation_enabled": not args.skip_validation,
        "splits": args.splits,
        "test_size_dates": args.test_size,
        "failures": failures,
    }
    import json
    (report_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str),
        encoding="utf-8",
    )

    temporal = result["temporal_metrics"]
    unseen = result["unseen_asset_metrics"]

    print("\nArtifacts")
    print("-" * 88)
    for artifact in result["artifacts"]:
        print(" ", artifact)

    if isinstance(temporal, pd.DataFrame) and not temporal.empty:
        print("\nCalendar walk-forward — mean by candidate")
        print("-" * 88)
        summary = (
            temporal.groupby(["task", "model", "horizon"], as_index=False)
            .agg(
                mean_primary_improvement_pct=("primary_improvement_pct", "mean"),
                median_primary_improvement_pct=("primary_improvement_pct", "median"),
            )
        )
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    if isinstance(unseen, pd.DataFrame) and not unseen.empty:
        print("\nUnseen-asset holdout — aggregate")
        print("-" * 88)
        summary = (
            unseen.groupby(["task", "model", "horizon"], as_index=False)
            .agg(
                assets=("ticker", "nunique"),
                win_count=("primary_improvement_pct", lambda s: int((s > 0).sum())),
                mean_improvement_pct=("primary_improvement_pct", "mean"),
                median_improvement_pct=("primary_improvement_pct", "median"),
            )
        )
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print(f"\nArtifact directory: {Path(args.output_dir).resolve()}")
    print(f"Report directory: {report_dir}")
    print(
        "\nRestart FastAPI after training. /api/v1/capabilities will then enable "
        "the Global scope when all 8 default artifacts are present."
    )


if __name__ == "__main__":
    main()
