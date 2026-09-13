from __future__ import annotations

import argparse
from pathlib import Path

from stock_ml_lab.reporting import unique_run_directory
from stock_ml_lab.visualization import build_analytics_report, discover_runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Milestone 9: build static experiment analytics from persisted Stock ML Lab reports."
    )
    parser.add_argument(
        "--source",
        action="append",
        default=[],
        help=(
            "Experiment run directory or parent directory to scan. Repeat the option "
            "to combine M6/M7/M8/M8.1 sources."
        ),
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Build a report from the bundled snapshot of the project's observed results.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: unique reports/m9/<timestamp>_research-analytics.",
    )
    parser.add_argument(
        "--title",
        default="Stock ML Lab — Research Analytics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sources = list(args.source)
    if args.demo:
        sources.append(str(Path(__file__).parent / "examples" / "m9_snapshot"))

    if not sources:
        raise SystemExit(
            "Provide at least one --source directory, or use --demo. "
            "Example: python3 visualize.py --demo"
        )

    runs = discover_runs(sources)
    if not runs:
        raise SystemExit("No compatible M6/M7/M8/M8.1 report runs were discovered.")

    if args.output_dir:
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
    else:
        output = unique_run_directory(
            base="reports",
            experiment="m9",
            label="research-analytics",
        )

    report = build_analytics_report(runs, output_dir=output, title=args.title)

    print("\nMilestone 9 — Visualization & Reporting")
    print("=" * 80)
    print(f"Discovered runs: {len(runs)}")
    for run in runs:
        print(f"  {run.kind:5s}  {run.path}")
    print(f"\nFigures generated: {len(report.figure_paths)}")
    for figure in report.figure_paths:
        print(f"  {figure.name}")
    print("\nExecutive findings:")
    for finding in report.findings:
        print(f"  - {finding}")
    print(f"\nHTML report: {report.html_path}")
    print(f"Markdown report: {report.markdown_path}")
    print(f"Report directory: {report.output_dir}")


if __name__ == "__main__":
    main()
