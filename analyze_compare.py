from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from stock_ml_lab.comparison.diagnostics import build_scope_diagnostics
from stock_ml_lab.comparison.report import write_scope_report
from stock_ml_lab.robustness.multiple_testing import benjamini_hochberg


def main() -> None:
    parser = argparse.ArgumentParser(description="Add explicitly post-hoc robustness diagnostics to an M13 run.")
    parser.add_argument("report_dir", type=Path)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    predictions = pd.read_csv(args.report_dir / "predictions.csv")
    outputs = build_scope_diagnostics(
        predictions,
        bootstrap_repetitions=args.bootstrap_repetitions,
        random_state=args.random_state,
        analysis_status="post_hoc_robustness_diagnostic",
    )
    inference = outputs["date_inference"].copy()
    inference["posthoc_hac_qvalue_across_configurations"] = benjamini_hochberg(
        inference["date_hac_pvalue_global_better"].to_numpy(dtype=float)
    )
    outputs["date_inference"] = inference
    for name, frame in outputs.items():
        frame.to_csv(args.report_dir / f"posthoc_{name}.csv", index=False)
    write_scope_report(
        args.report_dir / "report.md",
        title="M13 — Local vs Global Benchmark",
        summary=outputs["summary"],
        inference=inference,
        confirmatory=False,
    )
    manifest_path = args.report_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["posthoc_diagnostics"] = {
        "status": "post_hoc_robustness_diagnostic",
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "random_state": args.random_state,
        "date_clustered_hac": True,
        "multiplicity_family": "all eight M13 task/horizon configurations",
    }
    manifest.setdefault("notes", []).append(
        "These robustness diagnostics were specified after observing M13 and remain explicitly post hoc."
    )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Enriched M13 report: {args.report_dir}")


if __name__ == "__main__":
    main()
