from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
import json

import pandas as pd

from stock_ml_lab.visualization.normalize import (
    combine_m6_asset_results,
    combine_m8_runs,
    combine_m81_asset_results,
    combine_m81_candidate_summary,
    latest_m7_summary,
)
from stock_ml_lab.visualization.plots import (
    plot_confirmation_assets,
    plot_discovery_vs_confirmation,
    plot_m6_cross_asset,
    plot_m7_auc,
    plot_m8_feature_heatmap,
    plot_m8_horizon_classification,
    plot_m8_horizon_regression,
    plot_phase_win_rates,
)
from stock_ml_lab.visualization.sources import LoadedRun
from stock_ml_lab.visualization.story import (
    discovery_confirmation_metrics,
    narrative_findings,
)


@dataclass(frozen=True)
class AnalyticsReport:
    output_dir: Path
    html_path: Path
    markdown_path: Path
    figure_paths: tuple[Path, ...]
    findings: tuple[str, ...]


def _fmt(value, *, digits: int = 2, pct: bool = False) -> str:
    if pd.isna(value):
        return "N/A"
    if pct:
        return f"{100.0 * float(value):.{digits}f}%"
    return f"{float(value):.{digits}f}"


def _table_html(frame: pd.DataFrame, *, max_rows: int = 30) -> str:
    if frame.empty:
        return "<p class='muted'>No data available.</p>"
    view = frame.head(max_rows).copy()
    return view.to_html(index=False, border=0, classes="data-table", escape=True)


def _confirmation_summary_view(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    columns = [
        "task",
        "horizon",
        "model",
        "n_assets",
        "primary_win_count",
        "primary_win_rate",
        "primary_win_sign_test_p",
        "mean_primary_improvement_pct",
        "median_primary_improvement_pct",
        "primary_hac_fdr_sig_count",
        "primary_bootstrap_positive_count",
        "mean_phase_win_rate",
    ]
    out = frame[[c for c in columns if c in frame.columns]].copy()
    if "primary_win_rate" in out:
        out["primary_win_rate"] = out["primary_win_rate"].map(lambda x: _fmt(x, pct=True))
    if "mean_phase_win_rate" in out:
        out["mean_phase_win_rate"] = out["mean_phase_win_rate"].map(lambda x: _fmt(x, pct=True))
    for col in ["mean_primary_improvement_pct", "median_primary_improvement_pct"]:
        if col in out:
            out[col] = out[col].map(lambda x: f"{float(x):+.2f}%")
    if "primary_win_sign_test_p" in out:
        out["primary_win_sign_test_p"] = out["primary_win_sign_test_p"].map(
            lambda x: f"{float(x):.4f}"
        )
    return out


def _m8_discovery_view(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    mask = (
        frame["feature_set"].eq("legacy")
        & (
            (frame["task"].eq("classification") & frame["model_key"].eq("logistic"))
            | (frame["task"].eq("regression") & frame["model_key"].eq("ridge"))
        )
    )
    out = frame.loc[mask].copy()
    columns = [
        "task",
        "horizon",
        "model",
        "log_loss_improvement_vs_prior_pct",
        "roc_auc",
        "rmse_improvement_vs_zero_pct",
        "directional_accuracy",
    ]
    out = out[[c for c in columns if c in out.columns]].sort_values(["task", "horizon"])
    for col in ["log_loss_improvement_vs_prior_pct", "rmse_improvement_vs_zero_pct"]:
        if col in out:
            out[col] = out[col].map(
                lambda x: "N/A" if pd.isna(x) else f"{float(x):+.2f}%"
            )
    if "roc_auc" in out:
        out["roc_auc"] = out["roc_auc"].map(lambda x: "N/A" if pd.isna(x) else f"{float(x):.3f}")
    if "directional_accuracy" in out:
        out["directional_accuracy"] = out["directional_accuracy"].map(
            lambda x: "N/A" if pd.isna(x) else _fmt(x, pct=True)
        )
    return out


def _write_markdown(
    path: Path,
    *,
    findings: list[str],
    figures: list[Path],
    m8: pd.DataFrame,
    m81_summary: pd.DataFrame,
) -> None:
    lines = [
        "# Stock ML Lab — M9 Research Report",
        "",
        "## Executive findings",
        "",
    ]
    lines.extend([f"- {item}" for item in findings])
    lines += [
        "",
        "## Interpretation",
        "",
        "The reporting layer distinguishes exploratory discovery from locked confirmation. "
        "Positive discovery metrics are not promoted as robust predictive edge unless they "
        "survive the predeclared confirmation universe, dependence-aware inference, "
        "bootstrap uncertainty and non-overlapping phase diagnostics.",
        "",
        "## Figures",
        "",
    ]
    for fig in figures:
        lines.append(f"- `{fig.name}`")
    lines += [
        "",
        "## Discovery table",
        "",
        _m8_discovery_view(m8).to_markdown(index=False) if not m8.empty else "No M8 data.",
        "",
        "## Locked confirmation summary",
        "",
        _confirmation_summary_view(m81_summary).to_markdown(index=False)
        if not m81_summary.empty
        else "No M8.1 data.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _figure_card(fig: Path, title: str) -> str:
    return (
        "<section class='figure-card'>"
        f"<h3>{escape(title)}</h3>"
        f"<img src='figures/{escape(fig.name)}' alt='{escape(title)}'>"
        "</section>"
    )


def build_analytics_report(
    runs: list[LoadedRun],
    *,
    output_dir: str | Path,
    title: str = "Stock ML Lab — Research Analytics",
) -> AnalyticsReport:
    out = Path(output_dir)
    figures_dir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    m8 = combine_m8_runs(runs)
    m81_assets = combine_m81_asset_results(runs)
    m81_summary = combine_m81_candidate_summary(runs)
    m7 = latest_m7_summary(runs)
    m6 = combine_m6_asset_results(runs)

    metrics = discovery_confirmation_metrics(m8, m81_summary)
    findings = narrative_findings(m8, m81_summary, m81_assets)

    generated: list[tuple[Path, str]] = []

    def add(path: Path | None, title_text: str) -> None:
        if path is not None:
            generated.append((path, title_text))

    add(
        plot_discovery_vs_confirmation(
            metrics, figures_dir / "discovery_vs_confirmation.png"
        ),
        "Discovery vs locked confirmation",
    )
    add(
        plot_m8_horizon_classification(
            m8, figures_dir / "m8_logistic_auc_by_horizon.png"
        ),
        "Discovery: Logistic AUC by horizon",
    )
    add(
        plot_m8_horizon_regression(
            m8, figures_dir / "m8_ridge_rmse_by_horizon.png"
        ),
        "Discovery: Ridge RMSE improvement by horizon",
    )
    add(
        plot_m8_feature_heatmap(
            m8, figures_dir / "m8_feature_horizon_auc.png"
        ),
        "Exploratory feature-set × horizon map",
    )
    for task in ("classification", "regression"):
        for horizon in (10, 20):
            add(
                plot_confirmation_assets(
                    m81_assets,
                    task=task,
                    horizon=horizon,
                    path=figures_dir / f"m81_{task}_{horizon}d_assets.png",
                ),
                f"Locked confirmation: {task} {horizon}d across assets",
            )
    add(
        plot_phase_win_rates(
            m81_assets, figures_dir / "m81_phase_win_rates.png"
        ),
        "Locked confirmation: non-overlapping phase stability",
    )
    add(
        plot_m7_auc(m7, figures_dir / "m7_one_day_auc.png"),
        "M7 one-day classification benchmark",
    )
    add(
        plot_m6_cross_asset(m6, figures_dir / "m6_cross_asset_rmse.png"),
        "M6 daily cross-asset regression robustness",
    )

    figure_paths = [p for p, _ in generated]

    # Machine-readable normalized outputs make M9 useful for the later API/UI.
    if not m8.empty:
        m8.to_csv(out / "normalized_m8_results.csv", index=False)
    if not m81_assets.empty:
        m81_assets.to_csv(out / "normalized_m81_asset_results.csv", index=False)
    if not m81_summary.empty:
        m81_summary.to_csv(out / "normalized_m81_candidate_summary.csv", index=False)
    if not m7.empty:
        m7.to_csv(out / "normalized_m7_summary.csv", index=False)
    if not m6.empty:
        m6.to_csv(out / "normalized_m6_asset_results.csv", index=False)

    report_meta = {
        "title": title,
        "run_count": len(runs),
        "source_runs": [
            {"kind": run.kind, "path": str(run.path), "manifest": run.manifest}
            for run in runs
        ],
        "findings": findings,
        "figures": [p.name for p in figure_paths],
    }
    (out / "report_manifest.json").write_text(
        json.dumps(report_meta, indent=2, default=str), encoding="utf-8"
    )

    md_path = out / "report.md"
    _write_markdown(
        md_path,
        findings=findings,
        figures=figure_paths,
        m8=m8,
        m81_summary=m81_summary,
    )

    finding_html = "".join(f"<li>{escape(item)}</li>" for item in findings)
    figure_html = "".join(_figure_card(path, heading) for path, heading in generated)

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title>
<style>
:root {{
  --bg: #f5f5f3;
  --paper: #ffffff;
  --ink: #171717;
  --muted: #666666;
  --line: #ddddda;
  --accent: #25364a;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
               "Segoe UI", sans-serif;
  background: var(--bg);
  color: var(--ink);
  line-height: 1.55;
}}
header {{
  background: var(--accent);
  color: white;
  padding: 54px max(24px, calc((100vw - 1180px)/2));
}}
header h1 {{ margin: 0 0 10px; font-size: clamp(2rem, 4vw, 3.4rem); }}
header p {{ margin: 0; max-width: 850px; opacity: .9; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 34px 24px 70px; }}
section {{
  background: var(--paper);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 24px;
  margin: 0 0 22px;
}}
h2 {{ margin-top: 0; }}
.findings li {{ margin-bottom: 9px; }}
.figure-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
  gap: 20px;
}}
.figure-card {{ margin: 0; }}
.figure-card img {{ width: 100%; height: auto; display: block; }}
.data-table {{
  border-collapse: collapse;
  width: 100%;
  font-size: 0.9rem;
  overflow-x: auto;
  display: block;
}}
.data-table th, .data-table td {{
  border-bottom: 1px solid var(--line);
  padding: 8px 10px;
  text-align: right;
  white-space: nowrap;
}}
.data-table th:first-child, .data-table td:first-child {{ text-align: left; }}
.muted {{ color: var(--muted); }}
.note {{
  border-left: 4px solid var(--accent);
  padding-left: 16px;
  color: var(--muted);
}}
footer {{ color: var(--muted); margin-top: 30px; }}
</style>
</head>
<body>
<header>
  <h1>{escape(title)}</h1>
  <p>
    Leakage-aware experiment analytics. Exploratory discovery is shown separately
    from locked confirmation so attractive in-sample research findings are not
    mistaken for generalizable predictive edge.
  </p>
</header>
<main>
<section>
  <h2>Executive findings</h2>
  <ul class="findings">{finding_html or "<li>No narrative findings available.</li>"}</ul>
  <p class="note">
    Interpretation rule: a configuration is not promoted from one attractive ticker.
    Confirmation evidence is evaluated across assets, HAC/FDR inference, block-bootstrap
    intervals and non-overlapping phase stability.
  </p>
</section>
<section>
  <h2>Discovery candidates</h2>
  {_table_html(_m8_discovery_view(m8))}
</section>
<section>
  <h2>Locked confirmation</h2>
  {_table_html(_confirmation_summary_view(m81_summary))}
</section>
<div class="figure-grid">
  {figure_html}
</div>
<section>
  <h2>Provenance</h2>
  <p class="muted">This report was built from {len(runs)} persisted experiment run(s).</p>
  <ul>
    {''.join(f"<li><strong>{escape(r.kind)}</strong>: {escape(str(r.path))}</li>" for r in runs)}
  </ul>
</section>
<footer>
  Stock ML Lab · Milestone 9 · static research report generated from persisted experiment artifacts.
</footer>
</main>
</body>
</html>
"""
    html_path = out / "index.html"
    html_path.write_text(html, encoding="utf-8")

    return AnalyticsReport(
        output_dir=out,
        html_path=html_path,
        markdown_path=md_path,
        figure_paths=tuple(figure_paths),
        findings=tuple(findings),
    )
