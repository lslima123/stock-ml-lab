from __future__ import annotations

from pathlib import Path

from stock_ml_lab.visualization.report import build_analytics_report
from stock_ml_lab.visualization.sources import discover_runs


def test_demo_snapshot_builds_html_and_figures() -> None:
    snapshot = Path(__file__).parents[1] / "examples" / "m9_snapshot"
    runs = discover_runs([snapshot])
    assert any(run.kind == "m8" for run in runs)
    assert any(run.kind == "m8_1" for run in runs)

    output = snapshot.parent / "_test_output"
    if output.exists():
        import shutil
        shutil.rmtree(output)

    report = build_analytics_report(runs, output_dir=output, title="Test report")
    assert report.html_path.exists()
    assert report.markdown_path.exists()
    assert len(report.figure_paths) >= 6
    html = report.html_path.read_text(encoding="utf-8")
    assert "Discovery vs locked confirmation" in html
    assert "Locked confirmation" in html

    import shutil
    shutil.rmtree(output)
