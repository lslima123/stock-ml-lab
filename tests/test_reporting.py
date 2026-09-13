from __future__ import annotations

from datetime import datetime

from stock_ml_lab.reporting import unique_run_directory


def test_unique_report_directory_never_overwrites(tmp_path) -> None:
    ts = datetime(2026, 8, 27, 10, 0, 0)
    first = unique_run_directory(base=tmp_path, experiment="m7", label="PETR4.SA", timestamp=ts)
    second = unique_run_directory(base=tmp_path, experiment="m7", label="PETR4.SA", timestamp=ts)
    assert first != second
    assert first.exists() and second.exists()
