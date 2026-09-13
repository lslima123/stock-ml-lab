from __future__ import annotations

from pathlib import Path

import pandas as pd


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    selected = frame.loc[:, [column for column in columns if column in frame]].copy()
    if selected.empty:
        return "_No rows available._"
    for column in selected.columns:
        if pd.api.types.is_float_dtype(selected[column]):
            selected[column] = selected[column].map(lambda value: f"{value:.4f}")
    headers = [str(column) for column in selected.columns]
    rows = [[str(value) for value in row] for row in selected.itertuples(index=False, name=None)]
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
        *["| " + " | ".join(row) + " |" for row in rows],
    ])


def write_scope_report(
    output: str | Path,
    *,
    title: str,
    summary: pd.DataFrame,
    inference: pd.DataFrame,
    confirmatory: bool,
) -> Path:
    output = Path(output)
    boundary = (
        "This is a locked independent confirmation. Its two primary hypotheses and "
        "decision rule were fixed before downloading the confirmation assets."
        if confirmatory else
        "These diagnostics are POST HOC robustness analyses of the completed M13 benchmark. "
        "They strengthen interpretation but do not convert M13 into a new confirmation set."
    )
    body = f"""# {title}

## Evidence boundary

{boundary}

## Scope comparison

{_markdown_table(summary, [
    'task', 'horizon', 'assets', 'asset_win_count', 'fold_win_count', 'fold_count',
    'phase_win_count', 'phase_count', 'global_improvement_vs_local_pct',
    'bootstrap_improvement_ci_low_pct', 'bootstrap_improvement_ci_high_pct',
    'date_hac_pvalue_global_better', 'primary_hac_qvalue', 'confirmed',
])}

## Date-clustered inference

Rows sharing an OOS date are treated as a cluster. HAC lags are at least `h-1`, and the
circular block bootstrap resamples date clusters in blocks of at least `h` dates.

{_markdown_table(inference, [
    'task', 'horizon', 'primary_metric', 'rows', 'assets', 'dates',
    'global_improvement_vs_local_pct', 'date_hac_pvalue_global_better',
    'primary_hac_qvalue', 'bootstrap_improvement_ci_low_pct',
    'bootstrap_improvement_ci_high_pct', 'confirmed',
])}

## Interpretation

Positive improvement means lower global-model loss than the same-family local model. The
comparison identifies a training-scope effect; it is not a trading-performance claim.
"""
    output.write_text(body, encoding="utf-8")
    return output
