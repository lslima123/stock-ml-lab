from __future__ import annotations

import pandas as pd

from stock_ml_lab.visualization.sources import LoadedRun


def combine_m8_runs(runs: list[LoadedRun]) -> pd.DataFrame:
    frames = []
    for order, run in enumerate(runs):
        if run.kind != "m8":
            continue
        frame = run.tables.get("results", pd.DataFrame()).copy()
        if frame.empty:
            continue
        frame["_source_order"] = order
        frame["_source_path"] = str(run.path)
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True, sort=False)
    keys = ["task", "horizon", "feature_set", "model_key"]
    present = [key for key in keys if key in merged.columns]
    if present:
        merged = (
            merged.sort_values("_source_order")
            .drop_duplicates(subset=present, keep="last")
            .reset_index(drop=True)
        )
    return merged


def combine_m81_asset_results(runs: list[LoadedRun]) -> pd.DataFrame:
    frames = []
    for order, run in enumerate(runs):
        if run.kind != "m8_1":
            continue
        frame = run.tables.get("asset_results", pd.DataFrame()).copy()
        if frame.empty:
            continue
        frame["_source_order"] = order
        frame["_source_path"] = str(run.path)
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True, sort=False)
    keys = ["ticker", "task", "horizon", "feature_set", "model_key"]
    present = [key for key in keys if key in merged.columns]
    if present:
        merged = (
            merged.sort_values("_source_order")
            .drop_duplicates(subset=present, keep="last")
            .reset_index(drop=True)
        )
    return merged


def combine_m81_candidate_summary(runs: list[LoadedRun]) -> pd.DataFrame:
    frames = []
    for order, run in enumerate(runs):
        if run.kind != "m8_1":
            continue
        frame = run.tables.get("candidate_summary", pd.DataFrame()).copy()
        if frame.empty:
            continue
        frame["_source_order"] = order
        frame["_source_path"] = str(run.path)
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True, sort=False)
    keys = ["task", "horizon", "feature_set", "model"]
    present = [key for key in keys if key in merged.columns]
    if present:
        merged = (
            merged.sort_values("_source_order")
            .drop_duplicates(subset=present, keep="last")
            .reset_index(drop=True)
        )
    return merged


def combine_m81_phases(runs: list[LoadedRun]) -> pd.DataFrame:
    frames = []
    for order, run in enumerate(runs):
        if run.kind != "m8_1":
            continue
        frame = run.tables.get("phase_results", pd.DataFrame()).copy()
        if frame.empty:
            continue
        frame["_source_order"] = order
        frame["_source_path"] = str(run.path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True, sort=False)
    keys = ["ticker", "task", "horizon", "phase"]
    present = [key for key in keys if key in merged.columns]
    if present:
        merged = merged.sort_values("_source_order").drop_duplicates(present, keep="last")
    return merged.reset_index(drop=True)


def latest_m7_summary(runs: list[LoadedRun]) -> pd.DataFrame:
    candidates = [r for r in runs if r.kind == "m7" and not r.tables.get("summary", pd.DataFrame()).empty]
    if not candidates:
        return pd.DataFrame()
    # Paths include timestamps in normal use, so lexical order is a stable fallback.
    run = sorted(candidates, key=lambda r: str(r.path))[-1]
    frame = run.tables["summary"].copy()
    frame["_source_path"] = str(run.path)
    return frame


def combine_m6_asset_results(runs: list[LoadedRun]) -> pd.DataFrame:
    frames = []
    for order, run in enumerate(runs):
        if run.kind != "m6":
            continue
        frame = run.tables.get("asset_results", pd.DataFrame()).copy()
        if frame.empty:
            continue
        frame["_source_order"] = order
        frame["_source_path"] = str(run.path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True, sort=False)
    keys = ["ticker", "model_key"]
    present = [key for key in keys if key in merged.columns]
    if present:
        merged = merged.sort_values("_source_order").drop_duplicates(present, keep="last")
    return merged.reset_index(drop=True)
