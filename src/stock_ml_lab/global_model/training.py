from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import pandas as pd

from stock_ml_lab.global_model.artifacts import GlobalArtifactStore
from stock_ml_lab.global_model.dataset import GlobalPanelBundle, build_global_panel_from_ohlcv
from stock_ml_lab.global_model.models import (
    GLOBAL_CLASSIFICATION_MODEL,
    GLOBAL_HYPERPARAMETERS,
    GLOBAL_REGRESSION_MODEL,
    build_global_estimator,
)
from stock_ml_lab.global_model.validation import (
    evaluate_global_temporal,
    evaluate_unseen_asset_holdout,
)


def fit_global_artifact(
    bundle: GlobalPanelBundle,
    *,
    task: str,
    model: str,
    store: GlobalArtifactStore,
) -> Path:
    estimator = build_global_estimator(task, model)
    y = bundle.regression_y if task == "regression" else bundle.classification_y
    estimator.fit(bundle.X, y)

    metadata = {
        "scope": "global",
        "training_mode": "pretrained_global_artifact",
        "feature_names": list(bundle.feature_names),
        "feature_set": "legacy",
        "universe": list(bundle.tickers),
        "universe_size": len(bundle.tickers),
        "training_rows": len(bundle.frame),
        "training_start": pd.Timestamp(bundle.frame["date"].min()).date().isoformat(),
        "training_end": pd.Timestamp(bundle.frame["date"].max()).date().isoformat(),
        "hyperparameters": GLOBAL_HYPERPARAMETERS[(task, model)],
        "ticker_identity_used": False,
        "generalization_note": (
            "The pooled model uses only dimensionless legacy technical features and does "
            "not encode ticker identity, so it can be applied to tickers outside the training universe."
        ),
    }
    return store.save(
        task=task,
        model=model,
        horizon=bundle.horizon,
        estimator=estimator,
        metadata=metadata,
    )


def train_global_artifacts(
    data_by_ticker: Mapping[str, pd.DataFrame],
    *,
    horizons: tuple[int, ...] = (1, 5, 10, 20),
    output_dir: str | Path = "artifacts/global",
    validate: bool = True,
    n_splits: int = 5,
    test_size_dates: int = 252,
) -> dict[str, pd.DataFrame | list[str] | str]:
    store = GlobalArtifactStore(output_dir)
    temporal_frames: list[pd.DataFrame] = []
    per_asset_frames: list[pd.DataFrame] = []
    unseen_frames: list[pd.DataFrame] = []
    artifacts: list[str] = []

    for horizon in horizons:
        bundle = build_global_panel_from_ohlcv(data_by_ticker, horizon=horizon)

        for task, model in (
            ("regression", GLOBAL_REGRESSION_MODEL),
            ("classification", GLOBAL_CLASSIFICATION_MODEL),
        ):
            if validate:
                folds, assets = evaluate_global_temporal(
                    bundle,
                    task=task,
                    model=model,
                    n_splits=n_splits,
                    test_size_dates=test_size_dates,
                )
                unseen = evaluate_unseen_asset_holdout(
                    bundle,
                    task=task,
                    model=model,
                    test_size_dates=test_size_dates,
                )
                temporal_frames.append(folds)
                per_asset_frames.append(assets)
                unseen_frames.append(unseen)

            artifact = fit_global_artifact(
                bundle,
                task=task,
                model=model,
                store=store,
            )
            artifacts.append(str(artifact))

    manifest = {
        "milestone": "M12",
        "scope": "global",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "universe": sorted(t.strip().upper() for t in data_by_ticker),
        "horizons": list(horizons),
        "models": {
            "regression": GLOBAL_REGRESSION_MODEL,
            "classification": GLOBAL_CLASSIFICATION_MODEL,
        },
        "feature_set": "legacy",
        "ticker_identity_used": False,
        "artifact_count": len(artifacts),
    }
    store.write_manifest(manifest)

    def concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
        good = [f for f in frames if not f.empty]
        return pd.concat(good, ignore_index=True) if good else pd.DataFrame()

    return {
        "artifact_dir": str(store.root),
        "artifacts": artifacts,
        "temporal_metrics": concat(temporal_frames),
        "per_asset_metrics": concat(per_asset_frames),
        "unseen_asset_metrics": concat(unseen_frames),
    }
