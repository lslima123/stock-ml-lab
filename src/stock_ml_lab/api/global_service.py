from __future__ import annotations

from datetime import date, datetime

import numpy as np
import pandas as pd

from stock_ml_lab.api.schemas import PredictionRequest
from stock_ml_lab.data.loader import download_ohlcv
from stock_ml_lab.features.technical import FEATURE_COLUMNS, build_features
from stock_ml_lab.global_model.artifacts import GlobalArtifactStore


def _latest_global_features(
    *,
    ticker: str,
    start: date | datetime | str,
    end: date | datetime | str | None,
    ohlcv: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, float, pd.Timestamp]:
    if ohlcv is None:
        ohlcv = download_ohlcv(ticker=ticker, start=start, end=end, interval="1d")
    features = build_features(ohlcv).replace([np.inf, -np.inf], np.nan)
    valid = features.dropna(subset=FEATURE_COLUMNS)
    if valid.empty:
        raise ValueError("No valid latest feature row is available for global inference.")
    latest_X = valid.iloc[[-1]].loc[:, FEATURE_COLUMNS].copy()
    as_of = pd.Timestamp(latest_X.index[-1])
    latest_close = float(ohlcv.loc[as_of, "close"])
    return latest_X, latest_close, as_of


def predict_global(
    request: PredictionRequest,
    *,
    store: GlobalArtifactStore,
    ohlcv: pd.DataFrame | None = None,
) -> dict:
    artifact = store.load(request.task, request.model, request.horizon)
    metadata = artifact.metadata
    feature_names = tuple(metadata.get("feature_names", ()))
    if feature_names != tuple(FEATURE_COLUMNS):
        raise ValueError(
            "Global artifact feature contract does not match the current application."
        )

    latest_X, latest_close, as_of = _latest_global_features(
        ticker=request.ticker,
        start=request.start,
        end=request.end,
        ohlcv=ohlcv,
    )
    estimator = artifact.estimator

    if request.task == "regression":
        predicted_return = float(np.asarray(estimator.predict(latest_X)).reshape(-1)[0])
        predicted_price = latest_close * (1.0 + predicted_return)
        prediction = {
            "predicted_return": predicted_return,
            "predicted_price": float(predicted_price),
            "probability_up": None,
            "predicted_direction": (
                "up" if predicted_return > 1e-12 else "down" if predicted_return < -1e-12 else "flat"
            ),
        }
    else:
        probabilities = np.asarray(estimator.predict_proba(latest_X), dtype=float)
        classes = list(estimator.classes_)
        if 1 not in classes:
            raise ValueError("Global classifier artifact has no positive class.")
        probability_up = float(probabilities[0, classes.index(1)])
        prediction = {
            "predicted_return": None,
            "predicted_price": None,
            "probability_up": probability_up,
            "predicted_direction": "up" if probability_up >= 0.5 else "down",
        }

    return {
        "ticker": request.ticker,
        "scope": "global",
        "task": request.task,
        "model": request.model,
        "horizon": request.horizon,
        "as_of": as_of.date(),
        "latest_close": latest_close,
        "prediction": prediction,
        "training": {
            "training_mode": "pretrained_global_artifact",
            "training_rows": int(metadata["training_rows"]),
            "training_start": metadata["training_start"],
            "training_end": metadata["training_end"],
            "feature_count": len(FEATURE_COLUMNS),
            "feature_set": metadata.get("feature_set", "legacy"),
            "hyperparameters": metadata.get("hyperparameters", {}),
            "artifact_id": artifact.path.name,
            "universe": metadata.get("universe", []),
            "universe_size": int(metadata.get("universe_size", 0)),
            "note": (
                "The prediction uses a pretrained pooled multi-asset artifact. "
                "Ticker identity is not encoded, so the same global model can be "
                "applied to assets outside its training universe. The artifact is "
                "trained offline; no global retraining occurs inside this HTTP request."
            ),
        },
        "disclaimer": (
            "Research/educational output only. Forecasts are uncertain and are not investment advice."
        ),
    }
