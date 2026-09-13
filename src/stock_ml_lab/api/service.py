from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd

from stock_ml_lab.api.registry import build_local_estimator, get_spec
from stock_ml_lab.api.schemas import PredictionRequest
from stock_ml_lab.data.loader import download_ohlcv
from stock_ml_lab.features.technical import FEATURE_COLUMNS, build_features


@dataclass(frozen=True)
class LocalTrainingBundle:
    X: pd.DataFrame
    y: pd.Series
    latest_X: pd.DataFrame
    latest_close: float
    as_of: pd.Timestamp


def _build_local_training_bundle(
    *,
    ticker: str,
    start: date | datetime | str,
    end: date | datetime | str | None,
    horizon: int,
    task: str,
    ohlcv: pd.DataFrame | None = None,
) -> LocalTrainingBundle:
    if ohlcv is None:
        ohlcv = download_ohlcv(ticker=ticker, start=start, end=end, interval="1d")
    if len(ohlcv) < 80 + horizon:
        raise ValueError("Not enough history for local training and inference.")

    features = build_features(ohlcv).replace([np.inf, -np.inf], np.nan)
    close = ohlcv["close"].astype(float)
    forward = (close.shift(-horizon) / close - 1.0).rename(f"forward_return_{horizon}d")

    supervised = features.join(forward, how="inner").dropna(
        subset=list(FEATURE_COLUMNS) + [forward.name]
    )
    if len(supervised) < 60:
        raise ValueError("Too few supervised observations after feature construction.")

    X = supervised.loc[:, FEATURE_COLUMNS].copy()
    if task == "regression":
        y = supervised[forward.name].astype(float).copy()
    elif task == "classification":
        y = (supervised[forward.name] > 0.0).astype(int).rename("direction")
    else:
        raise ValueError(f"Unsupported task: {task}")

    latest_candidates = features.dropna(subset=FEATURE_COLUMNS)
    if latest_candidates.empty:
        raise ValueError("No valid latest feature row is available for inference.")
    latest_X = latest_candidates.iloc[[-1]].loc[:, FEATURE_COLUMNS].copy()
    as_of = pd.Timestamp(latest_X.index[-1])
    latest_close = float(close.loc[as_of])

    return LocalTrainingBundle(
        X=X,
        y=y,
        latest_X=latest_X,
        latest_close=latest_close,
        as_of=as_of,
    )


def _direction_from_value(value: float, *, epsilon: float = 1e-12) -> str:
    if value > epsilon:
        return "up"
    if value < -epsilon:
        return "down"
    return "flat"


def predict_local(
    request: PredictionRequest,
    *,
    ohlcv: pd.DataFrame | None = None,
) -> dict:
    bundle = _build_local_training_bundle(
        ticker=request.ticker,
        start=request.start,
        end=request.end,
        horizon=request.horizon,
        task=request.task,
        ohlcv=ohlcv,
    )
    estimator = build_local_estimator(request.task, request.model, random_state=42)
    estimator.fit(bundle.X, bundle.y)
    spec = get_spec(request.task, request.model)

    if request.task == "regression":
        predicted_return = float(np.asarray(estimator.predict(bundle.latest_X)).reshape(-1)[0])
        predicted_price = bundle.latest_close * (1.0 + predicted_return)
        prediction = {
            "predicted_return": predicted_return,
            "predicted_price": float(predicted_price),
            "probability_up": None,
            "predicted_direction": _direction_from_value(predicted_return),
        }
    else:
        probabilities = np.asarray(estimator.predict_proba(bundle.latest_X), dtype=float)
        classes = list(estimator.classes_)
        try:
            up_index = classes.index(1)
            probability_up = float(probabilities[0, up_index])
        except ValueError:
            probability_up = float(bundle.y.mean())
        prediction = {
            "predicted_return": None,
            "predicted_price": None,
            "probability_up": probability_up,
            "predicted_direction": "up" if probability_up >= 0.5 else "down",
        }

    return {
        "ticker": request.ticker,
        "scope": "local",
        "task": request.task,
        "model": request.model,
        "horizon": request.horizon,
        "as_of": bundle.as_of.date(),
        "latest_close": bundle.latest_close,
        "prediction": prediction,
        "training": {
            "training_mode": "on_demand_local_fit",
            "training_rows": len(bundle.X),
            "training_start": pd.Timestamp(bundle.X.index[0]).date(),
            "training_end": pd.Timestamp(bundle.X.index[-1]).date(),
            "feature_count": len(FEATURE_COLUMNS),
            "feature_set": "legacy",
            "hyperparameters": spec.hyperparameters,
            "note": (
                "The API fits the selected local model on demand using all supervised history "
                "available before the inference row. This is a product inference preset, not a "
                "claim that the exact fit reproduces nested-CV research metrics."
            ),
        },
        "disclaimer": (
            "Research/educational output only. Forecasts are uncertain and are not investment advice."
        ),
    }


def market_snapshot(
    *,
    ticker: str,
    start: date | datetime | str,
    end: date | datetime | str | None,
) -> dict:
    data = download_ohlcv(ticker=ticker, start=start, end=end, interval="1d")
    close = data["close"].astype(float)
    last = data.iloc[-1]
    ret = close.pct_change(fill_method=None).iloc[-1]
    return {
        "ticker": ticker.strip().upper(),
        "as_of": pd.Timestamp(data.index[-1]).date(),
        "close": float(last["close"]),
        "return_1d": None if pd.isna(ret) else float(ret),
        "volume": float(last["volume"]),
    }
