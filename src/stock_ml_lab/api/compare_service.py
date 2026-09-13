from __future__ import annotations

from stock_ml_lab.api.global_service import predict_global
from stock_ml_lab.api.schemas import PredictionRequest
from stock_ml_lab.api.service import predict_local
from stock_ml_lab.data.loader import download_ohlcv
from stock_ml_lab.global_model.artifacts import GlobalArtifactStore

COMPARE_MODELS = {
    "regression": ("ridge", "ridge"),
    "classification": ("logistic", "logistic"),
}


def predict_compare(request: PredictionRequest, *, store: GlobalArtifactStore) -> dict:
    if request.task not in COMPARE_MODELS:
        raise ValueError(f"Unsupported comparison task: {request.task}")
    local_model, global_model = COMPARE_MODELS[request.task]
    if request.model != local_model:
        raise ValueError(
            f"M13 same-family comparison requires local model '{local_model}' for {request.task}."
        )
    if not store.exists(request.task, global_model, request.horizon):
        raise FileNotFoundError(
            f"Global artifact unavailable for {request.task}/{global_model}/{request.horizon}d."
        )

    local_request = request.model_copy(update={"scope": "local", "model": local_model})
    global_request = request.model_copy(update={"scope": "global", "model": global_model})
    ohlcv = download_ohlcv(
        ticker=request.ticker,
        start=request.start,
        end=request.end,
        interval="1d",
    )
    local = predict_local(local_request, ohlcv=ohlcv)
    global_ = predict_global(global_request, store=store, ohlcv=ohlcv)

    if local["as_of"] != global_["as_of"]:
        raise ValueError("Local and global inference rows are not aligned.")

    if request.task == "regression":
        local_value = local["prediction"]["predicted_return"]
        global_value = global_["prediction"]["predicted_return"]
        delta = float(global_value - local_value)
        comparison = {
            "metric": "predicted_return",
            "local_value": float(local_value),
            "global_value": float(global_value),
            "global_minus_local": delta,
        }
    else:
        local_value = local["prediction"]["probability_up"]
        global_value = global_["prediction"]["probability_up"]
        delta = float(global_value - local_value)
        comparison = {
            "metric": "probability_up",
            "local_value": float(local_value),
            "global_value": float(global_value),
            "global_minus_local": delta,
        }

    return {
        "ticker": request.ticker,
        "task": request.task,
        "horizon": request.horizon,
        "as_of": local["as_of"],
        "local": local,
        "global": global_,
        "comparison": comparison,
        "disclaimer": local["disclaimer"],
    }
