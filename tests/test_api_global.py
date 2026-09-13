from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from stock_ml_lab.api.global_service import predict_global
from stock_ml_lab.api.schemas import PredictionRequest
from stock_ml_lab.global_model.artifacts import GlobalArtifactStore
from stock_ml_lab.global_model.dataset import build_global_panel_from_ohlcv
from stock_ml_lab.global_model.training import fit_global_artifact


def make(seed: int, n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2022-01-03", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, n)))
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": rng.lognormal(14, 0.2, n),
        },
        index=index,
    )


def test_predict_global_uses_pretrained_artifact(monkeypatch, tmp_path) -> None:
    source = {"AAA": make(1), "BBB": make(2)}
    bundle = build_global_panel_from_ohlcv(source, horizon=5)
    store = GlobalArtifactStore(tmp_path)
    fit_global_artifact(bundle, task="regression", model="ridge", store=store)

    inference_data = make(3)
    monkeypatch.setattr(
        "stock_ml_lab.api.global_service.download_ohlcv",
        lambda **kwargs: inference_data,
    )

    request = PredictionRequest(
        ticker="NEW",
        task="regression",
        model="ridge",
        scope="global",
        horizon=5,
        start=date(2022, 1, 1),
    )
    response = predict_global(request, store=store)
    assert response["scope"] == "global"
    assert response["ticker"] == "NEW"
    assert response["training"]["training_mode"] == "pretrained_global_artifact"
    assert response["training"]["universe_size"] == 2
    assert response["prediction"]["predicted_return"] is not None
