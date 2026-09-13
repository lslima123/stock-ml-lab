from datetime import date

import numpy as np
import pandas as pd

from stock_ml_lab.api.compare_service import predict_compare
from stock_ml_lab.api.schemas import PredictionRequest
from stock_ml_lab.global_model.artifacts import GlobalArtifactStore
from stock_ml_lab.global_model.dataset import build_global_panel_from_ohlcv
from stock_ml_lab.global_model.training import fit_global_artifact


def make(seed: int, n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    close = 100 * np.exp(np.cumsum(rng.normal(.0002, .01, n)))
    return pd.DataFrame({
        "open": close,
        "high": close * 1.01,
        "low": close * .99,
        "close": close,
        "volume": rng.lognormal(14, .2, n),
    }, index=idx)


def test_predict_compare_returns_both_scopes(monkeypatch, tmp_path) -> None:
    source = {"AAA": make(1), "BBB": make(2)}
    bundle = build_global_panel_from_ohlcv(source, horizon=5)
    store = GlobalArtifactStore(tmp_path)
    fit_global_artifact(bundle, task="regression", model="ridge", store=store)

    inference = make(3)
    downloads = []

    def fake_download(**kwargs):
        downloads.append(kwargs)
        return inference

    monkeypatch.setattr("stock_ml_lab.api.compare_service.download_ohlcv", fake_download)

    request = PredictionRequest(
        ticker="NEW",
        task="regression",
        model="ridge",
        scope="compare",
        horizon=5,
        start=date(2022, 1, 1),
    )
    payload = predict_compare(request, store=store)
    assert payload["local"]["scope"] == "local"
    assert payload["global"]["scope"] == "global"
    assert payload["local"]["as_of"] == payload["global"]["as_of"]
    assert len(downloads) == 1
    assert payload["comparison"]["metric"] == "predicted_return"
    assert payload["comparison"]["global_minus_local"] == payload["global"]["prediction"]["predicted_return"] - payload["local"]["prediction"]["predicted_return"]


def test_predict_compare_classification_returns_probability_delta(monkeypatch, tmp_path) -> None:
    source = {"AAA": make(1), "BBB": make(2)}
    bundle = build_global_panel_from_ohlcv(source, horizon=20)
    store = GlobalArtifactStore(tmp_path)
    fit_global_artifact(bundle, task="classification", model="logistic", store=store)

    inference = make(3)
    downloads = []

    def fake_download(**kwargs):
        downloads.append(kwargs)
        return inference

    monkeypatch.setattr("stock_ml_lab.api.compare_service.download_ohlcv", fake_download)

    request = PredictionRequest(
        ticker="NEW",
        task="classification",
        model="logistic",
        scope="compare",
        horizon=20,
        start=date(2022, 1, 1),
    )
    payload = predict_compare(request, store=store)

    local_probability = payload["local"]["prediction"]["probability_up"]
    global_probability = payload["global"]["prediction"]["probability_up"]
    assert payload["local"]["scope"] == "local"
    assert payload["global"]["scope"] == "global"
    assert payload["local"]["as_of"] == payload["global"]["as_of"]
    assert len(downloads) == 1
    assert 0.0 <= local_probability <= 1.0
    assert 0.0 <= global_probability <= 1.0
    assert payload["comparison"]["metric"] == "probability_up"
    assert payload["comparison"]["local_value"] == local_probability
    assert payload["comparison"]["global_value"] == global_probability
    assert payload["comparison"]["global_minus_local"] == (
        global_probability - local_probability
    )
