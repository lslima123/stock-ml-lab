from __future__ import annotations

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from stock_ml_lab.api.app import app, global_store
from stock_ml_lab.api import service


client = TestClient(app)


def make_ohlcv(n: int = 320) -> pd.DataFrame:
    index = pd.bdate_range("2024-01-02", periods=n)
    x = np.arange(n, dtype=float)
    close = 80.0 + 0.03 * x + 3.0 * np.sin(x / 7.0) + 1.0 * np.sin(x / 2.5)
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": 500_000.0 + 20_000.0 * (1 + np.sin(x / 4.0)),
        },
        index=index,
    )


def test_health_and_docs_contract() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "stock-ml-lab-api"


def test_capabilities_report_unavailable_scopes_without_artifacts(monkeypatch) -> None:
    monkeypatch.setattr(global_store, "is_complete", lambda: False)
    response = client.get("/api/v1/capabilities")
    assert response.status_code == 200
    payload = response.json()
    scopes = {item["name"]: item["available"] for item in payload["scopes"]}
    assert scopes == {"local": True, "global": False, "compare": False}
    assert 20 in payload["horizons"]


def test_global_prediction_requires_pretrained_artifact(monkeypatch) -> None:
    monkeypatch.setattr(global_store, "exists", lambda *_: False)
    response = client.post(
        "/api/v1/predict",
        json={
            "ticker": "PETR4.SA",
            "task": "regression",
            "model": "ridge",
            "scope": "global",
            "horizon": 20,
        },
    )
    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "global_artifact_unavailable"


def test_local_prediction_endpoint(monkeypatch) -> None:
    data = make_ohlcv()
    monkeypatch.setattr(service, "download_ohlcv", lambda **_: data)
    response = client.post(
        "/api/v1/predict",
        json={
            "ticker": "PETR4.SA",
            "task": "regression",
            "model": "ridge",
            "scope": "local",
            "horizon": 10,
            "start": "2024-01-01",
        },
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["ticker"] == "PETR4.SA"
    assert payload["horizon"] == 10
    assert payload["prediction"]["predicted_return"] is not None


def test_research_endpoints_have_bundled_snapshot() -> None:
    summary = client.get("/api/v1/research/summary")
    assert summary.status_code == 200
    assert len(summary.json()["findings"]) >= 3
    assert len(summary.json()["scope_confirmation"]) == 2

    confirmation = client.get(
        "/api/v1/research/confirmation", params={"task": "regression", "horizon": 20}
    )
    assert confirmation.status_code == 200
    assert confirmation.json()["count"] == 9

    scope = client.get(
        "/api/v1/research/scope-benchmark",
        params={"task": "classification", "horizon": 5, "level": "summary"},
    )
    assert scope.status_code == 200
    assert scope.json()["count"] == 1
    assert scope.json()["records"][0]["analysis_status"] == "post_hoc_robustness_diagnostic"

    scope_confirmation = client.get(
        "/api/v1/research/scope-confirmation",
        params={"task": "regression", "horizon": 5, "level": "summary"},
    )
    assert scope_confirmation.status_code == 200
    assert scope_confirmation.json()["count"] == 1
    record = scope_confirmation.json()["records"][0]
    assert record["analysis_status"] == "locked_independent_scope_confirmation"
    assert record["confirmed"] is True
