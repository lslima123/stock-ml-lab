from __future__ import annotations

import pytest
from pydantic import ValidationError

from stock_ml_lab.api.schemas import PredictionRequest


def test_prediction_request_normalizes_ticker_and_model() -> None:
    request = PredictionRequest(
        ticker=" petr4.sa ", task="regression", model="RIDGE", horizon=20
    )
    assert request.ticker == "PETR4.SA"
    assert request.model == "ridge"


def test_prediction_request_rejects_model_for_wrong_task() -> None:
    with pytest.raises(ValidationError):
        PredictionRequest(ticker="AAPL", task="classification", model="ridge")


def test_prediction_request_accepts_future_global_contract() -> None:
    request = PredictionRequest(
        ticker="AAPL", task="classification", model="logistic", scope="global"
    )
    assert request.scope == "global"
