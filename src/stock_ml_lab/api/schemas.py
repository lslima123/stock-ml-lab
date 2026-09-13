from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


TaskName = Literal["regression", "classification"]
ScopeName = Literal["local", "global", "compare"]
Horizon = Literal[1, 5, 10, 20]
Direction = Literal["up", "down", "flat"]

REGRESSION_MODELS = ("zero", "ridge", "random_forest", "xgboost", "catboost")
CLASSIFICATION_MODELS = ("prior", "logistic", "random_forest", "xgboost", "catboost")


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    service: str = "stock-ml-lab-api"
    version: str


class ScopeCapability(BaseModel):
    name: ScopeName
    available: bool
    description: str


class ModelCapability(BaseModel):
    key: str
    label: str
    task: TaskName
    scope: Literal["local", "global"]
    available: bool = True


class CapabilitiesResponse(BaseModel):
    version: str
    tasks: tuple[TaskName, ...]
    horizons: tuple[int, ...]
    feature_sets: tuple[str, ...]
    scopes: tuple[ScopeCapability, ...]
    models: tuple[ModelCapability, ...]
    docs_url: str = "/docs"


class PredictionRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    ticker: str = Field(min_length=1, max_length=32, examples=["PETR4.SA"])
    task: TaskName
    model: str = Field(min_length=1, max_length=64)
    scope: ScopeName = "local"
    horizon: Horizon = 1
    start: date = date(2018, 1, 1)
    end: date | None = None
    feature_set: Literal["legacy"] = "legacy"

    @model_validator(mode="after")
    def validate_model_for_task(self) -> "PredictionRequest":
        self.ticker = self.ticker.upper()
        self.model = self.model.lower()
        allowed = REGRESSION_MODELS if self.task == "regression" else CLASSIFICATION_MODELS
        if self.model not in allowed:
            raise ValueError(
                f"Model '{self.model}' is not valid for task '{self.task}'. "
                f"Allowed: {', '.join(allowed)}."
            )
        if self.end is not None and self.end <= self.start:
            raise ValueError("end must be later than start. yfinance end semantics are exclusive.")
        return self


class PredictionValue(BaseModel):
    predicted_return: float | None = None
    predicted_price: float | None = None
    probability_up: float | None = None
    predicted_direction: Direction


class TrainingContext(BaseModel):
    training_mode: Literal["on_demand_local_fit", "pretrained_global_artifact"]
    training_rows: int
    training_start: date
    training_end: date
    feature_count: int
    feature_set: str
    hyperparameters: dict[str, Any]
    note: str
    artifact_id: str | None = None
    universe: list[str] | None = None
    universe_size: int | None = None


class PredictionResponse(BaseModel):
    ticker: str
    scope: Literal["local", "global"]
    task: TaskName
    model: str
    horizon: int
    as_of: date
    latest_close: float
    prediction: PredictionValue
    training: TrainingContext
    disclaimer: str



class ComparePredictionResponse(BaseModel):
    ticker: str
    task: TaskName
    horizon: int
    as_of: date
    local: PredictionResponse
    global_: PredictionResponse = Field(alias="global")
    comparison: dict[str, Any]
    disclaimer: str

    model_config = ConfigDict(populate_by_name=True)


class GlobalArtifactStatusResponse(BaseModel):
    root: str
    complete: bool
    records: list[dict[str, Any]]


class MarketSnapshotResponse(BaseModel):
    ticker: str
    as_of: date
    close: float
    return_1d: float | None
    volume: float


class ResearchSummaryResponse(BaseModel):
    source: str
    findings: list[str]
    discovery_candidates: list[dict[str, Any]]
    locked_confirmation: list[dict[str, Any]]
    scope_benchmark: list[dict[str, Any]]
    scope_confirmation: list[dict[str, Any]]


class RecordsResponse(BaseModel):
    source: str
    count: int
    records: list[dict[str, Any]]
