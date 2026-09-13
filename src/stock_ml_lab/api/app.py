from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError, version

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from stock_ml_lab.api.analytics import AnalyticsRepository
from stock_ml_lab.api.compare_service import predict_compare
from stock_ml_lab.api.frontend import frontend_dist_dir, frontend_is_built
from stock_ml_lab.api.global_service import predict_global
from stock_ml_lab.api.registry import all_specs
from stock_ml_lab.api.schemas import (
    CapabilitiesResponse,
    ComparePredictionResponse,
    GlobalArtifactStatusResponse,
    HealthResponse,
    MarketSnapshotResponse,
    ModelCapability,
    PredictionRequest,
    PredictionResponse,
    RecordsResponse,
    ResearchSummaryResponse,
    ScopeCapability,
)
from stock_ml_lab.api.service import market_snapshot, predict_local
from stock_ml_lab.data.loader import MarketDataError
from stock_ml_lab.global_model.artifacts import GlobalArtifactStore


def _package_version() -> str:
    try:
        return version("stock-ml-lab")
    except PackageNotFoundError:
        return "0.15.0"


def _cors_origins() -> list[str]:
    raw = os.getenv(
        "STOCK_ML_LAB_CORS_ORIGINS",
        "http://localhost:3000,http://localhost:5173,http://127.0.0.1:5173",
    )
    return [value.strip() for value in raw.split(",") if value.strip()]


APP_VERSION = _package_version()
analytics = AnalyticsRepository.from_environment()
global_store = GlobalArtifactStore.from_environment()

app = FastAPI(
    title="Stock ML Lab API",
    summary="Leakage-aware stock forecasting research API",
    description=(
        "Local and pooled global-model inference plus research analytics for Stock ML Lab. "
        "M13 adds same-family local-vs-global inference; M14 exposes the completed "
        "independent confirmation; M15 consolidates the publication release."
    ),
    version=APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/", tags=["system"])
def root() -> dict:
    return {
        "service": "stock-ml-lab-api",
        "version": APP_VERSION,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health() -> HealthResponse:
    return HealthResponse(version=APP_VERSION)


@app.get("/api/v1/capabilities", response_model=CapabilitiesResponse, tags=["metadata"])
def capabilities() -> CapabilitiesResponse:
    local_models = tuple(
        ModelCapability(
            key=spec.key,
            label=spec.label,
            task=spec.task,
            scope="local",
            available=True,
        )
        for spec in all_specs()
    )
    global_models = (
        ModelCapability(
            key="ridge",
            label="Global Ridge",
            task="regression",
            scope="global",
            available=global_store.is_complete(),
        ),
        ModelCapability(
            key="logistic",
            label="Global Logistic Regression",
            task="classification",
            scope="global",
            available=global_store.is_complete(),
        ),
    )
    models = local_models + global_models
    scopes = (
        ScopeCapability(
            name="local",
            available=True,
            description="Asset-specific model fit using only the requested ticker history.",
        ),
        ScopeCapability(
            name="global",
            available=global_store.is_complete(),
            description=(
                "Pretrained pooled multi-asset Ridge/Logistic artifacts."
                if global_store.is_complete()
                else "Run global_train.py to create the M12 pooled multi-asset artifacts."
            ),
        ),
        ScopeCapability(
            name="compare",
            available=global_store.is_complete(),
            description=(
                "Same-family local-vs-global side-by-side inference."
                if global_store.is_complete()
                else "Run global_train.py to enable local-vs-global comparison."
            ),
        ),
    )
    return CapabilitiesResponse(
        version=APP_VERSION,
        tasks=("regression", "classification"),
        horizons=(1, 5, 10, 20),
        feature_sets=("legacy",),
        scopes=scopes,
        models=models,
    )


@app.post("/api/v1/predict", response_model=PredictionResponse | ComparePredictionResponse, tags=["inference"])
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        if request.scope == "compare":
            if not global_store.is_complete():
                raise HTTPException(
                    status_code=503,
                    detail={
                        "code": "global_artifact_unavailable",
                        "scope": "compare",
                        "message": "Run global_train.py before using comparison scope.",
                    },
                )
            payload = predict_compare(request, store=global_store)
            return ComparePredictionResponse.model_validate(payload)
        if request.scope == "local":
            payload = predict_local(request)
        else:
            if not global_store.exists(request.task, request.model, request.horizon):
                raise HTTPException(
                    status_code=503,
                    detail={
                        "code": "global_artifact_unavailable",
                        "scope": "global",
                        "message": (
                            "The requested pretrained global artifact is unavailable. "
                            "Run global_train.py and restart the API."
                        ),
                    },
                )
            payload = predict_global(request, store=global_store)
        return PredictionResponse.model_validate(payload)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail={"code": "global_artifact_unavailable", "message": str(exc)},
        ) from exc
    except MarketDataError as exc:
        raise HTTPException(
            status_code=404,
            detail={"code": "market_data_unavailable", "message": str(exc)},
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "inference_input_error", "message": str(exc)},
        ) from exc


@app.get(
    "/api/v1/global/status",
    response_model=GlobalArtifactStatusResponse,
    tags=["metadata"],
)
def global_status() -> GlobalArtifactStatusResponse:
    return GlobalArtifactStatusResponse(
        root=str(global_store.root),
        complete=global_store.is_complete(),
        records=global_store.available_records(),
    )


@app.get(
    "/api/v1/market/{ticker}/snapshot",
    response_model=MarketSnapshotResponse,
    tags=["market-data"],
)
def snapshot(
    ticker: str,
    start: str = Query("2025-01-01"),
    end: str | None = Query(None),
) -> MarketSnapshotResponse:
    try:
        payload = market_snapshot(ticker=ticker, start=start, end=end)
        return MarketSnapshotResponse.model_validate(payload)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail={"code": "global_artifact_unavailable", "message": str(exc)},
        ) from exc
    except MarketDataError as exc:
        raise HTTPException(
            status_code=404,
            detail={"code": "market_data_unavailable", "message": str(exc)},
        ) from exc


@app.get("/api/v1/research/summary", response_model=ResearchSummaryResponse, tags=["research"])
def research_summary() -> ResearchSummaryResponse:
    return ResearchSummaryResponse(
        source=analytics.source_label,
        findings=analytics.findings(),
        discovery_candidates=analytics.discovery(),
        locked_confirmation=analytics.confirmation_summary(),
        scope_benchmark=analytics.scope_benchmark(),
        scope_confirmation=analytics.scope_confirmation(),
    )


@app.get("/api/v1/research/discovery", response_model=RecordsResponse, tags=["research"])
def research_discovery(
    task: str | None = Query(None, pattern="^(classification|regression)$"),
    horizon: int | None = Query(None, ge=1, le=20),
) -> RecordsResponse:
    records = analytics.discovery(task=task, horizon=horizon)
    return RecordsResponse(source=analytics.source_label, count=len(records), records=records)


@app.get("/api/v1/research/confirmation", response_model=RecordsResponse, tags=["research"])
def research_confirmation(
    task: str | None = Query(None, pattern="^(classification|regression)$"),
    horizon: int | None = Query(None, ge=1, le=20),
) -> RecordsResponse:
    records = analytics.confirmation_assets(task=task, horizon=horizon)
    return RecordsResponse(source=analytics.source_label, count=len(records), records=records)


@app.get("/api/v1/research/scope-benchmark", response_model=RecordsResponse, tags=["research"])
def research_scope_benchmark(
    task: str | None = Query(None, pattern="^(classification|regression)$"),
    horizon: int | None = Query(None, ge=1, le=20),
    level: str = Query("summary", pattern="^(summary|asset)$"),
) -> RecordsResponse:
    records = (
        analytics.scope_benchmark(task=task, horizon=horizon)
        if level == "summary"
        else analytics.scope_assets(task=task, horizon=horizon)
    )
    return RecordsResponse(source=analytics.source_label, count=len(records), records=records)


@app.get("/api/v1/research/scope-confirmation", response_model=RecordsResponse, tags=["research"])
def research_scope_confirmation(
    task: str | None = Query(None, pattern="^(classification|regression)$"),
    horizon: int | None = Query(None, ge=1, le=20),
    level: str = Query("summary", pattern="^(summary|asset)$"),
) -> RecordsResponse:
    records = analytics.scope_confirmation(
        task=task,
        horizon=horizon,
        level=level,
    )
    return RecordsResponse(source=analytics.source_label, count=len(records), records=records)

# The React/Vite production build is optional during development. When
# frontend/dist exists, FastAPI serves it under /app without changing API routes.
if frontend_is_built():
    app.mount(
        "/app",
        StaticFiles(directory=str(frontend_dist_dir()), html=True),
        name="frontend",
    )
