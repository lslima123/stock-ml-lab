import type {
  ApiErrorShape,
  CapabilitiesResponse,
  ForecastResponse,
  MarketSnapshot,
  PredictionRequest,
  PredictionResponse,
  ResearchSummary
} from "../types";

const configuredBase = (import.meta.env.VITE_API_BASE_URL as string | undefined)?.trim();
const API_BASE = configuredBase ? configuredBase.replace(/\/$/, "") : "";

export class ApiError extends Error {
  status: number;
  code?: string;

  constructor(message: string, status: number, code?: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {})
    }
  });

  if (!response.ok) {
    let payload: ApiErrorShape | undefined;
    try {
      payload = (await response.json()) as ApiErrorShape;
    } catch {
      payload = undefined;
    }
    const detail = payload?.detail;
    const message =
      typeof detail === "string"
        ? detail
        : detail?.message ?? `Request failed with HTTP ${response.status}.`;
    const code = typeof detail === "object" ? detail?.code : undefined;
    throw new ApiError(message, response.status, code);
  }

  return (await response.json()) as T;
}

export function getHealth(): Promise<{ status: string; version: string }> {
  return requestJson("/health");
}

export function getCapabilities(): Promise<CapabilitiesResponse> {
  return requestJson("/api/v1/capabilities");
}

export function getMarketSnapshot(
  ticker: string,
  start = "2025-01-01"
): Promise<MarketSnapshot> {
  const params = new URLSearchParams({ start });
  return requestJson(`/api/v1/market/${encodeURIComponent(ticker)}/snapshot?${params}`);
}

export function getResearchSummary(): Promise<ResearchSummary> {
  return requestJson("/api/v1/research/summary");
}

export function predict(payload: PredictionRequest): Promise<ForecastResponse> {
  return requestJson("/api/v1/predict", {
    method: "POST",
    body: JSON.stringify(payload)
  });
}
