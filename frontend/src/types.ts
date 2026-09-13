export type TaskName = "regression" | "classification";
export type ScopeName = "local" | "global" | "compare";
export type Direction = "up" | "down" | "flat";

export interface ScopeCapability {
  name: ScopeName;
  available: boolean;
  description: string;
}

export interface ModelCapability {
  key: string;
  label: string;
  task: TaskName;
  scope: "local" | "global";
  available: boolean;
}

export interface CapabilitiesResponse {
  version: string;
  tasks: TaskName[];
  horizons: number[];
  feature_sets: string[];
  scopes: ScopeCapability[];
  models: ModelCapability[];
  docs_url: string;
}

export interface MarketSnapshot {
  ticker: string;
  as_of: string;
  close: number;
  return_1d: number | null;
  volume: number;
}

export interface PredictionRequest {
  ticker: string;
  task: TaskName;
  model: string;
  scope: ScopeName;
  horizon: number;
  start: string;
  end?: string | null;
  feature_set: "legacy";
}

export interface PredictionValue {
  predicted_return: number | null;
  predicted_price: number | null;
  probability_up: number | null;
  predicted_direction: Direction;
}

export interface TrainingContext {
  training_mode: "on_demand_local_fit" | "pretrained_global_artifact";
  training_rows: number;
  training_start: string;
  training_end: string;
  feature_count: number;
  feature_set: string;
  hyperparameters: Record<string, unknown>;
  note: string;
  artifact_id?: string | null;
  universe?: string[] | null;
  universe_size?: number | null;
}

export interface PredictionResponse {
  ticker: string;
  scope: "local" | "global";
  task: TaskName;
  model: string;
  horizon: number;
  as_of: string;
  latest_close: number;
  prediction: PredictionValue;
  training: TrainingContext;
  disclaimer: string;
}

export interface ResearchSummary {
  source: string;
  findings: string[];
  discovery_candidates: ResearchRecord[];
  locked_confirmation: ResearchRecord[];
  scope_benchmark: ResearchRecord[];
  scope_confirmation: ResearchRecord[];
}

export type ResearchRecord = Record<string, unknown>;

export interface ApiErrorShape {
  detail?: {
    code?: string;
    message?: string;
    scope?: string;
  } | string;
}

export interface ComparePredictionResponse {
  ticker: string;
  task: TaskName;
  horizon: number;
  as_of: string;
  local: PredictionResponse;
  global: PredictionResponse;
  comparison: {
    metric: string;
    local_value: number;
    global_value: number;
    global_minus_local: number;
  };
  disclaimer: string;
}

export type ForecastResponse = PredictionResponse | ComparePredictionResponse;
