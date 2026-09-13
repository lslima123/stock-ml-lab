import { FormEvent, useEffect, useMemo, useState } from "react";
import { getCapabilities, getHealth, getMarketSnapshot, getResearchSummary, predict, ApiError } from "./lib/api";
import {
  directionLabel,
  formatInteger,
  formatPctPoint,
  formatPercent,
  formatPrice
} from "./lib/format";
import { Metric } from "./components/Metric";
import { ResearchEvidence } from "./components/ResearchEvidence";
import { ScopeSelector } from "./components/ScopeSelector";
import type {
  CapabilitiesResponse,
  MarketSnapshot,
  ComparePredictionResponse,
  ForecastResponse,
  PredictionResponse,
  ResearchSummary,
  ScopeName,
  TaskName
} from "./types";

const DEFAULT_TICKER = "PETR4.SA";
const DEFAULT_START = "2018-01-01";

function App() {
  const [capabilities, setCapabilities] = useState<CapabilitiesResponse | null>(null);
  const [research, setResearch] = useState<ResearchSummary | null>(null);
  const [apiOnline, setApiOnline] = useState(false);

  const [ticker, setTicker] = useState(DEFAULT_TICKER);
  const [task, setTask] = useState<TaskName>("regression");
  const [model, setModel] = useState("ridge");
  const [horizon, setHorizon] = useState(20);
  const [scope, setScope] = useState<ScopeName>("local");
  const [start, setStart] = useState(DEFAULT_START);

  const [snapshot, setSnapshot] = useState<MarketSnapshot | null>(null);
  const [prediction, setPrediction] = useState<ForecastResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [snapshotLoading, setSnapshotLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.allSettled([getHealth(), getCapabilities(), getResearchSummary()]).then(
      ([healthResult, capabilityResult, researchResult]) => {
        setApiOnline(healthResult.status === "fulfilled");
        if (capabilityResult.status === "fulfilled") {
          setCapabilities(capabilityResult.value);
        }
        if (researchResult.status === "fulfilled") {
          setResearch(researchResult.value);
        }
      }
    );
  }, []);

  const modelScope = scope === "compare" ? "local" : scope;
  const models = useMemo(
    () =>
      capabilities?.models.filter(
        (item) =>
          item.task === task &&
          item.scope === modelScope &&
          item.available &&
          (scope !== "compare" || item.key === (task === "regression" ? "ridge" : "logistic"))
      ) ?? [],
    [capabilities, task, modelScope, scope]
  );

  useEffect(() => {
    if (models.length > 0 && !models.some((item) => item.key === model)) {
      const preferred = task === "regression" ? "ridge" : "logistic";
      setModel(models.find((item) => item.key === preferred)?.key ?? models[0].key);
    }
  }, [models, model, task]);

  async function refreshSnapshot(symbol = ticker) {
    if (!symbol.trim()) return;
    setSnapshotLoading(true);
    try {
      const next = await getMarketSnapshot(symbol.trim().toUpperCase());
      setSnapshot(next);
    } catch {
      setSnapshot(null);
    } finally {
      setSnapshotLoading(false);
    }
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    setPrediction(null);

    const normalizedTicker = ticker.trim().toUpperCase();
    setTicker(normalizedTicker);

    try {
      const [nextPrediction, nextSnapshot] = await Promise.all([
        predict({
          ticker: normalizedTicker,
          task,
          model,
          scope,
          horizon,
          start,
          feature_set: "legacy"
        }),
        getMarketSnapshot(normalizedTicker)
      ]);
      setPrediction(nextPrediction);
      setSnapshot(nextSnapshot);
    } catch (cause) {
      if (cause instanceof ApiError) {
        setError(`${cause.message}${cause.code ? ` (${cause.code})` : ""}`);
      } else {
        setError("The request could not be completed. Check that the API is running.");
      }
    } finally {
      setLoading(false);
    }
  }

  const scopeCapabilities =
    capabilities?.scopes ?? [
      {
        name: "local" as const,
        available: true,
        description: "Ticker-specific local fit."
      },
      {
        name: "global" as const,
        available: false,
        description: "Reserved for pooled multi-asset model."
      },
      {
        name: "compare" as const,
        available: false,
        description: "Reserved for local-vs-global comparison."
      }
    ];

  const selectedModelLabel =
    models.find((item) => item.key === model)?.label ?? model;

  const isCompareResponse = (value: ForecastResponse): value is ComparePredictionResponse =>
    "local" in value && "global" in value;

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">S</div>
          <div>
            <strong>Stock ML Lab</strong>
            <span>Forecasting research workspace</span>
          </div>
        </div>
        <div className="topbar-actions">
          <span className={`status-pill ${apiOnline ? "online" : "offline"}`}>
            <span className="status-dot" />
            API {apiOnline ? "online" : "offline"}
          </span>
          <a className="docs-link" href="/docs" target="_blank" rel="noreferrer">
            API docs ↗
          </a>
        </div>
      </header>

      <main className="main-layout">
        <section className="hero">
          <div className="hero-copy">
            <span className="eyebrow">Leakage-aware forecasting</span>
            <h1>Test a market forecast without hiding the evidence.</h1>
            <p>
              Run a ticker-specific model, inspect its current forecast, and compare the
              result with the discovery and locked-confirmation evidence from the research pipeline.
            </p>
          </div>
          <div className="hero-stat">
            <span>Current architecture</span>
            <strong>Local ready.</strong>
            <strong>{scopeCapabilities.find((item) => item.name === "global")?.available ? "Global ready." : "Global artifact pending."}</strong>
            <small>M14 confirmed the h=5 pooled Ridge scope effect; Logistic was not confirmed.</small>
          </div>
        </section>

        <div className="workspace-grid">
          <form className="panel controls-panel" onSubmit={handleSubmit}>
            <div className="section-heading">
              <div>
                <span className="eyebrow">Forecast setup</span>
                <h2>Configure the experiment</h2>
              </div>
              <span className="version-badge">API {capabilities?.version ?? "…"}</span>
            </div>

            <label className="field">
              <span>Ticker</span>
              <div className="ticker-input-row">
                <input
                  aria-label="Ticker"
                  value={ticker}
                  onChange={(event) => setTicker(event.target.value.toUpperCase())}
                  onBlur={() => refreshSnapshot()}
                  placeholder="PETR4.SA"
                  maxLength={32}
                />
                <button
                  className="ghost-button"
                  type="button"
                  onClick={() => refreshSnapshot()}
                  disabled={snapshotLoading}
                >
                  {snapshotLoading ? "Loading…" : "Snapshot"}
                </button>
              </div>
            </label>

            {snapshot ? (
              <div className="snapshot-strip">
                <div>
                  <span>{snapshot.ticker}</span>
                  <strong>{formatPrice(snapshot.ticker, snapshot.close)}</strong>
                </div>
                <div className={snapshot.return_1d != null && snapshot.return_1d >= 0 ? "positive" : "negative"}>
                  <span>1d return</span>
                  <strong>{formatPercent(snapshot.return_1d, 2, true)}</strong>
                </div>
                <div>
                  <span>Volume</span>
                  <strong>{formatInteger(snapshot.volume)}</strong>
                </div>
                <small>as of {snapshot.as_of}</small>
              </div>
            ) : null}

            <div className="field">
              <span>Task</span>
              <div className="segmented">
                <button
                  type="button"
                  className={task === "regression" ? "active" : ""}
                  onClick={() => setTask("regression")}
                >
                  Regression
                </button>
                <button
                  type="button"
                  className={task === "classification" ? "active" : ""}
                  onClick={() => setTask("classification")}
                >
                  Direction
                </button>
              </div>
            </div>

            <label className="field">
              <span>Model</span>
              <select value={model} onChange={(event) => setModel(event.target.value)}>
                {models.map((item) => (
                  <option key={item.key} value={item.key}>
                    {item.label}
                  </option>
                ))}
              </select>
            </label>

            <div className="field">
              <span>Forecast horizon</span>
              <div className="horizon-grid">
                {(capabilities?.horizons ?? [1, 5, 10, 20]).map((value) => (
                  <button
                    type="button"
                    key={value}
                    className={horizon === value ? "active" : ""}
                    onClick={() => setHorizon(value)}
                  >
                    {value}d
                  </button>
                ))}
              </div>
            </div>

            <div className="field">
              <span>Model scope</span>
              <ScopeSelector scopes={scopeCapabilities} value={scope} onChange={setScope} />
            </div>

            <details className="advanced">
              <summary>Advanced settings</summary>
              <label className="field compact">
                <span>Training history starts</span>
                <input
                  type="date"
                  value={start}
                  onChange={(event) => setStart(event.target.value)}
                />
              </label>
              <p>
                Production inference uses a fixed documented preset, not nested Optuna tuning
                inside the HTTP request.
              </p>
            </details>

            {error ? <div className="error-banner">{error}</div> : null}

            <button className="primary-button" type="submit" disabled={loading || !apiOnline || (scope === "compare" && model !== (task === "regression" ? "ridge" : "logistic"))}>
              {loading
                ? scope === "compare"
                  ? "Running local + global…"
                  : scope === "global"
                    ? "Loading global artifact…"
                    : "Fitting local model…"
                : scope === "compare"
                  ? "Compare local + global"
                  : `Run ${selectedModelLabel}`}
            </button>

            <p className="form-footnote">
              Research/educational output only. Forecasts are uncertain and are not investment advice.
            </p>
          </form>

          <section className="panel result-panel">
            <div className="section-heading">
              <div>
                <span className="eyebrow">Current inference</span>
                <h2>{prediction ? `${prediction.ticker} forecast` : "Awaiting forecast"}</h2>
              </div>
              {prediction ? (
                isCompareResponse(prediction) ? (
                  <span className="direction-chip">Local vs global</span>
                ) : (
                  <span className={`direction-chip ${prediction.prediction.predicted_direction}`}>
                    {directionLabel(prediction.prediction.predicted_direction)}
                  </span>
                )
              ) : null}
            </div>

            {!prediction && !loading ? (
              <div className="empty-state">
                <div className="empty-orbit">
                  <span />
                </div>
                <h3>Configure a model and run the forecast.</h3>
                <p>
                  The API will build the latest feature row and run the selected local, global, or comparison scope.
                </p>
              </div>
            ) : null}

            {loading ? (
              <div className="loading-state">
                <div className="loader" />
                <h3>Building the inference dataset</h3>
                <p>Downloading market data and running the selected inference scope.</p>
              </div>
            ) : null}

            {prediction ? (
              isCompareResponse(prediction) ? (
                <div className="compare-result">
                  <div className="compare-hero-grid">
                    {[prediction.local, prediction.global].map((item) => (
                      <article className="compare-card" key={`${item.scope}-${item.model}`}>
                        <span className="eyebrow">{item.scope === "local" ? "Local model" : "Global model"}</span>
                        <h3>{item.training.training_mode === "pretrained_global_artifact" ? "Global pooled model" : "Ticker-specific model"}</h3>
                        {item.task === "regression" ? (
                          <strong>{formatPercent(item.prediction.predicted_return, 2, true)}</strong>
                        ) : (
                          <strong>{formatPercent(item.prediction.probability_up, 1)}</strong>
                        )}
                        <span>
                          {item.task === "regression"
                            ? `Predicted ${item.horizon}d return`
                            : `Probability of positive ${item.horizon}d return`}
                        </span>
                        <small>{item.model} · as of {item.as_of}</small>
                      </article>
                    ))}
                  </div>
                  <div className="comparison-delta">
                    <span>Global − Local</span>
                    <strong>{formatPctPoint(prediction.comparison.global_minus_local)}</strong>
                    <small>
                      {prediction.comparison.metric === "probability_up"
                        ? "probability of positive return"
                        : "predicted return"}
                    </small>
                  </div>
                  <div className="metric-grid">
                    <Metric label="Latest close" value={formatPrice(prediction.ticker, prediction.local.latest_close)} />
                    <Metric label="Horizon" value={`${prediction.horizon} trading days`} />
                    <Metric label="Local rows" value={formatInteger(prediction.local.training.training_rows)} />
                    <Metric label="Global assets" value={formatInteger(prediction.global.training.universe_size ?? 0)} />
                  </div>
                  <div className="training-card">
                    <div className="training-heading">
                      <div>
                        <span className="eyebrow">Comparison contract</span>
                        <h3>Same model family, different scope</h3>
                      </div>
                      <span>{prediction.local.model} vs {prediction.global.model}</span>
                    </div>
                    <p>
                      The local model is fitted only on {prediction.ticker}; the global model is a
                      pretrained pooled artifact. This isolates the effect of training scope rather
                      than mixing model-family differences.
                    </p>
                  </div>
                </div>
              ) : (
                <>
                  <div className="result-hero">
                    {prediction.task === "regression" ? (
                      <>
                        <span>Predicted {prediction.horizon}d return</span>
                        <strong>{formatPercent(prediction.prediction.predicted_return, 2, true)}</strong>
                        <small>implied price {formatPrice(prediction.ticker, prediction.prediction.predicted_price)}</small>
                      </>
                    ) : (
                      <>
                        <span>Probability of positive return</span>
                        <strong>{formatPercent(prediction.prediction.probability_up, 1)}</strong>
                        <small>{prediction.horizon}-trading-day direction</small>
                      </>
                    )}
                  </div>
                  <div className="metric-grid">
                    <Metric label="Latest close" value={formatPrice(prediction.ticker, prediction.latest_close)} />
                    <Metric label="As of" value={prediction.as_of} />
                    <Metric label="Scope" value={prediction.scope} />
                    <Metric label="Model" value={selectedModelLabel} />
                  </div>
                  <div className="training-card">
                    <div className="training-heading">
                      <div>
                        <span className="eyebrow">Training context</span>
                        <h3>{prediction.training.training_mode === "pretrained_global_artifact" ? "Pretrained global artifact" : "On-demand local fit"}</h3>
                      </div>
                      <span>{prediction.training.training_rows} rows</span>
                    </div>
                    <div className="training-grid">
                      <span><small>Window</small>{prediction.training.training_start} → {prediction.training.training_end}</span>
                      <span><small>Features</small>{prediction.training.feature_count} · {prediction.training.feature_set}</span>
                      <span><small>Horizon</small>{prediction.horizon} trading days</span>
                    </div>
                    {prediction.training.universe_size ? (
                      <div className="global-artifact-note">
                        <small>Global training universe</small>
                        <strong>{prediction.training.universe_size} assets</strong>
                        <span>{prediction.training.universe?.join(", ")}</span>
                        {prediction.training.artifact_id ? <code>{prediction.training.artifact_id}</code> : null}
                      </div>
                    ) : null}
                    <details>
                      <summary>Hyperparameters & methodology note</summary>
                      <pre>{JSON.stringify(prediction.training.hyperparameters, null, 2)}</pre>
                      <p>{prediction.training.note}</p>
                    </details>
                  </div>
                </>
              )
            ) : null}
          </section>
        </div>

        <ResearchEvidence research={research} task={task} horizon={horizon} />

        <section className="panel roadmap-panel">
          <div>
            <span className="eyebrow">Architecture roadmap</span>
            <h2>Local and global models coexist by design.</h2>
            <p>
              M13 keeps the model family fixed while changing only the training scope.
              M14 independently confirmed the pooled Ridge scope effect at h=5, while
              Logistic was not confirmed. M15 consolidates this validated state for publication;
              the interface still makes no automatic recommendation.
            </p>
          </div>
          <div className="roadmap-steps">
            <article className="done">
              <span>M11</span>
              <strong>Local</strong>
              <small>One ticker, ticker-specific fit</small>
            </article>
            <article className={scopeCapabilities.find((item) => item.name === "global")?.available ? "done" : ""}>
              <span>M12</span>
              <strong>Global</strong>
              <small>Pooled multi-asset training</small>
            </article>
            <article className={scopeCapabilities.find((item) => item.name === "compare")?.available ? "done" : ""}>
              <span>M13</span>
              <strong>Compare</strong>
              <small>Local vs global, side by side</small>
            </article>
            <article className="done">
              <span>M14</span>
              <strong>Confirm</strong>
              <small>Independent h=5 result incorporated</small>
            </article>
            <article className="done">
              <span>M15</span>
              <strong>Release</strong>
              <small>Tested, documented, publication-ready</small>
            </article>
          </div>
        </section>
      </main>

      <footer>
        <span>Stock ML Lab · Milestone 15</span>
        <span>Research software — not investment advice.</span>
      </footer>
    </div>
  );
}

export default App;
