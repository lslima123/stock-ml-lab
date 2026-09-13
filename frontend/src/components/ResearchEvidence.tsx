import type { ResearchRecord, ResearchSummary, TaskName } from "../types";
import { formatPercentValue, safeNumber, safeString } from "../lib/format";

interface Props {
  research: ResearchSummary | null;
  task: TaskName;
  horizon: number;
}

function match(records: ResearchRecord[], task: TaskName, horizon: number) {
  return records.filter(
    (record) => safeString(record, "task") === task && safeNumber(record, "horizon") === horizon
  );
}

function discoveryRecord(records: ResearchRecord[], task: TaskName, horizon: number) {
  const candidates = match(records, task, horizon).filter(
    (record) => safeString(record, "feature_set") === "legacy"
  );
  const model = task === "classification" ? "logistic" : "ridge";
  return candidates.find((record) => safeString(record, "model_key") === model) ?? candidates[0];
}

function safeBoolean(record: ResearchRecord | undefined, key: string): boolean | null {
  const value = record?.[key];
  if (typeof value === "boolean") return value;
  if (typeof value === "string") {
    if (value.toLowerCase() === "true") return true;
    if (value.toLowerCase() === "false") return false;
  }
  return null;
}

export function ResearchEvidence({ research, task, horizon }: Props) {
  if (!research) {
    return (
      <section className="panel evidence-panel">
        <span className="eyebrow">Research context</span>
        <h2>Evidence chain</h2>
        <p className="muted">Research analytics are not available.</p>
      </section>
    );
  }

  const discovery = discoveryRecord(research.discovery_candidates, task, horizon);
  const confirmation = match(research.locked_confirmation, task, horizon)[0];
  const scope = match(research.scope_benchmark, task, horizon)[0];
  const scopeConfirmation = match(research.scope_confirmation, task, horizon)[0];
  const scopeConfirmed = safeBoolean(scopeConfirmation, "confirmed");
  const discoveryImprovement = task === "classification"
    ? safeNumber(discovery, "log_loss_improvement_vs_prior_pct")
    : safeNumber(discovery, "rmse_improvement_vs_zero_pct");

  return (
    <section className="panel evidence-panel">
      <div className="section-heading">
        <div>
          <span className="eyebrow">Research context</span>
          <h2>Evidence chain</h2>
        </div>
        <span className="evidence-badge">Not the live fit</span>
      </div>
      <p className="panel-intro">
        Historical OOS evidence is separated into discovery, locked confirmation and
        same-family scope comparison. It contextualizes this forecast; it is not a return claim.
      </p>

      <div className="evidence-grid">
        <article className="evidence-card discovery">
          <span className="evidence-label">PETR4 discovery · exploratory</span>
          <strong>{formatPercentValue(discoveryImprovement)}</strong>
          <span>{task === "classification" ? "Log Loss improvement" : "RMSE improvement"}</span>
          <small>Single-asset hypothesis generation</small>
        </article>

        <article className="evidence-card confirmation">
          <span className="evidence-label">M8.1 · locked confirmation</span>
          <strong>{formatPercentValue(safeNumber(confirmation, "median_primary_improvement_pct"))}</strong>
          <span>Median improvement across confirmation assets</span>
          <small>
            {safeNumber(confirmation, "primary_win_count") ?? "N/A"}/
            {safeNumber(confirmation, "n_assets") ?? "N/A"} asset wins · HAC/FDR 0
          </small>
        </article>

        <article className="evidence-card scope">
          <span className="evidence-label">M13 · global vs local</span>
          <strong>{formatPercentValue(safeNumber(scope, "global_improvement_vs_local_pct"))}</strong>
          <span>Same-family pooled OOS improvement</span>
          <small>
            {safeNumber(scope, "asset_win_count") ?? "N/A"}/{safeNumber(scope, "assets") ?? "N/A"} assets · {safeNumber(scope, "fold_win_count") ?? "N/A"}/{safeNumber(scope, "fold_count") ?? "N/A"} folds · {safeNumber(scope, "phase_win_count") ?? "N/A"}/{safeNumber(scope, "phase_count") ?? "N/A"} phases
          </small>
        </article>

        <article className={`evidence-card scope-confirmation ${scopeConfirmed === true ? "confirmed" : "not-confirmed"}`}>
          <span className="evidence-label">M14 · independent confirmation</span>
          <strong>
            {scopeConfirmation ? (scopeConfirmed ? "Confirmed" : "Not confirmed") : "Not tested"}
          </strong>
          <span>
            {scopeConfirmation
              ? `${formatPercentValue(safeNumber(scopeConfirmation, "global_improvement_vs_local_pct"))} ${task === "classification" ? "Log Loss" : "RMSE"} improvement`
              : "Locked protocol evaluated h=5 only"}
          </span>
          {scopeConfirmation ? (
            <small>
              q {safeNumber(scopeConfirmation, "primary_hac_qvalue")?.toFixed(4) ?? "N/A"} · 95% CI [{formatPercentValue(safeNumber(scopeConfirmation, "bootstrap_improvement_ci_low_pct"), 2, true)}, {formatPercentValue(safeNumber(scopeConfirmation, "bootstrap_improvement_ci_high_pct"), 2, true)}] · {safeNumber(scopeConfirmation, "asset_win_count") ?? "N/A"}/{safeNumber(scopeConfirmation, "assets") ?? "N/A"} assets
            </small>
          ) : null}
        </article>
      </div>

      <p className="posthoc-note">
        M13 date-clustered HAC/bootstrap, fold, phase and regional checks are post hoc.
        M14 is the separate locked test: Ridge scope was confirmed at h=5; Logistic scope was not.
        This is forecasting-loss evidence, not a trading-performance claim.
      </p>

      <details className="findings">
        <summary>Project findings</summary>
        <ul>{research.findings.map((finding) => <li key={finding}>{finding}</li>)}</ul>
      </details>
    </section>
  );
}
