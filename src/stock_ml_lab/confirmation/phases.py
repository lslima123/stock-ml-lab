from __future__ import annotations

import numpy as np
import pandas as pd

from stock_ml_lab.classification.metrics import classification_metrics
from stock_ml_lab.evaluation.metrics import regression_metrics


def phase_labels(
    full_index: pd.Index,
    oos_index: pd.Index,
    *,
    horizon: int,
) -> pd.Series:
    """Assign each OOS timestamp to one of h non-overlapping target phases.

    Positions are defined on the full supervised dataset. Within a phase,
    consecutive observations are h rows apart, so [t, t+h] forward-return
    windows do not overlap (they can share only a boundary price).
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1.")
    if not oos_index.isin(full_index).all():
        raise ValueError("Every OOS timestamp must belong to full_index.")

    positions = pd.Series(np.arange(len(full_index)), index=full_index)
    labels = (positions.loc[oos_index].to_numpy(dtype=int) % horizon).astype(int)
    return pd.Series(labels, index=oos_index, name="phase")


def classification_phase_table(
    *,
    full_index: pd.Index,
    actual: pd.Series,
    model_probability: pd.Series,
    prior_probability: pd.Series,
    horizon: int,
) -> pd.DataFrame:
    if not (
        actual.index.equals(model_probability.index)
        and actual.index.equals(prior_probability.index)
    ):
        raise ValueError("Classification OOS series must share an identical index.")

    phases = phase_labels(full_index, actual.index, horizon=horizon)
    rows = []
    for phase in range(horizon):
        idx = phases.index[phases == phase]
        if len(idx) == 0:
            continue
        y = actual.loc[idx]
        model = model_probability.loc[idx]
        prior = prior_probability.loc[idx]
        m = classification_metrics(y, model)
        b = classification_metrics(y, prior)
        rows.append(
            {
                "phase": phase,
                "n": len(idx),
                "start": idx.min(),
                "end": idx.max(),
                "model_log_loss": m.log_loss,
                "prior_log_loss": b.log_loss,
                "log_loss_improvement_vs_prior_pct": 100.0
                * (b.log_loss - m.log_loss)
                / b.log_loss,
                "model_brier": m.brier,
                "prior_brier": b.brier,
                "brier_improvement_vs_prior_pct": 100.0
                * (b.brier - m.brier)
                / b.brier,
                "balanced_accuracy": m.balanced_accuracy,
                "roc_auc": m.roc_auc,
            }
        )
    return pd.DataFrame(rows)


def regression_phase_table(
    *,
    full_index: pd.Index,
    actual: pd.Series,
    model_forecast: pd.Series,
    zero_forecast: pd.Series,
    horizon: int,
) -> pd.DataFrame:
    if not (
        actual.index.equals(model_forecast.index)
        and actual.index.equals(zero_forecast.index)
    ):
        raise ValueError("Regression OOS series must share an identical index.")

    phases = phase_labels(full_index, actual.index, horizon=horizon)
    rows = []
    for phase in range(horizon):
        idx = phases.index[phases == phase]
        if len(idx) == 0:
            continue
        y = actual.loc[idx]
        model = model_forecast.loc[idx]
        zero = zero_forecast.loc[idx]
        m = regression_metrics(y, model)
        b = regression_metrics(y, zero)
        rows.append(
            {
                "phase": phase,
                "n": len(idx),
                "start": idx.min(),
                "end": idx.max(),
                "model_rmse": m.rmse,
                "zero_rmse": b.rmse,
                "rmse_improvement_vs_zero_pct": 100.0
                * (b.rmse - m.rmse)
                / b.rmse,
                "model_mae": m.mae,
                "zero_mae": b.mae,
                "mae_improvement_vs_zero_pct": 100.0
                * (b.mae - m.mae)
                / b.mae,
                "directional_accuracy": m.directional_accuracy,
            }
        )
    return pd.DataFrame(rows)


def summarize_phase_stability(
    phase_table: pd.DataFrame,
    *,
    primary_column: str,
) -> dict[str, float | int]:
    values = pd.to_numeric(phase_table[primary_column], errors="coerce")
    values = values[np.isfinite(values.to_numpy(dtype=float))]
    if values.empty:
        return {
            "phase_count": 0,
            "phase_primary_median": float("nan"),
            "phase_primary_q25": float("nan"),
            "phase_primary_q75": float("nan"),
            "phase_primary_min": float("nan"),
            "phase_primary_max": float("nan"),
            "phase_win_count": 0,
            "phase_win_rate": float("nan"),
        }

    return {
        "phase_count": int(len(values)),
        "phase_primary_median": float(values.median()),
        "phase_primary_q25": float(values.quantile(0.25)),
        "phase_primary_q75": float(values.quantile(0.75)),
        "phase_primary_min": float(values.min()),
        "phase_primary_max": float(values.max()),
        "phase_win_count": int((values > 0.0).sum()),
        "phase_win_rate": float((values > 0.0).mean()),
    }
