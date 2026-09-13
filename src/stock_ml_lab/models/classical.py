from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd
import pmdarima as pm
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import VAR


@dataclass(frozen=True)
class AutoARIMAConfig:
    start_p: int = 0
    start_q: int = 0
    max_p: int = 3
    max_q: int = 3
    max_d: int = 1
    information_criterion: str = "aic"


def fit_arima(
    series: pd.Series | np.ndarray,
    *,
    order: tuple[int, int, int] = (1, 0, 1),
):
    """Fit a fixed-order ARIMA model to a univariate return series."""
    values = np.asarray(series, dtype=float).reshape(-1)
    if values.size < 30:
        raise ValueError("ARIMA requires at least 30 training observations.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = ARIMA(
            values,
            order=order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        return model.fit()


def select_auto_arima_order(
    series: pd.Series | np.ndarray,
    *,
    config: AutoARIMAConfig = AutoARIMAConfig(),
) -> tuple[int, int, int]:
    """Select a non-seasonal ARIMA order on training data only."""
    values = np.asarray(series, dtype=float).reshape(-1)
    if values.size < 30:
        raise ValueError("AutoARIMA requires at least 30 training observations.")

    model = pm.auto_arima(
        values,
        seasonal=False,
        start_p=config.start_p,
        start_q=config.start_q,
        max_p=config.max_p,
        max_q=config.max_q,
        max_d=config.max_d,
        stepwise=True,
        information_criterion=config.information_criterion,
        suppress_warnings=True,
        error_action="ignore",
        trace=False,
    )
    return tuple(int(value) for value in model.order)


def build_var_frame(X: pd.DataFrame) -> pd.DataFrame:
    """Build a compact stationary-ish endogenous system for VAR.

    The return series is paired with log volume growth.  If
    ``volume_change_1d = V_t / V_{t-1} - 1``, then log volume growth is
    ``log(V_t / V_{t-1}) = log1p(volume_change_1d)``.
    """
    required = {"return_1d", "volume_change_1d"}
    missing = required.difference(X.columns)
    if missing:
        raise ValueError(f"VAR requires features: {sorted(required)}; missing: {sorted(missing)}")

    volume_change = X["volume_change_1d"].astype(float)
    # Exact -1 can occur if volume falls to zero.  Keep the transform finite.
    safe_volume_change = volume_change.clip(lower=-0.999999999)

    frame = pd.DataFrame(
        {
            "return_1d": X["return_1d"].astype(float),
            "log_volume_change_1d": np.log1p(safe_volume_change),
        },
        index=X.index,
    )

    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("VAR endogenous variables must be finite.")
    return frame


def fit_var(
    endog: pd.DataFrame | np.ndarray,
    *,
    lags: int = 5,
):
    """Fit a fixed-lag VAR model; lag order is not tuned in Milestone 3.1."""
    if lags < 1:
        raise ValueError("VAR lags must be >= 1.")

    values = np.asarray(endog, dtype=float)
    if values.ndim != 2:
        raise ValueError("VAR endog must be a two-dimensional array.")
    if len(values) <= max(30, 5 * lags):
        raise ValueError("VAR training history is too short for the requested lag order.")

    return VAR(values).fit(maxlags=lags, ic=None, trend="c")
