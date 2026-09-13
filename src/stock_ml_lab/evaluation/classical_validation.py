from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from stock_ml_lab.evaluation.metrics import regression_metrics
from stock_ml_lab.evaluation.validation import CVResult, FoldResult
from stock_ml_lab.models.classical import (
    AutoARIMAConfig,
    build_var_frame,
    fit_arima,
    fit_var,
    select_auto_arima_order,
)


def _validate_inputs(X: pd.DataFrame, y: pd.Series, n_splits: int, gap: int) -> None:
    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame.")
    if not isinstance(y, pd.Series):
        raise TypeError("y must be a pandas Series.")
    if not X.index.equals(y.index):
        raise ValueError("X and y must have identical indices.")
    if not X.index.is_monotonic_increasing:
        raise ValueError("X and y must be ordered chronologically.")
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2.")
    if gap < 1:
        raise ValueError("gap must be >= 1 for the one-step-ahead next_return target.")


def _make_result(
    *,
    model_name: str,
    fold_results: list[FoldResult],
    prediction_parts: list[pd.Series],
    actual_parts: list[pd.Series],
    n_splits: int,
    gap: int,
    fold_model_details: pd.DataFrame | None,
) -> CVResult:
    predictions = pd.concat(prediction_parts).sort_index()
    actuals = pd.concat(actual_parts).sort_index()

    if predictions.index.has_duplicates:
        raise RuntimeError("Cross-validation produced duplicate prediction timestamps.")

    return CVResult(
        model_name=model_name,
        metrics=regression_metrics(actuals.to_numpy(), predictions.to_numpy()),
        folds=tuple(fold_results),
        predictions=predictions,
        actuals=actuals,
        n_splits=n_splits,
        gap=gap,
        fold_feature_importances=None,
        fold_model_details=fold_model_details,
    )


def evaluate_arima_cv(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    n_splits: int = 5,
    gap: int = 1,
    test_size: int | None = None,
    order: tuple[int, int, int] = (1, 0, 1),
    auto: bool = False,
    auto_config: AutoARIMAConfig = AutoARIMAConfig(),
) -> CVResult:
    """Evaluate ARIMA with fixed parameters and rolling one-step state updates.

    Parameters are estimated once at the beginning of each fold.  During the
    test fold, realized returns are appended with ``refit=False`` before each
    next-day forecast.  This mirrors the ML setting: model parameters stay
    fixed, but information observed by the current close is available.
    """
    _validate_inputs(X, y, n_splits, gap)
    if "return_1d" not in X.columns:
        raise ValueError("ARIMA rolling evaluation requires X['return_1d'].")

    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap, test_size=test_size)
    fold_results: list[FoldResult] = []
    prediction_parts: list[pd.Series] = []
    actual_parts: list[pd.Series] = []
    detail_rows: list[dict[str, object]] = []

    for fold_number, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        selected_order = (
            select_auto_arima_order(y_train, config=auto_config) if auto else order
        )
        fitted = fit_arima(y_train, order=selected_order)

        predictions: list[float] = []
        # y_train[-1] is the return observed one row after train_idx[-1].
        # Append any remaining purged/current observations without re-fitting.
        next_observation_pos = int(train_idx[-1]) + 2

        for test_pos in test_idx:
            test_pos = int(test_pos)
            if next_observation_pos <= test_pos:
                observed = X["return_1d"].iloc[next_observation_pos : test_pos + 1].to_numpy()
                if observed.size:
                    fitted = fitted.append(observed, refit=False)
            forecast = np.asarray(fitted.forecast(steps=1), dtype=float).reshape(-1)
            predictions.append(float(forecast[0]))
            next_observation_pos = test_pos + 1

        y_pred = np.asarray(predictions, dtype=float)
        fold_metrics = regression_metrics(y_test.to_numpy(), y_pred)

        fold_results.append(
            FoldResult(
                fold=fold_number,
                train_start=X_train.index[0],
                train_end=X_train.index[-1],
                test_start=X.index[test_idx[0]],
                test_end=X.index[test_idx[-1]],
                n_train=len(train_idx),
                n_test=len(test_idx),
                metrics=fold_metrics,
            )
        )
        prediction_parts.append(pd.Series(y_pred, index=X.index[test_idx], name="prediction"))
        actual_parts.append(y_test.rename("actual"))
        detail_rows.append(
            {
                "fold": fold_number,
                "order": str(tuple(selected_order)),
                "selection": "AIC/AutoARIMA" if auto else "fixed",
            }
        )

    details = pd.DataFrame(detail_rows).set_index("fold")
    return _make_result(
        model_name=model_name,
        fold_results=fold_results,
        prediction_parts=prediction_parts,
        actual_parts=actual_parts,
        n_splits=n_splits,
        gap=gap,
        fold_model_details=details,
    )


def evaluate_var_cv(
    *,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str = "VAR",
    n_splits: int = 5,
    gap: int = 1,
    test_size: int | None = None,
    lags: int = 5,
) -> CVResult:
    """Evaluate a fixed VAR(p) with rolling one-step forecasts.

    The endogenous vector is ``[return_1d, log_volume_change_1d]``. Parameters
    are estimated once per fold, while the state is updated with observations
    known at each close.  The return equation supplies the next-day prediction.
    """
    _validate_inputs(X, y, n_splits, gap)
    var_frame = build_var_frame(X)
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=gap, test_size=test_size)

    fold_results: list[FoldResult] = []
    prediction_parts: list[pd.Series] = []
    actual_parts: list[pd.Series] = []
    detail_rows: list[dict[str, object]] = []

    for fold_number, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        X_train = X.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # y_train at feature row t corresponds to return_1d at row t+1.
        shifted_train_positions = np.asarray(train_idx, dtype=int) + 1
        train_endog = var_frame.iloc[shifted_train_positions].copy()
        fitted = fit_var(train_endog, lags=lags)

        history = train_endog.to_numpy(dtype=float).tolist()
        next_observation_pos = int(train_idx[-1]) + 2
        predictions: list[float] = []

        for test_pos in test_idx:
            test_pos = int(test_pos)
            if next_observation_pos <= test_pos:
                observed = var_frame.iloc[next_observation_pos : test_pos + 1].to_numpy(dtype=float)
                history.extend(observed.tolist())

            state = np.asarray(history[-fitted.k_ar :], dtype=float)
            forecast = fitted.forecast(state, steps=1)
            predictions.append(float(forecast[0, 0]))
            next_observation_pos = test_pos + 1

        y_pred = np.asarray(predictions, dtype=float)
        fold_metrics = regression_metrics(y_test.to_numpy(), y_pred)
        fold_results.append(
            FoldResult(
                fold=fold_number,
                train_start=X_train.index[0],
                train_end=X_train.index[-1],
                test_start=X.index[test_idx[0]],
                test_end=X.index[test_idx[-1]],
                n_train=len(train_idx),
                n_test=len(test_idx),
                metrics=fold_metrics,
            )
        )
        prediction_parts.append(pd.Series(y_pred, index=X.index[test_idx], name="prediction"))
        actual_parts.append(y_test.rename("actual"))
        detail_rows.append(
            {
                "fold": fold_number,
                "order": f"VAR({fitted.k_ar})",
                "variables": "return_1d + log_volume_change_1d",
            }
        )

    details = pd.DataFrame(detail_rows).set_index("fold")
    return _make_result(
        model_name=model_name,
        fold_results=fold_results,
        prediction_parts=prediction_parts,
        actual_parts=actual_parts,
        n_splits=n_splits,
        gap=gap,
        fold_model_details=details,
    )
