from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    roc_auc_score,
)

from stock_ml_lab.global_model.dataset import GlobalPanelBundle
from stock_ml_lab.global_model.models import build_global_estimator


@dataclass(frozen=True)
class PanelFold:
    fold: int
    train_index: np.ndarray
    test_index: np.ndarray
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def panel_walk_forward_splits(
    bundle: GlobalPanelBundle,
    *,
    n_splits: int = 5,
    test_size_dates: int = 252,
) -> list[PanelFold]:
    if n_splits < 1:
        raise ValueError("n_splits must be >= 1.")
    if test_size_dates < 1:
        raise ValueError("test_size_dates must be >= 1.")

    frame = bundle.frame
    dates = pd.DatetimeIndex(sorted(pd.unique(frame["date"])))
    required = n_splits * test_size_dates + 80
    if len(dates) < required:
        raise ValueError(
            f"Not enough unique panel dates for {n_splits} folds of "
            f"{test_size_dates} dates; have {len(dates)}, need at least {required}."
        )

    first_test_pos = len(dates) - n_splits * test_size_dates
    folds: list[PanelFold] = []
    for fold in range(n_splits):
        start_pos = first_test_pos + fold * test_size_dates
        end_pos = start_pos + test_size_dates
        test_dates = dates[start_pos:end_pos]
        test_start = pd.Timestamp(test_dates[0])
        test_end = pd.Timestamp(test_dates[-1])

        # Critical global purge rule: every training label must have matured
        # before the first feature date in the test fold.
        train_mask = pd.to_datetime(frame["target_end_date"]) < test_start
        test_mask = pd.to_datetime(frame["date"]).isin(test_dates)

        train_idx = np.flatnonzero(train_mask.to_numpy())
        test_idx = np.flatnonzero(test_mask.to_numpy())
        if len(train_idx) == 0 or len(test_idx) == 0:
            raise ValueError(f"Fold {fold + 1} is empty after panel purge.")

        folds.append(
            PanelFold(
                fold=fold + 1,
                train_index=train_idx,
                test_index=test_idx,
                test_start=test_start,
                test_end=test_end,
            )
        )
    return folds


def _classification_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-8, 1 - 1e-8)
    pred = (probability >= 0.5).astype(int)
    out = {
        "log_loss": float(log_loss(y, probability, labels=[0, 1])),
        "brier": float(brier_score_loss(y, probability)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "accuracy": float(np.mean(pred == y)),
    }
    out["roc_auc"] = (
        float(roc_auc_score(y, probability))
        if len(np.unique(y)) == 2
        else float("nan")
    )
    return out


def _regression_metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
        "mae": float(mean_absolute_error(y, prediction)),
        "directional_accuracy": float(np.mean((prediction > 0) == (y > 0))),
    }


def evaluate_global_temporal(
    bundle: GlobalPanelBundle,
    *,
    task: str,
    model: str,
    n_splits: int = 5,
    test_size_dates: int = 252,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    folds = panel_walk_forward_splits(
        bundle,
        n_splits=n_splits,
        test_size_dates=test_size_dates,
    )
    fold_rows: list[dict] = []
    asset_rows: list[dict] = []

    X = bundle.X
    y = bundle.regression_y if task == "regression" else bundle.classification_y

    for split in folds:
        estimator = build_global_estimator(task, model)
        X_train = X.iloc[split.train_index]
        y_train = y.iloc[split.train_index]
        X_test = X.iloc[split.test_index]
        y_test = y.iloc[split.test_index]
        test_meta = bundle.frame.iloc[split.test_index][["ticker", "date"]].reset_index(drop=True)

        estimator.fit(X_train, y_train)

        if task == "regression":
            pred = np.asarray(estimator.predict(X_test), dtype=float)
            metrics = _regression_metrics(y_test.to_numpy(), pred)
            baseline = np.zeros(len(y_test), dtype=float)
            baseline_metrics = _regression_metrics(y_test.to_numpy(), baseline)
            metrics["primary_improvement_pct"] = (
                100.0 * (baseline_metrics["rmse"] - metrics["rmse"]) / baseline_metrics["rmse"]
            )
            score = pred
        else:
            proba = np.asarray(estimator.predict_proba(X_test), dtype=float)
            classes = list(estimator.classes_)
            up_index = classes.index(1)
            score = proba[:, up_index]
            metrics = _classification_metrics(y_test.to_numpy(), score)
            prior = float(y_train.mean())
            baseline_probability = np.full(len(y_test), prior, dtype=float)
            baseline_metrics = _classification_metrics(
                y_test.to_numpy(), baseline_probability
            )
            metrics["primary_improvement_pct"] = (
                100.0
                * (baseline_metrics["log_loss"] - metrics["log_loss"])
                / baseline_metrics["log_loss"]
            )

        fold_rows.append(
            {
                "task": task,
                "model": model,
                "horizon": bundle.horizon,
                "fold": split.fold,
                "train_rows": len(split.train_index),
                "test_rows": len(split.test_index),
                "test_start": split.test_start.date(),
                "test_end": split.test_end.date(),
                **metrics,
            }
        )

        scored = test_meta.copy()
        scored["y"] = y_test.to_numpy()
        scored["score"] = score
        for ticker, group in scored.groupby("ticker", sort=True):
            if task == "regression":
                gm = _regression_metrics(
                    group["y"].to_numpy(dtype=float),
                    group["score"].to_numpy(dtype=float),
                )
            else:
                gm = _classification_metrics(
                    group["y"].to_numpy(dtype=int),
                    group["score"].to_numpy(dtype=float),
                )
            asset_rows.append(
                {
                    "task": task,
                    "model": model,
                    "horizon": bundle.horizon,
                    "fold": split.fold,
                    "ticker": ticker,
                    "n": len(group),
                    **gm,
                }
            )

    return pd.DataFrame(fold_rows), pd.DataFrame(asset_rows)


def evaluate_unseen_asset_holdout(
    bundle: GlobalPanelBundle,
    *,
    task: str,
    model: str,
    test_size_dates: int = 252,
) -> pd.DataFrame:
    """Train on other assets before the holdout period; test an unseen ticker.

    The held-out ticker contributes zero training observations. Other assets may
    train only on labels whose target_end_date is strictly before the held-out
    test window, preventing both cross-asset and temporal leakage.
    """
    rows: list[dict] = []
    frame = bundle.frame
    X = bundle.X
    y = bundle.regression_y if task == "regression" else bundle.classification_y

    for ticker in bundle.tickers:
        ticker_dates = pd.DatetimeIndex(
            sorted(pd.unique(frame.loc[frame["ticker"] == ticker, "date"]))
        )
        if len(ticker_dates) <= test_size_dates:
            continue
        test_dates = ticker_dates[-test_size_dates:]
        test_start = pd.Timestamp(test_dates[0])

        train_mask = (
            (frame["ticker"] != ticker)
            & (pd.to_datetime(frame["target_end_date"]) < test_start)
        )
        test_mask = (
            (frame["ticker"] == ticker)
            & (pd.to_datetime(frame["date"]).isin(test_dates))
        )
        train_idx = np.flatnonzero(train_mask.to_numpy())
        test_idx = np.flatnonzero(test_mask.to_numpy())
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        estimator = build_global_estimator(task, model)
        estimator.fit(X.iloc[train_idx], y.iloc[train_idx])

        y_test = y.iloc[test_idx].to_numpy()
        if task == "regression":
            score = np.asarray(estimator.predict(X.iloc[test_idx]), dtype=float)
            metrics = _regression_metrics(y_test, score)
            baseline = np.zeros(len(y_test), dtype=float)
            baseline_metrics = _regression_metrics(y_test, baseline)
            primary = 100.0 * (baseline_metrics["rmse"] - metrics["rmse"]) / baseline_metrics["rmse"]
        else:
            probabilities = np.asarray(estimator.predict_proba(X.iloc[test_idx]), dtype=float)
            up_index = list(estimator.classes_).index(1)
            score = probabilities[:, up_index]
            metrics = _classification_metrics(y_test.astype(int), score)
            prior = float(y.iloc[train_idx].mean())
            base = np.full(len(y_test), prior, dtype=float)
            baseline_metrics = _classification_metrics(y_test.astype(int), base)
            primary = (
                100.0
                * (baseline_metrics["log_loss"] - metrics["log_loss"])
                / baseline_metrics["log_loss"]
            )

        rows.append(
            {
                "ticker": ticker,
                "task": task,
                "model": model,
                "horizon": bundle.horizon,
                "train_assets": len(bundle.tickers) - 1,
                "train_rows": len(train_idx),
                "test_rows": len(test_idx),
                "test_start": pd.Timestamp(test_dates[0]).date(),
                "test_end": pd.Timestamp(test_dates[-1]).date(),
                "primary_improvement_pct": primary,
                **metrics,
            }
        )

    return pd.DataFrame(rows)
