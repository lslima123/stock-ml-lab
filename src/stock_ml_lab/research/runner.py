from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd

from stock_ml_lab.classification.nested import run_classification_comparison
from stock_ml_lab.research.dataset import FEATURE_SETS, ResearchDatasetBundle, build_research_dataset
from stock_ml_lab.tuning.nested import run_tuning_comparison


CLASSIFICATION_TASK = "classification"
REGRESSION_TASK = "regression"
RESEARCH_TASKS = (CLASSIFICATION_TASK, REGRESSION_TASK)


@dataclass(frozen=True)
class ResearchGridResult:
    results: pd.DataFrame
    fold_results: pd.DataFrame
    feature_importance: pd.DataFrame
    dataset_summary: pd.DataFrame
    failures: pd.DataFrame
    manifest: dict

    def save(self, output_dir: str | Path) -> None:
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        self.results.to_csv(path / "results.csv", index=False)
        self.fold_results.to_csv(path / "fold_results.csv", index=False)
        self.feature_importance.to_csv(path / "feature_importance.csv", index=False)
        self.dataset_summary.to_csv(path / "dataset_summary.csv", index=False)
        self.failures.to_csv(path / "failures.csv", index=False)
        (path / "manifest.json").write_text(
            json.dumps(self.manifest, indent=2, default=str), encoding="utf-8"
        )

    def classification_view(self) -> pd.DataFrame:
        if self.results.empty:
            return self.results
        cols = [
            "horizon",
            "feature_set",
            "model",
            "n_predictions",
            "log_loss",
            "log_loss_improvement_vs_prior_pct",
            "brier",
            "brier_improvement_vs_prior_pct",
            "balanced_accuracy",
            "roc_auc",
        ]
        frame = self.results[self.results["task"] == CLASSIFICATION_TASK]
        return frame.loc[:, [c for c in cols if c in frame.columns]].copy()

    def regression_view(self) -> pd.DataFrame:
        if self.results.empty:
            return self.results
        cols = [
            "horizon",
            "feature_set",
            "model",
            "n_predictions",
            "rmse",
            "rmse_improvement_vs_zero_pct",
            "mae",
            "mae_improvement_vs_zero_pct",
            "directional_accuracy",
        ]
        frame = self.results[self.results["task"] == REGRESSION_TASK]
        return frame.loc[:, [c for c in cols if c in frame.columns]].copy()


def _classification_rows(
    bundle: ResearchDatasetBundle,
    *,
    feature_set: str,
    models: tuple[str, ...],
    outer_splits: int,
    inner_splits: int,
    test_size: int | None,
    n_trials: int,
    random_state: int,
):
    dataset = bundle.classification_dataset(feature_set)
    comparison = run_classification_comparison(
        dataset,
        models=models,
        outer_splits=outer_splits,
        inner_splits=inner_splits,
        gap=bundle.gap,
        test_size=test_size,
        n_trials=n_trials,
        random_state=random_state,
    )

    rows = []
    folds = []
    importance_rows = []
    prior = comparison.summary_frame().loc["Prior Baseline"]
    rows.append(
        {
            "task": CLASSIFICATION_TASK,
            "horizon": bundle.horizon,
            "gap": bundle.gap,
            "feature_set": feature_set,
            "model_key": "prior",
            "model": "Prior Baseline",
            **prior.to_dict(),
        }
    )

    for result in comparison.tuned_results:
        summary = comparison.summary_frame().loc[result.model_name]
        rows.append(
            {
                "task": CLASSIFICATION_TASK,
                "horizon": bundle.horizon,
                "gap": bundle.gap,
                "feature_set": feature_set,
                "model_key": result.model_key,
                "model": result.model_name,
                **summary.to_dict(),
            }
        )
        f = result.fold_frame().reset_index()
        f.insert(0, "model", result.model_name)
        f.insert(0, "model_key", result.model_key)
        f.insert(0, "feature_set", feature_set)
        f.insert(0, "horizon", bundle.horizon)
        f.insert(0, "task", CLASSIFICATION_TASK)
        folds.append(f)

        importance = result.feature_importance_frame()
        if importance is not None:
            imp = importance.reset_index().rename(columns={"index": "feature"})
            imp.insert(0, "model", result.model_name)
            imp.insert(0, "model_key", result.model_key)
            imp.insert(0, "feature_set", feature_set)
            imp.insert(0, "horizon", bundle.horizon)
            imp.insert(0, "task", CLASSIFICATION_TASK)
            importance_rows.append(imp)

    return rows, folds, importance_rows


def _regression_rows(
    bundle: ResearchDatasetBundle,
    *,
    feature_set: str,
    models: tuple[str, ...],
    outer_splits: int,
    inner_splits: int,
    test_size: int | None,
    n_trials: int,
    random_state: int,
):
    dataset = bundle.regression_dataset(feature_set)
    comparison = run_tuning_comparison(
        dataset,
        models=models,
        outer_splits=outer_splits,
        inner_splits=inner_splits,
        gap=bundle.gap,
        test_size=test_size,
        n_trials=n_trials,
        random_state=random_state,
    )

    rows = []
    folds = []
    importance_rows = []
    zero = comparison.summary_frame().loc["Zero Return"]
    rows.append(
        {
            "task": REGRESSION_TASK,
            "horizon": bundle.horizon,
            "gap": bundle.gap,
            "feature_set": feature_set,
            "model_key": "zero",
            "model": "Zero Return",
            **zero.to_dict(),
        }
    )

    for result in comparison.tuned_results:
        summary = comparison.summary_frame().loc[result.model_name]
        rows.append(
            {
                "task": REGRESSION_TASK,
                "horizon": bundle.horizon,
                "gap": bundle.gap,
                "feature_set": feature_set,
                "model_key": result.model_key,
                "model": result.model_name,
                **summary.to_dict(),
            }
        )
        f = result.fold_frame().reset_index()
        f.insert(0, "model", result.model_name)
        f.insert(0, "model_key", result.model_key)
        f.insert(0, "feature_set", feature_set)
        f.insert(0, "horizon", bundle.horizon)
        f.insert(0, "task", REGRESSION_TASK)
        folds.append(f)

        importance = result.feature_importance_frame()
        if importance is not None:
            imp = importance.reset_index().rename(columns={"index": "feature"})
            imp.insert(0, "model", result.model_name)
            imp.insert(0, "model_key", result.model_key)
            imp.insert(0, "feature_set", feature_set)
            imp.insert(0, "horizon", bundle.horizon)
            imp.insert(0, "task", REGRESSION_TASK)
            importance_rows.append(imp)

    return rows, folds, importance_rows


def run_research_grid(
    *,
    ticker: str,
    start: str,
    end: str | None,
    benchmark_ticker: str | None = None,
    horizons: tuple[int, ...] = (1, 5, 10, 20),
    feature_sets: tuple[str, ...] = ("legacy", "extended", "market", "regime"),
    tasks: tuple[str, ...] = (CLASSIFICATION_TASK,),
    classification_models: tuple[str, ...] = ("logistic", "catboost"),
    regression_models: tuple[str, ...] = ("ridge", "catboost"),
    outer_splits: int = 5,
    inner_splits: int = 3,
    test_size: int | None = 252,
    n_trials: int = 10,
    random_state: int = 42,
    continue_on_error: bool = True,
    progress: bool = True,
) -> ResearchGridResult:
    horizons = tuple(sorted(set(int(h) for h in horizons)))
    feature_sets = tuple(dict.fromkeys(feature_sets))
    tasks = tuple(dict.fromkeys(tasks))

    if not horizons or any(h < 1 for h in horizons):
        raise ValueError("horizons must contain positive integers.")
    unknown_sets = set(feature_sets).difference(FEATURE_SETS)
    if unknown_sets:
        raise ValueError(f"Unknown feature sets: {sorted(unknown_sets)}")
    unknown_tasks = set(tasks).difference(RESEARCH_TASKS)
    if unknown_tasks:
        raise ValueError(f"Unknown research tasks: {sorted(unknown_tasks)}")

    rows: list[dict] = []
    fold_frames: list[pd.DataFrame] = []
    importance_frames: list[pd.DataFrame] = []
    dataset_rows: list[dict] = []
    failures: list[dict] = []

    for h_idx, horizon in enumerate(horizons, start=1):
        if progress:
            print(f"\n[Horizon {horizon}d | gap={horizon}] building dataset...")
        try:
            bundle = build_research_dataset(
                ticker=ticker,
                benchmark_ticker=benchmark_ticker,
                start=start,
                end=end,
                horizon=horizon,
            )
        except Exception as exc:
            failures.append(
                {
                    "task": "dataset",
                    "horizon": horizon,
                    "feature_set": "all",
                    "model_key": "",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            if not continue_on_error:
                raise
            continue

        dataset_rows.append(
            {
                "ticker": bundle.ticker,
                "benchmark_ticker": bundle.benchmark_ticker,
                "horizon": horizon,
                "gap": bundle.gap,
                "rows": len(bundle.X_all),
                "start": bundle.X_all.index.min(),
                "end": bundle.X_all.index.max(),
                "positive_rate": bundle.directions.mean(),
                "mean_forward_return": bundle.forward_returns.mean(),
                "forward_return_std": bundle.forward_returns.std(),
            }
        )

        for fs_idx, feature_set in enumerate(feature_sets, start=1):
            if progress:
                print(
                    f"  [{fs_idx}/{len(feature_sets)}] feature_set={feature_set} "
                    f"({len(FEATURE_SETS[feature_set])} features)"
                )

            for task in tasks:
                try:
                    seed = random_state + horizon * 1000 + fs_idx * 100
                    if task == CLASSIFICATION_TASK:
                        r, f, imp = _classification_rows(
                            bundle,
                            feature_set=feature_set,
                            models=classification_models,
                            outer_splits=outer_splits,
                            inner_splits=inner_splits,
                            test_size=test_size,
                            n_trials=n_trials,
                            random_state=seed,
                        )
                    else:
                        r, f, imp = _regression_rows(
                            bundle,
                            feature_set=feature_set,
                            models=regression_models,
                            outer_splits=outer_splits,
                            inner_splits=inner_splits,
                            test_size=test_size,
                            n_trials=n_trials,
                            random_state=seed,
                        )
                    rows.extend(r)
                    fold_frames.extend(f)
                    importance_frames.extend(imp)

                    if progress:
                        nonbaseline = [x for x in r if x["model_key"] not in {"prior", "zero"}]
                        for item in nonbaseline:
                            if task == CLASSIFICATION_TASK:
                                print(
                                    f"      {item['model']}: "
                                    f"ΔLogLoss={item['log_loss_improvement_vs_prior_pct']:+.2f}% | "
                                    f"AUC={item['roc_auc']:.3f}"
                                )
                            else:
                                print(
                                    f"      {item['model']}: "
                                    f"ΔRMSE={item['rmse_improvement_vs_zero_pct']:+.2f}% | "
                                    f"DA={100*item['directional_accuracy']:.2f}%"
                                )
                except Exception as exc:
                    failures.append(
                        {
                            "task": task,
                            "horizon": horizon,
                            "feature_set": feature_set,
                            "model_key": "grid",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                    if progress:
                        print(f"      FAILED {task}: {type(exc).__name__}: {exc}")
                    if not continue_on_error:
                        raise

    result_frame = pd.DataFrame(rows)
    if not result_frame.empty:
        result_frame.insert(0, "ticker", ticker.strip().upper())
        if "n_predictions" in result_frame.columns:
            result_frame["n_predictions"] = pd.to_numeric(
                result_frame["n_predictions"], errors="coerce"
            )

    fold_frame = (
        pd.concat(fold_frames, ignore_index=True)
        if fold_frames
        else pd.DataFrame()
    )
    importance_frame = (
        pd.concat(importance_frames, ignore_index=True)
        if importance_frames
        else pd.DataFrame()
    )
    dataset_frame = pd.DataFrame(dataset_rows)
    failure_frame = pd.DataFrame(
        failures,
        columns=[
            "task",
            "horizon",
            "feature_set",
            "model_key",
            "error_type",
            "error",
        ],
    )

    manifest = {
        "milestone": 8,
        "purpose": "feature and forecast-horizon research",
        "ticker": ticker.strip().upper(),
        "benchmark_ticker": benchmark_ticker,
        "start": start,
        "end": end,
        "horizons": list(horizons),
        "feature_sets": list(feature_sets),
        "feature_set_columns": {k: list(FEATURE_SETS[k]) for k in feature_sets},
        "tasks": list(tasks),
        "classification_models": list(classification_models),
        "regression_models": list(regression_models),
        "outer_splits": outer_splits,
        "inner_splits": inner_splits,
        "test_size": test_size,
        "trials": n_trials,
        "purge_rule": "gap equals forecast horizon",
        "common_index_rule": "all feature sets share the full regime-feature supervised index within each horizon",
        "random_state": random_state,
        "failures": len(failure_frame),
    }

    return ResearchGridResult(
        results=result_frame,
        fold_results=fold_frame,
        feature_importance=importance_frame,
        dataset_summary=dataset_frame,
        failures=failure_frame,
        manifest=manifest,
    )
