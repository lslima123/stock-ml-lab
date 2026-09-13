from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from stock_ml_lab.dataset import DatasetBundle
from stock_ml_lab.evaluation.classical_validation import evaluate_arima_cv, evaluate_var_cv
from stock_ml_lab.evaluation.validation import CVResult, evaluate_time_series_cv
from stock_ml_lab.models.baseline import PersistenceRegressor, ZeroReturnRegressor
from stock_ml_lab.models.linear import make_ridge_pipeline
from stock_ml_lab.models.tree import make_catboost, make_random_forest, make_xgboost


AVAILABLE_MODELS = (
    "zero",
    "persistence",
    "arima",
    "auto_arima",
    "var",
    "ridge",
    "random_forest",
    "xgboost",
    "catboost",
)

DISPLAY_NAMES = {
    "zero": "Zero Return",
    "persistence": "Persistence",
    "arima": "ARIMA(1,0,1)",
    "auto_arima": "AutoARIMA",
    "var": "VAR(5)",
    "ridge": "Ridge",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
}


@dataclass(frozen=True)
class ComparisonResult:
    results: tuple[CVResult, ...]

    def get(self, model_name: str) -> CVResult:
        for result in self.results:
            if result.model_name == model_name:
                return result
        raise KeyError(f"Model result '{model_name}' not found.")

    def summary_frame(self) -> pd.DataFrame:
        rows = []
        for result in self.results:
            rows.append(
                {
                    "model": result.model_name,
                    **result.metrics.to_dict(),
                    "n_predictions": len(result.predictions),
                    "n_splits": result.n_splits,
                    "gap": result.gap,
                }
            )

        frame = pd.DataFrame(rows).set_index("model")

        if "Zero Return" in frame.index:
            baseline_mae = float(frame.loc["Zero Return", "mae"])
            baseline_rmse = float(frame.loc["Zero Return", "rmse"])
            frame["mae_improvement_vs_zero_pct"] = (
                (baseline_mae - frame["mae"]) / baseline_mae * 100.0
            )
            frame["rmse_improvement_vs_zero_pct"] = (
                (baseline_rmse - frame["rmse"]) / baseline_rmse * 100.0
            )

        return frame.sort_values("rmse")

    def assert_common_protocol(self) -> None:
        if not self.results:
            return

        reference = self.results[0]
        reference_boundaries = [
            (fold.train_start, fold.train_end, fold.test_start, fold.test_end)
            for fold in reference.folds
        ]
        reference_index = reference.predictions.index

        for result in self.results[1:]:
            boundaries = [
                (fold.train_start, fold.train_end, fold.test_start, fold.test_end)
                for fold in result.folds
            ]
            if boundaries != reference_boundaries:
                raise RuntimeError(
                    f"Model '{result.model_name}' used different temporal fold boundaries."
                )
            if not result.predictions.index.equals(reference_index):
                raise RuntimeError(
                    f"Model '{result.model_name}' used different test timestamps."
                )


def _build_estimator(model_key: str, *, ridge_alpha: float):
    factories = {
        "zero": lambda: ZeroReturnRegressor(),
        "persistence": lambda: PersistenceRegressor(),
        "ridge": lambda: make_ridge_pipeline(alpha=ridge_alpha),
        "random_forest": lambda: make_random_forest(),
        "xgboost": lambda: make_xgboost(),
        "catboost": lambda: make_catboost(),
    }
    return factories[model_key]()


def run_model_comparison(
    dataset: DatasetBundle,
    *,
    ridge_alpha: float = 1.0,
    n_splits: int = 5,
    gap: int = 1,
    test_size: int | None = None,
    models: tuple[str, ...] = AVAILABLE_MODELS,
) -> ComparisonResult:
    unknown = set(models).difference(AVAILABLE_MODELS)
    if unknown:
        raise ValueError(f"Unknown model keys: {sorted(unknown)}")
    if not models:
        raise ValueError("At least one model must be selected.")

    sklearn_models = {"zero", "persistence", "ridge", "random_forest", "xgboost", "catboost"}
    results: list[CVResult] = []

    for model_key in models:
        if model_key in sklearn_models:
            estimator = _build_estimator(model_key, ridge_alpha=ridge_alpha)
            result = evaluate_time_series_cv(
                X=dataset.X,
                y=dataset.y,
                estimator=estimator,
                model_name=DISPLAY_NAMES[model_key],
                n_splits=n_splits,
                gap=gap,
                test_size=test_size,
            )
        elif model_key == "arima":
            result = evaluate_arima_cv(
                X=dataset.X,
                y=dataset.y,
                model_name=DISPLAY_NAMES[model_key],
                n_splits=n_splits,
                gap=gap,
                test_size=test_size,
                order=(1, 0, 1),
                auto=False,
            )
        elif model_key == "auto_arima":
            result = evaluate_arima_cv(
                X=dataset.X,
                y=dataset.y,
                model_name=DISPLAY_NAMES[model_key],
                n_splits=n_splits,
                gap=gap,
                test_size=test_size,
                auto=True,
            )
        elif model_key == "var":
            result = evaluate_var_cv(
                X=dataset.X,
                y=dataset.y,
                model_name=DISPLAY_NAMES[model_key],
                n_splits=n_splits,
                gap=gap,
                test_size=test_size,
                lags=5,
            )
        else:  # pragma: no cover - protected by validation above
            raise RuntimeError(f"Unhandled model key: {model_key}")

        results.append(result)

    comparison = ComparisonResult(results=tuple(results))
    comparison.assert_common_protocol()
    return comparison
