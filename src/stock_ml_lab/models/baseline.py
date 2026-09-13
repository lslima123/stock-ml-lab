from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


class ZeroReturnRegressor(RegressorMixin, BaseEstimator):
    """Baseline that always predicts a next-period return of zero."""

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "n_features_in_")
        return np.zeros(len(X), dtype=float)


class PersistenceRegressor(RegressorMixin, BaseEstimator):
    """Predict the next return using the current day's return."""

    def __init__(self, feature_name: str = "return_1d"):
        self.feature_name = feature_name

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("PersistenceRegressor requires a pandas DataFrame.")
        if self.feature_name not in X.columns:
            raise ValueError(f"Feature '{self.feature_name}' not found in X.")

        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "n_features_in_")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("PersistenceRegressor requires a pandas DataFrame.")
        if self.feature_name not in X.columns:
            raise ValueError(f"Feature '{self.feature_name}' not found in X.")

        return X[self.feature_name].to_numpy(dtype=float)
