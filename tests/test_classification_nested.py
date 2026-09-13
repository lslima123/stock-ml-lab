from __future__ import annotations

import numpy as np
import pandas as pd

from stock_ml_lab.classification.dataset import DirectionDatasetBundle
from stock_ml_lab.classification.nested import (
    evaluate_prior_baseline,
    run_nested_classification,
)


def make_dataset(n: int = 180) -> DirectionDatasetBundle:
    index = pd.bdate_range("2023-01-02", periods=n)
    rng = np.random.default_rng(12)
    X = pd.DataFrame(
        {
            "return_1d": rng.normal(0, 0.01, n),
            "lag": rng.normal(0, 0.01, n),
            "ma": rng.normal(0, 0.02, n),
        },
        index=index,
    )
    score = 0.8 * X["return_1d"] - 0.4 * X["lag"] + rng.normal(0, 0.012, n)
    returns = pd.Series(score.to_numpy(), index=index, name="next_return")
    y = (returns > 0).astype(int).rename("next_direction")
    frame = X.copy()
    frame["next_return"] = returns
    frame["next_direction"] = y
    return DirectionDatasetBundle(
        ticker="TEST",
        X=X,
        y=y,
        next_returns=returns,
        frame=frame,
        feature_names=tuple(X.columns),
    )


def test_prior_and_nested_logistic_share_outer_timestamps() -> None:
    ds = make_dataset()
    prior = evaluate_prior_baseline(ds, outer_splits=3, gap=1, test_size=30)
    result = run_nested_classification(
        ds,
        model_key="logistic",
        outer_splits=3,
        inner_splits=2,
        gap=1,
        test_size=30,
        n_trials=2,
    )
    assert prior.probabilities.index.equals(result.probabilities.index)
    assert len(result.probabilities) == 90
    assert not result.probabilities.index.has_duplicates


def test_nested_classification_records_purged_folds_and_params() -> None:
    ds = make_dataset()
    result = run_nested_classification(
        ds,
        model_key="logistic",
        outer_splits=3,
        inner_splits=2,
        gap=1,
        test_size=30,
        n_trials=2,
    )
    positions = {ts: i for i, ts in enumerate(ds.X.index)}
    for fold in result.folds:
        assert positions[fold.test_start] - positions[fold.train_end] >= 2
        assert "C" in fold.best_params
        assert fold.best_inner_log_loss > 0
