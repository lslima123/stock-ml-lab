from __future__ import annotations

import pandas as pd

from stock_ml_lab.robustness.cross_asset import aggregate_cross_asset_results


def test_aggregate_cross_asset_results_counts_wins() -> None:
    frame = pd.DataFrame(
        {
            "model_key": ["catboost"] * 4,
            "model": ["Tuned CatBoost"] * 4,
            "rmse_improvement_vs_zero_pct": [1.0, 0.2, -0.1, 0.4],
            "directional_accuracy": [0.53, 0.52, 0.49, 0.54],
            "dm_mse_fdr_significant_05": [True, False, False, False],
            "pt_fdr_significant_05": [False, False, False, True],
            "rmse_bootstrap_positive_95": [True, False, False, False],
            "strategy_beats_buy_hold_return": [False, True, False, False],
            "strategy_beats_buy_hold_sharpe": [True, True, False, False],
            "nested_beats_buy_hold_return": [False, False, False, False],
            "nested_beats_buy_hold_sharpe": [False, True, False, False],
        }
    )

    summary = aggregate_cross_asset_results(frame).iloc[0]
    assert summary["n_assets"] == 4
    assert summary["rmse_win_count"] == 3
    assert summary["rmse_win_rate"] == 0.75
    assert summary["dm_mse_fdr_sig_count"] == 1
    assert summary["strategy_sharpe_beats_bh_count"] == 2
