import numpy as np
import pandas as pd

from stock_ml_lab.comparison.benchmark import run_comparison
from stock_ml_lab.global_model.dataset import build_global_panel_from_ohlcv


def make(seed: int, n: int = 520) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-04", periods=n)
    common = 0.0002 * np.sin(np.arange(n) / 13)
    ret = common + rng.normal(0, 0.01, n)
    close = 80 * np.exp(np.cumsum(ret))
    return pd.DataFrame({
        "open": close,
        "high": close * 1.01,
        "low": close * 0.99,
        "close": close,
        "volume": rng.lognormal(14, .2, n),
    }, index=idx)


def test_local_global_use_same_oos_dates_and_return_rows() -> None:
    bundle = build_global_panel_from_ohlcv({"AAA": make(1), "BBB": make(2)}, horizon=5)
    folds, assets, predictions = run_comparison(
        bundle,
        task="regression",
        local_model="ridge",
        global_model="ridge",
        n_splits=2,
        test_size_dates=60,
    )
    assert not folds.empty
    assert set(assets["ticker"]) == {"AAA", "BBB"}
    assert not predictions.empty
    assert (predictions["local_score"].notna() & predictions["global_score"].notna()).all()
    assert set(folds["test_start"]) and set(folds["test_end"])
    assert "qvalue_global_better" in assets.columns


def test_classification_comparison_uses_log_loss_and_horizon_hac() -> None:
    bundle = build_global_panel_from_ohlcv(
        {"AAA": make(1), "BBB": make(2)},
        horizon=20,
    )
    folds, assets, predictions = run_comparison(
        bundle,
        task="classification",
        local_model="logistic",
        global_model="logistic",
        n_splits=2,
        test_size_dates=60,
    )

    assert not folds.empty
    assert set(assets["criterion"]) == {"log_loss"}
    assert (assets["hac_lags"] >= 19).all()
    assert predictions["local_score"].between(0.0, 1.0).all()
    assert predictions["global_score"].between(0.0, 1.0).all()
    assert not predictions.duplicated(["ticker", "date", "fold"]).any()


def test_comparison_rejects_cross_family_pair() -> None:
    bundle = build_global_panel_from_ohlcv({"AAA": make(1), "BBB": make(2)}, horizon=5)
    try:
        run_comparison(
            bundle,
            task="regression",
            local_model="ridge",
            global_model="logistic",
            n_splits=1,
            test_size_dates=60,
        )
    except ValueError as exc:
        assert "same-family" in str(exc)
    else:
        raise AssertionError("cross-family comparison must be rejected")
