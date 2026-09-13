# Stock ML Lab — Full Research History

> This document preserves the detailed milestone-by-milestone record from the
> v0.15.0 release. For the project overview, installation instructions and main
> findings, return to the [README](../README.md).

The sections below document the decisions, commands, outputs and interpretation
rules accumulated throughout M1–M15. They are retained for auditability and
reproducibility; the milestone sequence should not be read as a set of escalating
claims of predictive performance.

---

Platform for comparing financial time-series forecasting models under rigorous
temporal validation. The research tasks cover 1, 5, 10 and 20 trading-day
forward returns and their direction; the project does not claim an automated
trading edge.

At the close of trading day `t`, an experiment uses only information available
through that close. For horizon `h`, every training label must mature before an
OOS test begins (`target_end_date < test_start`).

Current status: **M15 publication release completed**. M15 adds no new model or
hypothesis test: it consolidates the complete codebase, canonical evidence,
API, front-end, tests and release automation. The scientific endpoint remains
M14, whose independent test confirmed the h=5 pooled Ridge training-scope
effect on OOS RMSE; the corresponding pooled Logistic hypothesis was not
confirmed.

## Current model families

### Baselines

- Zero Return: `r_hat[t+1] = 0`
- Persistence: `r_hat[t+1] = r[t]`

### Classical time-series benchmarks

- ARIMA(1,0,1): fixed-order univariate benchmark on returns
- AutoARIMA: non-seasonal `(p,d,q)` selected by AIC on each training fold only
- VAR(5): bivariate system using `return_1d` and log volume growth

Classical models use rolling one-step-ahead forecasting. Parameters are estimated
once per fold. During the test period, newly observed data update the forecasting
state without re-estimating the parameters. This makes the information set
comparable to the tabular ML models, whose test-row features are built from data
known through the current close.

AutoARIMA uses `pmdarima` for order selection and `statsmodels` ARIMA for the
rolling state-space forecasts after the order has been selected.

### Machine learning

- Ridge
- Random Forest
- XGBoost
- CatBoost

All models share the same expanding-window `TimeSeriesSplit` boundaries and the
same default `gap=1` purge.

## Features

- current daily return
- return lags 1–3
- distance from 5/10/20-day moving averages
- 5/20-day rolling volatility
- RSI(14)
- daily volume change

## Metrics

- MAE
- RMSE
- directional accuracy
- directional coverage
- MAE improvement vs Zero Return
- RMSE improvement vs Zero Return

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Tests

```bash
pytest -q
```

## Build dataset

```bash
python main.py PETR4.SA \
    --start 2018-01-01 \
    --end 2026-01-01
```

## Full comparison

```bash
python experiment.py PETR4.SA \
    --start 2018-01-01 \
    --end 2026-01-01 \
    --test-size 252
```

## Classical benchmarks only

```bash
python experiment.py PETR4.SA \
    --start 2018-01-01 \
    --end 2026-01-01 \
    --test-size 252 \
    --models zero persistence arima auto_arima var
```

## Notes on interpretation

A model should not be considered useful merely because its directional accuracy
exceeds 50% in one fold. The project emphasizes out-of-fold aggregate metrics,
fold-to-fold stability and improvement relative to the zero-return benchmark.

Tree feature importances are normalized within each fold. Their numeric values
should not be compared directly across Random Forest, XGBoost and CatBoost,
because the libraries use different importance definitions.


## Milestone 4 — Nested Hyperparameter Optimization

Milestone 4 introduces Optuna while preserving the out-of-sample meaning of the
outer folds.

For each outer fold:

1. the outer test block is held out completely;
2. Optuna evaluates hyperparameters only through an inner expanding-window
   `TimeSeriesSplit` on the outer training block;
3. the selected configuration is refit on the complete outer training block;
4. the outer test block is evaluated exactly once.

This is nested temporal cross-validation. It avoids selecting hyperparameters on
the same observations later reported as out-of-sample performance.

Tunable models:

- Ridge
- Random Forest
- XGBoost
- CatBoost

The optimization objective is inner-CV RMSE. The search spaces are intentionally
bounded; Milestone 4 tests whether tuning improves generalization rather than
running an unrestricted parameter hunt.

### Quick tuning smoke test

```bash
python tune.py PETR4.SA \
    --start 2018-01-01 \
    --end 2026-01-01 \
    --test-size 252 \
    --models ridge \
    --trials 10
```

### Full Milestone 4 experiment

```bash
python tune.py PETR4.SA \
    --start 2018-01-01 \
    --end 2026-01-01 \
    --test-size 252 \
    --models ridge random_forest xgboost catboost \
    --trials 20
```

The full experiment performs a separate Optuna study for every model and every
outer fold. Start with 10 trials if you only want to verify the pipeline before
running the more expensive comparison.

Outputs include:

- untouched outer-fold MAE / RMSE / directional accuracy;
- improvement relative to Zero Return;
- inner-CV best RMSE for each outer fold;
- best hyperparameters selected independently in each outer fold;
- tuned tree feature importance.

## Milestone 5 — Statistical Validation & Economic Backtesting

Milestone 5 stops adding forecasting algorithms and asks a harder question:
**does the small out-of-sample advantage survive statistical uncertainty and
realistic trading frictions?**

### Statistical forecast validation

`validate.py` reruns the nested temporal tuning from Milestone 4 and evaluates
only the untouched outer-fold predictions.

For every selected tuned model it reports:

- Diebold–Mariano comparison against the Zero Return forecast under MSE loss;
- Diebold–Mariano comparison under MAE loss;
- a one-sided p-value for the alternative that the model has lower forecast loss;
- Pesaran–Timmermann directional predictive-accuracy test;
- circular moving-block bootstrap confidence intervals for MAE, RMSE,
  directional accuracy and the improvement relative to Zero Return.

The project targets the current stable statsmodels 0.14.6 release. On that
release, the validation layer uses its local Newey–West/HAC Diebold–Mariano and
Pesaran–Timmermann implementations. If a future statsmodels release exposes
official equivalents, the code will use them automatically when available.

The block bootstrap resamples contiguous chunks rather than independent days,
so the interval calculation preserves local temporal dependence better than an
i.i.d. bootstrap.

### Economic backtest

A prediction indexed by day `t` is the forecast for `return[t+1]`. Therefore the
position generated from that prediction earns the already-aligned supervised
`next_return`; no future return is used to create the signal.

Two position rules are supported:

```text
long_short:
    prediction > +threshold  -> +1
    prediction < -threshold  -> -1
    otherwise                ->  0

long_flat:
    prediction > +threshold  -> +1
    otherwise                ->  0
```

Transaction costs are charged in basis points per unit of position turnover:

```text
0 -> +1    = 1 unit of turnover
+1 -> -1   = 2 units of turnover
-1 -> 0    = 1 unit of turnover
```

Backtest metrics include:

- total return;
- annualized return;
- annualized volatility;
- Sharpe ratio;
- Sortino ratio;
- maximum drawdown;
- market exposure;
- annualized turnover;
- number of position changes;
- cumulative transaction-cost drag;
- active-day hit rate.

Buy & Hold over the exact same outer-test dates is reported as an economic
benchmark.

### Threshold analysis without outer-test leakage

The fixed-threshold table is explicitly **descriptive**. It shows how the final
OOS predictions behave at thresholds such as 0, 0.10%, 0.25% and 0.50%, but it
is not used to claim an optimized result.

The actual optimized strategy uses nested threshold selection:

1. take one outer fold from the already-nested tuned model;
2. reconstruct inner OOF forecasts using that fold's selected hyperparameters;
3. choose the threshold using only those outer-training forecasts;
4. apply that threshold to the untouched outer-test predictions;
5. repeat independently for every outer fold;
6. concatenate only the outer-test strategy returns.

Thus neither model hyperparameters nor the trading threshold are selected from
the returns later reported as final OOS performance.

### Run Milestone 5

Start with the current two strongest models:

```bash
python validate.py PETR4.SA \
    --start 2018-01-01 \
    --end 2026-01-01 \
    --test-size 252 \
    --models catboost xgboost \
    --trials 20 \
    --bootstrap 2000 \
    --block-length 20 \
    --cost-bps 10 \
    --thresholds 0 0.001 0.0025 0.005
```

For a faster pipeline check:

```bash
python validate.py PETR4.SA \
    --start 2018-01-01 \
    --end 2026-01-01 \
    --test-size 252 \
    --models catboost \
    --trials 5 \
    --bootstrap 200
```

To test a long-only implementation rather than shorting the stock:

```bash
python validate.py PETR4.SA \
    --start 2018-01-01 \
    --end 2026-01-01 \
    --models catboost \
    --strategy long_flat
```

### Important limitations

This remains a research backtest, not a trading simulator. Long-short results
include configurable transaction costs but do **not** yet model stock-borrow
fees, financing, bid/ask spread dynamics, market impact, taxes, order latency or
execution slippage beyond the chosen basis-point cost. These belong in later
execution-oriented work if the statistical signal survives Milestone 5.


## Milestone 6 — Cross-Asset Robustness

Milestone 6 tests whether conclusions obtained on one ticker generalize across a
predeclared universe. It does not search for the asset with the prettiest result.

Default core universe:

### Brazil
- PETR4.SA
- VALE3.SA
- ITUB4.SA
- WEGE3.SA
- BOVA11.SA

### United States
- AAPL
- MSFT
- NVDA
- JPM
- SPY

Every asset uses the same nested temporal protocol, hyperparameter search space,
transaction-cost assumption, statistical tests and reporting logic.

### Multiple-testing control

Testing many assets creates a multiple-comparisons problem. Milestone 6 applies
Benjamini-Hochberg FDR correction separately for each model across assets to:

- Diebold-Mariano MSE one-sided p-values;
- Diebold-Mariano MAE one-sided p-values;
- Pesaran-Timmermann directional p-values.

Raw p-values are still retained.

The aggregate report also includes an exact one-sided sign test for the number of
assets on which RMSE improves relative to Zero Return.

### Cross-asset outputs

The run writes machine-readable files to `reports/m6/`:

- `asset_results.csv` — one row per asset/model;
- `model_summary.csv` — aggregate robustness statistics;
- `fold_results.csv` — outer-fold diagnostics and selected parameters;
- `failures.csv` — tickers that could not be evaluated;
- `manifest.json` — complete experimental configuration.

### Recommended first M6 run

Start with the current strongest model, CatBoost:

```bash
python3 cross_asset.py \
    --start 2018-01-01 \
    --end 2026-01-01 \
    --models catboost \
    --trials 20 \
    --bootstrap 1000 \
    --cost-bps 10
```

This runs the complete 10-asset core universe.

For a faster smoke test:

```bash
python3 cross_asset.py \
    --tickers PETR4.SA VALE3.SA AAPL SPY \
    --start 2018-01-01 \
    --end 2026-01-01 \
    --models catboost \
    --trials 5 \
    --bootstrap 200 \
    --skip-nested-threshold
```

After CatBoost, compare the two boosting families:

```bash
python3 cross_asset.py \
    --start 2018-01-01 \
    --end 2026-01-01 \
    --models catboost xgboost \
    --trials 20 \
    --bootstrap 1000 \
    --cost-bps 10
```

### Interpretation discipline

A model is not considered robust merely because it wins on one or two tickers.
The primary M6 questions are:

1. On how many predeclared assets does RMSE beat Zero Return?
2. Is the median improvement positive?
3. Does the exact sign test support a cross-asset win rate above 50%?
4. Which DM/PT results survive FDR correction?
5. How often does the trading rule beat Buy & Hold after costs?
6. Do conclusions differ systematically between Brazil, U.S. equities and ETFs?

This milestone is specifically designed to prevent ticker selection from becoming
another hidden source of overfitting.

## Milestone 7 — Direct Direction Classification

Milestone 7 reframes the forecasting task. Instead of estimating the magnitude of
`next_return`, the target is now

```text
next_direction[t] = 1 if next_return[t] > 0 else 0
```

The information set is unchanged: at the close of day `t`, features use only data
available through that close and the label depends on the return at `t+1`.
`gap=1` therefore remains mandatory at every temporal train/test boundary.

### Why a separate classification task?

Milestones 1–6 showed that return-magnitude regressors had little robust
cross-asset edge, while a few directional results were more suggestive. M7 tests
that hypothesis directly rather than asking a regressor to solve two problems at
once (sign and magnitude).

### Models

The probabilistic baseline predicts the positive-class prior estimated from each
outer training fold. Tuned classifiers are:

- Logistic Regression
- Random Forest Classifier
- XGBoost Classifier
- CatBoost Classifier

Every tuned model uses nested expanding-window temporal CV. Optuna minimizes
**inner out-of-fold log loss**, while the outer test blocks remain untouched.

### Metrics

M7 reports both hard-label and probabilistic diagnostics:

- accuracy;
- balanced accuracy;
- precision;
- recall;
- F1;
- ROC-AUC;
- Brier score;
- log loss;
- actual positive rate;
- predicted positive rate;
- mean predicted probability;
- expected calibration error (ECE).

Brier score and log loss are proper probabilistic scoring rules, but neither is a
pure calibration metric by itself. Therefore M7 also writes quantile reliability
bins (`calibration.csv`) comparing mean predicted probability with observed
positive frequency.

### Probability-based trading rules

Let `p_t = P(next_return > 0 | X_t)`. For a confidence margin `m`, the long-short
rule is

```text
p_t > 0.5 + m  -> long
p_t < 0.5 - m  -> short
otherwise      -> cash
```

The fixed-margin table is descriptive only. The deployable research result uses a
nested probability-margin selector: each outer fold selects its margin using only
inner OOF predictions from the outer training block, then applies the selected
margin to the untouched outer test block.

Default probability margins are:

```text
0.000  -> 50/50 decision boundary
0.025  -> long above 52.5%, short below 47.5%
0.050  -> long above 55%,   short below 45%
0.100  -> long above 60%,   short below 40%
```

### Statistical uncertainty

For every tuned classifier, M7 applies a circular block bootstrap against the
training-prior baseline and reports 95% intervals for:

- accuracy improvement (percentage points);
- balanced-accuracy improvement (percentage points);
- Brier improvement (%);
- log-loss improvement (%);
- ROC-AUC excess above 0.5 (percentage points).

Pesaran–Timmermann is also reported on `p(up) - 0.5` to preserve continuity with
the directional forecast tests from M5/M6.

### Reports never overwrite silently

M7 automatically creates a unique directory such as:

```text
reports/m7/20260827T101500_PETR4.SA_catboost/
```

and writes:

- `summary.csv`
- `fold_results.csv`
- `predictions.csv`
- `bootstrap_intervals.csv`
- `calibration.csv`
- `backtest.csv`
- `fixed_probability_margins.csv`
- `nested_probability_margins.csv`
- `feature_importance.csv` (tree models)
- `manifest.json`

The M6 CLI was also updated so future cross-asset runs get unique output folders
unless `--output-dir` is explicitly supplied.

### Recommended M7 smoke test

Start with logistic regression because it is fast and gives a strong linear
probabilistic benchmark:

```bash
python3 classify.py PETR4.SA \
    --start 2018-01-01 \
    --end 2026-01-01 \
    --test-size 252 \
    --models logistic \
    --trials 10 \
    --bootstrap 500 \
    --cost-bps 10
```

### Boosting classification

Then test the two families motivated by M4–M6:

```bash
python3 classify.py PETR4.SA \
    --start 2018-01-01 \
    --end 2026-01-01 \
    --test-size 252 \
    --models xgboost catboost \
    --trials 20 \
    --bootstrap 1000 \
    --cost-bps 10
```

### Full M7 comparison

```bash
python3 classify.py PETR4.SA \
    --start 2018-01-01 \
    --end 2026-01-01 \
    --test-size 252 \
    --models logistic random_forest xgboost catboost \
    --trials 20 \
    --bootstrap 1000 \
    --cost-bps 10 \
    --probability-margins 0 0.025 0.05 0.10
```

The central M7 question is not simply whether accuracy exceeds 50%. It is whether
probabilistic classifiers improve on the time-varying class-prior baseline,
remain calibrated enough to support confidence thresholds, and create an
out-of-sample economic rule that survives transaction costs.

## Milestone 8 — Feature & Horizon Research

Milestone 8 changes the formulation of the prediction problem instead of adding
another algorithm. It studies two dimensions under the same nested temporal
protocol:

1. **forecast horizon**: 1, 5, 10 and 20 trading days;
2. **information set**: legacy technical features, extended asset features,
   market context and regime features.

### Multi-horizon targets

For a horizon `h`, the regression target is

```text
forward_return_h[t] = close[t + h] / close[t] - 1
```

and the classification target is

```text
forward_direction_h[t] = 1(forward_return_h[t] > 0)
```

The model still uses only information available through the close at `t`.

### Horizon-aware purge

The temporal purge grows with the target horizon:

```text
horizon=1   -> gap=1
horizon=5   -> gap=5
horizon=10  -> gap=10
horizon=20  -> gap=20
```

This is required because a training label at time `t` depends on the future
close at `t+h`. Keeping `gap=h` prevents the last training labels from consuming
prices belonging to the next validation/test block.

### Feature sets

Feature sets are cumulative and are evaluated on the **same supervised index
within a horizon**, so improvements cannot come merely from changing the test
calendar.

#### `legacy`

The original M1 feature set:

- daily return and return lags 1–3;
- moving-average distances 5/10/20;
- volatility 5/20;
- RSI(14);
- daily volume change.

#### `extended`

Adds asset-only information:

- momentum 5/20/60;
- drawdown 20/60;
- intraday high-low range;
- 20-day volume z-score.

#### `market`

Adds benchmark context:

- benchmark return;
- benchmark momentum 5/20/60;
- benchmark volatility 5/20;
- asset-minus-market daily return;
- relative 20-day momentum.

Default benchmark selection:

```text
Brazil (.SA) -> ^BVSP
U.S. equities -> SPY
SPY -> ^GSPC
```

A benchmark can be overridden explicitly with `--benchmark`.

#### `regime`

Adds slower state variables:

- asset short/long volatility ratio;
- market short/long volatility ratio;
- rolling 60-day beta;
- rolling 60-day correlation;
- asset 20/60 trend;
- market 20/60 trend;
- 60-day relative strength.

All calculations are backward-looking through date `t` only.

### Tasks and models

M8 supports both tasks while reusing the nested tuning infrastructure already
validated in M4/M7.

Classification can use:

- Logistic Regression;
- Random Forest Classifier;
- XGBoost Classifier;
- CatBoost Classifier.

Regression can use:

- Ridge;
- Random Forest;
- XGBoost;
- CatBoost.

Classification tuning continues to minimize inner-OOF log loss. Regression
tuning continues to minimize inner-OOF RMSE.

### Recommended experimental sequence

Do **not** immediately run every model × feature set × horizon. That turns the
research grid into another source of configuration overfitting.

#### Stage A — horizon screen with simple models

```bash
python3 research.py PETR4.SA \
    --start 2018-01-01 \
    --end 2026-01-01 \
    --horizons 1 5 10 20 \
    --feature-sets legacy \
    --tasks classification regression \
    --classification-models logistic \
    --regression-models ridge \
    --trials 10
```

This asks whether changing `h` alone improves the problem before adding more
information.

#### Stage B — feature ablation at all horizons

```bash
python3 research.py PETR4.SA \
    --start 2018-01-01 \
    --end 2026-01-01 \
    --horizons 1 5 10 20 \
    --feature-sets legacy extended market regime \
    --tasks classification \
    --classification-models logistic \
    --trials 10
```

This isolates the value of richer information while keeping the model simple.

#### Stage C — nonlinear confirmation

After Stage A/B identify a small number of plausible configurations, confirm
them with CatBoost/XGBoost, for example:

```bash
python3 research.py PETR4.SA \
    --start 2018-01-01 \
    --end 2026-01-01 \
    --horizons 5 20 \
    --feature-sets market regime \
    --tasks classification regression \
    --classification-models catboost \
    --regression-models catboost \
    --trials 20
```

### Reports

Each run receives a unique directory under `reports/m8/` and writes:

- `results.csv` — one row per task/horizon/feature-set/model;
- `fold_results.csv` — outer-fold diagnostics;
- `feature_importance.csv` — available tree importances;
- `dataset_summary.csv` — row counts, class prevalence and target dispersion;
- `failures.csv` — failed configurations;
- `manifest.json` — full experiment specification.

### Interpretation discipline

M8 is explicitly **exploratory configuration research**. Sorting feature sets
and horizons by OOS metrics does not create a new final holdout. Any configuration
selected from this grid must later be confirmed on untouched data or in a new
predeclared cross-asset experiment before being called superior.

For horizons greater than one day, M8 does not convert the overlapping forward
returns directly into a trading backtest. A correct multi-horizon portfolio
simulation requires an explicit holding/rebalancing convention and is left for a
later economic-validation stage.


## Milestone 8.1 — Locked Confirmation

M8 identified a medium-horizon signal on PETR4.SA, with the strongest exploratory
results coming from the simplest formulation rather than more complex feature sets:

- 10d + legacy + Logistic Regression;
- 20d + legacy + Logistic Regression;
- 10d + legacy + Ridge;
- 20d + legacy + Ridge.

M8.1 **freezes those four candidates**. It does not search new horizons, feature
sets, or model families.

### Discovery vs confirmation

PETR4.SA is the discovery series and is excluded from the default M8.1 sample.
The predeclared confirmation universe is:

- VALE3.SA
- ITUB4.SA
- WEGE3.SA
- BOVA11.SA
- AAPL
- MSFT
- NVDA
- JPM
- SPY

These assets were not used to select the 10d/20d + legacy formulation in M8.

### Overlapping-target correction

For an h-day forward target, adjacent labels share most of their return window.
M8.1 therefore distinguishes four separate controls:

1. **Leakage purge:** `gap = h` in both outer and inner temporal CV.
2. **HAC inference:** loss-differential tests use Newey-West lags `h - 1`.
3. **Block bootstrap:** the effective block length is at least `h`.
4. **Non-overlapping phases:** OOS timestamps are split by supervised-row
   position modulo `h`. Within each phase, successive forward-return windows are
   h rows apart and do not overlap (they can share only a boundary price).

For every asset/candidate, M8.1 reports the median, IQR, minimum, maximum, and win
rate across all 10 or 20 non-overlapping phases. No phase is selected after seeing
its result.

### Primary confirmation metrics

Classification:
- primary metric: Log Loss improvement vs training-prior baseline;
- secondary metric: Brier improvement;
- ranking diagnostic: ROC-AUC.

Regression:
- primary metric: RMSE improvement vs Zero Return;
- secondary metric: MAE improvement;
- diagnostic: directional accuracy.

The primary loss differential receives a horizon-aware HAC p-value. Across the
nine confirmation assets, Benjamini-Hochberg FDR is applied separately to each
locked task/horizon candidate.

### Run the locked confirmation

```bash
python3 confirm.py \
    --start 2018-01-01 \
    --end 2026-01-01 \
    --trials 10 \
    --bootstrap 1000 \
    --block-length 20
```

The default command runs all nine confirmation assets. A smoke-test subset is
allowed without changing the locked formulation:

```bash
python3 confirm.py \
    --tickers VALE3.SA AAPL SPY \
    --start 2018-01-01 \
    --end 2026-01-01 \
    --trials 3 \
    --bootstrap 200
```

### Outputs

Every run creates a unique directory under `reports/m8_1/` containing:

- `asset_results.csv`
- `candidate_summary.csv`
- `phase_results.csv`
- `fold_results.csv`
- `dataset_summary.csv`
- `failures.csv`
- `manifest.json`

### Interpretation discipline

A candidate is not confirmed because one asset has a high AUC or directional
accuracy. Evidence should be read jointly from:

- cross-asset win rate and median improvement;
- HAC p-values and FDR-adjusted q-values;
- block-bootstrap confidence intervals;
- non-overlapping phase stability.

The exact cross-asset sign test is retained as a compact robustness diagnostic,
but assets are economically correlated, so it should not be interpreted as if
all nine assets were independent experiments.


## Milestone 9 — Visualization & Reporting

Milestone 9 freezes the modeling research and adds a static analytics/reporting
layer over persisted experiment artifacts.

The reporting layer is deliberately **evidence-aware**:

- exploratory M8 discovery is shown separately from M8.1 locked confirmation;
- naive baselines remain visible;
- bootstrap intervals and FDR/HAC confirmation results are not hidden;
- negative confirmation is treated as a result, not as a visualization problem;
- charts are generated from persisted CSVs, not from notebook state.

### Outputs

`visualize.py` generates a unique static report directory containing:

```text
index.html
report.md
report_manifest.json
normalized_m8_results.csv
normalized_m81_asset_results.csv
normalized_m81_candidate_summary.csv
normalized_m7_summary.csv              # when supplied
normalized_m6_asset_results.csv        # when supplied
figures/
    discovery_vs_confirmation.png
    m8_logistic_auc_by_horizon.png
    m8_ridge_rmse_by_horizon.png
    m8_feature_horizon_auc.png
    m81_classification_10d_assets.png
    m81_classification_20d_assets.png
    m81_regression_10d_assets.png
    m81_regression_20d_assets.png
    m81_phase_win_rates.png
    ...
```

The HTML is self-contained apart from the adjacent PNG figures and can be opened
locally in any browser. No web server is required.

### Immediate demo

A bundled snapshot reproduces the research story observed through M7/M8/M8.1:

```bash
python3 visualize.py --demo
```

This is intended only for smoke testing and portfolio demonstration. For the
authoritative report, point M9 at the actual persisted experiment directories.

### Build from the actual runs

`--source` may point at either a single run directory or a parent directory. It
can be repeated. M9 auto-detects M6, M7, M8 and M8.1 schemas and de-duplicates
overlapping M8 grids, preferring the later supplied result for an identical
`task × horizon × feature_set × model` key.

Example using the M8 discovery and M8.1 confirmation folders:

```bash
python3 visualize.py \
    --source ../stock-ml-lab-milestone-8/reports/m8 \
    --source ../stock-ml-lab-milestone-8-1/reports/m8_1
```

Add M7 if desired:

```bash
python3 visualize.py \
    --source ../stock-ml-lab-milestone-7/reports/m7 \
    --source ../stock-ml-lab-milestone-8/reports/m8 \
    --source ../stock-ml-lab-milestone-8-1/reports/m8_1
```

If preserved M6 runs are available, they can be included with another
`--source`.

### Flagship M9 visual

The key plot is `discovery_vs_confirmation.png`. It places the PETR4 exploratory
gain beside the median locked-confirmation gain across the nine untouched assets.

This prevents a portfolio viewer from seeing only the attractive discovery
number while missing that the predeclared confirmation stage did not reproduce
the edge.

### Why static reporting before API/front-end?

M9 establishes a stable analytical contract for later product layers:

1. experiments save machine-readable artifacts;
2. M9 normalizes them into reporting tables;
3. M10 can expose those tables through an API;
4. M11 can render the API in an interactive front-end.

That keeps statistical research, application logic and presentation concerns
separate.

## Milestone 10 — FastAPI Application API

Milestone 10 turns the research code into an application-facing HTTP API while
keeping statistical research and product inference separate.

### Design goals

- expose a stable contract for the M11 front-end;
- provide real local-model predictions for a requested ticker;
- expose the M9/M8/M8.1 research evidence as machine-readable JSON;
- preserve naive baselines in the model registry;
- expose `local`, `global` and same-family `compare` scopes through one stable contract;
- avoid running Optuna or nested CV inside an HTTP request.

The API uses FastAPI + Pydantic and is available through `api.py`.

### Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -q
```

Then start the development server:

```bash
uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000/docs
http://127.0.0.1:8000/redoc
http://127.0.0.1:8000/health
```

### API contract

#### Capabilities

```http
GET /api/v1/capabilities
```

The response lists supported tasks, horizons, local models and scopes.

Current scope availability:

```text
local    available
global   available when the eight pretrained artifacts are present
compare  available with the same global-artifact requirement
```

Availability is reported dynamically from the artifact store.

#### Local prediction

```http
POST /api/v1/predict
Content-Type: application/json
```

Regression example:

```json
{
  "ticker": "PETR4.SA",
  "task": "regression",
  "model": "ridge",
  "scope": "local",
  "horizon": 20,
  "start": "2018-01-01",
  "feature_set": "legacy"
}
```

Classification example:

```json
{
  "ticker": "AAPL",
  "task": "classification",
  "model": "logistic",
  "scope": "local",
  "horizon": 10,
  "start": "2018-01-01"
}
```

The local endpoint downloads the requested asset's adjusted OHLCV history,
constructs the same leakage-safe legacy features used in the research pipeline,
creates an `h`-day forward target, fits the selected model on the supervised
history and predicts from the latest feature row.

For an `h`-day target, the final `h` rows do not have known training labels and
therefore are not included in `X_train`. The inference row is the latest valid
feature row. The API response reports both `training_end` and `as_of` so this
separation is observable.

Supported local regression models:

```text
zero
ridge
random_forest
xgboost
catboost
```

Supported local classification models:

```text
prior
logistic
random_forest
xgboost
catboost
```

Important: HTTP inference uses fixed, documented product presets and fits on
demand. It does **not** run Optuna inside the request, and the returned forecast
must not be confused with the nested-CV research metric for a historical run.

#### Global contract

The following request already validates structurally:

```json
{
  "ticker": "PETR4.SA",
  "task": "regression",
  "model": "ridge",
  "scope": "global",
  "horizon": 20
}
```

M12 implements this request from a pretrained pooled artifact. M13 implements
`scope="compare"`, returning local and global same-family predictions side by side.

### Market snapshot

```http
GET /api/v1/market/PETR4.SA/snapshot
```

Returns the latest adjusted close, date, one-day return and volume for use in the
M11 interface.

### Research analytics endpoints

```http
GET /api/v1/research/summary
GET /api/v1/research/discovery
GET /api/v1/research/confirmation
GET /api/v1/research/scope-benchmark
GET /api/v1/research/scope-confirmation
```

Filters are available, for example:

```http
GET /api/v1/research/confirmation?task=regression&horizon=20
```

A bundled M9 snapshot is included so these endpoints work immediately. To serve
the authoritative report generated from your real experiment runs, set:

```bash
export STOCK_ML_LAB_ANALYTICS_DIR="../stock-ml-lab-milestone-9/reports/m9/<RUN_DIRECTORY>"
uvicorn api:app --reload
```

The target directory should contain the M9 normalized CSV files such as:

```text
normalized_m8_results.csv
normalized_m81_asset_results.csv
normalized_m81_candidate_summary.csv
report_manifest.json
```

### CORS for M11

Default allowed browser origins are:

```text
http://localhost:3000
http://localhost:5173
http://127.0.0.1:5173
```

Override them with:

```bash
export STOCK_ML_LAB_CORS_ORIGINS="http://localhost:5173,https://your-ui.example"
```

### Docker

A minimal container is included:

```bash
docker build -t stock-ml-lab-api .
docker run --rm -p 8000:8000 stock-ml-lab-api
```

Then visit `http://localhost:8000/docs`.

### M10 architecture

```text
M1–M8.1 research pipeline
        │
        ├── experiment artifacts
        │
M9 reporting / normalized CSVs
        │
        ├───────────────┐
        │               │
M10 research API   M10 inference service
        │               │
        └───────┬───────┘
                │
              M11 UI
                │
        ┌───────┴────────┐
        │                │
   M12 global model   local models
        │                │
        └────── M13 compare ──────
```

The product boundary is now explicit: experimental validation remains in the
research modules, while application requests go through a typed API contract.


## Milestone 11 — React Front-end

Milestone 11 adds the product interface while preserving the separation between
live inference and research evidence.

The front-end lives in `frontend/` and uses:

- React 19;
- TypeScript;
- Vite;
- the M10 FastAPI contract;
- capability-driven model scopes.

The interface exposes:

- ticker lookup and market snapshot;
- regression vs direction classification;
- model selection filtered by task;
- 1d / 5d / 10d / 20d horizons;
- `local`, `global`, and `compare` scope cards;
- live local inference results;
- training context and hyperparameters;
- M8 discovery vs M8.1 locked-confirmation context.

The UI reads `GET /api/v1/capabilities`; Global and Compare turn on when the
complete artifact set is available. The evidence panel now separates PETR4
discovery, M8.1 locked confirmation and the real M13 scope benchmark.

### Run backend + front-end in development

Shell 1:

```bash
source .venv/bin/activate
uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

Shell 2:

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://127.0.0.1:5173/app/
```

### Production build served by FastAPI

```bash
cd frontend
npm install
npm run build
cd ..
uvicorn api:app --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000/app/
```

### Docker

The M11 Dockerfile is multi-stage: Node builds the React application, then the
final Python image installs Stock ML Lab and serves both the API and the built UI.

```bash
docker build -t stock-ml-lab .
docker run --rm -p 8000:8000 stock-ml-lab
```

Then open `http://127.0.0.1:8000/app/`.


## Milestone 12 — Global Multi-Asset Model

M12 adds a true pooled model. This is **cross-asset training**, not merely
cross-asset evaluation.

The default core universe is:

```text
PETR4.SA, VALE3.SA, ITUB4.SA, WEGE3.SA, BOVA11.SA,
AAPL, MSFT, NVDA, JPM, SPY
```

For every horizon (`1, 5, 10, 20` trading days), M12 builds one stacked panel:

```text
date       ticker    legacy features ...    forward_return    target_end_date
2023-...   PETR4.SA  ...                    ...               ...
2023-...   VALE3.SA  ...                    ...               ...
2023-...   AAPL      ...                    ...               ...
```

The initial global candidates are deliberately simple:

- regression: pooled Ridge;
- classification: pooled Logistic Regression.

Ticker identity is **not** used as a feature. The model therefore learns a
shared mapping from dimensionless technical features to the target and can be
applied to a ticker that was not in its training universe.

### Leakage control

The key panel rule is stronger than a row gap:

```text
training target_end_date < first test feature date
```

Therefore no h-day training label is allowed to mature inside the test window,
even when observations from many tickers are stacked together.

M12 runs two complementary evaluations:

1. **calendar walk-forward** — all assets train before a common future test window;
2. **unseen-asset holdout** — one ticker is removed entirely from training, while
   the other assets may train only on labels matured before that ticker's final
   test window.

This separates temporal generalization from cross-series generalization.

### Train real global artifacts

```bash
python3 global_train.py \
    --start 2018-01-01 \
    --end 2026-01-01 \
    --horizons 1 5 10 20 \
    --output-dir artifacts/global
```

Outputs:

```text
artifacts/global/
    global_regression_ridge_h1.joblib
    global_regression_ridge_h5.joblib
    global_regression_ridge_h10.joblib
    global_regression_ridge_h20.joblib
    global_classification_logistic_h1.joblib
    global_classification_logistic_h5.joblib
    global_classification_logistic_h10.joblib
    global_classification_logistic_h20.joblib
    manifest.json

reports/m12/<run>/
    temporal_metrics.csv
    per_asset_metrics.csv
    unseen_asset_metrics.csv
    failures.csv
    manifest.json
```

After training, restart FastAPI. `GET /api/v1/capabilities` dynamically switches
the Global scope to `available=true` once all eight default artifacts exist.

You can inspect the artifact contract directly with:

```text
GET /api/v1/global/status
```

### Global inference

The existing endpoint now accepts:

```json
{
  "ticker": "PETR4.SA",
  "task": "regression",
  "model": "ridge",
  "scope": "global",
  "horizon": 20,
  "start": "2018-01-01",
  "feature_set": "legacy"
}
```

Unlike local inference, this request does not fit a model. It loads the
pretrained global artifact and applies it to the latest valid feature row for
the requested ticker.

M13 adds the first-class `compare` response and controlled local-vs-global benchmark.

## Milestone 13 — Local vs Global Benchmark

M13 compares **training scope while holding model family fixed**. The supported pairs are:

- Regression: Local Ridge vs Global Ridge.
- Classification: Local Logistic Regression vs Global Logistic Regression.

Both scopes are evaluated on identical out-of-sample calendar folds with the same feature contract and target-maturity purge. Asset-level comparisons include HAC loss tests, Benjamini-Hochberg FDR adjustment, and an exact sign test for the number of global wins.

For horizons greater than one trading day, HAC lags are at least `horizon - 1` because forward-return targets overlap. The benchmark therefore does not treat overlapping observations as independent.

The HTTP API also exposes `scope="compare"`, returning local and global predictions together. This comparison intentionally rejects cross-family pairs: comparing Local XGBoost against Global Ridge would mix algorithm and training-scope effects. The product can add more global model families later without weakening this identification principle.

### Observed M13 benchmark

The real ten-asset benchmark produced the following aggregate results. Positive
improvement means lower Global loss than the same-family Local model.

| Task | h | Global wins | Mean improvement | Median improvement | Asset HAC/FDR < 5% |
| --- | ---: | ---: | ---: | ---: | ---: |
| Classification | 1 | 8/10 | +0.389% | +0.423% | 3 |
| Classification | 5 | 10/10 | +0.820% | +0.907% | 1 |
| Classification | 10 | 8/10 | +1.069% | +1.065% | 0 |
| Classification | 20 | 9/10 | +2.572% | +1.711% | 0 |
| Regression | 1 | 8/10 | +0.755% | +0.325% | 3 |
| Regression | 5 | 9/10 | +0.921% | +0.867% | 2 |
| Regression | 10 | 9/10 | +1.045% | +0.975% | 0 |
| Regression | 20 | 9/10 | +1.740% | +1.741% | 0 |

Date-clustered HAC, circular date-block bootstrap, fold, phase and regional
checks were added after observing the benchmark. They are therefore labeled
**post hoc robustness diagnostics**, even though all eight pooled comparisons
were supportive after correction. The h=5 pair was selected for M14 because it
was the most stable across assets, folds and non-overlapping phases—not because
it had the largest headline effect.

The curated M13 snapshot lives in `examples/m13_snapshot/`. Its compressed
prediction file contains the full 97,770-row OOS comparison contract.

## Milestone 14 — Locked Independent Scope Confirmation

M14 tested exactly two hypotheses at `h=5` with legacy features:

- Local Ridge vs Global Ridge, primary metric RMSE;
- Local Logistic vs Global Logistic, primary metric Log Loss.

The pooled models train only on the original M13 universe. Confirmation uses ten
different assets:

```text
BBAS3.SA, ABEV3.SA, AXIA3.SA, SUZB3.SA, RENT3.SA,
GOOGL, AMZN, META, XOM, UNH
```

The date window, assets, model families, horizon, features and inference rule are
frozen in `configs/m14_locked_scope.json`. No tuning or ticker substitutions are
allowed. Training labels obey the target-maturity purge; primary inference
clusters the panel by OOS date, uses four HAC lags, applies BH FDR across the two
locked hypotheses and uses a circular date-block bootstrap of length five.

Protocol amendment dated 2026-09-08: the provider identifier `ELET3.SA` was
corrected to `AXIA3.SA` after the official Eletrobras/AXIA ticker change effective
2025-11-10. This preserves the same ordinary-share issuer and is not an asset
substitution. The first locked preflight aborted before any model was evaluated.

A task is confirmed only if:

```text
one-sided date-clustered HAC q < 0.05
AND
95% bootstrap improvement interval lower bound > 0
```

### Locked results

| Task | Global improvement | HAC/FDR q | 95% block-bootstrap CI | Asset wins | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Regression / Ridge | +1.7166% RMSE | 0.000132 | [+0.9096%, +2.5647%] | 9/10 | **Confirmed** |
| Classification / Logistic | +0.4098% Log Loss | 0.157742 | [-0.3876%, +1.2215%] | 7/10 | Not confirmed |

The regression advantage was positive in all 5 calendar folds, all 5
non-overlapping phases and both regional blocks. Five regression assets survived
the supporting within-task asset-level HAC/FDR adjustment. AMZN was the only
asset with a negative point estimate.

Classification was positive in only 2/5 calendar folds and was negative for the
Brazilian regional block. Although its five phase point estimates were positive,
phases are supporting diagnostics and do not override the nonsignificant locked
primary test or the bootstrap interval crossing zero.

The first valid run contains 24,442 aligned predictions across 1,260 OOS dates
per task. The protocol hash was reproduced independently, all 110 training-audit
rows passed `target_end_date < test_start`, and pooled fits contained zero
confirmation-asset rows. The curated evidence is stored in
`examples/m14_snapshot/`, with predictions compressed as `predictions.csv.gz`.

The completed study was executed with:

```bash
python3 scope_confirm.py --output-base reports
```

The run at `20260907T215310_locked-scope-confirmation` is the canonical first
valid confirmation and should not be rerun for model selection. M14 confirms a
forecasting-loss training-scope effect for Ridge only. It does not establish
economic profitability, extend the conclusion to other horizons or model
families, implement a router, or justify an automatic scope recommendation.

## Milestone 15 — Publication-ready consolidation

M15 is a release-engineering milestone, not another research comparison. It
freezes the validated M14 state in one self-contained source package and adds:

- synchronized backend and front-end version `0.15.0`;
- the complete M1–M14 source tree and test suite;
- curated M9, M13 and canonical M14 evidence under `examples/`;
- a GitHub Actions workflow for Python tests and the production front-end build;
- an MIT license and a concise changelog;
- a release-contract test guarding version alignment and the canonical M14
  protocol/results;
- explicit separation between source-controlled evidence and regenerable model
  binaries or raw experiment output.

The eight pretrained M12 `.joblib` files are intentionally not committed. They
are generated artifacts tied to a data cutoff, are ignored by Git, and can be
rebuilt with `global_train.py`. Without them, Local inference and all bundled
research endpoints still work; Global and Compare correctly report themselves
as unavailable. Copy an existing complete artifact set into `artifacts/global/`
or train it again to enable those scopes.

### Release validation

Run the same checks used by continuous integration:

```bash
pytest -q
python3 frontend/validate_source.py
cd frontend
npm ci
npm run build
```

The M14 confirmation remains immutable inside this release. M15 must not be
interpreted as a second confirmation run or as evidence beyond the locked h=5
same-family scope comparison.

## License

Released under the MIT License. See `LICENSE`.
