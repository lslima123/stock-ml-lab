# Methodology

This document defines the scientific contract used by Stock ML Lab. It explains
how targets, time splits, model selection, statistical inference and economic
evaluation are handled. For the project overview and operating instructions, see
the [README](../README.md). For the complete milestone record, see
[RESEARCH_HISTORY.md](RESEARCH_HISTORY.md).

## 1. Scope and interpretation

Stock ML Lab is a research platform for comparing forecasting methods under
realistic temporal validation. It is not an execution system, an investment
adviser or a claim that historical predictive performance will persist.

The research asks a narrower question:

> Given information available at the close of trading day $t$, how do specified
> models compare when forecasting a future return or its direction on untouched
> out-of-sample observations?

The answer is reported as an empirical comparison. Statistical significance,
forecasting-loss improvement and trading profitability are separate claims and
are never treated as interchangeable.

## 2. Information timing and targets

Let $P_t$ be the adjusted closing price on trading day $t$. For horizon $h$:

$$
r_t^{(h)} = \frac{P_{t+h}}{P_t} - 1,
\qquad h \in \{1,5,10,20\}.
$$

Regression uses $r_t^{(h)}$ directly. Classification uses:

$$
y_t^{(h)} = \mathbf{1}\!\left(r_t^{(h)} > 0\right).
$$

The feature timestamp remains $t$. Consequently, the label attached to a row at
$t$ is not known until $t+h$. Every panel row stores this maturity explicitly as
`target_end_date`.

At each test boundary, the admissible training set satisfies:

```text
target_end_date < test_start
```

The strict inequality excludes every label whose realization reaches into the
test period. For single-series experiments the corresponding split contract is
`gap=h`.

## 3. Market data and supervised alignment

Daily OHLCV data are downloaded through yfinance with adjusted prices. Before
feature construction, the loader normalizes column names, orders the time index,
checks duplicate timestamps and validates positive close prices and admissible
volume values.

Features are computed from contemporaneous and lagged observations only. The
legacy contract contains 11 variables:

- one-day return and three return lags;
- distance from 5-, 10- and 20-day moving averages;
- 5- and 20-day volatility;
- 14-day RSI;
- one-day volume change.

Research-only feature groups add momentum, drawdown, range, standardized volume,
benchmark-relative variables and regime descriptors. Missing and non-finite rows
created by rolling windows or forward targets are removed only after features and
targets have been aligned.

A dedicated look-ahead test changes future prices and verifies that features at
earlier dates remain unchanged.

## 4. Temporal evaluation

Random train/test shuffling is prohibited. The project uses expanding temporal
windows so every training observation precedes the associated OOS block.

### Single-asset evaluation

For local models, `TimeSeriesSplit` produces ordered folds. The default daily
benchmark design uses five outer splits, 252 observations per test fold and
`gap=1`. Multi-horizon experiments replace that gap with the target horizon.

All learned preprocessing is part of the estimator pipeline. Scalers and other
transformations are fitted on the current training fold and applied to its test
fold; they are never fitted on the complete dataset.

### Panel evaluation

Global models use a calendar-aligned panel with these identifying fields:

```text
date, ticker, target_end_date, features, target
```

Calendar walk-forward folds are defined from unique market dates. Training rows
from every asset are admitted only when their target has matured before the first
test date. The same calendar test block is then used for all available assets.

Ticker identity is metadata, not a model feature. This makes the pooled model a
shared feature-to-target mapping rather than a collection of ticker-specific
intercepts.

### Unseen-asset evaluation

The M12 holdout evaluation removes the target asset from pooled training
entirely. Other assets may contribute only matured labels preceding the held-out
test window. This measures transfer to a series unseen during fitting.

## 5. Fixed models, tuning and external OOS data

Three evaluation roles are kept distinct:

| Role | Purpose | May select hyperparameters? |
| --- | --- | --- |
| Fixed-model comparison | Evaluate a prespecified estimator | No |
| Inner validation | Select candidate hyperparameters or a trading threshold | Yes |
| Outer/external OOS | Estimate final comparative performance | No |

When hyperparameters are tuned, the outer training block is split again with an
inner temporal cross-validation loop. Optuna minimizes inner OOF RMSE for
regression; classification candidates are selected by inner OOF Log Loss. The
winning parameters are refitted on the full outer-training block and evaluated
once on the outer test block.

Outer predictions remain untouched by tuning and are emitted once per timestamp.
Any choice made after viewing those predictions is labeled exploratory or post
hoc and requires a new independent confirmation sample.

## 6. Local and pooled global models

The scope comparison holds model family and features fixed:

| Task | Local | Global | Primary loss |
| --- | --- | --- | --- |
| Regression | Ridge | Ridge | RMSE / squared loss inference |
| Classification | Logistic | Logistic | Log Loss |

The Local model fits only observations from the target ticker. The Global model
fits the pooled training universe. Both receive the same 11 legacy features and
are scored on identical OOS timestamps. Comparisons such as Local XGBoost versus
Global Ridge are excluded because they confound model family with training scope.

The API follows a separate inference contract:

- Local estimators are fitted on demand for the requested ticker;
- Global estimators are loaded from pretrained artifacts;
- Compare returns the same-family Local and Global predictions side by side;
- no global retraining or hyperparameter search occurs inside an HTTP request.

## 7. Classical forecasting models

ARIMA, AutoARIMA and VAR are evaluated with rolling one-step-ahead forecasts.
Parameters are estimated from the external training block. During the test block,
the forecasting state may be updated with newly realized observations, but model
parameters are not re-estimated using test data.

This distinction permits operational one-step forecasting without converting the
test set into an expanding refit sample.

## 8. Metrics

### Regression

- RMSE (primary for the Local/Global benchmark);
- MAE;
- directional accuracy and coverage where applicable.

### Classification

- Log Loss (primary for the Local/Global benchmark);
- Brier score;
- ROC-AUC;
- accuracy, balanced accuracy, precision, recall and F1;
- calibration error and reliability bins;
- predicted-positive rate and probability margins.

AUC assesses ranking across thresholds. Balanced accuracy evaluates decisions at
a particular threshold. A high AUC and near-random balanced accuracy can therefore
coexist without contradiction.

For lower-is-better metrics, relative Global improvement over Local is:

$$
100 \times \frac{L_{\text{local}} - L_{\text{global}}}{L_{\text{local}}}.
$$

A positive value favors Global. It is a percentage reduction in forecasting
loss—not a percentage-point predicted return or portfolio return.

## 9. Dependence-aware inference

For $h>1$, adjacent forward returns overlap. Setting `gap=h` prevents a training
label from crossing into a test window, but it does not make consecutive OOS
targets independent.

The project therefore uses complementary checks:

- HAC/Newey–West loss-differential inference with at least `h-1` lags;
- circular block bootstrap with blocks of at least `h` dates;
- $h$ non-overlapping phases as stability diagnostics;
- fold, asset and regional summaries where relevant.

In pooled comparisons, losses are first clustered by calendar date. This avoids
treating simultaneous asset observations as independent evidence. The primary HAC
test is applied to the date-level mean loss advantage.

Regression comparisons use squared-error loss for the primary RMSE question and
may also report Diebold–Mariano tests under MSE or MAE. Directional work includes
the Pesaran–Timmermann test. Cross-asset win counts use an exact one-sided sign
test.

When several hypotheses are evaluated together, Benjamini–Hochberg controls the
false discovery rate. Unadjusted $p$-values are not promoted as confirmed results
when the stated multiplicity family requires adjusted $q$-values.

## 10. Economic evaluation

Forecast significance is not assumed to imply tradability. Long/short and
long/flat strategies are evaluated with position-change transaction costs:

$$
R_{t+1}^{\text{net}}
= s_t r_{t+1} - c\lvert s_t-s_{t-1}\rvert.
$$

Reported diagnostics include total and annualized return, volatility, Sharpe,
Sortino, maximum drawdown, exposure, turnover, rebalances, cost drag and active
hit rate. Threshold and cost sensitivity are reported explicitly. When a
threshold is selected, that selection occurs inside nested training data rather
than on the final OOS sequence.

Buy & Hold is included as a reference, not as a target that the model is assumed
to beat.

## 11. Discovery, confirmation and the M14 decision rule

The evidence hierarchy is explicit:

1. **Exploration** may identify a candidate configuration.
2. **Locking** freezes horizon, features, models, assets, metrics and inference.
3. **Confirmation** evaluates the frozen hypothesis on untouched data.
4. **Post hoc diagnostics** can explain a result but cannot upgrade its status.

M8 found attractive 10- and 20-day patterns for PETR4. M8.1 then excluded PETR4
and tested four frozen candidates on nine other assets; none met the primary
HAC/FDR criterion.

M13 compared Local and Global scopes across the original ten-asset universe. Its
date-clustered analyses were labeled post hoc because they followed inspection of
the benchmark.

M14 locked one configuration before downloading its confirmation universe:

- horizon 5;
- legacy features;
- Ridge vs Ridge and Logistic vs Logistic;
- five calendar walk-forward folds of 252 dates;
- original ten assets for pooled training;
- ten disjoint assets for confirmation;
- HAC lags 4 and 10,000 circular bootstrap replications with five-date blocks;
- BH FDR across the two primary task hypotheses.

A task was confirmed only if both conditions held:

```text
one-sided date-clustered HAC BH q < 0.05
AND
95% circular date-block-bootstrap improvement CI lower bound > 0
```

Under that locked rule, h=5 Global Ridge was confirmed against Local Ridge;
Global Logistic was not. The exact protocol is stored in
[`configs/m14_locked_scope.json`](../configs/m14_locked_scope.json), and the
canonical evidence is under [`examples/m14_snapshot/`](../examples/m14_snapshot/).

The provider failure for `ELET3.SA` occurred during preflight, before any model
was evaluated. The identifier was amended to `AXIA3.SA`, the official ticker for
the same issuer, and the amendment is recorded in the locked configuration and
canonical report.

## 12. Reporting and reproducibility

Research runs write fold-, asset- and prediction-level outputs plus summaries,
manifests and graceful-failure tables. Generated run directories are ignored by
Git; selected immutable snapshots are copied into `examples/` for review and API
serving.

The M15 release contract checks that:

- backend and frontend versions agree;
- the frozen M14 protocol hash remains unchanged;
- training and confirmation universes remain disjoint;
- the canonical regression and classification decisions are preserved.

The frontend consumes typed API responses and does not reproduce research logic.
This keeps scientific rules in one backend implementation and makes the UI a
presentation layer rather than an independent calculation path.

## 13. Interpretation policy

Results are described with the narrowest claim supported by their design:

- feature importance is not predictive power;
- one strong asset is not cross-asset robustness;
- an unadjusted significant test is not multiplicity-controlled confirmation;
- lower forecasting loss is not trading profitability;
- a confirmed specification does not validate untested horizons, features or
  algorithms;
- negative and inconclusive results remain part of the project record.

The intended portfolio narrative is therefore:

> A platform for comparison and evaluation of financial time-series forecasting
> models under rigorous temporal validation.
