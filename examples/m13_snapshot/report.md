# M13 — Local vs Global Benchmark

## Evidence boundary

These diagnostics are POST HOC robustness analyses of the completed M13 benchmark. They strengthen interpretation but do not convert M13 into a new confirmation set.

## Scope comparison

| task | horizon | assets | asset_win_count | fold_win_count | fold_count | phase_win_count | phase_count | global_improvement_vs_local_pct | bootstrap_improvement_ci_low_pct | bootstrap_improvement_ci_high_pct | date_hac_pvalue_global_better |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classification | 1 | 10 | 8 | 4 | 5 | 1 | 1 | 0.3932 | 0.1774 | 0.6094 | 0.0003 |
| classification | 5 | 10 | 10 | 5 | 5 | 5 | 5 | 0.8228 | 0.3313 | 1.3198 | 0.0008 |
| classification | 10 | 10 | 8 | 3 | 5 | 10 | 10 | 1.0841 | 0.1014 | 2.0752 | 0.0197 |
| classification | 20 | 10 | 9 | 5 | 5 | 20 | 20 | 2.6598 | 0.7535 | 4.5239 | 0.0046 |
| regression | 1 | 10 | 8 | 4 | 5 | 1 | 1 | 0.5359 | 0.2490 | 0.8299 | 0.0001 |
| regression | 5 | 10 | 9 | 5 | 5 | 5 | 5 | 0.8931 | 0.3819 | 1.4113 | 0.0006 |
| regression | 10 | 10 | 9 | 4 | 5 | 10 | 10 | 0.8051 | 0.0724 | 1.5416 | 0.0169 |
| regression | 20 | 10 | 9 | 4 | 5 | 20 | 20 | 1.4373 | 0.3455 | 2.5089 | 0.0073 |

## Date-clustered inference

Rows sharing an OOS date are treated as a cluster. HAC lags are at least `h-1`, and the
circular block bootstrap resamples date clusters in blocks of at least `h` dates.

| task | horizon | primary_metric | rows | assets | dates | global_improvement_vs_local_pct | date_hac_pvalue_global_better | bootstrap_improvement_ci_low_pct | bootstrap_improvement_ci_high_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classification | 1 | log_loss | 12225 | 10 | 1260 | 0.3932 | 0.0003 | 0.1774 | 0.6094 |
| classification | 5 | log_loss | 12220 | 10 | 1260 | 0.8228 | 0.0008 | 0.3313 | 1.3198 |
| classification | 10 | log_loss | 12220 | 10 | 1260 | 1.0841 | 0.0197 | 0.1014 | 2.0752 |
| classification | 20 | log_loss | 12220 | 10 | 1260 | 2.6598 | 0.0046 | 0.7535 | 4.5239 |
| regression | 1 | rmse | 12225 | 10 | 1260 | 0.5359 | 0.0001 | 0.2490 | 0.8299 |
| regression | 5 | rmse | 12220 | 10 | 1260 | 0.8931 | 0.0006 | 0.3819 | 1.4113 |
| regression | 10 | rmse | 12220 | 10 | 1260 | 0.8051 | 0.0169 | 0.0724 | 1.5416 |
| regression | 20 | rmse | 12220 | 10 | 1260 | 1.4373 | 0.0073 | 0.3455 | 2.5089 |

## Interpretation

Positive improvement means lower global-model loss than the same-family local model. The
comparison identifies a training-scope effect; it is not a trading-performance claim.
