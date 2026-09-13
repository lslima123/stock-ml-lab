# M14 — Locked Independent Scope Confirmation

## Evidence boundary

This is a locked independent confirmation. Its two primary hypotheses and decision rule were fixed before downloading the confirmation assets.

## Scope comparison

| task | horizon | assets | asset_win_count | fold_win_count | fold_count | phase_win_count | phase_count | global_improvement_vs_local_pct | bootstrap_improvement_ci_low_pct | bootstrap_improvement_ci_high_pct | date_hac_pvalue_global_better | primary_hac_qvalue | confirmed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classification | 5 | 10 | 7 | 2 | 5 | 5 | 5 | 0.4098 | -0.3876 | 1.2215 | 0.1577 | 0.1577 | False |
| regression | 5 | 10 | 9 | 5 | 5 | 5 | 5 | 1.7166 | 0.9096 | 2.5647 | 0.0001 | 0.0001 | True |

## Date-clustered inference

Rows sharing an OOS date are treated as a cluster. HAC lags are at least `h-1`, and the
circular block bootstrap resamples date clusters in blocks of at least `h` dates.

| task | horizon | primary_metric | rows | assets | dates | global_improvement_vs_local_pct | date_hac_pvalue_global_better | primary_hac_qvalue | bootstrap_improvement_ci_low_pct | bootstrap_improvement_ci_high_pct | confirmed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| classification | 5 | log_loss | 12221 | 10 | 1260 | 0.4098 | 0.1577 | 0.1577 | -0.3876 | 1.2215 | False |
| regression | 5 | rmse | 12221 | 10 | 1260 | 1.7166 | 0.0001 | 0.0001 | 0.9096 | 2.5647 | True |

## Interpretation

Positive improvement means lower global-model loss than the same-family local model. The
comparison identifies a training-scope effect; it is not a trading-performance claim.
