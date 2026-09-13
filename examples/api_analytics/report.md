# Stock ML Lab — M9 Research Report

## Executive findings

- Discovery: PETR4.SA 20d Logistic reached AUC 0.628 and +2.24% Log Loss improvement versus the prior.
- Discovery: PETR4.SA 20d Ridge reached +4.47% RMSE improvement versus Zero Return.
- Locked confirmation: 20d Logistic won on 1/9 assets; median improvement -0.34%.
- Locked confirmation: 20d Ridge won on 4/9 assets; median improvement -0.11%.
- Confirmatory inference: 0 locked asset-level result(s) survived primary HAC/FDR significance at 5%.
- Bootstrap robustness: 0 asset/candidate result(s) had a fully positive 95% primary-improvement interval.

## Interpretation

The reporting layer distinguishes exploratory discovery from locked confirmation. Positive discovery metrics are not promoted as robust predictive edge unless they survive the predeclared confirmation universe, dependence-aware inference, bootstrap uncertainty and non-overlapping phase diagnostics.

## Figures

- `discovery_vs_confirmation.png`
- `m8_logistic_auc_by_horizon.png`
- `m8_ridge_rmse_by_horizon.png`
- `m8_feature_horizon_auc.png`
- `m81_classification_10d_assets.png`
- `m81_classification_20d_assets.png`
- `m81_regression_10d_assets.png`
- `m81_regression_20d_assets.png`
- `m81_phase_win_rates.png`
- `m7_one_day_auc.png`

## Discovery table

| task           |   horizon | model                     | log_loss_improvement_vs_prior_pct   | roc_auc   | rmse_improvement_vs_zero_pct   | directional_accuracy   |
|:---------------|----------:|:--------------------------|:------------------------------------|:----------|:-------------------------------|:-----------------------|
| classification |         1 | Tuned Logistic Regression | +0.02%                              | 0.497     | N/A                            | N/A                    |
| classification |         5 | Tuned Logistic Regression | +0.44%                              | 0.558     | N/A                            | N/A                    |
| classification |        10 | Tuned Logistic Regression | +1.86%                              | 0.599     | N/A                            | N/A                    |
| classification |        20 | Tuned Logistic Regression | +2.24%                              | 0.628     | N/A                            | N/A                    |
| regression     |         1 | Tuned Ridge               | N/A                                 | N/A       | -1.29%                         | 50.48%                 |
| regression     |         5 | Tuned Ridge               | N/A                                 | N/A       | +0.74%                         | 55.32%                 |
| regression     |        10 | Tuned Ridge               | N/A                                 | N/A       | +1.81%                         | 58.89%                 |
| regression     |        20 | Tuned Ridge               | N/A                                 | N/A       | +4.47%                         | 64.92%                 |

## Locked confirmation summary

| task           |   horizon | model                     |   n_assets |   primary_win_count | primary_win_rate   |   primary_win_sign_test_p | mean_primary_improvement_pct   | median_primary_improvement_pct   |   primary_hac_fdr_sig_count |   primary_bootstrap_positive_count | mean_phase_win_rate   |
|:---------------|----------:|:--------------------------|-----------:|--------------------:|:-------------------|--------------------------:|:-------------------------------|:---------------------------------|----------------------------:|-----------------------------------:|:----------------------|
| classification |        10 | Tuned Logistic Regression |          9 |                   5 | 55.56%             |                    0.5    | -0.21%                         | +0.02%                           |                           0 |                                  0 | 44.44%                |
| classification |        20 | Tuned Logistic Regression |          9 |                   1 | 11.11%             |                    0.998  | -0.71%                         | -0.34%                           |                           0 |                                  0 | 24.44%                |
| regression     |        10 | Tuned Ridge               |          9 |                   3 | 33.33%             |                    0.9102 | -0.62%                         | -0.61%                           |                           0 |                                  0 | 37.78%                |
| regression     |        20 | Tuned Ridge               |          9 |                   4 | 44.44%             |                    0.7461 | -0.49%                         | -0.11%                           |                           0 |                                  0 | 50.00%                |
