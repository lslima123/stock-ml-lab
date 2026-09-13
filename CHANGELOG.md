# Changelog

All notable project milestones are summarized here. The project follows a
single research lineage; milestone numbers describe completed capabilities and
locked studies rather than claims of profitable forecasting.

## 0.15.0 — 2026-09-08

### Publication-ready consolidation

- Consolidated the complete M1–M14 source tree into one standalone release.
- Synchronized the Python API and React front-end at version 0.15.0.
- Added the canonical M14 evidence and complete compressed OOS predictions.
- Added release-contract tests for version alignment and M14 protocol integrity.
- Added GitHub Actions checks for the Python suite and production front-end build.
- Documented the policy for regenerable global-model binaries and raw reports.

No new model selection, hypothesis test or economic-performance claim was
introduced in M15.

## 0.14.0 — 2026-09-08

### Locked independent scope confirmation

- Confirmed Global Ridge over Local Ridge at h=5 on ten assets disjoint from the
  global training universe: +1.7166% OOS RMSE improvement, BH-adjusted HAC
  q=0.000132, block-bootstrap 95% CI [+0.9096%, +2.5647%], 9/10 asset wins.
- Did not confirm Global Logistic over Local Logistic: +0.4098% Log Loss
  improvement, q=0.157742, bootstrap interval crossing zero.
- Added the scope-confirmation API evidence endpoint and UI evidence card.

## 0.13.0 — 2026-09-02

### Same-family local-vs-global benchmark

- Added controlled Ridge-vs-Ridge and Logistic-vs-Logistic scope comparisons.
- Added identical OOS timestamps, target-maturity purge, HAC inference, BH FDR,
  sign tests and post hoc fold/phase/regional diagnostics.
- Added Local, Global and Compare API/UI scopes.

## 0.12.0 and earlier

- Added pooled global Ridge and Logistic models, temporal and unseen-asset
  evaluation, React/Vite UI, FastAPI, reporting, multi-horizon research,
  classification, cross-asset robustness, statistical/economic validation,
  nested tuning, classical models and the leakage-tested dataset engine.
