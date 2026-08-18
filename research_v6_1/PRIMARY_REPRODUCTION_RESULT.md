# Research V6.1 — primary-model clean reproduction

## Decision

The user-selected V6.1 portfolio reproduces exactly when its selected model
components are refitted from scratch under the frozen code, parameters, seeds,
features, and chronology. This rules out accidental software or solver
non-reproducibility under the published environment. It does **not** establish
that the 20-month historical result is free of model-selection or sample luck.

The reproduced policy is the return-ranking selectors plus the FF5+momentum
alpha sign gate and the frozen cost/risk optimizer at the 10% forecast-risk
ceiling. The evidence window remains January 2024 through August 2025.

## What was rebuilt

The clean selector run reused only the authenticated V6 feature matrix. It did
not reuse a fitted model, selector prediction, or selector checkpoint. It
refitted the five components actually consumed by the portfolio:

- CAM-free continuous MLP for CAPM alpha, retained because the common portfolio
  input pipeline requires all four target files even though this policy does
  not use the CAPM gate;
- CAM-free pairwise MLP for FF5+momentum alpha;
- CAM-free continuous MLP for the long-return rank;
- correlation-reduced pairwise XGBoost for the short-return rank; and
- correlation-reduced pairwise MLP for the short-return rank.

The direct point-in-time CAM rank was rematerialized from the authenticated
feature matrix. The original frozen objective choices were retained; discarded
research-grid candidates were not rerun because they do not enter the selected
portfolio and cannot affect computational reproducibility of that portfolio.

Clean selector root:
`artifacts/research_v6_1/primary_selector_clean_reproduction/481e306553c2-16046ee58521-d5dadd9cd9c3`

## Exact reproduction checks

- CAPM signal: maximum absolute difference 0; mean and minimum monthly rank
  correlation 1.000.
- FF5+momentum signal: maximum absolute difference 0; mean and minimum monthly
  rank correlation 1.000.
- Long-return signal: maximum absolute difference 0; mean and minimum monthly
  rank correlation 1.000.
- Short-return signal: maximum absolute difference 0; mean and minimum monthly
  rank correlation 1.000.
- The new and original portfolio monthly-result values are exactly equal after
  aligning their identical column sets. A checkpoint-only resume may serialize
  those columns in a different order; that storage-order difference has no
  economic content.
- The new and original aggregate summary rows are exactly equal.
- All 20 monthly position files are byte-for-byte identical.

Clean portfolio root:
`artifacts/research_v6_1/portfolio_step_02_clean_reproduction/56e3e0612cc5-8e196ed77924-b2bf0629f719`

The exact reproduced statistics are:

| Metric | Reproduced value |
|---|---:|
| Annualized gross excess return | 10.94% |
| Annualized net excess return | 9.94% |
| Annualized net excess return, stressed costs | 8.33% |
| Annualized realized volatility | 5.77% |
| Net Sharpe | 1.723 |
| Stressed-cost Sharpe | 1.449 |
| Maximum drawdown | -1.96% |
| Annualized primary cost | 0.99% |
| Mean gross exposure | 0.972 |
| Mean net exposure | 0.277 |
| Mean long holdings | 76.7 |
| Mean short holdings | 69.4 |

The repeated `optimal_inaccurate` solver notices are the same practical
numerical condition already audited in Step 02. They did not create a new
solution: every monthly position file is identical.

## Prespecified seed sensitivity

The official historical-test selector averages the three frozen fit seeds 17,
42, and 101. As a diagnostic—not a model-selection exercise—the same portfolio
was also run using each seed's individual test-period rank while leaving the
validation history and every portfolio rule unchanged.

| Test-fit signal | Net return | Volatility | Sharpe | Max drawdown |
|---|---:|---:|---:|---:|
| Individual seed 17 | 9.70% | 5.14% | 1.886 | -1.42% |
| Individual seed 42 | 3.87% | 5.41% | 0.716 | -3.21% |
| Individual seed 101 | 7.50% | 5.39% | 1.392 | -1.64% |
| **Frozen three-seed average** | **9.94%** | **5.77%** | **1.723** | **-1.96%** |

This is meaningful seed sensitivity. A single neural-network fit would not be
reliable enough: the individual-seed Sharpe range is 0.716--1.886. The frozen
three-seed averaging rule removes dependence on any one seed and itself
reproduces exactly, but three seeds cannot establish stability across every
possible initialization. These rows are diagnostics only; the best individual
seed must not replace the frozen average.

## Historical-sample uncertainty

The fixed portfolio still has only 20 post-diagnostic monthly observations.
Simple sensitivity checks show:

- first ten months Sharpe: 1.96;
- last ten months Sharpe: 1.41;
- removing the best month: 7.32% annualized net excess return and 1.53 Sharpe;
- removing the two best months: 5.83% annualized net excess return and 1.28
  Sharpe; and
- leave-one-month-out Sharpe range: 1.53--2.18.

A circular three-month block bootstrap gives a descriptive 95% Sharpe interval
of approximately 0.41--3.12 and an 86.9% resampled fraction above 1.0. This
interval is intentionally labeled descriptive: it reuses the same short,
post-diagnostic sample and does not correct for the earlier model-development
search. It cannot substitute for untouched forward evidence.

## Interpretation

The answer to the narrow reproducibility question is **yes**: rerunning the
frozen selected model from scratch produces the same signals, holdings, monthly
returns, and 1.723 Sharpe. None of the published result is attributable to a
one-off software execution under the fixed seed ensemble.

The honest remaining uncertainty is economic rather than computational.
Individual fits are seed-sensitive, and 20 favorable historical months are too
few to determine how much of the 1.723 Sharpe is sample luck. The portfolio
therefore remains a frozen historical-development candidate. The decisive test
is unchanged deployment on newly available months without modifying the
selectors, seed averaging, alpha gate, calibration, optimizer, or costs.
