# Research V6.2 — direct portfolio learning result

## Decision

The experiment is complete, but direct portfolio learning does **not** replace
the V6.1 rank-plus-optimizer pipeline.

The full-feature primary direct policy earned 1.17% annualized net excess
return with a 0.888 Sharpe. Applying the frozen FF5+momentum sign gate to the
same fitted policy raised net excess return to 1.43% and Sharpe to 0.934. The
best diagnostic result, CAM-free plus FF5, reached a 1.004 primary-cost Sharpe
and 0.984 stressed-cost Sharpe. All are materially below the frozen V6.1 Step
02 joint-alpha primary at 10.39% net excess return and a 1.671 Sharpe, and the
FF5-only descriptive control at a 1.723 Sharpe.

## Final lineage

- Contract SHA-256: `99cf7f2098b718a81ac4e3d562177837159b49c1c7ee26b24b60c5089088d358`
- Input identity: `f0bf8995d641742ae4b68e8f424ad85d67f7fe0d4c4d9b163e31119c2d129399`
- Code identity: `872cb86dba162f463ad2d2fc720034b6668435cdd4638a7ab5db24fb10e7eee4`
- Result root: `artifacts/research_v6_2/direct_portfolio_step_01/99cf7f2098b7-f0bf8995d641-872cb86dba16`
- Completion: 12 validation candidates, six test variants, and 120/120 test
  variant-months
- Reused V6 or V6.1 model weights: none

The earlier `c08d2079582e-f0bf8995d641-360172133323` run is preliminary
engineering history. A pre-publication comparison audit found that its gated
and ungated variants did not share the same fitted-policy seed. No conclusion
was frozen from it. The corrected lineage trains both variants identically and
applies the FF5 gate only to validation/test formation. Within every feature
view and annual refit, paired model files are byte-identical.

## Historical-test results

January 2024 through August 2025 contains 20 monthly observations. Returns,
volatility, and costs are annualized.

| Feature view and policy | Net excess return | Volatility | Net Sharpe | Stress Sharpe | Max drawdown | Primary cost |
|---|---:|---:|---:|---:|---:|---:|
| CAM-free, direct + FF5 gate | 1.61% | 1.61% | 1.004 | 0.984 | -0.53% | 0.05% |
| Correlation-reduced, direct + FF5 gate | 1.51% | 1.53% | 0.986 | 0.962 | -0.40% | 0.05% |
| **Full, direct + FF5 gate** | **1.43%** | **1.53%** | **0.934** | **0.913** | **-0.39%** | **0.05%** |
| **Full, direct only** | **1.17%** | **1.32%** | **0.888** | **0.879** | **-0.17%** | **0.02%** |
| CAM-free, direct only | 1.08% | 1.36% | 0.788 | 0.779 | -0.31% | 0.02% |
| Correlation-reduced, direct only | 0.94% | 1.31% | 0.723 | 0.712 | -0.17% | 0.02% |

The full view is the prespecified primary information set. CAM-free and
correlation-reduced are diagnostic challengers, so the CAM-free row cannot be
promoted merely because it has the highest observed direct-policy Sharpe.

## What the model learned

- The return-focused profile won validation for all six paths. The balanced
  profile, which imposed more forecast-risk penalty, was consistently worse.
- Every historical-test portfolio was long-only. The learned short-gross head
  remained below the frozen activation threshold, so short gross and short
  return contribution were zero. This is a valid outcome because shorts were
  permitted, not required.
- The model requested approximately 92%--100% mean long gross before repair.
  The hard 0.10 market-beta limit was usually the dominant scaling constraint,
  leaving only 10.7%--12.2% mean implemented gross in the direct paths and
  11.9%--12.2% in the gated paths.
- The FF5 gate modestly improved all three views. On the full primary view it
  added about 0.26 percentage points of annualized net excess return and 0.046
  Sharpe. This is useful corroboration of the alpha overlay, not a replacement
  result.
- Removing CAM helped only the gated policy in the observed test and hurt the
  ungated direct policy. Correlation reduction likewise did not reliably
  improve the direct model. There is no evidence that a single manual feature
  deletion rule dominates across policies.

## Engineering, cost, and risk audit

- The run manifest is `COMPLETE`; no error artifact exists.
- Gross contribution and primary/stress net-return accounting tie exactly at
  machine precision across all 120 rows.
- All six variants contain all 20 test months and 73,238 saved position rows.
- Maximum one-way turnover is 9.05%, below the 35% limit. Maximum realized
  one-way trade participation is 1.00% of ADV, below the 10% limit.
- Maximum position weight is 0.125%, below 1%. Maximum gross is 15.56%; net
  remains between 9.71% and 15.56%.
- Maximum forecast volatility is 2.82%, below 10%. Maximum absolute market
  beta is 0.10000000000000002; the residual is immaterial numerical dust at
  the 0.10 bound. All other factor exposures are below 0.056.
- Maximum SIC2 net and gross exposures are 3.26%, below their 5% and 30%
  limits.

## Honest interpretation

Direct weight learning answered the conceptual question, but did not solve the
economic transfer problem. It learned that most of the available capital
should remain in cash once the frozen beta and risk limits were imposed. That
produced small drawdowns and low costs, but also very low absolute return. The
model's refusal to short is further evidence that the historical information
set did not support a profitable standalone short sleeve under this objective.

The research conclusion is therefore conservative:

1. retain the frozen V6.1 Step 02 joint-alpha optimizer as the confirmation
   candidate;
2. retain the V6.1 FF5-only optimizer as a descriptive control, not an
   outcome-selected replacement;
3. close Step 03 and V6.2 as useful mechanism/robustness evidence;
4. do not tune another historical portfolio on the same 20 months;
5. evaluate the completely frozen candidate on untouched observations no
   earlier than September 2026.

This is historical post-diagnostic evidence, not proof of production
performance or a live-investment recommendation.
