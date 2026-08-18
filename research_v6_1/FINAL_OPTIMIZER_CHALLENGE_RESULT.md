# Research V6.1 — final optimizer challenge

## Decision

No bounded optimizer alternative earned the right to challenge the frozen
portfolio in the January 2024--August 2025 historical test. The unchanged
control had the highest net Sharpe in the separate February 2022--December 2023
validation window. The final decision is therefore:

`RETAIN_FROZEN_1P723_CONTROL_AND_CLOSE_OPTIMIZER_RESEARCH`

The retained thesis portfolio remains the return-ranking selectors plus the
FF5+momentum alpha sign gate and the frozen expected-net-return optimizer. The
study found no statistically superior portfolio maker with a defensible
economic mechanism. Portfolio-optimizer searching should stop; the next
research evidence should be untouched forward performance, not another
post-outcome rescue.

Corrected run root:
`artifacts/research_v6_1/final_optimizer_challenge/72e9dc82f89e-67cac227c15c-a8f3e5acf95a`

## What was tested

The selectors, features, ranks, FF5+momentum gate, candidate tails, short
floor, risk limits, cost model, and monthly schedule were held fixed. Only the
optimizer objective was varied. The bounded set contained:

- the unchanged control;
- three levels of shrinkage in the cross-sectional expected-return dispersion;
- three mean-variance penalties;
- three concentration penalties; and
- two conservative transaction-cost multipliers.

All 12 profiles were run over February 2022--December 2023. Exactly one profile
could be selected by highest validation net Sharpe. Only that profile could be
run over January 2024--August 2025. This prevents choosing an optimizer after
seeing the same 20 portfolio outcomes used to report the thesis result.

## Validation result

| Profile | Economic intervention | Net Sharpe | Net excess return | Realized volatility |
|---|---|---:|---:|---:|
| **Control** | unchanged optimizer | **0.497** | **2.63%** | **5.29%** |
| Concentration 0.10 | moderate weight dispersion penalty | 0.435 | 2.18% | 5.01% |
| Mean shrink 0.25 | retain 25% of within-side mean dispersion | 0.404 | 2.13% | 5.26% |
| Cost buffer 2.0 | double costs in the objective | 0.398 | 2.01% | 5.04% |
| Equal side mean | remove within-side mean dispersion | 0.393 | 2.16% | 5.49% |
| Concentration 0.05 | light weight dispersion penalty | 0.387 | 2.00% | 5.17% |
| Mean shrink 0.50 | retain 50% of within-side mean dispersion | 0.364 | 1.93% | 5.30% |
| Mean-variance 0.5 | light explicit variance penalty | 0.352 | 1.62% | 4.61% |
| Cost buffer 1.5 | 1.5 times costs in the objective | 0.337 | 1.74% | 5.17% |
| Concentration 0.25 | stronger weight dispersion penalty | 0.332 | 1.61% | 4.83% |
| Mean-variance 1.0 | moderate explicit variance penalty | 0.165 | 0.73% | 4.43% |
| Mean-variance 3.0 | strong explicit variance penalty | 0.003 | 0.01% | 4.40% |

The alternatives often reduced realized volatility, but they reduced payoff by
more. That is why their Sharpe ratios deteriorated. The result does not imply
that variance or cost awareness is unnecessary: the frozen optimizer already
has a volatility ceiling, factor and industry controls, turnover and capacity
limits, and explicit costs. Adding another objective penalty mostly duplicated
those controls and distorted the selector's cross-sectional payoff ordering.

## Exact control checksum and retained statistics

Because the control won validation, the 20-month historical-test execution was
an exact reproduction check. Its monthly returns and aggregate Sharpe are
identical to the frozen portfolio. The paired differences are zero in every
month.

| Metric | Retained result |
|---|---:|
| Annualized gross excess return | 10.94% |
| Annualized net excess return | 9.94% |
| Annualized stressed-cost net excess return | 8.33% |
| Annualized realized volatility | 5.77% |
| Net Sharpe | 1.723 |
| Stressed-cost Sharpe | 1.449 |
| HAC t-statistic of mean net excess return | 2.641 |
| Maximum drawdown | -1.96% |
| Annualized primary cost | 0.99% |
| Net excess return without the best two months | 5.83% |
| Mean gross exposure | 0.972 |
| Mean net exposure | 0.277 |
| Mean monthly one-way turnover | 0.500 |
| Mean long holdings | 76.7 |
| Mean short holdings | 69.4 |

The 2.641 statistic supports rejection of zero mean net excess return within
this historical sample. It is not a test showing superiority over the
no-short-floor portfolio or any newly selected challenger. No alternative
survived validation, so the study makes no claim of pairwise optimizer
superiority.

## Why apparently small changes could look large

The optimizer frequently sits on turnover, beta, net-exposure, side-gross, and
individual-position boundaries. Small objective changes can therefore exchange
whole 1% positions rather than move every holding smoothly. Relative to the
control, the alternatives changed an average of 6.5% to 36.8% of portfolio
weight on a one-way basis per validation month. Depending on the profile, an
average of roughly 3 to 67 names per month moved by more than 50 basis points.
Those are economically meaningful portfolio changes, even when the source is
a single regularization coefficient.

An initial implementation exposed a separate numerical problem: algebraically
zero penalty terms altered the conic solver's expression graph. The weights
were nearly identical through March 2024, then the stateful path selected a
different constrained corner in April and compounded thereafter. That run was
rejected. The control was changed to execute the original solver expression
and original monthly path directly. January through April then had zero weight
difference, and the completed 20-month control reproduced the frozen 1.723
Sharpe exactly. The invalid preliminary run is not research evidence.

## Constraint interpretation and audit

The generic first-pass audit compared every holding with the 1% candidate
entry cap and falsely labeled one inherited short as a breach. In February
2024, PERMNO 93427 entered the rebalance at -1.1469% after return drift and was
reduced to -1.1466%. Under the frozen rules, a holding that is no longer a
candidate is reduction-only: it may temporarily remain beyond the entry cap,
but it cannot increase in magnitude or change sign.

The corrected holdings-level audit checks candidate caps and inherited
reduction-only positions separately. Its maximum positive residual is
2.22e-08 against a practical tolerance of 5e-07, so all economically relevant
constraints pass. The bound audit is stored as
`constraint_audit_addendum.json` under the corrected run root.

## Thesis interpretation

The final result is economically interesting, reproducible, cost-aware, and
statistically positive within the 20-month historical sample. The optimizer
challenge adds useful negative evidence: straightforward mean uncertainty,
variance, concentration, and cost-buffer modifications did not improve the
portfolio on a separate validation window.

It remains post-diagnostic historical development evidence, not proven live
production quality. V6.1 was designed after earlier outcomes had been
inspected, the historical test contains only 20 months, and individual neural
network fits remain seed-sensitive. September 2025--August 2026 stays
quarantined. The frozen policy's decisive next test is untouched confirmation
beginning no earlier than September 2026, without changing the selector,
ensemble, gate, calibration, optimizer, or cost assumptions.
