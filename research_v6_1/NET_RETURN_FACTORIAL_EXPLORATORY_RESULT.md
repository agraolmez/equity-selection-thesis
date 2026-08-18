# Research V6.1 — post-diagnostic constraint-factorial curiosity study

## Decision

The user explicitly authorized one small curiosity study on 14 August 2026:
can the validation-selected 1.5% candidate-position portfolio earn more net
return or a higher Sharpe through modest, economically distinct constraint
relaxations, without changing any selector or information input?

The corrected result is:

`NO_REPLACEMENT_RETAIN_1P5_CONTROL`

A modest factor-limit relaxation produces a small return increase, and adding
slightly more liquidity capacity increases it marginally further. Neither
variant improves Sharpe, and neither paired return improvement is statistically
distinguishable from the 1.5% control over the 20-month historical test. The
existing 1.5% portfolio therefore remains the confirmation candidate.

## Frozen design

The genuine V6.1 return selectors, all 318 registered feature sources, fitted
rankings, FF5+momentum alpha sign gate, candidate tails, causal rank-to-payoff
calibration, cost model, turnover rule, gross/net rules, short floor, and risk
objective are unchanged. Only three binary portfolio limits vary around the
1.5% control:

- absolute factor exposure: 10.0% or 12.5%;
- SIC2 net/gross exposure: 5%/30% or 7.5%/40%;
- maximum one-way ADV participation: 10.0% or 12.5%.

This creates eight profiles rather than an open-ended search. All eight are
evaluated on February 2022--December 2023 validation data. Return and Sharpe
winners are selected separately within a practical 10% ADV tier and an
expanded 12.5% ADV tier. Only the four distinct selected profiles, including
the unchanged control, are evaluated on January 2024--August 2025. Historical-
test outcomes do not select a profile.

## Validation selection

| Profile | Change from 1.5% control | Annualized net excess | Net Sharpe |
|---|---|---:|---:|
| Control | none | 4.12% | 0.831 |
| Factor 12.5% | factor limits only | 4.66% | 0.889 |
| Industry 7.5%/40% | SIC2 limits only | 2.89% | 0.598 |
| ADV 12.5% | liquidity limit only | 3.52% | 0.695 |
| Factor + industry | factor and SIC2 limits | 4.64% | 0.941 |
| Factor + ADV | factor and liquidity limits | **4.89%** | **0.941** |
| Industry + ADV | SIC2 and liquidity limits | 3.26% | 0.674 |
| Factor + industry + ADV | all three limits | 4.07% | 0.828 |

Validation chooses factor-only for practical return, factor-plus-industry for
practical Sharpe, and factor-plus-ADV for both expanded objectives. The result
already shows why the bounded design matters: relaxing industry limits alone
is harmful, and broad simultaneous relaxation does not improve the control.

## Historical-test result

| Metric | 1.5% control | Factor 12.5% | Factor + industry | Factor + ADV |
|---|---:|---:|---:|---:|
| Annualized net excess return | 14.67% | 15.00% | 13.17% | **15.13%** |
| Annualized stressed net excess | 12.96% | 13.30% | 11.52% | **13.43%** |
| Total-return CAGR | 21.34% | 21.71% | 19.50% | **21.87%** |
| Realized volatility | **6.32%** | 6.74% | 7.11% | 6.73% |
| Net Sharpe | **2.320** | 2.224 | 1.852 | 2.250 |
| Stressed-cost Sharpe | **2.054** | 1.975 | 1.623 | 2.000 |
| Maximum drawdown | **-1.34%** | -1.39% | -2.73% | -1.39% |
| Annualized return without own best two months | **10.21%** | 10.04% | 7.68% | 10.12% |
| Maximum one-way ADV | 10.00% | 10.00% | 10.00% | 11.35% |

The factor-plus-ADV variant beats the control in 12 of 20 months and has a
positive median monthly difference of 0.038 percentage points. The gain is not
solely one winning month, but it is small and does not survive the more
conservative comparison that removes each path's own best two months.

## Paired evidence against the 1.5% control

| Challenger | Annualized net-return difference | Paired HAC t | 95% block-bootstrap interval | Sharpe difference | Probability Sharpe difference is positive |
|---|---:|---:|---:|---:|---:|
| Factor 12.5% | +0.332 pp | 0.305 | -2.164 to +2.099 pp | -0.096 | 24.9% |
| Factor + industry | -1.499 pp | -1.326 | -3.934 to +0.501 pp | -0.468 | 0.3% |
| Factor + ADV | +0.464 pp | 0.416 | -2.066 to +2.244 pp | -0.070 | 32.3% |

The challengers' own positive average returns remain statistically supported,
but that is not the question here. Their *incremental* return over the already
strong control is not supported. The 20-month paired intervals are wide and
cross zero.

## Economic interpretation

Looser factor limits let the optimizer accept slightly more systematic risk
when the frozen stock rankings indicate adequate payoff. Under factor-plus-ADV,
the annualized short contribution improves from -0.54% to +0.11%, while the
long contribution falls from 16.28% to 16.08%. The resulting gross-return gain
is about 0.46 percentage points and costs are essentially unchanged. This is a
coherent mechanism, but it raises realized volatility by about 0.40 percentage
points and consumes 11.35% rather than 10% of one-way ADV.

The factor-plus-industry validation Sharpe winner performs materially worse in
the later period. That failure is evidence of regime instability and is the
main reason not to continue expanding the grid. The bounded experiment found
a small return frontier, not a free improvement in risk-adjusted performance.

## Audit and evidence boundary

The corrected lineage contains all 184 validation and 80 historical-test
profile-months at the expected unique grain. Every numeric field is finite;
primary and stressed accounting identities reconcile to floating-point dust;
and the control is numerically identical to the previously published 1.5%
portfolio in every common monthly numeric field. Candidate-position, factor,
ADV, turnover, and gross constraints reconcile within immaterial solver dust.

The first exploratory lineage is marked `SUPERSEDED` because its tier filter
mistook an inherited reduction-only position for a new candidate-position cap
breach. No economic outcome changed; the corrected selection logic and a fresh
identity produced the result reported here.

This is explicitly post-diagnostic historical-development evidence. It cannot
replace the 1.5% confirmation candidate based on January 2024--August 2025.
September 2025--August 2026 remains quarantined, and untouched confirmation can
begin no earlier than September 2026 under the already frozen 1.5% policy.

## Reproducible sources

- Contract: `configs/research_v6_1/net_return_factorial_exploratory.json`
- Runner: `scripts/run_v6_1_net_return_frontier.py`
- Corrected complete result:
  `artifacts/research_v6_1/net_return_factorial_exploratory/63162896eac1-2ea69f5c5a54-a6bdeb1de9c4`
- Superseded selection lineage:
  `artifacts/research_v6_1/net_return_factorial_exploratory/e4e2099068aa-70e2253a4a6c-918c33b5926d`
