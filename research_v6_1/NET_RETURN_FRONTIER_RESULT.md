# Research V6.1 — frozen-selector net-return frontier

## Decision

The user explicitly reopened portfolio-only post-diagnostic research on 14
August 2026 to ask a narrower question: how much implementable net return can
the frozen genuine selectors produce if portfolio profiles are selected solely
by validation-period annualized net excess return?

The validation-selected answer is a single transparent change: raise the
candidate position cap from 1.0% to 1.5%. All practical, expanded-institutional,
and unrestricted selection tiers chose that same profile. The current
historical-development decision is:

`RETAIN_1P5_POSITION_CAP_AS_CONFIRMATION_CANDIDATE`

This supersedes the 1.0% cap as the best historical-development portfolio
candidate. It does not create an untouched or production claim.

## Frozen information and test design

The V6.1 return selectors, all model features, fitted rankings, FF5+momentum
alpha sign gate, top/bottom candidate tails, causal rank-to-payoff calibration,
monthly schedule, risk model, and primary/stressed cost models are unchanged.
The optimizer still maximizes calibrated expected stock excess return minus
spread, square-root impact, financing, and stock-level borrow costs. Only
transparent portfolio constraints vary.

- Selection window: February 2022--December 2023, 23 months.
- Historical test: January 2024--August 2025, 20 months.
- Validation frontier: 25 profiles and 575 complete profile-months.
- Historical test: only the frozen control and the distinct validation winner,
  40 complete profile-months.
- Capital base: $100 million.
- Selection metric: highest annualized primary-cost net excess return within
  each feasibility tier. Historical-test outcomes never select a profile.

The frontier varies the short floor, discretionary sides, turnover, ADV,
position cap, net exposure, factor limits, SIC2 limits, forecast-risk ceiling,
gross exposure, and bounded joint institutional/aggressive profiles. It is a
constraint study, not a selector or feature retune.

## Validation result

| Profile | Annualized net excess | Stress net excess | Net Sharpe | Stress Sharpe | Max drawdown |
|---|---:|---:|---:|---:|---:|
| **1.5% position cap** | **4.12%** | **2.32%** | **0.831** | **0.466** | -2.77% |
| Factor limits at 15% | 4.08% | 2.39% | 0.740 | 0.432 | -2.46% |
| 2.0% position cap | 4.01% | 2.18% | 0.812 | 0.441 | -3.23% |
| Frozen 1.0% control | 2.63% | 0.89% | 0.497 | 0.168 | -2.42% |
| Free short floor | 1.96% | 0.36% | 0.388 | 0.072 | -3.61% |
| Turnover at 75% | 1.19% | -0.92% | 0.220 | -0.170 | -2.66% |
| Joint aggressive profile | 0.57% | -5.32% | 0.041 | -0.379 | -15.26% |

The complete 25-profile table is stored in `validation_summary.parquet` under
the result root. Greater turnover, leverage, net freedom, and broad simultaneous
relaxation generally destroyed net return after costs. The useful change is
concentration in the strongest frozen ranks, not unrestricted risk-taking.

## Historical-test result

| Metric | Frozen 1.0% control | Validation-selected 1.5% cap | Change |
|---|---:|---:|---:|
| Annualized net excess return | 9.94% | **14.67%** | **+4.73 pp** |
| Annualized stressed-cost net excess return | 8.33% | **12.96%** | **+4.63 pp** |
| Total-return CAGR | 15.85% | **21.34%** | **+5.49 pp** |
| Realized volatility | 5.77% | 6.32% | +0.55 pp |
| Net Sharpe | 1.723 | **2.320** | **+0.597** |
| Stressed-cost Sharpe | 1.449 | **2.054** | **+0.606** |
| Maximum drawdown | -1.96% | **-1.34%** | +0.62 pp |
| Positive-month fraction | 65% | **75%** | +10 pp |
| Net excess return without best two months | 5.83% | **10.21%** | **+4.38 pp** |
| Annualized primary cost | 0.99% | 1.07% | +0.07 pp |
| Mean gross exposure | 0.972 | 1.013 | +0.041 |
| Mean net exposure | 0.277 | 0.279 | +0.002 |
| Mean one-way monthly turnover | 0.500 | 0.500 | 0.000 |
| Mean long / short holdings | 76.7 / 69.4 | 59.7 / 62.3 | fewer, larger positions |

The winner compounded to 38.04% and a 21.339% total-return CAGR. SPY compounded
to 38.02% and a 21.330% CAGR over the same unusually strong 20-month window,
but at 10.98% volatility and a -6.89% maximum drawdown. Matching SPY's realized
return in this short period is descriptive, not proof of stable market
outperformance.

## Economic mechanism

The improvement is not a hidden leverage or cost assumption. Mean net exposure
is effectively unchanged and mean gross rises only from 0.972 to 1.013. Annual
primary cost rises by only 0.07 percentage points.

The 1.5% cap allows the optimizer to allocate more to its strongest frozen
ranked opportunities and hold fewer names. Annualized long contribution rises
from 14.13% to 16.28%. Short contribution improves from -3.19% to -0.54%, so the
portfolio preserves its hedge while avoiding much of the prior short-sleeve
drag. The challenger beats the control in 17 of 20 months; the median monthly
improvement is 0.30 percentage points.

## Paired inference and robustness

- HAC t-statistic for the challenger's own mean net excess return: **3.487**.
- Paired HAC t-statistic for the return improvement over the control: **5.135**.
- Annualized paired mean improvement: **4.726 percentage points**.
- Three-month circular-block bootstrap 95% interval for the return improvement:
  **+2.984 to +6.524 percentage points**.
- Sharpe improvement: **+0.597**; block-bootstrap 95% interval **+0.295 to
  +0.905**.
- Bootstrap probability that the Sharpe improvement is positive: **99.978%**.

These statistics are encouraging within the observed sample. They do not erase
the multiple-research and post-diagnostic history of V6.1, and 20 months is too
short to establish regime durability.

## Capacity, constraints, and accounting audit

All 575 validation and 40 historical-test profile-months are present, unique at
the profile-month grain, finite, and paired with atomic JSON receipts and
position Parquets. Monthly primary and stressed accounting identities reconcile
exactly. The re-run control has zero numerical difference from the frozen
published control across all common numeric fields.

For the 1.5% profile, candidate weights respect the 1.5% cap to numerical dust.
Maximum one-way ADV is 10.0000001%, turnover is 50.0000023%, factor exposures
are within 10.0000013%, and SIC2 net exposure is within 5.0000002%. Mean and
maximum gross are 1.013 and 1.280. The 10% ADV constraint binds, so $100 million
is the supported research scale; the same weights should not be scaled upward
without a new capacity analysis.

One inherited February 2024 short is displayed at 1.723% after return drift.
This is not a new-position cap breach: it entered the month above the cap after
the prior holding-period move, had left the candidate set, could not be
increased or reversed, and was reduced under the frozen liquidity/turnover
repair rule. All 20 monthly solver statuses are `optimal_inaccurate`, reflecting
the existing conic solver label, but realized economic constraint residuals are
immaterial and the accounting is exact.

## Interpretation and boundary

This is the first validation-selected portfolio change in the final frontier
that improves both validation return and the separate historical test with a
clear mechanism. It is therefore the strongest current thesis portfolio result
and a defensible candidate to freeze for confirmation.

It remains post-diagnostic historical development. V6.1 and its portfolio
research were designed after earlier results had been inspected, the test has
only 20 months, and the 1.5% profile uses more concentration and reaches the
10% ADV boundary. September 2025--August 2026 remains quarantined. The next
valid step is untouched confirmation from September 2026 under this complete
frozen policy, not another historical optimizer search.

## Reproducible sources

- Contract: `configs/research_v6_1/net_return_frontier.json`
- Runner: `scripts/run_v6_1_net_return_frontier.py`
- Complete result root:
  `artifacts/research_v6_1/net_return_frontier/880c4b16de97-6e6c91767c46-eac64c8cf6db`
- Result summary: `result.json`
- Monthly validation and test results: `validation_monthly.parquet` and
  `historical_test_monthly.parquet`
