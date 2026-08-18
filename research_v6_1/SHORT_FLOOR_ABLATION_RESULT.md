# Research V6.1 — exact no-forced-short ablation

## Correction and decision

This is the clean experiment the user requested. It preserves the selected
V6.1 Step 02 FF5/momentum portfolio and removes mandatory short exposure. It
supersedes any attempt to use Step 03 or Step 04 as evidence about the isolated
short-floor intervention because those studies changed several portfolio
rules simultaneously.

The ablation is complete and does **not** improve the portfolio. The original
Step 02 FF5/momentum portfolio remains the stronger historical-development
result.

The complete run is:

`artifacts/research_v6_1/portfolio_short_floor_ablation/01ff21f472f6-53a6425a3d92-5c3a2cf90201`

## What changed

Exactly two linked constraint values changed:

1. Minimum short gross changed from 30% to zero.
2. Maximum net exposure changed from 30% to 100% because retaining a 50%
   minimum long gross and 30% maximum net would otherwise impose an implicit
   20% minimum short allocation.

The second change is mechanical and necessary to implement “do not force the
portfolio to short.” It does not create a new signal, objective, calibration,
or risk model.

Everything else is inherited unchanged from Step 02: frozen long- and
short-return selectors, FF5/momentum sign gate, 20% candidate tails, causal
monotonic payoff calibration, expected-net-return objective, 10% forecast-risk
ceiling, 50% minimum long gross, 150% maximum gross, -10% minimum net, 1%
position cap, 50% turnover cap, 10% ADV cap, market-beta and factor limits,
SIC2 limits, costs, borrow, $100 million NAV, and January 2024--August 2025.
There is no validation, retuning, or selector retraining.

## Result

| January 2024--August 2025 | Original required-short control | Exact no-forced-short ablation | Change |
|---|---:|---:|---:|
| Annualized net excess return | **9.94%** | 7.00% | -2.94 pp |
| Realized volatility | **5.77%** | 7.74% | +1.97 pp |
| Net Sharpe | **1.723** | 0.904 | -0.819 |
| Stress Sharpe | **1.449** | 0.732 | -0.717 |
| Maximum drawdown | **-1.96%** | -4.72% | -2.76 pp |
| Annualized primary cost | 0.99% | **0.93%** | -0.06 pp |
| Annualized long contribution | **14.13%** | 10.91% | -3.22 pp |
| Annualized short contribution | -3.19% | **-2.97%** | +0.22 pp |
| Mean long gross | 62.4% | 75.9% | +13.4 pp |
| Mean short gross | 34.7% | **22.0%** | -12.8 pp |
| Mean net exposure | 27.7% | 53.9% | +26.2 pp |

The no-forced-short path wins only 9 of 20 months on paired primary net excess
return. Its annualized gross-return deficit is 3.00 percentage points, while
the annualized cost saving is only 0.06 points.

## Did the experiment actually remove forced shorts?

Yes. Short gross is below 30% in 19 of 20 months, below 20% in six months, and
reaches a minimum of 6.5%. No month sits at the old 30% short floor. The
optimizer is not reproducing the old fixed sleeve.

It nevertheless holds some shorts every month because the unchanged absolute
market-beta limit of 0.10 binds in all 20 months. Those positions are therefore
retained by the original expected-return/risk problem rather than by a side
gross floor. Removing that beta control would be a different experiment and
would no longer mean “keep everything else the same.”

## Interpretation

The user's diagnosis was partly correct: the 30% minimum short sleeve made the
short book more negative than necessary. Removing it improves short
contribution by 0.22 percentage points annually and lowers costs slightly.
However, that is much too small to offset what happens elsewhere.

The original short and net constraints acted as strong portfolio
regularization. Once net exposure can rise, the optimizer places more capital
in long names using noisy calibrated return magnitudes. Mean net exposure
doubles, the market-beta ceiling binds every month, realized volatility rises,
and the realized long contribution falls despite higher long gross. The main
loss is therefore not the remaining short book; it is worse long-side sizing
and poorer realized diversification after the exposure constraint is relaxed.

This result also illustrates the distinction between ranking and cardinal
sizing. The frozen model can order stocks reasonably well, but the optimizer's
estimated differences in expected payoff are not accurate enough to use the
extra freedom reliably. The original constraints made the portfolio less
sensitive to those magnitude errors.

## Research boundary

This is a post-diagnostic historical ablation, not untouched evidence. It
should be reported as mechanism evidence and not used to claim that the
original 1.723 Sharpe will persist. Outcomes from September 2025 through August
2026 remain quarantined, and clean confirmation remains September 2026 or
later under a fully frozen policy.
