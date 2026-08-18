# Research V6.6 — intramonth intervention study

## Executive Summary

- **Decision:** `RETAIN_MONTHLY_REBALANCE_CONTROL_NO_INTRAMONTH_INTERVENTION_QUALIFIES`.
- **Monthly control:** 21.63% annualized net excess return, 1.961 Sharpe, and -8.21% maximum drawdown in January 2024–August 2025.
- **Best historical intervention:** `safety_only`. Historical evidence cannot reselect a policy; promotion requires all frozen validation, return, Sharpe, drawdown, cost, and paired-inference gates.
- **Learning rule:** the predictive overlay refits quarterly with expanding data, but only labels observable by that refit's monthly signal date enter training.

## What was tested

The frozen monthly V6.1 stock selectors, FF5/momentum gate, optimizer, and
relative monthly targets are unchanged. At each rebalance, targets are scaled
uniformly to no more than 1.80x gross. An intervention can only trim or close
an existing position; it cannot add a stock or re-enter before the next
monthly rebalance. A close signal executes at the following trading-session
close.

2022 is the initial predictive fit, 2023 selects one transparent-rule policy
and one predictive policy, and January 2024–August 2025 evaluates only the
prespecified control, safety policy, selected policies, and their frozen
predictive-plus-safety combination. The selected rule was `rule__stop_loss_15_exit`;
the selected predictive policy was `predictive__xgb_depth_2__h5__selective`.

## Historical performance

| Policy | Ann. net excess | Volatility | Sharpe | Max drawdown | Actions | Paired HAC t |
|---|---:|---:|---:|---:|---:|---:|
| monthly_control | 21.63% | 11.03% | 1.961 | -8.21% | 0 | nan |
| safety_only | 21.64% | 11.03% | 1.962 | -8.21% | 1 | 1.32 |
| selected_rule | 15.78% | 11.96% | 1.320 | -7.73% | 380 | -1.54 |
| selected_predictive | 23.00% | 12.03% | 1.913 | -6.58% | 311 | 0.62 |
| selected_predictive_plus_safety | 23.00% | 12.03% | 1.913 | -6.58% | 311 | 0.62 |

## Interpretation

No validation-frozen intervention cleared every economic and paired-inference gate. The evidence therefore favors preserving monthly discipline and using the daily layer only for operational exposure safety.

## Data and implementation checks

- Daily stock returns compound to the authenticated monthly outcomes; explicit market-accounting fallbacks remain untradeable until their exit-date accounting observation.
- Daily return coverage before fallback is 99.53% of active position-days and 99.47% of absolute starting weight.
- The untouched daily control's maximum absolute monthly return difference from the predecessor monthly-cost replay is 0.0633%; the documented difference is daily drift-aware financing and borrow accrual.
- The complete authenticated 850-column monthly feature matrix is reused, augmented with causal price, liquidity, position-state, market, and rolling direct/family/industry news variables.

## Caveats and next step

This is post-diagnostic historical development evidence, not untouched or live
confirmation. September 2025–August 2026 remains quarantined. A policy may be
paper-traded only as a shadow alongside the monthly control until forward
evidence beginning no earlier than September 2026 confirms that its turnover
and execution assumptions survive in real time.
