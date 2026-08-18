# Research V6.8 daily re-optimization result

## Decision

`RETAIN_MONTHLY_PRIMARY_AND_V6_7_CASH_OVERLAY_AS_SHADOW_ONLY`

The survivor re-optimizer does not clear every frozen economic and paired-inference gate. It therefore does not replace the monthly primary or promote the V6.7 overlay beyond shadow monitoring.

## What changed

The saved V6.7 warnings, monthly stocks, initial weights, and next-close execution
are unchanged.  After each executed warning batch, V6.8 maximizes the remaining
frozen monthly expected net payoff using only monotone increases in surviving,
never-warned holdings.  It cannot open a new stock, reverse a side, re-enter an
exit, or enlarge a warned name.  Refill is capped by the gross removed in that
same batch, and cash remains feasible.

## Validation: January--December 2023

- Monthly control: -3.82% annualized net excess return, -0.586 Sharpe.
- V6.7 cash-holding warning overlay: -1.71%, -0.232 Sharpe.
- V6.8 survivor re-optimization: 3.73%, 0.560 Sharpe.

## Historical evidence: January 2024--August 2025

- Monthly control: 21.63% annualized net excess return, 11.03% volatility, 1.961 Sharpe, -8.21% maximum drawdown.
- V6.7 cash overlay: 25.18%, 11.81%, 2.132, -6.47%.
- V6.8 re-optimization: 25.67%, 12.14%, 2.114, -6.40%.
- Refilled 41.7% of contemporaneously released gross across 157 action batches.
- Paired annualized difference versus V6.7: 0.49%, HAC t 0.64.

## Evidence boundary

This is a user-authorized post-diagnostic mechanism study on already observed
2024--August 2025 outcomes.  Even if the economic gates pass, it can only become
a prospective paper-trading shadow candidate.  The monthly strategy remains the
trading primary until clean evidence beginning no earlier than September 2026.
