# V6.1 frozen-selector net-return frontier

This is a user-authorized post-diagnostic portfolio study. The V6.1 selectors,
features, ranks, FF5+momentum alpha gate, candidate tails, causal rank-to-payoff
calibration, cost model, and monthly schedule are unchanged. Profiles are
selected only on February 2022--December 2023 annualized net excess return;
January 2024--August 2025 outcomes never select a profile.

| Selection tier | Validation-selected profile | Annualized net excess | Total-return CAGR | Net Sharpe | Stress Sharpe | Max drawdown | Max gross | Max ADV | SPY CAGR |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| frozen_control | control | 9.94% | 15.85% | 1.723 | 1.449 | -1.96% | 1.24 | 7.43% | 21.33% |
| practical_100m | position_1p5 | 14.67% | 21.34% | 2.320 | 2.054 | -1.34% | 1.28 | 10.00% | 21.33% |
| expanded_institutional_100m | position_1p5 | 14.67% | 21.34% | 2.320 | 2.054 | -1.34% | 1.28 | 10.00% | 21.33% |
| unrestricted_research | position_1p5 | 14.67% | 21.34% | 2.320 | 2.054 | -1.34% | 1.28 | 10.00% | 21.33% |

The unrestricted tier is diagnostic rather than deployable. The practical and
expanded-institutional tiers are defined by validation-period realized gross,
liquidity, position, turnover, factor, and SIC2 limits. All reported costs are
stateful and include spread, square-root impact, financing, and borrow.

September 2025--August 2026 remains quarantined. None of these historical
results is untouched confirmation or a production forecast.
