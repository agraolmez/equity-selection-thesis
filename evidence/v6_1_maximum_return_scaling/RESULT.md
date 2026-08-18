# Frozen-selector maximum-return scaling frontier

The frozen 1.5% position-cap portfolio is replayed at constant exposure
multipliers.  Stock selection, ranks, FF5+momentum gate, relative holdings, and
signal dates do not change.  Spread, square-root impact, financing, borrow,
drift, and NAV are recomputed at every scale.

The search does not stop when SPY or QQQ is crossed.  The absolute validation
ceiling is **9.35x**, where validation net
total-return CAGR reaches **22.17%**.

| Validation-selected tier | Scale | Historical CAGR | SPY CAGR | QQQ CAGR | Sharpe | Stress Sharpe | Max drawdown | Max gross | Max ADV | Beats both |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| realistic_100m | 1.00x | 21.34% | 21.33% | 23.26% | 2.32 | 2.05 | -1.34% | 1.28 | 10.0% | no |
| expanded_institutional_100m | 1.20x | 24.65% | 21.33% | 23.26% | 2.31 | 2.02 | -1.70% | 1.54 | 12.4% | yes |
| balanced_high_return | 2.05x | 39.17% | 21.33% | 23.26% | 2.26 | 1.90 | -3.76% | 2.62 | 24.4% | yes |
| stress_positive_ceiling | 5.85x | 114.36% | 21.33% | 23.26% | 2.16 | 1.66 | -14.14% | 7.49 | 120.4% | yes |
| absolute_return_ceiling | 9.35x | 194.30% | 21.33% | 23.26% | 2.08 | 1.50 | -24.48% | 11.97 | 294.4% | yes |

`realistic_100m` and `expanded_institutional_100m` are constrained capacity
views.  `balanced_high_return` retains a validation Sharpe floor.
`stress_positive_ceiling` is the largest return selected while stressed excess
return remains positive.  `absolute_return_ceiling` is a research limit, not a
deployable mandate: its leverage, turnover, ADV participation, and drawdown
must be read together with its return.

All tier choices use February 2022--December 2023 only.  January 2024--August
2025 is evaluated after the tier winners are frozen.  Prospective confirmation
remains separate.
