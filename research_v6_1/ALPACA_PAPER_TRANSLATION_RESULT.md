# Research V6.1 Alpaca paper-account translation

> Deployment boundary: the thesis reports the uncapped, validation-selected
> 2.05-times research specification as its high-return model. This 1.80-times
> capped translation is retained separately for Alpaca paper trading and must
> not be substituted for the thesis result.

**Status:** Complete post-diagnostic deployment translation
**Result root:** `artifacts/research_v6_1/alpaca_paper_translation/7a85da08ca8c-ca8bc83ff816-da182525b7d6`

This exercise does not retrain a selector, rerun the optimizer, or change any stock's relative monthly weight. It replays the frozen V6.1 `position_1p5` holdings from a $100,000 starting NAV. The translated policy applies the common 2.05 research multiplier and proportionally reduces all weights whenever necessary so rebalance gross does not exceed 1.80 times NAV.

## Historical-test result: January 2024-August 2025

| Cost view | CAGR | Volatility | Sharpe | Maximum drawdown | Ending NAV |
|---|---:|---:|---:|---:|---:|
| Frozen institutional financing | 35.34% | 11.05% | 2.370 | -2.28% | $165,593 |
| Broker-current financing sensitivity | 29.62% | 11.04% | 1.972 | -3.37% | $154,097 |
| SPY | 21.33% | 10.98% | - | -6.89% | $138,021 |
| QQQ | 23.26% | 14.57% | - | -8.63% | $141,693 |

## Validation result: February 2022-December 2023

Under the broker-current financing sensitivity, the translation earns a 7.23% CAGR at 9.39% volatility and a 0.437 Sharpe ratio. It ends at $114,315, compared with $107,670 for SPY and $111,716 for QQQ. The weak validation Sharpe prevents the scaling policy from replacing the original validation-selected institutional strategy.

## Operating boundary

Every rebalance respects the 1.80-times cap. Return drift nevertheless creates four validation and one historical-test month-end breaches of the 1.95-times safety trigger. Two validation month ends exceed two times NAV. A paper trader therefore needs daily exposure, buying-power, maintenance-margin, and shortability checks plus proportional deleveraging. Monthly replay cannot certify intramonth compliance.

The broker-current case uses Alpaca's published 6.25% non-elite annual margin rate accessed on 14 August 2026. It remains a sensitivity rather than a historical rate series, and Alpaca paper trading does not simulate every live execution or borrowing friction.
