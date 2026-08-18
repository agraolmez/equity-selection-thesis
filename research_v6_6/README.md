# Research V6.6 — intramonth intervention study

V6.6 tests whether intervening between monthly rebalances improves the frozen
V6.1 1.80x Alpaca-compatible portfolio. It does not alter the stock selectors,
FF5/momentum alpha gate, monthly optimizer, stock set, or relative monthly
target weights.

The study compares the monthly hold control with an operational gross-exposure
safety rule, bounded take-profit/stop-loss/trailing-stop rules, and a causal
hold/trim/exit predictor. The predictive candidates receive all 850
authenticated encoded monthly features plus 38 daily position, price,
liquidity, market, and related-news variables. They fit initially on 2022,
select one policy during 2023, and refit quarterly with expanding,
label-available history during the January 2024--August 2025 evaluation.

No intervention qualifies. The monthly control produces 21.63% annualized net
excess return, 11.03% volatility, a 1.961 Sharpe, and a -8.21% maximum
drawdown measured from daily NAV. The selected predictive XGBoost overlay
raises annualized net excess return to 23.00% and improves drawdown to -6.58%,
but its 12.03% volatility lowers Sharpe to 1.913 and the paired improvement is
not statistically reliable (HAC t 0.62). The validation-selected 15% stop-loss
does not generalize.

Decision:
`RETAIN_MONTHLY_REBALANCE_CONTROL_NO_INTRAMONTH_INTERVENTION_QUALIFIES`.
Use daily exposure monitoring as an operational safeguard and record the
predictive overlay in shadow mode only.

- Contract: [`intrmonth_interventions.json`](../configs/research_v6_6/intrmonth_interventions.json)
- Runner: [`run_v6_6_intramonth_interventions.py`](../scripts/run_v6_6_intramonth_interventions.py)
- Result: [`RESULT.md`](../evidence/v6_6_intramonth/RESULT.md)
- Machine-readable result: [`result.json`](../evidence/v6_6_intramonth/result.json)

The generated visual report and audit notebook remain in the external artifact
bundle bound by the checked-in run manifest.

The first preliminary lineage admitted solver-dust positions below the
established practical tolerance and is superseded. It must not be cited.
