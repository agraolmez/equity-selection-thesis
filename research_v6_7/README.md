# Research V6.7 — Daily MLP Overlay

Research V6.7 is closed under run
`4108d1f64761-16046ee58521-22f799195507`.

The study answered the specific V6.6 coverage gap: whether a compact neural
network could improve the causal daily hold/trim/exit layer when the monthly
portfolio, 1.80x gross translation, inputs, costs, and execution timing stayed
unchanged. It tested signed-return regression and adverse-return classification
at one- and five-day horizons, three seeds per model, and prespecified equal
five-day MLP-XGBoost ensembles.

The 2023 selection winner was the selective five-day equal signed-return MLP
plus frozen depth-2 XGBoost ensemble. On January 2024--August 2025 historical
evidence it produced 25.18% annualized net excess return, 11.81% volatility,
2.132 Sharpe, and -6.47% daily maximum drawdown. The monthly control produced
21.63%, 11.03%, 1.961, and -8.21%.

The ensemble's paired improvement versus the control has HAC t 0.96, below the
1.96 promotion gate. The terminal decision is
`RETAIN_MONTHLY_REBALANCE_CONTROL_NO_DAILY_MLP_PROMOTION`. The ensemble may be
logged in paper-trading shadow mode but may not replace the monthly trading
policy from this post-diagnostic evidence.

Primary files:

- Contract: [`daily_mlp_overlay.json`](../configs/research_v6_7/daily_mlp_overlay.json)
- Runner: [`run_v6_7_daily_mlp_overlay.py`](../scripts/run_v6_7_daily_mlp_overlay.py)
- Result: [`RESULT.md`](../evidence/v6_7_daily_overlay/RESULT.md)
- Machine-readable result: [`result.json`](../evidence/v6_7_daily_overlay/result.json)

The generated technical report and executed audit notebook remain in the
external artifact bundle bound by the checked-in run manifest.
