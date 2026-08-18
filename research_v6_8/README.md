# Research V6.8 — Daily Survivor Reoptimization

Research V6.8 is closed under run
`d55a1db689f9-2ef39b0755c5-2e422abd519f`.

This study answers one narrow implementation question left by V6.7: after a
daily trim or exit executes, should the portfolio compensate by re-optimizing
the remaining holdings? It reproduces the exact V6.7 warning schedule and
next-session-close execution. The optimizer may increase only unflagged stocks
already held in that month's portfolio, cannot re-enter the warned stock or
change a position's side, cannot replace more gross than the warning batch
removed, and may leave the released capital in cash.

On 2023 validation, survivor reoptimization produced 3.73% annualized net
excess return and a 0.560 Sharpe, versus -1.71% and -0.232 for the V6.7 cash
overlay. On January 2024--August 2025 historical evidence it produced 25.67%
annualized net excess return, 12.14% volatility, a 2.114 Sharpe, and -6.40%
maximum drawdown. The cash overlay produced 25.18%, 11.81%, 2.132, and -6.47%.

The reoptimizer restored 41.7% of released gross exposure. Its incremental
historical return was 0.49 percentage points annualized with paired HAC t 0.64,
and its Sharpe was slightly lower. The terminal decision is
`RETAIN_MONTHLY_PRIMARY_AND_V6_7_CASH_OVERLAY_AS_SHADOW_ONLY`: the monthly
portfolio remains the trading primary, V6.7 remains a shadow overlay, and V6.8
is not promoted.

On 16 August 2026, the user made a separate operating choice to evaluate the
unchanged V6.8 policy prospectively in Alpaca paper trading despite that failed
historical promotion gate. The implementation is documented in
[`paper_trading/research_v6_8/README.md`](../paper_trading/research_v6_8/README.md).
No order has been placed because a current signal and runtime paper credentials
are not yet available; the adapter rejects the historical 2025 weights.

The production-data audit has now progressed beyond raw entitlement checks.
The January 2024--August 2025 primitive market gate passes 20/20 months and
400/400 consolidated mappings. A broader 47-feature stock-level formula audit
resolves 399/400 instrument-months: 34 features pass the prespecified
feature-level rule, 13 fail, and four market features still require
full-universe lifecycle, industry, or factor context. Paper orders therefore
remain blocked; failed definitions use the trained missing-value path and are
never silently replaced.

Primary files:

- Contract: [`daily_reoptimization.json`](../configs/research_v6_8/daily_reoptimization.json)
- Runner: [`run_v6_8_daily_reoptimization.py`](../scripts/run_v6_8_daily_reoptimization.py)
- Result: [`RESULT.md`](../evidence/v6_8_daily_reoptimization/RESULT.md)
- Machine-readable result: [`result.json`](../evidence/v6_8_daily_reoptimization/result.json)
- Independent audit: [`independent_audit.json`](../evidence/v6_8_daily_reoptimization/independent_audit.json)
- Paper deployment contract: [`alpaca_paper_deployment.json`](../configs/research_v6_8/alpaca_paper_deployment.json)
- Paper adapter: [`run_v6_8_alpaca_paper_deployment.py`](../scripts/run_v6_8_alpaca_paper_deployment.py)

The generated technical and production-data reports, plus detailed market
formula receipts, remain in the external artifact bundle.
