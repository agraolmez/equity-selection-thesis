# Research V6.1 — Step 02 portfolio result

## Decision

The corrected portfolio study is complete.  The frozen primary policy—joint
CAPM and FF5+momentum alpha sign gates, followed by cost/risk optimization at a
10% forecast-volatility ceiling—passes every prespecified historical-development
promotion gate.

This is a **proceed-to-confirmation** result, not a production claim.  The
portfolio evidence contains only 20 post-diagnostic months from January 2024
through August 2025.  September 2025 through August 2026 remains quarantined;
the earliest clean confirmation starts in September 2026.

## Frozen lineage

- Contract SHA-256: `48f429eb0d7f305a899e432dfe952cf2c192cd450d1cd661b679f11c24b6c47f`
- Input identity: `53a6425a3d928fca16110a013d650a84b42b19155f94cb4a87cd5c9e8a47aa5f`
- Code identity: `7256b91cc3938dd6b60368c7a21ed8ec97d8230bb00ebf623f6b094fcd0a7b77`
- Result root: `artifacts/research_v6_1/portfolio_step_02/48f429eb0d7f-53a6425a3d92-7256b91cc393`
- Completion: 22 paths and 440/440 variant-months

The earlier `6bf45ef92bf1-53a6425a3d92-2c13774898a8` lineage is diagnostic-only.
Its audit exposed that the one-period cost objective could indefinitely retain
stocks after they left the candidate set.  The corrected lineage first
minimizes stale gross subject to feasibility, then maximizes expected net
return.  It removes 95.56% of stale gross at each rebalance in aggregate; any
remainder is mechanically limited by turnover, liquidity, and portfolio
feasibility.

## All preregistered results

Returns, volatility, and costs are annualized.  `Net` is excess of cash after
trading, financing, and borrow costs.  `Stress Sharpe` applies the frozen
stressed costs to the same monthly holdings without re-optimization.

| Policy and weighting | Net | Volatility | Net Sharpe | Stress Sharpe | Max drawdown | Primary cost |
|---|---:|---:|---:|---:|---:|---:|
| FF5+momentum, optimized 10% | 9.94% | 5.77% | 1.723 | 1.449 | -1.96% | 0.99% |
| Both alphas, optimized 12% | 10.60% | 6.21% | 1.707 | 1.449 | -4.15% | 0.99% |
| Both alphas, optimized 8% | 10.29% | 6.03% | 1.706 | 1.443 | -3.71% | 0.98% |
| **Both alphas, optimized 10% (primary)** | **10.39%** | **6.22%** | **1.671** | **1.414** | **-4.05%** | **0.99%** |
| FF5+momentum, linear tail | 7.97% | 6.28% | 1.269 | 0.874 | -2.00% | 1.16% |
| FF5+momentum, squared tail | 8.08% | 6.41% | 1.260 | 0.904 | -2.76% | 1.12% |
| FF5+momentum, Gaussian tail | 7.91% | 6.32% | 1.251 | 0.886 | -3.02% | 1.12% |
| CAPM, optimized 10% | 7.36% | 5.90% | 1.249 | 0.945 | -3.78% | 1.06% |
| FF5+momentum, equal | 7.72% | 6.38% | 1.210 | 0.767 | -2.45% | 1.26% |
| Both alphas, equal | 8.19% | 6.77% | 1.210 | 0.802 | -3.03% | 1.25% |
| Both alphas, linear tail | 7.69% | 6.83% | 1.126 | 0.770 | -3.31% | 1.15% |
| Both alphas, squared tail | 7.66% | 6.89% | 1.112 | 0.786 | -3.66% | 1.11% |
| Both alphas, Gaussian tail | 7.57% | 6.84% | 1.107 | 0.775 | -3.92% | 1.11% |
| Return only, squared tail | 6.14% | 6.69% | 0.918 | 0.568 | -3.85% | 1.12% |
| Return only, linear tail | 5.77% | 6.57% | 0.879 | 0.483 | -3.22% | 1.18% |
| Return only, Gaussian tail | 5.80% | 6.65% | 0.873 | 0.513 | -4.10% | 1.13% |
| CAPM, squared tail | 5.99% | 7.06% | 0.848 | 0.524 | -4.24% | 1.10% |
| Return only, optimized 10% | 4.15% | 5.05% | 0.821 | 0.429 | -2.75% | 1.14% |
| CAPM, linear tail | 5.75% | 7.02% | 0.819 | 0.461 | -4.02% | 1.15% |
| CAPM, Gaussian tail | 5.64% | 7.06% | 0.799 | 0.469 | -4.56% | 1.11% |
| CAPM, equal | 5.28% | 7.08% | 0.747 | 0.337 | -4.06% | 1.27% |
| Return only, equal | 4.52% | 6.46% | 0.700 | 0.229 | -3.32% | 1.32% |

## Primary-policy gate audit

- Net Sharpe is 1.671, above the strict 1.0 threshold.
- Annualized net excess return is 10.39%, strictly positive.
- Removing the two best months still leaves 6.65% annualized net excess return.
- Stressed costs leave 8.77% annualized net excess return and a 1.414 Sharpe.
- The 8%, 10%, and 12% forecast-volatility specifications all produce net
  Sharpes between 1.67 and 1.71.  The risk ceiling is not binding: mean forecast
  volatility is about 4.2%, while turnover, side-gross, and factor constraints
  determine the allocations.
- The primary path has 50% mean monthly one-way turnover, 0.95 mean gross,
  0.26 mean net, about 74 long and 70 short positions, and 0.99% annualized
  primary cost.  Turnover is at its frozen cap and is the main deployment
  caution.
- The $100 million path ends at $128.68 million after 20 months, including the
  cash return.  Its geometrically compounded excess return is 18.47%.

## Interpretation

The return selector alone is not enough: its best direct rank rule is the
squared-tail book at a 0.92 Sharpe, while the return-only optimizer falls to
0.82.  The FF5+momentum alpha gate is the economically important overlay.  At
the optimized weighting it raises annualized net excess return by 5.80
percentage points versus return-only and lifts Sharpe from 0.82 to 1.72.

CAPM alone helps the optimizer, but adds little once FF5+momentum is present.
The FF5-only optimizer has the highest descriptive Sharpe (1.723); the frozen
joint-alpha primary has slightly greater net return but slightly lower Sharpe
(1.671).  This does not authorize an ex-post switch away from the primary
policy.  It identifies FF5+momentum as the main source of incremental portfolio
value and CAPM as a robustness/consensus filter rather than an independent
performance engine.

For deterministic ranks, squared-tail is best for return-only, while linear
tail is best after the FF5 gate.  No single nonlinear rank transform dominates
across policies.  Cost/risk optimization adds material value only once an alpha
gate removes weak candidates.

The long book supplies the profits.  In the primary path, annualized gross
long contribution is 12.66%, short contribution is -1.28%, and annualized
cost is 0.99%.  The short selector is therefore useful for risk construction
and diversification here, but is not independently profitable over this short
historical window.

## Engineering and capacity audit

- All 440 result keys are unique and every path has all 20 months.
- Ten first-day corporate-action exits use their bound observed CRSP terminal
  settlement returns.  The portfolio window has zero unpriced held outcomes.
- Candidate positions obey the 1% formation cap.  Thirty-nine inherited
  position-month rows temporarily exceed 1% while being unwound; the maximum
  is 1.159%.  No new or increased candidate breaches the cap.
- Maximum realized one-way ADV participation is 7.56%, below the 10% limit.
- Optimized gross is at most 1.426; long and short gross floors are satisfied;
  net exposure remains between 0.192 and 0.300.
- Maximum absolute six-factor exposure is 0.100000023, maximum SIC2 net is
  0.050000004, and maximum SIC2 gross is 0.272.  These are within practical
  numerical tolerance of the frozen 0.10, 0.05, and 0.30 limits.
- CLARABEL reports 119/120 optimized months as `optimal_inaccurate` under the
  deliberately tight numerical tolerances.  The direct constraint audit shows
  only immaterial residuals (roughly 1e-8 in portfolio weights); no economic
  limit or conclusion changes.

## Honest limitation

The result is encouraging enough to continue, but not enough to claim a
production strategy.  Twenty monthly observations are a small, favorable,
post-diagnostic sample; the 2025 January–August subperiod is especially strong.
SPY and QQQ also performed well in this window, although the portfolio is a
low-beta long/short strategy and is not directly comparable to a long-only
index.  The primary path's HAC t-statistic for mean monthly net excess return
is approximately 2.25, useful but not decisive after model-development and
multiple-comparison considerations.

The next valid action is to freeze this complete policy and evaluate newly
available months without changing selectors, gates, calibration, costs, or
portfolio rules.  Portfolio optimization research is complete; further tuning
on January 2024–August 2025 would convert the result into another development
exercise rather than stronger confirmation.
