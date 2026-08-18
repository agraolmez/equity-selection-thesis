# Research V6.1 — Step 03 portfolio repair result

## Decision

Step 03 is complete. Allowing shorts to be optional, adding persistent
entry/holding bands, and replacing stock-level return magnitudes with three
conservative rank-to-payoff mappings did **not** improve on the frozen V6.1
Step 02 optimizer.

The best Step 03 path is the joint-alpha direct-rank portfolio. Over January
2024 through August 2025 it earned 7.03% annualized net excess return at 6.82%
realized volatility, for a 1.030 net Sharpe. The same holdings have a 0.926
Sharpe under stressed costs. This is economically positive historical
development evidence, but the frozen Step 02 joint-alpha optimizer remains
materially stronger at a 1.671 primary-cost Sharpe and 1.414 stressed Sharpe.

## Frozen lineage

- Contract SHA-256: `e6253d84f60c6ed955781ed1e37c73467472306ca510ae9384b6dedac1bc7745`
- Input identity: `53a6425a3d928fca16110a013d650a84b42b19155f94cb4a87cd5c9e8a47aa5f`
- Code identity: `4981f8103e7b86e2e4a4f4930204cbe67f2c73963c2da019c9aa48c7a1dbde2f`
- Result root: `artifacts/research_v6_1/portfolio_step_03/e6253d84f60c-53a6425a3d92-4981f8103e7b`
- Completion: six paths and 120/120 portfolio months
- Selector retraining: none

## All Step 03 paths

Returns, volatility, and costs are annualized. All paths permit long-only,
short-only, two-sided, or cash allocations; neither side has a minimum gross
requirement.

| Alpha gate and rank mapping | Net excess return | Volatility | Net Sharpe | Stress Sharpe | Max drawdown | Primary cost |
|---|---:|---:|---:|---:|---:|---:|
| Both alphas, direct rank | 7.03% | 6.82% | 1.030 | 0.926 | -4.17% | 0.61% |
| FF5+momentum, direct rank | 6.74% | 6.83% | 0.987 | 0.882 | -4.13% | 0.61% |
| Both alphas, shrunk linear | 6.54% | 7.14% | 0.915 | 0.820 | -5.02% | 0.57% |
| FF5+momentum, shrunk linear | 6.37% | 7.70% | 0.827 | 0.735 | -4.40% | 0.59% |
| Both alphas, broad bucket | 5.00% | 7.47% | 0.669 | 0.579 | -6.10% | 0.56% |
| FF5+momentum, broad bucket | 4.90% | 7.58% | 0.647 | 0.554 | -5.20% | 0.58% |

## What the repair experiment answered

- Direct rank strength transfers better than a fitted slope or broad payoff
  buckets in this short historical window.
- No candidate alpha short cleared the frozen stressed trading-and-borrow
  hurdle. Mean alpha-short gross and alpha-short contribution are therefore
  exactly zero for every path.
- The separate risk-hedge short sleeve was active, averaging 18.8% gross in
  the best path, but detracted 5.85 percentage points of annualized gross
  return. The corresponding long sleeve contributed 13.49% before costs.
- Lower turnover reduced annualized primary cost from roughly 0.99% in the
  frozen Step 02 joint optimizer to 0.61%, but the lost gross return was much
  larger than the saving.
- The result rejects the idea that mandatory-looking shorts alone caused the
  rank-to-portfolio mismatch. Shorts were already optional here; the alpha
  shorts stayed at zero and hedge shorts were a return drag.

## Interpretation and next boundary

Step 03 is a completed robustness and mechanism study. It does not replace or
retune Step 02. The strongest frozen V6.1 portfolio remains the Step 02
joint-alpha primary for confirmation; the Step 02 FF5-only optimizer remains
the highest descriptive Sharpe control and is not promoted after outcomes.

All evidence ends in August 2025 and was designed after earlier results were
known. September 2025 through August 2026 remains quarantined. Clean
confirmation can start no earlier than September 2026 under a completely
frozen policy.
