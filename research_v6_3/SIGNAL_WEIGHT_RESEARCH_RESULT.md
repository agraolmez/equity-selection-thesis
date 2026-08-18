# Research V6.3 — signal-to-weight result

## Decision

Retain the V6.1 1.5% incumbent. None of the six V6.3 ideas improved both net
return and Sharpe in the validation-selected January 2024--August 2025
historical comparison.

The experiment supports the user's central intuition: the strongest observed
portfolio remains the direct, disciplined use of the frozen selector ranks.
The incumbent converts those ranks into causal expected long and short payoff,
then maximizes expected net return subject to the existing cost, capacity, and
risk contract. Adding generic uncertainty discounts or extra routing layers
did not improve that conversion.

## Final lineage

- Contract SHA-256:
  `69841fc46fbdf17ee7ab95c1d15233e564bbb790eae365b587a835aca69c12ca`
- Input identity:
  `c1dc1cd6be2e4bbbca3191d49241b5b049707f6851f3ccae73997e300adb2ce8`
- Scientific code identity:
  `ec470fe0dfb4484cdb31a367e7f36388e0c0484ae1350ea2ba0e5056201e1a93`
- Result root:
  `artifacts/research_v6_3/signal_weight_research/69841fc46fbd-c1dc1cd6be2e-ec470fe0dfb4`
- Exact incumbent monthly reproduction error: `0.0`
- Focused implementation tests: 5/5 pass
- Result status: `COMPLETE`

A publication-only Markdown field-name correction was applied after all
scientific outputs were written. It changed no signal, position, return,
metric, selection, or conclusion.

## Historical-test comparison

January 2024--August 2025 contains 20 monthly observations. Returns,
volatility, and costs are annualized.

| Portfolio | Net excess return | Volatility | Sharpe | Stress Sharpe | Max drawdown | Primary cost |
|---|---:|---:|---:|---:|---:|---:|
| **Unchanged 1.5% incumbent** | **14.67%** | **6.32%** | **2.320** | **2.054** | **-1.34%** | **1.07%** |
| 3-factor, 50%-share latent risk | 14.55% | 6.38% | 2.282 | 2.016 | -1.36% | 1.07% |

The latent challenger loses 0.115 percentage point of annualized net excess
return and 0.038 Sharpe. Its paired HAC t statistic for the return difference
is -0.79. The 95% circular-block-bootstrap interval for its annualized return
difference is -0.447% to +0.204%, and the bootstrap probability of a positive
Sharpe difference is 19.6%. It therefore supplies no evidence for replacing
the incumbent.

## What happened to each idea

1. **Confidence-adjusted payoff failed.** Penalties of 0.5, 1.0, and 2.0
   produced validation Sharpes of 0.294, 0.205, and -0.077 versus 0.831 for the
   control. Model disagreement was not a useful generic reason to reduce an
   otherwise strong calibrated rank.
2. **Opportunity-breadth scaling failed.** The best of the three settings
   reached only 1.83% annualized validation net excess return and 0.381 Sharpe;
   it also missed the practical constraint audit. Scaling every expected edge
   by a monthly breadth statistic changed holdings and turnover without adding
   reliable payoff.
3. **Specialist mixtures were infeasible.** All three temperatures admitted
   no FF5-gated long candidates in the first validation month under the frozen
   sleeve contract. The control admitted 170. The experiment records this as
   a challenger failure rather than weakening constraints after observing it.
4. **Multitask/multihorizon networks were infeasible.** All three horizon
   blends failed the same common portfolio feasibility requirement. A shared
   representation did not preserve a usable cross-target gate.
5. **Decision-aware masks were infeasible.** All six blend/threshold settings
   made the already selective candidate set too sparse for the unchanged
   portfolio contract.
6. **Latent residual risk was the only validation survivor.** Three factors at
   50% variance share raised validation return from 4.12% to 4.33% and Sharpe
   from 0.831 to 0.863. The improvement did not transfer: historical-test
   return, Sharpe, stress Sharpe, and drawdown were all slightly worse.

## Incumbent portfolio mechanics

The incumbent averaged 101.3% gross exposure and 27.9% net exposure, with
about 59.7 long and 62.3 short holdings. Mean one-way turnover was 50%, and
maximum one-way participation reached the frozen 10% of ADV limit. Annualized
long contribution was +16.28%; short contribution was -0.54%. The short sleeve
therefore remained primarily a hedge/covariance instrument rather than a
standalone return engine.

This clarifies “trust the selectors”: the portfolio does not use arbitrary
rank-proportional weights. It uses the ranks to identify candidates and to
estimate cardinal long/short payoff, then lets costs and risk determine the
final weight up to the 1.5% stock cap. The successful policy is the simplest
tested version of that principle—without an extra confidence haircut.

## Research boundary

This is reproducible post-diagnostic historical development evidence, not
untouched confirmation or proof of production quality. No V6 or V6.1 artifact
was overwritten. September 2025--August 2026 remains quarantined. The next
scientifically valid step is to freeze the unchanged 1.5% incumbent and test
newly available observations beginning no earlier than September 2026, rather
than tune another transformation on the same 20 historical months.
