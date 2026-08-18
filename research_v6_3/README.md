# Research V6.3 — signal-to-weight research

V6.3 is a separate, user-authorized, post-diagnostic successor experiment. It
asks whether six economically distinct changes can improve the mapping from
the frozen V6.1 selector ranks to implementable portfolio weights while
keeping the universe, monthly schedule, costs, capacity limits, and incumbent
portfolio constraints fixed.

## Incumbent and interpretation

The common control is the V6.1 return-ranking selectors plus the FF5/momentum
alpha sign gate and the expected-net-return optimizer, with a 1.5% candidate
position cap. The control already trusts the selectors: stronger ranks are
mapped through causal monotonic calibration into expected long and short
payoffs, and the optimizer allocates more capital where calibrated payoff is
largest after trading, borrow, financing, and risk costs. V6.3 tests whether
that mapping should also depend on model confidence, opportunity breadth,
specialized information blocks, multiple horizons, tail probabilities, or a
latent residual-risk model.

## Six bounded ideas

1. Discount expected payoff when component models disagree or a stock's rank
   is unstable.
2. Scale expected payoff by a causal monthly opportunity-breadth percentile.
3. Route all 850 encoded features through market, fundamental/corporate,
   analyst, and alternative-data XGBoost specialists.
4. Train an all-feature shared neural network for long, short, FF5/momentum,
   three-month rank, and tail-probability targets.
5. Use the neural tail heads to abstain before the unchanged optimizer.
6. Add causal trailing PCA factors from residual stock returns to the frozen
   six-factor risk model.

All candidate settings are selected only on February 2022--December 2023.
Only the validation winner within each idea may enter the January
2024--August 2025 historical test. The incumbent is an explicit fallback.
September 2025--August 2026 remains quarantined, and untouched confirmation
cannot begin before September 2026.

## Completed result

The completed lineage is
`artifacts/research_v6_3/signal_weight_research/69841fc46fbd-c1dc1cd6be2e-ec470fe0dfb4`.
All six ideas were attempted under the common portfolio contract. The only
new profile selected by validation was the 3-factor/50%-share latent-risk
layer. On the historical test it produced 14.55% annualized net excess return
and a 2.282 Sharpe, below the unchanged incumbent's 14.67% and 2.320.

Decision: `RETAIN_V6_1_1P5_INCUMBENT`.

See [`SIGNAL_WEIGHT_RESEARCH_RESULT.md`](SIGNAL_WEIGHT_RESEARCH_RESULT.md) for
the complete result and interpretation.
