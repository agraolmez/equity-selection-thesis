# Thesis Reproducibility and Result Audit

**Audit date:** 17 August 2026
**Scope:** Final Research V6-V6.8 selector, portfolio, scaling, and intramonth lineage
**Decision:** **Ready for final thesis preparation with explicit limitations.** No material error was found that changes the frozen selector or monthly portfolio conclusion.

## Bottom line

The independent review reconciled the final data identities, point-in-time chronology, prediction keys, portfolio accounting, position constraints, cost calculations, benchmark comparison, and the exact V6.3 reproduction of the incumbent monthly return path. The realistic reference remains the V6.1 return-ranking portfolio with the FF5-plus-momentum alpha sign gate, causal rank-to-payoff calibration, expected-net-return optimizer, and a validation-selected 1.5% candidate position cap. The thesis high-return model is the separately frozen 2.05-times scaling tier selected on validation only.

The review does **not** upgrade the result to production quality. The January 2024–August 2025 historical test contains only 20 months and belongs to a post-diagnostic research lineage. The current local Python environment also differs from the project’s pinned dependency contract. Both facts must remain visible in the thesis and must be addressed before paper trading.

## Audited authorities

- V6 selector run: `artifacts/research_v6/selector_step_01/841b5f23ca9d-01c2e95ca6b9-dab475636cfc`
- V6.1 selector run: `artifacts/research_v6_1/selector_step_01/481e306553c2-16046ee58521-d5dadd9cd9c3`
- V6.1 net-return frontier: `artifacts/research_v6_1/net_return_frontier/880c4b16de97-6e6c91767c46-eac64c8cf6db`
- V6.1 maximum-return scaling frontier: `artifacts/research_v6_1/maximum_return_scaling_frontier/b50c39dfbd62-ca8bc83ff816-e5468ec50cbf`
- V6.3 final challenge: `artifacts/research_v6_3/signal_weight_research/69841fc46fbd-c1dc1cd6be2e-ec470fe0dfb4`
- V6.6 intramonth intervention study: `artifacts/research_v6_6/intrmonth_interventions/566f3749ee9b-16046ee58521-a82fbba1983b`
- V6.7 daily MLP study: `artifacts/research_v6_7/daily_mlp_overlay/4108d1f64761-16046ee58521-22f799195507`
- V6.8 survivor reoptimization study: `artifacts/research_v6_8/daily_reoptimization/d55a1db689f9-2ef39b0755c5-2e422abd519f`
- External Stage 13 inputs: `/Users/agraemirolmez/Desktop/Thesis/processed/phase_06_final_test/stage13-final-inputs-v2`
- V6 model-feature build: `/Users/agraemirolmez/Desktop/Thesis/processed/research_v6/model_features_v2`
- Alpaca archive: `/Users/agraemirolmez/Desktop/Thesis/raw/research_v6/alpaca_news_v1`
- LSEG USD archive: `/Users/agraemirolmez/Desktop/Thesis/raw/research_v6/lseg_usd_v1`

## Data and chronology checks

| Check | Result |
|---|---|
| Run completion | All four final manifests report `COMPLETE`. |
| Bound source files | All 14 portfolio source SHA-256 hashes match their frozen contracts. |
| Feature cache | 278,897 stock-month rows, 128 months, January 2015–August 2025. |
| December 2019 | Present with 2,051 rows; not interpolated. |
| Uniqueness | No duplicate month–PERMNO keys in the model cache or selected predictions. |
| Features | 318 raw feature sources and 850 finite encoded model columns. |
| Outcome timing | Every nonmissing `label_available_date` is strictly after its signal and only enters later eligible fits. |
| Missing outcomes | 777 rows, 0.279% of the cache, lack a realized label/availability date. These rows are unavailable outcomes, not evidence of leakage. |
| Prediction span | All four selected target files cover January 2020–August 2025 with unique keys and finite selected scores. |
| Quarantine | No September 2025–August 2026 outcome is used in the final historical result. |

An initial mechanical audit incorrectly treated the 777 unavailable outcomes as a timing failure because it required an availability date on every stock-month. The corrected test conditions only on observed labels. It finds zero equal-date or future-information violations. The thesis reports the missing-outcome rate rather than silently deleting it.

## Alternative-data checks

| Source | Result |
|---|---|
| Alpaca news | 128 monthly manifests and 128 monthly JSONL payloads; 1,952,966 articles across 39,129 API pages. Sample payload hashes for January 2015, December 2019, and August 2025 match. Manifests state that credentials were not persisted. |
| December 2019 news | 9,407 articles across 189 pages. |
| LSEG analyst data | `COMPLETE`, 128/128 months. Every receipt was reconciled to the SHA-256 hash of its monthly payload. |
| News features | All 128 model months present. |
| Credential scan | No hard-coded Alpaca credential candidate or workspace `.env` file was found in source, configuration, documentation, or research directories. |

The audit authenticates the downloaded archives and derived monthly coverage. It does not claim that Alpaca covers every economically relevant article or that every security receives stock-specific news.

## Selector checks

The selected models and their frozen historical-test statistics are:

| Target | Frozen selected score | Mean monthly rank IC | HAC t | Validation IC | Validation HAC t |
|---|---|---:|---:|---:|---:|
| CAPM alpha | CAM-free continuous MLP + direct CAM, equal rank blend | 0.0578 | 3.68 | 0.0271 | 1.55 |
| FF5 + momentum alpha | CAM-free pairwise MLP + direct CAM, equal rank blend | 0.0475 | 7.51 | 0.0219 | 2.83 |
| Long net payoff | CAM-free continuous MLP + direct CAM, equal rank blend | 0.0674 | 3.82 | 0.0591 | 2.59 |
| Short net payoff | Correlation-reduced pairwise XGBoost + MLP, equal rank blend | 0.0574 | 3.39 | 0.0256 | 0.97 |

The short-payoff selector’s validation evidence is directionally positive but weak. This is a limitation, not a data-integrity error. It helps explain why the final short sleeve should be interpreted primarily as a jointly optimized hedge rather than a separately proven source of short alpha.

## Portfolio and accounting checks

| Check | Result |
|---|---|
| Profile-month rows | 40 unique rows: 20 months for the 1.0% control and 20 for the 1.5% candidate. |
| Position rows | 9,495 position rows reviewed with unique monthly PERMNOs. |
| Accounting identity | Maximum monthly return reconciliation error: `6.94e-18`. |
| NAV chain | Exact; aggregate discrepancy `$0.00`. |
| Receipt reconciliation | Monthly receipt summaries equal published results. |
| Candidate cap | Newly formed candidate weights respect 1.5% to numerical dust. |
| Inherited position | One February 2024 short reached 1.723% after return drift and was reduced under the frozen stale-position/liquidity rule; it was not a new-position cap violation. |
| ADV | Maximum one-way participation is 10.0000001%; the 10% ADV constraint binds. |
| Turnover | Maximum one-way turnover is 50.0000023%; economically equal to the 50% limit. |
| Factors/industries | Six-factor and SIC2 exposures remain within their frozen limits to immaterial solver dust. |
| Solver | All months report `optimal_inaccurate`; realized constraints and accounting remain economically compliant. |
| V6.3 reproduction | Exact incumbent monthly net-return difference: `0.0`. |

## Final frozen result

| Metric | Result |
|---|---:|
| Annualized net excess return | 14.67% |
| Annualized stressed-cost net excess return | 12.96% |
| Total-return CAGR | 21.34% |
| Cumulative total return | 38.04% |
| Realized annualized volatility | 6.32% |
| Net Sharpe ratio | 2.320 |
| Stressed-cost Sharpe ratio | 2.054 |
| Maximum drawdown | -1.34% |
| Positive months | 75% |
| Own-return HAC t-statistic | 3.487 |
| Mean gross / net exposure | 1.013 / 0.279 |
| Mean long / short names | 59.7 / 62.3 |
| Annualized long / short contribution | +16.28% / -0.54% |
| Annualized primary / stressed cost | 1.07% / 2.77% |

The 1.5% candidate was chosen using February 2022–December 2023 validation data, not January 2024–August 2025 outcomes. Relative to the frozen 1.0% control, its annualized net excess-return improvement is 4.726 percentage points; the paired HAC t-statistic is 5.135 and the three-month block-bootstrap 95% interval is +2.984 to +6.524 percentage points. These statistics describe the observed post-diagnostic sample and do not correct for the full history of research decisions.

SPY compounded 38.02% at a 21.33% CAGR over the same unusually strong 20 months, versus 38.04% and 21.34% for the portfolio. The economically defensible claim is therefore **similar realized return with much lower volatility and drawdown**, not meaningful SPY outperformance.

## Validation-selected 2.05-times thesis model

The maximum-return frontier preserves the frozen stocks, signs, relative monthly weights, selectors, alpha gate, and optimizer output. It applies constant multipliers and recomputes stateful spread, square-root impact, financing, borrow, turnover, and NAV at $100 million. The balanced high-return tier maximizes validation total-return CAGR subject to a 0.75 validation-Sharpe floor and positive stressed-cost excess return. It selects 2.05 times without using January 2024-August 2025 outcomes.

| Metric | Validation | Historical test |
|---|---:|---:|
| Total-return CAGR | 10.99% | 39.17% |
| Annualized net excess return | 7.65% | 29.27% |
| Annualized volatility | 10.17% | 12.96% |
| Sharpe ratio | 0.752 | 2.259 |
| Stressed-cost Sharpe | 0.266 | 1.895 |
| Maximum drawdown | -6.16% | -3.76% |
| Maximum gross exposure | 2.765 | 2.624 |
| Maximum one-way ADV participation | 21.99% | 24.38% |

The historical result exceeds SPY’s 21.33% and QQQ’s 23.26% CAGR. The corresponding active-return block-bootstrap intervals include zero, so realized benchmark outperformance is economically large but not statistically established. The 2.05-times model also violates the original institutional capacity and exposure limits; it is the thesis return frontier, not the $100 million deployable winner and not the Alpaca execution policy.

## Separate $100,000 Alpaca deployment translation

The broker translation is a separate operating study and is not the high-return result reported in the thesis. It reads the authenticated 1.5% position-cap monthly holdings, preserves signs and relative security weights, applies a common 2.05 multiplier, and proportionally caps every rebalance at 1.80 times gross exposure. It does not refit a selector, change the alpha gate, select new securities, or rerun the optimizer. The translation configuration is identity-bound to the frozen maximum-return frontier, and both focused cap-invariance tests pass.

| Metric | Validation, broker-current | Historical test, broker-current |
|---|---:|---:|
| Starting NAV | $100,000 | $100,000 |
| Ending NAV | $114,315 | $154,097 |
| Total-return CAGR | 7.23% | 29.62% |
| Annualized volatility | 9.39% | 11.04% |
| Sharpe ratio | 0.437 | 1.972 |
| Maximum drawdown | -4.94% | -3.37% |
| Annualized modeled cost | 6.19% | 6.10% |
| Annualized financing cost | 4.89% | 4.80% |
| Maximum rebalance gross | 1.800 | 1.800 |
| Maximum month-end drift gross | 2.069 | 1.961 |
| Month-end 1.95x safety breaches | 4 | 1 |

The broker-current case substitutes Alpaca’s published 6.25% non-elite margin rate as of 14 August 2026. It is a sensitivity, not an authenticated historical rate series. Alpaca paper trading also omits material live effects. Two validation month ends exceed the nominal two-times overnight boundary after return drift, so the 1.80 rebalance cap cannot be considered deployable without daily exposure monitoring and proportional deleveraging. The historical market outperformance is genuine within the replay arithmetic, but it is not an untouched confirmation or proof that live orders could reproduce the path.

## Intramonth intervention studies

V6.6-V6.8 reuse the same 1.80-times monthly translation, daily position panel, broker-current cost model, and next-session-close execution. V6.7 adds a compact five-day MLP-XGBoost warning ensemble; V6.8 preserves those exact warnings and reoptimizes only among surviving positions. No V6.8 refill may add a stock, re-enter a warned name, reverse a side, or exceed the gross exposure removed by the contemporaneous warning batch.

| Historical-test policy | Ann. net excess return | Volatility | Sharpe | Maximum drawdown |
|---|---:|---:|---:|---:|
| Monthly 1.80x replay | 21.63% | 11.03% | 1.961 | -8.21% |
| V6.6 five-day XGBoost warning | 23.00% | 12.03% | 1.913 | -6.58% |
| V6.7 MLP-XGBoost warning | 25.18% | 11.81% | 2.132 | -6.47% |
| V6.8 survivor reoptimization | 25.67% | 12.14% | 2.114 | -6.40% |

V6.8 improves annualized return by 4.04 percentage points relative to the monthly control, with paired HAC t 0.92. Its 0.49 percentage-point gain over V6.7 has paired HAC t 0.64 and comes with slightly lower Sharpe. The independent V6.8 audit passes 18/18 identity, warning-schedule, accounting, and survivor-only checks with zero headline-metric and NAV reconciliation differences. These are post-diagnostic mechanism results and do not replace the monthly thesis primary.

## Test status and remaining engineering caveats

After the final-code extraction and legacy removal, the complete retained suite
passes **133 tests**. One prospective-portfolio integration test skips when the
licensed, authenticated historical parquet artifacts are absent; it runs when
those exact hash-bound inputs are mounted. Static import analysis is clean, all
retained modules compile, and every command-line entry point imports and exposes
its help interface. The checked-in V6.8 independent audit still passes 18/18
identity, schedule, accounting, and survivor-only checks.

This supports a reproducible research repository, not a deployment-ready claim.
Before paper trading:

1. recreate the exact environment pinned in `pyproject.toml`;
2. run the final-focused scientific suite and one realistic end-to-end paper-order smoke test;
3. freeze the repository state and external data/source manifest in version control;
4. verify broker-symbol mapping, short availability, order status, partial fills, and realized costs;
5. retain immutable signal-time snapshots and decision logs.

## Final audit judgment

The project is sufficiently coherent and reproducible to support a final thesis manuscript. The evidence supports a qualified positive result: machine-learning rankings contain economically useful cross-sectional information, one frozen cost-aware portfolio translated that information into a strong 20-month historical path, and a validation-selected 2.05-times specification raised realized return without changing security selection. Daily warning and survivor-reoptimization studies improve selected historical economics but do not clear their paired promotion gates. The evidence does **not** yet support a claim of proven production alpha, general regime durability, mechanically safe leverage, or scalable performance above $100 million.
