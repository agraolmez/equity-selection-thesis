# Research V6 — selector-first successor

V6 is a completed, frozen research lineage. It does not modify or reinterpret
closed V3, V4, V5, or V5.1 evidence. Its current scope ends at the selector
comparison. The selector backtest completed for all four targets; no V6
portfolio or optimizer was run. The exact findings and successor boundary are
recorded in [`CLOSURE.md`](CLOSURE.md).

## Current status

The V6 model-input audit and selector run are complete and the data status is
`BACKTEST_READY`. The canonical panel covers all 128 holding months from
January 2015 through August 2025, including an exact December 2019
reconstruction with no interpolation.

- Eligible stock-month rows: 278,897.
- Primary feature view: 318 raw features (303 authenticated base plus 15
  outcome-blind engineered features).
- Predecessor economic features: 158.
- LSEG analyst and expectation features: 89, including 52 currency-sensitive
  fields downloaded with `Curn=USD`.
- Alpaca related-news features: 56.
- December 2019: 2,051 eligible feature rows and 2,050 valid realized-label
  rows.
- Data checks: 128/128 canonical monthly files and receipts, zero duplicate
  stock-month keys, zero infinities, and zero all-empty included features.

The exact registry is stored at
`/Users/agraemirolmez/Desktop/Thesis/processed/research_v6/model_features_v2/feature_registry.csv`.
The full audit and feature policy are documented in
[`FEATURE_AUDIT.md`](FEATURE_AUDIT.md).

The executed frozen selector contract is
[`selector_step_01.json`](../configs/research_v6/selector_step_01.json). Its
full encoded input cache is prepared and receipt-bound: 278,897 rows by 850
finite columns, with all four targets and 2,051 December 2019 feature rows.
published run is
`artifacts/research_v6/selector_step_01/841b5f23ca9d-01c2e95ca6b9-dab475636cfc`.
Re-running or retuning V6 is no longer authorized; successor work belongs to
V6.1.

## Frozen selector comparison

For each CAPM-alpha, FF5-plus-momentum-alpha, long-net, and short-net target,
V6 compares:

- pairwise XGBoost and a continuous centered-rank XGBoost control;
- pairwise RankNet MLP and a continuous centered-rank MLP control;
- a causal dynamic ensemble of the validation-selected XGBoost and MLP; and
- a causally tuned Elastic Net benchmark.

The continuous controls predict the numerical centered monthly rank, not an
exact percentage return. XGBoost depth is tuned over 1, 2, and 3. This follows
the shallow-weak-learner design in Gu, Kelly, and Xiu and closely aligned
cross-sectional equity implementations, which use boosted-tree depths 1--3.
V5 used depth 3 and V5.1 used depths 2--3; depth 5 is therefore not part of the
V6 grid.

Validation is 2020--2023 and selects the objective and operating policy.
Prespecified operating candidates compare annual versus quarterly refits,
annual versus biennial retuning, and expanding versus rolling 60-month
history. Every retune uses only a trailing 12-month internal holdout whose
labels are available by the refit signal. The chosen family policies are then
locked for the January 2024--August 2025 historical test. That period remains
historical development evidence, not untouched confirmation.

Dynamic ensemble weights use a strictly prior 12-month-half-life IC history,
require 24 months before adapting, and stay within 20%--80%. Negative component
quality is floored at zero; if neither component has positive prior quality,
weights are 50/50. The current month's outcome is never used.

The historical runner used the stable isolated XGBoost runtime, ran two targets
in parallel, checkpointed each refit block, and resumed compatible work. The
command below is retained for reproducibility but must not be used to rescue or
retune the closed V6 lineage:

```bash
python scripts/run_v6_selector_step_01.py
```

Readiness without fitting can be checked with `--check-only`; the full input
cache can be authenticated or prepared with `--prepare-only`. Neither command
starts a model backtest.

## Prespecified information views

The primary view supplies all 318 registered features to each model. Feature
selection is not performed by a manually chosen subset: Elastic Net may shrink
uninformative encoded inputs to zero, while XGBoost and the neural network use
their own regularization and learned importance.

Four information views are registered before outcomes are inspected:

1. `primary_full`: all 318 features.
2. `cam_free`: 313 features, excluding LSEG's five Combined Alpha Model fields.
3. `cam_only`: the five CAM level/age/change fields as a vendor-model benchmark.
4. `news_free`: all non-news features, used only to measure Alpaca's incremental
   contribution.

These views test whether the third-party composite model adds information; they
are not permission to choose a feature set after seeing the final test results.

## Missing values and timing

Complete-case filtering is not used: no stock-month has every one of the 318
raw fields observed. The full eligible universe is retained.

- Continuous firm inputs are clipped at the monthly 1st/99th percentiles,
  converted to centered cross-sectional ranks, assigned neutral zero when
  missing, and paired with a missingness indicator.
- Binary inputs retain 0/1, use neutral zero when missing, and receive a
  missingness indicator.
- Categorical inputs use the frozen predecessor vocabulary and an explicit
  unknown state.
- Aggregate state inputs use strictly past expanding standardization, are
  clipped to ±5, and receive missingness indicators.

All inputs are bounded by the monthly signal-close cutoff. Alpaca articles are
usable only when both creation and update timestamps are no later than the
cutoff. LSEG values use the exact authenticated signal date. Raw vendor dates
are represented only as causal information ages. Local-currency LSEG levels
are not model inputs.

Raw news remains outside the repository at
`/Users/agraemirolmez/Desktop/Thesis/raw/research_v6/alpaca_news`. LSEG's
authenticated USD repair is under
`/Users/agraemirolmez/Desktop/Thesis/raw/research_v6/lseg_usd_v1`. Credentials
are not persisted in either feature output or repository files.

## Scope boundary

The older preliminary selector screen used only predecessor inputs and is not
a result for the completed 318-feature V6 dataset. It is retained as inactive
historical engineering evidence. The real walk-forward selector run is
complete and frozen. Portfolio construction and optimization remain out of
scope.
