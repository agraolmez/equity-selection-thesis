# Research V6.2 — direct portfolio learning

V6.2 is a separate, user-authorized successor experiment. It asks whether a
model trained directly on portfolio-level net utility can convert the V6
information set into better implementable weights than the frozen V6.1
rank-then-optimize pipeline.

## What was trained

The model is a linear cross-sectional portfolio policy, not another stock
return or rank predictor. It learns a stock score and feature-conditioned long
and short gross targets from monthly realized stock excess return after
point-in-time trading, borrow, financing, and forecast-risk penalties. The
resulting target weights are only capacity-clipped and proportionally repaired
to the hard risk limits; they are not passed through a separate expected-return
optimizer.

The full primary view uses all 318 registered V6 sources and all 850 encoded
columns. Two prespecified diagnostic views are also reported:

- `cam_free`: 313 raw sources and 840 encoded columns, excluding the five
  direct LSEG StarMine Combined Alpha Model fields;
- `correlation_reduced`: 202 raw sources and 618 encoded columns, using the
  frozen outcome-blind pre-2020 redundancy screen.

Missing values use the existing V6 neutral-value plus explicit-missingness
encoding. No complete-case stock filter is introduced. The authentic December
2019 feature, outcome, and risk/cost rows enter training after their label
availability date; nothing is interpolated.

## Comparisons and chronology

Each feature view has an ungated direct policy and the identical fitted policy
with the frozen causal FF5+momentum sign gate applied at portfolio formation.
The gate becomes active only after 24 completed forecast months. The paired
test model files are byte-identical within feature view and refit year, so the
comparison isolates the overlay.

Models fit from March 2017 through November 2019, profiles are selected on
January 2020 through December 2023, and the historical test is January 2024
through August 2025 with annual expanding refits. Realized labels are admitted
only after their availability date. September 2025 through August 2026 remains
quarantined, and untouched confirmation can start no earlier than September
2026.

## Completed result

The final corrected run is complete at
`artifacts/research_v6_2/direct_portfolio_step_01/99cf7f2098b7-f0bf8995d641-872cb86dba16`.
All 12 validation candidates and 120/120 test variant-months completed. The
full-feature primary direct policy has a 0.888 net Sharpe; adding the FF5 gate
raises it to 0.934. The best diagnostic path is CAM-free plus FF5 at 1.004,
but it does not beat the frozen V6.1 Step 02 optimizer and is not promoted.

The full result and audit are in
[`DIRECT_PORTFOLIO_STEP_01_RESULT.md`](DIRECT_PORTFOLIO_STEP_01_RESULT.md).
