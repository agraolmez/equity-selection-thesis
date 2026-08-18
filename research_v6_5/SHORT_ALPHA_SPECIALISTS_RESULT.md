# Research V6.5 Side-Specific Short-Alpha Specialists

## Question and design

Research V6.5 tests the asymmetric design requested by the user: retain the
existing positive-alpha models for the long side, while training separate
models whose target is specifically the negative-alpha tail for the short
side. This is not a sign reversal of the positive-alpha rank.

For each month and each alpha definition, the short target ranks
`max(-realized alpha, 0)`. Stocks with zero or positive alpha tie at zero;
increasingly negative alpha receives an increasingly attractive short rank.
The experiment covers both CAPM alpha and FF5+momentum alpha and repeats the
complete selector comparison across:

- full, direct-CAM-free, and outcome-blind correlation-reduced feature views;
- XGBoost, MLP, and Elastic Net;
- pairwise and continuous-rank objectives where supported; and
- all prespecified equal and causal dynamic ensembles, with the frozen
  reversed V6.1 alpha selector retained as the control.

Model selection uses January 2020--December 2023 validation only. January
2024--August 2025 is historical test evidence and cannot change the selected
model. No portfolio or optimizer was constructed.

## CAPM short specialist

Validation selected the causal dynamic ensemble of CAM-free XGBoost, MLP,
Elastic Net, and direct CAM. Its downside-target mean monthly IC was 0.10871,
versus 0.05296 for the frozen reversed CAPM control. The paired improvement
was 0.05575 with HAC t = 2.73. Its validation top-decile candidates also had
more negative mean CAPM alpha (-0.508% versus -0.289% per month) and a slightly
higher negative-alpha fraction (52.90% versus 52.24%). It therefore passed all
prespecified validation promotion conditions.

The validation-selected challenger retained higher downside-target IC in the
20-month historical test: 0.12130 versus 0.07975, a paired improvement of
0.04155 with HAC t = 2.48. However, the economically decisive top-decile
comparison reversed. The challenger still found negative-alpha stocks on
average, but its top decile realized -0.650% mean CAPM alpha per month versus
-1.653% for the frozen control. Its negative-alpha fraction was also lower,
54.59% versus 57.41%. The CAPM specialist therefore did not confirm as a
better short gate.

The selected challenger and reversed control were not duplicates. Their mean
top-decile overlap was 46.62% in validation and 53.22% in the historical test.

## FF5+momentum short specialist

Validation selected correlation-reduced Elastic Net. Its downside-target mean
monthly IC was 0.09703, versus 0.04899 for the frozen reversed FF5 control.
The paired improvement was 0.04804 with HAC t = 5.50.

Despite that strong rank statistic, it failed the validation tail conditions.
The selected top decile realized slightly positive mean FF5+momentum alpha
(+0.059% per month) versus -0.082% for the control, and its negative-alpha
fraction was lower (51.64% versus 52.42%). The historical test reinforced the
failure: the challenger IC was 0.10807 versus 0.07356, but its selected tail
realized +0.877% mean alpha per month and only 49.88% negative-alpha stocks.
The control's tail realized -0.548% mean alpha and contained 54.40%
negative-alpha stocks.

This apparent IC/tail tension is not a contradiction. The downside target
ties every non-negative-alpha observation at the same low target value. A
model can separate the broad negative-alpha group from the tied remainder and
therefore earn high cross-sectional IC without placing the most economically
negative observations in its extreme top decile. Short implementation needs
the latter.

The FF5 challenger was clearly distinct from the reversed control: mean
top-decile overlap was 27.70% in validation and 28.68% in the historical test.

## Decision

`RETAIN_FROZEN_REVERSED_CAPM_AND_FF5_CONTROLS_NO_V6_5_PORTFOLIO_TEST`

V6.5 confirms that separately trained long and short selectors are a coherent
and materially different design. It does not show that the new short
specialists improve the actual negative-alpha tail. The CAPM challenger
passed validation but failed historical tail confirmation; the FF5 challenger
failed the tail test already in validation. Neither is eligible to replace or
augment the current V6.1 alpha gate, and no portfolio test is justified.

## Reproducibility

- Contract:
  `configs/research_v6_5/short_alpha_specialists_step_01.json`
- Runner:
  `scripts/run_v6_5_short_alpha_specialists_step_01.py`
- Completed run:
  `artifacts/research_v6_5/short_alpha_specialists_step_01/6eaede17732d-16046ee58521-b5eb1bef540b`
- Run status: `COMPLETE`
- Prediction rows per target: 155,226
- Monthly cohorts: 68 (48 validation, 20 historical test)
- Duplicate stock-month keys: 0
- Missing official model scores: 0
- Label-shuffle controls: passed for both targets
- Portfolio constructed: no
- Optimizer run: no
