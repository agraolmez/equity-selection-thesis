# Research V6.4 Negative-Alpha Selector Result

## Scope

Research V6.4 is a separate, selector-only, post-diagnostic experiment. It
tests whether a dedicated model can rank stocks by subsequently negative
FF5+momentum alpha better than reversing the frozen V6.1 positive-alpha score.
It does not construct a portfolio or run an optimizer.

The target is the exact sign reversal of the existing FF5+momentum alpha rank,
so a higher V6.4 score means a more attractive candidate for the short side.
The experiment repeats the V6.1 selector process across:

- full, direct-CAM-free, and outcome-blind correlation-reduced feature views;
- XGBoost, MLP, and Elastic Net models;
- pairwise and continuous-rank objectives where supported; and
- prespecified equal-weight and causal dynamic ensembles, including CAM.

Model choice uses January 2020--December 2023 validation only. January
2024--August 2025 is a historical test and is not used for selection.

## Result

Validation selected the equal-weight ensemble of CAM-free XGBoost and the
sign-reversed direct CAM score. Its mean monthly rank IC was 0.02394 versus
0.02186 for the frozen reversed-alpha control. The paired difference was only
0.00208 with HAC t = 0.32, so validation did not establish a reliable
improvement.

On the 20-month historical test, the selected challenger fell to a mean monthly
rank IC of 0.01112 (HAC t = 0.88), while the frozen reversed-alpha control
reached 0.04753 (HAC t = 7.51). The challenger's paired IC deficit was -0.03641
with HAC t = -2.52.

The tail evidence agrees with the IC comparison. In the challenger's top 10%
short-candidate tail, 50.25% of stocks realized negative FF5+momentum alpha and
mean realized alpha was +0.454% per month. For the frozen control, 54.40% were
negative-alpha stocks and mean realized alpha was -0.548% per month. The
selected challenger therefore failed the economically relevant downside-tail
test.

A CAM-free MLP plus CAM ensemble was descriptively strong on the historical
test (IC 0.04222), but it was not selected on validation and remained below the
frozen control. It cannot be promoted after observing the test period.

## Decision

`RETAIN_FROZEN_REVERSED_FF5_ALPHA_CONTROL_NO_V6_4_PORTFOLIO_TEST`

The dedicated negative-alpha selector does not replace or augment the current
V6.1 FF5+momentum alpha gate. V6.4 is closed as a completed selector mechanism
study. Any portfolio experiment using a V6.4 challenger would require a new,
explicitly authorized lineage and cannot be justified by the current result.

## Reproducibility

- Contract: `configs/research_v6_4/negative_alpha_selector_step_01.json`
- Runner: `scripts/run_v6_4_negative_alpha_selector_step_01.py`
- Completed run:
  `artifacts/research_v6_4/negative_alpha_selector_step_01/f249781d221c-16046ee58521-b46b9c024d0e`
- Run status: `COMPLETE`
- Unit checks: 3 passed
- Prediction rows: 155,226
- Monthly cohorts: 68 (48 validation, 20 historical test)
- Duplicate stock-month keys: 0
- Missing model scores: 0
- Label-shuffle smoke control: passed
