# Research V6 closure

Research V6 is closed as completed selector research on 13 August 2026. Its
published run is:

`artifacts/research_v6/selector_step_01/841b5f23ca9d-01c2e95ca6b9-dab475636cfc`

The run manifest is complete for CAPM alpha, FF5-plus-momentum alpha, long-net,
and short-net. It used the authenticated 318-feature, 128-month input panel
from January 2015 through August 2025, including the exact December 2019
reconstruction. No portfolio or optimizer was constructed.

## Frozen historical-test findings

The January 2024--August 2025 historical test is development evidence rather
than untouched confirmation.

| Target | strongest V6 selector | mean monthly IC | HAC t | economic top-minus-bottom |
|---|---:|---:|---:|---:|
| CAPM alpha | MLP | 0.0359 | 2.35 | 0.0113 |
| FF5+momentum alpha | MLP | 0.0249 | 2.34 | 0.0021 |
| Long-net | Elastic Net by IC; MLP by the full gate | 0.0396 / 0.0381 | 1.42 / 2.82 | -0.0024 / 0.0039 |
| Short-net | dynamic XGBoost+MLP ensemble | 0.0409 | 2.43 | 0.0073 |

Six of the sixteen official model-target evaluations passed the complete V6
interesting-selector gate. XGBoost was weak for the two alpha targets and did
not pass a net-target gate. The MLP was the most consistent individual model.

The five-field CAM-only diagnostic was an Elastic Net trained on LSEG StarMine
Combined Alpha Model level, age, and change fields; it was not the unmodified
vendor CAM rank. A later MLP-plus-CAM diagnostic is retained as explicitly
post-outcome exploratory evidence and cannot amend V6.

## Successor boundary

V6 results, predictions, summaries, checkpoints, and model choices are frozen.
V6.1 is a separate post-diagnostic successor. It may reuse the authenticated
V6 input cache and use V6 predictions only as named frozen comparators. It may
not overwrite V6 or describe V6.1 evidence as untouched confirmation. The
earliest intended clean confirmation remains September 2026 onward under a
fully frozen policy.
