# Historical 15-stage crosswalk

The former README described a sealed Methodology v1 workflow with 15 stages.
V6 is a later successor and is not “Stage 16” or a continuation of that model
selection exercise. The public repository retains the old upstream machinery
only where it is needed to reconstruct the authenticated inputs consumed by
V6.

| Legacy stage | Historical purpose | Retained public implementation | Relationship to final V6 |
|---:|---|---|---|
| 1 | Identity/share-history discovery | Phase 01/02 discovery and verification modules | Upstream data authority |
| 2 | Typed canonical CRSP panel | Phase 02 typed-backbone builders and verifiers | Upstream data authority |
| 3 | Return engine and calendar | Return-reconciliation and calendar builders/verifiers | Supplies executable timing and returns |
| 4 | Investable universe | Signal-date-universe builders/verifiers | Supplies eligible stock-months |
| 5 | Auxiliary datasets | Compustat, OptionMetrics, SEC, ownership, and factor linkage | Supplies point-in-time feature inputs |
| 6 | Feature store | Stage 06 builders/verifiers | Predecessor features consumed by V6 |
| 7 | Labels | Stage 07 builders/verifiers | Supplies authenticated realized targets |
| 8 | Risk and costs | Stage 08 builders/verifiers | Supplies risk, costs, and portfolio inputs |
| 9 | Legacy selectors | Stage 09 selector implementation and verifiers | Historical predecessor only; V6 retrains its own selectors from 2015 |
| 10 | Legacy allocators | Stage 10 allocation implementation and verifiers | Historical predecessor and shared mechanics only |
| 11 | 2015–2019 legacy validation | Stage 11 materialization, execution, and verification | Supplies early V6 predecessor predictors; not V6 model selection |
| 12 | Legacy final freeze | Stage 12 freeze and verification implementation | Authenticates the Stage 13 source lineage |
| 13 | Legacy 2020–2025 final test | `portfolio_bot.final_test` runtime/publication modules | Supplies later predecessor predictors consumed by V6 |
| 14 | Legacy inference and robustness | Frozen inference module and report contract | Not required to build or evaluate V6 |
| 15 | Legacy thesis packaging | Superseded by the checked-in final V6 thesis package | Current thesis reports V6/V6.1, not Methodology v1 |

## Compatibility statement

The repository structurally preserves all 15 historical responsibilities and
compiles their retained modules. The authenticated V6 handoff is additionally
checked against the real external feature authority: 128 months, 278,897 rows,
and the exact frozen V6 build identity.

This does **not** mean that the old 15-stage Methodology v1 workflow was rerun
end to end during repository cleanup. Stages 14 and 15 were never prerequisites
for V6, the legacy Stage 14 empirical program was not promoted into the V6
lineage, and the current Stage 15-equivalent deliverable is the V6 thesis.
Claiming a fresh 15-stage numerical replay would therefore be inaccurate.

For final-result reproduction, follow `V6_REPRODUCTION.md`: rebuild the
upstream authorities through the Stage 13 predictor handoff, then build V6 data,
refit V6/V6.1, and reproduce the frozen portfolio and frontier result.
