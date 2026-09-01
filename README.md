# V6 Machine-Learning Portfolio Thesis

This is the final, simplified research repository. The thesis is the V6/V6.1
lineage: point-in-time stock selection with XGBoost and MLP models, followed by
a cost- and risk-aware long/short portfolio. Earlier v1-v5 frameworks and the
old 2000-2025 narrative are not the final design and have been removed.

## Final research design

V6 rebuilt the selector data and retrained every candidate; it did **not** reuse
predecessor model checkpoints. The authenticated panel contains 278,897
stock-month rows, 128 months, 318 raw features, and 850 encoded columns. The
feature package includes point-in-time Alpaca news and LSEG company-analyst
metrics (CAM), alongside market, accounting, risk, liquidity, ownership, and
identity features.

| Period | Role |
|---|---|
| January 2015-November 2019 | Initial V6 model training |
| January 2020-December 2023 | Validation and leakage-safe model/portfolio selection |
| January 2024-August 2025 | Historical test reported in the thesis |
| September 2025-August 2026 | Quarantined while the final policy was developed |
| September 2026 onward | Earliest clean prospective confirmation |

The final selector uses the equal ensemble of the CAM-free MLP and CAM model
for FF5+momentum, CAPM, and long-net targets. The short-net target instead uses
the equal correlation-reduced XGBoost/MLP ensemble because direct CAM features
were not reliable on the short side.

## Thesis result

The primary result is the validation-selected V6.1 portfolio with a 1.5%
candidate position cap at $100 million NAV. From January 2024 through August
2025 it produced:

- 14.67% annualized net excess return;
- 6.32% annualized volatility;
- 2.320 net Sharpe and 2.054 stressed-cost Sharpe;
- -1.34% maximum drawdown;
- 21.34% total-return CAGR; and
- a 4.73 percentage-point annualized net improvement over the reproduced
  control (HAC t = 5.14).

The separate 2.05x high-return research scaling produced 39.17% CAGR and a
2.259 Sharpe, but exceeded the original exposure and capacity limits. It is a
qualified research result, not the deployable thesis primary.

V6.3 did not beat the 2.320-Sharpe incumbent. V6.6-V6.8 tested intramonth exits,
a daily MLP/XGBoost overlay, and survivor re-optimization. None passed the
frozen promotion gates. The monthly V6.1 portfolio therefore remains primary;
V6.7/V6.8 are shadow or paper-research mechanisms only. V6.8 paper execution
is fail-closed when current authenticated signals, source data, broker state,
or credentials are absent.

## Post-thesis V6.8 replay

After the thesis was closed, the unchanged V6.8 policy was reconstructed
through July 2026 as an operational diagnostic. Stitching its January
2024-August 2025 historical-development returns to the separately rebuilt
September 2025-July 2026 segment produces a 111.63% cumulative net total return,
a 33.67% CAGR, and a 2.028 monthly net-excess Sharpe over 31 holding months.

This is **not** an additional thesis result, a live track record, or untouched
prospective confirmation. The later segment was reconstructed retrospectively;
July 2026 uses documented factor approximations, and borrow eligibility remains
modelled rather than supported by a contemporaneous locate archive. The exact
boundary, monthly returns, formulas, source hashes, and limitations are in
[`POST_THESIS_REPLAY.md`](POST_THESIS_REPLAY.md).

The concise source-backed result files are in [`evidence/`](evidence/). The
complete paper is [`thesis/MASTER_DRAFT.md`](thesis/MASTER_DRAFT.md), with claim
provenance in [`thesis/EVIDENCE_MAP.md`](thesis/EVIDENCE_MAP.md).

For the exact public-release contents and exclusions, see
[`PUBLICATION.md`](PUBLICATION.md). Security and credential handling are covered
in [`SECURITY.md`](SECURITY.md).

The earlier README's 15 stages belong to the sealed Methodology v1 workflow,
not to V6. Their retained responsibilities and the exact V6 handoff are
explained in [`LEGACY_15_STAGE_CROSSWALK.md`](LEGACY_15_STAGE_CROSSWALK.md).

## Repository map

- `src/portfolio_bot/research_v6/`: selector data, features, preprocessing,
  models, evaluation, and checkpointed runtime.
- `src/portfolio_bot/research_v6_1/`: final feature views, models, and portfolio
  optimizer.
- `src/portfolio_bot/research_v6_3/`: bounded signal-to-weight challenger.
- `src/portfolio_bot/research_v6_8/`: prospective data, portfolio, broker, and
  fail-closed paper-trading components.
- `src/portfolio_bot/data`, `features`, `risk_cost`, `selectors`, `allocation`,
  `validation`, `final_freeze`, and `final_test`: the retained Phase 01-Stage 13
  upstream dependency closure used to construct the V6 inputs from raw data.
- `scripts/`: upstream builders/verifiers, V6 data acquisition and feature
  materialization, final V6/V6.1 runs, and closed V6.2-V6.8 studies.
- `configs/`: immutable experiment and live/paper contracts.
- `research_v6*` and `paper_trading/`: compact decision records.
- `tests/`: focused final-lineage tests.
- `evidence/`: small checked-in manifests and result summaries. Large parquet
  inputs, trained models, checkpoints, and generated reports remain external.

The repository is roughly 354,000 source/configuration/test/document/evidence
lines, down from the approximately 640,000-line working research archive. This
is larger than the earlier lean cut because the complete raw-data-to-V6
dependency closure has been restored. Generated artifacts, superseded v1-v5
experiments, duplicate reports, caches, and vendored data remain excluded.

## Install and test

Python 3.11 or newer is required.

```bash
python -m pip install -e ".[research]"
pytest -q
```

GitHub CI runs the publication audit, repository-closure check, compilation,
and focused test suite without requiring licensed data or credentials.
The suite has one expected external-artifact skip: the prospective V6.8
portfolio integration test requires authenticated historical artifacts that
cannot be distributed on GitHub.

Install live LSEG support only for authenticated prospective-data work:

```bash
python -m pip install -e ".[live]"
```

## Reproduction boundary

The public repository now includes the complete code needed to rebuild V6 from
the licensed raw research inputs, not merely replay from a frozen feature
matrix. The data themselves are not committed. Their paths and SHA-256
identities are frozen in configs and manifests; runners stop rather than
silently substitute mismatched inputs.

Start with the safe, read-only closure check and the full runbook:

```bash
python scripts/check_v6_reproduction.py --repository-only
python scripts/check_v6_reproduction.py
```

See [`V6_REPRODUCTION.md`](V6_REPRODUCTION.md) for the raw Phase 01-Stage 13,
V6 feature, model-refit, clean-selector, portfolio, and result-confirmation
sequence.

Core entry points:

```bash
python scripts/run_v6_selector_step_01.py --native-runtime --check-only
python scripts/run_v6_1_selector_step_01.py --native-runtime --check-only
python scripts/run_v6_1_portfolio_step_02.py --help
python scripts/run_v6_1_net_return_frontier.py --help
python scripts/run_v6_1_maximum_return_scaling_frontier.py --help
python scripts/audit_v6_8_daily_reoptimization.py --help
```

Long computations checkpoint atomically by natural units of progress. Resume
is allowed only when configuration, code, and input identities match.

## Interpretation

All 2024-August 2025 figures are historical development evidence, not live
performance or untouched confirmation. The final design was selected using
realized evidence through August 2025 and later post-diagnostic studies.
Prospective confirmation beginning no earlier than September 2026 is still
required.

## License

Copyright © 2026 Ağra Emir Ölmez. All rights reserved. The repository is public
for viewing and academic review; reuse requires prior written permission. See
[`LICENSE`](LICENSE) for the complete terms.
