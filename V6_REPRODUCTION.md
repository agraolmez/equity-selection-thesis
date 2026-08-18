# Reproducing the final V6 thesis

This repository retains the complete code path needed to rebuild and confirm
the final V6/V6.1 thesis. Large generated artifacts and licensed CRSP,
Compustat, OptionMetrics, LSEG, Alpaca, and factor inputs are deliberately not
committed to Git. They remain part of the reproduction input, are authenticated
by frozen manifests, and cannot be replaced silently.

## What “from scratch” means

There are three reproducibility layers:

1. **Raw research rebuild:** Phase 01 through Stage 13 rebuild the point-in-time
   predecessor feature, label, risk, validation, and final-test authorities.
2. **V6 rebuild:** the LSEG and Alpaca downloaders, December 2019 repair, news
   builder, 303-feature base builder, and 15-feature engineering extension
   create the frozen 318-feature, 850-column encoded selector input.
3. **Final-result confirmation:** V6 and V6.1 refit the frozen model families;
   the clean selector reproduction rebuilds signals without fitted-model or
   prediction reuse; the frozen portfolio and 1.5% net-return frontier rebuild
   the reported 14.67% annualized net excess return and 2.320 Sharpe result.

Every long stage checkpoints by its natural unit. Resume only into a checkpoint
whose code, config, and input identities match.

## Before running anything

Use Python 3.11 and install the frozen environment:

```bash
python -m pip install -e ".[research,live]"
python scripts/check_v6_reproduction.py --repository-only
python scripts/check_v6_reproduction.py
pytest -q
```

“Ready to start” authenticates the repository and frozen source/feature
authorities. A clean checkout will not yet contain the large V6 and V6.1 model
artifacts; each stage creates the artifact required by the next stage.

The historical frozen builders bind the canonical external database at
`/Users/agraemirolmez/Desktop/Thesis`. Keep that path (or mount the licensed
database there) for an identity-compatible replay. The checker accepts a
different root for inventory inspection, but the frozen builders intentionally
do not rewrite paths or contracts.

## Rebuild sequence

Run the raw pipeline in its numbered order, using the corresponding build and
verify scripts retained under `scripts/`:

```text
Phase 01 raw snapshot
Phase 02 typed canonical sources
Phase 03 feature/label contract
Stage 06 feature store
Stage 07 label store
Stage 08 risk and cost model
Stage 09 selectors
Stage 10 allocation inputs
Stage 11 validation data and execution
Stage 12 frozen final system
Stage 13 final predictors and diagnostics
```

The immutable configs for those stages are under `configs/phase_*`; approval,
source-materialization, and verification files are retained with them. Do not
skip a verifier or substitute a newly discovered input.

Then rebuild the V6 data package in this order:

```bash
python scripts/build_v6_december_2019.py
python scripts/download_lseg_selector_data.py --help
python scripts/download_lseg_usd_repair.py
python scripts/download_v6_alpaca_news.py --start-month 2015-01 --end-month 2025-08
python scripts/materialize_v6_news_features.py
python scripts/materialize_v6_model_features.py
python scripts/materialize_v6_engineered_features.py
python scripts/verify_v6_engineered_features.py
python scripts/check_v6_reproduction.py
```

The LSEG and Alpaca steps require the original licensed access/credentials.
Downloaders publish atomic monthly files and receipts and resume completed
months. The two historical feature builders intentionally have no `--help`
mode: invoking them runs or resumes the build.

Finally, refit and confirm the frozen final lineage:

```bash
python scripts/run_v6_selector_step_01.py --native-runtime --check-only
python scripts/run_v6_selector_step_01.py --native-runtime
python scripts/run_v6_1_selector_step_01.py --native-runtime --check-only
python scripts/run_v6_1_selector_step_01.py --native-runtime
python scripts/reproduce_v6_1_primary_selectors.py --native-runtime
python scripts/run_v6_1_primary_portfolio_reproduction.py
python scripts/run_v6_1_net_return_frontier.py
```

`--native-runtime` uses the currently active frozen Python environment instead
of the original machine-specific interpreter path. Do not change seeds,
targets, feature views, selection rules, costs, or portfolio constraints. That
would open a new research lineage rather than confirm V6.

## Expected confirmation

- Feature input: 278,897 rows, 128 months, January 2015-August 2025, 318 raw
  features, 850 encoded columns.
- V6 run: `841b5f23ca9d-01c2e95ca6b9-dab475636cfc`.
- Clean primary reproduction: identical signals, all 20 monthly positions,
  monthly returns, and aggregate statistics for the frozen three-seed model.
- Final 1.5% frontier candidate: 14.67% annualized net excess return, 6.32%
  volatility, 2.320 net Sharpe, 2.054 stressed-cost Sharpe, and -1.34% maximum
  drawdown over January 2024-August 2025.

These are reproducible post-diagnostic historical results, not untouched or
live evidence. Clean prospective confirmation begins no earlier than September
2026 under the already frozen policy.
