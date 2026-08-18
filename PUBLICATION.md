# GitHub publication manifest

The public release is a clean snapshot of the final V6/V6.1 thesis repository.
It should be published to a new repository or an orphan/squashed release branch;
the private development history must not be pushed.

## Included

- `src/portfolio_bot/`: the complete raw-input-to-V6 implementation closure,
  the frozen V6/V6.1 selectors and portfolio, and the closed V6.2-V6.8 mechanism
  studies.
- `scripts/`: acquisition, materialization, verification, reproduction,
  reporting, and fail-closed paper-research entry points.
- `configs/`: frozen contracts, identities, schedules, costs, constraints, and
  upstream Phase 01-Stage 13 policies required by V6.
- `tests/`: focused tests for the retained final lineage.
- `evidence/`: compact result summaries and manifests that support the thesis
  claims without publishing large ledgers or vendor data.
- `research_v6*`, `thesis/`, and `paper_trading/`: research decisions, paper
  source, provenance maps, and the safe operating boundary.
- `README.md`, `V6_REPRODUCTION.md`, `SECURITY.md`, packaging metadata, and the
  GitHub Actions workflow.

## Excluded

- Raw or derived licensed CRSP, Compustat, OptionMetrics, LSEG, Alpaca, factor,
  or broker datasets.
- Credentials, `.env` files, account identifiers, current positions, orders,
  and broker responses.
- Trained model weights, encoded matrices, prediction caches, checkpoints,
  portfolio ledgers, and large generated artifacts.
- Generated Word, PDF, HTML, image, notebook, spreadsheet, archive, and binary
  deliverables.
- Superseded v1-v5 experiments, duplicate reports, scratch work, runtime caches,
  and the private development Git history.

## Frozen absolute paths

Some historical contracts retain `/Users/agraemirolmez/...` paths because their
exact bytes participate in frozen identities. They contain no credential, but
they are machine-specific. Reproduction uses the canonical external mount
described in `V6_REPRODUCTION.md`; changing those frozen files would create a
different lineage.

## Release procedure

1. Run `python scripts/check_publication.py`.
2. Run `python scripts/check_v6_reproduction.py --repository-only`.
3. Run `python -m pytest -q` and build the wheel.
4. Confirm that the all-rights-reserved `LICENSE` remains appropriate.
5. Export only the current tree into a new Git history.
6. Review the exported tree once more before making the GitHub repository
   public.
