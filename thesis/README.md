# Thesis package

This directory contains the submission-focused manuscript and the deterministic
sources used to rebuild its figures, report, and Bocconi-formatted document.

## Source of truth

- `MASTER_DRAFT.md`: complete substantive manuscript.
- `EVIDENCE_MAP.md`: claim-to-result provenance.
- `REPRODUCIBILITY_AUDIT.md`: scientific, accounting, and engineering audit.
- `references.bib`: cited bibliography.
- `build_figures.py`: deterministic thesis-figure builder.
- `build_report_artifact.py` and `report/artifact.json`: portable reading-copy
  source.
- `bocconi/BOCCONI_FINAL_PAPER.md`: submission-body source.
- `bocconi/build_bocconi_final_paper.py`: formatted DOCX builder.
- `bocconi/ABSTRACT_FOR_ONLINE_FORM.txt`: separate abstract.
- `bocconi/FORMAT_COMPLIANCE.md`: formatting checklist.

Generated DOCX/PDF/HTML files and figure images are intentionally excluded from
the simplified repository. Rebuild them from these sources in the pinned
research environment.

## Reported result

The manuscript reports both the realistic 1.5%-position-cap, $100 million V6.1
reference and the validation-selected 2.05x high-return research model. The
reference earns a 2.320 historical-test Sharpe. The 2.05x model earns a 39.17%
CAGR and 2.259 Sharpe but exceeds the original capacity and exposure limits.
V6.6-V6.8 remain post-diagnostic robustness studies and are not promoted over
the monthly primary.

The simplified checkout passes 133 retained tests. One external-artifact
integration test skips unless the exact licensed/frozen historical inputs are
mounted. See `REPRODUCIBILITY_AUDIT.md` for the full boundary.
