# Checked-in result evidence

This directory contains the small, reviewable result summaries and run
manifests cited by the thesis. Files are copied unchanged from the authoritative
completed runs. They are evidence, not inputs to a new model-selection pass.

`v6_selector/feature_registry.csv` is the portable, metadata-only definition of
the frozen 318-feature input used by repository tests. It contains feature
names and provenance rules, but no security observations or licensed values.

`post_thesis_v6_8_replay/` is a later operational addendum, not thesis evidence.
It publishes the 31 monthly returns behind the January 2024-July 2026 V6.8
diagnostic, together with formulas, source hashes, and required limitations.

Large parquet ledgers, fitted models, checkpoints, and licensed source data are
excluded from Git. Their identities remain frozen in the configs and manifests.
If a full external artifact conflicts with a prose summary, the authenticated
machine-readable artifact controls.
