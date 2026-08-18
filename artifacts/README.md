# External generated artifacts

Large parquet inputs, model checkpoints, monthly portfolios, and generated
reports are intentionally excluded from Git. Final runners authenticate them
against the paths and SHA-256 hashes frozen in `configs/` and their run
manifests.

Small, reviewable result summaries and manifests are committed under
[`../evidence/`](../evidence/). A missing or identity-mismatched external
artifact causes the relevant reproduction or paper-trading command to stop.
