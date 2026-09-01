# V6.8 post-thesis replay evidence

These files support the separately labelled post-thesis V6.8 operational
diagnostic. They do not extend the thesis evaluation period and are not
prospective or broker-realized performance.

- `monthly_returns.csv`: 31 unique, continuous holding months from January 2024
  through July 2026.
- `summary.json`: metric definitions, source hashes, evidence boundary, and
  limitations.

Run `python -m pytest -q tests/unit/test_post_thesis_v6_8_evidence.py` to
recompute the headline figures from the CSV.
