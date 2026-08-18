"""Independent, disk-backed reconciliation of Stage-2 auxiliary link payloads.

This module deliberately does not import the primary linkage builder.  It
re-expresses the frozen as-of resolution rules in DuckDB so the independent
verifier can compare every published linkage payload with the normalized link
tables without materializing their high-cardinality joins in Python.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Mapping

import duckdb


class AuxiliaryLinkageReconciliationError(RuntimeError):
    """Raised when the closed linkage inputs cannot be reconciled safely."""


_REQUIRED_PATHS = (
    "auxiliary_signal_linkage",
    "ccm_link_interval",
    "optionmetrics_crsp_link_interval",
    "restatement_event_observation",
    "restatement_crsp_link",
)

_LINKAGE_QUERY = r"""
WITH
linkage AS MATERIALIZED (
  SELECT row_number() OVER () AS verifier_row_id, *
  FROM read_parquet(?)
),
ccm_source AS MATERIALIZED (
  SELECT
    permno,
    link_start_date,
    link_end_date,
    coalesce(link_precedence, 32767) AS precedence,
    gvkey,
    liid,
    linktype,
    linkprim,
    cik
  FROM read_parquet(?)
  WHERE link_allowed
),
ccm_active AS (
  SELECT l.verifier_row_id, c.*
  FROM linkage l
  JOIN ccm_source c
    ON c.permno = l.permno
   AND c.link_start_date <= l.signal_date
   AND (c.link_end_date IS NULL OR l.signal_date <= c.link_end_date)
),
ccm_minimum_precedence AS (
  SELECT verifier_row_id, min(precedence) AS precedence
  FROM ccm_active
  GROUP BY verifier_row_id
),
ccm_best AS (
  SELECT a.*
  FROM ccm_active a
  JOIN ccm_minimum_precedence p
    USING (verifier_row_id, precedence)
),
ccm_rollup AS (
  SELECT
    verifier_row_id,
    count(DISTINCT gvkey) AS gvkey_count,
    first(gvkey ORDER BY liid, linktype) AS chosen_gvkey,
    first(linktype ORDER BY liid, linktype) AS chosen_linktype,
    first(linkprim ORDER BY liid, linktype) AS chosen_linkprim,
    count(DISTINCT cik) FILTER (WHERE cik IS NOT NULL AND cik <> '') AS cik_count,
    min(cik) FILTER (WHERE cik IS NOT NULL AND cik <> '') AS chosen_cik
  FROM ccm_best
  GROUP BY verifier_row_id
),
option_source AS MATERIALIZED (
  SELECT permno, secid, link_start_date, link_end_date, link_status
  FROM read_parquet(?)
),
option_rollup AS (
  SELECT
    l.verifier_row_id,
    count(DISTINCT o.secid) AS secid_count,
    min(o.secid) AS chosen_secid,
    bool_or(o.link_status = 'AMBIGUOUS') AS has_ambiguous,
    bool_or(o.link_status <> 'MATCHED') AS has_nonmatched
  FROM linkage l
  JOIN option_source o
    ON o.permno = l.permno
   AND o.link_start_date <= l.signal_date
   AND l.signal_date <= o.link_end_date
  GROUP BY l.verifier_row_id
),
expected AS (
  SELECT
    l.*,
    CASE
      WHEN c.verifier_row_id IS NULL THEN 'UNMATCHED'
      WHEN c.gvkey_count <> 1 THEN 'AMBIGUOUS'
      ELSE 'MATCHED'
    END AS expected_ccm_status,
    CASE WHEN c.gvkey_count = 1 THEN c.chosen_gvkey END AS expected_ccm_gvkey,
    CASE WHEN c.gvkey_count = 1 THEN c.chosen_linktype END AS expected_ccm_linktype,
    CASE WHEN c.gvkey_count = 1 THEN c.chosen_linkprim END AS expected_ccm_linkprim,
    CASE
      WHEN c.verifier_row_id IS NULL THEN 'UNMATCHED'
      WHEN c.gvkey_count <> 1 THEN 'AMBIGUOUS'
      WHEN c.cik_count > 1 THEN 'AMBIGUOUS'
      WHEN c.cik_count = 1 THEN 'MATCHED'
      ELSE 'UNMATCHED'
    END AS expected_sec_status,
    CASE
      WHEN c.gvkey_count = 1 AND c.cik_count = 1 THEN c.chosen_cik
    END AS expected_sec_cik,
    CASE
      WHEN o.verifier_row_id IS NULL THEN 'NOT_COVERED'
      WHEN o.has_ambiguous OR o.secid_count <> 1 THEN 'AMBIGUOUS'
      WHEN o.has_nonmatched THEN '__INVALID_SOURCE_DISPOSITION__'
      ELSE 'MATCHED'
    END AS expected_option_status,
    CASE
      WHEN NOT o.has_ambiguous AND o.secid_count = 1 AND NOT o.has_nonmatched
      THEN o.chosen_secid
    END AS expected_option_secid,
    coalesce(
      o.verifier_row_id IS NOT NULL
      AND NOT o.has_ambiguous
      AND o.secid_count = 1
      AND o.has_nonmatched,
      false
    ) AS unexpected_option_disposition
  FROM linkage l
  LEFT JOIN ccm_rollup c USING (verifier_row_id)
  LEFT JOIN option_rollup o USING (verifier_row_id)
)
SELECT
  count(*) FILTER (
    WHERE ccm_match_status IS DISTINCT FROM expected_ccm_status
       OR ccm_gvkey IS DISTINCT FROM expected_ccm_gvkey
       OR ccm_linktype IS DISTINCT FROM expected_ccm_linktype
       OR ccm_linkprim IS DISTINCT FROM expected_ccm_linkprim
  )::BIGINT AS linkage_ccm_payload_mismatches,
  count(*) FILTER (
    WHERE sec_match_status IS DISTINCT FROM expected_sec_status
       OR sec_cik IS DISTINCT FROM expected_sec_cik
  )::BIGINT AS linkage_sec_payload_mismatches,
  count(*) FILTER (
    WHERE optionmetrics_match_status IS DISTINCT FROM expected_option_status
       OR optionmetrics_secid IS DISTINCT FROM expected_option_secid
  )::BIGINT AS linkage_optionmetrics_payload_mismatches,
  count(*) FILTER (
    WHERE any_auxiliary_ambiguity IS DISTINCT FROM (
      expected_ccm_status = 'AMBIGUOUS'
      OR expected_sec_status = 'AMBIGUOUS'
      OR expected_option_status = 'AMBIGUOUS'
    )
  )::BIGINT AS linkage_ambiguity_flag_mismatches,
  count(*) FILTER (WHERE unexpected_option_disposition)::BIGINT
    AS optionmetrics_source_disposition_violations
FROM expected
"""

_RESTATEMENT_QUERY = r"""
WITH
events AS MATERIALIZED (
  SELECT restatement_notification_key, public_availability_date
  FROM read_parquet(?)
),
links AS MATERIALIZED (
  SELECT
    restatement_notification_key,
    permno,
    link_status,
    link_basis,
    candidate_count,
    matched_identity_start_date,
    matched_identity_end_date
  FROM read_parquet(?)
),
joined AS (
  SELECT
    e.restatement_notification_key AS event_key,
    l.restatement_notification_key AS link_key,
    e.public_availability_date,
    l.permno,
    l.link_status,
    l.link_basis,
    l.candidate_count,
    l.matched_identity_start_date,
    l.matched_identity_end_date
  FROM events e
  FULL OUTER JOIN links l USING (restatement_notification_key)
)
SELECT
  count(*) FILTER (WHERE event_key IS NULL OR link_key IS NULL)::BIGINT
    AS restatement_event_link_key_mismatches,
  count(*) FILTER (
    WHERE event_key IS NOT NULL
      AND link_key IS NOT NULL
      AND NOT coalesce(CASE
        WHEN link_status = 'MATCHED' THEN
          permno IS NOT NULL
          AND candidate_count = 1
          AND link_basis IN (
            'normalized_cusip8_date_valid',
            'company_key_bridge_date_valid'
          )
          AND matched_identity_start_date IS NOT NULL
          AND matched_identity_end_date IS NOT NULL
          AND matched_identity_start_date <= public_availability_date
          AND public_availability_date <= matched_identity_end_date
        WHEN link_status = 'AMBIGUOUS' THEN
          permno IS NULL
          AND candidate_count > 1
          AND link_basis IN (
            'normalized_cusip8_date_valid',
            'company_key_bridge_date_valid'
          )
          AND matched_identity_start_date IS NULL
          AND matched_identity_end_date IS NULL
        WHEN link_status = 'UNMATCHED' THEN
          permno IS NULL
          AND candidate_count = 0
          AND link_basis = 'UNMATCHED'
          AND matched_identity_start_date IS NULL
          AND matched_identity_end_date IS NULL
        ELSE false
      END, false)
  )::BIGINT AS restatement_link_shape_or_date_violations
FROM joined
"""


def _closed_paths(output_paths: Mapping[str, Path]) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for name in _REQUIRED_PATHS:
        raw = output_paths.get(name)
        if raw is None:
            raise AuxiliaryLinkageReconciliationError(
                f"Required linkage reconciliation input is absent: {name}"
            )
        path = Path(raw).resolve()
        if not path.is_file():
            raise AuxiliaryLinkageReconciliationError(
                f"Required linkage reconciliation input is not a file: {name}"
            )
        paths[name] = path
    return paths


def reconcile_linkage_payloads(output_paths: Mapping[str, Path]) -> dict[str, int]:
    """Recompute every published linkage payload from normalized as-of inputs.

    Counts are row-level mismatches.  The caller should require every returned
    value to be zero.  DuckDB may spill to the temporary directory, keeping the
    reconciliation bounded for the complete signal spine and link histories.
    """

    paths = _closed_paths(output_paths)
    with tempfile.TemporaryDirectory(prefix="auxiliary-linkage-verifier-") as scratch:
        connection = duckdb.connect(":memory:")
        try:
            connection.execute("SET memory_limit = '1GB'")
            connection.execute("SET threads = 1")
            connection.execute("SET preserve_insertion_order = false")
            connection.execute("SET temp_directory = ?", [scratch])
            linkage_row = connection.execute(
                _LINKAGE_QUERY,
                [
                    str(paths["auxiliary_signal_linkage"]),
                    str(paths["ccm_link_interval"]),
                    str(paths["optionmetrics_crsp_link_interval"]),
                ],
            ).fetchone()
            restatement_row = connection.execute(
                _RESTATEMENT_QUERY,
                [
                    str(paths["restatement_event_observation"]),
                    str(paths["restatement_crsp_link"]),
                ],
            ).fetchone()
        except duckdb.Error as exc:
            raise AuxiliaryLinkageReconciliationError(
                "DuckDB linkage reconciliation failed"
            ) from exc
        finally:
            connection.close()

    if linkage_row is None or restatement_row is None:
        raise AuxiliaryLinkageReconciliationError(
            "DuckDB linkage reconciliation returned no aggregate row"
        )
    names = (
        "linkage_ccm_payload_mismatches",
        "linkage_sec_payload_mismatches",
        "linkage_optionmetrics_payload_mismatches",
        "linkage_ambiguity_flag_mismatches",
        "optionmetrics_source_disposition_violations",
        "restatement_event_link_key_mismatches",
        "restatement_link_shape_or_date_violations",
    )
    return {
        name: int(value or 0)
        for name, value in zip(names, (*linkage_row, *restatement_row), strict=True)
    }


__all__ = [
    "AuxiliaryLinkageReconciliationError",
    "reconcile_linkage_payloads",
]
