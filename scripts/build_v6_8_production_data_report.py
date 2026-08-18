#!/usr/bin/env python3
"""Build the reproducible notebook and canonical report artifact for V6.8 data readiness."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import nbformat as nbf
import pandas as pd
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/research_v6_8/production_data_readiness/20260816"


CLASS_LABELS = {
    "SAME_SOURCE_REFRESH_PATH": "Same-source refresh path",
    "PROVIDER_BRIDGE_REQUIRES_PARITY": "Provider bridge · parity required",
    "LICENSE_OR_RETRAIN_REQUIRED": "License or retraining required",
    "CAUSAL_NOWCAST_OR_MISSING_POLICY_REQUIRED": "Causal nowcast or missing policy",
    "PRESERVE_FROZEN_STRUCTURAL_MISSINGNESS": "Preserve structural missingness",
}


def _source_query(description: str, tables: list[str]) -> dict:
    return {
        "engine": "DuckDB",
        "language": "sql",
        "sql": f"SELECT * FROM {tables[0]}",
        "description": description,
        "tables_used": tables,
    }


def build_artifact() -> dict:
    matrix = pd.read_csv(OUTPUT / "feature_source_matrix.csv")
    groups = pd.read_csv(OUTPUT / "source_group_summary.csv")
    readiness = json.loads((OUTPUT / "readiness.json").read_text("utf-8"))
    probe_path = ROOT / "artifacts/research_v6_8/lseg_production_probe/20260816/receipt.json"
    probe = json.loads(probe_path.read_text("utf-8"))
    replay = probe["evidence"]["same_source_historical_replay"]
    market_pilot = probe["evidence"]["market_price_bridge_pilot"]
    market_parity_path = (
        ROOT / "artifacts/research_v6_8/lseg_market_parity/20260817/receipt.json"
    )
    market_parity = json.loads(market_parity_path.read_text("utf-8"))
    market_monthly = pd.read_csv(market_parity_path.with_name("monthly_summary.csv"))
    market_minimum_rank_correlation = float(
        market_monthly["minimum_rank_correlation"].min()
    )
    derived_path = (
        ROOT
        / "artifacts/research_v6_8/lseg_market_derived_parity/20260817/feature_summary_receipt.json"
    )
    derived = json.loads(derived_path.read_text("utf-8"))
    derived_features = pd.read_csv(
        derived_path.with_name("feature_window_summary.csv")
    )
    derived_features["parity_status"] = derived_features[
        "feature_level_parity_pass"
    ].map({True: "Pass", False: "Fail"})
    market_formula_status = pd.DataFrame(
        [
            {
                "status": "Feature-level parity pass",
                "feature_count": int(derived["features_passing_feature_level_gate"]),
            },
            {
                "status": "Feature-level parity fail",
                "feature_count": int(derived["features_failing_feature_level_gate"]),
            },
            {
                "status": "Cross-sectional gate pending",
                "feature_count": int(len(derived["excluded_cross_sectional_features"])),
            },
        ]
    )
    groups["refresh_label"] = groups["refresh_class"].map(CLASS_LABELS)
    class_counts = (
        groups.groupby(["refresh_class", "refresh_label"], sort=False)["feature_count"]
        .sum()
        .reset_index()
        .sort_values("feature_count", ascending=False)
    )
    provider_rows = [
        {"check": key.replace("_", " ").title(), "value": value}
        for key, value in readiness["provider_checks"].items()
    ]
    generated = datetime.now(timezone.utc).isoformat()
    sources = [
        {
            "id": "frozen_registry_audit",
            "label": "Frozen V6 feature registry, production source contract, and parity receipts",
            "query": _source_query(
                "Verified the immutable registry hash, assigned every model ordinal exactly once to one live-source policy, and incorporated the current LSEG replay and market-parity receipts.",
                [
                    "research_v6.model_features_v2.feature_registry",
                    "research_v6_8.production_feature_sources_v2",
                    "research_v6_8.lseg_production_probe_v1",
                    "research_v6_8.lseg_market_parity_v1",
                    "research_v6_8.lseg_market_derived_feature_summary_v1",
                ],
            ),
        },
        {
            "id": "local_provider_preflight",
            "label": "Secret-free local provider and archive preflight",
            "query": _source_query(
                "Checked only credential presence flags, licensed desktop process state, local receipt coverage, and legacy-source file presence. No secret values were read into the report.",
                ["research_v6_8.production_data_readiness_v4"],
            ),
        },
        {
            "id": "lseg_capability_probe",
            "label": "Checkpointed LSEG capability, replay, and 20-month market parity evidence",
            "query": _source_query(
                "Replayed archived point-in-time LSEG fields, recorded capability counts for current content families, compared 10 frozen market primitives, and audited 47 stock-level market formulas on 20 deterministic monthly samples.",
                [
                    "research_v6_8.lseg_production_probe_v1",
                    "research_v6_8.lseg_market_parity_v1",
                    "research_v6_8.lseg_market_derived_feature_summary_v1",
                ],
            ),
        },
        {
            "id": "alpaca_bars_docs",
            "label": "Alpaca historical stock bars",
            "href": "https://docs.alpaca.markets/us/v1.4.2/reference/stockbars",
        },
        {
            "id": "alpaca_news_docs",
            "label": "Alpaca historical news data",
            "href": "https://docs.alpaca.markets/us/docs/historical-news-data",
        },
        {
            "id": "alpaca_options_docs",
            "label": "Alpaca historical option data",
            "href": "https://docs.alpaca.markets/us/docs/historical-option-data",
        },
        {
            "id": "lseg_docs",
            "label": "LSEG Data Library through Workspace",
            "href": "https://developers.lseg.com/en/api-catalog/lseg-data-platform/lseg-data-library-for-python/tutorials/content-tutorials/pricing/realtime-snapshot",
        },
        {
            "id": "lseg_opra_docs",
            "label": "LSEG OPRA exchange-derivatives coverage",
            "href": "https://www.lseg.com/en/data-catalogue/derivatives/pricing/exchange-derivatives-options-price-reporting-authority-opra",
        },
        {
            "id": "lseg_fundamentals_docs",
            "label": "LSEG Company Fundamentals and Estimates",
            "href": "https://developers.lseg.com/en/article-catalog/article/fundamentals-estimates-dcf",
        },
        {
            "id": "sec_docs",
            "label": "SEC EDGAR application programming interfaces",
            "href": "https://www.sec.gov/search-filings/edgar-application-programming-interfaces",
        },
        {
            "id": "french_docs",
            "label": "Kenneth R. French Data Library",
            "href": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html",
        },
        {
            "id": "yfinance_docs",
            "label": "yfinance project usage notice",
            "href": "https://github.com/ranaroussi/yfinance/blob/main/README.md",
        },
    ]
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "V6.8 Paper Trading Data Readiness",
            "description": "Feature-by-feature decision record for reconstructing the frozen V6.8 inputs prospectively without silent vendor substitution.",
            "generatedAt": generated,
            "blocks": [
                {
                    "id": "title",
                    "type": "markdown",
                    "layout": "full",
                    "body": "# V6.8 Paper Trading Data Readiness",
                },
                {
                    "id": "decision",
                    "type": "markdown",
                    "layout": "full",
                    "sourceId": "frozen_registry_audit",
                    "body": "## Technical summary\n\n**LSEG access is sufficient to continue engineering, but the frozen model is still blocked from paper orders.** The 20-month primitive gate passes 20/20 months with 400/400 consolidated mappings. The broader stock-level audit resolves 399/400 instrument-months and gives feature-level parity to 34 of 47 tested market features; 13 fail the prespecified 95%-of-months rule and four cross-sectional features remain pending. The clearest definition failure is share turnover, because LSEG issue shares can combine share classes while CRSP uses the traded security line. Fundamentals, ownership, OPRA aggregates, causal factor timing, the current signal package, and paper credentials also remain incomplete.",
                },
                {
                    "id": "headline_metrics",
                    "type": "metric-strip",
                    "layout": "full",
                    "cardIds": [
                        "feature_schema",
                        "same_source",
                        "bridge_required",
                        "hard_source_policy",
                    ],
                },
                {
                    "id": "source_split_finding",
                    "type": "markdown",
                    "layout": "full",
                    "sourceId": "frozen_registry_audit",
                    "body": "## LSEG covers the raw ingredients; definition parity is now the constraint\n\nLSEG can refresh the 97 analyst/CAM fields from the same vendor and provides the current prices, volumes, fundamentals, classifications, volatility, option volume, option open interest, and option history needed for the bridged families. Alpaca remains the same-source authority for 56 news fields, and SEC EDGAR remains the authority for 16 filing/event fields. LSEG values are not automatically identical to the CRSP, Compustat, and OptionMetrics definitions on which the frozen model learned, so those fields must pass overlap rank, distribution, missingness, and category parity before use.",
                },
                {
                    "id": "lseg_probe_finding",
                    "type": "markdown",
                    "layout": "full",
                    "sourceId": "lseg_capability_probe",
                    "body": "## Primitive market data pass, while 13 derived definitions remain unsafe\n\nThe initial same-source replay matched all 34 comparable numeric LSEG fields exactly. The primitive run spans January 2024 through August 2025 and passes all 20 months: 400/400 sampled securities resolve and all 200 primitive checks pass. The broader audit tests 47 stock-level market formulas across the same window. Thirty-four pass in at least 19 of 20 months; 13 do not. Three turnover features fail most clearly because LSEG issue shares can aggregate multiple share classes and ADR units that CRSP treats as distinct traded lines. Long-history path features mostly retain near-perfect rank agreement but fail the strict missingness rule in several months. Four lifecycle, industry, or factor-context features remain outside this stock-level gate.",
                },
                {
                    "id": "lseg_probe_table",
                    "type": "table",
                    "layout": "full",
                    "tableId": "lseg_probe_summary",
                },
                {
                    "id": "market_formula_finding",
                    "type": "markdown",
                    "layout": "full",
                    "sourceId": "lseg_capability_probe",
                    "body": "## The market bridge is mostly reproducible, not complete\n\nPassing means a feature met the rank, median-rank-distance, and missingness gate in at least 95% of the 20 overlap months. A failed or pending feature must use the trained missing-value path in a dry run; it cannot be silently replaced by a convenient LSEG or Yahoo proxy.",
                },
                {
                    "id": "market_formula_chart",
                    "type": "chart",
                    "layout": "full",
                    "chartId": "market_formula_status",
                },
                {
                    "id": "market_formula_table_finding",
                    "type": "markdown",
                    "layout": "full",
                    "sourceId": "lseg_capability_probe",
                    "body": "## Feature-level evidence identifies the exact remaining market work\n\nThe 13 failures are concentrated in turnover, market-cap definition, and long-history coverage rather than short-window returns, volatility, dollar volume, spread, or Amihud illiquidity. The next engineering step is therefore a narrow security-line and full-universe bridge, not another broad data download.",
                },
                {
                    "id": "market_formula_table",
                    "type": "table",
                    "layout": "full",
                    "tableId": "market_formula_features",
                },
                {
                    "id": "source_class_chart_finding",
                    "type": "markdown",
                    "layout": "full",
                    "sourceId": "frozen_registry_audit",
                    "body": "## Most frozen inputs have an identified refresh route\n\nThe refresh-class chart covers all 318 economic features exactly once. Same-source access is the largest block, but provider availability alone does not remove the parity and signal-timing requirements shown in the other classes.",
                },
                {
                    "id": "source_split_chart",
                    "type": "chart",
                    "layout": "full",
                    "chartId": "source_class_counts",
                },
                {
                    "id": "group_table_finding",
                    "type": "markdown",
                    "layout": "full",
                    "sourceId": "frozen_registry_audit",
                    "body": "## Production policy by feature family\n\nThe table is the operational rulebook. A fallback of “missing” uses the already-trained missing-value path; it does not mean zero economic information. No row permits Yahoo Finance as a primary source.",
                },
                {
                    "id": "group_table",
                    "type": "table",
                    "layout": "full",
                    "tableId": "source_groups",
                },
                {
                    "id": "local_preflight",
                    "type": "markdown",
                    "layout": "full",
                    "sourceId": "local_provider_preflight",
                    "body": "## Current local preflight\n\nThe LSEG Workspace session, same-source replay, capability probe, 20-month primitive gate, and 47-feature stock-level audit are complete. The historical LSEG, USD-repair, and Alpaca-news archives each contain all 128 months through August 2025. Thirteen market definitions remain non-parity and four market features still require cross-sectional construction. Rotated Alpaca credentials are not present in this process, the fundamental/ownership/OPRA bridges are incomplete, the factor policy is not frozen, and no current 318-column signal package exists. The order adapter therefore remains fail-closed.",
                },
                {
                    "id": "provider_table",
                    "type": "table",
                    "layout": "full",
                    "tableId": "provider_checks",
                },
                {
                    "id": "implementation_path",
                    "type": "markdown",
                    "layout": "full",
                    "body": "## The remaining path is narrower but still blocks orders\n\n1. Build the full eligible-universe security-line bridge and either certify or deliberately mark missing the 13 failed market formulas; construct the four lifecycle/industry/factor-context features separately.\n2. Run the Compustat-to-LSEG fundamentals, SEC 13F ownership, and OptionMetrics-to-LSEG OPRA aggregate-feature parity gates.\n3. Freeze the causal factor policy and append current LSEG, Alpaca news, and SEC data into a new prospective lineage.\n4. Configure rotated Alpaca credentials only through environment variables.\n5. Materialize the exact 318 columns, run the frozen encoder, and generate an identity-bound signal manifest.\n6. Complete one full monthly and daily-warning dry run before permitting paper orders.",
                },
                {
                    "id": "yahoo_policy",
                    "type": "markdown",
                    "layout": "full",
                    "sourceId": "yfinance_docs",
                    "body": "## Why Yahoo Finance is not the production authority\n\nyfinance is useful for a human sanity check, but its own project notice says it is not affiliated with, endorsed by, or vetted by Yahoo and is intended for research and educational use. It therefore cannot silently replace CRSP, LSEG, Alpaca SIP, or OptionMetrics inside the frozen model.",
                },
                {
                    "id": "caveats",
                    "type": "markdown",
                    "layout": "full",
                    "body": "## Caveats\n\nA complete feature schema is not the same as complete non-missing coverage. The model was trained with explicit missing indicators, so authentic missingness is acceptable; new provider semantics are not. Paper trading should remain shadow/dry-run until the data gate passes. This work prepares prospective deployment and does not reopen or reinterpret the closed historical V6.8 research result.",
                },
            ],
            "cards": [
                {
                    "id": "feature_schema",
                    "description": "Every frozen feature ordinal is assigned exactly once.",
                    "dataset": "headline",
                    "sourceId": "frozen_registry_audit",
                    "metrics": [{"label": "Frozen features mapped", "field": "mapped_features", "format": "number"}],
                },
                {
                    "id": "same_source",
                    "description": "LSEG, Alpaca news, and SEC fields with a same-source refresh path.",
                    "dataset": "headline",
                    "sourceId": "frozen_registry_audit",
                    "metrics": [{"label": "Same-source path", "field": "same_source_features", "format": "number"}],
                },
                {
                    "id": "bridge_required",
                    "description": "Current APIs exist, but definitions must match CRSP/Compustat training inputs.",
                    "dataset": "headline",
                    "sourceId": "frozen_registry_audit",
                    "metrics": [{"label": "Provider parity required", "field": "bridge_features", "format": "number"}],
                },
                {
                    "id": "hard_source_policy",
                    "description": "Factor-dependent fields still need an explicit signal-close causal policy.",
                    "dataset": "headline",
                    "sourceId": "frozen_registry_audit",
                    "metrics": [{"label": "Causal policy required", "field": "hard_policy_features", "format": "number"}],
                },
            ],
            "charts": [
                {
                    "id": "source_class_counts",
                    "title": "Frozen features by live refresh class",
                    "subtitle": "A schema can be complete even while some economic fields remain authentically missing.",
                    "type": "bar",
                    "dataset": "class_counts",
                    "sourceId": "frozen_registry_audit",
                    "valueFormat": "number",
                    "unit": "features",
                    "encodings": {
                        "x": {"field": "refresh_label", "type": "nominal", "label": "Refresh class"},
                        "y": {"field": "feature_count", "type": "quantitative", "label": "Feature count"},
                        "tooltip": [
                            {"field": "refresh_label", "type": "nominal", "label": "Refresh class"},
                            {"field": "feature_count", "type": "quantitative", "label": "Features", "format": "number"},
                        ],
                    },
                    "compatibleTypes": ["bar"],
                    "maxRows": 5,
                },
                {
                    "id": "market_formula_status",
                    "title": "Market features by production-parity status",
                    "subtitle": "January 2024–August 2025; 51 market-family features in total.",
                    "type": "bar",
                    "dataset": "market_formula_status",
                    "sourceId": "lseg_capability_probe",
                    "valueFormat": "number",
                    "unit": "features",
                    "encodings": {
                        "x": {"field": "status", "type": "nominal", "label": "Parity status"},
                        "y": {"field": "feature_count", "type": "quantitative", "label": "Feature count"},
                        "tooltip": [
                            {"field": "status", "type": "nominal", "label": "Status"},
                            {"field": "feature_count", "type": "quantitative", "label": "Features", "format": "number"},
                        ],
                    },
                    "compatibleTypes": ["bar"],
                    "maxRows": 3,
                }
            ],
            "tables": [
                {
                    "id": "source_groups",
                    "title": "Feature-family production source decisions",
                    "subtitle": "Ten mutually exclusive groups cover all 318 registered features.",
                    "dataset": "source_groups",
                    "sourceId": "frozen_registry_audit",
                    "defaultSort": {"field": "feature_count", "direction": "desc"},
                    "density": "dense",
                    "columns": [
                        {"field": "production_group_label", "label": "Feature group", "type": "text"},
                        {"field": "feature_count", "label": "Features", "format": "number"},
                        {"field": "refresh_label", "label": "Refresh class", "type": "text"},
                        {"field": "current_state", "label": "Current state", "type": "text"},
                        {"field": "primary_source", "label": "Primary source", "type": "text"},
                        {"field": "required_action", "label": "Required action", "type": "text"},
                    ],
                },
                {
                    "id": "lseg_probe_summary",
                    "title": "LSEG production probe evidence",
                    "subtitle": "June 2025 historical replay sample plus current Workspace entitlement checks on 16 August 2026.",
                    "dataset": "lseg_probe_summary",
                    "sourceId": "lseg_capability_probe",
                    "defaultSort": {"field": "check", "direction": "asc"},
                    "density": "spacious",
                    "columns": [
                        {"field": "check", "label": "Check", "type": "text"},
                        {"field": "evidence", "label": "Observed evidence", "type": "text"},
                        {"field": "status", "label": "Status", "type": "text"},
                    ],
                },
                {
                    "id": "provider_checks",
                    "title": "Local provider and archive checks",
                    "subtitle": "Secret-free status at report generation time.",
                    "dataset": "provider_checks",
                    "sourceId": "local_provider_preflight",
                    "defaultSort": {"field": "check", "direction": "asc"},
                    "density": "spacious",
                    "columns": [
                        {"field": "check", "label": "Check", "type": "text"},
                        {"field": "value", "label": "Observed value", "type": "text"},
                    ],
                },
                {
                    "id": "market_formula_features",
                    "title": "Stock-level market formula parity",
                    "subtitle": "47 features tested over 20 monthly samples; pass requires at least 19 passing months.",
                    "dataset": "market_formula_features",
                    "sourceId": "lseg_capability_probe",
                    "defaultSort": {"field": "passing_month_share", "direction": "asc"},
                    "density": "dense",
                    "columns": [
                        {"field": "feature", "label": "Feature", "type": "text"},
                        {"field": "parity_status", "label": "Status", "type": "text"},
                        {"field": "months_passing", "label": "Passing months", "format": "number"},
                        {"field": "passing_month_share", "label": "Pass share", "format": "percent"},
                        {"field": "minimum_rank_correlation", "label": "Minimum rank correlation", "format": "decimal"},
                        {"field": "maximum_missingness_difference", "label": "Max missingness difference", "format": "percent"},
                    ],
                },
            ],
            "sources": sources,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated,
            "status": "ready",
            "datasets": {
                "headline": [
                    {
                        "mapped_features": int(len(matrix)),
                        "same_source_features": int((matrix["refresh_class"] == "SAME_SOURCE_REFRESH_PATH").sum()),
                        "bridge_features": int((matrix["refresh_class"] == "PROVIDER_BRIDGE_REQUIRES_PARITY").sum()),
                        "hard_policy_features": int(matrix["refresh_class"].isin(["LICENSE_OR_RETRAIN_REQUIRED", "CAUSAL_NOWCAST_OR_MISSING_POLICY_REQUIRED"]).sum()),
                    }
                ],
                "class_counts": class_counts.to_dict("records"),
                "market_formula_status": market_formula_status.to_dict("records"),
                "market_formula_features": derived_features[
                    [
                        "feature",
                        "parity_status",
                        "months_passing",
                        "passing_month_share",
                        "minimum_rank_correlation",
                        "maximum_missingness_difference",
                    ]
                ].to_dict("records"),
                "source_groups": groups[
                    [
                        "production_group_label",
                        "feature_count",
                        "refresh_label",
                        "current_state",
                        "primary_source",
                        "required_action",
                    ]
                ].to_dict("records"),
                "provider_checks": provider_rows,
                "lseg_probe_summary": [
                    {
                        "check": "Same-source historical replay",
                        "evidence": f"{replay['compared_fields']} fields; {replay['weighted_numeric_exact_share']:.1%} weighted numeric exact match; {replay['maximum_missingness_difference']:.1%} maximum missingness difference",
                        "status": "Pass",
                    },
                    {
                        "check": "Current market and fundamentals",
                        "evidence": "Six representative US equities returned prices, volumes, fundamentals, analyst/CAM, identity, and volatility content",
                        "status": "Pass",
                    },
                    {
                        "check": "CRSP-to-LSEG market bridge pilot",
                        "evidence": f"{market_pilot['resolved_consolidated_rics']}/{market_pilot['sample_size']} consolidated instruments resolved; {market_pilot['features_passing_pilot_gate']}/{market_pilot['compared_features']} features passed; minimum rank correlation {market_pilot['minimum_rank_correlation']:.4f}",
                        "status": "Pilot pass",
                    },
                    {
                        "check": "CRSP-to-LSEG full-window sampled gate",
                        "evidence": f"{market_parity['months_passing']}/{market_parity['months_expected']} months passed; {market_parity['all_month_resolved_instrument_share']:.1%} instrument resolution; minimum monthly feature rank correlation {market_minimum_rank_correlation:.4f}; zero maximum missingness difference",
                        "status": market_parity["status"],
                    },
                    {
                        "check": "Stock-level derived market formulas",
                        "evidence": f"{derived['features_passing_feature_level_gate']}/{derived['features_tested']} features pass in at least 95% of months; {derived['features_failing_feature_level_gate']} fail; {len(derived['excluded_cross_sectional_features'])} cross-sectional features remain pending; 399/400 instrument-month mappings resolved",
                        "status": "Partial pass; bundle fail",
                    },
                    {
                        "check": "Current OPRA options",
                        "evidence": f"{probe['evidence']['opra_discovery']['underlyings_with_contract']}/{probe['evidence']['opra_discovery']['underlyings_requested']} representative underlyings returned contracts plus snapshot and daily-history fields",
                        "status": "Pass",
                    },
                    {
                        "check": "Replacement-source parity",
                        "evidence": "The primitive gate passes. The broader market bridge has 34 feature-level passes, 13 failures, and four cross-sectional features pending; full-universe identity/category coverage, fundamentals, ownership, and option aggregates remain incomplete",
                        "status": "Pending",
                    },
                ],
            },
        },
        "sources": sources,
    }
    return artifact


def build_notebook() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {"display_name": "Python (finance)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    notebook["cells"] = [
        nbf.v4.new_markdown_cell(
            "# V6.8 production-data readiness audit\n\n"
            "This notebook reproduces the 318-feature source classification and the local, secret-free provider preflight."
        ),
        nbf.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import pandas as pd\n"
            "import matplotlib.pyplot as plt\n\n"
            "ROOT = Path.cwd()\n"
            "OUTPUT = ROOT / 'artifacts/research_v6_8/production_data_readiness/20260816'\n"
            "matrix = pd.read_csv(OUTPUT / 'feature_source_matrix.csv')\n"
            "groups = pd.read_csv(OUTPUT / 'source_group_summary.csv')\n"
            "readiness = json.loads((OUTPUT / 'readiness.json').read_text())\n"
            "probe = json.loads((ROOT / 'artifacts/research_v6_8/lseg_production_probe/20260816/receipt.json').read_text())\n"
            "market_parity = json.loads((ROOT / 'artifacts/research_v6_8/lseg_market_parity/20260817/receipt.json').read_text())\n"
            "derived = json.loads((ROOT / 'artifacts/research_v6_8/lseg_market_derived_parity/20260817/feature_summary_receipt.json').read_text())\n"
            "derived_features = pd.read_csv(ROOT / 'artifacts/research_v6_8/lseg_market_derived_parity/20260817/feature_window_summary.csv')\n"
            "len(matrix), readiness['status'], probe['status'], market_parity['status'], derived['status']"
        ),
        nbf.v4.new_markdown_cell("## Coverage and classification"),
        nbf.v4.new_code_cell(
            "class_counts = matrix.groupby('refresh_class').size().sort_values(ascending=False)\n"
            "display(class_counts.rename('feature_count').to_frame())\n"
            "ax = class_counts.sort_values().plot.barh(figsize=(9, 4), color='#2563eb')\n"
            "ax.set_xlabel('Frozen feature count')\n"
            "ax.set_ylabel('Live refresh class')\n"
            "ax.set_title('All 318 V6 features are assigned to one production policy')\n"
            "plt.tight_layout(); plt.show()"
        ),
        nbf.v4.new_markdown_cell("## Source groups"),
        nbf.v4.new_code_cell(
            "display(groups[['production_group_label','feature_count','refresh_class','current_state','primary_source']])"
        ),
        nbf.v4.new_markdown_cell("## Local provider preflight"),
        nbf.v4.new_code_cell(
            "display(pd.Series(readiness['provider_checks'], name='observed').to_frame())\n"
            "display(pd.Series(probe['checks'], name='observed').to_frame())\n"
            "display(pd.Series(probe['evidence']['same_source_historical_replay'], name='observed').to_frame())\n"
            "display(pd.Series(probe['evidence']['market_price_bridge_pilot'], name='observed').to_frame())\n"
            "display(pd.Series(market_parity, name='observed').to_frame())\n"
            "display(pd.Series(derived, name='observed').to_frame())\n"
            "display(derived_features.sort_values(['feature_level_parity_pass','passing_month_share']))\n"
            "display(pd.Series(readiness['blockers'], name='paper_order_blocker').to_frame())"
        ),
        nbf.v4.new_markdown_cell("## Reproducibility assertions"),
        nbf.v4.new_code_cell(
            "assert len(matrix) == 318\n"
            "assert matrix.model_ordinal.tolist() == list(range(1, 319))\n"
            "assert matrix.feature_name.is_unique\n"
            "assert groups.feature_count.sum() == 318\n"
            "assert not matrix.yfinance_primary_allowed.any()\n"
            "assert len(matrix[matrix.feature_name.str.startswith('pastor_stambaugh')]) == 2\n"
            "assert probe['checks']['capability_pass']\n"
            "assert probe['checks']['same_source_replay_pass']\n"
            "assert probe['checks']['market_price_bridge_pilot_pass']\n"
            "assert market_parity['status'] == 'PASS'\n"
            "assert market_parity['months_passing'] == market_parity['months_expected'] == 20\n"
            "assert market_parity['all_month_resolved_instrument_share'] == 1.0\n"
            "assert derived['months'] == 20\n"
            "assert derived['features_tested'] == 47\n"
            "assert derived['features_passing_feature_level_gate'] == 34\n"
            "assert derived['features_failing_feature_level_gate'] == 13\n"
            "assert not probe['checks']['provider_replacement_parity_complete']\n"
            "print('PASS: registry coverage, LSEG replay, primitive parity, 47-feature audit accounting, Yahoo policy, and structural missingness checks')"
        ),
        nbf.v4.new_markdown_cell(
            "## Conclusion\n\n"
            "The full schema is reproducible and the 20-month sampled market-primitives bridge passes. Thirty-four of 47 tested stock-level market formulas meet the feature-level rule; 13 fail and four cross-sectional market features remain pending. Current paper orders remain blocked until the security-line/full-universe, fundamental, ownership, option, and factor-timing gates; credentials; and the current signal package are complete. Authentic missingness is allowed by the trained encoder; silent provider substitution is not."
        ),
    ]
    notebook_path = OUTPUT / "production_data_readiness.ipynb"
    notebook_path.write_text(nbf.writes(notebook), "utf-8")
    executed = NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    ).execute()
    notebook_path.write_text(nbf.writes(executed), "utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    artifact = build_artifact()
    (OUTPUT / "artifact.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", "utf-8"
    )
    build_notebook()


if __name__ == "__main__":
    main()
