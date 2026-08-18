#!/usr/bin/env python3
"""Check structural coverage of the historical 15-stage roadmap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGES = {
    1: ("scripts/phase_02_discover_crsp_numeric_units.py", "scripts/audit_phase_01_post_publication.py"),
    2: ("scripts/build_phase_02_crsp_typed_backbone.py", "scripts/verify_phase_02_crsp_typed_backbone.py"),
    3: ("scripts/build_phase_02_crsp_return_reconciliation.py", "scripts/build_phase_02_calendar.py"),
    4: ("scripts/build_phase_02_signal_date_universe.py", "scripts/verify_phase_02_signal_date_universe.py"),
    5: ("scripts/build_phase_02_auxiliary.py", "src/portfolio_bot/data/phase_02_auxiliary_core.py"),
    6: ("scripts/build_stage_06_feature_store.py", "scripts/verify_stage_06_feature_store.py"),
    7: ("scripts/build_stage_07_label_store.py", "scripts/verify_stage_07_label_store.py"),
    8: ("scripts/build_stage_08_risk_cost.py", "scripts/verify_stage_08_risk_cost.py"),
    9: ("scripts/run_stage_09_selector_search.py", "scripts/verify_stage_09_selectors.py"),
    10: ("scripts/build_stage_10_allocators.py", "scripts/verify_stage_10_allocators.py"),
    11: ("scripts/run_stage_11_validation.py", "scripts/verify_stage_11_validation_data.py"),
    12: ("scripts/build_stage_12_final_freeze.py", "scripts/verify_stage_12_final_freeze.py"),
    13: ("src/portfolio_bot/final_test/orchestration.py", "src/portfolio_bot/final_test/publication.py"),
    14: ("src/portfolio_bot/final_test/inference.py", "configs/phase_06_final_test/stage_14_report_template_v1.md"),
    15: ("thesis/MASTER_DRAFT.md", "thesis/REPRODUCIBILITY_AUDIT.md"),
}


def audit() -> dict[str, object]:
    stages: dict[str, dict[str, object]] = {}
    for stage, paths in STAGES.items():
        missing = [path for path in paths if not (ROOT / path).is_file()]
        stages[str(stage)] = {
            "status": "PRESENT" if not missing else "MISSING",
            "required_paths": list(paths),
            "missing_paths": missing,
        }
    complete = all(value["status"] == "PRESENT" for value in stages.values())
    return {
        "status": "PASS" if complete else "FAIL",
        "scope": "structural_coverage_not_fresh_numerical_replay",
        "stages": stages,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    result = audit()
    if args.as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"Historical 15-stage structural coverage: {result['status']}")
        for stage, value in result["stages"].items():
            print(f"Stage {stage}: {value['status']}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
