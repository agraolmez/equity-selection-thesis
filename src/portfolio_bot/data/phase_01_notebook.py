"""Canonical, thin Phase 1 review-notebook specification."""

from __future__ import annotations

import os
from pathlib import Path

import nbformat as nbf

from portfolio_bot.data.phase_01_review import PROJECT_ROOT, load_review_context


PHASE_01_NOTEBOOK_RELATIVE_PATH = Path(
    "notebooks/01_phase_01_raw_snapshot_review.ipynb"
)


def _cell(cell_id: str, cell: nbf.NotebookNode) -> nbf.NotebookNode:
    cell["id"] = cell_id
    return cell


def build_phase_01_notebook(project_root: Path = PROJECT_ROOT) -> nbf.NotebookNode:
    """Build the deterministic notebook source and bind the verified source IDs."""
    project_root = project_root.resolve()
    summary = load_review_context(project_root)["summary"]
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3"},
        "phase": "01_raw_snapshot",
        "role": "thin review of saved immutable evidence; no production logic",
        "raw_snapshot_id": summary["raw_snapshot_id"],
        "phase_01_build_id": summary["phase_01_build_id"],
        "verification_id": summary["verification_id"],
        "phase_status": "verified_pending_explicit_phase_freeze",
    }
    notebook["cells"] = [
        _cell(
            "phase1-title",
            nbf.v4.new_markdown_cell(
                "# Phase 1 raw snapshot review\n\n"
                "This notebook reads the published primary and independent-verification bundles. "
                "It is a pre-freeze review: it does not scan raw data, construct features, fit a model, or run a backtest, "
                "and it does not authorize Phase 2."
            ),
        ),
        _cell(
            "phase1-load",
            nbf.v4.new_code_cell(
                "from pathlib import Path\n"
                "import sys\n"
                "import pandas as pd\n"
                "from IPython.display import display\n\n"
                "WORKING_DIRECTORY = Path.cwd().resolve()\n"
                "PROJECT_ROOT = WORKING_DIRECTORY if (WORKING_DIRECTORY / 'src').is_dir() else WORKING_DIRECTORY.parent\n"
                "if not (PROJECT_ROOT / 'src').is_dir():\n"
                "    raise RuntimeError('Could not locate the project root from the notebook working directory.')\n"
                "sys.path.insert(0, str(PROJECT_ROOT / 'src'))\n\n"
                "from portfolio_bot.data.phase_01_review import load_review_context, validation_failures\n\n"
                "review = load_review_context(PROJECT_ROOT)\n"
                "summary = review['summary']\n"
                "summary"
            ),
        ),
        _cell(
            "phase1-gate-heading",
            nbf.v4.new_markdown_cell(
                "## Gate result\n\n"
                "Every required invariant is recomputed from the saved evidence before this notebook displays the result."
            ),
        ),
        _cell(
            "phase1-gate",
            nbf.v4.new_code_cell(
                "failures = validation_failures(review)\n"
                "assert failures == [], failures\n"
                "pd.DataFrame([{\n"
                "    'gate': 'Primary and independent raw verification',\n"
                "    'result': 'VERIFIED; READY TO FREEZE',\n"
                "    'files': summary['file_count'],\n"
                "    'parsed_rows': summary['parsed_rows'],\n"
                "    'raw_size_gb': summary['size_gb'],\n"
                "    'primary_passes': summary['primary_pass_count'],\n"
                "    'warnings': summary['primary_warning_count'],\n"
                "    'failures': summary['primary_failure_count'],\n"
                "    'independent_differences': summary['independent_difference_count'],\n"
                "}])"
            ),
        ),
        _cell(
            "phase1-family-heading",
            nbf.v4.new_markdown_cell(
                "## Source-family footprint\n\n"
                "Rows and bytes are descriptive scale measures. Coverage start and end are observation axes, not public-availability dates."
            ),
        ),
        _cell(
            "phase1-family",
            nbf.v4.new_code_cell(
                "family = pd.DataFrame(review['family_summary'])\n"
                "display(family[['family', 'file_count', 'parsed_rows', 'size_gb', 'coverage_start', 'coverage_end']])"
            ),
        ),
        _cell(
            "phase1-files-heading",
            nbf.v4.new_markdown_cell(
                "## Exact 25-file registry\n\n"
                "This table is the compact review surface for file identity, scale, coverage, and independent status."
            ),
        ),
        _cell(
            "phase1-files",
            nbf.v4.new_code_cell(
                "files = pd.DataFrame(review['file_summary'])\n"
                "display(files[['filename', 'family', 'parsed_rows', 'size_gb', 'coverage_axis', 'coverage_start', 'coverage_end', 'independent_status']])"
            ),
        ),
        _cell(
            "phase1-reconciliation-heading",
            nbf.v4.new_markdown_cell(
                "## Reconciliation checks\n\n"
                "The highest-impact totals and independent registries are checked explicitly below."
            ),
        ),
        _cell(
            "phase1-reconciliation",
            nbf.v4.new_code_cell(
                "check_counts = pd.Series([row['status'] for row in review['checks']]).value_counts().to_dict()\n"
                "reconciliation = {\n"
                "    'quality_check_counts': check_counts,\n"
                "    'difference_rows': len(review['differences']),\n"
                "    'hash_registries_equal': review['primary_hashes'] == review['verified_hashes'],\n"
                "    'date_profiles_reconcile': all(row['category_count_reconciles'] == 'True' for row in review['dates']),\n"
                "    'files_stable_through_verification': summary['all_files_stable_through_verification'],\n"
                "}\n"
                "assert reconciliation == {\n"
                "    'quality_check_counts': {'PASS': 585},\n"
                "    'difference_rows': 0,\n"
                "    'hash_registries_equal': True,\n"
                "    'date_profiles_reconcile': True,\n"
                "    'files_stable_through_verification': True,\n"
                "}\n"
                "reconciliation"
            ),
        ),
        _cell(
            "phase1-scope-heading",
            nbf.v4.new_markdown_cell(
                "## Scope boundary\n\n"
                "A successful raw-data gate does not by itself make a row eligible for a historical portfolio decision."
            ),
        ),
        _cell(
            "phase1-scope",
            nbf.v4.new_code_cell("display(pd.DataFrame(review['scope_rows']))"),
        ),
        _cell(
            "phase1-conclusion",
            nbf.v4.new_markdown_cell(
                "## Conclusion\n\n"
                "Primary and independent verification pass, so Phase 1 is ready for its explicit freeze. Phase 2 remains "
                "prohibited until that freeze is published and verified. After the freeze, the only permitted implementation "
                "stage is the availability-aligned canonical panel; point-in-time joins, current-vintage exclusions, canonical "
                "uniqueness, return reconciliation, feature leakage, models, and backtests remain gated and untested."
            ),
        ),
        _cell(
            "phase1-result",
            nbf.v4.new_code_cell(
                "phase_01_gate_result = {\n"
                "    'phase': '01_raw_snapshot',\n"
                "    'status': 'verified_pending_explicit_phase_freeze',\n"
                "    'raw_snapshot_id': summary['raw_snapshot_id'],\n"
                "    'phase_01_build_id': summary['phase_01_build_id'],\n"
                "    'verification_id': summary['verification_id'],\n"
                "    'phase_2_permitted': False,\n"
                "}\n"
                "assert phase_01_gate_result['status'] == summary['phase_status']\n"
                "phase_01_gate_result"
            ),
        ),
    ]
    return notebook


def write_phase_01_notebook(
    project_root: Path = PROJECT_ROOT,
    output: Path | None = None,
) -> Path:
    project_root = project_root.resolve()
    output = output or project_root / PHASE_01_NOTEBOOK_RELATIVE_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".ipynb.tmp")
    temporary.write_bytes(canonical_phase_01_notebook_bytes(project_root))
    os.replace(temporary, output)
    return output


def canonical_phase_01_notebook_bytes(
    project_root: Path = PROJECT_ROOT,
) -> bytes:
    """Return the exact unexecuted bytes emitted by the canonical notebook writer."""
    rendered = nbf.writes(build_phase_01_notebook(project_root.resolve()))
    if not rendered.endswith("\n"):
        rendered += "\n"
    return rendered.encode("utf-8")
