#!/usr/bin/env python3
"""Sealed, no-argument Stage 11 Gate-A source materialization command."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from portfolio_bot.validation.validation_source_extraction import (  # noqa: E402
    Stage11SourceExtractionError,
    materialize_normalized_stage11_sources,
)
from portfolio_bot.validation.validation_source_materialization import (  # noqa: E402
    Stage11SourceMaterializationSealedError,
    open_stage11_validation_source_materialization,
)
from portfolio_bot.validation.validation_source_publication import (  # noqa: E402
    Stage11SourcePublicationError,
    publish_stage11_normalized_sources,
)


def main(argv: list[str] | None = None) -> int:
    selected = sys.argv[1:] if argv is None else argv
    if selected:
        print(
            json.dumps(
                {
                    "status": "source_materialization_sealed",
                    "error": "caller-supplied paths and options are prohibited",
                    "validation_or_reserve_rows_read": 0,
                    "labels_forecasts_weights_trades_pnl_rows_created": 0,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    try:
        with open_stage11_validation_source_materialization() as authority:
            build = materialize_normalized_stage11_sources(authority)
            publication = publish_stage11_normalized_sources(authority, build)
            print(
                json.dumps(
                    {
                        "status": "STAGE11_SOURCE_MATERIALIZATION_COMPLETE",
                        "build_id": publication.build_id,
                        "bundle_path": str(publication.bundle_path),
                        "manifest_sha256": publication.manifest_sha256,
                        "pointer_sha256": publication.pointer_sha256,
                        "reused": publication.reused,
                        "gate_b_execution_authorized": False,
                    },
                    sort_keys=True,
                    indent=2,
                )
            )
        return 0
    except (
        Stage11SourceMaterializationSealedError,
        Stage11SourceExtractionError,
        Stage11SourcePublicationError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(
            json.dumps(
                {
                    "status": "source_materialization_sealed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "labels_forecasts_weights_trades_pnl_rows_created": 0,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
