#!/usr/bin/env python3
"""Create the canonical thin Phase 1 evidence-review notebook."""

from __future__ import annotations

from portfolio_bot.data.phase_01_notebook import write_phase_01_notebook


def main() -> int:
    print(write_phase_01_notebook())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
