#!/usr/bin/env python3
"""Validate and save the complete Phase 1 handoff assessment."""

from __future__ import annotations

from portfolio_bot.data.phase_01_handoff import write_validation_report


def main() -> int:
    print(write_validation_report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
