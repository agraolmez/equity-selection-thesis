#!/usr/bin/env python3
"""Perform the final raw-byte recheck and publish the immutable Phase 1 freeze."""

from __future__ import annotations

from portfolio_bot.data.phase_01_freeze import publish_phase_01_freeze


def main() -> int:
    publish_phase_01_freeze()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
