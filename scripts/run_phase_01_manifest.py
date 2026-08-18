#!/usr/bin/env python3
"""Thin entry point for the Phase 1 primary raw-snapshot build."""

from portfolio_bot.data.raw_manifest import main


if __name__ == "__main__":
    raise SystemExit(main())
