"""Post-diagnostic V6.3 signal-to-weight research."""

from portfolio_bot.research_v6_3.signals import (
    MultiTaskSelector,
    add_confidence_diagnostics,
    build_three_month_target,
    feature_expert_indices,
)

__all__ = [
    "MultiTaskSelector",
    "add_confidence_diagnostics",
    "build_three_month_target",
    "feature_expert_indices",
]
