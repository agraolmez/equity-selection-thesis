"""Pre-unblind contracts for the Stage 13 final walk-forward.

The package is intentionally filesystem-agnostic except for the explicitly
authorized access and checkpoint layers.  Importing it cannot inspect final
period data.
"""

from portfolio_bot.final_test.authorization import (
    Stage12VerifiedBinding,
    Stage13ApprovalReceipt,
    Stage13AuthorizationError,
    issue_stage13_unblind_approval,
    require_stage13_unblind_authorization,
)
from portfolio_bot.final_test.calendar import (
    CORE_COMPLETE_HOLDING_MONTHS,
    FINAL_COMPLETE_HOLDING_MONTHS,
    FINAL_HOLDING_MONTHS,
    PARTIAL_DECEMBER_HOLDING_MONTH,
    PRIMARY_FULL_HOLDING_MONTHS,
    FinalPeriodSegment,
)

__all__ = [
    "CORE_COMPLETE_HOLDING_MONTHS",
    "FINAL_COMPLETE_HOLDING_MONTHS",
    "FINAL_HOLDING_MONTHS",
    "PARTIAL_DECEMBER_HOLDING_MONTH",
    "PRIMARY_FULL_HOLDING_MONTHS",
    "FinalPeriodSegment",
    "Stage12VerifiedBinding",
    "Stage13ApprovalReceipt",
    "Stage13AuthorizationError",
    "issue_stage13_unblind_approval",
    "require_stage13_unblind_authorization",
]
