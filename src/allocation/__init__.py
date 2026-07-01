"""Canonical fixed-section allocation inputs and local assignment state.

This package intentionally stops at local fixed-section assignment mechanics.
It does not choose students, fill alternates, evaluate final outcomes, or run
global optimization.
"""

from .assignment_models import (
    AssignmentFeasibility,
    AssignmentRecord,
    AssignmentRejectionReason,
    AssignmentResult,
    StateConsistencyIssue,
)
from .baseline_models import (
    AlternateRequestStatus,
    BaselineInternalConsistencyError,
    BaselineResult,
    CandidateAttempt,
    HighDemandCandidateDemand,
    MandatoryFallbackOutcome,
    MandatoryFallbackStatus,
    PolicyReport,
    PrimaryRequestStatus,
    RequestOutcome,
    SectionRosterSummary,
    StudentOutcome,
)
from .input_adapter import canonicalize_allocation_input
from .input_models import (
    AllocationInputError,
    AllocationInputIssue,
    CanonicalAllocationInput,
    CanonicalStudent,
    CourseMetadata,
    LogicalRequest,
    LogicalSection,
    SectionMember,
    SourceRequestRow,
)
from .math_policy import (
    evaluate_math_policy,
    load_math_fallback_rules,
    math_course_ids_from_catalog,
    parse_math_fallback_rules,
)
from .math_policy_models import (
    MathCoverageStatus,
    MathFallbackConfigError,
    MathFallbackRule,
    MathPolicyReport,
    MathPolicyViolationType,
    StudentMathPolicyOutcome,
)
from .random_baseline import run_seeded_random_baseline
from .state import AllocationState

__all__ = [
    "AllocationState",
    "AlternateRequestStatus",
    "AssignmentFeasibility",
    "AllocationInputError",
    "AllocationInputIssue",
    "AssignmentRecord",
    "AssignmentRejectionReason",
    "AssignmentResult",
    "BaselineInternalConsistencyError",
    "BaselineResult",
    "CanonicalAllocationInput",
    "CanonicalStudent",
    "CandidateAttempt",
    "CourseMetadata",
    "HighDemandCandidateDemand",
    "MandatoryFallbackOutcome",
    "MandatoryFallbackStatus",
    "LogicalRequest",
    "LogicalSection",
    "MathCoverageStatus",
    "MathFallbackConfigError",
    "MathFallbackRule",
    "MathPolicyReport",
    "MathPolicyViolationType",
    "PolicyReport",
    "PrimaryRequestStatus",
    "RequestOutcome",
    "SectionRosterSummary",
    "SectionMember",
    "SourceRequestRow",
    "StateConsistencyIssue",
    "StudentOutcome",
    "StudentMathPolicyOutcome",
    "canonicalize_allocation_input",
    "evaluate_math_policy",
    "load_math_fallback_rules",
    "math_course_ids_from_catalog",
    "parse_math_fallback_rules",
    "run_seeded_random_baseline",
]
