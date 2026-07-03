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
    StudentDifficultyProfile,
    StudentOutcome,
)
from .cp_sat_models import (
    CpSatAllocationResult,
    CpSatModelStats,
    CpSatModelScope,
    CpSatObjectiveValues,
    CpSatSolveStatus,
    CpSatStageDiagnostic,
    CpSatStageName,
)
from .cp_sat_solver import run_fair_cp_sat_solver
from .constrained_first_baseline import (
    build_ordering_context,
    build_student_difficulty_profiles,
    candidate_section_priority,
    primary_request_priority,
    run_constrained_first_baseline,
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
    "CpSatAllocationResult",
    "CpSatModelStats",
    "CpSatModelScope",
    "CpSatObjectiveValues",
    "CpSatSolveStatus",
    "CpSatStageDiagnostic",
    "CpSatStageName",
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
    "StudentDifficultyProfile",
    "StudentOutcome",
    "StudentMathPolicyOutcome",
    "build_ordering_context",
    "build_student_difficulty_profiles",
    "candidate_section_priority",
    "canonicalize_allocation_input",
    "evaluate_math_policy",
    "load_math_fallback_rules",
    "math_course_ids_from_catalog",
    "parse_math_fallback_rules",
    "primary_request_priority",
    "run_fair_cp_sat_solver",
    "run_constrained_first_baseline",
    "run_seeded_random_baseline",
]
