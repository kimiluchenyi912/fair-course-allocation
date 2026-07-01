from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MathCoverageStatus(str, Enum):
    NO_MATH_REQUIRED = "no_math_required"
    SATISFIED_BY_PRIMARY = "satisfied_by_primary"
    SATISFIED_BY_MANDATORY_FALLBACK = "satisfied_by_mandatory_fallback"
    PENDING_MANDATORY_FALLBACK = "pending_mandatory_fallback"
    VIOLATED_SINGLE_MATH_REQUIRED = "violated_single_math_required"
    VIOLATED_MULTIPLE_MATH_AT_LEAST_ONE = "violated_multiple_math_at_least_one"
    VIOLATED_MANDATORY_FALLBACK_FAILED = "violated_mandatory_fallback_failed"


class MathPolicyViolationType(str, Enum):
    NONE = "none"
    SINGLE_MATH_REQUIRED = "single_math_required"
    MULTIPLE_MATH_AT_LEAST_ONE = "multiple_math_at_least_one"
    MANDATORY_FALLBACK_FAILED = "mandatory_fallback_failed"


@dataclass(frozen=True)
class MathFallbackRule:
    source_course_id: str
    fallback_course_id: str
    policy_type: str
    enabled: bool
    notes: str


@dataclass(frozen=True)
class StudentMathPolicyOutcome:
    student_id: str
    math_primary_request_keys: tuple[str, ...]
    math_primary_candidate_keys: tuple[str, ...]
    math_primary_count: int
    assigned_math_primary_request_keys: tuple[str, ...]
    assigned_math_primary_count: int
    coverage_status: MathCoverageStatus
    violation_type: MathPolicyViolationType
    fallback_source_request_key: str | None
    fallback_source_course_id: str | None
    fallback_target_course_id: str | None
    fallback_assignment_key: str | None
    fallback_outcome_status: str | None


@dataclass(frozen=True)
class MathPolicyReport:
    math_course_ids: tuple[str, ...]
    no_math_primary_student_ids: tuple[str, ...]
    single_math_primary_student_ids: tuple[str, ...]
    multiple_math_primary_student_ids: tuple[str, ...]
    coverage_satisfied_student_ids: tuple[str, ...]
    coverage_satisfied_by_primary_student_ids: tuple[str, ...]
    coverage_satisfied_by_mandatory_fallback_student_ids: tuple[str, ...]
    coverage_pending_fallback_student_ids: tuple[str, ...]
    direct_violation_student_ids: tuple[str, ...]
    fallback_failure_violation_student_ids: tuple[str, ...]
    fallback_failure_violating_request_keys: tuple[str, ...]
    current_math_coverage_violation_student_ids: tuple[str, ...]
    policy_currently_satisfied: bool
    single_math_required_student_ids: tuple[str, ...]
    single_math_required_violation_student_ids: tuple[str, ...]
    single_math_required_violating_request_keys: tuple[str, ...]
    multiple_math_student_ids: tuple[str, ...]
    multiple_math_at_least_one_violation_student_ids: tuple[str, ...]
    multiple_math_at_least_one_violating_request_keys: tuple[str, ...]
    fallback_source_course_id: str | None
    fallback_target_course_id: str | None
    fallback_required_student_ids: tuple[str, ...]
    fallback_required_source_request_keys: tuple[str, ...]
    fallback_pending_student_ids: tuple[str, ...]
    fallback_assigned_student_ids: tuple[str, ...]
    fallback_failed_student_ids: tuple[str, ...]
    fallback_mapping: tuple[MathFallbackRule, ...]
    student_outcomes: tuple[StudentMathPolicyOutcome, ...]


class MathFallbackConfigError(ValueError):
    """Raised when math fallback configuration is not internally valid."""
