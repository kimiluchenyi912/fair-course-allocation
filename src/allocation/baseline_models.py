from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .assignment_models import AssignmentRecord, AssignmentRejectionReason, StateConsistencyIssue


class PrimaryRequestStatus(str, Enum):
    ASSIGNED = "assigned"
    UNMET_NO_CANDIDATES = "unmet_no_candidates"
    UNMET_ALL_CANDIDATES_REJECTED = "unmet_all_candidates_rejected"


class AlternateRequestStatus(str, Enum):
    ASSIGNED = "assigned"
    NOT_NEEDED = "not_needed"
    DOES_NOT_FIT_REMAINING_LOAD = "does_not_fit_remaining_load"
    UNASSIGNED_NO_CANDIDATES = "unassigned_no_candidates"
    UNASSIGNED_ALL_CANDIDATES_REJECTED = "unassigned_all_candidates_rejected"


class MandatoryFallbackStatus(str, Enum):
    NOT_REQUIRED_SOURCE_ASSIGNED = "not_required_source_assigned"
    NOT_REQUIRED_MATH_COVERAGE_ALREADY_SATISFIED = "not_required_math_coverage_already_satisfied"
    ASSIGNED = "assigned"
    UNASSIGNED_NO_CANDIDATES = "unassigned_no_candidates"
    UNASSIGNED_ALL_CANDIDATES_REJECTED = "unassigned_all_candidates_rejected"


@dataclass(frozen=True)
class CandidateAttempt:
    attempt_index: int
    linked_section_group_id: str
    success: bool
    rejection_reasons: tuple[AssignmentRejectionReason, ...]
    assignment_key: str | None


@dataclass(frozen=True)
class RequestOutcome:
    request_key: str
    student_id: str
    request_type: str
    alternate_rank: int | None
    candidate_key: str
    period_units: int
    status: PrimaryRequestStatus | AlternateRequestStatus
    assignment_key: str | None
    assigned_linked_section_group_id: str | None
    candidate_attempts: tuple[CandidateAttempt, ...]
    remaining_units_before: int
    remaining_units_after: int


@dataclass(frozen=True)
class MandatoryFallbackOutcome:
    student_id: str
    source_request_key: str
    source_course_id: str
    fallback_request_key: str
    fallback_course_id: str
    status: MandatoryFallbackStatus
    assignment_key: str | None
    assigned_linked_section_group_id: str | None
    candidate_attempts: tuple[CandidateAttempt, ...]
    remaining_units_before: int
    remaining_units_after: int


@dataclass(frozen=True)
class StudentOutcome:
    student_id: str
    grade: int
    target_period_units: int
    assigned_period_units: int
    remaining_period_units: int
    assignment_keys: tuple[str, ...]
    primary_request_count: int
    primary_assigned_count: int
    primary_unmet_count: int
    primary_unmet_request_keys: tuple[str, ...]
    primary_unmet_period_units: int
    alternate_request_count: int
    alternate_assigned_count: int
    alternate_assigned_period_units: int
    mandatory_fallback_assigned_count: int
    mandatory_fallback_assigned_period_units: int
    mandatory_fallback_assignment_keys: tuple[str, ...]
    fully_scheduled: bool
    priority_protected: bool
    ordinary_fairness_violation: bool
    protected_fairness_violation: bool
    high_demand_guarantee_violation_count: int
    high_demand_violating_request_keys: tuple[str, ...]
    target_logical_course_count: int | None = None
    assigned_logical_course_count: int | None = None
    logical_schedule_gap_count: int | None = None
    logical_fully_scheduled: bool | None = None


@dataclass(frozen=True)
class HighDemandCandidateDemand:
    candidate_key: str
    logical_primary_demand: int


@dataclass(frozen=True)
class PolicyReport:
    ordinary_policy_satisfied: bool
    protected_policy_satisfied: bool
    high_demand_policy_satisfied: bool
    all_reported_policies_satisfied: bool
    ordinary_violation_student_ids: tuple[str, ...]
    protected_violation_student_ids: tuple[str, ...]
    high_demand_candidate_keys: tuple[str, ...]
    high_demand_demands: tuple[HighDemandCandidateDemand, ...]
    high_demand_violating_request_keys: tuple[str, ...]
    high_demand_violating_student_ids: tuple[str, ...]
    high_demand_violating_candidate_keys: tuple[str, ...]
    high_demand_violation_count: int


@dataclass(frozen=True)
class SectionRosterSummary:
    linked_section_group_id: str
    assigned_count: int
    capacity: int
    remaining_capacity: int
    student_ids: tuple[str, ...]


@dataclass(frozen=True)
class StudentDifficultyProfile:
    student_id: str
    protected: bool
    math_risk_score: int
    high_demand_primary_count: int
    scarce_primary_count_le_1: int
    scarce_primary_count_le_2: int
    total_candidate_flexibility: float
    average_candidate_flexibility: float
    double_period_primary_count: int
    target_period_units: int
    seeded_tie_break: float
    ordering_rank: int


@dataclass(frozen=True)
class BaselineResult:
    algorithm_name: str
    seed: int
    student_processing_order: tuple[str, ...]
    assignments: tuple[AssignmentRecord, ...]
    mandatory_fallback_outcomes: tuple[MandatoryFallbackOutcome, ...]
    request_outcomes: tuple[RequestOutcome, ...]
    student_outcomes: tuple[StudentOutcome, ...]
    policy_report: PolicyReport
    section_roster_summary: tuple[SectionRosterSummary, ...]
    consistency_issues: tuple[StateConsistencyIssue, ...]
    student_difficulty_profiles: tuple[StudentDifficultyProfile, ...] = ()


class BaselineInternalConsistencyError(RuntimeError):
    """Raised when a baseline run leaves AllocationState internally inconsistent."""

    def __init__(self, issues: tuple[StateConsistencyIssue, ...]):
        self.issues = issues
        details = "\n".join(f"{issue.code}: {issue.identifier} {issue.message}" for issue in issues)
        super().__init__(f"Allocation baseline internal consistency failed:\n{details}")
