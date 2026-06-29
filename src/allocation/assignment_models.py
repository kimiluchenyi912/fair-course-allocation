from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AssignmentRejectionReason(str, Enum):
    UNKNOWN_STUDENT = "unknown_student"
    UNKNOWN_REQUEST = "unknown_request"
    UNKNOWN_SECTION = "unknown_section"
    UNKNOWN_ASSIGNMENT = "unknown_assignment"
    REQUEST_NOT_OWNED_BY_STUDENT = "request_not_owned_by_student"
    SECTION_NOT_CANDIDATE_FOR_REQUEST = "section_not_candidate_for_request"
    REQUEST_ALREADY_ASSIGNED = "request_already_assigned"
    DUPLICATE_LOGICAL_COURSE_OR_BLOCK = "duplicate_logical_course_or_block"
    PERIOD_CONFLICT = "period_conflict"
    TARGET_LOAD_EXCEEDED = "target_load_exceeded"
    SECTION_FULL = "section_full"
    PERIOD_UNIT_MISMATCH = "period_unit_mismatch"


@dataclass(frozen=True)
class AssignmentRecord:
    assignment_key: str
    student_id: str
    request_key: str
    request_type: str
    alternate_rank: int | None
    request_candidate_key: str
    linked_section_group_id: str
    member_section_ids: tuple[str, ...]
    occupied_periods: tuple[str, ...]
    period_units: int
    structure_type: str
    logical_block_id: str
    course_ids: tuple[str, ...]


@dataclass(frozen=True)
class AssignmentFeasibility:
    student_id: str
    request_key: str
    linked_section_group_id: str
    allowed: bool
    reasons: tuple[AssignmentRejectionReason, ...] = ()


@dataclass(frozen=True)
class AssignmentResult:
    student_id: str
    request_key: str
    linked_section_group_id: str
    allowed: bool
    reasons: tuple[AssignmentRejectionReason, ...] = ()
    assignment: AssignmentRecord | None = None


@dataclass(frozen=True)
class StateConsistencyIssue:
    code: str
    message: str
    identifier: str = ""
