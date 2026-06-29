from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class AllocationInputIssue:
    code: str
    source: str
    message: str
    row: int | None = None
    identifier: str | None = None

    def format(self) -> str:
        parts = [self.code, self.source]
        if self.row is not None:
            parts.append(f"row {self.row}")
        if self.identifier:
            parts.append(self.identifier)
        return f"{' | '.join(parts)}: {self.message}"


class AllocationInputError(ValueError):
    """Raised after canonicalization collects one or more input violations."""

    def __init__(self, issues: tuple[AllocationInputIssue, ...]):
        self.issues = issues
        super().__init__("\n".join(issue.format() for issue in issues))


@dataclass(frozen=True)
class CourseMetadata:
    course_id: str
    period_units: int
    schedule_structure: str


@dataclass(frozen=True)
class SourceRequestRow:
    course_id: str
    request_group: str
    must_share_block_id: str
    request_rank: int | None


@dataclass(frozen=True)
class LogicalRequest:
    request_key: str
    student_id: str
    request_type: str
    candidate_key: str
    course_ids: tuple[str, ...]
    source_rows: tuple[SourceRequestRow, ...]
    request_rank: int | None
    period_units: int


@dataclass(frozen=True)
class CanonicalStudent:
    student_id: str
    grade: int
    target_period_units: int
    unscheduled_preference: str
    priority_protected: bool
    priority_reason: str
    priority_valid_school_year: str
    primary_requests: tuple[LogicalRequest, ...]
    alternate_requests: tuple[LogicalRequest, ...]


@dataclass(frozen=True)
class SectionMember:
    section_id: str
    course_id: str
    period_1: str
    period_2: str
    semester: str
    block_id: str
    logical_block_id: str
    semester_content: str


@dataclass(frozen=True)
class LogicalSection:
    """One future assignment target, keyed by linked_section_group_id.

    Gov/Econ keeps two member rows here but still represents one roster seat,
    one occupied period, and one future assignment.
    """

    linked_section_group_id: str
    member_sections: tuple[SectionMember, ...]
    course_ids: tuple[str, ...]
    logical_block_id: str
    occupied_periods: tuple[str, ...]
    period_units: int
    capacity: int
    structure_type: str


@dataclass(frozen=True)
class CanonicalAllocationInput:
    students: tuple[CanonicalStudent, ...]
    logical_requests: tuple[LogicalRequest, ...]
    logical_sections: tuple[LogicalSection, ...]
    candidate_index: Mapping[str, tuple[str, ...]]
    students_by_id: Mapping[str, CanonicalStudent]
    requests_by_key: Mapping[str, LogicalRequest]
    logical_sections_by_id: Mapping[str, LogicalSection]
    courses_by_id: Mapping[str, CourseMetadata]
