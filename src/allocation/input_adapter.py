from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from types import MappingProxyType
from typing import Iterable, Mapping

import pandas as pd

from src.section_planning.demand import GOV_ECON_COURSES, logical_block_id_from_request
from src.validation.constants import (
    VALID_GRADES,
    VALID_PERIODS,
    VALID_PRIORITY_REASONS,
    VALID_REQUEST_TYPES,
    VALID_SEMESTERS,
    VALID_UNSCHEDULED_PREFERENCES,
)
from src.validation.helpers import parse_bool, parse_int, text

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


STUDENT_COLUMNS = (
    "student_id",
    "grade",
    "target_course_count",
    "unscheduled_preference",
    "priority_protected",
    "priority_reason",
    "priority_valid_school_year",
)
REQUEST_COLUMNS = (
    "student_id",
    "course_id",
    "request_type",
    "request_rank",
    "request_group",
    "must_share_block_id",
)
SECTION_COLUMNS = (
    "section_id",
    "course_id",
    "period_1",
    "period_2",
    "semester",
    "capacity",
    "block_id",
    "linked_section_group_id",
    "logical_block_id",
    "semester_content",
)
CATALOG_COLUMNS = ("course_id", "periods_required", "schedule_structure")


@dataclass(frozen=True)
class _RawStudent:
    student_id: str
    grade: int
    target_period_units: int
    unscheduled_preference: str
    priority_protected: bool
    priority_reason: str
    priority_valid_school_year: str


@dataclass(frozen=True)
class _RawRequest:
    student_id: str
    course_id: str
    request_type: str
    request_rank: int | None
    request_group: str
    candidate_key: str
    source: SourceRequestRow
    row: int


@dataclass(frozen=True)
class _RawSection:
    member: SectionMember
    linked_section_group_id: str
    capacity: int
    row: int


def canonicalize_allocation_input(
    students: pd.DataFrame,
    requests: pd.DataFrame,
    sections: pd.DataFrame,
    catalog: pd.DataFrame,
) -> CanonicalAllocationInput:
    """Create deterministic, immutable allocation inputs without assigning anyone.

    A future assignment always targets one ``LogicalSection`` identified by its
    ``linked_section_group_id``. For Gov/Econ this means one roster seat and one
    period even though the logical section retains two semester member rows.
    """

    issues: list[AllocationInputIssue] = []
    _require_columns(students, STUDENT_COLUMNS, "students.csv", issues)
    _require_columns(requests, REQUEST_COLUMNS, "requests.csv", issues)
    _require_columns(sections, SECTION_COLUMNS, "sections.csv", issues)
    _require_columns(catalog, CATALOG_COLUMNS, "course_catalog.csv", issues)
    if issues:
        _raise_if_issues(issues)

    courses = _canonicalize_catalog(catalog, issues)
    raw_students = _canonicalize_students(students, issues)
    student_ids = set(raw_students)
    raw_requests = _canonicalize_request_rows(requests, student_ids, courses, issues)
    logical_requests = _build_logical_requests(raw_requests, courses, issues)
    logical_sections = _canonicalize_sections(sections, courses, issues)
    _raise_if_issues(issues)

    requests_by_student: dict[str, list[LogicalRequest]] = defaultdict(list)
    for request in logical_requests:
        requests_by_student[request.student_id].append(request)

    students_result = tuple(
        CanonicalStudent(
            student_id=student.student_id,
            grade=student.grade,
            target_period_units=student.target_period_units,
            unscheduled_preference=student.unscheduled_preference,
            priority_protected=student.priority_protected,
            priority_reason=student.priority_reason,
            priority_valid_school_year=student.priority_valid_school_year,
            primary_requests=tuple(
                request
                for request in requests_by_student.get(student.student_id, [])
                if request.request_type == "primary"
            ),
            alternate_requests=tuple(
                request
                for request in requests_by_student.get(student.student_id, [])
                if request.request_type == "alternate"
            ),
        )
        for student in sorted(raw_students.values(), key=lambda item: item.student_id)
    )
    candidate_index = _build_candidate_index(logical_requests, logical_sections)

    return CanonicalAllocationInput(
        students=students_result,
        logical_requests=logical_requests,
        logical_sections=logical_sections,
        candidate_index=_freeze_mapping(candidate_index),
        students_by_id=_freeze_mapping({student.student_id: student for student in students_result}),
        requests_by_key=_freeze_mapping({request.request_key: request for request in logical_requests}),
        logical_sections_by_id=_freeze_mapping(
            {section.linked_section_group_id: section for section in logical_sections}
        ),
        courses_by_id=_freeze_mapping(courses),
    )


def _canonicalize_catalog(
    catalog: pd.DataFrame,
    issues: list[AllocationInputIssue],
) -> dict[str, CourseMetadata]:
    courses: dict[str, CourseMetadata] = {}
    for row_number, (_, row) in enumerate(catalog.iterrows(), start=2):
        course_id = text(row["course_id"])
        units = parse_int(row["periods_required"])
        structure = text(row["schedule_structure"])
        if not course_id:
            _issue(issues, "EMPTY_COURSE_ID", "course_catalog.csv", "course_id cannot be blank.", row_number)
            continue
        if course_id in courses:
            _issue(issues, "DUPLICATE_COURSE_ID", "course_catalog.csv", "course_id must be unique.", row_number, course_id)
            continue
        if units is None or units <= 0:
            _issue(issues, "INVALID_COURSE_PERIOD_UNITS", "course_catalog.csv", "periods_required must be a positive integer.", row_number, course_id)
            continue
        if not structure:
            _issue(issues, "EMPTY_COURSE_SCHEDULE_STRUCTURE", "course_catalog.csv", "schedule_structure cannot be blank.", row_number, course_id)
            continue
        courses[course_id] = CourseMetadata(course_id, units, structure)
    return dict(sorted(courses.items()))


def _canonicalize_students(
    students: pd.DataFrame,
    issues: list[AllocationInputIssue],
) -> dict[str, _RawStudent]:
    result: dict[str, _RawStudent] = {}
    for row_number, (_, row) in enumerate(students.iterrows(), start=2):
        student_id = text(row["student_id"])
        grade = parse_int(row["grade"])
        target_units = parse_int(row["target_course_count"])
        preference = text(row["unscheduled_preference"])
        protected = parse_bool(row["priority_protected"])
        reason = text(row["priority_reason"])
        valid_year = text(row["priority_valid_school_year"])
        if not student_id:
            _issue(issues, "EMPTY_STUDENT_ID", "students.csv", "student_id cannot be blank.", row_number)
            continue
        if student_id in result:
            _issue(issues, "DUPLICATE_STUDENT_ID", "students.csv", "student_id must be unique.", row_number, student_id)
            continue
        if grade is None or str(grade) not in VALID_GRADES:
            _issue(issues, "INVALID_STUDENT_GRADE", "students.csv", "grade must be 9, 10, 11, or 12.", row_number, student_id)
        if target_units is None or target_units <= 0:
            _issue(issues, "INVALID_TARGET_PERIOD_UNITS", "students.csv", "target_course_count must be a positive period-unit target.", row_number, student_id)
        if preference not in VALID_UNSCHEDULED_PREFERENCES:
            _issue(issues, "INVALID_UNSCHEDULED_PREFERENCE", "students.csv", "Invalid unscheduled_preference.", row_number, student_id)
        if protected is None:
            _issue(issues, "INVALID_PRIORITY_PROTECTED", "students.csv", "priority_protected must be true or false.", row_number, student_id)
        elif reason not in VALID_PRIORITY_REASONS:
            _issue(issues, "INVALID_PRIORITY_REASON", "students.csv", "Invalid priority_reason.", row_number, student_id)
        elif not protected and (reason or valid_year):
            _issue(issues, "INACTIVE_PRIORITY_FIELDS", "students.csv", "Unprotected students must leave priority fields blank.", row_number, student_id)
        elif protected and (not reason or not valid_year):
            _issue(issues, "MISSING_PRIORITY_FIELDS", "students.csv", "Protected students need reason and valid school year.", row_number, student_id)
        if None not in (grade, target_units, protected):
            result[student_id] = _RawStudent(student_id, grade, target_units, preference, protected, reason, valid_year)
    return result


def _canonicalize_request_rows(
    requests: pd.DataFrame,
    student_ids: set[str],
    courses: Mapping[str, CourseMetadata],
    issues: list[AllocationInputIssue],
) -> tuple[_RawRequest, ...]:
    result: list[_RawRequest] = []
    for row_number, (_, row) in enumerate(requests.iterrows(), start=2):
        student_id = text(row["student_id"])
        course_id = text(row["course_id"])
        request_type = text(row["request_type"])
        rank_text = text(row["request_rank"])
        request_group = text(row["request_group"])
        block_id = text(row["must_share_block_id"])
        if student_id not in student_ids:
            _issue(issues, "UNKNOWN_REQUEST_STUDENT", "requests.csv", "Request references an unknown student_id.", row_number, student_id)
        if course_id not in courses:
            _issue(issues, "UNKNOWN_REQUEST_COURSE", "requests.csv", "Request references an unknown course_id.", row_number, course_id)
        if request_type not in VALID_REQUEST_TYPES:
            _issue(issues, "INVALID_REQUEST_TYPE", "requests.csv", "request_type must be primary or alternate.", row_number, student_id)
            continue
        rank = parse_int(rank_text) if rank_text else None
        if request_type == "primary" and rank_text:
            _issue(issues, "PRIMARY_RANK_NOT_BLANK", "requests.csv", "Primary request_rank must be blank.", row_number, student_id)
        if request_type == "alternate" and (rank is None or rank not in {1, 2, 3}):
            _issue(issues, "INVALID_ALTERNATE_RANK", "requests.csv", "Alternate request_rank must be 1, 2, or 3.", row_number, student_id)
        if student_id not in student_ids or course_id not in courses:
            continue
        source = SourceRequestRow(course_id, request_group, block_id, rank)
        candidate_key = logical_block_id_from_request(row)
        if not candidate_key:
            _issue(issues, "EMPTY_LOGICAL_REQUEST_KEY", "requests.csv", "Request has no course or logical block key.", row_number, student_id)
            continue
        result.append(_RawRequest(student_id, course_id, request_type, rank, request_group, candidate_key, source, row_number))
    return tuple(result)


def _build_logical_requests(
    raw_requests: tuple[_RawRequest, ...],
    courses: Mapping[str, CourseMetadata],
    issues: list[AllocationInputIssue],
) -> tuple[LogicalRequest, ...]:
    primary_groups: dict[tuple[str, str], list[_RawRequest]] = defaultdict(list)
    alternate_groups: dict[tuple[str, str], list[_RawRequest]] = defaultdict(list)
    alternate_ranks: dict[tuple[str, int], list[_RawRequest]] = defaultdict(list)
    for request in raw_requests:
        if request.request_type == "primary":
            primary_groups[(request.student_id, request.candidate_key)].append(request)
        else:
            alternate_groups[(request.student_id, request.candidate_key)].append(request)
            if request.request_rank is not None:
                alternate_ranks[(request.student_id, request.request_rank)].append(request)

    result: list[LogicalRequest] = []
    for (student_id, candidate_key), group in sorted(primary_groups.items()):
        source_rows = _sorted_source_rows(group)
        course_ids = tuple(sorted({request.course_id for request in group}))
        course_id = course_ids[0] if len(course_ids) == 1 else ""
        if len(course_ids) != 1:
            _issue(issues, "AMBIGUOUS_LOGICAL_PRIMARY_COURSE", "requests.csv", "A logical primary must resolve to one course_id.", min(request.row for request in group), candidate_key)
            continue
        if course_id in GOV_ECON_COURSES:
            _validate_gov_econ_request_group(student_id, candidate_key, group, issues)
        elif len(group) != 1:
            _issue(issues, "DUPLICATE_LOGICAL_PRIMARY", "requests.csv", "Duplicate logical primary requests are not allowed.", min(request.row for request in group), candidate_key)
            continue
        metadata = courses[course_id]
        result.append(
            LogicalRequest(
                request_key=f"primary:{student_id}:{candidate_key}",
                student_id=student_id,
                request_type="primary",
                candidate_key=candidate_key,
                course_ids=course_ids,
                source_rows=source_rows,
                request_rank=None,
                period_units=metadata.period_units,
            )
        )

    for (student_id, candidate_key), group in sorted(alternate_groups.items()):
        if len(group) != 1:
            _issue(issues, "DUPLICATE_ALTERNATE_LOGICAL_REQUEST", "requests.csv", "Duplicate alternate course or logical block is not allowed.", min(request.row for request in group), candidate_key)
            continue
        request = group[0]
        if request.request_rank is None or request.request_rank not in {1, 2, 3}:
            continue
        metadata = courses[request.course_id]
        result.append(
            LogicalRequest(
                request_key=f"alternate:{student_id}:{request.request_rank}:{candidate_key}",
                student_id=student_id,
                request_type="alternate",
                candidate_key=candidate_key,
                course_ids=(request.course_id,),
                source_rows=(request.source,),
                request_rank=request.request_rank,
                period_units=metadata.period_units,
            )
        )

    for (student_id, rank), group in sorted(alternate_ranks.items()):
        if len(group) > 1:
            _issue(issues, "DUPLICATE_ALTERNATE_RANK", "requests.csv", "Alternate ranks must be unique within each student.", min(request.row for request in group), f"{student_id}:{rank}")

    return tuple(sorted(result, key=_request_sort_key))


def _validate_gov_econ_request_group(
    student_id: str,
    candidate_key: str,
    group: list[_RawRequest],
    issues: list[AllocationInputIssue],
) -> None:
    row = min(request.row for request in group)
    if len(group) != 2:
        _issue(issues, "INVALID_GOV_ECON_REQUEST_ROWS", "requests.csv", "Gov/Econ logical primary must have exactly two semester request rows.", row, candidate_key)
    if candidate_key not in GOV_ECON_COURSES:
        _issue(issues, "INVALID_GOV_ECON_REQUEST_BLOCK", "requests.csv", "Gov/Econ must_share_block_id must match its supported logical block.", row, candidate_key)
    if any(request.request_group != "gov_econ_block" for request in group):
        _issue(issues, "INVALID_GOV_ECON_REQUEST_GROUP", "requests.csv", "Gov/Econ semester request rows must use request_group=gov_econ_block.", row, student_id)


def _canonicalize_sections(
    sections: pd.DataFrame,
    courses: Mapping[str, CourseMetadata],
    issues: list[AllocationInputIssue],
) -> tuple[LogicalSection, ...]:
    raw_sections: list[_RawSection] = []
    section_ids: set[str] = set()
    for row_number, (_, row) in enumerate(sections.iterrows(), start=2):
        section_id = text(row["section_id"])
        course_id = text(row["course_id"])
        period_1 = text(row["period_1"])
        period_2 = text(row["period_2"])
        semester = text(row["semester"])
        capacity = parse_int(row["capacity"])
        group_id = text(row["linked_section_group_id"])
        logical_block_id = text(row["logical_block_id"])
        if not section_id:
            _issue(issues, "EMPTY_SECTION_ID", "sections.csv", "section_id cannot be blank.", row_number)
        elif section_id in section_ids:
            _issue(issues, "DUPLICATE_SECTION_ID", "sections.csv", "section_id must be unique.", row_number, section_id)
        section_ids.add(section_id)
        if not group_id:
            _issue(issues, "EMPTY_LINKED_SECTION_GROUP_ID", "sections.csv", "linked_section_group_id cannot be blank.", row_number, section_id)
        if course_id not in courses:
            _issue(issues, "UNKNOWN_SECTION_COURSE", "sections.csv", "Section references an unknown course_id.", row_number, course_id)
        if not logical_block_id:
            _issue(issues, "EMPTY_LOGICAL_BLOCK_ID", "sections.csv", "logical_block_id cannot be blank.", row_number, section_id)
        if period_1 not in VALID_PERIODS:
            _issue(issues, "INVALID_SECTION_PERIOD", "sections.csv", "period_1 must be P1 through P7.", row_number, section_id)
        if period_2 and period_2 not in VALID_PERIODS:
            _issue(issues, "INVALID_SECTION_PERIOD", "sections.csv", "period_2 must be P1 through P7 or blank.", row_number, section_id)
        if period_1 and period_1 == period_2:
            _issue(issues, "DUPLICATE_OCCUPIED_PERIOD", "sections.csv", "period_1 and period_2 cannot be the same.", row_number, section_id)
        if semester not in VALID_SEMESTERS:
            _issue(issues, "INVALID_SECTION_SEMESTER", "sections.csv", "Invalid semester value.", row_number, section_id)
        if capacity is None or capacity <= 0:
            _issue(issues, "INVALID_SECTION_CAPACITY", "sections.csv", "capacity must be a positive integer.", row_number, section_id)
        if (
            section_id
            and group_id
            and course_id in courses
            and logical_block_id
            and period_1 in VALID_PERIODS
            and (not period_2 or period_2 in VALID_PERIODS)
            and period_1 != period_2
            and semester in VALID_SEMESTERS
            and capacity is not None
            and capacity > 0
        ):
            raw_sections.append(
                _RawSection(
                    SectionMember(
                        section_id=section_id,
                        course_id=course_id,
                        period_1=period_1,
                        period_2=period_2,
                        semester=semester,
                        block_id=text(row["block_id"]),
                        logical_block_id=logical_block_id,
                        semester_content=text(row["semester_content"]),
                    ),
                    group_id,
                    capacity,
                    row_number,
                )
            )

    groups: dict[str, list[_RawSection]] = defaultdict(list)
    for section in raw_sections:
        groups[section.linked_section_group_id].append(section)
    logical_sections = [
        _build_logical_section(group_id, group, courses, issues)
        for group_id, group in sorted(groups.items())
    ]
    return tuple(section for section in logical_sections if section is not None)


def _build_logical_section(
    group_id: str,
    group: list[_RawSection],
    courses: Mapping[str, CourseMetadata],
    issues: list[AllocationInputIssue],
) -> LogicalSection | None:
    row = min(section.row for section in group)
    members = tuple(sorted((section.member for section in group), key=lambda member: member.section_id))
    capacities = {section.capacity for section in group}
    course_ids = tuple(sorted({member.course_id for member in members}))
    logical_block_ids = {member.logical_block_id for member in members}
    semesters = {member.semester for member in members}
    if len(capacities) != 1:
        _issue(issues, "INCONSISTENT_LOGICAL_SECTION_CAPACITY", "sections.csv", "All rows in a logical section must share capacity.", row, group_id)
    if len(course_ids) != 1:
        _issue(issues, "AMBIGUOUS_LOGICAL_SECTION_COURSE", "sections.csv", "A logical section must resolve to one course_id.", row, group_id)
    if len(logical_block_ids) != 1:
        _issue(issues, "INCONSISTENT_LOGICAL_BLOCK_ID", "sections.csv", "All rows in a logical section must share logical_block_id.", row, group_id)
    if len(capacities) != 1 or len(course_ids) != 1 or len(logical_block_ids) != 1:
        return None

    course_id = course_ids[0]
    metadata = courses[course_id]
    logical_block_id = next(iter(logical_block_ids))
    capacity = next(iter(capacities))
    if semesters & {"semester_1", "semester_2"}:
        return _build_linked_semester_section(group_id, members, metadata, logical_block_id, capacity, issues, row)
    if "paired" in semesters:
        return _build_paired_content_section(group_id, members, metadata, logical_block_id, capacity, issues, row)
    if metadata.schedule_structure == "double_period" or any(member.period_2 for member in members):
        return _build_double_period_section(group_id, members, metadata, logical_block_id, capacity, issues, row)
    return _build_normal_section(group_id, members, metadata, logical_block_id, capacity, issues, row)


def _build_normal_section(
    group_id: str,
    members: tuple[SectionMember, ...],
    metadata: CourseMetadata,
    logical_block_id: str,
    capacity: int,
    issues: list[AllocationInputIssue],
    row: int,
) -> LogicalSection | None:
    if len(members) != 1 or members[0].semester != "full_year" or members[0].period_2:
        _issue(issues, "INVALID_NORMAL_SECTION_STRUCTURE", "sections.csv", "A normal logical section must be one full_year row with one period.", row, group_id)
        return None
    if metadata.period_units != 1 or metadata.schedule_structure != "standard":
        _issue(issues, "NORMAL_SECTION_CATALOG_MISMATCH", "sections.csv", "Normal section must match a one-period standard catalog course.", row, group_id)
        return None
    if members[0].course_id in GOV_ECON_COURSES:
        _issue(issues, "GOV_ECON_SECTION_STRUCTURE", "sections.csv", "Gov/Econ requires two linked semester rows.", row, group_id)
        return None
    if members[0].course_id == "AP_PHYSC":
        _issue(issues, "AP_PHYSC_SECTION_STRUCTURE", "sections.csv", "AP Physics C requires one paired-content row.", row, group_id)
        return None
    return LogicalSection(group_id, members, (members[0].course_id,), logical_block_id, (members[0].period_1,), 1, capacity, "normal")


def _build_double_period_section(
    group_id: str,
    members: tuple[SectionMember, ...],
    metadata: CourseMetadata,
    logical_block_id: str,
    capacity: int,
    issues: list[AllocationInputIssue],
    row: int,
) -> LogicalSection | None:
    if len(members) != 1 or members[0].semester != "full_year" or not members[0].period_2:
        _issue(issues, "INVALID_DOUBLE_PERIOD_STRUCTURE", "sections.csv", "A double-period logical section must be one full_year row with two periods.", row, group_id)
        return None
    first = int(members[0].period_1[1:])
    second = int(members[0].period_2[1:])
    if second - first != 1:
        _issue(issues, "NONCONSECUTIVE_DOUBLE_PERIOD", "sections.csv", "Double-period sections must use consecutive periods.", row, group_id)
        return None
    if metadata.period_units != 2 or metadata.schedule_structure != "double_period":
        _issue(issues, "DOUBLE_PERIOD_CATALOG_MISMATCH", "sections.csv", "Double-period section must match a two-period double_period catalog course.", row, group_id)
        return None
    return LogicalSection(group_id, members, (members[0].course_id,), logical_block_id, (members[0].period_1, members[0].period_2), 2, capacity, "double_period")


def _build_linked_semester_section(
    group_id: str,
    members: tuple[SectionMember, ...],
    metadata: CourseMetadata,
    logical_block_id: str,
    capacity: int,
    issues: list[AllocationInputIssue],
    row: int,
) -> LogicalSection | None:
    periods = {member.period_1 for member in members}
    block_ids = {member.block_id for member in members}
    if (
        len(members) != 2
        or {member.semester for member in members} != {"semester_1", "semester_2"}
        or len(periods) != 1
        or any(member.period_2 for member in members)
        or len(block_ids) != 1
        or not next(iter(block_ids))
    ):
        _issue(issues, "INVALID_LINKED_SEMESTER_STRUCTURE", "sections.csv", "Linked semester section must contain two matching semester rows in one period.", row, group_id)
        return None
    course_id = members[0].course_id
    if course_id not in GOV_ECON_COURSES:
        _issue(issues, "UNSUPPORTED_LINKED_SEMESTER_COURSE", "sections.csv", "Only configured Gov/Econ courses use two-row linked semester sections in V1.", row, group_id)
        return None
    if logical_block_id != course_id or metadata.period_units != 1 or metadata.schedule_structure != "semester_block":
        _issue(issues, "GOV_ECON_SECTION_CATALOG_MISMATCH", "sections.csv", "Gov/Econ section must match its one-period semester_block catalog course.", row, group_id)
        return None
    return LogicalSection(group_id, members, (course_id,), logical_block_id, (next(iter(periods)),), 1, capacity, "linked_semester")


def _build_paired_content_section(
    group_id: str,
    members: tuple[SectionMember, ...],
    metadata: CourseMetadata,
    logical_block_id: str,
    capacity: int,
    issues: list[AllocationInputIssue],
    row: int,
) -> LogicalSection | None:
    if len(members) != 1 or members[0].semester != "paired" or members[0].period_2:
        _issue(issues, "INVALID_PAIRED_CONTENT_STRUCTURE", "sections.csv", "Paired-content section must be one row with one period.", row, group_id)
        return None
    if metadata.period_units != 1 or metadata.schedule_structure != "semester_block":
        _issue(issues, "PAIRED_CONTENT_CATALOG_MISMATCH", "sections.csv", "Paired-content section must match a one-period semester_block catalog course.", row, group_id)
        return None
    return LogicalSection(group_id, members, (members[0].course_id,), logical_block_id, (members[0].period_1,), 1, capacity, "paired_content")


def _build_candidate_index(
    requests: tuple[LogicalRequest, ...],
    sections: tuple[LogicalSection, ...],
) -> dict[str, tuple[str, ...]]:
    by_candidate_key: dict[str, list[str]] = defaultdict(list)
    for section in sections:
        by_candidate_key[section.logical_block_id].append(section.linked_section_group_id)
    return {
        request.request_key: tuple(sorted(by_candidate_key.get(request.candidate_key, [])))
        for request in requests
    }


def _sorted_source_rows(group: Iterable[_RawRequest]) -> tuple[SourceRequestRow, ...]:
    return tuple(
        request.source
        for request in sorted(
            group,
            key=lambda item: (
                item.course_id,
                item.request_group,
                item.candidate_key,
                item.request_rank if item.request_rank is not None else -1,
            ),
        )
    )


def _request_sort_key(request: LogicalRequest) -> tuple[str, int, int, str]:
    type_order = 0 if request.request_type == "primary" else 1
    rank = request.request_rank if request.request_rank is not None else 0
    return (request.student_id, type_order, rank, request.candidate_key)


def _require_columns(
    df: pd.DataFrame,
    expected: Iterable[str],
    source: str,
    issues: list[AllocationInputIssue],
) -> None:
    missing = sorted(set(expected) - set(df.columns))
    if missing:
        _issue(issues, "MISSING_COLUMNS", source, f"Missing required columns: {', '.join(missing)}.")


def _issue(
    issues: list[AllocationInputIssue],
    code: str,
    source: str,
    message: str,
    row: int | None = None,
    identifier: str | None = None,
) -> None:
    issues.append(AllocationInputIssue(code, source, message, row, identifier))


def _raise_if_issues(issues: list[AllocationInputIssue]) -> None:
    if issues:
        raise AllocationInputError(tuple(issues))


def _freeze_mapping(values: Mapping[str, object]) -> Mapping[str, object]:
    return MappingProxyType(dict(sorted(values.items())))
