from __future__ import annotations

from collections import Counter
from types import MappingProxyType

from .assignment_models import (
    AssignmentFeasibility,
    AssignmentRecord,
    AssignmentRejectionReason,
    AssignmentResult,
    StateConsistencyIssue,
)
from .input_models import CanonicalAllocationInput, LogicalRequest, LogicalSection


MANDATORY_FALLBACK_REQUEST_TYPE = "mandatory_fallback"


class AllocationState:
    """Mutable fixed-section assignment state.

    This class only checks local assignment feasibility and applies atomic
    assignment/unassignment mutations. It does not choose students, fill
    alternates, evaluate fairness policy, or compute final outcomes.
    """

    def __init__(
        self,
        allocation_input: CanonicalAllocationInput,
        *,
        supplemental_requests: tuple[LogicalRequest, ...] = (),
        supplemental_candidate_index: dict[str, tuple[str, ...]] | None = None,
    ):
        self._input = allocation_input
        self._requests_by_key = _build_request_index(allocation_input, supplemental_requests)
        self._candidate_index = _build_candidate_index(allocation_input, supplemental_requests, supplemental_candidate_index or {})
        self._assignments: dict[str, AssignmentRecord] = {}
        self._request_assignment_key: dict[str, str] = {}
        self._student_assignment_keys: dict[str, set[str]] = {
            student.student_id: set() for student in allocation_input.students
        }
        self._student_request_keys: dict[str, set[str]] = {
            student.student_id: set() for student in allocation_input.students
        }
        self._student_candidate_counts: dict[str, Counter[str]] = {
            student.student_id: Counter() for student in allocation_input.students
        }
        self._student_period_counts: dict[str, Counter[str]] = {
            student.student_id: Counter() for student in allocation_input.students
        }
        self._student_used_units: dict[str, int] = {
            student.student_id: 0 for student in allocation_input.students
        }
        self._section_rosters: dict[str, set[str]] = {
            section.linked_section_group_id: set() for section in allocation_input.logical_sections
        }
        self._section_assigned_counts: dict[str, int] = {
            section.linked_section_group_id: 0 for section in allocation_input.logical_sections
        }

    def student_assignments(self, student_id: str) -> tuple[AssignmentRecord, ...]:
        self._require_student(student_id)
        return tuple(self._assignments[key] for key in sorted(self._student_assignment_keys[student_id]))

    def student_occupied_periods(self, student_id: str) -> frozenset[str]:
        self._require_student(student_id)
        return frozenset(period for period, count in self._student_period_counts[student_id].items() if count > 0)

    def student_used_period_units(self, student_id: str) -> int:
        self._require_student(student_id)
        return self._student_used_units[student_id]

    def student_remaining_period_units(self, student_id: str) -> int:
        self._require_student(student_id)
        student = self._input.students_by_id[student_id]
        return student.target_period_units - self._student_used_units[student_id]

    def section_roster(self, linked_section_group_id: str) -> tuple[str, ...]:
        self._require_section(linked_section_group_id)
        return tuple(sorted(self._section_rosters[linked_section_group_id]))

    def section_assigned_count(self, linked_section_group_id: str) -> int:
        self._require_section(linked_section_group_id)
        return self._section_assigned_counts[linked_section_group_id]

    def section_remaining_capacity(self, linked_section_group_id: str) -> int:
        self._require_section(linked_section_group_id)
        section = self._input.logical_sections_by_id[linked_section_group_id]
        return section.capacity - self._section_assigned_counts[linked_section_group_id]

    def is_request_assigned(self, request_key: str) -> bool:
        self._require_request(request_key)
        return request_key in self._request_assignment_key

    def get_assignment(self, assignment_key: str) -> AssignmentRecord:
        try:
            return self._assignments[assignment_key]
        except KeyError as exc:
            raise KeyError(f"Unknown assignment_key: {assignment_key}") from exc

    def all_assignments(self) -> tuple[AssignmentRecord, ...]:
        return tuple(self._assignments[key] for key in sorted(self._assignments))

    def check_assignment(
        self,
        student_id: str,
        request_key: str,
        linked_section_group_id: str,
    ) -> AssignmentFeasibility:
        reasons = self._assignment_rejection_reasons(student_id, request_key, linked_section_group_id)
        return AssignmentFeasibility(
            student_id=student_id,
            request_key=request_key,
            linked_section_group_id=linked_section_group_id,
            allowed=not reasons,
            reasons=reasons,
        )

    def try_assign(
        self,
        student_id: str,
        request_key: str,
        linked_section_group_id: str,
    ) -> AssignmentResult:
        check = self.check_assignment(student_id, request_key, linked_section_group_id)
        if not check.allowed:
            return AssignmentResult(
                student_id=student_id,
                request_key=request_key,
                linked_section_group_id=linked_section_group_id,
                allowed=False,
                reasons=check.reasons,
            )

        request = self._requests_by_key[request_key]
        section = self._input.logical_sections_by_id[linked_section_group_id]
        assignment = self._build_assignment_record(student_id, request, section)
        self._apply_assignment(assignment)
        return AssignmentResult(
            student_id=student_id,
            request_key=request_key,
            linked_section_group_id=linked_section_group_id,
            allowed=True,
            assignment=assignment,
        )

    def unassign(self, assignment_key: str) -> AssignmentResult:
        assignment = self._assignments.get(assignment_key)
        if assignment is None:
            return AssignmentResult(
                student_id="",
                request_key="",
                linked_section_group_id="",
                allowed=False,
                reasons=(AssignmentRejectionReason.UNKNOWN_ASSIGNMENT,),
            )

        self._remove_assignment(assignment)
        return AssignmentResult(
            student_id=assignment.student_id,
            request_key=assignment.request_key,
            linked_section_group_id=assignment.linked_section_group_id,
            allowed=True,
            assignment=assignment,
        )

    def validate_internal_consistency(self) -> tuple[StateConsistencyIssue, ...]:
        issues: list[StateConsistencyIssue] = []
        expected_student_assignments: dict[str, set[str]] = {
            student.student_id: set() for student in self._input.students
        }
        expected_student_requests: dict[str, set[str]] = {
            student.student_id: set() for student in self._input.students
        }
        expected_candidate_counts: dict[str, Counter[str]] = {
            student.student_id: Counter() for student in self._input.students
        }
        expected_period_counts: dict[str, Counter[str]] = {
            student.student_id: Counter() for student in self._input.students
        }
        expected_used_units: dict[str, int] = {
            student.student_id: 0 for student in self._input.students
        }
        expected_rosters: dict[str, set[str]] = {
            section.linked_section_group_id: set() for section in self._input.logical_sections
        }
        expected_section_counts: dict[str, int] = {
            section.linked_section_group_id: 0 for section in self._input.logical_sections
        }

        seen_requests: set[str] = set()
        for key, assignment in sorted(self._assignments.items()):
            if key != assignment.assignment_key:
                issues.append(_consistency_issue("ASSIGNMENT_KEY_MISMATCH", "Assignment table key does not match record.", key))
            if assignment.student_id not in self._input.students_by_id:
                issues.append(_consistency_issue("UNKNOWN_ASSIGNMENT_STUDENT", "Assignment points to unknown student.", key))
                continue
            if assignment.request_key not in self._requests_by_key:
                issues.append(_consistency_issue("UNKNOWN_ASSIGNMENT_REQUEST", "Assignment points to unknown request.", key))
                continue
            if assignment.linked_section_group_id not in self._input.logical_sections_by_id:
                issues.append(_consistency_issue("UNKNOWN_ASSIGNMENT_SECTION", "Assignment points to unknown section.", key))
                continue
            if assignment.request_key in seen_requests:
                issues.append(_consistency_issue("REQUEST_ASSIGNED_MULTIPLE_TIMES", "A request has more than one assignment.", assignment.request_key))
            seen_requests.add(assignment.request_key)

            expected_student_assignments[assignment.student_id].add(key)
            expected_student_requests[assignment.student_id].add(assignment.request_key)
            expected_candidate_counts[assignment.student_id][assignment.request_candidate_key] += 1
            for period in assignment.occupied_periods:
                expected_period_counts[assignment.student_id][period] += 1
            expected_used_units[assignment.student_id] += assignment.period_units
            expected_rosters[assignment.linked_section_group_id].add(assignment.student_id)
            expected_section_counts[assignment.linked_section_group_id] += 1

        self._compare_mapping(issues, "STUDENT_ASSIGNMENT_INDEX_MISMATCH", self._student_assignment_keys, expected_student_assignments)
        self._compare_mapping(issues, "STUDENT_REQUEST_INDEX_MISMATCH", self._student_request_keys, expected_student_requests)
        self._compare_mapping(issues, "STUDENT_LOGICAL_IDENTITY_INDEX_MISMATCH", self._student_candidate_counts, expected_candidate_counts)
        self._compare_mapping(issues, "STUDENT_PERIOD_INDEX_MISMATCH", self._student_period_counts, expected_period_counts)
        self._compare_mapping(issues, "STUDENT_USED_UNITS_MISMATCH", self._student_used_units, expected_used_units)
        self._compare_mapping(issues, "SECTION_ROSTER_MISMATCH", self._section_rosters, expected_rosters)
        self._compare_mapping(issues, "SECTION_COUNT_MISMATCH", self._section_assigned_counts, expected_section_counts)
        for section_id, count in sorted(self._section_assigned_counts.items()):
            capacity = self._input.logical_sections_by_id[section_id].capacity
            if count > capacity:
                issues.append(_consistency_issue("SECTION_OVER_CAPACITY", "Section assigned count exceeds capacity.", section_id))
        if set(self._request_assignment_key) != seen_requests:
            issues.append(_consistency_issue("REQUEST_ASSIGNMENT_TABLE_MISMATCH", "Request assignment table does not match assignments.", ""))
        for request_key, assignment_key in sorted(self._request_assignment_key.items()):
            assignment = self._assignments.get(assignment_key)
            if assignment is None or assignment.request_key != request_key:
                issues.append(_consistency_issue("REQUEST_ASSIGNMENT_KEY_MISMATCH", "Request assignment points to the wrong assignment.", request_key))
        return tuple(issues)

    @property
    def assignment_index(self):
        return MappingProxyType(dict(sorted(self._assignments.items())))

    def _assignment_rejection_reasons(
        self,
        student_id: str,
        request_key: str,
        linked_section_group_id: str,
    ) -> tuple[AssignmentRejectionReason, ...]:
        reasons: list[AssignmentRejectionReason] = []
        student = self._input.students_by_id.get(student_id)
        request = self._requests_by_key.get(request_key)
        section = self._input.logical_sections_by_id.get(linked_section_group_id)

        if student is None:
            reasons.append(AssignmentRejectionReason.UNKNOWN_STUDENT)
        if request is None:
            reasons.append(AssignmentRejectionReason.UNKNOWN_REQUEST)
        if section is None:
            reasons.append(AssignmentRejectionReason.UNKNOWN_SECTION)
        if student is None or request is None or section is None:
            return tuple(reasons)

        if request.student_id != student_id:
            reasons.append(AssignmentRejectionReason.REQUEST_NOT_OWNED_BY_STUDENT)
            if linked_section_group_id not in self._candidate_index.get(request_key, ()):
                reasons.append(AssignmentRejectionReason.SECTION_NOT_CANDIDATE_FOR_REQUEST)
            return tuple(reasons)

        if linked_section_group_id not in self._candidate_index.get(request_key, ()):
            reasons.append(AssignmentRejectionReason.SECTION_NOT_CANDIDATE_FOR_REQUEST)
        if request_key in self._request_assignment_key:
            reasons.append(AssignmentRejectionReason.REQUEST_ALREADY_ASSIGNED)
        if self._student_candidate_counts[student_id][request.candidate_key] > 0:
            reasons.append(AssignmentRejectionReason.DUPLICATE_LOGICAL_COURSE_OR_BLOCK)
        if set(self._student_period_counts[student_id]) & set(section.occupied_periods):
            reasons.append(AssignmentRejectionReason.PERIOD_CONFLICT)
        if self._student_used_units[student_id] + request.period_units > student.target_period_units:
            reasons.append(AssignmentRejectionReason.TARGET_LOAD_EXCEEDED)
        if self._section_assigned_counts[linked_section_group_id] >= section.capacity:
            reasons.append(AssignmentRejectionReason.SECTION_FULL)
        if request.period_units != section.period_units:
            reasons.append(AssignmentRejectionReason.PERIOD_UNIT_MISMATCH)
        return tuple(reasons)

    def _build_assignment_record(
        self,
        student_id: str,
        request: LogicalRequest,
        section: LogicalSection,
    ) -> AssignmentRecord:
        return AssignmentRecord(
            assignment_key=_assignment_key(student_id, request.request_key, section.linked_section_group_id),
            student_id=student_id,
            request_key=request.request_key,
            request_type=request.request_type,
            alternate_rank=request.request_rank,
            request_candidate_key=request.candidate_key,
            linked_section_group_id=section.linked_section_group_id,
            member_section_ids=tuple(member.section_id for member in section.member_sections),
            occupied_periods=section.occupied_periods,
            period_units=section.period_units,
            structure_type=section.structure_type,
            logical_block_id=section.logical_block_id,
            course_ids=section.course_ids,
        )

    def _apply_assignment(self, assignment: AssignmentRecord) -> None:
        self._assignments[assignment.assignment_key] = assignment
        self._request_assignment_key[assignment.request_key] = assignment.assignment_key
        self._student_assignment_keys[assignment.student_id].add(assignment.assignment_key)
        self._student_request_keys[assignment.student_id].add(assignment.request_key)
        self._student_candidate_counts[assignment.student_id][assignment.request_candidate_key] += 1
        for period in assignment.occupied_periods:
            self._student_period_counts[assignment.student_id][period] += 1
        self._student_used_units[assignment.student_id] += assignment.period_units
        self._section_rosters[assignment.linked_section_group_id].add(assignment.student_id)
        self._section_assigned_counts[assignment.linked_section_group_id] += 1

    def _remove_assignment(self, assignment: AssignmentRecord) -> None:
        del self._assignments[assignment.assignment_key]
        del self._request_assignment_key[assignment.request_key]
        self._student_assignment_keys[assignment.student_id].remove(assignment.assignment_key)
        self._student_request_keys[assignment.student_id].remove(assignment.request_key)
        _decrement_counter(self._student_candidate_counts[assignment.student_id], assignment.request_candidate_key)
        for period in assignment.occupied_periods:
            _decrement_counter(self._student_period_counts[assignment.student_id], period)
        self._student_used_units[assignment.student_id] -= assignment.period_units
        self._section_rosters[assignment.linked_section_group_id].remove(assignment.student_id)
        self._section_assigned_counts[assignment.linked_section_group_id] -= 1

    def _require_student(self, student_id: str) -> None:
        if student_id not in self._input.students_by_id:
            raise KeyError(f"Unknown student_id: {student_id}")

    def _require_request(self, request_key: str) -> None:
        if request_key not in self._requests_by_key:
            raise KeyError(f"Unknown request_key: {request_key}")

    def _require_section(self, linked_section_group_id: str) -> None:
        if linked_section_group_id not in self._input.logical_sections_by_id:
            raise KeyError(f"Unknown linked_section_group_id: {linked_section_group_id}")

    @staticmethod
    def _compare_mapping(
        issues: list[StateConsistencyIssue],
        code: str,
        actual: dict,
        expected: dict,
    ) -> None:
        if actual != expected:
            issues.append(_consistency_issue(code, "Internal index does not match assignments.", ""))


def _assignment_key(student_id: str, request_key: str, linked_section_group_id: str) -> str:
    return f"{student_id}|{request_key}|{linked_section_group_id}"


def _build_request_index(
    allocation_input: CanonicalAllocationInput,
    supplemental_requests: tuple[LogicalRequest, ...],
) -> MappingProxyType:
    requests = dict(allocation_input.requests_by_key)
    seen_supplemental: set[str] = set()
    for request in supplemental_requests:
        if request.request_key in requests or request.request_key in seen_supplemental:
            raise ValueError(f"Supplemental request_key collides with an existing request: {request.request_key}")
        if request.request_type != MANDATORY_FALLBACK_REQUEST_TYPE:
            raise ValueError(f"Supplemental request_type must be {MANDATORY_FALLBACK_REQUEST_TYPE}: {request.request_key}")
        if request.student_id not in allocation_input.students_by_id:
            raise ValueError(f"Supplemental request references unknown student_id: {request.student_id}")
        if not request.course_ids:
            raise ValueError(f"Supplemental request must contain at least one course_id: {request.request_key}")
        for course_id in request.course_ids:
            if course_id not in allocation_input.courses_by_id:
                raise ValueError(f"Supplemental request references unknown course_id: {course_id}")
        if len(request.course_ids) == 1:
            course = allocation_input.courses_by_id[request.course_ids[0]]
            if request.period_units != course.period_units:
                raise ValueError(f"Supplemental request period_units do not match course metadata: {request.request_key}")
        seen_supplemental.add(request.request_key)
        requests[request.request_key] = request
    return MappingProxyType(dict(sorted(requests.items())))


def _build_candidate_index(
    allocation_input: CanonicalAllocationInput,
    supplemental_requests: tuple[LogicalRequest, ...],
    supplemental_candidate_index: dict[str, tuple[str, ...]],
) -> MappingProxyType:
    candidate_index = dict(allocation_input.candidate_index)
    supplemental_keys = {request.request_key for request in supplemental_requests}
    unexpected_keys = set(supplemental_candidate_index) - supplemental_keys
    if unexpected_keys:
        raise ValueError(f"Supplemental candidate index references unknown supplemental requests: {sorted(unexpected_keys)}")
    for request in supplemental_requests:
        section_ids = supplemental_candidate_index.get(request.request_key, ())
        stable_section_ids = tuple(sorted(section_ids))
        if len(stable_section_ids) != len(set(stable_section_ids)):
            raise ValueError(f"Supplemental candidate index contains duplicate section ids: {request.request_key}")
        for section_id in stable_section_ids:
            section = allocation_input.logical_sections_by_id.get(section_id)
            if section is None:
                raise ValueError(f"Supplemental candidate index references unknown section: {section_id}")
            if section.logical_block_id != request.candidate_key:
                raise ValueError(
                    f"Supplemental candidate section {section_id} does not match request candidate_key {request.candidate_key}."
                )
        candidate_index[request.request_key] = stable_section_ids
    return MappingProxyType(dict(sorted(candidate_index.items())))


def _decrement_counter(counter: Counter[str], key: str) -> None:
    counter[key] -= 1
    if counter[key] <= 0:
        del counter[key]


def _consistency_issue(code: str, message: str, identifier: str) -> StateConsistencyIssue:
    return StateConsistencyIssue(code=code, message=message, identifier=identifier)
