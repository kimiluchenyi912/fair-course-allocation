from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.allocation import (
    AlternateRequestStatus,
    AssignmentRejectionReason,
    CandidateAttempt,
    PrimaryRequestStatus,
    RequestOutcome,
    StudentOutcome,
)
from src.benchmark_runner import (
    REQUEST_OUTCOMES_FIELDNAMES,
    STUDENT_OUTCOMES_FIELDNAMES,
    _assignment_failure_summary_rows,
    _request_failure_diagnostics,
    _request_outcome_rows,
    _student_schedule_gap_rows,
    _student_outcome_rows,
)


ORIGINAL_REQUEST_OUTCOME_COLUMNS = (
    "algorithm_name",
    "request_key",
    "student_id",
    "request_type",
    "alternate_rank",
    "candidate_key",
    "period_units",
    "status",
    "assignment_key",
    "assigned_linked_section_group_id",
    "candidate_attempts_count",
    "remaining_units_before",
    "remaining_units_after",
)

ORIGINAL_STUDENT_OUTCOME_COLUMNS = (
    "algorithm_name",
    "student_id",
    "grade",
    "target_period_units",
    "assigned_period_units",
    "remaining_period_units",
    "assignment_keys",
    "primary_request_count",
    "primary_assigned_count",
    "primary_unmet_count",
    "primary_unmet_request_keys",
    "primary_unmet_period_units",
    "alternate_request_count",
    "alternate_assigned_count",
    "alternate_assigned_period_units",
    "mandatory_fallback_assigned_count",
    "mandatory_fallback_assigned_period_units",
    "mandatory_fallback_assignment_keys",
    "fully_scheduled",
    "priority_protected",
    "ordinary_fairness_violation",
    "protected_fairness_violation",
    "high_demand_guarantee_violation_count",
    "high_demand_violating_request_keys",
)


def _attempt(index: int, *reasons: AssignmentRejectionReason, success: bool = False) -> CandidateAttempt:
    return CandidateAttempt(
        attempt_index=index,
        linked_section_group_id=f"SEC_{index}",
        success=success,
        rejection_reasons=tuple(reasons),
        assignment_key=f"ASSIGN_{index}" if success else None,
    )


def _request_outcome(
    status=PrimaryRequestStatus.UNMET_ALL_CANDIDATES_REJECTED,
    *,
    request_key: str = "primary:STU_1:CORE_A",
    student_id: str = "STU_1",
    request_type: str = "primary",
    alternate_rank: int | None = None,
    attempts: tuple[CandidateAttempt, ...] = (),
    assignment_key: str | None = None,
) -> RequestOutcome:
    return RequestOutcome(
        request_key=request_key,
        student_id=student_id,
        request_type=request_type,
        alternate_rank=alternate_rank,
        candidate_key="CORE_A",
        period_units=1,
        status=status,
        assignment_key=assignment_key,
        assigned_linked_section_group_id="SEC_ASSIGNED" if assignment_key else None,
        candidate_attempts=attempts,
        remaining_units_before=1,
        remaining_units_after=0 if assignment_key else 1,
    )


def _student_outcome(*, remaining: int, primary_unmet: int = 1) -> StudentOutcome:
    return StudentOutcome(
        student_id="STU_1",
        grade=11,
        target_period_units=2,
        assigned_period_units=2 - remaining,
        remaining_period_units=remaining,
        assignment_keys=("ASSIGN_1",) if remaining < 2 else (),
        primary_request_count=1,
        primary_assigned_count=1 - primary_unmet,
        primary_unmet_count=primary_unmet,
        primary_unmet_request_keys=("primary:STU_1:CORE_A",) if primary_unmet else (),
        primary_unmet_period_units=primary_unmet,
        alternate_request_count=1,
        alternate_assigned_count=1 if remaining == 0 else 0,
        alternate_assigned_period_units=1 if remaining == 0 else 0,
        mandatory_fallback_assigned_count=0,
        mandatory_fallback_assigned_period_units=0,
        mandatory_fallback_assignment_keys=(),
        fully_scheduled=remaining == 0,
        priority_protected=False,
        ordinary_fairness_violation=False,
        protected_fairness_violation=False,
        high_demand_guarantee_violation_count=0,
        high_demand_violating_request_keys=(),
    )


def test_request_failure_diagnostics_classifies_rejected_candidates_once() -> None:
    outcome = _request_outcome(
        attempts=(
            _attempt(1, AssignmentRejectionReason.PERIOD_CONFLICT, AssignmentRejectionReason.SECTION_FULL),
            _attempt(2, AssignmentRejectionReason.SECTION_FULL),
            _attempt(
                3,
                AssignmentRejectionReason.DUPLICATE_LOGICAL_COURSE_OR_BLOCK,
                AssignmentRejectionReason.PERIOD_CONFLICT,
            ),
            _attempt(4, AssignmentRejectionReason.TARGET_LOAD_EXCEEDED),
            _attempt(5, AssignmentRejectionReason.PERIOD_UNIT_MISMATCH),
        )
    )

    diagnostics = _request_failure_diagnostics(outcome)

    assert diagnostics["candidate_attempts_count"] == 5
    assert diagnostics["candidate_rejections_count"] == 5
    assert diagnostics["rejected_section_at_capacity_count"] == 1
    assert diagnostics["rejected_period_conflict_count"] == 1
    assert diagnostics["rejected_duplicate_logical_course_count"] == 1
    assert diagnostics["rejected_student_load_limit_count"] == 1
    assert diagnostics["rejected_other_count"] == 1
    assert diagnostics["terminal_unmet_reason"] == "mixed_rejections"


@pytest.mark.parametrize(
    ("reason", "count_field", "terminal_reason"),
    [
        pytest.param(
            AssignmentRejectionReason.SECTION_FULL,
            "rejected_section_at_capacity_count",
            "all_section_at_capacity",
            id="section_at_capacity",
        ),
        pytest.param(
            AssignmentRejectionReason.PERIOD_CONFLICT,
            "rejected_period_conflict_count",
            "all_period_conflict",
            id="period_conflict",
        ),
        pytest.param(
            AssignmentRejectionReason.DUPLICATE_LOGICAL_COURSE_OR_BLOCK,
            "rejected_duplicate_logical_course_count",
            "all_duplicate_logical_course",
            id="duplicate_logical_course",
        ),
        pytest.param(
            AssignmentRejectionReason.TARGET_LOAD_EXCEEDED,
            "rejected_student_load_limit_count",
            "all_student_load_limit",
            id="student_load_limit",
        ),
        pytest.param(
            AssignmentRejectionReason.PERIOD_UNIT_MISMATCH,
            "rejected_other_count",
            "other_rejection",
            id="other_rejection",
        ),
    ],
)
def test_each_rejection_reason_maps_to_one_count_field(
    reason: AssignmentRejectionReason,
    count_field: str,
    terminal_reason: str,
) -> None:
    diagnostics = _request_failure_diagnostics(_request_outcome(attempts=(_attempt(1, reason),)))
    reason_count_fields = [
        "rejected_section_at_capacity_count",
        "rejected_period_conflict_count",
        "rejected_duplicate_logical_course_count",
        "rejected_student_load_limit_count",
        "rejected_other_count",
    ]

    assert diagnostics[count_field] == 1
    assert sum(diagnostics[field] for field in reason_count_fields) == 1
    assert diagnostics["candidate_rejections_count"] == 1
    assert diagnostics["terminal_unmet_reason"] == terminal_reason


def test_request_failure_diagnostics_terminal_categories() -> None:
    capacity_only = _request_outcome(
        attempts=(
            _attempt(1, AssignmentRejectionReason.SECTION_FULL),
            _attempt(2, AssignmentRejectionReason.SECTION_FULL),
        )
    )
    no_candidates = _request_outcome(status=PrimaryRequestStatus.UNMET_NO_CANDIDATES)
    not_attempted = _request_outcome(
        AlternateRequestStatus.NOT_NEEDED,
        request_key="alternate:STU_1:1:ALT1",
        request_type="alternate",
        alternate_rank=1,
    )
    assigned_after_rejection = _request_outcome(
        PrimaryRequestStatus.ASSIGNED,
        attempts=(
            _attempt(1, AssignmentRejectionReason.SECTION_FULL),
            _attempt(2, success=True),
        ),
        assignment_key="ASSIGN_2",
    )

    assert _request_failure_diagnostics(capacity_only)["terminal_unmet_reason"] == "all_section_at_capacity"
    assert _request_failure_diagnostics(no_candidates)["terminal_unmet_reason"] == "no_candidate_sections"
    assert _request_failure_diagnostics(not_attempted)["terminal_unmet_reason"] == "not_attempted"
    assigned_diagnostics = _request_failure_diagnostics(assigned_after_rejection)
    assert assigned_diagnostics["terminal_unmet_reason"] == ""
    assert assigned_diagnostics["candidate_attempts_count"] == 2
    assert assigned_diagnostics["candidate_rejections_count"] == 1


def test_unmet_attempts_equal_rejections_when_every_candidate_fails() -> None:
    diagnostics = _request_failure_diagnostics(
        _request_outcome(
            attempts=(
                _attempt(1, AssignmentRejectionReason.SECTION_FULL),
                _attempt(2, AssignmentRejectionReason.SECTION_FULL),
            )
        )
    )

    assert diagnostics["candidate_attempts_count"] == 2
    assert diagnostics["candidate_rejections_count"] == 2


def test_mixed_rejections_requires_at_least_two_nonzero_reason_counts() -> None:
    diagnostics = _request_failure_diagnostics(
        _request_outcome(
            attempts=(
                _attempt(1, AssignmentRejectionReason.SECTION_FULL),
                _attempt(2, AssignmentRejectionReason.PERIOD_CONFLICT),
            )
        )
    )
    nonzero_reason_counts = [
        diagnostics[field]
        for field in (
            "rejected_section_at_capacity_count",
            "rejected_period_conflict_count",
            "rejected_duplicate_logical_course_count",
            "rejected_student_load_limit_count",
            "rejected_other_count",
        )
        if diagnostics[field] > 0
    ]

    assert diagnostics["terminal_unmet_reason"] == "mixed_rejections"
    assert len(nonzero_reason_counts) == 2


def test_not_attempted_alternate_keeps_capacity_and_conflict_counts_zero() -> None:
    diagnostics = _request_failure_diagnostics(
        _request_outcome(
            AlternateRequestStatus.NOT_NEEDED,
            request_key="alternate:STU_1:1:ALT1",
            request_type="alternate",
            alternate_rank=1,
        )
    )

    assert diagnostics["terminal_unmet_reason"] == "not_attempted"
    assert diagnostics["candidate_attempts_count"] == 0
    assert diagnostics["candidate_rejections_count"] == 0
    assert diagnostics["rejected_section_at_capacity_count"] == 0
    assert diagnostics["rejected_period_conflict_count"] == 0


def test_request_outcome_rows_include_unactivated_alternate_with_blank_assigned_terminal() -> None:
    assigned = _request_outcome(
        PrimaryRequestStatus.ASSIGNED,
        attempts=(
            _attempt(1, AssignmentRejectionReason.SECTION_FULL),
            _attempt(2, success=True),
        ),
        assignment_key="ASSIGN_2",
    )
    not_attempted = _request_outcome(
        AlternateRequestStatus.NOT_NEEDED,
        request_key="alternate:STU_1:1:ALT1",
        request_type="alternate",
        alternate_rank=1,
    )
    rows = _request_outcome_rows((SimpleNamespace(algorithm_name="alg", request_outcomes=(assigned, not_attempted)),))
    row_by_key = {row["request_key"]: row for row in rows}

    assert row_by_key["primary:STU_1:CORE_A"]["terminal_unmet_reason"] == ""
    assert row_by_key["primary:STU_1:CORE_A"]["candidate_attempts_count"] == 2
    assert row_by_key["primary:STU_1:CORE_A"]["candidate_rejections_count"] == 1
    assert row_by_key["alternate:STU_1:1:ALT1"]["terminal_unmet_reason"] == "not_attempted"
    assert row_by_key["alternate:STU_1:1:ALT1"]["rejected_section_at_capacity_count"] == 0
    assert row_by_key["alternate:STU_1:1:ALT1"]["rejected_period_conflict_count"] == 0
    for field in (
        "candidate_rejections_count",
        "rejected_section_at_capacity_count",
        "rejected_period_conflict_count",
        "rejected_duplicate_logical_course_count",
        "rejected_student_load_limit_count",
        "rejected_other_count",
    ):
        assert isinstance(row_by_key["primary:STU_1:CORE_A"][field], int)


def test_original_request_and_student_outcome_column_prefixes_are_preserved() -> None:
    assert REQUEST_OUTCOMES_FIELDNAMES[: len(ORIGINAL_REQUEST_OUTCOME_COLUMNS)] == ORIGINAL_REQUEST_OUTCOME_COLUMNS
    assert STUDENT_OUTCOMES_FIELDNAMES[: len(ORIGINAL_STUDENT_OUTCOME_COLUMNS)] == ORIGINAL_STUDENT_OUTCOME_COLUMNS


def test_assignment_failure_summary_uses_execution_request_and_terminal_order() -> None:
    first_result = SimpleNamespace(
        algorithm_name="z_first_by_execution",
        request_outcomes=(
            _request_outcome(PrimaryRequestStatus.UNMET_NO_CANDIDATES, request_key="primary:STU_1:NOSEC"),
            _request_outcome(PrimaryRequestStatus.UNMET_NO_CANDIDATES, request_key="primary:STU_1:NOSEC_2"),
            _request_outcome(
                request_key="alternate:STU_1:1:ALT1",
                request_type="alternate",
                alternate_rank=1,
                attempts=(_attempt(1, AssignmentRejectionReason.SECTION_FULL),),
            ),
        ),
    )
    second_result = SimpleNamespace(
        algorithm_name="a_second_by_execution",
        request_outcomes=(
            _request_outcome(attempts=(_attempt(1, AssignmentRejectionReason.PERIOD_CONFLICT),)),
        ),
    )

    rows = _assignment_failure_summary_rows((first_result, second_result))

    assert [row["algorithm_name"] for row in rows] == [
        "z_first_by_execution",
        "z_first_by_execution",
        "a_second_by_execution",
    ]
    assert rows[0]["request_kind"] == "primary"
    assert rows[0]["terminal_unmet_reason"] == "no_candidate_sections"
    assert rows[0]["unmet_request_count"] == 2
    assert rows[0]["affected_student_count"] == 1
    assert rows[1]["request_kind"] == "alternate"
    assert rows[1]["terminal_unmet_reason"] == "all_section_at_capacity"


def test_assignment_failure_summary_is_deterministic_and_does_not_leak_between_algorithms() -> None:
    first = SimpleNamespace(
        algorithm_name="first_algorithm",
        request_outcomes=(
            _request_outcome(attempts=(_attempt(1, AssignmentRejectionReason.SECTION_FULL),)),
        ),
    )
    second = SimpleNamespace(
        algorithm_name="second_algorithm",
        request_outcomes=(
            _request_outcome(attempts=(_attempt(1, AssignmentRejectionReason.PERIOD_CONFLICT),)),
        ),
    )

    first_rows = _assignment_failure_summary_rows((first, second))
    second_rows = _assignment_failure_summary_rows((first, second))

    assert first_rows == second_rows
    assert first_rows[0]["terminal_unmet_reason"] == "all_section_at_capacity"
    assert first_rows[1]["terminal_unmet_reason"] == "all_period_conflict"
    assert first_rows[0]["total_candidate_rejections"] == 1
    assert first_rows[1]["total_candidate_rejections"] == 1


def test_alternate_filling_primary_gap_removes_student_from_gap_artifact() -> None:
    result = SimpleNamespace(
        algorithm_name="seeded_random_greedy",
        student_outcomes=(_student_outcome(remaining=0, primary_unmet=1),),
        request_outcomes=(
            _request_outcome(PrimaryRequestStatus.UNMET_NO_CANDIDATES),
            _request_outcome(
                AlternateRequestStatus.ASSIGNED,
                request_key="alternate:STU_1:1:ALT1",
                request_type="alternate",
                alternate_rank=1,
                attempts=(_attempt(1, success=True),),
                assignment_key="ASSIGN_ALT",
            ),
        ),
    )

    assert _student_schedule_gap_rows((result,)) == ()


def test_primary_and_alternate_failure_leaves_positive_gap_with_json_lists() -> None:
    gap_student = _student_outcome(remaining=1)
    full_student = _student_outcome(remaining=0, primary_unmet=1)
    full_student = StudentOutcome(**{**full_student.__dict__, "student_id": "STU_2"})
    result = SimpleNamespace(
        algorithm_name="seeded_random_greedy",
        student_outcomes=(gap_student, full_student),
        request_outcomes=(
            _request_outcome(PrimaryRequestStatus.UNMET_NO_CANDIDATES),
            _request_outcome(
                AlternateRequestStatus.NOT_NEEDED,
                request_key="alternate:STU_1:1:ALT1",
                request_type="alternate",
                alternate_rank=1,
            ),
            _request_outcome(
                PrimaryRequestStatus.UNMET_NO_CANDIDATES,
                request_key="primary:STU_2:NOSEC",
                student_id="STU_2",
            ),
        ),
    )

    rows = _student_schedule_gap_rows((result,))

    assert len(rows) == 1
    assert rows[0]["student_id"] == "STU_1"
    assert rows[0]["schedule_gap_count"] == 1
    assert rows[0]["unmet_primary_request_ids"] == '["primary:STU_1:CORE_A"]'
    assert rows[0]["unmet_alternate_request_ids"] == '["alternate:STU_1:1:ALT1"]'
    assert rows[0]["terminal_unmet_reasons"] == '["no_candidate_sections","not_attempted"]'


def test_student_outcome_rows_add_gap_fields_without_changing_existing_values() -> None:
    result = SimpleNamespace(algorithm_name="alg", student_outcomes=(_student_outcome(remaining=1),))

    row = _student_outcome_rows((result,))[0]

    assert row["target_period_units"] == 2
    assert row["assigned_period_units"] == 1
    assert row["remaining_period_units"] == 1
    assert row["target_course_count"] == 2
    assert row["assigned_course_count"] == 1
    assert row["schedule_gap_count"] == 1
    assert row["assigned_alternate_count"] == row["alternate_assigned_count"]


def test_exception_while_reading_candidate_attempt_is_not_swallowed_as_other_rejection() -> None:
    class BrokenAttempt:
        success = False

        @property
        def rejection_reasons(self):
            raise RuntimeError("candidate diagnostic exploded")

    outcome = _request_outcome(attempts=(BrokenAttempt(),))  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="candidate diagnostic exploded"):
        _request_failure_diagnostics(outcome)
