from __future__ import annotations

import pandas as pd

from src.allocation import (
    AssignmentRejectionReason,
    MandatoryFallbackStatus,
    MathFallbackRule,
    PrimaryRequestStatus,
    canonicalize_allocation_input,
    run_grade_priority_baseline,
)


COURSE_ROWS = [
    ("CORE_A", 1, "standard"),
    ("CORE_B", 1, "standard"),
    ("CORE_C", 1, "standard"),
    ("POPULAR_ELECTIVE", 1, "standard"),
    ("MATH2", 1, "standard"),
    ("MATH2_3_HA", 2, "double_period"),
]


def catalog() -> pd.DataFrame:
    return pd.DataFrame(COURSE_ROWS, columns=["course_id", "periods_required", "schedule_structure"])


def students(rows: list[tuple[str, int, int, bool]] | None = None) -> pd.DataFrame:
    rows = rows or [("STU_1", 12, 7, False)]
    return pd.DataFrame(
        [
            (
                student_id,
                grade,
                target,
                "none",
                str(protected).lower(),
                "prior_year_unmet_primary" if protected else "",
                "2026-2027" if protected else "",
            )
            for student_id, grade, target, protected in rows
        ],
        columns=[
            "student_id",
            "grade",
            "target_course_count",
            "unscheduled_preference",
            "priority_protected",
            "priority_reason",
            "priority_valid_school_year",
        ],
    )


def request_row(
    student_id: str,
    course_id: str,
    request_type: str = "primary",
    rank: int | str = "",
    group: str = "",
    block_id: str = "",
) -> tuple:
    return (student_id, course_id, request_type, rank, group, block_id)


def requests(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=[
            "student_id",
            "course_id",
            "request_type",
            "request_rank",
            "request_group",
            "must_share_block_id",
        ],
    )


def section_row(
    section_id: str,
    course_id: str,
    period_1: str,
    period_2: str = "",
    semester: str = "full_year",
    capacity: int = 40,
    block_id: str = "",
    group_id: str = "",
    logical_block_id: str = "",
    semester_content: str = "",
) -> tuple:
    return (
        section_id,
        course_id,
        period_1,
        period_2,
        semester,
        capacity,
        block_id,
        group_id or section_id,
        logical_block_id or course_id,
        semester_content,
    )


def sections(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=[
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
        ],
    )


def base_section_rows() -> list[tuple]:
    return [
        section_row("SEC_CORE_A_1", "CORE_A", "P1", capacity=1, group_id="CORE_A_1"),
        section_row("SEC_CORE_B_1", "CORE_B", "P2", capacity=1, group_id="CORE_B_1"),
        section_row("SEC_CORE_C_1", "CORE_C", "P3", capacity=1, group_id="CORE_C_1"),
        section_row("SEC_POPULAR_1", "POPULAR_ELECTIVE", "P2", capacity=1, group_id="POPULAR_1"),
        section_row("SEC_MATH2_1", "MATH2", "P3", capacity=1, group_id="MATH2_1"),
        section_row("SEC_MATH_1", "MATH2_3_HA", "P5", "P6", capacity=1, group_id="MATH_1"),
    ]


def canonical(
    student_rows: list[tuple[str, int, int, bool]] | None = None,
    request_rows: list[tuple] | None = None,
    section_rows: list[tuple] | None = None,
):
    return canonicalize_allocation_input(
        students(student_rows),
        requests(request_rows or [request_row("STU_1", "CORE_A")]),
        sections(section_rows or base_section_rows()),
        catalog(),
    )


def outcome(result, request_key: str):
    return next(item for item in result.request_outcomes if item.request_key == request_key)


def key(student_id: str, course_id: str) -> str:
    return f"primary:{student_id}:{course_id}"


def run(input_data, seed: int = 20260630):
    return run_grade_priority_baseline(input_data, seed)


def test_grade_priority_processes_seniors_before_juniors_before_sophomores_before_freshmen() -> None:
    data = canonical(
        [
            ("STU_9", 9, 1, False),
            ("STU_10", 10, 1, False),
            ("STU_11", 11, 1, False),
            ("STU_12", 12, 1, False),
        ],
        [
            request_row("STU_9", "CORE_A"),
            request_row("STU_10", "CORE_A"),
            request_row("STU_11", "CORE_A"),
            request_row("STU_12", "CORE_A"),
        ],
    )

    result = run(data)

    assert result.student_processing_order == ("STU_12", "STU_11", "STU_10", "STU_9")
    assert outcome(result, key("STU_12", "CORE_A")).status == PrimaryRequestStatus.ASSIGNED
    for student_id in ("STU_11", "STU_10", "STU_9"):
        assert outcome(result, key(student_id, "CORE_A")).status == PrimaryRequestStatus.UNMET_ALL_CANDIDATES_REJECTED


def test_grade_priority_uses_deterministic_student_id_order_within_grade() -> None:
    data = canonical(
        [("STU_B", 12, 1, False), ("STU_A", 12, 1, False)],
        [request_row("STU_B", "CORE_A"), request_row("STU_A", "CORE_A")],
    )

    result = run(data)

    assert result.student_processing_order == ("STU_A", "STU_B")
    assert outcome(result, key("STU_A", "CORE_A")).status == PrimaryRequestStatus.ASSIGNED
    assert outcome(result, key("STU_B", "CORE_A")).status == PrimaryRequestStatus.UNMET_ALL_CANDIDATES_REJECTED


def test_grade_priority_is_deterministic_across_different_seeds() -> None:
    data = canonical(
        [("STU_9", 9, 1, False), ("STU_12", 12, 1, False)],
        [request_row("STU_9", "CORE_A"), request_row("STU_12", "CORE_A")],
    )

    first = run(data, seed=1)
    second = run(data, seed=999999)

    assert first.student_processing_order == second.student_processing_order
    assert [item.status for item in first.request_outcomes] == [item.status for item in second.request_outcomes]
    assert first.assignments == second.assignments


def test_grade_priority_respects_capacity_through_allocation_state() -> None:
    data = canonical(
        [("STU_12A", 12, 1, False), ("STU_12B", 12, 1, False)],
        [request_row("STU_12A", "CORE_A"), request_row("STU_12B", "CORE_A")],
    )

    result = run(data)

    winner = outcome(result, key("STU_12A", "CORE_A"))
    loser = outcome(result, key("STU_12B", "CORE_A"))
    assert winner.status == PrimaryRequestStatus.ASSIGNED
    assert loser.status == PrimaryRequestStatus.UNMET_ALL_CANDIDATES_REJECTED
    assert loser.candidate_attempts[0].rejection_reasons == (AssignmentRejectionReason.SECTION_FULL,)


def test_grade_priority_respects_period_conflicts_through_allocation_state() -> None:
    # CORE_B and POPULAR_ELECTIVE both occupy P2 with only one candidate each.
    # candidate_key sort order assigns CORE_B first, then POPULAR_ELECTIVE
    # must be rejected for a period conflict.
    data = canonical(
        [("STU_1", 12, 7, False)],
        [request_row("STU_1", "CORE_B"), request_row("STU_1", "POPULAR_ELECTIVE")],
    )

    result = run(data)

    core_b = outcome(result, key("STU_1", "CORE_B"))
    popular = outcome(result, key("STU_1", "POPULAR_ELECTIVE"))
    assert core_b.status == PrimaryRequestStatus.ASSIGNED
    assert popular.status == PrimaryRequestStatus.UNMET_ALL_CANDIDATES_REJECTED
    assert popular.candidate_attempts[0].rejection_reasons == (AssignmentRejectionReason.PERIOD_CONFLICT,)


def test_grade_priority_math2_3_ha_unmet_falls_back_to_math2() -> None:
    data = canonical(
        [("STU_1", 12, 7, False)],
        [request_row("STU_1", "MATH2_3_HA")],
        section_rows=[row for row in base_section_rows() if row[1] != "MATH2_3_HA"],
    )
    result = run_grade_priority_baseline(
        data,
        seed=1,
        math_fallback_rules=(MathFallbackRule("MATH2_3_HA", "MATH2", "mandatory_fallback", True, "test"),),
        math_course_ids=("MATH2", "MATH2_3_HA"),
    )

    source = outcome(result, key("STU_1", "MATH2_3_HA"))
    fallback = next(
        item for item in result.mandatory_fallback_outcomes if item.source_request_key == key("STU_1", "MATH2_3_HA")
    )
    assert source.status == PrimaryRequestStatus.UNMET_NO_CANDIDATES
    assert fallback.status == MandatoryFallbackStatus.ASSIGNED
    assert fallback.fallback_course_id == "MATH2"


def test_grade_priority_does_not_modify_canonical_input() -> None:
    data = canonical(request_rows=[request_row("STU_1", "CORE_A"), request_row("STU_1", "CORE_C")])
    before_requests = data.logical_requests
    before_sections = data.logical_sections

    run(data)

    assert data.logical_requests == before_requests
    assert data.logical_sections == before_sections
