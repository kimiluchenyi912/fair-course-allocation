from __future__ import annotations

import random

import pandas as pd

from src.allocation import (
    AlternateRequestStatus,
    AllocationInputError,
    AssignmentRejectionReason,
    MandatoryFallbackStatus,
    MathCoverageStatus,
    MathFallbackRule,
    PrimaryRequestStatus,
    canonicalize_allocation_input,
    evaluate_math_policy,
    run_seeded_random_baseline,
)


COURSE_ROWS = [
    ("CORE_A", 1, "standard"),
    ("CORE_B", 1, "standard"),
    ("CORE_C", 1, "standard"),
    ("CORE_D", 1, "standard"),
    ("ALT1", 1, "standard"),
    ("ALT2", 1, "standard"),
    ("ALT3", 1, "standard"),
    ("ALT2UNIT", 2, "double_period"),
    ("MATH2", 1, "standard"),
    ("MATH2_3_HA", 2, "double_period"),
    ("MATH_NOSEC", 1, "standard"),
    ("GOV_ECON_REG", 1, "semester_block"),
    ("AP_PHYSC", 1, "semester_block"),
    ("NOSEC", 1, "standard"),
    ("HIGH", 1, "standard"),
    ("POPULAR_ELECTIVE", 1, "standard"),
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


def gov_request_rows(student_id: str) -> list[tuple]:
    return [
        request_row(student_id, "GOV_ECON_REG", "primary", "", "gov_econ_block", "GOV_ECON_REG"),
        request_row(student_id, "GOV_ECON_REG", "primary", "", "gov_econ_block", "GOV_ECON_REG"),
    ]


def base_section_rows() -> list[tuple]:
    return [
        section_row("SEC_CORE_A_1", "CORE_A", "P1", capacity=1, group_id="CORE_A_1"),
        section_row("SEC_CORE_A_2", "CORE_A", "P2", capacity=1, group_id="CORE_A_2"),
        section_row("SEC_CORE_B_1", "CORE_B", "P2", capacity=1, group_id="CORE_B_1"),
        section_row("SEC_CORE_C_1", "CORE_C", "P3", capacity=1, group_id="CORE_C_1"),
        section_row("SEC_CORE_D_1", "CORE_D", "P4", capacity=1, group_id="CORE_D_1"),
        section_row("SEC_ALT1_1", "ALT1", "P5", capacity=1, group_id="ALT1_1"),
        section_row("SEC_ALT2_1", "ALT2", "P6", capacity=1, group_id="ALT2_1"),
        section_row("SEC_ALT3_1", "ALT3", "P7", capacity=1, group_id="ALT3_1"),
        section_row("SEC_ALT2UNIT_1", "ALT2UNIT", "P6", "P7", capacity=1, group_id="ALT2UNIT_1"),
        section_row("SEC_MATH2_1", "MATH2", "P3", capacity=1, group_id="MATH2_1"),
        section_row("SEC_MATH_1", "MATH2_3_HA", "P5", "P6", capacity=1, group_id="MATH_1"),
        section_row("SEC_GOV_1_S1", "GOV_ECON_REG", "P7", semester="semester_1", capacity=1, block_id="GOV_1", group_id="GOV_1", logical_block_id="GOV_ECON_REG", semester_content="Government"),
        section_row("SEC_GOV_1_S2", "GOV_ECON_REG", "P7", semester="semester_2", capacity=1, block_id="GOV_1", group_id="GOV_1", logical_block_id="GOV_ECON_REG", semester_content="Economics"),
        section_row("SEC_AP_PHYSC_1", "AP_PHYSC", "P4", semester="paired", capacity=1, block_id="AP_PHYSC_1", group_id="AP_PHYSC_1", logical_block_id="AP_PHYSC", semester_content="Mechanics / Electricity and Magnetism"),
        section_row("SEC_HIGH_1", "HIGH", "P1", capacity=200, group_id="HIGH_1"),
        section_row("SEC_POPULAR_1", "POPULAR_ELECTIVE", "P2", capacity=200, group_id="POPULAR_1"),
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


def student_outcome(result, student_id: str):
    return next(item for item in result.student_outcomes if item.student_id == student_id)


def key(student_id: str, course_id: str) -> str:
    return f"primary:{student_id}:{course_id}"


def alt_key(student_id: str, rank: int, course_id: str) -> str:
    return f"alternate:{student_id}:{rank}:{course_id}"


def run(input_data, seed: int = 20260630):
    return run_seeded_random_baseline(input_data, seed)


def fallback_rules() -> tuple[MathFallbackRule, ...]:
    return (MathFallbackRule("MATH2_3_HA", "MATH2", "mandatory_fallback", True, "test"),)


def math_ids() -> tuple[str, ...]:
    return ("MATH2", "MATH2_3_HA", "MATH_NOSEC")


def run_with_fallback(input_data, seed: int = 20260630):
    return run_seeded_random_baseline(
        input_data,
        seed,
        math_fallback_rules=fallback_rules(),
        math_course_ids=math_ids(),
    )


def fallback_outcome(result, source_request_key: str):
    return next(item for item in result.mandatory_fallback_outcomes if item.source_request_key == source_request_key)


def without_courses(section_rows: list[tuple], *course_ids: str) -> list[tuple]:
    return [row for row in section_rows if row[1] not in set(course_ids)]


def test_single_student_all_primary_requests_are_assigned() -> None:
    data = canonical(request_rows=[request_row("STU_1", "CORE_A"), request_row("STU_1", "CORE_C")])
    result = run(data)

    assert [item.status for item in result.request_outcomes] == [PrimaryRequestStatus.ASSIGNED, PrimaryRequestStatus.ASSIGNED]
    assert student_outcome(result, "STU_1").primary_unmet_count == 0


def test_two_students_compete_for_one_capacity_one_section() -> None:
    data = canonical(
        [("STU_1", 12, 1, False), ("STU_2", 12, 1, False)],
        [request_row("STU_1", "CORE_B"), request_row("STU_2", "CORE_B")],
    )
    result = run(data, seed=1)

    assigned = [item for item in result.request_outcomes if item.status == PrimaryRequestStatus.ASSIGNED]
    unmet = [item for item in result.request_outcomes if item.status != PrimaryRequestStatus.ASSIGNED]
    assert len(assigned) == 1
    assert len(unmet) == 1
    assert unmet[0].candidate_attempts[0].rejection_reasons == (AssignmentRejectionReason.SECTION_FULL,)


def test_first_candidate_failure_then_second_candidate_success_is_recorded() -> None:
    data = canonical(
        [("STU_1", 12, 3, False)],
        [request_row("STU_1", "CORE_B"), request_row("STU_1", "CORE_A")],
    )

    for seed in range(200):
        result = run(data, seed=seed)
        target = outcome(result, key("STU_1", "CORE_A"))
        if target.status == PrimaryRequestStatus.ASSIGNED and len(target.candidate_attempts) == 2:
            assert target.candidate_attempts[0].success is False
            assert AssignmentRejectionReason.PERIOD_CONFLICT in target.candidate_attempts[0].rejection_reasons
            assert target.candidate_attempts[1].success is True
            break
    else:
        raise AssertionError("No tested seed produced the intended candidate-attempt order.")


def test_normal_math_gov_econ_and_ap_physics_c_can_all_be_assigned() -> None:
    data = canonical(
        [("STU_1", 12, 5, False)],
        [
            request_row("STU_1", "CORE_A"),
            request_row("STU_1", "MATH2_3_HA"),
            *gov_request_rows("STU_1"),
            request_row("STU_1", "AP_PHYSC"),
        ],
    )
    result = run(data)

    assert all(item.status == PrimaryRequestStatus.ASSIGNED for item in result.request_outcomes)
    assert {assignment.structure_type for assignment in result.assignments} == {"normal", "double_period", "linked_semester", "paired_content"}


def test_math2_3_counts_as_one_primary_assigned_and_two_period_units() -> None:
    data = canonical([("STU_1", 12, 2, False)], [request_row("STU_1", "MATH2_3_HA")])
    result = run(data)
    student = student_outcome(result, "STU_1")

    assert student.primary_assigned_count == 1
    assert student.assigned_period_units == 2
    assert result.assignments[0].period_units == 2


def test_gov_econ_creates_one_assignment_and_one_roster_seat() -> None:
    data = canonical([("STU_1", 12, 1, False)], gov_request_rows("STU_1"))
    result = run(data)

    assert len(result.assignments) == 1
    assert result.assignments[0].member_section_ids == ("SEC_GOV_1_S1", "SEC_GOV_1_S2")
    assert next(row for row in result.section_roster_summary if row.linked_section_group_id == "GOV_1").assigned_count == 1


def test_primary_and_alternate_use_same_assignment_engine() -> None:
    data = canonical(
        [("STU_1", 12, 2, False)],
        [request_row("STU_1", "CORE_A"), request_row("STU_1", "ALT1", "alternate", 1, "alternate")],
    )
    result = run(data)

    assert outcome(result, alt_key("STU_1", 1, "ALT1")).status == AlternateRequestStatus.ASSIGNED
    assert any(assignment.request_type == "alternate" for assignment in result.assignments)


def test_same_input_and_seed_produce_identical_baseline_result() -> None:
    data = canonical(
        [("STU_1", 12, 2, False), ("STU_2", 12, 2, False)],
        [request_row("STU_1", "CORE_A"), request_row("STU_2", "CORE_A")],
    )

    assert run(data, seed=22) == run(data, seed=22)


def test_different_seed_changes_student_order_or_assignments_on_competition_fixture() -> None:
    data = canonical(
        [("STU_1", 12, 1, False), ("STU_2", 12, 1, False), ("STU_3", 12, 1, False)],
        [request_row("STU_1", "CORE_B"), request_row("STU_2", "CORE_B"), request_row("STU_3", "CORE_B")],
    )
    first = run(data, seed=1)
    second = run(data, seed=5)

    assert first.student_processing_order != second.student_processing_order or first.assignments != second.assignments


def test_baseline_does_not_modify_canonical_input() -> None:
    data = canonical()
    before = data

    run(data)

    assert data == before


def test_baseline_uses_local_rng_and_does_not_pollute_global_random_state() -> None:
    random.seed(99)
    before = random.getstate()

    run(canonical(), seed=3)

    assert random.getstate() == before


def test_equivalent_raw_row_order_with_same_seed_produces_identical_result() -> None:
    student_df = students([("STU_1", 12, 3, False), ("STU_2", 12, 3, False)])
    request_df = requests(
        [
            request_row("STU_1", "CORE_A"),
            request_row("STU_1", "ALT1", "alternate", 1, "alternate"),
            request_row("STU_2", "CORE_C"),
            request_row("STU_2", "ALT2", "alternate", 1, "alternate"),
        ]
    )
    section_df = sections(base_section_rows())
    first = canonicalize_allocation_input(student_df, request_df, section_df, catalog())
    second = canonicalize_allocation_input(
        student_df.sample(frac=1, random_state=1).reset_index(drop=True),
        request_df.sample(frac=1, random_state=2).reset_index(drop=True),
        section_df.sample(frac=1, random_state=3).reset_index(drop=True),
        catalog().sample(frac=1, random_state=4).reset_index(drop=True),
    )

    assert run(first, seed=11) == run(second, seed=11)


def test_assignments_outcomes_and_student_outcomes_are_stably_sorted() -> None:
    data = canonical(
        [("STU_2", 12, 2, False), ("STU_1", 12, 2, False)],
        [request_row("STU_2", "CORE_A"), request_row("STU_1", "CORE_C"), request_row("STU_1", "ALT1", "alternate", 1, "alternate")],
    )
    result = run(data)

    assert [item.assignment_key for item in result.assignments] == sorted(item.assignment_key for item in result.assignments)
    assert [(item.student_id, 0 if item.request_type == "primary" else 1, item.alternate_rank or 0, item.candidate_key) for item in result.request_outcomes] == sorted(
        (item.student_id, 0 if item.request_type == "primary" else 1, item.alternate_rank or 0, item.candidate_key) for item in result.request_outcomes
    )
    assert [item.student_id for item in result.student_outcomes] == sorted(item.student_id for item in result.student_outcomes)


def test_primary_no_candidates_outcome_is_recorded() -> None:
    data = canonical([("STU_1", 12, 1, False)], [request_row("STU_1", "NOSEC")])
    result = run(data)

    assert outcome(result, key("STU_1", "NOSEC")).status == PrimaryRequestStatus.UNMET_NO_CANDIDATES
    assert outcome(result, key("STU_1", "NOSEC")).candidate_attempts == ()


def test_primary_all_candidates_rejected_with_full_diagnostics() -> None:
    data = canonical(
        [("STU_1", 12, 1, False), ("STU_2", 12, 1, False)],
        [request_row("STU_1", "CORE_B"), request_row("STU_2", "CORE_B")],
    )
    result = run(data, seed=1)
    unmet = next(item for item in result.request_outcomes if item.status != PrimaryRequestStatus.ASSIGNED)

    assert unmet.status == PrimaryRequestStatus.UNMET_ALL_CANDIDATES_REJECTED
    assert unmet.candidate_attempts[0].rejection_reasons == (AssignmentRejectionReason.SECTION_FULL,)


def test_primary_period_conflict_diagnostics_are_preserved() -> None:
    same_period_rows = [
        row if row[1] != "CORE_C" else section_row("SEC_CORE_C_1", "CORE_C", "P2", capacity=1, group_id="CORE_C_1")
        for row in base_section_rows()
    ]
    data = canonical(
        [("STU_1", 12, 3, False)],
        [request_row("STU_1", "CORE_B"), request_row("STU_1", "CORE_C")],
        same_period_rows,
    )
    result = run(data)
    rejected_attempts = [
        attempt
        for item in result.request_outcomes
        for attempt in item.candidate_attempts
        if AssignmentRejectionReason.PERIOD_CONFLICT in attempt.rejection_reasons
    ]

    assert rejected_attempts


def test_primary_target_load_diagnostics_are_preserved() -> None:
    data = canonical([("STU_1", 12, 1, False)], [request_row("STU_1", "CORE_A"), request_row("STU_1", "CORE_C")])
    result = run(data)

    assert any(
        AssignmentRejectionReason.TARGET_LOAD_EXCEEDED in attempt.rejection_reasons
        for item in result.request_outcomes
        for attempt in item.candidate_attempts
    )


def test_math2_3_unmet_counts_one_request_and_two_period_units() -> None:
    data = canonical([("STU_1", 12, 1, False)], [request_row("STU_1", "CORE_A"), request_row("STU_1", "MATH2_3_HA")])
    result = run(data)
    student = student_outcome(result, "STU_1")

    assert student.primary_unmet_count == 1
    assert student.primary_unmet_period_units == 2


def test_alternate_assignment_does_not_change_primary_unmet_outcome() -> None:
    data = canonical(
        [("STU_1", 12, 2, False)],
        [request_row("STU_1", "NOSEC"), request_row("STU_1", "ALT1", "alternate", 1, "alternate")],
    )
    result = run(data)

    assert outcome(result, key("STU_1", "NOSEC")).status == PrimaryRequestStatus.UNMET_NO_CANDIDATES
    assert outcome(result, alt_key("STU_1", 1, "ALT1")).status == AlternateRequestStatus.ASSIGNED
    assert student_outcome(result, "STU_1").primary_unmet_count == 1


def test_every_primary_has_exactly_one_outcome_and_success_attempt_is_last() -> None:
    data = canonical([("STU_1", 12, 3, False)], [request_row("STU_1", "CORE_A"), request_row("STU_1", "CORE_C")])
    result = run(data)
    primary = [item for item in result.request_outcomes if item.request_type == "primary"]

    assert len(primary) == 2
    assert {item.request_key for item in primary} == {key("STU_1", "CORE_A"), key("STU_1", "CORE_C")}
    for item in primary:
        if item.status == PrimaryRequestStatus.ASSIGNED:
            assert item.candidate_attempts[-1].success is True


def test_alternate_phase_runs_after_all_students_primary_phase() -> None:
    data = canonical(
        [("STU_1", 12, 2, False), ("STU_2", 12, 1, False)],
        [
            request_row("STU_1", "NOSEC"),
            request_row("STU_1", "CORE_B", "alternate", 1, "alternate"),
            request_row("STU_2", "CORE_B"),
        ],
    )
    result = run(data, seed=0)

    assert outcome(result, key("STU_2", "CORE_B")).status == PrimaryRequestStatus.ASSIGNED
    assert outcome(result, alt_key("STU_1", 1, "CORE_B")).status == AlternateRequestStatus.UNASSIGNED_ALL_CANDIDATES_REJECTED


def test_mandatory_math_fallback_phase_runs_after_all_primary_requests() -> None:
    data = canonical(
        [("STU_1", 12, 1, False)],
        [request_row("STU_1", "MATH2_3_HA")],
        without_courses(base_section_rows(), "MATH2_3_HA"),
    )

    result = run_with_fallback(data)

    assert outcome(result, key("STU_1", "MATH2_3_HA")).status == PrimaryRequestStatus.UNMET_NO_CANDIDATES
    assert fallback_outcome(result, key("STU_1", "MATH2_3_HA")).status == MandatoryFallbackStatus.ASSIGNED
    assert result.assignments[0].request_type == "mandatory_fallback"


def test_mandatory_math_fallback_phase_runs_before_any_ordinary_alternate() -> None:
    data = canonical(
        [("STU_1", 12, 1, False)],
        [
            request_row("STU_1", "MATH2_3_HA"),
            request_row("STU_1", "ALT1", "alternate", 1, "alternate"),
        ],
        without_courses(base_section_rows(), "MATH2_3_HA"),
    )

    result = run_with_fallback(data)

    assert fallback_outcome(result, key("STU_1", "MATH2_3_HA")).status == MandatoryFallbackStatus.ASSIGNED
    assert outcome(result, alt_key("STU_1", 1, "ALT1")).status == AlternateRequestStatus.NOT_NEEDED


def test_fallback_success_leaves_only_remaining_units_for_alternates() -> None:
    data = canonical(
        [("STU_1", 12, 2, False)],
        [
            request_row("STU_1", "MATH2_3_HA"),
            request_row("STU_1", "ALT1", "alternate", 1, "alternate"),
        ],
        without_courses(base_section_rows(), "MATH2_3_HA"),
    )

    result = run_with_fallback(data)
    student = student_outcome(result, "STU_1")

    assert fallback_outcome(result, key("STU_1", "MATH2_3_HA")).status == MandatoryFallbackStatus.ASSIGNED
    assert outcome(result, alt_key("STU_1", 1, "ALT1")).status == AlternateRequestStatus.ASSIGNED
    assert student.assigned_period_units == 2
    assert student.mandatory_fallback_assigned_period_units == 1
    assert student.alternate_assigned_period_units == 1


def test_fallback_success_does_not_change_source_primary_or_alternate_counts() -> None:
    data = canonical(
        [("STU_1", 12, 2, False)],
        [
            request_row("STU_1", "MATH2_3_HA"),
            request_row("STU_1", "ALT1", "alternate", 1, "alternate"),
        ],
        without_courses(base_section_rows(), "MATH2_3_HA"),
    )

    result = run_with_fallback(data)
    student = student_outcome(result, "STU_1")

    assert student.primary_assigned_count == 0
    assert student.primary_unmet_count == 1
    assert student.alternate_assigned_count == 1
    assert student.mandatory_fallback_assigned_count == 1
    assert outcome(result, key("STU_1", "MATH2_3_HA")).status == PrimaryRequestStatus.UNMET_NO_CANDIDATES


def test_math2_3_source_assigned_makes_fallback_not_required() -> None:
    data = canonical([("STU_1", 12, 2, False)], [request_row("STU_1", "MATH2_3_HA")])
    result = run_with_fallback(data)

    assert fallback_outcome(result, key("STU_1", "MATH2_3_HA")).status == MandatoryFallbackStatus.NOT_REQUIRED_SOURCE_ASSIGNED
    assert student_outcome(result, "STU_1").mandatory_fallback_assigned_count == 0


def test_math2_3_source_unmet_attempts_math2_fallback_and_records_roster_seat() -> None:
    data = canonical(
        [("STU_1", 12, 1, False)],
        [request_row("STU_1", "MATH2_3_HA")],
        without_courses(base_section_rows(), "MATH2_3_HA"),
    )
    result = run_with_fallback(data)
    fallback = fallback_outcome(result, key("STU_1", "MATH2_3_HA"))

    assert fallback.status == MandatoryFallbackStatus.ASSIGNED
    assert fallback.assignment_key is not None
    assert next(row for row in result.section_roster_summary if row.linked_section_group_id == "MATH2_1").assigned_count == 1


def test_fallback_no_candidate_has_clear_outcome() -> None:
    data = canonical(
        [("STU_1", 12, 1, False)],
        [request_row("STU_1", "MATH2_3_HA")],
        without_courses(base_section_rows(), "MATH2_3_HA", "MATH2"),
    )

    result = run_with_fallback(data)
    fallback = fallback_outcome(result, key("STU_1", "MATH2_3_HA"))

    assert fallback.status == MandatoryFallbackStatus.UNASSIGNED_NO_CANDIDATES
    assert fallback.candidate_attempts == ()


def test_fallback_all_candidates_rejected_by_period_conflict_has_attempts() -> None:
    data = canonical(
        [("STU_1", 12, 2, False)],
        [request_row("STU_1", "CORE_C"), request_row("STU_1", "MATH2_3_HA")],
        without_courses(base_section_rows(), "MATH2_3_HA"),
    )

    result = run_with_fallback(data)
    fallback = fallback_outcome(result, key("STU_1", "MATH2_3_HA"))

    assert fallback.status == MandatoryFallbackStatus.UNASSIGNED_ALL_CANDIDATES_REJECTED
    assert fallback.candidate_attempts[0].rejection_reasons == (AssignmentRejectionReason.PERIOD_CONFLICT,)


def test_fallback_all_candidates_rejected_when_section_full_has_attempts() -> None:
    data = canonical(
        [("STU_1", 12, 1, False), ("STU_2", 12, 1, False)],
        [request_row("STU_1", "MATH2_3_HA"), request_row("STU_2", "MATH2_3_HA")],
        without_courses(base_section_rows(), "MATH2_3_HA"),
    )

    result = run_with_fallback(data, seed=1)
    statuses = [item.status for item in result.mandatory_fallback_outcomes]
    rejected = next(item for item in result.mandatory_fallback_outcomes if item.status != MandatoryFallbackStatus.ASSIGNED)

    assert statuses.count(MandatoryFallbackStatus.ASSIGNED) == 1
    assert rejected.candidate_attempts[0].rejection_reasons == (AssignmentRejectionReason.SECTION_FULL,)


def test_fallback_target_load_failure_keeps_state_consistent() -> None:
    data = canonical(
        [("STU_1", 12, 1, False)],
        [request_row("STU_1", "CORE_A"), request_row("STU_1", "MATH2_3_HA")],
        without_courses(base_section_rows(), "MATH2_3_HA"),
    )

    result = run_with_fallback(data)
    fallback = fallback_outcome(result, key("STU_1", "MATH2_3_HA"))

    assert fallback.status == MandatoryFallbackStatus.UNASSIGNED_ALL_CANDIDATES_REJECTED
    assert fallback.candidate_attempts[0].rejection_reasons == (AssignmentRejectionReason.TARGET_LOAD_EXCEEDED,)
    assert result.consistency_issues == ()


def test_fallback_assignment_prevents_duplicate_math2_logical_identity() -> None:
    data = canonical(
        [("STU_1", 12, 2, False)],
        [
            request_row("STU_1", "MATH2_3_HA"),
            request_row("STU_1", "MATH2", "alternate", 1, "alternate"),
        ],
        without_courses(base_section_rows(), "MATH2_3_HA"),
    )

    result = run_with_fallback(data)
    alternate = outcome(result, alt_key("STU_1", 1, "MATH2"))

    assert fallback_outcome(result, key("STU_1", "MATH2_3_HA")).status == MandatoryFallbackStatus.ASSIGNED
    assert alternate.status == AlternateRequestStatus.UNASSIGNED_ALL_CANDIDATES_REJECTED
    assert AssignmentRejectionReason.DUPLICATE_LOGICAL_COURSE_OR_BLOCK in alternate.candidate_attempts[0].rejection_reasons


def test_existing_assigned_math_primary_makes_math2_3_fallback_not_required() -> None:
    data = canonical(
        [("STU_1", 12, 1, False)],
        [request_row("STU_1", "MATH2"), request_row("STU_1", "MATH2_3_HA")],
        without_courses(base_section_rows(), "MATH2_3_HA"),
    )

    result = run_with_fallback(data)

    assert fallback_outcome(result, key("STU_1", "MATH2_3_HA")).status == MandatoryFallbackStatus.NOT_REQUIRED_MATH_COVERAGE_ALREADY_SATISFIED


def test_multiple_math_all_unmet_can_be_satisfied_by_mandatory_fallback() -> None:
    data = canonical(
        [("STU_1", 12, 1, False)],
        [request_row("STU_1", "MATH_NOSEC"), request_row("STU_1", "MATH2_3_HA")],
        without_courses(base_section_rows(), "MATH2_3_HA"),
    )

    result = run_with_fallback(data)
    report = evaluate_math_policy(data, result, math_ids(), fallback_rules())

    assert fallback_outcome(result, key("STU_1", "MATH2_3_HA")).status == MandatoryFallbackStatus.ASSIGNED
    assert next(item for item in report.student_outcomes if item.student_id == "STU_1").coverage_status == MathCoverageStatus.SATISFIED_BY_MANDATORY_FALLBACK


def test_multiple_math_all_unmet_has_violation_when_mandatory_fallback_fails() -> None:
    data = canonical(
        [("STU_1", 12, 1, False)],
        [request_row("STU_1", "MATH_NOSEC"), request_row("STU_1", "MATH2_3_HA")],
        without_courses(base_section_rows(), "MATH2_3_HA", "MATH2"),
    )

    result = run_with_fallback(data)
    report = evaluate_math_policy(data, result, math_ids(), fallback_rules())

    assert fallback_outcome(result, key("STU_1", "MATH2_3_HA")).status == MandatoryFallbackStatus.UNASSIGNED_NO_CANDIDATES
    assert next(item for item in report.student_outcomes if item.student_id == "STU_1").coverage_status == MathCoverageStatus.VIOLATED_MANDATORY_FALLBACK_FAILED


def test_alternates_are_processed_by_rank_and_later_rank_is_not_needed_after_schedule_is_full() -> None:
    data = canonical(
        [("STU_1", 12, 1, False)],
        [
            request_row("STU_1", "NOSEC"),
            request_row("STU_1", "ALT1", "alternate", 1, "alternate"),
            request_row("STU_1", "ALT2", "alternate", 2, "alternate"),
        ],
    )
    result = run(data)

    assert outcome(result, alt_key("STU_1", 1, "ALT1")).status == AlternateRequestStatus.ASSIGNED
    assert outcome(result, alt_key("STU_1", 2, "ALT2")).status == AlternateRequestStatus.NOT_NEEDED


def test_rank_one_alternate_failure_continues_to_rank_two() -> None:
    data = canonical(
        [("STU_1", 12, 1, False)],
        [
            request_row("STU_1", "NOSEC"),
            request_row("STU_1", "NOSEC", "alternate", 1, "alternate"),
            request_row("STU_1", "ALT2", "alternate", 2, "alternate"),
        ],
    )
    result = run(data)

    assert outcome(result, alt_key("STU_1", 1, "NOSEC")).status == AlternateRequestStatus.UNASSIGNED_NO_CANDIDATES
    assert outcome(result, alt_key("STU_1", 2, "ALT2")).status == AlternateRequestStatus.ASSIGNED


def test_two_unit_alternate_does_not_fit_one_remaining_unit() -> None:
    data = canonical(
        [("STU_1", 12, 2, False)],
        [
            request_row("STU_1", "CORE_A"),
            request_row("STU_1", "NOSEC"),
            request_row("STU_1", "ALT2UNIT", "alternate", 1, "alternate"),
        ],
    )
    result = run(data)

    assert outcome(result, alt_key("STU_1", 1, "ALT2UNIT")).status == AlternateRequestStatus.DOES_NOT_FIT_REMAINING_LOAD


def test_alternate_no_candidates_and_all_candidates_rejected_are_distinct() -> None:
    data = canonical(
        [("STU_1", 12, 2, False), ("STU_2", 12, 1, False)],
        [
            request_row("STU_1", "NOSEC"),
            request_row("STU_1", "NOSEC", "alternate", 1, "alternate"),
            request_row("STU_1", "CORE_B", "alternate", 2, "alternate"),
            request_row("STU_2", "CORE_B"),
        ],
    )
    result = run(data, seed=0)

    assert outcome(result, alt_key("STU_1", 1, "NOSEC")).status == AlternateRequestStatus.UNASSIGNED_NO_CANDIDATES
    assert outcome(result, alt_key("STU_1", 2, "CORE_B")).status == AlternateRequestStatus.UNASSIGNED_ALL_CANDIDATES_REJECTED


def test_every_alternate_has_one_outcome_and_rank_is_preserved_in_assignment() -> None:
    data = canonical(
        [("STU_1", 12, 1, False)],
        [
            request_row("STU_1", "NOSEC"),
            request_row("STU_1", "ALT1", "alternate", 1, "alternate"),
            request_row("STU_1", "ALT2", "alternate", 2, "alternate"),
            request_row("STU_1", "ALT3", "alternate", 3, "alternate"),
        ],
    )
    result = run(data)
    alternates = [item for item in result.request_outcomes if item.request_type == "alternate"]

    assert [item.alternate_rank for item in alternates] == [1, 2, 3]
    assert len(alternates) == 3
    assert next(assignment for assignment in result.assignments if assignment.request_type == "alternate").alternate_rank == 1


def test_student_outcome_units_counts_and_full_schedule_are_period_unit_based() -> None:
    data = canonical(
        [("STU_1", 12, 3, False)],
        [request_row("STU_1", "MATH2_3_HA"), request_row("STU_1", "CORE_A")],
    )
    result = run(data)
    student = student_outcome(result, "STU_1")

    assert student.fully_scheduled
    assert student.assigned_period_units == 3
    assert student.remaining_period_units == 0
    assert student.primary_assigned_count == 2
    assert student.primary_unmet_count == 0


def test_student_outcome_alternate_counts_and_units_are_accurate() -> None:
    data = canonical(
        [("STU_1", 12, 2, False)],
        [request_row("STU_1", "NOSEC"), request_row("STU_1", "ALT1", "alternate", 1, "alternate")],
    )
    result = run(data)
    student = student_outcome(result, "STU_1")

    assert student.alternate_assigned_count == 1
    assert student.alternate_assigned_period_units == 1
    assert student.primary_unmet_count == 1
    assert student.fully_scheduled is False


def test_ordinary_student_zero_or_one_primary_unmet_is_not_violation() -> None:
    data = canonical([("STU_1", 12, 2, False)], [request_row("STU_1", "NOSEC")])
    result = run(data)

    assert student_outcome(result, "STU_1").ordinary_fairness_violation is False
    assert result.policy_report.ordinary_policy_satisfied


def test_ordinary_student_more_than_one_primary_unmet_is_violation_but_baseline_returns() -> None:
    data = canonical([("STU_1", 12, 1, False)], [request_row("STU_1", "NOSEC"), request_row("STU_1", "CORE_B"), request_row("STU_1", "CORE_C")])
    result = run(data)
    student = student_outcome(result, "STU_1")

    assert student.primary_unmet_count > 1
    assert student.ordinary_fairness_violation
    assert result.policy_report.ordinary_violation_student_ids == ("STU_1",)


def test_protected_student_zero_unmet_is_not_violation() -> None:
    data = canonical([("STU_P", 12, 1, True)], [request_row("STU_P", "CORE_A")])
    result = run(data)

    assert student_outcome(result, "STU_P").protected_fairness_violation is False
    assert result.policy_report.protected_policy_satisfied


def test_protected_student_one_unmet_is_violation_and_not_ordinary_violation() -> None:
    data = canonical([("STU_P", 12, 1, True)], [request_row("STU_P", "NOSEC")])
    result = run(data)
    student = student_outcome(result, "STU_P")

    assert student.protected_fairness_violation
    assert student.ordinary_fairness_violation is False
    assert result.policy_report.protected_violation_student_ids == ("STU_P",)
    assert result.policy_report.ordinary_violation_student_ids == ()


def high_demand_input(count: int, course_id: str = "HIGH", include_section: bool = True):
    student_rows = [(f"STU_{index:03d}", 12, 1, False) for index in range(1, count + 1)]
    request_rows = [request_row(student_id, course_id) for student_id, *_ in student_rows]
    section_rows = base_section_rows() if include_section else [row for row in base_section_rows() if row[1] != course_id]
    return canonical(student_rows, request_rows, section_rows)


def test_high_demand_policy_demand_120_does_not_trigger() -> None:
    result = run(high_demand_input(120, include_section=False))

    assert result.policy_report.high_demand_candidate_keys == ()
    assert result.policy_report.high_demand_violation_count == 0


def test_high_demand_policy_demand_121_triggers() -> None:
    result = run(high_demand_input(121))

    assert result.policy_report.high_demand_candidate_keys == ("HIGH",)
    assert result.policy_report.high_demand_demands[0].logical_primary_demand == 121


def test_high_demand_primary_assigned_has_no_violation() -> None:
    result = run(high_demand_input(121))

    assert result.policy_report.high_demand_policy_satisfied
    assert result.policy_report.high_demand_violation_count == 0


def test_high_demand_primary_unmet_is_reported_and_alternate_fill_does_not_clear_violation() -> None:
    student_rows = [(f"STU_{index:03d}", 12, 1, False) for index in range(1, 122)]
    request_rows = []
    for student_id, *_ in student_rows:
        request_rows.append(request_row(student_id, "HIGH"))
        request_rows.append(request_row(student_id, "ALT1", "alternate", 1, "alternate"))
    section_rows = [row for row in base_section_rows() if row[1] != "HIGH"]
    data = canonical(student_rows, request_rows, section_rows)
    result = run(data)

    assert result.policy_report.high_demand_candidate_keys == ("HIGH",)
    assert result.policy_report.high_demand_violation_count == 121
    assert result.policy_report.high_demand_violating_student_ids[0] == "STU_001"
    assert outcome(result, alt_key("STU_001", 1, "ALT1")).status in {
        AlternateRequestStatus.ASSIGNED,
        AlternateRequestStatus.UNASSIGNED_ALL_CANDIDATES_REJECTED,
    }


def test_gov_econ_high_demand_counts_logical_requests_not_raw_semester_rows() -> None:
    student_rows = [(f"STU_{index:03d}", 12, 1, False) for index in range(1, 122)]
    request_rows = []
    for student_id, *_ in student_rows:
        request_rows.extend(gov_request_rows(student_id))
    result = run(canonical(student_rows, request_rows))

    demand = next(item for item in result.policy_report.high_demand_demands if item.candidate_key == "GOV_ECON_REG")
    assert demand.logical_primary_demand == 121


def test_popular_elective_demand_above_120_triggers_same_high_demand_policy() -> None:
    result = run(high_demand_input(121, course_id="POPULAR_ELECTIVE"))

    assert result.policy_report.high_demand_candidate_keys == ("POPULAR_ELECTIVE",)


def test_result_models_do_not_contain_global_infeasible_semantics() -> None:
    result = run(canonical([("STU_1", 12, 1, False)], [request_row("STU_1", "NOSEC")]))

    assert "infeasible" not in repr(result).lower()


def test_consistency_issues_empty_sections_not_over_capacity_and_counts_match_inputs() -> None:
    data = canonical(
        [("STU_1", 12, 2, False), ("STU_2", 12, 1, False)],
        [request_row("STU_1", "CORE_A"), request_row("STU_1", "ALT1", "alternate", 1, "alternate"), request_row("STU_2", "CORE_A")],
    )
    result = run(data)

    assert result.consistency_issues == ()
    assert all(row.assigned_count <= row.capacity for row in result.section_roster_summary)
    assert len(result.request_outcomes) == len(data.logical_requests)
    assert len(result.student_outcomes) == len(data.students)
