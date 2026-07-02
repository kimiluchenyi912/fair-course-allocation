from __future__ import annotations

import random

import pandas as pd

from src.allocation import (
    AlternateRequestStatus,
    CpSatSolveStatus,
    CpSatStageName,
    MandatoryFallbackStatus,
    MathFallbackRule,
    PrimaryRequestStatus,
    canonicalize_allocation_input,
    math_course_ids_from_catalog,
    run_constrained_first_baseline,
    run_fair_cp_sat_solver,
    run_seeded_random_baseline,
)


def catalog() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("CORE_A", 1, "standard"),
            ("CORE_B", 1, "standard"),
            ("CORE_C", 1, "standard"),
            ("CORE_D", 1, "standard"),
            ("ALT1", 1, "standard"),
            ("ALT2", 1, "standard"),
            ("ALT3", 1, "standard"),
            ("MATH1", 1, "standard"),
            ("MATH2", 1, "standard"),
            ("MATH2_3_HA", 2, "double_period"),
            ("CALC_D_LINALG", 1, "standard"),
            ("DOUBLE", 2, "double_period"),
            ("GOV_ECON_REG", 1, "semester_block"),
            ("AP_PHYSC", 1, "semester_block"),
            ("HIGH", 1, "standard"),
            ("NOSEC", 1, "standard"),
        ],
        columns=["course_id", "periods_required", "schedule_structure"],
    )


def catalog_with_department() -> pd.DataFrame:
    rows = []
    math = {"MATH1", "MATH2", "MATH2_3_HA", "CALC_D_LINALG"}
    for _, row in catalog().iterrows():
        rows.append((row["course_id"], "Mathematics" if row["course_id"] in math else "Other"))
    return pd.DataFrame(rows, columns=["course_id", "department"])


def students(rows: list[tuple[str, int, int, bool]] | None = None) -> pd.DataFrame:
    rows = rows or [("STU_1", 12, 1, False)]
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
    capacity: int = 1,
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


def base_sections() -> list[tuple]:
    return [
        section_row("SEC_CORE_A_1", "CORE_A", "P1", capacity=10, group_id="CORE_A_1"),
        section_row("SEC_CORE_A_2", "CORE_A", "P2", capacity=10, group_id="CORE_A_2"),
        section_row("SEC_CORE_B_1", "CORE_B", "P1", capacity=10, group_id="CORE_B_1"),
        section_row("SEC_CORE_C_1", "CORE_C", "P2", capacity=10, group_id="CORE_C_1"),
        section_row("SEC_CORE_D_1", "CORE_D", "P3", capacity=10, group_id="CORE_D_1"),
        section_row("SEC_ALT1_1", "ALT1", "P4", capacity=10, group_id="ALT1_1"),
        section_row("SEC_ALT2_1", "ALT2", "P5", capacity=10, group_id="ALT2_1"),
        section_row("SEC_ALT3_1", "ALT3", "P6", capacity=10, group_id="ALT3_1"),
        section_row("SEC_MATH1_1", "MATH1", "P1", capacity=10, group_id="MATH1_1"),
        section_row("SEC_MATH2_1", "MATH2", "P3", capacity=10, group_id="MATH2_1"),
        section_row("SEC_MATH23_1", "MATH2_3_HA", "P5", "P6", capacity=10, group_id="MATH23_1"),
        section_row("SEC_CALCD_1", "CALC_D_LINALG", "P7", capacity=45, group_id="CALCD_1"),
        section_row("SEC_DOUBLE_1", "DOUBLE", "P1", "P2", capacity=10, group_id="DOUBLE_1"),
        section_row(
            "SEC_GOV_1_S1",
            "GOV_ECON_REG",
            "P7",
            semester="semester_1",
            capacity=10,
            block_id="GOV_1",
            group_id="GOV_1",
            logical_block_id="GOV_ECON_REG",
            semester_content="Government",
        ),
        section_row(
            "SEC_GOV_1_S2",
            "GOV_ECON_REG",
            "P7",
            semester="semester_2",
            capacity=10,
            block_id="GOV_1",
            group_id="GOV_1",
            logical_block_id="GOV_ECON_REG",
            semester_content="Economics",
        ),
        section_row(
            "SEC_AP_PHYSC_1",
            "AP_PHYSC",
            "P4",
            semester="paired",
            capacity=10,
            block_id="AP_PHYSC_1",
            group_id="AP_PHYSC_1",
            logical_block_id="AP_PHYSC",
            semester_content="Mechanics / Electricity and Magnetism",
        ),
        section_row("SEC_HIGH_1", "HIGH", "P1", capacity=200, group_id="HIGH_1"),
    ]


def canonical(
    student_rows: list[tuple[str, int, int, bool]] | None = None,
    request_rows: list[tuple] | None = None,
    section_rows: list[tuple] | None = None,
):
    return canonicalize_allocation_input(
        students(student_rows),
        requests(request_rows or [request_row("STU_1", "CORE_A")]),
        sections(section_rows or base_sections()),
        catalog(),
    )


def fallback_rules() -> tuple[MathFallbackRule, ...]:
    return (MathFallbackRule("MATH2_3_HA", "MATH2", "mandatory_fallback", True, "test"),)


def math_ids() -> tuple[str, ...]:
    dept = catalog().merge(catalog_with_department(), on="course_id")
    return math_course_ids_from_catalog(dept)


def run_solver(input_data, seed: int = 20260630, max_time: float = 2.0):
    return run_fair_cp_sat_solver(
        input_data,
        seed=seed,
        math_fallback_rules=fallback_rules(),
        math_course_ids=math_ids(),
        max_time_seconds_per_stage=max_time,
    )


def outcome(result, request_key: str):
    return next(item for item in result.request_outcomes if item.request_key == request_key)


def fallback_outcome(result, source_request_key: str):
    return next(item for item in result.mandatory_fallback_outcomes if item.source_request_key == source_request_key)


def student_outcome(result, student_id: str):
    return next(item for item in result.student_outcomes if item.student_id == student_id)


def key(student_id: str, course_id: str) -> str:
    return f"primary:{student_id}:{course_id}"


def alt_key(student_id: str, rank: int, course_id: str) -> str:
    return f"alternate:{student_id}:{rank}:{course_id}"


def gov_request_rows(student_id: str) -> list[tuple]:
    return [
        request_row(student_id, "GOV_ECON_REG", "primary", "", "gov_econ_block", "GOV_ECON_REG"),
        request_row(student_id, "GOV_ECON_REG", "primary", "", "gov_econ_block", "GOV_ECON_REG"),
    ]


def test_basic_model_constraints_and_valid_candidate_assignment() -> None:
    data = canonical([("STU_1", 12, 1, False)], [request_row("STU_1", "CORE_A")])
    result = run_solver(data)

    assert result.solve_status == CpSatSolveStatus.OPTIMAL
    assert result.assignments[0].linked_section_group_id in data.candidate_index[key("STU_1", "CORE_A")]
    assert outcome(result, key("STU_1", "CORE_A")).status == PrimaryRequestStatus.ASSIGNED


def test_request_assigned_at_most_once_even_with_multiple_candidate_sections() -> None:
    data = canonical([("STU_1", 12, 1, False)], [request_row("STU_1", "CORE_A")])
    result = run_solver(data)

    assert len([assignment for assignment in result.assignments if assignment.request_key == key("STU_1", "CORE_A")]) == 1


def test_section_capacity_and_normal_shortage_are_feasible_not_model_invalid() -> None:
    data = canonical(
        [("STU_1", 12, 1, False), ("STU_2", 12, 1, False)],
        [request_row("STU_1", "CORE_A"), request_row("STU_2", "CORE_A")],
        [section_row("SEC_CORE_A_1", "CORE_A", "P1", capacity=1, group_id="CORE_A_1")],
    )
    result = run_solver(data)

    assert result.solve_status == CpSatSolveStatus.OPTIMAL
    assert result.solve_status != CpSatSolveStatus.MODEL_INVALID
    assert sum(row.assigned_count for row in result.section_roster_summary if row.linked_section_group_id == "CORE_A_1") == 1


def test_student_period_conflict_and_target_units_are_respected() -> None:
    data = canonical(
        [("STU_1", 12, 1, False)],
        [request_row("STU_1", "CORE_A"), request_row("STU_1", "CORE_B")],
        [
            section_row("SEC_A", "CORE_A", "P1", capacity=10, group_id="A"),
            section_row("SEC_B", "CORE_B", "P1", capacity=10, group_id="B"),
        ],
    )
    result = run_solver(data)

    assert student_outcome(result, "STU_1").assigned_period_units == 1
    assert student_outcome(result, "STU_1").primary_unmet_count == 1


def test_duplicate_logical_identity_cannot_be_assigned_via_primary_and_alternate() -> None:
    data = canonical(
        [("STU_1", 12, 2, False)],
        [request_row("STU_1", "CORE_A"), request_row("STU_1", "CORE_A", "alternate", 1, "alternate")],
    )
    result = run_solver(data)

    assert len(result.assignments) == 1
    assert outcome(result, alt_key("STU_1", 1, "CORE_A")).status != AlternateRequestStatus.ASSIGNED


def test_no_candidate_request_remains_unmet() -> None:
    data = canonical([("STU_1", 12, 1, False)], [request_row("STU_1", "NOSEC")])
    result = run_solver(data)

    assert outcome(result, key("STU_1", "NOSEC")).status == PrimaryRequestStatus.UNMET_NO_CANDIDATES


def test_solver_does_not_modify_canonical_input_or_global_random_state() -> None:
    data = canonical()
    before = data
    random.seed(99)
    random_state = random.getstate()

    run_solver(data, seed=7)

    assert data == before
    assert random.getstate() == random_state


def test_math2_3_consumes_two_periods_and_two_units() -> None:
    data = canonical([("STU_1", 12, 2, False)], [request_row("STU_1", "MATH2_3_HA")])
    result = run_solver(data)

    assert result.assignments[0].occupied_periods == ("P5", "P6")
    assert result.assignments[0].period_units == 2
    assert student_outcome(result, "STU_1").assigned_period_units == 2


def test_gov_econ_linked_rows_use_one_seat_and_one_period() -> None:
    data = canonical([("STU_1", 12, 1, False)], gov_request_rows("STU_1"))
    result = run_solver(data)

    assert len(result.assignments) == 1
    assert result.assignments[0].member_section_ids == ("SEC_GOV_1_S1", "SEC_GOV_1_S2")
    assert result.assignments[0].occupied_periods == ("P7",)
    assert next(row for row in result.section_roster_summary if row.linked_section_group_id == "GOV_1").assigned_count == 1


def test_ap_physics_c_is_single_logical_period_assignment() -> None:
    data = canonical([("STU_1", 12, 1, False)], [request_row("STU_1", "AP_PHYSC")])
    result = run_solver(data)

    assert result.assignments[0].request_candidate_key == "AP_PHYSC"
    assert result.assignments[0].period_units == 1
    assert result.assignments[0].occupied_periods == ("P4",)


def test_protected_student_all_primary_requests_are_hard_required() -> None:
    data = canonical(
        [("PRO", 12, 2, True)],
        [request_row("PRO", "CORE_A"), request_row("PRO", "CORE_D")],
    )
    result = run_solver(data)

    assert student_outcome(result, "PRO").protected_fairness_violation is False
    assert student_outcome(result, "PRO").primary_unmet_count == 0


def test_ordinary_student_max_one_primary_unmet_is_hard_required() -> None:
    data = canonical(
        [("STU_1", 12, 2, False)],
        [request_row("STU_1", "CORE_A"), request_row("STU_1", "CORE_B"), request_row("STU_1", "CORE_D")],
        [
            section_row("SEC_A", "CORE_A", "P1", capacity=10, group_id="A"),
            section_row("SEC_B", "CORE_B", "P1", capacity=10, group_id="B"),
            section_row("SEC_D", "CORE_D", "P2", capacity=10, group_id="D"),
        ],
    )
    result = run_solver(data)

    assert student_outcome(result, "STU_1").ordinary_fairness_violation is False
    assert student_outcome(result, "STU_1").primary_unmet_count <= 1


def test_high_demand_primary_requests_are_hard_required() -> None:
    student_rows = [("STU_1", 12, 1, False)]
    student_rows.extend((f"FILL_{index:03d}", 12, 1, False) for index in range(121))
    request_rows = [request_row("STU_1", "HIGH")]
    request_rows.extend(request_row(f"FILL_{index:03d}", "HIGH") for index in range(121))
    data = canonical(student_rows, request_rows, [section_row("SEC_HIGH", "HIGH", "P1", capacity=122, group_id="HIGH_1")])
    result = run_solver(data)

    assert result.policy_report is not None
    assert result.policy_report.high_demand_violation_count == 0
    assert all(outcome(result, key(student_id, "HIGH")).status == PrimaryRequestStatus.ASSIGNED for student_id, *_ in student_rows)


def test_true_hard_policy_conflict_returns_infeasible() -> None:
    data = canonical(
        [("P1", 12, 1, True), ("P2", 12, 1, True)],
        [request_row("P1", "CORE_A"), request_row("P2", "CORE_A")],
        [section_row("SEC_CORE_A", "CORE_A", "P1", capacity=1, group_id="CORE_A_1")],
    )
    result = run_solver(data)

    assert result.solve_status == CpSatSolveStatus.INFEASIBLE
    assert result.assignments == ()


def test_unique_math_soft_priority_beats_equal_ordinary_primary() -> None:
    data = canonical(
        [("STU_1", 12, 1, False)],
        [request_row("STU_1", "MATH1"), request_row("STU_1", "CORE_A")],
        [
            section_row("SEC_MATH", "MATH1", "P1", capacity=1, group_id="MATH1_1"),
            section_row("SEC_CORE", "CORE_A", "P1", capacity=1, group_id="CORE_A_1"),
        ],
    )
    result = run_solver(data)

    assert outcome(result, key("STU_1", "MATH1")).status == PrimaryRequestStatus.ASSIGNED
    assert result.math_policy_report.current_math_coverage_violation_student_ids == ()


def test_math_coverage_priority_respects_hard_ordinary_max_one_policy() -> None:
    data = canonical(
        [("STU_1", 12, 2, False)],
        [request_row("STU_1", "MATH2_3_HA"), request_row("STU_1", "CORE_A")],
        [
            section_row("SEC_MATH", "MATH2_3_HA", "P1", "P2", capacity=1, group_id="MATH23_1"),
            section_row("SEC_A", "CORE_A", "P1", capacity=1, group_id="CORE_A_1"),
        ],
    )
    result = run_solver(data)

    assert outcome(result, key("STU_1", "MATH2_3_HA")).status == PrimaryRequestStatus.ASSIGNED
    assert student_outcome(result, "STU_1").primary_assigned_count == 1
    assert student_outcome(result, "STU_1").primary_unmet_count == 1


def test_multiple_math_primary_needs_at_least_one_for_coverage() -> None:
    data = canonical(
        [("STU_1", 12, 1, False)],
        [request_row("STU_1", "MATH1"), request_row("STU_1", "MATH2")],
        [
            section_row("SEC_M1", "MATH1", "P1", capacity=1, group_id="MATH1_1"),
            section_row("SEC_M2", "MATH2", "P1", capacity=1, group_id="MATH2_1"),
        ],
    )
    result = run_solver(data)

    assert result.math_policy_report.current_math_coverage_violation_student_ids == ()
    assert student_outcome(result, "STU_1").primary_assigned_count == 1


def test_no_math_primary_student_is_exempt_from_math_violation() -> None:
    result = run_solver(canonical([("STU_1", 12, 1, False)], [request_row("STU_1", "CORE_A")]))

    assert result.math_policy_report.no_math_primary_student_ids == ("STU_1",)
    assert result.math_policy_report.current_math_coverage_violation_student_ids == ()


def test_single_calc_d_linalg_capacity_gap_remains_feasible_with_math_violations() -> None:
    student_rows = [(f"STU_{index:02d}", 12, 1, False) for index in range(50)]
    request_rows = [request_row(student_id, "CALC_D_LINALG") for student_id, *_ in student_rows]
    data = canonical(
        student_rows,
        request_rows,
        [section_row("SEC_CALCD_1", "CALC_D_LINALG", "P1", capacity=45, group_id="CALCD_1")],
    )
    result = run_solver(data)

    assert result.solve_status == CpSatSolveStatus.OPTIMAL
    assert sum(1 for item in result.request_outcomes if item.status == PrimaryRequestStatus.ASSIGNED) == 45
    assert len(result.math_policy_report.current_math_coverage_violation_student_ids) == 5
    assert len(data.logical_sections) == 1


def test_math_classification_uses_department_not_title() -> None:
    dept = catalog().merge(catalog_with_department(), on="course_id")
    assert "MATH1" in math_course_ids_from_catalog(dept)
    assert "CORE_A" not in math_course_ids_from_catalog(dept)


def test_solver_does_not_infer_math_from_course_title_string() -> None:
    data = canonical([("STU_1", 12, 1, False)], [request_row("STU_1", "CORE_A")])
    result = run_fair_cp_sat_solver(data, seed=1, math_course_ids=(), math_fallback_rules=(), max_time_seconds_per_stage=2)

    assert result.math_policy_report.no_math_primary_student_ids == ("STU_1",)


def test_fallback_not_assigned_when_source_primary_assigned() -> None:
    data = canonical([("STU_1", 12, 2, False)], [request_row("STU_1", "MATH2_3_HA")])
    result = run_solver(data)

    assert fallback_outcome(result, key("STU_1", "MATH2_3_HA")).status == MandatoryFallbackStatus.NOT_REQUIRED_SOURCE_ASSIGNED


def test_fallback_not_assigned_when_other_math_primary_satisfies_coverage() -> None:
    data = canonical(
        [("STU_1", 12, 1, False)],
        [request_row("STU_1", "MATH2_3_HA"), request_row("STU_1", "MATH1")],
        [row for row in base_sections() if row[1] != "MATH2_3_HA"],
    )
    result = run_solver(data)

    assert outcome(result, key("STU_1", "MATH1")).status == PrimaryRequestStatus.ASSIGNED
    assert fallback_outcome(result, key("STU_1", "MATH2_3_HA")).status == MandatoryFallbackStatus.NOT_REQUIRED_MATH_COVERAGE_ALREADY_SATISFIED


def test_fallback_can_assign_when_source_unmet_and_no_other_math_coverage() -> None:
    data = canonical(
        [("STU_1", 12, 1, False)],
        [request_row("STU_1", "MATH2_3_HA")],
        [row for row in base_sections() if row[1] != "MATH2_3_HA"],
    )
    result = run_solver(data)

    assert fallback_outcome(result, key("STU_1", "MATH2_3_HA")).status == MandatoryFallbackStatus.ASSIGNED
    assert student_outcome(result, "STU_1").mandatory_fallback_assigned_count == 1


def test_fallback_uses_real_capacity_period_and_target_units() -> None:
    data = canonical(
        [("STU_1", 12, 1, False)],
        [request_row("STU_1", "MATH2_3_HA"), request_row("STU_1", "ALT1", "alternate", 1, "alternate")],
        [
            section_row("SEC_MATH2", "MATH2", "P1", capacity=1, group_id="MATH2_1"),
            section_row("SEC_ALT1", "ALT1", "P1", capacity=1, group_id="ALT1_1"),
        ],
    )
    result = run_solver(data)

    assert fallback_outcome(result, key("STU_1", "MATH2_3_HA")).status == MandatoryFallbackStatus.ASSIGNED
    assert outcome(result, alt_key("STU_1", 1, "ALT1")).status == AlternateRequestStatus.NOT_NEEDED
    assert student_outcome(result, "STU_1").remaining_period_units == 0


def test_fallback_failure_is_math_violation_not_global_infeasible() -> None:
    data = canonical(
        [("STU_1", 12, 1, False)],
        [request_row("STU_1", "MATH2_3_HA")],
        [section_row("SEC_CORE_A", "CORE_A", "P1", capacity=1, group_id="CORE_A_1")],
    )
    result = run_solver(data)

    assert result.solve_status == CpSatSolveStatus.OPTIMAL
    assert fallback_outcome(result, key("STU_1", "MATH2_3_HA")).status == MandatoryFallbackStatus.UNASSIGNED_NO_CANDIDATES
    assert result.math_policy_report.current_math_coverage_violation_student_ids == ("STU_1",)


def test_alternate_ranks_are_lexicographic_and_do_not_change_primary_stats() -> None:
    data = canonical(
        [("STU_1", 12, 1, False)],
        [
            request_row("STU_1", "NOSEC"),
            request_row("STU_1", "ALT1", "alternate", 1, "alternate"),
            request_row("STU_1", "ALT2", "alternate", 2, "alternate"),
            request_row("STU_1", "ALT3", "alternate", 3, "alternate"),
        ],
    )
    result = run_solver(data)

    assert outcome(result, alt_key("STU_1", 1, "ALT1")).status == AlternateRequestStatus.ASSIGNED
    assert outcome(result, alt_key("STU_1", 2, "ALT2")).status == AlternateRequestStatus.NOT_NEEDED
    assert student_outcome(result, "STU_1").primary_unmet_count == 1


def test_rank2_is_prioritized_before_rank3_when_rank1_absent() -> None:
    data = canonical(
        [("STU_1", 12, 1, False)],
        [
            request_row("STU_1", "NOSEC"),
            request_row("STU_1", "ALT2", "alternate", 2, "alternate"),
            request_row("STU_1", "ALT3", "alternate", 3, "alternate"),
        ],
    )
    result = run_solver(data)

    assert outcome(result, alt_key("STU_1", 2, "ALT2")).status == AlternateRequestStatus.ASSIGNED
    assert outcome(result, alt_key("STU_1", 3, "ALT3")).status == AlternateRequestStatus.NOT_NEEDED


def test_alternate_can_fill_remaining_units_without_exceeding_target() -> None:
    data = canonical(
        [("STU_1", 12, 2, False)],
        [request_row("STU_1", "NOSEC"), request_row("STU_1", "CORE_A"), request_row("STU_1", "ALT1", "alternate", 1, "alternate")],
    )
    result = run_solver(data)

    assert outcome(result, alt_key("STU_1", 1, "ALT1")).status == AlternateRequestStatus.ASSIGNED
    assert student_outcome(result, "STU_1").assigned_period_units == 2


def test_fallback_math_priority_beats_ordinary_alternate() -> None:
    data = canonical(
        [("STU_1", 12, 1, False)],
        [request_row("STU_1", "MATH2_3_HA"), request_row("STU_1", "ALT1", "alternate", 1, "alternate")],
        [
            section_row("SEC_MATH2", "MATH2", "P1", capacity=1, group_id="MATH2_1"),
            section_row("SEC_ALT1", "ALT1", "P1", capacity=1, group_id="ALT1_1"),
        ],
    )
    result = run_solver(data)

    assert fallback_outcome(result, key("STU_1", "MATH2_3_HA")).status == MandatoryFallbackStatus.ASSIGNED
    assert outcome(result, alt_key("STU_1", 1, "ALT1")).status != AlternateRequestStatus.ASSIGNED


def test_primary_unmet_count_dominates_unmet_period_units() -> None:
    data = canonical(
        [("STU_1", 12, 2, False)],
        [request_row("STU_1", "DOUBLE"), request_row("STU_1", "CORE_A"), request_row("STU_1", "CORE_C")],
        [
            section_row("SEC_DOUBLE", "DOUBLE", "P1", "P2", capacity=1, group_id="DOUBLE_1"),
            section_row("SEC_A", "CORE_A", "P1", capacity=1, group_id="CORE_A_1"),
            section_row("SEC_C", "CORE_C", "P2", capacity=1, group_id="CORE_C_1"),
        ],
    )
    result = run_solver(data)

    assert outcome(result, key("STU_1", "CORE_A")).status == PrimaryRequestStatus.ASSIGNED
    assert outcome(result, key("STU_1", "CORE_C")).status == PrimaryRequestStatus.ASSIGNED
    assert student_outcome(result, "STU_1").primary_unmet_count == 1


def test_unmet_units_break_tie_after_primary_count() -> None:
    data = canonical(
        [("STU_1", 12, 2, False)],
        [request_row("STU_1", "DOUBLE"), request_row("STU_1", "CORE_A")],
        [
            section_row("SEC_DOUBLE", "DOUBLE", "P1", "P2", capacity=1, group_id="DOUBLE_1"),
            section_row("SEC_A", "CORE_A", "P1", capacity=1, group_id="CORE_A_1"),
        ],
    )
    result = run_solver(data)

    assert outcome(result, key("STU_1", "DOUBLE")).status == PrimaryRequestStatus.ASSIGNED
    assert student_outcome(result, "STU_1").primary_unmet_period_units == 1


def test_schedule_completion_runs_after_higher_objectives() -> None:
    data = canonical(
        [("STU_1", 12, 2, False)],
        [request_row("STU_1", "NOSEC"), request_row("STU_1", "CORE_A"), request_row("STU_1", "ALT1", "alternate", 1, "alternate")],
    )
    result = run_solver(data)

    assert student_outcome(result, "STU_1").fully_scheduled is True
    assert result.objective_values.fully_scheduled_students == 1


def test_stage_diagnostics_record_all_lexicographic_stages() -> None:
    result = run_solver(canonical())

    assert [item.stage_name for item in result.stage_diagnostics] == [
        CpSatStageName.MATH_COVERAGE,
        CpSatStageName.PRIMARY_SATISFACTION,
        CpSatStageName.ALTERNATE_RANK_1,
        CpSatStageName.ALTERNATE_RANK_2,
        CpSatStageName.ALTERNATE_RANK_3,
        CpSatStageName.FULLY_SCHEDULED,
        CpSatStageName.REMAINING_PERIOD_UNITS,
        CpSatStageName.SEEDED_TIE_BREAK,
    ]
    assert all(item.status == CpSatSolveStatus.OPTIMAL for item in result.stage_diagnostics)


def test_same_seed_full_result_and_assignments_are_identical() -> None:
    data = canonical([("STU_1", 12, 1, False)], [request_row("STU_1", "CORE_A")])

    assert run_solver(data, seed=13) == run_solver(data, seed=13)


def test_equivalent_request_row_order_keeps_objectives_and_result() -> None:
    student_rows = [("STU_1", 12, 2, False)]
    first = canonical(student_rows, [request_row("STU_1", "CORE_A"), request_row("STU_1", "CORE_C")])
    second = canonical(student_rows, [request_row("STU_1", "CORE_C"), request_row("STU_1", "CORE_A")])

    assert run_solver(first, seed=14).objective_values == run_solver(second, seed=14).objective_values


def test_different_seed_keeps_objective_vector_on_proven_optimal_tie() -> None:
    data = canonical([("STU_1", 12, 1, False)], [request_row("STU_1", "CORE_A")])
    first = run_solver(data, seed=1)
    second = run_solver(data, seed=2)

    assert first.lexicographic_optimality_proven is True
    assert second.lexicographic_optimality_proven is True
    assert first.objective_values.math_coverage_violations == second.objective_values.math_coverage_violations
    assert first.objective_values.primary_penalty == second.objective_values.primary_penalty
    assert first.objective_values.fully_scheduled_students == second.objective_values.fully_scheduled_students


def test_solution_replays_into_allocation_state_and_has_no_consistency_or_capacity_errors() -> None:
    result = run_solver(canonical([("STU_1", 12, 2, False)], [request_row("STU_1", "CORE_A"), request_row("STU_1", "CORE_C")]))

    assert result.consistency_issues == ()
    assert all(row.assigned_count <= row.capacity for row in result.section_roster_summary)


def test_random_and_constrained_baseline_regressions_are_unchanged() -> None:
    data = canonical(
        [("STU_1", 12, 2, False)],
        [request_row("STU_1", "CORE_A"), request_row("STU_1", "CORE_C")],
    )

    assert run_seeded_random_baseline(data, seed=0) == run_seeded_random_baseline(data, seed=0)
    assert run_constrained_first_baseline(data, seed=0, math_fallback_rules=fallback_rules(), math_course_ids=math_ids()) == run_constrained_first_baseline(
        data,
        seed=0,
        math_fallback_rules=fallback_rules(),
        math_course_ids=math_ids(),
    )
