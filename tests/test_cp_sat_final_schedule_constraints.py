from __future__ import annotations

import pytest

import src.allocation.cp_sat_solver as cp_sat_solver_module
from src.allocation import (
    CpSatBootstrapStatus,
    CpSatStageName,
    CpSatModelStats,
    CpSatSolveStatus,
    MandatoryFallbackStatus,
    PrimaryRequestStatus,
    run_fair_cp_sat_solver,
)
from src.allocation.cp_sat_solver import CpSatFinalSchedulePolicyConsistencyError
from src.final_schedule_policy import (
    FinalSchedulePolicyReport,
    FinalSchedulePolicySummary,
    MAXIMUM_ORDINARY_PRIMARY_UNMET_COUNT,
    MAXIMUM_PROTECTED_PRIMARY_UNMET_COUNT,
    MAXIMUM_SCHEDULE_GAP_COUNT,
    MINIMUM_ASSIGNED_LOGICAL_COURSE_COUNT,
    SCHEMA_VERSION,
    evaluate_final_schedule_policy,
)
from tests.test_cp_sat_solver import (
    alt_key,
    canonical,
    fallback_rules,
    gov_request_rows,
    key,
    math_ids,
    outcome,
    request_row,
    section_row,
    student_outcome,
)


def run_final_solver(data):
    return run_fair_cp_sat_solver(
        data,
        seed=20260630,
        math_fallback_rules=fallback_rules(),
        math_course_ids=math_ids(),
        max_time_seconds_per_stage=2,
        use_constrained_first_hint=False,
    )


def primary_rows(student_id: str, *course_ids: str) -> list[tuple]:
    return [request_row(student_id, course_id) for course_id in course_ids]


def test_policy_constants_are_shared_with_cp_sat_metadata() -> None:
    stats = CpSatModelStats(0, 0, 0.0, 0.0)

    assert stats.final_schedule_policy_schema_version == SCHEMA_VERSION
    assert stats.minimum_assigned_logical_course_count == MINIMUM_ASSIGNED_LOGICAL_COURSE_COUNT
    assert stats.maximum_logical_schedule_gap_count == MAXIMUM_SCHEDULE_GAP_COUNT
    assert stats.maximum_ordinary_primary_unmet_count == MAXIMUM_ORDINARY_PRIMARY_UNMET_COUNT
    assert stats.maximum_protected_primary_unmet_count == MAXIMUM_PROTECTED_PRIMARY_UNMET_COUNT


@pytest.mark.parametrize(
    ("target", "courses", "expected_assigned"),
    [
        (7, ("CORE_B", "CORE_C", "CORE_D", "ALT1", "ALT2", "ALT3"), 6),
        (6, ("CORE_B", "CORE_C", "CORE_D", "ALT1", "ALT2"), 5),
        (5, ("CORE_B", "CORE_C", "CORE_D", "ALT1", "ALT2"), 5),
    ],
)
def test_final_model_accepts_allowed_logical_schedule_boundaries(target, courses, expected_assigned) -> None:
    data = canonical([("STU", 12, target, False)], primary_rows("STU", *courses))

    result = run_final_solver(data)

    assert result.solve_status == CpSatSolveStatus.OPTIMAL
    assert result.model_stats.final_schedule_hard_constraints_enabled is True
    assert result.model_stats.post_solve_policy_gate_pass is True
    assert len(student_outcome(result, "STU").assignment_keys) == expected_assigned
    assert evaluate_final_schedule_policy(result.algorithm_name, result.student_outcomes).summary.final_schedule_policy_pass


@pytest.mark.parametrize(
    ("target", "courses"),
    [
        (7, ("CORE_B", "CORE_C", "CORE_D", "ALT1", "ALT2")),
        (6, ("CORE_B", "CORE_C", "CORE_D", "ALT1")),
        (5, ("CORE_B", "CORE_C", "CORE_D", "ALT1")),
    ],
)
def test_final_model_rejects_too_few_logical_courses(target, courses) -> None:
    data = canonical([("STU", 12, target, False)], primary_rows("STU", *courses))

    result = run_final_solver(data)
    full_stage = next(
        item
        for item in result.stage_diagnostics
        if item.stage_name == CpSatStageName.FULL_MODEL_FEASIBILITY_INCUMBENT
    )

    assert result.solve_status == CpSatSolveStatus.INFEASIBLE
    assert result.assignments == ()
    assert full_stage.status == CpSatSolveStatus.INFEASIBLE


def test_ordinary_student_may_have_one_primary_unmet_when_alternate_fills_load() -> None:
    data = canonical(
        [("STU", 12, 5, False)],
        [
            *primary_rows("STU", "CORE_B", "CORE_C", "CORE_D", "ALT1"),
            request_row("STU", "NOSEC"),
            request_row("STU", "ALT2", "alternate", 1, "alternate"),
        ],
    )

    result = run_final_solver(data)

    assert result.solve_status == CpSatSolveStatus.OPTIMAL
    assert student_outcome(result, "STU").primary_unmet_count == 1
    assert outcome(result, alt_key("STU", 1, "ALT2")).status == "assigned"
    assert result.model_stats.post_solve_policy_gate_pass is True


def test_ordinary_student_primary_unmet_two_is_infeasible_even_if_alternates_could_fill() -> None:
    data = canonical(
        [("STU", 12, 5, False)],
        [
            *primary_rows("STU", "CORE_A", "CORE_B", "CORE_C", "CORE_D"),
            request_row("STU", "NOSEC"),
            request_row("STU", "ALT1", "alternate", 1, "alternate"),
            request_row("STU", "ALT2", "alternate", 2, "alternate"),
        ],
    )

    result = run_final_solver(data)

    assert result.solve_status == CpSatSolveStatus.INFEASIBLE
    assert result.model_stats.bootstrap_status == CpSatBootstrapStatus.INFEASIBLE


def test_protected_student_primary_unmet_one_is_infeasible() -> None:
    data = canonical(
        [("PRO", 12, 5, True)],
        [
            *primary_rows("PRO", "CORE_B", "CORE_C", "CORE_D", "ALT1"),
            request_row("PRO", "NOSEC"),
            request_row("PRO", "ALT2", "alternate", 1, "alternate"),
        ],
    )

    result = run_final_solver(data)

    assert result.solve_status == CpSatSolveStatus.INFEASIBLE


def test_math2_3_counts_as_one_logical_course_but_two_period_units() -> None:
    data = canonical(
        [("STU", 12, 7, False)],
        primary_rows("STU", "CORE_B", "CORE_C", "CORE_D", "ALT1", "CALC_D_LINALG", "MATH2_3_HA"),
    )

    result = run_final_solver(data)
    outcome_row = student_outcome(result, "STU")

    assert result.solve_status == CpSatSolveStatus.OPTIMAL
    assert len(outcome_row.assignment_keys) == 6
    assert outcome_row.assigned_period_units == 7
    assert outcome(result, key("STU", "MATH2_3_HA")).status == PrimaryRequestStatus.ASSIGNED


def test_linked_gov_econ_counts_as_one_logical_course() -> None:
    data = canonical(
        [("STU", 12, 5, False)],
        [
            *gov_request_rows("STU"),
            *primary_rows("STU", "CORE_B", "CORE_C", "CORE_D", "ALT1"),
        ],
    )

    result = run_final_solver(data)
    assignment = next(item for item in result.assignments if item.request_candidate_key == "GOV_ECON_REG")

    assert result.solve_status == CpSatSolveStatus.OPTIMAL
    assert len(student_outcome(result, "STU").assignment_keys) == 5
    assert assignment.member_section_ids == ("SEC_GOV_1_S1", "SEC_GOV_1_S2")


def test_duplicate_logical_course_is_not_double_counted_for_final_load() -> None:
    data = canonical(
        [("STU", 12, 5, False)],
        [
            *primary_rows("STU", "CORE_A", "CORE_B", "CORE_C", "CORE_D", "ALT1", "ALT2"),
            request_row("STU", "CORE_A", "alternate", 1, "alternate"),
        ],
    )

    result = run_final_solver(data)

    assert result.solve_status == CpSatSolveStatus.OPTIMAL
    assert len(student_outcome(result, "STU").assignment_keys) == 5
    assert outcome(result, alt_key("STU", 1, "CORE_A")).status != "assigned"


def test_math_fallback_counts_for_logical_load_but_not_primary_satisfaction() -> None:
    data = canonical(
        [("STU", 12, 5, False)],
        [
            request_row("STU", "MATH2_3_HA"),
            *primary_rows("STU", "CORE_B", "CORE_C", "CORE_D", "ALT1"),
        ],
        [
            section_row("SEC_MATH2", "MATH2", "P5", capacity=1, group_id="MATH2_1"),
            section_row("SEC_B", "CORE_B", "P1", capacity=1, group_id="CORE_B_1"),
            section_row("SEC_C", "CORE_C", "P2", capacity=1, group_id="CORE_C_1"),
            section_row("SEC_D", "CORE_D", "P3", capacity=1, group_id="CORE_D_1"),
            section_row("SEC_ALT1", "ALT1", "P4", capacity=1, group_id="ALT1_1"),
        ],
    )

    result = run_final_solver(data)
    outcome_row = student_outcome(result, "STU")

    assert result.solve_status == CpSatSolveStatus.OPTIMAL
    assert outcome_row.primary_unmet_count == 1
    assert outcome_row.mandatory_fallback_assigned_count == 1
    assert len(outcome_row.assignment_keys) == 5
    assert result.mandatory_fallback_outcomes[0].status == MandatoryFallbackStatus.ASSIGNED


def test_high_demand_primary_hard_policy_still_triggers_above_120() -> None:
    student_rows = [(f"STU_{index:03d}", 12, 5, False) for index in range(121)]
    request_rows = [request_row(student_id, "HIGH") for student_id, *_ in student_rows]
    data = canonical(
        student_rows,
        request_rows,
        [section_row("SEC_HIGH", "HIGH", "P1", capacity=120, group_id="HIGH_1")],
    )

    result = run_final_solver(data)

    assert result.solve_status == CpSatSolveStatus.INFEASIBLE
    assert result.model_stats.bootstrap_status == CpSatBootstrapStatus.INFEASIBLE


def test_bootstrap_feasible_is_not_treated_as_final_policy_solution() -> None:
    data = canonical(
        [("STU", 12, 5, False)],
        [
            request_row("STU", "CORE_B"),
            request_row("STU", "CORE_C", "alternate", 1, "alternate"),
            request_row("STU", "CORE_D", "alternate", 2, "alternate"),
            request_row("STU", "ALT1", "alternate", 3, "alternate"),
        ],
    )

    result = run_final_solver(data)

    assert result.model_stats.bootstrap_status == CpSatBootstrapStatus.FEASIBLE_FOUND
    assert result.solve_status == CpSatSolveStatus.INFEASIBLE
    assert result.assignments == ()


def test_unknown_does_not_export_assignment_or_policy_pass() -> None:
    result = run_fair_cp_sat_solver(
        canonical([("STU", 12, 5, False)], primary_rows("STU", "CORE_B", "CORE_C", "CORE_D", "ALT1", "ALT2")),
        seed=1,
        math_fallback_rules=fallback_rules(),
        math_course_ids=math_ids(),
        max_time_seconds_per_stage=2,
        max_total_time_seconds=0.0,
        use_constrained_first_hint=False,
    )

    assert result.solve_status == CpSatSolveStatus.UNKNOWN
    assert result.assignments == ()
    assert result.model_stats.post_solve_policy_gate_pass is None


def test_post_solve_policy_gate_failure_is_internal_error(monkeypatch) -> None:
    summary = FinalSchedulePolicySummary(
        algorithm_name="fair_cp_sat_solver_v1_2",
        final_schedule_policy_pass=False,
        violating_student_count=1,
        protected_primary_unmet_violation_count=0,
        ordinary_primary_unmet_violation_count=0,
        schedule_gap_over_limit_count=0,
        below_minimum_course_count=1,
        minimum_assigned_course_count=4,
        maximum_schedule_gap_count=1,
        maximum_primary_unmet_count=0,
        logical_fully_scheduled_student_count=0,
        students_with_logical_schedule_gap=1,
        total_logical_schedule_gap=1,
    )
    monkeypatch.setattr(
        cp_sat_solver_module,
        "evaluate_final_schedule_policy",
        lambda *args, **kwargs: FinalSchedulePolicyReport(summary=summary, violations=()),
    )
    data = canonical([("STU", 12, 5, False)], primary_rows("STU", "CORE_B", "CORE_C", "CORE_D", "ALT1", "ALT2"))

    with pytest.raises(CpSatFinalSchedulePolicyConsistencyError, match="Final Schedule Policy Gate v1"):
        run_final_solver(data)
