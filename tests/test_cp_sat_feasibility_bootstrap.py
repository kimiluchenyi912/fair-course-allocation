from __future__ import annotations

import pandas as pd

import src.allocation.cp_sat_solver as cp_sat_solver_module
from src.allocation import (
    CpSatBootstrapStatus,
    CpSatModelScope,
    CpSatSolveStatus,
    CpSatStageName,
    MathFallbackRule,
    PrimaryRequestStatus,
    canonicalize_allocation_input,
    run_fair_cp_sat_solver,
)
from tests.test_cp_sat_solver import (
    base_sections,
    catalog,
    fallback_rules,
    key,
    math_ids,
    request_row,
    section_row,
)


def students(rows: list[tuple[str, int, int, bool]]) -> pd.DataFrame:
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


def canonical(student_rows, request_rows, section_rows=None):
    return canonicalize_allocation_input(
        students(student_rows),
        requests(request_rows),
        sections(section_rows or base_sections()),
        catalog(),
    )


def test_bootstrap_model_only_creates_primary_assignment_vars() -> None:
    data = canonical(
        [("STU_1", 12, 1, False)],
        [
            request_row("STU_1", "CORE_A"),
            request_row("STU_1", "ALT1", "alternate", 1, "alternate"),
        ],
    )

    build = cp_sat_solver_module._build_feasibility_bootstrap_model(data)

    assert build.assignment_vars
    assert all(key.request_key.startswith("primary:") for key in build.assignment_vars)
    assert not any("mandatory_fallback" in key.request_key for key in build.assignment_vars)
    assert not any(key.request_key.startswith("alternate:") for key in build.assignment_vars)
    assert "math_violation" not in str(build.model.Proto())
    assert "fully_scheduled" not in str(build.model.Proto())


def test_bootstrap_single_candidate_primary_uses_sparse_assigned_expression() -> None:
    data = canonical(
        [("STU_1", 12, 1, False)],
        [request_row("STU_1", "CORE_A")],
        [section_row("SEC_CORE_A", "CORE_A", "P1", capacity=1, group_id="CORE_A_1")],
    )

    build = cp_sat_solver_module._build_feasibility_bootstrap_model(data)

    assert len(build.assignment_vars) == 1
    assert "boot_assigned" not in str(build.model.Proto())


def test_core_model_single_candidate_primary_uses_sparse_assigned_expression() -> None:
    data = canonical(
        [("STU_1", 12, 1, False)],
        [request_row("STU_1", "CORE_A")],
        [section_row("SEC_CORE_A", "CORE_A", "P1", capacity=1, group_id="CORE_A_1")],
    )

    build = cp_sat_solver_module._build_core_cp_sat_model(data, (), math_ids(), seed=1)

    assert len(build.assignment_vars) == 1
    assert "assigned__primary__STU_1__CORE_A" not in str(build.model.Proto())


def test_bootstrap_zero_candidate_primary_creates_no_assignment_var() -> None:
    data = canonical(
        [("STU_1", 12, 1, False)],
        [request_row("STU_1", "NOSEC")],
        [section_row("SEC_CORE_A", "CORE_A", "P1", capacity=1, group_id="CORE_A_1")],
    )

    build = cp_sat_solver_module._build_feasibility_bootstrap_model(data)

    assert build.assignment_vars == {}
    assert build.assigned_vars[key("STU_1", "NOSEC")] == 0


def test_bootstrap_records_feasible_incumbent_but_not_objective_vector() -> None:
    result = run_fair_cp_sat_solver(
        canonical([("STU_1", 12, 1, True)], [request_row("STU_1", "CORE_A")]),
        seed=1,
        math_fallback_rules=fallback_rules(),
        math_course_ids=math_ids(),
        max_time_seconds_per_stage=2,
    )

    assert result.model_stats.bootstrap_enabled is True
    assert result.model_stats.bootstrap_status == CpSatBootstrapStatus.FEASIBLE_FOUND
    assert result.model_stats.bootstrap_incumbent_found is True
    assert result.model_stats.total_build_time_seconds >= (
        result.model_stats.bootstrap_build_time_seconds
        + result.model_stats.core_build_time_seconds
        + result.model_stats.enrichment_build_time_seconds
        - 0.000003
    )
    assert result.stage_diagnostics[0].model_scope == CpSatModelScope.BOOTSTRAP
    assert result.stage_diagnostics[0].stage_name == CpSatStageName.FEASIBILITY_BOOTSTRAP
    assert CpSatStageName.FEASIBILITY_BOOTSTRAP not in dict(result.model_stats.objective_vector)
    assert result.model_stats.highest_globally_proven_stage != CpSatStageName.FEASIBILITY_BOOTSTRAP


def test_bootstrap_feasible_with_no_external_hint() -> None:
    result = run_fair_cp_sat_solver(
        canonical([("STU_1", 12, 1, True)], [request_row("STU_1", "CORE_A")]),
        seed=1,
        math_fallback_rules=fallback_rules(),
        math_course_ids=math_ids(),
        max_time_seconds_per_stage=2,
        use_constrained_first_hint=False,
    )

    assert result.model_stats.bootstrap_status == CpSatBootstrapStatus.FEASIBLE_FOUND
    assert result.model_stats.bootstrap_hint_strategy == "none"
    assert result.model_stats.core_hint_source == "bootstrap"


def test_bootstrap_feasible_hint_does_not_constrain_core_math_choice() -> None:
    data = canonical(
        [("STU_1", 12, 1, False)],
        [request_row("STU_1", "MATH1"), request_row("STU_1", "CORE_A")],
        [
            section_row("SEC_MATH", "MATH1", "P1", capacity=1, group_id="MATH1_1"),
            section_row("SEC_CORE", "CORE_A", "P1", capacity=1, group_id="CORE_A_1"),
        ],
    )

    result = run_fair_cp_sat_solver(
        data,
        seed=1,
        math_fallback_rules=fallback_rules(),
        math_course_ids=math_ids(),
        max_time_seconds_per_stage=2,
        use_constrained_first_hint=False,
    )

    assert next(item for item in result.request_outcomes if item.request_key == key("STU_1", "MATH1")).status == PrimaryRequestStatus.ASSIGNED


def test_bootstrap_period_conflict_can_make_protected_student_infeasible() -> None:
    data = canonical(
        [("PRO", 12, 2, True)],
        [request_row("PRO", "CORE_A"), request_row("PRO", "CORE_B")],
        [
            section_row("SEC_A", "CORE_A", "P1", capacity=1, group_id="CORE_A_1"),
            section_row("SEC_B", "CORE_B", "P1", capacity=1, group_id="CORE_B_1"),
        ],
    )

    result = run_fair_cp_sat_solver(
        data,
        seed=1,
        math_fallback_rules=fallback_rules(),
        math_course_ids=math_ids(),
        max_time_seconds_per_stage=2,
    )

    assert result.solve_status == CpSatSolveStatus.INFEASIBLE
    assert result.model_stats.bootstrap_status == CpSatBootstrapStatus.INFEASIBLE


def test_bootstrap_target_load_can_make_protected_student_infeasible() -> None:
    data = canonical(
        [("PRO", 12, 1, True)],
        [request_row("PRO", "CORE_A"), request_row("PRO", "CORE_B")],
        [
            section_row("SEC_A", "CORE_A", "P1", capacity=1, group_id="CORE_A_1"),
            section_row("SEC_B", "CORE_B", "P2", capacity=1, group_id="CORE_B_1"),
        ],
    )

    result = run_fair_cp_sat_solver(
        data,
        seed=1,
        math_fallback_rules=fallback_rules(),
        math_course_ids=math_ids(),
        max_time_seconds_per_stage=2,
    )

    assert result.solve_status == CpSatSolveStatus.INFEASIBLE
    assert result.model_stats.bootstrap_status == CpSatBootstrapStatus.INFEASIBLE


def test_bootstrap_ordinary_student_may_not_have_more_than_one_primary_unmet() -> None:
    data = canonical(
        [("STU_1", 12, 1, False)],
        [request_row("STU_1", "CORE_A"), request_row("STU_1", "CORE_B"), request_row("STU_1", "CORE_C")],
        [
            section_row("SEC_A", "CORE_A", "P1", capacity=1, group_id="CORE_A_1"),
            section_row("SEC_B", "CORE_B", "P2", capacity=1, group_id="CORE_B_1"),
            section_row("SEC_C", "CORE_C", "P3", capacity=1, group_id="CORE_C_1"),
        ],
    )

    result = run_fair_cp_sat_solver(
        data,
        seed=1,
        math_fallback_rules=fallback_rules(),
        math_course_ids=math_ids(),
        max_time_seconds_per_stage=2,
    )

    assert result.solve_status == CpSatSolveStatus.INFEASIBLE
    assert result.model_stats.bootstrap_status == CpSatBootstrapStatus.INFEASIBLE


def test_bootstrap_high_demand_primary_policy_is_hard() -> None:
    student_rows = [(f"STU_{index:03d}", 12, 1, False) for index in range(121)]
    request_rows = [request_row(student_id, "HIGH") for student_id, *_ in student_rows]
    data = canonical(
        student_rows,
        request_rows,
        [section_row("SEC_HIGH", "HIGH", "P1", capacity=120, group_id="HIGH_1")],
    )

    result = run_fair_cp_sat_solver(
        data,
        seed=1,
        math_fallback_rules=fallback_rules(),
        math_course_ids=math_ids(),
        max_time_seconds_per_stage=2,
    )

    assert result.solve_status == CpSatSolveStatus.INFEASIBLE
    assert result.model_stats.bootstrap_status == CpSatBootstrapStatus.INFEASIBLE


def test_high_demand_policy_does_not_trigger_at_exactly_120() -> None:
    student_rows = [(f"STU_{index:03d}", 12, 1, False) for index in range(120)]
    request_rows = [request_row(student_id, "HIGH") for student_id, *_ in student_rows]
    data = canonical(
        student_rows,
        request_rows,
        [section_row("SEC_HIGH", "HIGH", "P1", capacity=119, group_id="HIGH_1")],
    )

    result = run_fair_cp_sat_solver(
        data,
        seed=1,
        math_fallback_rules=fallback_rules(),
        math_course_ids=math_ids(),
        max_time_seconds_per_stage=2,
    )

    assert result.solve_status != CpSatSolveStatus.INFEASIBLE
    assert result.policy_report.high_demand_policy_satisfied is True


def test_bootstrap_infeasible_returns_empty_hard_model_result() -> None:
    data = canonical(
        [("P1", 12, 1, True), ("P2", 12, 1, True)],
        [request_row("P1", "CORE_A"), request_row("P2", "CORE_A")],
        [section_row("SEC_CORE_A", "CORE_A", "P1", capacity=1, group_id="CORE_A_1")],
    )

    result = run_fair_cp_sat_solver(
        data,
        seed=1,
        math_fallback_rules=fallback_rules(),
        math_course_ids=math_ids(),
        max_time_seconds_per_stage=2,
    )

    assert result.solve_status == CpSatSolveStatus.INFEASIBLE
    assert result.model_stats.bootstrap_status == CpSatBootstrapStatus.INFEASIBLE
    assert result.assignments == ()


def test_bootstrap_does_not_harden_math_coverage_or_fallback() -> None:
    data = canonical(
        [("STU_1", 12, 1, False)],
        [request_row("STU_1", "MATH2_3_HA")],
        [section_row("SEC_CORE_A", "CORE_A", "P1", capacity=1, group_id="CORE_A_1")],
    )

    result = run_fair_cp_sat_solver(
        data,
        seed=1,
        math_fallback_rules=(MathFallbackRule("MATH2_3_HA", "MATH2", "mandatory_fallback", True, "test"),),
        math_course_ids=math_ids(),
        max_time_seconds_per_stage=2,
    )

    assert result.model_stats.bootstrap_status == CpSatBootstrapStatus.FEASIBLE_FOUND
    assert result.solve_status == CpSatSolveStatus.OPTIMAL
    assert result.math_policy_report.current_math_coverage_violation_student_ids == ("STU_1",)


def test_bootstrap_dangling_candidate_returns_model_invalid() -> None:
    data = canonical([("STU_1", 12, 1, False)], [request_row("STU_1", "CORE_A")])
    broken = data.__class__(
        students=data.students,
        logical_requests=data.logical_requests,
        logical_sections=data.logical_sections,
        candidate_index={key("STU_1", "CORE_A"): ("MISSING_SECTION",)},
        students_by_id=data.students_by_id,
        requests_by_key=data.requests_by_key,
        logical_sections_by_id=data.logical_sections_by_id,
        courses_by_id=data.courses_by_id,
    )

    result = run_fair_cp_sat_solver(
        broken,
        seed=1,
        math_fallback_rules=fallback_rules(),
        math_course_ids=math_ids(),
        max_time_seconds_per_stage=2,
    )

    assert result.solve_status == CpSatSolveStatus.MODEL_INVALID
    assert result.model_stats.bootstrap_status == CpSatBootstrapStatus.MODEL_INVALID


def test_candidate_section_must_match_request_identity() -> None:
    data = canonical([("STU_1", 12, 1, False)], [request_row("STU_1", "CORE_A")])
    broken = data.__class__(
        students=data.students,
        logical_requests=data.logical_requests,
        logical_sections=data.logical_sections,
        candidate_index={key("STU_1", "CORE_A"): ("CORE_B_1",)},
        students_by_id=data.students_by_id,
        requests_by_key=data.requests_by_key,
        logical_sections_by_id=data.logical_sections_by_id,
        courses_by_id=data.courses_by_id,
    )

    result = run_fair_cp_sat_solver(
        broken,
        seed=1,
        math_fallback_rules=fallback_rules(),
        math_course_ids=math_ids(),
        max_time_seconds_per_stage=2,
    )

    assert result.solve_status == CpSatSolveStatus.MODEL_INVALID
    assert result.model_stats.bootstrap_status == CpSatBootstrapStatus.MODEL_INVALID


def test_bootstrap_unknown_without_incumbent_falls_back_to_core(monkeypatch) -> None:
    def fake_bootstrap(*args, **kwargs):
        return cp_sat_solver_module._BootstrapRun(
            build=None,
            solver=None,
            status=CpSatBootstrapStatus.UNKNOWN_NO_INCUMBENT,
            diagnostic=None,
            selected_keys=(),
            solve_time_seconds=0.0,
            time_to_first_hard_feasible_solution_seconds=None,
            hint_strategy="none",
        )

    monkeypatch.setattr(cp_sat_solver_module, "_run_feasibility_bootstrap", fake_bootstrap)

    result = run_fair_cp_sat_solver(
        canonical([("STU_1", 12, 1, True)], [request_row("STU_1", "CORE_A")]),
        seed=1,
        math_fallback_rules=fallback_rules(),
        math_course_ids=math_ids(),
        max_time_seconds_per_stage=2,
    )

    assert result.solve_status == CpSatSolveStatus.OPTIMAL
    assert result.assignments
    assert result.model_stats.bootstrap_status == CpSatBootstrapStatus.UNKNOWN_NO_INCUMBENT


def test_global_budget_exhaustion_without_incumbent_returns_unknown_and_skips_stages() -> None:
    result = run_fair_cp_sat_solver(
        canonical([("STU_1", 12, 1, False)], [request_row("STU_1", "CORE_A")]),
        seed=1,
        math_fallback_rules=fallback_rules(),
        math_course_ids=math_ids(),
        max_time_seconds_per_stage=2,
        max_total_time_seconds=0.0,
    )

    assert result.solve_status == CpSatSolveStatus.UNKNOWN
    assert result.model_stats.total_budget_exhausted is True
    assert any(item.skipped for item in result.stage_diagnostics)
    assert result.model_stats.skipped_stage_count == sum(1 for item in result.stage_diagnostics if item.skipped)
    assert all(item.status != CpSatSolveStatus.OPTIMAL for item in result.stage_diagnostics if item.skipped)


def test_global_budget_exhaustion_after_core_incumbent_returns_feasible(monkeypatch) -> None:
    original = cp_sat_solver_module._GlobalTimeBudget.effective_limit
    calls = 0

    def fake_effective_limit(self, requested):
        nonlocal calls
        calls += 1
        if calls <= 2:
            return original(self, requested)
        self.exhausted = True
        return 0.0

    monkeypatch.setattr(cp_sat_solver_module._GlobalTimeBudget, "effective_limit", fake_effective_limit)

    result = run_fair_cp_sat_solver(
        canonical([("STU_1", 12, 1, True)], [request_row("STU_1", "CORE_A")]),
        seed=1,
        math_fallback_rules=fallback_rules(),
        math_course_ids=math_ids(),
        max_time_seconds_per_stage=2,
        max_total_time_seconds=None,
    )

    assert result.solve_status == CpSatSolveStatus.FEASIBLE
    assert result.assignments
    assert result.model_stats.total_budget_exhausted is True
    assert any(item.skipped for item in result.stage_diagnostics)
    assert result.model_stats.skipped_stage_count == sum(1 for item in result.stage_diagnostics if item.skipped)
    assert result.model_stats.objective_vector == ((CpSatStageName.MATH_COVERAGE, 0),)


def test_partial_core_objective_values_reflect_incumbent_not_default_zero(monkeypatch) -> None:
    original = cp_sat_solver_module._solve_status
    calls = 0

    def fake_solve_status(raw_status):
        nonlocal calls
        calls += 1
        status = original(raw_status)
        if calls == 1 and status == CpSatSolveStatus.OPTIMAL:
            return CpSatSolveStatus.FEASIBLE
        return status

    monkeypatch.setattr(cp_sat_solver_module, "_solve_status", fake_solve_status)

    result = run_fair_cp_sat_solver(
        canonical([("STU_1", 12, 1, False)], [request_row("STU_1", "NOSEC")]),
        seed=1,
        math_fallback_rules=fallback_rules(),
        math_course_ids=math_ids(),
        max_time_seconds_per_stage=2,
        continue_after_feasible=False,
        use_feasibility_bootstrap=False,
        use_constrained_first_hint=False,
    )

    assert result.solve_status == CpSatSolveStatus.FEASIBLE
    assert result.objective_values.math_coverage_violations == 0
    assert result.objective_values.primary_unmet_count == 1
    assert result.objective_values.primary_unmet_period_units == 1
    assert result.model_stats.objective_vector == ((CpSatStageName.MATH_COVERAGE, 0),)
    assert result.model_stats.skipped_stage_count == sum(1 for item in result.stage_diagnostics if item.skipped)
