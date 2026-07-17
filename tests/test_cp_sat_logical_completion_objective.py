from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

import src.allocation.cp_sat_solver as cp_sat_solver_module
from src.allocation import (
    AlternateRequestStatus,
    CpSatModelScope,
    CpSatSolveStatus,
    CpSatStageName,
    PrimaryRequestStatus,
    run_fair_cp_sat_solver,
)
from src.allocation.cp_sat_solver import CpSatFinalSchedulePolicyConsistencyError
from tests.test_cp_sat_solver import (
    alt_key,
    canonical,
    fallback_rules,
    gov_request_rows,
    key,
    math_ids,
    outcome,
    request_row,
    run_solver,
    student_outcome,
)


def stage(result, name: CpSatStageName):
    return next(item for item in result.stage_diagnostics if item.stage_name == name)


def assigned_logical_count(result) -> int:
    return sum(
        outcome.assigned_logical_course_count
        if outcome.assigned_logical_course_count is not None
        else len(outcome.assignment_keys)
        for outcome in result.student_outcomes
    )


def test_logical_completion_prefers_more_logical_courses_before_alternate_rank() -> None:
    data = canonical(
        [("STU_1", 12, 2, False)],
        [
            request_row("STU_1", "NOSEC"),
            request_row("STU_1", "DOUBLE", "alternate", 1, "alternate"),
            request_row("STU_1", "CORE_A", "alternate", 2, "alternate"),
            request_row("STU_1", "CORE_C", "alternate", 3, "alternate"),
        ],
    )

    result = run_solver(data)

    assert outcome(result, alt_key("STU_1", 1, "DOUBLE")).status != AlternateRequestStatus.ASSIGNED
    assert outcome(result, alt_key("STU_1", 2, "CORE_A")).status == AlternateRequestStatus.ASSIGNED
    assert outcome(result, alt_key("STU_1", 3, "CORE_C")).status == AlternateRequestStatus.ASSIGNED
    assert result.objective_values.logical_assigned_course_count == 2
    assert student_outcome(result, "STU_1").fully_scheduled is True


def test_primary_assignment_remains_higher_priority_than_logical_completion() -> None:
    data = canonical(
        [("STU_1", 12, 2, False)],
        [
            request_row("STU_1", "DOUBLE"),
            request_row("STU_1", "CORE_A", "alternate", 1, "alternate"),
            request_row("STU_1", "CORE_C", "alternate", 2, "alternate"),
        ],
    )

    result = run_solver(data)

    assert outcome(result, key("STU_1", "DOUBLE")).status == PrimaryRequestStatus.ASSIGNED
    assert outcome(result, alt_key("STU_1", 1, "CORE_A")).status != AlternateRequestStatus.ASSIGNED
    assert outcome(result, alt_key("STU_1", 2, "CORE_C")).status != AlternateRequestStatus.ASSIGNED
    assert result.objective_values.primary_unmet_count == 0
    assert result.objective_values.logical_assigned_course_count == 1


def test_logical_completion_stage_is_fixed_before_alternate_rank_stages() -> None:
    data = canonical(
        [("STU_1", 12, 2, False)],
        [
            request_row("STU_1", "NOSEC"),
            request_row("STU_1", "DOUBLE", "alternate", 1, "alternate"),
            request_row("STU_1", "CORE_A", "alternate", 2, "alternate"),
            request_row("STU_1", "CORE_C", "alternate", 3, "alternate"),
        ],
    )

    result = run_solver(data)
    logical_stage = stage(result, CpSatStageName.LOGICAL_SCHEDULE_COMPLETION)
    rank1_stage = stage(result, CpSatStageName.ALTERNATE_RANK_1)

    assert logical_stage.model_scope == CpSatModelScope.ENRICHMENT
    assert logical_stage.objective_value == 2
    assert dict(rank1_stage.fixed_higher_priority_values)[CpSatStageName.LOGICAL_SCHEDULE_COMPLETION] == 2
    assert result.model_stats.logical_schedule_completion_objective_enabled is True
    assert result.model_stats.logical_schedule_completion_stage_status == logical_stage.status
    assert result.model_stats.logical_schedule_completion_objective_value == 2
    assert result.model_stats.logical_schedule_completion_fixed_value == 2


def test_logical_completion_counts_double_period_course_once() -> None:
    result = run_solver(canonical([("STU_1", 12, 2, False)], [request_row("STU_1", "MATH2_3_HA")]))

    assert outcome(result, key("STU_1", "MATH2_3_HA")).status == PrimaryRequestStatus.ASSIGNED
    assert student_outcome(result, "STU_1").fully_scheduled is True
    assert result.objective_values.logical_assigned_course_count == 1


def test_logical_completion_counts_linked_group_once() -> None:
    result = run_solver(canonical([("STU_1", 12, 1, False)], gov_request_rows("STU_1")))

    assert result.objective_values.logical_assigned_course_count == 1
    assert assigned_logical_count(result) == 1


def test_multiple_candidate_sections_do_not_double_count_logical_request() -> None:
    result = run_solver(canonical([("STU_1", 12, 1, False)], [request_row("STU_1", "CORE_A")]))

    assert result.objective_values.logical_assigned_course_count == 1
    assert len(student_outcome(result, "STU_1").assignment_keys) == 1


def test_logical_objective_proto_uses_bounded_per_student_counters() -> None:
    data = canonical(
        [("STU_1", 12, 2, False), ("STU_2", 12, 1, False)],
        [request_row("STU_1", "CORE_A"), request_row("STU_2", "CORE_C")],
    )
    build = cp_sat_solver_module._build_enrichment_cp_sat_model(data, (), math_ids(), 1, {})
    build.model.Maximize(build.stage_exprs[CpSatStageName.LOGICAL_SCHEDULE_COMPLETION])

    objective = build.model.Proto().objective
    logical_indices = {build.logical_assigned_course_total_var.Index()}

    assert set(objective.vars) == logical_indices
    assert set(objective.coeffs) == {-1}
    assert [
        tuple(build.model.Proto().variables[variable.Index()].domain)
        for variable in build.logical_assigned_course_vars.values()
    ] == [
        (0, 2),
        (0, 1),
    ]
    assert tuple(build.model.Proto().variables[build.logical_assigned_course_total_var.Index()].domain) == (0, 3)
    assert cp_sat_solver_module._logical_completion_upper_bound(build) == 3


def test_logical_completion_bound_rejects_impossible_best_bound() -> None:
    diagnostic = SimpleNamespace(objective_value=17137, best_objective_bound=30107)

    with pytest.raises(cp_sat_solver_module.CpSatStageIncumbentConsistencyError, match="objective bound"):
        cp_sat_solver_module._validate_logical_completion_bound(diagnostic, 17365)


def test_logical_completion_matches_replayed_logical_assignments_with_final_gate() -> None:
    data = canonical(
        [("STU_1", 12, 5, False)],
        [
            request_row("STU_1", "CORE_A"),
            request_row("STU_1", "CORE_C"),
            request_row("STU_1", "CORE_D"),
            request_row("STU_1", "ALT1"),
            request_row("STU_1", "ALT2"),
        ],
    )

    result = run_fair_cp_sat_solver(
        data,
        seed=20260630,
        math_fallback_rules=fallback_rules(),
        math_course_ids=math_ids(),
        max_time_seconds_per_stage=2,
        use_constrained_first_hint=False,
    )

    assert result.solve_status == CpSatSolveStatus.OPTIMAL
    assert result.model_stats.post_solve_policy_gate_pass is True
    assert result.objective_values.logical_assigned_course_count == assigned_logical_count(result)
    assert result.model_stats.logical_schedule_completion_fixed_value == assigned_logical_count(result)


def test_full_model_feasibility_incumbent_is_not_an_objective_stage() -> None:
    data = canonical(
        [("STU_1", 12, 5, False)],
        [
            request_row("STU_1", "CORE_A"),
            request_row("STU_1", "CORE_C"),
            request_row("STU_1", "CORE_D"),
            request_row("STU_1", "ALT1"),
            request_row("STU_1", "ALT2"),
        ],
    )

    result = run_fair_cp_sat_solver(
        data,
        seed=20260630,
        math_fallback_rules=fallback_rules(),
        math_course_ids=math_ids(),
        max_time_seconds_per_stage=2,
        use_constrained_first_hint=False,
    )

    full_stage = stage(result, CpSatStageName.FULL_MODEL_FEASIBILITY_INCUMBENT)
    assert full_stage.model_scope == CpSatModelScope.ENRICHMENT
    assert full_stage.status in {CpSatSolveStatus.OPTIMAL, CpSatSolveStatus.FEASIBLE}
    assert CpSatStageName.FULL_MODEL_FEASIBILITY_INCUMBENT not in dict(result.model_stats.objective_vector)
    assert all(
        CpSatStageName.FULL_MODEL_FEASIBILITY_INCUMBENT not in dict(item.fixed_higher_priority_values)
        for item in result.stage_diagnostics
    )


def test_logical_completion_consistency_mismatch_fails_clearly(monkeypatch) -> None:
    original = cp_sat_solver_module._objective_values

    def fake_objective_values(*args, **kwargs):
        values = original(*args, **kwargs)
        return replace(values, logical_assigned_course_count=values.logical_assigned_course_count + 1)

    monkeypatch.setattr(cp_sat_solver_module, "_objective_values", fake_objective_values)
    data = canonical(
        [("STU_1", 12, 5, False)],
        [
            request_row("STU_1", "CORE_A"),
            request_row("STU_1", "CORE_C"),
            request_row("STU_1", "CORE_D"),
            request_row("STU_1", "ALT1"),
            request_row("STU_1", "ALT2"),
        ],
    )

    with pytest.raises(CpSatFinalSchedulePolicyConsistencyError, match="logical completion objective"):
        run_fair_cp_sat_solver(
            data,
            seed=20260630,
            math_fallback_rules=fallback_rules(),
            math_course_ids=math_ids(),
            max_time_seconds_per_stage=2,
            use_constrained_first_hint=False,
        )


def test_feasible_logical_completion_stage_is_not_reported_as_optimal(monkeypatch) -> None:
    original = cp_sat_solver_module._solve_status
    call_count = 0

    def fake_solve_status(raw_status):
        nonlocal call_count
        call_count += 1
        status = original(raw_status)
        if call_count == 5 and status == CpSatSolveStatus.OPTIMAL:
            return CpSatSolveStatus.FEASIBLE
        return status

    monkeypatch.setattr(cp_sat_solver_module, "_solve_status", fake_solve_status)

    result = run_solver(
        canonical(
            [("STU_1", 12, 2, False)],
            [
                request_row("STU_1", "NOSEC"),
                request_row("STU_1", "CORE_A", "alternate", 1, "alternate"),
                request_row("STU_1", "CORE_C", "alternate", 2, "alternate"),
            ],
        )
    )

    logical_stage = stage(result, CpSatStageName.LOGICAL_SCHEDULE_COMPLETION)
    assert logical_stage.status == CpSatSolveStatus.FEASIBLE
    assert logical_stage.optimum_proven is False
    assert result.solve_status == CpSatSolveStatus.FEASIBLE
    assert result.lexicographic_optimality_proven is False


def test_incomplete_core_objectives_do_not_export_partial_core_as_final_schedule(monkeypatch) -> None:
    original = cp_sat_solver_module._solve_status
    call_count = 0

    def fake_solve_status(raw_status):
        nonlocal call_count
        call_count += 1
        status = original(raw_status)
        if call_count == 3 and status == CpSatSolveStatus.OPTIMAL:
            return CpSatSolveStatus.FEASIBLE
        return status

    monkeypatch.setattr(cp_sat_solver_module, "_solve_status", fake_solve_status)
    data = canonical(
        [("STU_1", 12, 5, False)],
        [
            request_row("STU_1", "CORE_A"),
            request_row("STU_1", "CORE_C"),
            request_row("STU_1", "CORE_D"),
            request_row("STU_1", "ALT1"),
            request_row("STU_1", "ALT2"),
        ],
    )

    result = run_fair_cp_sat_solver(
        data,
        seed=20260630,
        math_fallback_rules=fallback_rules(),
        math_course_ids=math_ids(),
        max_time_seconds_per_stage=2,
        continue_after_feasible=False,
        use_constrained_first_hint=False,
    )

    assert result.solve_status == CpSatSolveStatus.UNKNOWN
    assert result.assignments == ()
    assert result.request_outcomes == ()
    assert result.model_stats.post_solve_policy_gate_pass is None
    assert stage(result, CpSatStageName.LOGICAL_SCHEDULE_COMPLETION).skipped is True
