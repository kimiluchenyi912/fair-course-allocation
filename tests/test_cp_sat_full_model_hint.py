from __future__ import annotations

from types import SimpleNamespace

import pytest
from ortools.sat.python import cp_model

import src.allocation.constrained_first_baseline as constrained_first_module
from src.allocation.cp_sat_solver import (
    _HintSeed,
    _VariableKey,
    _apply_complete_key_hint,
    _apply_complete_model_hint,
    _audit_complete_hint,
    _build_core_cp_sat_model,
    _build_full_feasibility_cp_sat_model,
    _build_mandatory_fallback_plans,
    _constrained_first_full_hint_seed,
    _convert_fallback_plans,
    _full_model_hint_values,
    _new_solver,
    run_fair_cp_sat_solver,
)
from src.allocation import CpSatStageName, CpSatSolveStatus
from tests.test_cp_sat_solver import canonical, fallback_rules, math_ids, request_row


def _full_fixture():
    return canonical(
        [("STU_1", 12, 5, False)],
        [
            request_row("STU_1", "CORE_A"),
            request_row("STU_1", "CORE_C"),
            request_row("STU_1", "CORE_D"),
            request_row("STU_1", "ALT1"),
            request_row("STU_1", "ALT2"),
        ],
    )


def _full_build(data):
    plans = _convert_fallback_plans(_build_mandatory_fallback_plans(data, fallback_rules()))
    return _build_full_feasibility_cp_sat_model(data, plans, math_ids(), 20260630)


def test_full_model_hint_covers_candidate_and_auxiliary_variables() -> None:
    data = _full_fixture()
    build = _full_build(data)
    seed = _constrained_first_full_hint_seed(data, fallback_rules(), math_ids(), 20260630)
    mapped = tuple(key for key in seed.keys if key in build.assignment_vars)

    values, _variables = _full_model_hint_values(build, mapped)

    assert len(values) == len(build.model.Proto().variables)
    non_boolean_values = {
        index: value for index, value in values.items() if value not in {0, 1}
    }
    logical_counter_indices = {variable.Index() for variable in build.logical_assigned_course_vars.values()}
    logical_counter_indices.add(build.logical_assigned_course_total_var.Index())
    assert set(non_boolean_values) <= logical_counter_indices
    assert values[build.logical_assigned_course_total_var.Index()] == sum(
        values[variable.Index()] for variable in build.logical_assigned_course_vars.values()
    )
    _apply_complete_model_hint(build, mapped)
    hint = build.model.Proto().solution_hint
    assert len(hint.vars) == len(build.model.Proto().variables)


def test_known_complete_full_model_hint_produces_real_incumbent() -> None:
    build = _full_build(_full_fixture())
    seed = _constrained_first_full_hint_seed(_full_fixture(), fallback_rules(), math_ids(), 20260630)
    _apply_complete_model_hint(build, tuple(key for key in seed.keys if key in build.assignment_vars))
    solver = _new_solver(2, 1, False, 20260630)
    solver.parameters.stop_after_first_solution = True

    status = solver.Solve(build.model)

    assert solver.StatusName(status) in {"FEASIBLE", "OPTIMAL"}


def test_hint_audit_reports_unmapped_keys_without_silent_mapping() -> None:
    data = _full_fixture()
    build = _build_core_cp_sat_model(data, (), math_ids(), seed=1)
    valid = next(iter(build.assignment_vars))
    unknown = _VariableKey("primary:STU_1:NOT_IN_MODEL", "MISSING_SECTION")

    audit = _audit_complete_hint(
        build.assignment_vars,
        _HintSeed("test", (valid, unknown), replay_policy_pass=False, violation_students=1),
    )

    assert audit.unknown_or_unmapped_assignments == 1
    assert audit.replay_policy_pass is False
    assert audit.violation_students == 1


def test_duplicate_complete_hint_keys_are_rejected() -> None:
    data = _full_fixture()
    build = _build_core_cp_sat_model(data, (), math_ids(), seed=1)
    valid = next(iter(build.assignment_vars))

    with pytest.raises(ValueError, match="Duplicate"):
        _apply_complete_key_hint(build.model, build.assignment_vars, (valid, valid))


def test_full_model_duplicate_hint_keys_are_rejected() -> None:
    build = _full_build(_full_fixture())
    valid = next(iter(build.assignment_vars))

    with pytest.raises(ValueError, match="Duplicate"):
        _apply_complete_model_hint(build, (valid, valid))


def test_hint_is_guidance_and_solver_can_choose_a_different_assignment() -> None:
    data = canonical(
        [("STU_1", 12, 1, False)],
        [request_row("STU_1", "CORE_A"), request_row("STU_1", "CORE_C")],
    )
    build = _build_core_cp_sat_model(data, (), math_ids(), seed=1)
    key_a = _VariableKey("primary:STU_1:CORE_A", "CORE_A_1")
    key_c = _VariableKey("primary:STU_1:CORE_C", "CORE_C_1")
    build.model.Maximize(build.assignment_vars[key_c])
    _apply_complete_key_hint(build.model, build.assignment_vars, (key_a,))
    solver = _new_solver(2, 1, False, 1)

    status = solver.Solve(build.model)

    assert status in {cp_model.OPTIMAL, cp_model.FEASIBLE}
    assert solver.BooleanValue(build.assignment_vars[key_c]) is True


def test_no_hint_and_full_hint_keep_model_structure_identical() -> None:
    data = _full_fixture()
    first = _full_build(data)
    second = _full_build(data)
    seed = _constrained_first_full_hint_seed(data, fallback_rules(), math_ids(), 20260630)
    _apply_complete_model_hint(second, tuple(key for key in seed.keys if key in second.assignment_vars))

    assert len(first.model.Proto().variables) == len(second.model.Proto().variables)
    assert len(first.model.Proto().constraints) == len(second.model.Proto().constraints)


def test_constrained_first_full_hint_records_policy_failure_and_all_request_kinds(monkeypatch) -> None:
    fake_assignments = tuple(
        SimpleNamespace(
            request_key=request_key,
            linked_section_group_id=section_id,
        )
        for request_key, section_id in (
            ("primary:STU_1:PRIMARY", "PRIMARY_1"),
            ("alternate:STU_1:1:ALT", "ALT_1"),
            ("mandatory_fallback:STU_1:PRIMARY:MATH", "MATH_1"),
        )
    )
    fake = SimpleNamespace(
        algorithm_name="constrained_first_greedy",
        assignments=fake_assignments,
        student_outcomes=(),
    )
    monkeypatch.setattr(constrained_first_module, "run_constrained_first_baseline", lambda *args, **kwargs: fake)

    seed = _constrained_first_full_hint_seed(_full_fixture(), (), (), 1)

    assert {key.request_key.split(":", 1)[0] for key in seed.keys} == {
        "primary",
        "alternate",
        "mandatory_fallback",
    }
    assert seed.replay_policy_pass is True


def test_logical_completion_can_be_disabled_without_changing_hard_stage_order() -> None:
    result = run_fair_cp_sat_solver(
        _full_fixture(),
        seed=20260630,
        math_fallback_rules=fallback_rules(),
        math_course_ids=math_ids(),
        max_time_seconds_per_stage=2,
        use_constrained_first_hint=False,
        logical_schedule_completion_enabled=False,
    )

    assert result.model_stats.logical_schedule_completion_objective_enabled is False
    assert all(item.stage_name != CpSatStageName.LOGICAL_SCHEDULE_COMPLETION for item in result.stage_diagnostics)
    assert result.solve_status in {CpSatSolveStatus.OPTIMAL, CpSatSolveStatus.FEASIBLE}


def test_full_model_seed_metadata_distinguishes_hint_from_solver_repair() -> None:
    result = run_fair_cp_sat_solver(
        _full_fixture(),
        seed=20260630,
        math_fallback_rules=fallback_rules(),
        math_course_ids=math_ids(),
        max_time_seconds_per_stage=2,
    )

    assert result.model_stats.hint_source == "constrained_first_greedy_full"
    assert result.model_stats.hint_coverage_rate == 1.0
    assert result.model_stats.hint_unknown_or_unmapped_assignments == 0
    assert result.model_stats.hint_duplicate_keys == 0
    assert result.model_stats.full_model_seed_policy_pass is not None
