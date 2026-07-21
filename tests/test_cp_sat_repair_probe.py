from __future__ import annotations

import json

import pytest

from src.allocation import CpSatSolveStatus, CpSatStageName, run_fair_cp_sat_solver
from src.allocation.cp_sat_solver import (
    _VariableKey,
    _assignment_distance_stats,
    _build_full_feasibility_cp_sat_model,
    _convert_fallback_plans,
    _hamming_distance_expression,
    _mapped_hint_keys,
)
from src.allocation.random_baseline import _build_mandatory_fallback_plans
from src.cp_sat_robustness_runner import (
    CpSatEvaluationError,
    _result_metrics,
    load_repair_probe_manifest,
)

from tests.test_cp_sat_cold_start_recovery import (
    _complete_fixture,
    fallback_rules,
    math_ids,
)


def _probe_result():
    return run_fair_cp_sat_solver(
        _complete_fixture(),
        seed=20260630,
        math_fallback_rules=fallback_rules(),
        math_course_ids=math_ids(),
        max_time_seconds_per_stage=1,
        bootstrap_time_seconds=0,
        max_total_time_seconds=3,
        num_search_workers=1,
        use_feasibility_bootstrap=False,
        use_constrained_first_hint=False,
        logical_schedule_completion_enabled=False,
        internal_feasibility_hint_strategy="constrained_first",
        internal_repair_time_seconds=1,
        internal_repair_objective_strategy="hamming_to_constrained_first",
        stop_after_first_valid_solution=True,
    )


def test_hamming_strategy_is_explicit_and_default_is_none() -> None:
    result = _probe_result()
    assert result.model_stats.internal_repair_objective_strategy == "hamming_to_constrained_first"
    with pytest.raises(ValueError, match="internal repair objective"):
        run_fair_cp_sat_solver(
            _complete_fixture(),
            seed=1,
            internal_feasibility_hint_strategy="none",
            internal_repair_objective_strategy="hamming_to_constrained_first",
            max_time_seconds_per_stage=1,
        )


def test_hamming_objective_uses_candidate_variables_only() -> None:
    allocation_input = _complete_fixture()
    plans = _convert_fallback_plans(_build_mandatory_fallback_plans(allocation_input, fallback_rules()))
    build = _build_full_feasibility_cp_sat_model(allocation_input, plans, math_ids(), 20260630)
    selected = tuple(sorted(build.assignment_vars, key=lambda item: (item.request_key, item.section_id))[:1])
    build.model.Minimize(_hamming_distance_expression(build, selected))
    objective = build.model.Proto().objective
    assert len(objective.vars) == len(build.assignment_vars)
    assert set(objective.vars) == {variable.Index() for variable in build.assignment_vars.values()}
    assert set(objective.coeffs) <= {-1, 1}


def test_hamming_distance_covers_assigned_and_unassigned_candidates_once() -> None:
    result = _probe_result()
    stats = result.model_stats
    assert stats.internal_repair_hamming_distance == (
        stats.internal_repair_greedy_assignments_removed
        + stats.internal_repair_new_assignments_added
    )
    assert stats.internal_hint_candidate_variables == stats.internal_hint_candidate_variables_hinted
    assert stats.internal_hint_candidate_coverage_rate == 1.0


def test_assignment_distance_stats_resolves_variable_key_request_id() -> None:
    allocation_input = _complete_fixture()
    request = allocation_input.logical_requests[0]
    section_id = allocation_input.candidate_index[request.request_key][0]
    key = _VariableKey(request.request_key, section_id)

    stats = _assignment_distance_stats(
        allocation_input,
        (),
        (key,),
        {request.request_key: request},
    )

    assert stats["changed_students"] == 1
    assert stats["changed_requests"] == 1
    assert stats["changed_sections"] == 1


def test_auxiliary_variables_are_not_in_distance_universe() -> None:
    result = _probe_result()
    stats = result.model_stats
    assert stats.internal_repair_variable_hash
    assert stats.internal_repair_domain_hash
    assert stats.internal_repair_constraint_hash
    assert stats.internal_repair_candidate_mapping_hash
    assert stats.internal_hint_auxiliary_variables_hinted >= 0


def test_model_invariance_strips_distance_and_hint() -> None:
    result = _probe_result()
    stats = result.model_stats
    assert stats.model_invariance_distance_stripped_equal is True
    assert stats.model_invariance_equal is True
    assert stats.model_invariance_without_distance_hash
    assert stats.model_invariance_distance_stripped_hash


def test_probe_has_only_internal_repair_stage_and_no_external_seed() -> None:
    result = _probe_result()
    assert [item.stage_name for item in result.stage_diagnostics] == [CpSatStageName.INTERNAL_REPAIR_FEASIBILITY]
    assert result.model_stats.bootstrap_status.value == "DISABLED"
    assert result.model_stats.external_hint_used is False
    assert result.model_stats.initial_solution_seed_enabled is False


def test_first_solver_response_is_final_source_for_small_probe() -> None:
    result = _probe_result()
    assert result.solve_status in {CpSatSolveStatus.FEASIBLE, CpSatSolveStatus.OPTIMAL}
    assert result.student_outcomes
    diagnostic = result.stage_diagnostics[0]
    assert diagnostic.response_proto_hash
    assert result.model_stats.internal_repair_response_proto_hash == diagnostic.response_proto_hash
    assert result.model_stats.internal_repair_time_to_first_solution_seconds is not None


def test_repair_hint_is_not_a_hard_assignment_constraint() -> None:
    result = _probe_result()
    assert result.model_stats.internal_hint_policy_violation_count is not None
    assert result.model_stats.internal_repair_hamming_distance is not None
    assert result.model_stats.internal_repair_validation_failure == ""


def test_result_metrics_reads_final_policy_summary_from_repair_result() -> None:
    allocation_input = _complete_fixture()
    result = _probe_result()

    metrics = _result_metrics(result, allocation_input)

    assert metrics["final_schedule_policy_pass"] is True
    assert metrics["ordinary_policy_violations"] == 0
    assert metrics["protected_policy_violations"] == 0
    assert metrics["high_demand_violations"] == 0


def test_probe_manifest_is_stable_reference_only() -> None:
    payload = load_repair_probe_manifest()
    assert payload["scenario_groups"] == {"normal": ["normal_dev_reference_2026"]}
    config = payload["solver_configuration"]
    assert config["solver_seed"] == 20260630
    assert config["workers"] == 1
    assert config["total_time_limit_seconds"] == 300
    assert config["external_persisted_seed"] is False
    assert config["internal_repair_objective_strategy"] == "hamming_to_constrained_first"
    assert config["stop_after_first_valid_solution"] is True


def test_probe_manifest_rejects_nonstable_or_holdout_scenario(tmp_path) -> None:
    payload = load_repair_probe_manifest()
    payload["scenarios"] = [{**payload["scenarios"][0], "scenario_id": "holdout_bad"}]
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CpSatEvaluationError):
        load_repair_probe_manifest(path)


def test_probe_manifest_permits_tuning_only_for_this_probe() -> None:
    payload = load_repair_probe_manifest()
    assert payload["tuning_allowed"] is True
    assert payload["holdout_execution_allowed"] is False
    assert payload["solver_configuration"]["stage_order"] == ["internal_repair_feasibility"]


def test_unknown_status_semantics_are_not_collapsed() -> None:
    from src.allocation import CpSatSolveStatus

    assert CpSatSolveStatus.UNKNOWN.value == "UNKNOWN"
    assert CpSatSolveStatus.FEASIBLE.value == "FEASIBLE"
    assert CpSatSolveStatus.OPTIMAL.value == "OPTIMAL"
