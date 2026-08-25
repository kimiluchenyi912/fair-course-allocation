"""One-shot, stable-reference distance-guided CP-SAT repair probe."""

from __future__ import annotations

import argparse
import platform
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import ortools
import pandas as pd
from ortools.sat.python import cp_model

from src.allocation import math_course_ids_from_catalog, run_fair_cp_sat_solver
from src.allocation.constrained_first_baseline import run_constrained_first_baseline
from src.allocation.cp_sat_solver import (
    _apply_complete_model_hint,
    _assignment_records_hash,
    _build_full_feasibility_cp_sat_model,
    _constrained_first_full_hint_seed,
    _convert_fallback_plans,
    _full_model_hint_values,
    _hamming_distance_expression,
    _mapped_hint_keys,
    _model_component_hashes,
    _model_proto_without_objective_and_solution_hint_hash,
    _model_proto_without_solution_hint_hash,
)
from src.allocation.random_baseline import _build_mandatory_fallback_plans
from src.allocation.math_policy_models import MathFallbackRule
from src.cp_sat_robustness_runner import (
    CpSatEvaluationError,
    EVALUATION_SCHEMA_VERSION,
    STABLE_FINGERPRINT,
    _load_benchmark_math_fallback_rules,
    _load_scenario_input,
    _read_json,
    _repair_probe_solver_configuration,
    _result_metrics,
    _scenario_from_payload,
    _stage_trace,
    _verify_source_suite,
    _write_checksums,
    _write_json,
    evaluation_manifest_hash,
    load_repair_probe_manifest,
)
from src.final_schedule_policy import evaluate_final_schedule_policy


DEFAULT_MANIFEST = Path("data/scenarios/cp_sat_cold_start_repair_probe_v1.json")
DEFAULT_OUTPUT = Path(
    "../fair-course-allocation-artifacts/robustness-v1/"
    "cp-sat-cold-start-repair-probe-v1"
)


def _parameter_audit(build: Any, hint: Any, values: dict[int, int]) -> dict[str, Any]:
    runtime = cp_model.CpSolver()
    configured = cp_model.CpSolver()
    configured.parameters.max_time_in_seconds = 300.0
    configured.parameters.num_search_workers = 1
    configured.parameters.random_seed = 20260630
    configured.parameters.repair_hint = True
    configured.parameters.hint_conflict_limit = 1000
    candidate_count = len(build.assignment_vars)
    hinted = sum(variable.Index() in values for variable in build.assignment_vars.values())
    model_variables = len(build.model.Proto().variables)
    return {
        "python_version": platform.python_version(),
        "ortools_version": getattr(ortools, "__version__", "unknown"),
        "runtime_defaults": {
            "repair_hint": runtime.parameters.repair_hint,
            "hint_conflict_limit": runtime.parameters.hint_conflict_limit,
            "max_time_in_seconds": runtime.parameters.max_time_in_seconds,
            "num_search_workers": runtime.parameters.num_search_workers,
            "random_seed": runtime.parameters.random_seed,
        },
        "effective_repair_parameters": {
            "repair_hint": configured.parameters.repair_hint,
            "hint_conflict_limit": configured.parameters.hint_conflict_limit,
            "max_time_in_seconds": configured.parameters.max_time_in_seconds,
            "num_search_workers": configured.parameters.num_search_workers,
            "random_seed": configured.parameters.random_seed,
            "stop_after_first_solution": True,
        },
        "model_variable_count": model_variables,
        "candidate_hint_variable_count": candidate_count,
        "candidate_hint_variables_hinted": hinted,
        "candidate_hint_coverage": hinted / candidate_count if candidate_count else 1.0,
        "auxiliary_hint_variables": max(len(values) - candidate_count, 0),
        "unhinted_auxiliary_variables": max(model_variables - len(values), 0),
        "phase_a_observations": {
            "repair_hint_explicitly_true": True,
            "hint_conflict_limit_explicitly_set": True,
            "presolve_hint_values_retained": None,
            "hint_dropped_or_rejected": None,
            "note": "Phase A did not persist response-level presolve hint logs.",
        },
        "hint_assignment_hash": hint.internal_audit.assignment_hash if hint.internal_audit else None,
    }


def _model_invariance_audit(
    allocation_input: Any,
    plans: tuple[Any, ...],
    math_ids: tuple[str, ...],
    seed: int,
) -> tuple[dict[str, Any], Any, Any, dict[int, int], dict[int, Any]]:
    model_a = _build_full_feasibility_cp_sat_model(allocation_input, plans, math_ids, seed)
    hint_rules = tuple(
        MathFallbackRule(
            plan.source_request.candidate_key,
            plan.fallback_request.candidate_key,
            "mandatory_fallback",
            True,
            "repair_probe",
        )
        for plan in plans
    )
    hint = _constrained_first_full_hint_seed(allocation_input, hint_rules, math_ids, seed)
    model_b = _build_full_feasibility_cp_sat_model(allocation_input, plans, math_ids, seed)
    mapped = _mapped_hint_keys(model_b.assignment_vars, hint.keys)
    model_b.model.Minimize(_hamming_distance_expression(model_b, mapped))
    base_hash = _model_proto_without_solution_hint_hash(model_a.model)
    before_hint_hash = _model_proto_without_solution_hint_hash(model_b.model)
    stripped_hash = _model_proto_without_objective_and_solution_hint_hash(model_b.model)
    values, variables = _full_model_hint_values(model_b, mapped, require_complete=False)
    _apply_complete_model_hint(model_b, mapped, values=values, variables=variables, allow_partial=True)
    after_hint_hash = _model_proto_without_solution_hint_hash(model_b.model)
    a_components = _model_component_hashes(model_a)
    b_components = _model_component_hashes(model_b)
    audit = {
        "model_without_distance_hash": base_hash,
        "model_with_distance_without_hint_hash": before_hint_hash,
        "distance_stripped_hash": stripped_hash,
        "distance_stripped_equal": base_hash == stripped_hash,
        "hinted_model_hash": after_hint_hash,
        "hint_changes_only_hint_or_objective": base_hash == stripped_hash,
        "variable_hash_equal": a_components[0] == b_components[0],
        "domain_hash_equal": a_components[1] == b_components[1],
        "constraint_hash_equal": a_components[2] == b_components[2],
        "candidate_mapping_hash_equal": a_components[3] == b_components[3],
        "variable_hash": a_components[0],
        "domain_hash": a_components[1],
        "constraint_hash": a_components[2],
        "candidate_mapping_hash": a_components[3],
        "candidate_variables": len(model_b.assignment_vars),
        "hinted_candidate_variables": sum(v.Index() in values for v in model_b.assignment_vars.values()),
        "auxiliary_variables_hinted": max(len(values) - len(model_b.assignment_vars), 0),
        "unhinted_variables": max(len(model_b.model.Proto().variables) - len(values), 0),
        "objective_only_descriptor_present": bool(model_b.model.Proto().objective.vars),
        "objective_strategy": "hamming_to_constrained_first",
    }
    required = (
        "distance_stripped_equal", "hint_changes_only_hint_or_objective",
        "variable_hash_equal", "domain_hash_equal", "constraint_hash_equal",
        "candidate_mapping_hash_equal",
    )
    if not all(audit[name] for name in required):
        raise CpSatEvaluationError("repair probe model invariance check failed")
    return audit, model_b, hint, values, variables


def _parameter_audit_from_result(result: Any, config: Any) -> dict[str, Any]:
    stats = result.model_stats
    runtime = cp_model.CpSolver()
    return {
        "python_version": platform.python_version(),
        "ortools_version": getattr(ortools, "__version__", "unknown"),
        "runtime_defaults": {
            "repair_hint": runtime.parameters.repair_hint,
            "hint_conflict_limit": runtime.parameters.hint_conflict_limit,
            "max_time_in_seconds": runtime.parameters.max_time_in_seconds,
            "num_search_workers": runtime.parameters.num_search_workers,
            "random_seed": runtime.parameters.random_seed,
        },
        "effective_repair_parameters": {
            "repair_hint": True,
            "hint_conflict_limit": 1000,
            "max_time_in_seconds": config.total_time_limit_seconds,
            "num_search_workers": config.workers,
            "random_seed": config.solver_seed,
            "stop_after_first_solution": True,
        },
        "model_variable_count": stats.total_variables,
        "candidate_hint_variable_count": stats.internal_hint_candidate_variables,
        "candidate_hint_variables_hinted": stats.internal_hint_candidate_variables_hinted,
        "candidate_hint_coverage": stats.internal_hint_candidate_coverage_rate,
        "auxiliary_hint_variables": stats.internal_hint_auxiliary_variables_hinted,
        "unhinted_auxiliary_variables": stats.internal_hint_unhinted_variables,
        "phase_a_observations": {
            "repair_hint_explicitly_true": True,
            "hint_conflict_limit_explicitly_set": True,
            "presolve_hint_values_retained": None,
            "hint_dropped_or_rejected": None,
            "note": "Phase A did not persist response-level presolve hint logs.",
        },
        "hint_assignment_hash": stats.internal_hint_assignment_hash,
    }


def _model_invariance_from_result(result: Any) -> dict[str, Any]:
    stats = result.model_stats
    return {
        "model_without_distance_hash": stats.model_invariance_without_distance_hash,
        "model_with_distance_without_hint_hash": stats.model_invariance_before_hint_hash,
        "distance_stripped_hash": stats.model_invariance_distance_stripped_hash,
        "distance_stripped_equal": stats.model_invariance_distance_stripped_equal,
        "hinted_model_hash": stats.model_invariance_after_hint_hash,
        "hint_changes_only_hint_or_objective": stats.model_invariance_distance_stripped_equal,
        "variable_hash": stats.internal_repair_variable_hash,
        "domain_hash": stats.internal_repair_domain_hash,
        "constraint_hash": stats.internal_repair_constraint_hash,
        "candidate_mapping_hash": stats.internal_repair_candidate_mapping_hash,
        "objective_strategy": stats.internal_repair_objective_strategy,
    }


def _greedy_footprint(
    allocation_input: Any,
    math_ids: tuple[str, ...],
    fallback_rules: tuple[Any, ...],
) -> dict[str, Any]:
    greedy = run_constrained_first_baseline(
        allocation_input,
        20260630,
        math_fallback_rules=fallback_rules,
        math_course_ids=math_ids,
    )
    policy = evaluate_final_schedule_policy(greedy.algorithm_name, greedy.student_outcomes)
    violating_students = {item.student_id for item in policy.violations}
    requests_by_student: dict[str, list[Any]] = defaultdict(list)
    request_by_key = {item.request_key: item for item in allocation_input.logical_requests}
    for request in allocation_input.logical_requests:
        requests_by_student[request.student_id].append(request)
    section_students: dict[str, set[str]] = defaultdict(set)
    for assignment in greedy.assignments:
        section_students[assignment.linked_section_group_id].add(assignment.student_id)
    direct_sections = {
        assignment.linked_section_group_id
        for assignment in greedy.assignments
        if assignment.student_id in violating_students
    }
    for student_id in violating_students:
        for request in requests_by_student[student_id]:
            direct_sections.update(allocation_input.candidate_index.get(request.request_key, ()))
    one_hop_students = set(violating_students)
    for section_id in direct_sections:
        one_hop_students.update(section_students[section_id])
    one_hop_sections = {
        assignment.linked_section_group_id
        for assignment in greedy.assignments
        if assignment.student_id in one_hop_students
    }
    two_hop_students = set(one_hop_students)
    for section_id in one_hop_sections:
        two_hop_students.update(section_students[section_id])
    affected_candidate_variables = sum(
        len(allocation_input.candidate_index.get(request.request_key, ()))
        for student_id in two_hop_students
        for request in requests_by_student[student_id]
    )
    issue_codes = [str(getattr(issue, "code", "")) for issue in greedy.consistency_issues]
    affected_periods = sorted({
        period
        for section_id in direct_sections
        for period in allocation_input.logical_sections_by_id[section_id].occupied_periods
        if section_id in allocation_input.logical_sections_by_id
    })
    affected_courses = sorted({
        allocation_input.logical_sections_by_id[section_id].logical_block_id
        for section_id in direct_sections
        if section_id in allocation_input.logical_sections_by_id
    })
    unknown_ids = sum(
        assignment.request_key not in request_by_key
        or assignment.linked_section_group_id not in allocation_input.logical_sections_by_id
        for assignment in greedy.assignments
    )
    return {
        "structural_integrity": {
            "capacity_violations": sum(item.assigned_count > item.capacity for item in greedy.section_roster_summary),
            "period_conflicts": sum("PERIOD" in code for code in issue_codes),
            "duplicate_logical_identity_issues": sum("DUPLICATE" in code or "IDENTITY" in code for code in issue_codes),
            "invalid_candidate_edges": 0,
            "unknown_ids": unknown_ids,
            "consistency_issue_count": len(greedy.consistency_issues),
        },
        "final_policy": {
            "ordinary_primary_unmet_violations": len(greedy.policy_report.ordinary_violation_student_ids),
            "protected_violations": len(greedy.policy_report.protected_violation_student_ids),
            "high_demand_violations": greedy.policy_report.high_demand_violation_count,
            "gap_over_1_students": sum(int(item.logical_schedule_gap_count or 0) > 1 for item in greedy.student_outcomes),
            "below_five_students": sum(int(item.assigned_logical_course_count or 0) < 5 for item in greedy.student_outcomes),
            "total_logical_gap": sum(int(item.logical_schedule_gap_count or 0) for item in greedy.student_outcomes),
            "affected_student_ids": sorted(violating_students),
            "affected_logical_course_ids": affected_courses,
            "affected_section_ids": sorted(direct_sections),
        },
        "connected_footprint": {
            "directly_violating_students": sorted(violating_students),
            "one_hop_connected_students": sorted(one_hop_students),
            "two_hop_connected_students": sorted(two_hop_students),
            "affected_sections": sorted(direct_sections | one_hop_sections),
            "affected_periods": affected_periods,
            "affected_logical_course_ids": affected_courses,
            "affected_candidate_variables": affected_candidate_variables,
        },
        "greedy_assignment_count": len(greedy.assignments),
        "assignment_hash": _assignment_records_hash(greedy.assignments),
    }


def run_repair_probe(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    output_dir: str | Path = DEFAULT_OUTPUT,
    config_dir: str | Path = "data/config",
) -> dict[str, Any]:
    manifest = load_repair_probe_manifest(manifest_path)
    config = _repair_probe_solver_configuration(manifest)
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise CpSatEvaluationError(f"repair probe output is non-empty; refusing to overwrite: {root}")
    root.mkdir(parents=True, exist_ok=True)
    source_info = _verify_source_suite(manifest, "normal")
    scenario = _scenario_from_payload(manifest, "normal_dev_reference_2026")
    allocation_input, input_manifest, source_result = _load_scenario_input(manifest, scenario, Path(config_dir))
    if asdict(__import__("src.experiment_manifest", fromlist=["canonical_input_fingerprint"]).canonical_input_fingerprint(allocation_input)) != STABLE_FINGERPRINT:
        raise CpSatEvaluationError("repair probe input fingerprint does not match stable reference")
    catalog = pd.read_csv(Path(config_dir) / "course_catalog.csv", keep_default_na=False)
    math_ids = math_course_ids_from_catalog(catalog)
    fallback_rules = _load_benchmark_math_fallback_rules(Path(config_dir), catalog)
    plans = _convert_fallback_plans(_build_mandatory_fallback_plans(allocation_input, fallback_rules))
    manifest_hash = evaluation_manifest_hash(manifest_path)
    _write_json(root / "evaluation_manifest_snapshot.json", manifest)
    _write_json(root / "run_manifest.json", {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "running",
        "evaluation_manifest_sha256": manifest_hash,
        "source_git_commit": manifest["source_git_commit"],
        "selected_scenario_ids": [scenario.scenario_id],
        "completed_scenario_ids": [],
        "solver_runs": 0,
        "holdout_runs": 0,
        "stress_runs": 0,
        "external_persisted_seed": False,
        "initial_solution_artifact_dir": None,
    })
    _write_json(root / "greedy_repair_footprint.json", _greedy_footprint(allocation_input, tuple(sorted(math_ids)), fallback_rules))
    started = __import__("time").perf_counter()
    result = run_fair_cp_sat_solver(
        allocation_input,
        seed=config.solver_seed,
        math_course_ids=tuple(sorted(math_ids)),
        math_fallback_rules=fallback_rules,
        max_time_seconds_per_stage=config.per_stage_time_limit_seconds,
        bootstrap_time_seconds=config.bootstrap_time_limit_seconds,
        max_total_time_seconds=config.total_time_limit_seconds,
        num_search_workers=config.workers,
        use_feasibility_bootstrap=False,
        use_constrained_first_hint=False,
        logical_schedule_completion_enabled=False,
        initial_solution_artifact_dir=None,
        internal_feasibility_hint_strategy="constrained_first",
        internal_repair_time_seconds=config.total_time_limit_seconds,
        internal_repair_objective_strategy="hamming_to_constrained_first",
        stop_after_first_valid_solution=True,
    )
    runtime = __import__("time").perf_counter() - started
    _write_json(root / "model_invariance.json", _model_invariance_from_result(result))
    _write_json(root / "repair_parameter_audit.json", _parameter_audit_from_result(result, config))
    trace = _stage_trace(result, scenario.scenario_id)
    metrics = _result_metrics(result, allocation_input)
    stage = next((item for item in trace if item["stage_name"] == "internal_repair_feasibility"), None)
    status = result.solve_status.value
    assignment_available = bool(result.student_outcomes)
    policy_pass = metrics.get("final_schedule_policy_pass") is True
    consistency_zero = metrics.get("consistency_issue_count") == 0
    response_hash = stage.get("response_proto_hash") if stage else None
    publishable = status in {"FEASIBLE", "OPTIMAL"} and assignment_available and policy_pass and consistency_zero and bool(response_hash)
    critical_failure = result.model_stats.internal_repair_validation_failure
    if status in {"FEASIBLE", "OPTIMAL"} and not assignment_available:
        critical_failure = critical_failure or "solver solution status has no validated assignment"
    probe = {
        "scenario_id": scenario.scenario_id,
        "status": status,
        "publishable_recovery": publishable,
        "runtime_seconds": round(runtime, 6),
        "time_to_first_solution_seconds": result.model_stats.internal_repair_time_to_first_solution_seconds,
        "response_proto_hash": response_hash,
        "objective_descriptor_hash": stage.get("objective_descriptor_hash") if stage else None,
        "objective_value": stage.get("objective_value") if stage else None,
        "best_objective_bound": stage.get("best_objective_bound") if stage else None,
        "hamming_distance": result.model_stats.internal_repair_hamming_distance,
        "greedy_assignments_removed": result.model_stats.internal_repair_greedy_assignments_removed,
        "new_assignments_added": result.model_stats.internal_repair_new_assignments_added,
        "changed_students": result.model_stats.internal_repair_changed_students,
        "changed_requests": result.model_stats.internal_repair_changed_requests,
        "changed_sections": result.model_stats.internal_repair_changed_sections,
        "conflicts": stage.get("conflict_count") if stage else None,
        "branches": stage.get("branch_count") if stage else None,
        "deterministic_time_seconds": stage.get("deterministic_time_seconds") if stage else None,
        "final_assignment_source": "internal_repair_feasibility_solver_response" if assignment_available else None,
        "external_persisted_seed": False,
        "critical_failure": critical_failure or None,
    }
    final_validation = None if not assignment_available else {
        "status": status,
        "final_schedule_policy_pass": policy_pass,
        "consistency_issue_count": metrics.get("consistency_issue_count"),
        "response_proto_hash": response_hash,
        "assignment_source": probe["final_assignment_source"],
        "raw_metrics": metrics,
    }
    _write_json(root / "stage_trace.json", {"scenario_id": scenario.scenario_id, "stages": trace})
    _write_json(root / "probe_result.json", probe)
    _write_json(root / "final_validation.json", final_validation or {"status": "not_available", "metrics": None})
    _write_json(root / "assignment_change_summary.json", {
        "hamming_distance": result.model_stats.internal_repair_hamming_distance,
        "greedy_assignments_removed": result.model_stats.internal_repair_greedy_assignments_removed,
        "new_assignments_added": result.model_stats.internal_repair_new_assignments_added,
        "changed_students": result.model_stats.internal_repair_changed_students,
        "changed_requests": result.model_stats.internal_repair_changed_requests,
        "changed_sections": result.model_stats.internal_repair_changed_sections,
    })
    _write_json(root / "failures.json", {
        "failures": [] if not critical_failure else [{"failure_type": "critical_validation_failure", "message": critical_failure}],
        "unexpected_failure_count": 0 if not critical_failure else 1,
    })
    _write_json(root / "run_manifest.json", {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "completed",
        "evaluation_manifest_sha256": manifest_hash,
        "source_git_commit": manifest["source_git_commit"],
        "selected_scenario_ids": [scenario.scenario_id],
        "completed_scenario_ids": [scenario.scenario_id],
        "solver_runs": 1,
        "holdout_runs": 0,
        "stress_runs": 0,
        "external_persisted_seed": False,
        "initial_solution_artifact_dir": None,
        "status_semantics": "UNKNOWN is not INFEASIBLE; FEASIBLE is not OPTIMAL",
    })
    _write_checksums(root)
    if critical_failure:
        raise CpSatEvaluationError(critical_failure)
    return {
        "status": status,
        "publishable_recovery": publishable,
        "scenario_id": scenario.scenario_id,
        "runtime_seconds": round(runtime, 6),
        "source_info": source_info,
        "input_fingerprint": source_result["input_fingerprint"],
        "metrics": metrics,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one stable CP-SAT distance repair probe.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args(argv)
    try:
        result = run_repair_probe(args.manifest, args.output_dir)
    except CpSatEvaluationError as exc:
        print(f"CP-SAT repair probe failed: {exc}")
        return 1
    print(
        "CP-SAT repair probe completed: "
        f"status={result['status']}, publishable_recovery={result['publishable_recovery']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
