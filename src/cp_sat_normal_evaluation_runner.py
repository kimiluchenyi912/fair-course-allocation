"""Cold-Start Feasibility Recovery v1 -- Phase C.

Frozen 12-normal-scenario development evaluation of the distance-guided
full-hard-model repair method (constrained-first internal hint, unweighted
Hamming objective, stop after the first solver solution).

This module never re-implements CP-SAT model construction or repair logic:
it calls ``run_fair_cp_sat_solver`` directly, exactly as
``src/cp_sat_repair_probe.py`` does for the single stable-reference scenario.
The stable reference itself is never re-solved here -- its already-completed
result is imported from the frozen ``cp-sat-cold-start-repair-probe-v1``
artifact once solver configuration and input provenance are verified to
match.
"""

from __future__ import annotations

import argparse
import csv
import tempfile
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.allocation import math_course_ids_from_catalog, run_fair_cp_sat_solver
from src.allocation.constrained_first_baseline import run_constrained_first_baseline
from src.experiment_manifest import canonical_input_fingerprint
from src.final_schedule_policy import evaluate_final_schedule_policy
from src.cp_sat_robustness_runner import (
    EVALUATION_SCHEMA_VERSION,
    GREEDY_ALGORITHMS,
    STABLE_FINGERPRINT,
    CpSatEvaluationCorrectnessError,
    CpSatEvaluationError,
    EvaluationScenario,
    _assignment_hash,
    _grade_rows,
    _greedy_rows,
    _jsonable,
    _json_hash,
    _load_benchmark_math_fallback_rules,
    _load_scenario_input,
    _paired_rows,
    _read_completed_summary,
    _read_json,
    _repair_probe_solver_configuration,
    _result_metrics,
    _scenario_from_payload,
    _sha256_file,
    _stage_trace,
    _validate_objective_bounds,
    _validate_result,
    _verify_sha256_manifest,
    _verify_source_suite,
    _write_checksums,
    _write_csv,
    _write_json,
    evaluation_manifest_hash,
)

DEFAULT_MANIFEST = Path("data/scenarios/cp_sat_cold_start_normal_evaluation_v1.json")
DEFAULT_OUTPUT = Path(
    "../fair-course-allocation-artifacts/robustness-v1/cp-sat-cold-start-normal-development-v1"
)
DEFAULT_STABLE_PROBE_ARTIFACT = Path(
    "../fair-course-allocation-artifacts/robustness-v1/cp-sat-cold-start-repair-probe-v1"
)
STABLE_SCENARIO_ID = "normal_dev_reference_2026"

# Fields that must agree between this evaluation's frozen per-scenario solver
# configuration and the configuration recorded inside the stable-reference
# probe artifact before that artifact's result may be imported without a
# re-solve.
_STABLE_IMPORT_CONFIG_FIELDS = (
    "algorithm", "solver_seed", "workers", "total_time_limit_seconds",
    "per_stage_time_limit_seconds", "internal_feasibility_hint_strategy",
    "internal_repair_objective_strategy", "stop_after_first_valid_solution",
    "external_persisted_seed", "initial_solution_artifact_dir",
    "stage_order", "objective_order",
)

# Every result row -- whether imported or freshly solved -- must carry this
# exact field set with the same semantics. A row missing any of these fails
# closed (raises) rather than letting a reporting step silently infer a
# default value such as ``False`` for a gate field.
_REQUIRED_ROW_FIELDS = (
    "scenario_id", "scenario_group", "result_origin", "solver_rerun",
    "status", "final_assignment_available", "publishable_assignment_available",
    "publishable_recovery", "final_schedule_policy_pass", "consistency_issue_count",
    "assignment_nonpublishable", "response_proto_hash",
)


# The CP-SAT repair objective minimizes Hamming distance to the Constrained
# First hint subject to the unchanged hard model, including the Final
# Schedule Policy hard constraints the hint itself may violate. That
# construction means every successful repair *removes* whatever policy
# violations the hint had, while sometimes assigning fewer raw primary/
# logical requests than the (policy-violating) hint did. That is a
# structural property of the comparison, true for every row, not a
# per-scenario judgment call -- so it is reported as a fixed classification
# rather than a per-row subjective preference label.
_COMPARISON_INTERPRETATION = "policy_compliance_tradeoff"


def _is_publishable(
    status: str | None,
    assignment_available: bool,
    response_hash: str | None,
    policy_pass: Any,
    consistency_issue_count: Any,
) -> bool:
    """Single, shared publishable-assignment gate used by both the imported
    stable-reference row and every freshly solved row, so the two paths can
    never silently diverge on what counts as publishable."""
    return bool(
        status in {"FEASIBLE", "OPTIMAL"}
        and assignment_available
        and response_hash
        and policy_pass is True
        and consistency_issue_count == 0
    )


def _validate_row_schema(row: dict[str, Any], scenario_id: str) -> None:
    missing = [field for field in _REQUIRED_ROW_FIELDS if field not in row]
    if missing:
        raise CpSatEvaluationError(
            f"result row for {scenario_id} is missing required fields (fail-closed, not defaulted): {missing}"
        )


def _repair_legacy_row(row: dict[str, Any]) -> dict[str, Any]:
    """Backfill fields a pre-fix raw artifact's persisted row is missing, by
    *computing* them from that same row's own already-persisted fields --
    never inferring them from anything outside the row, and never silently
    defaulting to ``False`` when the inputs needed to compute the true value
    are themselves missing (that case fails closed instead)."""
    repaired = dict(row)
    scenario_id = repaired.get("scenario_id", "<unknown>")
    if "publishable_assignment_available" not in repaired or "publishable_recovery" not in repaired:
        required_inputs = ("status", "final_assignment_available", "response_proto_hash", "final_schedule_policy_pass", "consistency_issue_count")
        missing_inputs = [field for field in required_inputs if field not in repaired]
        if missing_inputs:
            raise CpSatEvaluationError(
                f"cannot repair publishable field for {scenario_id} (fail-closed, not defaulted): "
                f"source fields are themselves missing: {missing_inputs}"
            )
        publishable = _is_publishable(
            repaired["status"], repaired["final_assignment_available"], repaired["response_proto_hash"],
            repaired["final_schedule_policy_pass"], repaired["consistency_issue_count"],
        )
        repaired.setdefault("publishable_assignment_available", publishable)
        repaired.setdefault("publishable_recovery", publishable)
    if "global_infeasibility_proven" not in repaired or "solver_global_infeasibility_proven" not in repaired:
        if "status" not in repaired:
            raise CpSatEvaluationError(
                f"cannot repair infeasibility-proof field for {scenario_id} (fail-closed, not defaulted): status is itself missing"
            )
        proven = repaired["status"] == "INFEASIBLE"
        repaired.setdefault("global_infeasibility_proven", proven)
        repaired.setdefault("solver_global_infeasibility_proven", proven)
    return repaired


def load_normal_evaluation_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    payload = _read_json(Path(path))
    required = {
        "evaluation_name", "evaluation_version", "baseline_evaluation",
        "stable_probe_evaluation", "source_normal_suite", "stable_probe_artifact",
        "source_git_commit", "split", "development_data",
        "holdout_execution_allowed", "stress_execution_allowed", "tuning_allowed",
        "solver_configuration", "scenario_groups", "scenarios", "notes",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise CpSatEvaluationError("normal evaluation manifest missing: " + ", ".join(missing))
    if payload["split"] != "development" or payload["development_data"] is not True:
        raise CpSatEvaluationError("normal evaluation manifest must use development data")
    if payload["holdout_execution_allowed"] is not False:
        raise CpSatEvaluationError("normal evaluation manifest must forbid holdout execution")
    if payload["stress_execution_allowed"] is not False:
        raise CpSatEvaluationError("normal evaluation manifest must forbid stress execution")
    if payload["tuning_allowed"] is not True:
        raise CpSatEvaluationError("normal evaluation manifest must permit only this tuning probe")
    _repair_probe_solver_configuration(payload)
    groups = payload["scenario_groups"]
    scenarios = payload["scenarios"]
    if not isinstance(groups, dict) or set(groups) != {"normal"} or not isinstance(scenarios, list):
        raise CpSatEvaluationError("normal evaluation manifest must contain only the normal scenario group")
    expected_order = [
        "normal_dev_reference_2026", "normal_dev_01", "normal_dev_02", "normal_dev_03",
        "normal_dev_04", "normal_dev_05", "normal_dev_06", "normal_dev_07",
        "normal_dev_08", "normal_dev_09", "normal_dev_10", "normal_dev_11",
    ]
    if [str(item) for item in groups["normal"]] != expected_order or len(scenarios) != 12:
        raise CpSatEvaluationError("normal evaluation manifest must contain exactly the 12 frozen normal scenarios in order")
    parsed_ids: list[str] = []
    for raw in scenarios:
        fields = {"scenario_id", "group", "source_suite", "source_scenario_id", "paired_normal_scenario_id", "expected_feasibility"}
        if not isinstance(raw, dict) or fields - set(raw):
            raise CpSatEvaluationError("normal evaluation scenario is incomplete")
        scenario = EvaluationScenario(**{field: str(raw[field]) for field in fields})
        if scenario.group != "normal" or scenario.source_suite != "normal" or scenario.expected_feasibility != "unknown":
            raise CpSatEvaluationError(f"invalid normal evaluation scenario: {scenario.scenario_id}")
        if any(token in scenario.scenario_id for token in ("stress", "holdout", "negative")):
            raise CpSatEvaluationError(f"forbidden scenario id in normal evaluation manifest: {scenario.scenario_id}")
        if scenario.scenario_id in parsed_ids:
            raise CpSatEvaluationError(f"duplicate normal evaluation scenario: {scenario.scenario_id}")
        parsed_ids.append(scenario.scenario_id)
    if parsed_ids != expected_order:
        raise CpSatEvaluationError("normal evaluation manifest scenario order does not match the frozen order")
    return payload


def _greedy_diagnostics(allocation_input: Any, math_ids: tuple[str, ...], fallback_rules: tuple[Any, ...]) -> dict[str, Any]:
    """Run the production Constrained First baseline once, purely to audit its
    structural integrity and to report it as an internal-hint diagnostic. Its
    output is never a candidate final assignment."""
    started = time.perf_counter()
    greedy = run_constrained_first_baseline(
        allocation_input, 20260630, math_fallback_rules=fallback_rules, math_course_ids=math_ids,
    )
    elapsed = time.perf_counter() - started
    policy = evaluate_final_schedule_policy(greedy.algorithm_name, greedy.student_outcomes)
    issue_codes = [str(getattr(issue, "code", "")) for issue in greedy.consistency_issues]
    structural = {
        "capacity_violations": sum(item.assigned_count > item.capacity for item in greedy.section_roster_summary),
        "period_conflicts": sum("PERIOD" in code for code in issue_codes),
        "duplicate_logical_identity_issues": sum("DUPLICATE" in code or "IDENTITY" in code for code in issue_codes),
        "invalid_candidate_edges": 0,
        "consistency_issue_count": len(greedy.consistency_issues),
    }
    logical_gaps = [int(item.logical_schedule_gap_count or 0) for item in greedy.student_outcomes]
    primary_outcomes = [item for item in greedy.request_outcomes if item.request_type == "primary"]
    primary_assigned = sum(item.status.value == "assigned" for item in primary_outcomes)
    return {
        "generation_seconds": elapsed,
        "structural_integrity": structural,
        "policy_violation_count": policy.summary.violating_student_count,
        "greedy_primary_assigned": primary_assigned,
        "greedy_primary_unmet": len(primary_outcomes) - primary_assigned,
        "greedy_logical_assigned": sum(int(item.assigned_logical_course_count or 0) for item in greedy.student_outcomes),
        "greedy_logical_gap": sum(logical_gaps),
        "greedy_logical_full": sum(gap == 0 for gap in logical_gaps),
    }


def _import_stable_reference(
    manifest: dict[str, Any],
    config_dir: Path,
    probe_artifact_dir: Path,
) -> dict[str, Any]:
    """Import the already-completed stable-reference probe result without
    re-solving it. Raises CpSatEvaluationError (fail closed) on any
    provenance or configuration mismatch."""
    probe_root = Path(probe_artifact_dir)
    tree_hash, files, directories, bytes_total = _verify_sha256_manifest(probe_root)
    probe_manifest = _read_json(probe_root / "evaluation_manifest_snapshot.json")
    probe_run_manifest = _read_json(probe_root / "run_manifest.json")
    probe_result = _read_json(probe_root / "probe_result.json")
    probe_final_validation = _read_json(probe_root / "final_validation.json")
    probe_stage_trace = _read_json(probe_root / "stage_trace.json")
    probe_model_invariance = _read_json(probe_root / "model_invariance.json")
    probe_parameter_audit = _read_json(probe_root / "repair_parameter_audit.json")
    probe_failures = _read_json(probe_root / "failures.json")

    probe_config = probe_manifest.get("solver_configuration", {})
    this_config = manifest["solver_configuration"]
    mismatches = [
        field for field in _STABLE_IMPORT_CONFIG_FIELDS
        if probe_config.get(field) != this_config.get(field)
    ]
    if mismatches:
        raise CpSatEvaluationError(
            "stable reference import refused: solver configuration mismatch on " + ", ".join(mismatches)
        )
    if probe_run_manifest.get("solver_runs") != 1:
        raise CpSatEvaluationError("stable reference import refused: probe provenance is not exactly one solver run")
    if probe_run_manifest.get("external_persisted_seed") is not False:
        raise CpSatEvaluationError("stable reference import refused: probe used an external persisted seed")
    if probe_run_manifest.get("status") != "completed":
        raise CpSatEvaluationError("stable reference import refused: probe run is not completed")
    if probe_run_manifest.get("completed_scenario_ids") != [STABLE_SCENARIO_ID]:
        raise CpSatEvaluationError("stable reference import refused: probe scenario id mismatch")
    response_hash = probe_result.get("response_proto_hash")
    if not response_hash or response_hash != probe_final_validation.get("response_proto_hash"):
        raise CpSatEvaluationError("stable reference import refused: missing or inconsistent response hash")
    if not probe_result.get("publishable_recovery"):
        raise CpSatEvaluationError("stable reference import refused: probe result is not publishable")
    if probe_failures.get("unexpected_failure_count"):
        raise CpSatEvaluationError("stable reference import refused: probe artifact recorded failures")

    scenario = _scenario_from_payload(manifest, STABLE_SCENARIO_ID)
    allocation_input, _input_manifest, source_result = _load_scenario_input(manifest, scenario, config_dir)
    fingerprint = asdict(canonical_input_fingerprint(allocation_input))
    if fingerprint != STABLE_FINGERPRINT:
        raise CpSatEvaluationError("stable reference import refused: canonical fingerprint is not the frozen stable fingerprint")
    if source_result.get("input_fingerprint") != fingerprint:
        raise CpSatEvaluationError("stable reference import refused: source scenario fingerprint mismatch")

    canonical_edges = sum(len(value) for value in allocation_input.candidate_index.values())
    model_candidate_variables = probe_parameter_audit.get("candidate_hint_variable_count")
    mandatory_fallback_candidate_variables = (
        max(model_candidate_variables - canonical_edges, 0) if model_candidate_variables is not None else None
    )

    metrics = dict(probe_final_validation.get("raw_metrics") or {})
    stable_stage = next(iter(probe_stage_trace.get("stages", [])), None)
    solver_wall_time = stable_stage.get("wall_time_seconds") if stable_stage else None
    auxiliary_hinted = metrics.get("internal_hint_auxiliary_variables_hinted")
    unhinted = metrics.get("internal_hint_unhinted_variables")
    auxiliary_hint_coverage = None
    if auxiliary_hinted is not None and unhinted is not None and (auxiliary_hinted + unhinted):
        auxiliary_hint_coverage = round(auxiliary_hinted / (auxiliary_hinted + unhinted), 6)

    status = probe_result.get("status")
    publishable = _is_publishable(
        status,
        True,
        response_hash,
        metrics.get("final_schedule_policy_pass"),
        metrics.get("consistency_issue_count"),
    )
    row = {
        **metrics,
        "scenario_id": STABLE_SCENARIO_ID,
        "scenario_group": "normal",
        "result_origin": "imported_frozen_probe",
        "solver_rerun": False,
        "source_artifact_path": str(probe_root),
        "source_response_hash": response_hash,
        "source_manifest_hash": evaluation_manifest_hash(probe_root / "evaluation_manifest_snapshot.json"),
        "status": status,
        "final_assignment_available": True,
        "publishable_assignment_available": publishable,
        "publishable_recovery": publishable,
        "global_infeasibility_proven": False,
        "solver_global_infeasibility_proven": False,
        "response_proto_hash": response_hash,
        "end_to_end_scenario_runtime_seconds": probe_result.get("runtime_seconds"),
        "runtime_seconds": probe_result.get("runtime_seconds"),
        "solver_wall_time_seconds": solver_wall_time,
        "time_to_first_solution_seconds": probe_result.get("time_to_first_solution_seconds"),
        "input_provenance_validation_seconds": None,
        "constrained_first_generation_seconds": None,
        "hint_conversion_seconds": None,
        "model_build_seconds": None,
        "post_solve_extraction_seconds": None,
        "policy_consistency_validation_seconds": None,
        "artifact_export_seconds": None,
        "hamming_distance": probe_result.get("hamming_distance"),
        "greedy_assignments_removed": probe_result.get("greedy_assignments_removed"),
        "new_assignments_added": probe_result.get("new_assignments_added"),
        "changed_students": probe_result.get("changed_students"),
        "changed_requests": probe_result.get("changed_requests"),
        "changed_sections": probe_result.get("changed_sections"),
        "branches": probe_result.get("branches"),
        "conflicts": probe_result.get("conflicts"),
        "deterministic_time_seconds": probe_result.get("deterministic_time_seconds"),
        "canonical_input_candidate_edges": canonical_edges,
        "model_candidate_variables": model_candidate_variables,
        "mandatory_fallback_candidate_variables": mandatory_fallback_candidate_variables,
        "candidate_hint_coverage": probe_parameter_audit.get("candidate_hint_coverage"),
        "auxiliary_hint_coverage": auxiliary_hint_coverage,
        "greedy_primary_assigned": metrics.get("internal_hint_primary_assigned"),
        "greedy_primary_unmet": metrics.get("internal_hint_primary_unmet"),
        "greedy_logical_assigned": metrics.get("internal_hint_logical_assigned"),
        "greedy_logical_gap": metrics.get("internal_hint_logical_gap"),
        "greedy_logical_full": metrics.get("internal_hint_logical_full"),
        "greedy_policy_violation_count": metrics.get("internal_hint_policy_violation_count"),
    }
    row.update(_audit_row_timing(row, stable_stage))
    _validate_row_schema(row, STABLE_SCENARIO_ID)
    return {
        "row": row,
        "probe_artifact_verification": {
            "artifact_dir": str(probe_root),
            "files": files,
            "directories": directories,
            "bytes": bytes_total,
            "tree_hash": tree_hash,
        },
        "stage_trace": probe_stage_trace,
        "final_validation": probe_final_validation,
        "model_invariance": probe_model_invariance,
        "repair_parameter_audit": probe_parameter_audit,
        "probe_failures": probe_failures,
    }


def _solve_scenario(
    manifest: dict[str, Any],
    config: Any,
    scenario: EvaluationScenario,
    config_dir: Path,
) -> dict[str, Any]:
    end_to_end_started = time.perf_counter()

    provenance_started = time.perf_counter()
    allocation_input, input_manifest, source_result = _load_scenario_input(manifest, scenario, config_dir)
    provenance_seconds = time.perf_counter() - provenance_started

    catalog = pd.read_csv(config_dir / "course_catalog.csv", keep_default_na=False)
    math_ids = math_course_ids_from_catalog(catalog)
    fallback_rules = _load_benchmark_math_fallback_rules(config_dir, catalog)

    greedy = _greedy_diagnostics(allocation_input, math_ids, fallback_rules)
    structural = greedy["structural_integrity"]
    if any(structural[key] for key in ("capacity_violations", "period_conflicts", "duplicate_logical_identity_issues", "invalid_candidate_edges")):
        raise CpSatEvaluationCorrectnessError(
            f"Greedy structural integrity failed before solve in {scenario.scenario_id}: {structural}"
        )

    canonical_edges = sum(len(value) for value in allocation_input.candidate_index.values())

    solve_started = time.perf_counter()
    result = run_fair_cp_sat_solver(
        allocation_input,
        seed=config.solver_seed,
        math_course_ids=math_ids,
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
    solve_wall_seconds = time.perf_counter() - solve_started

    extraction_started = time.perf_counter()
    trace = _stage_trace(result, scenario.scenario_id)
    metrics = _result_metrics(result, allocation_input)
    extraction_seconds = time.perf_counter() - extraction_started

    validation_started = time.perf_counter()
    _validate_objective_bounds(trace, int(metrics["theoretical_maximum"]))
    _validate_result(result, metrics, scenario)
    validation_seconds = time.perf_counter() - validation_started

    end_to_end_seconds = time.perf_counter() - end_to_end_started

    status = result.solve_status.value
    assignment_available = bool(result.student_outcomes)
    stage = next((item for item in trace if item["stage_name"] == "internal_repair_feasibility"), None)
    response_hash = stage.get("response_proto_hash") if stage else None
    stats = result.model_stats
    mandatory_fallback_candidate_variables = max(stats.internal_hint_candidate_variables - canonical_edges, 0)
    auxiliary_total = stats.internal_hint_auxiliary_variables_hinted + stats.internal_hint_unhinted_variables
    auxiliary_hint_coverage = round(stats.internal_hint_auxiliary_variables_hinted / auxiliary_total, 6) if auxiliary_total else None
    publishable = _is_publishable(
        status,
        assignment_available,
        response_hash,
        metrics.get("final_schedule_policy_pass"),
        metrics.get("consistency_issue_count"),
    )
    # A full-hard-model INFEASIBLE from the sole internal_repair_feasibility
    # stage is the only stage in this evaluation's frozen stage_order, so an
    # INFEASIBLE status here is a direct proof against the unchanged
    # production hard model -- never a fixed-objective or bootstrap-stage
    # result, and never conditioned on an earlier stage's incumbent.
    global_infeasibility_proven = status == "INFEASIBLE"

    row = {
        **{key: value for key, value in metrics.items() if key != "student_count"},
        "student_count": metrics["student_count"],
        "scenario_id": scenario.scenario_id,
        "scenario_group": scenario.group,
        "result_origin": "solved",
        "solver_rerun": True,
        "publishable_recovery": publishable,
        "global_infeasibility_proven": global_infeasibility_proven,
        "solver_global_infeasibility_proven": global_infeasibility_proven,
        "status": status,
        "final_assignment_available": assignment_available,
        "publishable_assignment_available": publishable,
        "response_proto_hash": response_hash,
        "full_hard_model_infeasibility_proven": status == "INFEASIBLE",
        "end_to_end_scenario_runtime_seconds": round(end_to_end_seconds, 6),
        "runtime_seconds": round(end_to_end_seconds, 6),
        "input_provenance_validation_seconds": round(provenance_seconds, 6),
        "constrained_first_generation_seconds": round(greedy["generation_seconds"], 6),
        "hint_conversion_seconds": None,
        "model_build_seconds": round(stats.enrichment_build_time_seconds, 6),
        "solver_wall_time_seconds": round(stats.internal_repair_runtime_seconds or solve_wall_seconds, 6),
        "time_to_first_solution_seconds": stats.internal_repair_time_to_first_solution_seconds,
        "post_solve_extraction_seconds": round(extraction_seconds, 6),
        "policy_consistency_validation_seconds": round(validation_seconds, 6),
        "artifact_export_seconds": None,
        "hamming_distance": stats.internal_repair_hamming_distance,
        "greedy_assignments_removed": stats.internal_repair_greedy_assignments_removed,
        "new_assignments_added": stats.internal_repair_new_assignments_added,
        "changed_students": stats.internal_repair_changed_students,
        "changed_requests": stats.internal_repair_changed_requests,
        "changed_sections": stats.internal_repair_changed_sections,
        "branches": stage.get("branch_count") if stage else None,
        "conflicts": stage.get("conflict_count") if stage else None,
        "deterministic_time_seconds": stage.get("deterministic_time_seconds") if stage else None,
        "objective_descriptor_hash": stage.get("objective_descriptor_hash") if stage else None,
        "objective_value": stage.get("objective_value") if stage else None,
        "best_objective_bound": stage.get("best_objective_bound") if stage else None,
        "canonical_input_candidate_edges": canonical_edges,
        "model_candidate_variables": stats.internal_hint_candidate_variables,
        "mandatory_fallback_candidate_variables": mandatory_fallback_candidate_variables,
        "candidate_hint_coverage": stats.internal_hint_candidate_coverage_rate,
        "auxiliary_hint_coverage": auxiliary_hint_coverage,
        "greedy_primary_assigned": greedy["greedy_primary_assigned"],
        "greedy_primary_unmet": greedy["greedy_primary_unmet"],
        "greedy_logical_assigned": greedy["greedy_logical_assigned"],
        "greedy_logical_gap": greedy["greedy_logical_gap"],
        "greedy_logical_full": greedy["greedy_logical_full"],
        "greedy_policy_violation_count": greedy["policy_violation_count"],
        "greedy_structural_capacity_violations": structural["capacity_violations"],
        "greedy_structural_period_conflicts": structural["period_conflicts"],
        "greedy_structural_duplicate_logical_identity_issues": structural["duplicate_logical_identity_issues"],
        "greedy_structural_invalid_candidate_edges": structural["invalid_candidate_edges"],
    }
    row.update(_audit_row_timing(row, stage))
    _validate_row_schema(row, scenario.scenario_id)
    return {
        "row": row,
        "trace": trace,
        "result": result,
        "allocation_input": allocation_input,
        "input_manifest": input_manifest,
        "source_result": source_result,
    }


def _cleanup_temp_dir(temporary: Path) -> None:
    if temporary.exists():
        for path in sorted(temporary.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        temporary.rmdir()


def _percentile(sorted_values: list[float], fraction: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = (len(sorted_values) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    if lower == upper:
        return sorted_values[lower]
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (position - lower)


def _stats(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = sorted(float(row[key]) for row in rows if row.get(key) is not None)
    if not values:
        return {"count": 0, "median": None, "p90": None, "min": None, "max": None}
    middle = len(values) // 2
    median = values[middle] if len(values) % 2 else (values[middle - 1] + values[middle]) / 2
    return {
        "count": len(values),
        "median": round(median, 6),
        "p90": round(_percentile(values, 0.9), 6),
        "min": round(values[0], 6),
        "max": round(values[-1], 6),
    }


_TIMING_TOLERANCE_SECONDS = 1.0


def _audit_row_timing(row: dict[str, Any], stage: dict[str, Any] | None) -> dict[str, Any]:
    """Reporting-layer-only cross-check of one scenario's timing fields.

    This never touches solve_status, never reruns anything, and never
    decides correctness of the solve itself -- it only decides whether the
    *reported timing numbers* for this scenario are internally consistent,
    so that an implausible one (e.g. a stage-level wall-clock reading that
    exceeds its own configured time limit) can be excluded from aggregate
    timing statistics while the raw value is still preserved for audit.
    """
    reasons: list[str] = []
    first = row.get("time_to_first_solution_seconds")
    wall = row.get("solver_wall_time_seconds")
    end_to_end = row.get("end_to_end_scenario_runtime_seconds")
    stage_wall = stage.get("wall_time_seconds") if stage else None
    effective_limit = stage.get("effective_time_limit_seconds") if stage else None

    if first is not None and first < -_TIMING_TOLERANCE_SECONDS:
        reasons.append("negative_time_to_first_solution")
    if wall is not None and first is not None and wall + _TIMING_TOLERANCE_SECONDS < first:
        reasons.append("solver_wall_time_less_than_time_to_first_solution")
    if end_to_end is not None and wall is not None and end_to_end + _TIMING_TOLERANCE_SECONDS < wall:
        reasons.append("end_to_end_runtime_less_than_solver_wall_time")
    if stage_wall is not None and effective_limit is not None and stage_wall > effective_limit + _TIMING_TOLERANCE_SECONDS:
        reasons.append("stage_reported_wall_time_exceeds_configured_time_limit")

    return {
        "timing_diagnostic_valid": not reasons,
        "timing_anomaly_reasons": reasons,
        "time_to_first_solution_seconds": first,
        "solver_wall_time_seconds": wall,
        "end_to_end_scenario_runtime_seconds": end_to_end,
        "stage_reported_wall_time_seconds": stage_wall,
        "effective_time_limit_seconds": effective_limit,
    }


def _hint_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "scenario_id": row["scenario_id"],
        "result_origin": row.get("result_origin"),
        "greedy_primary_assigned": row.get("greedy_primary_assigned"),
        "greedy_primary_unmet": row.get("greedy_primary_unmet"),
        "greedy_logical_assigned": row.get("greedy_logical_assigned"),
        "greedy_logical_gap": row.get("greedy_logical_gap"),
        "greedy_logical_full": row.get("greedy_logical_full"),
        "greedy_policy_violation_count": row.get("greedy_policy_violation_count"),
        "candidate_hint_coverage": row.get("candidate_hint_coverage"),
        "auxiliary_hint_coverage": row.get("auxiliary_hint_coverage"),
        "canonical_input_candidate_edges": row.get("canonical_input_candidate_edges"),
        "model_candidate_variables": row.get("model_candidate_variables"),
        "mandatory_fallback_candidate_variables": row.get("mandatory_fallback_candidate_variables"),
    }


def _timing_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "scenario_id": row["scenario_id"],
        "result_origin": row.get("result_origin"),
        "input_provenance_validation_seconds": row.get("input_provenance_validation_seconds"),
        "constrained_first_generation_seconds": row.get("constrained_first_generation_seconds"),
        "hint_conversion_seconds": row.get("hint_conversion_seconds"),
        "model_build_seconds": row.get("model_build_seconds"),
        "solver_wall_time_seconds": row.get("solver_wall_time_seconds"),
        "time_to_first_solution_seconds": row.get("time_to_first_solution_seconds"),
        "post_solve_extraction_seconds": row.get("post_solve_extraction_seconds"),
        "policy_consistency_validation_seconds": row.get("policy_consistency_validation_seconds"),
        "artifact_export_seconds": row.get("artifact_export_seconds"),
        "end_to_end_scenario_runtime_seconds": row.get("end_to_end_scenario_runtime_seconds"),
    }


def _read_completed_normal_summary(path: Path, manifest_hash: str, scenario_id: str) -> dict[str, Any]:
    cached = _read_completed_summary(path)
    if cached.get("evaluation_manifest_sha256") != manifest_hash or cached.get("scenario", {}).get("scenario_id") != scenario_id:
        raise CpSatEvaluationError(f"cached scenario provenance mismatch: {scenario_id}")
    return cached


class CpSatNormalEvaluationRunner:
    """Phase C: import the stable reference, solve the remaining 11 normal
    development scenarios once each under the identical frozen configuration."""

    def __init__(
        self,
        manifest_path: str | Path = DEFAULT_MANIFEST,
        config_dir: str | Path = "data/config",
        stable_probe_artifact_dir: str | Path = DEFAULT_STABLE_PROBE_ARTIFACT,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.config_dir = Path(config_dir)
        self.stable_probe_artifact_dir = Path(stable_probe_artifact_dir)
        self.manifest = load_normal_evaluation_manifest(self.manifest_path)
        self.manifest_hash = evaluation_manifest_hash(self.manifest_path)
        self.config = _repair_probe_solver_configuration(self.manifest)

    def select(self, scenario_id: str | None = None, max_scenarios: int | None = None) -> list[EvaluationScenario]:
        scenarios = [_scenario_from_payload(self.manifest, item["scenario_id"]) for item in self.manifest["scenarios"]]
        if scenario_id is not None:
            scenarios = [item for item in scenarios if item.scenario_id == scenario_id]
            if not scenarios:
                raise CpSatEvaluationError(f"scenario is not in normal evaluation manifest: {scenario_id}")
        if max_scenarios is not None:
            if max_scenarios <= 0:
                raise CpSatEvaluationError("max_scenarios must be positive")
            scenarios = scenarios[:max_scenarios]
        return scenarios

    def verify_sources(self, scenarios: Iterable[EvaluationScenario]) -> dict[str, Any]:
        source_info = _verify_source_suite(self.manifest, "normal")
        for scenario in scenarios:
            _load_scenario_input(self.manifest, scenario, self.config_dir)
        return source_info

    def dry_run(self, scenario_id: str | None = None, max_scenarios: int | None = None) -> list[str]:
        scenarios = self.select(scenario_id, max_scenarios)
        self.verify_sources(scenarios)
        return [scenario.scenario_id for scenario in scenarios]

    def run(
        self,
        output_dir: str | Path = DEFAULT_OUTPUT,
        *,
        scenario_id: str | None = None,
        max_scenarios: int | None = None,
        resume: bool = False,
        verify_only: bool = False,
    ) -> dict[str, Any]:
        scenarios = self.select(scenario_id, max_scenarios)
        source_info = self.verify_sources(scenarios)
        if verify_only:
            return {"verified_scenarios": [scenario.scenario_id for scenario in scenarios], "source_info": source_info}
        root = Path(output_dir)
        if root.exists() and any(root.iterdir()) and not resume:
            raise CpSatEvaluationError(f"normal evaluation output is non-empty; refusing to overwrite: {root}")
        root.mkdir(parents=True, exist_ok=True)
        run_manifest_path = root / "run_manifest.json"
        selected_ids = [scenario.scenario_id for scenario in scenarios]
        if resume:
            run_manifest = _read_json(run_manifest_path)
            expected = {
                "evaluation_manifest_sha256": self.manifest_hash,
                "source_git_commit": self.manifest["source_git_commit"],
                "solver_configuration_hash": _json_hash(self.manifest["solver_configuration"]),
                "selected_scenario_ids": selected_ids,
            }
            mismatches = [key for key, value in expected.items() if run_manifest.get(key) != value]
            if mismatches:
                raise CpSatEvaluationError("normal evaluation resume provenance mismatch: " + ", ".join(mismatches))
            if run_manifest.get("holdout_runs") != 0 or run_manifest.get("stress_runs") != 0 or run_manifest.get("external_persisted_seed") is not False:
                raise CpSatEvaluationError("resume safety flags do not match development-only evaluation")
            completed = set(run_manifest.get("completed_scenario_ids", []))
            unknown_completed = completed - set(selected_ids)
            if unknown_completed:
                raise CpSatEvaluationError("resume contains unknown completed scenarios: " + ", ".join(sorted(unknown_completed)))
        else:
            completed = set()
            _write_json(root / "evaluation_manifest_snapshot.json", self.manifest)
            _write_json(run_manifest_path, {
                "schema_version": EVALUATION_SCHEMA_VERSION,
                "status": "running",
                "evaluation_manifest_sha256": self.manifest_hash,
                "source_git_commit": self.manifest["source_git_commit"],
                "solver_configuration_hash": _json_hash(self.manifest["solver_configuration"]),
                "selected_scenario_ids": selected_ids,
                "completed_scenario_ids": [],
                "failed_scenario_ids": [],
                "imported_scenario_ids": [],
                "solved_scenario_ids": [],
                "holdout_runs": 0,
                "stress_runs": 0,
                "external_persisted_seed": False,
                "source_info": source_info,
            })
        failures: list[dict[str, Any]] = []
        for scenario in scenarios:
            if scenario.scenario_id in completed:
                _read_completed_normal_summary(
                    root / "scenarios" / scenario.scenario_id / "scenario_summary.json",
                    self.manifest_hash,
                    scenario.scenario_id,
                )
                continue
            try:
                if scenario.scenario_id == STABLE_SCENARIO_ID:
                    self._import_scenario(root, scenario)
                else:
                    self._solve_and_export_scenario(root, scenario)
                completed.add(scenario.scenario_id)
                self._update_run_manifest(root, scenarios, completed, failures, status="running")
            except CpSatEvaluationCorrectnessError as exc:
                failures.append({"scenario_id": scenario.scenario_id, "failure_type": "critical_correctness_failure", "message": str(exc)})
                self._update_run_manifest(root, scenarios, completed, failures, status="failed")
                _write_json(root / "failures.json", {"failures": failures, "unexpected_failure_count": len(failures)})
                raise
            except Exception as exc:
                failures.append({"scenario_id": scenario.scenario_id, "failure_type": "runner_failure", "message": str(exc)})
                self._update_run_manifest(root, scenarios, completed, failures, status="failed")
                _write_json(root / "failures.json", {"failures": failures, "unexpected_failure_count": len(failures)})
                raise CpSatEvaluationError(f"scenario failed: {scenario.scenario_id}: {exc}") from exc
        source_info_after = _verify_source_suite(self.manifest, "normal")
        if source_info_after != source_info:
            raise CpSatEvaluationCorrectnessError("source artifact changed during evaluation")
        _write_json(root / "failures.json", {"failures": failures, "unexpected_failure_count": len(failures)})
        self._write_aggregates(root, scenarios, source_info, failures)
        _write_json(run_manifest_path, {
            **_read_json(run_manifest_path),
            "status": "completed",
            "completed_scenario_ids": sorted(completed),
            "failed_scenario_ids": [item["scenario_id"] for item in failures],
        })
        _write_checksums(root)
        return _read_json(root / "aggregate_summary.json")

    def _update_run_manifest(self, root: Path, scenarios: list[EvaluationScenario], completed: set[str], failures: list[dict[str, Any]], *, status: str) -> None:
        current = _read_json(root / "run_manifest.json")
        imported = sorted(s.scenario_id for s in scenarios if s.scenario_id == STABLE_SCENARIO_ID and s.scenario_id in completed)
        solved = sorted(s.scenario_id for s in scenarios if s.scenario_id != STABLE_SCENARIO_ID and s.scenario_id in completed)
        current.update({
            "status": status,
            "completed_scenario_ids": sorted(completed),
            "failed_scenario_ids": [item["scenario_id"] for item in failures],
            "imported_scenario_ids": imported,
            "solved_scenario_ids": solved,
        })
        _write_json(root / "run_manifest.json", current)

    def _import_scenario(self, root: Path, scenario: EvaluationScenario) -> None:
        imported = _import_stable_reference(self.manifest, self.config_dir, self.stable_probe_artifact_dir)
        row = imported["row"]
        base = root / "scenarios"
        base.mkdir(parents=True, exist_ok=True)
        destination = base / scenario.scenario_id
        temporary = Path(tempfile.mkdtemp(prefix=f".{scenario.scenario_id}.", dir=base))
        export_started = time.perf_counter()
        try:
            _write_json(temporary / "stage_trace.json", imported["stage_trace"])
            _write_json(temporary / "final_validation.json", imported["final_validation"])
            _write_json(temporary / "model_invariance.json", imported["model_invariance"])
            _write_json(temporary / "repair_parameter_audit.json", imported["repair_parameter_audit"])
            _write_json(temporary / "source_probe_failures.json", imported["probe_failures"])
            row["artifact_export_seconds"] = round(time.perf_counter() - export_started, 6)
            scenario_summary = {
                "schema_version": EVALUATION_SCHEMA_VERSION,
                "status": "completed_with_assignment",
                "evaluation_manifest_sha256": self.manifest_hash,
                "source_git_commit": self.manifest["source_git_commit"],
                "scenario": asdict(scenario),
                "result_origin": "imported_frozen_probe",
                "solver_rerun": False,
                "result": row,
                "solver_configuration": self.manifest["solver_configuration"],
                "source_probe_artifact_verification": imported["probe_artifact_verification"],
            }
            _write_json(temporary / "scenario_summary.json", scenario_summary)
            temporary.replace(destination)
        finally:
            _cleanup_temp_dir(temporary)

    def _solve_and_export_scenario(self, root: Path, scenario: EvaluationScenario) -> None:
        solved = _solve_scenario(self.manifest, self.config, scenario, self.config_dir)
        row = solved["row"]
        result = solved["result"]
        trace = solved["trace"]
        allocation_input = solved["allocation_input"]
        source_result = solved["source_result"]

        base = root / "scenarios"
        base.mkdir(parents=True, exist_ok=True)
        destination = base / scenario.scenario_id
        temporary = Path(tempfile.mkdtemp(prefix=f".{scenario.scenario_id}.", dir=base))
        export_started = time.perf_counter()
        try:
            solver_dir = temporary / "solver"
            solver_dir.mkdir()
            _write_json(temporary / "stage_trace.json", {"scenario_id": scenario.scenario_id, "stages": trace})
            _write_json(temporary / "final_validation.json", {
                "final_schedule_policy_pass": row["final_schedule_policy_pass"],
                "consistency_issue_count": row["consistency_issue_count"],
                "assignment_nonpublishable": row["assignment_nonpublishable"],
                "status": row["status"],
                "raw_metrics": {key: value for key, value in row.items()},
            })
            _write_json(solver_dir / "solver_result.json", {
                "status": row["status"],
                "objective_values": result.objective_values,
                "model_stats": result.model_stats,
                "response_proto_hash": row["response_proto_hash"],
                "assignment_hash": _assignment_hash(result),
            })
            _write_csv(temporary / "grade_subgroup_results.csv", _grade_rows(allocation_input, result, scenario.scenario_id))
            if result.student_outcomes:
                _write_csv(solver_dir / "assignments.csv", [_jsonable(asdict(item)) for item in result.assignments])
                _write_csv(solver_dir / "request_outcomes.csv", [_jsonable(asdict(item)) for item in result.request_outcomes])
                _write_csv(solver_dir / "student_outcomes.csv", [_jsonable(asdict(item)) for item in result.student_outcomes])
            row["artifact_export_seconds"] = round(time.perf_counter() - export_started, 6)
            scenario_summary = {
                "schema_version": EVALUATION_SCHEMA_VERSION,
                "status": "completed_with_assignment" if result.student_outcomes else "completed_without_assignment",
                "evaluation_manifest_sha256": self.manifest_hash,
                "source_git_commit": self.manifest["source_git_commit"],
                "scenario": asdict(scenario),
                "result_origin": "solved",
                "solver_rerun": True,
                "result": row,
                "solver_configuration": self.manifest["solver_configuration"],
                "input_fingerprint": source_result.get("input_fingerprint"),
                "assignment_hash": _assignment_hash(result),
            }
            _write_json(temporary / "scenario_summary.json", scenario_summary)
            temporary.replace(destination)
        finally:
            _cleanup_temp_dir(temporary)

    def _write_aggregates(self, root: Path, scenarios: list[EvaluationScenario], source_info: dict[str, Any], failures: list[dict[str, Any]]) -> None:
        rows = []
        stage_rows = []
        grade_rows: list[dict[str, Any]] = []
        for scenario in scenarios:
            summary_path = root / "scenarios" / scenario.scenario_id / "scenario_summary.json"
            if not summary_path.is_file():
                continue
            summary = _read_completed_summary(summary_path)
            rows.append(summary["result"])
            trace_payload = _read_json(root / "scenarios" / scenario.scenario_id / "stage_trace.json")
            stages = trace_payload.get("stages", []) if isinstance(trace_payload, dict) else trace_payload
            stage_rows.extend({"scenario_id": scenario.scenario_id, **row} for row in stages)
            grade_path = root / "scenarios" / scenario.scenario_id / "grade_subgroup_results.csv"
            if grade_path.is_file() and grade_path.stat().st_size:
                with grade_path.open(newline="") as handle:
                    grade_rows.extend(csv.DictReader(handle))
        _write_csv(root / "scenario_results.csv", rows)
        _write_csv(root / "stage_results.csv", stage_rows)
        _write_csv(root / "grade_subgroup_results.csv", grade_rows)
        _write_csv(root / "internal_hint_results.csv", [_hint_row(row) for row in rows])
        _write_csv(root / "timing_results.csv", [_timing_row(row) for row in rows])
        _write_csv(root / "paired_recovery_vs_phase_c.csv", self._paired_vs_phase_c_baseline(rows))
        _write_csv(root / "paired_recovery_vs_constrained_first.csv", self._paired_vs_constrained_first(rows))
        aggregate = self._aggregate_summary(rows, failures, source_info)
        _write_json(root / "aggregate_summary.json", aggregate)
        _write_json(root / "readiness_assessment.json", self._success_gate(rows, failures))

    def _paired_vs_phase_c_baseline(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        baseline_root = Path(self.manifest["phase_c_baseline_artifact"]["artifact_dir"])
        baseline_path = baseline_root / "normal_results.csv"
        baseline_by_id: dict[str, dict[str, Any]] = {}
        if baseline_path.is_file():
            with baseline_path.open(newline="") as handle:
                baseline_by_id = {item["scenario_id"]: item for item in csv.DictReader(handle)}
        output = []
        for row in rows:
            scenario_id = row["scenario_id"]
            baseline = baseline_by_id.get(scenario_id, {})
            baseline_available = str(baseline.get("final_assignment_available", "")).strip().lower() == "true"
            recovery_publishable = bool(
                row.get("final_assignment_available")
                and row.get("final_schedule_policy_pass") is True
                and row.get("consistency_issue_count") == 0
            )
            output.append({
                "scenario_id": scenario_id,
                "phase_c_baseline_status": baseline.get("status") or baseline.get("raw_result_status"),
                "phase_c_baseline_publishable": baseline_available,
                "phase_c_baseline_runtime_seconds": baseline.get("runtime_seconds"),
                "recovery_status": row.get("status"),
                "recovery_publishable": recovery_publishable,
                "assignment_recovered": recovery_publishable and not baseline_available,
                "recovery_first_solution_time_seconds": row.get("time_to_first_solution_seconds"),
                "recovery_end_to_end_runtime_seconds": row.get("end_to_end_scenario_runtime_seconds"),
                "status_changed": baseline.get("status") != row.get("status"),
            })
        return output

    def _paired_vs_constrained_first(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        normal_root = Path(self.manifest["source_normal_suite"]["artifact_dir"])
        return _compute_paired_vs_constrained_first(normal_root, rows)

    def _aggregate_summary(self, rows: list[dict[str, Any]], failures: list[dict[str, Any]], source_info: dict[str, Any]) -> dict[str, Any]:
        return _compute_aggregate_summary(rows, failures, source_info)

    def _success_gate(self, rows: list[dict[str, Any]], failures: list[dict[str, Any]]) -> dict[str, Any]:
        return _compute_success_gate(rows, failures)


def _row_is_publishable(row: dict[str, Any]) -> bool:
    """Read the row's own validated publishable field rather than
    recomputing the gate condition ad hoc -- imported and solved rows are
    schema-validated (``_validate_row_schema``) to always carry it."""
    if "publishable_assignment_available" not in row:
        raise CpSatEvaluationError(
            f"row for {row.get('scenario_id', '<unknown>')} is missing publishable_assignment_available "
            "(fail-closed: refusing to infer publishable=False)"
        )
    return bool(row["publishable_assignment_available"])


def _compute_aggregate_summary(rows: list[dict[str, Any]], failures: list[dict[str, Any]], source_info: dict[str, Any]) -> dict[str, Any]:
    status_counts = Counter(row.get("status") for row in rows)
    publishable = [row for row in rows if _row_is_publishable(row)]
    policy_pass = [row for row in rows if row.get("final_schedule_policy_pass") is True]
    imported = [row for row in rows if row.get("result_origin") == "imported_frozen_probe"]
    solved = [row for row in rows if row.get("result_origin") == "solved"]
    invalid_timing_rows = [row for row in rows if not row.get("timing_diagnostic_valid", True)]
    # The timing-validity flag only ever fires on the stage-reported solver
    # wall time (a value sourced from the CP-SAT engine's own internal
    # clock); the independently perf_counter-measured end-to-end scenario
    # runtime is never the anomalous measurement, so it is aggregated over
    # every attempted scenario regardless of that flag.
    valid_solver_timing = [row for row in publishable if row.get("timing_diagnostic_valid", True)]
    return {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "development_only": True,
        "parameter_tuning_performed": False,
        "holdout_runs": 0,
        "stress_runs": 0,
        "negative_runs": 0,
        "external_persisted_seed": False,
        "scenarios": len(rows),
        "attempted_count": len(rows),
        "new_solver_runs": len(solved),
        "imported_result_count": len(imported),
        "status_counts": dict(sorted((key, value) for key, value in status_counts.items() if key is not None)),
        "feasible_count": status_counts.get("FEASIBLE", 0),
        "optimal_count": status_counts.get("OPTIMAL", 0),
        "unknown_count": status_counts.get("UNKNOWN", 0),
        "infeasible_count": status_counts.get("INFEASIBLE", 0),
        "publishable_assignment_count": len(publishable),
        "publishable_assignment_rate": round(len(publishable) / len(rows), 6) if rows else None,
        "publishable_assignment_denominator": len(rows),
        "policy_pass_count": len(policy_pass),
        "policy_pass_rate": round(len(policy_pass) / len(rows), 6) if rows else None,
        "critical_correctness_failures": sum(item["failure_type"] == "critical_correctness_failure" for item in failures),
        "first_solution_time_seconds": _stats(valid_solver_timing, "time_to_first_solution_seconds"),
        "solver_wall_time_seconds": _stats(valid_solver_timing, "solver_wall_time_seconds"),
        "end_to_end_runtime_seconds": _stats(rows, "end_to_end_scenario_runtime_seconds"),
        "primary_satisfaction_rate": _stats(publishable, "primary_satisfaction_rate"),
        "logical_full_rate": _stats(publishable, "logical_full_rate"),
        "total_logical_gap": _stats(publishable, "total_logical_gap"),
        "hamming_distance": _stats(publishable, "hamming_distance"),
        "changed_students": _stats(publishable, "changed_students"),
        "quality_metric_denominator": len(publishable),
        "timing_excluded_scenario_count": len(invalid_timing_rows),
        "comparison_interpretation": _COMPARISON_INTERPRETATION,
        "source_info": source_info,
        "failures": failures,
    }


def _compute_success_gate(rows: list[dict[str, Any]], failures: list[dict[str, Any]]) -> dict[str, Any]:
    stable = next((row for row in rows if row["scenario_id"] == STABLE_SCENARIO_ID), None)
    stable_ok = bool(stable and _row_is_publishable(stable))
    publishable = [row for row in rows if _row_is_publishable(row)]
    structural_ok = all(
        row.get("final_schedule_policy_pass") is True and row.get("consistency_issue_count") == 0
        for row in publishable
    )
    critical_failures = sum(item["failure_type"] == "critical_correctness_failure" for item in failures)
    blocking: list[str] = []
    if not stable_ok:
        blocking.append("stable_reference_result_not_retained_or_not_publishable")
    if len(publishable) < 7:
        blocking.append("fewer_than_7_of_12_normal_scenarios_publishable")
    if not structural_ok:
        blocking.append("a_publishable_assignment_failed_policy_or_structural_integrity")
    if critical_failures:
        blocking.append("critical_correctness_failures_present")
    passed = not blocking
    return {
        "gate": "PASS" if passed else "FAIL",
        "publishable_count": len(publishable),
        "publishable_denominator": len(rows),
        "stable_reference_retained": stable_ok,
        "critical_correctness_failures": critical_failures,
        "blocking_reasons": blocking,
        "ready_for_stress_development": passed,
        "ready_for_holdout": False,
        "no_stress_run": True,
        "no_holdout_run": True,
        "no_negative_run": True,
        "external_persisted_seed": False,
        "comparison_interpretation": _COMPARISON_INTERPRETATION,
        "cautionary_notes": [
            "This is a development evaluation, not a generalization proof.",
            "FEASIBLE is a validated incumbent without an optimality proof; it is not OPTIMAL.",
            "UNKNOWN means no incumbent was found under the frozen budget; it is not INFEASIBLE.",
            "The Hamming distance objective value is not proven optimal.",
            "Results depend on the internal Constrained First search used only as a hint.",
            "Successful repairs are a policy-compliance tradeoff: primary/logical completion counts can rise or fall relative to the hint while every Final Schedule Policy violation is removed.",
        ],
    }


def _compute_paired_vs_constrained_first(normal_root: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        if not _row_is_publishable(row):
            continue
        scenario_id = row["scenario_id"]
        try:
            greedy = _greedy_rows(normal_root / "scenarios" / scenario_id)
        except CpSatEvaluationError:
            continue
        scenario = EvaluationScenario(scenario_id, "normal", "normal", scenario_id, scenario_id, "unknown")
        students = int(row.get("student_count") or 0)
        paired = _paired_rows(row, greedy, scenario, students)
        output.extend(
            {**item, "comparison_interpretation": _COMPARISON_INTERPRETATION}
            for item in paired
            if item["algorithm"] == "constrained_first_greedy"
        )
    return output


def _infeasibility_scope_row(row: dict[str, Any], stage: dict[str, Any] | None) -> dict[str, Any]:
    """Confirm an INFEASIBLE scenario's proof scope is exactly the single,
    unmodified full-hard-model ``internal_repair_feasibility`` stage -- never
    a fixed-objective or bootstrap-stage result, and never conditioned on an
    earlier stage's incumbent (this evaluation's frozen stage_order has only
    the one stage, so there is no earlier stage to condition on)."""
    stage_name = stage.get("stage_name") if stage else None
    fixed_prior = stage.get("fixed_prior_objectives") if stage else None
    conditional = stage.get("conditional_on_unproven_incumbent") if stage else None
    scope_confirmed = bool(
        row.get("status") == "INFEASIBLE"
        and stage_name == "internal_repair_feasibility"
        and not fixed_prior
        and conditional is False
        and row.get("response_proto_hash")
        and row.get("final_assignment_available") is False
        and row.get("global_infeasibility_proven") is True
        and row.get("solver_global_infeasibility_proven") is True
    )
    return {
        "scenario_id": row["scenario_id"],
        "status": row.get("status"),
        "response_proto_hash": row.get("response_proto_hash"),
        "stage_name": stage_name,
        "fixed_prior_objectives": fixed_prior,
        "conditional_on_unproven_incumbent": conditional,
        "global_infeasibility_proven": row.get("global_infeasibility_proven"),
        "solver_global_infeasibility_proven": row.get("solver_global_infeasibility_proven"),
        "final_assignment_available": row.get("final_assignment_available"),
        "scope_confirmed": scope_confirmed,
    }


def rebuild_audited_normal_evaluation(source_dir: str | Path, output_dir: str | Path) -> dict[str, Any]:
    """Rebuild a corrected reporting layer for an already-completed Phase C
    normal-development evaluation using only the existing raw artifact on
    disk. Never solves anything, never re-imports the stable reference
    result, and never writes to the source artifact -- it is verified
    byte-for-byte unchanged before and after this function runs."""
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    if not source_dir.is_dir():
        raise CpSatEvaluationError(f"source artifact directory does not exist: {source_dir}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise CpSatEvaluationError(f"audited output is non-empty; refusing to overwrite: {output_dir}")

    before = _verify_sha256_manifest(source_dir)
    source_sha256sums_hash = _sha256_file(source_dir / "SHA256SUMS.txt")

    run_manifest = _read_json(source_dir / "run_manifest.json")
    if run_manifest.get("status") != "completed":
        raise CpSatEvaluationError("source artifact run_manifest.json is not status=completed")
    scenario_ids = run_manifest.get("selected_scenario_ids")
    if not scenario_ids:
        raise CpSatEvaluationError("source artifact run_manifest.json has no selected_scenario_ids")
    manifest_snapshot = _read_json(source_dir / "evaluation_manifest_snapshot.json")
    normal_root = Path(manifest_snapshot["source_normal_suite"]["artifact_dir"])

    rows: list[dict[str, Any]] = []
    stage_by_scenario: dict[str, dict[str, Any] | None] = {}
    for scenario_id in scenario_ids:
        scenario_dir = source_dir / "scenarios" / scenario_id
        summary = _read_completed_summary(scenario_dir / "scenario_summary.json")
        row = _repair_legacy_row(dict(summary["result"]))
        _validate_row_schema(row, scenario_id)
        trace_payload = _read_json(scenario_dir / "stage_trace.json")
        stages = trace_payload.get("stages", []) if isinstance(trace_payload, dict) else trace_payload
        stage_by_scenario[scenario_id] = stages[0] if stages else None
        rows.append(row)

    timing_audit_entries = []
    corrected_rows = []
    for row in rows:
        stage = stage_by_scenario.get(row["scenario_id"])
        diagnostic = _audit_row_timing(row, stage)
        timing_audit_entries.append({"scenario_id": row["scenario_id"], **diagnostic})
        corrected = dict(row)
        corrected.update(diagnostic)
        corrected_rows.append(corrected)

    infeasible_rows = [row for row in corrected_rows if row.get("status") == "INFEASIBLE"]
    infeasibility_scope_entries = [
        _infeasibility_scope_row(row, stage_by_scenario.get(row["scenario_id"]))
        for row in infeasible_rows
    ]
    unconfirmed = [entry["scenario_id"] for entry in infeasibility_scope_entries if not entry["scope_confirmed"]]
    if unconfirmed:
        raise CpSatEvaluationError("INFEASIBLE scope could not be confirmed for: " + ", ".join(unconfirmed))

    failures = _read_json(source_dir / "failures.json").get("failures", [])
    source_info = run_manifest.get("source_info", {})

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "scenario_results.csv", corrected_rows)
    _write_csv(output_dir / "timing_results.csv", [_timing_row(row) for row in corrected_rows])
    _write_csv(
        output_dir / "paired_recovery_vs_constrained_first.csv",
        _compute_paired_vs_constrained_first(normal_root, corrected_rows),
    )
    aggregate = _compute_aggregate_summary(corrected_rows, failures, source_info)
    _write_json(output_dir / "aggregate_summary.json", aggregate)
    readiness = _compute_success_gate(corrected_rows, failures)
    _write_json(output_dir / "readiness_assessment.json", readiness)
    _write_json(output_dir / "timing_anomaly_audit.json", {
        "tolerance_seconds": _TIMING_TOLERANCE_SECONDS,
        "scenarios": timing_audit_entries,
        "invalid_scenario_ids": sorted(
            entry["scenario_id"] for entry in timing_audit_entries if not entry["timing_diagnostic_valid"]
        ),
    })
    _write_json(output_dir / "infeasibility_scope_audit.json", {
        "stage_order": ["internal_repair_feasibility"],
        "scenarios": infeasibility_scope_entries,
        "all_scopes_confirmed": all(entry["scope_confirmed"] for entry in infeasibility_scope_entries),
    })
    _write_json(output_dir / "provenance.json", {
        "source_artifact_path": str(source_dir),
        "source_artifact_sha256sums_hash": source_sha256sums_hash,
        "source_artifact_tree_hash": before[0],
        "source_artifact_files": before[1],
        "source_artifact_directories": before[2],
        "source_artifact_bytes": before[3],
        "no_new_solver_runs": True,
        "normal_solver_runs_added": 0,
        "stress_runs": 0,
        "holdout_runs": 0,
        "rebuild_reads_only_existing_scenario_summary_json": True,
    })
    _write_checksums(output_dir)

    after = _verify_sha256_manifest(source_dir)
    if after != before:
        raise CpSatEvaluationCorrectnessError("source artifact was modified during the audited rebuild")

    return {
        "output_dir": str(output_dir),
        "aggregate_summary": aggregate,
        "readiness_assessment": readiness,
        "timing_anomaly_audit": {"invalid_scenario_ids": [e["scenario_id"] for e in timing_audit_entries if not e["timing_diagnostic_valid"]]},
        "infeasibility_scope_audit": {"all_scopes_confirmed": all(e["scope_confirmed"] for e in infeasibility_scope_entries)},
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the frozen Phase C 12-normal development evaluation.")
    parser.add_argument("--evaluation-manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--scenario-id")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--max-scenarios", type=int)
    parser.add_argument("--stable-probe-artifact-dir", default=str(DEFAULT_STABLE_PROBE_ARTIFACT))
    parser.add_argument("--audit-source-dir", help="Rebuild a corrected reporting layer from an existing raw artifact without solving")
    parser.add_argument("--audit-output-dir", help="Destination for the audited artifact")
    args = parser.parse_args(tuple(argv) if argv is not None else None)
    try:
        if args.audit_source_dir:
            if not args.audit_output_dir:
                raise CpSatEvaluationError("--audit-output-dir is required with --audit-source-dir")
            summary = rebuild_audited_normal_evaluation(args.audit_source_dir, args.audit_output_dir)
            print(
                "Phase C audited rebuild PASS: "
                f"{summary['aggregate_summary']['publishable_assignment_count']}/{summary['aggregate_summary']['attempted_count']} publishable, "
                f"gate={summary['readiness_assessment']['gate']}, no new solver runs"
            )
            return 0
        runner = CpSatNormalEvaluationRunner(args.evaluation_manifest, stable_probe_artifact_dir=args.stable_probe_artifact_dir)
        if args.dry_run:
            selected = runner.dry_run(args.scenario_id, args.max_scenarios)
            print(f"Phase C dry-run PASS: {len(selected)} scenario(s): {', '.join(selected)}")
            return 0
        if args.verify_only:
            verified = runner.run(args.output_dir, scenario_id=args.scenario_id, max_scenarios=args.max_scenarios, verify_only=True)
            print(f"Phase C verify PASS: {len(verified['verified_scenarios'])} scenario(s)")
            return 0
        summary = runner.run(args.output_dir, scenario_id=args.scenario_id, max_scenarios=args.max_scenarios, resume=args.resume)
        print(f"Phase C PASS: {summary['publishable_assignment_count']}/{summary['attempted_count']} publishable, gate={_read_json(Path(args.output_dir) / 'readiness_assessment.json')['gate']}")
        return 0
    except (CpSatEvaluationError, ValueError) as exc:
        print(f"Phase C normal evaluation failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
