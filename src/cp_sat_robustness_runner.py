"""Frozen, development-only CP-SAT robustness evaluation runner.

This module consumes the already persisted Phase A and Phase B inputs.  It
never regenerates students, sections, or Greedy results, and it calls the
production ``run_fair_cp_sat_solver`` entry point directly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import tempfile
import time
from collections import Counter
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import ortools

from src.allocation import (
    CpSatAllocationResult,
    CpSatSolveStatus,
    CpSatStageName,
    math_course_ids_from_catalog,
    run_fair_cp_sat_solver,
)
from src.benchmark_runner import _load_math_fallback_rules as _load_benchmark_math_fallback_rules
from src.experiment_manifest import (
    ExperimentManifestError,
    ExperimentSeeds,
    build_experiment_manifest,
    canonical_input_fingerprint,
    verify_experiment_manifest,
)
from src.infeasibility_certificates import validate_certificate
from src.final_schedule_policy import evaluate_final_schedule_policy


EVALUATION_SCHEMA_VERSION = 1
DEFAULT_MANIFEST = Path("data/scenarios/cp_sat_development_evaluation_v1.json")
DEFAULT_OUTPUT = Path(
    "/Users/klu/Projects/fair-course-allocation-artifacts/robustness-v1/cp-sat-development-v1"
)
REPO_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_GROUPS = ("normal", "stress", "negative", "all")
SOLVER_STAGE_ORDER = (
    "feasibility_bootstrap",
    "full_model_feasibility_incumbent",
    "math_coverage",
    "primary_unmet_count",
    "primary_unmet_period_units",
    "logical_schedule_completion",
    "alternate_rank_1",
    "alternate_rank_2",
    "alternate_rank_3",
    "fully_scheduled",
    "remaining_period_units",
    "seeded_tie_break",
)
OBJECTIVE_SENSE = {
    "math_coverage": "min",
    "primary_unmet_count": "min",
    "primary_unmet_period_units": "min",
    "logical_schedule_completion": "max",
    "alternate_rank_1": "max",
    "alternate_rank_2": "max",
    "alternate_rank_3": "max",
    "fully_scheduled": "max",
    "remaining_period_units": "min",
    "seeded_tie_break": "min",
}
GREEDY_ALGORITHMS = (
    "seeded_random_greedy",
    "first_come_first_served_greedy",
    "grade_priority_greedy",
    "constrained_first_greedy",
)


class CpSatEvaluationError(ValueError):
    """Raised when an evaluation cannot proceed without ambiguity."""


class CpSatEvaluationCorrectnessError(CpSatEvaluationError):
    """Raised when a solver response violates an integrity contract."""


@dataclass(frozen=True)
class SolverConfiguration:
    algorithm: str
    solver_seed: int
    workers: int
    bootstrap_time_limit_seconds: int
    per_stage_time_limit_seconds: int
    total_time_limit_seconds: int
    use_feasibility_bootstrap: bool
    use_constrained_first_hint: bool
    logical_schedule_completion_enabled: bool
    initial_solution_artifact_dir: None
    external_persisted_seed: bool
    stage_order: tuple[str, ...]
    objective_order: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationScenario:
    scenario_id: str
    group: str
    source_suite: str
    source_scenario_id: str
    paired_normal_scenario_id: str
    expected_feasibility: str


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _json_hash(payload: Any) -> str:
    return _sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CpSatEvaluationError(f"Cannot read JSON file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise CpSatEvaluationError(f"JSON root must be an object: {path}")
    return payload


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = tuple(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _solver_configuration(payload: dict[str, Any]) -> SolverConfiguration:
    raw = payload.get("solver_configuration")
    if not isinstance(raw, dict):
        raise CpSatEvaluationError("solver_configuration must be an object")
    required = {
        "algorithm", "solver_seed", "workers", "bootstrap_time_limit_seconds",
        "per_stage_time_limit_seconds", "total_time_limit_seconds",
        "use_feasibility_bootstrap", "use_constrained_first_hint",
        "logical_schedule_completion_enabled", "initial_solution_artifact_dir",
        "external_persisted_seed", "stage_order", "objective_order",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise CpSatEvaluationError("solver_configuration missing: " + ", ".join(missing))
    config = SolverConfiguration(
        algorithm=str(raw["algorithm"]),
        solver_seed=int(raw["solver_seed"]),
        workers=int(raw["workers"]),
        bootstrap_time_limit_seconds=int(raw["bootstrap_time_limit_seconds"]),
        per_stage_time_limit_seconds=int(raw["per_stage_time_limit_seconds"]),
        total_time_limit_seconds=int(raw["total_time_limit_seconds"]),
        use_feasibility_bootstrap=bool(raw["use_feasibility_bootstrap"]),
        use_constrained_first_hint=bool(raw["use_constrained_first_hint"]),
        logical_schedule_completion_enabled=bool(raw["logical_schedule_completion_enabled"]),
        initial_solution_artifact_dir=raw["initial_solution_artifact_dir"],
        external_persisted_seed=bool(raw["external_persisted_seed"]),
        stage_order=tuple(str(item) for item in raw["stage_order"]),
        objective_order=tuple(str(item) for item in raw["objective_order"]),
    )
    if config.algorithm != "cp_sat" or config.solver_seed != 20260630 or config.workers != 1:
        raise CpSatEvaluationError("evaluation solver configuration is not the frozen CP-SAT configuration")
    if (config.bootstrap_time_limit_seconds, config.per_stage_time_limit_seconds, config.total_time_limit_seconds) != (30, 30, 300):
        raise CpSatEvaluationError("evaluation time budgets are not the frozen 30/30/300 configuration")
    if not config.use_feasibility_bootstrap or not config.use_constrained_first_hint or not config.logical_schedule_completion_enabled:
        raise CpSatEvaluationError("required production CP-SAT stages or hint configuration are disabled")
    if config.initial_solution_artifact_dir is not None or config.external_persisted_seed:
        raise CpSatEvaluationError("external persisted seeds are forbidden in this evaluation")
    if config.stage_order != SOLVER_STAGE_ORDER:
        raise CpSatEvaluationError("evaluation stage_order does not match the production stage order")
    if config.objective_order != tuple(item for item in SOLVER_STAGE_ORDER if item != "feasibility_bootstrap" and item != "full_model_feasibility_incumbent"):
        raise CpSatEvaluationError("evaluation objective_order does not match the production objective order")
    return config


def load_evaluation_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    payload = _read_json(Path(path))
    required = {
        "evaluation_name", "evaluation_version", "source_normal_suite", "source_stress_suite",
        "source_git_commit", "split", "solver_configuration", "scenario_groups", "scenarios",
        "holdout_execution_allowed", "tuning_allowed", "notes",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise CpSatEvaluationError("evaluation manifest missing: " + ", ".join(missing))
    if payload["split"] != "development" or payload["holdout_execution_allowed"] is not False or payload["tuning_allowed"] is not False:
        raise CpSatEvaluationError("evaluation manifest must be development-only and holdout/tuning disabled")
    _solver_configuration(payload)
    groups = payload["scenario_groups"]
    if not isinstance(groups, dict) or set(groups) != {"normal", "stress", "negative"}:
        raise CpSatEvaluationError("scenario_groups must contain exactly normal, stress, and negative")
    scenarios = payload["scenarios"]
    if not isinstance(scenarios, list):
        raise CpSatEvaluationError("scenarios must be a list")
    parsed: dict[str, EvaluationScenario] = {}
    for raw in scenarios:
        if not isinstance(raw, dict):
            raise CpSatEvaluationError("each evaluation scenario must be an object")
        fields = {"scenario_id", "group", "source_suite", "source_scenario_id", "paired_normal_scenario_id", "expected_feasibility"}
        if fields - set(raw):
            raise CpSatEvaluationError("evaluation scenario missing: " + ", ".join(sorted(fields - set(raw))))
        scenario = EvaluationScenario(**{field: str(raw[field]) for field in fields})
        if scenario.scenario_id in parsed:
            raise CpSatEvaluationError(f"duplicate evaluation scenario: {scenario.scenario_id}")
        if scenario.group not in {"normal", "stress", "negative"} or scenario.source_suite not in {"normal", "stress"}:
            raise CpSatEvaluationError(f"invalid group/source suite: {scenario.scenario_id}")
        if scenario.expected_feasibility not in {"unknown", "structurally_infeasible"}:
            raise CpSatEvaluationError(f"invalid expected_feasibility: {scenario.scenario_id}")
        if scenario.group == "negative" and scenario.expected_feasibility != "structurally_infeasible":
            raise CpSatEvaluationError("negative scenarios must be structurally_infeasible")
        if scenario.group != "negative" and scenario.expected_feasibility != "unknown":
            raise CpSatEvaluationError("ordinary scenarios must have expected_feasibility=unknown")
        if "holdout" in scenario.scenario_id:
            raise CpSatEvaluationError("holdout scenarios are forbidden in the evaluation manifest")
        parsed[scenario.scenario_id] = scenario
    group_ids = {group: [str(item) for item in groups[group]] for group in groups}
    if set(parsed) != set(sum(group_ids.values(), [])):
        raise CpSatEvaluationError("scenario_groups and scenarios disagree")
    if any(len(group_ids[group]) != len(set(group_ids[group])) for group in group_ids):
        raise CpSatEvaluationError("scenario_groups contain duplicate IDs")
    if tuple(group_ids["normal"]) != tuple(s.scenario_id for s in parsed.values() if s.group == "normal"):
        raise CpSatEvaluationError("scenario order does not match the scenarios list")
    if len(group_ids["normal"]) != 12 or len(group_ids["stress"]) != 12 or len(group_ids["negative"]) != 3:
        raise CpSatEvaluationError("evaluation must contain 12 normal, 12 stress, and 3 negative scenarios")
    normal_ids = set(group_ids["normal"])
    for scenario in parsed.values():
        if scenario.paired_normal_scenario_id not in normal_ids:
            raise CpSatEvaluationError(f"missing paired normal scenario: {scenario.scenario_id}")
    return payload


def evaluation_manifest_hash(payload_or_path: dict[str, Any] | str | Path) -> str:
    if isinstance(payload_or_path, dict):
        return _json_hash(payload_or_path)
    return _sha256_file(Path(payload_or_path))


def _verify_sha256_manifest(root: Path) -> tuple[str, int, int, int]:
    checksum_path = root / "SHA256SUMS.txt"
    if not root.is_dir() or not checksum_path.is_file():
        raise CpSatEvaluationError(f"source artifact is missing root or SHA256SUMS.txt: {root}")
    expected: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or "  " not in line:
            raise CpSatEvaluationError(f"malformed checksum line in {checksum_path}")
        digest, relative = line.split("  ", 1)
        expected[relative] = digest
    files = [path for path in root.rglob("*") if path.is_file() and path != checksum_path]
    actual = {str(path.relative_to(root)): _sha256_file(path) for path in files}
    if set(expected) != set(actual):
        raise CpSatEvaluationError(f"checksum file set mismatch for {root}")
    bad = [name for name in actual if actual[name] != expected[name]]
    if bad:
        raise CpSatEvaluationError(f"SHA256 mismatch in {root}: {bad[:3]}")
    tree_hash = _json_hash(sorted(actual.items()))
    directories = [path for path in root.rglob("*") if path.is_dir()]
    return tree_hash, len(files) + 1, len(directories), sum(path.stat().st_size for path in files) + checksum_path.stat().st_size


def _verify_source_suite(payload: dict[str, Any], key: str) -> dict[str, Any]:
    source = payload["source_normal_suite" if key == "normal" else "source_stress_suite"]
    root = Path(source["artifact_dir"])
    manifest_path = root / "suite_manifest_snapshot.json"
    sums_path = root / "SHA256SUMS.txt"
    if _sha256_file(manifest_path) != source["manifest_sha256"]:
        raise CpSatEvaluationError(f"{key} source suite manifest hash mismatch")
    if _sha256_file(sums_path) != source["sha256sums_sha256"]:
        raise CpSatEvaluationError(f"{key} source suite SHA256SUMS hash mismatch")
    tree_hash, files, directories, bytes_total = _verify_sha256_manifest(root)
    return {
        "artifact_dir": str(root),
        "manifest_sha256": source["manifest_sha256"],
        "sha256sums_sha256": source["sha256sums_sha256"],
        "tree_hash": tree_hash,
        "files": files,
        "directories": directories,
        "bytes": bytes_total,
    }


def _source_root(payload: dict[str, Any], source_suite: str) -> Path:
    return Path(payload["source_normal_suite" if source_suite == "normal" else "source_stress_suite"]["artifact_dir"])


def _scenario_from_payload(payload: dict[str, Any], scenario_id: str) -> EvaluationScenario:
    for raw in payload["scenarios"]:
        if raw["scenario_id"] == scenario_id:
            fields = ("scenario_id", "group", "source_suite", "source_scenario_id", "paired_normal_scenario_id", "expected_feasibility")
            return EvaluationScenario(**{field: str(raw[field]) for field in fields})
    raise CpSatEvaluationError(f"unknown evaluation scenario: {scenario_id}")


def _load_scenario_input(
    payload: dict[str, Any],
    scenario: EvaluationScenario,
    config_dir: Path,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    root = _source_root(payload, scenario.source_suite) / "scenarios" / scenario.source_scenario_id
    scenario_result = _read_json(root / "scenario_result.json")
    if scenario_result.get("status") != "completed":
        raise CpSatEvaluationError(f"source scenario is not completed: {scenario.scenario_id}")
    generated = root / "generated"
    sections = root / "sections"
    generation_metadata = _read_json(generated / "generation_metadata.json")
    section_metadata = _read_json(sections / "section_planning_metadata.json")
    data_seed = int(generation_metadata.get("data_generation_seed", generation_metadata.get("seed")))
    section_seed = int(section_metadata["seed"])
    seeds = ExperimentSeeds(data_seed, section_seed, int(payload["solver_configuration"]["solver_seed"]))
    try:
        manifest = build_experiment_manifest(
            generated,
            sections,
            config_dir,
            scenario_id=str(generation_metadata.get("scenario_id", "stable_year")),
            seeds=seeds,
            repo_root=REPO_ROOT,
        )
        allocation_input = verify_experiment_manifest(
            manifest,
            config_dir=config_dir,
            repo_root=REPO_ROOT,
            require_same_git=True,
        )
    except ExperimentManifestError as exc:
        raise CpSatEvaluationError(f"source scenario provenance failed: {scenario.scenario_id}: {exc}") from exc
    actual = asdict(canonical_input_fingerprint(allocation_input))
    expected = scenario_result.get("input_fingerprint")
    if actual != expected:
        raise CpSatEvaluationError(f"source scenario fingerprint mismatch: {scenario.scenario_id}")
    return allocation_input, manifest.to_dict(), scenario_result


def _stage_sense(stage_name: str) -> str | None:
    if stage_name in {"feasibility_bootstrap", "full_model_feasibility_incumbent"}:
        return None
    return OBJECTIVE_SENSE.get(stage_name)


def _stage_reason(status: str, skipped: bool, skip_reason: str, optimum: bool) -> str:
    if skipped:
        return skip_reason or "skipped"
    if status == "OPTIMAL":
        return "optimal_proven" if optimum else "optimal_status"
    if status == "FEASIBLE":
        return "incumbent_found_not_proven"
    if status == "INFEASIBLE":
        return "infeasible"
    if status == "UNKNOWN":
        return "unknown_no_incumbent"
    return status.lower()


def _stage_trace(result: CpSatAllocationResult, scenario_id: str) -> list[dict[str, Any]]:
    trace = []
    for index, diagnostic in enumerate(result.stage_diagnostics):
        status = diagnostic.status.value
        response_hash = diagnostic.response_proto_hash or ""
        token = _json_hash({"scenario_id": scenario_id, "index": index, "stage": diagnostic.stage_name.value, "response": response_hash})
        trace.append({
            "stage_name": diagnostic.stage_name.value,
            "stage_run_token": token,
            "skipped": diagnostic.skipped,
            "executed": not diagnostic.skipped,
            "status": status,
            "wall_time_seconds": diagnostic.wall_time_seconds,
            "deterministic_time_seconds": None,
            "objective_value": diagnostic.objective_value,
            "best_objective_bound": diagnostic.best_objective_bound,
            "objective_sense": _stage_sense(diagnostic.stage_name.value),
            "objective_descriptor_hash": diagnostic.objective_descriptor_hash or "",
            "response_proto_hash": response_hash,
            "incumbent_found": status in {"FEASIBLE", "OPTIMAL"},
            "incumbent_source": "solver_response" if response_hash else None,
            "hint_source": result.model_stats.bootstrap_hint_strategy if diagnostic.stage_name == CpSatStageName.FEASIBILITY_BOOTSTRAP else result.model_stats.hint_source,
            "hint_variable_count": result.model_stats.hint_variables_supplied if diagnostic.stage_name != CpSatStageName.FEASIBILITY_BOOTSTRAP else None,
            "branch_count": diagnostic.branches,
            "conflict_count": diagnostic.conflicts,
            "fixed_prior_objectives": [
                {"stage_name": name.value, "value": value}
                for name, value in diagnostic.fixed_higher_priority_values
            ],
            "budget_remaining_at_start_seconds": diagnostic.remaining_global_budget_at_start_seconds,
            "effective_time_limit_seconds": diagnostic.effective_time_limit_seconds,
            "termination_reason": _stage_reason(status, diagnostic.skipped, diagnostic.skip_reason, diagnostic.optimum_proven),
            "conditional_on_unproven_incumbent": diagnostic.conditional_on_unproven_incumbent,
        })
    return trace


def _assignment_hash(result: CpSatAllocationResult) -> str | None:
    if not result.assignments:
        return None
    rows = [
        {
            "request_key": item.request_key,
            "linked_section_group_id": item.linked_section_group_id,
            "assignment_key": item.assignment_key,
        }
        for item in result.assignments
    ]
    return _json_hash(sorted(rows, key=lambda row: (row["request_key"], row["linked_section_group_id"])))


def _final_stage(trace: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates = [
        row for row in trace
        if row["stage_name"] not in {"feasibility_bootstrap"} and row["response_proto_hash"] and row["incumbent_found"]
    ]
    return candidates[-1] if candidates else None


def _null_metrics(allocation_input: Any) -> dict[str, Any]:
    return {
        "student_count": len(allocation_input.students),
        "primary_assigned": None, "primary_unmet": None, "primary_satisfaction_rate": None,
        "primary_unmet_period_units": None, "protected_primary_unmet": None,
        "ordinary_max_primary_unmet": None, "ordinary_policy_violations": None,
        "protected_policy_violations": None, "high_demand_violations": None,
        "target_logical_total": sum(student.target_period_units for student in allocation_input.students),
        "assigned_logical_total": None, "total_logical_gap": None,
        "logical_fully_scheduled_students": None, "logical_full_rate": None,
        "gap_1_students": None, "gap_over_1_students": None, "below_five_students": None,
        "over_target_students": None, "logical_objective": None, "logical_best_bound": None,
        "objective_to_bound_gap": None, "theoretical_maximum": sum(student.target_period_units for student in allocation_input.students),
        "alternates_assigned": None, "alternate_rank_1": None, "alternate_rank_2": None, "alternate_rank_3": None,
        "math_soft_policy_violations": None, "capacity_utilization": None,
        "final_schedule_policy_pass": None, "consistency_issue_count": None,
        "policy_violation_count": None,
        "assignment_nonpublishable": True,
        "certificate_valid": None,
    }


def _result_metrics(result: CpSatAllocationResult, allocation_input: Any) -> dict[str, Any]:
    if not result.student_outcomes:
        return _null_metrics(allocation_input)
    students = result.student_outcomes
    primary_outcomes = [outcome for outcome in result.request_outcomes if outcome.request_type == "primary"]
    primary_assigned = sum(outcome.status.value == "assigned" for outcome in primary_outcomes)
    policy = evaluate_final_schedule_policy(result.algorithm_name, students)
    logical_assigned = sum(int(outcome.assigned_logical_course_count or 0) for outcome in students)
    target_total = sum(int(outcome.target_logical_course_count or outcome.target_period_units) for outcome in students)
    logical_gaps = [int(outcome.logical_schedule_gap_count or 0) for outcome in students]
    logical_stage = next((row for row in result.stage_diagnostics if row.stage_name == CpSatStageName.LOGICAL_SCHEDULE_COMPLETION), None)
    logical_objective = result.model_stats.logical_schedule_completion_objective_value
    if logical_objective is None:
        logical_objective = result.objective_values.logical_assigned_course_count
    logical_bound = result.model_stats.logical_schedule_completion_best_bound
    if logical_bound is None and logical_stage is not None:
        logical_bound = logical_stage.best_objective_bound
    capacity = sum(row.capacity for row in result.section_roster_summary)
    assigned_seats = sum(row.assigned_count for row in result.section_roster_summary)
    protected_unmet = sum(outcome.priority_protected and outcome.primary_unmet_count > 0 for outcome in students)
    ordinary_max = max((outcome.primary_unmet_count for outcome in students if not outcome.priority_protected), default=0)
    rank_counts = Counter(
        outcome.alternate_rank
        for outcome in result.request_outcomes
        if outcome.request_type == "alternate" and outcome.status.value == "assigned"
    )
    return {
        "student_count": len(students),
        "primary_assigned": primary_assigned,
        "primary_unmet": len(primary_outcomes) - primary_assigned,
        "primary_satisfaction_rate": round(primary_assigned / len(primary_outcomes), 6) if primary_outcomes else None,
        "primary_unmet_period_units": sum(outcome.period_units for outcome in primary_outcomes if outcome.status.value != "assigned"),
        "protected_primary_unmet": protected_unmet,
        "ordinary_max_primary_unmet": ordinary_max,
        "ordinary_policy_violations": len(result.policy_report.ordinary_violation_student_ids) if result.policy_report else 0,
        "protected_policy_violations": len(result.policy_report.protected_violation_student_ids) if result.policy_report else 0,
        "high_demand_violations": result.policy_report.high_demand_violation_count if result.policy_report else 0,
        "target_logical_total": target_total,
        "assigned_logical_total": logical_assigned,
        "total_logical_gap": sum(logical_gaps),
        "logical_fully_scheduled_students": sum(gap == 0 for gap in logical_gaps),
        "logical_full_rate": round(sum(gap == 0 for gap in logical_gaps) / len(students), 6),
        "gap_1_students": sum(gap == 1 for gap in logical_gaps),
        "gap_over_1_students": sum(gap > 1 for gap in logical_gaps),
        "below_five_students": sum(int(outcome.assigned_logical_course_count or 0) < 5 for outcome in students),
        "over_target_students": sum(int(outcome.assigned_logical_course_count or 0) > int(outcome.target_logical_course_count or outcome.target_period_units) for outcome in students),
        "logical_objective": logical_objective,
        "logical_best_bound": logical_bound,
        "objective_to_bound_gap": (logical_bound - logical_objective) if logical_objective is not None and logical_bound is not None else None,
        "theoretical_maximum": target_total,
        "alternates_assigned": sum(rank_counts.values()),
        "alternate_rank_1": rank_counts.get(1, 0),
        "alternate_rank_2": rank_counts.get(2, 0),
        "alternate_rank_3": rank_counts.get(3, 0),
        "math_soft_policy_violations": len(result.math_policy_report.current_math_coverage_violation_student_ids) if result.math_policy_report else 0,
        "capacity_utilization": round(assigned_seats / capacity, 6) if capacity else None,
        "final_schedule_policy_pass": policy.summary.final_schedule_policy_pass,
        "policy_violation_count": policy.summary.violating_student_count,
        "consistency_issue_count": len(result.consistency_issues),
        "assignment_nonpublishable": not policy.summary.final_schedule_policy_pass or bool(result.consistency_issues),
        "certificate_valid": None,
    }


def _validate_objective_bounds(trace: list[dict[str, Any]], theoretical_maximum: int) -> None:
    for row in trace:
        objective = row["objective_value"]
        bound = row["best_objective_bound"]
        if objective is None or bound is None:
            continue
        sense = row["objective_sense"]
        if sense == "max" and objective > bound:
            raise CpSatEvaluationCorrectnessError(f"max objective exceeds best bound in {row['stage_name']}")
        if sense == "min" and bound > objective:
            raise CpSatEvaluationCorrectnessError(f"min best bound exceeds objective in {row['stage_name']}")
        if row["stage_name"] == "logical_schedule_completion" and (objective > theoretical_maximum or bound > theoretical_maximum):
            raise CpSatEvaluationCorrectnessError("logical completion objective/bound exceeds theoretical maximum")


def _validate_result(result: CpSatAllocationResult, metrics: dict[str, Any], scenario: EvaluationScenario) -> None:
    status = result.solve_status.value
    if status == "MODEL_INVALID":
        raise CpSatEvaluationCorrectnessError(f"MODEL_INVALID in {scenario.scenario_id}")
    assignment = bool(result.student_outcomes)
    if status in {"FEASIBLE", "OPTIMAL"} and not assignment:
        raise CpSatEvaluationCorrectnessError(f"{status} has no final assignment in {scenario.scenario_id}")
    if assignment and status not in {"FEASIBLE", "OPTIMAL"}:
        raise CpSatEvaluationCorrectnessError(f"assignment exists with non-solution status {status}")
    if scenario.group == "negative":
        if status in {"FEASIBLE", "OPTIMAL"} or assignment:
            raise CpSatEvaluationCorrectnessError(f"negative scenario produced a publishable-looking result: {scenario.scenario_id}")
    elif assignment:
        if metrics["final_schedule_policy_pass"] is not True or metrics["consistency_issue_count"] != 0:
            raise CpSatEvaluationCorrectnessError(f"ordinary CP-SAT assignment failed policy or consistency: {scenario.scenario_id}")


def _greedy_rows(scenario_dir: Path) -> dict[str, dict[str, Any]]:
    path = scenario_dir / "benchmark" / "algorithm_summary.csv"
    if not path.is_file():
        raise CpSatEvaluationError(f"Greedy benchmark summary is missing: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {row["algorithm_name"]: row for row in rows}
    if set(result) != set(GREEDY_ALGORITHMS):
        raise CpSatEvaluationError(f"Greedy result universe mismatch: {path}")
    return result


def _number(row: dict[str, Any], key: str, default: Any = None) -> Any:
    value = row.get(key, "")
    if value in ("", None):
        return default
    try:
        return float(value) if "." in str(value) else int(value)
    except ValueError:
        return default


def _paired_rows(metrics: dict[str, Any], greedy: dict[str, dict[str, Any]] | None, scenario: EvaluationScenario, students: int) -> list[dict[str, Any]]:
    if greedy is None:
        return []
    rows = []
    for algorithm in GREEDY_ALGORITHMS:
        base = greedy[algorithm]
        rows.append({
            "scenario_id": scenario.scenario_id,
            "scenario_group": scenario.group,
            "algorithm": algorithm,
            "cp_sat_status": metrics["status"],
            "cp_sat_assignment_available": metrics["final_assignment_available"],
            "primary_assigned_delta": _delta(metrics["primary_assigned"], _number(base, "primary_assigned")),
            "primary_satisfaction_delta": _delta(metrics["primary_satisfaction_rate"], _number(base, "primary_satisfaction_rate")),
            "logical_full_students_delta": _delta(metrics["logical_fully_scheduled_students"], _number(base, "logical_fully_scheduled_students")),
            "logical_full_rate_delta": _delta(metrics["logical_full_rate"], _ratio(_number(base, "logical_fully_scheduled_students"), students)),
            "total_gap_delta": _delta(metrics["total_logical_gap"], _number(base, "total_logical_schedule_gap")),
            "gap_over_1_delta": _delta(metrics["gap_over_1_students"], _number(base, "students_with_schedule_gap_over_limit")),
            "below_five_delta": _delta(metrics["below_five_students"], _number(base, "students_below_minimum_course_count")),
            "policy_violation_delta": _delta(metrics["policy_violation_count"], _number(base, "final_schedule_policy_violation_students")),
        })
    return rows


def _delta(left: Any, right: Any) -> Any:
    return left - right if left is not None and right is not None else None


def _ratio(value: Any, denominator: int) -> float | None:
    return round(value / denominator, 6) if value is not None and denominator else None


def _scenario_row(scenario: EvaluationScenario, result: CpSatAllocationResult, metrics: dict[str, Any], trace: list[dict[str, Any]], manifest: dict[str, Any], input_fingerprint: dict[str, Any], runtime: float) -> dict[str, Any]:
    final_stage = _final_stage(trace)
    row = {
        "scenario_id": scenario.scenario_id,
        "scenario_group": scenario.group,
        "expected_feasibility": scenario.expected_feasibility,
        "status": result.solve_status.value,
        "final_assignment_available": bool(result.student_outcomes),
        "assignment_nonpublishable": metrics["assignment_nonpublishable"],
        "runtime_seconds": round(runtime, 6),
        "completed_stages": sum(not item["skipped"] for item in trace),
        "failed_or_incomplete_stage": next((item["stage_name"] for item in trace if item["status"] in {"UNKNOWN", "INFEASIBLE", "MODEL_INVALID"} or item["skipped"]), None),
        "final_assignment_source_stage": final_stage["stage_name"] if final_stage and result.student_outcomes else None,
        "final_response_proto_hash": final_stage["response_proto_hash"] if final_stage and result.student_outcomes else None,
        "final_objective_descriptor_hash": final_stage["objective_descriptor_hash"] if final_stage and result.student_outcomes else None,
        "assignment_hash": _assignment_hash(result),
        "solver_seed": manifest["solver_configuration"]["solver_seed"],
        "workers": manifest["solver_configuration"]["workers"],
        "total_time_budget_seconds": manifest["solver_configuration"]["total_time_limit_seconds"],
        "ortools_version": getattr(ortools, "__version__", "unknown"),
        "python_version": platform.python_version(),
        "source_git_commit": manifest["source_git_commit"],
        "data_generation_seed": input_fingerprint.get("data_generation_seed"),
        "section_planning_seed": input_fingerprint.get("section_planning_seed"),
        **{key: value for key, value in metrics.items() if key not in {"student_count"}},
        "students": input_fingerprint["students"],
        "logical_requests": input_fingerprint["logical_requests"],
        "logical_primaries": input_fingerprint["logical_primaries"],
        "alternates": input_fingerprint["alternates"],
        "logical_sections": input_fingerprint["logical_sections"],
        "section_rows": input_fingerprint["section_rows"],
        "candidate_edges": input_fingerprint["candidate_edges"],
        "canonical_input_hash": input_fingerprint["canonical_input_hash"],
    }
    return row


def _grade_rows(allocation_input: Any, result: CpSatAllocationResult, scenario_id: str) -> list[dict[str, Any]]:
    by_grade = Counter(student.grade for student in allocation_input.students)
    outcomes = {student.grade: [row for row in result.student_outcomes if row.grade == student.grade] for student in allocation_input.students}
    rows = []
    for grade in sorted(by_grade):
        group = outcomes.get(grade, [])
        if not group:
            rows.append({"scenario_id": scenario_id, "grade": grade, "student_count": by_grade[grade], "primary_satisfaction_rate": None, "logical_full_rate": None, "mean_gap": None, "gap_over_1_count": None, "below_five_count": None, "policy_violation_count": None})
            continue
        gaps = [int(row.logical_schedule_gap_count or 0) for row in group]
        primary_total = sum(row.primary_request_count for row in group)
        primary_assigned = sum(row.primary_assigned_count for row in group)
        rows.append({
            "scenario_id": scenario_id,
            "grade": grade,
            "student_count": len(group),
            "primary_satisfaction_rate": round(primary_assigned / primary_total, 6) if primary_total else None,
            "logical_full_rate": round(sum(gap == 0 for gap in gaps) / len(group), 6),
            "mean_gap": round(sum(gaps) / len(group), 6),
            "gap_over_1_count": sum(gap > 1 for gap in gaps),
            "below_five_count": sum(int(row.assigned_logical_course_count or 0) < 5 for row in group),
            "policy_violation_count": sum(
                bool(row.ordinary_fairness_violation or row.protected_fairness_violation or row.logical_schedule_gap_count is not None and row.logical_schedule_gap_count > 1)
                for row in group
            ),
        })
    return rows


def _aggregate(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    def values(key: str) -> list[float]:
        return sorted(float(row[key]) for row in rows if row.get(key) is not None)
    def stats(key: str) -> dict[str, Any]:
        items = values(key)
        if not items:
            return {"count": 0, "median": None, "min": None, "max": None}
        middle = len(items) // 2
        median = items[middle] if len(items) % 2 else (items[middle - 1] + items[middle]) / 2
        return {"count": len(items), "median": round(median, 6), "min": round(items[0], 6), "max": round(items[-1], 6)}
    status_counts = Counter(row["status"] for row in rows)
    assigned = [row for row in rows if row.get("final_assignment_available")]
    return {
        "scope": label,
        "scenarios_attempted": len(rows),
        "status_counts": dict(sorted(status_counts.items())),
        "final_assignment_count": len(assigned),
        "final_assignment_rate": round(len(assigned) / len(rows), 6) if rows else None,
        "policy_pass_count": sum(row.get("final_schedule_policy_pass") is True for row in rows),
        "policy_pass_rate_all": round(sum(row.get("final_schedule_policy_pass") is True for row in rows) / len(rows), 6) if rows else None,
        "policy_pass_rate_with_assignment": round(sum(row.get("final_schedule_policy_pass") is True for row in assigned) / len(assigned), 6) if assigned else None,
        "primary_satisfaction_rate": stats("primary_satisfaction_rate"),
        "logical_full_rate": stats("logical_full_rate"),
        "total_logical_gap": stats("total_logical_gap"),
        "runtime_seconds": stats("runtime_seconds"),
        "objective_to_bound_gap": stats("objective_to_bound_gap"),
    }


def _stage_scope(stage_name: str) -> str:
    if stage_name == "feasibility_bootstrap":
        return "bootstrap"
    if stage_name == "full_model_feasibility_incumbent":
        return "full_hard_model"
    if stage_name in {"math_coverage", "primary_unmet_count", "primary_unmet_period_units"}:
        return "core_model"
    return "enrichment_model"


def _audit_status_semantics(
    scenario: EvaluationScenario,
    raw_result: dict[str, Any],
    trace: list[dict[str, Any]],
) -> dict[str, Any]:
    """Classify an existing trace without rerunning or reinterpreting CP-SAT."""
    executed = [row for row in trace if not row.get("skipped")]
    terminal = executed[-1] if executed else None
    first_infeasible = next((row for row in executed if row.get("status") == "INFEASIBLE"), None)
    full_model = next(
        (row for row in trace if row.get("stage_name") == "full_model_feasibility_incumbent"),
        None,
    )
    full_model_infeasible = bool(full_model and full_model.get("status") == "INFEASIBLE")
    full_model_incumbent = bool(full_model and full_model.get("status") in {"FEASIBLE", "OPTIMAL"})
    publishable = bool(raw_result.get("final_assignment_available"))
    certificate_valid = raw_result.get("certificate_valid") is True
    any_incumbent = any(row.get("incumbent_found") is True for row in executed)
    first_stage = first_infeasible.get("stage_name") if first_infeasible else None
    first_fixed = first_infeasible.get("fixed_prior_objectives", []) if first_infeasible else []
    fixed_objective_infeasible_stages = [
        row["stage_name"]
        for row in executed
        if row.get("status") == "INFEASIBLE" and row.get("fixed_prior_objectives")
    ]

    if certificate_valid:
        outcome = "STRUCTURAL_CERTIFICATE_INFEASIBLE"
        scope = "structural_certificate"
        explanation = (
            "The source scenario has a valid structural infeasibility certificate. "
            "The certificate is kept separate from any CP-SAT full-model proof."
        )
    elif full_model_infeasible:
        outcome = "GLOBAL_INFEASIBLE"
        scope = "full_hard_model"
        explanation = "The full hard-model feasibility stage returned INFEASIBLE."
    elif first_infeasible:
        if first_stage == "feasibility_bootstrap":
            outcome = "BOOTSTRAP_STAGE_INFEASIBLE"
            scope = "unknown"
            explanation = (
                "The bootstrap subset returned INFEASIBLE, but the full hard-model "
                "feasibility stage did not prove global infeasibility."
            )
        elif first_fixed:
            outcome = "LEXICOGRAPHIC_STAGE_INFEASIBLE"
            scope = "fixed_objective_stage"
            explanation = (
                "A later stage returned INFEASIBLE under fixed prior objective values; "
                "this is not a proof that the original hard model is infeasible."
            )
        else:
            outcome = "CORE_STAGE_INFEASIBLE"
            scope = "unknown"
            explanation = (
                "A core objective stage returned INFEASIBLE, but no full hard-model "
                "feasibility proof was recorded."
            )
    elif publishable:
        outcome = "PUBLISHABLE_ASSIGNMENT"
        scope = "none"
        explanation = "A final solver assignment was exported and passed the evaluation gate."
    elif full_model_incumbent:
        outcome = "NO_FINAL_ASSIGNMENT_AFTER_INCUMBENT"
        scope = "none"
        explanation = "A full-model incumbent existed, but no final publishable assignment was exported."
    else:
        outcome = "UNKNOWN_NO_FINAL_ASSIGNMENT"
        scope = "unknown"
        explanation = "No publishable assignment or full-model infeasibility proof was recorded."

    stage_statuses = {
        row["stage_name"]: row.get("status")
        for row in trace
        if row.get("stage_name")
    }
    return {
        "raw_result_status": raw_result.get("status"),
        "raw_terminal_solver_status": terminal.get("status") if terminal else None,
        "terminal_stage": terminal.get("stage_name") if terminal else None,
        "terminal_stage_scope": _stage_scope(terminal["stage_name"]) if terminal else None,
        "evaluation_outcome": outcome,
        "global_infeasibility_proven": full_model_infeasible,
        "solver_global_infeasibility_proven": full_model_infeasible,
        "infeasibility_scope": scope,
        "complete_incumbent_found": full_model_incumbent or publishable,
        "full_model_incumbent_found": full_model_incumbent,
        "any_stage_incumbent_found": any_incumbent,
        "publishable_assignment_available": publishable,
        "assignment_nonpublishable": bool(raw_result.get("assignment_nonpublishable", not publishable)),
        "certificate_proof_valid": certificate_valid,
        "first_infeasible_stage": first_stage,
        "first_infeasible_fixed_prior_objectives": first_fixed,
        "fixed_objective_infeasible_stages": fixed_objective_infeasible_stages,
        "fixed_objective_infeasible_stage_count": len(fixed_objective_infeasible_stages),
        "full_model_feasibility_status": full_model.get("status") if full_model else None,
        "stage_statuses": stage_statuses,
        "explanation": explanation,
        "scenario_id": scenario.scenario_id,
        "scenario_group": scenario.group,
    }


def _audited_result_row(raw_result: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    row = dict(raw_result)
    row.update({key: value for key, value in audit.items() if key not in {"scenario_id", "scenario_group", "stage_statuses"}})
    row["scenario_id"] = audit["scenario_id"]
    row["scenario_group"] = audit["scenario_group"]
    for stage_name, status in audit["stage_statuses"].items():
        row[f"stage_status_{stage_name}"] = status
    return row


def _audit_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "scenarios": len(rows),
        "raw_terminal_status_counts": dict(sorted(Counter(row["raw_terminal_solver_status"] for row in rows).items())),
        "evaluation_outcome_counts": dict(sorted(Counter(row["evaluation_outcome"] for row in rows).items())),
        "infeasibility_scope_counts": dict(sorted(Counter(row["infeasibility_scope"] for row in rows).items())),
        "full_hard_model_proven_infeasible_count": sum(row["global_infeasibility_proven"] for row in rows),
        "fixed_objective_stage_infeasible_count": sum(row["fixed_objective_infeasible_stage_count"] for row in rows),
        "bootstrap_stage_infeasible_unproven_count": sum(row["evaluation_outcome"] == "BOOTSTRAP_STAGE_INFEASIBLE" for row in rows),
        "core_stage_infeasible_unproven_count": sum(row["evaluation_outcome"] == "CORE_STAGE_INFEASIBLE" for row in rows),
        "unknown_no_final_assignment_count": sum(row["evaluation_outcome"] == "UNKNOWN_NO_FINAL_ASSIGNMENT" for row in rows),
        "complete_incumbent_count": sum(row["complete_incumbent_found"] for row in rows),
        "publishable_assignment_count": sum(row["publishable_assignment_available"] for row in rows),
    }


def _audited_readiness(
    rows: list[dict[str, Any]],
    negative: list[dict[str, Any]],
    source_info: dict[str, Any],
    failures: list[dict[str, Any]],
) -> dict[str, Any]:
    normal = [row for row in rows if row["scenario_group"] == "normal"]
    stress = [row for row in rows if row["scenario_group"] == "stress"]
    attempted = len(normal)
    publishable = sum(bool(row["publishable_assignment_available"]) for row in normal)
    no_assignment = attempted - publishable
    majority = attempted > 0 and no_assignment > attempted / 2
    blocking: list[str] = []
    if attempted != 12:
        blocking.append("not_all_normal_development_scenarios_attempted")
    if len(stress) != 12:
        blocking.append("not_all_stress_development_scenarios_attempted")
    if failures:
        blocking.append("critical_or_runner_failures_present")
    if any(row["raw_result_status"] in {"FEASIBLE", "OPTIMAL"} for row in negative):
        blocking.append("negative_scenario_returned_feasible_or_optimal")
    if majority:
        blocking.extend([
            "zero_publishable_assignments_across_normal_development_scenarios" if publishable == 0 else "majority_normal_scenarios_have_no_publishable_assignment",
            "cold_start_solver_failed_to_establish_usable_incumbents_under_frozen_300_second_budget",
            "development_evaluation_does_not_yet_support_a_holdout_test",
        ])
    return {
        "ready_for_holdout": not blocking,
        "blocking_reasons": blocking,
        "normal_scenarios_attempted": attempted,
        "normal_publishable_assignments": publishable,
        "normal_publishable_assignment_rate": round(publishable / attempted, 6) if attempted else None,
        "normal_no_assignment_count": no_assignment,
        "majority_normal_without_assignment": majority,
        "stress_scenarios_attempted": len(stress),
        "negative_scenarios_attempted": len(negative),
        "holdout_runs": 0,
        "source_artifact_hashes_verified": bool(source_info),
        "solver_configuration_frozen": True,
        "cautionary_notes": [
            "Development results are not final generalization evidence.",
            "Holdout scenarios remain frozen, unviewed, and unrun.",
            "UNKNOWN means no incumbent was returned within the frozen budget; it is not INFEASIBLE.",
            "A fixed-objective-stage or bootstrap INFEASIBLE result is not a full hard-model proof.",
        ],
    }


def _read_completed_summary(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("status") not in {"completed_with_assignment", "completed_without_assignment"}:
        raise CpSatEvaluationError(f"cached scenario is not completed: {path}")
    return payload


class CpSatRobustnessRunner:
    def __init__(self, manifest_path: str | Path = DEFAULT_MANIFEST, config_dir: str | Path = "data/config") -> None:
        self.manifest_path = Path(manifest_path)
        self.config_dir = Path(config_dir)
        self.manifest = load_evaluation_manifest(self.manifest_path)
        self.manifest_hash = evaluation_manifest_hash(self.manifest_path)
        self.config = _solver_configuration(self.manifest)

    def select(self, group: str = "all", scenario_id: str | None = None, max_scenarios: int | None = None) -> list[EvaluationScenario]:
        if group not in ALLOWED_GROUPS:
            raise CpSatEvaluationError(f"group must be one of {ALLOWED_GROUPS}")
        scenarios = [_scenario_from_payload(self.manifest, item["scenario_id"]) for item in self.manifest["scenarios"]]
        if group != "all":
            scenarios = [item for item in scenarios if item.group == group]
        if scenario_id is not None:
            scenarios = [item for item in scenarios if item.scenario_id == scenario_id]
            if not scenarios:
                raise CpSatEvaluationError(f"scenario is not in development evaluation: {scenario_id}")
        if max_scenarios is not None:
            if max_scenarios <= 0:
                raise CpSatEvaluationError("max_scenarios must be positive")
            scenarios = scenarios[:max_scenarios]
        return scenarios

    def verify_sources(self, scenarios: Iterable[EvaluationScenario]) -> dict[str, Any]:
        source_info = {key: _verify_source_suite(self.manifest, key) for key in ("normal", "stress")}
        for scenario in scenarios:
            _load_scenario_input(self.manifest, scenario, self.config_dir)
        return source_info

    def dry_run(self, group: str = "all", scenario_id: str | None = None, max_scenarios: int | None = None) -> list[str]:
        scenarios = self.select(group, scenario_id, max_scenarios)
        self.verify_sources(scenarios)
        return [scenario.scenario_id for scenario in scenarios]

    def run(self, output_dir: str | Path = DEFAULT_OUTPUT, *, group: str = "all", scenario_id: str | None = None, max_scenarios: int | None = None, resume: bool = False, verify_only: bool = False) -> dict[str, Any]:
        scenarios = self.select(group, scenario_id, max_scenarios)
        source_info = self.verify_sources(scenarios)
        if verify_only:
            return {"verified_scenarios": [scenario.scenario_id for scenario in scenarios], "source_info": source_info}
        root = Path(output_dir)
        if root.exists() and any(root.iterdir()) and not resume:
            raise CpSatEvaluationError(f"evaluation output is non-empty; refusing to overwrite: {root}")
        root.mkdir(parents=True, exist_ok=True)
        run_manifest_path = root / "run_manifest.json"
        if resume:
            run_manifest = _read_json(run_manifest_path)
            expected = {"evaluation_manifest_sha256": self.manifest_hash, "source_git_commit": self.manifest["source_git_commit"], "solver_configuration_hash": _json_hash(self.manifest["solver_configuration"])}
            mismatches = [key for key, value in expected.items() if run_manifest.get(key) != value]
            if mismatches:
                raise CpSatEvaluationError("resume provenance mismatch: " + ", ".join(mismatches))
            selected_ids = [scenario.scenario_id for scenario in scenarios]
            if run_manifest.get("selected_scenario_ids") != selected_ids:
                raise CpSatEvaluationError("resume scenario selection mismatch")
            if run_manifest.get("holdout_runs") != 0 or run_manifest.get("external_persisted_seed") is not False:
                raise CpSatEvaluationError("resume safety flags do not match development-only evaluation")
            completed = set(run_manifest.get("completed_scenario_ids", []))
            unknown_completed = completed - set(selected_ids)
            if unknown_completed:
                raise CpSatEvaluationError("resume contains unknown completed scenarios: " + ", ".join(sorted(unknown_completed)))
        else:
            completed = set()
            _write_json(root / "evaluation_manifest_snapshot.json", self.manifest)
            _write_json(root / "run_manifest.json", {
                "schema_version": EVALUATION_SCHEMA_VERSION,
                "status": "running",
                "evaluation_manifest_sha256": self.manifest_hash,
                "source_git_commit": self.manifest["source_git_commit"],
                "solver_configuration_hash": _json_hash(self.manifest["solver_configuration"]),
                "source_info": source_info,
                "selected_scenario_ids": [scenario.scenario_id for scenario in scenarios],
                "completed_scenario_ids": [],
                "failed_scenario_ids": [],
                "holdout_runs": 0,
                "external_persisted_seed": False,
            })
        failures: list[dict[str, Any]] = []
        for scenario in scenarios:
            if scenario.scenario_id in completed:
                cached = _read_completed_summary(root / "scenarios" / scenario.scenario_id / "scenario_summary.json")
                if (
                    cached.get("evaluation_manifest_sha256") != self.manifest_hash
                    or cached.get("scenario", {}).get("scenario_id") != scenario.scenario_id
                ):
                    raise CpSatEvaluationError(f"cached scenario provenance mismatch: {scenario.scenario_id}")
                continue
            try:
                self._run_scenario(root, scenario)
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
        source_info_after = {key: _verify_source_suite(self.manifest, key) for key in ("normal", "stress")}
        if source_info_after != source_info:
            raise CpSatEvaluationCorrectnessError("source artifact changed during evaluation")
        _write_json(root / "failures.json", {"failures": failures, "unexpected_failure_count": len(failures)})
        self._write_aggregates(root, scenarios, source_info, failures)
        _write_json(root / "run_manifest.json", {**_read_json(root / "run_manifest.json"), "status": "completed", "completed_scenario_ids": sorted(completed), "failed_scenario_ids": [item["scenario_id"] for item in failures]})
        _write_checksums(root)
        return _read_json(root / "aggregate_summary.json")

    def _update_run_manifest(self, root: Path, scenarios: list[EvaluationScenario], completed: set[str], failures: list[dict[str, Any]], *, status: str) -> None:
        current = _read_json(root / "run_manifest.json")
        current.update({"status": status, "completed_scenario_ids": sorted(completed), "failed_scenario_ids": [item["scenario_id"] for item in failures]})
        _write_json(root / "run_manifest.json", current)

    def _run_scenario(self, root: Path, scenario: EvaluationScenario) -> None:
        allocation_input, input_manifest, source_result = _load_scenario_input(self.manifest, scenario, self.config_dir)
        certificate_valid = None
        if scenario.group == "negative":
            certificate_path = _source_root(self.manifest, scenario.source_suite) / "scenarios" / scenario.source_scenario_id / "infeasibility_certificate.json"
            certificate = _read_json(certificate_path)
            valid, reason = validate_certificate(certificate, allocation_input)
            if not valid:
                raise CpSatEvaluationCorrectnessError(f"negative certificate invalid before solve: {scenario.scenario_id}: {reason}")
            certificate_valid = True
        catalog = pd.read_csv(self.config_dir / "course_catalog.csv", keep_default_na=False)
        math_ids = math_course_ids_from_catalog(catalog)
        fallback_rules = _load_benchmark_math_fallback_rules(self.config_dir, catalog)
        started = time.perf_counter()
        result = run_fair_cp_sat_solver(
            allocation_input,
            seed=self.config.solver_seed,
            math_course_ids=math_ids,
            math_fallback_rules=fallback_rules,
            max_time_seconds_per_stage=self.config.per_stage_time_limit_seconds,
            bootstrap_time_seconds=self.config.bootstrap_time_limit_seconds,
            max_total_time_seconds=self.config.total_time_limit_seconds,
            num_search_workers=self.config.workers,
            use_feasibility_bootstrap=self.config.use_feasibility_bootstrap,
            use_constrained_first_hint=self.config.use_constrained_first_hint,
            logical_schedule_completion_enabled=self.config.logical_schedule_completion_enabled,
            initial_solution_artifact_dir=None,
        )
        runtime = time.perf_counter() - started
        trace = _stage_trace(result, scenario.scenario_id)
        metrics = _result_metrics(result, allocation_input)
        metrics.update({
            "status": result.solve_status.value,
            "final_assignment_available": bool(result.student_outcomes),
            "certificate_valid": certificate_valid,
        })
        _validate_objective_bounds(trace, int(metrics["theoretical_maximum"]))
        _validate_result(result, metrics, scenario)
        input_fingerprint = source_result["input_fingerprint"]
        row = _scenario_row(
            scenario,
            result,
            metrics,
            trace,
            self.manifest,
            {**input_fingerprint, **input_manifest},
            runtime,
        )
        row["scenario_family"] = source_result.get("scenario_family")
        row["source_scenario_id"] = scenario.source_scenario_id
        stage = _final_stage(trace)
        scenario_summary = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "status": "completed_with_assignment" if result.student_outcomes else "completed_without_assignment",
            "evaluation_manifest_sha256": self.manifest_hash,
            "source_git_commit": self.manifest["source_git_commit"],
            "scenario": asdict(scenario),
            "input_manifest": input_manifest,
            "input_fingerprint": source_result.get("input_fingerprint"),
            "source_result_fingerprint": source_result.get("input_fingerprint"),
            "result": row,
            "solver_configuration": self.manifest["solver_configuration"],
            "final_assignment_source_stage": stage["stage_name"] if stage and result.student_outcomes else None,
            "final_assignment_response_proto_hash": stage["response_proto_hash"] if stage and result.student_outcomes else None,
            "assignment_hash": _assignment_hash(result),
            "trace_file": "stage_trace.json",
            "final_validation_file": "final_validation.json",
        }
        base = root / "scenarios"
        base.mkdir(parents=True, exist_ok=True)
        destination = base / scenario.scenario_id
        temporary = Path(tempfile.mkdtemp(prefix=f".{scenario.scenario_id}.", dir=base))
        try:
            solver_dir = temporary / "solver"
            solver_dir.mkdir()
            _write_json(temporary / "stage_trace.json", {"scenario_id": scenario.scenario_id, "stages": trace})
            _write_json(temporary / "final_validation.json", {
                "final_schedule_policy_pass": metrics["final_schedule_policy_pass"],
                "consistency_issue_count": metrics["consistency_issue_count"],
                "assignment_nonpublishable": metrics["assignment_nonpublishable"],
                "status": result.solve_status.value,
                "evaluation_status": scenario_summary["status"],
            })
            _write_json(solver_dir / "solver_result.json", {
                "status": result.solve_status.value,
                "objective_values": result.objective_values,
                "model_stats": result.model_stats,
                "assignment_source_stage": scenario_summary["final_assignment_source_stage"],
                "response_proto_hash": scenario_summary["final_assignment_response_proto_hash"],
                "assignment_hash": scenario_summary["assignment_hash"],
            })
            _write_json(temporary / "scenario_summary.json", scenario_summary)
            _write_csv(
                temporary / "grade_subgroup_results.csv",
                _grade_rows(allocation_input, result, scenario.scenario_id),
            )
            if result.student_outcomes:
                _write_csv(temporary / "solver" / "assignments.csv", [_jsonable(asdict(item)) for item in result.assignments])
                _write_csv(temporary / "solver" / "request_outcomes.csv", [_jsonable(asdict(item)) for item in result.request_outcomes])
                _write_csv(temporary / "solver" / "student_outcomes.csv", [_jsonable(asdict(item)) for item in result.student_outcomes])
            temporary.replace(destination)
        finally:
            if temporary.exists():
                for path in sorted(temporary.rglob("*"), reverse=True):
                    if path.is_file():
                        path.unlink()
                    elif path.is_dir():
                        path.rmdir()
                temporary.rmdir()

    def _write_aggregates(self, root: Path, scenarios: list[EvaluationScenario], source_info: dict[str, Any], failures: list[dict[str, Any]]) -> None:
        rows = []
        grades = []
        for scenario in scenarios:
            summary_path = root / "scenarios" / scenario.scenario_id / "scenario_summary.json"
            if not summary_path.is_file():
                continue
            summary = _read_completed_summary(summary_path)
            rows.append(summary["result"])
            grade_path = root / "scenarios" / scenario.scenario_id / "grade_subgroup_results.csv"
            if grade_path.is_file():
                with grade_path.open(newline="") as handle:
                    grades.extend(csv.DictReader(handle))
        normal = [row for row in rows if row["scenario_group"] == "normal"]
        stress = [row for row in rows if row["scenario_group"] == "stress"]
        negative = [row for row in rows if row["scenario_group"] == "negative"]
        _write_csv(root / "normal_results.csv", normal)
        _write_csv(root / "stress_results.csv", stress)
        _write_csv(root / "negative_results.csv", negative)
        _write_csv(root / "grade_subgroup_results.csv", grades)
        stage_rows = []
        for scenario in scenarios:
            path = root / "scenarios" / scenario.scenario_id / "stage_trace.json"
            if path.is_file():
                stage_rows.extend({"scenario_id": scenario.scenario_id, **row} for row in _read_json(path)["stages"])
        _write_csv(root / "stage_results.csv", stage_rows)
        cp_greedy = []
        for row in rows:
            scenario = _scenario_from_payload(self.manifest, row["scenario_id"])
            greedy = _greedy_rows(_source_root(self.manifest, scenario.source_suite) / "scenarios" / scenario.source_scenario_id)
            cp_greedy.extend(_paired_rows(row, greedy, scenario, int(row["students"])))
        _write_csv(root / "paired_cp_sat_vs_greedy.csv", cp_greedy)
        normal_by_id = {row["scenario_id"]: row for row in normal}
        normal_stress = []
        for row in stress:
            scenario = _scenario_from_payload(self.manifest, row["scenario_id"])
            base = normal_by_id.get(scenario.paired_normal_scenario_id)
            if base is None:
                raise CpSatEvaluationError(
                    f"missing paired normal CP-SAT result: {row['scenario_id']} -> {scenario.paired_normal_scenario_id}"
                )
            normal_stress.append({
                "stress_scenario_id": row["scenario_id"],
                "normal_scenario_id": scenario.paired_normal_scenario_id,
                "stress_status": row["status"],
                "normal_status": base.get("status") if base else None,
                "stress_runtime_seconds": row["runtime_seconds"],
                "normal_runtime_seconds": base.get("runtime_seconds") if base else None,
                "runtime_delta": _delta(row["runtime_seconds"], base.get("runtime_seconds") if base else None),
                "primary_satisfaction_delta": _delta(row["primary_satisfaction_rate"], base.get("primary_satisfaction_rate") if base else None),
                "logical_full_rate_delta": _delta(row["logical_full_rate"], base.get("logical_full_rate") if base else None),
                "total_gap_delta": _delta(row["total_logical_gap"], base.get("total_logical_gap") if base else None),
                "objective_to_bound_gap_delta": _delta(row["objective_to_bound_gap"], base.get("objective_to_bound_gap") if base else None),
                "policy_status_changed": row.get("final_schedule_policy_pass") != (base.get("final_schedule_policy_pass") if base else None),
            })
        _write_csv(root / "paired_normal_stress_cp_sat.csv", normal_stress)
        aggregate = {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "development_only": True,
            "holdout_runs": 0,
            "external_persisted_seed": False,
            "normal": _aggregate(normal, "normal development"),
            "stress": _aggregate(stress, "ordinary stress development"),
            "negative": {
                "certificate_valid_count": sum(row.get("certificate_valid") is True for row in negative),
                "cp_sat_infeasible_count": sum(row["status"] == "INFEASIBLE" for row in negative),
                "unknown_count": sum(row["status"] == "UNKNOWN" for row in negative),
                "unexpected_feasible_count": sum(row["status"] in {"FEASIBLE", "OPTIMAL"} for row in negative),
                "publishable_assignment_count": sum(bool(row["final_assignment_available"]) for row in negative),
            },
            "by_stress_family": self._family_aggregate(stress),
            "source_info": source_info,
            "failures": failures,
        }
        _write_json(root / "aggregate_summary.json", aggregate)
        readiness = self._readiness(rows, negative, source_info, failures)
        _write_json(root / "holdout_readiness_assessment.json", readiness)

    def _family_aggregate(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        result = {}
        for family in sorted({str(row.get("scenario_family", "unknown")) for row in rows}):
            result[family] = _aggregate([row for row in rows if str(row.get("scenario_family", "unknown")) == family], family)
        return result

    def _readiness(self, rows: list[dict[str, Any]], negative: list[dict[str, Any]], source_info: dict[str, Any], failures: list[dict[str, Any]]) -> dict[str, Any]:
        enriched = []
        for row in rows:
            if "publishable_assignment_available" in row:
                enriched.append(row)
            else:
                enriched.append({
                    **row,
                    "raw_result_status": row.get("status"),
                    "publishable_assignment_available": bool(row.get("final_assignment_available")),
                })
        return _audited_readiness(enriched, negative, source_info, failures)


def audit_existing_artifact(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    allow_existing_audit: bool = False,
) -> dict[str, Any]:
    """Rebuild Phase C summaries from raw traces without invoking CP-SAT."""
    source = Path(source_dir).resolve()
    output = Path(output_dir).resolve()
    if source == output:
        raise CpSatEvaluationError("audited output must differ from the raw source artifact")
    if output.exists() and any(output.iterdir()):
        if not allow_existing_audit:
            raise CpSatEvaluationError(f"audited output is non-empty; refusing to overwrite: {output}")
        provenance_path = output / "audit_provenance.json"
        if not provenance_path.is_file():
            raise CpSatEvaluationError("existing audit output has no provenance; refusing to overwrite")
        existing_provenance = _read_json(provenance_path)
        if existing_provenance.get("source_artifact_path") != str(source) or existing_provenance.get("no_new_solver_runs") is not True:
            raise CpSatEvaluationError("existing audit provenance does not match this raw source")
    output.mkdir(parents=True, exist_ok=True)

    manifest = load_evaluation_manifest(manifest_path)
    source_info = _verify_source_artifacts_for_audit(source)
    run_manifest_path = source / "run_manifest.json"
    run_manifest = _read_json(run_manifest_path)
    if run_manifest.get("status") != "completed":
        raise CpSatEvaluationError("raw Phase C artifact is not completed")
    if run_manifest.get("evaluation_manifest_sha256") != evaluation_manifest_hash(manifest_path):
        raise CpSatEvaluationError("raw Phase C evaluation manifest hash does not match the frozen manifest")
    selected_ids = list(run_manifest.get("selected_scenario_ids", []))
    expected_ids = [str(item["scenario_id"]) for item in manifest["scenarios"]]
    if selected_ids != expected_ids or set(run_manifest.get("completed_scenario_ids", [])) != set(expected_ids):
        raise CpSatEvaluationError("raw Phase C artifact does not contain the complete frozen development selection")
    if run_manifest.get("holdout_runs") != 0 or run_manifest.get("external_persisted_seed") is not False:
        raise CpSatEvaluationError("raw Phase C artifact violates development-only provenance flags")

    audit_rows: list[dict[str, Any]] = []
    result_rows: list[dict[str, Any]] = []
    for raw_scenario in manifest["scenarios"]:
        scenario = EvaluationScenario(
            scenario_id=str(raw_scenario["scenario_id"]),
            group=str(raw_scenario["group"]),
            source_suite=str(raw_scenario["source_suite"]),
            source_scenario_id=str(raw_scenario["source_scenario_id"]),
            paired_normal_scenario_id=str(raw_scenario["paired_normal_scenario_id"]),
            expected_feasibility=str(raw_scenario["expected_feasibility"]),
        )
        scenario_root = source / "scenarios" / scenario.scenario_id
        summary = _read_completed_summary(scenario_root / "scenario_summary.json")
        trace = _read_json(scenario_root / "stage_trace.json")["stages"]
        semantics = _audit_status_semantics(scenario, summary["result"], trace)
        audit_row = {key: value for key, value in semantics.items() if key != "stage_statuses"}
        audit_row["stage_statuses"] = semantics["stage_statuses"]
        audit_row.update({f"stage_status_{name}": status for name, status in semantics["stage_statuses"].items()})
        audit_row["stage_trace"] = trace
        audit_rows.append(audit_row)
        result_rows.append(_audited_result_row(summary["result"], semantics))

    normal = [row for row in result_rows if row["scenario_group"] == "normal"]
    stress = [row for row in result_rows if row["scenario_group"] == "stress"]
    negative = [row for row in result_rows if row["scenario_group"] == "negative"]
    source_failures = _read_json(source / "failures.json") if (source / "failures.json").is_file() else {"failures": []}
    failures = list(source_failures.get("failures", []))
    provenance = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "source_artifact_path": str(source),
        "source_artifact_sha256": source_info["sha256sums_sha256"],
        "source_tree_hash": source_info["tree_hash"],
        "source_run_manifest_sha256": _sha256_file(run_manifest_path),
        "source_evaluation_manifest_hash": run_manifest["evaluation_manifest_sha256"],
        "audit_git_commit": run_manifest.get("source_git_commit"),
        "no_new_solver_runs": True,
        "source_artifact_stats": source_info,
    }
    corrected_aggregate = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "development_only": True,
        "holdout_runs": 0,
        "no_new_solver_runs": True,
        "normal": {**_aggregate(normal, "normal development"), "status_semantics": _audit_counts(normal)},
        "stress": {**_aggregate(stress, "ordinary stress development"), "status_semantics": _audit_counts(stress)},
        "negative": {**_audit_counts(negative), "certificate_valid_count": sum(row["certificate_proof_valid"] for row in negative)},
        "all_status_semantics": _audit_counts(result_rows),
        "source_info": source_info,
        "failures": failures,
    }
    readiness = _audited_readiness(result_rows, negative, source_info, failures)
    audit_json = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "provenance": provenance,
        "scenario_count": len(audit_rows),
        "scenarios": audit_rows,
        "counts": _audit_counts(result_rows),
    }
    _write_json(output / "audit_provenance.json", provenance)
    _write_json(output / "status_semantics_audit.json", audit_json)
    _write_csv(
        output / "status_semantics_audit.csv",
        [{key: value for key, value in row.items() if key not in {"stage_statuses", "stage_trace"}} for row in audit_rows],
    )
    _write_csv(output / "normal_results.csv", normal)
    _write_csv(output / "stress_results.csv", stress)
    _write_csv(output / "negative_results.csv", negative)
    _write_json(output / "aggregate_summary.json", corrected_aggregate)
    _write_json(output / "holdout_readiness_assessment.json", readiness)
    _write_json(output / "failures.json", {
        "source_failures": failures,
        "unexpected_failure_count": len(failures),
        "status_summary": _audit_counts(result_rows),
    })
    source_info_after = _verify_source_artifacts_for_audit(source)
    if source_info_after != source_info:
        raise CpSatEvaluationCorrectnessError("raw Phase C artifact changed during status audit")
    _write_checksums(output)
    return corrected_aggregate


def _verify_source_artifacts_for_audit(source: Path) -> dict[str, Any]:
    tree_hash, files, directories, bytes_total = _verify_sha256_manifest(source)
    return {
        "artifact_dir": str(source),
        "tree_hash": tree_hash,
        "sha256sums_sha256": _sha256_file(source / "SHA256SUMS.txt"),
        "files": files,
        "directories": directories,
        "bytes": bytes_total,
    }


def _write_checksums(root: Path) -> None:
    checksum = root / "SHA256SUMS.txt"
    rows = []
    for path in sorted(path for path in root.rglob("*") if path.is_file() and path != checksum):
        rows.append(f"{_sha256_file(path)}  {path.relative_to(root)}")
    checksum.write_text("\n".join(rows) + "\n", encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run frozen development CP-SAT robustness evaluation.")
    parser.add_argument("--evaluation-manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--group", choices=ALLOWED_GROUPS, default="all")
    parser.add_argument("--scenario-id")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-scenarios", type=int)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--audit-source-dir", help="Rebuild summaries from an existing raw Phase C artifact without solving")
    parser.add_argument("--audit-output-dir", help="Destination for a status-semantics audited artifact")
    parser.add_argument("--audit-overwrite-existing", action="store_true", help="Refresh an existing matching audited summary only")
    args = parser.parse_args(tuple(argv) if argv is not None else None)
    try:
        runner = CpSatRobustnessRunner(args.evaluation_manifest)
        if args.audit_source_dir:
            if not args.audit_output_dir:
                raise CpSatEvaluationError("--audit-output-dir is required with --audit-source-dir")
            summary = audit_existing_artifact(
                args.audit_source_dir,
                args.audit_output_dir,
                manifest_path=args.evaluation_manifest,
                allow_existing_audit=args.audit_overwrite_existing,
            )
            print(
                "CP-SAT robustness status audit PASS: "
                f"{summary['all_status_semantics']['scenarios']} scenario(s), no new solver runs"
            )
            return 0
        if args.dry_run:
            selected = runner.dry_run(args.group, args.scenario_id, args.max_scenarios)
            print(f"CP-SAT robustness dry-run PASS: {len(selected)} development scenario(s): {', '.join(selected)}")
            return 0
        if args.verify_only:
            verified = runner.run(args.output_dir, group=args.group, scenario_id=args.scenario_id, max_scenarios=args.max_scenarios, verify_only=True)
            print(f"CP-SAT robustness verify PASS: {len(verified['verified_scenarios'])} scenario(s)")
            return 0
        summary = runner.run(args.output_dir, group=args.group, scenario_id=args.scenario_id, max_scenarios=args.max_scenarios, resume=args.resume)
        print(f"CP-SAT robustness PASS: {summary['normal']['scenarios_attempted']} normal, {summary['stress']['scenarios_attempted']} stress, {summary['negative']['publishable_assignment_count']} negative publishable assignments")
        return 0
    except (CpSatEvaluationError, ExperimentManifestError, ValueError) as exc:
        print(f"CP-SAT robustness runner failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
