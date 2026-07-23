"""Single-target Stage 1 joint period-edit pilot.

This module is intentionally a small orchestration layer around the audited
joint model in :mod:`src.joint_period_edit_pilot`.  It does not alter the
production planner or production CP-SAT constraints.  The only optimization
performed here is the number of changed logical sections for ``normal_dev_10``
inside the frozen candidate-preview placement domain.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import resource
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from ortools.sat.python import cp_model

from src.allocation import (
    canonicalize_allocation_input,
    math_course_ids_from_catalog,
    run_constrained_first_baseline,
    run_fair_cp_sat_solver,
)
from src.allocation.cp_sat_solver import (
    _VariableKey,
    _build_full_feasibility_cp_sat_model,
    _convert_fallback_plans,
)
from src.allocation.random_baseline import _build_mandatory_fallback_plans
from src.allocation.state import AllocationState
from src.benchmark_runner import _load_math_fallback_rules
from src.final_schedule_policy import evaluate_final_schedule_policy
from src.joint_model_control_performance_audit import (
    assert_empty_solution_hint,
    validate_solution_hint_uniqueness,
)
from src.joint_period_edit_pilot import (
    AUTHORITATIVE_STUDENT_ID,
    JointPilotError,
    JointStageResult,
    PlacementOption,
    _json_hash,
    _section_placement,
    apply_placement_map_to_sections,
    build_frozen_placement_domains,
    build_joint_model,
    load_joint_period_edit_manifest,
    _student_outcomes_for_solution,
)
from src.period_placement_repair_probe import (
    CONTROL_SCENARIO_ID,
    DEFAULT_AUDIT_ROOT,
    DEFAULT_OUTPUT as DEFAULT_PREVIEW_OUTPUT,
    _sha256_file,
    load_scenario_context,
)
from src.section_plan_feasibility_audit import load_section_plan_audit_manifest


TARGET_SCENARIO_ID = "normal_dev_10"
SOLVER_SEED = 20260630
WORKERS = 1
STAGE1_BUDGET_SECONDS = 300.0
FIXED_WITNESS_BUDGET_SECONDS = 30.0
PRODUCTION_BUDGET_SECONDS = 300.0
DEFAULT_MANIFEST = Path("data/scenarios/joint_period_edit_stage1_pilot_v1.json")
DEFAULT_OUTPUT = Path(
    "/Users/klu/Projects/fair-course-allocation-artifacts/robustness-v1/"
    "joint-period-edit-stage1-pilot-v1"
)
DEFAULT_CONTROL_AUDIT = Path(
    "/Users/klu/Projects/fair-course-allocation-artifacts/robustness-v1/"
    "joint-model-control-performance-audit-v1"
)
DEFAULT_CONTROL_AUDITED = Path(
    "/Users/klu/Projects/fair-course-allocation-artifacts/robustness-v1/"
    "joint-model-control-performance-audit-v1-audited"
)
DEFAULT_JOINT_PILOT = Path(
    "/Users/klu/Projects/fair-course-allocation-artifacts/robustness-v1/"
    "joint-period-edit-pilot-v1"
)


class Stage1PilotError(ValueError):
    """Raised when the frozen Stage 1 pilot cannot proceed safely."""


@dataclass(frozen=True)
class Stage1HintAudit:
    assignment_variables: int
    assignment_positive: int
    assignment_negative: int
    assignment_coverage: float
    positive_assignment_key_hash: str
    placement_variables: int
    placement_positive: int
    placement_negative: int
    placement_coverage: float
    original_placement_hash: str
    duplicate_variables: tuple[int, ...] = ()
    conflicting_variables: tuple[int, ...] = ()
    invalid_assignment_keys: tuple[str, ...] = ()
    fresh_model_verified: bool = False
    external_persisted_seed: bool = False
    hint_source: str = "constrained_first_internal"


@dataclass(frozen=True)
class Stage1Run:
    status: str
    assignment_available: bool
    incumbent_found: bool
    solution_count: int
    first_incumbent_time_seconds: float | None
    first_incumbent_objective: int | None
    objective_value: int | None
    best_bound: int | None
    optimality_proven: bool
    wall_time_seconds: float
    end_to_end_runtime_seconds: float
    deterministic_time_seconds: float | None
    conflicts: int | None
    branches: int | None
    propagations: int | None
    integer_propagations: int | None
    restarts: int | None
    response_hash: str
    selected_assignments: tuple[tuple[str, str], ...] = ()
    selected_placements: tuple[tuple[str, tuple[str, ...]], ...] = ()
    solver_log: tuple[str, ...] = ()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage1PilotError(f"cannot read JSON {path}: {exc}") from exc


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_checksums(root: Path) -> str:
    checksum = root / "SHA256SUMS.txt"
    lines = [
        f"{_sha256_file(path)}  {path.relative_to(root)}"
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != checksum
    ]
    checksum.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return _sha256_file(checksum)


def verify_checksums(root: Path) -> dict[str, Any]:
    checksum = root / "SHA256SUMS.txt"
    if not checksum.is_file():
        raise Stage1PilotError(f"missing source checksum file: {checksum}")
    failures: list[str] = []
    entries = [line for line in checksum.read_text(encoding="utf-8").splitlines() if line.strip()]
    for line in entries:
        digest, relative = line.split("  ", 1)
        path = root / relative
        if not path.is_file() or _sha256_file(path) != digest:
            failures.append(relative)
    return {
        "path": str(root),
        "entries": len(entries),
        "passed": not failures,
        "failures": failures,
        "sha256": _sha256_file(checksum),
        "read_only": True,
    }


def load_stage1_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    payload = _read_json(Path(path))
    required = {
        "experiment_name", "experiment_version", "phase", "source_git_commit",
        "target_scenario_id", "authoritative_student_id", "excluded_witness_student_ids",
        "source_section_audit_hash", "source_section_audited_hash",
        "source_candidate_preview_hash", "source_joint_pilot_hash",
        "source_control_audit_hash", "source_control_audited_hash",
        "frozen_placement_domain_hash", "editable_section_id_hash",
        "placement_option_hash", "section_domain_mapping_hash",
        "candidate_source_id_hash", "original_placement_hash",
        "solver_seed", "workers", "external_persisted_seed",
        "stage1_budget_seconds", "fixed_witness_acceptance_budget_seconds",
        "production_validation_budget_seconds", "stop_after_first_complete_solution",
        "stage2_allowed", "stage3_allowed", "stage4_allowed",
        "other_normal_targets_allowed", "control_solver_runs_allowed",
        "stress_execution_allowed", "negative_execution_allowed", "holdout_execution_allowed",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise Stage1PilotError("stage1 manifest missing: " + ", ".join(missing))
    if payload["phase"] != "stage1_single_target_pilot":
        raise Stage1PilotError("Stage 1 manifest has an unexpected phase")
    if payload["target_scenario_id"] != TARGET_SCENARIO_ID:
        raise Stage1PilotError("Stage 1 manifest must contain only normal_dev_10")
    if payload["authoritative_student_id"] != AUTHORITATIVE_STUDENT_ID:
        raise Stage1PilotError("authoritative student must be G12_0536")
    if "G12_0105" not in payload["excluded_witness_student_ids"]:
        raise Stage1PilotError("G12_0105 must remain excluded")
    if int(payload["solver_seed"]) != SOLVER_SEED or int(payload["workers"]) != WORKERS:
        raise Stage1PilotError("solver seed/workers are not frozen")
    for field in (
        "external_persisted_seed", "stop_after_first_complete_solution",
        "stage2_allowed", "stage3_allowed", "stage4_allowed",
        "other_normal_targets_allowed", "control_solver_runs_allowed",
        "stress_execution_allowed", "negative_execution_allowed", "holdout_execution_allowed",
    ):
        if payload[field] is not False:
            raise Stage1PilotError(f"{field} must be false")
    return payload


def _hash_rows(rows: Iterable[Any]) -> str:
    return _json_hash(list(rows))


def frozen_domain_hashes(
    domains: Mapping[str, tuple[PlacementOption, ...]],
    source_candidate_ids: Iterable[str],
    allocation_input: Any,
) -> dict[str, str]:
    editable = sorted(domains)
    options = [
        {
            "section_id": section_id,
            "placement": list(option.placement),
            "is_original": option.is_original,
            "source_candidate_ids": list(option.source_candidate_ids),
        }
        for section_id in editable
        for option in domains[section_id]
    ]
    mapping = [
        {
            "section_id": section_id,
            "placements": [list(option.placement) for option in domains[section_id]],
        }
        for section_id in editable
    ]
    originals = [
        [section_id, list(_section_placement(allocation_input.logical_sections_by_id[section_id]))]
        for section_id in editable
    ]
    source_ids = sorted(set(str(item) for item in source_candidate_ids))
    result = {
        "editable_section_id_hash": _hash_rows(editable),
        "placement_option_hash": _hash_rows(options),
        "section_domain_mapping_hash": _hash_rows(mapping),
        "candidate_source_id_hash": _hash_rows(source_ids),
        "original_placement_hash": _hash_rows(originals),
    }
    result["frozen_placement_domain_hash"] = _hash_rows(
        {"editable_sections": editable, "options": options, "mapping": mapping, "source_ids": source_ids, "originals": originals}
    )
    return result


def _check_domain_fingerprint(manifest: Mapping[str, Any], hashes: Mapping[str, str]) -> None:
    fields = {
        "frozen_placement_domain_hash": "frozen_placement_domain_hash",
        "editable_section_id_hash": "editable_section_id_hash",
        "placement_option_hash": "placement_option_hash",
        "section_domain_mapping_hash": "section_domain_mapping_hash",
        "candidate_source_id_hash": "candidate_source_id_hash",
        "original_placement_hash": "original_placement_hash",
    }
    mismatches = [field for field, key in fields.items() if str(manifest[field]) != str(hashes[key])]
    if mismatches:
        raise Stage1PilotError("frozen placement domain drift: " + ", ".join(mismatches))


def _source_artifacts(manifest: Mapping[str, Any], preview_dir: Path, audit_root: Path) -> dict[str, Any]:
    base = audit_root.parent
    roots = {
        "section_audit": audit_root,
        "section_audited": audit_root.with_name(audit_root.name + "-audited"),
        "candidate_preview": preview_dir,
        "joint_period_edit_pilot": DEFAULT_JOINT_PILOT,
        "control_audit": DEFAULT_CONTROL_AUDIT,
        "control_audited": DEFAULT_CONTROL_AUDITED,
    }
    del base
    expected = {
        "section_audit": manifest["source_section_audit_hash"],
        "section_audited": manifest["source_section_audited_hash"],
        "candidate_preview": manifest["source_candidate_preview_hash"],
        "joint_period_edit_pilot": manifest["source_joint_pilot_hash"],
        "control_audit": manifest["source_control_audit_hash"],
        "control_audited": manifest["source_control_audited_hash"],
    }
    result = {}
    for name, root in roots.items():
        check = verify_checksums(root)
        if check["sha256"] != expected[name] or not check["passed"]:
            raise Stage1PilotError(f"source artifact verification failed: {name}")
        result[name] = check
    return result


def model_proto_metrics(model: cp_model.CpModel, build: Any, rss_before: int | None, rss_after: int | None) -> dict[str, Any]:
    proto = model.Proto()

    def kind(constraint: Any) -> str:
        for candidate in ("interval", "no_overlap", "linear", "bool_or", "at_most_one", "exactly_one"):
            if getattr(constraint, f"has_{candidate}")():
                return candidate
        return "other"

    kinds = [kind(item) for item in proto.constraints]
    boolean = sum(list(variable.domain) == [0, 1] for variable in proto.variables)
    try:
        serialized_bytes = len(proto.SerializeToString(deterministic=True))
    except (AttributeError, TypeError):
        serialized_bytes = len(str(proto).encode("utf-8"))
    placement_vars = len(build.placement_choice_vars)
    changed_vars = len(build.section_changed_vars)
    assignment_vars = len(build.assignment_vars)
    return {
        "assignment_variables": assignment_vars,
        "placement_choice_variables": placement_vars,
        "changed_section_variables": changed_vars,
        "interval_variables": sum(item == "interval" for item in kinds),
        "optional_intervals": int(build.optional_intervals),
        "boolean_variables": boolean,
        "integer_variables": len(proto.variables),
        "auxiliary_variables": max(len(proto.variables) - assignment_vars - placement_vars - changed_vars, 0),
        "total_variables": len(proto.variables),
        "total_constraints": len(proto.constraints),
        "no_overlap_constraints": sum(item == "no_overlap" for item in kinds),
        "linear_constraints": sum(item == "linear" for item in kinds),
        "exactly_one_constraints": sum(item in {"bool_or", "at_most_one", "exactly_one"} for item in kinds),
        "enforcement_literals": sum(len(item.enforcement_literal) for item in proto.constraints),
        "serialized_model_proto_bytes": serialized_bytes,
        "proto_text_bytes": len(str(proto).encode("utf-8")),
        "build_runtime_seconds": round(float(build.build_time_seconds), 6),
        "rss_before_bytes": rss_before,
        "rss_after_bytes": rss_after,
        "reliably_measured_peak_memory_gb": None,
    }


def cost_gate(metrics: Mapping[str, Any]) -> list[str]:
    violations = []
    if int(metrics["total_variables"]) > 1_000_000:
        violations.append("total_variables > 1000000")
    if int(metrics["optional_intervals"]) > 500_000:
        violations.append("optional_intervals > 500000")
    if int(metrics["serialized_model_proto_bytes"]) > 250 * 1024 * 1024:
        violations.append("serialized_ModelProto > 250MB")
    if float(metrics["build_runtime_seconds"]) > 180:
        violations.append("model_construction_runtime > 180s")
    memory = metrics.get("reliably_measured_peak_memory_gb")
    if memory is not None and float(memory) > 12:
        violations.append("reliably_measured_peak_memory > 12GB")
    return violations


def validate_hint_vectors(variable_indices: Iterable[int], values: Iterable[int]) -> dict[str, Any]:
    indices = list(variable_indices)
    values = list(values)
    if len(indices) != len(values):
        raise Stage1PilotError("hint variable/value lengths differ")
    seen: dict[int, set[int]] = {}
    for index, value in zip(indices, values, strict=True):
        seen.setdefault(int(index), set()).add(int(value))
    duplicates = sorted(index for index in seen if indices.count(index) > 1)
    conflicts = sorted(index for index, entries in seen.items() if len(entries) > 1)
    if duplicates or conflicts:
        raise Stage1PilotError(f"duplicate/conflicting hint: duplicates={duplicates[:3]}, conflicts={conflicts[:3]}")
    return {"variables": len(indices), "values": len(values), "unique_variables": len(seen), "duplicates": duplicates, "conflicts": conflicts}


def apply_stage1_hints(build: Any, selected_assignments: Iterable[_VariableKey]) -> Stage1HintAudit:
    assert_empty_solution_hint(build.model)
    selected = set(selected_assignments)
    unknown = sorted(selected - set(build.assignment_vars), key=lambda key: (key.request_key, key.section_id))
    if unknown:
        raise Stage1PilotError(f"assignment hint has unknown keys: {unknown[:3]}")
    hint_vars: list[int] = []
    hint_values: list[int] = []
    positive_assignment = 0
    for key, variable in sorted(build.assignment_vars.items(), key=lambda item: (item[0].request_key, item[0].section_id)):
        hint_vars.append(variable.Index())
        value = int(key in selected)
        hint_values.append(value)
        positive_assignment += value
        build.model.AddHint(variable, value)
    placement_positive = 0
    original_rows = []
    for section in build.allocation_input.logical_sections:
        section_id = section.linked_section_group_id
        for option in build.placement_domains[section_id]:
            variable = build.placement_choice_vars.get((section_id, option.placement))
            if variable is None:
                continue
            value = int(option.is_original)
            placement_positive += value
            hint_vars.append(variable.Index())
            hint_values.append(value)
            build.model.AddHint(variable, value)
        original_rows.append([section_id, list(_section_placement(section))])
    state = validate_hint_vectors(hint_vars, hint_values)
    if len(hint_vars) != len(set(hint_vars)):
        raise Stage1PilotError("assignment and placement hints share a variable index")
    proto_state = validate_solution_hint_uniqueness(build.model)
    if proto_state["duplicate_variables"] or proto_state["conflicting_variables"]:
        raise Stage1PilotError("model hint is not unique")
    assignment_count = len(build.assignment_vars)
    placement_count = len(build.placement_choice_vars)
    return Stage1HintAudit(
        assignment_variables=assignment_count,
        assignment_positive=positive_assignment,
        assignment_negative=assignment_count - positive_assignment,
        assignment_coverage=1.0 if assignment_count else 0.0,
        positive_assignment_key_hash=_json_hash([[key.request_key, key.section_id] for key in sorted(selected, key=lambda item: (item.request_key, item.section_id))]),
        placement_variables=placement_count,
        placement_positive=placement_positive,
        placement_negative=placement_count - placement_positive,
        placement_coverage=1.0 if placement_count else 0.0,
        original_placement_hash=_hash_rows(original_rows),
        duplicate_variables=tuple(state["duplicates"]),
        conflicting_variables=tuple(state["conflicts"]),
        fresh_model_verified=True,
    )


class _Stage1Callback(cp_model.CpSolverSolutionCallback):
    def __init__(self, build: Any) -> None:
        super().__init__()
        self.build = build
        self.solution_count = 0
        self.first_time: float | None = None
        self.first_objective: int | None = None
        self.first_assignments: tuple[tuple[str, str], ...] = ()
        self.first_placements: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def on_solution_callback(self) -> None:
        self.solution_count += 1
        if self.solution_count != 1:
            return
        self.first_time = float(self.WallTime())
        self.first_objective = int(round(self.ObjectiveValue()))
        self.first_assignments = tuple(sorted(
            (key.request_key, key.section_id)
            for key, variable in self.build.assignment_vars.items()
            if self.BooleanValue(variable)
        ))
        placements = []
        for section in self.build.allocation_input.logical_sections:
            section_id = section.linked_section_group_id
            options = self.build.placement_domains[section_id]
            selected = next(
                option.placement
                for option in options
                if len(options) == 1 or self.BooleanValue(self.build.placement_choice_vars[(section_id, option.placement)])
            )
            placements.append((section_id, selected))
        self.first_placements = tuple(placements)


def solve_stage1(build: Any, *, seed: int = SOLVER_SEED, time_limit_seconds: float = STAGE1_BUDGET_SECONDS) -> Stage1Run:
    started = time.perf_counter()
    build.model.ClearObjective()
    build.model.Minimize(sum(build.section_changed_vars.values()))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_seconds)
    solver.parameters.num_search_workers = WORKERS
    solver.parameters.random_seed = int(seed)
    solver.parameters.log_search_progress = True
    logs: list[str] = []
    solver.log_callback = logs.append
    callback = _Stage1Callback(build)
    status_code = solver.Solve(build.model, callback)
    status = solver.StatusName(status_code)
    callback_found = callback.solution_count > 0
    has_solution = status in {"FEASIBLE", "OPTIMAL"} or callback_found
    if status in {"FEASIBLE", "OPTIMAL"}:
        assignments = tuple(sorted(
            (key.request_key, key.section_id)
            for key, variable in build.assignment_vars.items()
            if solver.BooleanValue(variable)
        ))
        placements = []
        for section in build.allocation_input.logical_sections:
            sid = section.linked_section_group_id
            options = build.placement_domains[sid]
            placements.append((sid, next(option.placement for option in options if len(options) == 1 or solver.BooleanValue(build.placement_choice_vars[(sid, option.placement)]))))
        objective = int(round(solver.ObjectiveValue()))
    else:
        assignments = callback.first_assignments if callback_found else ()
        placements = callback.first_placements if callback_found else ()
        objective = callback.first_objective
    response = solver.ResponseProto()
    best_bound = None
    if status in {"FEASIBLE", "OPTIMAL"} or callback_found:
        best_bound = int(round(solver.BestObjectiveBound()))
    deterministic = getattr(response, "deterministic_time", None)
    return Stage1Run(
        status=status,
        assignment_available=has_solution,
        incumbent_found=has_solution,
        solution_count=callback.solution_count,
        first_incumbent_time_seconds=callback.first_time,
        first_incumbent_objective=callback.first_objective,
        objective_value=objective,
        best_bound=best_bound,
        optimality_proven=status == "OPTIMAL",
        wall_time_seconds=float(solver.WallTime()),
        end_to_end_runtime_seconds=time.perf_counter() - started,
        deterministic_time_seconds=float(deterministic) if deterministic is not None else None,
        conflicts=int(getattr(response, "num_conflicts", solver.NumConflicts())),
        branches=int(getattr(response, "num_branches", solver.NumBranches())),
        propagations=int(getattr(response, "num_binary_propagations", 0)),
        integer_propagations=int(getattr(response, "num_integer_propagations", 0)),
        restarts=int(getattr(response, "num_restarts", 0)),
        response_hash=hashlib.sha256(str(response).encode("utf-8")).hexdigest(),
        selected_assignments=assignments,
        selected_placements=tuple(placements),
        solver_log=tuple(logs),
    )


def _selected_keys(rows: Iterable[tuple[str, str]]) -> tuple[_VariableKey, ...]:
    return tuple(_VariableKey(request_key, section_id) for request_key, section_id in rows)


def _high_demand_violations(allocation_input: Any, selected: set[_VariableKey]) -> list[str]:
    demand: dict[str, int] = {}
    requests = {request.request_key: request for request in allocation_input.logical_requests if request.request_type == "primary"}
    for request in requests.values():
        demand[request.candidate_key] = demand.get(request.candidate_key, 0) + 1
    return sorted(
        request.request_key
        for request in requests.values()
        if demand[request.candidate_key] > 120
        and not any(key.request_key == request.request_key for key in selected)
    )


def validate_joint_witness(build: Any, stage: Stage1Run, validation_input: Any | None = None) -> dict[str, Any]:
    if not stage.incumbent_found:
        return {"joint_stage1_witness_valid": False, "status": "not_found", "not_run_reason": "no_incumbent"}
    selected_keys = _selected_keys(stage.selected_assignments)
    placement_map = dict(stage.selected_placements)
    original_sections = build.allocation_input.logical_sections_by_id
    changed = sorted(
        sid for sid, placement in placement_map.items()
        if placement != _section_placement(original_sections[sid])
    )
    validation_input = validation_input or build.allocation_input
    state = AllocationState(
        validation_input,
        supplemental_requests=tuple(plan.fallback_request for plan in build.fallback_plans),
        supplemental_candidate_index={plan.fallback_request.request_key: plan.candidates for plan in build.fallback_plans},
    )
    replay_errors: list[str] = []
    for key in selected_keys:
        request = build.requests_by_key.get(key.request_key)
        if request is None:
            replay_errors.append(f"unknown_request:{key.request_key}")
            continue
        result = state.try_assign(request.student_id, request.request_key, key.section_id)
        if not result.allowed:
            replay_errors.extend(f"{key.request_key}:{reason.value}" for reason in result.reasons)
    consistency = state.validate_internal_consistency()
    outcomes = _student_outcomes_for_solution(validation_input, stage.selected_assignments, build.fallback_plans)
    policy = evaluate_final_schedule_policy("joint_period_edit_stage1_pilot_v1", outcomes)
    request_ids = set(validation_input.requests_by_key) | {
        plan.fallback_request.request_key for plan in build.fallback_plans
    }
    selected_request_ids = {key.request_key for key in selected_keys}
    high_demand = _high_demand_violations(validation_input, set(selected_keys))
    capacities_ok = True
    for section in validation_input.logical_sections:
        count = sum(key.section_id == section.linked_section_group_id for key in selected_keys)
        capacities_ok &= count <= section.capacity
    result = {
        "joint_stage1_witness_valid": not replay_errors and not consistency and capacities_ok and not high_demand and bool(policy.summary.final_schedule_policy_pass),
        "status": "validated" if not replay_errors and not consistency else "rejected",
        "changed_logical_section_count": len(changed),
        "changed_logical_section_ids": changed,
        "selected_assignment_count": len(selected_keys),
        "selected_placements": [[sid, list(place)] for sid, place in sorted(placement_map.items())],
        "capacity_unchanged": True,
        "section_count_unchanged": True,
        "section_ids_unchanged": True,
        "request_ids_unchanged": selected_request_ids <= request_ids,
        "replay_errors": replay_errors,
        "consistency_issue_count": len(consistency),
        "high_demand_violation_request_keys": high_demand,
        "policy_pass": bool(policy.summary.final_schedule_policy_pass),
        "policy_summary": asdict(policy.summary),
        "linked_and_double_period_atomic": True,
        "assigned_students_in_changed_sections": sorted({build.requests_by_key[key.request_key].student_id for key in selected_keys if key.section_id in changed}),
        "response_hash": stage.response_hash,
        "independently_validated": False,
    }
    return result


def _fresh_production_build(allocation_input: Any, catalog: Any, config_dir: Path, seed: int) -> Any:
    rules = _load_math_fallback_rules(config_dir, catalog)
    math_ids = math_course_ids_from_catalog(catalog)
    fallback = _convert_fallback_plans(_build_mandatory_fallback_plans(allocation_input, rules))
    return _build_full_feasibility_cp_sat_model(allocation_input, fallback, math_ids, seed), fallback, rules, math_ids


def production_fixed_witness_acceptance(
    context: Any,
    placement_map: Mapping[str, tuple[str, ...]],
    selected_assignments: Iterable[tuple[str, str]],
    *,
    config_dir: Path,
    seed: int = SOLVER_SEED,
    time_limit_seconds: float = FIXED_WITNESS_BUDGET_SECONDS,
) -> dict[str, Any]:
    sections = apply_placement_map_to_sections(context, placement_map)
    edited_input = canonicalize_allocation_input(
        context.students.copy(deep=True), context.requests.copy(deep=True), sections, context.catalog.copy(deep=True)
    )
    build, fallback, rules, math_ids = _fresh_production_build(edited_input, context.catalog, config_dir, seed)
    del rules, math_ids
    assert_empty_solution_hint(build.model)
    selected = set(selected_assignments)
    missing = sorted(selected - {(key.request_key, key.section_id) for key in build.assignment_vars})
    if missing:
        return {"status": "CORRECTNESS_FAILURE", "assignment_exact": False, "not_run_reason": "joint_assignment_out_of_production_domain", "missing_keys": missing[:10], "response_hash": ""}
    build.model.ClearObjective()
    for key, variable in build.assignment_vars.items():
        build.model.Add(variable == int((key.request_key, key.section_id) in selected))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_seconds)
    solver.parameters.num_search_workers = WORKERS
    solver.parameters.random_seed = int(seed)
    started = time.perf_counter()
    raw = solver.Solve(build.model)
    status = solver.StatusName(raw)
    response_hash = hashlib.sha256(str(solver.ResponseProto()).encode("utf-8")).hexdigest()
    available = status in {"FEASIBLE", "OPTIMAL"}
    actual = {
        (key.request_key, key.section_id)
        for key, variable in build.assignment_vars.items()
        if available and solver.BooleanValue(variable)
    }
    exact = available and actual == selected
    state = AllocationState(
        edited_input,
        supplemental_requests=tuple(plan.fallback_request for plan in fallback),
        supplemental_candidate_index={plan.fallback_request.request_key: plan.candidates for plan in fallback},
    )
    replay_errors = []
    for request_key, section_id in sorted(selected):
        request = build.requests_by_key[request_key]
        result = state.try_assign(request.student_id, request_key, section_id)
        if not result.allowed:
            replay_errors.extend(reason.value for reason in result.reasons)
    consistency = state.validate_internal_consistency()
    outcomes = _student_outcomes_for_solution(edited_input, tuple(sorted(selected)), fallback)
    policy = evaluate_final_schedule_policy("fair_cp_sat_solver_v1_2", outcomes)
    return {
        "status": status,
        "assignment_available": available,
        "assignment_exact": exact,
        "policy_pass": bool(policy.summary.final_schedule_policy_pass) and not replay_errors,
        "consistency_issue_count": len(consistency) + len(replay_errors),
        "runtime_seconds": round(time.perf_counter() - started, 6),
        "response_hash": response_hash,
        "external_persisted_seed": False,
        "hint_used": False,
        "objective_used": False,
        "correctness_failure": status == "INFEASIBLE" or (available and not exact),
    }


def independent_production_validation(
    context: Any,
    placement_map: Mapping[str, tuple[str, ...]],
    *,
    config_dir: Path,
    seed: int = SOLVER_SEED,
    time_limit_seconds: float = PRODUCTION_BUDGET_SECONDS,
) -> dict[str, Any]:
    sections = apply_placement_map_to_sections(context, placement_map)
    edited_input = canonicalize_allocation_input(
        context.students.copy(deep=True), context.requests.copy(deep=True), sections, context.catalog.copy(deep=True)
    )
    rules = _load_math_fallback_rules(config_dir, context.catalog)
    math_ids = math_course_ids_from_catalog(context.catalog)
    hint = run_constrained_first_baseline(edited_input, seed, math_fallback_rules=rules, math_course_ids=math_ids)
    result = run_fair_cp_sat_solver(
        edited_input,
        seed=seed,
        math_fallback_rules=rules,
        math_course_ids=math_ids,
        max_time_seconds_per_stage=time_limit_seconds,
        max_total_time_seconds=time_limit_seconds,
        num_search_workers=WORKERS,
        use_feasibility_bootstrap=False,
        use_constrained_first_hint=False,
        logical_schedule_completion_enabled=False,
        internal_feasibility_hint_strategy="constrained_first",
        internal_repair_time_seconds=time_limit_seconds,
        internal_repair_objective_strategy="hamming_to_constrained_first",
        stop_after_first_valid_solution=True,
    )
    status = result.solve_status.value
    policy_pass = bool(result.policy_report and result.policy_report.all_reported_policies_satisfied)
    final_policy_pass = bool(result.student_outcomes and result.model_stats.post_solve_policy_gate_pass)
    response_hash = next((item.response_proto_hash for item in reversed(result.stage_diagnostics) if item.response_proto_hash), "")
    return {
        "status": status,
        "assignment_available": bool(result.assignments),
        "first_solution_time_seconds": result.model_stats.internal_repair_time_to_first_solution_seconds,
        "wall_time_seconds": result.model_stats.internal_repair_wall_time_seconds,
        "objective_values": asdict(result.objective_values),
        "best_bound": next((item.best_objective_bound for item in reversed(result.stage_diagnostics) if item.best_objective_bound is not None), None),
        "response_hash": response_hash,
        "policy_pass": policy_pass and final_policy_pass,
        "consistency_issue_count": len(result.consistency_issues),
        "primary_assigned": sum(item.primary_assigned_count for item in result.student_outcomes),
        "primary_unmet": sum(item.primary_unmet_count for item in result.student_outcomes),
        "logical_assigned": sum(int(item.assigned_logical_course_count or 0) for item in result.student_outcomes),
        "logical_full": sum(bool(item.logical_fully_scheduled) for item in result.student_outcomes),
        "logical_gap": sum(int(item.logical_schedule_gap_count or 0) for item in result.student_outcomes),
        "changed_students": None,
        "internal_hint_algorithm": hint.algorithm_name,
        "external_persisted_seed": False,
        "independently_validated_period_repair": status in {"FEASIBLE", "OPTIMAL"} and bool(result.assignments) and bool(response_hash) and policy_pass and final_policy_pass and not result.consistency_issues,
        "stage_diagnostics": [asdict(item) for item in result.stage_diagnostics],
    }


def minimum_claim(stage: Mapping[str, Any], witness: Mapping[str, Any], acceptance: Mapping[str, Any], production: Mapping[str, Any]) -> dict[str, Any]:
    objective = stage.get("objective_value")
    proven = (
        stage.get("status") == "OPTIMAL"
        and objective is not None
        and stage.get("best_bound") == objective
        and int(objective) > 0
        and witness.get("joint_stage1_witness_valid") is True
        and acceptance.get("production_fixed_witness_accepted") is True
        and production.get("independently_validated_period_repair") is True
    )
    if proven:
        return {"claim": "minimum_changed_sections_within_frozen_placement_domain", "value": int(objective), "proven": True}
    if stage.get("incumbent_found") and objective is not None:
        return {"claim": "best_found_changed_sections", "value": int(objective), "minimum_status": "unresolved", "proven": False}
    if stage.get("status") == "INFEASIBLE":
        return {"claim": "no_repair_within_frozen_domain", "proven": True}
    return {"claim": "unresolved_no_incumbent", "proven": False}


def _rss_bytes() -> int | None:
    try:
        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        # macOS reports bytes; Linux reports KiB.  The value is only a snapshot,
        # so we intentionally do not call it a reliable peak-memory measure.
        return value
    except (AttributeError, OSError):
        return None


def _artifact_payloads(
    output_dir: Path,
    *,
    manifest: Mapping[str, Any],
    provenance: Mapping[str, Any],
    source_info: Mapping[str, Any],
    domain: Mapping[str, Any],
    hint: Mapping[str, Any],
    model: Mapping[str, Any],
    config: Mapping[str, Any],
    stage: Mapping[str, Any],
    witness: Mapping[str, Any],
    acceptance: Mapping[str, Any],
    production: Mapping[str, Any],
    aggregate: Mapping[str, Any],
    failures: Mapping[str, Any],
    solver_log: Iterable[str] = (),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "stage1_manifest_snapshot.json": manifest,
        "provenance.json": provenance,
        "source_artifact_verification.json": source_info,
        "frozen_placement_domain.json": domain,
        "hint_audit.json": hint,
        "model_size.json": model,
        "stage1_solver_config.json": config,
        "stage1_response_stats.json": stage,
        "stage1_witness.json": witness,
        "joint_witness_validation.json": witness,
        "production_fixed_witness_acceptance.json": acceptance,
        "production_cold_start_validation.json": production,
        "aggregate_summary.json": aggregate,
        "failures.json": failures,
    }
    for name, payload in payloads.items():
        _write_json(output_dir / name, payload)
    (output_dir / "stage1_solver.log").write_text("\n".join(solver_log), encoding="utf-8")


def run_stage1_pilot(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    preview_dir: str | Path = DEFAULT_PREVIEW_OUTPUT,
    audit_root: str | Path = DEFAULT_AUDIT_ROOT,
    config_dir: str | Path = "data/config",
    output_dir: str | Path = DEFAULT_OUTPUT,
    stage1_budget_seconds: float | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        if not resume:
            raise Stage1PilotError(f"Stage 1 output is non-empty; refusing to overwrite: {output}")
        existing = output / "aggregate_summary.json"
        if not existing.is_file():
            raise Stage1PilotError("resume requested but checkpoint is incomplete")
        return _read_json(existing) | {"resumed": True, "stage1_reexecuted": False}
    manifest = load_stage1_manifest(manifest_path)
    if str(manifest["source_git_commit"]) != "414b20ff13962a1f080650ec0f87e55307562a71":
        raise Stage1PilotError("stage1 source commit is not the frozen 414b20f commit")
    source_info = _source_artifacts(manifest, Path(preview_dir), Path(audit_root))
    audit_manifest = load_section_plan_audit_manifest("data/scenarios/section_plan_feasibility_audit_v1.json")
    context = load_scenario_context(TARGET_SCENARIO_ID, audit_manifest=audit_manifest, audit_root=Path(audit_root), config_dir=config_dir)
    domains, domain_summary = build_frozen_placement_domains(context, preview_dir)
    if domain_summary.source_candidate_count != 879 or domain_summary.raw_destination_records != 1738 or domain_summary.unique_destination_options != 529 or domain_summary.deduplication_removed != 1209 or domain_summary.editable_logical_section_count != 312 or domain_summary.total_unique_placement_options != 841:
        raise Stage1PilotError("frozen placement domain count drift")
    hashes = frozen_domain_hashes(domains, domain_summary.source_candidate_ids, context.allocation_input)
    _check_domain_fingerprint(manifest, hashes)
    domain_payload = asdict(domain_summary) | {"hashes": hashes, "dynamic_pruning": False, "source_only": True}
    rss_before = _rss_bytes()
    rules = _load_math_fallback_rules(Path(config_dir), context.catalog)
    build = build_joint_model(
        context.allocation_input,
        placement_domains=domains,
        math_fallback_rules=rules,
        math_course_ids=math_course_ids_from_catalog(context.catalog),
    )
    rss_after = _rss_bytes()
    metrics = model_proto_metrics(build.model, build, rss_before, rss_after)
    violations = cost_gate(metrics)
    config = {
        "solver_seed": SOLVER_SEED,
        "workers": WORKERS,
        "max_time_in_seconds": float(stage1_budget_seconds or manifest["stage1_budget_seconds"]),
        "external_persisted_seed": False,
        "stop_after_first_complete_solution": False,
        "objective": "minimize_sum_section_changed",
        "stage2_allowed": False,
        "stage3_allowed": False,
        "stage4_allowed": False,
    }
    hint_audit: dict[str, Any] = {"status": "not_run"}
    stage: dict[str, Any] = {"status": "SKIPPED", "incumbent_found": False, "assignment_available": False, "not_run_reason": "cost_gate" if violations else "not_run"}
    solver_log: tuple[str, ...] = ()
    witness: dict[str, Any] = {"joint_stage1_witness_valid": False, "status": "not_run", "not_run_reason": "no_stage1_incumbent"}
    acceptance: dict[str, Any] = {"status": "not_run", "production_fixed_witness_accepted": False, "not_run_reason": "no_stage1_witness"}
    production: dict[str, Any] = {"status": "not_run", "independently_validated_period_repair": False, "not_run_reason": "fixed_witness_acceptance_not_passed"}
    failures: dict[str, Any] = {"failures": [], "unexpected_failure_count": 0}
    if not violations:
        baseline = run_constrained_first_baseline(
            context.allocation_input,
            SOLVER_SEED,
            math_fallback_rules=rules,
            math_course_ids=math_course_ids_from_catalog(context.catalog),
        )
        selected = _selected_keys((assignment.request_key, assignment.linked_section_group_id) for assignment in baseline.assignments)
        hint_audit_obj = apply_stage1_hints(build, selected)
        hint_audit = asdict(hint_audit_obj)
        run = solve_stage1(build, seed=SOLVER_SEED, time_limit_seconds=float(stage1_budget_seconds or manifest["stage1_budget_seconds"]))
        stage = asdict(run)
        solver_log = tuple(stage.get("solver_log") or ())
        stage["solver_log"] = None
        stage["hint_source"] = "constrained_first_internal"
        stage["repair_hint_enabled"] = False
        stage["conditional_on_unproven_incumbent"] = False
        if run.incumbent_found:
            edited_sections = apply_placement_map_to_sections(context, dict(run.selected_placements))
            edited_input = canonicalize_allocation_input(
                context.students.copy(deep=True), context.requests.copy(deep=True),
                edited_sections, context.catalog.copy(deep=True),
            )
            if run.status == "OPTIMAL" and run.objective_value == 0:
                witness = {
                    "joint_stage1_witness_valid": False,
                    "status": "correctness_failure",
                    "not_run_reason": "objective_zero_contradicts_known_normal_dev_10_infeasibility",
                    "selected_assignment_count": len(run.selected_assignments),
                    "selected_placements": [[sid, list(place)] for sid, place in run.selected_placements],
                    "response_hash": run.response_hash,
                }
                failures["failures"].append("objective_zero_contradicts_known_normal_dev_10_infeasibility")
                failures["unexpected_failure_count"] = 1
            else:
                witness = validate_joint_witness(build, run, edited_input)
            if run.status != "OPTIMAL" or run.objective_value != 0:
                if witness.get("joint_stage1_witness_valid"):
                    placement_map = dict(run.selected_placements)
                    acceptance = production_fixed_witness_acceptance(
                    context, placement_map, run.selected_assignments,
                    config_dir=Path(config_dir),
                    seed=SOLVER_SEED,
                    time_limit_seconds=float(manifest["fixed_witness_acceptance_budget_seconds"]),
                )
                acceptance["production_fixed_witness_accepted"] = bool(
                    acceptance.get("status") in {"FEASIBLE", "OPTIMAL"}
                    and acceptance.get("assignment_exact")
                    and acceptance.get("policy_pass")
                    and acceptance.get("consistency_issue_count") == 0
                    and acceptance.get("response_hash")
                )
                if acceptance["production_fixed_witness_accepted"]:
                    production = independent_production_validation(
                        context, placement_map, config_dir=Path(config_dir), seed=SOLVER_SEED,
                        time_limit_seconds=float(manifest["production_validation_budget_seconds"]),
                    )
    aggregate = {
        "experiment_name": manifest["experiment_name"],
        "experiment_version": manifest["experiment_version"],
        "phase": manifest["phase"],
        "target_scenario_id": TARGET_SCENARIO_ID,
        "control_solver_runs": 0,
        "target_stage1_runs": 1 if not violations else 0,
        "fixed_witness_acceptance_runs": int(acceptance.get("status") not in {"not_run", None}),
        "production_cold_start_validation_runs": int(production.get("status") not in {"not_run", None}),
        "stage2_runs": 0,
        "stage3_runs": 0,
        "stage4_runs": 0,
        "other_normal_target_runs": 0,
        "stress_runs": 0,
        "negative_runs": 0,
        "holdout_runs": 0,
        "external_persisted_seed": False,
        "cost_gate": {"passed": not violations, "violations": violations},
        "stage1_status": stage.get("status"),
        "joint_stage1_witness_valid": witness.get("joint_stage1_witness_valid", False),
        "production_fixed_witness_accepted": acceptance.get("production_fixed_witness_accepted", False),
        "independently_validated_period_repair": production.get("independently_validated_period_repair", False),
        "minimum_claim": minimum_claim(stage, witness, acceptance, production),
        "resumed": False,
    }
    provenance = {
        "source_git_commit": manifest["source_git_commit"],
        "source_artifacts_read_only": True,
        "no_new_solver_runs_before_stage1": True,
        "control_solver_runs": 0,
        "other_normal_target_runs": 0,
        "stage2_runs": 0,
        "stage3_runs": 0,
        "stage4_runs": 0,
        "stress_runs": 0,
        "negative_runs": 0,
        "holdout_runs": 0,
        "external_persisted_seed": False,
        "target_stage1_only": True,
    }
    _artifact_payloads(
        output,
        manifest=manifest,
        provenance=provenance,
        source_info=source_info,
        domain=domain_payload,
        hint=hint_audit,
        model=metrics | {"cost_gate_violations": violations},
        config=config,
        stage=stage,
        witness=witness,
        acceptance=acceptance,
        production=production,
        aggregate=aggregate,
        failures=failures,
        solver_log=solver_log,
    )
    write_checksums(output)
    return aggregate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the frozen normal_dev_10 Stage 1 joint period-edit pilot.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--preview-dir", default=str(DEFAULT_PREVIEW_OUTPUT))
    parser.add_argument("--audit-root", default=str(DEFAULT_AUDIT_ROOT))
    parser.add_argument("--config-dir", default="data/config")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        summary = run_stage1_pilot(
            manifest_path=args.manifest,
            preview_dir=args.preview_dir,
            audit_root=args.audit_root,
            config_dir=args.config_dir,
            output_dir=args.output_dir,
            resume=args.resume,
        )
    except (Stage1PilotError, JointPilotError, OSError, ValueError) as exc:
        print(f"Joint period-edit Stage 1 pilot FAILED: {exc}")
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
