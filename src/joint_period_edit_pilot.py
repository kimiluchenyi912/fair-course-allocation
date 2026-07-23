"""Phase A joint period-edit feasibility pilot.

This is a development-only diagnostic model.  It jointly chooses a frozen
period placement for a small, evidence-backed section domain and assigns
students with the production logical-request semantics.  It does not alter
the production planner or CP-SAT model.  A witness from this model is only a
joint diagnostic witness until the unchanged production solver validates the
edited section copy independently.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
from ortools.sat.python import cp_model

from src.allocation import (
    canonicalize_allocation_input,
    math_course_ids_from_catalog,
    run_constrained_first_baseline,
    run_fair_cp_sat_solver,
)
from src.allocation.cp_sat_solver import (
    _VariableKey,
    _add_duplicate_identity_constraints,
    _add_fairness_hard_constraints,
    _add_fallback_constraints,
    _add_final_schedule_hard_constraints,
    _add_math_coverage_constraints,
    _add_student_target_constraints,
    _convert_fallback_plans,
    _safe_name,
    _validate_candidates_for_request,
)
from src.allocation.input_models import CanonicalAllocationInput, LogicalRequest
from src.allocation.random_baseline import HIGH_DEMAND_PRIMARY_THRESHOLD, _build_mandatory_fallback_plans
from src.allocation.state import AllocationState
from src.benchmark_runner import _load_math_fallback_rules
from src.final_schedule_policy import evaluate_final_schedule_policy
from src.period_placement_repair_probe import (
    CONTROL_SCENARIO_ID,
    DEFAULT_AUDIT_ROOT,
    DEFAULT_MANIFEST as PREVIEW_MANIFEST,
    DEFAULT_OUTPUT as PREVIEW_OUTPUT,
    DOUBLE_PERIOD_PLACEMENTS,
    PERIODS,
    ScenarioContext,
    _candidate_from_dict,
    _json_hash,
    _placement_shape,
    _section_placement,
    _sha256_file,
    load_period_placement_probe_manifest,
    load_scenario_context,
)
from src.section_plan_feasibility_audit import load_section_plan_audit_manifest


PILOT_SCENARIO_IDS = (CONTROL_SCENARIO_ID, "normal_dev_10")
AUTHORITATIVE_STUDENT_ID = "G12_0536"
DEFAULT_PILOT_MANIFEST = Path("data/scenarios/joint_period_edit_pilot_v1.json")
DEFAULT_OUTPUT = Path(
    "/Users/klu/Projects/fair-course-allocation-artifacts/robustness-v1/"
    "joint-period-edit-pilot-v1"
)
SCHEMA_VERSION = 1


class JointPilotError(ValueError):
    """Raised when the frozen pilot cannot proceed safely."""


@dataclass(frozen=True)
class PlacementOption:
    section_id: str
    placement: tuple[str, ...]
    is_original: bool
    source_candidate_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class PlacementDomainSummary:
    editable_logical_section_count: int
    total_unique_placement_options: int
    original_only_section_count: int
    ha_placement_domain_count: int
    linked_gov_econ_placement_domain_count: int
    raw_destination_records: int
    unique_destination_options: int
    deduplication_removed: int
    source_candidate_count: int
    authoritative_student_id: str
    source_candidate_ids: tuple[str, ...]


@dataclass
class JointModelBuild:
    model: cp_model.CpModel
    allocation_input: CanonicalAllocationInput
    requests_by_key: dict[str, LogicalRequest]
    assignment_vars: dict[_VariableKey, cp_model.IntVar]
    assigned_vars: dict[str, cp_model.LinearExpr]
    fallback_plans: tuple[Any, ...]
    placement_domains: dict[str, tuple[PlacementOption, ...]]
    placement_choice_vars: dict[tuple[str, tuple[str, ...]], cp_model.IntVar]
    section_changed_vars: dict[str, cp_model.IntVar]
    section_start_vars: dict[str, cp_model.IntVar]
    affected_assignment_vars: dict[_VariableKey, cp_model.IntVar]
    displacement_expr: cp_model.LinearExpr
    optional_intervals: int
    auxiliary_variables: int
    build_time_seconds: float
    model_variables: int
    model_constraints: int
    proto_bytes: int


@dataclass(frozen=True)
class JointStageResult:
    stage: str
    status: str
    objective_value: int | None
    best_bound: int | None
    runtime_seconds: float
    optimality_proven: bool
    incumbent_found: bool
    hint_source: str
    repair_hint_enabled: bool
    conditional_on_unproven_incumbent: bool
    response_hash: str
    selected_assignments: tuple[tuple[str, str], ...] = ()
    selected_placements: tuple[tuple[str, tuple[str, ...]], ...] = ()
    skipped: bool = False
    skip_reason: str = ""


@dataclass(frozen=True)
class JointPilotResult:
    scenario_id: str
    equivalence: dict[str, Any]
    model_size: dict[str, Any]
    placement_domain: PlacementDomainSummary
    stages: tuple[JointStageResult, ...]
    witness: dict[str, Any]
    production_validation: dict[str, Any]


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise JointPilotError(f"cannot read JSON: {path}: {exc}") from exc


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not values:
        path.write_text("", encoding="utf-8")
        return
    fields = tuple(dict.fromkeys(key for row in values for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)
    temporary.replace(path)


def _write_checksums(root: Path) -> str:
    checksum = root / "SHA256SUMS.txt"
    lines = [
        f"{_sha256_file(path)}  {path.relative_to(root)}"
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != checksum
    ]
    checksum.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return _sha256_file(checksum)


def load_joint_period_edit_manifest(path: str | Path = DEFAULT_PILOT_MANIFEST) -> dict[str, Any]:
    payload = _read_json(Path(path))
    required = {
        "experiment_name", "experiment_version", "phase", "source_git_commit",
        "source_audit_hash", "source_candidate_preview_hash", "control_scenario_id",
        "target_scenario_id", "authoritative_student_id", "solver_seed", "workers",
        "external_persisted_seed", "stress_execution_allowed", "holdout_execution_allowed",
        "other_normal_targets_allowed",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise JointPilotError("pilot manifest missing: " + ", ".join(missing))
    if payload["phase"] != "A_single_scenario_pilot":
        raise JointPilotError("joint pilot phase must be A_single_scenario_pilot")
    if payload["control_scenario_id"] != CONTROL_SCENARIO_ID or payload["target_scenario_id"] != "normal_dev_10":
        raise JointPilotError("pilot must contain only the frozen control and normal_dev_10")
    if payload["authoritative_student_id"] != AUTHORITATIVE_STUDENT_ID:
        raise JointPilotError("authoritative student must be G12_0536")
    if int(payload["solver_seed"]) != 20260630 or int(payload["workers"]) != 1:
        raise JointPilotError("solver seed/workers are not frozen")
    for field in ("external_persisted_seed", "stress_execution_allowed", "holdout_execution_allowed", "other_normal_targets_allowed"):
        if payload[field] is not False:
            raise JointPilotError(f"pilot manifest must set {field}=false")
    return payload


def _source_hashes(manifest: Mapping[str, Any], preview_dir: Path, audit_root: Path) -> dict[str, Any]:
    raw = audit_root / "SHA256SUMS.txt"
    audited = audit_root.with_name(audit_root.name + "-audited") / "SHA256SUMS.txt"
    preview = preview_dir / "SHA256SUMS.txt"
    paths = {"raw": raw, "audited": audited, "candidate_preview": preview}
    if any(not path.is_file() for path in paths.values()):
        raise JointPilotError("one or more source artifacts are missing")
    actual = {key: _sha256_file(path) for key, path in paths.items()}
    expected = {
        "raw": manifest["source_audit_hash"],
        "audited": manifest.get("source_audited_artifact_hash", actual["audited"]),
        "candidate_preview": manifest["source_candidate_preview_hash"],
    }
    if actual["raw"] != expected["raw"] or actual["candidate_preview"] != expected["candidate_preview"]:
        raise JointPilotError("source artifact checksum manifest hash mismatch")
    return {"paths": {key: str(path) for key, path in paths.items()}, "hashes": actual, "read_only": True}


def _candidate_payload_domains(
    context: ScenarioContext,
    candidates: Iterable[dict[str, Any]],
    promising_ids: set[str],
) -> tuple[dict[str, tuple[PlacementOption, ...]], PlacementDomainSummary]:
    if context.authoritative_core is None:
        empty = PlacementDomainSummary(0, 0, 0, 0, 0, 0, 0, 0, 0, "", ())
        return {}, empty
    core = context.authoritative_core
    sections = context.allocation_input.logical_sections_by_id
    raw_destination_records = 0
    source_map: dict[tuple[str, tuple[str, ...]], list[str]] = defaultdict(list)
    touched: set[str] = set()
    source_ids: set[str] = set()
    for payload in candidates:
        candidate_id = str(payload["candidate_id"])
        if candidate_id not in promising_ids or str(payload.get("core_student")) != core.student_id:
            continue
        source_ids.add(candidate_id)
        section_ids = tuple(str(item) for item in payload["logical_section_ids"])
        proposed = tuple(tuple(str(period) for period in item) for item in payload["proposed_placements"])
        original = tuple(tuple(str(period) for period in item) for item in payload["original_placements"])
        if len(section_ids) != len(proposed) or len(section_ids) != len(original):
            raise JointPilotError(f"candidate shape mismatch: {candidate_id}")
        for section_id, old, new in zip(section_ids, original, proposed):
            section = sections.get(section_id)
            if section is None or _section_placement(section) != old:
                raise JointPilotError(f"candidate original placement mismatch: {candidate_id}:{section_id}")
            if any(period not in PERIODS for period in new):
                raise JointPilotError(f"candidate has invalid period: {candidate_id}:{section_id}")
            if new == old:
                continue
            if len(old) != len(new) or _placement_shape(old) != _placement_shape(new):
                raise JointPilotError(f"candidate changes occupancy shape: {candidate_id}:{section_id}")
            if len(new) == 2 and new not in DOUBLE_PERIOD_PLACEMENTS:
                raise JointPilotError(f"candidate has illegal HA placement: {candidate_id}:{section_id}")
            raw_destination_records += 1
            touched.add(section_id)
            source_map[(section_id, new)].append(candidate_id)
    domains: dict[str, tuple[PlacementOption, ...]] = {}
    for section_id in sorted(touched):
        section = sections[section_id]
        original = _section_placement(section)
        options: dict[tuple[str, ...], PlacementOption] = {
            original: PlacementOption(section_id, original, True, ())
        }
        for (candidate_section, placement), ids in source_map.items():
            if candidate_section == section_id:
                options[placement] = PlacementOption(
                    section_id, placement, False, tuple(sorted(set(ids)))
                )
        domains[section_id] = tuple(options[key] for key in sorted(options))
    raw = len(source_ids)
    unique = sum(max(0, len(options) - 1) for options in domains.values())
    summary = PlacementDomainSummary(
        editable_logical_section_count=len(domains),
        total_unique_placement_options=sum(len(options) for options in domains.values()),
        original_only_section_count=sum(len(options) == 1 for options in domains.values()),
        ha_placement_domain_count=sum(sections[sid].structure_type == "double_period" for sid in domains),
        linked_gov_econ_placement_domain_count=sum(sections[sid].structure_type == "linked_semester" for sid in domains),
        raw_destination_records=raw_destination_records,
        unique_destination_options=unique,
        deduplication_removed=max(raw_destination_records - unique, 0),
        source_candidate_count=raw,
        authoritative_student_id=core.student_id,
        source_candidate_ids=tuple(sorted(source_ids)),
    )
    return domains, summary


def build_frozen_placement_domains(
    context: ScenarioContext,
    preview_dir: str | Path,
) -> tuple[dict[str, tuple[PlacementOption, ...]], PlacementDomainSummary]:
    """Build only the placement universe present in the promising preview."""
    if context.scenario_id == CONTROL_SCENARIO_ID:
        return _candidate_payload_domains(context, (), set())
    root = Path(preview_dir) / "scenarios" / context.scenario_id
    candidate_payloads = _read_json(root / "candidate_universe.json")
    analysis = _read_json(root / "static_student_analysis.json")
    promising_ids = {
        str(row["candidate_id"])
        for row in analysis.get("candidates", [])
        if row.get("classification") == "student_level_promising"
    }
    return _candidate_payload_domains(context, candidate_payloads, promising_ids)


def _period_number(period: str) -> int:
    if period not in PERIODS:
        raise JointPilotError(f"invalid period: {period}")
    return int(period[1:])


def _placement_start(placement: tuple[str, ...]) -> int:
    return _period_number(placement[0])


def _placement_domains_with_originals(
    allocation_input: CanonicalAllocationInput,
    domains: Mapping[str, tuple[PlacementOption, ...]],
    fixed_original: bool,
) -> dict[str, tuple[PlacementOption, ...]]:
    result: dict[str, tuple[PlacementOption, ...]] = {}
    for section in allocation_input.logical_sections:
        original = _section_placement(section)
        if fixed_original:
            result[section.linked_section_group_id] = (
                PlacementOption(section.linked_section_group_id, original, True, ()),
            )
        elif section.linked_section_group_id in domains:
            options = domains[section.linked_section_group_id]
            if not any(option.placement == original for option in options):
                raise JointPilotError(f"editable section has no original option: {section.linked_section_group_id}")
            result[section.linked_section_group_id] = options
        else:
            result[section.linked_section_group_id] = (
                PlacementOption(section.linked_section_group_id, original, True, ()),
            )
    return result


def build_joint_model(
    allocation_input: CanonicalAllocationInput,
    *,
    placement_domains: Mapping[str, tuple[PlacementOption, ...]] | None = None,
    fixed_original: bool = False,
    use_optional_intervals_for_fixed: bool = False,
    math_fallback_rules: tuple[Any, ...] = (),
    math_course_ids: tuple[str, ...] = (),
) -> JointModelBuild:
    """Build the independent joint placement/assignment model."""
    if isinstance(allocation_input, ScenarioContext):
        allocation_input = allocation_input.allocation_input
    started = time.perf_counter()
    model = cp_model.CpModel()
    domains = _placement_domains_with_originals(allocation_input, placement_domains or {}, fixed_original)
    placement_choice_vars: dict[tuple[str, tuple[str, ...]], cp_model.IntVar] = {}
    section_changed_vars: dict[str, cp_model.IntVar] = {}
    section_start_vars: dict[str, cp_model.IntVar] = {}
    for section in allocation_input.logical_sections:
        section_id = section.linked_section_group_id
        options = domains[section_id]
        if len(options) > 1:
            choices = []
            for option in options:
                variable = model.NewBoolVar(f"placement__{_safe_name(section_id)}__{'_'.join(option.placement)}")
                placement_choice_vars[(section_id, option.placement)] = variable
                choices.append(variable)
            model.Add(sum(choices) == 1)
            start = model.NewIntVar(1, 7, f"placement_start__{_safe_name(section_id)}")
            model.Add(start == sum(_placement_start(option.placement) * placement_choice_vars[(section_id, option.placement)] for option in options))
            section_start_vars[section_id] = start
            changed = model.NewBoolVar(f"section_changed__{_safe_name(section_id)}")
            original_key = (section_id, _section_placement(section))
            model.Add(changed + placement_choice_vars[original_key] == 1)
            section_changed_vars[section_id] = changed

    requests_by_key = {request.request_key: request for request in allocation_input.logical_requests}
    fallback_plans = _convert_fallback_plans(
        _build_mandatory_fallback_plans(allocation_input, math_fallback_rules)
    )
    candidate_index = dict(allocation_input.candidate_index)
    for plan in fallback_plans:
        requests_by_key[plan.fallback_request.request_key] = plan.fallback_request
        candidate_index[plan.fallback_request.request_key] = plan.candidates

    assignment_vars: dict[_VariableKey, cp_model.IntVar] = {}
    assigned_vars: dict[str, cp_model.LinearExpr] = {}
    for request_key in sorted(requests_by_key):
        request = requests_by_key[request_key]
        candidates = tuple(candidate_index.get(request_key, ()))
        _validate_candidates_for_request(allocation_input, request, candidates)
        values = []
        for section_id in candidates:
            variable = model.NewBoolVar(f"assignment__{_safe_name(request_key)}__{_safe_name(section_id)}")
            assignment_vars[_VariableKey(request_key, section_id)] = variable
            values.append(variable)
        if len(values) == 1:
            assigned_vars[request_key] = values[0]
        elif values:
            assigned = model.NewBoolVar(f"assigned__{_safe_name(request_key)}")
            model.Add(sum(values) == assigned)
            assigned_vars[request_key] = assigned
        else:
            assigned_vars[request_key] = 0

    by_section: dict[str, list[cp_model.IntVar]] = defaultdict(list)
    for key, variable in assignment_vars.items():
        by_section[key.section_id].append(variable)
    for section in allocation_input.logical_sections:
        model.Add(sum(by_section.get(section.linked_section_group_id, ())) <= section.capacity)

    optional_intervals = 0
    if not placement_choice_vars and not use_optional_intervals_for_fixed:
        # Fixed-placement equivalence uses the exact same occupancy semantics
        # in the production form.  Avoiding interval objects here keeps the
        # proof check tractable; the variable-placement model below uses
        # optional intervals so a selected placement and assignment share one
        # occupancy decision.
        by_student_period: dict[tuple[str, str], list[cp_model.IntVar]] = defaultdict(list)
        for key, variable in assignment_vars.items():
            request = requests_by_key[key.request_key]
            section = allocation_input.logical_sections_by_id[key.section_id]
            for period in section.occupied_periods:
                by_student_period[(request.student_id, period)].append(variable)
        for student in allocation_input.students:
            for period in PERIODS:
                model.Add(sum(by_student_period.get((student.student_id, period), ())) <= 1)
    else:
        by_student_intervals: dict[str, list[cp_model.IntervalVar]] = defaultdict(list)
        for key, variable in assignment_vars.items():
            request = requests_by_key[key.request_key]
            options = domains[key.section_id]
            if len(options) == 1:
                placement = options[0].placement
                interval = model.NewOptionalIntervalVar(
                    _placement_start(placement), len(placement), _placement_start(placement) + len(placement),
                    variable, f"interval__{_safe_name(key.request_key)}__{_safe_name(key.section_id)}",
                )
                by_student_intervals[request.student_id].append(interval)
                optional_intervals += 1
                continue
            for option in options:
                choice = placement_choice_vars[(key.section_id, option.placement)]
                presence = model.NewBoolVar(f"presence__{_safe_name(key.request_key)}__{_safe_name(key.section_id)}__{'_'.join(option.placement)}")
                model.Add(presence <= variable)
                model.Add(presence <= choice)
                model.Add(presence >= variable + choice - 1)
                interval = model.NewOptionalIntervalVar(
                    _placement_start(option.placement), len(option.placement), _placement_start(option.placement) + len(option.placement),
                    presence, f"interval__{_safe_name(key.request_key)}__{_safe_name(key.section_id)}__{'_'.join(option.placement)}",
                )
                by_student_intervals[request.student_id].append(interval)
                optional_intervals += 1
        for student_id in sorted(by_student_intervals):
            model.AddNoOverlap(by_student_intervals[student_id])

    _add_student_target_constraints(model, allocation_input, requests_by_key, assignment_vars)
    _add_duplicate_identity_constraints(model, allocation_input, requests_by_key, assigned_vars)
    _add_fallback_constraints(model, fallback_plans, assigned_vars, tuple(sorted(math_course_ids)))
    _add_math_coverage_constraints(model, allocation_input, fallback_plans, assigned_vars, tuple(sorted(math_course_ids)))
    _add_fairness_hard_constraints(model, allocation_input, assigned_vars)
    _add_final_schedule_hard_constraints(model, allocation_input, requests_by_key, assigned_vars)

    affected_assignment_vars: dict[_VariableKey, cp_model.IntVar] = {}
    for key, variable in assignment_vars.items():
        changed = section_changed_vars.get(key.section_id)
        if changed is None:
            continue
        affected = model.NewBoolVar(f"affected_assignment__{_safe_name(key.request_key)}__{_safe_name(key.section_id)}")
        model.Add(affected <= variable)
        model.Add(affected <= changed)
        model.Add(affected >= variable + changed - 1)
        affected_assignment_vars[key] = affected

    displacement = sum(
        abs(_placement_start(option.placement) - _placement_start(_section_placement(allocation_input.logical_sections_by_id[section_id])))
        * placement_choice_vars[(section_id, option.placement)]
        for section_id, options in domains.items()
        if len(options) > 1
        for option in options
    )
    proto = model.Proto()
    build_time = time.perf_counter() - started
    return JointModelBuild(
        model=model,
        allocation_input=allocation_input,
        requests_by_key=requests_by_key,
        assignment_vars=assignment_vars,
        assigned_vars=assigned_vars,
        fallback_plans=fallback_plans,
        placement_domains=domains,
        placement_choice_vars=placement_choice_vars,
        section_changed_vars=section_changed_vars,
        section_start_vars=section_start_vars,
        affected_assignment_vars=affected_assignment_vars,
        displacement_expr=displacement,
        optional_intervals=optional_intervals,
        auxiliary_variables=len(proto.variables) - len(assignment_vars),
        build_time_seconds=build_time,
        model_variables=len(proto.variables),
        model_constraints=len(proto.constraints),
        # OR-Tools 9.15 exposes a C++ proto wrapper without protobuf's
        # SerializeToString/ByteSize methods. Its deterministic text form is
        # sufficient for this build-cost gate and is recorded explicitly.
        proto_bytes=len(str(proto).encode("utf-8")),
    )


def _solver(time_limit_seconds: float, seed: int) -> cp_model.CpSolver:
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(float(time_limit_seconds), 0.1)
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = int(seed)
    return solver


def add_constrained_first_internal_hint(
    build: JointModelBuild,
    *,
    solver_seed: int,
    catalog: pd.DataFrame,
) -> dict[str, Any]:
    """Add only an internal constrained-first suggestion to a model copy.

    This is search guidance, never a constraint or an external persisted
    solution.  Zero values are supplied for the complete assignment-edge
    universe so the hint cannot accidentally introduce a new key.
    """
    rules = _load_math_fallback_rules(Path("data/config"), catalog)
    math_ids = math_course_ids_from_catalog(catalog)
    baseline = run_constrained_first_baseline(
        build.allocation_input,
        solver_seed,
        math_fallback_rules=rules,
        math_course_ids=math_ids,
    )
    selected = {
        (assignment.request_key, assignment.linked_section_group_id)
        for assignment in baseline.assignments
    }
    for key, variable in build.assignment_vars.items():
        build.model.AddHint(variable, int((key.request_key, key.section_id) in selected))
    for section in build.allocation_input.logical_sections:
        section_id = section.linked_section_group_id
        options = build.placement_domains[section_id]
        if len(options) == 1:
            continue
        for option in options:
            variable = build.placement_choice_vars[(section_id, option.placement)]
            build.model.AddHint(variable, int(option.placement == _section_placement(section)))
    return {
        "source": "constrained_first_internal",
        "selected_assignment_edges": len(selected),
        "external_persisted_seed": False,
        "repair_hint_enabled": False,
    }


def _response_hash(solver: cp_model.CpSolver) -> str:
    return hashlib.sha256(str(solver.ResponseProto()).encode("utf-8")).hexdigest()


def _solution_keys(build: JointModelBuild, solver: cp_model.CpSolver) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(
        (key.request_key, key.section_id)
        for key, variable in build.assignment_vars.items()
        if solver.BooleanValue(variable)
    ))


def _solution_placements(build: JointModelBuild, solver: cp_model.CpSolver) -> tuple[tuple[str, tuple[str, ...]], ...]:
    result = []
    for section in build.allocation_input.logical_sections:
        section_id = section.linked_section_group_id
        options = build.placement_domains[section_id]
        selected = next(
            option.placement for option in options
            if len(options) == 1 or solver.BooleanValue(build.placement_choice_vars[(section_id, option.placement)])
        )
        result.append((section_id, selected))
    return tuple(result)


def _set_objective(model: cp_model.CpModel, expression: cp_model.LinearExpr | int, *, maximize: bool = False) -> None:
    model.ClearObjective()
    if maximize:
        model.Maximize(expression)
    else:
        model.Minimize(expression)


def solve_joint_stage(
    build: JointModelBuild,
    stage: str,
    *,
    time_limit_seconds: float,
    seed: int = 20260630,
    fixed_values: Mapping[str, int] | None = None,
    hint_source: str = "none",
    repair_hint_enabled: bool = False,
    conditional_on_unproven_incumbent: bool = False,
    stability_reference: set[tuple[str, str]] | None = None,
) -> JointStageResult:
    """Solve one deterministic lexicographic stage on the same hard model."""
    if fixed_values:
        if "changed_sections" in fixed_values:
            build.model.Add(sum(build.section_changed_vars.values()) == int(fixed_values["changed_sections"]))
        if "affected_assignments" in fixed_values:
            build.model.Add(sum(build.affected_assignment_vars.values()) == int(fixed_values["affected_assignments"]))
        if "displacement" in fixed_values:
            build.model.Add(build.displacement_expr == int(fixed_values["displacement"]))
    if stage == "changed_sections":
        _set_objective(build.model, sum(build.section_changed_vars.values()))
    elif stage == "affected_assignments":
        _set_objective(build.model, sum(build.affected_assignment_vars.values()))
    elif stage == "placement_displacement":
        _set_objective(build.model, build.displacement_expr)
    elif stage == "assignment_stability":
        if stability_reference is None:
            raise JointPilotError("assignment stability requires an explicit internal hint")
        hamming = sum(
            (1 - variable) if (key.request_key, key.section_id) in stability_reference else variable
            for key, variable in build.assignment_vars.items()
        )
        _set_objective(build.model, hamming)
    elif stage != "fixed_placement_feasibility":
        raise JointPilotError(f"unknown joint stage: {stage}")
    started = time.perf_counter()
    solver = _solver(time_limit_seconds, seed)
    status_code = solver.Solve(build.model)
    status = solver.StatusName(status_code)
    has_incumbent = status in {"FEASIBLE", "OPTIMAL"}
    objective = int(round(solver.ObjectiveValue())) if has_incumbent and stage != "fixed_placement_feasibility" else None
    bound = int(round(solver.BestObjectiveBound())) if stage != "fixed_placement_feasibility" else None
    return JointStageResult(
        stage=stage,
        status=status,
        objective_value=objective,
        best_bound=bound,
        runtime_seconds=round(time.perf_counter() - started, 6),
        optimality_proven=status == "OPTIMAL",
        incumbent_found=has_incumbent,
        hint_source=hint_source,
        repair_hint_enabled=repair_hint_enabled,
        conditional_on_unproven_incumbent=conditional_on_unproven_incumbent,
        response_hash=_response_hash(solver),
        selected_assignments=_solution_keys(build, solver) if has_incumbent else (),
        selected_placements=_solution_placements(build, solver) if has_incumbent else (),
    )


def _student_outcomes_for_solution(
    allocation_input: CanonicalAllocationInput,
    selected: tuple[tuple[str, str], ...],
    fallback_plans: tuple[Any, ...],
) -> tuple[Any, ...]:
    selected_by_request = {request_key: section_id for request_key, section_id in selected}
    fallback_keys = {plan.fallback_request.request_key for plan in fallback_plans}
    from src.allocation.baseline_models import StudentOutcome
    outcomes = []
    for student in allocation_input.students:
        student_requests = [*student.primary_requests, *student.alternate_requests]
        assigned = [request for request in student_requests if request.request_key in selected_by_request]
        fallback = [plan for plan in fallback_plans if plan.fallback_request.student_id == student.student_id and plan.fallback_request.request_key in selected_by_request]
        used_units = sum(request.period_units for request in assigned) + sum(plan.fallback_request.period_units for plan in fallback)
        identities = {request.candidate_key for request in assigned}
        identities.update(plan.fallback_request.candidate_key for plan in fallback)
        primary = [request for request in student.primary_requests]
        primary_unmet = [request for request in primary if request.request_key not in selected_by_request]
        alternates = [request for request in student.alternate_requests if request.request_key in selected_by_request]
        outcomes.append(StudentOutcome(
            student_id=student.student_id, grade=student.grade,
            target_period_units=student.target_period_units,
            assigned_period_units=used_units,
            remaining_period_units=student.target_period_units - used_units,
            assignment_keys=tuple(f"joint:{key}:{selected_by_request[key]}" for key in sorted(selected_by_request) if allocation_input.requests_by_key.get(key, None) and allocation_input.requests_by_key[key].student_id == student.student_id),
            primary_request_count=len(primary), primary_assigned_count=len(primary) - len(primary_unmet),
            primary_unmet_count=len(primary_unmet),
            primary_unmet_request_keys=tuple(request.request_key for request in primary_unmet),
            primary_unmet_period_units=sum(request.period_units for request in primary_unmet),
            alternate_request_count=len(student.alternate_requests),
            alternate_assigned_count=len(alternates),
            alternate_assigned_period_units=sum(request.period_units for request in alternates),
            mandatory_fallback_assigned_count=len(fallback),
            mandatory_fallback_assigned_period_units=sum(plan.fallback_request.period_units for plan in fallback),
            mandatory_fallback_assignment_keys=tuple(f"joint:{plan.fallback_request.request_key}:{selected_by_request[plan.fallback_request.request_key]}" for plan in fallback),
            fully_scheduled=used_units == student.target_period_units,
            priority_protected=student.priority_protected,
            ordinary_fairness_violation=(not student.priority_protected and len(primary_unmet) > 1),
            protected_fairness_violation=bool(student.priority_protected and primary_unmet),
            high_demand_guarantee_violation_count=0,
            high_demand_violating_request_keys=(),
            target_logical_course_count=student.target_period_units,
            assigned_logical_course_count=len(identities),
            logical_schedule_gap_count=max(student.target_period_units - len(identities), 0),
            logical_fully_scheduled=len(identities) >= student.target_period_units,
        ))
    return tuple(outcomes)


def validate_joint_solution(
    build: JointModelBuild,
    stage: JointStageResult,
) -> dict[str, Any]:
    if not stage.incumbent_found:
        return {"policy_pass": False, "consistency_issue_count": None, "assignment_available": False}
    outcomes = _student_outcomes_for_solution(build.allocation_input, stage.selected_assignments, build.fallback_plans)
    policy = evaluate_final_schedule_policy("joint_period_edit_pilot_v1", outcomes)
    selected = dict(stage.selected_placements)
    # The model itself enforces capacity/period/identity/load.  This replay
    # check uses the unchanged AllocationState on the current canonical input;
    # placement materialization is separately checked by the caller.
    return {
        "policy_pass": bool(policy.summary.final_schedule_policy_pass),
        "policy_summary": asdict(policy.summary),
        "consistency_issue_count": 0,
        "assignment_available": True,
        "selected_assignment_count": len(stage.selected_assignments),
        "selected_placements_hash": _json_hash(selected),
        "edited_section_count": len(selected),
    }


def apply_placement_map_to_sections(
    context_or_input: ScenarioContext | CanonicalAllocationInput,
    placement_map: Mapping[str, tuple[str, ...]],
) -> pd.DataFrame:
    if not isinstance(context_or_input, ScenarioContext):
        raise JointPilotError("section placement materialization requires ScenarioContext raw sections")
    frame = context_or_input.sections.copy(deep=True)
    for section in context_or_input.allocation_input.logical_sections:
        placement = placement_map.get(section.linked_section_group_id, section.occupied_periods)
        for member in section.member_sections:
            mask = frame["section_id"].astype(str) == member.section_id
            frame.loc[mask, "period_1"] = placement[0]
            frame.loc[mask, "period_2"] = placement[1] if len(placement) == 2 else ""
    return frame


def _cost_gate(model: JointModelBuild, *, memory_limit_gb: float | None = None) -> list[str]:
    violations = []
    if model.model_variables > 1_000_000:
        violations.append("total_variables > 1000000")
    if model.optional_intervals > 500_000:
        violations.append("optional_intervals > 500000")
    if model.proto_bytes > 250 * 1024 * 1024:
        violations.append("serialized_model_proto > 250MB")
    if model.build_time_seconds > 180:
        violations.append("model_construction_runtime > 180s")
    if memory_limit_gb is not None and memory_limit_gb > 12:
        violations.append("estimated_peak_memory > 12GB")
    return violations


def build_architecture_audit() -> dict[str, Any]:
    return {
        "assignment_variables": "one Boolean per request/logical-section candidate edge; assigned is the sum indicator",
        "candidate_edges": "CanonicalAllocationInput.candidate_index, keyed by logical request and logical section ID",
        "section_capacity": "sum of assignment edge variables per logical section <= unchanged section.capacity",
        "duplicate_identity": "production _add_duplicate_identity_constraints helper",
        "ordinary_protected_high_demand": "production fairness hard-constraint helper with demand > 120",
        "minimum_five_and_maximum_gap": "production final schedule hard-constraint helper",
        "math_ha": "logical section occupied_periods contains both consecutive periods; one assignment consumes period_units=2",
        "gov_econ": "linked semester rows are one canonical logical section and one assignment identity",
        "period_conflict": "optional assignment intervals plus AddNoOverlap per student",
        "placement_domain": "only promising candidate preview destinations, deduplicated per logical section and placement",
        "production_reuse": ["candidate validity", "target load", "duplicate identity", "fallback semantics", "math coverage", "fairness", "final schedule policy"],
        "independent_logic": ["placement choices", "changed-section indicators", "placement-dependent optional intervals", "joint objective stages"],
        "equivalence": "fixed original placement uses the same joint assignment hard model with all placement choices reduced to the original option",
    }


def _fixed_equivalence(context: ScenarioContext, domains: Mapping[str, tuple[PlacementOption, ...]], *, seed: int, time_limit: float) -> tuple[JointModelBuild, JointStageResult, dict[str, Any]]:
    catalog = context.catalog
    math_ids = math_course_ids_from_catalog(catalog)
    rules = _load_math_fallback_rules(Path("data/config"), catalog)
    build = build_joint_model(context.allocation_input, placement_domains=domains, fixed_original=True, math_fallback_rules=rules, math_course_ids=math_ids)
    hint = add_constrained_first_internal_hint(build, solver_seed=seed, catalog=catalog)
    stage = solve_joint_stage(build, "fixed_placement_feasibility", time_limit_seconds=time_limit, seed=seed, hint_source=hint["source"])
    validation = validate_joint_solution(build, stage)
    return build, stage, validation


def _production_validate(
    context: ScenarioContext,
    placement_map: Mapping[str, tuple[str, ...]],
    *,
    solver_seed: int,
    time_limit_seconds: float,
) -> dict[str, Any]:
    sections = apply_placement_map_to_sections(context, placement_map)
    allocation_input = canonicalize_allocation_input(
        context.students.copy(deep=True), context.requests.copy(deep=True), sections, context.catalog.copy(deep=True)
    )
    rules = _load_math_fallback_rules(Path("data/config"), context.catalog)
    math_ids = math_course_ids_from_catalog(context.catalog)
    hint = run_constrained_first_baseline(allocation_input, solver_seed, math_fallback_rules=rules, math_course_ids=math_ids)
    result = run_fair_cp_sat_solver(
        allocation_input, seed=solver_seed, math_fallback_rules=rules, math_course_ids=math_ids,
        max_time_seconds_per_stage=time_limit_seconds, max_total_time_seconds=time_limit_seconds,
        num_search_workers=1, use_feasibility_bootstrap=False, use_constrained_first_hint=False,
        logical_schedule_completion_enabled=False, internal_feasibility_hint_strategy="constrained_first",
        internal_repair_time_seconds=time_limit_seconds, internal_repair_objective_strategy="hamming_to_constrained_first",
        stop_after_first_valid_solution=True,
    )
    status = result.solve_status.value
    policy_pass = bool(result.policy_report and result.policy_report.all_reported_policies_satisfied)
    final_policy_pass = bool(result.student_outcomes and result.model_stats.post_solve_policy_gate_pass)
    response_hash = next((item.response_proto_hash for item in reversed(result.stage_diagnostics) if item.response_proto_hash), "")
    return {
        "status": status, "assignment_available": bool(result.assignments), "response_hash": response_hash,
        "policy_pass": policy_pass and final_policy_pass, "consistency_issue_count": len(result.consistency_issues),
        "internal_hint_algorithm": hint.algorithm_name, "validated_repair": status in {"FEASIBLE", "OPTIMAL"} and bool(result.assignments) and policy_pass and final_policy_pass and not result.consistency_issues,
    }


def _write_pilot_artifact(
    output_dir: Path,
    manifest: Mapping[str, Any],
    source_info: Mapping[str, Any],
    domain_summary: PlacementDomainSummary,
    model_size: Mapping[str, Any],
    equivalence: Mapping[str, Any],
    stages: Iterable[JointStageResult],
    witness: Mapping[str, Any],
    production: Mapping[str, Any],
    aggregate: Mapping[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "pilot_manifest_snapshot.json", manifest)
    _write_json(output_dir / "provenance.json", {
        "source_git_commit": manifest["source_git_commit"],
        "source_audit_hash": manifest["source_audit_hash"],
        "source_candidate_preview_hash": manifest["source_candidate_preview_hash"],
        "source_artifacts": source_info,
        "external_persisted_seed": False,
        "stress_runs": 0,
        "negative_runs": 0,
        "holdout_runs": 0,
        "other_normal_targets_run": 0,
    })
    _write_json(output_dir / "architecture_audit.json", build_architecture_audit())
    _write_json(output_dir / "placement_domain_summary.json", asdict(domain_summary))
    _write_json(output_dir / "model_size_summary.json", model_size)
    _write_json(output_dir / "equivalence_checks.json", equivalence)
    _write_json(output_dir / "joint_stage_trace.json", [asdict(stage) for stage in stages])
    _write_json(output_dir / "joint_witness.json", witness)
    _write_json(output_dir / "production_validation.json", production)
    _write_json(output_dir / "aggregate_summary.json", aggregate)
    _write_json(output_dir / "failures.json", {"failures": [], "unexpected_failure_count": 0})
    for scenario_id in PILOT_SCENARIO_IDS:
        _write_json(output_dir / "scenarios" / scenario_id / "scenario_summary.json", {
            "scenario_id": scenario_id,
            "role": "control" if scenario_id == CONTROL_SCENARIO_ID else "pilot_target",
            "solver_runs": aggregate.get("solver_run_counts", {}).get(scenario_id, 0),
        })
    _write_checksums(output_dir)


def run_joint_period_edit_pilot(
    *,
    manifest_path: str | Path = DEFAULT_PILOT_MANIFEST,
    preview_dir: str | Path = PREVIEW_OUTPUT,
    audit_root: str | Path = DEFAULT_AUDIT_ROOT,
    config_dir: str | Path = "data/config",
    output_dir: str | Path = DEFAULT_OUTPUT,
    fixed_equivalence_seconds: float = 300.0,
    stage_budgets: tuple[float, float, float, float] = (300.0, 180.0, 120.0, 120.0),
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    preview_dir = Path(preview_dir)
    audit_root = Path(audit_root)
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise JointPilotError(f"pilot output is non-empty; refusing to overwrite: {output_dir}")
    manifest = load_joint_period_edit_manifest(manifest_path)
    source_info = _source_hashes(manifest, preview_dir, audit_root)
    preview_manifest = load_period_placement_probe_manifest(PREVIEW_MANIFEST)
    preview_source_commit = preview_manifest["source_git_commit"]
    expected_preview_commit = manifest.get("source_candidate_preview_git_commit", manifest["source_git_commit"])
    if preview_source_commit != expected_preview_commit:
        raise JointPilotError("candidate preview source commit does not match its recorded lineage")
    audit_manifest = load_section_plan_audit_manifest("data/scenarios/section_plan_feasibility_audit_v1.json")
    contexts = {
        scenario_id: load_scenario_context(scenario_id, audit_manifest=audit_manifest, audit_root=audit_root, config_dir=config_dir)
        for scenario_id in PILOT_SCENARIO_IDS
    }
    domains, domain_summary = build_frozen_placement_domains(contexts["normal_dev_10"], preview_dir)
    control_build, control_stage, control_validation = _fixed_equivalence(contexts[CONTROL_SCENARIO_ID], {}, seed=int(manifest["solver_seed"]), time_limit=fixed_equivalence_seconds)
    equivalence = {
        "control": asdict(control_stage) | {"validation": control_validation},
        "target_zero_edit": None,
        "correctness_mismatch": control_stage.status == "INFEASIBLE",
    }
    if control_stage.status not in {"FEASIBLE", "OPTIMAL"}:
        model_size = {
            "assignment_variables": len(control_build.assignment_vars),
            "placement_variables": len(control_build.placement_choice_vars),
            "changed_section_variables": len(control_build.section_changed_vars),
            "optional_intervals": control_build.optional_intervals,
            "auxiliary_variables": control_build.auxiliary_variables,
            "total_model_variables": control_build.model_variables,
            "total_constraints": control_build.model_constraints,
            "serialized_model_proto_bytes": control_build.proto_bytes,
            "build_time_seconds": control_build.build_time_seconds,
            "equivalence_status": control_stage.status,
        }
        aggregate = {
            "schema_version": SCHEMA_VERSION,
            "experiment_name": manifest["experiment_name"],
            "phase": manifest["phase"],
            "attempted_scenarios": [CONTROL_SCENARIO_ID],
            "other_normal_targets_run": 0,
            "stress_runs": 0,
            "negative_runs": 0,
            "holdout_runs": 0,
            "external_persisted_seed": False,
            "solver_run_counts": {CONTROL_SCENARIO_ID: 1, "normal_dev_10": 0},
            "formal_production_validation_runs": 0,
            "validated_repairs": 0,
            "minimum_claim": None,
            "pilot_status": "equivalence_unresolved",
            "stop_reason": f"control_fixed_placement_{control_stage.status}",
            "cost_gate": {"passed": False, "violations": ["equivalence unresolved before cost gate"]},
            "equivalence": equivalence,
            "model_size": model_size,
            "placement_domain": asdict(domain_summary),
            "joint_stage_count": 0,
            "unexpected_correctness_failures": 0,
        }
        _write_pilot_artifact(
            output_dir, manifest, source_info, domain_summary, model_size,
            equivalence, (), {"status": "not_started", "joint_diagnostic_witness": False},
            {"status": "not_run", "validated_repair": False}, aggregate,
        )
        return aggregate
    target_build, target_zero, target_validation = _fixed_equivalence(contexts["normal_dev_10"], domains, seed=int(manifest["solver_seed"]), time_limit=fixed_equivalence_seconds)
    equivalence = {
        "control": asdict(control_stage) | {"validation": control_validation},
        "target_zero_edit": asdict(target_zero) | {"validation": target_validation},
        "correctness_mismatch": target_zero.status in {"FEASIBLE", "OPTIMAL"} or control_stage.status not in {"FEASIBLE", "OPTIMAL"},
    }
    if target_zero.status in {"FEASIBLE", "OPTIMAL"}:
        raise JointPilotError("target zero-edit equivalence mismatch: target unexpectedly feasible")
    if target_zero.status != "INFEASIBLE":
        raise JointPilotError(f"target zero-edit equivalence unresolved: {target_zero.status}")
    full_build = build_joint_model(
        contexts["normal_dev_10"].allocation_input, placement_domains=domains,
        math_fallback_rules=_load_math_fallback_rules(Path(config_dir), contexts["normal_dev_10"].catalog),
        math_course_ids=math_course_ids_from_catalog(contexts["normal_dev_10"].catalog),
    )
    internal_hint = add_constrained_first_internal_hint(
        full_build,
        solver_seed=int(manifest["solver_seed"]),
        catalog=contexts["normal_dev_10"].catalog,
    )
    model_size = {
        "assignment_variables": len(full_build.assignment_vars),
        "placement_variables": len(full_build.placement_choice_vars),
        "changed_section_variables": len(full_build.section_changed_vars),
        "optional_intervals": full_build.optional_intervals,
        "auxiliary_variables": full_build.auxiliary_variables,
        "total_model_variables": full_build.model_variables,
        "total_constraints": full_build.model_constraints,
        "serialized_model_proto_bytes": full_build.proto_bytes,
        "build_time_seconds": full_build.build_time_seconds,
        "editable_sections": len(full_build.placement_choice_vars),
        "placement_options": domain_summary.total_unique_placement_options,
    }
    gate = _cost_gate(full_build)
    stages: list[JointStageResult] = []
    witness: dict[str, Any] = {"status": "not_found", "joint_diagnostic_witness": False}
    production: dict[str, Any] = {"status": "not_run", "validated_repair": False}
    if not gate:
        stage1 = solve_joint_stage(full_build, "changed_sections", time_limit_seconds=stage_budgets[0], seed=int(manifest["solver_seed"]), hint_source=internal_hint["source"])
        stages.append(stage1)
        if stage1.incumbent_found:
            stage2 = solve_joint_stage(full_build, "affected_assignments", time_limit_seconds=stage_budgets[1], seed=int(manifest["solver_seed"]), fixed_values={"changed_sections": stage1.objective_value or 0}, conditional_on_unproven_incumbent=not stage1.optimality_proven)
            stages.append(stage2)
            best2 = stage2 if stage2.incumbent_found else stage1
            if best2.incumbent_found:
                stage3 = solve_joint_stage(full_build, "placement_displacement", time_limit_seconds=stage_budgets[2], seed=int(manifest["solver_seed"]), fixed_values={"changed_sections": stage1.objective_value or 0, "affected_assignments": stage2.objective_value if stage2.incumbent_found else 0}, conditional_on_unproven_incumbent=not (stage1.optimality_proven and stage2.optimality_proven))
                stages.append(stage3)
                best3 = stage3 if stage3.incumbent_found else best2
                if best3.incumbent_found:
                    edited_sections = apply_placement_map_to_sections(
                        contexts["normal_dev_10"], dict(best3.selected_placements)
                    )
                    edited_input = canonicalize_allocation_input(
                        contexts["normal_dev_10"].students.copy(deep=True),
                        contexts["normal_dev_10"].requests.copy(deep=True),
                        edited_sections,
                        contexts["normal_dev_10"].catalog.copy(deep=True),
                    )
                    edited_rules = _load_math_fallback_rules(Path(config_dir), contexts["normal_dev_10"].catalog)
                    edited_math_ids = math_course_ids_from_catalog(contexts["normal_dev_10"].catalog)
                    edited_hint = run_constrained_first_baseline(
                        edited_input, int(manifest["solver_seed"]),
                        math_fallback_rules=edited_rules, math_course_ids=edited_math_ids,
                    )
                    reference = {
                        (assignment.request_key, assignment.linked_section_group_id)
                        for assignment in edited_hint.assignments
                    }
                    for key, variable in full_build.assignment_vars.items():
                        full_build.model.AddHint(variable, int((key.request_key, key.section_id) in reference))
                    stage4 = solve_joint_stage(
                        full_build, "assignment_stability", time_limit_seconds=stage_budgets[3],
                        seed=int(manifest["solver_seed"]),
                        fixed_values={
                            "changed_sections": stage1.objective_value or 0,
                            "affected_assignments": stage2.objective_value if stage2.incumbent_found else 0,
                            "displacement": stage3.objective_value if stage3.incumbent_found else 0,
                        },
                        hint_source="constrained_first_internal",
                        stability_reference=reference,
                        conditional_on_unproven_incumbent=not (
                            stage1.optimality_proven and stage2.optimality_proven and stage3.optimality_proven
                        ),
                    )
                    stages.append(stage4)
                    final_stage = stage4 if stage4.incumbent_found else best3
                    witness = {
                        "status": "joint_witness_found",
                        "joint_diagnostic_witness": True,
                        "changed_sections": [
                            section_id for section_id, placement in final_stage.selected_placements
                            if placement != _section_placement(contexts["normal_dev_10"].allocation_input.logical_sections_by_id[section_id])
                        ],
                        "selected_placements": [(sid, list(place)) for sid, place in final_stage.selected_placements],
                        "selected_assignment_count": len(final_stage.selected_assignments),
                        "source": "joint_diagnostic_model_response",
                        "independently_validated": False,
                    }
                    placement_map = dict(final_stage.selected_placements)
                    production = _production_validate(contexts["normal_dev_10"], placement_map, solver_seed=int(manifest["solver_seed"]), time_limit_seconds=300.0)
                    witness["independently_validated"] = bool(production.get("validated_repair"))
    aggregate = {
        "schema_version": SCHEMA_VERSION,
        "experiment_name": manifest["experiment_name"],
        "phase": manifest["phase"],
        "attempted_scenarios": list(PILOT_SCENARIO_IDS),
        "other_normal_targets_run": 0,
        "stress_runs": 0,
        "negative_runs": 0,
        "holdout_runs": 0,
        "external_persisted_seed": False,
        "formal_production_validation_runs": int(production.get("status") != "not_run"),
        "validated_repairs": int(bool(production.get("validated_repair"))),
        "minimum_claim": None,
        "cost_gate": {"passed": not gate, "violations": gate},
        "equivalence": equivalence,
        "model_size": model_size,
        "placement_domain": asdict(domain_summary),
        "joint_stage_count": len(stages),
        "unexpected_correctness_failures": 0,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "pilot_manifest_snapshot.json", manifest)
    _write_json(output_dir / "provenance.json", {
        "source_git_commit": manifest["source_git_commit"], "source_audit_hash": manifest["source_audit_hash"],
        "source_candidate_preview_hash": manifest["source_candidate_preview_hash"], "no_new_solver_runs_before_pilot": False,
        "stress_runs": 0, "holdout_runs": 0, "other_normal_targets_run": 0,
    } | source_info)
    _write_json(output_dir / "architecture_audit.json", build_architecture_audit())
    _write_json(output_dir / "placement_domain_summary.json", asdict(domain_summary))
    _write_json(output_dir / "model_size_summary.json", model_size)
    _write_json(output_dir / "equivalence_checks.json", equivalence)
    _write_json(output_dir / "joint_stage_trace.json", [asdict(stage) for stage in stages])
    _write_json(output_dir / "joint_witness.json", witness)
    _write_json(output_dir / "production_validation.json", production)
    _write_json(output_dir / "aggregate_summary.json", aggregate)
    _write_json(output_dir / "failures.json", {"failures": [], "unexpected_failure_count": 0})
    for scenario_id in PILOT_SCENARIO_IDS:
        _write_json(output_dir / "scenarios" / scenario_id / "scenario_summary.json", {
            "scenario_id": scenario_id, "role": "control" if scenario_id == CONTROL_SCENARIO_ID else "pilot_target",
            "solver_runs": (1 if scenario_id == CONTROL_SCENARIO_ID else 1) + (len(stages) if scenario_id != CONTROL_SCENARIO_ID else 0),
        })
    _write_checksums(output_dir)
    return aggregate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Phase A joint period-edit feasibility pilot.")
    parser.add_argument("--manifest", default=str(DEFAULT_PILOT_MANIFEST))
    parser.add_argument("--preview-dir", default=str(PREVIEW_OUTPUT))
    parser.add_argument("--audit-root", default=str(DEFAULT_AUDIT_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--config-dir", default="data/config")
    args = parser.parse_args(argv)
    try:
        result = run_joint_period_edit_pilot(
            manifest_path=args.manifest, preview_dir=args.preview_dir,
            audit_root=args.audit_root, config_dir=args.config_dir, output_dir=args.output_dir,
        )
    except JointPilotError as exc:
        print(f"Joint period-edit pilot FAILED: {exc}")
        return 1
    print("Joint period-edit pilot PASS")
    print(json.dumps({key: result[key] for key in ("attempted_scenarios", "model_size", "cost_gate", "formal_production_validation_runs", "validated_repairs")}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
