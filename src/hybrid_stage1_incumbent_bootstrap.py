"""Bounded incumbent searches for the frozen hybrid period-edit domain.

This module is a development-only orchestration layer.  It keeps the full
312-section/841-option model and adds only ``sum(section_changed) <= K``.
Portfolio hints and Hamming objectives guide search; they never restrict the
feasible region. The explicit ``sum(section_changed) <= K`` cap is the sole
deliberate feasibility restriction added to each bounded search model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from ortools.sat.python import cp_model

from src.allocation import canonicalize_allocation_input, math_course_ids_from_catalog, run_constrained_first_baseline
from src.allocation.cp_sat_solver import _VariableKey
from src.benchmark_runner import _load_math_fallback_rules
from src.final_schedule_policy import evaluate_final_schedule_policy
from src.joint_period_edit_pilot import (
    AUTHORITATIVE_STUDENT_ID,
    PlacementOption,
    _json_hash,
    _section_placement,
    apply_placement_map_to_sections,
    build_frozen_placement_domains,
    build_joint_model,
)
from src.joint_period_edit_stage1_pilot import (
    DEFAULT_CONTROL_AUDIT,
    DEFAULT_CONTROL_AUDITED,
    DEFAULT_JOINT_PILOT,
    Stage1Run,
    _selected_keys,
    frozen_domain_hashes,
    production_fixed_witness_acceptance,
    validate_joint_witness,
    verify_checksums,
    independent_production_validation,
)
from src.period_placement_repair_probe import (
    DEFAULT_AUDIT_ROOT,
    DEFAULT_OUTPUT as DEFAULT_PREVIEW_OUTPUT,
    CandidateEdit,
    _candidate_from_dict as preview_candidate_from_dict,
    apply_candidate_to_input,
    apply_candidate_to_sections,
    exact_student_level_analysis,
    load_scenario_context,
)
from src.section_plan_feasibility_audit import load_section_plan_audit_manifest


TARGET_SCENARIO_ID = "normal_dev_10"
SOLVER_SEED = 20260630
WORKERS = 1
K1_BUDGET_SECONDS = 180.0
K2_BUDGET_SECONDS = 180.0
FIXED_WITNESS_BUDGET_SECONDS = 30.0
PRODUCTION_BUDGET_SECONDS = 300.0
DEFAULT_MANIFEST = Path("data/scenarios/hybrid_stage1_incumbent_bootstrap_v1.json")
DEFAULT_OUTPUT = Path(
    "/Users/klu/Projects/fair-course-allocation-artifacts/robustness-v1/"
    "hybrid-stage1-incumbent-bootstrap-v1"
)
DEFAULT_PREVIOUS_STAGE1 = Path(
    "/Users/klu/Projects/fair-course-allocation-artifacts/robustness-v1/"
    "hybrid-joint-period-edit-stage1-execution-v1"
)
DEFAULT_SIZE_AUDIT = Path(
    "/Users/klu/Projects/fair-course-allocation-artifacts/robustness-v1/"
    "joint-stage1-model-size-reduction-audit-v1"
)


class BootstrapError(ValueError):
    """Raised when bootstrap provenance or a frozen invariant fails closed."""


@dataclass(frozen=True)
class SearchResult:
    run_id: str
    k: int
    hint_id: str
    status: str
    assignment_available: bool
    incumbent_found: bool
    solution_count: int
    first_solution_time_seconds: float | None
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


@dataclass(frozen=True)
class _PortfolioCandidate:
    candidate: CandidateEdit
    core_primary_unmet: int
    core_schedule_gap: int
    affected_student_count: int
    changed_candidate_period_relationships: int
    absolute_period_displacement: int
    source_classification: str


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BootstrapError(f"cannot read JSON: {path}: {exc}") from exc


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
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
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(values)
    temporary.replace(path)


def write_checksums(root: Path) -> str:
    checksum = root / "SHA256SUMS.txt"
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != checksum:
            lines.append(f"{_sha256(path)}  {path.relative_to(root)}")
    checksum.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return _sha256(checksum)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_bootstrap_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    payload = _read_json(Path(path))
    required = {
        "experiment_name", "experiment_version", "phase", "source_git_commit",
        "target_scenario_id", "authoritative_student_id", "excluded_student_ids",
        "frozen_placement_domain_hash", "editable_section_count", "placement_option_count",
        "candidate_edge_count", "solver_seed", "workers", "external_persisted_seed",
        "k1_hint_portfolio_size_max", "k2_hint_portfolio_size_max", "k1_run_budget_seconds",
        "k2_run_budget_seconds", "production_fixed_witness_budget_seconds",
        "production_validation_budget_seconds", "stop_after_first_complete_solution",
        "stage2_allowed", "stage3_allowed", "stage4_allowed", "control_runs_allowed",
        "other_normal_targets_allowed", "stress_execution_allowed",
        "negative_execution_allowed", "holdout_execution_allowed",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise BootstrapError("bootstrap manifest missing: " + ", ".join(missing))
    if payload["phase"] != "single_target_incumbent_bootstrap":
        raise BootstrapError("unexpected bootstrap phase")
    if payload["target_scenario_id"] != TARGET_SCENARIO_ID:
        raise BootstrapError("only normal_dev_10 is allowed")
    if payload["authoritative_student_id"] != AUTHORITATIVE_STUDENT_ID:
        raise BootstrapError("authoritative student must be G12_0536")
    if "G12_0105" not in payload["excluded_student_ids"]:
        raise BootstrapError("G12_0105 must remain excluded")
    if int(payload["solver_seed"]) != SOLVER_SEED or int(payload["workers"]) != WORKERS:
        raise BootstrapError("solver seed or workers drifted")
    if payload["external_persisted_seed"] is not False:
        raise BootstrapError("external persisted seed is forbidden")
    for field in (
        "stage2_allowed", "stage3_allowed", "stage4_allowed", "control_runs_allowed",
        "other_normal_targets_allowed", "stress_execution_allowed",
        "negative_execution_allowed", "holdout_execution_allowed",
    ):
        if payload[field] is not False:
            raise BootstrapError(f"{field} must be false")
    if payload["stop_after_first_complete_solution"] is not True:
        raise BootstrapError("bootstrap must stop after the first complete solution")
    if int(payload["k1_hint_portfolio_size_max"]) != 3 or int(payload["k2_hint_portfolio_size_max"]) != 2:
        raise BootstrapError("portfolio sizes are not frozen")
    if float(payload["k1_run_budget_seconds"]) != K1_BUDGET_SECONDS or float(payload["k2_run_budget_seconds"]) != K2_BUDGET_SECONDS:
        raise BootstrapError("bootstrap budgets are not frozen")
    return payload


def _number(text: str) -> int:
    return int(text.replace("'", ""))


def audit_previous_stage1_log(
    log_path: str | Path,
    response_stats_path: str | Path,
    hint_audit_path: str | Path,
    solver_config_path: str | Path,
) -> dict[str, Any]:
    """Separate structured response facts from observations parsed from logs."""
    log = Path(log_path).read_text(encoding="utf-8")
    stats = _read_json(Path(response_stats_path))
    hint = _read_json(Path(hint_audit_path))
    config = _read_json(Path(solver_config_path))
    initial = re.search(r"Initial optimization model.*?\n#Variables:\s*([\d']+)", log, re.S)
    presolved = re.search(r"Presolved optimization model.*?\n#Variables:\s*([\d']+)", log, re.S)
    presolved_block = ""
    if presolved:
        presolved_block = log[presolved.end():]
        presolved_block = presolved_block.split("[Symmetry]", 1)[0]
    constraint_counts = {
        match.group(1): _number(match.group(2))
        for match in re.finditer(r"#k([A-Za-z0-9]+):\s*([\d']+)", presolved_block)
    }
    hint_incomplete = bool(re.search(r"solution hint is incomplete", log, re.I))
    hint_infeasible = bool(re.search(r"(?:hint|solution hint).*infeasible", log, re.I))
    search_started = bool(re.search(r"Starting search at", log))
    solution_repository = re.search(r"'best_solutions':\s*([\d']+)", log)
    first_solution_signal = bool(re.search(r"#\d+\s+[^\n]*best:\s*(?!inf)", log))
    objective_bound = re.search(r"next:\[[^,]+,\s*([\d']+)\]", log)
    structured = {
        "status": stats.get("status"),
        "incumbent_found": bool(stats.get("incumbent_found")),
        "assignment_available": bool(stats.get("assignment_available")),
        "solution_count": stats.get("solution_count"),
        "objective_value": stats.get("objective_value"),
        "best_bound": stats.get("best_bound"),
        "wall_time_seconds": stats.get("wall_time_seconds"),
        "deterministic_time_seconds": stats.get("deterministic_time_seconds"),
        "conflicts": stats.get("conflicts"),
        "branches": stats.get("branches"),
        "propagations": stats.get("propagations"),
        "integer_propagations": stats.get("integer_propagations"),
        "restarts": stats.get("restarts"),
        "response_hash": stats.get("response_hash"),
    }
    return {
        "source_files": {"log": str(log_path), "response_stats": str(response_stats_path), "hint_audit": str(hint_audit_path), "solver_config": str(solver_config_path)},
        "structured_response": structured,
        "hint_audit": hint,
        "solver_config": config,
        "log_evidence": {
            "hint_complete": not hint_incomplete,
            "hint_incomplete_message_seen": hint_incomplete,
            "hint_infeasible_message_seen": hint_infeasible,
            "hint_repair_or_violation_messages": [line for line in log.splitlines() if re.search(r"hint|repair|violation", line, re.I) and re.search(r"infeasible|violation|repair", line, re.I)],
            "initial_variable_count": _number(initial.group(1)) if initial else None,
            "presolved_variable_count": _number(presolved.group(1)) if presolved else None,
            "presolved_constraint_counts": constraint_counts,
            "presolved_constraint_count": sum(constraint_counts.values()) if constraint_counts else None,
            "symmetry_graph_skipped": "Graph too large. Skipping" in log,
            "search_started": search_started,
            "best_solutions_added": _number(solution_repository.group(1)) if solution_repository else None,
            "first_solution_signal": first_solution_signal,
            "objective_upper_bound_seen": _number(objective_bound.group(1)) if objective_bound else None,
            "evidence_source": "parsed_solver_log",
        },
        "inference": {
            "hint_infeasibility_proven": False,
            "first_partial_solution_observed": first_solution_signal,
            "interpretation": "search started but the structured response and log show no incumbent",
        },
    }


def _period_number(period: tuple[str, ...]) -> int:
    return int(period[0][1:])


def _candidate_metric(row: Mapping[str, Any], key: str, fallback: int) -> int:
    value = row.get(key)
    if value is None:
        return fallback
    return int(value)


def _portfolio_sort_key(item: _PortfolioCandidate) -> tuple[Any, ...]:
    return (
        item.core_primary_unmet,
        item.core_schedule_gap,
        item.affected_student_count,
        item.changed_candidate_period_relationships,
        item.absolute_period_displacement,
        item.candidate.candidate_id,
    )


def candidate_sort_key(candidate: CandidateEdit, analysis: Mapping[str, Any]) -> tuple[Any, ...]:
    return _portfolio_sort_key(_portfolio_item(candidate, analysis))


def _portfolio_item(candidate: CandidateEdit, analysis: Mapping[str, Any]) -> _PortfolioCandidate:
    displacement = sum(
        abs(_period_number(old) - _period_number(new))
        for old, new in zip(candidate.original_placements, candidate.proposed_placements)
    )
    return _PortfolioCandidate(
        candidate=candidate,
        core_primary_unmet=int(analysis.get("edited_primary_unmet", 999999)),
        core_schedule_gap=int(analysis.get("edited_schedule_gap", 999999)),
        affected_student_count=int(analysis.get("affected_student_count", candidate.affected_student_count)),
        changed_candidate_period_relationships=_candidate_metric(analysis, "changed_candidate_period_relationships", candidate.affected_candidate_edge_count),
        absolute_period_displacement=displacement,
        source_classification=str(analysis.get("classification", "unknown")),
    )


def _transition_key(candidate: CandidateEdit) -> tuple[Any, ...]:
    return tuple(sorted(
        (course, tuple(old), tuple(new))
        for course, old, new in zip(candidate.logical_course_ids, candidate.original_placements, candidate.proposed_placements)
    ))


def select_single_edit_portfolio(
    candidates: Iterable[CandidateEdit],
    analyses: Mapping[str, Mapping[str, Any]],
    *,
    max_size: int = 3,
) -> tuple[_PortfolioCandidate, ...]:
    items = []
    for candidate in candidates:
        analysis = analyses.get(candidate.candidate_id, {})
        if candidate.edit_type != "single_section_move":
            continue
        if analysis.get("classification") != "student_level_promising":
            continue
        if not all(analysis.get(field) is True for field in ("ordinary_primary_unmet_at_most_one", "minimum_five_policy_reached", "maximum_gap_one_policy_reached")):
            continue
        if int(analysis.get("edited_primary_unmet", 999999)) > 1 or int(analysis.get("edited_schedule_gap", 999999)) > 1:
            continue
        items.append(_portfolio_item(candidate, analysis))
    items.sort(key=_portfolio_sort_key)
    selected: list[_PortfolioCandidate] = []
    seen_destinations: set[tuple[str, tuple[str, ...]]] = set()
    seen_transitions: set[tuple[Any, ...]] = set()
    for item in items:
        destination = (item.candidate.logical_section_ids[0], item.candidate.proposed_placements[0])
        transition = _transition_key(item.candidate)
        if destination in seen_destinations or transition in seen_transitions:
            continue
        seen_destinations.add(destination)
        seen_transitions.add(transition)
        selected.append(item)
        if len(selected) >= max_size:
            break
    return tuple(selected)


def build_pair_hint_portfolio(
    single_candidates: Iterable[_PortfolioCandidate],
    allocation_input: Any,
    analyses: Mapping[str, Mapping[str, Any]],
    *,
    max_size: int = 2,
    source_limit: int = 20,
) -> tuple[_PortfolioCandidate, ...]:
    source = list(single_candidates)[:source_limit]
    # The source list is intentionally the frozen top-20 hint source, not a
    # domain filter.  The model below still receives all 312/841 options.
    pairs: list[_PortfolioCandidate] = []
    for index, left in enumerate(source):
        for right in source[index + 1:]:
            first, second = left.candidate, right.candidate
            if set(first.logical_section_ids) & set(second.logical_section_ids):
                continue
            candidate = CandidateEdit(
                candidate_id=f"bootstrap_pair:{first.candidate_id}__{second.candidate_id}",
                edit_type="bootstrap_pair",
                logical_section_ids=first.logical_section_ids + second.logical_section_ids,
                logical_course_ids=first.logical_course_ids + second.logical_course_ids,
                original_placements=first.original_placements + second.original_placements,
                proposed_placements=first.proposed_placements + second.proposed_placements,
                valid_period_source="frozen single-section move composition",
                occupancy_shape=first.occupancy_shape + second.occupancy_shape,
                core_student=first.core_student,
                core_period_relevance=tuple(sorted(set(first.core_period_relevance + right.candidate.core_period_relevance))),
                affected_candidate_edge_count=first.affected_candidate_edge_count + second.affected_candidate_edge_count,
                affected_student_count=first.affected_student_count + second.affected_student_count,
            )
            edited = apply_candidate_to_input(allocation_input, candidate)
            exact = exact_student_level_analysis(edited, candidate.core_student)
            primary_count = exact["primary_request_count"]
            analysis = {
                "classification": "student_level_promising" if (
                    exact["original_primary_unmet"] <= 1
                    and exact["original_max_primary_assignments"] >= min(5, primary_count)
                    and exact["original_max_schedule_gap"] <= 1
                ) else "student_level_no_effect",
                "edited_primary_unmet": exact["original_primary_unmet"],
                "edited_schedule_gap": exact["original_max_schedule_gap"],
                "changed_candidate_period_relationships": candidate.affected_candidate_edge_count,
            }
            if analysis["classification"] != "student_level_promising":
                continue
            item = _portfolio_item(candidate, analysis)
            pairs.append(item)
    pairs.sort(key=_portfolio_sort_key)
    selected: list[_PortfolioCandidate] = []
    seen_transitions: set[tuple[Any, ...]] = set()
    for item in pairs:
        transition = _transition_key(item.candidate)
        if transition in seen_transitions:
            continue
        seen_transitions.add(transition)
        selected.append(item)
        if len(selected) >= max_size:
            break
    return tuple(selected)


def candidate_payload(item: _PortfolioCandidate) -> dict[str, Any]:
    return asdict(item.candidate) | {
        "core_primary_unmet": item.core_primary_unmet,
        "core_schedule_gap": item.core_schedule_gap,
        "changed_candidate_period_relationships": item.changed_candidate_period_relationships,
        "absolute_period_displacement": item.absolute_period_displacement,
        "source_classification": item.source_classification,
        "portfolio_sort_key": list(_portfolio_sort_key(item)),
    }


def _assignment_hint_quality(context: Any, candidate: CandidateEdit, config_dir: Path, seed: int) -> tuple[dict[str, Any], tuple[_VariableKey, ...]]:
    sections = apply_candidate_to_sections(context.sections, candidate)
    edited_input = canonicalize_allocation_input(
        context.students.copy(deep=True), context.requests.copy(deep=True), sections, context.catalog.copy(deep=True)
    )
    rules = _load_math_fallback_rules(config_dir, context.catalog)
    math_ids = math_course_ids_from_catalog(context.catalog)
    baseline = run_constrained_first_baseline(edited_input, seed, math_fallback_rules=rules, math_course_ids=math_ids)
    selected = tuple(sorted((_VariableKey(item.request_key, item.linked_section_group_id) for item in baseline.assignments), key=lambda x: (x.request_key, x.section_id)))
    invalid = [f"{key.request_key}:{key.section_id}" for key in selected if key.section_id not in edited_input.candidate_index.get(key.request_key, ())]
    policy = baseline.policy_report
    roster_violations = [item.linked_section_group_id for item in baseline.section_roster_summary if item.assigned_count > item.capacity]
    consistency_codes = [issue.code for issue in baseline.consistency_issues]
    quality = {
        "candidate_id": candidate.candidate_id,
        "changed_logical_section_count": len(candidate.logical_section_ids),
        "constrained_first_assignment_count": len(selected),
        "primary_assigned": sum(item.primary_assigned_count for item in baseline.student_outcomes),
        "primary_unmet": sum(item.primary_unmet_count for item in baseline.student_outcomes),
        "logical_assigned": sum(int(item.assigned_logical_course_count or 0) for item in baseline.student_outcomes),
        "logical_full": sum(bool(item.logical_fully_scheduled) for item in baseline.student_outcomes),
        "ordinary_policy_violations": len(policy.ordinary_violation_student_ids),
        "protected_policy_violations": len(policy.protected_violation_student_ids),
        "high_demand_violations": policy.high_demand_violation_count,
        "period_conflict_count": sum("PERIOD" in code.upper() for code in consistency_codes),
        "capacity_violation_count": len(roster_violations),
        "duplicate_identity_count": sum("DUPLICATE" in code.upper() for code in consistency_codes),
        "policy_pass": policy.all_reported_policies_satisfied and not baseline.consistency_issues,
        "consistency_issue_count": len(baseline.consistency_issues),
        "invalid_assignment_keys": invalid,
        "hint_positive_key_count": len(selected),
        "hint_positive_key_hash": _json_hash([[key.request_key, key.section_id] for key in selected]),
        "hint_coverage": 1.0 if selected else 0.0,
        "assignment_algorithm": baseline.algorithm_name,
    }
    return quality, selected


def apply_bootstrap_hints(
    build: Any,
    candidate: CandidateEdit,
    assignment_hint: Iterable[_VariableKey],
) -> dict[str, Any]:
    from src.joint_model_control_performance_audit import assert_empty_solution_hint, validate_solution_hint_uniqueness

    assert_empty_solution_hint(build.model)
    selected = set(assignment_hint)
    unknown = sorted(selected - set(build.assignment_vars), key=lambda key: (key.request_key, key.section_id))
    if unknown:
        raise BootstrapError(f"assignment hint contains unknown keys: {unknown[:3]}")
    target_placements = {sid: placement for sid, placement in zip(candidate.logical_section_ids, candidate.proposed_placements)}
    hint_indices: list[int] = []
    hint_values: list[int] = []
    for key, variable in sorted(build.assignment_vars.items(), key=lambda item: (item[0].request_key, item[0].section_id)):
        value = int(key in selected)
        build.model.AddHint(variable, value)
        hint_indices.append(variable.Index())
        hint_values.append(value)
    placement_positive = 0
    for section in build.allocation_input.logical_sections:
        sid = section.linked_section_group_id
        target = target_placements.get(sid, _section_placement(section))
        for option in build.placement_domains[sid]:
            variable = build.placement_choice_vars.get((sid, option.placement))
            if variable is None:
                continue
            value = int(option.placement == target)
            placement_positive += value
            build.model.AddHint(variable, value)
            hint_indices.append(variable.Index())
            hint_values.append(value)
    if len(hint_indices) != len(set(hint_indices)):
        raise BootstrapError("duplicate variable ownership in bootstrap hint")
    state = validate_solution_hint_uniqueness(build.model)
    if state["duplicate_variables"] or state["conflicting_variables"]:
        raise BootstrapError("duplicate/conflicting bootstrap hint")
    return {
        "hint_source": "edited_plan_constrained_first",
        "candidate_id": candidate.candidate_id,
        "assignment_variables": len(build.assignment_vars),
        "assignment_positive": len(selected),
        "assignment_negative": len(build.assignment_vars) - len(selected),
        "assignment_coverage": 1.0 if build.assignment_vars else 0.0,
        "placement_variables": len(build.placement_choice_vars),
        "placement_positive": placement_positive,
        "placement_negative": len(build.placement_choice_vars) - placement_positive,
        "placement_coverage": 1.0 if build.placement_choice_vars else 0.0,
        "positive_assignment_key_hash": _json_hash([[key.request_key, key.section_id] for key in sorted(selected, key=lambda x: (x.request_key, x.section_id))]),
        "duplicate_variables": [],
        "conflicting_variables": [],
        "fresh_model_verified": True,
        "external_persisted_seed": False,
        "repair_hint_enabled": False,
    }


def build_bootstrap_model(allocation_input: Any, domains: Mapping[str, tuple[PlacementOption, ...]], candidate: CandidateEdit, assignment_hint: Iterable[_VariableKey], *, k: int, math_fallback_rules: tuple[Any, ...], math_course_ids: tuple[str, ...]) -> tuple[Any, dict[str, Any]]:
    build = build_joint_model(
        allocation_input,
        placement_domains=domains,
        math_fallback_rules=math_fallback_rules,
        math_course_ids=math_course_ids,
        occupancy_mode="hybrid_sparse_linear_occupancy",
    )
    if k not in {1, 2}:
        raise BootstrapError("bootstrap cap must be 1 or 2")
    add_change_cap(build, k)
    build.model.Minimize(hamming_expression(build, assignment_hint))
    hint = apply_bootstrap_hints(build, candidate, assignment_hint)
    hint["cardinality_cap"] = k
    hint["hamming_objective"] = "unweighted_assignment_distance_to_edited_plan_constrained_first"
    hint["full_domain_preserved"] = True
    hint["candidate_pruning"] = False
    return build, hint


def add_change_cap(build: Any, k: int) -> None:
    """Add the sole bootstrap feasibility restriction to a fresh model."""
    if k not in {1, 2}:
        raise BootstrapError("bootstrap cap must be 1 or 2")
    build.model.Add(sum(build.section_changed_vars.values()) <= k)


def hamming_expression(build: Any, assignment_hint: Iterable[_VariableKey]) -> Any:
    selected = set(assignment_hint)
    return sum((1 - variable) if key in selected else variable for key, variable in build.assignment_vars.items())


class _FirstSolutionCallback(cp_model.CpSolverSolutionCallback):
    def __init__(self, build: Any) -> None:
        super().__init__()
        self.build = build
        self.count = 0
        self.first_time: float | None = None
        self.first_objective: int | None = None
        self.assignments: tuple[tuple[str, str], ...] = ()
        self.placements: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def on_solution_callback(self) -> None:
        self.count += 1
        if self.count != 1:
            return
        self.first_time = float(self.WallTime())
        self.first_objective = int(round(self.ObjectiveValue()))
        self.assignments = tuple(sorted((key.request_key, key.section_id) for key, variable in self.build.assignment_vars.items() if self.BooleanValue(variable)))
        self.placements = tuple((section.linked_section_group_id, next(option.placement for option in self.build.placement_domains[section.linked_section_group_id] if len(self.build.placement_domains[section.linked_section_group_id]) == 1 or self.BooleanValue(self.build.placement_choice_vars[(section.linked_section_group_id, option.placement)]))) for section in self.build.allocation_input.logical_sections)


def solve_bootstrap(build: Any, *, run_id: str, k: int, hint_id: str, seed: int = SOLVER_SEED, time_limit_seconds: float = 180.0) -> SearchResult:
    started = time.perf_counter()
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_seconds)
    solver.parameters.num_search_workers = WORKERS
    solver.parameters.random_seed = int(seed)
    solver.parameters.stop_after_first_solution = True
    solver.parameters.log_search_progress = True
    logs: list[str] = []
    solver.log_callback = logs.append
    callback = _FirstSolutionCallback(build)
    status_code = solver.Solve(build.model, callback)
    status = solver.StatusName(status_code)
    response = solver.ResponseProto()
    available = status in {"FEASIBLE", "OPTIMAL"} or callback.count > 0
    if status in {"FEASIBLE", "OPTIMAL"}:
        assignments = tuple(sorted((key.request_key, key.section_id) for key, variable in build.assignment_vars.items() if solver.BooleanValue(variable)))
        placements = tuple((section.linked_section_group_id, next(option.placement for option in build.placement_domains[section.linked_section_group_id] if len(build.placement_domains[section.linked_section_group_id]) == 1 or solver.BooleanValue(build.placement_choice_vars[(section.linked_section_group_id, option.placement)]))) for section in build.allocation_input.logical_sections)
        objective = int(round(solver.ObjectiveValue()))
    else:
        assignments, placements, objective = callback.assignments, callback.placements, callback.first_objective
    best_bound = int(round(solver.BestObjectiveBound())) if available else None
    deterministic = getattr(response, "deterministic_time", None)
    return SearchResult(
        run_id=run_id, k=k, hint_id=hint_id, status=status, assignment_available=available,
        incumbent_found=available, solution_count=callback.count,
        first_solution_time_seconds=callback.first_time, objective_value=objective,
        best_bound=best_bound, optimality_proven=status == "OPTIMAL",
        wall_time_seconds=float(solver.WallTime()), end_to_end_runtime_seconds=time.perf_counter() - started,
        deterministic_time_seconds=float(deterministic) if deterministic is not None else None,
        conflicts=int(getattr(response, "num_conflicts", solver.NumConflicts())),
        branches=int(getattr(response, "num_branches", solver.NumBranches())),
        propagations=int(getattr(response, "num_binary_propagations", 0)),
        integer_propagations=int(getattr(response, "num_integer_propagations", 0)),
        restarts=int(getattr(response, "num_restarts", 0)),
        response_hash=hashlib.sha256(str(response).encode("utf-8")).hexdigest(),
        selected_assignments=assignments, selected_placements=placements, solver_log=tuple(logs),
    )


def _stage1_run_from_search(result: SearchResult) -> Stage1Run:
    return Stage1Run(
        status=result.status, assignment_available=result.assignment_available, incumbent_found=result.incumbent_found,
        solution_count=result.solution_count, first_incumbent_time_seconds=result.first_solution_time_seconds,
        first_incumbent_objective=result.objective_value, objective_value=result.objective_value,
        best_bound=result.best_bound, optimality_proven=result.optimality_proven,
        wall_time_seconds=result.wall_time_seconds, end_to_end_runtime_seconds=result.end_to_end_runtime_seconds,
        deterministic_time_seconds=result.deterministic_time_seconds, conflicts=result.conflicts, branches=result.branches,
        propagations=result.propagations, integer_propagations=result.integer_propagations, restarts=result.restarts,
        response_hash=result.response_hash, selected_assignments=result.selected_assignments,
        selected_placements=result.selected_placements, solver_log=result.solver_log,
    )


def validate_bootstrap_witness(context: Any, build: Any, result: SearchResult, *, config_dir: Path, k: int) -> dict[str, Any]:
    if not result.assignment_available:
        return {"joint_bootstrap_witness_valid": False, "status": "not_run", "not_run_reason": "no_incumbent"}
    placement_map = dict(result.selected_placements)
    edited_sections = apply_placement_map_to_sections(context, placement_map)
    edited_input = canonicalize_allocation_input(context.students.copy(deep=True), context.requests.copy(deep=True), edited_sections, context.catalog.copy(deep=True))
    witness = validate_joint_witness(build, _stage1_run_from_search(result), edited_input)
    witness["joint_bootstrap_witness_valid"] = bool(witness.get("joint_stage1_witness_valid")) and int(witness.get("changed_logical_section_count", 999)) <= k
    witness["cap_bound"] = k
    witness["response_hash"] = result.response_hash
    return witness


def _source_artifacts(manifest: Mapping[str, Any], previous: Path, size_audit: Path, preview: Path, audit_root: Path) -> dict[str, Any]:
    roots = {
        "previous_stage1": previous,
        "hybrid_size_audit": size_audit,
        "candidate_preview": preview,
        "section_audit": audit_root,
        "section_audited": audit_root.with_name(audit_root.name + "-audited"),
        "control_audit": DEFAULT_CONTROL_AUDIT,
        "control_audited": DEFAULT_CONTROL_AUDITED,
    }
    expected = {
        "previous_stage1": manifest["source_previous_stage1_hash"], "hybrid_size_audit": manifest["source_hybrid_audit_hash"],
        "candidate_preview": manifest["source_candidate_preview_hash"], "section_audit": manifest["source_section_audit_hash"],
        "section_audited": manifest["source_section_audited_hash"], "control_audit": manifest["source_control_audit_hash"],
        "control_audited": manifest["source_control_audited_hash"],
    }
    result = {}
    for name, root in roots.items():
        check = verify_checksums(root)
        if not check["passed"] or check["sha256"] != expected[name]:
            raise BootstrapError(f"source artifact verification failed: {name}")
        result[name] = check
    return result


def _load_portfolios(context: Any, preview_dir: Path, domains: Mapping[str, tuple[PlacementOption, ...]]) -> tuple[tuple[_PortfolioCandidate, ...], tuple[_PortfolioCandidate, ...]]:
    root = preview_dir / "scenarios" / TARGET_SCENARIO_ID
    candidate_rows = _read_json(root / "candidate_universe.json")
    analysis_rows = _read_json(root / "static_student_analysis.json")["candidates"]
    analyses = {str(row["candidate_id"]): row for row in analysis_rows}
    candidates = [preview_candidate_from_dict(row) for row in candidate_rows if str(row.get("core_student")) == AUTHORITATIVE_STUDENT_ID]
    single_source = select_single_edit_portfolio(
        candidates,
        analyses,
        max_size=20,
    )
    if not single_source:
        raise BootstrapError("no exact promising single-section candidates")
    # A portfolio selection is capped after sorting; keep the top-20 source
    # available for deterministic pair construction.
    pair_portfolio = build_pair_hint_portfolio(single_source, context.allocation_input, analyses, max_size=2, source_limit=20)
    return tuple(single_source[:3]), pair_portfolio


def _write_portfolio(path: Path, items: Iterable[_PortfolioCandidate]) -> None:
    _write_json(path, {"count": len(tuple(items)), "candidates": [candidate_payload(item) for item in items]})


def _claim(k1: list[dict[str, Any]], k2: list[dict[str, Any]], accepted: bool, validated: bool) -> dict[str, Any]:
    if validated and accepted:
        valid_k = 1 if any(row["k"] == 1 and row.get("witness_valid") for row in k1) else 2
        if valid_k == 1:
            return {"claim": "minimum_changed_sections_within_frozen_placement_domain", "value": 1, "proven": True}
        if any(row["status"] == "INFEASIBLE" for row in k1):
            return {"claim": "minimum_changed_sections_within_frozen_placement_domain", "value": 2, "proven": True}
        return {"claim": "validated_repair_with_at_most_2_changes", "proven": False, "minimum_status": "unresolved"}
    executed = k1 + k2
    if any(row.get("status") == "MODEL_INVALID" for row in executed):
        return {"claim": "correctness_failure", "proven": False}
    if k2 and all(row.get("status") == "INFEASIBLE" for row in k2) and any(row.get("status") == "INFEASIBLE" for row in k1):
        return {"claim": "no_repair_within_cap_2_and_frozen_domain", "proven": True}
    return {"claim": "unresolved_no_incumbent", "proven": False}


def run_bootstrap(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    output_dir: str | Path = DEFAULT_OUTPUT,
    preview_dir: str | Path = DEFAULT_PREVIEW_OUTPUT,
    audit_root: str | Path = DEFAULT_AUDIT_ROOT,
    previous_stage1: str | Path = DEFAULT_PREVIOUS_STAGE1,
    size_audit: str | Path = DEFAULT_SIZE_AUDIT,
    config_dir: str | Path = "data/config",
    resume: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir)
    resume_checkpoint: dict[str, Any] | None = None
    if output.exists() and any(output.iterdir()):
        if not resume:
            raise BootstrapError(f"bootstrap output is non-empty; refusing overwrite: {output}")
        aggregate = output / "aggregate_summary.json"
        if aggregate.is_file():
            return _read_json(aggregate) | {"resumed": True, "search_reexecuted": False}
        checkpoint = output / "checkpoint.json"
        if not checkpoint.is_file():
            raise BootstrapError("resume requested without an atomic checkpoint")
        resume_checkpoint = _read_json(checkpoint)
        if resume_checkpoint.get("schema_version") != 1:
            raise BootstrapError("unsupported bootstrap checkpoint schema")
    manifest = load_bootstrap_manifest(manifest_path)
    source_info = _source_artifacts(manifest, Path(previous_stage1), Path(size_audit), Path(preview_dir), Path(audit_root))
    previous_audit = audit_previous_stage1_log(
        Path(previous_stage1) / "stage1_solver.log", Path(previous_stage1) / "stage1_response_stats.json",
        Path(previous_stage1) / "hint_audit.json", Path(previous_stage1) / "stage1_solver_config.json",
    )
    audit_manifest = load_section_plan_audit_manifest("data/scenarios/section_plan_feasibility_audit_v1.json")
    context = load_scenario_context(TARGET_SCENARIO_ID, audit_manifest=audit_manifest, audit_root=Path(audit_root), config_dir=config_dir)
    domains, domain_summary = build_frozen_placement_domains(context, preview_dir)
    hashes = frozen_domain_hashes(domains, domain_summary.source_candidate_ids, context.allocation_input)
    for key in ("frozen_placement_domain_hash", "editable_section_id_hash", "placement_option_hash", "section_domain_mapping_hash", "candidate_source_id_hash", "original_placement_hash"):
        if hashes[key] != manifest[key]:
            raise BootstrapError(f"frozen domain hash drift: {key}")
    if domain_summary.editable_logical_section_count != int(manifest["editable_section_count"]) or domain_summary.total_unique_placement_options != int(manifest["placement_option_count"]):
        raise BootstrapError("frozen domain count drift")
    rules = _load_math_fallback_rules(Path(config_dir), context.catalog)
    math_ids = math_course_ids_from_catalog(context.catalog)
    singles, pairs = _load_portfolios(context, Path(preview_dir), domains)
    all_portfolio = singles + pairs
    quality_rows: list[dict[str, Any]] = []
    quality_by_id: dict[str, tuple[dict[str, Any], tuple[_VariableKey, ...]]] = {}
    for item in all_portfolio:
        quality, assignment_hint = _assignment_hint_quality(context, item.candidate, Path(config_dir), SOLVER_SEED)
        quality_rows.append(quality)
        quality_by_id[item.candidate.candidate_id] = (quality, assignment_hint)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "bootstrap_manifest_snapshot.json", manifest)
    _write_json(output / "provenance.json", {"source_git_commit": manifest["source_git_commit"], "previous_stage1_reruns": 0, "control_runs": 0, "other_normal_target_runs": 0, "stage2_runs": 0, "stage3_runs": 0, "stage4_runs": 0, "stress_runs": 0, "negative_runs": 0, "holdout_runs": 0, "external_persisted_seed": False})
    _write_json(output / "source_artifact_verification.json", source_info)
    _write_json(output / "previous_stage1_log_audit.json", previous_audit)
    _write_json(output / "frozen_domain_verification.json", {"counts": asdict(domain_summary), "hashes": hashes, "candidate_pruning": False, "no_preview_external_placements": True, "authoritative_student_id": AUTHORITATIVE_STUDENT_ID, "excluded_student_ids": ["G12_0105"]})
    _write_portfolio(output / "single_hint_portfolio.json", singles)
    _write_portfolio(output / "pair_hint_portfolio.json", pairs)
    _write_csv(output / "edited_plan_hint_quality.csv", quality_rows)
    _write_json(output / "joint_witness.json", {"status": "not_run", "joint_bootstrap_witness_valid": False, "not_run_reason": "no_validated_incumbent"})
    _write_json(output / "production_fixed_witness_acceptance.json", {"status": "not_run", "production_fixed_witness_accepted": False, "not_run_reason": "no_validated_incumbent"})
    _write_json(output / "production_cold_start_validation.json", {"status": "not_run", "independently_validated_period_repair": False, "not_run_reason": "fixed_witness_acceptance_not_passed"})

    search_rows: list[dict[str, Any]] = []
    k1_rows: list[dict[str, Any]] = []
    k2_rows: list[dict[str, Any]] = []
    witness = {"status": "not_run", "joint_bootstrap_witness_valid": False, "not_run_reason": "no_validated_incumbent"}
    acceptance = {"status": "not_run", "production_fixed_witness_accepted": False, "not_run_reason": "no_validated_incumbent"}
    production = {"status": "not_run", "independently_validated_period_repair": False, "not_run_reason": "fixed_witness_acceptance_not_passed"}
    failures: list[str] = []
    accepted = False
    validated = False
    stop_reason = "portfolio_exhausted"

    if resume_checkpoint is not None:
        k1_rows.extend(resume_checkpoint.get("k1_rows", []))
        k2_rows.extend(resume_checkpoint.get("k2_rows", []))
        search_rows.extend(k1_rows)
        search_rows.extend(k2_rows)
        witness = dict(resume_checkpoint.get("witness", witness))
        acceptance = dict(resume_checkpoint.get("acceptance", acceptance))
        production = dict(resume_checkpoint.get("production", production))
        failures.extend(str(value) for value in resume_checkpoint.get("failures", []))
        accepted = bool(resume_checkpoint.get("accepted", False))
        validated = bool(resume_checkpoint.get("validated", False))
        stop_reason = str(resume_checkpoint.get("stop_reason", stop_reason))
        if search_rows != list(resume_checkpoint.get("search_rows", search_rows)):
            raise BootstrapError("bootstrap checkpoint search rows are inconsistent")

    def checkpoint_state() -> None:
        search_rows[:] = list(k1_rows) + list(k2_rows)
        _write_csv(output / "search_runs.csv", search_rows)
        _write_json(output / "checkpoint.json", {
            "schema_version": 1,
            "stage": "search",
            "search_rows": search_rows,
            "k1_rows": k1_rows,
            "k2_rows": k2_rows,
            "witness": witness,
            "acceptance": acceptance,
            "production": production,
            "failures": failures,
            "accepted": accepted,
            "validated": validated,
            "stop_reason": stop_reason,
            "search_reexecuted": False,
        })

    checkpoint_state()

    def run_portfolio(items: Iterable[_PortfolioCandidate], k: int, rows: list[dict[str, Any]]) -> bool:
        nonlocal witness, acceptance, production, accepted, validated, stop_reason
        completed_run_ids = {str(row.get("run_id")) for row in search_rows}
        if accepted:
            return True
        for index, item in enumerate(items, start=1):
            run_id = f"k{k}_{index:02d}"
            if run_id in completed_run_ids:
                continue
            quality, assignment_hint = quality_by_id[item.candidate.candidate_id]
            build, hint = build_bootstrap_model(context.allocation_input, domains, item.candidate, assignment_hint, k=k, math_fallback_rules=rules, math_course_ids=math_ids)
            result = solve_bootstrap(build, run_id=run_id, k=k, hint_id=item.candidate.candidate_id, seed=SOLVER_SEED, time_limit_seconds=K1_BUDGET_SECONDS if k == 1 else K2_BUDGET_SECONDS)
            run_dir = output / "runs" / run_id
            _write_json(run_dir / "solver_config.json", {"seed": SOLVER_SEED, "workers": WORKERS, "max_time_in_seconds": K1_BUDGET_SECONDS if k == 1 else K2_BUDGET_SECONDS, "cardinality_cap": k, "objective": "hamming_to_edited_plan_constrained_first", "stop_after_first_complete_solution": True, "external_persisted_seed": False})
            _write_json(run_dir / "hint_audit.json", hint | {"quality": quality})
            _write_json(run_dir / "response_stats.json", {key: value for key, value in asdict(result).items() if key != "solver_log"})
            (run_dir / "solver.log").write_text("\n".join(result.solver_log), encoding="utf-8")
            row = {"run_id": run_id, "k": k, "hint_id": item.candidate.candidate_id, "status": result.status, "incumbent_found": result.incumbent_found, "objective_value": result.objective_value, "best_bound": result.best_bound, "first_solution_time_seconds": result.first_solution_time_seconds, "runtime_seconds": result.end_to_end_runtime_seconds, "response_hash": result.response_hash, "witness_valid": False}
            if result.assignment_available:
                witness = validate_bootstrap_witness(context, build, result, config_dir=Path(config_dir), k=k)
                row["witness_valid"] = bool(witness.get("joint_bootstrap_witness_valid"))
                if row["witness_valid"]:
                    accepted = True
                    placement_map = dict(result.selected_placements)
                    acceptance = production_fixed_witness_acceptance(context, placement_map, result.selected_assignments, config_dir=Path(config_dir), seed=SOLVER_SEED, time_limit_seconds=FIXED_WITNESS_BUDGET_SECONDS)
                    acceptance["production_fixed_witness_accepted"] = bool(acceptance.get("status") in {"FEASIBLE", "OPTIMAL"} and acceptance.get("assignment_exact") and acceptance.get("policy_pass") and acceptance.get("consistency_issue_count") == 0 and acceptance.get("response_hash"))
                    if acceptance["production_fixed_witness_accepted"]:
                        production = independent_production_validation(context, placement_map, config_dir=Path(config_dir), seed=SOLVER_SEED, time_limit_seconds=PRODUCTION_BUDGET_SECONDS)
                        validated = bool(production.get("independently_validated_period_repair"))
                    stop_reason = "validated_repair" if validated else "witness_acceptance_or_validation_unresolved"
                    rows.append(row)
                    checkpoint_state()
                    return True
            rows.append(row)
            if result.status == "MODEL_INVALID":
                failures.append(f"{run_id}:MODEL_INVALID")
                stop_reason = "correctness_failure"
                checkpoint_state()
                return True
            checkpoint_state()
            if result.status == "INFEASIBLE":
                stop_reason = f"{run_id}_infeasible_global_cap_proof"
                checkpoint_state()
                return False
        return False

    k1_stop = run_portfolio(singles, 1, k1_rows)
    if not k1_stop and not failures and not any(row.get("status") == "INFEASIBLE" for row in k1_rows):
        stop_reason = "k1_portfolio_exhausted"
    if not k1_stop and not failures:
        run_portfolio(pairs, 2, k2_rows)
    search_rows.extend(k1_rows)
    search_rows.extend(k2_rows)
    checkpoint_state()
    claim = _claim(k1_rows, k2_rows, accepted, validated)
    aggregate = {
        "experiment_name": manifest["experiment_name"], "phase": manifest["phase"], "target_scenario_id": TARGET_SCENARIO_ID,
        "result_classification": claim["claim"],
        "k1_runs": len(k1_rows), "k2_runs": len(k2_rows), "previous_stage1_reruns": 0,
        "fixed_witness_acceptance_runs": int(acceptance.get("status") not in {"not_run", None}),
        "production_validation_runs": int(production.get("status") not in {"not_run", None}),
        "control_runs": 0, "other_normal_target_runs": 0, "stage2_runs": 0, "stage3_runs": 0, "stage4_runs": 0,
        "stress_runs": 0, "negative_runs": 0, "holdout_runs": 0,
        "external_persisted_seed": False, "joint_bootstrap_witness_valid": bool(witness.get("joint_bootstrap_witness_valid")),
        "production_fixed_witness_accepted": bool(acceptance.get("production_fixed_witness_accepted")),
        "independently_validated_period_repair": validated, "minimum_claim": claim, "stop_reason": stop_reason,
        "failures": failures,
    }
    _write_json(output / "joint_witness.json", witness)
    _write_json(output / "production_fixed_witness_acceptance.json", acceptance)
    _write_json(output / "production_cold_start_validation.json", production)
    _write_json(output / "failures.json", {"failures": failures, "unexpected_failure_count": len(failures)})
    _write_json(output / "aggregate_summary.json", aggregate)
    write_checksums(output)
    return aggregate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preview-dir", type=Path, default=DEFAULT_PREVIEW_OUTPUT)
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--previous-stage1", type=Path, default=DEFAULT_PREVIOUS_STAGE1)
    parser.add_argument("--size-audit", type=Path, default=DEFAULT_SIZE_AUDIT)
    parser.add_argument("--config-dir", type=Path, default=Path("data/config"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(run_bootstrap(manifest_path=args.manifest, output_dir=args.output_dir, preview_dir=args.preview_dir, audit_root=args.audit_root, previous_stage1=args.previous_stage1, size_audit=args.size_audit, config_dir=args.config_dir, resume=args.resume), indent=2, sort_keys=True, default=str))
    except (BootstrapError, OSError, ValueError) as exc:
        print(f"Hybrid Stage 1 incumbent bootstrap FAILED: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
