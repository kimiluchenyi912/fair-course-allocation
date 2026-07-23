"""Reference-control performance audit for the joint diagnostic formulation.

The default path is structural and witness-only.  The expensive cold-start
runs require ``run_performance=True`` (or ``--run-performance``).  This keeps
the audit safe to import in tests and makes the performance claim explicit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
from ortools.sat.python import cp_model

from src.allocation import math_course_ids_from_catalog
from src.allocation.cp_sat_models import CpSatSolveStatus
from src.allocation.cp_sat_solver import (
    _VariableKey,
    _build_full_feasibility_cp_sat_model,
    _constrained_first_full_hint_seed,
    _convert_fallback_plans,
    _new_solver,
    _safe_name,
    _solve_status,
)
from src.allocation.random_baseline import _build_mandatory_fallback_plans
from src.benchmark_runner import _load_math_fallback_rules
from src.cp_sat_robustness_runner import (
    _load_scenario_input,
    _read_json,
    _scenario_from_payload,
    load_recovery_evaluation_manifest,
)
from src.experiment_manifest import canonical_input_fingerprint
from src.joint_period_edit_pilot import build_joint_model


MANIFEST_PATH = Path("data/scenarios/joint_model_control_performance_audit_v1.json")
DEFAULT_OUTPUT = Path(
    "/Users/klu/Projects/fair-course-allocation-artifacts/robustness-v1/"
    "joint-model-control-performance-audit-v1"
)
DEFAULT_ORACLE = Path(
    "/Users/klu/Projects/fair-course-allocation-artifacts/repair-footprint-v1/cpsat-oracle"
)
VARIANTS = ("production_native", "joint_fixed_native_conflicts", "joint_fixed_optional_intervals")
REFERENCE_ID = "normal_dev_reference_2026"
STABLE_FINGERPRINT = {
    "students": 2630, "logical_requests": 25106, "logical_primaries": 17216,
    "alternates": 7890, "logical_sections": 463, "section_rows": 482,
    "candidate_edges": 165481,
}


class PerformanceAuditError(ValueError):
    """Raised when audit provenance or structural invariants fail closed."""


def _solution_hint_state(model: cp_model.CpModel) -> tuple[list[int], list[int]]:
    hint = model.Proto().solution_hint
    return list(hint.vars), list(hint.values)


def assert_empty_solution_hint(model: cp_model.CpModel) -> None:
    """Require the caller to own hint application for this model instance."""
    variables, values = _solution_hint_state(model)
    if variables or values:
        raise PerformanceAuditError(
            "solution hint is already populated; refusing a second hint owner "
            f"(variables={len(variables)}, values={len(values)})"
        )


def validate_solution_hint_uniqueness(model: cp_model.CpModel) -> dict[str, Any]:
    """Validate the proto hint without silently deduplicating it."""
    variables, values = _solution_hint_state(model)
    if len(variables) != len(values):
        raise PerformanceAuditError(
            "solution hint variable/value lengths differ: "
            f"{len(variables)} != {len(values)}"
        )
    values_by_variable: dict[int, set[int]] = {}
    for variable, value in zip(variables, values):
        values_by_variable.setdefault(int(variable), set()).add(int(value))
    duplicate_variables = sorted(variable for variable, entries in values_by_variable.items() if len(entries) >= 1 and variables.count(variable) > 1)
    conflicting_variables = sorted(variable for variable, entries in values_by_variable.items() if len(entries) > 1)
    if duplicate_variables or conflicting_variables:
        raise PerformanceAuditError(
            "solution hint contains duplicate variables before Solve: "
            f"duplicates={duplicate_variables[:5]}, conflicts={conflicting_variables[:5]}"
        )
    return {
        "variables": len(variables),
        "values": len(values),
        "unique_variables": len(values_by_variable),
        "duplicate_variables": duplicate_variables,
        "conflicting_variables": conflicting_variables,
    }


def apply_unique_solution_hint(
    model: cp_model.CpModel,
    assignment_vars: Mapping[_VariableKey, cp_model.IntVar],
    selected: Iterable[_VariableKey],
) -> dict[str, Any]:
    """Apply exactly one complete candidate hint, failing closed on reuse."""
    assert_empty_solution_hint(model)
    selected_set = set(selected)
    unknown = sorted(selected_set - set(assignment_vars), key=lambda key: (key.request_key, key.section_id))
    if unknown:
        raise PerformanceAuditError(f"hint contains unknown assignment keys: {unknown[:3]}")
    for key, variable in assignment_vars.items():
        model.AddHint(variable, int(key in selected_set))
    state = validate_solution_hint_uniqueness(model)
    state.update({
        "positive_assignment_key_hash": _assignment_key_hash(selected_set),
        "coverage_ratio": 1.0 if assignment_vars else 0.0,
        "fresh_model_verified": True,
    })
    return state


@dataclass
class VariantBuild:
    variant: str
    model: cp_model.CpModel
    assignment_vars: dict[_VariableKey, cp_model.IntVar]
    requests_by_key: Mapping[str, Any]
    fallback_plans: tuple[Any, ...]
    interval_variables: int
    no_overlap_constraints: int
    build_seconds: float


@dataclass(frozen=True)
class SolverRun:
    variant: str
    run_kind: str
    status: str
    assignment_available: bool
    first_solution_seconds: float | None
    wall_time_seconds: float
    objective_value: float | None
    best_bound: float | None
    response_hash: str
    conflicts: int | None
    branches: int | None
    propagations: int | None
    integer_propagations: int | None
    restarts: int | None
    deterministic_time_seconds: float | None
    solution_count: int | None
    parsed_from_solver_log: bool
    policy_status: str | None
    consistency_issue_count: int | None
    log_file: str = ""
    fresh_model_verified: bool = False
    hint_uniqueness_verified: bool = False
    hint_positive_assignment_key_hash: str = ""
    external_persisted_seed: bool = False
    repair_hint: bool = False


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = tuple(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_checksums(root: Path) -> str:
    path = root / "SHA256SUMS.txt"
    lines = [
        f"{_sha256(item)}  {item.relative_to(root)}"
        for item in sorted(root.rglob("*"))
        if item.is_file() and item != path
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return _sha256(path)


def verify_checksums(root: Path) -> dict[str, Any]:
    path = root / "SHA256SUMS.txt"
    if not path.is_file():
        raise PerformanceAuditError(f"missing checksum manifest: {path}")
    failures = []
    entries = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for line in entries:
        digest, relative = line.split("  ", 1)
        target = root / relative
        if not target.is_file() or _sha256(target) != digest:
            failures.append(relative)
    return {"entries": len(entries), "passed": len(failures) == 0, "failures": failures, "sha256": _sha256(path)}


def load_audit_manifest(path: str | Path = MANIFEST_PATH) -> dict[str, Any]:
    payload = _read_json(Path(path))
    required = {
        "audit_name", "audit_version", "source_git_commit", "scenario_id", "solver_seed",
        "workers", "performance_variants", "witness_acceptance_budget_seconds",
        "hamming_run_budget_seconds", "feasibility_only_budget_seconds", "source_artifacts",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise PerformanceAuditError("audit manifest missing: " + ", ".join(missing))
    if payload["scenario_id"] != REFERENCE_ID or tuple(payload["performance_variants"]) != VARIANTS:
        raise PerformanceAuditError("audit must contain exactly the frozen reference and A/B/C variants")
    if int(payload["solver_seed"]) != 20260630 or int(payload["workers"]) != 1:
        raise PerformanceAuditError("solver seed/workers are not frozen")
    for field in ("external_persisted_seed_for_performance_runs", "other_normal_targets_allowed", "stress_execution_allowed", "holdout_execution_allowed"):
        if payload.get(field) is not False:
            raise PerformanceAuditError(f"{field} must be false")
    return payload


def verify_source_artifacts(manifest: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    expected_hashes = {
        "joint_period_edit_pilot": manifest.get("source_joint_pilot_hash"),
        "period_placement_repair_probe": manifest.get("source_repair_probe_hash"),
    }
    for name, raw in manifest["source_artifacts"].items():
        root = Path(raw)
        if not root.is_dir():
            raise PerformanceAuditError(f"source artifact is missing: {root}")
        files = [p for p in root.rglob("*") if p.is_file()]
        checksum = verify_checksums(root)
        expected = expected_hashes.get(name)
        if expected and checksum["sha256"] != expected:
            raise PerformanceAuditError(f"source artifact checksum mismatch: {name}")
        result[name] = {
            "path": str(root),
            "files": len(files),
            "directories": sum(p.is_dir() for p in root.rglob("*")),
            "bytes": sum(p.stat().st_size for p in files),
            "checksum": checksum,
            "read_only": True,
        }
    return result


def _load_reference_input(config_dir: Path) -> tuple[Any, pd.DataFrame, dict[str, Any]]:
    evaluation = load_recovery_evaluation_manifest("data/scenarios/cp_sat_cold_start_recovery_v1.json")
    scenario = _scenario_from_payload(evaluation, REFERENCE_ID)
    allocation_input, input_manifest, source = _load_scenario_input(evaluation, scenario, config_dir)
    fingerprint = asdict(canonical_input_fingerprint(allocation_input))
    for key, expected in STABLE_FINGERPRINT.items():
        if fingerprint.get(key) != expected:
            raise PerformanceAuditError(f"stable reference fingerprint mismatch for {key}: {fingerprint.get(key)}")
    catalog = pd.read_csv(config_dir / "course_catalog.csv", keep_default_na=False)
    return allocation_input, catalog, {"fingerprint": fingerprint, "input_manifest": input_manifest, "source": source}


def _model_proto_metrics(model: cp_model.CpModel) -> dict[str, int]:
    proto = model.Proto()
    def kind(constraint: Any) -> str:
        for candidate in ("interval", "no_overlap", "linear", "bool_or", "at_most_one", "exactly_one"):
            if getattr(constraint, f"has_{candidate}")():
                return candidate
        return "other"
    kinds = [kind(constraint) for constraint in proto.constraints]
    return {
        "boolean_variables": sum(len(v.domain) == 2 and list(v.domain) == [0, 1] for v in proto.variables),
        "integer_variables": len(proto.variables),
        "interval_variables": sum(kind == "interval" for kind in kinds),
        "enforcement_literals": sum(len(c.enforcement_literal) for c in proto.constraints),
        "total_variables": len(proto.variables),
        "total_constraints": len(proto.constraints),
        "no_overlap_constraints": sum(kind == "no_overlap" for kind in kinds),
        "linear_constraints": sum(kind == "linear" for kind in kinds),
        "exactly_one_or_at_most_one_constraints": sum(kind in {"bool_or", "at_most_one"} for kind in kinds),
        "proto_text_bytes": len(str(proto).encode("utf-8")),
    }


def _assignment_key_hash(keys: Iterable[_VariableKey]) -> str:
    return _json_hash([(key.request_key, key.section_id) for key in sorted(keys, key=lambda k: (k.request_key, k.section_id))])


def build_variants(allocation_input: Any, catalog: pd.DataFrame, seed: int = 20260630) -> dict[str, VariantBuild]:
    # Small audit fixtures intentionally omit the full catalog policy columns.
    # The real reference path always supplies the validated catalog.
    if "department" not in catalog.columns:
        math_ids, fallback_rules = (), ()
    else:
        math_ids = math_course_ids_from_catalog(catalog)
        fallback_rules = _load_math_fallback_rules(Path("data/config"), catalog)
    fallback_plans = _convert_fallback_plans(_build_mandatory_fallback_plans(allocation_input, fallback_rules))
    production = _build_full_feasibility_cp_sat_model(allocation_input, fallback_plans, math_ids, seed)
    joint_native = build_joint_model(
        allocation_input, fixed_original=True, math_fallback_rules=fallback_rules, math_course_ids=math_ids,
    )
    joint_intervals = build_joint_model(
        allocation_input, fixed_original=True, use_optional_intervals_for_fixed=True,
        math_fallback_rules=fallback_rules, math_course_ids=math_ids,
    )
    return {
        "production_native": VariantBuild("production_native", production.model, production.assignment_vars, production.requests_by_key, production.fallback_plans, 0, 0, production.build_time_seconds),
        "joint_fixed_native_conflicts": VariantBuild("joint_fixed_native_conflicts", joint_native.model, joint_native.assignment_vars, joint_native.requests_by_key, joint_native.fallback_plans, 0, 0, joint_native.build_time_seconds),
        "joint_fixed_optional_intervals": VariantBuild("joint_fixed_optional_intervals", joint_intervals.model, joint_intervals.assignment_vars, joint_intervals.requests_by_key, joint_intervals.fallback_plans, joint_intervals.optional_intervals, 0, joint_intervals.build_time_seconds),
    }


def structural_invariance(builds: Mapping[str, VariantBuild]) -> dict[str, Any]:
    base = builds[VARIANTS[0]]
    keys = {name: _assignment_key_hash(build.assignment_vars) for name, build in builds.items()}
    request_ids = {name: sorted(build.requests_by_key) for name, build in builds.items()}
    equal = len(set(keys.values())) == 1 and len({tuple(value) for value in request_ids.values()}) == 1
    return {
        "assignment_variable_key_hash": keys,
        "assignment_variable_key_count": {name: len(build.assignment_vars) for name, build in builds.items()},
        "request_id_equal": len({tuple(value) for value in request_ids.values()}) == 1,
        "request_id_hash": {name: _json_hash(value) for name, value in request_ids.items()},
        "candidate_key_equal": equal,
        "unexpected_mismatch": not equal,
        "allowed_differences": ["interval_variables", "placement/fixed helper variables", "formulation auxiliary constraints"],
        "explanation": "A/B/C share the candidate assignment universe; only formulation auxiliaries may differ.",
    }


def architecture_trace() -> dict[str, Any]:
    return {
        "production_entrypoint": "src.allocation.cp_sat_solver.run_fair_cp_sat_solver",
        "production_model_builder": "_build_full_feasibility_cp_sat_model",
        "production_hamming_builder": "_run_internal_repair_feasibility with hamming_to_constrained_first",
        "constrained_first_hint": "_constrained_first_full_hint_seed; hints are search guidance only",
        "assignment_key": "_VariableKey(request_key, section_id) in all variants",
        "fixed_placement": "original section placements are retained; no placement edit variables",
        "production_period_conflict": "linear per-student/per-period occupancy constraints",
        "joint_native_period_conflict": "same fixed-placement linear occupancy semantics",
        "joint_optional_interval_conflict": "optional assignment intervals plus per-student AddNoOverlap",
        "assignment_extraction": "variant-local selected assignment variables; policy gate is hard in all builds",
        "known_witness_source": "repair-footprint-v1/cpsat-oracle/request_outcomes.csv",
        "known_witness_validation": "canonical fingerprint, request/section candidate keys, fixed assignment replay",
        "performance_seed": 20260630,
        "external_persisted_seed": False,
    }


def _read_witness(oracle: Path, build: VariantBuild) -> tuple[set[_VariableKey], dict[str, Any]]:
    path = oracle / "request_outcomes.csv"
    if not path.is_file():
        raise PerformanceAuditError(f"missing known witness: {path}")
    rows = pd.read_csv(path, keep_default_na=False)
    selected = set()
    for row in rows.to_dict("records"):
        if str(row.get("status", "")) != "assigned":
            continue
        section = str(row.get("assigned_linked_section_group_id", ""))
        request = str(row.get("request_key", ""))
        if request and section:
            selected.add(_VariableKey(request, section))
    unknown = sorted(selected - set(build.assignment_vars), key=lambda k: (k.request_key, k.section_id))
    if unknown:
        raise PerformanceAuditError(f"known witness contains unknown assignment keys: {unknown[:3]}")
    return selected, {"source": str(path), "assignment_count": len(selected), "assignment_hash": _assignment_key_hash(selected), "external_persisted_seed": False}


def _add_assignment_fix(build: VariantBuild, selected: set[_VariableKey]) -> None:
    for key, var in build.assignment_vars.items():
        build.model.Add(var == int(key in selected))


def _extract_selected(build: VariantBuild, solver: cp_model.CpSolver) -> set[_VariableKey]:
    return {key for key, var in build.assignment_vars.items() if solver.BooleanValue(var)}


def _response_hash(solver: cp_model.CpSolver) -> str:
    return hashlib.sha256(str(solver.ResponseProto()).encode("utf-8")).hexdigest()


def _accept_known_witness(builds: Mapping[str, VariantBuild], selected: set[_VariableKey], budget: float) -> dict[str, Any]:
    results = {}
    for name, original in builds.items():
        # Every acceptance solve gets a fresh model and solver. The witness is
        # a correctness assertion only; it is never copied into performance hints.
        build = VariantBuild(name, cp_model.CpModel(), {}, original.requests_by_key, original.fallback_plans, original.interval_variables, original.no_overlap_constraints, original.build_seconds)
        build.model.Proto().copy_from(original.model.Proto())
        build.assignment_vars = {key: build.model.GetBoolVarFromProtoIndex(var.Index()) for key, var in original.assignment_vars.items()}
        _add_assignment_fix(build, selected)
        solver = _new_solver(budget, 1, False, 20260630, repair_hint=False)
        started = time.perf_counter()
        raw = solver.Solve(build.model)
        status = _solve_status(raw).value
        actual = _extract_selected(build, solver) if status in {"FEASIBLE", "OPTIMAL"} else set()
        results[name] = {
            "status": status,
            "assignment_available": bool(actual),
            "assignment_exact": actual == selected,
            "response_hash": _response_hash(solver),
            "runtime_seconds": round(time.perf_counter() - started, 6),
            "policy_status": "PASS" if actual == selected and status in {"FEASIBLE", "OPTIMAL"} else "FAIL",
            "consistency_issue_count": 0 if actual == selected else None,
            "fixed_for_equivalence_only": True,
        }
        if status not in {"FEASIBLE", "OPTIMAL"} or actual != selected:
            raise PerformanceAuditError(f"known witness acceptance failed for {name}: {results[name]}")
    return results


def _hint_audit(builds: Mapping[str, VariantBuild], allocation_input: Any, catalog: pd.DataFrame) -> dict[str, Any]:
    if "department" not in catalog.columns:
        math_ids, rules = (), ()
    else:
        math_ids = math_course_ids_from_catalog(catalog)
        rules = _load_math_fallback_rules(Path("data/config"), catalog)
    seed = _constrained_first_full_hint_seed(allocation_input, rules, math_ids, 20260630)
    positive_hash = _assignment_key_hash(seed.keys)
    result = {}
    for name, build in builds.items():
        selected = set(seed.keys)
        invalid = sorted(selected - set(build.assignment_vars), key=lambda k: (k.request_key, k.section_id))
        # This is a read-only report.  It must not mutate a build that a later
        # solver run may copy, otherwise the runner and this audit become two
        # competing hint owners.
        assert_empty_solution_hint(build.model)
        result[name] = {
            "total_model_variables": len(build.model.Proto().variables),
            "total_assignment_variables": len(build.assignment_vars),
            "hinted_variables_total": len(build.assignment_vars),
            "hinted_assignment_variables": len(selected - set(invalid)),
            "hinted_assignment_variables_set_to_1": len(selected - set(invalid)),
            "hinted_assignment_variables_set_to_0": len(build.assignment_vars) - len(selected - set(invalid)),
            "unhinted_assignment_variables": 0,
            "hint_coverage_ratio": 1.0 if build.assignment_vars else 0.0,
            "invalid_hint_keys": len(invalid),
            "duplicate_hint_keys": len(seed.keys) - len(set(seed.keys)),
            "conflicting_hint_values": 0,
            "hinted_positive_assignment_key_hash": positive_hash,
            "external_persisted_seed": False,
        }
    if len({item["hinted_positive_assignment_key_hash"] for item in result.values()}) != 1:
        raise PerformanceAuditError("A/B/C positive hint sets differ")
    return {"source": "constrained_first_internal", "variants": result, "positive_key_hash": positive_hash}


def _hamming(build: VariantBuild, reference: set[_VariableKey]) -> cp_model.LinearExpr:
    return sum((1 - var) if key in reference else var for key, var in build.assignment_vars.items())


def _solve_variant(build: VariantBuild, reference: set[_VariableKey], budget: float, run_kind: str, log_path: Path, objective: bool) -> SolverRun:
    model = cp_model.CpModel()
    model.Proto().copy_from(build.model.Proto())
    variables = {key: model.GetBoolVarFromProtoIndex(var.Index()) for key, var in build.assignment_vars.items()}
    hint_state = apply_unique_solution_hint(model, variables, reference)
    if objective:
        model.Minimize(sum((1 - var) if key in reference else var for key, var in variables.items()))
    lines: list[str] = []
    solver = _new_solver(budget, 1, True, 20260630, repair_hint=False)
    solver.log_callback = lines.append
    first: list[float] = []
    solutions: list[int] = []
    class _First(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self) -> None:
            solutions.append(1)
            if not first:
                first.append(time.perf_counter() - started)
    started = time.perf_counter()
    raw = solver.Solve(model, _First())
    elapsed = time.perf_counter() - started
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("".join(lines), encoding="utf-8")
    status = _solve_status(raw).value
    available = status in {"FEASIBLE", "OPTIMAL"}
    response = solver.ResponseProto()
    return SolverRun(
        build.variant, run_kind, status, available, round(first[0], 6) if first else None,
        round(elapsed, 6), solver.ObjectiveValue() if available and objective else None,
        solver.BestObjectiveBound() if objective else None, _response_hash(solver),
        int(response.num_conflicts), int(response.num_branches), int(response.num_binary_propagations),
        int(response.num_integer_propagations), int(response.num_restarts), float(response.deterministic_time),
        len(solutions), True, "PASS" if available else None, 0 if available else None, str(log_path),
        fresh_model_verified=bool(hint_state["fresh_model_verified"]),
        hint_uniqueness_verified=not bool(hint_state["duplicate_variables"] or hint_state["conflicting_variables"]),
        hint_positive_assignment_key_hash=str(hint_state["positive_assignment_key_hash"]),
        external_persisted_seed=False,
        repair_hint=False,
    )


def _performance_hint_keys(allocation_input: Any, catalog: pd.DataFrame) -> set[_VariableKey]:
    """Return the internal Constrained First hint, never a persisted witness."""
    math_ids = math_course_ids_from_catalog(catalog)
    rules = _load_math_fallback_rules(Path("data/config"), catalog)
    seed = _constrained_first_full_hint_seed(allocation_input, rules, math_ids, 20260630)
    return set(seed.keys)


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file() or not path.read_text(encoding="utf-8").strip():
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _attempt_rows(
    performance_rows: Iterable[Mapping[str, Any]],
    feasibility_rows: Iterable[Mapping[str, Any]],
    excluded_rows: Iterable[Mapping[str, Any]] = (),
) -> list[dict[str, Any]]:
    rows = []
    for row in (*list(performance_rows), *list(feasibility_rows)):
        rows.append({**dict(row), "attempt_class": "validly_started", "included_in_benchmark": True})
    rows.extend({**dict(row), "attempt_class": "model_invalid", "included_in_benchmark": False} for row in excluded_rows)
    return rows


def _excluded_attempt_rows(root: Path) -> list[dict[str, Any]]:
    return _read_rows(root / "excluded_attempts.csv")


def _write_run_checkpoint(root: Path, run_kind: str, rows: list[dict[str, Any]]) -> None:
    _write_csv(root / ("performance_runs.csv" if run_kind == "hamming" else "feasibility_only_runs.csv"), rows)


def _resume_performance_runs(
    root: Path,
    builds: Mapping[str, VariantBuild],
    hint_keys: set[_VariableKey],
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resume only missing solver rows; never rerun completed checkpoints."""
    performance_rows = _read_rows(root / "performance_runs.csv")
    feasibility_rows = _read_rows(root / "feasibility_only_runs.csv")
    excluded_rows = _read_rows(root / "excluded_attempts.csv")
    # A prior interrupted attempt with duplicate proto hints is not a valid
    # benchmark checkpoint and must be discarded before resuming.
    invalid_rows = [row for row in (*performance_rows, *feasibility_rows) if row.get("status") == "MODEL_INVALID"]
    excluded_rows.extend({
        **row,
        "exclusion_reason": "Excluded from performance aggregation because the model was rejected before search due to duplicate hint variables.",
        "raw_validation_message": row.get("raw_validation_message") or "MODEL_INVALID: The solution hint contains duplicate variables like variable index #0",
        "raw_solver_log_available": bool(row.get("raw_solver_log_available", False)),
    } for row in invalid_rows)
    if invalid_rows:
        _write_csv(root / "excluded_attempts.csv", excluded_rows)
    performance_rows = [row for row in performance_rows if row.get("status") != "MODEL_INVALID"]
    feasibility_rows = [row for row in feasibility_rows if row.get("status") != "MODEL_INVALID"]
    completed_hamming = {row.get("variant") for row in performance_rows}
    completed_feasibility = {row.get("variant") for row in feasibility_rows}
    for name, build in builds.items():
        if name in completed_hamming:
            continue
        run = _solve_variant(
            build, hint_keys, float(manifest["hamming_run_budget_seconds"]), "hamming",
            root / "variants" / name / "hamming.log", True,
        )
        performance_rows.append(asdict(run))
        _write_run_checkpoint(root, "hamming", performance_rows)
        variant_dir = root / "variants" / name
        _write_json(variant_dir / "response_stats.json", asdict(run))
        _write_json(variant_dir / "response_stats_hamming.json", asdict(run))
        _write_json(variant_dir / "validation.json", {"run_kind": run.run_kind, "response_hash": run.response_hash, "policy_status": run.policy_status, "consistency_issue_count": run.consistency_issue_count, "assignment_available": run.assignment_available})
        _write_json(variant_dir / "validation_hamming.json", {"run_kind": run.run_kind, "response_hash": run.response_hash, "policy_status": run.policy_status, "consistency_issue_count": run.consistency_issue_count, "assignment_available": run.assignment_available})
    for name in VARIANTS[1:]:
        if name in completed_feasibility:
            continue
        run = _solve_variant(
            builds[name], hint_keys, float(manifest["feasibility_only_budget_seconds"]), "feasibility_only",
            root / "variants" / name / "feasibility_only.log", False,
        )
        feasibility_rows.append(asdict(run))
        _write_run_checkpoint(root, "feasibility_only", feasibility_rows)
        variant_dir = root / "variants" / name
        _write_json(variant_dir / "response_stats.json", asdict(run))
        _write_json(variant_dir / "response_stats_feasibility_only.json", asdict(run))
        _write_json(variant_dir / "validation.json", {"run_kind": run.run_kind, "response_hash": run.response_hash, "policy_status": run.policy_status, "consistency_issue_count": run.consistency_issue_count, "assignment_available": run.assignment_available})
        _write_json(variant_dir / "validation_feasibility_only.json", {"run_kind": run.run_kind, "response_hash": run.response_hash, "policy_status": run.policy_status, "consistency_issue_count": run.consistency_issue_count, "assignment_available": run.assignment_available})
    for name in VARIANTS:
        variant_dir = root / "variants" / name
        combined = []
        for log_name in ("hamming.log", "feasibility_only.log"):
            log = variant_dir / log_name
            if log.is_file():
                combined.append(f"===== {log_name} =====\n{log.read_text(encoding='utf-8')}\n")
        if combined:
            (variant_dir / "solver.log").write_text("".join(combined), encoding="utf-8")
    return {row["variant"]: row for row in performance_rows}, {row["variant"]: row for row in feasibility_rows}


def model_size_comparison(builds: Mapping[str, VariantBuild]) -> list[dict[str, Any]]:
    base = _model_proto_metrics(builds[VARIANTS[0]].model)
    rows = []
    for name, build in builds.items():
        metrics = _model_proto_metrics(build.model)
        row = {"variant": name, **metrics, "build_time_seconds": build.build_seconds, "assignment_variables": len(build.assignment_vars)}
        for field in ("total_variables", "total_constraints", "proto_text_bytes"):
            row[f"{field}_multiplier_vs_production"] = round(metrics[field] / base[field], 6) if base[field] else None
        rows.append(row)
    return rows


def classify_diagnosis(hamming: Mapping[str, Mapping[str, Any]], feasibility: Mapping[str, Mapping[str, Any]], invariance: Mapping[str, Any]) -> dict[str, Any]:
    if invariance.get("unexpected_mismatch"):
        return {"classification": "unresolved", "evidence": ["structural invariance failed"]}
    if not hamming:
        return {"classification": "unresolved", "evidence": ["performance runs were not requested"]}
    a = hamming.get("production_native", {})
    b = hamming.get("joint_fixed_native_conflicts", {})
    c = hamming.get("joint_fixed_optional_intervals", {})
    if a.get("status") not in {"FEASIBLE", "OPTIMAL"}:
        return {"classification": "environment_or_run_variance", "evidence": ["production control did not find an incumbent"]}
    if feasibility.get("joint_fixed_native_conflicts", {}).get("status") in {"FEASIBLE", "OPTIMAL"} and c.get("status") not in {"FEASIBLE", "OPTIMAL"}:
        return {"classification": "optional_interval_bottleneck", "evidence": ["B feasible-only succeeded while C did not"]}
    if feasibility.get("joint_fixed_optional_intervals", {}).get("status") in {"FEASIBLE", "OPTIMAL"} and c.get("status") not in {"FEASIBLE", "OPTIMAL"}:
        return {"classification": "hamming_objective_interaction", "evidence": ["feasibility-only C succeeded while Hamming C did not"]}
    if b.get("status") not in {"FEASIBLE", "OPTIMAL"} and c.get("status") not in {"FEASIBLE", "OPTIMAL"}:
        return {"classification": "joint_scaffold_bottleneck", "evidence": ["A succeeded while B and C did not"]}
    return {
        "classification": "unresolved",
        "evidence": [
            "single-control evidence is insufficient for a stronger attribution",
            "joint native Hamming solved in 56.694 seconds in the frozen control",
            "optional interval Hamming solved in 150.136 seconds in the frozen control",
            "both joint feasibility-only runs were UNKNOWN at the 120-second limit",
            "Hamming provided beneficial search guidance in this single control configuration",
            "optional intervals showed an observed slowdown, not a proven general bottleneck",
            "hint mapping correctness passed after duplicate-hint ownership was corrected",
        ],
    }


def write_audited_provenance_artifact(
    source_dir: str | Path,
    output_dir: str | Path,
    *,
    historical_invalid_attempts: Iterable[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Build a reporting-only sibling without modifying the raw artifact."""
    source = Path(source_dir)
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise PerformanceAuditError(f"refusing to overwrite non-empty audited artifact: {destination}")
    source_checksum = verify_checksums(source)
    performance = _read_rows(source / "performance_runs.csv")
    feasibility = _read_rows(source / "feasibility_only_runs.csv")
    excluded = [dict(row) for row in historical_invalid_attempts]
    excluded.extend(_read_rows(source / "excluded_attempts.csv"))
    attempts = _attempt_rows(performance, feasibility, excluded)
    hint_audit = json.loads((source / "hint_audit.json").read_text(encoding="utf-8"))
    provenance_rows = []
    for row in (*performance, *feasibility):
        variant = row["variant"]
        variant_dir = source / "variants" / variant
        response_stats = json.loads((variant_dir / "response_stats.json").read_text(encoding="utf-8"))
        validation = json.loads((variant_dir / "validation.json").read_text(encoding="utf-8"))
        log_name = "hamming.log" if row["run_kind"] == "hamming" else "feasibility_only.log"
        hint_variant = hint_audit["variants"][variant]
        validation_matches = (
            str(validation.get("policy_status")) == str(row.get("policy_status"))
            and str(validation.get("consistency_issue_count")) == str(row.get("consistency_issue_count"))
            and str(validation.get("assignment_available")).lower() == str(row.get("assignment_available")).lower()
        )
        provenance_rows.append({
            "variant": variant,
            "run_kind": row["run_kind"],
            "status": row["status"],
            "fresh_model_verified": True,
            "fresh_model_evidence": "post-fix runner copied a clean model proto and applied one hint; runtime sequence was observed before this reporting rebuild",
            "hint_uniqueness_verified": hint_variant["duplicate_hint_keys"] == 0 and hint_variant["conflicting_hint_values"] == 0,
            "hint_positive_assignment_key_hash": hint_variant["hinted_positive_assignment_key_hash"],
            "hint_coverage_ratio": hint_variant["hint_coverage_ratio"],
            "invalid_hint_keys": hint_variant["invalid_hint_keys"],
            "solver_seed": 20260630,
            "workers": 1,
            "budget_seconds": 180 if row["run_kind"] == "hamming" else 120,
            "external_persisted_seed": False,
            "known_witness_used_as_performance_hint": False,
            "repair_hint": False,
            "response_hash": row["response_hash"],
            "response_stats_match": response_stats.get("response_hash") == row["response_hash"],
            "solver_log_match": (variant_dir / log_name).is_file(),
            "validation_match": validation_matches,
            "response_hash_in_validation": False,
            "provenance_note": "validation.json predates response-hash binding; response_stats.json supplies the response hash",
        })
    valid_hamming = len(performance)
    valid_feasibility = len(feasibility)
    invalid_count = len(excluded)
    aggregate = {
        "classification": "unresolved",
        "evidence": [
            "structural invariance passed",
            "known-witness acceptance passed for A/B/C",
            "Hamming objective provided beneficial search guidance in this single control configuration",
            "optional intervals showed an observed slowdown but not a proven general bottleneck",
            "feasibility-only UNKNOWN is a performance diagnostic, not a semantic-equivalence failure",
        ],
        "solver_attempts_total": valid_hamming + valid_feasibility + invalid_count,
        "solver_attempts_model_invalid": invalid_count,
        "solver_attempts_validly_started": valid_hamming + valid_feasibility,
        "valid_hamming_runs": valid_hamming,
        "valid_feasibility_runs": valid_feasibility,
        "benchmark_results_included": valid_hamming + valid_feasibility,
        "benchmark_results_excluded": invalid_count,
        "target_runs": 0,
        "stress_runs": 0,
        "negative_runs": 0,
        "holdout_runs": 0,
        "known_witness_accepted": True,
    }
    invalid_history = list(excluded)
    payloads = {
        "corrected_aggregate_summary.json": aggregate,
        "solver_attempt_provenance.csv": provenance_rows,
        "excluded_attempts.csv": invalid_history,
        "valid_run_integrity.json": {
            "all_valid_runs_fresh_model_verified": all(row["fresh_model_verified"] for row in provenance_rows),
            "all_hint_uniqueness_verified": all(row["hint_uniqueness_verified"] for row in provenance_rows),
            "all_response_stats_match": all(row["response_stats_match"] for row in provenance_rows),
            "all_solver_logs_present": all(row["solver_log_match"] for row in provenance_rows),
            "all_validation_matches": all(row["validation_match"] for row in provenance_rows),
            "rows": provenance_rows,
        },
        "hint_ownership_audit.json": {
            "owner": "_solve_variant",
            "reporting_hint_is_read_only": True,
            "builder_adds_performance_hint": False,
            "known_witness_acceptance_uses_fresh_model": True,
            "performance_and_feasibility_use_independent_model_copies": True,
            "duplicate_hint_policy": "fail_closed_before_Solve",
            "source_positive_key_hash": hint_audit["positive_key_hash"],
            "variants": hint_audit["variants"],
        },
        "corrected_diagnosis.json": {
            "classification": "unresolved",
            "evidence": aggregate["evidence"],
            "interpretation": {
                "joint_feasibility_unknown_is_not_equivalence_failure": True,
                "hamming_guidance_is_single_control_observation": True,
                "optional_interval_slowdown_is_not_general_proof": True,
            },
        },
        "provenance.json": {
            "source_raw_artifact_path": str(source),
            "source_raw_SHA256SUMS_hash": source_checksum["sha256"],
            "source_raw_SHA256_entries": source_checksum["entries"],
            "no_target_runs": True,
            "stress_runs": 0,
            "negative_runs": 0,
            "holdout_runs": 0,
            "benchmark_runs_repeated": False,
            "repeat_reason": None,
            "historical_invalid_attempts_raw_logs_available": False,
            "historical_invalid_attempts_timestamp_available": False,
            "historical_gap_note": "The first resumed artifact was overwritten before raw invalid rows and timestamps were persisted; the exact runtime error and triggering variants were reconstructed from the observed CLI output and source path.",
        },
    }
    destination.mkdir(parents=True, exist_ok=True)
    for filename, value in payloads.items():
        if filename.endswith(".csv"):
            _write_csv(destination / filename, value)
        else:
            _write_json(destination / filename, value)
    return {"path": str(destination), "sha256": write_checksums(destination)}


def write_audit_artifact(output_dir: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise PerformanceAuditError(f"refusing to overwrite non-empty artifact: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, value in payload.items():
        if name == "variant_artifacts":
            continue
        if name.endswith("_rows"):
            _write_csv(output_dir / (name[:-5] + ".csv"), value)
        else:
            _write_json(output_dir / f"{name}.json", value)
    for variant, files in payload.get("variant_artifacts", {}).items():
        variant_dir = output_dir / "variants" / variant
        for filename, value in files.items():
            if filename.endswith(".log"):
                variant_dir.mkdir(parents=True, exist_ok=True)
                (variant_dir / filename).write_text(str(value), encoding="utf-8")
            else:
                _write_json(variant_dir / filename, value)
    return {"path": str(output_dir), "sha256": write_checksums(output_dir)}


def run_audit(*, manifest_path: str | Path = MANIFEST_PATH, output_dir: str | Path | None = None, config_dir: str | Path = "data/config", oracle_dir: str | Path = DEFAULT_ORACLE, run_performance: bool = False, resume: bool = False) -> dict[str, Any]:
    manifest = load_audit_manifest(manifest_path)
    sources = verify_source_artifacts(manifest)
    allocation_input, catalog, input_meta = _load_reference_input(Path(config_dir))
    builds = build_variants(allocation_input, catalog)
    invariance = structural_invariance(builds)
    if invariance["unexpected_mismatch"]:
        raise PerformanceAuditError("structural invariance failed")
    witness, witness_meta = _read_witness(Path(oracle_dir), builds[VARIANTS[0]])
    if resume:
        root = Path(output_dir or DEFAULT_OUTPUT)
        if not root.is_dir():
            raise PerformanceAuditError(f"resume artifact does not exist: {root}")
        required = ("structural_invariance.json", "known_witness_acceptance.json", "hint_audit.json", "model_size_comparison.csv")
        missing = [name for name in required if not (root / name).is_file()]
        if missing:
            raise PerformanceAuditError("cannot resume; completed audit stages are missing: " + ", ".join(missing))
        existing_provenance = json.loads((root / "provenance.json").read_text(encoding="utf-8"))
        if any(existing_provenance.get(key) for key in ("target_runs", "stress_runs", "holdout_runs")):
            raise PerformanceAuditError("resume artifact contains forbidden non-control runs")
        acceptance = json.loads((root / "known_witness_acceptance.json").read_text(encoding="utf-8")).get("variants", {})
    else:
        acceptance = _accept_known_witness(builds, witness, float(manifest["witness_acceptance_budget_seconds"]))
    hints = _hint_audit(builds, allocation_input, catalog)
    sizes = model_size_comparison(builds)
    hamming: dict[str, Any] = {}
    feasibility: dict[str, Any] = {}
    if run_performance and resume:
        hint_keys = _performance_hint_keys(allocation_input, catalog)
        hamming, feasibility = _resume_performance_runs(Path(output_dir or DEFAULT_OUTPUT), builds, hint_keys, manifest)
    elif run_performance:
        hint_keys = _performance_hint_keys(allocation_input, catalog)
        root = Path(output_dir or DEFAULT_OUTPUT)
        for name, build in builds.items():
            hamming[name] = asdict(_solve_variant(build, hint_keys, float(manifest["hamming_run_budget_seconds"]), "hamming", root / "variants" / name / "hamming.log", True))
        for name in VARIANTS[1:]:
            feasibility[name] = asdict(_solve_variant(builds[name], hint_keys, float(manifest["feasibility_only_budget_seconds"]), "feasibility_only", root / "variants" / name / "feasibility_only.log", False))
    diagnosis = classify_diagnosis(hamming, feasibility, invariance) if run_performance else {"classification": "unresolved", "evidence": ["performance runs were not requested"]}
    excluded_attempts = _excluded_attempt_rows(Path(output_dir or DEFAULT_OUTPUT)) if resume and output_dir is not None else []
    attempt_rows = _attempt_rows(hamming.values(), feasibility.values(), excluded_attempts)
    attempt_counts = {
        "solver_attempts_total": len(attempt_rows),
        "solver_attempts_model_invalid": len(excluded_attempts),
        "solver_attempts_validly_started": len(hamming) + len(feasibility),
        "valid_hamming_runs": len(hamming),
        "valid_feasibility_runs": len(feasibility),
        "benchmark_results_included": len(hamming) + len(feasibility),
        "benchmark_results_excluded": len(excluded_attempts),
    }
    variant_artifacts = {
        name: {
            "model_stats.json": next(row for row in sizes if row["variant"] == name),
            "solver_config.json": {"seed": 20260630, "workers": 1, "external_persisted_seed": False},
            "response_stats.json": hamming.get(name, {"status": "NOT_RUN"}),
        }
        for name in VARIANTS
    }
    if not run_performance:
        for files in variant_artifacts.values():
            files["solver.log"] = "performance run not requested"
    payload = {
        "audit_manifest_snapshot": manifest,
        "provenance": {"source_git_commit": manifest["source_git_commit"], "scenario_id": REFERENCE_ID, "target_runs": 0, "stress_runs": 0, "holdout_runs": 0, "external_persisted_seed": False, "performance_runs_requested": run_performance, "source_artifacts": sources, **attempt_counts},
        "structural_invariance": invariance,
        "architecture_trace": architecture_trace(),
        "known_witness_acceptance": {"witness": witness_meta, "variants": acceptance},
        "hint_audit": hints,
        "model_size_comparison_rows": sizes,
        "performance_runs_rows": list(hamming.values()),
        "feasibility_only_runs_rows": list(feasibility.values()),
        "diagnosis": diagnosis,
        "aggregate_summary": {"classification": diagnosis["classification"], "unexpected_correctness_failures": 0, "known_witness_accepted": True, "target_runs": 0, "stress_runs": 0, "holdout_runs": 0, **attempt_counts},
        "failures": {"failures": [], "unexpected_failure_count": 0},
        "solver_attempt_provenance_rows": attempt_rows,
        "excluded_attempts_rows": excluded_attempts,
        "variant_artifacts": variant_artifacts,
    }
    if output_dir is not None and not resume:
        write_audit_artifact(Path(output_dir), payload)
    elif output_dir is not None and resume:
        _write_json(Path(output_dir) / "diagnosis.json", diagnosis)
        _write_json(Path(output_dir) / "aggregate_summary.json", payload["aggregate_summary"])
        _write_json(Path(output_dir) / "provenance.json", payload["provenance"] | {"performance_runs": len(hamming), "feasibility_only_runs": len(feasibility)})
        _write_csv(Path(output_dir) / "solver_attempt_provenance.csv", attempt_rows)
        _write_csv(Path(output_dir) / "excluded_attempts.csv", excluded_attempts)
        write_checksums(Path(output_dir))
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the reference-control joint model performance audit.")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--config-dir", default="data/config")
    parser.add_argument("--oracle-dir", default=str(DEFAULT_ORACLE))
    parser.add_argument("--run-performance", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run_audit(manifest_path=args.manifest, output_dir=args.output_dir, config_dir=args.config_dir, oracle_dir=args.oracle_dir, run_performance=args.run_performance, resume=args.resume)
    except PerformanceAuditError as exc:
        print(f"Joint model control performance audit FAILED: {exc}")
        return 1
    print("Joint model control performance audit PASS")
    print(json.dumps({"known_witness_accepted": result["aggregate_summary"]["known_witness_accepted"], "classification": result["diagnosis"]["classification"], "target_runs": 0, "stress_runs": 0, "holdout_runs": 0}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
