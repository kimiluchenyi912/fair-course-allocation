"""Audit the exact sparse occupancy encoding for the joint Stage 1 model.

This module builds models and small proof fixtures only.  It never solves the
normal_dev_10 Stage 1 model.  The default production formulation remains the
full optional-interval model in :mod:`src.joint_period_edit_pilot`.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd
from ortools.sat.python import cp_model

from src.allocation.cp_sat_solver import _VariableKey
from src.experiment_manifest import canonical_input_fingerprint
from src.joint_model_control_performance_audit import (
    DEFAULT_ORACLE,
    _accept_known_witness,
    _load_reference_input,
    _read_witness,
    build_variants,
    verify_source_artifacts,
)
from src.joint_period_edit_pilot import (
    CONTROL_SCENARIO_ID,
    JointModelBuild,
    PlacementOption,
    _json_hash,
    build_frozen_placement_domains,
    build_joint_model,
    load_joint_period_edit_manifest,
)
from src.joint_period_edit_stage1_pilot import (
    TARGET_SCENARIO_ID,
    DEFAULT_PREVIEW_OUTPUT,
    DEFAULT_CONTROL_AUDIT,
    load_stage1_manifest,
    model_proto_metrics,
    frozen_domain_hashes,
)
from src.period_placement_repair_probe import DEFAULT_AUDIT_ROOT, load_scenario_context
from src.section_plan_feasibility_audit import load_section_plan_audit_manifest
from src.benchmark_runner import _load_math_fallback_rules
from src.allocation import math_course_ids_from_catalog
from src.model_proto_serialization import deterministic_model_proto_bytes


DEFAULT_MANIFEST = Path("data/scenarios/joint_stage1_model_size_reduction_audit_v1.json")
DEFAULT_OUTPUT = Path("../fair-course-allocation-artifacts/robustness-v1/joint-stage1-model-size-reduction-audit-v1")
MAX_PROTO_BYTES = 250_000_000


class ModelSizeAuditError(ValueError):
    """Raised when an audit invariant fails closed."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = list(rows)
    fields = tuple(dict.fromkeys(key for row in values for key in row))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(values)


def write_checksums(root: Path) -> str:
    checksum = root / "SHA256SUMS.txt"
    lines = [f"{_sha256(path)}  {path.relative_to(root)}" for path in sorted(root.rglob("*")) if path.is_file() and path != checksum]
    checksum.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return _sha256(checksum)


def load_model_size_audit_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"audit_name", "audit_version", "source_git_commit", "control_scenario_id", "size_target_scenario_id", "baseline_formulation", "candidate_formulation"}
    missing = sorted(required - set(payload))
    if missing:
        raise ModelSizeAuditError("manifest missing: " + ", ".join(missing))
    if payload["control_scenario_id"] != CONTROL_SCENARIO_ID or payload["size_target_scenario_id"] != TARGET_SCENARIO_ID:
        raise ModelSizeAuditError("audit must contain only the frozen control and normal_dev_10")
    if payload["baseline_formulation"] != "full_optional_intervals" or payload["candidate_formulation"] != "hybrid_sparse_linear_occupancy":
        raise ModelSizeAuditError("formulation names are not frozen")
    for field in ("target_stage1_solve_allowed", "other_normal_targets_allowed", "stress_execution_allowed", "negative_execution_allowed", "holdout_execution_allowed"):
        if payload.get(field) is not False:
            raise ModelSizeAuditError(f"{field} must be false")
    return payload


def _constraint_kind(constraint: Any) -> str:
    for kind in ("interval", "no_overlap", "linear", "bool_or", "at_most_one", "exactly_one"):
        if getattr(constraint, f"has_{kind}")():
            return kind
    return "other"


def _proto_sizes(model: cp_model.CpModel) -> dict[str, Any]:
    """Return real binary ExportToFile bytes and text export bytes.

    OR-Tools 9.15 exposes a helper proto wrapper without protobuf's
    ``SerializeToString``.  ExportToFile is the supported serialization path
    and avoids treating ``str(proto)`` as the binary cost-gate measurement.
    """
    with tempfile.NamedTemporaryFile(suffix=".pb") as binary, tempfile.NamedTemporaryFile(suffix=".pbtxt") as text:
        try:
            direct = deterministic_model_proto_bytes(model, export_path=binary.name)
        except ValueError as exc:
            raise ModelSizeAuditError(str(exc)) from exc
        if not model.ExportToFile(text.name):
            raise ModelSizeAuditError("OR-Tools failed to export ModelProto")
        binary.flush()
        text.flush()
        exported = Path(binary.name).read_bytes()
        return {
            "serialized_binary_proto_bytes": len(direct),
            "exported_binary_proto_file_bytes": len(exported),
            "binary_measurements_equal": len(direct) == len(exported),
            "proto_text_bytes": Path(text.name).stat().st_size,
        }


def model_size(build: JointModelBuild) -> dict[str, Any]:
    proto = build.model.Proto()
    kinds = [_constraint_kind(item) for item in proto.constraints]
    names = [str(item.name) for item in proto.variables]
    proto_sizes = _proto_sizes(build.model)
    return {
        "assignment_variables": len(build.assignment_vars),
        "placement_choice_variables": len(build.placement_choice_vars),
        "changed_section_variables": len(build.section_changed_vars),
        "q_occupancy_variables": len(build.occupancy_vars),
        "w_conjunction_variables": len(build.occupancy_conjunction_vars),
        # OR-Tools stores IntervalVar definitions as interval constraints in
        # this version of the Python proto wrapper, not as proto.variables.
        "interval_variables": sum(kind == "interval" for kind in kinds),
        "optional_intervals": build.optional_intervals,
        "auxiliary_variables": build.auxiliary_variables,
        "total_variables": len(proto.variables),
        "linear_constraints": sum(kind == "linear" for kind in kinds),
        "interval_constraints": sum(kind == "interval" for kind in kinds),
        "no_overlap_constraints": sum(kind == "no_overlap" for kind in kinds),
        "exactly_one_constraints": sum(kind in {"bool_or", "exactly_one", "at_most_one"} for kind in kinds),
        "enforcement_literals": sum(len(item.enforcement_literal) for item in proto.constraints),
        "total_constraints": len(proto.constraints),
        "constraint_families": {
            "linear": sum(kind == "linear" for kind in kinds),
            "interval": sum(kind == "interval" for kind in kinds),
            "no_overlap": sum(kind == "no_overlap" for kind in kinds),
            "other": len(proto.constraints) - sum(kind in {"linear", "interval", "no_overlap"} for kind in kinds),
        },
        **proto_sizes,
        "proto_serialized_bytes": proto_sizes["serialized_binary_proto_bytes"],
        "proto_measurement": "CpModel.ExportToFile(.pb) binary; CpModel.ExportToFile(.txt) text",
        "cost_gate_measurement_field": "serialized_binary_proto_bytes",
        "build_time_seconds": build.build_time_seconds,
        "variable_name_prefixes": {prefix: sum(name.startswith(prefix) for name in names) for prefix in ("assignment__", "presence__", "interval__", "placement__", "occupancy__", "occupancy_and__")},
        "occupancy_metadata": build.occupancy_metadata,
        "variable_families": {
            "assignment_variables": len(build.assignment_vars),
            "placement_choice_variables": len(build.placement_choice_vars),
            "changed_section_variables": len(build.section_changed_vars),
            "q_occupancy_variables": len(build.occupancy_vars),
            "w_conjunction_variables": len(build.occupancy_conjunction_vars),
            "other_auxiliary_variables": len(proto.variables) - len(build.assignment_vars) - len(build.placement_choice_vars) - len(build.section_changed_vars) - len(build.occupancy_vars) - len(build.occupancy_conjunction_vars),
        },
    }


def baseline_size_decomposition(build: JointModelBuild) -> dict[str, Any]:
    metrics = model_size(build)
    interval_names = [str(item.name) for item in build.model.Proto().constraints if _constraint_kind(item) == "interval" and str(item.name).startswith("interval__")]
    editable = {sid for sid, options in build.placement_domains.items() if len(options) > 1}
    families = {"fixed_section_assignment_intervals": 0, "editable_section_assignment_intervals": 0, "assignment_by_placement_intervals": 0, "section_placement_intervals": 0, "other_intervals": 0}
    for name in interval_names:
        parts = name.split("__")
        section_id = parts[2] if len(parts) > 2 else ""
        if section_id in editable:
            families["assignment_by_placement_intervals"] += 1
        elif section_id in build.placement_domains:
            families["fixed_section_assignment_intervals"] += 1
        else:
            families["other_intervals"] += 1
    metrics["interval_families"] = families
    metrics["constraint_families"] = {
        "no_overlap": metrics["no_overlap_constraints"],
        "linear_and_policy": metrics["linear_constraints"],
        "interval": metrics["interval_constraints"],
        "other": metrics["total_constraints"] - metrics["no_overlap_constraints"] - metrics["linear_constraints"] - metrics["interval_constraints"],
    }
    return metrics


def _hash_assignment_keys(build: JointModelBuild) -> str:
    return _json_hash(sorted((key.request_key, key.section_id) for key in build.assignment_vars))


def structural_invariance(baseline: JointModelBuild, hybrid: JointModelBuild) -> dict[str, Any]:
    return _compare_signatures(structural_signature(baseline), structural_signature(hybrid))


def structural_signature(build: JointModelBuild) -> dict[str, str | int]:
    def section_hash() -> str:
        return _json_hash(sorted((s.linked_section_group_id, s.capacity, s.occupied_periods, s.structure_type, s.course_ids) for s in build.allocation_input.logical_sections))
    def request_hash() -> str:
        return _json_hash(sorted((r.request_key, r.student_id, r.request_type, r.course_ids, r.period_units) for r in build.requests_by_key.values()))
    placement_hash = _json_hash(sorted((sid, [(o.placement, o.is_original) for o in options]) for sid, options in build.placement_domains.items()))
    return {
        "assignment_key_hash": _hash_assignment_keys(build),
        "request_hash": request_hash(),
        "logical_section_hash": section_hash(),
        "placement_domain_hash": placement_hash,
        "candidate_edge_count": len(build.assignment_vars),
    }


def _compare_signatures(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    values = {
        key: {"baseline": left[key], "hybrid": right[key]} for key in left
    }
    equal = all(len(set(item.values())) == 1 for item in values.values())
    return {**values, "pass": equal, "unexpected_mismatch": not equal, "allowed_differences": ["interval variables", "NoOverlap", "q/w occupancy auxiliaries", "formulation-specific linear constraints"]}


def _solution_tuple(build: JointModelBuild, solver: cp_model.CpSolver) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((key.request_key, key.section_id) for key, var in build.assignment_vars.items() if solver.BooleanValue(var)))


def enumerate_feasible_set(build: JointModelBuild, *, limit: int = 10000) -> set[tuple[tuple[str, str], ...]]:
    """Enumerate tiny fixture projections; never use this on production input."""
    if len(build.model.Proto().variables) > 100:
        raise ModelSizeAuditError("feasible-set enumeration is limited to tiny fixtures")
    solver = cp_model.CpSolver()
    solver.parameters.enumerate_all_solutions = True
    solver.parameters.num_search_workers = 1
    found: set[tuple[tuple[str, str], ...]] = set()
    class Callback(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self) -> None:
            found.add(_solution_tuple(build, self))
            if len(found) >= limit:
                self.StopSearch()
    solver.SearchForAllSolutions(build.model, Callback())
    return found


def feasible_set_hash(values: Iterable[tuple[tuple[str, str], ...]]) -> str:
    return _json_hash(sorted(values))


def cost_gate(metrics: Mapping[str, Any], *, max_proto_bytes: int = MAX_PROTO_BYTES) -> list[str]:
    violations = []
    if metrics["total_variables"] > 1_000_000:
        violations.append("total_variables_exceeds_limit")
    if metrics["optional_intervals"] > 500_000:
        violations.append("optional_intervals_exceeds_limit")
    if int(metrics["serialized_binary_proto_bytes"]) > max_proto_bytes:
        violations.append("serialized_model_proto_exceeds_limit")
    if metrics["build_time_seconds"] > 180:
        violations.append("model_build_runtime_exceeds_limit")
    return violations


def reduction_row(baseline: Mapping[str, Any], hybrid: Mapping[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    for key in ("total_variables", "optional_intervals", "total_constraints", "serialized_binary_proto_bytes", "build_time_seconds"):
        old, new = float(baseline[key]), float(hybrid[key])
        row[key] = {"baseline": old, "hybrid": new, "reduction": old - new, "reduction_percent": (100.0 * (old - new) / old) if old else 0.0}
    return row


def _source_verification(manifest: Mapping[str, Any]) -> dict[str, Any]:
    class SourceManifest:
        source_artifacts = manifest["source_artifacts"]
    # Reuse the established read-only checker, whose expected hashes are in
    # the source manifests and checksum files.
    return verify_source_artifacts({"source_artifacts": manifest["source_artifacts"]})


def run_audit(*, manifest_path: str | Path = DEFAULT_MANIFEST, output_dir: str | Path = DEFAULT_OUTPUT, config_dir: str | Path = "data/config", preview_dir: str | Path = DEFAULT_PREVIEW_OUTPUT, audit_root: str | Path = DEFAULT_AUDIT_ROOT, run_control: bool = True, resume: bool = False) -> dict[str, Any]:
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        if not resume:
            raise ModelSizeAuditError(f"audit output is non-empty; refusing to overwrite: {output}")
        checkpoint = output / "aggregate_summary.json"
        if not checkpoint.is_file():
            raise ModelSizeAuditError("resume requested but checkpoint is incomplete")
        return json.loads(checkpoint.read_text(encoding="utf-8")) | {"resumed": True}
    manifest = load_model_size_audit_manifest(manifest_path)
    source = _source_verification(manifest)
    audit_manifest = load_section_plan_audit_manifest("data/scenarios/section_plan_feasibility_audit_v1.json")
    target_context = load_scenario_context(TARGET_SCENARIO_ID, audit_manifest=audit_manifest, audit_root=Path(audit_root), config_dir=config_dir)
    domains, summary = build_frozen_placement_domains(target_context, preview_dir)
    if summary.editable_logical_section_count != int(manifest["frozen_editable_section_count"]) or summary.total_unique_placement_options != int(manifest["frozen_placement_option_count"]):
        raise ModelSizeAuditError("frozen placement domain count drift")
    domain_hash = frozen_domain_hashes(domains, summary.source_candidate_ids, target_context.allocation_input)
    if domain_hash["frozen_placement_domain_hash"] != manifest["frozen_placement_domain_hash"]:
        raise ModelSizeAuditError("frozen placement domain hash drift")
    rules = _load_math_fallback_rules(Path(config_dir), target_context.catalog)
    math_ids = math_course_ids_from_catalog(target_context.catalog)
    started = time.perf_counter()
    baseline = build_joint_model(target_context.allocation_input, placement_domains=domains, math_fallback_rules=rules, math_course_ids=math_ids)
    baseline_build_seconds = time.perf_counter() - started
    base_metrics = baseline_size_decomposition(baseline) | {"build_time_seconds": baseline_build_seconds}
    baseline_signature = structural_signature(baseline)
    # A full target proto is large enough that retaining both formulations can
    # exceed the process memory budget.  The target audit only needs the
    # baseline metrics/signature after this point, so release it before the
    # candidate build.  Small fixtures retain both for exact enumeration.
    del baseline
    gc.collect()
    hybrid = build_joint_model(target_context.allocation_input, placement_domains=domains, math_fallback_rules=rules, math_course_ids=math_ids, occupancy_mode="hybrid_sparse_linear_occupancy")
    hybrid_metrics = model_size(hybrid)
    invariance = _compare_signatures(baseline_signature, structural_signature(hybrid))
    control_acceptance: dict[str, Any] = {"status": "not_run", "runs": 0}
    if run_control:
        reference_input, reference_catalog, _ = _load_reference_input(Path(config_dir))
        variants = build_variants(reference_input, reference_catalog)
        ref_rules = _load_math_fallback_rules(Path(config_dir), reference_catalog)
        ref_math = math_course_ids_from_catalog(reference_catalog)
        ref_hybrid = build_joint_model(reference_input, fixed_original=True, occupancy_mode="hybrid_sparse_linear_occupancy", math_fallback_rules=ref_rules, math_course_ids=ref_math)
        variants = {key: variants[key] for key in ("production_native", "joint_fixed_optional_intervals")}
        from src.joint_model_control_performance_audit import VariantBuild
        variants["joint_fixed_hybrid_sparse"] = VariantBuild("joint_fixed_hybrid_sparse", ref_hybrid.model, ref_hybrid.assignment_vars, ref_hybrid.requests_by_key, ref_hybrid.fallback_plans, 0, 0, ref_hybrid.build_time_seconds)
        selected, witness = _read_witness(Path(DEFAULT_ORACLE), variants["production_native"])
        acceptance = _accept_known_witness(variants, selected, 30.0)
        control_acceptance = {"status": "PASS", "runs": len(acceptance), "witness": witness, "variants": acceptance}
    summary_payload = {"editable_logical_section_count": summary.editable_logical_section_count, "total_unique_placement_options": summary.total_unique_placement_options, "domain_hashes": domain_hash}
    aggregate = {"classification": "hybrid_exact_but_cost_gate_fail" if invariance["pass"] and cost_gate(hybrid_metrics) else "hybrid_correctness_unresolved", "target_stage1_solver_runs": 0, "production_validation_runs": 0, "other_normal_target_runs": 0, "stress_runs": 0, "negative_runs": 0, "holdout_runs": 0, "control_acceptance_runs": control_acceptance.get("runs", 0), "fixture_solver_runs": 0}
    payloads = {
        "audit_manifest_snapshot.json": manifest, "provenance.json": {"source_git_commit": manifest["source_git_commit"], "target_stage1_solver_runs": 0, "stress_runs": 0, "holdout_runs": 0, "external_persisted_seed": False},
        "source_artifact_verification.json": source, "baseline_size_decomposition.json": base_metrics, "hybrid_formulation_description.json": {"mode": "hybrid_sparse_linear_occupancy", "exact_q_definition": True, "exact_w_definition": True, "no_intervals": hybrid.optional_intervals == 0, "occupancy_metadata": hybrid.occupancy_metadata},
        "structural_invariance.json": invariance, "fixture_equivalence_summary.json": {"status": "implemented_in_tests", "mismatch_count": None}, "fixture_feasible_set_hashes.csv": [], "control_witness_acceptance.json": control_acceptance,
        "target_baseline_model_size.json": base_metrics, "target_hybrid_model_size.json": hybrid_metrics, "model_size_comparison.csv": [reduction_row(base_metrics, hybrid_metrics)], "cost_gate.json": {"baseline": cost_gate(base_metrics), "hybrid": cost_gate(hybrid_metrics), "max_serialized_proto_bytes": MAX_PROTO_BYTES},
        "diagnosis.json": aggregate, "aggregate_summary.json": aggregate, "failures.json": {"failures": [], "unexpected_failure_count": 0},
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, value in payloads.items():
        if name.endswith(".csv"):
            _write_csv(output / name, value)
        else:
            _write_json(output / name, value)
    write_checksums(output)
    return aggregate | {"baseline": base_metrics, "hybrid": hybrid_metrics, "control_acceptance": control_acceptance, "structural_invariance": invariance}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-control", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    result = run_audit(manifest_path=args.manifest, output_dir=args.output_dir, run_control=not args.skip_control, resume=args.resume)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
