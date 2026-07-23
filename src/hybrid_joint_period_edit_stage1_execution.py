"""Execute the single frozen hybrid Stage 1 period-edit target.

Only ``normal_dev_10`` is allowed.  Stage 1 is invoked at most once per
artifact directory; later stages are deliberately outside this runner's API.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

from src.allocation import math_course_ids_from_catalog
from src.benchmark_runner import _load_math_fallback_rules
from src.joint_period_edit_pilot import (
    build_frozen_placement_domains,
    build_joint_model,
    _json_hash,
)
from src.joint_period_edit_stage1_pilot import (
    DEFAULT_PREVIEW_OUTPUT,
    STAGE1_BUDGET_SECONDS,
    FIXED_WITNESS_BUDGET_SECONDS,
    PRODUCTION_BUDGET_SECONDS,
    TARGET_SCENARIO_ID,
    apply_stage1_hints,
    independent_production_validation,
    load_stage1_manifest,
    production_fixed_witness_acceptance,
    solve_stage1,
    validate_joint_witness,
    verify_checksums,
    frozen_domain_hashes,
)
from src.joint_stage1_model_size_reduction_audit import (
    DEFAULT_MANIFEST as SIZE_MANIFEST,
    MAX_PROTO_BYTES,
    cost_gate,
    model_size,
)
from src.period_placement_repair_probe import DEFAULT_AUDIT_ROOT, load_scenario_context
from src.section_plan_feasibility_audit import load_section_plan_audit_manifest


DEFAULT_MANIFEST = Path("data/scenarios/hybrid_joint_period_edit_stage1_execution_v1.json")
DEFAULT_OUTPUT = Path("/Users/klu/Projects/fair-course-allocation-artifacts/robustness-v1/hybrid-joint-period-edit-stage1-execution-v1")
DEFAULT_SIZE_ARTIFACT = Path("/Users/klu/Projects/fair-course-allocation-artifacts/robustness-v1/joint-stage1-model-size-reduction-audit-v1")


class HybridExecutionError(ValueError):
    """Raised when execution provenance or a frozen invariant fails closed."""


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_checksums(root: Path) -> str:
    import hashlib
    checksum = root / "SHA256SUMS.txt"
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != checksum:
            lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root)}")
    checksum.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return hashlib.sha256(checksum.read_bytes()).hexdigest()


def load_execution_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    payload = _read_json(Path(path))
    if payload.get("target_scenario_id") != TARGET_SCENARIO_ID:
        raise HybridExecutionError("only normal_dev_10 is allowed")
    if payload.get("formulation") != "hybrid_sparse_linear_occupancy":
        raise HybridExecutionError("hybrid formulation is not frozen")
    forbidden = ("stage2_allowed", "stage3_allowed", "stage4_allowed", "control_runs_allowed", "other_normal_targets_allowed", "stress_execution_allowed", "negative_execution_allowed", "holdout_execution_allowed", "external_persisted_seed")
    if any(payload.get(field) is not False for field in forbidden):
        raise HybridExecutionError("execution manifest enables a forbidden run")
    if int(payload.get("solver_seed")) != 20260630 or int(payload.get("workers")) != 1:
        raise HybridExecutionError("solver seed or worker count drift")
    return payload


def verify_frozen_domain(manifest: Mapping[str, Any], context: Any, domains: Mapping[str, Any], summary: Any) -> dict[str, Any]:
    if summary.editable_logical_section_count != int(manifest["editable_section_count"]):
        raise HybridExecutionError("editable section count drift")
    if summary.total_unique_placement_options != int(manifest["placement_option_count"]):
        raise HybridExecutionError("placement option count drift")
    if context.authoritative_core is None or context.authoritative_core.student_id != manifest["authoritative_student_id"]:
        raise HybridExecutionError("authoritative student drift")
    hashes = frozen_domain_hashes(domains, summary.source_candidate_ids, context.allocation_input)
    for key in ("frozen_placement_domain_hash", "section_domain_mapping_hash", "original_placement_hash"):
        if hashes.get(key) != manifest[key]:
            raise HybridExecutionError(f"frozen domain hash drift: {key}")
    if len(context.allocation_input.candidate_index) == 0:
        raise HybridExecutionError("candidate index is empty")
    candidate_edges = sum(len(values) for values in context.allocation_input.candidate_index.values())
    # The audit count is assignment-variable edges after fallback expansion;
    # the runner verifies it after build, where that universe is available.
    return {"counts": {"editable_sections": summary.editable_logical_section_count, "placement_options": summary.total_unique_placement_options}, "hashes": hashes, "candidate_edges_from_input": candidate_edges, "candidate_pruning": False, "authoritative_student_id": manifest["authoritative_student_id"], "excluded_student_ids": list(manifest["excluded_student_ids"])}


def _stage1_config(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {"solver_seed": manifest["solver_seed"], "workers": manifest["workers"], "max_time_in_seconds": manifest["stage1_budget_seconds"], "objective": "minimize_sum_section_changed", "external_persisted_seed": False, "repair_hint_enabled": False, "stop_after_first_complete_solution": False, "stage2_allowed": False, "stage3_allowed": False, "stage4_allowed": False}


def _not_run(reason: str) -> dict[str, Any]:
    return {"status": "not_run", "assignment_available": False, "not_run_reason": reason, "response_hash": ""}


def run_execution(*, manifest_path: str | Path = DEFAULT_MANIFEST, output_dir: str | Path = DEFAULT_OUTPUT, size_artifact: str | Path = DEFAULT_SIZE_ARTIFACT, preview_dir: str | Path = DEFAULT_PREVIEW_OUTPUT, audit_root: str | Path = DEFAULT_AUDIT_ROOT, config_dir: str | Path = "data/config", resume: bool = False) -> dict[str, Any]:
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        if not resume:
            raise HybridExecutionError(f"execution output is non-empty; refusing overwrite: {output}")
        aggregate = output / "aggregate_summary.json"
        if not aggregate.is_file():
            raise HybridExecutionError("resume requested without completed aggregate checkpoint")
        return _read_json(aggregate) | {"resumed": True, "stage1_reexecuted": False}
    manifest = load_execution_manifest(manifest_path)
    size_root = Path(size_artifact)
    size_check = verify_checksums(size_root)
    if not size_check["passed"]:
        raise HybridExecutionError("hybrid size audit checksum failed")
    manifest_size_hash = load_execution_manifest(manifest_path).get("source_hybrid_audit_hash")
    if manifest_size_hash and size_check.get("sha256") != manifest_size_hash:
        raise HybridExecutionError("hybrid size audit source hash mismatch")
    section_manifest = load_section_plan_audit_manifest("data/scenarios/section_plan_feasibility_audit_v1.json")
    context = load_scenario_context(TARGET_SCENARIO_ID, audit_manifest=section_manifest, audit_root=Path(audit_root), config_dir=config_dir)
    domains, domain_summary = build_frozen_placement_domains(context, preview_dir)
    frozen = verify_frozen_domain(manifest, context, domains, domain_summary)
    rules = _load_math_fallback_rules(Path(config_dir), context.catalog)
    math_ids = math_course_ids_from_catalog(context.catalog)
    build = build_joint_model(context.allocation_input, placement_domains=domains, math_fallback_rules=rules, math_course_ids=math_ids, occupancy_mode="hybrid_sparse_linear_occupancy")
    size = model_size(build)
    candidate_edges = len(build.assignment_vars)
    if candidate_edges != int(manifest["candidate_edge_count"]):
        raise HybridExecutionError("assignment candidate edge count drift")
    gate = cost_gate(size, max_proto_bytes=MAX_PROTO_BYTES)
    failures: list[str] = []
    hint_audit: dict[str, Any] = {"status": "not_run"}
    stage = _not_run("cost_gate" if gate else "not_run")
    witness = _not_run("no_stage1_incumbent")
    acceptance = _not_run("no_joint_witness")
    production = _not_run("fixed_witness_acceptance_not_passed")
    solver_log: tuple[str, ...] = ()
    if not gate:
        from src.joint_period_edit_stage1_pilot import _selected_keys
        baseline = __import__("src.allocation", fromlist=["run_constrained_first_baseline"]).run_constrained_first_baseline(context.allocation_input, int(manifest["solver_seed"]), math_fallback_rules=rules, math_course_ids=math_ids)
        selected = _selected_keys((item.request_key, item.linked_section_group_id) for item in baseline.assignments)
        hint_audit_obj = apply_stage1_hints(build, selected)
        hint_audit = asdict(hint_audit_obj)
        run = solve_stage1(build, seed=int(manifest["solver_seed"]), time_limit_seconds=float(manifest["stage1_budget_seconds"]))
        stage = asdict(run)
        solver_log = tuple(stage.pop("solver_log", ()) or ())
        stage["hint_source"] = "constrained_first_internal"
        stage["repair_hint_enabled"] = False
        if run.objective_value == 0:
            failures.append("objective_zero_contradicts_known_normal_dev_10_infeasibility")
            witness = {"status": "correctness_failure", "joint_stage1_witness_valid": False, "not_run_reason": "objective_zero"}
        elif run.incumbent_found:
            witness = validate_joint_witness(build, run)
            if witness.get("joint_stage1_witness_valid"):
                placement_map = dict(run.selected_placements)
                acceptance = production_fixed_witness_acceptance(context, placement_map, run.selected_assignments, config_dir=Path(config_dir), seed=int(manifest["solver_seed"]), time_limit_seconds=float(manifest["fixed_witness_acceptance_budget_seconds"]))
                if acceptance.get("status") in {"FEASIBLE", "OPTIMAL"} and acceptance.get("assignment_exact") and acceptance.get("policy_pass") and acceptance.get("consistency_issue_count") == 0:
                    production = independent_production_validation(context, placement_map, config_dir=Path(config_dir), seed=int(manifest["solver_seed"]), time_limit_seconds=float(manifest["production_validation_budget_seconds"]))
    aggregate = {
        "experiment_name": manifest["experiment_name"], "phase": manifest["phase"], "target_scenario_id": TARGET_SCENARIO_ID,
        "stage1_runs": 1 if stage.get("status") not in {"not_run", "SKIPPED"} else 0,
        "fixed_witness_acceptance_runs": 1 if acceptance.get("status") not in {"not_run", "SKIPPED"} else 0,
        "production_validation_runs": 1 if production.get("status") not in {"not_run", "SKIPPED"} else 0,
        "control_runs": 0, "stage2_runs": 0, "stage3_runs": 0, "stage4_runs": 0, "other_normal_runs": 0, "stress_runs": 0, "negative_runs": 0, "holdout_runs": 0,
        "stage1_status": stage.get("status", "not_run"), "joint_witness_valid": witness.get("joint_stage1_witness_valid", False),
        "independently_validated_period_repair": production.get("independently_validated_period_repair", False), "minimum_claim": "unresolved" if stage.get("status") != "OPTIMAL" else "requires_production_gates",
        "failures": failures, "external_persisted_seed": False,
    }
    provenance = {"source_git_commit": manifest["source_git_commit"], "source_artifact_verification": size_check, "stage1_runs": aggregate["stage1_runs"], "fixed_witness_acceptance_runs": aggregate["fixed_witness_acceptance_runs"], "production_validation_runs": aggregate["production_validation_runs"], "control_runs": 0, "stage2_runs": 0, "stage3_runs": 0, "stage4_runs": 0, "other_normal_runs": 0, "stress_runs": 0, "negative_runs": 0, "holdout_runs": 0, "external_persisted_seed": False}
    payloads = {
        "execution_manifest_snapshot.json": manifest, "provenance.json": provenance, "source_artifact_verification.json": size_check, "frozen_domain_verification.json": frozen, "hybrid_model_size.json": size, "hint_audit.json": hint_audit, "stage1_solver_config.json": _stage1_config(manifest), "stage1_response_stats.json": stage, "stage1_witness.json": witness, "joint_witness_validation.json": witness, "production_fixed_witness_acceptance.json": acceptance, "production_cold_start_validation.json": production, "aggregate_summary.json": aggregate, "failures.json": {"failures": failures, "unexpected_failure_count": len(failures)},
    }
    output.mkdir(parents=True, exist_ok=True)
    for name, value in payloads.items():
        _write_json(output / name, value)
    (output / "stage1_solver.log").write_text("\n".join(solver_log), encoding="utf-8")
    _write_json(output / "aggregate_summary.json", aggregate)
    write_checksums(output)
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--size-artifact", type=Path, default=DEFAULT_SIZE_ARTIFACT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run_execution(manifest_path=args.manifest, output_dir=args.output_dir, size_artifact=args.size_artifact, resume=args.resume), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
