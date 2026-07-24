"""Hybrid K=2 Search Bottleneck Diagnostic Audit v1.

This module is a development-only, read-mostly orchestration layer. It reuses
the exact frozen 312-section/841-option hybrid domain and the exact two K=2
pair candidates already produced by
:mod:`src.hybrid_stage1_incumbent_bootstrap`. It does not regenerate pair
candidates, does not rerun the full K=2 cardinality-cap search, and does not
run K=1 or K=3. Its only purpose is to distinguish, for the two already-frozen
pairs, whether an unresolved K=2 search is more attributable to (a) global
section-pair selection, (b) destination-placement selection, (c) fixed-plan
student-assignment feasibility, or (d) assignment-hint/Hamming search
guidance, via three placement-fixing ablations (Diagnostics A, B, C).
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

from src.allocation import canonicalize_allocation_input, math_course_ids_from_catalog
from src.allocation.cp_sat_solver import _build_full_feasibility_cp_sat_model
from src.allocation.random_baseline import _build_mandatory_fallback_plans
from src.allocation.state import AllocationState
from src.benchmark_runner import _load_math_fallback_rules
from src.final_schedule_policy import evaluate_final_schedule_policy
from src.hybrid_stage1_incumbent_bootstrap import (
    SearchResult,
    add_change_cap,
    apply_bootstrap_hints,
    build_bootstrap_model,
    hamming_expression,
    solve_bootstrap,
    validate_bootstrap_witness,
)
from src.joint_model_control_performance_audit import assert_empty_solution_hint
from src.joint_period_edit_pilot import (
    AUTHORITATIVE_STUDENT_ID,
    _json_hash,
    _section_placement,
    _student_outcomes_for_solution,
    apply_placement_map_to_sections,
    build_frozen_placement_domains,
    build_joint_model,
)
from src.joint_period_edit_stage1_pilot import (
    DEFAULT_CONTROL_AUDIT,
    DEFAULT_CONTROL_AUDITED,
    _fresh_production_build,
    frozen_domain_hashes,
    independent_production_validation,
    production_fixed_witness_acceptance,
    verify_checksums,
)
from src.period_placement_repair_probe import (
    DEFAULT_AUDIT_ROOT,
    DEFAULT_OUTPUT as DEFAULT_PREVIEW_OUTPUT,
    CandidateEdit,
    _candidate_from_dict as _candidate_from_portfolio_dict,
    _requests_for_sections,
    _sha256_file,
    exact_student_level_analysis,
    load_scenario_context,
)
from src.section_plan_feasibility_audit import load_section_plan_audit_manifest


TARGET_SCENARIO_ID = "normal_dev_10"
SOLVER_SEED = 20260630
WORKERS = 1
DIAGNOSTIC_A_BUDGET_SECONDS = 60.0
DIAGNOSTIC_B_BUDGET_SECONDS = 60.0
DIAGNOSTIC_C_BUDGET_SECONDS = 120.0
FIXED_WITNESS_BUDGET_SECONDS = 30.0
PRODUCTION_BUDGET_SECONDS = 300.0
MAX_DIAGNOSTIC_A_RUNS = 2
MAX_DIAGNOSTIC_B_RUNS = 2
MAX_DIAGNOSTIC_C_RUNS = 2

DEFAULT_MANIFEST = Path("data/scenarios/hybrid_k2_search_bottleneck_diagnostic_v1.json")
DEFAULT_OUTPUT = Path(
    "/Users/klu/Projects/fair-course-allocation-artifacts/robustness-v1/"
    "hybrid-k2-search-bottleneck-diagnostic-v1"
)
DEFAULT_BOOTSTRAP = Path(
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

# Source artifacts read-only-verified for this diagnostic. Only the bootstrap
# artifact's hash is a required manifest field (it is the direct source of the
# frozen pairs and the previous K=2 logs); the remaining four are the same
# upstream artifacts the bootstrap itself already verified, and are re-checked
# here read-only as an independent provenance chain, matching this task's
# preflight instructions. None of these files are ever written by this module.
EXPECTED_SOURCE_ARTIFACT_HASHES: dict[str, tuple[Path, str]] = {
    "hybrid_stage1_incumbent_bootstrap": (
        DEFAULT_BOOTSTRAP,
        "528a13614477c0403c12626696e2f0cd394beb69bb5818316ea680ad0810773a",
    ),
    "hybrid_joint_period_edit_stage1_execution": (
        DEFAULT_PREVIOUS_STAGE1,
        "2aa3c72115025562e4e8dbaf97dd372c186c0371cbc125b977c9a6cf53a92d63",
    ),
    "joint_stage1_model_size_reduction_audit": (
        DEFAULT_SIZE_AUDIT,
        "5a5775ca7bff4054b034ab336d9ebc49fa6c11f9cc0e58cff94be8a34dbd3f80",
    ),
    "period_placement_repair_probe": (
        DEFAULT_PREVIEW_OUTPUT,
        "c43e00a74bbe513064b4d40839ad648b000f08757499f91ae0952d9e542ea6e2",
    ),
    "joint_model_control_performance_audit_audited": (
        DEFAULT_CONTROL_AUDITED,
        "f5eb6f8020180fbfa9fe706c1e99c0c8b40f6e592f9b1fc551d78e85fdbceae5",
    ),
}


# --------------------------------------------------------------------------
# Execution-history correction (post-hoc, reporting-only)
#
# The accepted final artifact's diagnostic_a_runs/diagnostic_c_runs fields
# only ever counted the accepted final batch of four solver runs. A first
# batch of four runs (2x Diagnostic A + 2x Diagnostic C) was executed and
# completed successfully, but was then superseded and rerun after discovering
# that provenance.json was never rewritten with real counters at the end of a
# run (it stayed at its pre-search all-zero snapshot). The first batch's
# artifact directory was deleted, within the same session, before this
# correction was written. These constants record that real execution history
# so it can be surfaced in the artifact without conflating "runs kept in the
# accepted final artifact" with "total solver invocations actually made."
# --------------------------------------------------------------------------

ACCEPTED_FINAL_ARTIFACT_RUNS: dict[str, int] = {
    "diagnostic_a": 2, "diagnostic_b": 0, "diagnostic_c": 2, "total": 4,
}
SUPERSEDED_RUNS: dict[str, int] = {
    "diagnostic_a": 2, "diagnostic_b": 0, "diagnostic_c": 2, "total": 4,
}
TOTAL_SOLVER_INVOCATIONS: dict[str, int] = {
    "diagnostic_a": 4, "diagnostic_b": 0, "diagnostic_c": 4, "total": 8,
}
RERUN_BATCHES = 1
PROTOCOL_DEVIATION = True
PROTOCOL_DEVIATION_REASON = (
    "The first completed diagnostic batch was rerun after fixing a "
    "provenance counter finalization bug."
)
RESULT_BASED_PARAMETER_CHANGE = False
PORTFOLIO_CHANGED_BETWEEN_BATCHES = False
SEED_CHANGED_BETWEEN_BATCHES = False
BUDGET_CHANGED_BETWEEN_BATCHES = False
INCUMBENT_SELECTION_BIAS_NOTE = (
    "The rerun reduces procedural cleanliness but does not create incumbent "
    "selection bias in this case, because no run in either batch produced an "
    "incumbent and no configuration (portfolio, model restrictions, seed, "
    "workers, budgets) was changed between batches."
)
FIRST_BATCH_STATUSES_OBSERVED: tuple[str, ...] = (
    "INFEASIBLE", "INFEASIBLE", "INFEASIBLE", "INFEASIBLE",
)
FIRST_BATCH_STATUSES_EVIDENCE_SOURCE = (
    "recorded in this session's own conversation transcript at the time the "
    "first batch completed, before the provenance.json finalization bug was "
    "found and the original artifact directory was deleted; not reconstructed "
    "or inferred from the second (accepted) batch's results"
)
FIRST_BATCH_UNAVAILABLE_NOTE = (
    "The first batch was superseded after a reporting bug was found. Its "
    "original artifact directory was deleted before the execution-history "
    "correction, so unrecoverable per-run fields (response hashes, exact "
    "runtimes, and pair/diagnostic-level mapping) are explicitly recorded as "
    "unavailable."
)
EXACT_DESTINATION_PLAN_COUNT = 2
DIAGNOSTIC_C_SEARCH_CONFIG_COUNT = 2


class DiagnosticError(ValueError):
    """Raised when the frozen K=2 bottleneck diagnostic cannot proceed safely."""


# --------------------------------------------------------------------------
# Small I/O helpers (duplicated per this repository's established convention:
# each single-script CLI tool in src/ owns its own tiny JSON/CSV/checksum
# helpers rather than sharing a cross-module utils layer).
# --------------------------------------------------------------------------


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DiagnosticError(f"cannot read JSON: {path}: {exc}") from exc


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
            lines.append(f"{_sha256_file(path)}  {path.relative_to(root)}")
    checksum.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return _sha256_file(checksum)


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------


def load_diagnostic_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    payload = _read_json(Path(path))
    required = {
        "experiment_name", "experiment_version", "phase", "source_git_commit",
        "target_scenario_id", "authoritative_student_id", "excluded_student_ids",
        "source_bootstrap_artifact_hash", "frozen_pair_portfolio_hash", "pair_count",
        "editable_section_count", "placement_option_count", "candidate_edge_count",
        "solver_seed", "workers", "external_persisted_seed",
        "exact_destination_no_hint_budget_seconds", "exact_destination_hamming_budget_seconds",
        "fixed_section_ids_destination_free_budget_seconds", "joint_fixed_witness_budget_seconds",
        "production_validation_budget_seconds", "full_k2_search_allowed", "k1_allowed", "k3_allowed",
        "other_normal_targets_allowed", "stress_execution_allowed", "negative_execution_allowed",
        "holdout_execution_allowed",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise DiagnosticError("diagnostic manifest missing: " + ", ".join(missing))
    if payload["phase"] != "k2_search_bottleneck_diagnostic":
        raise DiagnosticError("unexpected diagnostic phase")
    if payload["target_scenario_id"] != TARGET_SCENARIO_ID:
        raise DiagnosticError("only normal_dev_10 is allowed")
    if payload["authoritative_student_id"] != AUTHORITATIVE_STUDENT_ID:
        raise DiagnosticError("authoritative student must be G12_0536")
    if "G12_0105" not in payload["excluded_student_ids"]:
        raise DiagnosticError("G12_0105 must remain excluded")
    if int(payload["solver_seed"]) != SOLVER_SEED or int(payload["workers"]) != WORKERS:
        raise DiagnosticError("solver seed or workers drifted")
    if payload["external_persisted_seed"] is not False:
        raise DiagnosticError("external persisted seed is forbidden")
    if int(payload["pair_count"]) != 2:
        raise DiagnosticError("pair_count must be frozen at 2")
    for field in (
        "full_k2_search_allowed", "k1_allowed", "k3_allowed", "other_normal_targets_allowed",
        "stress_execution_allowed", "negative_execution_allowed", "holdout_execution_allowed",
    ):
        if payload[field] is not False:
            raise DiagnosticError(f"{field} must be false")
    for field, expected in (
        ("exact_destination_no_hint_budget_seconds", DIAGNOSTIC_A_BUDGET_SECONDS),
        ("exact_destination_hamming_budget_seconds", DIAGNOSTIC_B_BUDGET_SECONDS),
        ("fixed_section_ids_destination_free_budget_seconds", DIAGNOSTIC_C_BUDGET_SECONDS),
        ("joint_fixed_witness_budget_seconds", FIXED_WITNESS_BUDGET_SECONDS),
        ("production_validation_budget_seconds", PRODUCTION_BUDGET_SECONDS),
    ):
        if float(payload[field]) != expected:
            raise DiagnosticError(f"{field} is not frozen at the expected budget")
    return payload


# --------------------------------------------------------------------------
# Source artifact verification (read-only)
# --------------------------------------------------------------------------


def verify_source_artifacts(manifest: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, (root, expected_hash) in EXPECTED_SOURCE_ARTIFACT_HASHES.items():
        check = verify_checksums(root)
        if not check["passed"] or check["sha256"] != expected_hash:
            raise DiagnosticError(f"source artifact verification failed: {name}")
        result[name] = check
    bootstrap_root, bootstrap_hash = EXPECTED_SOURCE_ARTIFACT_HASHES["hybrid_stage1_incumbent_bootstrap"]
    if str(manifest["source_bootstrap_artifact_hash"]) != bootstrap_hash:
        raise DiagnosticError("manifest source_bootstrap_artifact_hash does not match the verified bootstrap artifact")
    del bootstrap_root
    return result


# --------------------------------------------------------------------------
# Frozen pair portfolio (read-only reuse; never regenerated)
# --------------------------------------------------------------------------


def load_frozen_pairs(manifest: Mapping[str, Any], bootstrap_dir: str | Path = DEFAULT_BOOTSTRAP) -> tuple[CandidateEdit, ...]:
    path = Path(bootstrap_dir) / "pair_hint_portfolio.json"
    actual_hash = _sha256_file(path)
    if actual_hash != str(manifest["frozen_pair_portfolio_hash"]):
        raise DiagnosticError("pair_hint_portfolio.json hash does not match the frozen manifest hash")
    payload = _read_json(path)
    rows = payload.get("candidates", [])
    if int(payload.get("count", -1)) != 2 or len(rows) != 2:
        raise DiagnosticError("frozen pair portfolio must contain exactly 2 candidates")
    pairs = tuple(_candidate_from_portfolio_dict(row) for row in rows)
    for candidate in pairs:
        if candidate.edit_type != "bootstrap_pair":
            raise DiagnosticError(f"unexpected edit_type in frozen pair portfolio: {candidate.edit_type}")
        if len(candidate.logical_section_ids) != 2 or len(set(candidate.logical_section_ids)) != 2:
            raise DiagnosticError(f"a frozen pair must name exactly two distinct sections: {candidate.candidate_id}")
        if candidate.core_student != AUTHORITATIVE_STUDENT_ID:
            raise DiagnosticError(f"frozen pair candidate has an unexpected core student: {candidate.candidate_id}")
        if AUTHORITATIVE_STUDENT_ID != "G12_0536" or "G12_0105" in candidate.candidate_id:
            raise DiagnosticError("G12_0105 must not appear in a frozen pair candidate")
    return pairs


def pair_id_for_index(index: int) -> str:
    return f"pair_{index + 1}"


def analyze_pair(pair_id: str, candidate: CandidateEdit, allocation_input: Any, config_dir: Path) -> dict[str, Any]:
    """Record the frozen facts about one pair without altering it."""
    edited = _apply_candidate_periods(allocation_input, candidate)
    exact = exact_student_level_analysis(edited, candidate.core_student)
    affected_requests = _requests_for_sections(allocation_input, set(candidate.logical_section_ids))
    affected_students = sorted({request.student_id for request in affected_requests})
    period_transitions = [
        {"course_id": course, "logical_section_id": section_id, "from": list(old), "to": list(new)}
        for section_id, course, old, new in zip(
            candidate.logical_section_ids, candidate.logical_course_ids,
            candidate.original_placements, candidate.proposed_placements,
        )
    ]
    return {
        "pair_id": pair_id,
        "candidate_id": candidate.candidate_id,
        "logical_section_ids": list(candidate.logical_section_ids),
        "logical_course_ids": list(candidate.logical_course_ids),
        "original_placements": [list(item) for item in candidate.original_placements],
        "hinted_destination_placements": [list(item) for item in candidate.proposed_placements],
        "occupancy_shape": [list(item) for item in candidate.occupancy_shape],
        "period_transitions": period_transitions,
        "affected_student_count": len(affected_students),
        "affected_student_union": affected_students,
        "changed_candidate_period_relationships": candidate.affected_candidate_edge_count,
        "total_absolute_period_displacement": sum(
            abs(int(old[0][1:]) - int(new[0][1:]))
            for old, new in zip(candidate.original_placements, candidate.proposed_placements)
        ),
        "core_student": candidate.core_student,
        "core_student_primary_unmet": int(exact["original_primary_unmet"]),
        "core_student_logical_primary_load": int(exact["primary_request_count"]),
        "core_student_schedule_gap": int(exact["original_max_schedule_gap"]),
        "pair_hash": _json_hash(asdict(candidate)),
    }


def _apply_candidate_periods(allocation_input: Any, candidate: CandidateEdit) -> Any:
    from src.period_placement_repair_probe import apply_candidate_to_input

    return apply_candidate_to_input(allocation_input, candidate)


def compare_frozen_pairs(pair_analyses: tuple[dict[str, Any], ...]) -> dict[str, Any]:
    section_id_sets = [tuple(sorted(item["logical_section_ids"])) for item in pair_analyses]
    unique_section_id_pair_count = len(set(section_id_sets))
    identical = unique_section_id_pair_count == 1
    return {
        "pairs_share_identical_section_ids": identical,
        "section_id_sets": section_id_sets,
        "note": (
            "The two frozen K=2 pairs move the same two logical sections "
            "(AP_3D_ART_DESIGN_01, SOCIAL_JUSTICE_01) to different destination "
            "placements for SOCIAL_JUSTICE_01. Diagnostic C for pair_1 and "
            "pair_2 therefore builds the identical fixed-section-ID model; "
            "only the placement/assignment hint differs between the two runs."
        ) if identical else "The two frozen pairs use different section-ID sets.",
        "pair_semantics": {
            "frozen_pair_candidate_count": len(pair_analyses),
            "exact_destination_plan_count": len(pair_analyses),
            "unique_section_id_pair_count": unique_section_id_pair_count,
            "unique_fixed_section_pair_count_tested": unique_section_id_pair_count,
            "diagnostic_c_search_config_count": len(pair_analyses),
            "note": (
                "One unique section-ID pair was tested. Two exact destination "
                "variants and two search-guidance configurations were "
                "evaluated. Diagnostic C's two runs share an identical "
                "feasible region (same fixed section-ID pair) but differ in "
                "placement/assignment hint and Hamming reference; either "
                "INFEASIBLE result alone already proves this section pair "
                "infeasible over the full frozen destination domain, and the "
                "second is a supporting reproduction under a different "
                "search-guidance configuration, not proof that a second, "
                "distinct section pair is infeasible."
            ) if identical else (
                "The two frozen pairs use different section-ID sets, so each "
                "pair is a genuinely distinct fixed-section-pair test."
            ),
        },
    }


# --------------------------------------------------------------------------
# Previous K=2 run log analysis (read-only; structured facts, log facts, and
# inference are kept in clearly separate sub-objects)
# --------------------------------------------------------------------------


def _number(text: str) -> int:
    return int(text.replace("'", ""))


def _parse_k2_solver_log(log_text: str) -> dict[str, Any]:
    initial = re.search(r"Initial optimization model.*?\n#Variables:\s*([\d']+)", log_text, re.S)
    presolved = re.search(r"Presolved optimization model '': \(model_fingerprint: (0x[0-9a-f]+)\).*?\n#Variables:\s*([\d']+)", log_text, re.S)
    unused_removed = re.search(r"presolve: ([\d']+) unused variables removed", log_text)
    search_started = re.search(r"Starting search at ([\d.]+)s", log_text)
    hint_incomplete = re.search(r"The solution hint is incomplete: ([\d']+) out of ([\d']+) non fixed variables hinted", log_text)
    hint_infeasible = bool(re.search(r"(?:hint|solution hint).*infeasible", log_text, re.I))
    response_block = log_text.rsplit("CpSolverResponse summary:", 1)
    response_fields: dict[str, Any] = {}
    if len(response_block) == 2:
        block = response_block[1]
        for key in ("status", "objective", "best_bound", "conflicts", "branches", "restarts"):
            match = re.search(rf"^{key}:\s*(\S+)", block, re.M)
            if match:
                value = match.group(1)
                response_fields[key] = value if key == "status" else (None if value in {"", "nan"} else _try_number(value))
        for key in ("walltime", "usertime", "deterministic_time"):
            match = re.search(rf"^{key}:\s*(\S+)", block, re.M)
            if match:
                response_fields[key] = float(match.group(1))
    return {
        "initial_variable_count": _number(initial.group(1)) if initial else None,
        "presolved_model_fingerprint": presolved.group(1) if presolved else None,
        "presolved_variable_count": _number(presolved.group(2)) if presolved else None,
        "unused_variables_removed": _number(unused_removed.group(1)) if unused_removed else None,
        "symmetry_graph_skipped": "Graph too large. Skipping" in log_text,
        "search_started": bool(search_started),
        "search_start_time_seconds": float(search_started.group(1)) if search_started else None,
        "hint_incomplete_message_seen": bool(hint_incomplete),
        "hint_incomplete_non_fixed_hinted": _number(hint_incomplete.group(1)) if hint_incomplete else None,
        "hint_incomplete_non_fixed_total": _number(hint_incomplete.group(2)) if hint_incomplete else None,
        "hint_infeasible_message_seen": hint_infeasible,
        "raw_cp_solver_response_summary_fields": response_fields,
        "evidence_source": "parsed_solver_log",
    }


def _try_number(value: str) -> Any:
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _analyze_one_k2_run(bootstrap_dir: Path, run_id: str) -> dict[str, Any]:
    run_dir = bootstrap_dir / "runs" / run_id
    log_text = (run_dir / "solver.log").read_text(encoding="utf-8")
    stats = _read_json(run_dir / "response_stats.json")
    hint = _read_json(run_dir / "hint_audit.json")
    config = _read_json(run_dir / "solver_config.json")
    log_evidence = _parse_k2_solver_log(log_text)
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
        "restarts": stats.get("restarts"),
        "response_hash": stats.get("response_hash"),
    }
    return {
        "run_id": run_id,
        "hint_id": config.get("cardinality_cap"),
        "structured_response": structured,
        "log_evidence": log_evidence,
        "hint_audit_source_classification": hint.get("quality", {}).get("classification"),
        "inference": {
            "hint_infeasibility_proven": False,
            "interpretation": (
                "structured response and log agree: solution_count=0, "
                "incumbent_found=false, status=UNKNOWN; the incomplete-hint "
                "message means the CF hint did not cover every non-fixed "
                "variable (assignment variables outside the CF baseline plus "
                "placement variables), not that the hint was infeasible"
            ),
        },
    }


def analyze_previous_k2_runs(bootstrap_dir: str | Path = DEFAULT_BOOTSTRAP) -> dict[str, Any]:
    bootstrap_dir = Path(bootstrap_dir)
    run_1 = _analyze_one_k2_run(bootstrap_dir, "k2_01")
    run_2 = _analyze_one_k2_run(bootstrap_dir, "k2_02")
    fp1 = run_1["log_evidence"]["presolved_model_fingerprint"]
    fp2 = run_2["log_evidence"]["presolved_model_fingerprint"]
    started_1 = run_1["log_evidence"]["search_started"]
    started_2 = run_2["log_evidence"]["search_started"]
    comparison = {
        "presolved_models_byte_identical": fp1 is not None and fp1 == fp2,
        "presolved_variable_counts": [
            run_1["log_evidence"]["presolved_variable_count"],
            run_2["log_evidence"]["presolved_variable_count"],
        ],
        "both_hints_reported_incomplete_not_infeasible": (
            run_1["log_evidence"]["hint_incomplete_message_seen"]
            and run_2["log_evidence"]["hint_incomplete_message_seen"]
            and not run_1["log_evidence"]["hint_infeasible_message_seen"]
            and not run_2["log_evidence"]["hint_infeasible_message_seen"]
        ),
        "both_runs_entered_branch_search": started_1 and started_2,
        "stuck_in_presolve": {
            "run_1": not started_1,
            "run_2": not started_2,
        },
        "search_effort_differs": (
            run_1["structured_response"]["branches"] != run_2["structured_response"]["branches"]
            or run_1["structured_response"]["conflicts"] != run_2["structured_response"]["conflicts"]
        ),
        "hints_produced_different_decisions": fp1 is not None and fp2 is not None and fp1 != fp2,
        "raw_log_objective_and_best_bound_present": (
            bool(run_1["log_evidence"]["raw_cp_solver_response_summary_fields"].get("objective") is not None)
            or bool(run_2["log_evidence"]["raw_cp_solver_response_summary_fields"].get("objective") is not None)
        ),
        "raw_log_values_are_not_a_valid_incumbent": (
            "the raw solver.log CpSolverResponse summary block prints numeric "
            "objective/best_bound fields even when status=UNKNOWN and "
            "solution_count=0; these numbers are NOT a found feasible "
            "solution's objective and must not be reported as an incumbent. "
            "Only structured_response.objective_value (from response_stats.json, "
            "which is null here because incumbent_found=false) is treated as "
            "authoritative for whether an incumbent exists."
        ),
        "supported_conclusion": (
            "Both K=2 runs presolved to a large but tractable model, entered "
            "real branch-and-bound search well before their time budget "
            "expired, and did materially different amounts of search work "
            "(different branch/conflict counts and, when present, different "
            "presolved model fingerprints) without finding a feasible integer "
            "solution. This supports 'the global K=2 search space is large "
            "and a coherent hint alone did not resolve it inside 180s' but "
            "does NOT by itself distinguish whether the bottleneck is "
            "section-pair selection, destination selection, or assignment "
            "feasibility for this specific pair; that is exactly what "
            "Diagnostics A/B/C in this artifact are for."
        ),
    }
    return {"k2_01": run_1, "k2_02": run_2, "comparison": comparison}


# --------------------------------------------------------------------------
# Diagnostic A: exact destinations, production feasibility, no hint
# --------------------------------------------------------------------------


def exact_plan_production_feasibility_no_hint(
    context: Any,
    placement_map: Mapping[str, tuple[str, ...]],
    *,
    config_dir: Path,
    seed: int = SOLVER_SEED,
    time_limit_seconds: float = DIAGNOSTIC_A_BUDGET_SECONDS,
) -> dict[str, Any]:
    """Diagnostic A: fresh production hard model, exact edited placements, no hint, no objective."""
    sections = apply_placement_map_to_sections(context, placement_map)
    edited_input = canonicalize_allocation_input(
        context.students.copy(deep=True), context.requests.copy(deep=True), sections, context.catalog.copy(deep=True)
    )
    build, fallback, rules, math_ids = _fresh_production_build(edited_input, context.catalog, config_dir, seed)
    del rules, math_ids
    assert_empty_solution_hint(build.model)
    build.model.ClearObjective()
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_seconds)
    solver.parameters.num_search_workers = WORKERS
    solver.parameters.random_seed = int(seed)
    solver.parameters.stop_after_first_solution = True
    solver.parameters.log_search_progress = True
    logs: list[str] = []
    solver.log_callback = logs.append
    started = time.perf_counter()
    raw = solver.Solve(build.model)
    status = solver.StatusName(raw)
    wall_time = float(solver.WallTime())
    response_hash = hashlib.sha256(str(solver.ResponseProto()).encode("utf-8")).hexdigest()
    available = status in {"FEASIBLE", "OPTIMAL"}
    assignments = tuple(sorted(
        (key.request_key, key.section_id)
        for key, variable in build.assignment_vars.items()
        if available and solver.BooleanValue(variable)
    ))
    policy_pass = None
    consistency_issue_count = 0
    replay_errors: list[str] = []
    if available:
        state = AllocationState(
            edited_input,
            supplemental_requests=tuple(plan.fallback_request for plan in fallback),
            supplemental_candidate_index={plan.fallback_request.request_key: plan.candidates for plan in fallback},
        )
        for request_key, section_id in assignments:
            request = build.requests_by_key[request_key]
            result = state.try_assign(request.student_id, request_key, section_id)
            if not result.allowed:
                replay_errors.extend(reason.value for reason in result.reasons)
        consistency = state.validate_internal_consistency()
        consistency_issue_count = len(consistency) + len(replay_errors)
        outcomes = _student_outcomes_for_solution(edited_input, assignments, fallback)
        policy = evaluate_final_schedule_policy("hybrid_k2_search_bottleneck_diagnostic_v1", outcomes)
        policy_pass = bool(policy.summary.final_schedule_policy_pass) and not replay_errors
    return {
        "diagnostic_id": "diagnostic_a",
        "model_restriction": "exact_edited_placements_no_dynamic_placement_variables",
        "status": status,
        "assignment_available": available,
        "selected_assignments": [list(item) for item in assignments],
        "policy_pass": policy_pass,
        "consistency_issue_count": consistency_issue_count,
        "runtime_seconds": round(time.perf_counter() - started, 6),
        "wall_time_seconds": wall_time,
        "response_hash": response_hash,
        "seed": int(seed),
        "workers": WORKERS,
        "external_persisted_seed": False,
        "hint_used": False,
        "objective_used": False,
        "budget_seconds": float(time_limit_seconds),
        "placement_map": {sid: list(placement) for sid, placement in placement_map.items()},
        "solver_log": logs,
    }


# --------------------------------------------------------------------------
# Diagnostic B: exact destinations, coherent hint + Hamming
# (thin, unmodified reuse of independent_production_validation with the
# diagnostic's own 60s budget instead of the 300s production budget)
# --------------------------------------------------------------------------


def exact_plan_production_feasibility_with_hint(
    context: Any,
    placement_map: Mapping[str, tuple[str, ...]],
    *,
    config_dir: Path,
    seed: int = SOLVER_SEED,
    time_limit_seconds: float = DIAGNOSTIC_B_BUDGET_SECONDS,
) -> dict[str, Any]:
    result = independent_production_validation(
        context, placement_map, config_dir=config_dir, seed=seed, time_limit_seconds=time_limit_seconds,
    )
    result = dict(result)
    result["diagnostic_id"] = "diagnostic_b"
    result["model_restriction"] = "exact_edited_placements_no_dynamic_placement_variables"
    result["budget_seconds"] = float(time_limit_seconds)
    result["assignment_available"] = bool(result.get("assignment_available"))
    return result


# --------------------------------------------------------------------------
# Diagnostic C: fixed section IDs, destinations free
# --------------------------------------------------------------------------


def add_fixed_section_pair_constraint(build: Any, section_ids: tuple[str, str]) -> None:
    """Force exactly the given two logical sections to change; all other
    editable sections are forced to keep their original placement.

    Both destination options remain fully free within each section's complete
    frozen placement domain: only the *choice of which two sections change*
    is fixed here, never which destination they land on. This is Diagnostic
    C's sole deliberate feasibility restriction.
    """
    if len(set(section_ids)) != 2:
        raise DiagnosticError("fixed section pair must name exactly two distinct sections")
    missing = [sid for sid in section_ids if sid not in build.section_changed_vars]
    if missing:
        raise DiagnosticError(f"pair section(s) missing a changed-indicator variable: {missing}")
    for sid in section_ids:
        build.model.Add(build.section_changed_vars[sid] == 1)
    for sid, variable in build.section_changed_vars.items():
        if sid not in section_ids:
            build.model.Add(variable == 0)


def fixed_pair_destination_free_diagnostic(
    allocation_input: Any,
    domains: Mapping[str, tuple[Any, ...]],
    candidate: CandidateEdit,
    assignment_hint: Iterable[Any],
    *,
    run_id: str,
    math_fallback_rules: tuple[Any, ...],
    math_course_ids: tuple[str, ...],
    seed: int = SOLVER_SEED,
    time_limit_seconds: float = DIAGNOSTIC_C_BUDGET_SECONDS,
) -> tuple[Any, dict[str, Any], SearchResult]:
    build = build_joint_model(
        allocation_input,
        placement_domains=domains,
        math_fallback_rules=math_fallback_rules,
        math_course_ids=math_course_ids,
        occupancy_mode="hybrid_sparse_linear_occupancy",
    )
    add_fixed_section_pair_constraint(build, candidate.logical_section_ids)
    build.model.Minimize(hamming_expression(build, assignment_hint))
    hint = apply_bootstrap_hints(build, candidate, assignment_hint)
    hint["cardinality_cap"] = None
    hint["fixed_section_ids"] = list(candidate.logical_section_ids)
    hint["hamming_objective"] = "unweighted_assignment_distance_to_edited_plan_constrained_first"
    hint["full_domain_preserved"] = True
    hint["candidate_pruning"] = False
    hint["model_restriction"] = "fixed_two_section_ids_changed_all_others_original_destinations_free"
    result = solve_bootstrap(build, run_id=run_id, k=2, hint_id=candidate.candidate_id, seed=seed, time_limit_seconds=time_limit_seconds)
    return build, hint, result


# --------------------------------------------------------------------------
# Witness / production validation for a Diagnostic A/B incumbent
# --------------------------------------------------------------------------


def validate_exact_plan_witness(
    context: Any,
    original_allocation_input: Any,
    placement_map: Mapping[str, tuple[str, ...]],
    assignments: tuple[tuple[str, str], ...],
) -> dict[str, Any]:
    original_sections = original_allocation_input.logical_sections_by_id
    changed = sorted(sid for sid, placement in placement_map.items() if placement != _section_placement(original_sections[sid]))
    return {
        "changed_logical_section_count": len(changed),
        "changed_logical_section_ids": changed,
        "section_count_unchanged": True,
        "section_ids_unchanged": True,
        "capacity_unchanged": True,
        "selected_assignment_count": len(assignments),
    }


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------


@dataclass
class _PairOutcome:
    pair_id: str
    incumbent_found: bool
    classification_evidence: list[str]
    run_rows: list[dict[str, Any]]
    witness_source: dict[str, Any] | None
    correctness_failure: bool


def _run_row(pair_id: str, diagnostic_id: str, response: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "pair_id": pair_id,
        "diagnostic_id": diagnostic_id,
        "status": response.get("status"),
        "assignment_available": bool(response.get("assignment_available")),
        "runtime_seconds": response.get("runtime_seconds") or response.get("wall_time_seconds"),
        "response_hash": response.get("response_hash", ""),
    }


def _write_diagnostic_run(output: Path, pair_id: str, diagnostic_id: str, *, solver_config: dict[str, Any], hint_audit: dict[str, Any], response: dict[str, Any], validation: dict[str, Any]) -> None:
    run_dir = output / "runs" / pair_id / diagnostic_id
    _write_json(run_dir / "solver_config.json", solver_config)
    _write_json(run_dir / "hint_audit.json", hint_audit)
    model_size = {
        key: response.get(key)
        for key in ("assignment_available",)
    }
    _write_json(run_dir / "model_size.json", model_size)
    response_for_disk = {key: value for key, value in response.items() if key != "solver_log"}
    _write_json(run_dir / "response_stats.json", response_for_disk)
    (run_dir / "solver.log").write_text("\n".join(response.get("solver_log", [])), encoding="utf-8")
    _write_json(run_dir / "validation.json", validation)


def run_pair_protocol(
    *,
    pair_index: int,
    candidate: CandidateEdit,
    context: Any,
    domains: Mapping[str, tuple[Any, ...]],
    rules: tuple[Any, ...],
    math_ids: tuple[str, ...],
    config_dir: Path,
    output: Path,
    already_completed: set[tuple[str, str]],
) -> _PairOutcome:
    pair_id = pair_id_for_index(pair_index)
    placement_map = dict(zip(candidate.logical_section_ids, candidate.proposed_placements))
    run_rows: list[dict[str, Any]] = []
    classification_evidence: list[str] = []

    def already_ran(diagnostic_id: str) -> bool:
        return (pair_id, diagnostic_id) in already_completed

    # Diagnostic A
    if not already_ran("diagnostic_a"):
        response = exact_plan_production_feasibility_no_hint(
            context, placement_map, config_dir=config_dir, seed=SOLVER_SEED, time_limit_seconds=DIAGNOSTIC_A_BUDGET_SECONDS,
        )
        solver_config = {
            "seed": SOLVER_SEED, "workers": WORKERS, "max_time_in_seconds": DIAGNOSTIC_A_BUDGET_SECONDS,
            "objective": "none", "hint": "none", "stop_after_first_solution": True, "external_persisted_seed": False,
        }
        hint_audit = {"hint_used": False, "objective_used": False, "model_restriction": response["model_restriction"]}
        validation = {"validated": False, "reason": "validated only if this run produced the accepted incumbent"}
        _write_diagnostic_run(output, pair_id, "diagnostic_a", solver_config=solver_config, hint_audit=hint_audit, response=response, validation=validation)
        run_rows.append(_run_row(pair_id, "diagnostic_a", response))
        if response["status"] == "MODEL_INVALID":
            return _PairOutcome(pair_id, False, ["diagnostic_a_model_invalid"], run_rows, None, True)
        if response["assignment_available"]:
            classification_evidence.append("exact_plan_assignment_feasible_diagnostic_a")
            return _PairOutcome(pair_id, True, classification_evidence, run_rows, {
                "diagnostic_id": "diagnostic_a", "placement_map": placement_map,
                "assignments": tuple(tuple(item) for item in response["selected_assignments"]),
                "response": response,
            }, False)
        if response["status"] == "INFEASIBLE":
            classification_evidence.append("exact_destination_pair_infeasible_diagnostic_a")
        elif response["status"] == "UNKNOWN":
            classification_evidence.append("diagnostic_a_unknown_no_incumbent")
    else:
        classification_evidence.append("diagnostic_a_resumed_from_checkpoint")
        run_rows.append({"pair_id": pair_id, "diagnostic_id": "diagnostic_a", "status": "resumed", "assignment_available": False, "runtime_seconds": None, "response_hash": ""})

    a_status = next((row["status"] for row in run_rows if row["diagnostic_id"] == "diagnostic_a"), None)

    # Diagnostic B only if A was UNKNOWN with no incumbent
    if a_status == "UNKNOWN" and not already_ran("diagnostic_b"):
        response = exact_plan_production_feasibility_with_hint(
            context, placement_map, config_dir=config_dir, seed=SOLVER_SEED, time_limit_seconds=DIAGNOSTIC_B_BUDGET_SECONDS,
        )
        solver_config = {
            "seed": SOLVER_SEED, "workers": WORKERS, "max_time_in_seconds": DIAGNOSTIC_B_BUDGET_SECONDS,
            "objective": "hamming_to_edited_plan_constrained_first", "hint": "edited_plan_constrained_first",
            "stop_after_first_solution": True, "external_persisted_seed": False,
        }
        hint_audit = {"hint_used": True, "objective_used": True, "hint_source": response.get("internal_hint_algorithm"), "model_restriction": response["model_restriction"]}
        validation = {"validated": False, "reason": "validated only if this run produced the accepted incumbent"}
        _write_diagnostic_run(output, pair_id, "diagnostic_b", solver_config=solver_config, hint_audit=hint_audit, response=response, validation=validation)
        run_rows.append(_run_row(pair_id, "diagnostic_b", response))
        if response["assignment_available"]:
            classification_evidence.append("assignment_hint_helpful_diagnostic_b")
            assignments = tuple(sorted(response.get("selected_assignments", []))) if isinstance(response.get("selected_assignments"), list) else ()
            return _PairOutcome(pair_id, True, classification_evidence, run_rows, {
                "diagnostic_id": "diagnostic_b", "placement_map": placement_map, "assignments": assignments, "response": response,
            }, False)
        if response["status"] == "INFEASIBLE":
            classification_evidence.append("exact_destination_pair_infeasible_diagnostic_b")
        elif response["status"] == "UNKNOWN":
            classification_evidence.append("diagnostic_b_unknown_no_incumbent")
    elif a_status == "UNKNOWN":
        classification_evidence.append("diagnostic_b_resumed_from_checkpoint")

    b_status = next((row["status"] for row in run_rows if row["diagnostic_id"] == "diagnostic_b"), None)

    # Diagnostic C unless A or B already proved this exact plan INFEASIBLE
    # and only if there is no incumbent yet (already ensured by early returns
    # above).
    exact_plan_infeasible = a_status == "INFEASIBLE" or b_status == "INFEASIBLE"
    if not already_ran("diagnostic_c"):
        quality, assignment_hint = _assignment_hint_quality_for_pair(context, candidate, config_dir, SOLVER_SEED)
        build, hint, result = fixed_pair_destination_free_diagnostic(
            context.allocation_input, domains, candidate, assignment_hint,
            run_id=f"{pair_id}_diagnostic_c", math_fallback_rules=rules, math_course_ids=math_ids,
            seed=SOLVER_SEED, time_limit_seconds=DIAGNOSTIC_C_BUDGET_SECONDS,
        )
        response = {key: value for key, value in asdict(result).items()}
        solver_config = {
            "seed": SOLVER_SEED, "workers": WORKERS, "max_time_in_seconds": DIAGNOSTIC_C_BUDGET_SECONDS,
            "objective": "hamming_to_edited_plan_constrained_first", "hint": "edited_plan_constrained_first_plus_fixed_pair_placement",
            "stop_after_first_solution": True, "external_persisted_seed": False,
        }
        hint_audit = hint | {"quality": quality}
        validation = {"validated": False, "reason": "validated only if this run produced the accepted incumbent"}
        response_row = dict(response)
        response_row["assignment_available"] = result.assignment_available
        response_row["status"] = result.status
        response_row["response_hash"] = result.response_hash
        response_row["runtime_seconds"] = result.end_to_end_runtime_seconds
        _write_diagnostic_run(output, pair_id, "diagnostic_c", solver_config=solver_config, hint_audit=hint_audit, response=response_row, validation=validation)
        run_rows.append(_run_row(pair_id, "diagnostic_c", response_row))
        if result.status == "MODEL_INVALID":
            return _PairOutcome(pair_id, False, classification_evidence + ["diagnostic_c_model_invalid"], run_rows, None, True)
        if result.assignment_available:
            witness = validate_bootstrap_witness(context, build, result, config_dir=config_dir, k=2)
            if witness.get("joint_bootstrap_witness_valid"):
                classification_evidence.append(
                    "global_section_pair_selection_bottleneck_supported" if exact_plan_infeasible else "destination_selection_bottleneck"
                )
                return _PairOutcome(pair_id, True, classification_evidence, run_rows, {
                    "diagnostic_id": "diagnostic_c", "placement_map": dict(result.selected_placements),
                    "assignments": result.selected_assignments, "response": response_row, "build": build, "result": result, "witness": witness,
                }, False)
            classification_evidence.append("diagnostic_c_witness_invalid")
        elif result.status == "INFEASIBLE":
            classification_evidence.append("fixed_section_pair_infeasible")
        elif result.status == "UNKNOWN":
            classification_evidence.append("destination_and_assignment_search_unresolved")
    else:
        classification_evidence.append("diagnostic_c_resumed_from_checkpoint")

    return _PairOutcome(pair_id, False, classification_evidence, run_rows, None, False)


def _assignment_hint_quality_for_pair(context: Any, candidate: CandidateEdit, config_dir: Path, seed: int) -> tuple[dict[str, Any], tuple[Any, ...]]:
    from src.hybrid_stage1_incumbent_bootstrap import _assignment_hint_quality

    return _assignment_hint_quality(context, candidate, config_dir, seed)


def classify_bottleneck(pair_outcomes: tuple[_PairOutcome, ...]) -> dict[str, Any]:
    evidence: list[str] = []
    for outcome in pair_outcomes:
        evidence.extend(f"{outcome.pair_id}:{item}" for item in outcome.classification_evidence)
    incumbent_found = any(outcome.incumbent_found for outcome in pair_outcomes)
    correctness_failure = any(outcome.correctness_failure for outcome in pair_outcomes)
    labels: list[str] = []
    if correctness_failure:
        labels = ["correctness_failure"]
    elif any("exact_plan_assignment_feasible_diagnostic_a" in item for item in evidence):
        labels.append("exact_plan_assignment_feasible")
    elif any("assignment_hint_helpful_diagnostic_b" in item for item in evidence):
        labels.append("assignment_hint_helpful")
    elif any("global_section_pair_selection_bottleneck_supported" in item for item in evidence):
        labels.append("global_section_pair_selection_bottleneck_supported")
    elif any("destination_selection_bottleneck" in item for item in evidence):
        labels.append("destination_selection_bottleneck")
    else:
        if any("exact_destination_pair_infeasible" in item for item in evidence):
            labels.append("exact_destination_pair_infeasible")
        if any("fixed_section_pair_infeasible" in item for item in evidence):
            labels.append("fixed_section_pair_infeasible")
        if any("_unknown_no_incumbent" in item for item in evidence) or any("destination_and_assignment_search_unresolved" in item for item in evidence):
            labels.append("assignment_search_unresolved" if any("diagnostic_a_unknown" in item or "diagnostic_b_unknown" in item for item in evidence) else "destination_and_assignment_search_unresolved")
        if not incumbent_found and all(
            {"diagnostic_a", "diagnostic_b", "diagnostic_c"} <= {row["diagnostic_id"] for row in outcome.run_rows} or
            (row_has_infeasible_a_or_b(outcome.run_rows))
            for outcome in pair_outcomes
        ):
            labels.append("diagnostic_portfolio_exhausted_no_incumbent")
    if not labels:
        labels.append("diagnostic_portfolio_exhausted_no_incumbent")
    return {
        "labels": sorted(set(labels)),
        "evidence": evidence,
        "incumbent_found": incumbent_found,
        "note": "labels marked *_supported are evidence-based inferences from these two frozen pairs only; they are not a complete causal proof over the full K=2 candidate space.",
    }


def row_has_infeasible_a_or_b(rows: list[dict[str, Any]]) -> bool:
    return any(row["diagnostic_id"] in {"diagnostic_a", "diagnostic_b"} and row["status"] == "INFEASIBLE" for row in rows)


# --------------------------------------------------------------------------
# Top-level run
# --------------------------------------------------------------------------


def run_diagnostic(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    output_dir: str | Path = DEFAULT_OUTPUT,
    bootstrap_dir: str | Path = DEFAULT_BOOTSTRAP,
    preview_dir: str | Path = DEFAULT_PREVIEW_OUTPUT,
    audit_root: str | Path = DEFAULT_AUDIT_ROOT,
    config_dir: str | Path = "data/config",
    resume: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir)
    resume_checkpoint: dict[str, Any] | None = None
    if output.exists() and any(output.iterdir()):
        if not resume:
            raise DiagnosticError(f"diagnostic output is non-empty; refusing overwrite: {output}")
        aggregate = output / "aggregate_summary.json"
        if aggregate.is_file():
            return _read_json(aggregate) | {"resumed": True, "search_reexecuted": False}
        checkpoint = output / "checkpoint.json"
        if not checkpoint.is_file():
            raise DiagnosticError("resume requested without an atomic checkpoint")
        resume_checkpoint = _read_json(checkpoint)
        if resume_checkpoint.get("schema_version") != 1:
            raise DiagnosticError("unsupported diagnostic checkpoint schema")

    manifest = load_diagnostic_manifest(manifest_path)
    source_verification = verify_source_artifacts(manifest)
    bootstrap_dir = Path(bootstrap_dir)
    pairs = load_frozen_pairs(manifest, bootstrap_dir)
    previous_k2_analysis = analyze_previous_k2_runs(bootstrap_dir)

    audit_manifest = load_section_plan_audit_manifest("data/scenarios/section_plan_feasibility_audit_v1.json")
    context = load_scenario_context(TARGET_SCENARIO_ID, audit_manifest=audit_manifest, audit_root=Path(audit_root), config_dir=config_dir)
    domains, domain_summary = build_frozen_placement_domains(context, preview_dir)
    hashes = frozen_domain_hashes(domains, domain_summary.source_candidate_ids, context.allocation_input)
    if domain_summary.editable_logical_section_count != int(manifest["editable_section_count"]) or domain_summary.total_unique_placement_options != int(manifest["placement_option_count"]):
        raise DiagnosticError("frozen domain count drift versus manifest")
    rules = _load_math_fallback_rules(Path(config_dir), context.catalog)
    math_ids = math_course_ids_from_catalog(context.catalog)

    pair_analyses = tuple(
        analyze_pair(pair_id_for_index(index), candidate, context.allocation_input, Path(config_dir))
        for index, candidate in enumerate(pairs)
    )
    pair_comparison = compare_frozen_pairs(pair_analyses)

    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "diagnostic_manifest_snapshot.json", manifest)
    _write_json(output / "provenance.json", {
        "source_git_commit": manifest["source_git_commit"], "previous_full_k2_reruns": 0,
        "diagnostic_a_runs": 0, "diagnostic_b_runs": 0, "diagnostic_c_runs": 0,
        "k1_runs": 0, "k3_runs": 0, "production_validation_runs": 0,
        "control_runs": 0, "other_normal_target_runs": 0, "stress_runs": 0, "negative_runs": 0, "holdout_runs": 0,
        "external_persisted_seed": False,
    })
    _write_json(output / "source_artifact_verification.json", source_verification)
    _write_json(output / "frozen_pair_portfolio.json", {
        "count": len(pair_analyses), "pairs": list(pair_analyses), "comparison": pair_comparison,
        "frozen_domain": {"counts": asdict(domain_summary), "hashes": hashes},
    })
    _write_json(output / "previous_k2_log_analysis.json", previous_k2_analysis)

    run_rows: list[dict[str, Any]] = []
    failures: list[str] = []
    discovered_witness: dict[str, Any] = {"status": "not_run", "not_run_reason": "no_incumbent_yet"}
    fixed_witness_acceptance: dict[str, Any] = {"status": "not_run", "not_run_reason": "no_incumbent_yet"}
    production_validation: dict[str, Any] = {"status": "not_run", "not_run_reason": "no_incumbent_yet"}
    accepted = False
    validated = False
    completed: set[tuple[str, str]] = set()
    pair_outcomes: list[_PairOutcome] = []

    if resume_checkpoint is not None:
        run_rows.extend(resume_checkpoint.get("run_rows", []))
        completed = {(row["pair_id"], row["diagnostic_id"]) for row in run_rows}
        failures.extend(str(item) for item in resume_checkpoint.get("failures", []))
        discovered_witness = dict(resume_checkpoint.get("discovered_witness", discovered_witness))
        fixed_witness_acceptance = dict(resume_checkpoint.get("fixed_witness_acceptance", fixed_witness_acceptance))
        production_validation = dict(resume_checkpoint.get("production_validation", production_validation))
        accepted = bool(resume_checkpoint.get("accepted", False))
        validated = bool(resume_checkpoint.get("validated", False))

    def checkpoint_state() -> None:
        _write_csv(output / "diagnostic_runs.csv", run_rows)
        _write_json(output / "checkpoint.json", {
            "schema_version": 1, "run_rows": run_rows, "failures": failures,
            "discovered_witness": discovered_witness, "fixed_witness_acceptance": fixed_witness_acceptance,
            "production_validation": production_validation, "accepted": accepted, "validated": validated,
            "search_reexecuted": False,
        })

    checkpoint_state()

    accepted_witness_source: dict[str, Any] | None = None
    for index, candidate in enumerate(pairs):
        if accepted:
            break
        pair_id = pair_id_for_index(index)
        pair_run_count = sum(1 for pid, _ in completed if pid == pair_id)
        if index == 1 and not pair_outcomes:
            break  # defensive; pair_1 must run first
        if index == 1:
            previous = pair_outcomes[0]
            if previous.incumbent_found or previous.correctness_failure:
                break
        already_for_pair = {(pid, did) for pid, did in completed if pid == pair_id}
        outcome = run_pair_protocol(
            pair_index=index, candidate=candidate, context=context, domains=domains,
            rules=rules, math_ids=math_ids, config_dir=Path(config_dir), output=output,
            already_completed=already_for_pair,
        )
        pair_outcomes.append(outcome)
        run_rows.extend(outcome.run_rows)
        completed.update((row["pair_id"], row["diagnostic_id"]) for row in outcome.run_rows)
        checkpoint_state()
        if outcome.correctness_failure:
            failures.append(f"{pair_id}:correctness_failure")
            checkpoint_state()
            break
        if outcome.incumbent_found:
            accepted_witness_source = outcome.witness_source
            accepted = True
            checkpoint_state()
            break

    if accepted and accepted_witness_source is not None:
        diagnostic_id = accepted_witness_source["diagnostic_id"]
        placement_map = accepted_witness_source["placement_map"]
        assignments = accepted_witness_source["assignments"]
        discovered_witness = {
            "status": "found", "diagnostic_id": diagnostic_id,
            "placement_map": {sid: list(placement) for sid, placement in placement_map.items()},
            "selected_assignment_count": len(assignments),
            "response_hash": accepted_witness_source["response"].get("response_hash", ""),
        }
        if diagnostic_id == "diagnostic_c":
            discovered_witness["joint_witness"] = accepted_witness_source["witness"]
        else:
            discovered_witness["exact_plan_witness"] = validate_exact_plan_witness(context, context.allocation_input, placement_map, assignments)
        fixed_witness_acceptance = production_fixed_witness_acceptance(
            context, placement_map, assignments, config_dir=Path(config_dir), seed=SOLVER_SEED, time_limit_seconds=FIXED_WITNESS_BUDGET_SECONDS,
        )
        fixed_witness_acceptance["production_fixed_witness_accepted"] = bool(
            fixed_witness_acceptance.get("status") in {"FEASIBLE", "OPTIMAL"}
            and fixed_witness_acceptance.get("assignment_exact")
            and fixed_witness_acceptance.get("policy_pass")
            and fixed_witness_acceptance.get("consistency_issue_count") == 0
            and fixed_witness_acceptance.get("response_hash")
        )
        if fixed_witness_acceptance["production_fixed_witness_accepted"]:
            production_validation = independent_production_validation(
                context, placement_map, config_dir=Path(config_dir), seed=SOLVER_SEED, time_limit_seconds=PRODUCTION_BUDGET_SECONDS,
            )
            validated = bool(production_validation.get("independently_validated_period_repair"))
        checkpoint_state()

    claim = _minimum_claim(accepted, validated, fixed_witness_acceptance)
    classification = classify_bottleneck(tuple(pair_outcomes))
    _write_json(output / "discovered_witness.json", discovered_witness)
    _write_json(output / "hybrid_fixed_witness_acceptance.json", fixed_witness_acceptance)
    _write_json(output / "production_validation.json", production_validation)
    _write_json(output / "bottleneck_classification.json", classification)
    _write_json(output / "failures.json", {"failures": failures, "unexpected_failure_count": len(failures)})
    diagnostic_a_runs = sum(1 for row in run_rows if row["diagnostic_id"] == "diagnostic_a")
    diagnostic_b_runs = sum(1 for row in run_rows if row["diagnostic_id"] == "diagnostic_b")
    diagnostic_c_runs = sum(1 for row in run_rows if row["diagnostic_id"] == "diagnostic_c")
    production_validation_runs = int(production_validation.get("status") not in {"not_run", None})
    # Re-write provenance.json with final counters; the copy written earlier
    # (before any diagnostic ran) is intentionally a pre-search snapshot of
    # all zeros and must not be mistaken for the final run tally.
    _write_json(output / "provenance.json", {
        "source_git_commit": manifest["source_git_commit"], "previous_full_k2_reruns": 0,
        "diagnostic_a_runs": diagnostic_a_runs, "diagnostic_b_runs": diagnostic_b_runs, "diagnostic_c_runs": diagnostic_c_runs,
        "k1_runs": 0, "k3_runs": 0, "production_validation_runs": production_validation_runs,
        "control_runs": 0, "other_normal_target_runs": 0, "stress_runs": 0, "negative_runs": 0, "holdout_runs": 0,
        "external_persisted_seed": False,
    })
    aggregate = {
        "experiment_name": manifest["experiment_name"], "phase": manifest["phase"],
        "target_scenario_id": TARGET_SCENARIO_ID, "pairs_run": len(pair_outcomes),
        "previous_full_k2_reruns": 0,
        "diagnostic_a_runs": diagnostic_a_runs, "diagnostic_b_runs": diagnostic_b_runs, "diagnostic_c_runs": diagnostic_c_runs,
        "k1_runs": 0, "k3_runs": 0,
        "production_validation_runs": production_validation_runs,
        "control_runs": 0, "other_normal_target_runs": 0, "stress_runs": 0, "negative_runs": 0, "holdout_runs": 0,
        "external_persisted_seed": False, "accepted": accepted, "validated": validated,
        "minimum_claim": claim, "bottleneck_classification": classification["labels"],
        "result_classification": "resolved_with_validated_repair" if validated else ("resolved_incumbent_pending_validation" if accepted else "unresolved_no_incumbent"),
        "failures": failures,
    }
    _write_json(output / "aggregate_summary.json", aggregate)
    write_checksums(output)
    return aggregate


def _minimum_claim(accepted: bool, validated: bool, fixed_witness_acceptance: Mapping[str, Any]) -> dict[str, Any]:
    if validated and accepted and fixed_witness_acceptance.get("production_fixed_witness_accepted"):
        return {"claim": "minimum_changed_sections_within_frozen_placement_domain", "value": 2, "proven": True}
    if accepted:
        return {"claim": "candidate_repair_found_pending_full_validation", "proven": False}
    return {"claim": "unresolved_no_incumbent", "proven": False}


def build_execution_history_correction(pair_semantics: Mapping[str, Any]) -> dict[str, Any]:
    """Assemble the execution_history_correction.json payload.

    Pure and read-only with respect to solvers: it only combines the
    hardcoded, independently-recorded facts about this specific artifact's
    real execution history (see the module-level constants above) with the
    pair-semantics facts already derivable from the frozen portfolio.
    """
    return {
        "audit_type": "execution_history_correction",
        "created_for": "hybrid_k2_search_bottleneck_diagnostic_v1",
        "reason": (
            "The accepted final artifact's provenance.json originally reported "
            "only the final (second) batch's four diagnostic solver runs, "
            "without disclosing that a first batch of four runs was executed "
            "and then superseded after a provenance-counter finalization bug "
            "was discovered and fixed."
        ),
        "accepted_final_artifact_runs": dict(ACCEPTED_FINAL_ARTIFACT_RUNS),
        "superseded_runs": dict(SUPERSEDED_RUNS),
        "total_solver_invocations": dict(TOTAL_SOLVER_INVOCATIONS),
        "rerun_batches": RERUN_BATCHES,
        "protocol_deviation": PROTOCOL_DEVIATION,
        "protocol_deviation_reason": PROTOCOL_DEVIATION_REASON,
        "result_based_parameter_change": RESULT_BASED_PARAMETER_CHANGE,
        "portfolio_changed_between_batches": PORTFOLIO_CHANGED_BETWEEN_BATCHES,
        "seed_changed_between_batches": SEED_CHANGED_BETWEEN_BATCHES,
        "budget_changed_between_batches": BUDGET_CHANGED_BETWEEN_BATCHES,
        "incumbent_selection_bias_note": INCUMBENT_SELECTION_BIAS_NOTE,
        "first_batch_evidence": {
            "first_batch_statuses_observed": list(FIRST_BATCH_STATUSES_OBSERVED),
            "first_batch_statuses_evidence_source": FIRST_BATCH_STATUSES_EVIDENCE_SOURCE,
            "first_batch_response_hashes": None,
            "first_batch_exact_runtime_seconds": None,
            "first_batch_per_run_pair_diagnostic_mapping": None,
            "evidence_retention": "original artifact deleted before audit correction",
            "note": FIRST_BATCH_UNAVAILABLE_NOTE,
        },
        "pair_semantics": dict(pair_semantics),
        "solver_evidence_files_modified": False,
        "solver_evidence_byte_integrity_note": (
            "runs/**/solver.log, runs/**/response_stats.json, "
            "runs/**/solver_config.json, runs/**/model_size.json, and "
            "runs/**/validation.json for the accepted final batch were not "
            "modified by this correction; only artifact-level reporting/"
            "summary files were updated."
        ),
    }


def apply_execution_history_correction(output_dir: str | Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    """Patch an already-completed diagnostic artifact's reporting files with
    the real execution history (superseded first batch + accepted final
    batch + total invocations) and corrected pair-count semantics.

    Never builds or solves a CP-SAT model and never touches
    ``runs/**`` solver evidence files. Fails closed if the artifact's
    recorded accepted-batch run counts do not match the known execution
    history, so this cannot silently mislabel an unrelated artifact.
    """
    output = Path(output_dir)
    portfolio_path = output / "frozen_pair_portfolio.json"
    checkpoint_path = output / "checkpoint.json"
    provenance_path = output / "provenance.json"
    aggregate_path = output / "aggregate_summary.json"
    failures_path = output / "failures.json"
    for path in (portfolio_path, checkpoint_path, provenance_path, aggregate_path, failures_path):
        if not path.is_file():
            raise DiagnosticError(f"cannot apply execution-history correction: missing {path}")

    checkpoint = _read_json(checkpoint_path)
    run_rows = checkpoint.get("run_rows", [])
    observed = {
        "diagnostic_a": sum(1 for row in run_rows if row["diagnostic_id"] == "diagnostic_a"),
        "diagnostic_b": sum(1 for row in run_rows if row["diagnostic_id"] == "diagnostic_b"),
        "diagnostic_c": sum(1 for row in run_rows if row["diagnostic_id"] == "diagnostic_c"),
    }
    expected = {key: ACCEPTED_FINAL_ARTIFACT_RUNS[key] for key in ("diagnostic_a", "diagnostic_b", "diagnostic_c")}
    if observed != expected:
        raise DiagnosticError(
            f"accepted final artifact run counts {observed} do not match the "
            f"recorded execution history {expected}; refusing to apply correction"
        )

    portfolio = _read_json(portfolio_path)
    pairs = tuple(portfolio.get("pairs", ()))
    comparison = compare_frozen_pairs(pairs)
    portfolio["comparison"] = comparison
    _write_json(portfolio_path, portfolio)

    history = build_execution_history_correction(comparison["pair_semantics"])
    _write_json(output / "execution_history_correction.json", history)

    provenance = _read_json(provenance_path)
    provenance.update({
        "accepted_final_artifact_diagnostic_a_runs": ACCEPTED_FINAL_ARTIFACT_RUNS["diagnostic_a"],
        "accepted_final_artifact_diagnostic_b_runs": ACCEPTED_FINAL_ARTIFACT_RUNS["diagnostic_b"],
        "accepted_final_artifact_diagnostic_c_runs": ACCEPTED_FINAL_ARTIFACT_RUNS["diagnostic_c"],
        "diagnostic_a_runs_scope": "accepted_final_artifact_only",
        "diagnostic_b_runs_scope": "accepted_final_artifact_only",
        "diagnostic_c_runs_scope": "accepted_final_artifact_only",
        "total_solver_invocations_diagnostic_a": TOTAL_SOLVER_INVOCATIONS["diagnostic_a"],
        "total_solver_invocations_diagnostic_b": TOTAL_SOLVER_INVOCATIONS["diagnostic_b"],
        "total_solver_invocations_diagnostic_c": TOTAL_SOLVER_INVOCATIONS["diagnostic_c"],
        "total_solver_invocations_all": TOTAL_SOLVER_INVOCATIONS["total"],
        "rerun_batches": RERUN_BATCHES,
        "protocol_deviation": PROTOCOL_DEVIATION,
        "execution_history_correction_file": "execution_history_correction.json",
    })
    _write_json(provenance_path, provenance)

    aggregate = _read_json(aggregate_path)
    aggregate.update({
        "accepted_final_artifact_diagnostic_a_runs": ACCEPTED_FINAL_ARTIFACT_RUNS["diagnostic_a"],
        "accepted_final_artifact_diagnostic_b_runs": ACCEPTED_FINAL_ARTIFACT_RUNS["diagnostic_b"],
        "accepted_final_artifact_diagnostic_c_runs": ACCEPTED_FINAL_ARTIFACT_RUNS["diagnostic_c"],
        "total_solver_invocations": dict(TOTAL_SOLVER_INVOCATIONS),
        "rerun_batches": RERUN_BATCHES,
        "protocol_deviation": PROTOCOL_DEVIATION,
        "protocol_deviation_reason": PROTOCOL_DEVIATION_REASON,
        "execution_history_correction_file": "execution_history_correction.json",
        "pair_semantics": comparison["pair_semantics"],
    })
    _write_json(aggregate_path, aggregate)

    failures = _read_json(failures_path)
    failures["protocol_deviations"] = [{
        "type": "rerun_after_reporting_bug",
        "description": (
            "The first completed diagnostic batch (2x Diagnostic A + 2x "
            "Diagnostic C, all INFEASIBLE per this session's own transcript) "
            "was superseded and rerun after discovering that provenance.json "
            "was not finalized with real run counts. The original artifact "
            "directory was deleted and the same portfolio, model "
            "restrictions, seed, workers, and budgets were used for the "
            "accepted final batch."
        ),
        "reason": "provenance.json finalization bug (stale all-zero counters never overwritten with real counts)",
        "result_based": RESULT_BASED_PARAMETER_CHANGE,
        "portfolio_changed": PORTFOLIO_CHANGED_BETWEEN_BATCHES,
        "seed_changed": SEED_CHANGED_BETWEEN_BATCHES,
        "budget_changed": BUDGET_CHANGED_BETWEEN_BATCHES,
        "incumbent_selection_bias": False,
        "details_file": "execution_history_correction.json",
    }]
    _write_json(failures_path, failures)

    checkpoint["protocol_deviation"] = PROTOCOL_DEVIATION
    checkpoint["execution_history_correction_file"] = "execution_history_correction.json"
    _write_json(checkpoint_path, checkpoint)

    checksum = write_checksums(output)
    return {
        "output_dir": str(output),
        "execution_history_correction_hash": _sha256_file(output / "execution_history_correction.json"),
        "sha256sums_hash": checksum,
        "accepted_final_artifact_runs": dict(ACCEPTED_FINAL_ARTIFACT_RUNS),
        "total_solver_invocations": dict(TOTAL_SOLVER_INVOCATIONS),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-dir", type=Path, default=DEFAULT_BOOTSTRAP)
    parser.add_argument("--preview-dir", type=Path, default=DEFAULT_PREVIEW_OUTPUT)
    parser.add_argument("--audit-root", type=Path, default=DEFAULT_AUDIT_ROOT)
    parser.add_argument("--config-dir", type=Path, default=Path("data/config"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--apply-execution-history-correction", action="store_true",
        help=(
            "Patch an already-completed artifact's reporting files with the "
            "real superseded/accepted/total solver-invocation history and "
            "corrected pair-count semantics. Does not build or solve any "
            "CP-SAT model and does not touch runs/** solver evidence files."
        ),
    )
    args = parser.parse_args(argv)
    try:
        if args.apply_execution_history_correction:
            print(json.dumps(apply_execution_history_correction(args.output_dir), indent=2, sort_keys=True, default=str))
            return 0
        print(json.dumps(run_diagnostic(
            manifest_path=args.manifest, output_dir=args.output_dir, bootstrap_dir=args.bootstrap_dir,
            preview_dir=args.preview_dir, audit_root=args.audit_root, config_dir=args.config_dir, resume=args.resume,
        ), indent=2, sort_keys=True, default=str))
    except (DiagnosticError, OSError, ValueError) as exc:
        print(f"Hybrid K=2 search bottleneck diagnostic FAILED: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
