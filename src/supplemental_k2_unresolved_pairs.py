"""Fail-closed supplemental protocol for unresolved formal K=2 pairs.

Dry-run is solver-free. Execution remains disabled until a separate manifest
revision records an explicitly approved extended time budget.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src import formal_remaining_k2_batch as formal
from src.model_proto_serialization import deterministic_model_proto_bytes


SCHEMA_VERSION = 1
PAIR_COUNT = 2
SOLVER_SEED = 20260630
WORKERS = 1
SOURCE_FILE_COUNT = 22
SOURCE_CHECKSUM_ENTRIES = 21
SOURCE_SHA256SUMS_HASH = "ee8eaf6fc6a5e1bb4744b051aabc311929b4a8b8d22e0e68269addad8bb71f4a"
SOURCE_MANIFEST_HASH = "0e1a59fa10f718a36d8a5ca56a7bdc331a4f74d82e63d56b8003bbda43c14ec9"
SOURCE_ORDERING_HASH = "f1246ec8582d6925e26fcc9ee53583a7ac275d6063781b2f59865af788573927"
SOURCE_PAIR_ROWS_HASH = "03c72ef0489cb332fe59a33ce91230764379dec9f0273c4dfc80d86a5c51da0c"
RESPONSE_HASH_NAMESPACE = "supplemental_k2_unresolved_pairs_v1"
DEFAULT_MANIFEST = Path("data/scenarios/supplemental_k2_unresolved_pairs_v1.json")
DEFAULT_SOURCE_ARTIFACT = Path(
    "../fair-course-allocation-artifacts/robustness-v1/"
    "formal-remaining-k2-batch-v1"
)
SUPPLEMENTAL_OUTPUT = Path(
    "../fair-course-allocation-artifacts/robustness-v1/"
    "supplemental-k2-unresolved-pairs-v1"
)
EXPECTED_PAIRS = (
    {
        "frozen_non_original_destination_domains": {
            "AP_3D_ART_DESIGN_01": [["P1"], ["P2"], ["P4"], ["P5"], ["P6"]],
            "INTERMEDIATE_ACTING_01": [["P3"], ["P7"]],
        },
        "original_formal_order": 1,
        "original_placements": {
            "AP_3D_ART_DESIGN_01": ["P7"],
            "INTERMEDIATE_ACTING_01": ["P1"],
        },
        "pair_id": "AP_3D_ART_DESIGN_01__INTERMEDIATE_ACTING_01",
        "section_ids": ["AP_3D_ART_DESIGN_01", "INTERMEDIATE_ACTING_01"],
        "original_classification": "artifact_failure",
        "reason": "original_artifact_failure",
    },
    {
        "frozen_non_original_destination_domains": {
            "AP_3D_ART_DESIGN_01": [["P1"], ["P2"], ["P4"], ["P5"], ["P6"]],
            "FOOTBALL_01": [["P3"], ["P7"]],
        },
        "original_formal_order": 2,
        "original_placements": {
            "AP_3D_ART_DESIGN_01": ["P7"],
            "FOOTBALL_01": ["P1"],
        },
        "pair_id": "AP_3D_ART_DESIGN_01__FOOTBALL_01",
        "section_ids": ["AP_3D_ART_DESIGN_01", "FOOTBALL_01"],
        "original_classification": "unresolved_unknown_no_incumbent",
        "reason": "original_unknown_no_incumbent",
    },
)
CLASSIFICATIONS = {
    "supplemental_planned_not_run",
    "supplemental_fixed_pair_infeasible",
    "supplemental_incumbent_pending_validation",
    "supplemental_unresolved_unknown_no_incumbent",
    "supplemental_model_invalid",
    "supplemental_artifact_failure",
}


class SupplementalK2Error(ValueError):
    """Raised when the supplemental protocol cannot proceed safely."""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SupplementalK2Error(f"cannot read JSON: {path}") from exc


def load_manifest(path: str | Path = DEFAULT_MANIFEST) -> tuple[dict[str, Any], str]:
    manifest_path = Path(path)
    payload = _read_json(manifest_path)
    required = {
        "manifest_schema_version",
        "experiment_name",
        "experiment_version",
        "source_formal_artifact",
        "source_formal_artifact_file_count",
        "source_formal_artifact_checksum_entry_count",
        "source_formal_artifact_sha256sums_hash",
        "source_formal_manifest_hash",
        "source_formal_ordering_hash",
        "source_formal_pair_rows_hash",
        "supplemental_experiment",
        "original_run_preserved",
        "original_result_not_overwritten",
        "original_invocation_not_reclassified",
        "supplemental_pairs",
        "hint",
        "assignment_hint",
        "objective",
        "candidate_pruning",
        "feasibility_only",
        "same_production_model_semantics",
        "both_selected_sections_forced_changed",
        "other_editable_sections_fixed_original",
        "complete_frozen_non_original_destination_domains",
        "seed",
        "workers",
        "stop_after_first_solution",
        "run_b_allowed",
        "no_automatic_rerun",
        "extended_budget_policy",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise SupplementalK2Error("manifest missing: " + ", ".join(missing))
    frozen = {
        "manifest_schema_version": SCHEMA_VERSION,
        "experiment_name": "supplemental_k2_unresolved_pairs",
        "experiment_version": "v1",
        "source_formal_artifact": str(DEFAULT_SOURCE_ARTIFACT),
        "source_formal_artifact_file_count": SOURCE_FILE_COUNT,
        "source_formal_artifact_checksum_entry_count": SOURCE_CHECKSUM_ENTRIES,
        "source_formal_artifact_sha256sums_hash": SOURCE_SHA256SUMS_HASH,
        "source_formal_manifest_hash": SOURCE_MANIFEST_HASH,
        "source_formal_ordering_hash": SOURCE_ORDERING_HASH,
        "source_formal_pair_rows_hash": SOURCE_PAIR_ROWS_HASH,
        "supplemental_experiment": True,
        "original_run_preserved": True,
        "original_result_not_overwritten": True,
        "original_invocation_not_reclassified": True,
        "supplemental_pairs": list(EXPECTED_PAIRS),
        "hint": False,
        "assignment_hint": False,
        "objective": False,
        "candidate_pruning": False,
        "feasibility_only": True,
        "same_production_model_semantics": True,
        "both_selected_sections_forced_changed": True,
        "other_editable_sections_fixed_original": True,
        "complete_frozen_non_original_destination_domains": True,
        "seed": SOLVER_SEED,
        "workers": WORKERS,
        "stop_after_first_solution": True,
        "run_b_allowed": False,
        "no_automatic_rerun": True,
    }
    drift = [key for key, value in frozen.items() if payload.get(key) != value]
    if drift:
        raise SupplementalK2Error(f"frozen manifest drift: {drift}")
    policy = payload["extended_budget_policy"]
    if not isinstance(policy, dict) or policy.get("approval_status") not in {
        "pending_explicit_approval",
        "approved",
    }:
        raise SupplementalK2Error("invalid extended budget policy")
    if policy["approval_status"] == "pending_explicit_approval":
        if policy != {
            "approval_reference": None,
            "approval_status": "pending_explicit_approval",
            "approved": False,
            "per_pair_time_limit_seconds": None,
        }:
            raise SupplementalK2Error("pending budget policy drift")
    elif (
        policy.get("approved") is not True
        or not policy.get("approval_reference")
        or not isinstance(policy.get("per_pair_time_limit_seconds"), int)
        or policy["per_pair_time_limit_seconds"] <= 0
    ):
        raise SupplementalK2Error("approved budget policy is incomplete")
    return payload, formal.sha256_file(manifest_path)


@lru_cache(maxsize=1)
def _current_frozen_placements() -> dict[str, dict[str, list[list[str]] | list[str]]]:
    from src.hybrid_k2_section_pair_screening import load_target_context_and_domains

    _, domains, _ = load_target_context_and_domains()
    section_ids = sorted(
        {section_id for pair in EXPECTED_PAIRS for section_id in pair["section_ids"]}
    )
    return {
        section_id: {
            "original": list(next(option.placement for option in domains[section_id] if option.is_original)),
            "non_original": [
                list(option.placement) for option in domains[section_id] if not option.is_original
            ],
        }
        for section_id in section_ids
    }


def verify_source_formal_artifact(manifest: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(manifest["source_formal_artifact"])
    verification = formal.verify_checksums(root)
    observed = (
        verification["file_count"],
        verification["checksum_entry_count"],
        verification["sha256sums_sha256"],
    )
    if observed != (SOURCE_FILE_COUNT, SOURCE_CHECKSUM_ENTRIES, SOURCE_SHA256SUMS_HASH):
        raise SupplementalK2Error("source formal artifact identity drift")
    source_manifest = _read_json(root / "manifest.json")
    ordering = _read_json(root / "ordering.json")
    checkpoint = _read_json(root / "checkpoint.json")
    rows = formal.read_csv(root / "planned_pairs.csv")
    if source_manifest.get("manifest_hash") != SOURCE_MANIFEST_HASH:
        raise SupplementalK2Error("source formal manifest hash drift")
    if ordering.get("ordering_hash") != SOURCE_ORDERING_HASH:
        raise SupplementalK2Error("source formal ordering hash drift")
    if checkpoint.get("pair_rows_hash") != SOURCE_PAIR_ROWS_HASH:
        raise SupplementalK2Error("source formal pair rows hash drift")
    if len(rows) != 12 or len(checkpoint.get("pair_states", [])) != 12:
        raise SupplementalK2Error("source formal pair universe drift")
    if [int(row["formal_order"]) for row in rows] != list(range(1, 13)) or [
        row["pair_id"] for row in rows
    ] != ordering.get("pair_ids"):
        raise SupplementalK2Error("source formal planned pair order drift")
    states = checkpoint["pair_states"]
    first, second = states[:2]
    if not (
        checkpoint.get("total_solver_invocations") == 2
        and first.get("formal_order") == 1
        and first.get("state") == "artifact_failure"
        and first.get("result_classification") == "artifact_failure"
        and first.get("original_solver_invocation_consumed") is True
        and first.get("response_hash") is None
        and not (root / "runs" / first["pair_id"]).exists()
        and second.get("formal_order") == 2
        and second.get("state") == "completed"
        and second.get("result_classification") == "unresolved_unknown_no_incumbent"
    ):
        raise SupplementalK2Error("source unresolved pair state drift")
    if any(
        state.get("state") != "planned"
        or state.get("result_classification") != "planned_not_run"
        for state in states[2:]
    ):
        raise SupplementalK2Error("Orders 3-12 are not frozen as not-yet-run")
    second_result_path = root / "runs" / second["pair_id"] / "run_result.json"
    second_result = _read_json(second_result_path)
    if (
        second_result.get("result_classification") != "unresolved_unknown_no_incumbent"
        or second_result.get("response_hash") != second.get("response_hash")
        or second_result.get("response_hash_verified") is not True
    ):
        raise SupplementalK2Error("source Order 2 evidence drift")
    pairs = []
    current_placements = _current_frozen_placements()
    for expected in EXPECTED_PAIRS:
        index = expected["original_formal_order"] - 1
        row = rows[index]
        state = states[index]
        observed_pair = {
            "original_formal_order": index + 1,
            "original_placements": expected["original_placements"],
            "frozen_non_original_destination_domains": expected[
                "frozen_non_original_destination_domains"
            ],
            "pair_id": row["pair_id"],
            "section_ids": [row["section_id_a"], row["section_id_b"]],
            "original_classification": state["result_classification"],
            "reason": expected["reason"],
        }
        if observed_pair != expected:
            raise SupplementalK2Error("supplemental pair identity/classification drift")
        for section_id in expected["section_ids"]:
            if (
                current_placements[section_id]["original"]
                != expected["original_placements"][section_id]
                or current_placements[section_id]["non_original"]
                != expected["frozen_non_original_destination_domains"][section_id]
            ):
                raise SupplementalK2Error("current frozen destination domain drift")
        pairs.append(
            dict(row)
            | {
                "original_formal_order": index + 1,
                "original_classification": state["result_classification"],
                "supplemental_inclusion_reason": expected["reason"],
                "original_evidence_reference": (
                    "checkpoint.json#pair_states[0]"
                    if index == 0
                    else f"runs/{row['pair_id']}/run_result.json"
                ),
                "original_response_hash": state.get("response_hash"),
                "original_placements": expected["original_placements"],
                "frozen_non_original_destination_domains": expected[
                    "frozen_non_original_destination_domains"
                ],
            }
        )
    return {
        "artifact_root": str(root),
        "verification": verification,
        "source_manifest_hash": SOURCE_MANIFEST_HASH,
        "source_ordering_hash": SOURCE_ORDERING_HASH,
        "source_pair_rows_hash": SOURCE_PAIR_ROWS_HASH,
        "source_total_solver_invocations": 2,
        "original_run_preserved": True,
        "original_result_not_overwritten": True,
        "original_invocation_not_reclassified": True,
        "not_yet_run_formal_orders": list(range(3, 13)),
        "pairs": pairs,
    }


def supplemental_run_config(run: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    policy = manifest["extended_budget_policy"]
    return {
        "response_hash_namespace": RESPONSE_HASH_NAMESPACE,
        "supplemental_experiment": True,
        "pair_id": run["pair_id"],
        "original_formal_order": int(run["original_formal_order"]),
        "fixed_section_ids": [run["section_id_a"], run["section_id_b"]],
        "destination_domain_sizes": [
            int(run["destination_domain_size_a"]),
            int(run["destination_domain_size_b"]),
        ],
        "frozen_non_original_destination_domains": run[
            "frozen_non_original_destination_domains"
        ],
        "seed": manifest["seed"],
        "workers": manifest["workers"],
        "time_limit_seconds": policy["per_pair_time_limit_seconds"],
        "budget_approval_reference": policy["approval_reference"],
        "hint": "none",
        "assignment_hint": "none",
        "objective": "none",
        "candidate_pruning": False,
        "both_selected_sections_forced_changed": True,
        "all_other_editable_sections_fixed_original": True,
        "full_frozen_non_original_destination_domains": True,
        "stop_after_first_solution": True,
        "run_type": "supplemental_fixed_pair_run_a_feasibility_only",
    }


def planned_runs(source: Mapping[str, Any], manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    runs = []
    for supplemental_order, pair in enumerate(source["pairs"], 1):
        run = {
            "supplemental_order": supplemental_order,
            "pair_id": pair["pair_id"],
            "section_id_a": pair["section_id_a"],
            "section_id_b": pair["section_id_b"],
            "course_id_a": pair["course_id_a"],
            "course_id_b": pair["course_id_b"],
            "original_formal_order": int(pair["original_formal_order"]),
            "original_classification": pair["original_classification"],
            "supplemental_inclusion_reason": pair["supplemental_inclusion_reason"],
            "original_evidence_reference": pair["original_evidence_reference"],
            "original_response_hash": pair["original_response_hash"],
            "original_placements": pair["original_placements"],
            "frozen_non_original_destination_domains": pair[
                "frozen_non_original_destination_domains"
            ],
            "destination_domain_size_a": int(pair["destination_domain_size_a"]),
            "destination_domain_size_b": int(pair["destination_domain_size_b"]),
            "placement_combination_count": int(pair["placement_combination_count"]),
            "supplemental_result": "supplemental_planned_not_run",
        }
        run["config_fingerprint"] = formal.json_hash(supplemental_run_config(run, manifest))
        runs.append(run)
    return runs


def _initial_checkpoint(
    runs: Sequence[Mapping[str, Any]], manifest_hash: str, source: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_hash": manifest_hash,
        "source_formal_sha256sums_hash": source["verification"]["sha256sums_sha256"],
        "total_solver_invocations": 0,
        "pair_states": [
            {
                "supplemental_order": run["supplemental_order"],
                "pair_id": run["pair_id"],
                "state": "planned",
                "result_classification": "supplemental_planned_not_run",
                "config_fingerprint": run["config_fingerprint"],
                "response_hash": None,
                "original_formal_order": run["original_formal_order"],
                "original_classification": run["original_classification"],
            }
            for run in runs
        ],
    }


def _validate_checkpoint(
    checkpoint: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
    manifest_hash: str,
    source: Mapping[str, Any],
    output: Path,
) -> None:
    if checkpoint.get("manifest_hash") != manifest_hash or checkpoint.get(
        "source_formal_sha256sums_hash"
    ) != source["verification"]["sha256sums_sha256"]:
        raise SupplementalK2Error("resume source/manifest fingerprint drift")
    states = checkpoint.get("pair_states", [])
    if len(states) != PAIR_COUNT:
        raise SupplementalK2Error("checkpoint pair universe drift")
    for state, run in zip(states, runs):
        if (
            state.get("pair_id") != run["pair_id"]
            or state.get("config_fingerprint") != run["config_fingerprint"]
            or state.get("original_classification") != run["original_classification"]
        ):
            raise SupplementalK2Error("checkpoint pair/config/original-result drift")
        if state.get("state") in {"running", "artifact_failure"}:
            raise SupplementalK2Error(f"state forbids automatic rerun: {run['pair_id']}")
        if state.get("state") == "completed":
            run_dir = output / "runs" / run["pair_id"]
            result = _read_json(run_dir / "run_result.json")
            config = _read_json(run_dir / "solver_config.json")
            if formal.json_hash(config) != run["config_fingerprint"]:
                raise SupplementalK2Error("persisted supplemental config drift")
            if (
                result.get("response_hash") != state.get("response_hash")
                or result.get("response_hash_namespace") != RESPONSE_HASH_NAMESPACE
            ):
                raise SupplementalK2Error("persisted supplemental response drift")
    completed = sum(state.get("state") == "completed" for state in states)
    if checkpoint.get("total_solver_invocations") != completed:
        raise SupplementalK2Error("supplemental invocation count drift")


def _summary(
    *, dry_run: bool, checkpoint: Mapping[str, Any], manifest_hash: str, source: Mapping[str, Any]
) -> dict[str, Any]:
    counts = {classification: 0 for classification in CLASSIFICATIONS}
    for state in checkpoint["pair_states"]:
        counts[state["result_classification"]] += 1
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_name": "supplemental_k2_unresolved_pairs",
        "supplemental_experiment": True,
        "mode": "dry_run" if dry_run else "execution",
        "manifest_hash": manifest_hash,
        "source_formal_artifact_sha256sums_hash": source["verification"]["sha256sums_sha256"],
        "supplemental_pair_count": PAIR_COUNT,
        "solver_invocations": checkpoint["total_solver_invocations"],
        "pair_result_counts": counts,
        "original_run_preserved": True,
        "original_result_not_overwritten": True,
        "original_invocation_not_reclassified": True,
        "additional_independent_evidence_only": True,
        "supplemental_evidence_applied_to_global_proof": False,
        "finalization_performed": False,
        "global_k2_status": "unresolved",
        "proven_lower_bound": 2,
        "exact_minimum_claim": None,
        "not_yet_run_formal_orders": list(range(3, 13)),
    }


def _write_base_artifact(
    output: Path,
    manifest: Mapping[str, Any],
    manifest_hash: str,
    source: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
    checkpoint: Mapping[str, Any],
    *,
    dry_run: bool,
) -> dict[str, Any]:
    formal.write_json(output / "manifest.json", dict(manifest) | {"manifest_hash": manifest_hash})
    formal.write_json(
        output / "source_audit.json",
        {key: value for key, value in source.items() if key != "pairs"},
    )
    formal.write_csv(output / "planned_pairs.csv", runs)
    formal.write_json(output / "checkpoint.json", checkpoint)
    summary = _summary(
        dry_run=dry_run,
        checkpoint=checkpoint,
        manifest_hash=manifest_hash,
        source=source,
    )
    formal.write_json(output / "aggregate_summary.json", summary)
    formal.write_json(
        output / "provenance.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "supplemental_experiment": True,
            "mode": summary["mode"],
            "source_formal_artifact": source["artifact_root"],
            "source_formal_artifact_read_only": True,
            "original_run_preserved": True,
            "original_result_not_overwritten": True,
            "original_invocation_not_reclassified": True,
            "response_hash_namespace": RESPONSE_HASH_NAMESPACE,
            "finalization_is_separate": True,
            "real_solver_invocations": checkpoint["total_solver_invocations"],
        },
    )
    if not (output / "failures.json").exists():
        formal.write_json(output / "failures.json", {"failure_count": 0, "failures": []})
    formal.write_checksums(output)
    return summary


def _classify(result: Mapping[str, Any]) -> tuple[str, bool]:
    status = str(result.get("status", ""))
    incumbent = bool(result.get("incumbent_found") or result.get("assignment_available"))
    if incumbent:
        return "supplemental_incumbent_pending_validation", True
    if status == "INFEASIBLE":
        return "supplemental_fixed_pair_infeasible", False
    if status == "UNKNOWN":
        return "supplemental_unresolved_unknown_no_incumbent", True
    if status == "MODEL_INVALID":
        return "supplemental_model_invalid", True
    return "supplemental_artifact_failure", True


def _supplemental_response_hash(run: Mapping[str, Any], source_hash: str) -> str:
    return formal.json_hash(
        {
            "namespace": RESPONSE_HASH_NAMESPACE,
            "pair_id": run["pair_id"],
            "original_formal_order": run["original_formal_order"],
            "source_response_hash": source_hash,
        }
    )


def _write_run(
    output: Path,
    run: Mapping[str, Any],
    result: Mapping[str, Any],
    classification: str,
    *,
    manifest_hash: str,
    source: Mapping[str, Any],
) -> str:
    config = result["solver_config"]
    if formal.json_hash(config) != run["config_fingerprint"]:
        raise SupplementalK2Error("supplemental solver config fingerprint drift")
    domains = result["destination_domains"]
    if domains != run["frozen_non_original_destination_domains"]:
        raise SupplementalK2Error("supplemental destination domain drift")
    raw_hash = result.get("response_hash")
    if not raw_hash or result.get("response_hash_verified") is not True:
        raise SupplementalK2Error("supplemental raw response hash unavailable")
    response_hash = _supplemental_response_hash(run, str(raw_hash))
    response_stats = dict(result["response_stats"])
    response_stats.update(
        {
            "source_response_hash": raw_hash,
            "source_response_hash_verified": True,
            "response_hash": response_hash,
            "response_hash_verified": True,
            "response_hash_namespace": RESPONSE_HASH_NAMESPACE,
        }
    )
    run_dir = output / "runs" / run["pair_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    formal.write_json(run_dir / "solver_config.json", config)
    formal.write_json(run_dir / "destination_domains.json", domains)
    formal.write_json(run_dir / "model_fingerprint.json", result["model_fingerprint"])
    formal.write_json(run_dir / "response_stats.json", response_stats)
    formal._atomic_write_text(run_dir / "solver.log", "".join(result.get("solver_log", ())))
    formal.write_json(run_dir / "validation.json", result.get("validation", {"validated": False}))
    formal.write_json(run_dir / "hint_audit.json", result.get("hint_audit", {}))
    formal.write_json(
        run_dir / "provenance.json",
        {
            "supplemental_experiment": True,
            "manifest_hash": manifest_hash,
            "source_formal_artifact_sha256sums_hash": source["verification"]["sha256sums_sha256"],
            "pair_id": run["pair_id"],
            "original_formal_order": run["original_formal_order"],
            "original_classification": run["original_classification"],
            "original_evidence_reference": run["original_evidence_reference"],
            "original_result_not_overwritten": True,
            "additional_independent_evidence": True,
            "response_hash_namespace": RESPONSE_HASH_NAMESPACE,
            "run_b_allowed": False,
            "automatic_rerun_allowed": False,
        },
    )
    formal.write_json(
        run_dir / "run_result.json",
        {
            "supplemental_experiment": True,
            "pair_id": run["pair_id"],
            "original_formal_order": run["original_formal_order"],
            "original_classification": run["original_classification"],
            "original_result_not_overwritten": True,
            "result_classification": classification,
            "config_fingerprint": run["config_fingerprint"],
            "source_response_hash": raw_hash,
            "response_hash": response_hash,
            "response_hash_verified": True,
            "response_hash_namespace": RESPONSE_HASH_NAMESPACE,
            "global_k2_status": "unresolved",
            "proven_lower_bound": 2,
            "exact_minimum_claim": None,
            "finalization_performed": False,
        },
    )
    return response_hash


def _real_solver_runner(run: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Run one independent supplemental search; imported lazily for dry-run safety."""
    from src.allocation import math_course_ids_from_catalog
    from src.benchmark_runner import _load_math_fallback_rules
    from src.hybrid_k2_section_pair_screening import (
        _response_payload,
        fixed_pair_feasibility_run,
        load_target_context_and_domains,
    )
    from src.hybrid_stage1_incumbent_bootstrap import validate_bootstrap_witness
    from src.period_placement_repair_probe import CandidateEdit

    context, domains, _ = load_target_context_and_domains()
    section_ids = (run["section_id_a"], run["section_id_b"])
    original, proposed, domain_rows = [], [], []
    for section_id in section_ids:
        options = domains[section_id]
        original.append(next(option.placement for option in options if option.is_original))
        non_original = tuple(option.placement for option in options if not option.is_original)
        proposed.append(non_original[0])
        domain_rows.append([list(value) for value in non_original])
    observed_domains = dict(zip(section_ids, domain_rows))
    if observed_domains != run["frozen_non_original_destination_domains"]:
        raise SupplementalK2Error("real-run frozen destination domain drift")
    candidate = CandidateEdit(
        candidate_id=f"supplemental_k2:{run['supplemental_order']}:{run['pair_id']}",
        edit_type="supplemental_k2_full_destination_domain",
        logical_section_ids=section_ids,
        logical_course_ids=(run["course_id_a"], run["course_id_b"]),
        original_placements=tuple(original),
        proposed_placements=tuple(proposed),
        valid_period_source="supplemental frozen full non-original destination domain",
        occupancy_shape=tuple(len(value) for value in original),
        core_student="all_students_production_model",
        core_period_relevance=tuple(
            sorted({period for values in (*original, *proposed) for period in values})
        ),
        affected_candidate_edge_count=0,
        affected_student_count=0,
    )
    rules = _load_math_fallback_rules(Path("data/config"), context.catalog)
    build, hint_audit, search = fixed_pair_feasibility_run(
        context.allocation_input,
        domains,
        candidate,
        math_fallback_rules=rules,
        math_course_ids=math_course_ids_from_catalog(context.catalog),
        seed=int(manifest["seed"]),
        time_limit_seconds=float(
            manifest["extended_budget_policy"]["per_pair_time_limit_seconds"]
        ),
    )
    proto_bytes = deterministic_model_proto_bytes(build.model)
    response = _response_payload(search)
    validation = {"validated": False, "reason": "no incumbent"}
    if search.incumbent_found:
        validation = validate_bootstrap_witness(
            context, build, search, config_dir=Path("data/config"), k=2
        )
    return {
        "status": search.status,
        "incumbent_found": search.incumbent_found,
        "assignment_available": search.assignment_available,
        "response_hash": response.get("response_hash"),
        "response_hash_verified": response.get("response_hash_verified", False),
        "response_stats": response,
        "solver_log": search.solver_log,
        "solver_config": supplemental_run_config(run, manifest),
        "destination_domains": observed_domains,
        "model_fingerprint": {
            "sha256": hashlib.sha256(proto_bytes).hexdigest(),
            "binary_proto_bytes": len(proto_bytes),
            "total_variables": len(build.model.Proto().variables),
            "total_constraints": len(build.model.Proto().constraints),
            "full_model_persisted": False,
        },
        "validation": validation,
        "hint_audit": hint_audit,
    }


def run_experiment(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    output_dir: str | Path,
    dry_run: bool = False,
    execute: bool = False,
    resume: bool = False,
    max_new_solver_runs: int | None = None,
    solver_runner: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if int(dry_run) + int(execute) != 1:
        raise SupplementalK2Error("choose exactly one explicit mode: --dry-run or --execute")
    if max_new_solver_runs is not None and max_new_solver_runs < 0:
        raise SupplementalK2Error("max-new-solver-runs must be nonnegative")
    manifest, manifest_hash = load_manifest(manifest_path)
    source = verify_source_formal_artifact(manifest)
    runs = planned_runs(source, manifest)
    output = Path(output_dir)
    resolved_output = output.resolve()
    resolved_source = Path(source["artifact_root"]).resolve()
    if resolved_output == resolved_source or resolved_source in resolved_output.parents:
        raise SupplementalK2Error("supplemental output must be separate from formal source")
    if dry_run and resolved_output == SUPPLEMENTAL_OUTPUT.resolve():
        raise SupplementalK2Error("dry-run may not write the formal supplemental artifact")
    if execute:
        policy = manifest["extended_budget_policy"]
        if policy["approval_status"] != "approved" or policy["approved"] is not True:
            raise SupplementalK2Error("extended budget is not explicitly approved")
    nonempty = output.exists() and any(output.iterdir())
    if nonempty and not resume:
        raise SupplementalK2Error(f"output is non-empty; use --resume: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if nonempty:
        formal.verify_checksums(output)
        checkpoint = _read_json(output / "checkpoint.json")
        _validate_checkpoint(checkpoint, runs, manifest_hash, source, output)
    else:
        checkpoint = _initial_checkpoint(runs, manifest_hash, source)
    if dry_run:
        if checkpoint["total_solver_invocations"] != 0:
            raise SupplementalK2Error("dry-run cannot resume solver evidence")
        summary = _write_base_artifact(
            output, manifest, manifest_hash, source, runs, checkpoint, dry_run=True
        )
        summary["output_dir"] = str(output)
        summary["output_files"] = sorted(
            str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()
        )
        return summary

    states = checkpoint["pair_states"]
    terminal = next(
        (
            state["result_classification"]
            for state in states
            if state["state"] == "completed"
            and state["result_classification"]
            != "supplemental_fixed_pair_infeasible"
        ),
        None,
    )
    if terminal:
        summary = _summary(
            dry_run=False, checkpoint=checkpoint, manifest_hash=manifest_hash, source=source
        )
        summary.update(new_solver_invocations_this_call=0, stop_reason=terminal)
        return summary
    runner = solver_runner or _real_solver_runner
    new_invocations = 0
    stop_reason = "invocation_limit_reached"
    for state, run in zip(states, runs):
        if state["state"] == "completed":
            continue
        if max_new_solver_runs is not None and new_invocations >= max_new_solver_runs:
            break
        state["state"] = "running"
        checkpoint["total_solver_invocations"] += 1
        new_invocations += 1
        formal.write_json(output / "checkpoint.json", checkpoint)
        formal.write_checksums(output)
        try:
            result = dict(runner(run, manifest))
            classification, must_stop = _classify(result)
            response_hash = _write_run(
                output,
                run,
                result,
                classification,
                manifest_hash=manifest_hash,
                source=source,
            )
            state.update(
                state="completed",
                result_classification=classification,
                response_hash=response_hash,
            )
        except Exception as exc:
            state.update(
                state="artifact_failure",
                result_classification="supplemental_artifact_failure",
                failure=f"{type(exc).__name__}: {exc}",
            )
            formal.write_json(output / "checkpoint.json", checkpoint)
            failures_path = output / "failures.json"
            failures = _read_json(failures_path) if failures_path.is_file() else {"failures": []}
            history = list(failures.get("failures", []))
            history.append(state["failure"])
            formal.write_json(
                failures_path, {"failure_count": len(history), "failures": history}
            )
            formal.write_checksums(output)
            raise SupplementalK2Error(f"supplemental run failed closed: {run['pair_id']}") from exc
        _write_base_artifact(
            output, manifest, manifest_hash, source, runs, checkpoint, dry_run=False
        )
        if must_stop:
            stop_reason = classification
            break
    else:
        stop_reason = "all_supplemental_pairs_completed"
    summary = _summary(
        dry_run=False, checkpoint=checkpoint, manifest_hash=manifest_hash, source=source
    )
    summary.update(
        new_solver_invocations_this_call=new_invocations,
        stop_reason=stop_reason,
    )
    formal.write_json(output / "aggregate_summary.json", summary)
    formal.write_checksums(output)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan or execute the supplemental unresolved-pair K=2 experiment."
    )
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-solver-runs", type=int)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_experiment(
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            execute=args.execute,
            resume=args.resume,
            max_new_solver_runs=args.max_new_solver_runs,
        )
    except (SupplementalK2Error, formal.FormalK2BatchError) as exc:
        print(f"Supplemental unresolved-pair K=2 FAIL: {exc}")
        return 1
    print("Supplemental unresolved-pair K=2 PASS")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
