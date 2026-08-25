"""Fail-closed batch runner for the 12 formal remaining K=2 section pairs.

The pair universe and order come only from the accepted all-student blocker
safe-screen artifact.  Dry-run is solver-free; real execution requires the
explicit ``--execute`` flag and uses the existing production fixed-pair Run A.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src.model_proto_serialization import deterministic_model_proto_bytes


SCHEMA_VERSION = 1
COMPACT_EVIDENCE_SCHEMA_VERSION = 1
FAILURE_CONTINUATION_SCHEMA_VERSION = 1
PAIR_COUNT = 12
BATCH_SIZE = 4
CHECKPOINT_FREQUENCY = 4
GLOBAL_INVOCATION_BUDGET = 12
SOLVER_SEED = 20260630
WORKERS = 1
TIME_LIMIT_SECONDS = 75.0
ORDERING_HASH = "f1246ec8582d6925e26fcc9ee53583a7ac275d6063781b2f59865af788573927"
RESULT_HASH = "244c39b79d3b61c379d386aec203fb2bd73395f1af44ebc575c2f9d500169af5"
PROOF_HASH = "853223911427d9decccff42496022f0030c7cf0eba9e4fcc1184e6eec68f31b1"
SOURCE_SHA256SUMS_HASH = "9205ad79ff3248ac578362ca113cf0e1110a4bd677859b96c73eb5122416db95"
SOURCE_FILE_COUNT = 11
SOURCE_CHECKSUM_ENTRIES = 10
DEFAULT_MANIFEST = Path("data/scenarios/formal_remaining_k2_batch_v1.json")
DEFAULT_SOURCE_ARTIFACT = Path(
    "../fair-course-allocation-artifacts/robustness-v1/"
    "all-student-k2-blocker-safe-screen-v1"
)
FORMAL_OUTPUT = Path(
    "../fair-course-allocation-artifacts/robustness-v1/"
    "formal-remaining-k2-batch-v1"
)
FAILURE_CONTINUATION_FILE = "failure_continuation_authorization.json"
CONTINUATION_PROVENANCE_FILE = "failure_continuation_provenance.json"
CONTINUATION_CHUNKS = ((2, 3, 4, 5), (6, 7, 8, 9), (10, 11, 12))
ALLOWED_PAIR_RESULTS = {
    "fixed_pair_infeasible",
    "incumbent_pending_validation",
    "unresolved_unknown_no_incumbent",
    "model_invalid",
    "artifact_failure",
    "planned_not_run",
}
ANOMALY_MODEL_REASONS = {
    "model_invalid",
    "artifact_failure_debug_recovery",
    "configuration_fingerprint_mismatch",
    "unexpected_response_status",
    "explicit_reproducibility_escalation",
}


class FormalK2BatchError(ValueError):
    """Raised when the frozen batch protocol cannot proceed safely."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FormalK2BatchError(f"cannot read JSON: {path}") from exc


def read_csv(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    except OSError as exc:
        raise FormalK2BatchError(f"cannot read CSV: {path}") from exc


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, value: Any) -> None:
    _atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise FormalK2BatchError(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_checksums(root: Path) -> str:
    checksum = root / "SHA256SUMS.txt"
    lines = [
        f"{sha256_file(path)}  {path.relative_to(root)}"
        for path in sorted(root.rglob("*"))
        if path.is_file() and path != checksum
    ]
    _atomic_write_text(checksum, "\n".join(lines) + "\n")
    return sha256_file(checksum)


def verify_checksums(root: Path) -> dict[str, Any]:
    checksum = root / "SHA256SUMS.txt"
    if not checksum.is_file():
        raise FormalK2BatchError(f"missing SHA256SUMS.txt: {root}")
    entries: dict[str, str] = {}
    for number, line in enumerate(checksum.read_text(encoding="utf-8").splitlines(), 1):
        try:
            expected, relative_text = line.split("  ", 1)
        except ValueError as exc:
            raise FormalK2BatchError(f"malformed checksum line {number}") from exc
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts or relative_text in entries:
            raise FormalK2BatchError(f"unsafe or duplicate checksum path: {relative_text}")
        entries[relative_text] = expected
    files = sorted(path for path in root.rglob("*") if path.is_file())
    uncovered = [
        str(path.relative_to(root))
        for path in files
        if path != checksum and str(path.relative_to(root)) not in entries
    ]
    failures = [
        relative
        for relative, expected in entries.items()
        if not (root / relative).is_file() or sha256_file(root / relative) != expected
    ]
    if uncovered or failures or len(files) != len(entries) + 1:
        raise FormalK2BatchError(
            f"checksum verification failed: failures={failures}, uncovered={uncovered}"
        )
    return {
        "passed": True,
        "file_count": len(files),
        "checksum_entry_count": len(entries),
        "sha256sums_sha256": sha256_file(checksum),
    }


def load_manifest(path: str | Path = DEFAULT_MANIFEST) -> tuple[dict[str, Any], str]:
    manifest_path = Path(path)
    payload = read_json(manifest_path)
    required = {
        "manifest_schema_version",
        "experiment_name",
        "experiment_version",
        "phase",
        "formal_input_artifact",
        "source_artifact_file_count",
        "source_artifact_checksum_entry_count",
        "source_artifact_sha256sums_hash",
        "source_proof_hash",
        "result_hash",
        "max_pair_count",
        "ordering_hash",
        "seed",
        "workers",
        "per_pair_time_limit_seconds",
        "batch_size",
        "checkpoint_frequency",
        "global_invocation_budget",
        "hint",
        "objective",
        "candidate_pruning",
        "feasibility_only",
        "stop_after_first_solution",
        "run_b_allowed",
        "no_automatic_rerun",
        "both_selected_sections_forced_changed",
        "other_editable_sections_fixed_original",
        "complete_frozen_non_original_destination_domains",
        "compact_evidence_enabled",
        "compact_evidence_schema_version",
        "full_model_proto_default",
        "anomaly_model_proto_escalation_enabled",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise FormalK2BatchError("manifest missing: " + ", ".join(missing))
    expected_values = {
        "manifest_schema_version": SCHEMA_VERSION,
        "experiment_name": "formal_remaining_k2_batch",
        "experiment_version": "v1",
        "phase": "remaining_fixed_pair_run_a_batch",
        "formal_input_artifact": str(DEFAULT_SOURCE_ARTIFACT),
        "source_artifact_file_count": SOURCE_FILE_COUNT,
        "source_artifact_checksum_entry_count": SOURCE_CHECKSUM_ENTRIES,
        "source_artifact_sha256sums_hash": SOURCE_SHA256SUMS_HASH,
        "source_proof_hash": PROOF_HASH,
        "result_hash": RESULT_HASH,
        "max_pair_count": PAIR_COUNT,
        "ordering_hash": ORDERING_HASH,
        "seed": SOLVER_SEED,
        "workers": WORKERS,
        "per_pair_time_limit_seconds": int(TIME_LIMIT_SECONDS),
        "batch_size": BATCH_SIZE,
        "checkpoint_frequency": CHECKPOINT_FREQUENCY,
        "global_invocation_budget": GLOBAL_INVOCATION_BUDGET,
        "hint": False,
        "objective": False,
        "candidate_pruning": False,
        "feasibility_only": True,
        "stop_after_first_solution": True,
        "run_b_allowed": False,
        "no_automatic_rerun": True,
        "both_selected_sections_forced_changed": True,
        "other_editable_sections_fixed_original": True,
        "complete_frozen_non_original_destination_domains": True,
        "compact_evidence_enabled": True,
        "compact_evidence_schema_version": COMPACT_EVIDENCE_SCHEMA_VERSION,
        "full_model_proto_default": False,
        "anomaly_model_proto_escalation_enabled": True,
    }
    drift = [key for key, expected in expected_values.items() if payload.get(key) != expected]
    if drift:
        raise FormalK2BatchError(f"frozen manifest drift: {drift}")
    return payload, sha256_file(manifest_path)


def _ordering_hash(rows: Sequence[Mapping[str, str]]) -> str:
    ranking_records = [
        {
            "pair_id": row["pair_id"],
            "core_feasible_placement_combinations": int(row["core_feasible_placement_combinations"]),
            "affected_student_union_count": int(row["affected_student_union_count"]),
            "changed_candidate_period_relationships": int(row["changed_candidate_period_relationships"]),
            "total_absolute_period_displacement": int(row["total_absolute_period_displacement"]),
        }
        for row in rows
    ]
    return json_hash(ranking_records)


def verify_formal_input(manifest: Mapping[str, Any]) -> dict[str, Any]:
    root = Path(manifest["formal_input_artifact"])
    verification = verify_checksums(root)
    expected_verification = (
        SOURCE_FILE_COUNT,
        SOURCE_CHECKSUM_ENTRIES,
        SOURCE_SHA256SUMS_HASH,
    )
    observed_verification = (
        verification["file_count"],
        verification["checksum_entry_count"],
        verification["sha256sums_sha256"],
    )
    if observed_verification != expected_verification:
        raise FormalK2BatchError("formal input artifact identity drift")
    summary = read_json(root / "aggregate_summary.json")
    proof = read_json(root / "proof_audit.json")
    rows = read_csv(root / "blocker_screen_survivors.csv")
    if len(rows) != PAIR_COUNT or [int(row["formal_order"]) for row in rows] != list(range(1, PAIR_COUNT + 1)):
        raise FormalK2BatchError("formal pair count/order drift")
    if len({row["pair_id"] for row in rows}) != PAIR_COUNT:
        raise FormalK2BatchError("duplicate formal pair ID")
    if any(row["classification"] != "blocker_screen_survivor" or row["feasibility_claim"] != "none" for row in rows):
        raise FormalK2BatchError("formal pair claim drift")
    ordering_hash = _ordering_hash(rows)
    if ordering_hash != manifest["ordering_hash"] or summary.get("formal_remaining_pair_ordering_hash") != ordering_hash:
        raise FormalK2BatchError("formal ordering hash drift")
    if summary.get("screening_diagnostics", {}).get("result_hash") != manifest["result_hash"]:
        raise FormalK2BatchError("safe-screen result hash drift")
    if proof.get("proof_hash") != manifest["source_proof_hash"] or proof.get("proof_verified") is not True:
        raise FormalK2BatchError("blocker proof hash/status drift")
    return {
        "artifact_root": str(root),
        "verification": verification,
        "ordering_hash": ordering_hash,
        "result_hash": summary["screening_diagnostics"]["result_hash"],
        "proof_hash": proof["proof_hash"],
        "pair_count": len(rows),
        "pair_rows_hash": json_hash(rows),
        "pairs": rows,
    }


def frozen_run_config(run: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "pair_id": run["pair_id"],
        "fixed_section_ids": [run["section_id_a"], run["section_id_b"]],
        "destination_domain_sizes": [
            int(run["destination_domain_size_a"]),
            int(run["destination_domain_size_b"]),
        ],
        "seed": manifest["seed"],
        "workers": manifest["workers"],
        "time_limit_seconds": manifest["per_pair_time_limit_seconds"],
        "hint": "none",
        "assignment_hint": "none",
        "objective": "none",
        "candidate_pruning": False,
        "both_selected_sections_forced_changed": True,
        "all_other_editable_sections_fixed_original": True,
        "full_frozen_non_original_destination_domains": True,
        "stop_after_first_solution": True,
        "run_type": "fixed_pair_run_a_feasibility_only",
    }


def planned_runs(source: Mapping[str, Any], manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = []
    for row in source["pairs"]:
        order = int(row["formal_order"])
        domain_a = int(row["destination_domain_size_a"])
        domain_b = int(row["destination_domain_size_b"])
        if domain_a * domain_b != int(row["total_placement_combinations"]):
            raise FormalK2BatchError(f"placement combination drift: {row['pair_id']}")
        run = {
            "pair_id": row["pair_id"],
            "section_id_a": row["section_id_a"],
            "section_id_b": row["section_id_b"],
            "destination_domain_size_a": domain_a,
            "destination_domain_size_b": domain_b,
        }
        config = frozen_run_config(run, manifest)
        output.append(
            {
                "formal_order": order,
                "batch_index": (order - 1) // BATCH_SIZE + 1,
                "batch_position": (order - 1) % BATCH_SIZE + 1,
                "pair_id": row["pair_id"],
                "section_id_a": row["section_id_a"],
                "section_id_b": row["section_id_b"],
                "course_id_a": row["course_id_a"],
                "course_id_b": row["course_id_b"],
                "original_placement_a": row["original_placement_a"],
                "original_placement_b": row["original_placement_b"],
                "destination_domain_size_a": domain_a,
                "destination_domain_size_b": domain_b,
                "placement_combination_count": int(row["total_placement_combinations"]),
                "config_fingerprint": json_hash(config),
                "planned_result": "planned_not_run",
            }
        )
    return output


def _initial_checkpoint(
    runs: Sequence[Mapping[str, Any]], manifest_hash: str, source: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "manifest_hash": manifest_hash,
        "source_sha256sums_hash": source["verification"]["sha256sums_sha256"],
        "ordering_hash": source["ordering_hash"],
        "pair_rows_hash": source["pair_rows_hash"],
        "total_solver_invocations": 0,
        "completed_batch_indices": [],
        "pair_states": [
            {
                "formal_order": run["formal_order"],
                "pair_id": run["pair_id"],
                "state": "planned",
                "result_classification": "planned_not_run",
                "config_fingerprint": run["config_fingerprint"],
                "response_hash": None,
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
    *,
    approved_failure_orders: frozenset[int] = frozenset(),
) -> None:
    expected = {
        "manifest_hash": manifest_hash,
        "source_sha256sums_hash": source["verification"]["sha256sums_sha256"],
        "ordering_hash": source["ordering_hash"],
        "pair_rows_hash": source["pair_rows_hash"],
    }
    drift = [key for key, value in expected.items() if checkpoint.get(key) != value]
    if drift:
        raise FormalK2BatchError(f"resume fingerprint drift: {drift}")
    states = checkpoint.get("pair_states", [])
    if len(states) != len(runs):
        raise FormalK2BatchError("checkpoint pair universe drift")
    for state, run in zip(states, runs):
        if state.get("pair_id") != run["pair_id"] or state.get("config_fingerprint") != run["config_fingerprint"]:
            raise FormalK2BatchError("checkpoint pair/config fingerprint drift")
        if state.get("state") == "running":
            raise FormalK2BatchError(f"interrupted run requires manual recovery: {run['pair_id']}")
        if state.get("state") == "artifact_failure":
            if int(run["formal_order"]) not in approved_failure_orders:
                raise FormalK2BatchError(f"artifact failure forbids automatic rerun: {run['pair_id']}")
            if state.get("response_hash") is not None or (output / "runs" / run["pair_id"]).exists():
                raise FormalK2BatchError("approved artifact failure gained response evidence")
        if state.get("state") == "completed":
            run_dir = output / "runs" / run["pair_id"]
            result = read_json(run_dir / "run_result.json")
            response = read_json(run_dir / "response_stats.json")
            solver_config = read_json(run_dir / "solver_config.json")
            if result.get("config_fingerprint") != run["config_fingerprint"]:
                raise FormalK2BatchError("persisted config fingerprint drift")
            if json_hash(solver_config) != run["config_fingerprint"]:
                raise FormalK2BatchError("persisted solver config fingerprint drift")
            if result.get("response_hash") != state.get("response_hash") or response.get("response_hash") != state.get("response_hash"):
                raise FormalK2BatchError("persisted response fingerprint drift")
    total = int(checkpoint.get("total_solver_invocations", -1))
    if total < 0 or total > GLOBAL_INVOCATION_BUDGET:
        raise FormalK2BatchError("total invocation budget drift")


def _proof_blockers(checkpoint: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "formal_order": int(state["formal_order"]),
            "pair_id": state["pair_id"],
            "classification": "artifact_failure",
            "excluded_from_global_k2_proof": True,
        }
        for state in checkpoint["pair_states"]
        if state["result_classification"] == "artifact_failure"
    ]


def _validate_failure_continuation_authorization(
    output: Path,
    checkpoint: Mapping[str, Any],
    runs: Sequence[Mapping[str, Any]],
    manifest_hash: str,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    path = output / FAILURE_CONTINUATION_FILE
    if not path.is_file():
        raise FormalK2BatchError("approved failure continuation authorization is missing")
    authorization = read_json(path)
    observed_hash = authorization.get("authorization_hash")
    unsigned = {key: value for key, value in authorization.items() if key != "authorization_hash"}
    if not observed_hash or observed_hash != json_hash(unsigned):
        raise FormalK2BatchError("failure continuation authorization hash mismatch")
    failed_run = runs[0]
    failed_state = checkpoint["pair_states"][0]
    expected = {
        "schema_version": FAILURE_CONTINUATION_SCHEMA_VERSION,
        "artifact_path": str(output.resolve()),
        "manifest_hash": manifest_hash,
        "ordering_hash": source["ordering_hash"],
        "failed_pair_order": 1,
        "failed_pair_id": failed_run["pair_id"],
        "failed_pair_section_ids": [failed_run["section_id_a"], failed_run["section_id_b"]],
        "original_classification": "artifact_failure",
        "original_failure_reason": failed_state.get("failure"),
        "original_invocation_consumed": True,
        "total_solver_invocations_before_continuation": 1,
        "solver_rerun_authorized": False,
        "failed_pair_may_be_skipped_for_execution": True,
        "failed_pair_may_be_used_as_feasibility_evidence": False,
        "failed_pair_excluded_from_global_proof": True,
        "next_executable_order": 2,
        "remaining_executable_orders": list(range(2, PAIR_COUNT + 1)),
        "remaining_invocation_budget": GLOBAL_INVOCATION_BUDGET - 1,
        "continuation_chunks": [list(chunk) for chunk in CONTINUATION_CHUNKS],
        "global_k2_status": "unresolved",
        "proven_lower_bound": 2,
        "exact_minimum_claim": None,
        "source_hashes": {
            "sha256sums": source["verification"]["sha256sums_sha256"],
            "ordering": source["ordering_hash"],
            "result": source["result_hash"],
            "proof": source["proof_hash"],
            "pair_rows": source["pair_rows_hash"],
        },
    }
    drift = [key for key, value in expected.items() if authorization.get(key) != value]
    if drift:
        raise FormalK2BatchError(f"failure continuation authorization drift: {drift}")
    if failed_state.get("state") != "artifact_failure" or failed_state.get("response_hash") is not None:
        raise FormalK2BatchError("authorized failed pair checkpoint drift")
    failures_path = output / "failures.json"
    failures = read_json(failures_path)
    if (
        authorization.get("original_failures_sha256") != sha256_file(failures_path)
        or failed_state.get("failure") not in failures.get("failures", [])
    ):
        raise FormalK2BatchError("original artifact failure history drift")
    if authorization.get("authorization_reason") in (None, "") or not authorization.get("created_at_utc"):
        raise FormalK2BatchError("failure continuation authorization provenance is incomplete")
    repo_commit_hash = str(authorization.get("repo_commit_hash", ""))
    if len(repo_commit_hash) != 40 or any(character not in "0123456789abcdef" for character in repo_commit_hash):
        raise FormalK2BatchError("failure continuation repository commit is invalid")
    checkpoint_metadata = checkpoint.get("failure_continuation", {})
    if checkpoint_metadata.get("authorization_hash") != observed_hash:
        raise FormalK2BatchError("checkpoint continuation authorization hash mismatch")
    completed_after_failure = [
        state
        for state in checkpoint["pair_states"][1:]
        if state["state"] == "completed"
    ]
    expected_invocations = 1 + len(completed_after_failure)
    if int(checkpoint.get("total_solver_invocations", -1)) != expected_invocations:
        raise FormalK2BatchError("continuation invocation count mismatch")
    terminal = [
        state
        for state in completed_after_failure
        if state["result_classification"] != "fixed_pair_infeasible"
    ]
    if terminal:
        raise FormalK2BatchError("terminal pair result forbids further continuation")
    return authorization


def create_failure_continuation_authorization(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    output_dir: str | Path = FORMAL_OUTPUT,
) -> dict[str, Any]:
    output = Path(output_dir)
    if not output.exists() or not any(output.iterdir()):
        raise FormalK2BatchError("continuation authorization requires an existing artifact")
    verify_checksums(output)
    if (output / FAILURE_CONTINUATION_FILE).exists():
        raise FormalK2BatchError("failure continuation authorization already exists")
    manifest, manifest_hash = load_manifest(manifest_path)
    source = verify_formal_input(manifest)
    runs = planned_runs(source, manifest)
    checkpoint = read_json(output / "checkpoint.json")
    _validate_checkpoint(
        checkpoint,
        runs,
        manifest_hash,
        source,
        output,
        approved_failure_orders=frozenset({1}),
    )
    failed_state = checkpoint["pair_states"][0]
    failed_run = runs[0]
    failure_reason = failed_state.get("failure")
    failures = read_json(output / "failures.json")
    if (
        int(checkpoint["total_solver_invocations"]) != 1
        or failed_state.get("state") != "artifact_failure"
        or failed_state.get("result_classification") != "artifact_failure"
        or failed_state.get("response_hash") is not None
        or not failure_reason
        or failure_reason not in failures.get("failures", [])
    ):
        raise FormalK2BatchError("failed Order 1 evidence is not eligible for continuation")
    if any(state["state"] != "planned" for state in checkpoint["pair_states"][1:]):
        raise FormalK2BatchError("continuation authorization requires Orders 2-12 planned_not_run")
    before = verify_checksums(output)
    authorization = {
        "schema_version": FAILURE_CONTINUATION_SCHEMA_VERSION,
        "artifact_path": str(output.resolve()),
        "manifest_hash": manifest_hash,
        "ordering_hash": source["ordering_hash"],
        "source_hashes": {
            "sha256sums": source["verification"]["sha256sums_sha256"],
            "ordering": source["ordering_hash"],
            "result": source["result_hash"],
            "proof": source["proof_hash"],
            "pair_rows": source["pair_rows_hash"],
        },
        "failed_pair_order": 1,
        "failed_pair_id": failed_run["pair_id"],
        "failed_pair_section_ids": [failed_run["section_id_a"], failed_run["section_id_b"]],
        "original_classification": "artifact_failure",
        "original_failure_reason": failure_reason,
        "original_invocation_consumed": True,
        "total_solver_invocations_before_continuation": 1,
        "solver_rerun_authorized": False,
        "failed_pair_may_be_skipped_for_execution": True,
        "failed_pair_may_be_used_as_feasibility_evidence": False,
        "failed_pair_excluded_from_global_proof": True,
        "next_executable_order": 2,
        "remaining_executable_orders": list(range(2, PAIR_COUNT + 1)),
        "remaining_invocation_budget": GLOBAL_INVOCATION_BUDGET - 1,
        "continuation_chunks": [list(chunk) for chunk in CONTINUATION_CHUNKS],
        "global_k2_status": "unresolved",
        "proven_lower_bound": 2,
        "exact_minimum_claim": None,
        "authorization_reason": (
            "Allow evidence collection for Orders 2-12 without rerunning or using "
            "the failed Order 1 as feasibility evidence."
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repo_commit_hash": _git_head(),
        "original_checkpoint_sha256": sha256_file(output / "checkpoint.json"),
        "original_failures_sha256": sha256_file(output / "failures.json"),
        "original_sha256sums_hash": before["sha256sums_sha256"],
    }
    authorization["authorization_hash"] = json_hash(authorization)
    failed_state.update({
        "original_solver_invocation_consumed": True,
        "solver_rerun_authorized": False,
        "excluded_from_global_k2_proof": True,
        "may_be_skipped_for_execution": True,
        "may_be_used_as_feasibility_evidence": False,
    })
    checkpoint["failure_continuation"] = {
        "schema_version": FAILURE_CONTINUATION_SCHEMA_VERSION,
        "authorization_file": FAILURE_CONTINUATION_FILE,
        "authorization_hash": authorization["authorization_hash"],
        "approved_failed_orders": [1],
        "next_executable_order": 2,
        "remaining_invocation_budget": GLOBAL_INVOCATION_BUDGET - 1,
    }
    write_json(output / FAILURE_CONTINUATION_FILE, authorization)
    write_json(output / "checkpoint.json", checkpoint)
    write_json(output / CONTINUATION_PROVENANCE_FILE, {
        "schema_version": FAILURE_CONTINUATION_SCHEMA_VERSION,
        "created_at_utc": authorization["created_at_utc"],
        "repo_commit_hash": authorization["repo_commit_hash"],
        "authorization_hash": authorization["authorization_hash"],
        "mode": "explicit_manual_failure_continuation_authorization",
        "artifact_recovery": False,
        "solver_result_recovery": False,
        "solver_rerun": False,
        "new_solver_invocations": 0,
        "original_artifact_failure_preserved": True,
        "original_failure_reason": failure_reason,
    })
    checksum_hash = write_checksums(output)
    verification = verify_checksums(output)
    _validate_failure_continuation_authorization(
        output, checkpoint, runs, manifest_hash, source
    )
    return {
        "authorization": authorization,
        "verification": verification,
        "sha256sums_hash": checksum_hash,
        "new_solver_invocations": 0,
    }


def _summary(
    *, dry_run: bool, runs: Sequence[Mapping[str, Any]], checkpoint: Mapping[str, Any], source: Mapping[str, Any], manifest_hash: str
) -> dict[str, Any]:
    counts = {classification: 0 for classification in ALLOWED_PAIR_RESULTS}
    for state in checkpoint["pair_states"]:
        counts[state["result_classification"]] += 1
    blockers = _proof_blockers(checkpoint)
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_name": "formal_remaining_k2_batch",
        "mode": "dry_run" if dry_run else "execution",
        "manifest_hash": manifest_hash,
        "source_sha256sums_hash": source["verification"]["sha256sums_sha256"],
        "ordering_hash": source["ordering_hash"],
        "planned_pair_count": len(runs),
        "batch_count": len(runs) // BATCH_SIZE,
        "batch_size": BATCH_SIZE,
        "solver_invocations": checkpoint["total_solver_invocations"],
        "pair_result_counts": counts,
        "global_k2_status": "unresolved",
        "proven_lower_bound": 2,
        "exact_minimum_claim": None,
        "proof_blockers": blockers,
        "global_proof_closure_blocked": bool(blockers),
        "new_feasibility_results": counts["fixed_pair_infeasible"] + counts["incumbent_pending_validation"],
        "dry_run_produces_feasibility_result": False,
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
    write_json(output / "manifest.json", dict(manifest) | {"manifest_hash": manifest_hash})
    write_json(output / "source_audit.json", {key: value for key, value in source.items() if key != "pairs"})
    write_csv(output / "planned_pairs.csv", runs)
    write_json(output / "compact_evidence_plan.json", {
        "schema_version": COMPACT_EVIDENCE_SCHEMA_VERSION,
        "manifest_hash": manifest_hash,
        "source_hashes": {
            "sha256sums": source["verification"]["sha256sums_sha256"],
            "ordering": source["ordering_hash"],
            "result": source["result_hash"],
            "proof": source["proof_hash"],
            "pair_rows": source["pair_rows_hash"],
        },
        "runs": [compact_evidence_plan(run) for run in runs],
    })
    write_json(output / "ordering.json", {
        "ordering_hash": source["ordering_hash"],
        "pair_ids": [run["pair_id"] for run in runs],
        "batches": [
            [run["pair_id"] for run in runs if run["batch_index"] == batch]
            for batch in range(1, len(runs) // BATCH_SIZE + 1)
        ],
    })
    write_json(output / "checkpoint.json", checkpoint)
    summary = _summary(
        dry_run=dry_run,
        runs=runs,
        checkpoint=checkpoint,
        source=source,
        manifest_hash=manifest_hash,
    )
    write_json(output / "aggregate_summary.json", summary)
    provenance_path = output / "provenance.json"
    provenance = read_json(provenance_path) if provenance_path.is_file() else {}
    provenance.update({
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": summary["mode"],
        "formal_input_artifact": source["artifact_root"],
        "source_artifact_read_only": True,
        "compact_evidence_schema_version": COMPACT_EVIDENCE_SCHEMA_VERSION,
        "full_model_proto_default": False,
        "anomaly_model_proto_escalation_enabled": True,
        "real_solver_invocations": checkpoint["total_solver_invocations"],
    })
    if checkpoint.get("failure_continuation"):
        provenance["failure_continuation"] = checkpoint["failure_continuation"]
    write_json(provenance_path, provenance)
    if not (output / "failures.json").exists():
        write_json(output / "failures.json", {"failures": [], "failure_count": 0})
    write_checksums(output)
    return summary


def compact_evidence_plan(run: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "compact_evidence_schema_version": COMPACT_EVIDENCE_SCHEMA_VERSION,
        "pair_identity": run["pair_id"],
        "pair_order": run["formal_order"],
        "source_hashes": True,
        "manifest_hash": True,
        "destination_domains": True,
        "placement_combination_count": run["placement_combination_count"],
        "model_fingerprint": True,
        "config_fingerprint": run["config_fingerprint"],
        "solver_configuration": True,
        "response_status_and_statistics": True,
        "response_hash": True,
        "raw_solver_log": True,
        "validation_result": True,
        "run_result": True,
        "checkpoint_state": True,
        "provenance": True,
        "full_model_saved": False,
        "full_model_save_reason": None,
    }


def _classify_solver_result(result: Mapping[str, Any]) -> tuple[str, bool]:
    status = str(result.get("status", ""))
    incumbent = bool(result.get("incumbent_found") or result.get("assignment_available"))
    if incumbent:
        return "incumbent_pending_validation", True
    if status == "INFEASIBLE":
        return "fixed_pair_infeasible", False
    if status == "UNKNOWN":
        return "unresolved_unknown_no_incumbent", True
    if status == "MODEL_INVALID":
        return "model_invalid", True
    return "artifact_failure", True


def _write_compact_run(
    output: Path,
    run: Mapping[str, Any],
    result: Mapping[str, Any],
    classification: str,
    *,
    manifest_hash: str,
    source: Mapping[str, Any],
) -> None:
    run_dir = output / "runs" / run["pair_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    evidence = compact_evidence_plan(run)
    evidence["manifest_hash"] = manifest_hash
    evidence["source_hashes"] = {
        "sha256sums": source["verification"]["sha256sums_sha256"],
        "ordering": source["ordering_hash"],
        "result": source["result_hash"],
        "proof": source["proof_hash"],
        "pair_rows": source["pair_rows_hash"],
    }
    solver_config = result["solver_config"]
    if json_hash(solver_config) != run["config_fingerprint"]:
        raise FormalK2BatchError("solver config fingerprint drift before persistence")
    domains = result["destination_domains"]
    observed_domain_sizes = [
        len(domains[run["section_id_a"]]),
        len(domains[run["section_id_b"]]),
    ]
    expected_domain_sizes = [
        int(run["destination_domain_size_a"]),
        int(run["destination_domain_size_b"]),
    ]
    if observed_domain_sizes != expected_domain_sizes:
        raise FormalK2BatchError("destination domain drift before persistence")
    response_hash = result.get("response_hash")
    if not response_hash or not result.get("response_hash_verified"):
        raise FormalK2BatchError("response hash unavailable or unverified")
    if result["response_stats"].get("response_hash") != response_hash:
        raise FormalK2BatchError("response statistics hash drift")
    full_model_bytes = result.get("full_model_bytes")
    save_reason = result.get("full_model_save_reason")
    if full_model_bytes is not None:
        if save_reason not in ANOMALY_MODEL_REASONS:
            raise FormalK2BatchError("model.pb save reason is not an allowed anomaly")
        (run_dir / "model.pb").write_bytes(full_model_bytes)
        evidence["full_model_saved"] = True
        evidence["full_model_save_reason"] = save_reason
    write_json(run_dir / "solver_config.json", solver_config)
    write_json(run_dir / "destination_domains.json", domains)
    write_json(run_dir / "model_fingerprint.json", result["model_fingerprint"])
    write_json(run_dir / "response_stats.json", result["response_stats"])
    _atomic_write_text(run_dir / "solver.log", "".join(result.get("solver_log", ())))
    write_json(run_dir / "validation.json", result.get("validation", {"validated": False}))
    write_json(run_dir / "hint_audit.json", result.get("hint_audit", {
        "hint_used": False,
        "assignment_hint_used": False,
        "objective_used": False,
        "candidate_pruning": False,
    }))
    write_json(run_dir / "provenance.json", {
        "compact_evidence_schema_version": COMPACT_EVIDENCE_SCHEMA_VERSION,
        "manifest_hash": manifest_hash,
        "source_sha256sums_hash": source["verification"]["sha256sums_sha256"],
        "pair_id": run["pair_id"],
        "formal_order": run["formal_order"],
        "run_b_allowed": False,
        "automatic_rerun_allowed": False,
    })
    write_json(run_dir / "compact_evidence.json", evidence)
    write_json(run_dir / "run_result.json", {
        "pair_id": run["pair_id"],
        "result_classification": classification,
        "config_fingerprint": run["config_fingerprint"],
        "response_hash": result.get("response_hash"),
        "response_hash_verified": bool(result.get("response_hash_verified")),
        "global_k2_status": "unresolved",
        "proven_lower_bound": 2,
        "exact_minimum_claim": None,
    })


def _real_solver_runner(run: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Execute one frozen production Run A; imported lazily so dry-run is solver-free."""
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
    domain_rows = []
    original = []
    proposed = []
    for section_id in section_ids:
        options = domains[section_id]
        original.append(next(option.placement for option in options if option.is_original))
        non_original = tuple(option.placement for option in options if not option.is_original)
        domain_rows.append([list(value) for value in non_original])
        proposed.append(non_original[0])
    candidate = CandidateEdit(
        candidate_id=f"formal_remaining:{run['formal_order']}:{run['pair_id']}",
        edit_type="k2_section_pair_full_destination_domain",
        logical_section_ids=section_ids,
        logical_course_ids=(run["course_id_a"], run["course_id_b"]),
        original_placements=tuple(original),
        proposed_placements=tuple(proposed),
        valid_period_source="formal frozen full non-original destination domain",
        occupancy_shape=tuple(len(value) for value in original),
        core_student="all_students_production_model",
        core_period_relevance=tuple(sorted({period for values in (*original, *proposed) for period in values})),
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
        time_limit_seconds=float(manifest["per_pair_time_limit_seconds"]),
    )
    proto_bytes = deterministic_model_proto_bytes(build.model)
    model_fingerprint = {
        "sha256": hashlib.sha256(proto_bytes).hexdigest(),
        "binary_proto_bytes": len(proto_bytes),
        "total_variables": len(build.model.Proto().variables),
        "total_constraints": len(build.model.Proto().constraints),
        "full_model_persisted": False,
    }
    solver_config = frozen_run_config(run, manifest)
    if json_hash(frozen_run_config(run, manifest)) != run["config_fingerprint"]:
        raise FormalK2BatchError("real-run configuration fingerprint drift")
    validation = {"validated": False, "reason": "no incumbent"}
    if search.incumbent_found:
        validation = validate_bootstrap_witness(context, build, search, config_dir=Path("data/config"), k=2)
    response = _response_payload(search)
    result = {
        "status": search.status,
        "incumbent_found": search.incumbent_found,
        "assignment_available": search.assignment_available,
        "response_hash": response.get("response_hash"),
        "response_hash_verified": response.get("response_hash_verified", False),
        "response_stats": response,
        "solver_log": search.solver_log,
        "solver_config": solver_config,
        "destination_domains": {section_id: values for section_id, values in zip(section_ids, domain_rows)},
        "model_fingerprint": model_fingerprint,
        "validation": validation,
        "hint_audit": hint_audit,
    }
    if search.status == "MODEL_INVALID":
        result["full_model_bytes"] = proto_bytes
        result["full_model_save_reason"] = "model_invalid"
    return result


def run_batch(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    output_dir: str | Path,
    dry_run: bool = False,
    execute: bool = False,
    authorize_failure_continuation: bool = False,
    continue_after_approved_failure: bool = False,
    resume: bool = False,
    max_new_solver_runs: int | None = None,
    solver_runner: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if sum(bool(value) for value in (dry_run, execute, authorize_failure_continuation)) != 1:
        raise FormalK2BatchError(
            "choose exactly one explicit mode: --dry-run, --execute, or "
            "--authorize-failure-continuation"
        )
    if authorize_failure_continuation:
        if not resume or continue_after_approved_failure:
            raise FormalK2BatchError("authorization creation requires --resume only")
        return create_failure_continuation_authorization(
            manifest_path=manifest_path,
            output_dir=output_dir,
        )
    if continue_after_approved_failure and (not execute or not resume):
        raise FormalK2BatchError(
            "--continue-after-approved-failure requires --execute and --resume"
        )
    if max_new_solver_runs is not None and max_new_solver_runs < 0:
        raise FormalK2BatchError("max-new-solver-runs must be nonnegative")
    output = Path(output_dir)
    manifest, manifest_hash = load_manifest(manifest_path)
    source = verify_formal_input(manifest)
    runs = planned_runs(source, manifest)
    if output.resolve() == FORMAL_OUTPUT.resolve() and dry_run:
        raise FormalK2BatchError("dry-run may not write the formal batch artifact")
    nonempty = output.exists() and any(output.iterdir())
    if nonempty and not resume:
        raise FormalK2BatchError(f"output is non-empty; use --resume: {output}")
    output.mkdir(parents=True, exist_ok=True)
    approved_failure_orders: frozenset[int] = frozenset()
    if nonempty:
        verify_checksums(output)
        checkpoint = read_json(output / "checkpoint.json")
        if continue_after_approved_failure:
            _validate_failure_continuation_authorization(
                output, checkpoint, runs, manifest_hash, source
            )
            approved_failure_orders = frozenset({1})
        _validate_checkpoint(
            checkpoint,
            runs,
            manifest_hash,
            source,
            output,
            approved_failure_orders=approved_failure_orders,
        )
    else:
        checkpoint = _initial_checkpoint(runs, manifest_hash, source)
    if dry_run:
        if int(checkpoint["total_solver_invocations"]) != 0:
            raise FormalK2BatchError("dry-run cannot resume an artifact with solver invocations")
        summary = _write_base_artifact(
            output, manifest, manifest_hash, source, runs, checkpoint, dry_run=True
        )
        summary["output_dir"] = str(output)
        summary["output_files"] = sorted(
            str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()
        )
        return summary

    runner = solver_runner or _real_solver_runner
    new_invocations = 0
    stop_reason = "invocation_limit_reached"
    states = checkpoint["pair_states"]
    for state, run in zip(states, runs):
        if state["state"] == "completed":
            continue
        if state["state"] == "artifact_failure" and int(run["formal_order"]) in approved_failure_orders:
            continue
        if max_new_solver_runs is not None and new_invocations >= max_new_solver_runs:
            break
        if int(checkpoint["total_solver_invocations"]) >= GLOBAL_INVOCATION_BUDGET:
            stop_reason = "global_invocation_budget_exhausted"
            break
        state["state"] = "running"
        checkpoint["total_solver_invocations"] += 1
        new_invocations += 1
        write_json(output / "checkpoint.json", checkpoint)
        write_checksums(output)
        try:
            result = dict(runner(run, manifest))
            classification, must_stop = _classify_solver_result(result)
            _write_compact_run(
                output,
                run,
                result,
                classification,
                manifest_hash=manifest_hash,
                source=source,
            )
            state.update({
                "state": "completed",
                "result_classification": classification,
                "response_hash": result.get("response_hash"),
            })
        except Exception as exc:
            state.update({
                "state": "artifact_failure",
                "result_classification": "artifact_failure",
                "failure": f"{type(exc).__name__}: {exc}",
            })
            write_json(output / "checkpoint.json", checkpoint)
            failures_path = output / "failures.json"
            failures = read_json(failures_path) if failures_path.is_file() else {"failures": []}
            history = list(failures.get("failures", []))
            if state["failure"] not in history:
                history.append(state["failure"])
            write_json(failures_path, {"failures": history, "failure_count": len(history)})
            write_checksums(output)
            raise FormalK2BatchError(f"run failed closed: {run['pair_id']}: {exc}") from exc
        for batch_index in range(1, len(runs) // BATCH_SIZE + 1):
            batch_states = states[(batch_index - 1) * BATCH_SIZE : batch_index * BATCH_SIZE]
            if all(item["state"] == "completed" for item in batch_states):
                if batch_index not in checkpoint["completed_batch_indices"]:
                    checkpoint["completed_batch_indices"].append(batch_index)
        write_json(output / "checkpoint.json", checkpoint)
        write_checksums(output)
        if must_stop:
            stop_reason = classification
            break
    else:
        stop_reason = (
            "all_executable_pairs_completed_with_proof_blocker"
            if _proof_blockers(checkpoint)
            else "all_pairs_completed"
        )
    summary = _write_base_artifact(
        output, manifest, manifest_hash, source, runs, checkpoint, dry_run=False
    )
    summary.update({"new_solver_invocations_this_call": new_invocations, "stop_reason": stop_reason})
    write_json(output / "aggregate_summary.json", summary)
    write_checksums(output)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plan or execute the formal remaining K=2 Run A batch.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--continue-after-approved-failure", action="store_true")
    parser.add_argument("--max-new-solver-runs", type=int, default=None)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--authorize-failure-continuation", action="store_true")
    args = parser.parse_args(argv)
    try:
        summary = run_batch(
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            dry_run=args.dry_run,
            execute=args.execute,
            authorize_failure_continuation=args.authorize_failure_continuation,
            continue_after_approved_failure=args.continue_after_approved_failure,
            resume=args.resume,
            max_new_solver_runs=args.max_new_solver_runs,
        )
    except FormalK2BatchError as exc:
        print(f"Formal remaining K=2 batch FAIL: {exc}")
        return 1
    print("Formal remaining K=2 batch PASS")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
