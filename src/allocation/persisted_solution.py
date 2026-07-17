from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .input_models import CanonicalAllocationInput
from .state import AllocationState


REQUIRED_ARTIFACT_FILES = (
    "benchmark_manifest.json",
    "algorithm_summary.csv",
    "student_outcomes.csv",
    "request_outcomes.csv",
    "final_schedule_policy_summary.csv",
    "final_schedule_policy_violations.csv",
    "artifact_recovery_provenance.json",
    "SHA256SUMS.txt",
)


class PersistedSolutionArtifactError(ValueError):
    """Raised when a persisted solution cannot be trusted as a solver hint."""


@dataclass(frozen=True)
class PersistedSolutionSeed:
    selected_assignments: tuple[tuple[str, str], ...]
    source_commit: str
    source_algorithm: str
    source_status: str
    source_policy_pass: bool
    manifest_sha256: str
    request_outcomes_sha256: str
    provenance_sha256: str
    fingerprint: dict[str, Any]
    data_generation_seed: int
    section_planning_seed: int
    solver_seed: int
    hint_unknown_keys: int = 0
    hint_duplicate_keys: int = 0


def load_persisted_solution_seed(
    artifact_dir: str | Path,
    allocation_input: CanonicalAllocationInput,
) -> PersistedSolutionSeed:
    root = Path(artifact_dir)
    if not root.is_dir():
        raise PersistedSolutionArtifactError(f"persisted solution artifact directory is missing: {root}")

    hashes = _verify_sha256_manifest(root)
    missing = tuple(name for name in REQUIRED_ARTIFACT_FILES if not (root / name).is_file())
    if missing:
        raise PersistedSolutionArtifactError(f"required persisted artifact files are missing: {', '.join(missing)}")

    try:
        manifest_payload = _read_json(root / "benchmark_manifest.json")
        provenance = _read_json(root / "artifact_recovery_provenance.json")
        algorithm_summary = _read_one_row(root / "algorithm_summary.csv")
        policy_summary = _read_one_row(root / "final_schedule_policy_summary.csv")
        policy_violations = pd.read_csv(root / "final_schedule_policy_violations.csv", keep_default_na=False)
        students = pd.read_csv(root / "student_outcomes.csv", keep_default_na=False)
        requests = pd.read_csv(root / "request_outcomes.csv", keep_default_na=False)
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, pd.errors.ParserError) as exc:
        raise PersistedSolutionArtifactError(f"cannot read persisted solution artifact: {exc}") from exc

    manifest = manifest_payload.get("manifest")
    if not isinstance(manifest, dict):
        raise PersistedSolutionArtifactError("benchmark_manifest.json must contain a manifest object")
    actual_fingerprint = _canonical_fingerprint(allocation_input)
    _require_exact_fingerprint(manifest, actual_fingerprint)
    _require_exact_fingerprint(provenance.get("fingerprint"), actual_fingerprint)

    data_seed = _required_int(manifest, "data_generation_seed")
    section_seed = _required_int(manifest, "section_planning_seed")
    solver_seed = _required_int(manifest, "solver_seed")
    if provenance.get("data_generation_seed") != data_seed or provenance.get("section_planning_seed") != section_seed:
        raise PersistedSolutionArtifactError("persisted artifact seed metadata is inconsistent")
    if provenance.get("solver_seed") != solver_seed:
        raise PersistedSolutionArtifactError("persisted artifact solver seed metadata is inconsistent")

    source_status = str(algorithm_summary.get("status", "")).upper()
    if source_status not in {"FEASIBLE", "OPTIMAL"}:
        raise PersistedSolutionArtifactError(f"persisted source status must be FEASIBLE or OPTIMAL, got {source_status!r}")
    if str(provenance.get("status", "")).upper() != source_status:
        raise PersistedSolutionArtifactError("persisted source status disagrees with provenance")
    if str(provenance.get("solve_status", "")).upper() != source_status:
        raise PersistedSolutionArtifactError("persisted solve_status disagrees with source status")

    policy_pass = _parse_bool(policy_summary.get("final_schedule_policy_pass"))
    if not policy_pass:
        raise PersistedSolutionArtifactError("persisted source final schedule policy did not pass")
    if _as_int(policy_summary.get("violating_student_count")) != 0:
        raise PersistedSolutionArtifactError("persisted source has violating students")
    if len(policy_violations) != 0:
        raise PersistedSolutionArtifactError("persisted source contains final schedule policy violations")
    _require_zero(algorithm_summary, ("ordinary_violations", "protected_violations", "high_demand_violations"))
    _require_zero(algorithm_summary, ("section_over_capacity_count", "consistency_issue_count"))
    _require_zero(provenance, ("violating_student_count", "section_over_capacity_count", "consistency_issue_count"))
    if provenance.get("final_schedule_policy_pass") is not True:
        raise PersistedSolutionArtifactError("provenance does not confirm final schedule policy PASS")

    _validate_student_universe(students, allocation_input)
    selected = _validate_request_universe(requests, allocation_input)
    _replay_assignments(requests, allocation_input)
    _validate_student_outcomes(students, requests)
    return PersistedSolutionSeed(
        selected_assignments=tuple(selected),
        source_commit=str(provenance.get("source_git_commit") or provenance.get("git_subject", "")),
        source_algorithm=str(algorithm_summary.get("algorithm_name", provenance.get("algorithm_name", ""))),
        source_status=source_status,
        source_policy_pass=True,
        manifest_sha256=hashes["benchmark_manifest.json"],
        request_outcomes_sha256=hashes["request_outcomes.csv"],
        provenance_sha256=hashes["artifact_recovery_provenance.json"],
        fingerprint=dict(actual_fingerprint),
        data_generation_seed=data_seed,
        section_planning_seed=section_seed,
        solver_seed=solver_seed,
    )


def _verify_sha256_manifest(root: Path) -> dict[str, str]:
    sums_path = root / "SHA256SUMS.txt"
    if not sums_path.is_file():
        raise PersistedSolutionArtifactError(f"SHA256SUMS.txt is missing: {sums_path}")
    expected: dict[str, str] = {}
    for line_number, line in enumerate(sums_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or len(parts[0]) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in parts[0]):
            raise PersistedSolutionArtifactError(f"malformed SHA256SUMS.txt line {line_number}")
        name = parts[1].lstrip("*")
        name_path = Path(name)
        if not name or name in expected or name_path.is_absolute() or ".." in name_path.parts:
            raise PersistedSolutionArtifactError(f"duplicate or empty SHA256 entry on line {line_number}")
        expected[name] = parts[0].lower()
    for name, expected_hash in expected.items():
        path = root / name
        if not path.is_file() or _sha256(path) != expected_hash:
            raise PersistedSolutionArtifactError(f"SHA256 mismatch for persisted artifact file: {name}")
    missing = tuple(name for name in REQUIRED_ARTIFACT_FILES[:-1] if name not in expected)
    if missing:
        raise PersistedSolutionArtifactError(f"SHA256SUMS.txt does not cover required files: {', '.join(missing)}")
    return expected


def _canonical_fingerprint(allocation_input: CanonicalAllocationInput) -> dict[str, Any]:
    # Imported lazily to avoid the allocation package's import cycle.
    from src.experiment_manifest import canonical_input_fingerprint
    from dataclasses import asdict

    return asdict(canonical_input_fingerprint(allocation_input))


def _require_exact_fingerprint(source: Any, actual: dict[str, Any]) -> None:
    if not isinstance(source, dict):
        raise PersistedSolutionArtifactError("persisted artifact fingerprint is missing")
    fields = (
        "students", "logical_requests", "logical_primaries", "alternates",
        "logical_sections", "section_rows", "candidate_edges", "canonical_input_hash",
    )
    mismatches = [f"{field}: expected {actual[field]!r}, got {source.get(field)!r}" for field in fields if source.get(field) != actual[field]]
    if mismatches:
        raise PersistedSolutionArtifactError("persisted artifact fingerprint mismatch: " + "; ".join(mismatches))


def _validate_student_universe(students: pd.DataFrame, allocation_input: CanonicalAllocationInput) -> None:
    if "student_id" not in students or students["student_id"].duplicated().any():
        raise PersistedSolutionArtifactError("persisted student_outcomes has missing or duplicate student IDs")
    expected = {student.student_id for student in allocation_input.students}
    actual = set(students["student_id"].astype(str))
    if actual != expected:
        raise PersistedSolutionArtifactError("persisted student universe does not exactly match canonical input")


def _validate_request_universe(
    requests: pd.DataFrame,
    allocation_input: CanonicalAllocationInput,
) -> list[tuple[str, str]]:
    required = {"request_key", "student_id", "request_type", "candidate_key", "status", "assignment_key", "assigned_linked_section_group_id"}
    missing = sorted(required - set(requests.columns))
    if missing:
        raise PersistedSolutionArtifactError(f"request_outcomes is missing columns: {', '.join(missing)}")
    if requests["request_key"].duplicated().any():
        raise PersistedSolutionArtifactError("persisted request_outcomes contains duplicate request keys")
    expected = set(allocation_input.requests_by_key)
    actual = set(requests["request_key"].astype(str))
    if actual != expected:
        raise PersistedSolutionArtifactError("persisted logical request universe does not exactly match canonical input")

    selected: list[tuple[str, str]] = []
    assignment_keys: set[str] = set()
    for row in requests.to_dict("records"):
        request_key = str(row["request_key"])
        request = allocation_input.requests_by_key[request_key]
        if str(row["student_id"]) != request.student_id or str(row["request_type"]) != request.request_type:
            raise PersistedSolutionArtifactError(f"persisted request identity mismatch: {request_key}")
        if str(row["candidate_key"]) != request.candidate_key:
            raise PersistedSolutionArtifactError(f"persisted request candidate mismatch: {request_key}")
        status = str(row["status"]).lower()
        group_id = str(row["assigned_linked_section_group_id"])
        assignment_key = str(row["assignment_key"])
        if status == "assigned":
            if not group_id or not assignment_key:
                raise PersistedSolutionArtifactError(f"assigned request is missing assignment identity: {request_key}")
            if group_id not in allocation_input.candidate_index.get(request_key, ()):
                raise PersistedSolutionArtifactError(f"assigned section is not a candidate for request: {request_key} -> {group_id}")
            if assignment_key in assignment_keys:
                raise PersistedSolutionArtifactError(f"duplicate persisted assignment key: {assignment_key}")
            assignment_keys.add(assignment_key)
            selected.append((request_key, group_id))
        elif group_id or assignment_key:
            raise PersistedSolutionArtifactError(f"unassigned request contains a fake assignment: {request_key}")
    return sorted(selected)


def _validate_student_outcomes(students: pd.DataFrame, requests: pd.DataFrame) -> None:
    assigned = requests[requests["status"].astype(str).str.lower() == "assigned"]
    assigned_counts = assigned.groupby("student_id").size().to_dict()
    if "assigned_logical_course_count" not in students.columns:
        raise PersistedSolutionArtifactError("student_outcomes is missing assigned_logical_course_count")
    for row in students.to_dict("records"):
        expected = int(assigned_counts.get(str(row["student_id"]), 0))
        if _as_int(row["assigned_logical_course_count"]) != expected:
            raise PersistedSolutionArtifactError(
                f"student assigned logical count disagrees with request outcomes: {row['student_id']}"
            )


def _replay_assignments(requests: pd.DataFrame, allocation_input: CanonicalAllocationInput) -> None:
    state = AllocationState(allocation_input)
    assigned = [
        row for row in requests.to_dict("records")
        if str(row["status"]).lower() == "assigned"
    ]
    assigned.sort(
        key=lambda row: _assignment_sort_key(
            allocation_input.requests_by_key[str(row["request_key"])],
            str(row["assigned_linked_section_group_id"]),
        )
    )
    for row in assigned:
        result = state.try_assign(
            str(row["student_id"]),
            str(row["request_key"]),
            str(row["assigned_linked_section_group_id"]),
        )
        if not result.allowed:
            reasons = ", ".join(reason.value for reason in result.reasons)
            raise PersistedSolutionArtifactError(
                f"persisted assignments fail AllocationState replay: {row['request_key']} -> "
                f"{row['assigned_linked_section_group_id']} ({reasons})"
            )
    issues = state.validate_internal_consistency()
    if issues:
        raise PersistedSolutionArtifactError(f"persisted assignments fail consistency replay: {issues!r}")


def _assignment_sort_key(request: Any, section_id: str) -> tuple[Any, ...]:
    type_order = {"primary": 0, "mandatory_fallback": 1, "alternate": 2}.get(request.request_type, 9)
    return (request.student_id, type_order, request.request_rank or 0, request.request_key, section_id)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return value


def _read_one_row(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path, keep_default_na=False)
    if len(frame) != 1:
        raise ValueError(f"{path.name} must contain exactly one row")
    return frame.iloc[0].to_dict()


def _required_int(mapping: dict[str, Any], field: str) -> int:
    value = mapping.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PersistedSolutionArtifactError(f"persisted artifact field {field} must be a non-negative integer")
    return value


def _require_zero(mapping: dict[str, Any], fields: tuple[str, ...]) -> None:
    for field in fields:
        if field in mapping and _as_int(mapping[field]) != 0:
            raise PersistedSolutionArtifactError(f"persisted artifact field {field} must be zero")


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if str(value).strip().lower() in {"true", "1"}:
        return True
    if str(value).strip().lower() in {"false", "0"}:
        return False
    raise PersistedSolutionArtifactError(f"invalid boolean value in persisted artifact: {value!r}")


def _as_int(value: Any) -> int:
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise PersistedSolutionArtifactError(f"invalid integer value in persisted artifact: {value!r}") from exc
    if isinstance(value, float) and value != result:
        raise PersistedSolutionArtifactError(f"non-integral value in persisted artifact: {value!r}")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()
