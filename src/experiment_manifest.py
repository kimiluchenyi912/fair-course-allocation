from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.allocation import CanonicalAllocationInput, canonicalize_allocation_input


SCHEMA_VERSION = 1


class ExperimentManifestError(ValueError):
    def __init__(self, issues: tuple[str, ...]):
        self.issues = issues
        super().__init__("Experiment manifest validation failed:\n" + "\n".join(f"- {issue}" for issue in issues))


@dataclass(frozen=True)
class ExperimentSeeds:
    data_generation_seed: int
    section_planning_seed: int
    solver_seed: int

    def __post_init__(self) -> None:
        for field_name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer.")


STABLE_YEAR_BENCHMARK_SEEDS = ExperimentSeeds(
    data_generation_seed=2026,
    section_planning_seed=2026,
    solver_seed=20260630,
)


@dataclass(frozen=True)
class CanonicalInputFingerprint:
    students: int
    logical_requests: int
    logical_primaries: int
    alternates: int
    logical_sections: int
    section_rows: int
    candidate_edges: int
    canonical_input_hash: str


@dataclass(frozen=True)
class ExperimentManifest:
    scenario_id: str
    seeds: ExperimentSeeds
    generated_input_path: str
    section_input_path: str
    fingerprint: CanonicalInputFingerprint
    generated_input_hash: str
    section_input_hash: str
    config_fingerprint: str
    git_commit: str = ""
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scenario_id": self.scenario_id,
            **asdict(self.seeds),
            "generated_input_path": self.generated_input_path,
            "section_input_path": self.section_input_path,
            **asdict(self.fingerprint),
            "generated_input_hash": self.generated_input_hash,
            "section_input_hash": self.section_input_hash,
            "config_fingerprint": self.config_fingerprint,
            "git_commit": self.git_commit,
        }


def canonical_input_fingerprint(allocation_input: CanonicalAllocationInput) -> CanonicalInputFingerprint:
    request_keys = {request.request_key for request in allocation_input.logical_requests}
    if set(allocation_input.candidate_index) != request_keys:
        raise ExperimentManifestError(("candidate_index keys do not match canonical logical requests.",))
    primary_count = sum(request.request_type == "primary" for request in allocation_input.logical_requests)
    alternate_count = sum(request.request_type == "alternate" for request in allocation_input.logical_requests)
    return CanonicalInputFingerprint(
        students=len(allocation_input.students),
        logical_requests=len(allocation_input.logical_requests),
        logical_primaries=primary_count,
        alternates=alternate_count,
        logical_sections=len(allocation_input.logical_sections),
        section_rows=sum(len(section.member_sections) for section in allocation_input.logical_sections),
        candidate_edges=sum(len(section_ids) for section_ids in allocation_input.candidate_index.values()),
        canonical_input_hash=_sha256_json(_canonical_payload(allocation_input)),
    )


def build_experiment_manifest(
    generated_input_path: str | Path,
    section_input_path: str | Path,
    config_dir: str | Path,
    *,
    scenario_id: str,
    seeds: ExperimentSeeds,
    repo_root: str | Path | None = None,
) -> ExperimentManifest:
    manifest, _ = _build_manifest_and_input(
        generated_input_path,
        section_input_path,
        config_dir,
        scenario_id=scenario_id,
        seeds=seeds,
        repo_root=repo_root,
    )
    return manifest


def _build_manifest_and_input(
    generated_input_path: str | Path,
    section_input_path: str | Path,
    config_dir: str | Path,
    *,
    scenario_id: str,
    seeds: ExperimentSeeds,
    repo_root: str | Path | None,
) -> tuple[ExperimentManifest, CanonicalAllocationInput]:
    generated_dir = Path(generated_input_path)
    section_dir = Path(section_input_path)
    config_path = Path(config_dir)
    issues, generation_metadata, section_metadata = _validate_source_metadata(
        generated_dir,
        section_dir,
        scenario_id,
        seeds,
    )
    required_paths = (
        generated_dir / "students.csv",
        generated_dir / "requests.csv",
        section_dir / "sections.csv",
        config_path / "course_catalog.csv",
    )
    missing = tuple(str(path) for path in required_paths if not path.is_file())
    if missing:
        issues.extend(f"required input file is missing: {path}" for path in missing)
        raise ExperimentManifestError(tuple(sorted(issues)))

    students = pd.read_csv(required_paths[0], keep_default_na=False)
    requests = pd.read_csv(required_paths[1], keep_default_na=False)
    sections = pd.read_csv(required_paths[2], keep_default_na=False)
    catalog = pd.read_csv(required_paths[3], keep_default_na=False)
    allocation_input = canonicalize_allocation_input(students, requests, sections, catalog)
    fingerprint = canonical_input_fingerprint(allocation_input)
    if fingerprint.section_rows != len(sections):
        issues.append(f"canonical section_rows={fingerprint.section_rows} does not match sections.csv rows={len(sections)}.")
    _validate_metadata_counts(
        issues,
        generation_metadata,
        section_metadata,
        students,
        requests,
        sections,
        fingerprint,
    )
    if issues:
        raise ExperimentManifestError(tuple(sorted(issues)))

    return (
        ExperimentManifest(
            scenario_id=scenario_id,
            seeds=seeds,
            generated_input_path=str(generated_dir),
            section_input_path=str(section_dir),
            fingerprint=fingerprint,
            generated_input_hash=_hash_named_files(
                {
                    "students.csv": required_paths[0],
                    "requests.csv": required_paths[1],
                }
            ),
            section_input_hash=_hash_named_files({"sections.csv": required_paths[2]}),
            config_fingerprint=_hash_directory_csvs(config_path),
            git_commit=_git_commit(repo_root) if repo_root is not None else "",
        ),
        allocation_input,
    )


def verify_experiment_manifest(
    manifest: ExperimentManifest,
    *,
    config_dir: str | Path,
    repo_root: str | Path | None = None,
    require_same_git: bool = False,
) -> CanonicalAllocationInput:
    actual, allocation_input = _build_manifest_and_input(
        manifest.generated_input_path,
        manifest.section_input_path,
        config_dir,
        scenario_id=manifest.scenario_id,
        seeds=manifest.seeds,
        repo_root=repo_root,
    )
    expected_data = manifest.to_dict()
    actual_data = actual.to_dict()
    ignored = set() if require_same_git else {"git_commit"}
    mismatches = [
        f"{field}: expected {expected_data[field]!r}, actual {actual_data[field]!r}"
        for field in sorted(expected_data)
        if field not in ignored and expected_data[field] != actual_data[field]
    ]
    if mismatches:
        raise ExperimentManifestError(tuple(mismatches))
    return allocation_input


def write_experiment_manifest(manifest: ExperimentManifest, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(destination)


def load_experiment_manifest(path: str | Path) -> ExperimentManifest:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION}.")
        seeds = ExperimentSeeds(
            data_generation_seed=_manifest_int(data, "data_generation_seed"),
            section_planning_seed=_manifest_int(data, "section_planning_seed"),
            solver_seed=_manifest_int(data, "solver_seed"),
        )
        fingerprint = CanonicalInputFingerprint(
            students=_manifest_int(data, "students"),
            logical_requests=_manifest_int(data, "logical_requests"),
            logical_primaries=_manifest_int(data, "logical_primaries"),
            alternates=_manifest_int(data, "alternates"),
            logical_sections=_manifest_int(data, "logical_sections"),
            section_rows=_manifest_int(data, "section_rows"),
            candidate_edges=_manifest_int(data, "candidate_edges"),
            canonical_input_hash=_manifest_text(data, "canonical_input_hash"),
        )
        return ExperimentManifest(
            scenario_id=_manifest_text(data, "scenario_id"),
            seeds=seeds,
            generated_input_path=_manifest_text(data, "generated_input_path"),
            section_input_path=_manifest_text(data, "section_input_path"),
            fingerprint=fingerprint,
            generated_input_hash=_manifest_text(data, "generated_input_hash"),
            section_input_hash=_manifest_text(data, "section_input_hash"),
            config_fingerprint=_manifest_text(data, "config_fingerprint"),
            git_commit=str(data.get("git_commit", "")),
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ExperimentManifestError((str(exc),)) from exc


def _validate_source_metadata(
    generated_dir: Path,
    section_dir: Path,
    scenario_id: str,
    seeds: ExperimentSeeds,
) -> tuple[list[str], dict[str, Any] | None, dict[str, Any] | None]:
    issues: list[str] = []
    generation_metadata = _read_metadata(generated_dir / "generation_metadata.json", issues)
    section_metadata = _read_metadata(section_dir / "section_planning_metadata.json", issues)
    if generation_metadata is not None:
        _check_metadata_value(issues, "generation metadata scenario_id", generation_metadata.get("scenario_id"), scenario_id)
        _check_metadata_value(issues, "generation metadata data seed", generation_metadata.get("seed"), seeds.data_generation_seed)
        _check_recorded_hashes(
            issues,
            "generation metadata output",
            generation_metadata.get("output_file_hashes"),
            generated_dir,
            ("students.csv", "requests.csv"),
        )
    if section_metadata is not None:
        _check_metadata_value(issues, "section metadata scenario_id", section_metadata.get("scenario_id"), scenario_id)
        _check_metadata_value(
            issues,
            "section metadata section planning seed",
            section_metadata.get("seed"),
            seeds.section_planning_seed,
        )
        _check_recorded_hashes(
            issues,
            "section metadata input",
            section_metadata.get("input_file_hashes"),
            generated_dir,
            ("students.csv", "requests.csv"),
        )
        _check_recorded_hashes(
            issues,
            "section metadata output",
            section_metadata.get("output_file_hashes"),
            section_dir,
            ("sections.csv",),
        )
    return issues, generation_metadata, section_metadata


def _check_recorded_hashes(
    issues: list[str],
    label: str,
    recorded_hashes: Any,
    directory: Path,
    names: tuple[str, ...],
) -> None:
    if not isinstance(recorded_hashes, dict):
        issues.append(f"{label}_file_hashes is missing or invalid.")
        return
    for name in names:
        source = directory / name
        if source.is_file():
            _check_metadata_value(
                issues,
                f"{label} hash for {name}",
                recorded_hashes.get(name),
                _sha256_file(source),
            )


def _validate_metadata_counts(
    issues: list[str],
    generation_metadata: dict[str, Any] | None,
    section_metadata: dict[str, Any] | None,
    students: pd.DataFrame,
    requests: pd.DataFrame,
    sections: pd.DataFrame,
    fingerprint: CanonicalInputFingerprint,
) -> None:
    primary_rows = int((requests["request_type"] == "primary").sum())
    alternate_rows = int((requests["request_type"] == "alternate").sum())
    if generation_metadata is not None:
        _check_metadata_value(issues, "generation metadata total_students", generation_metadata.get("total_students"), len(students))
        _check_metadata_value(
            issues,
            "generation metadata primary_request_rows",
            generation_metadata.get("primary_request_rows"),
            primary_rows,
        )
        _check_metadata_value(
            issues,
            "generation metadata alternate_request_rows",
            generation_metadata.get("alternate_request_rows"),
            alternate_rows,
        )
    if section_metadata is not None:
        expected = {
            "student_count": len(students),
            "primary_request_rows": primary_rows,
            "total_section_rows": len(sections),
            "total_logical_sections": fingerprint.logical_sections,
            "total_primary_demand": fingerprint.logical_primaries,
        }
        for field, value in expected.items():
            _check_metadata_value(issues, f"section metadata {field}", section_metadata.get(field), value)


def _read_metadata(path: Path, issues: list[str]) -> dict[str, Any] | None:
    if not path.is_file():
        issues.append(f"required metadata file is missing: {path}")
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        issues.append(f"cannot read metadata {path}: {exc}")
        return None
    if not isinstance(value, dict):
        issues.append(f"metadata must be a JSON object: {path}")
        return None
    return value


def _check_metadata_value(issues: list[str], name: str, actual: Any, expected: Any) -> None:
    if actual != expected:
        issues.append(f"{name}: expected {expected!r}, actual {actual!r}.")


def _canonical_payload(allocation_input: CanonicalAllocationInput) -> dict[str, Any]:
    return {
        "students": [
            {
                "student_id": student.student_id,
                "grade": student.grade,
                "target_period_units": student.target_period_units,
                "unscheduled_preference": student.unscheduled_preference,
                "priority_protected": student.priority_protected,
                "priority_reason": student.priority_reason,
                "priority_valid_school_year": student.priority_valid_school_year,
            }
            for student in allocation_input.students
        ],
        "requests": [asdict(request) for request in allocation_input.logical_requests],
        "sections": [asdict(section) for section in allocation_input.logical_sections],
        "courses": [asdict(allocation_input.courses_by_id[key]) for key in sorted(allocation_input.courses_by_id)],
        "candidate_index": [
            [request_key, list(allocation_input.candidate_index[request_key])]
            for request_key in sorted(allocation_input.candidate_index)
        ],
    }


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _hash_named_files(files: dict[str, Path]) -> str:
    digest = hashlib.sha256()
    for name, path in sorted(files.items()):
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _hash_directory_csvs(directory: Path) -> str:
    files = {path.relative_to(directory).as_posix(): path for path in directory.rglob("*.csv")}
    if not files:
        raise ExperimentManifestError((f"config directory contains no CSV files: {directory}",))
    return _hash_named_files(files)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(repo_root: str | Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ExperimentManifestError((f"cannot determine git commit: {exc}",)) from exc
    return result.stdout.strip()


def _manifest_int(data: dict[str, Any], field: str) -> int:
    value = data[field]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer.")
    return value


def _manifest_text(data: dict[str, Any], field: str) -> str:
    value = data[field]
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string.")
    return value
