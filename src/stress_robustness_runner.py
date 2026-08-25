"""Phase B stress and structural-infeasibility robustness runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.allocation import canonicalize_allocation_input
from src.benchmark_runner import run_benchmark_suite
from src.experiment_manifest import ExperimentSeeds, build_experiment_manifest
from src.infeasibility_certificates import validate_certificate
from src.input_difficulty import build_input_difficulty
from src.robustness_runner import (
    ALGORITHM_LABELS,
    GREEDY_ALGORITHMS,
    ScenarioManifestError,
    ScenarioSpec,
    _difficulty_row,
    _fingerprint_from_dict,
    _scenario_result_rows,
    _sha256_file,
    _write_json,
    _write_rows_csv,
    load_scenario_suite,
    _algorithm_alias,
)
from src.scenario_transforms import (
    ScenarioTransformError,
    StressTransformSpec,
    TRANSFORM_SCHEMA_VERSION,
    apply_stress_transforms,
)


STRESS_SCHEMA_VERSION = 1
DEFAULT_STRESS_SUITE_PATH = Path("data/scenarios/stress_robustness_v1.json")
DEFAULT_NORMAL_SUITE_PATH = Path("data/scenarios/normal_year_robustness_v1.json")
DEFAULT_NORMAL_ARTIFACT_DIR = Path("../fair-course-allocation-artifacts/robustness-v1/normal-development-v1")
DEFAULT_STRESS_ARTIFACT_DIR = Path("../fair-course-allocation-artifacts/robustness-v1/stress-development-v1")
ALLOWED_TRANSFORMS = {
    "enrollment_surge",
    "popular_course_surge",
    "alternate_drop",
    "capacity_reduction",
    "section_outage",
    "protected_primary_no_candidate",
    "minimum_logical_load_max_four",
    "global_capacity_deficit",
}


class StressManifestError(ValueError):
    """Raised when the Phase B manifest is incomplete or ambiguous."""


class StressRunnerError(ValueError):
    """Raised when a Phase B run cannot proceed fail-closed."""


@dataclass(frozen=True)
class StressScenarioSpec:
    scenario_id: str
    split: str
    scenario_family: str
    base_scenario_id: str
    paired_normal_scenario_id: str
    transform_seed: int
    transforms: tuple[dict[str, Any], ...]
    transform_order: tuple[str, ...]
    expected_feasibility: str
    tuning_allowed: bool
    enabled: bool
    purpose: str
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BoundStressScenario:
    spec: StressScenarioSpec
    data_generation_seed: int
    section_planning_seed: int
    algorithm_seed: int

    def to_transform_spec(self) -> StressTransformSpec:
        return StressTransformSpec(
            scenario_id=self.spec.scenario_id,
            base_scenario_id=self.spec.base_scenario_id,
            transform_seed=self.spec.transform_seed,
            transforms=self.spec.transforms,
            transform_order=self.spec.transform_order,
            scenario_family=self.spec.scenario_family,
            expected_feasibility=self.spec.expected_feasibility,
        )


@dataclass(frozen=True)
class StressScenarioSuite:
    schema_version: int
    suite_name: str
    suite_version: str
    parent_suite: str
    transform_schema_version: int
    default_split: str
    scenarios: tuple[StressScenarioSpec, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "suite_name": self.suite_name,
            "suite_version": self.suite_version,
            "parent_suite": self.parent_suite,
            "transform_schema_version": self.transform_schema_version,
            "default_split": self.default_split,
            "scenarios": [scenario.to_dict() for scenario in self.scenarios],
        }


@dataclass(frozen=True)
class StressSuiteResult:
    suite_hash: str
    split: str
    scenario_ids: tuple[str, ...]
    algorithms: tuple[str, ...]
    output_dir: str | None
    dry_run: bool
    scenario_rows: tuple[dict[str, Any], ...] = ()


def load_stress_scenario_suite(path: str | Path = DEFAULT_STRESS_SUITE_PATH) -> StressScenarioSuite:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StressManifestError(f"Cannot read stress manifest {source}: {exc}") from exc
    return validate_stress_scenario_suite(payload)


def validate_stress_scenario_suite(payload: Any) -> StressScenarioSuite:
    if not isinstance(payload, dict):
        raise StressManifestError("Stress manifest must be a JSON object.")
    required = {"schema_version", "suite_name", "suite_version", "parent_suite", "transform_schema_version", "default_split", "scenarios"}
    missing = sorted(required - set(payload))
    if missing:
        raise StressManifestError(f"Stress manifest is missing: {', '.join(missing)}")
    if payload["schema_version"] != STRESS_SCHEMA_VERSION or payload["transform_schema_version"] != TRANSFORM_SCHEMA_VERSION:
        raise StressManifestError("Only stress and transform schema version 1 are supported.")
    if payload["parent_suite"] != "normal_year_robustness":
        raise StressManifestError("parent_suite must be normal_year_robustness.")
    if payload["default_split"] not in {"development", "holdout"}:
        raise StressManifestError("default_split must be development or holdout.")
    scenarios = []
    seen: set[str] = set()
    for index, raw in enumerate(payload["scenarios"], start=1):
        if not isinstance(raw, dict):
            raise StressManifestError(f"scenarios[{index}] must be an object.")
        fields = {
            "scenario_id", "split", "scenario_family", "base_scenario_id", "paired_normal_scenario_id",
            "transform_seed", "transforms", "transform_order", "expected_feasibility", "tuning_allowed",
            "enabled", "purpose", "notes",
        }
        missing = sorted(fields - set(raw))
        if missing:
            raise StressManifestError(f"scenarios[{index}] is missing: {', '.join(missing)}")
        scenario_id = _text(raw["scenario_id"], f"scenarios[{index}].scenario_id")
        if scenario_id in seen:
            raise StressManifestError(f"Duplicate stress scenario_id: {scenario_id}")
        seen.add(scenario_id)
        split = raw["split"]
        if split not in {"development", "holdout"}:
            raise StressManifestError(f"{scenario_id}: split must be development or holdout.")
        if not isinstance(raw["tuning_allowed"], bool) or not isinstance(raw["enabled"], bool):
            raise StressManifestError(f"{scenario_id}: tuning_allowed must be boolean.")
        if split == "holdout" and raw["tuning_allowed"]:
            raise StressManifestError(f"{scenario_id}: holdout cannot allow tuning.")
        expected = raw["expected_feasibility"]
        if expected not in {"unknown", "structurally_infeasible"}:
            raise StressManifestError(f"{scenario_id}: invalid expected_feasibility.")
        transforms = raw["transforms"]
        order = raw["transform_order"]
        if not isinstance(transforms, list) or not transforms or not isinstance(order, list) or len(order) != len(transforms):
            raise StressManifestError(f"{scenario_id}: transforms and transform_order must be non-empty matching lists.")
        for transform, transform_type in zip(transforms, order):
            if not isinstance(transform, dict) or transform.get("type") != transform_type or transform_type not in ALLOWED_TRANSFORMS:
                raise StressManifestError(f"{scenario_id}: invalid transform order or type.")
            if transform_type in {"enrollment_surge", "popular_course_surge", "alternate_drop", "capacity_reduction"}:
                percentage = transform.get("percentage")
                if isinstance(percentage, bool) or not isinstance(percentage, (int, float)) or not 0 <= float(percentage) <= 1:
                    raise StressManifestError(f"{scenario_id}: {transform_type} percentage must be in [0, 1].")
            if transform_type == "section_outage" and (isinstance(transform.get("count"), bool) or not isinstance(transform.get("count"), int) or transform["count"] <= 0):
                raise StressManifestError(f"{scenario_id}: section_outage count must be positive.")
        structural = {"protected_primary_no_candidate", "minimum_logical_load_max_four", "global_capacity_deficit"}
        if expected == "structurally_infeasible" and order[0] not in structural:
            raise StressManifestError(f"{scenario_id}: structural negative must start with a certificate transform.")
        if expected == "unknown" and any(item in structural for item in order):
            raise StressManifestError(f"{scenario_id}: ordinary stress cannot contain a structural negative transform.")
        scenarios.append(
            StressScenarioSpec(
                scenario_id=scenario_id,
                split=split,
                scenario_family=_text(raw["scenario_family"], f"{scenario_id}.scenario_family"),
                base_scenario_id=_text(raw["base_scenario_id"], f"{scenario_id}.base_scenario_id"),
                paired_normal_scenario_id=_text(raw["paired_normal_scenario_id"], f"{scenario_id}.paired_normal_scenario_id"),
                transform_seed=_nonnegative_int(raw["transform_seed"], f"{scenario_id}.transform_seed"),
                transforms=tuple(dict(item) for item in transforms),
                transform_order=tuple(str(item) for item in order),
                expected_feasibility=expected,
                tuning_allowed=raw["tuning_allowed"],
                enabled=raw["enabled"],
                purpose=_text(raw["purpose"], f"{scenario_id}.purpose"),
                notes=_text(raw["notes"], f"{scenario_id}.notes"),
            )
        )
    if not scenarios:
        raise StressManifestError("Stress manifest must contain scenarios.")
    return StressScenarioSuite(
        schema_version=STRESS_SCHEMA_VERSION,
        suite_name=_text(payload["suite_name"], "suite_name"),
        suite_version=_text(payload["suite_version"], "suite_version"),
        parent_suite=_text(payload["parent_suite"], "parent_suite"),
        transform_schema_version=TRANSFORM_SCHEMA_VERSION,
        default_split=payload["default_split"],
        scenarios=tuple(scenarios),
    )


def stress_suite_hash(suite: StressScenarioSuite) -> str:
    return _sha256_json(suite.to_dict())


def run_stress_robustness_suite(
    *,
    suite_path: str | Path = DEFAULT_STRESS_SUITE_PATH,
    normal_suite_path: str | Path = DEFAULT_NORMAL_SUITE_PATH,
    normal_artifact_dir: str | Path = DEFAULT_NORMAL_ARTIFACT_DIR,
    split: str = "development",
    scenario_id: str | None = None,
    algorithms: tuple[str, ...] = GREEDY_ALGORITHMS,
    output_dir: str | Path = DEFAULT_STRESS_ARTIFACT_DIR,
    max_scenarios: int | None = None,
    resume: bool = False,
    dry_run: bool = False,
    confirm_holdout_evaluation: bool = False,
    config_dir: str | Path = "data/config",
) -> StressSuiteResult:
    suite = load_stress_scenario_suite(suite_path)
    normal_suite = load_scenario_suite(normal_suite_path)
    selected = _select_stress_scenarios(suite, split, scenario_id, max_scenarios, confirm_holdout_evaluation)
    algorithms = _normalize_algorithms(algorithms)
    suite_hash = stress_suite_hash(suite)
    if dry_run:
        return StressSuiteResult(suite_hash, split, tuple(item.scenario_id for item in selected), algorithms, str(output_dir), True)
    root = Path(output_dir)
    _prepare_output_root(root, resume=resume)
    normal_root = Path(normal_artifact_dir)
    if not normal_root.is_dir():
        raise StressRunnerError(f"Persistent normal artifact directory does not exist: {normal_root}")
    config_fingerprint = _hash_directory_csvs(Path(config_dir))
    normal_by_id = {item.scenario_id: item for item in normal_suite.scenarios}
    source_commit = _git_commit()
    if resume:
        _verify_resume_manifest(root, suite_hash, split, selected, algorithms, config_fingerprint)
    run_manifest = {
        "schema_version": STRESS_SCHEMA_VERSION,
        "runner_name": "stress_robustness_runner_v1",
        "suite_hash": suite_hash,
        "suite_path": str(Path(suite_path)),
        "parent_suite": suite.suite_name,
        "split": split,
        "scenario_ids": [item.scenario_id for item in selected],
        "algorithm_aliases": list(algorithms),
        "algorithm_names": [ALGORITHM_LABELS[item] for item in algorithms],
        "uses_cp_sat": False,
        "normal_artifact_dir": str(normal_root),
        "config_fingerprint": config_fingerprint,
        "source_git_commit": source_commit,
        "source_git_dirty": bool(_git_status()),
        "holdout_confirmed": bool(confirm_holdout_evaluation and split == "holdout"),
        "created_at": _now(),
    }
    _write_json(root / "suite_manifest_snapshot.json", suite.to_dict())
    _write_json(root / "run_manifest.json", run_manifest)
    all_rows: list[dict[str, Any]] = []
    difficulty_rows: list[dict[str, Any]] = []
    transformation_rows: list[dict[str, Any]] = []
    negative_rows: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    completed_ids: list[str] = []
    for raw_spec in selected:
        if raw_spec.base_scenario_id not in normal_by_id:
            raise StressRunnerError(f"Base scenario is not in the normal manifest: {raw_spec.base_scenario_id}")
        base_normal = normal_by_id[raw_spec.base_scenario_id]
        bound = BoundStressScenario(raw_spec, base_normal.data_generation_seed, base_normal.section_planning_seed, base_normal.algorithm_seed)
        scenario_dir = root / "scenarios" / raw_spec.scenario_id
        if resume:
            cached = _load_cached_stress_scenario(
                scenario_dir,
                raw_spec,
                suite_hash=suite_hash,
                config_fingerprint=config_fingerprint,
                normal_root=normal_root,
            )
            if cached is not None:
                all_rows.extend(cached["scenario_rows"])
                difficulty_rows.append(cached["difficulty_row"])
                transformation_rows.append(cached["transformation_row"])
                paired_rows.extend(_paired_stress_rows(cached["scenario_rows"], raw_spec, normal_root))
                if cached["negative_row"] is not None:
                    negative_rows.append(cached["negative_row"])
                completed_ids.append(raw_spec.scenario_id)
                continue
        try:
            result = _run_stress_scenario(bound, suite_hash, normal_root, config_fingerprint, source_commit, Path(config_dir), scenario_dir, algorithms)
            all_rows.extend(result["scenario_rows"])
            difficulty_rows.append(result["difficulty_row"])
            transformation_rows.append(result["transformation_row"])
            paired_rows.extend(_paired_stress_rows(result["scenario_rows"], raw_spec, normal_root))
            if result["negative_row"] is not None:
                negative_rows.append(result["negative_row"])
            completed_ids.append(raw_spec.scenario_id)
        except Exception as exc:
            failure = {"scenario_id": raw_spec.scenario_id, "error": str(exc), "expected_feasibility": raw_spec.expected_feasibility}
            failures.append(failure)
            _write_json(scenario_dir / "scenario_result.json", {"schema_version": STRESS_SCHEMA_VERSION, "status": "failed", **failure})
            if raw_spec.expected_feasibility == "structurally_infeasible":
                raise StressRunnerError(f"Structural scenario failed to build: {raw_spec.scenario_id}: {exc}") from exc
    ordinary_rows = [row for row in all_rows if row.get("scenario_family") != "structural_infeasibility" and row.get("row_type") == "overall" and row.get("completed")]
    _write_rows_csv(root / "scenario_results.csv", [row for row in all_rows if row.get("row_type") == "overall"])
    _write_rows_csv(root / "grade_subgroup_results.csv", [row for row in all_rows if row.get("row_type") == "grade"])
    _write_rows_csv(root / "input_difficulty.csv", difficulty_rows)
    _write_rows_csv(root / "transformation_summary.csv", transformation_rows)
    _write_rows_csv(root / "paired_normal_stress_comparison.csv", paired_rows)
    _write_json(root / "aggregate_stress_summary.json", _aggregate_stress_summary(ordinary_rows, paired_rows))
    _write_rows_csv(root / "negative_scenario_summary.csv", negative_rows)
    _write_json(root / "negative_scenario_summary.json", {"scenario_count": len(negative_rows), "scenarios": negative_rows})
    run_manifest["completed_scenario_ids"] = completed_ids
    run_manifest["failed_scenario_ids"] = [item.scenario_id for item in selected if item.scenario_id not in completed_ids]
    _write_json(root / "run_manifest.json", run_manifest)
    _write_json(root / "failures.json", {"failures": failures, "unexpected_failure_count": len(failures)})
    _write_json(root / "artifact_provenance.json", {
        "source_git_commit": source_commit,
        "suite_name": suite.suite_name,
        "suite_version": suite.suite_version,
        "split": split,
        "scenario_count": len(selected),
        "overall_result_rows": len([row for row in all_rows if row.get("row_type") == "overall"]),
        "grade_subgroup_rows": len([row for row in all_rows if row.get("row_type") == "grade"]),
        "paired_comparison_rows": len(paired_rows),
        "holdout_scenarios_run": len(selected) if split == "holdout" else 0,
        "development_data": split == "development",
        "holdout_data": split == "holdout",
        "not_a_generalization_claim": True,
        "normal_artifact_dir": str(normal_root),
        "stable_reference_scenario_id": next((item.scenario_id for item in normal_suite.scenarios if item.expected_reference_fingerprint is not None), ""),
        "stable_reference_fingerprint": _read_json(
            normal_root / "scenarios" / next(item.scenario_id for item in normal_suite.scenarios if item.expected_reference_fingerprint is not None) / "input_fingerprint.json"
        ),
        "created_at": _now(),
        "suite_manifest_sha256": _sha256_file(root / "suite_manifest_snapshot.json"),
        "run_manifest_sha256": _sha256_file(root / "run_manifest.json"),
    })
    _write_sha256_sums(root)
    return StressSuiteResult(suite_hash, split, tuple(item.scenario_id for item in selected), algorithms, str(root), False, tuple(all_rows))


def _run_stress_scenario(bound, suite_hash, normal_root, config_fingerprint, source_commit, config_dir, scenario_dir, algorithms):
    spec = bound.spec
    base_dir = normal_root / "scenarios" / spec.base_scenario_id
    base_result_path = base_dir / "scenario_result.json"
    if not base_result_path.is_file():
        raise StressRunnerError(f"Base scenario is not completed: {spec.base_scenario_id}")
    base_result = json.loads(base_result_path.read_text(encoding="utf-8"))
    if base_result.get("status") != "completed":
        raise StressRunnerError(f"Base scenario status is not completed: {spec.base_scenario_id}")
    transform_report = apply_stress_transforms(
        base_dir,
        scenario_dir,
        bound.to_transform_spec(),
        config_dir=config_dir,
        source_git_commit=source_commit,
    )
    generated_dir = scenario_dir / "generated"
    sections_dir = scenario_dir / "sections"
    catalog = pd.read_csv(config_dir / "course_catalog.csv", keep_default_na=False)
    allocation_input = canonicalize_allocation_input(
        pd.read_csv(generated_dir / "students.csv", keep_default_na=False),
        pd.read_csv(generated_dir / "requests.csv", keep_default_na=False),
        pd.read_csv(sections_dir / "sections.csv", keep_default_na=False),
        catalog,
    )
    manifest = build_experiment_manifest(
        generated_dir,
        sections_dir,
        config_dir,
        scenario_id="stable_year",
        seeds=ExperimentSeeds(bound.data_generation_seed, bound.section_planning_seed, bound.algorithm_seed),
    )
    _write_json(scenario_dir / "input_fingerprint.json", asdict(manifest.fingerprint))
    difficulty = build_input_difficulty(allocation_input)
    _write_json(scenario_dir / "input_difficulty.json", difficulty)
    base_difficulty = _read_json(base_dir / "input_difficulty.json")
    benchmark_dir = scenario_dir / "benchmark"
    benchmark = run_benchmark_suite(
        generated_input_dir=generated_dir,
        sections_input_dir=sections_dir,
        config_dir=config_dir,
        seeds=ExperimentSeeds(bound.data_generation_seed, bound.section_planning_seed, bound.algorithm_seed),
        scenario_id="stable_year",
        algorithms=algorithms,
        output_json_path=benchmark_dir / "benchmark_summary.json",
        output_csv_path=benchmark_dir / "scenario_algorithm_results.csv",
        output_artifact_dir=benchmark_dir,
        include_large_tables=True,
    )
    phase_a_spec = ScenarioSpec(
        scenario_id=spec.scenario_id,
        split=spec.split,
        scenario_family=spec.scenario_family,
        data_generation_seed=bound.data_generation_seed,
        section_planning_seed=bound.section_planning_seed,
        algorithm_seed=bound.algorithm_seed,
        generation_scenario_id="stable_year",
        enabled=spec.enabled,
        purpose=spec.purpose,
        tuning_allowed=spec.tuning_allowed,
        expected_reference_fingerprint=None,
        notes=spec.notes,
    )
    rows = _scenario_result_rows(phase_a_spec, manifest, benchmark, benchmark_dir)
    shock_severity = _shock_severity(spec)
    rows = [{**row, "shock_severity": shock_severity, "expected_feasibility": spec.expected_feasibility} for row in rows]
    negative_row = None
    certificate = transform_report.get("certificate")
    if spec.expected_feasibility == "structurally_infeasible":
        if not certificate:
            raise StressRunnerError(f"Missing certificate for {spec.scenario_id}")
        valid, reason = validate_certificate(certificate, allocation_input)
        if not valid:
            raise StressRunnerError(f"Certificate validation failed for {spec.scenario_id}: {reason}")
        overall = [row for row in rows if row.get("row_type") == "overall"]
        negative_row = {
            "scenario_id": spec.scenario_id,
            "scenario_family": spec.scenario_family,
            "expected_feasibility": spec.expected_feasibility,
            "certificate_type": certificate["certificate_type"],
            "certificate_valid": True,
            "runner_completed": all(row.get("completed") for row in overall),
            "algorithm_count": len(overall),
            "policy_fail_count": sum(not bool(row.get("final_schedule_policy_pass")) for row in overall),
            "consistency_issue_count": sum(int(row.get("consistency_issue_count", 0)) for row in overall),
            "assignment_nonpublishable": True,
            "certificate_reason": certificate.get("proof", ""),
        }
    _write_json(scenario_dir / "scenario_result.json", {
        "schema_version": STRESS_SCHEMA_VERSION,
        "status": "completed",
        "suite_hash": suite_hash,
        "scenario_spec_hash": _sha256_json(spec.to_dict()),
        "base_result_sha256": _sha256_file(base_result_path),
        "scenario_id": spec.scenario_id,
        "scenario_family": spec.scenario_family,
        "base_scenario_id": spec.base_scenario_id,
        "paired_normal_scenario_id": spec.paired_normal_scenario_id,
        "config_fingerprint": config_fingerprint,
        "source_git_commit": source_commit,
        "input_fingerprint": asdict(manifest.fingerprint),
        "scenario_rows": rows,
        "transformation_report": transform_report,
        "negative_summary": negative_row,
        "difficulty_row": _stress_difficulty_row(phase_a_spec, difficulty, base_difficulty, allocation_input, spec),
        "transformation_row": _transformation_row(transform_report, spec),
    })
    return {
        "scenario_rows": rows,
        "difficulty_row": _stress_difficulty_row(phase_a_spec, difficulty, base_difficulty, allocation_input, spec),
        "transformation_row": _transformation_row(transform_report, spec),
        "negative_row": negative_row,
    }


def _verify_resume_manifest(root, suite_hash, split, selected, algorithms, config_fingerprint):
    path = root / "run_manifest.json"
    if not path.is_file():
        raise StressRunnerError(f"Cannot resume without {path}.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StressRunnerError(f"Invalid stress resume manifest: {path}: {exc}") from exc
    expected = {
        "suite_hash": suite_hash,
        "split": split,
        "scenario_ids": [item.scenario_id for item in selected],
        "algorithm_aliases": list(algorithms),
        "config_fingerprint": config_fingerprint,
    }
    mismatches = [key for key, value in expected.items() if payload.get(key) != value]
    if mismatches:
        raise StressRunnerError("Stress resume provenance mismatch: " + ", ".join(mismatches))


def _load_cached_stress_scenario(scenario_dir, spec, *, suite_hash, config_fingerprint, normal_root):
    path = scenario_dir / "scenario_result.json"
    if not path.is_file():
        if scenario_dir.exists() and any(scenario_dir.iterdir()):
            raise StressRunnerError(f"Cannot resume scenario without {path}.")
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StressRunnerError(f"Invalid cached stress scenario: {path}: {exc}") from exc
    expected = {
        "status": "completed",
        "suite_hash": suite_hash,
        "scenario_spec_hash": _sha256_json(spec.to_dict()),
        "config_fingerprint": config_fingerprint,
        "base_scenario_id": spec.base_scenario_id,
    }
    mismatches = [key for key, value in expected.items() if payload.get(key) != value]
    if mismatches:
        raise StressRunnerError(f"Cached stress scenario provenance mismatch for {spec.scenario_id}: {', '.join(mismatches)}")
    base_result = normal_root / "scenarios" / spec.base_scenario_id / "scenario_result.json"
    if payload.get("base_result_sha256") != _sha256_file(base_result):
        raise StressRunnerError(f"Cached stress scenario base provenance mismatch for {spec.scenario_id}")
    required = ("scenario_rows", "difficulty_row", "transformation_row")
    if any(not isinstance(payload.get(key), (list, dict)) for key in required):
        raise StressRunnerError(f"Cached stress scenario is missing exported rows: {path}")
    return {
        "scenario_rows": payload["scenario_rows"],
        "difficulty_row": payload["difficulty_row"],
        "transformation_row": payload["transformation_row"],
        "negative_row": payload.get("negative_summary"),
    }


def _paired_stress_rows(stress_rows, spec, normal_root):
    if spec.expected_feasibility == "structurally_infeasible":
        return []
    path = normal_root / "scenario_results.csv"
    if not path.is_file():
        raise StressRunnerError(f"Persistent normal scenario_results.csv is missing: {path}")
    normal = pd.read_csv(path, keep_default_na=False)
    normal = normal[(normal["scenario_id"] == spec.paired_normal_scenario_id) & (normal["row_type"] == "overall")]
    normal_by_algorithm = {str(row["algorithm"]): row for _, row in normal.iterrows()}
    rows = []
    for row in stress_rows:
        if row.get("row_type") != "overall":
            continue
        algorithm = str(row["algorithm"])
        base = normal_by_algorithm.get(algorithm)
        if base is None:
            raise StressRunnerError(f"Missing normal paired row for {spec.paired_normal_scenario_id}/{algorithm}")
        rows.append({
            "stress_scenario_id": spec.scenario_id,
            "normal_scenario_id": spec.paired_normal_scenario_id,
            "scenario_family": spec.scenario_family,
            "algorithm": algorithm,
            "algorithm_name": row.get("algorithm_name", ""),
            "primary_assigned_delta": int(row.get("primary_assigned", 0)) - int(base.get("primary_assigned", 0)),
            "stress_primary_satisfaction_rate": float(row.get("primary_satisfaction_rate", 0)),
            "normal_primary_satisfaction_rate": float(base.get("primary_satisfaction_rate", 0)),
            "primary_satisfaction_rate_delta": float(row.get("primary_satisfaction_rate", 0)) - float(base.get("primary_satisfaction_rate", 0)),
            "stress_logical_full_rate": float(row.get("logical_full_rate", 0)),
            "normal_logical_full_rate": float(base.get("logical_full_rate", 0)),
            "logical_full_rate_delta": float(row.get("logical_full_rate", 0)) - float(base.get("logical_full_rate", 0)),
            "logical_full_students_delta": int(row.get("logical_full_students", 0)) - int(base.get("logical_full_students", 0)),
            "stress_total_logical_gap": int(row.get("total_logical_gap", 0)),
            "normal_total_logical_gap": int(base.get("total_logical_gap", 0)),
            "total_logical_gap_delta": int(row.get("total_logical_gap", 0)) - int(base.get("total_logical_gap", 0)),
            "gap_over_1_students_delta": int(row.get("gap_over_1_students", 0)) - int(base.get("gap_over_1_students", 0)),
            "below_five_students_delta": int(row.get("below_five_students", 0)) - int(base.get("below_five_students", 0)),
            "stress_policy_violation_count": int(row.get("policy_violation_count", 0)),
            "normal_policy_violation_count": int(base.get("policy_violation_count", 0)),
            "policy_violation_delta": int(row.get("policy_violation_count", 0)) - int(base.get("policy_violation_count", 0)),
            "runtime_delta": float(row.get("runtime_seconds", 0)) - float(base.get("runtime_seconds", 0)),
            "shock_magnitude": json.dumps(spec.transforms, sort_keys=True),
            "shock_severity": _shock_severity(spec),
        })
    return rows


def _aggregate_stress_summary(rows, paired_rows=()):
    frame = pd.DataFrame(rows)
    result = {
        "schema_version": STRESS_SCHEMA_VERSION,
        "scope": "ordinary development stress scenarios only; structural negatives are excluded",
        "development_data": True,
        "not_a_generalization_claim": True,
        "algorithms": {},
    }
    if frame.empty:
        return result
    paired_frame = pd.DataFrame(paired_rows)
    for algorithm, group in frame.groupby("algorithm", sort=True):
        algorithm_paired = paired_frame[paired_frame["algorithm"] == algorithm] if not paired_frame.empty else pd.DataFrame()
        result["algorithms"][algorithm] = {
            "scenario_count": int(group["scenario_id"].nunique()),
            "completed_count": int(group["completed"].sum()),
            "policy_pass_count": int(group["final_schedule_policy_pass"].fillna(False).sum()),
            "policy_pass_rate": round(float(group["final_schedule_policy_pass"].fillna(False).mean()), 6),
            "metrics": {metric: _stats(group[metric]) for metric in ("primary_satisfaction_rate", "logical_full_rate", "total_logical_gap", "gap_over_1_students", "below_five_students")},
            "paired_degradation": {
                metric: _stats(algorithm_paired[metric])
                for metric in ("primary_satisfaction_rate_delta", "logical_full_rate_delta", "total_logical_gap_delta", "gap_over_1_students_delta", "below_five_students_delta", "runtime_delta")
            } if not algorithm_paired.empty else {},
        }
    result["by_scenario_family"] = {}
    for family, group in frame.groupby("scenario_family", sort=True):
        result["by_scenario_family"][family] = {}
        for severity, severity_group in group.groupby("shock_severity", sort=True):
            result["by_scenario_family"][family][str(severity)] = {
                "scenario_count": int(severity_group["scenario_id"].nunique()),
                "algorithms": sorted(severity_group["algorithm"].unique()),
                "metrics": {
                    "primary_satisfaction_rate": _stats(severity_group["primary_satisfaction_rate"]),
                    "logical_full_rate": _stats(severity_group["logical_full_rate"]),
                    "total_logical_gap": _stats(severity_group["total_logical_gap"]),
                    "gap_over_1_students": _stats(severity_group["gap_over_1_students"]),
                    "below_five_students": _stats(severity_group["below_five_students"]),
                },
            }
    return result


def _transformation_row(report, spec):
    return {
        "scenario_id": spec.scenario_id,
        "base_scenario_id": spec.base_scenario_id,
        "scenario_family": spec.scenario_family,
        "expected_feasibility": spec.expected_feasibility,
        "transform_order": ";".join(spec.transform_order),
        "transform_seed": spec.transform_seed,
        "students_before": report["rows_before"]["students"],
        "students_after": report["rows_after"]["students"],
        "requests_before": report["rows_before"]["requests"],
        "requests_after": report["rows_after"]["requests"],
        "section_rows_before": report["rows_before"]["section_rows"],
        "section_rows_after": report["rows_after"]["section_rows"],
        "students_added": report["students_added"],
        "requests_added": report["requests_added"],
        "alternate_requests_removed": report["alternate_requests_removed"],
        "logical_sections_changed": report["logical_sections_changed"],
        "logical_sections_removed": report["logical_sections_removed"],
        "capacity_removed": report["capacity_removed"],
        "base_canonical_input_hash": report["base_fingerprint"]["canonical_input_hash"],
        "transformed_canonical_input_hash": report["transformed_fingerprint"]["canonical_input_hash"],
        "deterministic_replay_hash": report["deterministic_replay_hash"],
        "validation_status": report["validation_status"],
        "shock_severity": _shock_severity(spec),
    }


def _stress_difficulty_row(phase_a_spec, difficulty, base_difficulty, allocation_input, spec):
    row = dict(_difficulty_row(phase_a_spec, difficulty))
    current_scale = difficulty["scale"]
    base_scale = base_difficulty["scale"]
    current_demand = difficulty["demand_capacity"]
    base_demand = base_difficulty["demand_capacity"]
    current_flexibility = difficulty["request_flexibility"]
    base_flexibility = base_difficulty["request_flexibility"]
    current_capacity = _planned_logical_capacity(difficulty)
    base_capacity = _planned_logical_capacity(base_difficulty)
    row.update({
        "paired_normal_scenario_id": spec.paired_normal_scenario_id,
        "expected_feasibility": spec.expected_feasibility,
        "shock_severity": _shock_severity(spec),
        "student_count_delta": current_scale["students"] - base_scale["students"],
        "primary_count_delta": current_scale["logical_primaries"] - base_scale["logical_primaries"],
        "alternate_count_delta": current_scale["alternates"] - base_scale["alternates"],
        "planned_logical_capacity": current_capacity,
        "paired_normal_planned_logical_capacity": base_capacity,
        "planned_logical_capacity_delta": current_capacity - base_capacity,
        "section_count_delta": current_scale["logical_sections"] - base_scale["logical_sections"],
        "candidate_edge_delta": current_scale["candidate_edges"] - base_scale["candidate_edges"],
        "zero_candidate_primary_delta": current_flexibility["primaries_with_zero_candidates"] - base_flexibility["primaries_with_zero_candidates"],
        "over_capacity_course_delta": current_demand["courses_with_primary_demand_over_capacity"] - base_demand["courses_with_primary_demand_over_capacity"],
        "capacity_only_primary_shortfall_delta": current_demand["total_capacity_only_primary_shortfall"] - base_demand["total_capacity_only_primary_shortfall"],
        "students_with_zero_alternates": _distribution_count(difficulty["request_flexibility"]["alternate_count_distribution"], "0"),
        "median_alternates_per_student": _distribution_median(difficulty["request_flexibility"]["alternate_count_distribution"]),
        "high_pressure_course_ratios": json.dumps(
            sorted(
                [
                    {"logical_course_key": item["logical_course_key"], "ratio": item["primary_demand_capacity_ratio"]}
                    for item in current_demand["courses"]
                    if item["primary_demand_capacity_ratio"] is not None
                ],
                key=lambda item: (-float(item["ratio"]), item["logical_course_key"]),
            )[: max(1, len(current_demand["courses"]) // 10)] if current_demand["courses"] else [],
            sort_keys=True,
        ),
        "minimum_five_capacity_margin": current_capacity - 5 * current_scale["students"],
        "certificate_status": "valid" if spec.expected_feasibility == "structurally_infeasible" else "not_applicable",
    })
    return row


def _planned_logical_capacity(difficulty):
    return int(sum(int(row.get("planned_capacity", 0)) for row in difficulty["demand_capacity"]["courses"]))


def _distribution_count(distribution, key):
    return int(distribution.get(key, 0))


def _distribution_median(distribution):
    values = []
    for key, count in distribution.items():
        values.extend([int(key)] * int(count))
    if not values:
        return 0.0
    values.sort()
    middle = len(values) // 2
    return float(values[middle]) if len(values) % 2 else round((values[middle - 1] + values[middle]) / 2, 6)


def _select_stress_scenarios(suite, split, scenario_id, max_scenarios, confirm_holdout):
    if split not in {"development", "holdout"}:
        raise StressRunnerError("split must be development or holdout")
    if split == "holdout" and not confirm_holdout:
        raise StressRunnerError("Holdout evaluation requires confirm_holdout_evaluation=True")
    selected = [item for item in suite.scenarios if item.enabled and item.split == split]
    if scenario_id:
        selected = [item for item in selected if item.scenario_id == scenario_id]
        if not selected:
            raise StressRunnerError(f"Enabled stress scenario '{scenario_id}' is not in split '{split}'.")
    if max_scenarios is not None:
        if max_scenarios <= 0:
            raise StressRunnerError("max_scenarios must be positive")
        selected = selected[:max_scenarios]
    if not selected:
        raise StressRunnerError("No enabled stress scenarios selected")
    return tuple(selected)


def _normalize_algorithms(algorithms):
    normalized = tuple(item.strip().lower() for item in algorithms if item.strip())
    if not normalized or any(item not in GREEDY_ALGORITHMS for item in normalized):
        raise StressRunnerError("Stress runner accepts the four Greedy algorithms only; CP-SAT is excluded")
    return tuple(dict.fromkeys(normalized))


def _shock_severity(spec):
    percentages = [float(item["percentage"]) for item in spec.transforms if "percentage" in item]
    counts = [int(item["count"]) for item in spec.transforms if "count" in item]
    return round(max(percentages, default=0.0) + max(counts, default=0) * 0.01, 6)


def _prepare_output_root(root, resume):
    if root.exists() and not root.is_dir():
        raise StressRunnerError(f"Output path is not a directory: {root}")
    if root.exists() and any(root.iterdir()) and not resume:
        raise StressRunnerError(f"Output directory is non-empty; use resume=True: {root}")
    root.mkdir(parents=True, exist_ok=True)


def _hash_directory_csvs(directory):
    digest = hashlib.sha256()
    if not directory.is_dir():
        raise StressRunnerError(f"Missing directory: {directory}")
    files = sorted(directory.rglob("*.csv"))
    if not files:
        raise StressRunnerError(f"No CSV files in: {directory}")
    for path in files:
        digest.update(path.relative_to(directory).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _read_json(path):
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StressRunnerError(f"Cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise StressRunnerError(f"JSON artifact must be an object: {path}")
    return payload


def _write_sha256_sums(root):
    paths = sorted(path for path in root.rglob("*") if path.is_file() and path.name != "SHA256SUMS.txt")
    lines = [f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root).as_posix()}" for path in paths]
    (root / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _stats(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"count": 0, "median": 0, "min": 0, "max": 0}
    return {"count": int(len(values)), "median": round(float(values.median()), 6), "min": round(float(values.min()), 6), "max": round(float(values.max()), 6)}


def _text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise StressManifestError(f"{field} must be non-empty text")
    return value.strip()


def _nonnegative_int(value, field):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise StressManifestError(f"{field} must be a non-negative integer")
    return value


def _sha256_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _git_status():
    try:
        return subprocess.check_output(["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return ""


def _now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase B Greedy stress robustness scenarios.")
    parser.add_argument("--suite", default=str(DEFAULT_STRESS_SUITE_PATH))
    parser.add_argument("--normal-suite", default=str(DEFAULT_NORMAL_SUITE_PATH))
    parser.add_argument("--normal-artifact-dir", default=str(DEFAULT_NORMAL_ARTIFACT_DIR))
    parser.add_argument("--split", choices=("development", "holdout"), default="development")
    parser.add_argument("--scenario-id")
    parser.add_argument("--algorithms", default=",".join(GREEDY_ALGORITHMS))
    parser.add_argument("--output-dir", default=str(DEFAULT_STRESS_ARTIFACT_DIR))
    parser.add_argument("--max-scenarios", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-holdout-evaluation", action="store_true")
    parser.add_argument("--config-dir", default="data/config")
    args = parser.parse_args(tuple(argv) if argv is not None else None)
    try:
        result = run_stress_robustness_suite(
            suite_path=args.suite,
            normal_suite_path=args.normal_suite,
            normal_artifact_dir=args.normal_artifact_dir,
            split=args.split,
            scenario_id=args.scenario_id,
            algorithms=tuple(args.algorithms.split(",")),
            output_dir=args.output_dir,
            max_scenarios=args.max_scenarios,
            resume=args.resume,
            dry_run=args.dry_run,
            confirm_holdout_evaluation=args.confirm_holdout_evaluation,
            config_dir=args.config_dir,
        )
    except (StressManifestError, StressRunnerError, ScenarioTransformError, ScenarioManifestError, ValueError) as exc:
        print(f"Stress robustness runner FAIL: {exc}")
        return 1
    print(f"Stress robustness runner PASS: {len(result.scenario_ids)} {result.split} scenario(s), algorithms={','.join(result.algorithms)}, dry_run={result.dry_run}")
    if result.output_dir:
        print(f"Artifacts: {result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
