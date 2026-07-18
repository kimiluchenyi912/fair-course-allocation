"""Run the frozen normal-year Greedy robustness benchmark suite.

This module owns experiment orchestration and provenance only.  Generation,
section planning, canonicalization, and allocation semantics remain in their
existing modules.
"""

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
from src.benchmark_runner import BenchmarkSuiteResult, run_benchmark_suite
from src.experiment_manifest import (
    CanonicalInputFingerprint,
    ExperimentManifest,
    ExperimentSeeds,
    build_experiment_manifest,
    canonical_input_fingerprint,
)
from src.generation.student_generator import generate_synthetic_dataset
from src.input_difficulty import build_input_difficulty
from src.section_planning.runner import plan_sections_from_files, write_result_atomic


ROBUSTNESS_SCHEMA_VERSION = 1
DEFAULT_SUITE_PATH = Path("data/scenarios/normal_year_robustness_v1.json")
GREEDY_ALGORITHMS = ("random", "fcfs", "grade_priority", "constrained")
ALGORITHM_LABELS = {
    "random": "seeded_random_greedy",
    "fcfs": "first_come_first_served_greedy",
    "grade_priority": "grade_priority_greedy",
    "constrained": "constrained_first_greedy",
}
SCENARIO_RESULT_FILE = "scenario_result.json"


class ScenarioManifestError(ValueError):
    """Raised when a frozen scenario manifest is malformed or ambiguous."""


class RobustnessRunnerError(ValueError):
    """Raised when a robustness run cannot proceed safely."""


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    split: str
    scenario_family: str
    data_generation_seed: int
    section_planning_seed: int
    algorithm_seed: int
    generation_scenario_id: str
    enabled: bool
    purpose: str
    tuning_allowed: bool
    expected_reference_fingerprint: dict[str, Any] | None
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ScenarioSuite:
    schema_version: int
    suite_name: str
    suite_version: str
    default_split: str
    scenarios: tuple[ScenarioSpec, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "suite_name": self.suite_name,
            "suite_version": self.suite_version,
            "default_split": self.default_split,
            "scenarios": [scenario.to_dict() for scenario in self.scenarios],
        }


@dataclass(frozen=True)
class RobustnessSuiteResult:
    suite_hash: str
    split: str
    scenario_ids: tuple[str, ...]
    algorithms: tuple[str, ...]
    dry_run: bool
    output_dir: str | None
    scenario_rows: tuple[dict[str, Any], ...] = ()
    aggregate_summary: dict[str, Any] | None = None
    paired_rows: tuple[dict[str, Any], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": ROBUSTNESS_SCHEMA_VERSION,
            "suite_hash": self.suite_hash,
            "split": self.split,
            "scenario_ids": list(self.scenario_ids),
            "algorithms": list(self.algorithms),
            "dry_run": self.dry_run,
            "output_dir": self.output_dir,
            "scenario_rows": list(self.scenario_rows),
            "aggregate_summary": self.aggregate_summary,
            "paired_rows": list(self.paired_rows),
        }


def load_scenario_suite(path: str | Path = DEFAULT_SUITE_PATH) -> ScenarioSuite:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScenarioManifestError(f"Cannot read scenario manifest {source}: {exc}") from exc
    return validate_scenario_suite(payload)


def validate_scenario_suite(payload: Any) -> ScenarioSuite:
    if not isinstance(payload, dict):
        raise ScenarioManifestError("Scenario manifest must be a JSON object.")
    required = {"schema_version", "suite_name", "suite_version", "default_split", "scenarios"}
    missing = sorted(required - set(payload))
    if missing:
        raise ScenarioManifestError(f"Scenario manifest is missing required fields: {', '.join(missing)}")
    if payload["schema_version"] != 1:
        raise ScenarioManifestError("Only scenario manifest schema_version=1 is supported.")
    if payload["default_split"] not in {"development", "holdout"}:
        raise ScenarioManifestError("default_split must be development or holdout.")
    raw_scenarios = payload["scenarios"]
    if not isinstance(raw_scenarios, list) or not raw_scenarios:
        raise ScenarioManifestError("scenarios must be a non-empty list.")

    scenario_required = {
        "scenario_id",
        "split",
        "scenario_family",
        "data_generation_seed",
        "section_planning_seed",
        "algorithm_seed",
        "generation_scenario_id",
        "enabled",
        "purpose",
        "tuning_allowed",
        "expected_reference_fingerprint",
        "notes",
    }
    scenarios: list[ScenarioSpec] = []
    ids: set[str] = set()
    seed_pairs: set[tuple[int, int]] = set()
    reference_count = 0
    for index, raw in enumerate(raw_scenarios, start=1):
        if not isinstance(raw, dict):
            raise ScenarioManifestError(f"scenarios[{index}] must be an object.")
        missing = sorted(scenario_required - set(raw))
        if missing:
            raise ScenarioManifestError(f"scenarios[{index}] is missing: {', '.join(missing)}")
        scenario_id = _required_text(raw["scenario_id"], f"scenarios[{index}].scenario_id")
        if scenario_id in ids:
            raise ScenarioManifestError(f"Duplicate scenario_id: {scenario_id}")
        ids.add(scenario_id)
        split = raw["split"]
        if split not in {"development", "holdout"}:
            raise ScenarioManifestError(f"{scenario_id}: split must be development or holdout.")
        data_seed = _nonnegative_int(raw["data_generation_seed"], f"{scenario_id}.data_generation_seed")
        section_seed = _nonnegative_int(raw["section_planning_seed"], f"{scenario_id}.section_planning_seed")
        algorithm_seed = _nonnegative_int(raw["algorithm_seed"], f"{scenario_id}.algorithm_seed")
        pair = (data_seed, section_seed)
        if pair in seed_pairs:
            raise ScenarioManifestError(
                f"{scenario_id}: data_generation_seed and section_planning_seed pair is not unique."
            )
        seed_pairs.add(pair)
        enabled = raw["enabled"]
        tuning_allowed = raw["tuning_allowed"]
        if not isinstance(enabled, bool) or not isinstance(tuning_allowed, bool):
            raise ScenarioManifestError(f"{scenario_id}: enabled and tuning_allowed must be booleans.")
        if split == "holdout" and tuning_allowed:
            raise ScenarioManifestError(f"{scenario_id}: holdout scenarios cannot allow tuning.")
        expected = raw["expected_reference_fingerprint"]
        if expected is not None:
            reference_count += 1
            _validate_reference_fingerprint(expected, scenario_id)
            if split != "development":
                raise ScenarioManifestError(f"{scenario_id}: the reference fingerprint must be development-only.")
        scenarios.append(
            ScenarioSpec(
                scenario_id=scenario_id,
                split=split,
                scenario_family=_required_text(raw["scenario_family"], f"{scenario_id}.scenario_family"),
                data_generation_seed=data_seed,
                section_planning_seed=section_seed,
                algorithm_seed=algorithm_seed,
                generation_scenario_id=_required_text(
                    raw["generation_scenario_id"], f"{scenario_id}.generation_scenario_id"
                ),
                enabled=enabled,
                purpose=_required_text(raw["purpose"], f"{scenario_id}.purpose"),
                tuning_allowed=tuning_allowed,
                expected_reference_fingerprint=expected,
                notes=_required_text(raw["notes"], f"{scenario_id}.notes"),
            )
        )
    if reference_count != 1:
        raise ScenarioManifestError("The suite must contain exactly one expected reference fingerprint.")
    if scenarios[0].expected_reference_fingerprint is None:
        raise ScenarioManifestError("The first scenario must be the frozen reference scenario.")
    return ScenarioSuite(
        schema_version=1,
        suite_name=_required_text(payload["suite_name"], "suite_name"),
        suite_version=_required_text(payload["suite_version"], "suite_version"),
        default_split=payload["default_split"],
        scenarios=tuple(scenarios),
    )


def scenario_suite_hash(suite: ScenarioSuite) -> str:
    return _sha256_json(suite.to_dict())


def run_robustness_suite(
    *,
    suite_path: str | Path = DEFAULT_SUITE_PATH,
    split: str = "development",
    scenario_id: str | None = None,
    algorithms: tuple[str, ...] = GREEDY_ALGORITHMS,
    output_dir: str | Path | None = None,
    max_scenarios: int | None = None,
    resume: bool = False,
    dry_run: bool = False,
    confirm_holdout_evaluation: bool = False,
    config_dir: str | Path = "data/config",
    templates_dir: str | Path = "data/templates",
) -> RobustnessSuiteResult:
    suite = load_scenario_suite(suite_path)
    selected = _select_scenarios(suite, split, scenario_id, max_scenarios, confirm_holdout_evaluation)
    algorithms = _normalize_greedy_algorithms(algorithms)
    suite_hash = scenario_suite_hash(suite)
    if dry_run:
        return RobustnessSuiteResult(
            suite_hash=suite_hash,
            split=split,
            scenario_ids=tuple(item.scenario_id for item in selected),
            algorithms=algorithms,
            dry_run=True,
            output_dir=str(output_dir) if output_dir is not None else None,
        )
    if output_dir is None:
        raise RobustnessRunnerError("output_dir is required unless --dry-run is used.")
    root = Path(output_dir)
    _prepare_output_root(root, resume=resume)
    config_fingerprint = _hash_input_files(Path(config_dir), Path(templates_dir))
    source_commit = _git_commit()
    run_manifest = {
        "schema_version": ROBUSTNESS_SCHEMA_VERSION,
        "runner_name": "robustness_runner_v1",
        "suite_hash": suite_hash,
        "suite_path": str(Path(suite_path)),
        "split": split,
        "scenario_ids": [item.scenario_id for item in selected],
        "algorithm_aliases": list(algorithms),
        "algorithm_names": [ALGORITHM_LABELS[item] for item in algorithms],
        "uses_cp_sat": False,
        "config_templates_fingerprint": config_fingerprint,
        "source_git_commit": source_commit,
        "source_git_dirty": bool(_git_status()),
        "holdout_confirmed": bool(confirm_holdout_evaluation and split == "holdout"),
        "resume": resume,
        "created_at": _now(),
    }
    _write_json(root / "suite_manifest_snapshot.json", suite.to_dict())
    _write_json(root / "run_manifest.json", run_manifest)

    scenario_rows: list[dict[str, Any]] = []
    difficulty_rows: list[dict[str, Any]] = []
    completed_ids: list[str] = []
    for spec in selected:
        scenario_dir = root / "scenarios" / spec.scenario_id
        cached = _load_cached_scenario(
            scenario_dir,
            spec,
            suite_hash=suite_hash,
            config_fingerprint=config_fingerprint,
            resume=resume,
        )
        if cached is not None:
            scenario_rows.extend(cached["scenario_rows"])
            difficulty_rows.append(cached["input_difficulty_row"])
            completed_ids.append(spec.scenario_id)
            continue
        try:
            result = _run_scenario(
                spec,
                suite_hash=suite_hash,
                config_fingerprint=config_fingerprint,
                config_dir=Path(config_dir),
                templates_dir=Path(templates_dir),
                scenario_dir=scenario_dir,
                algorithms=algorithms,
            )
        except Exception as exc:
            scenario_rows.extend(_failed_scenario_rows(spec, str(exc), algorithms))
            difficulty_rows.append(
                {
                    "scenario_id": spec.scenario_id,
                    "split": spec.split,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            _write_json(
                scenario_dir / SCENARIO_RESULT_FILE,
                {
                    "schema_version": ROBUSTNESS_SCHEMA_VERSION,
                    "status": "failed",
                    "scenario_id": spec.scenario_id,
                    "suite_hash": suite_hash,
                    "scenario_spec_hash": _sha256_json(spec.to_dict()),
                    "config_templates_fingerprint": config_fingerprint,
                    "error": str(exc),
                },
            )
            if spec.expected_reference_fingerprint is not None:
                raise RobustnessRunnerError(f"Frozen reference scenario failed: {spec.scenario_id}: {exc}") from exc
            continue
        scenario_rows.extend(result["scenario_rows"])
        difficulty_rows.append(result["input_difficulty_row"])
        completed_ids.append(spec.scenario_id)

    aggregate = build_aggregate_summary(scenario_rows)
    paired = build_paired_comparison(scenario_rows)
    _write_rows_csv(root / "scenario_results.csv", scenario_rows)
    _write_rows_csv(root / "input_difficulty.csv", difficulty_rows)
    _write_json(root / "aggregate_summary.json", aggregate)
    _write_rows_csv(root / "paired_algorithm_comparison.csv", paired)
    run_manifest["completed_scenario_ids"] = completed_ids
    run_manifest["failed_scenario_ids"] = [item.scenario_id for item in selected if item.scenario_id not in completed_ids]
    _write_json(root / "run_manifest.json", run_manifest)
    return RobustnessSuiteResult(
        suite_hash=suite_hash,
        split=split,
        scenario_ids=tuple(item.scenario_id for item in selected),
        algorithms=algorithms,
        dry_run=False,
        output_dir=str(root),
        scenario_rows=tuple(scenario_rows),
        aggregate_summary=aggregate,
        paired_rows=tuple(paired),
    )


def build_aggregate_summary(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame([row for row in rows if row.get("row_type") == "overall" and row.get("completed")])
    metrics = (
        "primary_satisfaction_rate",
        "logical_full_rate",
        "primary_assigned",
        "total_logical_gap",
        "ordinary_violations",
        "protected_violations",
        "high_demand_violations",
        "gap_one_students",
        "gap_over_1_students",
        "below_five_students",
        "over_target_students",
        "runtime_seconds",
    )
    result: dict[str, Any] = {
        "schema_version": ROBUSTNESS_SCHEMA_VERSION,
        "row_semantics": "Only completed overall scenario rows are included in numeric metric distributions.",
        "algorithms": {},
    }
    all_rows = pd.DataFrame([row for row in rows if row.get("row_type") == "overall"])
    for algorithm in sorted(all_rows["algorithm"].dropna().unique()) if not all_rows.empty else ():
        completed = frame[frame["algorithm"] == algorithm] if not frame.empty else pd.DataFrame()
        observed = all_rows[all_rows["algorithm"] == algorithm]
        summary: dict[str, Any] = {
            "scenario_count": int(observed["scenario_id"].nunique()),
            "completed_count": int(len(completed)),
            "failed_count": int(observed["scenario_id"].nunique() - completed["scenario_id"].nunique()),
            "policy_pass_count": int(completed["final_schedule_policy_pass"].fillna(False).sum()) if not completed.empty else 0,
            "policy_pass_rate": round(float(completed["final_schedule_policy_pass"].fillna(False).mean()), 6) if not completed.empty else 0.0,
            "metrics": {},
        }
        for metric in metrics:
            if completed.empty or metric not in completed:
                summary["metrics"][metric] = _empty_stats()
                continue
            values = pd.to_numeric(completed[metric], errors="coerce").dropna()
            summary["metrics"][metric] = _stats(values)
        result["algorithms"][algorithm] = summary
    return result


def build_paired_comparison(rows: Iterable[dict[str, Any]]) -> tuple[dict[str, Any], ...]:
    overall = [row for row in rows if row.get("row_type") == "overall" and row.get("completed")]
    by_scenario = {(row["scenario_id"], row["algorithm"]): row for row in overall}
    paired: list[dict[str, Any]] = []
    for scenario_id in sorted({row["scenario_id"] for row in overall}):
        for baseline_algorithm in ("random", "fcfs", "grade_priority"):
            baseline = by_scenario.get((scenario_id, baseline_algorithm))
            candidate = by_scenario.get((scenario_id, "constrained"))
            if baseline is None or candidate is None:
                continue
            paired.append(
                {
                    "scenario_id": scenario_id,
                    "baseline_algorithm": baseline_algorithm,
                    "comparison_algorithm": "constrained",
                    "baseline_algorithm_name": ALGORITHM_LABELS[baseline_algorithm],
                    "comparison_algorithm_name": ALGORITHM_LABELS["constrained"],
                    "primary_assigned_delta": int(candidate["primary_assigned"] - baseline["primary_assigned"]),
                    "logical_full_students_delta": int(candidate["logical_full_students"] - baseline["logical_full_students"]),
                    "policy_violation_delta": int(candidate["policy_violation_count"] - baseline["policy_violation_count"]),
                }
            )
    return tuple(paired)


def _run_scenario(
    spec: ScenarioSpec,
    *,
    suite_hash: str,
    config_fingerprint: str,
    config_dir: Path,
    templates_dir: Path,
    scenario_dir: Path,
    algorithms: tuple[str, ...],
) -> dict[str, Any]:
    generated_dir = scenario_dir / "generated"
    sections_dir = scenario_dir / "sections"
    benchmark_dir = scenario_dir / "benchmark"
    generated_dir.mkdir(parents=True, exist_ok=True)
    generated = generate_synthetic_dataset(
        config_dir,
        spec.generation_scenario_id,
        spec.data_generation_seed,
        templates_dir=templates_dir,
    )
    _write_generation_result(generated, generated_dir, spec)
    planned = plan_sections_from_files(
        generated_dir,
        config_dir,
        spec.generation_scenario_id,
        spec.section_planning_seed,
        templates_dir=templates_dir,
    )
    write_result_atomic(planned, sections_dir)
    manifest = build_experiment_manifest(
        generated_dir,
        sections_dir,
        config_dir,
        scenario_id=spec.generation_scenario_id,
        seeds=ExperimentSeeds(spec.data_generation_seed, spec.section_planning_seed, spec.algorithm_seed),
    )
    if spec.expected_reference_fingerprint is not None:
        _check_reference(manifest.fingerprint, spec.expected_reference_fingerprint, spec.scenario_id)
    allocation_input = canonicalize_allocation_input(
        pd.read_csv(generated_dir / "students.csv", keep_default_na=False),
        pd.read_csv(generated_dir / "requests.csv", keep_default_na=False),
        pd.read_csv(sections_dir / "sections.csv", keep_default_na=False),
        pd.read_csv(config_dir / "course_catalog.csv", keep_default_na=False),
    )
    _write_json(scenario_dir / "input_fingerprint.json", asdict(manifest.fingerprint))
    difficulty = build_input_difficulty(allocation_input)
    _write_json(scenario_dir / "input_difficulty.json", difficulty)
    benchmark = run_benchmark_suite(
        generated_input_dir=generated_dir,
        sections_input_dir=sections_dir,
        config_dir=config_dir,
        seeds=ExperimentSeeds(spec.data_generation_seed, spec.section_planning_seed, spec.algorithm_seed),
        scenario_id=spec.generation_scenario_id,
        expected_fingerprint=(
            _fingerprint_from_dict(spec.expected_reference_fingerprint)
            if spec.expected_reference_fingerprint is not None
            else None
        ),
        algorithms=algorithms,
        output_json_path=benchmark_dir / "benchmark_summary.json",
        output_csv_path=benchmark_dir / "scenario_algorithm_results.csv",
        output_artifact_dir=benchmark_dir,
        include_large_tables=True,
    )
    rows = _scenario_result_rows(spec, manifest, benchmark, benchmark_dir)
    difficulty_row = _difficulty_row(spec, difficulty)
    _write_json(
        scenario_dir / SCENARIO_RESULT_FILE,
        {
            "schema_version": ROBUSTNESS_SCHEMA_VERSION,
            "status": "completed",
            "scenario_id": spec.scenario_id,
            "suite_hash": suite_hash,
            "scenario_spec_hash": _sha256_json(spec.to_dict()),
            "config_templates_fingerprint": config_fingerprint,
            "source_git_commit": _git_commit(),
            "created_at": _now(),
            "resumed": False,
            "generated_input_hash": manifest.generated_input_hash,
            "section_input_hash": manifest.section_input_hash,
            "generated_file_hashes": {
                "students.csv": _sha256_file(generated_dir / "students.csv"),
                "requests.csv": _sha256_file(generated_dir / "requests.csv"),
            },
            "section_file_hashes": {
                "sections.csv": _sha256_file(sections_dir / "sections.csv"),
            },
            "input_fingerprint": asdict(manifest.fingerprint),
            "input_difficulty_row": difficulty_row,
            "scenario_rows": rows,
        },
    )
    return {"scenario_rows": rows, "input_difficulty_row": difficulty_row}


def _failed_scenario_rows(
    spec: ScenarioSpec,
    error: str,
    algorithms: tuple[str, ...],
) -> list[dict[str, Any]]:
    return [
        {
            "scenario_id": spec.scenario_id,
            "scenario_family": spec.scenario_family,
            "split": spec.split,
            "row_type": "overall",
            "grade": "",
            "algorithm": alias,
            "algorithm_name": ALGORITHM_LABELS[alias],
            "algorithm_label": ALGORITHM_LABELS[alias],
            "completed": False,
            "status": "failed",
            "data_generation_seed": spec.data_generation_seed,
            "section_planning_seed": spec.section_planning_seed,
            "algorithm_seed": spec.algorithm_seed,
            "error": error,
        }
        for alias in algorithms
    ]


def _scenario_result_rows(
    spec: ScenarioSpec,
    manifest: ExperimentManifest,
    benchmark: BenchmarkSuiteResult,
    benchmark_dir: Path,
) -> list[dict[str, Any]]:
    students = pd.read_csv(benchmark_dir / "student_outcomes.csv", keep_default_na=False)
    requests = pd.read_csv(benchmark_dir / "request_outcomes.csv", keep_default_na=False)
    rows: list[dict[str, Any]] = []
    for result in benchmark.results:
        student_rows = students[students["algorithm_name"] == result.algorithm_name]
        request_rows = requests[requests["algorithm_name"] == result.algorithm_name]
        logical_full = _sum_int(student_rows, "logical_fully_scheduled")
        gap_students = _sum_int(student_rows, "logical_schedule_gap_count", predicate=lambda value: value > 0)
        target_total = _sum_int(student_rows, "target_logical_course_count")
        assigned_total = _sum_int(student_rows, "assigned_logical_course_count")
        row = _overall_row(
            spec,
            manifest,
            result.algorithm_name,
            result.to_dict(),
            student_rows,
            request_rows,
            logical_full=logical_full,
            gap_students=gap_students,
            target_total=target_total,
            assigned_total=assigned_total,
        )
        rows.append(row)
        for grade, group in student_rows.groupby("grade", sort=True):
            rows.append(
                {
                    **row,
                    "row_type": "grade",
                    "grade": int(grade),
                    "students": int(len(group)),
                    "primary_assigned": _sum_int(group, "primary_assigned_count"),
                    "primary_unmet": _sum_int(group, "primary_unmet_count"),
                    "primary_satisfaction_rate": round(
                        _sum_int(group, "primary_assigned_count") / max(_sum_int(group, "primary_request_count"), 1), 6
                    ),
                    "logical_full_students": _sum_int(group, "logical_fully_scheduled"),
                    "logical_full_rate": round(_sum_int(group, "logical_fully_scheduled") / max(len(group), 1), 6),
                    "mean_logical_gap": round(
                        float(pd.to_numeric(group["logical_schedule_gap_count"], errors="coerce").fillna(0).mean()),
                        6,
                    ),
                    "total_logical_gap": _sum_int(group, "logical_schedule_gap_count"),
                    "policy_violation_count": int(
                        _sum_int(group, "ordinary_fairness_violation")
                        + _sum_int(group, "protected_fairness_violation")
                        + _sum_int(group, "high_demand_guarantee_violation_count")
                    ),
                }
            )
    return rows


def _overall_row(
    spec: ScenarioSpec,
    manifest: ExperimentManifest,
    algorithm: str,
    result: dict[str, Any],
    students: pd.DataFrame,
    requests: pd.DataFrame,
    *,
    logical_full: int,
    gap_students: int,
    target_total: int,
    assigned_total: int,
) -> dict[str, Any]:
    policy_violation_count = int(result.get("ordinary_violations", 0)) + int(result.get("protected_violations", 0)) + int(result.get("high_demand_violations", 0))
    return {
        "scenario_id": spec.scenario_id,
        "scenario_family": spec.scenario_family,
        "split": spec.split,
        "row_type": "overall",
        "grade": "",
        "algorithm": _algorithm_alias(algorithm),
        "algorithm_name": algorithm,
        "algorithm_label": _algorithm_label(algorithm),
        "completed": result.get("status") == "completed",
        "status": result.get("status"),
        "data_generation_seed": spec.data_generation_seed,
        "section_planning_seed": spec.section_planning_seed,
        "algorithm_seed": spec.algorithm_seed,
        "students": int(len(students)),
        "logical_requests": manifest.fingerprint.logical_requests,
        "logical_primaries": manifest.fingerprint.logical_primaries,
        "alternates": manifest.fingerprint.alternates,
        "logical_sections": manifest.fingerprint.logical_sections,
        "section_rows": manifest.fingerprint.section_rows,
        "candidate_edges": manifest.fingerprint.candidate_edges,
        "canonical_input_hash": manifest.fingerprint.canonical_input_hash,
        "primary_assigned": int(result.get("primary_assigned", 0)),
        "primary_unmet": int(result.get("primary_unmet", 0)),
        "primary_satisfaction_rate": float(result.get("primary_satisfaction_rate", 0.0)),
        "logical_full_students": logical_full,
        "logical_full_rate": round(logical_full / max(len(students), 1), 6),
        "target_logical_course_count": target_total,
        "assigned_logical_course_count": assigned_total,
        "total_logical_gap": int(result.get("total_logical_schedule_gap") or max(target_total - assigned_total, 0)),
        "gap_students": gap_students,
        "gap_one_students": _count_values(students, "logical_schedule_gap_count", lambda value: value == 1),
        "gap_over_1_students": _count_values(students, "logical_schedule_gap_count", lambda value: value > 1),
        "below_five_students": _count_values(students, "assigned_logical_course_count", lambda value: value < 5),
        "over_target_students": int(
            sum(
                assigned > target
                for assigned, target in zip(
                    pd.to_numeric(students["assigned_logical_course_count"], errors="coerce").fillna(0),
                    pd.to_numeric(students["target_logical_course_count"], errors="coerce").fillna(0),
                )
            )
        ) if not students.empty else 0,
        "max_primary_unmet_per_student": int(
            pd.to_numeric(students["primary_unmet_count"], errors="coerce").fillna(0).max()
        ) if not students.empty else 0,
        "ordinary_violations": int(result.get("ordinary_violations", 0)),
        "protected_violations": int(result.get("protected_violations", 0)),
        "high_demand_violations": int(result.get("high_demand_violations", 0)),
        "policy_violation_count": policy_violation_count,
        "final_schedule_policy_pass": result.get("final_schedule_policy_pass"),
        "consistency_issue_count": int(result.get("consistency_issue_count", 0)),
        "section_over_capacity_count": int(result.get("section_over_capacity_count", 0)),
        "period_conflict_count": _sum_int(requests, "rejected_period_conflict_count"),
        "duplicate_logical_course_rejection_count": _sum_int(requests, "rejected_duplicate_logical_course_count"),
        "total_alternates_assigned": int(result.get("total_alternates_assigned", 0)),
        "runtime_seconds": float(result.get("runtime_seconds", 0.0)),
    }


def _difficulty_row(spec: ScenarioSpec, difficulty: dict[str, Any]) -> dict[str, Any]:
    scale = difficulty["scale"]
    flexibility = difficulty["request_flexibility"]
    candidate = flexibility["primary_candidate_sections"]
    demand = difficulty["demand_capacity"]
    period = difficulty["period_candidate_structure"]
    return {
        "scenario_id": spec.scenario_id,
        "split": spec.split,
        "data_generation_seed": spec.data_generation_seed,
        "section_planning_seed": spec.section_planning_seed,
        "students": scale["students"],
        "logical_requests": scale["logical_requests"],
        "logical_primaries": scale["logical_primaries"],
        "alternates": scale["alternates"],
        "logical_sections": scale["logical_sections"],
        "section_rows": scale["section_rows"],
        "candidate_edges": scale["candidate_edges"],
        "canonical_input_hash": scale["canonical_input_hash"],
        "protected_student_count": difficulty["student_load"]["protected_student_count"],
        "primaries_with_zero_candidates": flexibility["primaries_with_zero_candidates"],
        "candidates_min": candidate["min"],
        "candidates_p10": candidate["p10"],
        "candidates_median": candidate["median"],
        "candidates_p90": candidate["p90"],
        "candidates_max": candidate["max"],
        "courses_over_capacity": demand["courses_with_primary_demand_over_capacity"],
        "capacity_only_primary_shortfall": demand["total_capacity_only_primary_shortfall"],
        "max_demand_capacity_ratio": demand["maximum_primary_demand_capacity_ratio"],
        "p90_demand_capacity_ratio": demand["p90_primary_demand_capacity_ratio"],
        "requests_with_one_period": period["requests_with_candidates_in_only_one_period"],
        "requests_with_multiple_periods": period["requests_with_candidates_across_multiple_periods"],
        "average_distinct_candidate_periods": period["average_distinct_candidate_periods"],
    }


def _select_scenarios(
    suite: ScenarioSuite,
    split: str,
    scenario_id: str | None,
    max_scenarios: int | None,
    confirm_holdout: bool,
) -> tuple[ScenarioSpec, ...]:
    if split not in {"development", "holdout"}:
        raise ScenarioManifestError("split must be development or holdout.")
    if split == "holdout" and not confirm_holdout:
        raise RobustnessRunnerError("Holdout evaluation requires confirm_holdout_evaluation=True.")
    selected = [item for item in suite.scenarios if item.enabled and item.split == split]
    if scenario_id is not None:
        selected = [item for item in selected if item.scenario_id == scenario_id]
        if not selected:
            raise RobustnessRunnerError(f"Enabled scenario '{scenario_id}' is not in split '{split}'.")
    if max_scenarios is not None:
        if max_scenarios <= 0:
            raise RobustnessRunnerError("max_scenarios must be positive.")
        selected = selected[:max_scenarios]
    if not selected:
        raise RobustnessRunnerError("No enabled scenarios selected.")
    return tuple(selected)


def _normalize_greedy_algorithms(algorithms: Iterable[str]) -> tuple[str, ...]:
    normalized = tuple(item.strip().lower() for item in algorithms if item.strip())
    invalid = tuple(item for item in normalized if item not in GREEDY_ALGORITHMS)
    if invalid:
        raise RobustnessRunnerError(
            "Robustness suite accepts Greedy algorithms only; unsupported: " + ", ".join(invalid)
        )
    if not normalized:
        raise RobustnessRunnerError("At least one Greedy algorithm is required.")
    return tuple(dict.fromkeys(normalized))


def _prepare_output_root(root: Path, *, resume: bool) -> None:
    if root.exists() and not root.is_dir():
        raise RobustnessRunnerError(f"Output path is not a directory: {root}")
    if root.exists() and any(root.iterdir()) and not resume:
        raise RobustnessRunnerError(f"Output directory is non-empty; use resume=True to reuse it: {root}")
    root.mkdir(parents=True, exist_ok=True)


def _load_cached_scenario(
    scenario_dir: Path,
    spec: ScenarioSpec,
    *,
    suite_hash: str,
    config_fingerprint: str,
    resume: bool,
) -> dict[str, Any] | None:
    if not scenario_dir.exists():
        return None
    result_path = scenario_dir / SCENARIO_RESULT_FILE
    if not resume:
        if any(scenario_dir.iterdir()):
            raise RobustnessRunnerError(f"Scenario output exists; use resume=True: {scenario_dir}")
        return None
    if not result_path.is_file():
        raise RobustnessRunnerError(f"Cannot resume scenario without {result_path}.")
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RobustnessRunnerError(f"Invalid cached scenario result: {result_path}: {exc}") from exc
    expected = {
        "status": "completed",
        "suite_hash": suite_hash,
        "scenario_spec_hash": _sha256_json(spec.to_dict()),
        "config_templates_fingerprint": config_fingerprint,
    }
    mismatches = [key for key, value in expected.items() if payload.get(key) != value]
    if mismatches:
        raise RobustnessRunnerError(f"Cached scenario provenance mismatch for {spec.scenario_id}: {', '.join(mismatches)}")
    if not isinstance(payload.get("scenario_rows"), list) or not isinstance(payload.get("input_difficulty_row"), dict):
        raise RobustnessRunnerError(f"Cached scenario is missing exported rows: {result_path}")
    return payload


def _write_generation_result(result: Any, output_dir: Path, spec: ScenarioSpec) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result.students.to_csv(output_dir / "students.csv", index=False)
    result.requests.to_csv(output_dir / "requests.csv", index=False)
    result.summary.to_csv(output_dir / "generation_summary.csv", index=False)
    metadata = {
        **dict(result.metadata),
        "robustness_scenario_id": spec.scenario_id,
        "data_generation_seed": spec.data_generation_seed,
        "generation_scenario_id": spec.generation_scenario_id,
        "output_file_hashes": {
            "students.csv": _sha256_file(output_dir / "students.csv"),
            "requests.csv": _sha256_file(output_dir / "requests.csv"),
        },
    }
    _write_json(output_dir / "generation_metadata.json", metadata)


def _check_reference(actual: CanonicalInputFingerprint, expected: dict[str, Any], scenario_id: str) -> None:
    expected_fingerprint = _fingerprint_from_dict(expected)
    if actual != expected_fingerprint:
        raise RobustnessRunnerError(
            f"Frozen reference fingerprint mismatch for {scenario_id}: "
            f"expected {asdict(expected_fingerprint)}, actual {asdict(actual)}"
        )


def _fingerprint_from_dict(payload: dict[str, Any]) -> CanonicalInputFingerprint:
    fields = (
        "students",
        "logical_requests",
        "logical_primaries",
        "alternates",
        "logical_sections",
        "section_rows",
        "candidate_edges",
        "canonical_input_hash",
    )
    return CanonicalInputFingerprint(**{field: payload[field] for field in fields})


def _validate_reference_fingerprint(payload: Any, scenario_id: str) -> None:
    if not isinstance(payload, dict):
        raise ScenarioManifestError(f"{scenario_id}: expected_reference_fingerprint must be an object or null.")
    required = {
        "students",
        "logical_requests",
        "logical_primaries",
        "alternates",
        "logical_sections",
        "section_rows",
        "candidate_edges",
        "canonical_input_hash",
    }
    if set(payload) != required:
        raise ScenarioManifestError(f"{scenario_id}: reference fingerprint fields must be exactly {sorted(required)}.")
    for field in required - {"canonical_input_hash"}:
        _nonnegative_int(payload[field], f"{scenario_id}.expected_reference_fingerprint.{field}")
    if not isinstance(payload["canonical_input_hash"], str) or len(payload["canonical_input_hash"]) != 64:
        raise ScenarioManifestError(f"{scenario_id}: canonical_input_hash must be a SHA-256 hex string.")


def _required_text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ScenarioManifestError(f"{field} must be a non-empty string.")
    return value.strip()


def _nonnegative_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ScenarioManifestError(f"{field} must be a non-negative integer.")
    return value


def _hash_input_files(*directories: Path) -> str:
    digest = hashlib.sha256()
    for directory in directories:
        if not directory.is_dir():
            raise RobustnessRunnerError(f"Input directory does not exist: {directory}")
        for path in sorted(directory.glob("*.csv")):
            digest.update(str(path.relative_to(directory)).encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _git_status() -> str:
    try:
        return subprocess.check_output(["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        return ""


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _algorithm_alias(algorithm_name: str) -> str:
    for alias, label in ALGORITHM_LABELS.items():
        if label == algorithm_name:
            return alias
    return algorithm_name


def _algorithm_label(algorithm_name: str) -> str:
    return ALGORITHM_LABELS.get(_algorithm_alias(algorithm_name), algorithm_name)


def _sum_int(frame: pd.DataFrame, column: str, predicate: Any = None) -> int:
    if frame.empty or column not in frame:
        return 0
    values = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    if predicate is not None:
        return int(sum(predicate(value) for value in values))
    return int(values.sum())


def _count_values(frame: pd.DataFrame, column: str, predicate: Any) -> int:
    if frame.empty or column not in frame:
        return 0
    values = pd.to_numeric(frame[column], errors="coerce").fillna(0)
    return int(sum(predicate(value) for value in values))


def _stats(values: pd.Series) -> dict[str, float | int]:
    return {
        "count": int(len(values)),
        "mean": round(float(values.mean()), 6),
        "median": round(float(values.median()), 6),
        "min": round(float(values.min()), 6),
        "max": round(float(values.max()), 6),
        "p10": round(float(values.quantile(0.1)), 6),
        "p90": round(float(values.quantile(0.9)), 6),
        "std": round(float(values.std(ddof=0)), 6),
    }


def _empty_stats() -> dict[str, float | int]:
    return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "p10": 0.0, "p90": 0.0, "std": 0.0}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _write_rows_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = tuple(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fieldnames = tuple(dict.fromkeys(key for row in rows for key in row))
    frame = pd.DataFrame(rows, columns=fieldnames)
    frame.to_csv(path, index=False)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the frozen Greedy normal-year robustness suite.")
    parser.add_argument("--suite", default=str(DEFAULT_SUITE_PATH))
    parser.add_argument("--split", choices=("development", "holdout"), default="development")
    parser.add_argument("--scenario-id")
    parser.add_argument("--algorithms", default=",".join(GREEDY_ALGORITHMS))
    parser.add_argument("--output-dir")
    parser.add_argument("--max-scenarios", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--confirm-holdout-evaluation", action="store_true")
    parser.add_argument("--config-dir", default="data/config")
    parser.add_argument("--templates-dir", default="data/templates")
    args = parser.parse_args(tuple(argv) if argv is not None else None)
    try:
        result = run_robustness_suite(
            suite_path=args.suite,
            split=args.split,
            scenario_id=args.scenario_id,
            algorithms=tuple(args.algorithms.split(",")),
            output_dir=args.output_dir,
            max_scenarios=args.max_scenarios,
            resume=args.resume,
            dry_run=args.dry_run,
            confirm_holdout_evaluation=args.confirm_holdout_evaluation,
            config_dir=args.config_dir,
            templates_dir=args.templates_dir,
        )
    except (ScenarioManifestError, RobustnessRunnerError, ValueError) as exc:
        print(f"Robustness runner FAIL: {exc}")
        return 1
    print(
        "Robustness runner PASS: "
        f"{len(result.scenario_ids)} {result.split} scenario(s), "
        f"algorithms={','.join(result.algorithms)}, dry_run={result.dry_run}"
    )
    if result.output_dir:
        print(f"Artifacts: {result.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
