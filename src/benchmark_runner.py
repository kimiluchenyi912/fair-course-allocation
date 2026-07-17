from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.allocation import (
    AlternateRequestStatus,
    AssignmentRejectionReason,
    BaselineResult,
    CanonicalAllocationInput,
    CpSatAllocationResult,
    MandatoryFallbackStatus,
    PrimaryRequestStatus,
    RequestOutcome,
    evaluate_math_policy,
    load_math_fallback_rules,
    math_course_ids_from_catalog,
    run_constrained_first_baseline,
    run_fair_cp_sat_solver,
    run_fcfs_baseline,
    run_grade_priority_baseline,
    run_seeded_random_baseline,
)
from src.allocation.math_policy_models import MathFallbackRule
from src.experiment_manifest import (
    CanonicalInputFingerprint,
    ExperimentManifest,
    ExperimentManifestError,
    ExperimentSeeds,
    build_experiment_manifest,
    verify_experiment_manifest,
)
from src.final_schedule_policy import (
    COURSE_COUNT_SEMANTICS,
    MAXIMUM_ORDINARY_PRIMARY_UNMET_COUNT,
    MAXIMUM_PROTECTED_PRIMARY_UNMET_COUNT,
    MAXIMUM_SCHEDULE_GAP_COUNT,
    MINIMUM_ASSIGNED_LOGICAL_COURSE_COUNT,
    REASON_CODES as FINAL_POLICY_REASON_CODES,
    SCHEMA_VERSION as FINAL_POLICY_SCHEMA_VERSION,
    SUMMARY_FIELDNAMES as FINAL_POLICY_SUMMARY_FIELDNAMES,
    VIOLATION_FIELDNAMES as FINAL_POLICY_VIOLATION_FIELDNAMES,
    evaluate_final_schedule_policy,
    summary_row as final_policy_summary_row,
    violation_row as final_policy_violation_row,
)


DEFAULT_ALGORITHMS = ("random", "constrained")
RUNNER_NAME = "benchmark_runner_v1"

DEFAULT_ARTIFACT_FILES = (
    "algorithm_summary.csv",
    "course_unmet_summary.csv",
    "section_utilization.csv",
    "assignment_failure_summary.csv",
    "student_schedule_gaps.csv",
    "final_schedule_policy_summary.csv",
    "final_schedule_policy_violations.csv",
    "benchmark_manifest.json",
)
LARGE_TABLE_ARTIFACT_FILES = (
    "student_outcomes.csv",
    "request_outcomes.csv",
)

COURSE_UNMET_SUMMARY_FIELDNAMES = (
    "algorithm_name",
    "candidate_key",
    "primary_demand",
    "primary_assigned",
    "primary_unmet",
    "primary_unmet_rate",
)
SECTION_UTILIZATION_FIELDNAMES = (
    "algorithm_name",
    "linked_section_group_id",
    "capacity",
    "assigned_count",
    "remaining_capacity",
    "utilization_rate",
)
ASSIGNMENT_FAILURE_SUMMARY_FIELDNAMES = (
    "algorithm_name",
    "request_kind",
    "terminal_unmet_reason",
    "unmet_request_count",
    "affected_student_count",
    "total_candidate_attempts",
    "total_candidate_rejections",
)
STUDENT_SCHEDULE_GAPS_FIELDNAMES = (
    "algorithm_name",
    "student_id",
    "grade",
    "target_course_count",
    "assigned_course_count",
    "schedule_gap_count",
    "primary_unmet_count",
    "alternates_assigned_count",
    "unmet_primary_request_ids",
    "unmet_alternate_request_ids",
    "terminal_unmet_reasons",
)
STUDENT_OUTCOMES_FIELDNAMES = (
    "algorithm_name",
    "student_id",
    "grade",
    "target_period_units",
    "assigned_period_units",
    "remaining_period_units",
    "assignment_keys",
    "primary_request_count",
    "primary_assigned_count",
    "primary_unmet_count",
    "primary_unmet_request_keys",
    "primary_unmet_period_units",
    "alternate_request_count",
    "alternate_assigned_count",
    "alternate_assigned_period_units",
    "mandatory_fallback_assigned_count",
    "mandatory_fallback_assigned_period_units",
    "mandatory_fallback_assignment_keys",
    "fully_scheduled",
    "priority_protected",
    "ordinary_fairness_violation",
    "protected_fairness_violation",
    "high_demand_guarantee_violation_count",
    "high_demand_violating_request_keys",
    "target_course_count",
    "assigned_course_count",
    "schedule_gap_count",
    "assigned_alternate_count",
    "target_logical_course_count",
    "assigned_logical_course_count",
    "logical_schedule_gap_count",
    "logical_fully_scheduled",
)
REQUEST_OUTCOMES_FIELDNAMES = (
    "algorithm_name",
    "request_key",
    "student_id",
    "request_type",
    "alternate_rank",
    "candidate_key",
    "period_units",
    "status",
    "assignment_key",
    "assigned_linked_section_group_id",
    "candidate_attempts_count",
    "remaining_units_before",
    "remaining_units_after",
    "candidate_rejections_count",
    "rejected_section_at_capacity_count",
    "rejected_period_conflict_count",
    "rejected_duplicate_logical_course_count",
    "rejected_student_load_limit_count",
    "rejected_other_count",
    "terminal_unmet_reason",
)

DIAGNOSTICS_SCHEMA_VERSION = "assignment_failure_diagnostics_v1"
DIAGNOSTIC_ARTIFACT_FILES = (
    "assignment_failure_summary.csv",
    "student_schedule_gaps.csv",
)
REJECTION_REASON_CODES = (
    "section_at_capacity",
    "period_conflict",
    "duplicate_logical_course",
    "student_load_limit",
    "other_rejection",
)
TERMINAL_UNMET_REASON_ORDER = (
    "no_candidate_sections",
    "all_section_at_capacity",
    "all_period_conflict",
    "all_duplicate_logical_course",
    "all_student_load_limit",
    "mixed_rejections",
    "other_rejection",
    "not_attempted",
)
REQUEST_KIND_ORDER = ("primary", "alternate", "mandatory_fallback")


class BenchmarkRunnerError(ValueError):
    pass


@dataclass(frozen=True)
class CpSatBenchmarkOptions:
    max_time_seconds_per_stage: float = 30.0
    max_total_time_seconds: float | None = None
    bootstrap_time_seconds: float | None = None
    use_feasibility_bootstrap: bool = True
    use_constrained_first_hint: bool = True
    num_search_workers: int = 1
    logical_schedule_completion_enabled: bool = True
    initial_solution_artifact_dir: str | Path | None = None


@dataclass(frozen=True)
class BenchmarkAlgorithmResult:
    algorithm_name: str
    status: str
    runtime_seconds: float
    primary_assigned: int
    primary_unmet: int
    primary_satisfaction_rate: float
    primary_unmet_period_units: int
    fallback_required: int
    fallback_assigned: int
    fallback_failed: int
    alternate_rank1_assigned: int
    alternate_rank2_assigned: int
    alternate_rank3_assigned: int
    total_alternates_assigned: int
    fully_scheduled_students: int
    students_with_remaining_units: int
    ordinary_violations: int
    protected_violations: int
    high_demand_violations: int
    math_coverage_violations: int
    section_over_capacity_count: int
    consistency_issue_count: int
    solve_status: str | None = None
    lexicographic_optimality_proven: bool | None = None
    highest_globally_proven_stage: str | None = None
    conditional_optimization_performed: bool | None = None
    skipped_stage_count: int | None = None
    bootstrap_status: str | None = None
    stage_diagnostics_summary: tuple[dict[str, Any], ...] = ()
    final_schedule_policy_pass: bool | None = None
    final_schedule_policy_violation_students: int | None = None
    students_below_minimum_course_count: int | None = None
    students_with_schedule_gap_over_limit: int | None = None
    logical_fully_scheduled_students: int | None = None
    students_with_logical_schedule_gap: int | None = None
    total_logical_schedule_gap: int | None = None
    logical_schedule_completion_objective_enabled: bool | None = None
    logical_schedule_completion_stage_status: str | None = None
    logical_schedule_completion_objective_value: int | None = None
    logical_schedule_completion_best_bound: int | None = None
    logical_schedule_completion_conditionally_optimized: bool | None = None
    logical_schedule_completion_fixed_value: int | None = None
    hint_source: str | None = None
    hint_total_model_variables: int | None = None
    hint_variables_supplied: int | None = None
    hint_coverage_rate: float | None = None
    hint_selected_variables: int | None = None
    hint_zero_variables: int | None = None
    hint_unknown_or_unmapped_assignments: int | None = None
    hint_duplicate_keys: int | None = None
    hint_replay_policy_pass: bool | None = None
    full_model_seed_strategy: str | None = None
    full_model_seed_policy_pass: bool | None = None
    full_model_seed_violation_students: int | None = None
    full_model_seed_repaired_by_solver: bool | None = None
    initial_solution_seed_enabled: bool = False
    initial_solution_seed_role: str = ""
    initial_solution_seed_source_commit: str = ""
    initial_solution_seed_source_algorithm: str = ""
    initial_solution_seed_source_status: str = ""
    initial_solution_seed_source_policy_pass: bool | None = None
    initial_solution_seed_manifest_sha256: str = ""
    initial_solution_seed_request_outcomes_sha256: str = ""
    initial_solution_seed_provenance_sha256: str = ""
    initial_solution_seed_hint_coverage: float | None = None
    initial_solution_seed_unknown_keys: int = 0
    initial_solution_seed_duplicate_keys: int = 0
    initial_solution_seed_selected_by_stage: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkSuiteResult:
    manifest: ExperimentManifest
    expected_fingerprint: CanonicalInputFingerprint | None
    fingerprint_verified: bool
    algorithms_run: tuple[str, ...]
    results: tuple[BenchmarkAlgorithmResult, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "created_by": RUNNER_NAME,
            "runner_name": RUNNER_NAME,
            "data_generation_seed": self.manifest.seeds.data_generation_seed,
            "section_planning_seed": self.manifest.seeds.section_planning_seed,
            "solver_seed": self.manifest.seeds.solver_seed,
            "generated_input_dir": self.manifest.generated_input_path,
            "sections_input_dir": self.manifest.section_input_path,
            "canonical_fingerprint": asdict(self.manifest.fingerprint),
            "expected_fingerprint": asdict(self.expected_fingerprint) if self.expected_fingerprint is not None else None,
            "fingerprint_verified": self.fingerprint_verified,
            "algorithms_run": list(self.algorithms_run),
            "results": [result.to_dict() for result in self.results],
        }

    def to_csv_rows(self) -> tuple[dict[str, Any], ...]:
        base = {
            "runner_name": RUNNER_NAME,
            "data_generation_seed": self.manifest.seeds.data_generation_seed,
            "section_planning_seed": self.manifest.seeds.section_planning_seed,
            "solver_seed": self.manifest.seeds.solver_seed,
            "students": self.manifest.fingerprint.students,
            "logical_requests": self.manifest.fingerprint.logical_requests,
            "logical_primaries": self.manifest.fingerprint.logical_primaries,
            "alternates": self.manifest.fingerprint.alternates,
            "logical_sections": self.manifest.fingerprint.logical_sections,
            "section_rows": self.manifest.fingerprint.section_rows,
            "candidate_edges": self.manifest.fingerprint.candidate_edges,
            "canonical_input_hash": self.manifest.fingerprint.canonical_input_hash,
            "fingerprint_verified": self.fingerprint_verified,
        }
        rows = []
        for result in self.results:
            row = {**base, **result.to_dict()}
            row.pop("stage_diagnostics_summary", None)
            rows.append(row)
        return tuple(rows)


def run_benchmark_suite(
    *,
    generated_input_dir: str | Path,
    sections_input_dir: str | Path,
    config_dir: str | Path,
    seeds: ExperimentSeeds,
    scenario_id: str = "stable_year",
    course_catalog_path: str | Path | None = None,
    expected_fingerprint: CanonicalInputFingerprint | None = None,
    algorithms: tuple[str, ...] = DEFAULT_ALGORITHMS,
    output_json_path: str | Path | None = None,
    output_csv_path: str | Path | None = None,
    output_artifact_dir: str | Path | None = None,
    include_large_tables: bool = False,
    cp_sat_options: CpSatBenchmarkOptions | None = None,
) -> BenchmarkSuiteResult:
    algorithms = _normalize_algorithms(algorithms)
    manifest = build_experiment_manifest(
        generated_input_dir,
        sections_input_dir,
        config_dir,
        scenario_id=scenario_id,
        seeds=seeds,
    )
    if expected_fingerprint is not None:
        _check_expected_fingerprint(manifest.fingerprint, expected_fingerprint)
    allocation_input = verify_experiment_manifest(manifest, config_dir=config_dir)
    catalog_path = Path(course_catalog_path) if course_catalog_path is not None else Path(config_dir) / "course_catalog.csv"
    catalog = pd.read_csv(catalog_path, keep_default_na=False)
    math_course_ids = math_course_ids_from_catalog(catalog) if "department" in catalog.columns else ()
    math_fallback_rules = _load_math_fallback_rules(Path(config_dir), catalog)

    algorithm_runs = tuple(
        _run_algorithm(name, allocation_input, seeds.solver_seed, math_course_ids, math_fallback_rules, cp_sat_options)
        for name in algorithms
    )
    results = tuple(summary for summary, _raw_result in algorithm_runs)
    suite = BenchmarkSuiteResult(
        manifest=manifest,
        expected_fingerprint=expected_fingerprint,
        fingerprint_verified=True,
        algorithms_run=algorithms,
        results=results,
    )
    if output_json_path is not None:
        _write_json(suite, Path(output_json_path))
    if output_csv_path is not None:
        _write_csv(suite, Path(output_csv_path))
    if output_artifact_dir is not None:
        raw_results = tuple(raw_result for _summary, raw_result in algorithm_runs)
        _export_artifacts(suite, raw_results, Path(output_artifact_dir), include_large_tables=include_large_tables)
    return suite


def _normalize_algorithms(algorithms: tuple[str, ...]) -> tuple[str, ...]:
    valid = {"random", "constrained", "fcfs", "grade_priority", "cp_sat"}
    normalized = tuple(item.strip().lower() for item in algorithms if item.strip())
    invalid = tuple(item for item in normalized if item not in valid)
    if invalid:
        raise BenchmarkRunnerError(f"Unsupported benchmark algorithm(s): {', '.join(invalid)}")
    if not normalized:
        raise BenchmarkRunnerError("At least one benchmark algorithm is required.")
    return normalized


def _check_expected_fingerprint(actual: CanonicalInputFingerprint, expected: CanonicalInputFingerprint) -> None:
    actual_data = asdict(actual)
    expected_data = asdict(expected)
    mismatches = tuple(
        f"{field}: expected {expected_data[field]!r}, actual {actual_data[field]!r}"
        for field in sorted(expected_data)
        if actual_data[field] != expected_data[field]
    )
    if mismatches:
        details = "\n".join(f"- {item}" for item in mismatches)
        raise BenchmarkRunnerError(
            "Canonical fingerprint mismatch; no algorithms were run.\n"
            f"{details}\n"
            "Check data_generation_seed, section_planning_seed, solver_seed, generated_input_dir, and sections_input_dir."
        )


def _load_math_fallback_rules(config_dir: Path, catalog: pd.DataFrame) -> tuple[MathFallbackRule, ...]:
    if not (config_dir / "math_fallbacks.csv").is_file():
        return ()
    return load_math_fallback_rules(config_dir, catalog)


def _run_algorithm(
    name: str,
    allocation_input: CanonicalAllocationInput,
    seed: int,
    math_course_ids: tuple[str, ...],
    math_fallback_rules: tuple[MathFallbackRule, ...],
    cp_sat_options: CpSatBenchmarkOptions | None,
) -> tuple[BenchmarkAlgorithmResult, BaselineResult | CpSatAllocationResult]:
    started = time.perf_counter()
    if name == "random":
        result = run_seeded_random_baseline(
            allocation_input,
            seed=seed,
            math_course_ids=math_course_ids,
            math_fallback_rules=math_fallback_rules,
        )
    elif name == "constrained":
        result = run_constrained_first_baseline(
            allocation_input,
            seed=seed,
            math_course_ids=math_course_ids,
            math_fallback_rules=math_fallback_rules,
        )
    elif name == "fcfs":
        result = run_fcfs_baseline(
            allocation_input,
            seed=seed,
            math_course_ids=math_course_ids,
            math_fallback_rules=math_fallback_rules,
        )
    elif name == "grade_priority":
        result = run_grade_priority_baseline(
            allocation_input,
            seed=seed,
            math_course_ids=math_course_ids,
            math_fallback_rules=math_fallback_rules,
        )
    elif name == "cp_sat":
        options = cp_sat_options or CpSatBenchmarkOptions()
        result = run_fair_cp_sat_solver(
            allocation_input,
            seed=seed,
            math_course_ids=math_course_ids,
            math_fallback_rules=math_fallback_rules,
            max_time_seconds_per_stage=options.max_time_seconds_per_stage,
            max_total_time_seconds=options.max_total_time_seconds,
            bootstrap_time_seconds=options.bootstrap_time_seconds,
            use_feasibility_bootstrap=options.use_feasibility_bootstrap,
            use_constrained_first_hint=options.use_constrained_first_hint,
            num_search_workers=options.num_search_workers,
            logical_schedule_completion_enabled=options.logical_schedule_completion_enabled,
            initial_solution_artifact_dir=options.initial_solution_artifact_dir,
        )
    else:  # pragma: no cover - _normalize_algorithms keeps this unreachable.
        raise BenchmarkRunnerError(f"Unsupported benchmark algorithm: {name}")
    summary = _summarize_result(allocation_input, result, round(time.perf_counter() - started, 6), math_course_ids, math_fallback_rules)
    return summary, result


def _summarize_result(
    allocation_input: CanonicalAllocationInput,
    result: BaselineResult | CpSatAllocationResult,
    runtime_seconds: float,
    math_course_ids: tuple[str, ...],
    math_fallback_rules: tuple[MathFallbackRule, ...],
) -> BenchmarkAlgorithmResult:
    primary_outcomes = tuple(outcome for outcome in result.request_outcomes if outcome.request_type == "primary")
    primary_assigned = sum(outcome.status == PrimaryRequestStatus.ASSIGNED for outcome in primary_outcomes)
    primary_unmet = len(primary_outcomes) - primary_assigned
    alternates = tuple(outcome for outcome in result.request_outcomes if outcome.request_type == "alternate")
    alternate_assigned_by_rank = {
        rank: sum(
            outcome.alternate_rank == rank and outcome.status == AlternateRequestStatus.ASSIGNED
            for outcome in alternates
        )
        for rank in (1, 2, 3)
    }
    fallback_failed_statuses = {
        MandatoryFallbackStatus.UNASSIGNED_NO_CANDIDATES,
        MandatoryFallbackStatus.UNASSIGNED_ALL_CANDIDATES_REJECTED,
    }
    fallback_required_statuses = fallback_failed_statuses | {MandatoryFallbackStatus.ASSIGNED}
    math_report = getattr(result, "math_policy_report", None)
    if math_report is None and result.request_outcomes:
        math_report = evaluate_math_policy(
            allocation_input,
            result,  # type: ignore[arg-type]
            math_course_ids,
            math_fallback_rules,
        )
    final_policy_report = (
        evaluate_final_schedule_policy(result.algorithm_name, result.student_outcomes)
        if result.student_outcomes
        else None
    )
    base = BenchmarkAlgorithmResult(
        algorithm_name=result.algorithm_name,
        status=_result_status(result),
        runtime_seconds=runtime_seconds,
        primary_assigned=primary_assigned,
        primary_unmet=primary_unmet,
        primary_satisfaction_rate=round(primary_assigned / len(primary_outcomes), 6) if primary_outcomes else 0.0,
        primary_unmet_period_units=sum(
            outcome.period_units for outcome in primary_outcomes if outcome.status != PrimaryRequestStatus.ASSIGNED
        ),
        fallback_required=sum(outcome.status in fallback_required_statuses for outcome in result.mandatory_fallback_outcomes),
        fallback_assigned=sum(outcome.status == MandatoryFallbackStatus.ASSIGNED for outcome in result.mandatory_fallback_outcomes),
        fallback_failed=sum(outcome.status in fallback_failed_statuses for outcome in result.mandatory_fallback_outcomes),
        alternate_rank1_assigned=alternate_assigned_by_rank[1],
        alternate_rank2_assigned=alternate_assigned_by_rank[2],
        alternate_rank3_assigned=alternate_assigned_by_rank[3],
        total_alternates_assigned=sum(alternate_assigned_by_rank.values()),
        fully_scheduled_students=sum(outcome.fully_scheduled for outcome in result.student_outcomes),
        students_with_remaining_units=sum(outcome.remaining_period_units > 0 for outcome in result.student_outcomes),
        ordinary_violations=len(result.policy_report.ordinary_violation_student_ids) if result.policy_report else 0,
        protected_violations=len(result.policy_report.protected_violation_student_ids) if result.policy_report else 0,
        high_demand_violations=result.policy_report.high_demand_violation_count if result.policy_report else 0,
        math_coverage_violations=(
            len(math_report.current_math_coverage_violation_student_ids)
            if math_report is not None
            else 0
        ),
        section_over_capacity_count=sum(row.assigned_count > row.capacity for row in result.section_roster_summary),
        consistency_issue_count=len(result.consistency_issues),
        final_schedule_policy_pass=(
            final_policy_report.summary.final_schedule_policy_pass if final_policy_report is not None else None
        ),
        final_schedule_policy_violation_students=(
            final_policy_report.summary.violating_student_count if final_policy_report is not None else None
        ),
        students_below_minimum_course_count=(
            final_policy_report.summary.below_minimum_course_count if final_policy_report is not None else None
        ),
        students_with_schedule_gap_over_limit=(
            final_policy_report.summary.schedule_gap_over_limit_count if final_policy_report is not None else None
        ),
        logical_fully_scheduled_students=(
            final_policy_report.summary.logical_fully_scheduled_student_count
            if final_policy_report is not None
            else None
        ),
        students_with_logical_schedule_gap=(
            final_policy_report.summary.students_with_logical_schedule_gap
            if final_policy_report is not None
            else None
        ),
        total_logical_schedule_gap=(
            final_policy_report.summary.total_logical_schedule_gap
            if final_policy_report is not None
            else None
        ),
    )
    if not isinstance(result, CpSatAllocationResult):
        return base
    stats = result.model_stats
    data = asdict(base)
    data.update(
        {
            "solve_status": result.solve_status.value,
            "lexicographic_optimality_proven": result.lexicographic_optimality_proven,
            "highest_globally_proven_stage": (
                stats.highest_globally_proven_stage.value if stats.highest_globally_proven_stage is not None else None
            ),
            "conditional_optimization_performed": stats.conditional_optimization_performed,
            "skipped_stage_count": stats.skipped_stage_count,
            "bootstrap_status": stats.bootstrap_status.value,
            "logical_schedule_completion_objective_enabled": stats.logical_schedule_completion_objective_enabled,
            "logical_schedule_completion_stage_status": (
                stats.logical_schedule_completion_stage_status.value
                if stats.logical_schedule_completion_stage_status is not None
                else None
            ),
            "logical_schedule_completion_objective_value": stats.logical_schedule_completion_objective_value,
            "logical_schedule_completion_best_bound": stats.logical_schedule_completion_best_bound,
            "logical_schedule_completion_conditionally_optimized": (
                stats.logical_schedule_completion_conditionally_optimized
            ),
            "logical_schedule_completion_fixed_value": stats.logical_schedule_completion_fixed_value,
            "hint_source": stats.hint_source,
            "hint_total_model_variables": stats.hint_total_model_variables,
            "hint_variables_supplied": stats.hint_variables_supplied,
            "hint_coverage_rate": stats.hint_coverage_rate,
            "hint_selected_variables": stats.hint_selected_variables,
            "hint_zero_variables": stats.hint_zero_variables,
            "hint_unknown_or_unmapped_assignments": stats.hint_unknown_or_unmapped_assignments,
            "hint_duplicate_keys": stats.hint_duplicate_keys,
            "hint_replay_policy_pass": stats.hint_replay_policy_pass,
            "full_model_seed_strategy": stats.full_model_seed_strategy,
            "full_model_seed_policy_pass": stats.full_model_seed_policy_pass,
            "full_model_seed_violation_students": stats.full_model_seed_violation_students,
            "full_model_seed_repaired_by_solver": stats.full_model_seed_repaired_by_solver,
            "initial_solution_seed_enabled": stats.initial_solution_seed_enabled,
            "initial_solution_seed_role": stats.initial_solution_seed_role,
            "initial_solution_seed_source_commit": stats.initial_solution_seed_source_commit,
            "initial_solution_seed_source_algorithm": stats.initial_solution_seed_source_algorithm,
            "initial_solution_seed_source_status": stats.initial_solution_seed_source_status,
            "initial_solution_seed_source_policy_pass": stats.initial_solution_seed_source_policy_pass,
            "initial_solution_seed_manifest_sha256": stats.initial_solution_seed_manifest_sha256,
            "initial_solution_seed_request_outcomes_sha256": stats.initial_solution_seed_request_outcomes_sha256,
            "initial_solution_seed_provenance_sha256": stats.initial_solution_seed_provenance_sha256,
            "initial_solution_seed_hint_coverage": stats.initial_solution_seed_hint_coverage,
            "initial_solution_seed_unknown_keys": stats.initial_solution_seed_unknown_keys,
            "initial_solution_seed_duplicate_keys": stats.initial_solution_seed_duplicate_keys,
            "initial_solution_seed_selected_by_stage": stats.initial_solution_seed_selected_by_stage,
            "stage_diagnostics_summary": tuple(
            {
                "stage_name": diagnostic.stage_name.value,
                "model_scope": diagnostic.model_scope.value,
                "status": diagnostic.status.value,
                "objective_value": diagnostic.objective_value,
                "best_objective_bound": diagnostic.best_objective_bound,
                "wall_time_seconds": diagnostic.wall_time_seconds,
                "optimum_proven": diagnostic.optimum_proven,
                "conditional_on_unproven_incumbent": diagnostic.conditional_on_unproven_incumbent,
                "fixed_higher_priority_values": tuple(
                    (stage_name.value, value)
                    for stage_name, value in diagnostic.fixed_higher_priority_values
                ),
                "effective_time_limit_seconds": diagnostic.effective_time_limit_seconds,
                "response_proto_hash": diagnostic.response_proto_hash,
                "objective_descriptor_hash": diagnostic.objective_descriptor_hash,
                "skipped": diagnostic.skipped,
                "skip_reason": diagnostic.skip_reason,
            }
            for diagnostic in result.stage_diagnostics
            ),
        }
    )
    return BenchmarkAlgorithmResult(**data)


def _result_status(result: BaselineResult | CpSatAllocationResult) -> str:
    if isinstance(result, CpSatAllocationResult):
        return result.solve_status.value
    return "completed"


def _write_json(result: BenchmarkSuiteResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(result: BenchmarkSuiteResult, path: Path) -> None:
    rows = result.to_csv_rows()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = tuple(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _export_artifacts(
    suite: BenchmarkSuiteResult,
    raw_results: tuple[BaselineResult | CpSatAllocationResult, ...],
    output_artifact_dir: Path,
    *,
    include_large_tables: bool,
) -> tuple[str, ...]:
    """Write manifest-bound benchmark artifacts to ``output_artifact_dir``.

    Callers only reach this after fingerprint/manifest verification has
    already succeeded (``run_benchmark_suite`` raises before this point on
    mismatch), so no partial or misleading artifact set is ever written for
    a rejected input.
    """
    course_unmet_rows = _course_unmet_summary_rows(raw_results)
    section_utilization_rows = _section_utilization_rows(raw_results)
    failure_summary_rows = _assignment_failure_summary_rows(raw_results)
    schedule_gap_rows = _student_schedule_gap_rows(raw_results)
    final_policy_reports = tuple(
        evaluate_final_schedule_policy(result.algorithm_name, result.student_outcomes)
        for result in raw_results
        if result.student_outcomes
    )
    written_files = DEFAULT_ARTIFACT_FILES + (LARGE_TABLE_ARTIFACT_FILES if include_large_tables else ())
    manifest_payload = _benchmark_manifest_payload(suite, written_files)
    student_outcome_rows = _student_outcome_rows(raw_results) if include_large_tables else ()
    request_outcome_rows = _request_outcome_rows(raw_results) if include_large_tables else ()

    output_artifact_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(suite, output_artifact_dir / "algorithm_summary.csv")
    _write_rows_csv(output_artifact_dir / "course_unmet_summary.csv", COURSE_UNMET_SUMMARY_FIELDNAMES, course_unmet_rows)
    _write_rows_csv(output_artifact_dir / "section_utilization.csv", SECTION_UTILIZATION_FIELDNAMES, section_utilization_rows)
    _write_rows_csv(
        output_artifact_dir / "assignment_failure_summary.csv",
        ASSIGNMENT_FAILURE_SUMMARY_FIELDNAMES,
        failure_summary_rows,
    )
    _write_rows_csv(
        output_artifact_dir / "student_schedule_gaps.csv",
        STUDENT_SCHEDULE_GAPS_FIELDNAMES,
        schedule_gap_rows,
    )
    _write_rows_csv(
        output_artifact_dir / "final_schedule_policy_summary.csv",
        FINAL_POLICY_SUMMARY_FIELDNAMES,
        tuple(final_policy_summary_row(report) for report in final_policy_reports),
    )
    _write_rows_csv(
        output_artifact_dir / "final_schedule_policy_violations.csv",
        FINAL_POLICY_VIOLATION_FIELDNAMES,
        tuple(final_policy_violation_row(violation) for report in final_policy_reports for violation in report.violations),
    )
    (output_artifact_dir / "benchmark_manifest.json").write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if include_large_tables:
        _write_rows_csv(output_artifact_dir / "student_outcomes.csv", STUDENT_OUTCOMES_FIELDNAMES, student_outcome_rows)
        _write_rows_csv(output_artifact_dir / "request_outcomes.csv", REQUEST_OUTCOMES_FIELDNAMES, request_outcome_rows)
    return written_files


def _write_rows_csv(path: Path, fieldnames: tuple[str, ...], rows: tuple[dict[str, Any], ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _course_unmet_summary_rows(
    raw_results: tuple[BaselineResult | CpSatAllocationResult, ...],
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for result in raw_results:
        grouped: dict[str, list] = defaultdict(list)
        for outcome in result.request_outcomes:
            if outcome.request_type != "primary":
                continue
            grouped[outcome.candidate_key].append(outcome)
        for candidate_key, outcomes in grouped.items():
            demand = len(outcomes)
            assigned = sum(outcome.status == PrimaryRequestStatus.ASSIGNED for outcome in outcomes)
            unmet = demand - assigned
            rows.append(
                {
                    "algorithm_name": result.algorithm_name,
                    "candidate_key": candidate_key,
                    "primary_demand": demand,
                    "primary_assigned": assigned,
                    "primary_unmet": unmet,
                    "primary_unmet_rate": round(unmet / demand, 6),
                }
            )
    return tuple(sorted(rows, key=lambda row: (row["algorithm_name"], row["candidate_key"])))


def _section_utilization_rows(
    raw_results: tuple[BaselineResult | CpSatAllocationResult, ...],
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for result in raw_results:
        for section in result.section_roster_summary:
            rows.append(
                {
                    "algorithm_name": result.algorithm_name,
                    "linked_section_group_id": section.linked_section_group_id,
                    "capacity": section.capacity,
                    "assigned_count": section.assigned_count,
                    "remaining_capacity": section.remaining_capacity,
                    "utilization_rate": round(section.assigned_count / section.capacity, 6),
                }
            )
    return tuple(sorted(rows, key=lambda row: (row["algorithm_name"], row["linked_section_group_id"])))


def _assignment_failure_summary_rows(
    raw_results: tuple[BaselineResult | CpSatAllocationResult, ...],
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for algorithm_index, result in enumerate(raw_results):
        groups: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]] = defaultdict(list)
        for outcome in result.request_outcomes:
            diagnostics = _request_failure_diagnostics(outcome)
            terminal_reason = diagnostics["terminal_unmet_reason"]
            if not terminal_reason:
                continue
            groups[(outcome.request_type, terminal_reason)].append((outcome.student_id, diagnostics))
        for (request_kind, terminal_reason), items in groups.items():
            rows.append(
                {
                    "_algorithm_order": algorithm_index,
                    "algorithm_name": result.algorithm_name,
                    "request_kind": request_kind,
                    "terminal_unmet_reason": terminal_reason,
                    "unmet_request_count": len(items),
                    "affected_student_count": len({student_id for student_id, _diagnostics in items}),
                    "total_candidate_attempts": sum(
                        diagnostics["candidate_attempts_count"] for _student_id, diagnostics in items
                    ),
                    "total_candidate_rejections": sum(
                        diagnostics["candidate_rejections_count"] for _student_id, diagnostics in items
                    ),
                }
            )
    return tuple(_without_internal_sort_keys(row) for row in sorted(rows, key=_assignment_failure_summary_sort_key))


def _student_schedule_gap_rows(
    raw_results: tuple[BaselineResult | CpSatAllocationResult, ...],
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for algorithm_index, result in enumerate(raw_results):
        unmet_by_student: dict[str, list[RequestOutcome]] = defaultdict(list)
        for outcome in result.request_outcomes:
            if _request_failure_diagnostics(outcome)["terminal_unmet_reason"]:
                unmet_by_student[outcome.student_id].append(outcome)
        for outcome in result.student_outcomes:
            schedule_gap = max(outcome.target_period_units - outcome.assigned_period_units, 0)
            if schedule_gap <= 0:
                continue
            unmet_outcomes = unmet_by_student.get(outcome.student_id, ())
            terminal_reasons = sorted(
                {
                    diagnostics["terminal_unmet_reason"]
                    for item in unmet_outcomes
                    if (diagnostics := _request_failure_diagnostics(item))["terminal_unmet_reason"]
                }
            )
            rows.append(
                {
                    "_algorithm_order": algorithm_index,
                    "algorithm_name": result.algorithm_name,
                    "student_id": outcome.student_id,
                    "grade": outcome.grade,
                    "target_course_count": outcome.target_period_units,
                    "assigned_course_count": outcome.assigned_period_units,
                    "schedule_gap_count": schedule_gap,
                    "primary_unmet_count": outcome.primary_unmet_count,
                    "alternates_assigned_count": outcome.alternate_assigned_count,
                    "unmet_primary_request_ids": _json_array(
                        item.request_key for item in unmet_outcomes if item.request_type == "primary"
                    ),
                    "unmet_alternate_request_ids": _json_array(
                        item.request_key for item in unmet_outcomes if item.request_type == "alternate"
                    ),
                    "terminal_unmet_reasons": _json_array(terminal_reasons),
                }
            )
    return tuple(_without_internal_sort_keys(row) for row in sorted(rows, key=_student_schedule_gap_sort_key))


def _student_outcome_rows(
    raw_results: tuple[BaselineResult | CpSatAllocationResult, ...],
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for result in raw_results:
        for outcome in result.student_outcomes:
            rows.append(
                {
                    "algorithm_name": result.algorithm_name,
                    "student_id": outcome.student_id,
                    "grade": outcome.grade,
                    "target_period_units": outcome.target_period_units,
                    "assigned_period_units": outcome.assigned_period_units,
                    "remaining_period_units": outcome.remaining_period_units,
                    "assignment_keys": "|".join(outcome.assignment_keys),
                    "primary_request_count": outcome.primary_request_count,
                    "primary_assigned_count": outcome.primary_assigned_count,
                    "primary_unmet_count": outcome.primary_unmet_count,
                    "primary_unmet_request_keys": "|".join(outcome.primary_unmet_request_keys),
                    "primary_unmet_period_units": outcome.primary_unmet_period_units,
                    "alternate_request_count": outcome.alternate_request_count,
                    "alternate_assigned_count": outcome.alternate_assigned_count,
                    "alternate_assigned_period_units": outcome.alternate_assigned_period_units,
                    "mandatory_fallback_assigned_count": outcome.mandatory_fallback_assigned_count,
                    "mandatory_fallback_assigned_period_units": outcome.mandatory_fallback_assigned_period_units,
                    "mandatory_fallback_assignment_keys": "|".join(outcome.mandatory_fallback_assignment_keys),
                    "fully_scheduled": outcome.fully_scheduled,
                    "priority_protected": outcome.priority_protected,
                    "ordinary_fairness_violation": outcome.ordinary_fairness_violation,
                    "protected_fairness_violation": outcome.protected_fairness_violation,
                    "high_demand_guarantee_violation_count": outcome.high_demand_guarantee_violation_count,
                    "high_demand_violating_request_keys": "|".join(outcome.high_demand_violating_request_keys),
                    "target_course_count": outcome.target_period_units,
                    "assigned_course_count": outcome.assigned_period_units,
                    "schedule_gap_count": max(outcome.target_period_units - outcome.assigned_period_units, 0),
                    "assigned_alternate_count": outcome.alternate_assigned_count,
                    "target_logical_course_count": (
                        outcome.target_logical_course_count
                        if outcome.target_logical_course_count is not None
                        else outcome.target_period_units
                    ),
                    "assigned_logical_course_count": (
                        outcome.assigned_logical_course_count
                        if outcome.assigned_logical_course_count is not None
                        else len(outcome.assignment_keys)
                    ),
                    "logical_schedule_gap_count": (
                        outcome.logical_schedule_gap_count
                        if outcome.logical_schedule_gap_count is not None
                        else max(outcome.target_period_units - len(outcome.assignment_keys), 0)
                    ),
                    "logical_fully_scheduled": (
                        outcome.logical_fully_scheduled
                        if outcome.logical_fully_scheduled is not None
                        else len(outcome.assignment_keys) >= outcome.target_period_units
                    ),
                }
            )
    return tuple(sorted(rows, key=lambda row: (row["algorithm_name"], row["student_id"])))


def _request_outcome_rows(
    raw_results: tuple[BaselineResult | CpSatAllocationResult, ...],
) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for result in raw_results:
        for outcome in result.request_outcomes:
            diagnostics = _request_failure_diagnostics(outcome)
            rows.append(
                {
                    "algorithm_name": result.algorithm_name,
                    "request_key": outcome.request_key,
                    "student_id": outcome.student_id,
                    "request_type": outcome.request_type,
                    "alternate_rank": outcome.alternate_rank,
                    "candidate_key": outcome.candidate_key,
                    "period_units": outcome.period_units,
                    "status": outcome.status.value,
                    "assignment_key": outcome.assignment_key,
                    "assigned_linked_section_group_id": outcome.assigned_linked_section_group_id,
                    "candidate_attempts_count": len(outcome.candidate_attempts),
                    "candidate_rejections_count": diagnostics["candidate_rejections_count"],
                    "rejected_section_at_capacity_count": diagnostics["rejected_section_at_capacity_count"],
                    "rejected_period_conflict_count": diagnostics["rejected_period_conflict_count"],
                    "rejected_duplicate_logical_course_count": diagnostics[
                        "rejected_duplicate_logical_course_count"
                    ],
                    "rejected_student_load_limit_count": diagnostics["rejected_student_load_limit_count"],
                    "rejected_other_count": diagnostics["rejected_other_count"],
                    "terminal_unmet_reason": diagnostics["terminal_unmet_reason"],
                    "remaining_units_before": outcome.remaining_units_before,
                    "remaining_units_after": outcome.remaining_units_after,
                }
            )
    return tuple(sorted(rows, key=lambda row: (row["algorithm_name"], row["request_key"])))


def _request_failure_diagnostics(outcome: RequestOutcome) -> dict[str, Any]:
    primary_reasons = [_primary_rejection_reason(attempt.rejection_reasons) for attempt in outcome.candidate_attempts if not attempt.success]
    counts = {reason: primary_reasons.count(reason) for reason in REJECTION_REASON_CODES}
    candidate_rejections_count = len(primary_reasons)
    terminal_unmet_reason = _terminal_unmet_reason(outcome, counts, candidate_rejections_count)
    return {
        "candidate_attempts_count": len(outcome.candidate_attempts),
        "candidate_rejections_count": candidate_rejections_count,
        "rejected_section_at_capacity_count": counts["section_at_capacity"],
        "rejected_period_conflict_count": counts["period_conflict"],
        "rejected_duplicate_logical_course_count": counts["duplicate_logical_course"],
        "rejected_student_load_limit_count": counts["student_load_limit"],
        "rejected_other_count": counts["other_rejection"],
        "terminal_unmet_reason": terminal_unmet_reason,
    }


def _primary_rejection_reason(reasons: tuple[AssignmentRejectionReason, ...]) -> str:
    if not reasons:
        return "other_rejection"
    first = reasons[0]
    if first == AssignmentRejectionReason.SECTION_FULL:
        return "section_at_capacity"
    if first == AssignmentRejectionReason.PERIOD_CONFLICT:
        return "period_conflict"
    if first == AssignmentRejectionReason.DUPLICATE_LOGICAL_COURSE_OR_BLOCK:
        return "duplicate_logical_course"
    if first == AssignmentRejectionReason.TARGET_LOAD_EXCEEDED:
        return "student_load_limit"
    return "other_rejection"


def _terminal_unmet_reason(
    outcome: RequestOutcome,
    counts: dict[str, int],
    candidate_rejections_count: int,
) -> str:
    if outcome.status in {PrimaryRequestStatus.ASSIGNED, AlternateRequestStatus.ASSIGNED}:
        return ""
    if not outcome.candidate_attempts:
        if outcome.status in {
            PrimaryRequestStatus.UNMET_NO_CANDIDATES,
            AlternateRequestStatus.UNASSIGNED_NO_CANDIDATES,
        }:
            return "no_candidate_sections"
        return "not_attempted"
    if candidate_rejections_count == 0:
        return "other_rejection"
    nonzero_reasons = [reason for reason in REJECTION_REASON_CODES if counts[reason] > 0]
    if len(nonzero_reasons) > 1:
        return "mixed_rejections"
    only_reason = nonzero_reasons[0] if nonzero_reasons else "other_rejection"
    if only_reason == "other_rejection":
        return "other_rejection"
    return f"all_{only_reason}"


def _assignment_failure_summary_sort_key(row: dict[str, Any]) -> tuple[int, int, int]:
    return (
        row["_algorithm_order"],
        _ordered_index(REQUEST_KIND_ORDER, row["request_kind"]),
        _ordered_index(TERMINAL_UNMET_REASON_ORDER, row["terminal_unmet_reason"]),
    )


def _student_schedule_gap_sort_key(row: dict[str, Any]) -> tuple[int, int, int, str]:
    return (row["_algorithm_order"], -int(row["schedule_gap_count"]), int(row["grade"]), row["student_id"])


def _ordered_index(order: tuple[str, ...], value: str) -> int:
    try:
        return order.index(value)
    except ValueError:
        return len(order)


def _json_array(values: Iterable[str]) -> str:
    return json.dumps(sorted(values), separators=(",", ":"))


def _without_internal_sort_keys(row: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def _benchmark_manifest_payload(suite: BenchmarkSuiteResult, written_files: tuple[str, ...]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "runner_name": RUNNER_NAME,
        "diagnostics_schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "manifest": suite.manifest.to_dict(),
        "expected_fingerprint": asdict(suite.expected_fingerprint) if suite.expected_fingerprint is not None else None,
        "fingerprint_verified": suite.fingerprint_verified,
        "algorithms_run": list(suite.algorithms_run),
        "artifact_files": list(written_files),
        "diagnostics_artifact_files": list(DIAGNOSTIC_ARTIFACT_FILES),
        "request_outcome_diagnostics_fields": [
            "candidate_rejections_count",
            "rejected_section_at_capacity_count",
            "rejected_period_conflict_count",
            "rejected_duplicate_logical_course_count",
            "rejected_student_load_limit_count",
            "rejected_other_count",
            "terminal_unmet_reason",
        ],
        "rejection_reason_codes": list(REJECTION_REASON_CODES),
        "rejection_precedence": [
            "Uses AllocationState._assignment_rejection_reasons order and maps the first returned reason.",
            "Normal mapped checks are duplicate_logical_course, period_conflict, student_load_limit, section_at_capacity.",
            "Unmapped state/candidate errors are reported as other_rejection.",
        ],
        "candidate_attempts_count_definition": (
            "Count of candidate sections actually entered in the assignment loop and evaluated with "
            "AllocationState.try_assign; includes the final successful candidate, excludes skipped candidates "
            "after success, and is 0 when no candidate loop is entered."
        ),
        "candidate_rejections_count_definition": (
            "Count of attempted candidates whose AllocationState.try_assign result was rejected. Each rejected "
            "candidate contributes exactly one primary rejection reason using the recorded first reason."
        ),
        "terminal_classification_rules": {
            "no_candidate_sections": "The request had no candidate sections and no candidate loop was entered.",
            "all_<reason>": "At least one candidate was rejected and every rejected candidate had the same primary reason.",
            "mixed_rejections": "Rejected candidates contained two or more primary rejection reasons.",
            "other_rejection": "A candidate was rejected through a real path not mapped to the stable taxonomy.",
            "not_attempted": "The request outcome row exists but no assignment candidate loop was entered.",
        },
        "request_kinds": list(REQUEST_KIND_ORDER),
        "known_diagnostics_limitation": (
            "Rejection diagnostics describe reasons observed by the evaluated greedy algorithm under its evolving "
            "assignment state. They do not prove global infeasibility and may change under a different assignment "
            "order or optimizer."
        ),
        "section_at_capacity_limitation": (
            "section_at_capacity means the candidate section was at capacity when checked; it does not prove "
            "global total-seat insufficiency."
        ),
        "period_conflict_limitation": (
            "period_conflict means the candidate overlapped the student's already-assigned periods at that "
            "moment; it does not prove a swap, repair, or global optimizer could not resolve the conflict."
        ),
        "final_schedule_policy_schema_version": FINAL_POLICY_SCHEMA_VERSION,
        "final_schedule_policy_artifact_files": [
            "final_schedule_policy_summary.csv",
            "final_schedule_policy_violations.csv",
        ],
        "final_schedule_policy_rules": {
            "protected_primary_unmet": "priority_protected students must have primary_unmet_count == 0.",
            "ordinary_primary_unmet_over_limit": (
                "non-protected students must have primary_unmet_count <= "
                f"{MAXIMUM_ORDINARY_PRIMARY_UNMET_COUNT}."
            ),
            "schedule_gap_over_limit": (
                "each student must have final logical schedule_gap_count <= "
                f"{MAXIMUM_SCHEDULE_GAP_COUNT}."
            ),
            "below_minimum_course_count": (
                "each student must have assigned_logical_course_count >= "
                f"{MINIMUM_ASSIGNED_LOGICAL_COURSE_COUNT}."
            ),
        },
        "minimum_assigned_course_count": MINIMUM_ASSIGNED_LOGICAL_COURSE_COUNT,
        "maximum_ordinary_primary_unmet_count": MAXIMUM_ORDINARY_PRIMARY_UNMET_COUNT,
        "maximum_protected_primary_unmet_count": MAXIMUM_PROTECTED_PRIMARY_UNMET_COUNT,
        "maximum_schedule_gap_count": MAXIMUM_SCHEDULE_GAP_COUNT,
        "final_schedule_policy_reason_codes": list(FINAL_POLICY_REASON_CODES),
        "course_count_semantics": COURSE_COUNT_SEMANTICS,
        "fully_scheduled_definition": (
            "Legacy period-unit metric: StudentOutcome.fully_scheduled is true when assigned_period_units "
            "equals target_period_units. A double-period logical course can make this true while the "
            "student still has a logical-course gap."
        ),
        "logical_fully_scheduled_definition": (
            "Logical-course metric: logical_fully_scheduled is true when assigned_logical_course_count "
            "is at least target_logical_course_count."
        ),
        "logical_schedule_gap_definition": (
            "logical_schedule_gap_count = max(target_logical_course_count - "
            "assigned_logical_course_count, 0); Final Schedule Policy Gate v1 uses this "
            "logical-course gap, not the legacy period-unit gap."
        ),
        "logical_schedule_completion_objective_definition": (
            "CP-SAT logical_schedule_completion maximizes assigned logical-course count after "
            "higher-priority math coverage and primary objective values are fixed. With Final "
            "Schedule Policy Gate v1 enabled, maximum logical schedule gap is 1, so maximizing "
            "assigned logical courses is equivalent to minimizing total logical schedule gap "
            "conditionally on the fixed higher-priority incumbent values."
        ),
        "final_schedule_policy_limitation": (
            "A benchmark algorithm may complete successfully while failing the final schedule policy gate. "
            "A failed gate means the result is not eligible to be described or exported as a publishable "
            "final schedule."
        ),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run manifest-guarded allocation benchmarks.")
    parser.add_argument("--generated-input-dir", required=True)
    parser.add_argument("--sections-input-dir", required=True)
    parser.add_argument("--config-dir", default="data/config")
    parser.add_argument("--course-catalog-path")
    parser.add_argument("--scenario", default="stable_year")
    parser.add_argument("--data-seed", type=int, required=True)
    parser.add_argument("--section-seed", type=int, required=True)
    parser.add_argument("--solver-seed", type=int, required=True)
    parser.add_argument("--algorithms", default="random,constrained")
    parser.add_argument("--output-json")
    parser.add_argument("--output-csv")
    parser.add_argument("--output-artifact-dir")
    parser.add_argument("--include-large-tables", action="store_true")
    parser.add_argument("--cp-sat-max-time-seconds-per-stage", type=float, default=30.0)
    parser.add_argument("--cp-sat-max-total-time-seconds", type=float)
    parser.add_argument("--cp-sat-bootstrap-time-seconds", type=float)
    parser.add_argument("--cp-sat-disable-feasibility-bootstrap", action="store_true")
    parser.add_argument("--cp-sat-disable-constrained-first-hint", action="store_true")
    parser.add_argument("--cp-sat-disable-logical-schedule-completion", action="store_true")
    parser.add_argument("--cp-sat-num-search-workers", type=int, default=1)
    parser.add_argument("--cp-sat-initial-solution-artifact-dir")
    args = parser.parse_args(tuple(argv) if argv is not None else None)
    seeds = ExperimentSeeds(args.data_seed, args.section_seed, args.solver_seed)
    algorithms = tuple(args.algorithms.split(","))
    try:
        result = run_benchmark_suite(
            generated_input_dir=args.generated_input_dir,
            sections_input_dir=args.sections_input_dir,
            config_dir=args.config_dir,
            course_catalog_path=args.course_catalog_path,
            scenario_id=args.scenario,
            seeds=seeds,
            algorithms=algorithms,
            output_json_path=args.output_json,
            output_csv_path=args.output_csv,
            output_artifact_dir=args.output_artifact_dir,
            include_large_tables=args.include_large_tables,
            cp_sat_options=CpSatBenchmarkOptions(
                max_time_seconds_per_stage=args.cp_sat_max_time_seconds_per_stage,
                max_total_time_seconds=args.cp_sat_max_total_time_seconds,
                bootstrap_time_seconds=args.cp_sat_bootstrap_time_seconds,
                use_feasibility_bootstrap=not args.cp_sat_disable_feasibility_bootstrap,
                use_constrained_first_hint=not args.cp_sat_disable_constrained_first_hint,
                num_search_workers=args.cp_sat_num_search_workers,
                logical_schedule_completion_enabled=not args.cp_sat_disable_logical_schedule_completion,
                initial_solution_artifact_dir=args.cp_sat_initial_solution_artifact_dir,
            ),
        )
    except (BenchmarkRunnerError, ExperimentManifestError, ValueError) as exc:
        print(f"Benchmark runner failed: {exc}")
        print("Check data_generation_seed, section_planning_seed, solver_seed, generated_input_dir, sections_input_dir, and input paths.")
        return 1
    print(
        "Benchmark PASS: "
        f"{', '.join(result.algorithms_run)} on "
        f"{result.manifest.fingerprint.students} students, "
        f"{result.manifest.fingerprint.logical_primaries} logical primaries"
    )
    if "cp_sat" not in result.algorithms_run:
        print("CP-SAT was not run; include --algorithms ... ,cp_sat to opt in.")
    for row in result.results:
        status = "PASS" if row.final_schedule_policy_pass else "FAIL"
        print(
            f"Final schedule policy: {row.algorithm_name} {status} "
            f"({row.final_schedule_policy_violation_students} violating student(s))"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
