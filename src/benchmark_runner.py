from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.allocation import (
    AlternateRequestStatus,
    BaselineResult,
    CanonicalAllocationInput,
    CpSatAllocationResult,
    MandatoryFallbackStatus,
    PrimaryRequestStatus,
    evaluate_math_policy,
    load_math_fallback_rules,
    math_course_ids_from_catalog,
    run_constrained_first_baseline,
    run_fair_cp_sat_solver,
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


DEFAULT_ALGORITHMS = ("random", "constrained")
RUNNER_NAME = "benchmark_runner_v1"


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

    results = tuple(
        _run_algorithm(name, allocation_input, seeds.solver_seed, math_course_ids, math_fallback_rules, cp_sat_options)
        for name in algorithms
    )
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
    return suite


def _normalize_algorithms(algorithms: tuple[str, ...]) -> tuple[str, ...]:
    valid = {"random", "constrained", "cp_sat"}
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
) -> BenchmarkAlgorithmResult:
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
        )
    else:  # pragma: no cover - _normalize_algorithms keeps this unreachable.
        raise BenchmarkRunnerError(f"Unsupported benchmark algorithm: {name}")
    return _summarize_result(allocation_input, result, round(time.perf_counter() - started, 6), math_course_ids, math_fallback_rules)


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
            "stage_diagnostics_summary": tuple(
            {
                "stage_name": diagnostic.stage_name.value,
                "model_scope": diagnostic.model_scope.value,
                "status": diagnostic.status.value,
                "optimum_proven": diagnostic.optimum_proven,
                "conditional_on_unproven_incumbent": diagnostic.conditional_on_unproven_incumbent,
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
    parser.add_argument("--cp-sat-max-time-seconds-per-stage", type=float, default=30.0)
    parser.add_argument("--cp-sat-max-total-time-seconds", type=float)
    parser.add_argument("--cp-sat-bootstrap-time-seconds", type=float)
    parser.add_argument("--cp-sat-disable-feasibility-bootstrap", action="store_true")
    parser.add_argument("--cp-sat-disable-constrained-first-hint", action="store_true")
    parser.add_argument("--cp-sat-num-search-workers", type=int, default=1)
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
            cp_sat_options=CpSatBenchmarkOptions(
                max_time_seconds_per_stage=args.cp_sat_max_time_seconds_per_stage,
                max_total_time_seconds=args.cp_sat_max_total_time_seconds,
                bootstrap_time_seconds=args.cp_sat_bootstrap_time_seconds,
                use_feasibility_bootstrap=not args.cp_sat_disable_feasibility_bootstrap,
                use_constrained_first_hint=not args.cp_sat_disable_constrained_first_hint,
                num_search_workers=args.cp_sat_num_search_workers,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
