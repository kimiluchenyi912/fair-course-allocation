from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "final_schedule_policy_gate_v1"
MINIMUM_ASSIGNED_LOGICAL_COURSE_COUNT = 5
MAXIMUM_SCHEDULE_GAP_COUNT = 1
MAXIMUM_ORDINARY_PRIMARY_UNMET_COUNT = 1
MAXIMUM_PROTECTED_PRIMARY_UNMET_COUNT = 0

BELOW_MINIMUM_COURSE_COUNT = "below_minimum_course_count"
SCHEDULE_GAP_OVER_LIMIT = "schedule_gap_over_limit"
PROTECTED_PRIMARY_UNMET = "protected_primary_unmet"
ORDINARY_PRIMARY_UNMET_OVER_LIMIT = "ordinary_primary_unmet_over_limit"

REASON_CODES = (
    BELOW_MINIMUM_COURSE_COUNT,
    SCHEDULE_GAP_OVER_LIMIT,
    PROTECTED_PRIMARY_UNMET,
    ORDINARY_PRIMARY_UNMET_OVER_LIMIT,
)
SEVERITY_ORDER = {
    BELOW_MINIMUM_COURSE_COUNT: 0,
    SCHEDULE_GAP_OVER_LIMIT: 1,
    PROTECTED_PRIMARY_UNMET: 2,
    ORDINARY_PRIMARY_UNMET_OVER_LIMIT: 3,
}
COURSE_COUNT_SEMANTICS = (
    "target_logical_course_count uses the student's configured target_course_count/target_period_units. "
    "assigned_logical_course_count counts assignment records, so Math 2/3 Honors Accelerated counts as "
    "one logical course even though it uses two period units."
)

SUMMARY_FIELDNAMES = (
    "algorithm_name",
    "final_schedule_policy_pass",
    "violating_student_count",
    "protected_primary_unmet_violation_count",
    "ordinary_primary_unmet_violation_count",
    "schedule_gap_over_limit_count",
    "below_minimum_course_count",
    "minimum_assigned_course_count",
    "maximum_schedule_gap_count",
    "maximum_primary_unmet_count",
    "logical_fully_scheduled_student_count",
    "students_with_logical_schedule_gap",
    "total_logical_schedule_gap",
)
VIOLATION_FIELDNAMES = (
    "algorithm_name",
    "student_id",
    "grade",
    "is_priority_protected",
    "target_logical_course_count",
    "assigned_logical_course_count",
    "schedule_gap_count",
    "primary_unmet_count",
    "alternates_assigned_count",
    "violation_reasons",
)


class FinalSchedulePolicyError(ValueError):
    pass


@dataclass(frozen=True)
class FinalScheduleStudentViolation:
    algorithm_name: str
    student_id: str
    grade: int
    is_priority_protected: bool
    target_logical_course_count: int
    assigned_logical_course_count: int
    schedule_gap_count: int
    primary_unmet_count: int
    alternates_assigned_count: int
    violation_reasons: tuple[str, ...]


@dataclass(frozen=True)
class FinalSchedulePolicySummary:
    algorithm_name: str
    final_schedule_policy_pass: bool
    violating_student_count: int
    protected_primary_unmet_violation_count: int
    ordinary_primary_unmet_violation_count: int
    schedule_gap_over_limit_count: int
    below_minimum_course_count: int
    minimum_assigned_course_count: int
    maximum_schedule_gap_count: int
    maximum_primary_unmet_count: int
    logical_fully_scheduled_student_count: int
    students_with_logical_schedule_gap: int
    total_logical_schedule_gap: int


@dataclass(frozen=True)
class FinalSchedulePolicyReport:
    summary: FinalSchedulePolicySummary
    violations: tuple[FinalScheduleStudentViolation, ...]


def evaluate_final_schedule_policy(algorithm_name: str, student_outcomes: Iterable[Any]) -> FinalSchedulePolicyReport:
    violations: list[FinalScheduleStudentViolation] = []
    minimum_assigned: int | None = None
    maximum_gap = 0
    maximum_primary_unmet = 0
    logical_fully_scheduled = 0
    logical_gap_students = 0
    total_logical_gap = 0
    for outcome in student_outcomes:
        student = _student_view(algorithm_name, outcome)
        minimum_assigned = (
            student.assigned_logical_course_count
            if minimum_assigned is None
            else min(minimum_assigned, student.assigned_logical_course_count)
        )
        maximum_gap = max(maximum_gap, student.schedule_gap_count)
        maximum_primary_unmet = max(maximum_primary_unmet, student.primary_unmet_count)
        if student.schedule_gap_count == 0:
            logical_fully_scheduled += 1
        else:
            logical_gap_students += 1
            total_logical_gap += student.schedule_gap_count
        reasons = _violation_reasons(student)
        if reasons:
            violations.append(
                FinalScheduleStudentViolation(
                    algorithm_name=algorithm_name,
                    student_id=student.student_id,
                    grade=student.grade,
                    is_priority_protected=student.is_priority_protected,
                    target_logical_course_count=student.target_logical_course_count,
                    assigned_logical_course_count=student.assigned_logical_course_count,
                    schedule_gap_count=student.schedule_gap_count,
                    primary_unmet_count=student.primary_unmet_count,
                    alternates_assigned_count=student.alternates_assigned_count,
                    violation_reasons=reasons,
                )
            )
    summary = FinalSchedulePolicySummary(
        algorithm_name=algorithm_name,
        final_schedule_policy_pass=not violations,
        violating_student_count=len(violations),
        protected_primary_unmet_violation_count=sum(PROTECTED_PRIMARY_UNMET in item.violation_reasons for item in violations),
        ordinary_primary_unmet_violation_count=sum(
            ORDINARY_PRIMARY_UNMET_OVER_LIMIT in item.violation_reasons for item in violations
        ),
        schedule_gap_over_limit_count=sum(SCHEDULE_GAP_OVER_LIMIT in item.violation_reasons for item in violations),
        below_minimum_course_count=sum(BELOW_MINIMUM_COURSE_COUNT in item.violation_reasons for item in violations),
        minimum_assigned_course_count=minimum_assigned if minimum_assigned is not None else 0,
        maximum_schedule_gap_count=maximum_gap,
        maximum_primary_unmet_count=maximum_primary_unmet,
        logical_fully_scheduled_student_count=logical_fully_scheduled,
        students_with_logical_schedule_gap=logical_gap_students,
        total_logical_schedule_gap=total_logical_gap,
    )
    return FinalSchedulePolicyReport(summary=summary, violations=tuple(sorted(violations, key=_violation_sort_key)))


def summary_row(report: FinalSchedulePolicyReport) -> dict[str, Any]:
    summary = report.summary
    return {
        "algorithm_name": summary.algorithm_name,
        "final_schedule_policy_pass": summary.final_schedule_policy_pass,
        "violating_student_count": summary.violating_student_count,
        "protected_primary_unmet_violation_count": summary.protected_primary_unmet_violation_count,
        "ordinary_primary_unmet_violation_count": summary.ordinary_primary_unmet_violation_count,
        "schedule_gap_over_limit_count": summary.schedule_gap_over_limit_count,
        "below_minimum_course_count": summary.below_minimum_course_count,
        "minimum_assigned_course_count": summary.minimum_assigned_course_count,
        "maximum_schedule_gap_count": summary.maximum_schedule_gap_count,
        "maximum_primary_unmet_count": summary.maximum_primary_unmet_count,
        "logical_fully_scheduled_student_count": summary.logical_fully_scheduled_student_count,
        "students_with_logical_schedule_gap": summary.students_with_logical_schedule_gap,
        "total_logical_schedule_gap": summary.total_logical_schedule_gap,
    }


def violation_row(violation: FinalScheduleStudentViolation) -> dict[str, Any]:
    return {
        "algorithm_name": violation.algorithm_name,
        "student_id": violation.student_id,
        "grade": violation.grade,
        "is_priority_protected": violation.is_priority_protected,
        "target_logical_course_count": violation.target_logical_course_count,
        "assigned_logical_course_count": violation.assigned_logical_course_count,
        "schedule_gap_count": violation.schedule_gap_count,
        "primary_unmet_count": violation.primary_unmet_count,
        "alternates_assigned_count": violation.alternates_assigned_count,
        "violation_reasons": json.dumps(list(violation.violation_reasons), separators=(",", ":")),
    }


def load_policy_reports_from_artifacts(artifact_dir: str | Path) -> tuple[FinalSchedulePolicyReport, ...]:
    path = _artifact_data_dir(Path(artifact_dir))
    student_outcomes = path / "student_outcomes.csv"
    if not student_outcomes.is_file():
        raise FinalSchedulePolicyError(f"Missing required artifact file: {student_outcomes}")
    rows = _read_student_outcome_rows(student_outcomes)
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["algorithm_name"], []).append(row)
    return tuple(evaluate_final_schedule_policy(algorithm_name, grouped[algorithm_name]) for algorithm_name in grouped)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check final schedule policy compliance for benchmark artifacts.")
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--algorithm")
    args = parser.parse_args(tuple(argv) if argv is not None else None)
    try:
        reports = load_policy_reports_from_artifacts(args.artifact_dir)
        if args.algorithm:
            reports = tuple(report for report in reports if report.summary.algorithm_name == args.algorithm)
            if not reports:
                raise FinalSchedulePolicyError(f"Algorithm not found in artifacts: {args.algorithm}")
    except FinalSchedulePolicyError as exc:
        print(f"Final schedule policy gate failed to run: {exc}")
        return 2

    failed = False
    for report in reports:
        summary = report.summary
        status = "PASS" if summary.final_schedule_policy_pass else "FAIL"
        print(
            f"{summary.algorithm_name}: {status} | "
            f"violating students={summary.violating_student_count} | "
            f"protected={summary.protected_primary_unmet_violation_count} | "
            f"ordinary={summary.ordinary_primary_unmet_violation_count} | "
            f"schedule_gap_over_limit={summary.schedule_gap_over_limit_count} | "
            f"below_minimum_course={summary.below_minimum_course_count}"
        )
        failed = failed or not summary.final_schedule_policy_pass
    return 1 if failed else 0


@dataclass(frozen=True)
class _StudentPolicyView:
    algorithm_name: str
    student_id: str
    grade: int
    is_priority_protected: bool
    target_logical_course_count: int
    assigned_logical_course_count: int
    schedule_gap_count: int
    primary_unmet_count: int
    alternates_assigned_count: int


def _student_view(algorithm_name: str, outcome: Any) -> _StudentPolicyView:
    target = _int_value(outcome, "target_logical_course_count" if isinstance(outcome, dict) else "target_period_units")
    assigned = _assigned_logical_course_count(outcome)
    return _StudentPolicyView(
        algorithm_name=algorithm_name,
        student_id=str(_value(outcome, "student_id")),
        grade=_int_value(outcome, "grade"),
        is_priority_protected=_bool_value(outcome, "priority_protected"),
        target_logical_course_count=target,
        assigned_logical_course_count=assigned,
        schedule_gap_count=max(target - assigned, 0),
        primary_unmet_count=_int_value(outcome, "primary_unmet_count"),
        alternates_assigned_count=_int_value(outcome, "alternate_assigned_count"),
    )


def _assigned_logical_course_count(outcome: Any) -> int:
    if isinstance(outcome, dict):
        return _int_value(outcome, "assigned_logical_course_count")
    assignment_keys = _value(outcome, "assignment_keys")
    return len(tuple(assignment_keys))


def _violation_reasons(student: _StudentPolicyView) -> tuple[str, ...]:
    reasons: list[str] = []
    if student.assigned_logical_course_count < MINIMUM_ASSIGNED_LOGICAL_COURSE_COUNT:
        reasons.append(BELOW_MINIMUM_COURSE_COUNT)
    if student.schedule_gap_count > MAXIMUM_SCHEDULE_GAP_COUNT:
        reasons.append(SCHEDULE_GAP_OVER_LIMIT)
    if student.is_priority_protected and student.primary_unmet_count > MAXIMUM_PROTECTED_PRIMARY_UNMET_COUNT:
        reasons.append(PROTECTED_PRIMARY_UNMET)
    if not student.is_priority_protected and student.primary_unmet_count > MAXIMUM_ORDINARY_PRIMARY_UNMET_COUNT:
        reasons.append(ORDINARY_PRIMARY_UNMET_OVER_LIMIT)
    return tuple(reason for reason in REASON_CODES if reason in reasons)


def _violation_sort_key(violation: FinalScheduleStudentViolation) -> tuple[int, int, int, int, str]:
    severity = min(SEVERITY_ORDER[reason] for reason in violation.violation_reasons)
    return (
        severity,
        -violation.schedule_gap_count,
        -violation.primary_unmet_count,
        violation.grade,
        violation.student_id,
    )


def _artifact_data_dir(path: Path) -> Path:
    if (path / "student_outcomes.csv").is_file():
        return path
    if (path / "artifacts" / "student_outcomes.csv").is_file():
        return path / "artifacts"
    return path


def _read_student_outcome_rows(path: Path) -> tuple[dict[str, str], ...]:
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(_REQUIRED_STUDENT_OUTCOME_COLUMNS) - set(reader.fieldnames or ())
        if missing:
            raise FinalSchedulePolicyError(f"{path.name} is missing required column(s): {', '.join(sorted(missing))}")
        rows = tuple(dict(row) for row in reader)
    if not rows:
        raise FinalSchedulePolicyError(f"{path.name} contains no student outcome rows.")
    return rows


_REQUIRED_STUDENT_OUTCOME_COLUMNS = (
    "algorithm_name",
    "student_id",
    "grade",
    "target_logical_course_count",
    "assigned_logical_course_count",
    "primary_unmet_count",
    "alternate_assigned_count",
    "priority_protected",
)


def _value(outcome: Any, name: str) -> Any:
    if isinstance(outcome, dict):
        if name not in outcome:
            raise FinalSchedulePolicyError(f"Missing student outcome field: {name}")
        return outcome[name]
    return getattr(outcome, name)


def _int_value(outcome: Any, name: str) -> int:
    value = _value(outcome, name)
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise FinalSchedulePolicyError(f"Invalid integer for {name}: {value!r}") from exc
    if number < 0:
        raise FinalSchedulePolicyError(f"Invalid negative integer for {name}: {value!r}")
    return number


def _bool_value(outcome: Any, name: str) -> bool:
    value = _value(outcome, name)
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no", ""}:
        return False
    raise FinalSchedulePolicyError(f"Invalid boolean for {name}: {value!r}")


if __name__ == "__main__":
    raise SystemExit(main())
