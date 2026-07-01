from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import pandas as pd

from src.validation.helpers import parse_bool, text

from .baseline_models import BaselineResult, MandatoryFallbackOutcome, MandatoryFallbackStatus, PrimaryRequestStatus
from .input_models import CanonicalAllocationInput, LogicalRequest
from .math_policy_models import (
    MathCoverageStatus,
    MathFallbackConfigError,
    MathFallbackRule,
    MathPolicyReport,
    MathPolicyViolationType,
    StudentMathPolicyOutcome,
)


MATH_DEPARTMENT = "Mathematics"
MANDATORY_FALLBACK_POLICY_TYPE = "mandatory_fallback"


def math_course_ids_from_catalog(catalog: pd.DataFrame) -> tuple[str, ...]:
    required = {"course_id", "department"}
    missing = required - set(catalog.columns)
    if missing:
        raise MathFallbackConfigError(f"course_catalog.csv missing columns: {sorted(missing)}")
    return tuple(sorted(text(row["course_id"]) for _, row in catalog.iterrows() if text(row["department"]) == MATH_DEPARTMENT))


def load_math_fallback_rules(
    config_dir: str | Path = "data/config",
    catalog: pd.DataFrame | None = None,
) -> tuple[MathFallbackRule, ...]:
    config_dir = Path(config_dir)
    catalog = pd.read_csv(config_dir / "course_catalog.csv", keep_default_na=False) if catalog is None else catalog
    fallback_df = pd.read_csv(config_dir / "math_fallbacks.csv", keep_default_na=False)
    return parse_math_fallback_rules(catalog, fallback_df)


def parse_math_fallback_rules(
    catalog: pd.DataFrame,
    fallback_df: pd.DataFrame,
) -> tuple[MathFallbackRule, ...]:
    required = {
        "source_course_id",
        "fallback_course_id",
        "policy_type",
        "enabled",
        "notes",
    }
    missing = required - set(fallback_df.columns)
    if missing:
        raise MathFallbackConfigError(f"math_fallbacks.csv missing columns: {sorted(missing)}")
    catalog_by_id = {text(row["course_id"]): row for _, row in catalog.iterrows()}
    enabled_sources: dict[str, str] = {}
    rules: list[MathFallbackRule] = []
    for index, row in fallback_df.iterrows():
        line = int(index) + 2
        source = text(row["source_course_id"])
        fallback = text(row["fallback_course_id"])
        policy_type = text(row["policy_type"])
        enabled = parse_bool(row["enabled"])
        notes = text(row["notes"])
        if not source:
            raise MathFallbackConfigError(f"math_fallbacks.csv line {line}: source_course_id cannot be blank.")
        if not fallback:
            raise MathFallbackConfigError(f"math_fallbacks.csv line {line}: fallback_course_id cannot be blank.")
        if policy_type != MANDATORY_FALLBACK_POLICY_TYPE:
            raise MathFallbackConfigError(f"math_fallbacks.csv line {line}: unsupported policy_type {policy_type!r}.")
        if enabled is None:
            raise MathFallbackConfigError(f"math_fallbacks.csv line {line}: enabled must be true or false.")
        _require_math_course(catalog_by_id, source, "source_course_id", line)
        _require_math_course(catalog_by_id, fallback, "fallback_course_id", line)
        if enabled:
            previous = enabled_sources.get(source)
            if previous is not None and previous != fallback:
                raise MathFallbackConfigError(
                    f"math_fallbacks.csv line {line}: enabled source {source} has conflicting mandatory fallbacks."
                )
            enabled_sources[source] = fallback
        rules.append(MathFallbackRule(source, fallback, policy_type, bool(enabled), notes))
    return tuple(sorted(rules, key=lambda item: (item.source_course_id, item.fallback_course_id, item.policy_type)))


def evaluate_math_policy(
    allocation_input: CanonicalAllocationInput,
    baseline_result: BaselineResult,
    math_course_ids: tuple[str, ...],
    math_fallback_rules: tuple[MathFallbackRule, ...],
) -> MathPolicyReport:
    math_ids = tuple(sorted(math_course_ids))
    math_id_set = set(math_ids)
    enabled_mandatory_fallbacks = {
        rule.source_course_id: rule
        for rule in math_fallback_rules
        if rule.enabled and rule.policy_type == MANDATORY_FALLBACK_POLICY_TYPE
    }
    outcomes_by_request = {outcome.request_key: outcome for outcome in baseline_result.request_outcomes}
    fallback_outcomes_by_source = {
        outcome.source_request_key: outcome
        for outcome in getattr(baseline_result, "mandatory_fallback_outcomes", ())
    }
    primary_math_by_student: dict[str, list[LogicalRequest]] = defaultdict(list)
    for request in allocation_input.logical_requests:
        if request.request_type == "primary" and request.candidate_key in math_id_set:
            primary_math_by_student[request.student_id].append(request)

    student_outcomes: list[StudentMathPolicyOutcome] = []
    for student in allocation_input.students:
        math_requests = tuple(sorted(primary_math_by_student.get(student.student_id, []), key=lambda item: item.request_key))
        assigned_math = tuple(
            request
            for request in math_requests
            if outcomes_by_request[request.request_key].status == PrimaryRequestStatus.ASSIGNED
        )
        coverage_status = MathCoverageStatus.NO_MATH_REQUIRED
        violation_type = MathPolicyViolationType.NONE
        fallback_source_request_key = None
        fallback_source_course_id = None
        fallback_target_course_id = None
        fallback_assignment_key = None
        fallback_outcome_status = None
        if len(math_requests) == 1:
            request = math_requests[0]
            outcome = outcomes_by_request[request.request_key]
            if outcome.status == PrimaryRequestStatus.ASSIGNED:
                coverage_status = MathCoverageStatus.SATISFIED_BY_PRIMARY
            elif request.candidate_key in enabled_mandatory_fallbacks:
                rule = enabled_mandatory_fallbacks[request.candidate_key]
                fallback_outcome = fallback_outcomes_by_source.get(request.request_key)
                coverage_status, violation_type = _status_from_fallback_outcome(fallback_outcome)
                fallback_source_request_key = request.request_key
                fallback_source_course_id = rule.source_course_id
                fallback_target_course_id = rule.fallback_course_id
                fallback_assignment_key = fallback_outcome.assignment_key if fallback_outcome is not None else None
                fallback_outcome_status = fallback_outcome.status.value if fallback_outcome is not None else None
            else:
                coverage_status = MathCoverageStatus.VIOLATED_SINGLE_MATH_REQUIRED
                violation_type = MathPolicyViolationType.SINGLE_MATH_REQUIRED
        elif len(math_requests) > 1:
            if assigned_math:
                coverage_status = MathCoverageStatus.SATISFIED_BY_PRIMARY
            else:
                fallback_outcome = _fallback_outcome_for_requests(math_requests, fallback_outcomes_by_source)
                if fallback_outcome is not None:
                    coverage_status, violation_type = _status_from_fallback_outcome(fallback_outcome)
                    fallback_source_request_key = fallback_outcome.source_request_key
                    fallback_source_course_id = fallback_outcome.source_course_id
                    fallback_target_course_id = fallback_outcome.fallback_course_id
                    fallback_assignment_key = fallback_outcome.assignment_key
                    fallback_outcome_status = fallback_outcome.status.value
                elif any(request.candidate_key in enabled_mandatory_fallbacks for request in math_requests):
                    request = next(request for request in math_requests if request.candidate_key in enabled_mandatory_fallbacks)
                    rule = enabled_mandatory_fallbacks[request.candidate_key]
                    coverage_status = MathCoverageStatus.PENDING_MANDATORY_FALLBACK
                    fallback_source_request_key = request.request_key
                    fallback_source_course_id = rule.source_course_id
                    fallback_target_course_id = rule.fallback_course_id
                else:
                    coverage_status = MathCoverageStatus.VIOLATED_MULTIPLE_MATH_AT_LEAST_ONE
                    violation_type = MathPolicyViolationType.MULTIPLE_MATH_AT_LEAST_ONE

        student_outcomes.append(
            StudentMathPolicyOutcome(
                student_id=student.student_id,
                math_primary_request_keys=tuple(request.request_key for request in math_requests),
                math_primary_candidate_keys=tuple(request.candidate_key for request in math_requests),
                math_primary_count=len(math_requests),
                assigned_math_primary_request_keys=tuple(request.request_key for request in assigned_math),
                assigned_math_primary_count=len(assigned_math),
                coverage_status=coverage_status,
                violation_type=violation_type,
                fallback_source_request_key=fallback_source_request_key,
                fallback_source_course_id=fallback_source_course_id,
                fallback_target_course_id=fallback_target_course_id,
                fallback_assignment_key=fallback_assignment_key,
                fallback_outcome_status=fallback_outcome_status,
            )
        )

    student_outcomes_tuple = tuple(sorted(student_outcomes, key=lambda item: item.student_id))
    no_math = tuple(item.student_id for item in student_outcomes_tuple if item.coverage_status == MathCoverageStatus.NO_MATH_REQUIRED)
    single_math = tuple(item.student_id for item in student_outcomes_tuple if item.math_primary_count == 1)
    multiple_math = tuple(item.student_id for item in student_outcomes_tuple if item.math_primary_count > 1)
    satisfied_by_primary = tuple(item.student_id for item in student_outcomes_tuple if item.coverage_status == MathCoverageStatus.SATISFIED_BY_PRIMARY)
    satisfied_by_fallback = tuple(
        item.student_id for item in student_outcomes_tuple if item.coverage_status == MathCoverageStatus.SATISFIED_BY_MANDATORY_FALLBACK
    )
    satisfied = tuple(sorted(satisfied_by_primary + satisfied_by_fallback))
    pending = tuple(item.student_id for item in student_outcomes_tuple if item.coverage_status == MathCoverageStatus.PENDING_MANDATORY_FALLBACK)
    single_violations = tuple(
        item.student_id for item in student_outcomes_tuple if item.coverage_status == MathCoverageStatus.VIOLATED_SINGLE_MATH_REQUIRED
    )
    multiple_violations = tuple(
        item.student_id for item in student_outcomes_tuple if item.coverage_status == MathCoverageStatus.VIOLATED_MULTIPLE_MATH_AT_LEAST_ONE
    )
    fallback_failures = tuple(
        item.student_id for item in student_outcomes_tuple if item.coverage_status == MathCoverageStatus.VIOLATED_MANDATORY_FALLBACK_FAILED
    )
    direct_violations = tuple(sorted(single_violations + multiple_violations))
    current_violations = tuple(sorted(direct_violations + fallback_failures))
    fallback_source = next((rule.source_course_id for rule in math_fallback_rules if rule.enabled), None)
    fallback_target = next((rule.fallback_course_id for rule in math_fallback_rules if rule.enabled), None)
    return MathPolicyReport(
        math_course_ids=math_ids,
        no_math_primary_student_ids=no_math,
        single_math_primary_student_ids=single_math,
        multiple_math_primary_student_ids=multiple_math,
        coverage_satisfied_student_ids=satisfied,
        coverage_satisfied_by_primary_student_ids=satisfied_by_primary,
        coverage_satisfied_by_mandatory_fallback_student_ids=satisfied_by_fallback,
        coverage_pending_fallback_student_ids=pending,
        direct_violation_student_ids=direct_violations,
        fallback_failure_violation_student_ids=fallback_failures,
        fallback_failure_violating_request_keys=tuple(
            item.fallback_source_request_key
            for item in student_outcomes_tuple
            if item.coverage_status == MathCoverageStatus.VIOLATED_MANDATORY_FALLBACK_FAILED
            and item.fallback_source_request_key is not None
        ),
        current_math_coverage_violation_student_ids=current_violations,
        policy_currently_satisfied=not current_violations and not pending,
        single_math_required_student_ids=single_math,
        single_math_required_violation_student_ids=single_violations,
        single_math_required_violating_request_keys=tuple(
            item.math_primary_request_keys[0] for item in student_outcomes_tuple if item.student_id in set(single_violations)
        ),
        multiple_math_student_ids=multiple_math,
        multiple_math_at_least_one_violation_student_ids=multiple_violations,
        multiple_math_at_least_one_violating_request_keys=tuple(
            request_key
            for item in student_outcomes_tuple
            if item.student_id in set(multiple_violations)
            for request_key in item.math_primary_request_keys
        ),
        fallback_source_course_id=fallback_source,
        fallback_target_course_id=fallback_target,
        fallback_required_student_ids=tuple(
            item.student_id
            for item in student_outcomes_tuple
            if item.fallback_source_request_key is not None
        ),
        fallback_required_source_request_keys=tuple(
            item.fallback_source_request_key
            for item in student_outcomes_tuple
            if item.fallback_source_request_key is not None
        ),
        fallback_pending_student_ids=pending,
        fallback_assigned_student_ids=satisfied_by_fallback,
        fallback_failed_student_ids=fallback_failures,
        fallback_mapping=math_fallback_rules,
        student_outcomes=student_outcomes_tuple,
    )


def _status_from_fallback_outcome(
    fallback_outcome: MandatoryFallbackOutcome | None,
) -> tuple[MathCoverageStatus, MathPolicyViolationType]:
    if fallback_outcome is None:
        return MathCoverageStatus.PENDING_MANDATORY_FALLBACK, MathPolicyViolationType.NONE
    if fallback_outcome.status == MandatoryFallbackStatus.ASSIGNED:
        return MathCoverageStatus.SATISFIED_BY_MANDATORY_FALLBACK, MathPolicyViolationType.NONE
    if fallback_outcome.status in {
        MandatoryFallbackStatus.UNASSIGNED_NO_CANDIDATES,
        MandatoryFallbackStatus.UNASSIGNED_ALL_CANDIDATES_REJECTED,
    }:
        return MathCoverageStatus.VIOLATED_MANDATORY_FALLBACK_FAILED, MathPolicyViolationType.MANDATORY_FALLBACK_FAILED
    return MathCoverageStatus.PENDING_MANDATORY_FALLBACK, MathPolicyViolationType.NONE


def _fallback_outcome_for_requests(
    math_requests: tuple[LogicalRequest, ...],
    fallback_outcomes_by_source: dict[str, MandatoryFallbackOutcome],
) -> MandatoryFallbackOutcome | None:
    for request in math_requests:
        fallback_outcome = fallback_outcomes_by_source.get(request.request_key)
        if fallback_outcome is not None:
            return fallback_outcome
    return None


def _require_math_course(catalog_by_id: dict[str, pd.Series], course_id: str, column: str, line: int) -> None:
    if course_id not in catalog_by_id:
        raise MathFallbackConfigError(f"math_fallbacks.csv line {line}: {column} {course_id} is not in course_catalog.csv.")
    if text(catalog_by_id[course_id]["department"]) != MATH_DEPARTMENT:
        raise MathFallbackConfigError(f"math_fallbacks.csv line {line}: {column} {course_id} must have department=Mathematics.")
