from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass

from .assignment_models import AssignmentResult
from .baseline_models import (
    AlternateRequestStatus,
    BaselineInternalConsistencyError,
    BaselineResult,
    CandidateAttempt,
    HighDemandCandidateDemand,
    MandatoryFallbackOutcome,
    MandatoryFallbackStatus,
    PolicyReport,
    PrimaryRequestStatus,
    RequestOutcome,
    SectionRosterSummary,
    StudentOutcome,
)
from .input_models import CanonicalAllocationInput, LogicalRequest, SourceRequestRow
from .math_policy_models import MathFallbackRule
from .state import MANDATORY_FALLBACK_REQUEST_TYPE, AllocationState


ALGORITHM_NAME = "seeded_random_greedy"
HIGH_DEMAND_PRIMARY_THRESHOLD = 120


@dataclass(frozen=True)
class _MandatoryFallbackPlan:
    source_request: LogicalRequest
    fallback_request: LogicalRequest
    candidates: tuple[str, ...]


def run_seeded_random_baseline(
    allocation_input: CanonicalAllocationInput,
    seed: int,
    *,
    math_fallback_rules: tuple[MathFallbackRule, ...] = (),
    math_course_ids: tuple[str, ...] = (),
) -> BaselineResult:
    """Run the seeded random greedy baseline.

    The baseline intentionally makes no repair, displacement, protected-first,
    or high-demand-first moves. It reports policy violations after the greedy
    run rather than treating them as infeasibility.
    """

    rng = random.Random(seed)
    fallback_plans = _build_mandatory_fallback_plans(allocation_input, math_fallback_rules)
    state = AllocationState(
        allocation_input,
        supplemental_requests=tuple(plan.fallback_request for plan in fallback_plans),
        supplemental_candidate_index={
            plan.fallback_request.request_key: plan.candidates for plan in fallback_plans
        },
    )
    students = list(allocation_input.students)
    rng.shuffle(students)
    student_order = tuple(student.student_id for student in students)
    outcomes: list[RequestOutcome] = []

    for student in students:
        primary_requests = list(student.primary_requests)
        rng.shuffle(primary_requests)
        for request in primary_requests:
            outcomes.append(_try_primary_request(state, allocation_input, request, rng))

    primary_outcomes = tuple(outcomes)
    mandatory_fallback_outcomes = _run_mandatory_fallback_phase(
        state,
        students,
        fallback_plans,
        primary_outcomes,
        math_course_ids,
        rng,
    )

    for student in students:
        for request in sorted(student.alternate_requests, key=lambda item: item.request_rank or 0):
            outcomes.append(_try_alternate_request(state, allocation_input, request, rng))

    consistency_issues = state.validate_internal_consistency()
    if consistency_issues:
        raise BaselineInternalConsistencyError(consistency_issues)

    assignments = state.all_assignments()
    request_outcomes = tuple(sorted(outcomes, key=_outcome_sort_key))
    policy_report = _build_policy_report(allocation_input, request_outcomes)
    student_outcomes = _build_student_outcomes(allocation_input, state, request_outcomes, mandatory_fallback_outcomes, policy_report)
    section_summary = _build_section_roster_summary(allocation_input, state)
    return BaselineResult(
        algorithm_name=ALGORITHM_NAME,
        seed=int(seed),
        student_processing_order=student_order,
        assignments=assignments,
        mandatory_fallback_outcomes=tuple(sorted(mandatory_fallback_outcomes, key=_fallback_outcome_sort_key)),
        request_outcomes=request_outcomes,
        student_outcomes=student_outcomes,
        policy_report=policy_report,
        section_roster_summary=section_summary,
        consistency_issues=consistency_issues,
    )


def _build_mandatory_fallback_plans(
    allocation_input: CanonicalAllocationInput,
    math_fallback_rules: tuple[MathFallbackRule, ...],
) -> tuple[_MandatoryFallbackPlan, ...]:
    enabled_rules = {
        rule.source_course_id: rule
        for rule in math_fallback_rules
        if rule.enabled and rule.policy_type == "mandatory_fallback"
    }
    if not enabled_rules:
        return ()

    sections_by_candidate = _logical_section_ids_by_candidate_key(allocation_input)
    plans: list[_MandatoryFallbackPlan] = []
    for source_request in allocation_input.logical_requests:
        if source_request.request_type != "primary":
            continue
        rule = enabled_rules.get(source_request.candidate_key)
        if rule is None:
            continue
        if rule.fallback_course_id not in allocation_input.courses_by_id:
            raise ValueError(f"Mandatory fallback target course is missing from allocation input: {rule.fallback_course_id}")
        metadata = allocation_input.courses_by_id[rule.fallback_course_id]
        fallback_request_key = _mandatory_fallback_request_key(
            source_request.student_id,
            source_request.request_key,
            rule.fallback_course_id,
        )
        fallback_request = LogicalRequest(
            request_key=fallback_request_key,
            student_id=source_request.student_id,
            request_type=MANDATORY_FALLBACK_REQUEST_TYPE,
            candidate_key=rule.fallback_course_id,
            course_ids=(rule.fallback_course_id,),
            source_rows=(
                SourceRequestRow(
                    course_id=rule.fallback_course_id,
                    request_group=MANDATORY_FALLBACK_REQUEST_TYPE,
                    must_share_block_id=source_request.request_key,
                    request_rank=None,
                ),
            ),
            request_rank=None,
            period_units=metadata.period_units,
        )
        plans.append(
            _MandatoryFallbackPlan(
                source_request=source_request,
                fallback_request=fallback_request,
                candidates=sections_by_candidate.get(rule.fallback_course_id, ()),
            )
        )
    return tuple(sorted(plans, key=lambda item: item.source_request.request_key))


def _run_mandatory_fallback_phase(
    state: AllocationState,
    students,
    fallback_plans: tuple[_MandatoryFallbackPlan, ...],
    primary_outcomes: tuple[RequestOutcome, ...],
    math_course_ids: tuple[str, ...],
    rng: random.Random,
) -> tuple[MandatoryFallbackOutcome, ...]:
    if not fallback_plans:
        return ()

    primary_outcomes_by_key = {outcome.request_key: outcome for outcome in primary_outcomes}
    plans_by_student: dict[str, list[_MandatoryFallbackPlan]] = defaultdict(list)
    for plan in fallback_plans:
        plans_by_student[plan.source_request.student_id].append(plan)
    assigned_math_primary_by_student = _assigned_math_primary_by_student(primary_outcomes, set(math_course_ids))

    outcomes: list[MandatoryFallbackOutcome] = []
    for student in students:
        for plan in sorted(plans_by_student.get(student.student_id, ()), key=lambda item: item.source_request.request_key):
            source_outcome = primary_outcomes_by_key[plan.source_request.request_key]
            before = state.student_remaining_period_units(student.student_id)
            if source_outcome.status == PrimaryRequestStatus.ASSIGNED:
                outcomes.append(
                    _mandatory_fallback_outcome(
                        plan,
                        MandatoryFallbackStatus.NOT_REQUIRED_SOURCE_ASSIGNED,
                        None,
                        None,
                        (),
                        before,
                        before,
                    )
                )
                continue
            assigned_primary_math = assigned_math_primary_by_student.get(student.student_id, set())
            if assigned_primary_math - {plan.source_request.request_key}:
                outcomes.append(
                    _mandatory_fallback_outcome(
                        plan,
                        MandatoryFallbackStatus.NOT_REQUIRED_MATH_COVERAGE_ALREADY_SATISFIED,
                        None,
                        None,
                        (),
                        before,
                        before,
                    )
                )
                continue

            candidates = list(plan.candidates)
            rng.shuffle(candidates)
            if not candidates:
                outcomes.append(
                    _mandatory_fallback_outcome(
                        plan,
                        MandatoryFallbackStatus.UNASSIGNED_NO_CANDIDATES,
                        None,
                        None,
                        (),
                        before,
                        state.student_remaining_period_units(student.student_id),
                    )
                )
                continue

            attempts, assignment = _try_candidates(state, plan.fallback_request, tuple(candidates))
            status = (
                MandatoryFallbackStatus.ASSIGNED
                if assignment is not None
                else MandatoryFallbackStatus.UNASSIGNED_ALL_CANDIDATES_REJECTED
            )
            if assignment is not None:
                assigned_math_primary_by_student[student.student_id].add(plan.fallback_request.request_key)
            outcomes.append(
                _mandatory_fallback_outcome(
                    plan,
                    status,
                    assignment.assignment_key if assignment is not None else None,
                    assignment.linked_section_group_id if assignment is not None else None,
                    attempts,
                    before,
                    state.student_remaining_period_units(student.student_id),
                )
            )
    return tuple(outcomes)


def _assigned_math_primary_by_student(
    primary_outcomes: tuple[RequestOutcome, ...],
    math_course_ids: set[str],
) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for outcome in primary_outcomes:
        if outcome.status == PrimaryRequestStatus.ASSIGNED and outcome.candidate_key in math_course_ids:
            result[outcome.student_id].add(outcome.request_key)
    return result


def _mandatory_fallback_outcome(
    plan: _MandatoryFallbackPlan,
    status: MandatoryFallbackStatus,
    assignment_key: str | None,
    assigned_section_id: str | None,
    attempts: tuple[CandidateAttempt, ...],
    before: int,
    after: int,
) -> MandatoryFallbackOutcome:
    return MandatoryFallbackOutcome(
        student_id=plan.source_request.student_id,
        source_request_key=plan.source_request.request_key,
        source_course_id=plan.source_request.candidate_key,
        fallback_request_key=plan.fallback_request.request_key,
        fallback_course_id=plan.fallback_request.candidate_key,
        status=status,
        assignment_key=assignment_key,
        assigned_linked_section_group_id=assigned_section_id,
        candidate_attempts=attempts,
        remaining_units_before=before,
        remaining_units_after=after,
    )


def _mandatory_fallback_request_key(student_id: str, source_request_key: str, fallback_course_id: str) -> str:
    return f"{MANDATORY_FALLBACK_REQUEST_TYPE}:{student_id}:{source_request_key}:{fallback_course_id}"


def _logical_section_ids_by_candidate_key(
    allocation_input: CanonicalAllocationInput,
) -> dict[str, tuple[str, ...]]:
    by_candidate: dict[str, list[str]] = defaultdict(list)
    for section in allocation_input.logical_sections:
        by_candidate[section.logical_block_id].append(section.linked_section_group_id)
    return {key: tuple(sorted(value)) for key, value in sorted(by_candidate.items())}


def _try_primary_request(
    state: AllocationState,
    allocation_input: CanonicalAllocationInput,
    request: LogicalRequest,
    rng: random.Random,
) -> RequestOutcome:
    before = state.student_remaining_period_units(request.student_id)
    candidates = _shuffled_candidates(allocation_input, request.request_key, rng)
    if not candidates:
        return _request_outcome(
            request,
            PrimaryRequestStatus.UNMET_NO_CANDIDATES,
            None,
            None,
            (),
            before,
            state.student_remaining_period_units(request.student_id),
        )

    attempts, assignment = _try_candidates(state, request, candidates)
    status = PrimaryRequestStatus.ASSIGNED if assignment is not None else PrimaryRequestStatus.UNMET_ALL_CANDIDATES_REJECTED
    return _request_outcome(
        request,
        status,
        assignment.assignment_key if assignment is not None else None,
        assignment.linked_section_group_id if assignment is not None else None,
        attempts,
        before,
        state.student_remaining_period_units(request.student_id),
    )


def _try_alternate_request(
    state: AllocationState,
    allocation_input: CanonicalAllocationInput,
    request: LogicalRequest,
    rng: random.Random,
) -> RequestOutcome:
    before = state.student_remaining_period_units(request.student_id)
    if before == 0:
        return _request_outcome(request, AlternateRequestStatus.NOT_NEEDED, None, None, (), before, before)
    if request.period_units > before:
        return _request_outcome(
            request,
            AlternateRequestStatus.DOES_NOT_FIT_REMAINING_LOAD,
            None,
            None,
            (),
            before,
            before,
        )

    candidates = _shuffled_candidates(allocation_input, request.request_key, rng)
    if not candidates:
        return _request_outcome(
            request,
            AlternateRequestStatus.UNASSIGNED_NO_CANDIDATES,
            None,
            None,
            (),
            before,
            state.student_remaining_period_units(request.student_id),
        )

    attempts, assignment = _try_candidates(state, request, candidates)
    status = AlternateRequestStatus.ASSIGNED if assignment is not None else AlternateRequestStatus.UNASSIGNED_ALL_CANDIDATES_REJECTED
    return _request_outcome(
        request,
        status,
        assignment.assignment_key if assignment is not None else None,
        assignment.linked_section_group_id if assignment is not None else None,
        attempts,
        before,
        state.student_remaining_period_units(request.student_id),
    )


def _try_candidates(
    state: AllocationState,
    request: LogicalRequest,
    candidates: tuple[str, ...],
):
    attempts: list[CandidateAttempt] = []
    for index, section_id in enumerate(candidates, start=1):
        result = state.try_assign(request.student_id, request.request_key, section_id)
        attempts.append(_candidate_attempt(index, section_id, result))
        if result.allowed:
            return tuple(attempts), result.assignment
    return tuple(attempts), None


def _candidate_attempt(
    attempt_index: int,
    section_id: str,
    result: AssignmentResult,
) -> CandidateAttempt:
    return CandidateAttempt(
        attempt_index=attempt_index,
        linked_section_group_id=section_id,
        success=result.allowed,
        rejection_reasons=result.reasons,
        assignment_key=result.assignment.assignment_key if result.assignment is not None else None,
    )


def _request_outcome(
    request: LogicalRequest,
    status: PrimaryRequestStatus | AlternateRequestStatus,
    assignment_key: str | None,
    assigned_section_id: str | None,
    attempts: tuple[CandidateAttempt, ...],
    before: int,
    after: int,
) -> RequestOutcome:
    return RequestOutcome(
        request_key=request.request_key,
        student_id=request.student_id,
        request_type=request.request_type,
        alternate_rank=request.request_rank,
        candidate_key=request.candidate_key,
        period_units=request.period_units,
        status=status,
        assignment_key=assignment_key,
        assigned_linked_section_group_id=assigned_section_id,
        candidate_attempts=attempts,
        remaining_units_before=before,
        remaining_units_after=after,
    )


def _shuffled_candidates(
    allocation_input: CanonicalAllocationInput,
    request_key: str,
    rng: random.Random,
) -> tuple[str, ...]:
    candidates = list(allocation_input.candidate_index.get(request_key, ()))
    rng.shuffle(candidates)
    return tuple(candidates)


def _build_policy_report(
    allocation_input: CanonicalAllocationInput,
    request_outcomes: tuple[RequestOutcome, ...],
) -> PolicyReport:
    primary_demand = Counter(
        request.candidate_key
        for request in allocation_input.logical_requests
        if request.request_type == "primary"
    )
    high_demand_keys = tuple(sorted(key for key, count in primary_demand.items() if count > HIGH_DEMAND_PRIMARY_THRESHOLD))
    high_demand_set = set(high_demand_keys)
    primary_by_student: dict[str, list[RequestOutcome]] = defaultdict(list)
    high_violations: list[RequestOutcome] = []
    for outcome in request_outcomes:
        if outcome.request_type != "primary":
            continue
        primary_by_student[outcome.student_id].append(outcome)
        if outcome.candidate_key in high_demand_set and outcome.status != PrimaryRequestStatus.ASSIGNED:
            high_violations.append(outcome)

    ordinary_violations: list[str] = []
    protected_violations: list[str] = []
    for student in allocation_input.students:
        unmet_count = sum(1 for outcome in primary_by_student[student.student_id] if outcome.status != PrimaryRequestStatus.ASSIGNED)
        if student.priority_protected:
            if unmet_count > 0:
                protected_violations.append(student.student_id)
        elif unmet_count > 1:
            ordinary_violations.append(student.student_id)

    high_demand_demands = tuple(
        HighDemandCandidateDemand(key, int(primary_demand[key]))
        for key in high_demand_keys
    )
    high_violation_request_keys = tuple(sorted(outcome.request_key for outcome in high_violations))
    high_violation_students = tuple(sorted({outcome.student_id for outcome in high_violations}))
    high_violation_candidates = tuple(sorted({outcome.candidate_key for outcome in high_violations}))
    return PolicyReport(
        ordinary_policy_satisfied=not ordinary_violations,
        protected_policy_satisfied=not protected_violations,
        high_demand_policy_satisfied=not high_violations,
        all_reported_policies_satisfied=not ordinary_violations and not protected_violations and not high_violations,
        ordinary_violation_student_ids=tuple(sorted(ordinary_violations)),
        protected_violation_student_ids=tuple(sorted(protected_violations)),
        high_demand_candidate_keys=high_demand_keys,
        high_demand_demands=high_demand_demands,
        high_demand_violating_request_keys=high_violation_request_keys,
        high_demand_violating_student_ids=high_violation_students,
        high_demand_violating_candidate_keys=high_violation_candidates,
        high_demand_violation_count=len(high_violations),
    )


def _build_student_outcomes(
    allocation_input: CanonicalAllocationInput,
    state: AllocationState,
    request_outcomes: tuple[RequestOutcome, ...],
    mandatory_fallback_outcomes: tuple[MandatoryFallbackOutcome, ...],
    policy_report: PolicyReport,
) -> tuple[StudentOutcome, ...]:
    outcomes_by_student: dict[str, list[RequestOutcome]] = defaultdict(list)
    for outcome in request_outcomes:
        outcomes_by_student[outcome.student_id].append(outcome)
    fallback_assigned_by_student: dict[str, list[MandatoryFallbackOutcome]] = defaultdict(list)
    for outcome in mandatory_fallback_outcomes:
        if outcome.status == MandatoryFallbackStatus.ASSIGNED:
            fallback_assigned_by_student[outcome.student_id].append(outcome)
    high_violation_keys = set(policy_report.high_demand_violating_request_keys)
    ordinary_violation_ids = set(policy_report.ordinary_violation_student_ids)
    protected_violation_ids = set(policy_report.protected_violation_student_ids)
    result: list[StudentOutcome] = []
    for student in allocation_input.students:
        student_outcomes = outcomes_by_student[student.student_id]
        primary = [outcome for outcome in student_outcomes if outcome.request_type == "primary"]
        alternates = [outcome for outcome in student_outcomes if outcome.request_type == "alternate"]
        primary_unmet = [outcome for outcome in primary if outcome.status != PrimaryRequestStatus.ASSIGNED]
        alternate_assigned = [outcome for outcome in alternates if outcome.status == AlternateRequestStatus.ASSIGNED]
        fallback_assigned = sorted(fallback_assigned_by_student[student.student_id], key=lambda item: item.fallback_request_key)
        assignments = state.student_assignments(student.student_id)
        assigned_units = state.student_used_period_units(student.student_id)
        remaining_units = state.student_remaining_period_units(student.student_id)
        high_violating = tuple(sorted(outcome.request_key for outcome in primary if outcome.request_key in high_violation_keys))
        result.append(
            StudentOutcome(
                student_id=student.student_id,
                grade=student.grade,
                target_period_units=student.target_period_units,
                assigned_period_units=assigned_units,
                remaining_period_units=remaining_units,
                assignment_keys=tuple(assignment.assignment_key for assignment in assignments),
                primary_request_count=len(primary),
                primary_assigned_count=sum(1 for outcome in primary if outcome.status == PrimaryRequestStatus.ASSIGNED),
                primary_unmet_count=len(primary_unmet),
                primary_unmet_request_keys=tuple(sorted(outcome.request_key for outcome in primary_unmet)),
                primary_unmet_period_units=sum(outcome.period_units for outcome in primary_unmet),
                alternate_request_count=len(alternates),
                alternate_assigned_count=len(alternate_assigned),
                alternate_assigned_period_units=sum(outcome.period_units for outcome in alternate_assigned),
                mandatory_fallback_assigned_count=len(fallback_assigned),
                mandatory_fallback_assigned_period_units=sum(
                    assignment.period_units
                    for assignment in assignments
                    if assignment.assignment_key in {outcome.assignment_key for outcome in fallback_assigned}
                ),
                mandatory_fallback_assignment_keys=tuple(
                    outcome.assignment_key for outcome in fallback_assigned if outcome.assignment_key is not None
                ),
                fully_scheduled=assigned_units == student.target_period_units,
                priority_protected=student.priority_protected,
                ordinary_fairness_violation=student.student_id in ordinary_violation_ids,
                protected_fairness_violation=student.student_id in protected_violation_ids,
                high_demand_guarantee_violation_count=len(high_violating),
                high_demand_violating_request_keys=high_violating,
            )
        )
    return tuple(result)


def _build_section_roster_summary(
    allocation_input: CanonicalAllocationInput,
    state: AllocationState,
) -> tuple[SectionRosterSummary, ...]:
    rows: list[SectionRosterSummary] = []
    for section in allocation_input.logical_sections:
        section_id = section.linked_section_group_id
        rows.append(
            SectionRosterSummary(
                linked_section_group_id=section_id,
                assigned_count=state.section_assigned_count(section_id),
                capacity=section.capacity,
                remaining_capacity=state.section_remaining_capacity(section_id),
                student_ids=state.section_roster(section_id),
            )
        )
    return tuple(sorted(rows, key=lambda item: item.linked_section_group_id))


def _outcome_sort_key(outcome: RequestOutcome) -> tuple[str, int, int, str]:
    type_order = 0 if outcome.request_type == "primary" else 1
    rank = outcome.alternate_rank if outcome.alternate_rank is not None else 0
    return (outcome.student_id, type_order, rank, outcome.candidate_key)


def _fallback_outcome_sort_key(outcome: MandatoryFallbackOutcome) -> tuple[str, str, str]:
    return (outcome.student_id, outcome.source_request_key, outcome.fallback_course_id)
