from __future__ import annotations

from collections import defaultdict

from .baseline_models import (
    AlternateRequestStatus,
    BaselineResult,
    MandatoryFallbackOutcome,
    MandatoryFallbackStatus,
    PrimaryRequestStatus,
    RequestOutcome,
)
from .input_models import CanonicalAllocationInput, LogicalRequest
from .math_policy_models import MathFallbackRule
from .random_baseline import (
    _MandatoryFallbackPlan,
    _assigned_math_primary_by_student,
    _build_mandatory_fallback_plans,
    _finalize_baseline_result,
    _mandatory_fallback_outcome,
    _request_outcome,
    _try_candidates,
)
from .state import AllocationState


ALGORITHM_NAME = "first_come_first_served_greedy"


def run_fcfs_baseline(
    allocation_input: CanonicalAllocationInput,
    seed: int,
    *,
    math_fallback_rules: tuple[MathFallbackRule, ...] = (),
    math_course_ids: tuple[str, ...] = (),
) -> BaselineResult:
    """Run the first-come-first-served greedy baseline.

    Students are processed in the canonical input order (sorted student_id),
    each student's primary requests in their existing logical request order,
    and each request's candidate sections in their existing candidate_index
    order. FCFS makes no repair, displacement, shuffling, or priority-based
    reordering; it accepts the first hard-feasible candidate that
    ``AllocationState`` allows. ``seed`` is accepted only for a uniform
    baseline interface with the other algorithms; FCFS ordering does not
    depend on it.
    """

    fallback_plans = _build_mandatory_fallback_plans(allocation_input, math_fallback_rules)
    state = AllocationState(
        allocation_input,
        supplemental_requests=tuple(plan.fallback_request for plan in fallback_plans),
        supplemental_candidate_index={
            plan.fallback_request.request_key: plan.candidates for plan in fallback_plans
        },
    )
    students = list(allocation_input.students)
    student_order = tuple(student.student_id for student in students)
    outcomes: list[RequestOutcome] = []

    for student in students:
        for request in student.primary_requests:
            outcomes.append(_try_primary_request_fcfs(state, allocation_input, request))

    primary_outcomes = tuple(outcomes)
    mandatory_fallback_outcomes = _run_fcfs_mandatory_fallback_phase(
        state,
        students,
        fallback_plans,
        primary_outcomes,
        math_course_ids,
    )

    for student in students:
        for request in sorted(student.alternate_requests, key=lambda item: item.request_rank or 0):
            outcomes.append(_try_alternate_request_fcfs(state, allocation_input, request))

    return _finalize_baseline_result(
        ALGORITHM_NAME,
        allocation_input,
        seed,
        student_order,
        state,
        tuple(outcomes),
        mandatory_fallback_outcomes,
    )


def _run_fcfs_mandatory_fallback_phase(
    state: AllocationState,
    students,
    fallback_plans: tuple[_MandatoryFallbackPlan, ...],
    primary_outcomes: tuple[RequestOutcome, ...],
    math_course_ids: tuple[str, ...],
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

            candidates = plan.candidates
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

            attempts, assignment = _try_candidates(state, plan.fallback_request, candidates)
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


def _try_primary_request_fcfs(
    state: AllocationState,
    allocation_input: CanonicalAllocationInput,
    request: LogicalRequest,
) -> RequestOutcome:
    before = state.student_remaining_period_units(request.student_id)
    candidates = allocation_input.candidate_index.get(request.request_key, ())
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


def _try_alternate_request_fcfs(
    state: AllocationState,
    allocation_input: CanonicalAllocationInput,
    request: LogicalRequest,
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

    candidates = allocation_input.candidate_index.get(request.request_key, ())
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
