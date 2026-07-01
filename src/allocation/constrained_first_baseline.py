from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from typing import Mapping

from .baseline_models import (
    AlternateRequestStatus,
    BaselineResult,
    CandidateAttempt,
    MandatoryFallbackOutcome,
    MandatoryFallbackStatus,
    PrimaryRequestStatus,
    RequestOutcome,
    StudentDifficultyProfile,
)
from .input_models import CanonicalAllocationInput, CanonicalStudent, LogicalRequest
from .math_policy_models import MathFallbackRule
from .random_baseline import (
    HIGH_DEMAND_PRIMARY_THRESHOLD,
    _MandatoryFallbackPlan,
    _assigned_math_primary_by_student,
    _build_mandatory_fallback_plans,
    _fallback_outcome_sort_key,
    _finalize_baseline_result,
    _mandatory_fallback_outcome,
    _request_outcome,
    _try_candidates,
)
from .state import AllocationState


ALGORITHM_NAME = "constrained_first_greedy"


@dataclass(frozen=True)
class _OrderingContext:
    high_demand_candidate_keys: frozenset[str]
    math_course_ids: frozenset[str]
    fallback_source_course_ids: frozenset[str]
    period_pressure: Mapping[str, int]
    student_tie_breaks: Mapping[str, float]
    request_tie_breaks: Mapping[str, float]
    candidate_tie_breaks: Mapping[tuple[str, str], float]


def run_constrained_first_baseline(
    allocation_input: CanonicalAllocationInput,
    seed: int,
    *,
    math_fallback_rules: tuple[MathFallbackRule, ...] = (),
    math_course_ids: tuple[str, ...] = (),
) -> BaselineResult:
    """Run the constrained-first greedy baseline without repair or backtracking."""

    fallback_plans = _build_mandatory_fallback_plans(allocation_input, math_fallback_rules)
    context = build_ordering_context(allocation_input, seed, math_fallback_rules, math_course_ids, fallback_plans)
    profiles = build_student_difficulty_profiles(allocation_input, seed, math_fallback_rules, math_course_ids, context)
    students = tuple(allocation_input.students_by_id[profile.student_id] for profile in profiles)
    student_order = tuple(student.student_id for student in students)
    state = AllocationState(
        allocation_input,
        supplemental_requests=tuple(plan.fallback_request for plan in fallback_plans),
        supplemental_candidate_index={plan.fallback_request.request_key: plan.candidates for plan in fallback_plans},
    )

    outcomes: list[RequestOutcome] = []
    for student in students:
        remaining = list(student.primary_requests)
        while remaining:
            request = min(
                remaining,
                key=lambda item: primary_request_priority(state, allocation_input, item, remaining, context),
            )
            candidates = _ordered_candidate_sections(state, allocation_input, request, remaining, context)
            outcomes.append(_try_primary_request_ordered(state, request, candidates))
            remaining.remove(request)

    primary_outcomes = tuple(outcomes)
    mandatory_fallback_outcomes = _run_constrained_mandatory_fallback_phase(
        state,
        allocation_input,
        students,
        fallback_plans,
        primary_outcomes,
        math_course_ids,
        context,
    )

    for student in students:
        for request in sorted(student.alternate_requests, key=lambda item: item.request_rank or 0):
            candidates = _ordered_candidate_sections(state, allocation_input, request, (), context)
            outcomes.append(_try_alternate_request_ordered(state, request, candidates))

    return _finalize_baseline_result(
        ALGORITHM_NAME,
        allocation_input,
        seed,
        student_order,
        state,
        tuple(outcomes),
        mandatory_fallback_outcomes,
        profiles,
    )


def build_student_difficulty_profiles(
    allocation_input: CanonicalAllocationInput,
    seed: int,
    math_fallback_rules: tuple[MathFallbackRule, ...],
    math_course_ids: tuple[str, ...],
    context: _OrderingContext | None = None,
) -> tuple[StudentDifficultyProfile, ...]:
    context = context or build_ordering_context(allocation_input, seed, math_fallback_rules, math_course_ids, ())
    profiles = tuple(_student_difficulty_profile(allocation_input, student, context) for student in allocation_input.students)
    ordered = sorted(profiles, key=_student_profile_sort_key)
    return tuple(replace(profile, ordering_rank=index) for index, profile in enumerate(ordered, start=1))


def primary_request_priority(
    state: AllocationState,
    allocation_input: CanonicalAllocationInput,
    request: LogicalRequest,
    remaining_requests: list[LogicalRequest] | tuple[LogicalRequest, ...],
    context: _OrderingContext,
) -> tuple:
    feasible_count = _feasible_candidate_count(state, allocation_input, request)
    policy_rank = _request_policy_rank(request, remaining_requests, context)
    static_candidate_count = len(allocation_input.candidate_index.get(request.request_key, ()))
    return (
        feasible_count,
        policy_rank,
        static_candidate_count,
        -request.period_units,
        context.request_tie_breaks.get(request.request_key, 0.0),
        request.request_key,
    )


def candidate_section_priority(
    state: AllocationState,
    allocation_input: CanonicalAllocationInput,
    request: LogicalRequest,
    section_id: str,
    remaining_requests: list[LogicalRequest] | tuple[LogicalRequest, ...],
    context: _OrderingContext,
) -> tuple:
    section = allocation_input.logical_sections_by_id[section_id]
    check = state.check_assignment(request.student_id, request.request_key, section_id)
    feasible_rank = 0 if check.allowed else 1
    remaining_capacity = state.section_remaining_capacity(section_id)
    capacity_ratio = remaining_capacity / section.capacity
    pressure = sum(context.period_pressure.get(period, 0) for period in section.occupied_periods)
    future_options = _future_options_after_candidate(state, allocation_input, request, section_id, remaining_requests)
    return (
        feasible_rank,
        -capacity_ratio,
        pressure,
        -future_options,
        -remaining_capacity,
        context.candidate_tie_breaks.get((request.request_key, section_id), 0.0),
        section_id,
    )


def build_ordering_context(
    allocation_input: CanonicalAllocationInput,
    seed: int,
    math_fallback_rules: tuple[MathFallbackRule, ...],
    math_course_ids: tuple[str, ...],
    fallback_plans: tuple[_MandatoryFallbackPlan, ...],
) -> _OrderingContext:
    primary_demand = Counter(
        request.candidate_key for request in allocation_input.logical_requests if request.request_type == "primary"
    )
    high_demand_keys = frozenset(
        key for key, count in primary_demand.items() if count > HIGH_DEMAND_PRIMARY_THRESHOLD
    )
    fallback_sources = frozenset(
        rule.source_course_id for rule in math_fallback_rules if rule.enabled and rule.policy_type == "mandatory_fallback"
    )
    rng = random.Random(seed)
    student_ties = {
        student.student_id: rng.random()
        for student in sorted(allocation_input.students, key=lambda item: item.student_id)
    }
    request_keys = sorted(request.request_key for request in allocation_input.logical_requests)
    request_ties = {request_key: rng.random() for request_key in request_keys}
    candidate_pairs = {
        (request.request_key, section_id)
        for request in allocation_input.logical_requests
        for section_id in allocation_input.candidate_index.get(request.request_key, ())
    }
    candidate_pairs.update(
        (plan.fallback_request.request_key, section_id)
        for plan in fallback_plans
        for section_id in plan.candidates
    )
    candidate_ties = {pair: rng.random() for pair in sorted(candidate_pairs)}
    return _OrderingContext(
        high_demand_candidate_keys=high_demand_keys,
        math_course_ids=frozenset(math_course_ids),
        fallback_source_course_ids=fallback_sources,
        period_pressure=_period_pressure(allocation_input),
        student_tie_breaks=student_ties,
        request_tie_breaks=request_ties,
        candidate_tie_breaks=candidate_ties,
    )


def _student_difficulty_profile(
    allocation_input: CanonicalAllocationInput,
    student: CanonicalStudent,
    context: _OrderingContext,
) -> StudentDifficultyProfile:
    primary = student.primary_requests
    candidate_counts = [len(allocation_input.candidate_index.get(request.request_key, ())) for request in primary]
    math_primary = [request for request in primary if request.candidate_key in context.math_course_ids]
    math_risk = 0
    if len(math_primary) == 1:
        math_risk += 2
    if any(request.candidate_key in context.fallback_source_course_ids for request in math_primary):
        math_risk += 2
    if len(math_primary) > 1 and any(len(allocation_input.candidate_index.get(request.request_key, ())) <= 1 for request in math_primary):
        math_risk += 1
    total_flexibility = sum(
        len(allocation_input.candidate_index.get(request.request_key, ())) / max(request.period_units, 1)
        for request in primary
    )
    return StudentDifficultyProfile(
        student_id=student.student_id,
        protected=student.priority_protected,
        math_risk_score=math_risk,
        high_demand_primary_count=sum(1 for request in primary if request.candidate_key in context.high_demand_candidate_keys),
        scarce_primary_count_le_1=sum(1 for count in candidate_counts if count <= 1),
        scarce_primary_count_le_2=sum(1 for count in candidate_counts if count <= 2),
        total_candidate_flexibility=float(total_flexibility),
        average_candidate_flexibility=float(total_flexibility / len(primary)) if primary else 0.0,
        double_period_primary_count=sum(1 for request in primary if request.period_units > 1),
        target_period_units=student.target_period_units,
        seeded_tie_break=context.student_tie_breaks[student.student_id],
        ordering_rank=0,
    )


def _student_profile_sort_key(profile: StudentDifficultyProfile) -> tuple:
    return (
        0 if profile.protected else 1,
        -profile.math_risk_score,
        -profile.high_demand_primary_count,
        -profile.scarce_primary_count_le_1,
        -profile.scarce_primary_count_le_2,
        profile.total_candidate_flexibility,
        -profile.double_period_primary_count,
        -profile.target_period_units,
        profile.seeded_tie_break,
        profile.student_id,
    )


def _period_pressure(allocation_input: CanonicalAllocationInput) -> Mapping[str, int]:
    pressure: Counter[str] = Counter()
    for request in allocation_input.logical_requests:
        if request.request_type != "primary":
            continue
        for section_id in allocation_input.candidate_index.get(request.request_key, ()):
            section = allocation_input.logical_sections_by_id[section_id]
            for period in section.occupied_periods:
                pressure[period] += 1
    return dict(sorted(pressure.items()))


def _request_policy_rank(
    request: LogicalRequest,
    remaining_requests: list[LogicalRequest] | tuple[LogicalRequest, ...],
    context: _OrderingContext,
) -> int:
    remaining_math = [item for item in remaining_requests if item.candidate_key in context.math_course_ids]
    if request.candidate_key in context.math_course_ids and (
        len(remaining_math) == 1 or request.candidate_key in context.fallback_source_course_ids
    ):
        return 0
    if request.candidate_key in context.high_demand_candidate_keys:
        return 1
    if request.period_units > 1:
        return 2
    return 3


def _feasible_candidate_count(
    state: AllocationState,
    allocation_input: CanonicalAllocationInput,
    request: LogicalRequest,
) -> int:
    return sum(
        1
        for section_id in allocation_input.candidate_index.get(request.request_key, ())
        if state.check_assignment(request.student_id, request.request_key, section_id).allowed
    )


def _ordered_candidate_sections(
    state: AllocationState,
    allocation_input: CanonicalAllocationInput,
    request: LogicalRequest,
    remaining_requests: list[LogicalRequest] | tuple[LogicalRequest, ...],
    context: _OrderingContext,
) -> tuple[str, ...]:
    candidates = allocation_input.candidate_index.get(request.request_key, ())
    return tuple(
        sorted(
            candidates,
            key=lambda section_id: candidate_section_priority(
                state,
                allocation_input,
                request,
                section_id,
                remaining_requests,
                context,
            ),
        )
    )


def _future_options_after_candidate(
    state: AllocationState,
    allocation_input: CanonicalAllocationInput,
    request: LogicalRequest,
    section_id: str,
    remaining_requests: list[LogicalRequest] | tuple[LogicalRequest, ...],
) -> int:
    section = allocation_input.logical_sections_by_id[section_id]
    blocked_periods = set(section.occupied_periods)
    remaining_units = state.student_remaining_period_units(request.student_id) - section.period_units
    if remaining_units < 0:
        return 0
    options = 0
    for other in remaining_requests:
        if other.request_key == request.request_key or other.period_units > remaining_units:
            continue
        for other_section_id in allocation_input.candidate_index.get(other.request_key, ()):
            other_section = allocation_input.logical_sections_by_id[other_section_id]
            if blocked_periods & set(other_section.occupied_periods):
                continue
            if state.check_assignment(other.student_id, other.request_key, other_section_id).allowed:
                options += 1
    return options


def _try_primary_request_ordered(
    state: AllocationState,
    request: LogicalRequest,
    candidates: tuple[str, ...],
) -> RequestOutcome:
    before = state.student_remaining_period_units(request.student_id)
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


def _try_alternate_request_ordered(
    state: AllocationState,
    request: LogicalRequest,
    candidates: tuple[str, ...],
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


def _run_constrained_mandatory_fallback_phase(
    state: AllocationState,
    allocation_input: CanonicalAllocationInput,
    students: tuple[CanonicalStudent, ...],
    fallback_plans: tuple[_MandatoryFallbackPlan, ...],
    primary_outcomes: tuple[RequestOutcome, ...],
    math_course_ids: tuple[str, ...],
    context: _OrderingContext,
) -> tuple[MandatoryFallbackOutcome, ...]:
    if not fallback_plans:
        return ()

    primary_outcomes_by_key = {outcome.request_key: outcome for outcome in primary_outcomes}
    plans_by_student: dict[str, list[_MandatoryFallbackPlan]] = defaultdict(list)
    for plan in fallback_plans:
        plans_by_student[plan.source_request.student_id].append(plan)
    assigned_math_by_student = _assigned_math_primary_by_student(primary_outcomes, set(math_course_ids))
    math_coverage_satisfied = {student_id for student_id, request_keys in assigned_math_by_student.items() if request_keys}

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
            if student.student_id in math_coverage_satisfied:
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
            candidates = tuple(
                sorted(
                    plan.candidates,
                    key=lambda section_id: candidate_section_priority(
                        state,
                        allocation_input,
                        plan.fallback_request,
                        section_id,
                        (),
                        context,
                    ),
                )
            )
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
                math_coverage_satisfied.add(student.student_id)
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
    return tuple(sorted(outcomes, key=_fallback_outcome_sort_key))
