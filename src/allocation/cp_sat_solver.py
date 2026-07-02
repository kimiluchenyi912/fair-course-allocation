from __future__ import annotations

import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Iterable

from ortools.sat.python import cp_model

from .baseline_models import (
    AlternateRequestStatus,
    CandidateAttempt,
    MandatoryFallbackOutcome,
    MandatoryFallbackStatus,
    PrimaryRequestStatus,
    RequestOutcome,
)
from .cp_sat_models import (
    CpSatAllocationResult,
    CpSatModelStats,
    CpSatObjectiveValues,
    CpSatSolveStatus,
    CpSatStageDiagnostic,
    CpSatStageName,
)
from .input_models import CanonicalAllocationInput, LogicalRequest, SourceRequestRow
from .math_policy import evaluate_math_policy
from .math_policy_models import MathFallbackRule
from .random_baseline import (
    HIGH_DEMAND_PRIMARY_THRESHOLD,
    _build_mandatory_fallback_plans,
    _finalize_baseline_result,
)
from .state import MANDATORY_FALLBACK_REQUEST_TYPE, AllocationState


ALGORITHM_NAME = "fair_cp_sat_solver_v1"


@dataclass(frozen=True)
class _FallbackPlan:
    source_request: LogicalRequest
    fallback_request: LogicalRequest
    candidates: tuple[str, ...]


@dataclass(frozen=True)
class _VariableKey:
    request_key: str
    section_id: str


@dataclass
class _ModelBuild:
    model: cp_model.CpModel
    assignment_vars: dict[_VariableKey, cp_model.IntVar]
    assigned_vars: dict[str, cp_model.IntVar]
    math_violation_vars: dict[str, cp_model.IntVar]
    fully_scheduled_vars: dict[str, cp_model.IntVar]
    fallback_plans: tuple[_FallbackPlan, ...]
    math_course_ids: tuple[str, ...]
    requests_by_key: dict[str, LogicalRequest]
    candidate_index: dict[str, tuple[str, ...]]
    stage_exprs: dict[CpSatStageName, cp_model.LinearExpr]
    primary_penalty_dominance_base: int
    build_time_seconds: float


def run_fair_cp_sat_solver(
    allocation_input: CanonicalAllocationInput,
    *,
    seed: int,
    math_fallback_rules: tuple[MathFallbackRule, ...] = (),
    math_course_ids: tuple[str, ...] = (),
    max_time_seconds_per_stage: float = 30.0,
    num_search_workers: int = 1,
    log_search_progress: bool = False,
) -> CpSatAllocationResult:
    """Solve the fixed-section allocation problem with CP-SAT.

    This solver never changes sections, capacities, periods, requests, or
    eligibility. It models fairness policies as hard constraints and then uses
    explicit lexicographic stages for soft goals.
    """

    started = time.perf_counter()
    build = _build_model(allocation_input, math_fallback_rules, tuple(sorted(math_course_ids)), seed)
    diagnostics: list[CpSatStageDiagnostic] = []
    incumbent: cp_model.CpSolver | None = None
    lexicographic_optimum = True

    stages = (
        (CpSatStageName.MATH_COVERAGE, "min"),
        (CpSatStageName.PRIMARY_SATISFACTION, "min"),
        (CpSatStageName.ALTERNATE_RANK_1, "max"),
        (CpSatStageName.ALTERNATE_RANK_2, "max"),
        (CpSatStageName.ALTERNATE_RANK_3, "max"),
        (CpSatStageName.FULLY_SCHEDULED, "max"),
        (CpSatStageName.REMAINING_PERIOD_UNITS, "min"),
        (CpSatStageName.SEEDED_TIE_BREAK, "min"),
    )
    status = CpSatSolveStatus.UNKNOWN
    fixed_objective_values: dict[CpSatStageName, int] = {}

    for stage_name, sense in stages:
        expr = build.stage_exprs[stage_name]
        if sense == "min":
            build.model.Minimize(expr)
        else:
            build.model.Maximize(expr)
        solver = _new_solver(max_time_seconds_per_stage, num_search_workers, log_search_progress, seed)
        raw_status = solver.Solve(build.model)
        status = _solve_status(raw_status)
        diagnostic = _stage_diagnostic(stage_name, status, solver)
        diagnostics.append(diagnostic)

        if status not in {CpSatSolveStatus.OPTIMAL, CpSatSolveStatus.FEASIBLE}:
            if incumbent is not None:
                lexicographic_optimum = False
                break
            return _empty_result(
                seed,
                status,
                tuple(diagnostics),
                build,
                time.perf_counter() - started,
                lexicographic_optimum and status == CpSatSolveStatus.OPTIMAL,
            )

        incumbent = solver
        value = int(round(solver.ObjectiveValue()))
        fixed_objective_values[stage_name] = value
        if status != CpSatSolveStatus.OPTIMAL:
            lexicographic_optimum = False
            break
        build.model.Add(expr == value)

    assert incumbent is not None
    selected = _selected_assignments(build, incumbent)
    try:
        state = _replay_solution(allocation_input, build, selected)
    except RuntimeError:
        return _empty_result(
            seed,
            CpSatSolveStatus.MODEL_INVALID,
            tuple(diagnostics),
            build,
            time.perf_counter() - started,
            False,
        )

    request_outcomes = _build_request_outcomes(allocation_input, build, incumbent, state)
    fallback_outcomes = _build_fallback_outcomes(build, incumbent, state)
    baseline_result = _finalize_baseline_result(
        ALGORITHM_NAME,
        allocation_input,
        seed,
        (),
        state,
        request_outcomes,
        fallback_outcomes,
    )
    math_report = evaluate_math_policy(allocation_input, baseline_result, tuple(sorted(math_course_ids)), math_fallback_rules)
    final_status = CpSatSolveStatus.OPTIMAL if lexicographic_optimum else CpSatSolveStatus.FEASIBLE
    model_proto = build.model.Proto()
    return CpSatAllocationResult(
        algorithm_name=ALGORITHM_NAME,
        seed=int(seed),
        solve_status=final_status,
        lexicographic_optimality_proven=lexicographic_optimum,
        stage_diagnostics=tuple(diagnostics),
        objective_values=_objective_values(build, incumbent, fixed_objective_values),
        model_stats=CpSatModelStats(
            total_variables=len(model_proto.variables),
            total_constraints=len(model_proto.constraints),
            build_time_seconds=round(build.build_time_seconds, 6),
            solve_time_seconds=round(time.perf_counter() - started - build.build_time_seconds, 6),
        ),
        assignments=baseline_result.assignments,
        mandatory_fallback_outcomes=baseline_result.mandatory_fallback_outcomes,
        request_outcomes=baseline_result.request_outcomes,
        student_outcomes=baseline_result.student_outcomes,
        policy_report=baseline_result.policy_report,
        math_policy_report=math_report,
        section_roster_summary=baseline_result.section_roster_summary,
        consistency_issues=baseline_result.consistency_issues,
    )


def _build_model(
    allocation_input: CanonicalAllocationInput,
    math_fallback_rules: tuple[MathFallbackRule, ...],
    math_course_ids: tuple[str, ...],
    seed: int,
) -> _ModelBuild:
    started = time.perf_counter()
    model = cp_model.CpModel()
    fallback_plans = _convert_fallback_plans(_build_mandatory_fallback_plans(allocation_input, math_fallback_rules))
    requests_by_key = dict(allocation_input.requests_by_key)
    candidate_index = {key: tuple(value) for key, value in allocation_input.candidate_index.items()}
    for plan in fallback_plans:
        requests_by_key[plan.fallback_request.request_key] = plan.fallback_request
        candidate_index[plan.fallback_request.request_key] = plan.candidates

    assignment_vars: dict[_VariableKey, cp_model.IntVar] = {}
    assigned_vars: dict[str, cp_model.IntVar] = {}
    for request_key in sorted(requests_by_key):
        request = requests_by_key[request_key]
        candidates = tuple(section_id for section_id in candidate_index.get(request_key, ()) if section_id in allocation_input.logical_sections_by_id)
        candidate_vars: list[cp_model.IntVar] = []
        for section_id in candidates:
            var = model.NewBoolVar(f"x__{_safe_name(request_key)}__{_safe_name(section_id)}")
            assignment_vars[_VariableKey(request_key, section_id)] = var
            candidate_vars.append(var)
        assigned = model.NewBoolVar(f"assigned__{_safe_name(request_key)}")
        assigned_vars[request_key] = assigned
        model.Add(sum(candidate_vars) == assigned)
        model.Add(sum(candidate_vars) <= 1)
        if request.period_units != allocation_input.courses_by_id[request.course_ids[0]].period_units:
            # Canonicalization should already prevent this for source requests;
            # fallback requests are built from the same metadata.
            model.Add(assigned == 0)

    _add_section_capacity_constraints(model, allocation_input, assignment_vars)
    _add_student_period_constraints(model, allocation_input, requests_by_key, assignment_vars)
    _add_student_target_constraints(model, allocation_input, requests_by_key, assignment_vars)
    _add_duplicate_identity_constraints(model, allocation_input, requests_by_key, assigned_vars)
    _add_fallback_constraints(model, fallback_plans, assigned_vars, math_course_ids)
    math_violation_vars = _add_math_coverage_constraints(
        model,
        allocation_input,
        fallback_plans,
        assigned_vars,
        math_course_ids,
    )
    _add_fairness_hard_constraints(model, allocation_input, assigned_vars)
    fully_scheduled_vars, remaining_exprs = _add_schedule_completion_vars(
        model,
        allocation_input,
        requests_by_key,
        assignment_vars,
    )
    stage_exprs, primary_base = _stage_expressions(
        allocation_input,
        requests_by_key,
        assigned_vars,
        math_violation_vars,
        fully_scheduled_vars,
        remaining_exprs,
        assignment_vars,
        seed,
    )
    _add_constrained_first_hint(
        model,
        allocation_input,
        assignment_vars,
        math_fallback_rules,
        math_course_ids,
        seed,
    )
    return _ModelBuild(
        model=model,
        assignment_vars=assignment_vars,
        assigned_vars=assigned_vars,
        math_violation_vars=math_violation_vars,
        fully_scheduled_vars=fully_scheduled_vars,
        fallback_plans=fallback_plans,
        math_course_ids=math_course_ids,
        requests_by_key=requests_by_key,
        candidate_index=candidate_index,
        stage_exprs=stage_exprs,
        primary_penalty_dominance_base=primary_base,
        build_time_seconds=time.perf_counter() - started,
    )


def _convert_fallback_plans(plans) -> tuple[_FallbackPlan, ...]:
    return tuple(_FallbackPlan(plan.source_request, plan.fallback_request, plan.candidates) for plan in plans)


def _add_section_capacity_constraints(
    model: cp_model.CpModel,
    allocation_input: CanonicalAllocationInput,
    assignment_vars: dict[_VariableKey, cp_model.IntVar],
) -> None:
    by_section: dict[str, list[cp_model.IntVar]] = defaultdict(list)
    for key, var in assignment_vars.items():
        by_section[key.section_id].append(var)
    for section in allocation_input.logical_sections:
        model.Add(sum(by_section.get(section.linked_section_group_id, ())) <= section.capacity)


def _add_student_period_constraints(
    model: cp_model.CpModel,
    allocation_input: CanonicalAllocationInput,
    requests_by_key: dict[str, LogicalRequest],
    assignment_vars: dict[_VariableKey, cp_model.IntVar],
) -> None:
    by_student_period: dict[tuple[str, str], list[cp_model.IntVar]] = defaultdict(list)
    for key, var in assignment_vars.items():
        request = requests_by_key[key.request_key]
        section = allocation_input.logical_sections_by_id[key.section_id]
        for period in section.occupied_periods:
            by_student_period[(request.student_id, period)].append(var)
    for student in allocation_input.students:
        for period in (f"P{index}" for index in range(1, 8)):
            model.Add(sum(by_student_period.get((student.student_id, period), ())) <= 1)


def _add_student_target_constraints(
    model: cp_model.CpModel,
    allocation_input: CanonicalAllocationInput,
    requests_by_key: dict[str, LogicalRequest],
    assignment_vars: dict[_VariableKey, cp_model.IntVar],
) -> None:
    by_student: dict[str, list[cp_model.LinearExpr]] = defaultdict(list)
    for key, var in assignment_vars.items():
        request = requests_by_key[key.request_key]
        by_student[request.student_id].append(var * request.period_units)
    for student in allocation_input.students:
        model.Add(sum(by_student.get(student.student_id, ())) <= student.target_period_units)


def _add_duplicate_identity_constraints(
    model: cp_model.CpModel,
    allocation_input: CanonicalAllocationInput,
    requests_by_key: dict[str, LogicalRequest],
    assigned_vars: dict[str, cp_model.IntVar],
) -> None:
    by_student_identity: dict[tuple[str, str], list[cp_model.IntVar]] = defaultdict(list)
    for request in requests_by_key.values():
        by_student_identity[(request.student_id, request.candidate_key)].append(assigned_vars[request.request_key])
    for student in allocation_input.students:
        identities = {
            identity
            for sid, identity in by_student_identity
            if sid == student.student_id
        }
        for identity in identities:
            model.Add(sum(by_student_identity[(student.student_id, identity)]) <= 1)


def _add_fallback_constraints(
    model: cp_model.CpModel,
    fallback_plans: tuple[_FallbackPlan, ...],
    assigned_vars: dict[str, cp_model.IntVar],
    math_course_ids: tuple[str, ...],
) -> None:
    math_set = set(math_course_ids)
    primary_math_by_student: dict[str, list[str]] = defaultdict(list)
    for request_key in assigned_vars:
        if not request_key.startswith("primary:"):
            continue
        _, student_id, candidate_key = request_key.split(":", 2)
        if candidate_key in math_set:
            primary_math_by_student[student_id].append(request_key)
    for plan in fallback_plans:
        fallback = assigned_vars[plan.fallback_request.request_key]
        source = assigned_vars[plan.source_request.request_key]
        model.Add(fallback + source <= 1)
        for other_math_request_key in primary_math_by_student.get(plan.source_request.student_id, ()):
            if other_math_request_key != plan.source_request.request_key:
                model.Add(fallback + assigned_vars[other_math_request_key] <= 1)


def _add_math_coverage_constraints(
    model: cp_model.CpModel,
    allocation_input: CanonicalAllocationInput,
    fallback_plans: tuple[_FallbackPlan, ...],
    assigned_vars: dict[str, cp_model.IntVar],
    math_course_ids: tuple[str, ...],
) -> dict[str, cp_model.IntVar]:
    math_set = set(math_course_ids)
    fallback_by_student: dict[str, list[cp_model.IntVar]] = defaultdict(list)
    for plan in fallback_plans:
        fallback_by_student[plan.fallback_request.student_id].append(assigned_vars[plan.fallback_request.request_key])
    violations: dict[str, cp_model.IntVar] = {}
    for student in allocation_input.students:
        math_primary = [
            assigned_vars[request.request_key]
            for request in student.primary_requests
            if request.candidate_key in math_set
        ]
        if not math_primary:
            continue
        coverage_terms = math_primary + fallback_by_student.get(student.student_id, [])
        coverage = model.NewBoolVar(f"math_coverage__{_safe_name(student.student_id)}")
        violation = model.NewBoolVar(f"math_violation__{_safe_name(student.student_id)}")
        model.Add(sum(coverage_terms) >= coverage)
        for term in coverage_terms:
            model.Add(coverage >= term)
        model.Add(coverage + violation == 1)
        violations[student.student_id] = violation
    return violations


def _add_fairness_hard_constraints(
    model: cp_model.CpModel,
    allocation_input: CanonicalAllocationInput,
    assigned_vars: dict[str, cp_model.IntVar],
) -> None:
    demand = Counter(request.candidate_key for request in allocation_input.logical_requests if request.request_type == "primary")
    high_demand = {key for key, count in demand.items() if count > HIGH_DEMAND_PRIMARY_THRESHOLD}
    for student in allocation_input.students:
        primary_assigned = [assigned_vars[request.request_key] for request in student.primary_requests]
        primary_count = len(student.primary_requests)
        if student.priority_protected:
            model.Add(sum(primary_assigned) == primary_count)
        else:
            model.Add(primary_count - sum(primary_assigned) <= 1)
        for request in student.primary_requests:
            if request.candidate_key in high_demand:
                model.Add(assigned_vars[request.request_key] == 1)


def _add_schedule_completion_vars(
    model: cp_model.CpModel,
    allocation_input: CanonicalAllocationInput,
    requests_by_key: dict[str, LogicalRequest],
    assignment_vars: dict[_VariableKey, cp_model.IntVar],
) -> tuple[dict[str, cp_model.IntVar], dict[str, cp_model.LinearExpr]]:
    by_student: dict[str, list[cp_model.LinearExpr]] = defaultdict(list)
    for key, var in assignment_vars.items():
        request = requests_by_key[key.request_key]
        by_student[request.student_id].append(var * request.period_units)
    fully_scheduled: dict[str, cp_model.IntVar] = {}
    remaining_exprs: dict[str, cp_model.LinearExpr] = {}
    for student in allocation_input.students:
        used = sum(by_student.get(student.student_id, ()))
        remaining = student.target_period_units - used
        remaining_exprs[student.student_id] = remaining
        full = model.NewBoolVar(f"fully_scheduled__{_safe_name(student.student_id)}")
        # Target-load constraints guarantee remaining >= 0, so full is exactly
        # the indicator for remaining == 0.
        model.Add(remaining == 0).OnlyEnforceIf(full)
        model.Add(remaining >= 1).OnlyEnforceIf(full.Not())
        fully_scheduled[student.student_id] = full
    return fully_scheduled, remaining_exprs


def _stage_expressions(
    allocation_input: CanonicalAllocationInput,
    requests_by_key: dict[str, LogicalRequest],
    assigned_vars: dict[str, cp_model.IntVar],
    math_violation_vars: dict[str, cp_model.IntVar],
    fully_scheduled_vars: dict[str, cp_model.IntVar],
    remaining_exprs: dict[str, cp_model.LinearExpr],
    assignment_vars: dict[_VariableKey, cp_model.IntVar],
    seed: int,
) -> tuple[dict[CpSatStageName, cp_model.LinearExpr], int]:
    primary = [request for request in allocation_input.logical_requests if request.request_type == "primary"]
    primary_unmet_count = sum(1 - assigned_vars[request.request_key] for request in primary)
    primary_unmet_units = sum((1 - assigned_vars[request.request_key]) * request.period_units for request in primary)
    max_possible_unmet_units = sum(request.period_units for request in primary)
    primary_base = max_possible_unmet_units + 1
    # primary_base is strictly greater than every possible period-unit tie
    # breaker, so one fewer unmet logical primary always dominates any
    # difference in unmet period units.
    primary_penalty = primary_unmet_count * primary_base + primary_unmet_units
    alternates_by_rank: dict[int, list[cp_model.IntVar]] = defaultdict(list)
    for request in allocation_input.logical_requests:
        if request.request_type == "alternate" and request.request_rank is not None:
            alternates_by_rank[request.request_rank].append(assigned_vars[request.request_key])
    tie_break = _seeded_tie_break_expr(assignment_vars, seed)
    return (
        {
            CpSatStageName.MATH_COVERAGE: sum(math_violation_vars.values()),
            CpSatStageName.PRIMARY_SATISFACTION: primary_penalty,
            CpSatStageName.ALTERNATE_RANK_1: sum(alternates_by_rank.get(1, ())),
            CpSatStageName.ALTERNATE_RANK_2: sum(alternates_by_rank.get(2, ())),
            CpSatStageName.ALTERNATE_RANK_3: sum(alternates_by_rank.get(3, ())),
            CpSatStageName.FULLY_SCHEDULED: sum(fully_scheduled_vars.values()),
            CpSatStageName.REMAINING_PERIOD_UNITS: sum(remaining_exprs.values()),
            CpSatStageName.SEEDED_TIE_BREAK: tie_break,
        },
        primary_base,
    )


def _seeded_tie_break_expr(
    assignment_vars: dict[_VariableKey, cp_model.IntVar],
    seed: int,
) -> cp_model.LinearExpr:
    keys = sorted(assignment_vars, key=lambda item: (item.request_key, item.section_id))
    shuffled = list(keys)
    random.Random(seed).shuffle(shuffled)
    weights = {key: index + 1 for index, key in enumerate(shuffled)}
    return sum(weights[key] * assignment_vars[key] for key in keys)


def _new_solver(
    max_time_seconds: float,
    workers: int,
    log_search_progress: bool,
    seed: int,
) -> cp_model.CpSolver:
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(max_time_seconds)
    solver.parameters.num_search_workers = int(workers)
    solver.parameters.random_seed = int(seed)
    solver.parameters.log_search_progress = bool(log_search_progress)
    solver.parameters.repair_hint = True
    solver.parameters.hint_conflict_limit = 1000
    return solver


def _add_constrained_first_hint(
    model: cp_model.CpModel,
    allocation_input: CanonicalAllocationInput,
    assignment_vars: dict[_VariableKey, cp_model.IntVar],
    math_fallback_rules: tuple[MathFallbackRule, ...],
    math_course_ids: tuple[str, ...],
    seed: int,
) -> None:
    # Hints only guide search. They do not relax any hard policy; CP-SAT may
    # repair or ignore them while optimizing the formal model.
    from .constrained_first_baseline import run_constrained_first_baseline

    greedy = run_constrained_first_baseline(
        allocation_input,
        seed,
        math_fallback_rules=math_fallback_rules,
        math_course_ids=math_course_ids,
    )
    hinted = {
        _VariableKey(assignment.request_key, assignment.linked_section_group_id)
        for assignment in greedy.assignments
    }
    for key, var in assignment_vars.items():
        if key in hinted:
            model.AddHint(var, 1)


def _stage_diagnostic(
    stage_name: CpSatStageName,
    status: CpSatSolveStatus,
    solver: cp_model.CpSolver,
) -> CpSatStageDiagnostic:
    has_solution = status in {CpSatSolveStatus.OPTIMAL, CpSatSolveStatus.FEASIBLE}
    objective = int(round(solver.ObjectiveValue())) if has_solution else None
    bound = int(round(solver.BestObjectiveBound())) if has_solution else None
    return CpSatStageDiagnostic(
        stage_name=stage_name,
        status=status,
        objective_value=objective,
        best_objective_bound=bound,
        wall_time_seconds=round(solver.WallTime(), 6),
        conflicts=int(solver.NumConflicts()),
        branches=int(solver.NumBranches()),
        optimum_proven=status == CpSatSolveStatus.OPTIMAL,
    )


def _solve_status(raw_status: int) -> CpSatSolveStatus:
    if raw_status == cp_model.OPTIMAL:
        return CpSatSolveStatus.OPTIMAL
    if raw_status == cp_model.FEASIBLE:
        return CpSatSolveStatus.FEASIBLE
    if raw_status == cp_model.INFEASIBLE:
        return CpSatSolveStatus.INFEASIBLE
    if raw_status == cp_model.MODEL_INVALID:
        return CpSatSolveStatus.MODEL_INVALID
    return CpSatSolveStatus.UNKNOWN


def _selected_assignments(
    build: _ModelBuild,
    solver: cp_model.CpSolver,
) -> tuple[_VariableKey, ...]:
    return tuple(
        key
        for key in sorted(build.assignment_vars, key=lambda item: (item.request_key, item.section_id))
        if solver.BooleanValue(build.assignment_vars[key])
    )


def _replay_solution(
    allocation_input: CanonicalAllocationInput,
    build: _ModelBuild,
    selected: tuple[_VariableKey, ...],
) -> AllocationState:
    state = AllocationState(
        allocation_input,
        supplemental_requests=tuple(plan.fallback_request for plan in build.fallback_plans),
        supplemental_candidate_index={plan.fallback_request.request_key: plan.candidates for plan in build.fallback_plans},
    )
    for key in sorted(selected, key=lambda item: _assignment_replay_sort_key(build.requests_by_key[item.request_key], item.section_id)):
        request = build.requests_by_key[key.request_key]
        result = state.try_assign(request.student_id, request.request_key, key.section_id)
        if not result.allowed:
            raise RuntimeError(
                "CP-SAT solution failed AllocationState replay: "
                f"{request.request_key} -> {key.section_id}: {[reason.value for reason in result.reasons]}"
            )
    issues = state.validate_internal_consistency()
    if issues:
        raise RuntimeError(f"CP-SAT solution failed AllocationState consistency: {issues!r}")
    return state


def _assignment_replay_sort_key(request: LogicalRequest, section_id: str) -> tuple:
    type_order = {"primary": 0, MANDATORY_FALLBACK_REQUEST_TYPE: 1, "alternate": 2}.get(request.request_type, 9)
    rank = request.request_rank or 0
    return (request.student_id, type_order, rank, request.request_key, section_id)


def _build_request_outcomes(
    allocation_input: CanonicalAllocationInput,
    build: _ModelBuild,
    solver: cp_model.CpSolver,
    state: AllocationState,
) -> tuple[RequestOutcome, ...]:
    outcomes: list[RequestOutcome] = []
    assignment_by_request = {assignment.request_key: assignment for assignment in state.all_assignments()}
    remaining_by_student = {
        student.student_id: state.student_remaining_period_units(student.student_id)
        for student in allocation_input.students
    }
    for request in allocation_input.logical_requests:
        assignment = assignment_by_request.get(request.request_key)
        before = remaining_by_student[request.student_id]
        if request.request_type == "primary":
            status = (
                PrimaryRequestStatus.ASSIGNED
                if assignment is not None
                else PrimaryRequestStatus.UNMET_NO_CANDIDATES
                if not build.candidate_index.get(request.request_key, ())
                else PrimaryRequestStatus.UNMET_ALL_CANDIDATES_REJECTED
            )
        else:
            status = _alternate_status(request, assignment, before, build)
        outcomes.append(
            RequestOutcome(
                request_key=request.request_key,
                student_id=request.student_id,
                request_type=request.request_type,
                alternate_rank=request.request_rank,
                candidate_key=request.candidate_key,
                period_units=request.period_units,
                status=status,
                assignment_key=assignment.assignment_key if assignment is not None else None,
                assigned_linked_section_group_id=assignment.linked_section_group_id if assignment is not None else None,
                candidate_attempts=(),
                remaining_units_before=before,
                remaining_units_after=before,
            )
        )
    return tuple(outcomes)


def _alternate_status(
    request: LogicalRequest,
    assignment,
    remaining_units: int,
    build: _ModelBuild,
) -> AlternateRequestStatus:
    if assignment is not None:
        return AlternateRequestStatus.ASSIGNED
    if remaining_units == 0:
        return AlternateRequestStatus.NOT_NEEDED
    if request.period_units > remaining_units:
        return AlternateRequestStatus.DOES_NOT_FIT_REMAINING_LOAD
    if not build.candidate_index.get(request.request_key, ()):
        return AlternateRequestStatus.UNASSIGNED_NO_CANDIDATES
    return AlternateRequestStatus.UNASSIGNED_ALL_CANDIDATES_REJECTED


def _build_fallback_outcomes(
    build: _ModelBuild,
    solver: cp_model.CpSolver,
    state: AllocationState,
) -> tuple[MandatoryFallbackOutcome, ...]:
    assignment_by_request = {assignment.request_key: assignment for assignment in state.all_assignments()}
    outcomes: list[MandatoryFallbackOutcome] = []
    for plan in build.fallback_plans:
        assignment = assignment_by_request.get(plan.fallback_request.request_key)
        status = _fallback_status(plan, build, solver, assignment is not None)
        remaining = state.student_remaining_period_units(plan.source_request.student_id)
        outcomes.append(
            MandatoryFallbackOutcome(
                student_id=plan.source_request.student_id,
                source_request_key=plan.source_request.request_key,
                source_course_id=plan.source_request.candidate_key,
                fallback_request_key=plan.fallback_request.request_key,
                fallback_course_id=plan.fallback_request.candidate_key,
                status=status,
                assignment_key=assignment.assignment_key if assignment is not None else None,
                assigned_linked_section_group_id=assignment.linked_section_group_id if assignment is not None else None,
                candidate_attempts=(),
                remaining_units_before=remaining,
                remaining_units_after=remaining,
            )
        )
    return tuple(outcomes)


def _fallback_status(
    plan: _FallbackPlan,
    build: _ModelBuild,
    solver: cp_model.CpSolver,
    assigned: bool,
) -> MandatoryFallbackStatus:
    if solver.BooleanValue(build.assigned_vars[plan.source_request.request_key]):
        return MandatoryFallbackStatus.NOT_REQUIRED_SOURCE_ASSIGNED
    math_set = set(build.math_course_ids)
    for request in build.requests_by_key.values():
        if (
            request.student_id == plan.source_request.student_id
            and request.request_type == "primary"
            and request.request_key != plan.source_request.request_key
            and request.candidate_key in math_set
            and solver.BooleanValue(build.assigned_vars[request.request_key])
        ):
            return MandatoryFallbackStatus.NOT_REQUIRED_MATH_COVERAGE_ALREADY_SATISFIED
    if assigned:
        return MandatoryFallbackStatus.ASSIGNED
    if not plan.candidates:
        return MandatoryFallbackStatus.UNASSIGNED_NO_CANDIDATES
    return MandatoryFallbackStatus.UNASSIGNED_ALL_CANDIDATES_REJECTED


def _objective_values(
    build: _ModelBuild,
    solver: cp_model.CpSolver,
    stage_values: dict[CpSatStageName, int],
) -> CpSatObjectiveValues:
    primary_penalty = stage_values.get(CpSatStageName.PRIMARY_SATISFACTION, 0)
    primary_unmet_count = primary_penalty // build.primary_penalty_dominance_base
    primary_unmet_units = primary_penalty % build.primary_penalty_dominance_base
    return CpSatObjectiveValues(
        math_coverage_violations=stage_values.get(CpSatStageName.MATH_COVERAGE, 0),
        primary_unmet_count=primary_unmet_count,
        primary_unmet_period_units=primary_unmet_units,
        primary_penalty=primary_penalty,
        alternate_rank1_assigned=stage_values.get(CpSatStageName.ALTERNATE_RANK_1, 0),
        alternate_rank2_assigned=stage_values.get(CpSatStageName.ALTERNATE_RANK_2, 0),
        alternate_rank3_assigned=stage_values.get(CpSatStageName.ALTERNATE_RANK_3, 0),
        fully_scheduled_students=stage_values.get(CpSatStageName.FULLY_SCHEDULED, 0),
        total_remaining_period_units=stage_values.get(CpSatStageName.REMAINING_PERIOD_UNITS, 0),
        seeded_tie_break_value=stage_values.get(CpSatStageName.SEEDED_TIE_BREAK, 0),
    )


def _empty_result(
    seed: int,
    status: CpSatSolveStatus,
    diagnostics: tuple[CpSatStageDiagnostic, ...],
    build: _ModelBuild,
    elapsed: float,
    optimality_proven: bool,
) -> CpSatAllocationResult:
    proto = build.model.Proto()
    return CpSatAllocationResult(
        algorithm_name=ALGORITHM_NAME,
        seed=int(seed),
        solve_status=status,
        lexicographic_optimality_proven=optimality_proven,
        stage_diagnostics=diagnostics,
        objective_values=CpSatObjectiveValues(),
        model_stats=CpSatModelStats(
            total_variables=len(proto.variables),
            total_constraints=len(proto.constraints),
            build_time_seconds=round(build.build_time_seconds, 6),
            solve_time_seconds=round(max(elapsed - build.build_time_seconds, 0.0), 6),
        ),
    )


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value)
