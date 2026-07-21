from __future__ import annotations

import hashlib
import copy
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable

from ortools.sat.python import cp_model

from src.final_schedule_policy import (
    MAXIMUM_ORDINARY_PRIMARY_UNMET_COUNT,
    MAXIMUM_PROTECTED_PRIMARY_UNMET_COUNT,
    MAXIMUM_SCHEDULE_GAP_COUNT,
    MINIMUM_ASSIGNED_LOGICAL_COURSE_COUNT,
    evaluate_final_schedule_policy,
)

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
    CpSatBootstrapStatus,
    CpSatModelStats,
    CpSatModelScope,
    CpSatObjectiveValues,
    CpSatSolveStatus,
    CpSatStageDiagnostic,
    CpSatStageName,
)
from .input_models import CanonicalAllocationInput, CanonicalStudent, LogicalRequest, SourceRequestRow
from .math_policy import evaluate_math_policy
from .math_policy_models import MathFallbackRule
from .persisted_solution import PersistedSolutionSeed, load_persisted_solution_seed
from .random_baseline import (
    HIGH_DEMAND_PRIMARY_THRESHOLD,
    _build_mandatory_fallback_plans,
    _finalize_baseline_result,
)
from .state import MANDATORY_FALLBACK_REQUEST_TYPE, AllocationState


ALGORITHM_NAME = "fair_cp_sat_solver_v1_2"


class CpSatFinalSchedulePolicyConsistencyError(RuntimeError):
    """Raised when a claimed CP-SAT final solution fails the policy gate."""


class CpSatStageIncumbentConsistencyError(RuntimeError):
    """Raised when a formal stage incumbent cannot be carried to the next stage."""


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
    model_scope: CpSatModelScope
    model: cp_model.CpModel
    assignment_vars: dict[_VariableKey, cp_model.IntVar]
    assigned_vars: dict[str, cp_model.LinearExpr]
    math_violation_vars: dict[str, cp_model.IntVar]
    logical_assigned_course_vars: dict[str, cp_model.IntVar]
    logical_assigned_course_total_var: cp_model.IntVar
    fully_scheduled_vars: dict[str, cp_model.IntVar]
    fallback_plans: tuple[_FallbackPlan, ...]
    math_course_ids: tuple[str, ...]
    requests_by_key: dict[str, LogicalRequest]
    students_by_id: dict[str, CanonicalStudent]
    candidate_index: dict[str, tuple[str, ...]]
    stage_exprs: dict[CpSatStageName, cp_model.LinearExpr]
    primary_penalty_dominance_base: int
    build_time_seconds: float


@dataclass(frozen=True)
class _SolveStage:
    stage_name: CpSatStageName
    sense: str


@dataclass(frozen=True)
class _StageRun:
    solver: cp_model.CpSolver | None
    status: CpSatSolveStatus
    diagnostics: tuple[CpSatStageDiagnostic, ...]
    stage_values: dict[CpSatStageName, int]
    lexicographic_optimum: bool
    conditional_optimization_performed: bool
    highest_globally_proven_stage: CpSatStageName | None
    stage_to_stage_hint_used: bool
    incumbent_candidates: tuple["_IncumbentCandidate", ...] = ()
    budget_exhausted: bool = False
    selected_candidate_sources: tuple[str, ...] = ()


@dataclass
class _BootstrapBuild:
    model: cp_model.CpModel
    assignment_vars: dict[_VariableKey, cp_model.IntVar]
    assigned_vars: dict[str, cp_model.LinearExpr]
    requests_by_key: dict[str, LogicalRequest]
    candidate_index: dict[str, tuple[str, ...]]
    build_time_seconds: float


@dataclass(frozen=True)
class _BootstrapRun:
    build: _BootstrapBuild | None
    solver: cp_model.CpSolver | None
    status: CpSatBootstrapStatus
    diagnostic: CpSatStageDiagnostic | None
    selected_keys: tuple[_VariableKey, ...]
    solve_time_seconds: float
    time_to_first_hard_feasible_solution_seconds: float | None
    hint_strategy: str
    budget_exhausted: bool = False


@dataclass(frozen=True)
class _FullModelFeasibilityRun:
    build: _ModelBuild | None
    solver: cp_model.CpSolver | None
    status: CpSatSolveStatus
    diagnostic: CpSatStageDiagnostic | None
    selected_keys: tuple[_VariableKey, ...]
    solve_time_seconds: float
    budget_exhausted: bool = False
    hint_audit: _HintAudit = field(default_factory=lambda: _HintAudit())


@dataclass(frozen=True)
class _HintSeed:
    source: str = "none"
    keys: tuple[_VariableKey, ...] = ()
    replay_policy_pass: bool | None = None
    violation_students: int | None = None
    persisted: PersistedSolutionSeed | None = None
    internal_audit: "_HintAudit | None" = None


@dataclass(frozen=True)
class _HintAudit:
    source: str = "none"
    total_model_variables: int = 0
    variables_supplied: int = 0
    coverage_rate: float = 0.0
    selected_variables: int = 0
    zero_variables: int = 0
    unknown_or_unmapped_assignments: int = 0
    duplicate_keys: int = 0
    replay_policy_pass: bool | None = None
    violation_students: int | None = None
    auxiliary_variables_derived: int = 0
    initial_solution_seed_enabled: bool = False
    initial_solution_seed_role: str = ""
    initial_solution_seed_source_commit: str = ""
    initial_solution_seed_source_algorithm: str = ""
    initial_solution_seed_source_status: str = ""
    initial_solution_seed_source_policy_pass: bool | None = None
    initial_solution_seed_manifest_sha256: str = ""
    initial_solution_seed_request_outcomes_sha256: str = ""
    initial_solution_seed_provenance_sha256: str = ""
    initial_solution_seed_fingerprint: tuple[tuple[str, object], ...] = ()
    initial_solution_seed_hint_coverage: float | None = None
    initial_solution_seed_unknown_keys: int = 0
    initial_solution_seed_duplicate_keys: int = 0
    assignment_hash: str = ""
    candidate_variables: int = 0
    candidate_variables_hinted: int = 0
    candidate_coverage_rate: float = 0.0
    unhinted_variables: int = 0
    out_of_domain_keys: int = 0
    structural_issue_count: int = 0
    primary_assigned: int | None = None
    primary_unmet: int | None = None
    logical_assigned: int | None = None
    logical_gap: int | None = None
    logical_full: int | None = None
    gap_over_1: int | None = None
    below_five: int | None = None
    policy_violation_count: int | None = None
    runtime_seconds: float = field(default=0.0, compare=False)


@dataclass(frozen=True)
class _InternalRepairRun:
    build: _ModelBuild | None
    solver: cp_model.CpSolver | None
    status: CpSatSolveStatus
    diagnostic: CpSatStageDiagnostic | None
    selected_keys: tuple[_VariableKey, ...]
    solve_time_seconds: float
    hint_audit: _HintAudit = field(default_factory=lambda: _HintAudit())
    model_before_hint_hash: str = ""
    model_after_hint_hash: str = ""
    model_invariance_equal: bool | None = None
    model_invariance_without_distance_hash: str = ""
    model_invariance_distance_stripped_hash: str = ""
    model_invariance_distance_stripped_equal: bool | None = None
    variable_hash: str = ""
    domain_hash: str = ""
    constraint_hash: str = ""
    candidate_mapping_hash: str = ""
    objective_strategy: str = "none"
    time_to_first_solution_seconds: float | None = None
    hamming_distance: int | None = None
    greedy_assignments_removed: int | None = None
    new_assignments_added: int | None = None
    changed_students: int | None = None
    changed_requests: int | None = None
    changed_sections: int | None = None
    response_proto_hash: str = ""
    validation_failure: str = ""
    validated: bool = False
    baseline_result: object | None = None


class _FirstSolutionCapture(cp_model.CpSolverSolutionCallback):
    """Capture the first complete CP-SAT response without becoming the result."""

    def __init__(self, build: _ModelBuild) -> None:
        super().__init__()
        self.build = build
        self.selected_keys: tuple[_VariableKey, ...] = ()
        self.time_to_first_solution_seconds: float | None = None

    def on_solution_callback(self) -> None:
        self.selected_keys = tuple(
            key
            for key in sorted(self.build.assignment_vars, key=lambda item: (item.request_key, item.section_id))
            if self.BooleanValue(self.build.assignment_vars[key])
        )
        self.time_to_first_solution_seconds = float(self.WallTime())
        self.StopSearch()


@dataclass(frozen=True)
class _StageIncumbent:
    """Complete model solution captured from one successful objective stage."""

    values_by_name: tuple[tuple[str, int], ...]
    domains_by_name: tuple[tuple[str, tuple[int, ...]], ...]


@dataclass(frozen=True)
class _IncumbentCandidate:
    candidate_id: str
    source: str
    model_scope: CpSatModelScope
    selected_keys: tuple[_VariableKey, ...]
    replay_policy_pass: bool | None = None
    snapshot: _StageIncumbent | None = None


@dataclass
class _GlobalTimeBudget:
    max_seconds: float | None
    started_at: float
    exhausted: bool = False

    def remaining(self) -> float | None:
        if self.max_seconds is None:
            return None
        return max(float(self.max_seconds) - (time.perf_counter() - self.started_at), 0.0)

    def effective_limit(self, requested: float) -> float:
        remaining = self.remaining()
        if remaining is None:
            return float(requested)
        if remaining <= 0:
            self.exhausted = True
            return 0.0
        return min(float(requested), remaining)

    def refresh(self) -> None:
        remaining = self.remaining()
        if remaining is not None and remaining <= 0:
            self.exhausted = True


def _enrichment_stages(*, logical_schedule_completion_enabled: bool = True) -> tuple[_SolveStage, ...]:
    stages = (
        _SolveStage(CpSatStageName.ALTERNATE_RANK_1, "max"),
        _SolveStage(CpSatStageName.ALTERNATE_RANK_2, "max"),
        _SolveStage(CpSatStageName.ALTERNATE_RANK_3, "max"),
        _SolveStage(CpSatStageName.FULLY_SCHEDULED, "max"),
        _SolveStage(CpSatStageName.REMAINING_PERIOD_UNITS, "min"),
        _SolveStage(CpSatStageName.SEEDED_TIE_BREAK, "min"),
    )
    if logical_schedule_completion_enabled:
        return (_SolveStage(CpSatStageName.LOGICAL_SCHEDULE_COMPLETION, "max"), *stages)
    return stages


def run_fair_cp_sat_solver(
    allocation_input: CanonicalAllocationInput,
    *,
    seed: int,
    math_fallback_rules: tuple[MathFallbackRule, ...] = (),
    math_course_ids: tuple[str, ...] = (),
    max_time_seconds_per_stage: float = 30.0,
    num_search_workers: int = 1,
    log_search_progress: bool = False,
    continue_after_feasible: bool = True,
    use_feasibility_bootstrap: bool = True,
    bootstrap_time_seconds: float | None = None,
    max_total_time_seconds: float | None = None,
    use_constrained_first_hint: bool = True,
    stage_to_stage_hints: bool = True,
    enforce_final_schedule_hard_constraints: bool = True,
    logical_schedule_completion_enabled: bool = True,
    initial_solution_artifact_dir: Path | str | None = None,
    internal_feasibility_hint_strategy: str = "none",
    internal_repair_time_seconds: float | None = None,
    internal_repair_objective_strategy: str = "none",
    stop_after_first_valid_solution: bool = False,
) -> CpSatAllocationResult:
    """Solve the fixed-section allocation problem with CP-SAT.

    This solver never changes sections, capacities, periods, requests, or
    eligibility. It models fairness policies as hard constraints and then uses
    explicit lexicographic stages for soft goals.
    """

    started = time.perf_counter()
    if internal_feasibility_hint_strategy not in {"none", "constrained_first"}:
        raise ValueError(
            "internal_feasibility_hint_strategy must be 'none' or 'constrained_first'"
        )
    if internal_repair_objective_strategy not in {"none", "hamming_to_constrained_first"}:
        raise ValueError(
            "internal_repair_objective_strategy must be 'none' or "
            "'hamming_to_constrained_first'"
        )
    if internal_repair_objective_strategy != "none" and internal_feasibility_hint_strategy != "constrained_first":
        raise ValueError("internal repair objective requires the constrained_first internal hint")
    if stop_after_first_valid_solution and internal_repair_objective_strategy == "none":
        raise ValueError("stop_after_first_valid_solution requires an internal repair objective")
    if internal_feasibility_hint_strategy != "none" and not enforce_final_schedule_hard_constraints:
        raise ValueError("internal feasibility recovery requires final schedule hard constraints")
    enrichment_stages = _enrichment_stages(
        logical_schedule_completion_enabled=logical_schedule_completion_enabled
    )
    math_course_ids = tuple(sorted(math_course_ids))
    budget = _GlobalTimeBudget(max_total_time_seconds, started)
    persisted_seed = (
        load_persisted_solution_seed(initial_solution_artifact_dir, allocation_input)
        if initial_solution_artifact_dir is not None
        else None
    )
    fallback_plans = _convert_fallback_plans(_build_mandatory_fallback_plans(allocation_input, math_fallback_rules))
    bootstrap_run = _BootstrapRun(
        build=None,
        solver=None,
        status=CpSatBootstrapStatus.DISABLED,
        diagnostic=None,
        selected_keys=(),
        solve_time_seconds=0.0,
        time_to_first_hard_feasible_solution_seconds=None,
        hint_strategy="none",
    )
    full_feasibility_run = _FullModelFeasibilityRun(
        build=None,
        solver=None,
        status=CpSatSolveStatus.SKIPPED,
        diagnostic=None,
        selected_keys=(),
        solve_time_seconds=0.0,
    )
    internal_repair_run = _InternalRepairRun(
        build=None,
        solver=None,
        status=CpSatSolveStatus.SKIPPED,
        diagnostic=None,
        selected_keys=(),
        solve_time_seconds=0.0,
    )
    full_model_hint_seed = _HintSeed()
    diagnostics: list[CpSatStageDiagnostic] = []
    try:
        _validate_candidate_index(allocation_input)
    except ValueError:
        if use_feasibility_bootstrap:
            bootstrap_run = _model_invalid_bootstrap_run("none")
            diagnostics.append(bootstrap_run.diagnostic)
        return _empty_result(
            seed,
            CpSatSolveStatus.MODEL_INVALID,
            tuple(diagnostics),
            None,
            None,
            bootstrap_run,
            time.perf_counter() - started,
            False,
            external_hint_used=False,
            stage_to_stage_hint_used=False,
            max_total_time_seconds=max_total_time_seconds,
            total_budget_exhausted=budget.exhausted,
            skipped_stage_count=0,
            core_hint_source="none",
            final_schedule_hard_constraints_enabled=enforce_final_schedule_hard_constraints,
        )
    external_hint_keys = (
        _constrained_first_partial_hint_keys(
            allocation_input,
            math_fallback_rules,
            math_course_ids,
            seed,
        )
        if use_constrained_first_hint
        else ()
    )
    external_hint_used = bool(external_hint_keys)
    if persisted_seed is not None:
        full_model_hint_seed = _HintSeed(
            source="persisted_feasible_seed",
            keys=tuple(
                _VariableKey(request_key, section_id)
                for request_key, section_id in persisted_seed.selected_assignments
            ),
            replay_policy_pass=True,
            violation_students=0,
            persisted=persisted_seed,
        )
    elif use_constrained_first_hint and enforce_final_schedule_hard_constraints:
        full_model_hint_seed = _constrained_first_full_hint_seed(
            allocation_input,
            math_fallback_rules,
            math_course_ids,
            seed,
        )

    if internal_feasibility_hint_strategy == "constrained_first":
        internal_repair_run = _run_internal_repair_feasibility(
            allocation_input,
            fallback_plans,
            math_course_ids,
            seed=seed,
            max_time_seconds=(
                internal_repair_time_seconds
                if internal_repair_time_seconds is not None
                else max_time_seconds_per_stage
            ),
            num_search_workers=num_search_workers,
            log_search_progress=log_search_progress,
            budget=budget,
            objective_strategy=internal_repair_objective_strategy,
            stop_after_first_solution=stop_after_first_valid_solution,
        )
        if internal_repair_run.diagnostic is not None:
            diagnostics.append(internal_repair_run.diagnostic)
        if internal_repair_objective_strategy != "none":
            if internal_repair_run.validated:
                return _validated_internal_repair_result(
                    allocation_input=allocation_input,
                    seed=seed,
                    math_course_ids=math_course_ids,
                    math_fallback_rules=math_fallback_rules,
                    diagnostics=tuple(diagnostics),
                    internal_repair_run=internal_repair_run,
                    bootstrap_run=bootstrap_run,
                    max_total_time_seconds=max_total_time_seconds,
                    total_budget_exhausted=budget.exhausted,
                    logical_schedule_completion_enabled=logical_schedule_completion_enabled,
                    final_schedule_hard_constraints_enabled=enforce_final_schedule_hard_constraints,
                    solve_status=internal_repair_run.status,
                )
            return _empty_result(
                seed,
                internal_repair_run.status if internal_repair_run.status in {
                    CpSatSolveStatus.INFEASIBLE,
                    CpSatSolveStatus.MODEL_INVALID,
                    CpSatSolveStatus.FEASIBLE,
                    CpSatSolveStatus.OPTIMAL,
                } else CpSatSolveStatus.UNKNOWN,
                tuple(diagnostics),
                internal_repair_run.build,
                None,
                bootstrap_run,
                time.perf_counter() - started,
                False,
                external_hint_used=False,
                stage_to_stage_hint_used=False,
                max_total_time_seconds=max_total_time_seconds,
                total_budget_exhausted=budget.exhausted,
                skipped_stage_count=0,
                core_hint_source="none",
                final_schedule_hard_constraints_enabled=enforce_final_schedule_hard_constraints,
                internal_repair_run=internal_repair_run,
            )
        if internal_repair_run.status in {CpSatSolveStatus.INFEASIBLE, CpSatSolveStatus.MODEL_INVALID}:
            return _empty_result(
                seed,
                internal_repair_run.status,
                tuple(diagnostics),
                internal_repair_run.build,
                None,
                bootstrap_run,
                time.perf_counter() - started,
                False,
                external_hint_used=external_hint_used,
                stage_to_stage_hint_used=False,
                max_total_time_seconds=max_total_time_seconds,
                total_budget_exhausted=budget.exhausted,
                skipped_stage_count=0,
                core_hint_source="none",
                final_schedule_hard_constraints_enabled=enforce_final_schedule_hard_constraints,
                internal_repair_run=internal_repair_run,
            )

    repair_validated = internal_repair_run.validated
    if use_feasibility_bootstrap and not repair_validated:
        bootstrap_run = _run_feasibility_bootstrap(
            allocation_input,
            seed=seed,
            max_time_seconds=bootstrap_time_seconds if bootstrap_time_seconds is not None else max_time_seconds_per_stage,
            num_search_workers=num_search_workers,
            log_search_progress=log_search_progress,
            initial_hint_keys=external_hint_keys,
            budget=budget,
        )
        if bootstrap_run.diagnostic is not None:
            diagnostics.append(bootstrap_run.diagnostic)
        if bootstrap_run.status == CpSatBootstrapStatus.INFEASIBLE:
            return _empty_result(
                seed,
                CpSatSolveStatus.INFEASIBLE,
                tuple(diagnostics),
                None,
                None,
                bootstrap_run,
                time.perf_counter() - started,
                False,
                external_hint_used=external_hint_used,
                stage_to_stage_hint_used=False,
                max_total_time_seconds=max_total_time_seconds,
                total_budget_exhausted=budget.exhausted,
                skipped_stage_count=0,
                core_hint_source="none",
                final_schedule_hard_constraints_enabled=enforce_final_schedule_hard_constraints,
                internal_repair_run=internal_repair_run,
            )
        if bootstrap_run.status == CpSatBootstrapStatus.MODEL_INVALID:
            return _empty_result(
                seed,
                CpSatSolveStatus.MODEL_INVALID,
                tuple(diagnostics),
                None,
                None,
                bootstrap_run,
                time.perf_counter() - started,
                False,
                external_hint_used=external_hint_used,
                stage_to_stage_hint_used=False,
                max_total_time_seconds=max_total_time_seconds,
                total_budget_exhausted=budget.exhausted,
                skipped_stage_count=0,
                core_hint_source="none",
                final_schedule_hard_constraints_enabled=enforce_final_schedule_hard_constraints,
                internal_repair_run=internal_repair_run,
            )

    if enforce_final_schedule_hard_constraints:
        if repair_validated:
            full_feasibility_run = _FullModelFeasibilityRun(
                build=None,
                solver=None,
                status=CpSatSolveStatus.SKIPPED,
                diagnostic=None,
                selected_keys=(),
                solve_time_seconds=0.0,
            )
        else:
            full_feasibility_run = _run_full_model_feasibility_incumbent(
                allocation_input,
                fallback_plans,
                math_course_ids,
                seed=seed,
                max_time_seconds=max_time_seconds_per_stage,
                num_search_workers=num_search_workers,
                log_search_progress=log_search_progress,
                initial_hint_seed=full_model_hint_seed,
                budget=budget,
            )
        if full_feasibility_run.diagnostic is not None:
            diagnostics.append(full_feasibility_run.diagnostic)

    core_build = _build_core_cp_sat_model(allocation_input, fallback_plans, math_course_ids, seed)
    incumbent_candidates: list[_IncumbentCandidate] = []
    if internal_repair_run.selected_keys and internal_repair_run.validated:
        incumbent_candidates.append(
            _IncumbentCandidate(
                candidate_id="internal_repair_feasibility",
                source="internal_repair_feasibility",
                model_scope=CpSatModelScope.ENRICHMENT,
                selected_keys=internal_repair_run.selected_keys,
                replay_policy_pass=True,
            )
        )
    if full_model_hint_seed.keys and full_model_hint_seed.replay_policy_pass is True:
        incumbent_candidates.append(
            _IncumbentCandidate(
                candidate_id="validated_initial_full_seed",
                source=full_model_hint_seed.source,
                model_scope=CpSatModelScope.ENRICHMENT,
                selected_keys=full_model_hint_seed.keys,
                replay_policy_pass=True,
            )
        )
    if full_feasibility_run.selected_keys:
        incumbent_candidates.append(
            _IncumbentCandidate(
                candidate_id="full_model_feasibility_incumbent",
                source="full_model_feasibility_incumbent",
                model_scope=CpSatModelScope.ENRICHMENT,
                selected_keys=full_feasibility_run.selected_keys,
                replay_policy_pass=True,
            )
        )
    core_initial_hint_source = set(external_hint_keys)
    core_hint_source = "constrained_first_partial" if external_hint_keys else "none"
    if internal_repair_run.selected_keys:
        core_initial_hint_source.update(internal_repair_run.selected_keys)
        core_hint_source = (
            f"internal_repair_feasibility+{core_hint_source}"
            if core_hint_source != "none"
            else "internal_repair_feasibility"
        )
    if bootstrap_run.selected_keys:
        core_initial_hint_source.update(bootstrap_run.selected_keys)
        core_hint_source = (
            "bootstrap+constrained_first_partial"
            if external_hint_keys
            else "bootstrap"
        )
    if full_feasibility_run.selected_keys:
        core_initial_hint_source.update(full_feasibility_run.selected_keys)
        core_hint_source = (
            f"full_model_feasibility_incumbent+{core_hint_source}"
            if core_hint_source != "none"
            else "full_model_feasibility_incumbent"
        )
    core_run = _solve_stage_sequence(
        core_build,
        (
            _SolveStage(CpSatStageName.MATH_COVERAGE, "min"),
            _SolveStage(CpSatStageName.PRIMARY_UNMET_COUNT, "min"),
            _SolveStage(CpSatStageName.PRIMARY_UNMET_PERIOD_UNITS, "min"),
        ),
        max_time_seconds_per_stage=max_time_seconds_per_stage,
        num_search_workers=num_search_workers,
        log_search_progress=log_search_progress,
        seed=seed,
        continue_after_feasible=continue_after_feasible,
        stage_to_stage_hints=stage_to_stage_hints,
        initial_hint_keys=tuple(sorted(core_initial_hint_source, key=lambda item: (item.request_key, item.section_id))),
        initial_fixed_values=(),
        already_conditional=False,
        budget=budget,
        incumbent_candidates=tuple(incumbent_candidates),
    )
    diagnostics.extend(core_run.diagnostics)
    if core_run.solver is None:
        if internal_repair_run.validated:
            return _validated_internal_repair_result(
                allocation_input=allocation_input,
                seed=seed,
                math_course_ids=math_course_ids,
                math_fallback_rules=math_fallback_rules,
                diagnostics=tuple(diagnostics),
                internal_repair_run=internal_repair_run,
                bootstrap_run=bootstrap_run,
                max_total_time_seconds=max_total_time_seconds,
                total_budget_exhausted=budget.exhausted,
                logical_schedule_completion_enabled=logical_schedule_completion_enabled,
                final_schedule_hard_constraints_enabled=enforce_final_schedule_hard_constraints,
            )
        skipped = _skipped_diagnostics(
            enrichment_stages,
            reason="time_budget_exhausted" if budget.exhausted else "no_incumbent",
            model_scope=CpSatModelScope.ENRICHMENT,
        )
        diagnostics.extend(skipped)
        return _empty_result(
            seed,
            core_run.status,
            tuple(diagnostics),
            core_build,
            None,
            bootstrap_run,
            time.perf_counter() - started,
            False,
            external_hint_used=external_hint_used,
            stage_to_stage_hint_used=core_run.stage_to_stage_hint_used,
            max_total_time_seconds=max_total_time_seconds,
            total_budget_exhausted=budget.exhausted,
            skipped_stage_count=sum(1 for item in diagnostics if item.skipped),
            core_hint_source=core_hint_source,
            final_schedule_hard_constraints_enabled=enforce_final_schedule_hard_constraints,
            hint_audit=full_feasibility_run.hint_audit,
            hint_selected_keys=full_feasibility_run.selected_keys,
            internal_repair_run=internal_repair_run,
        )

    required_core_stages = (
        CpSatStageName.MATH_COVERAGE,
        CpSatStageName.PRIMARY_UNMET_COUNT,
        CpSatStageName.PRIMARY_UNMET_PERIOD_UNITS,
    )
    core_values = {
        stage_name: core_run.stage_values[stage_name]
        for stage_name in required_core_stages
        if stage_name in core_run.stage_values
    }
    if len(core_values) != len(required_core_stages):
        final_build = core_build
        final_solver = core_run.solver
        final_stage_values = core_run.stage_values
        enrichment_build = None
        skipped = _skipped_diagnostics(
            enrichment_stages,
            reason="missing_core_objective",
            model_scope=CpSatModelScope.ENRICHMENT,
        )
        diagnostics.extend(skipped)
        enrichment_run = _StageRun(
            None,
            core_run.status,
            skipped,
            {},
            False,
            False,
            None,
            False,
            core_run.incumbent_candidates,
        )
        if enforce_final_schedule_hard_constraints:
            if internal_repair_run.validated:
                return _validated_internal_repair_result(
                    allocation_input=allocation_input,
                    seed=seed,
                    math_course_ids=math_course_ids,
                    math_fallback_rules=math_fallback_rules,
                    diagnostics=tuple(diagnostics),
                    internal_repair_run=internal_repair_run,
                    bootstrap_run=bootstrap_run,
                    max_total_time_seconds=max_total_time_seconds,
                    total_budget_exhausted=budget.exhausted,
                    logical_schedule_completion_enabled=logical_schedule_completion_enabled,
                    final_schedule_hard_constraints_enabled=enforce_final_schedule_hard_constraints,
                )
            final_status = (
                CpSatSolveStatus.INFEASIBLE
                if core_run.status == CpSatSolveStatus.INFEASIBLE
                else CpSatSolveStatus.UNKNOWN
            )
            return _empty_result(
                seed,
                final_status,
                tuple(diagnostics),
                core_build,
                None,
                bootstrap_run,
                time.perf_counter() - started,
                False,
                external_hint_used=external_hint_used,
                stage_to_stage_hint_used=core_run.stage_to_stage_hint_used,
                max_total_time_seconds=max_total_time_seconds,
                total_budget_exhausted=budget.exhausted,
                skipped_stage_count=sum(1 for item in diagnostics if item.skipped),
                core_hint_source=core_hint_source,
                final_schedule_hard_constraints_enabled=enforce_final_schedule_hard_constraints,
                hint_audit=full_feasibility_run.hint_audit,
                hint_selected_keys=full_feasibility_run.selected_keys,
                internal_repair_run=internal_repair_run,
            )
    else:
        core_selected = _selected_assignments(core_build, core_run.solver)
        enrichment_build = _build_enrichment_cp_sat_model(
            allocation_input,
            fallback_plans,
            math_course_ids,
            seed,
            core_values,
            enforce_final_schedule_hard_constraints=enforce_final_schedule_hard_constraints,
        )
        enrichment_hint_source = set(external_hint_keys)
        enrichment_hint_source.update(full_feasibility_run.selected_keys)
        enrichment_hint_source.update(internal_repair_run.selected_keys)
        if stage_to_stage_hints:
            enrichment_hint_source.update(core_selected)
        enrichment_hint_keys = tuple(sorted(enrichment_hint_source, key=lambda item: (item.request_key, item.section_id)))
        enrichment_run = _solve_stage_sequence(
            enrichment_build,
            enrichment_stages,
            max_time_seconds_per_stage=max_time_seconds_per_stage,
            num_search_workers=num_search_workers,
            log_search_progress=log_search_progress,
            seed=seed,
            continue_after_feasible=continue_after_feasible,
            stage_to_stage_hints=stage_to_stage_hints,
            initial_hint_keys=enrichment_hint_keys,
            initial_fixed_values=tuple(core_values.items()),
            already_conditional=not core_run.lexicographic_optimum,
            budget=budget,
            incumbent_candidates=core_run.incumbent_candidates,
        )
        diagnostics.extend(enrichment_run.diagnostics)

        final_build = enrichment_build
        final_solver = enrichment_run.solver
        final_stage_values = {**core_run.stage_values, **enrichment_run.stage_values}
        if final_solver is None:
            if enforce_final_schedule_hard_constraints:
                if internal_repair_run.validated:
                    return _validated_internal_repair_result(
                        allocation_input=allocation_input,
                        seed=seed,
                        math_course_ids=math_course_ids,
                        math_fallback_rules=math_fallback_rules,
                        diagnostics=tuple(diagnostics),
                        internal_repair_run=internal_repair_run,
                        bootstrap_run=bootstrap_run,
                        max_total_time_seconds=max_total_time_seconds,
                        total_budget_exhausted=budget.exhausted,
                        logical_schedule_completion_enabled=logical_schedule_completion_enabled,
                        final_schedule_hard_constraints_enabled=enforce_final_schedule_hard_constraints,
                    )
                final_status = (
                    CpSatSolveStatus.INFEASIBLE
                    if enrichment_run.status == CpSatSolveStatus.INFEASIBLE
                    else CpSatSolveStatus.UNKNOWN
                )
                return _empty_result(
                    seed,
                    final_status,
                    tuple(diagnostics),
                    enrichment_build,
                    core_build,
                    bootstrap_run,
                    time.perf_counter() - started,
                    False,
                    external_hint_used=external_hint_used,
                    stage_to_stage_hint_used=core_run.stage_to_stage_hint_used or enrichment_run.stage_to_stage_hint_used,
                    max_total_time_seconds=max_total_time_seconds,
                    total_budget_exhausted=budget.exhausted,
                    skipped_stage_count=sum(1 for item in diagnostics if item.skipped),
                    core_hint_source=core_hint_source,
                    final_schedule_hard_constraints_enabled=enforce_final_schedule_hard_constraints,
                    hint_audit=full_feasibility_run.hint_audit,
                    hint_selected_keys=full_feasibility_run.selected_keys,
                    internal_repair_run=internal_repair_run,
                )
            final_build = core_build
            final_solver = core_run.solver
            final_stage_values = core_run.stage_values

    selected = _selected_assignments(final_build, final_solver)
    try:
        state = _replay_solution(allocation_input, final_build, selected)
    except RuntimeError:
        return _empty_result(
            seed,
            CpSatSolveStatus.MODEL_INVALID,
            tuple(diagnostics),
            final_build,
            core_build if final_build is enrichment_build else None,
            bootstrap_run,
            time.perf_counter() - started,
            False,
            external_hint_used=external_hint_used,
            stage_to_stage_hint_used=core_run.stage_to_stage_hint_used or enrichment_run.stage_to_stage_hint_used,
            max_total_time_seconds=max_total_time_seconds,
            total_budget_exhausted=budget.exhausted,
            skipped_stage_count=0,
            core_hint_source=core_hint_source,
            final_schedule_hard_constraints_enabled=enforce_final_schedule_hard_constraints,
            hint_audit=full_feasibility_run.hint_audit,
            hint_selected_keys=full_feasibility_run.selected_keys,
            internal_repair_run=internal_repair_run,
        )

    request_outcomes = _build_request_outcomes(allocation_input, final_build, final_solver, state)
    fallback_outcomes = _build_fallback_outcomes(final_build, final_solver, state)
    baseline_result = _finalize_baseline_result(
        ALGORITHM_NAME,
        allocation_input,
        seed,
        (),
        state,
        request_outcomes,
        fallback_outcomes,
    )
    final_policy_report = None
    if enforce_final_schedule_hard_constraints:
        final_policy_report = evaluate_final_schedule_policy(ALGORITHM_NAME, baseline_result.student_outcomes)
        if not final_policy_report.summary.final_schedule_policy_pass:
            summary = final_policy_report.summary
            raise CpSatFinalSchedulePolicyConsistencyError(
                "CP-SAT final solution violates Final Schedule Policy Gate v1: "
                f"violating_students={summary.violating_student_count}, "
                f"protected={summary.protected_primary_unmet_violation_count}, "
                f"ordinary={summary.ordinary_primary_unmet_violation_count}, "
                f"schedule_gap_over_limit={summary.schedule_gap_over_limit_count}, "
                f"below_minimum_course={summary.below_minimum_course_count}"
            )
    math_report = evaluate_math_policy(allocation_input, baseline_result, tuple(sorted(math_course_ids)), math_fallback_rules)
    objective_values = _objective_values(
        final_build,
        final_solver,
        final_stage_values,
        logical_schedule_completion_enabled=logical_schedule_completion_enabled,
    )
    if final_policy_report is not None and logical_schedule_completion_enabled:
        _validate_logical_completion_consistency(
            baseline_result.student_outcomes,
            final_policy_report,
            objective_values.logical_assigned_course_count,
            final_stage_values.get(CpSatStageName.LOGICAL_SCHEDULE_COMPLETION),
        )
    lexicographic_optimum = core_run.lexicographic_optimum and enrichment_run.lexicographic_optimum and enrichment_build is not None
    final_status = CpSatSolveStatus.OPTIMAL if lexicographic_optimum else CpSatSolveStatus.FEASIBLE
    core_proto = core_build.model.Proto()
    enrichment_proto = enrichment_build.model.Proto() if enrichment_build is not None else None
    enrichment_variable_count = len(enrichment_proto.variables) if enrichment_proto is not None else 0
    enrichment_constraint_count = len(enrichment_proto.constraints) if enrichment_proto is not None else 0
    enrichment_build_time = enrichment_build.build_time_seconds if enrichment_build is not None else 0.0
    total_build_time = core_build.build_time_seconds + enrichment_build_time + _bootstrap_build_time(bootstrap_run)
    elapsed = time.perf_counter() - started
    conditional = core_run.conditional_optimization_performed or enrichment_run.conditional_optimization_performed
    highest_global = enrichment_run.highest_globally_proven_stage or core_run.highest_globally_proven_stage
    skipped_count = sum(1 for item in diagnostics if item.skipped)
    logical_metadata = _logical_completion_metadata(tuple(diagnostics), final_stage_values)
    return CpSatAllocationResult(
        algorithm_name=ALGORITHM_NAME,
        seed=int(seed),
        solve_status=final_status,
        lexicographic_optimality_proven=lexicographic_optimum,
        stage_diagnostics=tuple(diagnostics),
        objective_values=objective_values,
        model_stats=CpSatModelStats(
            total_variables=len(core_proto.variables) + enrichment_variable_count,
            total_constraints=len(core_proto.constraints) + enrichment_constraint_count,
            build_time_seconds=round(total_build_time, 6),
            solve_time_seconds=round(max(elapsed - total_build_time, 0.0), 6),
            core_model_variable_count=len(core_proto.variables),
            core_model_constraint_count=len(core_proto.constraints),
            enrichment_model_variable_count=enrichment_variable_count,
            enrichment_model_constraint_count=enrichment_constraint_count,
            bootstrap_enabled=use_feasibility_bootstrap,
            bootstrap_status=bootstrap_run.status,
            bootstrap_variable_count=_bootstrap_variable_count(bootstrap_run),
            bootstrap_constraint_count=_bootstrap_constraint_count(bootstrap_run),
            bootstrap_build_time_seconds=round(_bootstrap_build_time(bootstrap_run), 6),
            bootstrap_solve_time_seconds=round(bootstrap_run.solve_time_seconds, 6),
            bootstrap_hint_strategy=bootstrap_run.hint_strategy,
            bootstrap_incumbent_found=bootstrap_run.status == CpSatBootstrapStatus.FEASIBLE_FOUND,
            time_to_first_hard_feasible_solution_seconds=(
                round(bootstrap_run.time_to_first_hard_feasible_solution_seconds, 6)
                if bootstrap_run.time_to_first_hard_feasible_solution_seconds is not None
                else None
            ),
            core_hint_source=core_hint_source,
            max_total_time_seconds=max_total_time_seconds,
            total_budget_exhausted=budget.exhausted,
            skipped_stage_count=skipped_count,
            core_build_time_seconds=round(core_build.build_time_seconds, 6),
            enrichment_build_time_seconds=round(enrichment_build_time, 6),
            total_build_time_seconds=round(total_build_time, 6),
            total_solve_time_seconds=round(max(elapsed - total_build_time, 0.0), 6),
            time_to_first_feasible_solution_seconds=(
                round(bootstrap_run.time_to_first_hard_feasible_solution_seconds, 6)
                if bootstrap_run.time_to_first_hard_feasible_solution_seconds is not None
                else None
            ),
            warm_start_strategy=_warm_start_strategy(stage_to_stage_hints, bool(external_hint_keys)),
            external_hint_used=external_hint_used,
            stage_to_stage_hint_used=core_run.stage_to_stage_hint_used or enrichment_run.stage_to_stage_hint_used,
            highest_globally_proven_stage=highest_global,
            conditional_optimization_performed=conditional,
            objective_vector=_objective_vector(final_stage_values),
            final_schedule_hard_constraints_enabled=enforce_final_schedule_hard_constraints,
            post_solve_policy_gate_pass=(
                final_policy_report.summary.final_schedule_policy_pass if final_policy_report is not None else None
            ),
            logical_schedule_completion_objective_enabled=logical_metadata["enabled"],
            logical_schedule_completion_stage_status=logical_metadata["status"],
            logical_schedule_completion_objective_value=logical_metadata["objective_value"],
            logical_schedule_completion_best_bound=logical_metadata["best_bound"],
            logical_schedule_completion_conditionally_optimized=logical_metadata["conditionally_optimized"],
            logical_schedule_completion_fixed_value=logical_metadata["fixed_value"],
            hint_source=full_feasibility_run.hint_audit.source,
            hint_total_model_variables=full_feasibility_run.hint_audit.total_model_variables,
            hint_variables_supplied=full_feasibility_run.hint_audit.variables_supplied,
            hint_coverage_rate=full_feasibility_run.hint_audit.coverage_rate,
            hint_selected_variables=full_feasibility_run.hint_audit.selected_variables,
            hint_zero_variables=full_feasibility_run.hint_audit.zero_variables,
            hint_unknown_or_unmapped_assignments=full_feasibility_run.hint_audit.unknown_or_unmapped_assignments,
            hint_duplicate_keys=full_feasibility_run.hint_audit.duplicate_keys,
            hint_replay_policy_pass=full_feasibility_run.hint_audit.replay_policy_pass,
            full_model_seed_strategy=full_feasibility_run.hint_audit.source,
            full_model_seed_policy_pass=full_feasibility_run.hint_audit.replay_policy_pass,
            full_model_seed_violation_students=full_feasibility_run.hint_audit.violation_students,
            full_model_seed_repaired_by_solver=(
                full_feasibility_run.hint_audit.replay_policy_pass is False
                and bool(full_feasibility_run.selected_keys)
            ) if full_feasibility_run.hint_audit.source != "none" else None,
            initial_solution_seed_enabled=full_feasibility_run.hint_audit.initial_solution_seed_enabled,
            initial_solution_seed_role=full_feasibility_run.hint_audit.initial_solution_seed_role,
            initial_solution_seed_source_commit=full_feasibility_run.hint_audit.initial_solution_seed_source_commit,
            initial_solution_seed_source_algorithm=full_feasibility_run.hint_audit.initial_solution_seed_source_algorithm,
            initial_solution_seed_source_status=full_feasibility_run.hint_audit.initial_solution_seed_source_status,
            initial_solution_seed_source_policy_pass=full_feasibility_run.hint_audit.initial_solution_seed_source_policy_pass,
            initial_solution_seed_manifest_sha256=full_feasibility_run.hint_audit.initial_solution_seed_manifest_sha256,
            initial_solution_seed_request_outcomes_sha256=full_feasibility_run.hint_audit.initial_solution_seed_request_outcomes_sha256,
            initial_solution_seed_provenance_sha256=full_feasibility_run.hint_audit.initial_solution_seed_provenance_sha256,
            initial_solution_seed_fingerprint=full_feasibility_run.hint_audit.initial_solution_seed_fingerprint,
            initial_solution_seed_hint_coverage=full_feasibility_run.hint_audit.initial_solution_seed_hint_coverage,
            initial_solution_seed_unknown_keys=full_feasibility_run.hint_audit.initial_solution_seed_unknown_keys,
            initial_solution_seed_duplicate_keys=full_feasibility_run.hint_audit.initial_solution_seed_duplicate_keys,
            initial_solution_seed_selected_by_stage=tuple(
                ["full_model_feasibility"] if full_feasibility_run.hint_audit.initial_solution_seed_enabled else []
            ) + tuple(
                source
                for source in core_run.selected_candidate_sources + enrichment_run.selected_candidate_sources
                if "persisted_feasible_seed" in source
            ),
            internal_feasibility_hint_strategy=internal_feasibility_hint_strategy,
            internal_repair_objective_strategy=internal_repair_run.objective_strategy,
            internal_repair_hint_enabled=internal_feasibility_hint_strategy == "constrained_first",
            internal_repair_status=(internal_repair_run.status if internal_feasibility_hint_strategy != "none" else None),
            internal_repair_incumbent_found=bool(internal_repair_run.validated),
            internal_repair_runtime_seconds=round(internal_repair_run.solve_time_seconds, 6),
            internal_repair_time_to_first_solution_seconds=internal_repair_run.time_to_first_solution_seconds,
            internal_repair_hamming_distance=internal_repair_run.hamming_distance,
            internal_repair_greedy_assignments_removed=internal_repair_run.greedy_assignments_removed,
            internal_repair_new_assignments_added=internal_repair_run.new_assignments_added,
            internal_repair_changed_students=internal_repair_run.changed_students,
            internal_repair_changed_requests=internal_repair_run.changed_requests,
            internal_repair_changed_sections=internal_repair_run.changed_sections,
            internal_repair_response_proto_hash=internal_repair_run.response_proto_hash,
            internal_repair_validation_failure=internal_repair_run.validation_failure,
            internal_hint_assignment_hash=internal_repair_run.hint_audit.assignment_hash,
            internal_hint_primary_assigned=internal_repair_run.hint_audit.primary_assigned,
            internal_hint_primary_unmet=internal_repair_run.hint_audit.primary_unmet,
            internal_hint_logical_assigned=internal_repair_run.hint_audit.logical_assigned,
            internal_hint_logical_gap=internal_repair_run.hint_audit.logical_gap,
            internal_hint_logical_full=internal_repair_run.hint_audit.logical_full,
            internal_hint_gap_over_1=internal_repair_run.hint_audit.gap_over_1,
            internal_hint_below_five=internal_repair_run.hint_audit.below_five,
            internal_hint_policy_violation_count=internal_repair_run.hint_audit.policy_violation_count,
            internal_hint_structural_issue_count=internal_repair_run.hint_audit.structural_issue_count,
            internal_hint_candidate_variables=internal_repair_run.hint_audit.candidate_variables,
            internal_hint_candidate_variables_hinted=internal_repair_run.hint_audit.candidate_variables_hinted,
            internal_hint_candidate_coverage_rate=internal_repair_run.hint_audit.candidate_coverage_rate,
            internal_hint_auxiliary_variables_hinted=internal_repair_run.hint_audit.auxiliary_variables_derived,
            internal_hint_unhinted_variables=internal_repair_run.hint_audit.unhinted_variables,
            internal_hint_duplicate_keys=internal_repair_run.hint_audit.duplicate_keys,
            internal_hint_out_of_domain_keys=internal_repair_run.hint_audit.out_of_domain_keys,
            model_invariance_before_hint_hash=internal_repair_run.model_before_hint_hash,
            model_invariance_after_hint_hash=internal_repair_run.model_after_hint_hash,
            model_invariance_equal=internal_repair_run.model_invariance_equal,
            model_invariance_without_distance_hash=internal_repair_run.model_invariance_without_distance_hash,
            model_invariance_distance_stripped_hash=internal_repair_run.model_invariance_distance_stripped_hash,
            model_invariance_distance_stripped_equal=internal_repair_run.model_invariance_distance_stripped_equal,
            internal_repair_variable_hash=internal_repair_run.variable_hash,
            internal_repair_domain_hash=internal_repair_run.domain_hash,
            internal_repair_constraint_hash=internal_repair_run.constraint_hash,
            internal_repair_candidate_mapping_hash=internal_repair_run.candidate_mapping_hash,
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


def _validated_internal_repair_result(
    *,
    allocation_input: CanonicalAllocationInput,
    seed: int,
    math_course_ids: tuple[str, ...],
    math_fallback_rules: tuple[MathFallbackRule, ...],
    diagnostics: tuple[CpSatStageDiagnostic, ...],
    internal_repair_run: _InternalRepairRun,
    bootstrap_run: _BootstrapRun,
    max_total_time_seconds: float | None,
    total_budget_exhausted: bool,
    logical_schedule_completion_enabled: bool,
    final_schedule_hard_constraints_enabled: bool,
    solve_status: CpSatSolveStatus | None = None,
) -> CpSatAllocationResult:
    """Preserve a validated repair incumbent when later stages have no answer."""
    if not internal_repair_run.validated or internal_repair_run.build is None or internal_repair_run.solver is None:
        raise ValueError("validated internal repair result is unavailable")
    baseline_result = internal_repair_run.baseline_result
    if baseline_result is None:
        raise ValueError("validated internal repair result has no replayed baseline")
    build = internal_repair_run.build
    solver = internal_repair_run.solver
    final_policy = evaluate_final_schedule_policy(ALGORITHM_NAME, baseline_result.student_outcomes)
    objective_values = _objective_values(
        build,
        solver,
        {},
        logical_schedule_completion_enabled=logical_schedule_completion_enabled,
    )
    if logical_schedule_completion_enabled:
        _validate_logical_completion_consistency(
            baseline_result.student_outcomes,
            final_policy,
            objective_values.logical_assigned_course_count,
            None,
        )
    math_report = evaluate_math_policy(
        allocation_input,
        baseline_result,
        tuple(sorted(math_course_ids)),
        math_fallback_rules,
    )
    proto = build.model.Proto()
    hint = internal_repair_run.hint_audit
    return CpSatAllocationResult(
        algorithm_name=ALGORITHM_NAME,
        seed=int(seed),
        solve_status=solve_status or CpSatSolveStatus.UNKNOWN_WITH_VALIDATED_INCUMBENT,
        lexicographic_optimality_proven=False,
        stage_diagnostics=diagnostics,
        objective_values=objective_values,
        model_stats=CpSatModelStats(
            total_variables=len(proto.variables),
            total_constraints=len(proto.constraints),
            build_time_seconds=round(build.build_time_seconds, 6),
            solve_time_seconds=round(internal_repair_run.solve_time_seconds, 6),
            enrichment_model_variable_count=len(proto.variables),
            enrichment_model_constraint_count=len(proto.constraints),
            bootstrap_enabled=bootstrap_run.status != CpSatBootstrapStatus.DISABLED,
            bootstrap_status=bootstrap_run.status,
            bootstrap_variable_count=_bootstrap_variable_count(bootstrap_run),
            bootstrap_constraint_count=_bootstrap_constraint_count(bootstrap_run),
            bootstrap_build_time_seconds=round(_bootstrap_build_time(bootstrap_run), 6),
            bootstrap_solve_time_seconds=round(bootstrap_run.solve_time_seconds, 6),
            bootstrap_hint_strategy=bootstrap_run.hint_strategy,
            bootstrap_incumbent_found=bootstrap_run.status == CpSatBootstrapStatus.FEASIBLE_FOUND,
            max_total_time_seconds=max_total_time_seconds,
            total_budget_exhausted=total_budget_exhausted,
            skipped_stage_count=sum(item.skipped for item in diagnostics),
            enrichment_build_time_seconds=round(build.build_time_seconds, 6),
            total_build_time_seconds=round(build.build_time_seconds, 6),
            total_solve_time_seconds=round(internal_repair_run.solve_time_seconds, 6),
            warm_start_strategy="internal_repair_feasibility",
            external_hint_used=False,
            stage_to_stage_hint_used=False,
            final_schedule_hard_constraints_enabled=final_schedule_hard_constraints_enabled,
            post_solve_policy_gate_pass=final_policy.summary.final_schedule_policy_pass,
            logical_schedule_completion_objective_enabled=False,
            hint_source=hint.source,
            hint_total_model_variables=hint.total_model_variables,
            hint_variables_supplied=hint.variables_supplied,
            hint_coverage_rate=hint.coverage_rate,
            hint_selected_variables=hint.selected_variables,
            hint_zero_variables=hint.zero_variables,
            hint_unknown_or_unmapped_assignments=hint.unknown_or_unmapped_assignments,
            hint_duplicate_keys=hint.duplicate_keys,
            hint_replay_policy_pass=hint.replay_policy_pass,
            full_model_seed_strategy=hint.source,
            full_model_seed_policy_pass=hint.replay_policy_pass,
            full_model_seed_violation_students=hint.violation_students,
            internal_feasibility_hint_strategy="constrained_first",
            internal_repair_objective_strategy=internal_repair_run.objective_strategy,
            internal_repair_hint_enabled=True,
            internal_repair_status=internal_repair_run.status,
            internal_repair_incumbent_found=True,
            internal_repair_runtime_seconds=round(internal_repair_run.solve_time_seconds, 6),
            internal_repair_time_to_first_solution_seconds=internal_repair_run.time_to_first_solution_seconds,
            internal_repair_hamming_distance=internal_repair_run.hamming_distance,
            internal_repair_greedy_assignments_removed=internal_repair_run.greedy_assignments_removed,
            internal_repair_new_assignments_added=internal_repair_run.new_assignments_added,
            internal_repair_changed_students=internal_repair_run.changed_students,
            internal_repair_changed_requests=internal_repair_run.changed_requests,
            internal_repair_changed_sections=internal_repair_run.changed_sections,
            internal_repair_response_proto_hash=internal_repair_run.response_proto_hash,
            internal_repair_validation_failure=internal_repair_run.validation_failure,
            internal_hint_assignment_hash=hint.assignment_hash,
            internal_hint_primary_assigned=hint.primary_assigned,
            internal_hint_primary_unmet=hint.primary_unmet,
            internal_hint_logical_assigned=hint.logical_assigned,
            internal_hint_logical_gap=hint.logical_gap,
            internal_hint_logical_full=hint.logical_full,
            internal_hint_gap_over_1=hint.gap_over_1,
            internal_hint_below_five=hint.below_five,
            internal_hint_policy_violation_count=hint.policy_violation_count,
            internal_hint_structural_issue_count=hint.structural_issue_count,
            internal_hint_candidate_variables=hint.candidate_variables,
            internal_hint_candidate_variables_hinted=hint.candidate_variables_hinted,
            internal_hint_candidate_coverage_rate=hint.candidate_coverage_rate,
            internal_hint_auxiliary_variables_hinted=hint.auxiliary_variables_derived,
            internal_hint_unhinted_variables=hint.unhinted_variables,
            internal_hint_duplicate_keys=hint.duplicate_keys,
            internal_hint_out_of_domain_keys=hint.out_of_domain_keys,
            model_invariance_before_hint_hash=internal_repair_run.model_before_hint_hash,
            model_invariance_after_hint_hash=internal_repair_run.model_after_hint_hash,
            model_invariance_equal=internal_repair_run.model_invariance_equal,
            model_invariance_without_distance_hash=internal_repair_run.model_invariance_without_distance_hash,
            model_invariance_distance_stripped_hash=internal_repair_run.model_invariance_distance_stripped_hash,
            model_invariance_distance_stripped_equal=internal_repair_run.model_invariance_distance_stripped_equal,
            internal_repair_variable_hash=internal_repair_run.variable_hash,
            internal_repair_domain_hash=internal_repair_run.domain_hash,
            internal_repair_constraint_hash=internal_repair_run.constraint_hash,
            internal_repair_candidate_mapping_hash=internal_repair_run.candidate_mapping_hash,
        ),
        assignments=baseline_result.assignments,
        mandatory_fallback_outcomes=baseline_result.mandatory_fallback_outcomes,
        request_outcomes=baseline_result.request_outcomes,
        student_outcomes=baseline_result.student_outcomes,
        policy_report=final_policy,
        math_policy_report=math_report,
        section_roster_summary=baseline_result.section_roster_summary,
        consistency_issues=baseline_result.consistency_issues,
    )


def _build_core_cp_sat_model(
    allocation_input: CanonicalAllocationInput,
    fallback_plans: tuple[_FallbackPlan, ...],
    math_course_ids: tuple[str, ...],
    seed: int,
) -> _ModelBuild:
    return _build_model(
        allocation_input,
        fallback_plans,
        math_course_ids,
        seed,
        model_scope=CpSatModelScope.CORE,
        include_alternates=False,
        include_schedule_completion=False,
        fixed_core_values=None,
    )


def _build_enrichment_cp_sat_model(
    allocation_input: CanonicalAllocationInput,
    fallback_plans: tuple[_FallbackPlan, ...],
    math_course_ids: tuple[str, ...],
    seed: int,
    fixed_core_values: dict[CpSatStageName, int],
    *,
    enforce_final_schedule_hard_constraints: bool = True,
) -> _ModelBuild:
    return _build_model(
        allocation_input,
        fallback_plans,
        math_course_ids,
        seed,
        model_scope=CpSatModelScope.ENRICHMENT,
        include_alternates=True,
        include_schedule_completion=True,
        fixed_core_values=fixed_core_values,
        enforce_final_schedule_hard_constraints=enforce_final_schedule_hard_constraints,
    )


def _build_full_feasibility_cp_sat_model(
    allocation_input: CanonicalAllocationInput,
    fallback_plans: tuple[_FallbackPlan, ...],
    math_course_ids: tuple[str, ...],
    seed: int,
) -> _ModelBuild:
    return _build_model(
        allocation_input,
        fallback_plans,
        math_course_ids,
        seed,
        model_scope=CpSatModelScope.ENRICHMENT,
        include_alternates=True,
        include_schedule_completion=True,
        fixed_core_values=None,
        enforce_final_schedule_hard_constraints=True,
    )


def _build_feasibility_bootstrap_model(
    allocation_input: CanonicalAllocationInput,
) -> _BootstrapBuild:
    started = time.perf_counter()
    model = cp_model.CpModel()
    requests_by_key = {
        request.request_key: request
        for request in allocation_input.logical_requests
        if request.request_type == "primary"
    }
    candidate_index: dict[str, tuple[str, ...]] = {}
    assignment_vars: dict[_VariableKey, cp_model.IntVar] = {}
    assigned_vars: dict[str, cp_model.LinearExpr] = {}
    for request_key in sorted(requests_by_key):
        request = requests_by_key[request_key]
        raw_candidates = allocation_input.candidate_index.get(request_key, ())
        candidates = tuple(raw_candidates)
        _validate_candidates_for_request(allocation_input, request, candidates)
        candidate_index[request_key] = candidates
        candidate_vars: list[cp_model.IntVar] = []
        for section_id in candidates:
            var = model.NewBoolVar(f"boot_x__{_safe_name(request_key)}__{_safe_name(section_id)}")
            assignment_vars[_VariableKey(request_key, section_id)] = var
            candidate_vars.append(var)
        if len(candidate_vars) == 1:
            assigned_vars[request_key] = candidate_vars[0]
        elif candidate_vars:
            assigned = model.NewBoolVar(f"boot_assigned__{_safe_name(request_key)}")
            assigned_vars[request_key] = assigned
            model.Add(sum(candidate_vars) == assigned)
        else:
            assigned_vars[request_key] = 0
    _add_section_capacity_constraints(model, allocation_input, assignment_vars)
    _add_student_period_constraints(model, allocation_input, requests_by_key, assignment_vars)
    _add_student_target_constraints(model, allocation_input, requests_by_key, assignment_vars)
    _add_duplicate_identity_constraints(model, allocation_input, requests_by_key, assigned_vars)
    _add_fairness_hard_constraints(model, allocation_input, assigned_vars)
    return _BootstrapBuild(
        model=model,
        assignment_vars=assignment_vars,
        assigned_vars=assigned_vars,
        requests_by_key=requests_by_key,
        candidate_index=candidate_index,
        build_time_seconds=time.perf_counter() - started,
    )


def _validate_candidate_index(allocation_input: CanonicalAllocationInput) -> None:
    for request in allocation_input.logical_requests:
        candidates = tuple(allocation_input.candidate_index.get(request.request_key, ()))
        _validate_candidates_for_request(allocation_input, request, candidates)


def _validate_candidates_for_request(
    allocation_input: CanonicalAllocationInput,
    request: LogicalRequest,
    candidates: tuple[str, ...],
) -> None:
    if len(candidates) != len(set(candidates)):
        raise ValueError(f"duplicate candidate sections for {request.request_key}")
    for section_id in candidates:
        section = allocation_input.logical_sections_by_id.get(section_id)
        if section is None:
            raise ValueError(f"dangling candidate section {section_id} for {request.request_key}")
        if section.logical_block_id != request.candidate_key:
            raise ValueError(f"candidate section {section_id} does not match {request.request_key}")
        if section.period_units != request.period_units:
            raise ValueError(f"candidate section {section_id} has wrong period units for {request.request_key}")
        if tuple(sorted(section.course_ids)) != tuple(sorted(request.course_ids)):
            raise ValueError(f"candidate section {section_id} has wrong course identity for {request.request_key}")


def _run_internal_repair_feasibility(
    allocation_input: CanonicalAllocationInput,
    fallback_plans: tuple[_FallbackPlan, ...],
    math_course_ids: tuple[str, ...],
    *,
    seed: int,
    max_time_seconds: float,
    num_search_workers: int,
    log_search_progress: bool,
    budget: _GlobalTimeBudget,
    objective_strategy: str = "none",
    stop_after_first_solution: bool = False,
) -> _InternalRepairRun:
    """Try the unchanged full hard model with an internal Greedy hint.

    The constrained-first assignment is deliberately treated as a hint only.
    It may violate policy, but it must be structurally replayable.  Only the
    CP-SAT response can become a validated incumbent or final assignment.
    """
    hint_seed = _constrained_first_full_hint_seed(
        allocation_input,
        tuple(
            MathFallbackRule(
                plan.source_request.candidate_key,
                plan.fallback_request.candidate_key,
                "mandatory_fallback",
                True,
                "internal_repair",
            )
            for plan in fallback_plans
        ),
        math_course_ids,
        seed,
    )
    build = _build_full_feasibility_cp_sat_model(
        allocation_input,
        fallback_plans,
        math_course_ids,
        seed,
    )
    effective_limit = budget.effective_limit(max_time_seconds)
    remaining_at_start = budget.remaining()
    if effective_limit <= 0:
        return _InternalRepairRun(
            build=build,
            solver=None,
            status=CpSatSolveStatus.SKIPPED,
            diagnostic=_skipped_diagnostic(
                CpSatStageName.INTERNAL_REPAIR_FEASIBILITY,
                CpSatModelScope.ENRICHMENT,
                "time_budget_exhausted",
                remaining_at_start,
                effective_limit,
            ),
            selected_keys=(),
            solve_time_seconds=0.0,
            hint_audit=hint_seed.internal_audit or _HintAudit(source=hint_seed.source),
            model_invariance_equal=None,
            objective_strategy=objective_strategy,
        )

    without_distance_hash = _model_proto_without_solution_hint_hash(build.model)
    if objective_strategy == "hamming_to_constrained_first":
        distance = _hamming_distance_expression(build, hint_seed.keys)
        build.model.Minimize(distance)
    before_hint_hash = _model_proto_without_solution_hint_hash(build.model)
    # The objective is appended after the base model is built and before any
    # hint is written. Reusing the base hash avoids a second huge proto copy.
    distance_stripped_hash = without_distance_hash
    variable_hash, domain_hash, constraint_hash, candidate_mapping_hash = _model_component_hashes(build)
    mapped_keys = _mapped_hint_keys(build.assignment_vars, hint_seed.keys)
    values, variables = _full_model_hint_values(build, mapped_keys, require_complete=False)
    hint_audit = _audit_full_model_hint(build, hint_seed, values)
    hint_audit = replace(
        hint_audit,
        assignment_hash=hint_seed.internal_audit.assignment_hash if hint_seed.internal_audit else "",
        candidate_variables=len(build.assignment_vars),
        candidate_variables_hinted=sum(index in values for index in (var.Index() for var in build.assignment_vars.values())),
        candidate_coverage_rate=(
            sum(index in values for index in (var.Index() for var in build.assignment_vars.values()))
            / len(build.assignment_vars)
            if build.assignment_vars
            else 1.0
        ),
        unhinted_variables=max(len(build.model.Proto().variables) - len(values), 0),
        out_of_domain_keys=sum(key not in build.assignment_vars for key in set(hint_seed.keys)),
        primary_assigned=hint_seed.internal_audit.primary_assigned if hint_seed.internal_audit else None,
        primary_unmet=hint_seed.internal_audit.primary_unmet if hint_seed.internal_audit else None,
        logical_assigned=hint_seed.internal_audit.logical_assigned if hint_seed.internal_audit else None,
        logical_gap=hint_seed.internal_audit.logical_gap if hint_seed.internal_audit else None,
        logical_full=hint_seed.internal_audit.logical_full if hint_seed.internal_audit else None,
        gap_over_1=hint_seed.internal_audit.gap_over_1 if hint_seed.internal_audit else None,
        below_five=hint_seed.internal_audit.below_five if hint_seed.internal_audit else None,
        policy_violation_count=hint_seed.internal_audit.policy_violation_count if hint_seed.internal_audit else None,
        structural_issue_count=hint_seed.internal_audit.structural_issue_count if hint_seed.internal_audit else 0,
        runtime_seconds=hint_seed.internal_audit.runtime_seconds if hint_seed.internal_audit else 0.0,
    )
    _apply_complete_model_hint(
        build,
        mapped_keys,
        values=values,
        variables=variables,
        allow_partial=True,
    )
    after_hash = _model_proto_without_solution_hint_hash(build.model)
    solver = _new_solver(
        effective_limit,
        num_search_workers,
        log_search_progress,
        seed,
        repair_hint=True,
    )
    solver.parameters.stop_after_first_solution = bool(stop_after_first_solution)
    solve_started = time.perf_counter()
    capture = _FirstSolutionCapture(build) if stop_after_first_solution else None
    raw_status = (
        solver.solve(build.model, capture)
        if capture is not None
        else solver.Solve(build.model)
    )
    solve_time = time.perf_counter() - solve_started
    budget.refresh()
    status = _solve_status(raw_status)
    diagnostic = _stage_diagnostic(
        CpSatStageName.INTERNAL_REPAIR_FEASIBILITY,
        CpSatModelScope.ENRICHMENT,
        status,
        solver,
        conditional_on_unproven_incumbent=False,
        fixed_higher_priority_values=(),
        objective_descriptor_hash=_objective_descriptor_hash(build.model),
        remaining_global_budget_at_start_seconds=remaining_at_start,
        effective_time_limit_seconds=effective_limit,
        repair_hint_enabled=True,
        hint_assignment_hash=hint_audit.assignment_hash,
    )
    selected = _selected_assignments(build, solver) if status in {CpSatSolveStatus.OPTIMAL, CpSatSolveStatus.FEASIBLE} else ()
    first_solution_time = capture.time_to_first_solution_seconds if capture is not None else None
    validation_failure = ""
    baseline_result = None
    validated = False
    if status in {CpSatSolveStatus.OPTIMAL, CpSatSolveStatus.FEASIBLE}:
        try:
            state = _replay_solution(allocation_input, build, selected)
            request_outcomes = _build_request_outcomes(allocation_input, build, solver, state)
            fallback_outcomes = _build_fallback_outcomes(build, solver, state)
            baseline_result = _finalize_baseline_result(
                ALGORITHM_NAME,
                allocation_input,
                seed,
                (),
                state,
                request_outcomes,
                fallback_outcomes,
            )
            policy_report = evaluate_final_schedule_policy(ALGORITHM_NAME, baseline_result.student_outcomes)
            validated = policy_report.summary.final_schedule_policy_pass and not baseline_result.consistency_issues
            if not validated:
                validation_failure = (
                    "policy_or_consistency_failure: "
                    f"policy_pass={policy_report.summary.final_schedule_policy_pass}, "
                    f"consistency_issues={len(baseline_result.consistency_issues)}"
                )
        except Exception as exc:
            validation_failure = f"response_replay_failure: {type(exc).__name__}: {exc}"
    distance_stats = _assignment_distance_stats(
        allocation_input,
        hint_seed.keys,
        selected,
        build.requests_by_key,
    )
    return _InternalRepairRun(
        build=build,
        solver=solver if status in {CpSatSolveStatus.OPTIMAL, CpSatSolveStatus.FEASIBLE} else None,
        status=status,
        diagnostic=diagnostic,
        selected_keys=selected if validated else (),
        solve_time_seconds=solve_time,
        hint_audit=hint_audit,
        model_before_hint_hash=before_hint_hash,
        model_after_hint_hash=after_hash,
        model_invariance_equal=without_distance_hash == after_hash if objective_strategy == "none" else True,
        model_invariance_without_distance_hash=without_distance_hash,
        model_invariance_distance_stripped_hash=distance_stripped_hash,
        model_invariance_distance_stripped_equal=without_distance_hash == distance_stripped_hash,
        variable_hash=variable_hash,
        domain_hash=domain_hash,
        constraint_hash=constraint_hash,
        candidate_mapping_hash=candidate_mapping_hash,
        objective_strategy=objective_strategy,
        time_to_first_solution_seconds=first_solution_time,
        hamming_distance=distance_stats["hamming_distance"],
        greedy_assignments_removed=distance_stats["greedy_assignments_removed"],
        new_assignments_added=distance_stats["new_assignments_added"],
        changed_students=distance_stats["changed_students"],
        changed_requests=distance_stats["changed_requests"],
        changed_sections=distance_stats["changed_sections"],
        response_proto_hash=diagnostic.response_proto_hash if diagnostic else "",
        validation_failure=validation_failure,
        validated=validated,
        baseline_result=baseline_result if validated else None,
    )


def _hamming_distance_expression(
    build: _ModelBuild,
    greedy_keys: tuple[_VariableKey, ...],
) -> cp_model.LinearExpr:
    """Build an unweighted symmetric-difference objective over candidates only."""
    selected = set(greedy_keys)
    terms = [
        (1 - variable) if key in selected else variable
        for key, variable in sorted(build.assignment_vars.items(), key=lambda item: (item[0].request_key, item[0].section_id))
    ]
    return cp_model.LinearExpr.Sum(terms) if terms else 0


def _assignment_distance_stats(
    allocation_input: CanonicalAllocationInput,
    greedy_keys: tuple[_VariableKey, ...],
    selected_keys: tuple[_VariableKey, ...],
    requests_by_key: dict[str, LogicalRequest] | None = None,
) -> dict[str, int | None]:
    if not selected_keys:
        return {
            "hamming_distance": None,
            "greedy_assignments_removed": None,
            "new_assignments_added": None,
            "changed_students": None,
            "changed_requests": None,
            "changed_sections": None,
        }
    greedy = set(greedy_keys)
    selected = set(selected_keys)
    removed = greedy - selected
    added = selected - greedy
    request_by_key = requests_by_key or {
        request.request_key: request for request in allocation_input.logical_requests
    }
    changed_request_keys = {key.request_key for key in removed | added}
    changed_students = {
        request_by_key[key.request_key].student_id
        for key in removed | added
        if key.request_key in request_by_key
    }
    changed_sections = {key.section_id for key in removed | added}
    return {
        "hamming_distance": len(removed) + len(added),
        "greedy_assignments_removed": len(removed),
        "new_assignments_added": len(added),
        "changed_students": len(changed_students),
        "changed_requests": len(changed_request_keys),
        "changed_sections": len(changed_sections),
    }


def _model_proto_without_objective_and_solution_hint_hash(model: cp_model.CpModel) -> str:
    proto = copy.deepcopy(model.Proto())
    proto.solution_hint.vars.clear()
    proto.solution_hint.values.clear()
    proto.objective.vars.clear()
    proto.objective.coeffs.clear()
    proto.objective.clear_offset()
    proto.objective.clear_scaling_factor()
    proto.objective.clear_integer_before_offset()
    proto.objective.clear_integer_after_offset()
    proto.objective.clear_integer_scaling_factor()
    proto.objective.clear_scaling_was_exact()
    text = str(proto).replace("objective {\n}\n", "").replace("solution_hint {\n}\n", "")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _model_component_hashes(
    build: _ModelBuild,
) -> tuple[str, str, str, str]:
    proto = build.model.Proto()
    def digest(rows: Iterable[object]) -> str:
        hasher = hashlib.sha256()
        for row in rows:
            hasher.update(repr(row).encode("utf-8"))
            hasher.update(b"\0")
        return hasher.hexdigest()

    variable_hash = digest(
        (str(item.name), tuple(int(value) for value in item.domain))
        for item in proto.variables
    )
    domain_hash = digest(
        tuple(int(value) for value in item.domain)
        for item in proto.variables
    )
    constraint_hash = digest(str(item) for item in proto.constraints)
    candidate_mapping_hash = digest(
        (key.request_key, key.section_id, int(variable.Index()))
        for key, variable in sorted(
            build.assignment_vars.items(),
            key=lambda item: (item[0].request_key, item[0].section_id),
        )
    )
    return variable_hash, domain_hash, constraint_hash, candidate_mapping_hash


def _model_invalid_bootstrap_run(hint_strategy: str) -> _BootstrapRun:
    return _BootstrapRun(
        build=None,
        solver=None,
        status=CpSatBootstrapStatus.MODEL_INVALID,
        diagnostic=CpSatStageDiagnostic(
            stage_name=CpSatStageName.FEASIBILITY_BOOTSTRAP,
            model_scope=CpSatModelScope.BOOTSTRAP,
            status=CpSatSolveStatus.MODEL_INVALID,
            objective_value=None,
            best_objective_bound=None,
            wall_time_seconds=0.0,
            conflicts=0,
            branches=0,
            optimum_proven=False,
        ),
        selected_keys=(),
        solve_time_seconds=0.0,
        time_to_first_hard_feasible_solution_seconds=None,
        hint_strategy=hint_strategy,
    )


def _run_feasibility_bootstrap(
    allocation_input: CanonicalAllocationInput,
    *,
    seed: int,
    max_time_seconds: float,
    num_search_workers: int,
    log_search_progress: bool,
    initial_hint_keys: tuple[_VariableKey, ...],
    budget: _GlobalTimeBudget,
) -> _BootstrapRun:
    try:
        build = _build_feasibility_bootstrap_model(allocation_input)
    except ValueError:
        return _model_invalid_bootstrap_run(_bootstrap_hint_strategy(initial_hint_keys))
    effective_limit = budget.effective_limit(max_time_seconds)
    remaining_at_start = budget.remaining()
    if effective_limit <= 0:
        return _BootstrapRun(
            build=build,
            solver=None,
            status=CpSatBootstrapStatus.UNKNOWN_NO_INCUMBENT,
            diagnostic=_skipped_diagnostic(
                CpSatStageName.FEASIBILITY_BOOTSTRAP,
                CpSatModelScope.BOOTSTRAP,
                "time_budget_exhausted",
                remaining_at_start,
                effective_limit,
            ),
            selected_keys=(),
            solve_time_seconds=0.0,
            time_to_first_hard_feasible_solution_seconds=None,
            hint_strategy=_bootstrap_hint_strategy(initial_hint_keys),
            budget_exhausted=True,
        )
    if initial_hint_keys:
        _apply_complete_key_hint(
            build.model,
            build.assignment_vars,
            _mapped_hint_keys(build.assignment_vars, initial_hint_keys),
        )
    solver = _new_solver(effective_limit, num_search_workers, log_search_progress, seed)
    solver.parameters.stop_after_first_solution = True
    solve_started = time.perf_counter()
    raw_status = solver.Solve(build.model)
    solve_time = time.perf_counter() - solve_started
    budget.refresh()
    status = _solve_status(raw_status)
    diagnostic = _stage_diagnostic(
        CpSatStageName.FEASIBILITY_BOOTSTRAP,
        CpSatModelScope.BOOTSTRAP,
        status,
        solver,
        conditional_on_unproven_incumbent=False,
        fixed_higher_priority_values=(),
        objective_descriptor_hash=_objective_descriptor_hash(build.model),
        remaining_global_budget_at_start_seconds=remaining_at_start,
        effective_time_limit_seconds=effective_limit,
    )
    if status in {CpSatSolveStatus.OPTIMAL, CpSatSolveStatus.FEASIBLE}:
        selected = _selected_bootstrap_assignments(build, solver)
        try:
            _replay_bootstrap_solution(allocation_input, build, selected)
        except RuntimeError:
            return _BootstrapRun(
                build=build,
                solver=solver,
                status=CpSatBootstrapStatus.MODEL_INVALID,
                diagnostic=diagnostic,
                selected_keys=(),
                solve_time_seconds=solve_time,
                time_to_first_hard_feasible_solution_seconds=None,
                hint_strategy=_bootstrap_hint_strategy(initial_hint_keys),
            )
        return _BootstrapRun(
            build=build,
            solver=solver,
            status=CpSatBootstrapStatus.FEASIBLE_FOUND,
            diagnostic=diagnostic,
            selected_keys=selected,
            solve_time_seconds=solve_time,
            time_to_first_hard_feasible_solution_seconds=solve_time,
            hint_strategy=_bootstrap_hint_strategy(initial_hint_keys),
        )
    if status == CpSatSolveStatus.INFEASIBLE:
        return _BootstrapRun(
            build=build,
            solver=solver,
            status=CpSatBootstrapStatus.INFEASIBLE,
            diagnostic=diagnostic,
            selected_keys=(),
            solve_time_seconds=solve_time,
            time_to_first_hard_feasible_solution_seconds=None,
            hint_strategy=_bootstrap_hint_strategy(initial_hint_keys),
        )
    if status == CpSatSolveStatus.MODEL_INVALID:
        return _BootstrapRun(
            build=build,
            solver=solver,
            status=CpSatBootstrapStatus.MODEL_INVALID,
            diagnostic=diagnostic,
            selected_keys=(),
            solve_time_seconds=solve_time,
            time_to_first_hard_feasible_solution_seconds=None,
            hint_strategy=_bootstrap_hint_strategy(initial_hint_keys),
        )
    return _BootstrapRun(
        build=build,
        solver=None,
        status=CpSatBootstrapStatus.UNKNOWN_NO_INCUMBENT,
        diagnostic=diagnostic,
        selected_keys=(),
        solve_time_seconds=solve_time,
        time_to_first_hard_feasible_solution_seconds=None,
        hint_strategy=_bootstrap_hint_strategy(initial_hint_keys),
        budget_exhausted=budget.exhausted,
    )


def _run_full_model_feasibility_incumbent(
    allocation_input: CanonicalAllocationInput,
    fallback_plans: tuple[_FallbackPlan, ...],
    math_course_ids: tuple[str, ...],
    *,
    seed: int,
    max_time_seconds: float,
    num_search_workers: int,
    log_search_progress: bool,
    initial_hint_seed: _HintSeed,
    budget: _GlobalTimeBudget,
) -> _FullModelFeasibilityRun:
    build = _build_full_feasibility_cp_sat_model(allocation_input, fallback_plans, math_course_ids, seed)
    hint_audit = _HintAudit(source=initial_hint_seed.source)
    effective_limit = budget.effective_limit(max_time_seconds)
    remaining_at_start = budget.remaining()
    if effective_limit <= 0:
        return _FullModelFeasibilityRun(
            build=build,
            solver=None,
            status=CpSatSolveStatus.SKIPPED,
            diagnostic=_skipped_diagnostic(
                CpSatStageName.FULL_MODEL_FEASIBILITY_INCUMBENT,
                CpSatModelScope.ENRICHMENT,
                "time_budget_exhausted",
                remaining_at_start,
                effective_limit,
            ),
            selected_keys=(),
            solve_time_seconds=0.0,
            budget_exhausted=True,
            hint_audit=hint_audit,
        )
    if initial_hint_seed.keys:
        mapped_keys = _mapped_hint_keys(build.assignment_vars, initial_hint_seed.keys)
        values, variables = _full_model_hint_values(build, mapped_keys)
        hint_audit = _audit_full_model_hint(build, initial_hint_seed, values)
        _apply_complete_model_hint(build, mapped_keys, values=values, variables=variables)
    solver = _new_solver(effective_limit, num_search_workers, log_search_progress, seed)
    solver.parameters.stop_after_first_solution = True
    solve_started = time.perf_counter()
    raw_status = solver.Solve(build.model)
    solve_time = time.perf_counter() - solve_started
    budget.refresh()
    status = _solve_status(raw_status)
    diagnostic = _stage_diagnostic(
        CpSatStageName.FULL_MODEL_FEASIBILITY_INCUMBENT,
        CpSatModelScope.ENRICHMENT,
        status,
        solver,
        conditional_on_unproven_incumbent=False,
        fixed_higher_priority_values=(),
        objective_descriptor_hash=_objective_descriptor_hash(build.model),
        remaining_global_budget_at_start_seconds=remaining_at_start,
        effective_time_limit_seconds=effective_limit,
    )
    if status in {CpSatSolveStatus.OPTIMAL, CpSatSolveStatus.FEASIBLE}:
        selected = _selected_assignments(build, solver)
        try:
            state = _replay_solution(allocation_input, build, selected)
            request_outcomes = _build_request_outcomes(allocation_input, build, solver, state)
            fallback_outcomes = _build_fallback_outcomes(build, solver, state)
            baseline_result = _finalize_baseline_result(
                ALGORITHM_NAME,
                allocation_input,
                seed,
                (),
                state,
                request_outcomes,
                fallback_outcomes,
            )
            policy_report = evaluate_final_schedule_policy(ALGORITHM_NAME, baseline_result.student_outcomes)
        except RuntimeError:
            return _FullModelFeasibilityRun(
                build=build,
                solver=solver,
                status=CpSatSolveStatus.MODEL_INVALID,
                diagnostic=diagnostic,
                selected_keys=(),
                solve_time_seconds=solve_time,
                hint_audit=hint_audit,
            )
        if not policy_report.summary.final_schedule_policy_pass:
            return _FullModelFeasibilityRun(
                build=build,
                solver=solver,
                status=CpSatSolveStatus.MODEL_INVALID,
                diagnostic=diagnostic,
                selected_keys=(),
                solve_time_seconds=solve_time,
                hint_audit=hint_audit,
            )
        return _FullModelFeasibilityRun(
            build=build,
            solver=solver,
            status=status,
            diagnostic=diagnostic,
            selected_keys=selected,
            solve_time_seconds=solve_time,
            hint_audit=hint_audit,
        )
    return _FullModelFeasibilityRun(
        build=build,
        solver=None,
        status=status,
        diagnostic=diagnostic,
        selected_keys=(),
        solve_time_seconds=solve_time,
        budget_exhausted=budget.exhausted,
        hint_audit=hint_audit,
    )


def _build_model(
    allocation_input: CanonicalAllocationInput,
    fallback_plans: tuple[_FallbackPlan, ...],
    math_course_ids: tuple[str, ...],
    seed: int,
    *,
    model_scope: CpSatModelScope,
    include_alternates: bool,
    include_schedule_completion: bool,
    fixed_core_values: dict[CpSatStageName, int] | None,
    enforce_final_schedule_hard_constraints: bool = False,
) -> _ModelBuild:
    started = time.perf_counter()
    model = cp_model.CpModel()
    requests_by_key = {
        request.request_key: request
        for request in allocation_input.logical_requests
        if include_alternates or request.request_type != "alternate"
    }
    candidate_index = {key: tuple(value) for key, value in allocation_input.candidate_index.items()}
    for plan in fallback_plans:
        requests_by_key[plan.fallback_request.request_key] = plan.fallback_request
        candidate_index[plan.fallback_request.request_key] = plan.candidates

    assignment_vars: dict[_VariableKey, cp_model.IntVar] = {}
    assigned_vars: dict[str, cp_model.LinearExpr] = {}
    for request_key in sorted(requests_by_key):
        request = requests_by_key[request_key]
        candidates = tuple(candidate_index.get(request_key, ()))
        _validate_candidates_for_request(allocation_input, request, candidates)
        candidate_vars: list[cp_model.IntVar] = []
        for section_id in candidates:
            var = model.NewBoolVar(f"x__{_safe_name(request_key)}__{_safe_name(section_id)}")
            assignment_vars[_VariableKey(request_key, section_id)] = var
            candidate_vars.append(var)
        if len(candidate_vars) == 1:
            assigned_vars[request_key] = candidate_vars[0]
        elif candidate_vars:
            assigned = model.NewBoolVar(f"assigned__{_safe_name(request_key)}")
            assigned_vars[request_key] = assigned
            model.Add(sum(candidate_vars) == assigned)
        else:
            assigned_vars[request_key] = 0
        if request.period_units != allocation_input.courses_by_id[request.course_ids[0]].period_units:
            # Canonicalization should already prevent this for source requests;
            # fallback requests are built from the same metadata.
            if candidate_vars:
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
    if include_schedule_completion and enforce_final_schedule_hard_constraints:
        _add_final_schedule_hard_constraints(model, allocation_input, requests_by_key, assigned_vars)
    if include_schedule_completion:
        fully_scheduled_vars, remaining_exprs = _add_schedule_completion_vars(
            model,
            allocation_input,
            requests_by_key,
            assignment_vars,
        )
    else:
        fully_scheduled_vars = {}
        remaining_exprs = {}
    logical_assigned_course_vars, logical_assigned_course_total_var = _add_logical_assigned_course_vars(
        model,
        allocation_input,
        requests_by_key,
        assigned_vars,
    )
    stage_exprs, primary_base = _stage_expressions(
        allocation_input,
        requests_by_key,
        assigned_vars,
        math_violation_vars,
        logical_assigned_course_vars,
        logical_assigned_course_total_var,
        fully_scheduled_vars,
        remaining_exprs,
        assignment_vars,
        seed,
    )
    if fixed_core_values:
        for stage_name, value in fixed_core_values.items():
            model.Add(stage_exprs[stage_name] == value)
    return _ModelBuild(
        model_scope=model_scope,
        model=model,
        assignment_vars=assignment_vars,
        assigned_vars=assigned_vars,
        math_violation_vars=math_violation_vars,
        logical_assigned_course_vars=logical_assigned_course_vars,
        logical_assigned_course_total_var=logical_assigned_course_total_var,
        fully_scheduled_vars=fully_scheduled_vars,
        fallback_plans=fallback_plans,
        math_course_ids=math_course_ids,
        requests_by_key=requests_by_key,
        students_by_id=dict(allocation_input.students_by_id),
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
    assigned_vars: dict[str, cp_model.LinearExpr],
) -> None:
    by_student_identity: dict[tuple[str, str], list[cp_model.LinearExpr]] = defaultdict(list)
    for request in requests_by_key.values():
        by_student_identity[(request.student_id, request.candidate_key)].append(assigned_vars[request.request_key])
    for student in allocation_input.students:
        identities = {
            identity
            for sid, identity in by_student_identity
            if sid == student.student_id
        }
        for identity in identities:
            terms = by_student_identity[(student.student_id, identity)]
            if len(terms) > 1:
                model.Add(sum(terms) <= 1)


def _add_fallback_constraints(
    model: cp_model.CpModel,
    fallback_plans: tuple[_FallbackPlan, ...],
    assigned_vars: dict[str, cp_model.LinearExpr],
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
    assigned_vars: dict[str, cp_model.LinearExpr],
    math_course_ids: tuple[str, ...],
) -> tuple[dict[str, cp_model.IntVar], cp_model.IntVar]:
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
        violation = model.NewBoolVar(f"math_violation__{_safe_name(student.student_id)}")
        model.Add(sum(coverage_terms) + violation >= 1)
        for term in coverage_terms:
            model.Add(term + violation <= 1)
        violations[student.student_id] = violation
    return violations


def _add_fairness_hard_constraints(
    model: cp_model.CpModel,
    allocation_input: CanonicalAllocationInput,
    assigned_vars: dict[str, cp_model.LinearExpr],
) -> None:
    demand = Counter(request.candidate_key for request in allocation_input.logical_requests if request.request_type == "primary")
    high_demand = {key for key, count in demand.items() if count > HIGH_DEMAND_PRIMARY_THRESHOLD}
    for student in allocation_input.students:
        primary_assigned = [assigned_vars[request.request_key] for request in student.primary_requests]
        primary_count = len(student.primary_requests)
        if student.priority_protected:
            model.Add(primary_count - sum(primary_assigned) <= MAXIMUM_PROTECTED_PRIMARY_UNMET_COUNT)
        else:
            model.Add(primary_count - sum(primary_assigned) <= MAXIMUM_ORDINARY_PRIMARY_UNMET_COUNT)
        for request in student.primary_requests:
            if request.candidate_key in high_demand:
                model.Add(assigned_vars[request.request_key] == 1)


def _add_final_schedule_hard_constraints(
    model: cp_model.CpModel,
    allocation_input: CanonicalAllocationInput,
    requests_by_key: dict[str, LogicalRequest],
    assigned_vars: dict[str, cp_model.LinearExpr],
) -> None:
    by_student: dict[str, list[cp_model.LinearExpr]] = defaultdict(list)
    for request in requests_by_key.values():
        by_student[request.student_id].append(assigned_vars[request.request_key])
    for student in allocation_input.students:
        assigned_logical_courses = sum(by_student.get(student.student_id, ()))
        model.Add(assigned_logical_courses >= MINIMUM_ASSIGNED_LOGICAL_COURSE_COUNT)
        model.Add(assigned_logical_courses >= student.target_period_units - MAXIMUM_SCHEDULE_GAP_COUNT)


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


def _add_logical_assigned_course_vars(
    model: cp_model.CpModel,
    allocation_input: CanonicalAllocationInput,
    requests_by_key: dict[str, LogicalRequest],
    assigned_vars: dict[str, cp_model.LinearExpr],
) -> dict[str, cp_model.IntVar]:
    by_student: dict[str, list[cp_model.LinearExpr]] = defaultdict(list)
    for request in requests_by_key.values():
        by_student[request.student_id].append(assigned_vars[request.request_key])
    result: dict[str, cp_model.IntVar] = {}
    for student in allocation_input.students:
        count = model.NewIntVar(
            0,
            student.target_period_units,
            f"logical_assigned__{_safe_name(student.student_id)}",
        )
        model.Add(count == sum(by_student.get(student.student_id, ())))
        result[student.student_id] = count
    # Keep the aggregate objective's theoretical upper bound explicit.  The
    # per-student domains are authoritative; this redundant aggregate bound
    # also keeps CP-SAT's reported best bound inside the same metric range.
    target_total = sum(student.target_period_units for student in allocation_input.students)
    model.Add(sum(result.values()) <= target_total)
    total = model.NewIntVar(0, target_total, "logical_assigned_total")
    model.Add(total == sum(result.values()))
    return result, total


def _stage_expressions(
    allocation_input: CanonicalAllocationInput,
    requests_by_key: dict[str, LogicalRequest],
    assigned_vars: dict[str, cp_model.LinearExpr],
    math_violation_vars: dict[str, cp_model.IntVar],
    logical_assigned_course_vars: dict[str, cp_model.IntVar],
    logical_assigned_course_total_var: cp_model.IntVar,
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
    alternates_by_rank: dict[int, list[cp_model.LinearExpr]] = defaultdict(list)
    for request in allocation_input.logical_requests:
        if request.request_type == "alternate" and request.request_rank is not None and request.request_key in assigned_vars:
            alternates_by_rank[request.request_rank].append(assigned_vars[request.request_key])
    # Logical completion counts logical request assignments, not period units.
    # With Final Schedule Policy Gate v1 enabled (maximum logical gap == 1),
    # maximizing this expression is equivalent to minimizing total logical
    # schedule gap / gap-student count. The equivalence is intentionally not
    # assumed by tests that disable the final hard schedule constraints.
    logical_assigned_courses = logical_assigned_course_total_var
    tie_break = _seeded_tie_break_expr(assignment_vars, seed)
    return (
        {
            CpSatStageName.MATH_COVERAGE: sum(math_violation_vars.values()),
            CpSatStageName.PRIMARY_SATISFACTION: primary_penalty,
            CpSatStageName.PRIMARY_UNMET_COUNT: primary_unmet_count,
            CpSatStageName.PRIMARY_UNMET_PERIOD_UNITS: primary_unmet_units,
            CpSatStageName.LOGICAL_SCHEDULE_COMPLETION: logical_assigned_courses,
            CpSatStageName.ALTERNATE_RANK_1: sum(alternates_by_rank.get(1, ())),
            CpSatStageName.ALTERNATE_RANK_2: sum(alternates_by_rank.get(2, ())),
            CpSatStageName.ALTERNATE_RANK_3: sum(alternates_by_rank.get(3, ())),
            CpSatStageName.FULLY_SCHEDULED: sum(fully_scheduled_vars.values()),
            CpSatStageName.REMAINING_PERIOD_UNITS: sum(remaining_exprs.values()) if remaining_exprs else 0,
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
    *,
    repair_hint: bool = True,
) -> cp_model.CpSolver:
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(max_time_seconds)
    solver.parameters.num_search_workers = int(workers)
    solver.parameters.random_seed = int(seed)
    solver.parameters.log_search_progress = bool(log_search_progress)
    solver.parameters.repair_hint = bool(repair_hint)
    solver.parameters.hint_conflict_limit = 1000
    return solver


def _solve_stage_sequence(
    build: _ModelBuild,
    stages: tuple[_SolveStage, ...],
    *,
    max_time_seconds_per_stage: float,
    num_search_workers: int,
    log_search_progress: bool,
    seed: int,
    continue_after_feasible: bool,
    stage_to_stage_hints: bool,
    initial_hint_keys: tuple[_VariableKey, ...],
    initial_fixed_values: tuple[tuple[CpSatStageName, int], ...],
    already_conditional: bool,
    budget: _GlobalTimeBudget,
    incumbent_candidates: tuple[_IncumbentCandidate, ...] = (),
) -> _StageRun:
    diagnostics: list[CpSatStageDiagnostic] = []
    stage_values: dict[CpSatStageName, int] = {}
    fixed_values: dict[CpSatStageName, int] = dict(initial_fixed_values)
    incumbent: cp_model.CpSolver | None = None
    status = CpSatSolveStatus.UNKNOWN
    lexicographic_optimum = not already_conditional
    conditional = already_conditional
    conditional_performed = already_conditional
    highest_global: CpSatStageName | None = None
    stage_hint_used = False
    candidates = list(incumbent_candidates)
    prepared_cache: dict[str, tuple[_StageIncumbent, tuple[_VariableKey, ...]] | None] = {}
    selected_candidate_sources: list[str] = []

    if initial_hint_keys:
        _apply_key_hint(build.model, build.assignment_vars, initial_hint_keys)

    for index, stage in enumerate(stages):
        remaining_at_start = budget.remaining()
        effective_limit = budget.effective_limit(max_time_seconds_per_stage)
        if effective_limit <= 0:
            diagnostics.append(
                _skipped_diagnostic(
                    stage.stage_name,
                    build.model_scope,
                    "time_budget_exhausted",
                    remaining_at_start,
                    effective_limit,
                )
            )
            diagnostics.extend(_skipped_diagnostics(stages[index + 1 :], "time_budget_exhausted", build.model_scope))
            lexicographic_optimum = False
            break
        expr = build.stage_exprs[stage.stage_name]
        if stage_to_stage_hints:
            selected_candidate = _select_incumbent_candidate(
                build,
                stage,
                fixed_values,
                tuple(candidates),
                prepared_cache,
            )
            if selected_candidate is not None:
                _apply_complete_stage_incumbent(build.model, selected_candidate[1])
                selected_candidate_sources.append(
                    f"{stage.stage_name.value}:{selected_candidate[0].source}"
                )
        if stage.sense == "min":
            build.model.Minimize(expr)
        else:
            build.model.Maximize(expr)
        solver = _new_solver(
            effective_limit,
            num_search_workers,
            log_search_progress,
            seed,
            repair_hint=stage.stage_name != CpSatStageName.LOGICAL_SCHEDULE_COMPLETION,
        )
        raw_status = solver.Solve(build.model)
        budget.refresh()
        status = _solve_status(raw_status)
        diagnostic = _stage_diagnostic(
            stage.stage_name,
            build.model_scope,
            status,
            solver,
            conditional_on_unproven_incumbent=conditional,
            fixed_higher_priority_values=tuple(fixed_values.items()),
            objective_descriptor_hash=_objective_descriptor_hash(build.model),
            remaining_global_budget_at_start_seconds=remaining_at_start,
            effective_time_limit_seconds=effective_limit,
        )
        if stage.stage_name == CpSatStageName.LOGICAL_SCHEDULE_COMPLETION:
            _validate_logical_completion_bound(diagnostic, _logical_completion_upper_bound(build))
        diagnostics.append(diagnostic)

        if status not in {CpSatSolveStatus.OPTIMAL, CpSatSolveStatus.FEASIBLE}:
            diagnostics.extend(_skipped_diagnostics(stages[index + 1 :], "no_incumbent", build.model_scope))
            lexicographic_optimum = False
            break

        incumbent = solver
        stage_incumbent = _capture_stage_incumbent(build.model, solver)
        value = _stage_objective_value(solver, expr)
        stage_values[stage.stage_name] = value
        fixed_values[stage.stage_name] = value
        build.model.Add(expr == value)
        candidates.append(
            _IncumbentCandidate(
                candidate_id=f"formal_{stage.stage_name.value}_{len(candidates)}",
                source=f"formal_stage:{stage.stage_name.value}",
                model_scope=build.model_scope,
                selected_keys=_selected_assignments(build, solver),
                snapshot=stage_incumbent,
            )
        )

        if status == CpSatSolveStatus.OPTIMAL and not conditional:
            highest_global = stage.stage_name
        else:
            lexicographic_optimum = False
            conditional = True
            conditional_performed = True
            if not continue_after_feasible:
                diagnostics.extend(
                    _skipped_diagnostics(stages[index + 1 :], "conditional_continuation_disabled", build.model_scope)
                )
                break

        if stage_to_stage_hints:
            _apply_solution_hint(
                build.model,
                build.assignment_vars,
                solver,
                initial_hint_keys,
                incumbent=stage_incumbent,
            )
            stage_hint_used = True

    return _StageRun(
        solver=incumbent,
        status=status,
        diagnostics=tuple(diagnostics),
        stage_values=stage_values,
        lexicographic_optimum=lexicographic_optimum and len(stage_values) == len(stages),
        conditional_optimization_performed=conditional_performed,
        highest_globally_proven_stage=highest_global,
        stage_to_stage_hint_used=stage_hint_used,
        incumbent_candidates=tuple(candidates),
        budget_exhausted=budget.exhausted,
        selected_candidate_sources=tuple(selected_candidate_sources),
    )


def _apply_key_hint(
    model: cp_model.CpModel,
    assignment_vars: dict[_VariableKey, cp_model.IntVar],
    hinted_keys: tuple[_VariableKey, ...],
) -> None:
    _clear_hints(model)
    for key in sorted(set(hinted_keys), key=lambda item: (item.request_key, item.section_id)):
        var = assignment_vars.get(key)
        if var is not None:
            model.AddHint(var, 1)


def _select_incumbent_candidate(
    build: _ModelBuild,
    stage: _SolveStage,
    fixed_values: dict[CpSatStageName, int],
    candidates: tuple[_IncumbentCandidate, ...],
    prepared_cache: dict[str, tuple[_StageIncumbent, tuple[_VariableKey, ...]] | None] | None = None,
) -> tuple[_IncumbentCandidate, _StageIncumbent, int] | None:
    eligible: list[tuple[_IncumbentCandidate, tuple[_VariableKey, ...], int]] = []
    prepared_cache = prepared_cache if prepared_cache is not None else {}
    for candidate in candidates:
        mapped_keys = _mapped_candidate_keys(build, candidate)
        if mapped_keys is None:
            continue
        if any(
            _selected_stage_objective_value(build, stage_name, mapped_keys) != fixed_value
            for stage_name, fixed_value in fixed_values.items()
        ):
            continue
        current_value = _selected_stage_objective_value(build, stage.stage_name, mapped_keys)
        eligible.append((candidate, mapped_keys, current_value))
    if not eligible:
        return None

    source_priority = {
        "persisted_feasible_seed": 0,
        "full_model_feasibility_incumbent": 1,
    }
    if stage.sense == "min":
        selected_candidate, selected_keys, current_value = min(
            eligible,
            key=lambda item: (
                item[2],
                source_priority.get(item[0].source, 2),
                item[0].candidate_id,
            ),
        )
    else:
        selected_candidate, selected_keys, current_value = min(
            eligible,
            key=lambda item: (
                -item[2],
                source_priority.get(item[0].source, 2),
                item[0].candidate_id,
            ),
        )
    if selected_candidate.candidate_id not in prepared_cache:
        prepared_cache[selected_candidate.candidate_id] = _prepare_incumbent_candidate(
            build,
            selected_candidate,
            {},
        )
    prepared = prepared_cache[selected_candidate.candidate_id]
    if prepared is None:
        return None
    incumbent, _prepared_keys = prepared
    return selected_candidate, incumbent, current_value


def _mapped_candidate_keys(
    build: _ModelBuild,
    candidate: _IncumbentCandidate,
) -> tuple[_VariableKey, ...] | None:
    if candidate.model_scope == CpSatModelScope.CORE and build.model_scope == CpSatModelScope.ENRICHMENT:
        return None
    if len(candidate.selected_keys) != len(set(candidate.selected_keys)):
        return None
    mapped_keys = tuple(key for key in candidate.selected_keys if key in build.assignment_vars)
    if candidate.model_scope == build.model_scope and len(mapped_keys) != len(candidate.selected_keys):
        return None
    return mapped_keys


def _prepare_incumbent_candidate(
    build: _ModelBuild,
    candidate: _IncumbentCandidate,
    fixed_values: dict[CpSatStageName, int],
) -> tuple[_StageIncumbent, tuple[_VariableKey, ...]] | None:
    mapped_keys = _mapped_candidate_keys(build, candidate)
    if mapped_keys is None:
        return None
    if candidate.snapshot is not None and candidate.model_scope == build.model_scope:
        try:
            _validate_stage_incumbent_for_model(build.model, candidate.snapshot)
        except CpSatStageIncumbentConsistencyError:
            return None
        return candidate.snapshot, mapped_keys
    try:
        values, _variables = _full_model_hint_values(build, mapped_keys)
        incumbent = _stage_incumbent_from_values(build.model, values)
    except (KeyError, ValueError, CpSatStageIncumbentConsistencyError):
        return None
    for stage_name, fixed_value in fixed_values.items():
        if _selected_stage_objective_value(build, stage_name, mapped_keys) != fixed_value:
            return None
    return incumbent, mapped_keys


def _apply_complete_stage_incumbent(model: cp_model.CpModel, incumbent: _StageIncumbent) -> None:
    _validate_stage_incumbent_for_model(model, incumbent)
    values_by_name = dict(incumbent.values_by_name)
    _write_hint_values(
        model,
        {
            index: values_by_name[variable.name]
            for index, variable in enumerate(model.Proto().variables)
        },
    )


def _selected_stage_objective_value(
    build: _ModelBuild,
    stage_name: CpSatStageName,
    selected_keys: tuple[_VariableKey, ...],
) -> int:
    selected = {key for key in selected_keys if key in build.assignment_vars}
    selected_request_keys = {key.request_key for key in selected}
    math_set = set(build.math_course_ids)

    def assigned(request_key: str) -> int:
        return int(request_key in selected_request_keys)

    primary = tuple(request for request in build.requests_by_key.values() if request.request_type == "primary")
    primary_unmet = sum(1 - assigned(request.request_key) for request in primary)
    primary_unmet_units = sum(
        (1 - assigned(request.request_key)) * request.period_units for request in primary
    )
    if stage_name == CpSatStageName.PRIMARY_UNMET_COUNT:
        return primary_unmet
    if stage_name == CpSatStageName.PRIMARY_UNMET_PERIOD_UNITS:
        return primary_unmet_units
    if stage_name == CpSatStageName.PRIMARY_SATISFACTION:
        max_possible_units = sum(request.period_units for request in primary)
        return primary_unmet * (max_possible_units + 1) + primary_unmet_units
    if stage_name == CpSatStageName.MATH_COVERAGE:
        violations = 0
        for student_id in build.math_violation_vars:
            covered = any(
                key in selected
                and key.request_key in build.requests_by_key
                and build.requests_by_key[key.request_key].student_id == student_id
                and build.requests_by_key[key.request_key].candidate_key in math_set
                and build.requests_by_key[key.request_key].request_type
                in {"primary", MANDATORY_FALLBACK_REQUEST_TYPE}
                for key in selected
            )
            violations += int(not covered)
        return violations
    if stage_name == CpSatStageName.LOGICAL_SCHEDULE_COMPLETION:
        return _selected_logical_course_count(build, selected)
    if stage_name in {
        CpSatStageName.ALTERNATE_RANK_1,
        CpSatStageName.ALTERNATE_RANK_2,
        CpSatStageName.ALTERNATE_RANK_3,
    }:
        rank = {
            CpSatStageName.ALTERNATE_RANK_1: 1,
            CpSatStageName.ALTERNATE_RANK_2: 2,
            CpSatStageName.ALTERNATE_RANK_3: 3,
        }[stage_name]
        return sum(
            assigned(request.request_key)
            for request in build.requests_by_key.values()
            if request.request_type == "alternate" and request.request_rank == rank
        )
    used_units: Counter[str] = Counter()
    for key in selected:
        request = build.requests_by_key.get(key.request_key)
        if request is not None:
            used_units[request.student_id] += request.period_units
    if stage_name == CpSatStageName.FULLY_SCHEDULED:
        return sum(
            used_units[student.student_id] == student.target_period_units
            for student in build.students_by_id.values()
        )

    if stage_name == CpSatStageName.REMAINING_PERIOD_UNITS:
        return sum(student.target_period_units - used_units[student.student_id] for student in build.students_by_id.values())
    if stage_name == CpSatStageName.SEEDED_TIE_BREAK:
        keys = sorted(build.assignment_vars, key=lambda item: (item.request_key, item.section_id))
        weights = {key: index + 1 for index, key in enumerate(keys)}
        return sum(weights[key] for key in selected)
    raise KeyError(f"No candidate objective evaluator for {stage_name.value}")


def _selected_logical_course_count(
    build: _ModelBuild,
    selected_keys: tuple[_VariableKey, ...],
) -> int:
    identities_by_student: dict[str, set[str]] = defaultdict(set)
    for key in selected_keys:
        request = build.requests_by_key.get(key.request_key)
        if request is not None and key in build.assignment_vars:
            identities_by_student[request.student_id].add(request.candidate_key)
    return sum(len(identities) for identities in identities_by_student.values())


def _logical_completion_upper_bound(build: _ModelBuild) -> int:
    return sum(student.target_period_units for student in build.students_by_id.values())


def _validate_logical_completion_bound(
    diagnostic: CpSatStageDiagnostic,
    upper_bound: int,
) -> None:
    if diagnostic.objective_value is None or diagnostic.best_objective_bound is None:
        return
    objective = diagnostic.objective_value
    best_bound = diagnostic.best_objective_bound
    if not 0 <= objective <= best_bound <= upper_bound:
        raise CpSatStageIncumbentConsistencyError(
            "CP-SAT logical completion objective bound is inconsistent: "
            f"objective={objective}, best_bound={best_bound}, target_total={upper_bound}"
        )


def _apply_complete_key_hint(
    model: cp_model.CpModel,
    assignment_vars: dict[_VariableKey, cp_model.IntVar],
    hinted_keys: tuple[_VariableKey, ...],
) -> None:
    _clear_hints(model)
    duplicate_count = len(hinted_keys) - len(set(hinted_keys))
    if duplicate_count:
        raise ValueError(f"Duplicate CP-SAT hint keys: {duplicate_count}")
    unknown = tuple(key for key in hinted_keys if key not in assignment_vars)
    if unknown:
        raise ValueError(f"Unmapped CP-SAT hint keys: {unknown[:3]!r}")
    selected = set(hinted_keys)
    _write_hint_values(
        model,
        {
            assignment_vars[key].Index(): int(key in selected)
            for key in sorted(assignment_vars, key=lambda item: (item.request_key, item.section_id))
        },
    )


def _mapped_hint_keys(
    assignment_vars: dict[_VariableKey, cp_model.IntVar],
    hinted_keys: tuple[_VariableKey, ...],
) -> tuple[_VariableKey, ...]:
    return tuple(
        sorted(
            {key for key in hinted_keys if key in assignment_vars},
            key=lambda item: (item.request_key, item.section_id),
        )
    )


def _audit_complete_hint(
    assignment_vars: dict[_VariableKey, cp_model.IntVar],
    seed: _HintSeed,
) -> _HintAudit:
    if not seed.keys:
        return _HintAudit(source=seed.source)
    unique_keys = set(seed.keys)
    mapped_keys = _mapped_hint_keys(assignment_vars, seed.keys)
    total = len(assignment_vars)
    selected = len(mapped_keys)
    return _HintAudit(
        source=seed.source,
        total_model_variables=total,
        variables_supplied=total,
        coverage_rate=1.0 if total else 0.0,
        selected_variables=selected,
        zero_variables=max(total - selected, 0),
        unknown_or_unmapped_assignments=sum(key not in assignment_vars for key in unique_keys),
        duplicate_keys=len(seed.keys) - len(unique_keys),
        replay_policy_pass=seed.replay_policy_pass,
        violation_students=seed.violation_students,
    )


def _full_model_hint_values(
    build: _ModelBuild,
    selected_keys: tuple[_VariableKey, ...],
    *,
    require_complete: bool = True,
) -> tuple[dict[int, int], dict[int, cp_model.IntVar]]:
    """Derive all auxiliary Boolean values implied by a known assignment.

    The assignment vector is the source of truth.  ``assigned_*``, math
    violation, and legacy full-schedule indicators are deterministic helper
    variables in this model, so hinting them is safe and keeps the hint a
    search suggestion rather than adding constraints.
    """
    selected = set(selected_keys)
    values: dict[int, int] = {}
    variables: dict[int, cp_model.IntVar] = {}

    def add(var: cp_model.IntVar, value: int) -> None:
        index = var.Index()
        existing = values.get(index)
        if existing is not None and existing != int(value):
            raise ValueError(f"Conflicting hint value for CP-SAT variable index {index}")
        values[index] = int(value)
        variables[index] = var

    for key, var in build.assignment_vars.items():
        add(var, int(key in selected))

    for request_key, expression in build.assigned_vars.items():
        if isinstance(expression, cp_model.IntVar):
            add(expression, int(any(key.request_key == request_key for key in selected)))

    math_set = set(build.math_course_ids)
    for student_id, var in build.math_violation_vars.items():
        covered = any(
            key in selected
            and build.requests_by_key[key.request_key].student_id == student_id
            and build.requests_by_key[key.request_key].candidate_key in math_set
            and build.requests_by_key[key.request_key].request_type in {
                "primary",
                MANDATORY_FALLBACK_REQUEST_TYPE,
            }
            for key in selected
            if key.request_key in build.requests_by_key
        )
        add(var, int(not covered))

    logical_identities_by_student: dict[str, set[str]] = defaultdict(set)
    for key in selected:
        request = build.requests_by_key.get(key.request_key)
        if request is not None:
            logical_identities_by_student[request.student_id].add(request.candidate_key)
    for student_id, var in build.logical_assigned_course_vars.items():
        add(var, len(logical_identities_by_student[student_id]))
    add(
        build.logical_assigned_course_total_var,
        sum(len(identities) for identities in logical_identities_by_student.values()),
    )

    used_units: Counter[str] = Counter()
    for key in selected:
        request = build.requests_by_key.get(key.request_key)
        if request is not None:
            used_units[request.student_id] += request.period_units
    for student_id, var in build.fully_scheduled_vars.items():
        student = build.students_by_id[student_id]
        add(var, int(used_units[student_id] == student.target_period_units))

    expected = len(build.model.Proto().variables)
    for index, proto_var in enumerate(build.model.Proto().variables):
        if index in values:
            continue
        if len(proto_var.domain) == 2 and proto_var.domain[0] == proto_var.domain[1]:
            add(build.model.GetIntVarFromProtoIndex(index), int(proto_var.domain[0]))
    if require_complete and len(values) != expected:
        missing = expected - len(values)
        raise ValueError(f"Known assignment did not map {missing} CP-SAT model variables")
    return values, variables


def _audit_full_model_hint(
    build: _ModelBuild,
    seed: _HintSeed,
    values: dict[int, int],
) -> _HintAudit:
    if not seed.keys:
        return _HintAudit(source=seed.source)
    unique_keys = set(seed.keys)
    total = len(build.model.Proto().variables)
    persisted = seed.persisted
    return _HintAudit(
        source=seed.source,
        total_model_variables=total,
        variables_supplied=len(values),
        coverage_rate=(len(values) / total) if total else 0.0,
        selected_variables=sum(value == 1 for value in values.values()),
        zero_variables=sum(value == 0 for value in values.values()),
        unknown_or_unmapped_assignments=sum(key not in build.assignment_vars for key in unique_keys),
        duplicate_keys=len(seed.keys) - len(unique_keys),
        replay_policy_pass=seed.replay_policy_pass,
        violation_students=seed.violation_students,
        auxiliary_variables_derived=max(len(values) - len(build.assignment_vars), 0),
        initial_solution_seed_enabled=persisted is not None,
        initial_solution_seed_role="full_model_initial_hint" if persisted is not None else "",
        initial_solution_seed_source_commit=persisted.source_commit if persisted is not None else "",
        initial_solution_seed_source_algorithm=persisted.source_algorithm if persisted is not None else "",
        initial_solution_seed_source_status=persisted.source_status if persisted is not None else "",
        initial_solution_seed_source_policy_pass=persisted.source_policy_pass if persisted is not None else None,
        initial_solution_seed_manifest_sha256=persisted.manifest_sha256 if persisted is not None else "",
        initial_solution_seed_request_outcomes_sha256=persisted.request_outcomes_sha256 if persisted is not None else "",
        initial_solution_seed_provenance_sha256=persisted.provenance_sha256 if persisted is not None else "",
        initial_solution_seed_fingerprint=(tuple(sorted(persisted.fingerprint.items())) if persisted is not None else ()),
        initial_solution_seed_hint_coverage=(len(values) / total if total else 0.0) if persisted is not None else None,
        initial_solution_seed_unknown_keys=(sum(key not in build.assignment_vars for key in unique_keys) if persisted is not None else 0),
        initial_solution_seed_duplicate_keys=(len(seed.keys) - len(unique_keys) if persisted is not None else 0),
    )


def _apply_complete_model_hint(
    build: _ModelBuild,
    selected_keys: tuple[_VariableKey, ...],
    *,
    values: dict[int, int] | None = None,
    variables: dict[int, cp_model.IntVar] | None = None,
    allow_partial: bool = False,
) -> None:
    if len(selected_keys) != len(set(selected_keys)):
        raise ValueError("Duplicate CP-SAT full-model hint keys")
    unknown = tuple(key for key in selected_keys if key not in build.assignment_vars)
    if unknown:
        raise ValueError(f"Unmapped CP-SAT full-model hint keys: {unknown[:3]!r}")
    if values is None or variables is None:
        values, variables = _full_model_hint_values(
            build,
            selected_keys,
            require_complete=not allow_partial,
        )
    del variables
    expected = set(range(len(build.model.Proto().variables)))
    if not allow_partial and set(values) != expected:
        raise ValueError(
            "Complete CP-SAT model hint does not cover every model variable: "
            f"expected {len(expected)}, got {len(values)}"
        )
    _write_hint_values(build.model, values)


def _model_proto_without_solution_hint_hash(model: cp_model.CpModel) -> str:
    """Hash model structure while excluding the mutable CP-SAT hint field."""
    proto = copy.deepcopy(model.Proto())
    proto.solution_hint.vars.clear()
    proto.solution_hint.values.clear()
    text = str(proto).replace("solution_hint {\n}\n", "")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _selected_bootstrap_assignments(
    build: _BootstrapBuild,
    solver: cp_model.CpSolver,
) -> tuple[_VariableKey, ...]:
    return tuple(
        key
        for key in sorted(build.assignment_vars, key=lambda item: (item.request_key, item.section_id))
        if solver.BooleanValue(build.assignment_vars[key])
    )


def _replay_bootstrap_solution(
    allocation_input: CanonicalAllocationInput,
    build: _BootstrapBuild,
    selected: tuple[_VariableKey, ...],
) -> AllocationState:
    state = AllocationState(allocation_input)
    for key in sorted(selected, key=lambda item: _assignment_replay_sort_key(build.requests_by_key[item.request_key], item.section_id)):
        request = build.requests_by_key[key.request_key]
        result = state.try_assign(request.student_id, request.request_key, key.section_id)
        if not result.allowed:
            raise RuntimeError(
                "CP-SAT bootstrap solution failed AllocationState replay: "
                f"{request.request_key} -> {key.section_id}: {[reason.value for reason in result.reasons]}"
            )
    issues = state.validate_internal_consistency()
    if issues:
        raise RuntimeError(f"CP-SAT bootstrap solution failed AllocationState consistency: {issues!r}")
    return state


def _bootstrap_hint_strategy(hint_keys: tuple[_VariableKey, ...]) -> str:
    return "constrained_first_partial" if hint_keys else "none"


def _bootstrap_variable_count(bootstrap_run: _BootstrapRun) -> int:
    return len(bootstrap_run.build.model.Proto().variables) if bootstrap_run.build is not None else 0


def _bootstrap_constraint_count(bootstrap_run: _BootstrapRun) -> int:
    return len(bootstrap_run.build.model.Proto().constraints) if bootstrap_run.build is not None else 0


def _bootstrap_build_time(bootstrap_run: _BootstrapRun) -> float:
    return bootstrap_run.build.build_time_seconds if bootstrap_run.build is not None else 0.0


def _apply_solution_hint(
    model: cp_model.CpModel,
    assignment_vars: dict[_VariableKey, cp_model.IntVar],
    solver: cp_model.CpSolver,
    extra_hint_keys: tuple[_VariableKey, ...],
    *,
    incumbent: _StageIncumbent | None = None,
) -> None:
    del assignment_vars, extra_hint_keys
    incumbent = incumbent or _capture_stage_incumbent(model, solver)
    _validate_stage_incumbent_for_model(model, incumbent)
    values_by_name = dict(incumbent.values_by_name)
    _write_hint_values(
        model,
        {
            index: values_by_name[variable.name]
            for index, variable in enumerate(model.Proto().variables)
        },
    )


def _capture_stage_incumbent(
    model: cp_model.CpModel,
    solver: cp_model.CpSolver,
) -> _StageIncumbent:
    proto = model.Proto()
    expected = len(proto.variables)
    response_values = tuple(int(value) for value in solver.ResponseProto().solution)
    if response_values and len(response_values) != expected:
        raise CpSatStageIncumbentConsistencyError(
            "CP-SAT stage response has an incomplete solution vector: "
            f"expected {expected}, got {len(response_values)}"
        )
    values = tuple(int(solver.Value(model.GetIntVarFromProtoIndex(index))) for index in range(expected))
    if response_values and values != response_values:
        raise CpSatStageIncumbentConsistencyError(
            "CP-SAT stage response vector does not match CpSolver.Value()"
        )
    names = tuple(variable.name for variable in proto.variables)
    if len(names) != len(set(names)):
        raise CpSatStageIncumbentConsistencyError("CP-SAT stage model contains duplicate variable names")
    domains = tuple(tuple(int(value) for value in variable.domain) for variable in proto.variables)
    for index, (value, domain) in enumerate(zip(values, domains, strict=True)):
        if not _value_in_domain(value, domain):
            raise CpSatStageIncumbentConsistencyError(
                f"CP-SAT stage incumbent value {value} is outside variable domain at index {index}"
            )
    return _stage_incumbent_from_values(model, values)


def _stage_incumbent_from_values(
    model: cp_model.CpModel,
    values: tuple[int, ...] | dict[int, int],
) -> _StageIncumbent:
    variables = model.Proto().variables
    ordered_values = tuple(
        int(values[index]) if isinstance(values, dict) else int(values[index])
        for index in range(len(variables))
    )
    names = tuple(variable.name for variable in variables)
    if len(names) != len(set(names)):
        raise CpSatStageIncumbentConsistencyError("CP-SAT stage model contains duplicate variable names")
    domains = tuple(tuple(int(value) for value in variable.domain) for variable in variables)
    for index, (value, domain) in enumerate(zip(ordered_values, domains, strict=True)):
        if not _value_in_domain(value, domain):
            raise CpSatStageIncumbentConsistencyError(
                f"CP-SAT stage incumbent value {value} is outside variable domain at index {index}"
            )
    return _StageIncumbent(
        values_by_name=tuple(sorted(zip(names, ordered_values, strict=True))),
        domains_by_name=tuple(sorted(zip(names, domains, strict=True))),
    )


def _validate_stage_incumbent_for_model(
    model: cp_model.CpModel,
    incumbent: _StageIncumbent,
) -> None:
    variables = model.Proto().variables
    if len(variables) != len(incumbent.values_by_name):
        raise CpSatStageIncumbentConsistencyError(
            "CP-SAT next-stage variable count changed: "
            f"incumbent={len(incumbent.values_by_name)}, next_stage={len(variables)}"
        )
    values_by_name = dict(incumbent.values_by_name)
    incumbent_domains = dict(incumbent.domains_by_name)
    names = tuple(variable.name for variable in variables)
    if len(names) != len(set(names)):
        raise CpSatStageIncumbentConsistencyError("CP-SAT next-stage model contains duplicate variable names")
    if set(names) != set(values_by_name):
        raise CpSatStageIncumbentConsistencyError(
            "CP-SAT next-stage variable mapping changed; stable variable names are required"
        )
    for variable in variables:
        domain = tuple(int(value) for value in variable.domain)
        if domain != incumbent_domains[variable.name]:
            raise CpSatStageIncumbentConsistencyError(
                "CP-SAT next-stage variable domains changed for the previous incumbent: "
                f"{variable.name}"
            )
        if not _value_in_domain(values_by_name[variable.name], domain):
            raise CpSatStageIncumbentConsistencyError(
                "CP-SAT next-stage incumbent value is outside the variable domain: "
                f"{variable.name}"
            )


def _value_in_domain(value: int, domain: tuple[int, ...]) -> bool:
    return any(lower <= value <= upper for lower, upper in zip(domain[::2], domain[1::2], strict=True))


def _stage_objective_value(
    solver: cp_model.CpSolver,
    expression: cp_model.LinearExpr,
) -> int:
    solver_value = float(solver.ObjectiveValue())
    replayed_value = int(solver.Value(expression))
    normalized_value = int(round(solver_value))
    if normalized_value != replayed_value:
        raise CpSatStageIncumbentConsistencyError(
            "CP-SAT stage objective mismatch: "
            f"solver={normalized_value}, replay={replayed_value}"
        )
    return replayed_value


def _clear_hints(model: cp_model.CpModel) -> None:
    if not hasattr(model, "ClearHints"):
        raise RuntimeError("OR-Tools CpModel.ClearHints() is required for safe stage-to-stage hints")
    model.ClearHints()


def _write_hint_values(model: cp_model.CpModel, values_by_index: dict[int, int]) -> None:
    """Write a deterministic complete or partial hint without repeated API calls."""
    _clear_hints(model)
    hint = model.Proto().solution_hint
    for index in sorted(values_by_index):
        hint.vars.append(int(index))
        hint.values.append(int(values_by_index[index]))


def _constrained_first_partial_hint_keys(
    allocation_input: CanonicalAllocationInput,
    math_fallback_rules: tuple[MathFallbackRule, ...],
    math_course_ids: tuple[str, ...],
    seed: int,
) -> tuple[_VariableKey, ...]:
    # Hints only guide search. They do not relax any hard policy; CP-SAT may
    # repair or ignore them while optimizing the formal model.
    from .constrained_first_baseline import run_constrained_first_baseline

    greedy = run_constrained_first_baseline(
        allocation_input,
        seed,
        math_fallback_rules=math_fallback_rules,
        math_course_ids=math_course_ids,
    )
    excluded_students = set(greedy.policy_report.ordinary_violation_student_ids)
    excluded_students.update(greedy.policy_report.protected_violation_student_ids)
    excluded_students.update(greedy.policy_report.high_demand_violating_student_ids)
    hinted = tuple(
        _VariableKey(assignment.request_key, assignment.linked_section_group_id)
        for assignment in greedy.assignments
        if assignment.student_id not in excluded_students
    )
    return tuple(sorted(hinted, key=lambda item: (item.request_key, item.section_id)))


def _constrained_first_full_hint_seed(
    allocation_input: CanonicalAllocationInput,
    math_fallback_rules: tuple[MathFallbackRule, ...],
    math_course_ids: tuple[str, ...],
    seed: int,
) -> _HintSeed:
    """Build a complete candidate-vector seed from the unchanged greedy result.

    The returned assignment keys are only a hint.  In particular, a greedy
    policy failure is recorded as metadata and is never converted into a hard
    constraint for the CP-SAT model.
    """
    from .constrained_first_baseline import run_constrained_first_baseline

    started = time.perf_counter()
    greedy = run_constrained_first_baseline(
        allocation_input,
        seed,
        math_fallback_rules=math_fallback_rules,
        math_course_ids=math_course_ids,
    )
    policy_report = evaluate_final_schedule_policy(greedy.algorithm_name, greedy.student_outcomes)
    keys = tuple(
        _VariableKey(assignment.request_key, assignment.linked_section_group_id)
        for assignment in greedy.assignments
    )
    primary_outcomes = tuple(
        item
        for item in getattr(greedy, "request_outcomes", ())
        if item.request_type == "primary"
    )
    logical_assigned = sum(int(item.assigned_logical_course_count or 0) for item in greedy.student_outcomes)
    target_logical = sum(
        int(item.target_logical_course_count or item.target_period_units)
        for item in greedy.student_outcomes
    )
    logical_gap = sum(
        max(
            int(item.target_logical_course_count or item.target_period_units)
            - int(item.assigned_logical_course_count or 0),
            0,
        )
        for item in greedy.student_outcomes
    )
    final_policy = evaluate_final_schedule_policy(ALGORITHM_NAME, greedy.student_outcomes)
    internal_audit = _HintAudit(
        source="constrained_first_internal",
        assignment_hash=_assignment_records_hash(greedy.assignments),
        duplicate_keys=len(keys) - len(set(keys)),
        primary_assigned=sum(item.status == PrimaryRequestStatus.ASSIGNED for item in primary_outcomes),
        primary_unmet=sum(item.status != PrimaryRequestStatus.ASSIGNED for item in primary_outcomes),
        logical_assigned=logical_assigned,
        logical_gap=logical_gap,
        logical_full=sum(gap == 0 for gap in (
            int(item.logical_schedule_gap_count or 0) for item in greedy.student_outcomes
        )),
        gap_over_1=sum(int(item.logical_schedule_gap_count or 0) > 1 for item in greedy.student_outcomes),
        below_five=sum(int(item.assigned_logical_course_count or 0) < 5 for item in greedy.student_outcomes),
        policy_violation_count=final_policy.summary.violating_student_count,
        structural_issue_count=len(getattr(greedy, "consistency_issues", ())),
        runtime_seconds=time.perf_counter() - started,
    )
    return _HintSeed(
        source="constrained_first_greedy_full",
        keys=tuple(sorted(keys, key=lambda item: (item.request_key, item.section_id))),
        replay_policy_pass=policy_report.summary.final_schedule_policy_pass,
        violation_students=policy_report.summary.violating_student_count,
        internal_audit=internal_audit,
    )


def _assignment_records_hash(assignments: Iterable[object]) -> str:
    rows = sorted(
        (
            str(item.request_key),
            str(item.linked_section_group_id),
            str(getattr(item, "assignment_key", "")),
        )
        for item in assignments
    )
    return hashlib.sha256(repr(rows).encode("utf-8")).hexdigest()


def _stage_diagnostic(
    stage_name: CpSatStageName,
    model_scope: CpSatModelScope,
    status: CpSatSolveStatus,
    solver: cp_model.CpSolver,
    *,
    conditional_on_unproven_incumbent: bool,
    fixed_higher_priority_values: tuple[tuple[CpSatStageName, int], ...],
    objective_descriptor_hash: str = "",
    remaining_global_budget_at_start_seconds: float | None = None,
    effective_time_limit_seconds: float | None = None,
    repair_hint_enabled: bool = False,
    hint_assignment_hash: str = "",
) -> CpSatStageDiagnostic:
    has_solution = status in {CpSatSolveStatus.OPTIMAL, CpSatSolveStatus.FEASIBLE}
    objective = int(round(solver.ObjectiveValue())) if has_solution else None
    bound = int(round(solver.BestObjectiveBound())) if has_solution else None
    return CpSatStageDiagnostic(
        stage_name=stage_name,
        model_scope=model_scope,
        status=status,
        objective_value=objective,
        best_objective_bound=bound,
        wall_time_seconds=round(solver.WallTime(), 6),
        conflicts=int(solver.NumConflicts()),
        branches=int(solver.NumBranches()),
        optimum_proven=status == CpSatSolveStatus.OPTIMAL,
        conditional_on_unproven_incumbent=conditional_on_unproven_incumbent,
        fixed_higher_priority_values=fixed_higher_priority_values,
        remaining_global_budget_at_start_seconds=(
            round(remaining_global_budget_at_start_seconds, 6)
            if remaining_global_budget_at_start_seconds is not None
            else None
        ),
        effective_time_limit_seconds=(
            round(effective_time_limit_seconds, 6)
            if effective_time_limit_seconds is not None
            else None
        ),
        response_proto_hash=_response_proto_hash(solver),
        objective_descriptor_hash=objective_descriptor_hash,
        repair_hint_enabled=repair_hint_enabled,
        hint_assignment_hash=hint_assignment_hash,
    )


def _response_proto_hash(solver: cp_model.CpSolver) -> str:
    """Return a stable identity for the response produced by this solver."""
    return hashlib.sha256(str(solver.ResponseProto()).encode("utf-8")).hexdigest()


def _objective_descriptor_hash(model: cp_model.CpModel) -> str:
    """Hash only the current objective descriptor, excluding solution hints."""
    objective = model.Proto().objective
    descriptor = (
        tuple(objective.vars),
        tuple(objective.coeffs),
        float(objective.offset),
        float(objective.scaling_factor),
        tuple(objective.domain),
    )
    return hashlib.sha256(repr(descriptor).encode("utf-8")).hexdigest()


def _skipped_diagnostic(
    stage_name: CpSatStageName,
    model_scope: CpSatModelScope,
    reason: str,
    remaining_global_budget_at_start_seconds: float | None = None,
    effective_time_limit_seconds: float | None = None,
) -> CpSatStageDiagnostic:
    return CpSatStageDiagnostic(
        stage_name=stage_name,
        model_scope=model_scope,
        status=CpSatSolveStatus.SKIPPED,
        objective_value=None,
        best_objective_bound=None,
        wall_time_seconds=0.0,
        conflicts=0,
        branches=0,
        optimum_proven=False,
        skipped=True,
        skip_reason=reason,
        remaining_global_budget_at_start_seconds=(
            round(remaining_global_budget_at_start_seconds, 6)
            if remaining_global_budget_at_start_seconds is not None
            else None
        ),
        effective_time_limit_seconds=(
            round(effective_time_limit_seconds, 6)
            if effective_time_limit_seconds is not None
            else None
        ),
    )


def _skipped_diagnostics(
    stages: tuple[_SolveStage, ...],
    reason: str,
    model_scope: CpSatModelScope,
) -> tuple[CpSatStageDiagnostic, ...]:
    return tuple(
        _skipped_diagnostic(stage.stage_name, model_scope, reason)
        for stage in stages
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
                if not build.candidate_index.get(request.request_key, allocation_input.candidate_index.get(request.request_key, ()))
                else PrimaryRequestStatus.UNMET_ALL_CANDIDATES_REJECTED
            )
        else:
            status = _alternate_status(request, assignment, before, build, allocation_input)
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
    allocation_input: CanonicalAllocationInput,
) -> AlternateRequestStatus:
    if assignment is not None:
        return AlternateRequestStatus.ASSIGNED
    if remaining_units == 0:
        return AlternateRequestStatus.NOT_NEEDED
    if request.period_units > remaining_units:
        return AlternateRequestStatus.DOES_NOT_FIT_REMAINING_LOAD
    if not build.candidate_index.get(request.request_key, allocation_input.candidate_index.get(request.request_key, ())):
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
    if _solver_bool_value(solver, build.assigned_vars[plan.source_request.request_key]):
        return MandatoryFallbackStatus.NOT_REQUIRED_SOURCE_ASSIGNED
    math_set = set(build.math_course_ids)
    for request in build.requests_by_key.values():
        if (
            request.student_id == plan.source_request.student_id
            and request.request_type == "primary"
            and request.request_key != plan.source_request.request_key
            and request.candidate_key in math_set
            and _solver_bool_value(solver, build.assigned_vars[request.request_key])
        ):
            return MandatoryFallbackStatus.NOT_REQUIRED_MATH_COVERAGE_ALREADY_SATISFIED
    if assigned:
        return MandatoryFallbackStatus.ASSIGNED
    if not plan.candidates:
        return MandatoryFallbackStatus.UNASSIGNED_NO_CANDIDATES
    return MandatoryFallbackStatus.UNASSIGNED_ALL_CANDIDATES_REJECTED


def _solver_bool_value(solver: cp_model.CpSolver, expr: cp_model.LinearExpr) -> bool:
    if isinstance(expr, int):
        return bool(expr)
    return bool(solver.BooleanValue(expr))


def _solver_int_value(solver: cp_model.CpSolver, expr: cp_model.LinearExpr) -> int:
    if isinstance(expr, int):
        return int(expr)
    return int(round(solver.Value(expr)))


def _objective_values(
    build: _ModelBuild,
    solver: cp_model.CpSolver,
    stage_values: dict[CpSatStageName, int],
    *,
    logical_schedule_completion_enabled: bool = True,
) -> CpSatObjectiveValues:
    del stage_values
    stage_exprs = build.stage_exprs
    primary_unmet_count = _solver_int_value(solver, stage_exprs[CpSatStageName.PRIMARY_UNMET_COUNT])
    primary_unmet_units = _solver_int_value(solver, stage_exprs[CpSatStageName.PRIMARY_UNMET_PERIOD_UNITS])
    primary_penalty = _solver_int_value(solver, stage_exprs[CpSatStageName.PRIMARY_SATISFACTION])
    return CpSatObjectiveValues(
        math_coverage_violations=_solver_int_value(solver, stage_exprs[CpSatStageName.MATH_COVERAGE]),
        primary_unmet_count=primary_unmet_count,
        primary_unmet_period_units=primary_unmet_units,
        primary_penalty=primary_penalty,
        logical_assigned_course_count=(
            _solver_int_value(solver, stage_exprs[CpSatStageName.LOGICAL_SCHEDULE_COMPLETION])
            if logical_schedule_completion_enabled
            else 0
        ),
        alternate_rank1_assigned=_solver_int_value(solver, stage_exprs[CpSatStageName.ALTERNATE_RANK_1]),
        alternate_rank2_assigned=_solver_int_value(solver, stage_exprs[CpSatStageName.ALTERNATE_RANK_2]),
        alternate_rank3_assigned=_solver_int_value(solver, stage_exprs[CpSatStageName.ALTERNATE_RANK_3]),
        fully_scheduled_students=_solver_int_value(solver, stage_exprs[CpSatStageName.FULLY_SCHEDULED]),
        total_remaining_period_units=_solver_int_value(solver, stage_exprs[CpSatStageName.REMAINING_PERIOD_UNITS]),
        seeded_tie_break_value=_solver_int_value(solver, stage_exprs[CpSatStageName.SEEDED_TIE_BREAK]),
    )


def _validate_logical_completion_consistency(
    student_outcomes,
    final_policy_report,
    objective_value: int,
    fixed_stage_value: int | None,
) -> None:
    student_outcomes = tuple(student_outcomes)
    assigned_logical = sum(
        int(outcome.assigned_logical_course_count)
        if outcome.assigned_logical_course_count is not None
        else len(outcome.assignment_keys)
        for outcome in student_outcomes
    )
    target_logical = sum(
        int(outcome.target_logical_course_count)
        if outcome.target_logical_course_count is not None
        else int(outcome.target_period_units)
        for outcome in student_outcomes
    )
    total_gap = sum(
        max(
            (
                int(outcome.target_logical_course_count)
                if outcome.target_logical_course_count is not None
                else int(outcome.target_period_units)
            )
            - (
                int(outcome.assigned_logical_course_count)
                if outcome.assigned_logical_course_count is not None
                else len(outcome.assignment_keys)
            ),
            0,
        )
        for outcome in student_outcomes
    )
    over_target = tuple(
        outcome.student_id
        for outcome in student_outcomes
        if (
            int(outcome.assigned_logical_course_count)
            if outcome.assigned_logical_course_count is not None
            else len(outcome.assignment_keys)
        )
        > (
            int(outcome.target_logical_course_count)
            if outcome.target_logical_course_count is not None
            else int(outcome.target_period_units)
        )
    )
    if over_target:
        raise CpSatFinalSchedulePolicyConsistencyError(
            "CP-SAT logical schedule metrics are inconsistent: assigned logical courses exceed target "
            f"for students {over_target[:3]!r}"
        )
    if assigned_logical + total_gap != target_logical:
        raise CpSatFinalSchedulePolicyConsistencyError(
            "CP-SAT logical schedule metrics are inconsistent: "
            f"assigned={assigned_logical}, gap={total_gap}, target_total={target_logical}"
        )
    summary = final_policy_report.summary
    if summary.logical_fully_scheduled_student_count + summary.students_with_logical_schedule_gap != len(student_outcomes):
        raise CpSatFinalSchedulePolicyConsistencyError(
            "CP-SAT logical schedule metrics are inconsistent: "
            "logical_fully_scheduled + students_with_logical_schedule_gap != total_students"
        )
    if summary.maximum_schedule_gap_count <= MAXIMUM_SCHEDULE_GAP_COUNT:
        if summary.total_logical_schedule_gap != summary.students_with_logical_schedule_gap:
            raise CpSatFinalSchedulePolicyConsistencyError(
                "CP-SAT logical schedule metrics are inconsistent under max-gap hard policy: "
                "total_logical_schedule_gap != students_with_logical_schedule_gap"
            )
    if summary.total_logical_schedule_gap != total_gap:
        raise CpSatFinalSchedulePolicyConsistencyError(
            "CP-SAT logical schedule metrics are inconsistent with replayed outcomes: "
            f"summary_gap={summary.total_logical_schedule_gap}, replay_gap={total_gap}"
        )
    if objective_value != assigned_logical:
        raise CpSatFinalSchedulePolicyConsistencyError(
            "CP-SAT logical completion objective does not match replayed assignments: "
            f"objective={objective_value}, replay={assigned_logical}"
        )
    if fixed_stage_value is not None and fixed_stage_value != assigned_logical:
        raise CpSatFinalSchedulePolicyConsistencyError(
            "CP-SAT logical completion fixed stage value does not match replayed assignments: "
            f"fixed={fixed_stage_value}, replay={assigned_logical}"
        )


def _logical_completion_metadata(
    diagnostics: tuple[CpSatStageDiagnostic, ...],
    stage_values: dict[CpSatStageName, int],
) -> dict[str, object]:
    diagnostic = next(
        (item for item in diagnostics if item.stage_name == CpSatStageName.LOGICAL_SCHEDULE_COMPLETION),
        None,
    )
    return {
        "enabled": diagnostic is not None,
        "status": diagnostic.status if diagnostic is not None else None,
        "objective_value": diagnostic.objective_value if diagnostic is not None else None,
        "best_bound": diagnostic.best_objective_bound if diagnostic is not None else None,
        "conditionally_optimized": (
            diagnostic.conditional_on_unproven_incumbent if diagnostic is not None else False
        ),
        "fixed_value": stage_values.get(CpSatStageName.LOGICAL_SCHEDULE_COMPLETION),
    }


def _objective_vector(stage_values: dict[CpSatStageName, int]) -> tuple[tuple[CpSatStageName, int], ...]:
    order = (
        CpSatStageName.MATH_COVERAGE,
        CpSatStageName.PRIMARY_UNMET_COUNT,
        CpSatStageName.PRIMARY_UNMET_PERIOD_UNITS,
        CpSatStageName.LOGICAL_SCHEDULE_COMPLETION,
        CpSatStageName.ALTERNATE_RANK_1,
        CpSatStageName.ALTERNATE_RANK_2,
        CpSatStageName.ALTERNATE_RANK_3,
        CpSatStageName.FULLY_SCHEDULED,
        CpSatStageName.REMAINING_PERIOD_UNITS,
        CpSatStageName.SEEDED_TIE_BREAK,
    )
    return tuple((stage_name, stage_values[stage_name]) for stage_name in order if stage_name in stage_values)


def _warm_start_strategy(stage_to_stage_hints: bool, external_hint_used: bool) -> str:
    if stage_to_stage_hints and external_hint_used:
        return "stage_to_stage_incumbent+constrained_first_partial"
    if stage_to_stage_hints:
        return "stage_to_stage_incumbent"
    if external_hint_used:
        return "constrained_first_partial"
    return "none"


def _empty_result(
    seed: int,
    status: CpSatSolveStatus,
    diagnostics: tuple[CpSatStageDiagnostic, ...],
    build: _ModelBuild | None,
    other_build: _ModelBuild | None,
    bootstrap_run: _BootstrapRun,
    elapsed: float,
    optimality_proven: bool,
    *,
    external_hint_used: bool,
    stage_to_stage_hint_used: bool,
    max_total_time_seconds: float | None,
    total_budget_exhausted: bool,
    skipped_stage_count: int,
    core_hint_source: str,
    final_schedule_hard_constraints_enabled: bool = True,
    hint_audit: _HintAudit | None = None,
    hint_selected_keys: tuple[_VariableKey, ...] = (),
    internal_repair_run: _InternalRepairRun | None = None,
) -> CpSatAllocationResult:
    proto = build.model.Proto() if build is not None else None
    other_proto = other_build.model.Proto() if other_build is not None else None
    core_proto = proto if build is not None and build.model_scope == CpSatModelScope.CORE else other_proto
    enrichment_proto = proto if build is not None and build.model_scope == CpSatModelScope.ENRICHMENT else other_proto
    core_vars = len(core_proto.variables) if core_proto is not None else 0
    core_constraints = len(core_proto.constraints) if core_proto is not None else 0
    enrichment_vars = len(enrichment_proto.variables) if enrichment_proto is not None else 0
    enrichment_constraints = len(enrichment_proto.constraints) if enrichment_proto is not None else 0
    build_time = build.build_time_seconds if build is not None else 0.0
    other_build_time = other_build.build_time_seconds if other_build is not None else 0.0
    total_build_time = build_time + other_build_time + _bootstrap_build_time(bootstrap_run)
    logical_metadata = _logical_completion_metadata(diagnostics, {})
    hint_audit = hint_audit or _HintAudit()
    internal_repair_run = internal_repair_run or _InternalRepairRun(
        build=None,
        solver=None,
        status=CpSatSolveStatus.SKIPPED,
        diagnostic=None,
        selected_keys=(),
        solve_time_seconds=0.0,
    )
    return CpSatAllocationResult(
        algorithm_name=ALGORITHM_NAME,
        seed=int(seed),
        solve_status=status,
        lexicographic_optimality_proven=optimality_proven,
        stage_diagnostics=diagnostics,
        objective_values=CpSatObjectiveValues(),
        model_stats=CpSatModelStats(
            total_variables=core_vars + enrichment_vars,
            total_constraints=core_constraints + enrichment_constraints,
            build_time_seconds=round(total_build_time, 6),
            solve_time_seconds=round(max(elapsed - total_build_time, 0.0), 6),
            core_model_variable_count=core_vars,
            core_model_constraint_count=core_constraints,
            enrichment_model_variable_count=enrichment_vars,
            enrichment_model_constraint_count=enrichment_constraints,
            bootstrap_enabled=bootstrap_run.status != CpSatBootstrapStatus.DISABLED,
            bootstrap_status=bootstrap_run.status,
            bootstrap_variable_count=_bootstrap_variable_count(bootstrap_run),
            bootstrap_constraint_count=_bootstrap_constraint_count(bootstrap_run),
            bootstrap_build_time_seconds=round(_bootstrap_build_time(bootstrap_run), 6),
            bootstrap_solve_time_seconds=round(bootstrap_run.solve_time_seconds, 6),
            bootstrap_hint_strategy=bootstrap_run.hint_strategy,
            bootstrap_incumbent_found=bootstrap_run.status == CpSatBootstrapStatus.FEASIBLE_FOUND,
            time_to_first_hard_feasible_solution_seconds=(
                round(bootstrap_run.time_to_first_hard_feasible_solution_seconds, 6)
                if bootstrap_run.time_to_first_hard_feasible_solution_seconds is not None
                else None
            ),
            core_hint_source=core_hint_source,
            max_total_time_seconds=max_total_time_seconds,
            total_budget_exhausted=total_budget_exhausted,
            skipped_stage_count=skipped_stage_count,
            core_build_time_seconds=round(build_time if build is not None and build.model_scope == CpSatModelScope.CORE else other_build_time, 6),
            enrichment_build_time_seconds=round(build_time if build is not None and build.model_scope == CpSatModelScope.ENRICHMENT else other_build_time, 6),
            total_build_time_seconds=round(total_build_time, 6),
            total_solve_time_seconds=round(max(elapsed - total_build_time, 0.0), 6),
            time_to_first_feasible_solution_seconds=(
                round(bootstrap_run.time_to_first_hard_feasible_solution_seconds, 6)
                if bootstrap_run.time_to_first_hard_feasible_solution_seconds is not None
                else None
            ),
            warm_start_strategy=_warm_start_strategy(stage_to_stage_hint_used, external_hint_used),
            external_hint_used=external_hint_used,
            stage_to_stage_hint_used=stage_to_stage_hint_used,
            final_schedule_hard_constraints_enabled=final_schedule_hard_constraints_enabled,
            logical_schedule_completion_objective_enabled=logical_metadata["enabled"],
            logical_schedule_completion_stage_status=logical_metadata["status"],
            logical_schedule_completion_objective_value=logical_metadata["objective_value"],
            logical_schedule_completion_best_bound=logical_metadata["best_bound"],
            logical_schedule_completion_conditionally_optimized=logical_metadata["conditionally_optimized"],
            logical_schedule_completion_fixed_value=logical_metadata["fixed_value"],
            hint_source=hint_audit.source,
            hint_total_model_variables=hint_audit.total_model_variables,
            hint_variables_supplied=hint_audit.variables_supplied,
            hint_coverage_rate=hint_audit.coverage_rate,
            hint_selected_variables=hint_audit.selected_variables,
            hint_zero_variables=hint_audit.zero_variables,
            hint_unknown_or_unmapped_assignments=hint_audit.unknown_or_unmapped_assignments,
            hint_duplicate_keys=hint_audit.duplicate_keys,
            hint_replay_policy_pass=hint_audit.replay_policy_pass,
            full_model_seed_strategy=hint_audit.source,
            full_model_seed_policy_pass=hint_audit.replay_policy_pass,
            full_model_seed_violation_students=hint_audit.violation_students,
            full_model_seed_repaired_by_solver=(
                hint_audit.replay_policy_pass is False and bool(hint_selected_keys)
            ) if hint_audit.source != "none" else None,
            initial_solution_seed_enabled=hint_audit.initial_solution_seed_enabled,
            initial_solution_seed_role=hint_audit.initial_solution_seed_role,
            initial_solution_seed_source_commit=hint_audit.initial_solution_seed_source_commit,
            initial_solution_seed_source_algorithm=hint_audit.initial_solution_seed_source_algorithm,
            initial_solution_seed_source_status=hint_audit.initial_solution_seed_source_status,
            initial_solution_seed_source_policy_pass=hint_audit.initial_solution_seed_source_policy_pass,
            initial_solution_seed_manifest_sha256=hint_audit.initial_solution_seed_manifest_sha256,
            initial_solution_seed_request_outcomes_sha256=hint_audit.initial_solution_seed_request_outcomes_sha256,
            initial_solution_seed_provenance_sha256=hint_audit.initial_solution_seed_provenance_sha256,
            initial_solution_seed_fingerprint=hint_audit.initial_solution_seed_fingerprint,
            initial_solution_seed_hint_coverage=hint_audit.initial_solution_seed_hint_coverage,
            initial_solution_seed_unknown_keys=hint_audit.initial_solution_seed_unknown_keys,
            initial_solution_seed_duplicate_keys=hint_audit.initial_solution_seed_duplicate_keys,
            internal_feasibility_hint_strategy=(
                "constrained_first" if internal_repair_run.diagnostic is not None else "none"
            ),
            internal_repair_objective_strategy=internal_repair_run.objective_strategy,
            internal_repair_hint_enabled=internal_repair_run.diagnostic is not None,
            internal_repair_status=(
                internal_repair_run.status if internal_repair_run.diagnostic is not None else None
            ),
            internal_repair_incumbent_found=internal_repair_run.validated,
            internal_repair_runtime_seconds=round(internal_repair_run.solve_time_seconds, 6),
            internal_repair_time_to_first_solution_seconds=internal_repair_run.time_to_first_solution_seconds,
            internal_repair_hamming_distance=internal_repair_run.hamming_distance,
            internal_repair_greedy_assignments_removed=internal_repair_run.greedy_assignments_removed,
            internal_repair_new_assignments_added=internal_repair_run.new_assignments_added,
            internal_repair_changed_students=internal_repair_run.changed_students,
            internal_repair_changed_requests=internal_repair_run.changed_requests,
            internal_repair_changed_sections=internal_repair_run.changed_sections,
            internal_repair_response_proto_hash=internal_repair_run.response_proto_hash,
            internal_repair_validation_failure=internal_repair_run.validation_failure,
            internal_hint_assignment_hash=internal_repair_run.hint_audit.assignment_hash,
            internal_hint_primary_assigned=internal_repair_run.hint_audit.primary_assigned,
            internal_hint_primary_unmet=internal_repair_run.hint_audit.primary_unmet,
            internal_hint_logical_assigned=internal_repair_run.hint_audit.logical_assigned,
            internal_hint_logical_gap=internal_repair_run.hint_audit.logical_gap,
            internal_hint_logical_full=internal_repair_run.hint_audit.logical_full,
            internal_hint_gap_over_1=internal_repair_run.hint_audit.gap_over_1,
            internal_hint_below_five=internal_repair_run.hint_audit.below_five,
            internal_hint_policy_violation_count=internal_repair_run.hint_audit.policy_violation_count,
            internal_hint_structural_issue_count=internal_repair_run.hint_audit.structural_issue_count,
            internal_hint_candidate_variables=internal_repair_run.hint_audit.candidate_variables,
            internal_hint_candidate_variables_hinted=internal_repair_run.hint_audit.candidate_variables_hinted,
            internal_hint_candidate_coverage_rate=internal_repair_run.hint_audit.candidate_coverage_rate,
            internal_hint_auxiliary_variables_hinted=internal_repair_run.hint_audit.auxiliary_variables_derived,
            internal_hint_unhinted_variables=internal_repair_run.hint_audit.unhinted_variables,
            internal_hint_duplicate_keys=internal_repair_run.hint_audit.duplicate_keys,
            internal_hint_out_of_domain_keys=internal_repair_run.hint_audit.out_of_domain_keys,
            model_invariance_before_hint_hash=internal_repair_run.model_before_hint_hash,
            model_invariance_after_hint_hash=internal_repair_run.model_after_hint_hash,
            model_invariance_equal=internal_repair_run.model_invariance_equal,
            model_invariance_without_distance_hash=internal_repair_run.model_invariance_without_distance_hash,
            model_invariance_distance_stripped_hash=internal_repair_run.model_invariance_distance_stripped_hash,
            model_invariance_distance_stripped_equal=internal_repair_run.model_invariance_distance_stripped_equal,
            internal_repair_variable_hash=internal_repair_run.variable_hash,
            internal_repair_domain_hash=internal_repair_run.domain_hash,
            internal_repair_constraint_hash=internal_repair_run.constraint_hash,
            internal_repair_candidate_mapping_hash=internal_repair_run.candidate_mapping_hash,
        ),
    )


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value)
