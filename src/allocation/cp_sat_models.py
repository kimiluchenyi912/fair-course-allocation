from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from src.final_schedule_policy import (
    MAXIMUM_ORDINARY_PRIMARY_UNMET_COUNT,
    MAXIMUM_PROTECTED_PRIMARY_UNMET_COUNT,
    MAXIMUM_SCHEDULE_GAP_COUNT,
    MINIMUM_ASSIGNED_LOGICAL_COURSE_COUNT,
    SCHEMA_VERSION as FINAL_SCHEDULE_POLICY_SCHEMA_VERSION,
)

from .assignment_models import AssignmentRecord, StateConsistencyIssue
from .baseline_models import (
    MandatoryFallbackOutcome,
    PolicyReport,
    RequestOutcome,
    SectionRosterSummary,
    StudentOutcome,
)
from .math_policy_models import MathPolicyReport


class CpSatSolveStatus(str, Enum):
    OPTIMAL = "OPTIMAL"
    FEASIBLE = "FEASIBLE"
    INFEASIBLE = "INFEASIBLE"
    MODEL_INVALID = "MODEL_INVALID"
    UNKNOWN = "UNKNOWN"
    SKIPPED = "SKIPPED"


class CpSatBootstrapStatus(str, Enum):
    DISABLED = "DISABLED"
    FEASIBLE_FOUND = "FEASIBLE_FOUND"
    INFEASIBLE = "INFEASIBLE"
    UNKNOWN_NO_INCUMBENT = "UNKNOWN_NO_INCUMBENT"
    MODEL_INVALID = "MODEL_INVALID"


class CpSatStageName(str, Enum):
    FEASIBILITY_BOOTSTRAP = "feasibility_bootstrap"
    MATH_COVERAGE = "math_coverage"
    PRIMARY_SATISFACTION = "primary_satisfaction"
    PRIMARY_UNMET_COUNT = "primary_unmet_count"
    PRIMARY_UNMET_PERIOD_UNITS = "primary_unmet_period_units"
    ALTERNATE_RANK_1 = "alternate_rank_1"
    ALTERNATE_RANK_2 = "alternate_rank_2"
    ALTERNATE_RANK_3 = "alternate_rank_3"
    FULLY_SCHEDULED = "fully_scheduled"
    REMAINING_PERIOD_UNITS = "remaining_period_units"
    SEEDED_TIE_BREAK = "seeded_tie_break"


class CpSatModelScope(str, Enum):
    BOOTSTRAP = "bootstrap"
    CORE = "core"
    ENRICHMENT = "enrichment"


@dataclass(frozen=True)
class CpSatStageDiagnostic:
    stage_name: CpSatStageName
    model_scope: CpSatModelScope
    status: CpSatSolveStatus
    objective_value: int | None
    best_objective_bound: int | None
    wall_time_seconds: float = field(compare=False)
    conflicts: int
    branches: int
    optimum_proven: bool
    conditional_on_unproven_incumbent: bool = False
    fixed_higher_priority_values: tuple[tuple[CpSatStageName, int], ...] = ()
    skipped: bool = False
    skip_reason: str = ""
    remaining_global_budget_at_start_seconds: float | None = field(default=None, compare=False)
    effective_time_limit_seconds: float | None = field(default=None, compare=False)


@dataclass(frozen=True)
class CpSatObjectiveValues:
    math_coverage_violations: int = 0
    primary_unmet_count: int = 0
    primary_unmet_period_units: int = 0
    primary_penalty: int = 0
    alternate_rank1_assigned: int = 0
    alternate_rank2_assigned: int = 0
    alternate_rank3_assigned: int = 0
    fully_scheduled_students: int = 0
    total_remaining_period_units: int = 0
    seeded_tie_break_value: int = 0


@dataclass(frozen=True)
class CpSatModelStats:
    total_variables: int
    total_constraints: int
    build_time_seconds: float = field(compare=False)
    solve_time_seconds: float = field(compare=False)
    core_model_variable_count: int = 0
    core_model_constraint_count: int = 0
    enrichment_model_variable_count: int = 0
    enrichment_model_constraint_count: int = 0
    bootstrap_enabled: bool = False
    bootstrap_status: CpSatBootstrapStatus = CpSatBootstrapStatus.DISABLED
    bootstrap_variable_count: int = 0
    bootstrap_constraint_count: int = 0
    bootstrap_build_time_seconds: float = field(default=0.0, compare=False)
    bootstrap_solve_time_seconds: float = field(default=0.0, compare=False)
    bootstrap_hint_strategy: str = "none"
    bootstrap_incumbent_found: bool = False
    time_to_first_hard_feasible_solution_seconds: float | None = field(default=None, compare=False)
    core_hint_source: str = "none"
    max_total_time_seconds: float | None = None
    total_budget_exhausted: bool = False
    skipped_stage_count: int = 0
    core_build_time_seconds: float = field(default=0.0, compare=False)
    enrichment_build_time_seconds: float = field(default=0.0, compare=False)
    total_build_time_seconds: float = field(default=0.0, compare=False)
    total_solve_time_seconds: float = field(default=0.0, compare=False)
    time_to_first_feasible_solution_seconds: float | None = field(default=None, compare=False)
    warm_start_strategy: str = "none"
    external_hint_used: bool = False
    stage_to_stage_hint_used: bool = False
    highest_globally_proven_stage: CpSatStageName | None = None
    conditional_optimization_performed: bool = False
    objective_vector: tuple[tuple[CpSatStageName, int], ...] = ()
    final_schedule_hard_constraints_enabled: bool = True
    final_schedule_policy_schema_version: str = FINAL_SCHEDULE_POLICY_SCHEMA_VERSION
    minimum_assigned_logical_course_count: int = MINIMUM_ASSIGNED_LOGICAL_COURSE_COUNT
    maximum_logical_schedule_gap_count: int = MAXIMUM_SCHEDULE_GAP_COUNT
    maximum_ordinary_primary_unmet_count: int = MAXIMUM_ORDINARY_PRIMARY_UNMET_COUNT
    maximum_protected_primary_unmet_count: int = MAXIMUM_PROTECTED_PRIMARY_UNMET_COUNT
    post_solve_policy_gate_pass: bool | None = None


@dataclass(frozen=True)
class CpSatAllocationResult:
    algorithm_name: str
    seed: int
    solve_status: CpSatSolveStatus
    lexicographic_optimality_proven: bool
    stage_diagnostics: tuple[CpSatStageDiagnostic, ...]
    objective_values: CpSatObjectiveValues
    model_stats: CpSatModelStats
    assignments: tuple[AssignmentRecord, ...] = ()
    mandatory_fallback_outcomes: tuple[MandatoryFallbackOutcome, ...] = ()
    request_outcomes: tuple[RequestOutcome, ...] = ()
    student_outcomes: tuple[StudentOutcome, ...] = ()
    policy_report: PolicyReport | None = None
    math_policy_report: MathPolicyReport | None = None
    section_roster_summary: tuple[SectionRosterSummary, ...] = ()
    consistency_issues: tuple[StateConsistencyIssue, ...] = ()
