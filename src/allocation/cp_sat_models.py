from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

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


class CpSatStageName(str, Enum):
    MATH_COVERAGE = "math_coverage"
    PRIMARY_SATISFACTION = "primary_satisfaction"
    ALTERNATE_RANK_1 = "alternate_rank_1"
    ALTERNATE_RANK_2 = "alternate_rank_2"
    ALTERNATE_RANK_3 = "alternate_rank_3"
    FULLY_SCHEDULED = "fully_scheduled"
    REMAINING_PERIOD_UNITS = "remaining_period_units"
    SEEDED_TIE_BREAK = "seeded_tie_break"


@dataclass(frozen=True)
class CpSatStageDiagnostic:
    stage_name: CpSatStageName
    status: CpSatSolveStatus
    objective_value: int | None
    best_objective_bound: int | None
    wall_time_seconds: float = field(compare=False)
    conflicts: int
    branches: int
    optimum_proven: bool


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
