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
    UNKNOWN_WITH_VALIDATED_INCUMBENT = "UNKNOWN_WITH_VALIDATED_INCUMBENT"
    SKIPPED = "SKIPPED"


class CpSatBootstrapStatus(str, Enum):
    DISABLED = "DISABLED"
    FEASIBLE_FOUND = "FEASIBLE_FOUND"
    INFEASIBLE = "INFEASIBLE"
    UNKNOWN_NO_INCUMBENT = "UNKNOWN_NO_INCUMBENT"
    MODEL_INVALID = "MODEL_INVALID"


class CpSatStageName(str, Enum):
    FEASIBILITY_BOOTSTRAP = "feasibility_bootstrap"
    INTERNAL_REPAIR_FEASIBILITY = "internal_repair_feasibility"
    FULL_MODEL_FEASIBILITY_INCUMBENT = "full_model_feasibility_incumbent"
    MATH_COVERAGE = "math_coverage"
    PRIMARY_SATISFACTION = "primary_satisfaction"
    PRIMARY_UNMET_COUNT = "primary_unmet_count"
    PRIMARY_UNMET_PERIOD_UNITS = "primary_unmet_period_units"
    LOGICAL_SCHEDULE_COMPLETION = "logical_schedule_completion"
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
    # These hashes bind the reported stage metrics to the response and
    # objective descriptor used by that stage's Solve call.
    response_proto_hash: str = field(default="", compare=False)
    objective_descriptor_hash: str = field(default="", compare=False)
    repair_hint_enabled: bool = False
    hint_assignment_hash: str = field(default="", compare=False)


@dataclass(frozen=True)
class CpSatObjectiveValues:
    math_coverage_violations: int = 0
    primary_unmet_count: int = 0
    primary_unmet_period_units: int = 0
    primary_penalty: int = 0
    logical_assigned_course_count: int = 0
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
    logical_schedule_completion_objective_enabled: bool = False
    logical_schedule_completion_stage_status: CpSatSolveStatus | None = None
    logical_schedule_completion_objective_value: int | None = None
    logical_schedule_completion_best_bound: int | None = None
    logical_schedule_completion_conditionally_optimized: bool = False
    logical_schedule_completion_fixed_value: int | None = None
    # Hint metadata describes the candidate-assignment variable universe. It
    # does not imply that auxiliary CP-SAT variables were fixed or that the
    # hint is a feasible solution.
    hint_source: str = "none"
    hint_total_model_variables: int = 0
    hint_variables_supplied: int = 0
    hint_coverage_rate: float = 0.0
    hint_selected_variables: int = 0
    hint_zero_variables: int = 0
    hint_unknown_or_unmapped_assignments: int = 0
    hint_duplicate_keys: int = 0
    hint_replay_policy_pass: bool | None = None
    full_model_seed_strategy: str = "none"
    full_model_seed_policy_pass: bool | None = None
    full_model_seed_violation_students: int | None = None
    full_model_seed_repaired_by_solver: bool | None = None
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
    initial_solution_seed_selected_by_stage: tuple[str, ...] = ()
    internal_feasibility_hint_strategy: str = "none"
    internal_repair_objective_strategy: str = "none"
    internal_repair_hint_enabled: bool = False
    internal_repair_status: CpSatSolveStatus | None = None
    internal_repair_incumbent_found: bool = False
    internal_repair_runtime_seconds: float = field(default=0.0, compare=False)
    internal_repair_time_to_first_solution_seconds: float | None = field(default=None, compare=False)
    internal_repair_hamming_distance: int | None = None
    internal_repair_greedy_assignments_removed: int | None = None
    internal_repair_new_assignments_added: int | None = None
    internal_repair_changed_students: int | None = None
    internal_repair_changed_requests: int | None = None
    internal_repair_changed_sections: int | None = None
    internal_repair_response_proto_hash: str = ""
    internal_repair_validation_failure: str = ""
    internal_hint_assignment_hash: str = ""
    internal_hint_primary_assigned: int | None = None
    internal_hint_primary_unmet: int | None = None
    internal_hint_logical_assigned: int | None = None
    internal_hint_logical_gap: int | None = None
    internal_hint_logical_full: int | None = None
    internal_hint_gap_over_1: int | None = None
    internal_hint_below_five: int | None = None
    internal_hint_policy_violation_count: int | None = None
    internal_hint_structural_issue_count: int | None = None
    internal_hint_candidate_variables: int = 0
    internal_hint_candidate_variables_hinted: int = 0
    internal_hint_candidate_coverage_rate: float = 0.0
    internal_hint_auxiliary_variables_hinted: int = 0
    internal_hint_unhinted_variables: int = 0
    internal_hint_duplicate_keys: int = 0
    internal_hint_out_of_domain_keys: int = 0
    model_invariance_before_hint_hash: str = ""
    model_invariance_after_hint_hash: str = ""
    model_invariance_equal: bool | None = None
    model_invariance_without_distance_hash: str = ""
    model_invariance_distance_stripped_hash: str = ""
    model_invariance_distance_stripped_equal: bool | None = None
    internal_repair_variable_hash: str = ""
    internal_repair_domain_hash: str = ""
    internal_repair_constraint_hash: str = ""
    internal_repair_candidate_mapping_hash: str = ""


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
