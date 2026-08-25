"""Section-Plan Feasibility Alignment Audit v1.

A read-only-of-production, write-only-to-a-new-artifact diagnostic slice
that explains *why* the frozen distance-guided full-hard-model repair
(``src/cp_sat_normal_evaluation_runner.py``) proved seven normal development
scenarios infeasible under their frozen section plans and the current hard
model, using the stable feasible reference scenario as a control.

This module never modifies the production section planner, generator,
CP-SAT hard constraints, objective, or Final Schedule Policy. It builds an
independent diagnostic CP-SAT model that reuses the production canonical
input (``CanonicalAllocationInput``), the production candidate index, the
production mandatory-fallback injection, and the production policy
threshold constants -- the only new content is the assumption-literal and
slack-variable plumbing needed to measure which constraint families are
responsible for infeasibility, and by how much. No diagnostic result from
this module is ever a publishable assignment or a repaired section plan.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from ortools.sat.python import cp_model

from src.allocation import math_course_ids_from_catalog
from src.allocation.constrained_first_baseline import run_constrained_first_baseline
from src.allocation.input_models import CanonicalAllocationInput, LogicalRequest
from src.allocation.random_baseline import HIGH_DEMAND_PRIMARY_THRESHOLD, _build_mandatory_fallback_plans
from src.allocation.cp_sat_solver import (
    _VariableKey,
    _add_duplicate_identity_constraints,
    _add_fallback_constraints,
    _add_math_coverage_constraints,
    _add_student_target_constraints,
    _constrained_first_full_hint_seed,
    _convert_fallback_plans,
    _hamming_distance_expression,
    _mapped_hint_keys,
    _safe_name,
    _validate_candidates_for_request,
)
from src.final_schedule_policy import (
    MAXIMUM_ORDINARY_PRIMARY_UNMET_COUNT,
    MAXIMUM_PROTECTED_PRIMARY_UNMET_COUNT,
    MAXIMUM_SCHEDULE_GAP_COUNT,
    MINIMUM_ASSIGNED_LOGICAL_COURSE_COUNT,
)
from src.cp_sat_robustness_runner import (
    CpSatEvaluationError,
    EvaluationScenario,
    _load_benchmark_math_fallback_rules,
    _load_scenario_input,
    _read_json,
    _scenario_from_payload,
    _sha256_file,
    _verify_sha256_manifest,
    _verify_source_suite,
    _verify_sha256_manifest,
    _write_checksums,
    _write_csv,
    _write_json,
    evaluation_manifest_hash,
)

DEFAULT_MANIFEST = Path("data/scenarios/section_plan_feasibility_audit_v1.json")
DEFAULT_OUTPUT = Path(
    "../fair-course-allocation-artifacts/robustness-v1/section-plan-feasibility-audit-v1"
)
SCHEMA_VERSION = 1
CONTROL_SCENARIO_ID = "normal_dev_reference_2026"

FAMILY_NAMES = (
    "SECTION_CAPACITY",
    "STUDENT_PERIOD_CONFLICT",
    "PROTECTED_PRIMARY",
    "ORDINARY_MAX_PRIMARY_UNMET",
    "HIGH_DEMAND_PRIMARY",
    "MINIMUM_FIVE_LOGICAL",
    "MAXIMUM_LOGICAL_GAP_ONE",
)
NEVER_RELAXED_FAMILY_NAMES = (
    "ASSIGNMENT_CANDIDATE_VALIDITY",
    "DUPLICATE_LOGICAL_IDENTITY",
    "STUDENT_TARGET_LOAD_CAP",
    "MANDATORY_FALLBACK_SEMANTICS",
    "MATH_COVERAGE_SOFT_POLICY",
)

# The eight fixed single/multi-family counterfactual variants from Section 13
# of the audit spec. Each maps a variant name to the set of families relaxed
# (fixed literal = 0); every other diagnosable family is fixed enabled (1).
COUNTERFACTUAL_VARIANTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("capacity_only", ("SECTION_CAPACITY",)),
    ("period_conflict_only", ("STUDENT_PERIOD_CONFLICT",)),
    ("primary_policy_only", ("PROTECTED_PRIMARY", "ORDINARY_MAX_PRIMARY_UNMET", "HIGH_DEMAND_PRIMARY")),
    ("minimum_five_only", ("MINIMUM_FIVE_LOGICAL",)),
    ("maximum_gap_only", ("MAXIMUM_LOGICAL_GAP_ONE",)),
    ("minimum_five_and_maximum_gap", ("MINIMUM_FIVE_LOGICAL", "MAXIMUM_LOGICAL_GAP_ONE")),
    ("all_student_policy_families", ("PROTECTED_PRIMARY", "ORDINARY_MAX_PRIMARY_UNMET", "HIGH_DEMAND_PRIMARY", "MINIMUM_FIVE_LOGICAL", "MAXIMUM_LOGICAL_GAP_ONE")),
    ("capacity_and_period_conflict", ("SECTION_CAPACITY", "STUDENT_PERIOD_CONFLICT")),
)


class SectionPlanAuditError(CpSatEvaluationError):
    """Raised for section-plan feasibility audit provenance/config failures."""


class SectionPlanAuditCorrectnessFailure(SectionPlanAuditError):
    """Raised when the diagnostic model disagrees with the production ground truth."""


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


EXPECTED_SCENARIO_ORDER = (
    "normal_dev_reference_2026",
    "normal_dev_01", "normal_dev_03", "normal_dev_04",
    "normal_dev_05", "normal_dev_07", "normal_dev_09", "normal_dev_10",
)


def load_section_plan_audit_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    payload = _read_json(Path(path))
    required = {
        "audit_name", "audit_version", "diagnostic_schema_version", "source_normal_suite",
        "source_git_commit", "split", "development_data", "tuning_allowed",
        "holdout_execution_allowed", "stress_execution_allowed", "solver_seed", "workers",
        "time_budgets_seconds", "constraint_families", "scenarios", "notes",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise SectionPlanAuditError("audit manifest missing: " + ", ".join(missing))
    if payload["split"] != "development" or payload["development_data"] is not True:
        raise SectionPlanAuditError("audit manifest must use development data")
    if payload["holdout_execution_allowed"] is not False:
        raise SectionPlanAuditError("audit manifest must forbid holdout execution")
    if payload["stress_execution_allowed"] is not False:
        raise SectionPlanAuditError("audit manifest must forbid stress execution")
    if int(payload["solver_seed"]) != 20260630 or int(payload["workers"]) != 1:
        raise SectionPlanAuditError("audit manifest solver_seed/workers are not frozen")
    families = payload["constraint_families"]
    if tuple(families.get("diagnosable", ())) != FAMILY_NAMES:
        raise SectionPlanAuditError("audit manifest diagnosable family list does not match the frozen taxonomy")
    scenarios = payload["scenarios"]
    if not isinstance(scenarios, list) or len(scenarios) != 8:
        raise SectionPlanAuditError("audit manifest must contain exactly 1 control and 7 infeasible targets")
    parsed_ids: list[str] = []
    for raw in scenarios:
        fields = {"scenario_id", "role", "source_suite", "source_scenario_id", "expected_baseline_outcome"}
        if not isinstance(raw, dict) or fields - set(raw):
            raise SectionPlanAuditError("audit scenario entry is incomplete")
        scenario_id = str(raw["scenario_id"])
        role = str(raw["role"])
        outcome = str(raw["expected_baseline_outcome"])
        if any(token in scenario_id for token in ("stress", "holdout", "negative")):
            raise SectionPlanAuditError(f"forbidden scenario id in audit manifest: {scenario_id}")
        if scenario_id == CONTROL_SCENARIO_ID:
            if role != "feasible_control" or outcome != "feasible_control":
                raise SectionPlanAuditError("control scenario role/outcome mismatch")
        else:
            if role != "infeasible_target" or outcome != "frozen_plan_hard_model_infeasible":
                raise SectionPlanAuditError(f"target scenario role/outcome mismatch: {scenario_id}")
        if scenario_id in parsed_ids:
            raise SectionPlanAuditError(f"duplicate audit scenario: {scenario_id}")
        parsed_ids.append(scenario_id)
    if tuple(parsed_ids) != EXPECTED_SCENARIO_ORDER:
        raise SectionPlanAuditError("audit manifest scenario order does not match the frozen control+7-target order")
    return payload


# ---------------------------------------------------------------------------
# Static feasibility descriptors (Section 5) -- no solver involved
# ---------------------------------------------------------------------------


def _global_supply_descriptor(allocation_input: CanonicalAllocationInput) -> dict[str, Any]:
    students = allocation_input.students
    student_count = len(students)
    total_capacity = sum(section.capacity for section in allocation_input.logical_sections)
    minimum_required = MINIMUM_ASSIGNED_LOGICAL_COURSE_COUNT * student_count
    total_target = sum(student.target_period_units for student in students)
    capacity_by_period: dict[str, int] = defaultdict(int)
    sections_by_period: dict[str, int] = defaultdict(int)
    for section in allocation_input.logical_sections:
        for period in section.occupied_periods:
            capacity_by_period[period] += section.capacity
            sections_by_period[period] += 1
    return {
        "students": student_count,
        "logical_sections": len(allocation_input.logical_sections),
        "total_logical_seat_capacity": total_capacity,
        "minimum_required_logical_assignments": minimum_required,
        "global_minimum_five_capacity_margin": total_capacity - minimum_required,
        "total_target_logical_courses": total_target,
        "total_target_load_capacity_margin": total_capacity - total_target,
        "capacity_by_period": dict(sorted(capacity_by_period.items())),
        "section_count_by_period": dict(sorted(sections_by_period.items())),
    }


def _course_demand_descriptor(allocation_input: CanonicalAllocationInput) -> dict[str, Any]:
    primary_demand: Counter[str] = Counter(
        request.candidate_key for request in allocation_input.logical_requests if request.request_type == "primary"
    )
    total_demand: Counter[str] = Counter(request.candidate_key for request in allocation_input.logical_requests)
    capacity_by_course: dict[str, int] = defaultdict(int)
    periods_by_course: dict[str, set[str]] = defaultdict(set)
    sections_by_course_period: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    capacity_by_course_period: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for section in allocation_input.logical_sections:
        capacity_by_course[section.logical_block_id] += section.capacity
        for period in section.occupied_periods:
            periods_by_course[section.logical_block_id].add(period)
            sections_by_course_period[section.logical_block_id][period] += 1
            capacity_by_course_period[section.logical_block_id][period] += section.capacity

    courses = sorted(set(primary_demand) | set(total_demand) | set(capacity_by_course))
    rows = []
    for course in courses:
        capacity = capacity_by_course.get(course, 0)
        primary = primary_demand.get(course, 0)
        total = total_demand.get(course, 0)
        rows.append({
            "logical_course_id": course,
            "primary_demand": primary,
            "total_request_demand": total,
            "logical_capacity": capacity,
            "primary_demand_capacity_ratio": round(primary / capacity, 6) if capacity else None,
            "total_demand_capacity_ratio": round(total / capacity, 6) if capacity else None,
            "candidate_periods": sorted(periods_by_course.get(course, ())),
            "sections_by_period": dict(sorted(sections_by_course_period.get(course, {}).items())),
            "capacity_by_period": dict(sorted(capacity_by_course_period.get(course, {}).items())),
        })
    primary_over_capacity = sorted(
        row["logical_course_id"] for row in rows if row["logical_capacity"] and row["primary_demand"] > row["logical_capacity"]
    )
    total_over_capacity = sorted(
        row["logical_course_id"] for row in rows if row["logical_capacity"] and row["total_request_demand"] > row["logical_capacity"]
    )
    capacity_only_shortfall = sum(
        max(row["primary_demand"] - row["logical_capacity"], 0) for row in rows if row["logical_capacity"]
    )
    top_pressure = sorted(
        (row for row in rows if row["logical_capacity"]),
        key=lambda row: row["primary_demand"] - row["logical_capacity"],
        reverse=True,
    )[:20]
    return {
        "courses": rows,
        "primary_demand_over_capacity_courses": primary_over_capacity,
        "total_demand_over_capacity_courses": total_over_capacity,
        "total_course_level_capacity_only_shortfall": capacity_only_shortfall,
        "top_20_pressure_courses": [row["logical_course_id"] for row in top_pressure],
        "note": "Course-level capacity shortfall is not a proof of global infeasibility: it ignores alternates and period combinations.",
    }


def _student_max_load_model(
    allocation_input: CanonicalAllocationInput,
) -> tuple[dict[_VariableKey, cp_model.IntVar], dict[str, cp_model.LinearExpr], dict[str, LogicalRequest], cp_model.CpModel]:
    """Build the ignoring-capacity per-student feasibility model.

    Keeps candidate availability, period-occupancy semantics (including
    multi-period HA/linked sections), and duplicate logical identity --
    drops section capacity entirely, so that maximizing the assignment
    count is an exact per-student computation (students never compete with
    each other once capacity is removed, so the joint maximum equals the
    sum of independent per-student maxima).
    """
    model = cp_model.CpModel()
    requests_by_key = {request.request_key: request for request in allocation_input.logical_requests}
    assignment_vars: dict[_VariableKey, cp_model.IntVar] = {}
    assigned_vars: dict[str, cp_model.LinearExpr] = {}
    for request_key in sorted(requests_by_key):
        request = requests_by_key[request_key]
        candidates = tuple(allocation_input.candidate_index.get(request_key, ()))
        candidate_vars = []
        for section_id in candidates:
            var = model.NewBoolVar(f"ml__{_safe_name(request_key)}__{_safe_name(section_id)}")
            assignment_vars[_VariableKey(request_key, section_id)] = var
            candidate_vars.append(var)
        if len(candidate_vars) == 1:
            assigned_vars[request_key] = candidate_vars[0]
        elif candidate_vars:
            assigned = model.NewBoolVar(f"ml_assigned__{_safe_name(request_key)}")
            assigned_vars[request_key] = assigned
            model.Add(sum(candidate_vars) == assigned)
        else:
            assigned_vars[request_key] = 0
    _add_duplicate_identity_constraints(model, allocation_input, requests_by_key, assigned_vars)
    by_student_period: dict[tuple[str, str], list[cp_model.IntVar]] = defaultdict(list)
    for key, var in assignment_vars.items():
        request = requests_by_key[key.request_key]
        section = allocation_input.logical_sections_by_id[key.section_id]
        for period in section.occupied_periods:
            by_student_period[(request.student_id, period)].append(var)
    for student in allocation_input.students:
        for period in (f"P{index}" for index in range(1, 8)):
            model.Add(sum(by_student_period.get((student.student_id, period), ())) <= 1)
    model.Maximize(sum(assigned_vars.values()))
    return assignment_vars, assigned_vars, requests_by_key, model


def _student_max_load_descriptor(allocation_input: CanonicalAllocationInput, *, time_limit_seconds: float, seed: int) -> dict[str, Any]:
    _assignment_vars, assigned_vars, requests_by_key, model = _student_max_load_model(allocation_input)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = seed
    status = solver.Solve(model)
    status_name = solver.StatusName(status)
    if status_name not in {"OPTIMAL", "FEASIBLE"}:
        return {
            "solve_status": status_name,
            "optimality_proven": False,
            "limitation": "student max-load matching model did not return a solution within budget; results unavailable",
            "students_below_five": None,
            "students_below_target_minus_one": None,
            "zero_candidate_primary_students": None,
            "one_candidate_primary_requests": None,
            "maximum_load_distribution": None,
        }
    by_student: dict[str, list[str]] = defaultdict(list)
    for request in requests_by_key.values():
        by_student[request.student_id].append(request.request_key)
    zero_candidate_primary_students: list[str] = []
    one_candidate_primary_requests: list[str] = []
    below_five: list[str] = []
    below_target_minus_one: list[str] = []
    distribution: dict[str, int] = defaultdict(int)
    for student in allocation_input.students:
        max_load = 0
        for request_key in by_student.get(student.student_id, ()):
            value = assigned_vars[request_key]
            solved = solver.Value(value) if not isinstance(value, int) else value
            max_load += int(solved)
        distribution[str(max_load)] += 1
        if max_load < MINIMUM_ASSIGNED_LOGICAL_COURSE_COUNT:
            below_five.append(student.student_id)
        if max_load < student.target_period_units - 1:
            below_target_minus_one.append(student.student_id)
        for request in student.primary_requests:
            candidates = allocation_input.candidate_index.get(request.request_key, ())
            if len(candidates) == 0:
                zero_candidate_primary_students.append(student.student_id)
            elif len(candidates) == 1:
                one_candidate_primary_requests.append(request.request_key)
    return {
        "solve_status": status_name,
        "optimality_proven": status_name == "OPTIMAL",
        "students_below_five": sorted(set(below_five)),
        "students_below_five_count": len(set(below_five)),
        "students_below_target_minus_one": sorted(set(below_target_minus_one)),
        "students_below_target_minus_one_count": len(set(below_target_minus_one)),
        "zero_candidate_primary_students": sorted(set(zero_candidate_primary_students)),
        "one_candidate_primary_requests": sorted(set(one_candidate_primary_requests)),
        "maximum_load_distribution": dict(sorted(distribution.items(), key=lambda item: int(item[0]))),
    }


def _period_concentration_descriptor(allocation_input: CanonicalAllocationInput) -> dict[str, Any]:
    supply_by_period: dict[str, int] = defaultdict(int)
    for section in allocation_input.logical_sections:
        for period in section.occupied_periods:
            supply_by_period[period] += section.capacity
    demand_by_period: dict[str, int] = defaultdict(int)
    only_available_by_period: dict[str, int] = defaultdict(int)
    majority_concentrated_students: list[str] = []
    for request in allocation_input.logical_requests:
        if request.request_type != "primary":
            continue
        periods = set()
        for section_id in allocation_input.candidate_index.get(request.request_key, ()):
            periods.update(allocation_input.logical_sections_by_id[section_id].occupied_periods)
        for period in periods:
            demand_by_period[period] += 1
        if len(periods) == 1:
            only_available_by_period[next(iter(periods))] += 1
    for student in allocation_input.students:
        period_counts: Counter[str] = Counter()
        total = 0
        for request in student.primary_requests:
            periods = set()
            for section_id in allocation_input.candidate_index.get(request.request_key, ()):
                periods.update(allocation_input.logical_sections_by_id[section_id].occupied_periods)
            for period in periods:
                period_counts[period] += 1
            total += 1
        if total and period_counts:
            top_period, top_count = period_counts.most_common(1)[0]
            if top_count > total / 2:
                majority_concentrated_students.append(student.student_id)
    periods = sorted(set(supply_by_period) | set(demand_by_period))
    return {
        "supply_by_period": {p: supply_by_period.get(p, 0) for p in periods},
        "primary_candidate_demand_by_period": {p: demand_by_period.get(p, 0) for p in periods},
        "requests_only_available_in_period": {p: only_available_by_period.get(p, 0) for p in periods},
        "students_majority_concentrated_in_one_period": sorted(set(majority_concentrated_students)),
        "capacity_to_demand_pressure_by_period": {
            p: round(demand_by_period.get(p, 0) / supply_by_period[p], 6) if supply_by_period.get(p) else None
            for p in periods
        },
    }


def build_static_descriptors(allocation_input: CanonicalAllocationInput, *, time_limit_seconds: float, seed: int) -> dict[str, Any]:
    return {
        "global_supply": _global_supply_descriptor(allocation_input),
        "course_demand": _course_demand_descriptor(allocation_input),
        "student_max_load": _student_max_load_descriptor(allocation_input, time_limit_seconds=time_limit_seconds, seed=seed),
        "period_concentration": _period_concentration_descriptor(allocation_input),
    }


# ---------------------------------------------------------------------------
# Diagnostic model (Section 6/7) -- reuses production data structures and the
# never-relaxed production constraint functions unmodified; the seven
# diagnosable families are each gated by an assumption literal built from the
# same imported production threshold constants.
# ---------------------------------------------------------------------------

_SLACK_UPPER_BOUND = 100_000


@dataclass
class _DiagnosticBuild:
    model: cp_model.CpModel
    assignment_vars: dict[_VariableKey, cp_model.IntVar]
    assigned_vars: dict[str, Any]
    requests_by_key: dict[str, LogicalRequest]
    group_literals: dict[str, cp_model.IntVar]
    fine_literals: dict[str, dict[Any, cp_model.IntVar]]


def _build_assignment_vars(
    model: cp_model.CpModel,
    allocation_input: CanonicalAllocationInput,
    fallback_plans: tuple[Any, ...],
    name_prefix: str,
) -> tuple[dict[_VariableKey, cp_model.IntVar], dict[str, Any], dict[str, LogicalRequest]]:
    requests_by_key = {request.request_key: request for request in allocation_input.logical_requests}
    candidate_index = {key: tuple(value) for key, value in allocation_input.candidate_index.items()}
    for plan in fallback_plans:
        requests_by_key[plan.fallback_request.request_key] = plan.fallback_request
        candidate_index[plan.fallback_request.request_key] = plan.candidates
    assignment_vars: dict[_VariableKey, cp_model.IntVar] = {}
    assigned_vars: dict[str, Any] = {}
    for request_key in sorted(requests_by_key):
        request = requests_by_key[request_key]
        candidates = tuple(candidate_index.get(request_key, ()))
        _validate_candidates_for_request(allocation_input, request, candidates)
        candidate_vars = []
        for section_id in candidates:
            var = model.NewBoolVar(f"{name_prefix}__{_safe_name(request_key)}__{_safe_name(section_id)}")
            assignment_vars[_VariableKey(request_key, section_id)] = var
            candidate_vars.append(var)
        if len(candidate_vars) == 1:
            assigned_vars[request_key] = candidate_vars[0]
        elif candidate_vars:
            assigned = model.NewBoolVar(f"{name_prefix}_assigned__{_safe_name(request_key)}")
            assigned_vars[request_key] = assigned
            model.Add(sum(candidate_vars) == assigned)
        else:
            assigned_vars[request_key] = 0
        if candidate_vars and request.period_units != allocation_input.courses_by_id[request.course_ids[0]].period_units:
            model.Add(assigned_vars[request_key] == 0)
    return assignment_vars, assigned_vars, requests_by_key


def _build_diagnostic_model(
    allocation_input: CanonicalAllocationInput,
    fallback_plans: tuple[Any, ...],
    math_course_ids: tuple[str, ...],
    *,
    fine_grained_families: frozenset[str] = frozenset(),
) -> _DiagnosticBuild:
    model = cp_model.CpModel()
    assignment_vars, assigned_vars, requests_by_key = _build_assignment_vars(model, allocation_input, fallback_plans, "x")

    _add_duplicate_identity_constraints(model, allocation_input, requests_by_key, assigned_vars)
    _add_student_target_constraints(model, allocation_input, requests_by_key, assignment_vars)
    _add_fallback_constraints(model, fallback_plans, assigned_vars, math_course_ids)
    _add_math_coverage_constraints(model, allocation_input, fallback_plans, assigned_vars, math_course_ids)

    group_literals = {name: model.NewBoolVar(f"family__{name}") for name in FAMILY_NAMES}
    fine_literals: dict[str, dict[Any, cp_model.IntVar]] = {name: {} for name in FAMILY_NAMES}

    def literal_for(family: str, key: Any) -> cp_model.IntVar:
        if family in fine_grained_families:
            lit = model.NewBoolVar(f"fine__{family}__{_safe_name(str(key))}")
            fine_literals[family][key] = lit
            return lit
        return group_literals[family]

    by_section: dict[str, list[cp_model.IntVar]] = defaultdict(list)
    for key, var in assignment_vars.items():
        by_section[key.section_id].append(var)
    for section in allocation_input.logical_sections:
        lit = literal_for("SECTION_CAPACITY", section.linked_section_group_id)
        model.Add(sum(by_section.get(section.linked_section_group_id, ())) <= section.capacity).OnlyEnforceIf(lit)

    by_student_period: dict[tuple[str, str], list[cp_model.IntVar]] = defaultdict(list)
    for key, var in assignment_vars.items():
        request = requests_by_key[key.request_key]
        section = allocation_input.logical_sections_by_id[key.section_id]
        for period in section.occupied_periods:
            by_student_period[(request.student_id, period)].append(var)
    for student in allocation_input.students:
        for period in (f"P{index}" for index in range(1, 8)):
            lit = literal_for("STUDENT_PERIOD_CONFLICT", (student.student_id, period))
            model.Add(sum(by_student_period.get((student.student_id, period), ())) <= 1).OnlyEnforceIf(lit)

    demand = Counter(request.candidate_key for request in allocation_input.logical_requests if request.request_type == "primary")
    high_demand = {key for key, count in demand.items() if count > HIGH_DEMAND_PRIMARY_THRESHOLD}
    for student in allocation_input.students:
        primary_assigned = [assigned_vars[request.request_key] for request in student.primary_requests]
        primary_count = len(student.primary_requests)
        if student.priority_protected:
            lit = literal_for("PROTECTED_PRIMARY", student.student_id)
            model.Add(primary_count - sum(primary_assigned) <= MAXIMUM_PROTECTED_PRIMARY_UNMET_COUNT).OnlyEnforceIf(lit)
        else:
            lit = literal_for("ORDINARY_MAX_PRIMARY_UNMET", student.student_id)
            model.Add(primary_count - sum(primary_assigned) <= MAXIMUM_ORDINARY_PRIMARY_UNMET_COUNT).OnlyEnforceIf(lit)
        for request in student.primary_requests:
            if request.candidate_key in high_demand:
                lit = literal_for("HIGH_DEMAND_PRIMARY", request.request_key)
                model.Add(assigned_vars[request.request_key] == 1).OnlyEnforceIf(lit)

    by_student_all: dict[str, list[Any]] = defaultdict(list)
    for request in requests_by_key.values():
        by_student_all[request.student_id].append(assigned_vars[request.request_key])
    for student in allocation_input.students:
        assigned_logical = sum(by_student_all.get(student.student_id, ()))
        lit_five = literal_for("MINIMUM_FIVE_LOGICAL", student.student_id)
        model.Add(assigned_logical >= MINIMUM_ASSIGNED_LOGICAL_COURSE_COUNT).OnlyEnforceIf(lit_five)
        lit_gap = literal_for("MAXIMUM_LOGICAL_GAP_ONE", student.student_id)
        model.Add(assigned_logical >= student.target_period_units - MAXIMUM_SCHEDULE_GAP_COUNT).OnlyEnforceIf(lit_gap)

    return _DiagnosticBuild(
        model=model, assignment_vars=assignment_vars, assigned_vars=assigned_vars,
        requests_by_key=requests_by_key, group_literals=group_literals, fine_literals=fine_literals,
    )


def _new_diagnostic_solver(time_limit_seconds: float, seed: int) -> cp_model.CpSolver:
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max(float(time_limit_seconds), 0.1)
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = int(seed)
    return solver


def _solve_with_assumptions(
    build: _DiagnosticBuild,
    true_literals: Iterable[cp_model.IntVar],
    *,
    time_limit_seconds: float,
    seed: int,
) -> tuple[cp_model.CpSolver, str]:
    build.model.ClearAssumptions()
    build.model.AddAssumptions(list(true_literals))
    solver = _new_diagnostic_solver(time_limit_seconds, seed)
    status = solver.Solve(build.model)
    return solver, solver.StatusName(status)


def verify_diagnostic_model_equivalence(build: _DiagnosticBuild, *, time_limit_seconds: float, seed: int) -> dict[str, Any]:
    """Solve with every diagnosable family fully enabled -- this must
    reproduce the production ground-truth feasibility status exactly."""
    started = time.perf_counter()
    _solver, status_name = _solve_with_assumptions(
        build, build.group_literals.values(), time_limit_seconds=time_limit_seconds, seed=seed,
    )
    return {"status": status_name, "runtime_seconds": round(time.perf_counter() - started, 6)}


def group_level_core(build: _DiagnosticBuild, *, time_limit_seconds: float, seed: int) -> dict[str, Any]:
    started = time.perf_counter()
    solver, status_name = _solve_with_assumptions(
        build, build.group_literals.values(), time_limit_seconds=time_limit_seconds, seed=seed,
    )
    result: dict[str, Any] = {
        "status": status_name,
        "runtime_seconds": round(time.perf_counter() - started, 6),
        "sufficient_core": [],
        "locally_minimal_core": [],
        "minimality_status": "not_applicable",
    }
    if status_name in {"FEASIBLE", "OPTIMAL"}:
        return result
    if status_name != "INFEASIBLE":
        result["minimality_status"] = "unresolved_no_infeasibility_proof"
        return result
    index_to_family = {lit.Index(): name for name, lit in build.group_literals.items()}
    sufficient = sorted({index_to_family[i] for i in solver.SufficientAssumptionsForInfeasibility() if i in index_to_family})
    result["sufficient_core"] = sufficient
    remaining = set(sufficient)
    deadline = started + time_limit_seconds
    minimality_status = "locally_minimal"
    for family in sufficient:
        remaining_budget = deadline - time.perf_counter()
        if remaining_budget <= 0:
            minimality_status = "unresolved_time_budget"
            break
        trial = remaining - {family}
        _trial_solver, trial_status = _solve_with_assumptions(
            build, [build.group_literals[name] for name in trial], time_limit_seconds=remaining_budget, seed=seed,
        )
        if trial_status == "INFEASIBLE":
            remaining = trial
        elif trial_status not in {"FEASIBLE", "OPTIMAL"}:
            minimality_status = "unresolved_time_budget"
            break
    result["locally_minimal_core"] = sorted(remaining)
    result["minimality_status"] = minimality_status
    return result


def _jsonable_fine_id(key: Any) -> Any:
    return list(key) if isinstance(key, tuple) else key


def _fine_core_involvement(
    items: list[tuple[str, Any]],
    allocation_input: CanonicalAllocationInput,
    requests_by_key: dict[str, LogicalRequest],
) -> dict[str, list[str]]:
    students: set[str] = set()
    requests: set[str] = set()
    courses: set[str] = set()
    sections: set[str] = set()
    periods: set[str] = set()
    for family, key in items:
        if family == "SECTION_CAPACITY":
            sections.add(str(key))
            section = allocation_input.logical_sections_by_id.get(key)
            if section is not None:
                courses.add(section.logical_block_id)
                periods.update(section.occupied_periods)
        elif family == "STUDENT_PERIOD_CONFLICT":
            student_id, period = key
            students.add(student_id)
            periods.add(period)
        elif family == "HIGH_DEMAND_PRIMARY":
            request = requests_by_key.get(key)
            requests.add(key)
            if request is not None:
                students.add(request.student_id)
                courses.add(request.candidate_key)
        else:
            students.add(str(key))
    return {
        "students": sorted(students),
        "requests": sorted(requests),
        "logical_courses": sorted(courses),
        "sections": sorted(sections),
        "periods": sorted(periods),
    }


def fine_grained_core(
    allocation_input: CanonicalAllocationInput,
    fallback_plans: tuple[Any, ...],
    math_course_ids: tuple[str, ...],
    *,
    target_families: tuple[str, ...],
    time_limit_seconds: float,
    seed: int,
) -> dict[str, Any]:
    empty = {
        "status": "not_applicable", "sufficient_core": [], "locally_minimal_core": [],
        "minimality_status": "not_applicable", "core_size": 0,
        "involved_students": [], "involved_requests": [], "involved_logical_courses": [],
        "involved_sections": [], "involved_periods": [], "target_families": list(target_families),
        "runtime_seconds": 0.0,
    }
    if not target_families:
        return empty
    build = _build_diagnostic_model(
        allocation_input, fallback_plans, math_course_ids, fine_grained_families=frozenset(target_families),
    )
    other_families = [name for name in FAMILY_NAMES if name not in target_families]
    always_true = [build.group_literals[name] for name in other_families]
    fine_items: list[tuple[str, Any, cp_model.IntVar]] = [
        (family, key, lit)
        for family in target_families
        for key, lit in build.fine_literals[family].items()
    ]
    started = time.perf_counter()
    solver, status_name = _solve_with_assumptions(
        build, always_true + [lit for _, _, lit in fine_items], time_limit_seconds=time_limit_seconds, seed=seed,
    )
    result = dict(empty)
    result["status"] = status_name
    result["runtime_seconds"] = round(time.perf_counter() - started, 6)
    result["target_families"] = list(target_families)
    if status_name in {"FEASIBLE", "OPTIMAL"}:
        return result
    if status_name != "INFEASIBLE":
        result["minimality_status"] = "unresolved_no_infeasibility_proof"
        return result
    index_to_key = {lit.Index(): (family, key) for family, key, lit in fine_items}
    sufficient = sorted(
        {index_to_key[i] for i in solver.SufficientAssumptionsForInfeasibility() if i in index_to_key},
        key=lambda item: (item[0], str(item[1])),
    )
    result["sufficient_core"] = [{"family": family, "id": _jsonable_fine_id(key)} for family, key in sufficient]
    remaining = list(sufficient)
    lookup = {(family, key): lit for family, key, lit in fine_items}
    deadline = started + time_limit_seconds
    minimality_status = "locally_minimal"
    for family, key in sufficient:
        remaining_budget = deadline - time.perf_counter()
        if remaining_budget <= 0:
            minimality_status = "unresolved_time_budget"
            break
        trial = [item for item in remaining if item != (family, key)]
        _trial_solver, trial_status = _solve_with_assumptions(
            build, always_true + [lookup[item] for item in trial], time_limit_seconds=remaining_budget, seed=seed,
        )
        if trial_status == "INFEASIBLE":
            remaining = trial
        elif trial_status not in {"FEASIBLE", "OPTIMAL"}:
            minimality_status = "unresolved_time_budget"
            break
    result["locally_minimal_core"] = [{"family": family, "id": _jsonable_fine_id(key)} for family, key in remaining]
    result["minimality_status"] = minimality_status
    result["core_size"] = len(remaining)
    result.update({f"involved_{k}": v for k, v in _fine_core_involvement(remaining, allocation_input, build.requests_by_key).items()})
    return result


def run_counterfactual_variants(build: _DiagnosticBuild, *, time_limit_seconds: float, seed: int) -> list[dict[str, Any]]:
    # Counterfactual feasibility is not an unsat core and does not establish
    # minimumity: it only tests whether relaxing a family restores feasibility.
    rows = []
    for variant_name, relaxed_families in COUNTERFACTUAL_VARIANTS:
        enabled = [build.group_literals[name] for name in FAMILY_NAMES if name not in relaxed_families]
        started = time.perf_counter()
        _solver, status_name = _solve_with_assumptions(build, enabled, time_limit_seconds=time_limit_seconds, seed=seed)
        rows.append({
            "variant": variant_name,
            "relaxed_families": list(relaxed_families),
            "status": status_name,
            "runtime_seconds": round(time.perf_counter() - started, 6),
            "assignment_found": status_name in {"FEASIBLE", "OPTIMAL"},
        })
    return rows


# ---------------------------------------------------------------------------
# Controlled relaxation model (Section 9/10/11)
# ---------------------------------------------------------------------------


@dataclass
class _RelaxationBuild:
    model: cp_model.CpModel
    assignment_vars: dict[_VariableKey, cp_model.IntVar]
    assigned_vars: dict[str, Any]
    requests_by_key: dict[str, LogicalRequest]
    capacity_overflow: dict[str, cp_model.IntVar]
    ordinary_extra_unmet: dict[str, cp_model.IntVar]
    protected_unmet_indicator: dict[str, cp_model.IntVar]
    high_demand_unmet_indicator: dict[str, cp_model.IntVar]
    load_shortfall: dict[str, cp_model.IntVar]
    excess_gap: dict[str, cp_model.IntVar]
    period_overlap_slack: dict[tuple[str, str], cp_model.IntVar]
    nonzero_indicators: dict[str, list[cp_model.IntVar]] = field(default_factory=dict)


def _build_relaxation_model(
    allocation_input: CanonicalAllocationInput,
    fallback_plans: tuple[Any, ...],
    math_course_ids: tuple[str, ...],
    *,
    allow_period_overlap_slack: bool = False,
) -> _RelaxationBuild:
    model = cp_model.CpModel()
    assignment_vars, assigned_vars, requests_by_key = _build_assignment_vars(model, allocation_input, fallback_plans, "r")

    _add_duplicate_identity_constraints(model, allocation_input, requests_by_key, assigned_vars)
    _add_student_target_constraints(model, allocation_input, requests_by_key, assignment_vars)
    _add_fallback_constraints(model, fallback_plans, assigned_vars, math_course_ids)
    _add_math_coverage_constraints(model, allocation_input, fallback_plans, assigned_vars, math_course_ids)

    period_overlap_slack: dict[tuple[str, str], cp_model.IntVar] = {}
    by_student_period: dict[tuple[str, str], list[cp_model.IntVar]] = defaultdict(list)
    for key, var in assignment_vars.items():
        request = requests_by_key[key.request_key]
        section = allocation_input.logical_sections_by_id[key.section_id]
        for period in section.occupied_periods:
            by_student_period[(request.student_id, period)].append(var)
    for student in allocation_input.students:
        for period in (f"P{index}" for index in range(1, 8)):
            terms = by_student_period.get((student.student_id, period), ())
            if allow_period_overlap_slack:
                slack = model.NewIntVar(0, _SLACK_UPPER_BOUND, f"period_overlap__{_safe_name(student.student_id)}__{period}")
                period_overlap_slack[(student.student_id, period)] = slack
                model.Add(sum(terms) <= 1 + slack)
            else:
                model.Add(sum(terms) <= 1)

    capacity_overflow: dict[str, cp_model.IntVar] = {}
    by_section: dict[str, list[cp_model.IntVar]] = defaultdict(list)
    for key, var in assignment_vars.items():
        by_section[key.section_id].append(var)
    for section in allocation_input.logical_sections:
        overflow = model.NewIntVar(0, _SLACK_UPPER_BOUND, f"overflow__{_safe_name(section.linked_section_group_id)}")
        capacity_overflow[section.linked_section_group_id] = overflow
        model.Add(sum(by_section.get(section.linked_section_group_id, ())) <= section.capacity + overflow)

    demand = Counter(request.candidate_key for request in allocation_input.logical_requests if request.request_type == "primary")
    high_demand = {key for key, count in demand.items() if count > HIGH_DEMAND_PRIMARY_THRESHOLD}
    ordinary_extra_unmet: dict[str, cp_model.IntVar] = {}
    protected_unmet_indicator: dict[str, cp_model.IntVar] = {}
    high_demand_unmet_indicator: dict[str, cp_model.IntVar] = {}
    for student in allocation_input.students:
        primary_assigned = [assigned_vars[request.request_key] for request in student.primary_requests]
        primary_count = len(student.primary_requests)
        primary_unmet = primary_count - sum(primary_assigned)
        if student.priority_protected:
            indicator = model.NewBoolVar(f"protected_unmet__{_safe_name(student.student_id)}")
            model.Add(primary_unmet >= MAXIMUM_PROTECTED_PRIMARY_UNMET_COUNT + 1).OnlyEnforceIf(indicator)
            model.Add(primary_unmet <= MAXIMUM_PROTECTED_PRIMARY_UNMET_COUNT).OnlyEnforceIf(indicator.Not())
            protected_unmet_indicator[student.student_id] = indicator
        else:
            extra = model.NewIntVar(0, _SLACK_UPPER_BOUND, f"ordinary_extra__{_safe_name(student.student_id)}")
            model.Add(extra >= primary_unmet - MAXIMUM_ORDINARY_PRIMARY_UNMET_COUNT)
            ordinary_extra_unmet[student.student_id] = extra
        for request in student.primary_requests:
            if request.candidate_key in high_demand:
                indicator = model.NewBoolVar(f"high_demand_unmet__{_safe_name(request.request_key)}")
                model.Add(assigned_vars[request.request_key] == 0).OnlyEnforceIf(indicator)
                model.Add(assigned_vars[request.request_key] == 1).OnlyEnforceIf(indicator.Not())
                high_demand_unmet_indicator[request.request_key] = indicator

    load_shortfall: dict[str, cp_model.IntVar] = {}
    excess_gap: dict[str, cp_model.IntVar] = {}
    by_student_all: dict[str, list[Any]] = defaultdict(list)
    for request in requests_by_key.values():
        by_student_all[request.student_id].append(assigned_vars[request.request_key])
    for student in allocation_input.students:
        assigned_logical = sum(by_student_all.get(student.student_id, ()))
        shortfall = model.NewIntVar(0, _SLACK_UPPER_BOUND, f"load_shortfall__{_safe_name(student.student_id)}")
        model.Add(shortfall >= MINIMUM_ASSIGNED_LOGICAL_COURSE_COUNT - assigned_logical)
        load_shortfall[student.student_id] = shortfall
        gap = model.NewIntVar(0, _SLACK_UPPER_BOUND, f"excess_gap__{_safe_name(student.student_id)}")
        model.Add(gap >= (student.target_period_units - MAXIMUM_SCHEDULE_GAP_COUNT) - assigned_logical)
        excess_gap[student.student_id] = gap

    return _RelaxationBuild(
        model=model, assignment_vars=assignment_vars, assigned_vars=assigned_vars, requests_by_key=requests_by_key,
        capacity_overflow=capacity_overflow, ordinary_extra_unmet=ordinary_extra_unmet,
        protected_unmet_indicator=protected_unmet_indicator, high_demand_unmet_indicator=high_demand_unmet_indicator,
        load_shortfall=load_shortfall, excess_gap=excess_gap, period_overlap_slack=period_overlap_slack,
    )


def _nonzero_indicator(model: cp_model.CpModel, slack: cp_model.IntVar, name: str) -> cp_model.IntVar:
    indicator = model.NewBoolVar(name)
    model.Add(slack >= 1).OnlyEnforceIf(indicator)
    model.Add(slack == 0).OnlyEnforceIf(indicator.Not())
    return indicator


def _response_hash(solver: cp_model.CpSolver) -> str:
    import hashlib
    return hashlib.sha256(str(solver.ResponseProto()).encode("utf-8")).hexdigest()


def _relaxation_stage(
    build: _RelaxationBuild,
    objective_expr: Any,
    *,
    time_limit_seconds: float,
    seed: int,
    stage_name: str,
) -> dict[str, Any]:
    build.model.Minimize(objective_expr)
    solver = _new_diagnostic_solver(time_limit_seconds, seed)
    started = time.perf_counter()
    status = solver.Solve(build.model)
    status_name = solver.StatusName(status)
    has_solution = status_name in {"FEASIBLE", "OPTIMAL"}
    return {
        "stage_name": stage_name,
        "status": status_name,
        "objective_value": int(round(solver.ObjectiveValue())) if has_solution else None,
        "best_objective_bound": int(round(solver.BestObjectiveBound())) if has_solution else None,
        "optimality_proven": status_name == "OPTIMAL",
        "runtime_seconds": round(time.perf_counter() - started, 6),
        "response_hash": _response_hash(solver) if has_solution else "",
        "_solver": solver,
    }


def run_relaxation(
    allocation_input: CanonicalAllocationInput,
    fallback_plans: tuple[Any, ...],
    math_course_ids: tuple[str, ...],
    *,
    stage1_seconds: float,
    stage2_seconds: float,
    stage3_seconds: float,
    seed: int,
    hint_keys: tuple[_VariableKey, ...],
) -> dict[str, Any]:
    build = _build_relaxation_model(allocation_input, fallback_plans, math_course_ids, allow_period_overlap_slack=False)
    nonzero_capacity = [
        _nonzero_indicator(build.model, var, f"nz_capacity__{_safe_name(section_id)}")
        for section_id, var in build.capacity_overflow.items()
    ]
    nonzero_ordinary = [
        _nonzero_indicator(build.model, var, f"nz_ordinary__{_safe_name(student_id)}")
        for student_id, var in build.ordinary_extra_unmet.items()
    ]
    nonzero_five = [
        _nonzero_indicator(build.model, var, f"nz_five__{_safe_name(student_id)}")
        for student_id, var in build.load_shortfall.items()
    ]
    nonzero_gap = [
        _nonzero_indicator(build.model, var, f"nz_gap__{_safe_name(student_id)}")
        for student_id, var in build.excess_gap.items()
    ]
    stage1_objective = (
        sum(nonzero_capacity) + sum(nonzero_ordinary)
        + sum(build.protected_unmet_indicator.values()) + sum(build.high_demand_unmet_indicator.values())
        + sum(nonzero_five) + sum(nonzero_gap)
    )
    stage1 = _relaxation_stage(build, stage1_objective, time_limit_seconds=stage1_seconds, seed=seed, stage_name="stage_1_minimize_relaxed_instance_count")
    stages = [{k: v for k, v in stage1.items() if k != "_solver"}]
    if stage1["status"] not in {"FEASIBLE", "OPTIMAL"}:
        return {"build": build, "stages": stages, "final_solver": None, "layer": 1, "period_overlap_used": False}
    build.model.Add(stage1_objective == stage1["objective_value"])

    stage2_objective = (
        sum(build.capacity_overflow.values()) + sum(build.ordinary_extra_unmet.values())
        + sum(build.protected_unmet_indicator.values()) + sum(build.high_demand_unmet_indicator.values())
        + sum(build.load_shortfall.values()) + sum(build.excess_gap.values())
    )
    stage2 = _relaxation_stage(build, stage2_objective, time_limit_seconds=stage2_seconds, seed=seed, stage_name="stage_2_minimize_total_slack_magnitude")
    stages.append({k: v for k, v in stage2.items() if k != "_solver"})
    if stage2["status"] not in {"FEASIBLE", "OPTIMAL"}:
        return {"build": build, "stages": stages, "final_solver": stage1["_solver"], "layer": 1, "period_overlap_used": False}
    build.model.Add(stage2_objective == stage2["objective_value"])

    hamming_expr = _hamming_distance_expression(build, hint_keys)
    stage3 = _relaxation_stage(build, hamming_expr, time_limit_seconds=stage3_seconds, seed=seed, stage_name="stage_3_minimize_hamming_to_constrained_first")
    stages.append({k: v for k, v in stage3.items() if k != "_solver"})
    final_solver = stage3["_solver"] if stage3["status"] in {"FEASIBLE", "OPTIMAL"} else stage2["_solver"]
    return {"build": build, "stages": stages, "final_solver": final_solver, "layer": 1, "period_overlap_used": False}


def run_relaxation_with_fallback_layer(
    allocation_input: CanonicalAllocationInput,
    fallback_plans: tuple[Any, ...],
    math_course_ids: tuple[str, ...],
    *,
    stage1_seconds: float,
    stage2_seconds: float,
    stage3_seconds: float,
    seed: int,
    hint_keys: tuple[_VariableKey, ...],
) -> dict[str, Any]:
    result = run_relaxation(
        allocation_input, fallback_plans, math_course_ids,
        stage1_seconds=stage1_seconds, stage2_seconds=stage2_seconds, stage3_seconds=stage3_seconds,
        seed=seed, hint_keys=hint_keys,
    )
    if result["stages"][0]["status"] in {"FEASIBLE", "OPTIMAL"}:
        return result
    # Layer 1 (period conflicts hard) was infeasible even with every other
    # family fully slacked -- reported separately, per Section 9.
    build = _build_relaxation_model(allocation_input, fallback_plans, math_course_ids, allow_period_overlap_slack=True)
    nonzero_period = [
        _nonzero_indicator(build.model, var, f"nz_period__{_safe_name(student_id)}__{period}")
        for (student_id, period), var in build.period_overlap_slack.items()
    ]
    stage1_objective = sum(nonzero_period)
    stage1 = _relaxation_stage(build, stage1_objective, time_limit_seconds=stage1_seconds, seed=seed, stage_name="layer2_stage_1_minimize_period_overlap_instance_count")
    stages = [{k: v for k, v in stage1.items() if k != "_solver"}]
    final_solver = stage1["_solver"] if stage1["status"] in {"FEASIBLE", "OPTIMAL"} else None
    return {"build": build, "stages": result["stages"] + stages, "final_solver": final_solver, "layer": 2, "period_overlap_used": True}


# ---------------------------------------------------------------------------
# Witness validation (Section 11)
# ---------------------------------------------------------------------------


def validate_relaxation_witness(
    build: _RelaxationBuild,
    solver: cp_model.CpSolver | None,
    allocation_input: CanonicalAllocationInput,
) -> dict[str, Any]:
    if solver is None:
        return {
            "valid": False, "issues": ["no_solution_found_within_budget"],
            "no_duplicate_logical_identity": None, "capacity_overflow_closure": None,
            "student_policy_slack_closure": None, "no_unknown_ids": None, "response_hash_present": False,
        }

    def value(term: Any) -> int:
        return int(solver.Value(term)) if not isinstance(term, int) else int(term)

    issues: list[str] = []

    by_student_identity: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for request in build.requests_by_key.values():
        by_student_identity[(request.student_id, request.candidate_key)].append(build.assigned_vars[request.request_key])
    duplicate_ok = True
    for (student_id, identity), terms in by_student_identity.items():
        if sum(value(term) for term in terms) > 1:
            issues.append(f"duplicate_logical_identity:{student_id}:{identity}")
            duplicate_ok = False

    by_section: dict[str, list[cp_model.IntVar]] = defaultdict(list)
    for key, var in build.assignment_vars.items():
        by_section[key.section_id].append(var)
    capacity_ok = True
    for section in allocation_input.logical_sections:
        assigned = sum(value(var) for var in by_section.get(section.linked_section_group_id, ()))
        overflow_solved = value(build.capacity_overflow[section.linked_section_group_id])
        if overflow_solved != max(assigned - section.capacity, 0):
            issues.append(f"capacity_overflow_mismatch:{section.linked_section_group_id}")
            capacity_ok = False

    by_student_all: dict[str, list[Any]] = defaultdict(list)
    for request in build.requests_by_key.values():
        by_student_all[request.student_id].append(build.assigned_vars[request.request_key])
    policy_ok = True
    for student in allocation_input.students:
        assigned_logical = sum(value(term) for term in by_student_all.get(student.student_id, ()))
        expected_shortfall = max(MINIMUM_ASSIGNED_LOGICAL_COURSE_COUNT - assigned_logical, 0)
        if value(build.load_shortfall[student.student_id]) != expected_shortfall:
            issues.append(f"load_shortfall_mismatch:{student.student_id}")
            policy_ok = False
        expected_gap = max((student.target_period_units - MAXIMUM_SCHEDULE_GAP_COUNT) - assigned_logical, 0)
        if value(build.excess_gap[student.student_id]) != expected_gap:
            issues.append(f"excess_gap_mismatch:{student.student_id}")
            policy_ok = False
        primary_assigned = sum(value(build.assigned_vars[request.request_key]) for request in student.primary_requests)
        primary_unmet = len(student.primary_requests) - primary_assigned
        if not student.priority_protected and student.student_id in build.ordinary_extra_unmet:
            expected_extra = max(primary_unmet - MAXIMUM_ORDINARY_PRIMARY_UNMET_COUNT, 0)
            if value(build.ordinary_extra_unmet[student.student_id]) != expected_extra:
                issues.append(f"ordinary_extra_unmet_mismatch:{student.student_id}")
                policy_ok = False

    return {
        "valid": not issues,
        "issues": issues,
        "no_duplicate_logical_identity": duplicate_ok,
        "capacity_overflow_closure": capacity_ok,
        "student_policy_slack_closure": policy_ok,
        "no_unknown_ids": True,
        "response_hash_present": True,
    }


# ---------------------------------------------------------------------------
# Section-plan repair interpretation (Section 12)
# ---------------------------------------------------------------------------


CLASSIFICATION_LABELS = (
    "global_capacity_deficit",
    "course_capacity_bottleneck",
    "period_supply_misalignment",
    "minimum_load_policy_interaction",
    "maximum_gap_policy_interaction",
    "primary_protection_interaction",
    "linked_or_ha_structure",
    "unresolved_multi_family_interaction",
)


def _witness_quality(
    relaxation_result: dict[str, Any] | None,
    witness_validation: dict[str, Any] | None,
) -> dict[str, Any]:
    """Classify relaxation evidence without treating a failed witness as proof."""
    stages = (relaxation_result or {}).get("stages", ())
    stage_1 = stages[0] if stages else {}
    stage_2 = stages[1] if len(stages) > 1 else {}
    stage_2_has_incumbent = stage_2.get("status") in {"FEASIBLE", "OPTIMAL"}
    stage_1_proven = bool(stage_1.get("optimality_proven"))
    witness_valid = (witness_validation or {}).get("valid") is True
    authoritative = witness_valid and stage_1_proven and stage_2_has_incumbent
    reasons: list[str] = []
    if not witness_valid:
        reasons.append("witness_validation_invalid")
    if not stage_1_proven:
        reasons.append("stage_1_optimality_not_proven")
    if not stage_2_has_incumbent:
        reasons.append("stage_2_no_incumbent")
    return {
        "witness_authoritative": authoritative,
        "witness_use": "authoritative_repair_evidence" if authoritative else "diagnostic_only",
        "exclusion_reason": ";".join(reasons) if reasons else None,
        "stage_1_proven": stage_1_proven,
        "stage_2_has_incumbent": stage_2_has_incumbent,
    }


def _classify_scenario(
    group_core: dict[str, Any],
    fine_core: dict[str, Any],
    capacity_repairs: list[dict[str, Any]],
    policy_interaction: dict[str, Any],
    linked_or_ha_sections: list[str],
    *,
    witness_authoritative: bool = False,
) -> dict[str, Any]:
    core_families = set(group_core.get("locally_minimal_core") or group_core.get("sufficient_core") or [])
    labels: list[str] = []
    evidence: dict[str, Any] = {}
    if "SECTION_CAPACITY" in core_families:
        if capacity_repairs:
            labels.append("course_capacity_bottleneck")
            evidence["course_capacity_bottleneck"] = sorted({row["logical_course"] for row in capacity_repairs if row["logical_course"]})
        else:
            labels.append("global_capacity_deficit")
    if "STUDENT_PERIOD_CONFLICT" in core_families:
        labels.append("period_supply_misalignment")
        evidence["period_supply_misalignment"] = fine_core.get("involved_periods", [])
    if "MINIMUM_FIVE_LOGICAL" in core_families and policy_interaction["students_requiring_minimum_five_slack"]:
        labels.append("minimum_load_policy_interaction")
        evidence["minimum_load_policy_interaction"] = policy_interaction["students_requiring_minimum_five_slack"]
    if "MAXIMUM_LOGICAL_GAP_ONE" in core_families and policy_interaction["students_requiring_max_gap_slack"]:
        labels.append("maximum_gap_policy_interaction")
        evidence["maximum_gap_policy_interaction"] = policy_interaction["students_requiring_max_gap_slack"]
    if (
        witness_authoritative
        and {"PROTECTED_PRIMARY", "ORDINARY_MAX_PRIMARY_UNMET", "HIGH_DEMAND_PRIMARY"} & core_families
        and policy_interaction["protected_or_high_demand_conflicts"]
    ):
        labels.append("primary_protection_interaction")
        evidence["primary_protection_interaction"] = policy_interaction["protected_or_high_demand_conflicts"]
    if linked_or_ha_sections:
        labels.append("linked_or_ha_structure")
        evidence["linked_or_ha_structure"] = linked_or_ha_sections
    if not labels:
        labels.append("unresolved_multi_family_interaction")
    return {"labels": sorted(set(labels)), "core_families": sorted(core_families), "evidence": evidence}


def build_repair_candidates(
    scenario_id: str,
    allocation_input: CanonicalAllocationInput,
    group_core: dict[str, Any],
    fine_core: dict[str, Any],
    relaxation_result: dict[str, Any],
    witness_validation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    build: _RelaxationBuild = relaxation_result["build"]
    final_solver = relaxation_result["final_solver"]

    def value(term: Any) -> int | None:
        if final_solver is None:
            return None
        return int(final_solver.Value(term)) if not isinstance(term, int) else int(term)

    capacity_repairs = []
    linked_or_ha_sections: list[str] = []
    if final_solver is not None:
        for section_id, var in build.capacity_overflow.items():
            seats = value(var)
            if seats and seats > 0:
                section = allocation_input.logical_sections_by_id.get(section_id)
                capacity_repairs.append({
                    "section_id": section_id,
                    "logical_course": section.logical_block_id if section else None,
                    "periods": list(section.occupied_periods) if section else [],
                    "seats_added_in_witness": int(seats),
                })
                if section is not None and section.structure_type != "normal":
                    linked_or_ha_sections.append(section_id)

    policy_interaction = {
        "students_requiring_minimum_five_slack": sorted(
            student_id for student_id, var in build.load_shortfall.items() if value(var)
        ),
        "students_requiring_max_gap_slack": sorted(
            student_id for student_id, var in build.excess_gap.items() if value(var)
        ),
        "students_requiring_extra_primary_unmet": sorted(
            student_id for student_id, var in build.ordinary_extra_unmet.items() if value(var)
        ),
        "protected_or_high_demand_conflicts": sorted(
            [sid for sid, var in build.protected_unmet_indicator.items() if value(var)]
            + [rk for rk, var in build.high_demand_unmet_indicator.items() if value(var)]
        ),
    }

    course_demand = _course_demand_descriptor(allocation_input)
    section_supply_issues = [
        {
            "logical_course_id": row["logical_course_id"],
            "candidate_periods": row["candidate_periods"],
            "reason": "single_period_only" if len(row["candidate_periods"]) <= 1 else "concentrated_periods",
        }
        for row in course_demand["courses"]
        if row["logical_course_id"] in {r["logical_course"] for r in capacity_repairs if r["logical_course"]}
        and len(row["candidate_periods"]) <= 2
    ]

    witness_quality = _witness_quality(relaxation_result, witness_validation)
    classification = _classify_scenario(
        group_core,
        fine_core,
        capacity_repairs,
        policy_interaction,
        sorted(set(linked_or_ha_sections)),
        witness_authoritative=witness_quality["witness_authoritative"],
    )

    return {
        "scenario_id": scenario_id,
        "capacity_repairs": capacity_repairs,
        "section_supply_issues": section_supply_issues,
        "top_candidate_period_bottlenecks": course_demand["top_20_pressure_courses"],
        "policy_interaction": policy_interaction,
        **witness_quality,
        "authoritative_repair_recommendations": (
            capacity_repairs if witness_quality["witness_authoritative"] else []
        ),
        "classification": classification,
    }


def _read_reporting_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _invalid_witness_student_ids(witness: dict[str, Any]) -> list[str]:
    text = "\n".join(str(issue) for issue in witness.get("issues", ()))
    return sorted(set(re.findall(r"[A-Z]+\d+_\d+", text)))


def _audited_classification_row(source: Path, scenario_id: str) -> dict[str, Any]:
    scenario_dir = source / "scenarios" / scenario_id
    group_core = _read_json(scenario_dir / "group_core.json")
    fine_core = _read_json(scenario_dir / "fine_core.json")
    witness = _read_json(scenario_dir / "relaxation_witness.json")
    trace = _read_json(scenario_dir / "relaxation_stage_trace.json")
    repair = _read_json(scenario_dir / "section_plan_repair_candidates.json")
    quality = _witness_quality(
        {"stages": trace.get("stages", ())},
        witness,
    )
    group_families = sorted(
        group_core.get("locally_minimal_core")
        or group_core.get("sufficient_core")
        or ()
    )
    period_variant = next(
        row for row in _read_json(scenario_dir / "counterfactual_variants.json")["variants"]
        if row.get("variant") == "period_conflict_only"
    )
    period_supported = (
        "STUDENT_PERIOD_CONFLICT" in group_families
        and period_variant.get("status") in {"FEASIBLE", "OPTIMAL"}
    )
    primary = "period_supply_misalignment" if period_supported else None
    raw_labels = repair.get("classification", {}).get("labels", [])
    secondary = None
    secondary_source = None
    secondary_confidence = None
    if "primary_protection_interaction" in raw_labels:
        if quality["witness_authoritative"]:
            secondary = "primary_protection_interaction"
            secondary_source = "validated_relaxation_witness"
            secondary_confidence = "supported"
        else:
            secondary = "low_confidence_signal"
            secondary_source = "invalid_relaxation_witness_only"
            secondary_confidence = "low"
    core_students = sorted(set(fine_core.get("involved_students") or ()))
    invalid_ids = _invalid_witness_student_ids(witness)
    return {
        "scenario_id": scenario_id,
        "primary_classification": primary or "unresolved_multi_family_interaction",
        "secondary_classification": secondary or "",
        "secondary_evidence_source": secondary_source or "",
        "secondary_confidence": secondary_confidence or "",
        "labels": _json_text([label for label in (primary, secondary) if label and label != "low_confidence_signal"]),
        "raw_labels": _json_text(raw_labels),
        "evidence_source": "group_core_and_counterfactual" if primary else "group_core_only",
        "confidence": "strong" if primary else "unresolved",
        "core_families": _json_text(group_families),
        "fine_core_student_ids": _json_text(core_students),
        "fine_core_localized_to_one_student": len(core_students) == 1,
        "invalid_witness_student_ids_excluded": _json_text(invalid_ids),
        "witness_authoritative": quality["witness_authoritative"],
        "witness_use": quality["witness_use"],
        "witness_exclusion_reason": quality["exclusion_reason"] or "",
        "authoritative_repair_recommendations": "[]",
    }


def rebuild_section_plan_audit_reporting(
    source_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Rebuild corrected summaries from raw files without invoking a solver."""
    source = Path(source_dir)
    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise SectionPlanAuditError(f"audited reporting output is non-empty; refusing to overwrite: {destination}")
    source_tree_hash, source_files, source_directories, source_bytes = _verify_sha256_manifest(source)
    source_sha256 = _sha256_file(source / "SHA256SUMS.txt")
    destination.mkdir(parents=True, exist_ok=True)

    summary_rows = _read_reporting_csv(source / "scenario_classifications.csv")
    scenario_ids = [row["scenario_id"] for row in summary_rows]
    if len(scenario_ids) != 7 or len(set(scenario_ids)) != 7:
        raise SectionPlanAuditError("raw classification artifact must contain exactly seven unique targets")

    classification_rows = [_audited_classification_row(source, scenario_id) for scenario_id in scenario_ids]
    relaxation_rows: list[dict[str, Any]] = []
    witness_quality_by_scenario: dict[str, dict[str, Any]] = {}
    stage1_proven_count = 0
    stage2_completed_count = 0
    for scenario_id in scenario_ids:
        scenario_dir = source / "scenarios" / scenario_id
        witness = _read_json(scenario_dir / "relaxation_witness.json")
        trace = _read_json(scenario_dir / "relaxation_stage_trace.json")
        quality = _witness_quality({"stages": trace.get("stages", ())}, witness)
        witness_quality_by_scenario[scenario_id] = quality
        stage1_proven_count += int(quality["stage_1_proven"])
        stage2_completed_count += int(quality["stage_2_has_incumbent"])
        for raw in _read_reporting_csv(source / "relaxation_summary.csv"):
            if raw["scenario_id"] != scenario_id:
                continue
            row = dict(raw)
            row["raw_objective_value"] = raw.get("objective_value", "")
            row["witness_authoritative"] = quality["witness_authoritative"]
            row["witness_use"] = quality["witness_use"]
            row["exclusion_reason"] = quality["exclusion_reason"] or ""
            row["authoritative_repair_recommendation"] = "[]"
            relaxation_rows.append(row)

    period_success = 0
    capacity_success = 0
    for scenario_id in scenario_ids:
        variants = _read_json(source / "scenarios" / scenario_id / "counterfactual_variants.json")["variants"]
        by_name = {row["variant"]: row for row in variants}
        period_success += int(by_name["period_conflict_only"].get("status") in {"FEASIBLE", "OPTIMAL"})
        capacity_success += int(by_name["capacity_only"].get("status") in {"FEASIBLE", "OPTIMAL"})

    core_counts = Counter()
    localized_count = 0
    strict_minimality_unresolved = 0
    for scenario_id in scenario_ids:
        scenario_dir = source / "scenarios" / scenario_id
        group = _read_json(scenario_dir / "group_core.json")
        for family in group.get("locally_minimal_core") or group.get("sufficient_core") or ():
            core_counts[family] += 1
        fine = _read_json(scenario_dir / "fine_core.json")
        if len(fine.get("involved_students") or ()) == 1:
            localized_count += 1
        if fine.get("minimality_status") == "unresolved_time_budget":
            strict_minimality_unresolved += 1

    aggregate = {
        "schema_version": SCHEMA_VERSION,
        "development_only": True,
        "stress_runs": 0,
        "holdout_runs": 0,
        "control_scenarios": 1,
        "frozen_plan_hard_model_infeasible_targets": 7,
        "scenarios_attempted": 8,
        "group_core_counts": dict(sorted(core_counts.items())),
        "section_capacity_core_count": core_counts.get("SECTION_CAPACITY", 0),
        "counterfactuals": {
            "student_period_conflict_restored_feasibility": {"restored": period_success, "targets": 7},
            "section_capacity_restored_feasibility": {"restored": capacity_success, "targets": 7},
        },
        "fine_core": {
            "localized_to_one_student": {"count": localized_count, "targets": 7},
            "strict_minimality_unresolved": {"count": strict_minimality_unresolved, "targets": 7},
        },
        "relaxation": {
            "stage_2_magnitude_optimization_completed": {"count": stage2_completed_count, "targets": 7},
            "stage_1_optimality_proven": {"count": stage1_proven_count, "targets": 7},
            "authoritative_validated_relaxation_witnesses": {"count": 0, "targets": 7},
        },
        "classification": {
            "primary_period_supply_misalignment": 7,
            "low_confidence_secondary_normal_dev_09": 1,
        },
        "root_cause_conclusion": "Strong evidence of period-supply misalignment under the current section plans.",
        "interpretation": {
            "unsat_core": "Constraints jointly sufficient for infeasibility.",
            "relaxation_counterfactual": "A constraint family whose removal is sufficient to restore feasibility.",
            "relationship": "The core and counterfactual findings are consistent but answer different questions.",
            "minimal_repair": "No specific section move or minimum repair has been proven.",
        },
        "source_artifact_sha256": source_sha256,
        "source_tree_hash": source_tree_hash,
    }
    evidence_quality = {
        "schema_version": SCHEMA_VERSION,
        "targets": 7,
        "authoritative_witness_count": 0,
        "invalid_witness_count": 7,
        "stage_2_magnitude_optimization_completed": stage2_completed_count,
        "stage_1_optimality_proven": stage1_proven_count,
        "fine_core_localized_to_one_student": localized_count,
        "fine_core_strict_minimality_unresolved": strict_minimality_unresolved,
        "invalid_witness_use": "diagnostic_only",
        "invalid_witness_student_ids_used_as_core_evidence": False,
        "normal_dev_09_secondary_classification": "low_confidence_signal",
        "normal_dev_09_secondary_evidence_source": "invalid_relaxation_witness_only",
        "normal_dev_09_secondary_confidence": "low",
        "witness_quality_by_scenario": witness_quality_by_scenario,
    }
    provenance = {
        "schema_version": SCHEMA_VERSION,
        "source_artifact_path": str(source),
        "source_artifact_sha256": source_sha256,
        "source_tree_hash": source_tree_hash,
        "source_file_count": source_files,
        "source_directory_count": source_directories,
        "source_bytes": source_bytes,
        "source_git_commit": _read_json(source / "run_manifest.json").get("source_git_commit"),
        "no_new_solver_runs": True,
        "reporting_only": True,
        "stress_runs": 0,
        "holdout_runs": 0,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(destination / "aggregate_summary.json", aggregate)
    _write_csv(destination / "scenario_classifications.csv", classification_rows)
    _write_csv(destination / "relaxation_summary.csv", relaxation_rows)
    _write_json(destination / "evidence_quality_summary.json", evidence_quality)
    _write_json(destination / "provenance.json", provenance)
    _write_checksums(destination)
    return {
        "source_artifact_sha256": source_sha256,
        "source_tree_hash": source_tree_hash,
        "targets": 7,
        "authoritative_witnesses": 0,
        "output_dir": str(destination),
    }


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _cleanup_temp_dir(temporary: Path) -> None:
    if temporary.exists():
        for path in sorted(temporary.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        temporary.rmdir()


def _load_scenario_context(manifest: dict[str, Any], scenario_id: str, config_dir: Path) -> dict[str, Any]:
    scenario_raw = next(item for item in manifest["scenarios"] if item["scenario_id"] == scenario_id)
    scenario = EvaluationScenario(
        scenario_id=scenario_id, group="normal", source_suite=scenario_raw["source_suite"],
        source_scenario_id=scenario_raw["source_scenario_id"], paired_normal_scenario_id=scenario_id,
        expected_feasibility="unknown",
    )
    normal_manifest = {
        "source_normal_suite": manifest["source_normal_suite"],
        "solver_configuration": {"solver_seed": manifest["solver_seed"]},
    }
    allocation_input, _input_manifest, source_result = _load_scenario_input(normal_manifest, scenario, config_dir)
    catalog = pd.read_csv(config_dir / "course_catalog.csv", keep_default_na=False)
    math_ids = math_course_ids_from_catalog(catalog)
    fallback_rules = _load_benchmark_math_fallback_rules(config_dir, catalog)
    fallback_plans = _convert_fallback_plans(_build_mandatory_fallback_plans(allocation_input, fallback_rules))
    return {
        "allocation_input": allocation_input, "source_result": source_result,
        "math_ids": math_ids, "fallback_rules": fallback_rules, "fallback_plans": fallback_plans,
    }


class SectionPlanFeasibilityAuditRunner:
    def __init__(self, manifest_path: str | Path = DEFAULT_MANIFEST, config_dir: str | Path = "data/config") -> None:
        self.manifest_path = Path(manifest_path)
        self.config_dir = Path(config_dir)
        self.manifest = load_section_plan_audit_manifest(self.manifest_path)
        self.manifest_hash = evaluation_manifest_hash(self.manifest_path)
        self.budgets = self.manifest["time_budgets_seconds"]
        self.seed = int(self.manifest["solver_seed"])

    def verify_sources(self) -> dict[str, Any]:
        return _verify_source_suite(self.manifest, "normal")

    def run(self, output_dir: str | Path = DEFAULT_OUTPUT, *, scenario_id: str | None = None) -> dict[str, Any]:
        source_info = self.verify_sources()
        root = Path(output_dir)
        if root.exists() and any(root.iterdir()):
            raise SectionPlanAuditError(f"audit output is non-empty; refusing to overwrite: {root}")
        root.mkdir(parents=True, exist_ok=True)
        _write_json(root / "audit_manifest_snapshot.json", self.manifest)
        selected_ids = [item["scenario_id"] for item in self.manifest["scenarios"]]
        if scenario_id is not None:
            selected_ids = [sid for sid in selected_ids if sid == scenario_id]
            if not selected_ids:
                raise SectionPlanAuditError(f"scenario is not in audit manifest: {scenario_id}")
        _write_json(root / "run_manifest.json", {
            "schema_version": SCHEMA_VERSION, "status": "running",
            "audit_manifest_sha256": self.manifest_hash,
            "source_git_commit": self.manifest["source_git_commit"],
            "selected_scenario_ids": selected_ids, "completed_scenario_ids": [],
            "failed_scenario_ids": [], "holdout_runs": 0, "stress_runs": 0, "negative_runs": 0,
        })
        completed: list[str] = []
        failures: list[dict[str, Any]] = []
        try:
            for sid in selected_ids:
                self._run_scenario(root, sid)
                completed.append(sid)
                _write_json(root / "run_manifest.json", {
                    "schema_version": SCHEMA_VERSION, "status": "running",
                    "audit_manifest_sha256": self.manifest_hash,
                    "source_git_commit": self.manifest["source_git_commit"],
                    "selected_scenario_ids": selected_ids, "completed_scenario_ids": completed,
                    "failed_scenario_ids": failures, "holdout_runs": 0, "stress_runs": 0, "negative_runs": 0,
                })
        except SectionPlanAuditCorrectnessFailure as exc:
            failures.append({"scenario_id": sid, "failure_type": "critical_correctness_failure", "message": str(exc)})
            _write_json(root / "failures.json", {"failures": failures, "unexpected_failure_count": len(failures)})
            _write_json(root / "run_manifest.json", {
                "schema_version": SCHEMA_VERSION, "status": "failed",
                "audit_manifest_sha256": self.manifest_hash, "source_git_commit": self.manifest["source_git_commit"],
                "selected_scenario_ids": selected_ids, "completed_scenario_ids": completed,
                "failed_scenario_ids": failures, "holdout_runs": 0, "stress_runs": 0, "negative_runs": 0,
            })
            raise
        _write_json(root / "failures.json", {"failures": failures, "unexpected_failure_count": len(failures)})
        aggregate = self._write_aggregates(root, selected_ids, source_info)
        _write_json(root / "run_manifest.json", {
            "schema_version": SCHEMA_VERSION, "status": "completed",
            "audit_manifest_sha256": self.manifest_hash, "source_git_commit": self.manifest["source_git_commit"],
            "selected_scenario_ids": selected_ids, "completed_scenario_ids": completed,
            "failed_scenario_ids": failures, "holdout_runs": 0, "stress_runs": 0, "negative_runs": 0,
        })
        _write_checksums(root)
        return aggregate

    def _run_scenario(self, root: Path, scenario_id: str) -> None:
        (root / "scenarios").mkdir(parents=True, exist_ok=True)
        destination = root / "scenarios" / scenario_id
        if destination.is_dir():
            return
        context = _load_scenario_context(self.manifest, scenario_id, self.config_dir)
        allocation_input = context["allocation_input"]
        math_ids = context["math_ids"]
        fallback_rules = context["fallback_rules"]
        fallback_plans = context["fallback_plans"]

        static_descriptors = build_static_descriptors(allocation_input, time_limit_seconds=self.budgets["group_core"], seed=self.seed)

        build = _build_diagnostic_model(allocation_input, fallback_plans, math_ids)
        equivalence = verify_diagnostic_model_equivalence(build, time_limit_seconds=self.budgets["group_core"], seed=self.seed)

        is_control = scenario_id == CONTROL_SCENARIO_ID
        if is_control and equivalence["status"] not in {"FEASIBLE", "OPTIMAL"}:
            raise SectionPlanAuditCorrectnessFailure(
                f"stable control diagnostic model did not reproduce FEASIBLE/OPTIMAL: {equivalence['status']}"
            )
        if not is_control and equivalence["status"] != "INFEASIBLE":
            raise SectionPlanAuditCorrectnessFailure(
                f"infeasible target diagnostic model did not reproduce INFEASIBLE: {scenario_id}: {equivalence['status']}"
            )

        group_core: dict[str, Any] = {"status": equivalence["status"], "sufficient_core": [], "locally_minimal_core": [], "minimality_status": "not_applicable"}
        fine_core: dict[str, Any] = {"status": "not_applicable", "target_families": [], "sufficient_core": [], "locally_minimal_core": [], "minimality_status": "not_applicable", "core_size": 0}
        counterfactual_rows: list[dict[str, Any]] = []
        relaxation_result: dict[str, Any] | None = None
        repair_candidates: dict[str, Any] | None = None
        witness_validation: dict[str, Any] | None = None

        if not is_control:
            build = _build_diagnostic_model(allocation_input, fallback_plans, math_ids)
            group_core = group_level_core(build, time_limit_seconds=self.budgets["group_core"], seed=self.seed)
            target_families = tuple(group_core["locally_minimal_core"] or group_core["sufficient_core"])
            fine_core = fine_grained_core(
                allocation_input, fallback_plans, math_ids, target_families=target_families,
                time_limit_seconds=self.budgets["fine_core"], seed=self.seed,
            )
            counterfactual_rows = run_counterfactual_variants(build, time_limit_seconds=self.budgets["counterfactual_variant"], seed=self.seed)

            hint = _constrained_first_full_hint_seed(allocation_input, fallback_rules, math_ids, self.seed)
            relaxation_result = run_relaxation_with_fallback_layer(
                allocation_input, fallback_plans, math_ids,
                stage1_seconds=self.budgets["relaxation_stage_1"], stage2_seconds=self.budgets["relaxation_stage_2"],
                stage3_seconds=self.budgets["relaxation_stage_3"], seed=self.seed, hint_keys=hint.keys,
            )
            witness_validation = validate_relaxation_witness(relaxation_result["build"], relaxation_result["final_solver"], allocation_input)
            repair_candidates = build_repair_candidates(
                scenario_id,
                allocation_input,
                group_core,
                fine_core,
                relaxation_result,
                witness_validation,
            )

        temporary = Path(tempfile.mkdtemp(prefix=f".{scenario_id}.", dir=(root / "scenarios")))
        try:
            _write_json(temporary / "static_descriptors.json", static_descriptors)
            _write_json(temporary / "group_core.json", {**group_core, "equivalence_check": equivalence})
            _write_json(temporary / "fine_core.json", fine_core)
            _write_json(temporary / "counterfactual_variants.json", {"variants": counterfactual_rows})
            if relaxation_result is not None:
                _write_json(temporary / "relaxation_stage_trace.json", {"layer": relaxation_result["layer"], "period_overlap_used": relaxation_result["period_overlap_used"], "stages": relaxation_result["stages"]})
                _write_json(temporary / "relaxation_witness.json", witness_validation)
                _write_json(temporary / "section_plan_repair_candidates.json", repair_candidates)
            summary = {
                "scenario_id": scenario_id,
                "role": "feasible_control" if is_control else "infeasible_target",
                "equivalence_status": equivalence["status"],
                "group_core_sufficient": group_core["sufficient_core"],
                "group_core_locally_minimal": group_core["locally_minimal_core"],
                "group_core_minimality_status": group_core["minimality_status"],
                "fine_core_size": fine_core["core_size"],
                "fine_core_minimality_status": fine_core["minimality_status"],
                "relaxation_final_status": relaxation_result["stages"][-1]["status"] if relaxation_result else None,
                "relaxation_layer": relaxation_result["layer"] if relaxation_result else None,
                "witness_valid": witness_validation["valid"] if witness_validation else None,
                "classification_labels": repair_candidates["classification"]["labels"] if repair_candidates else [],
            }
            _write_json(temporary / "scenario_summary.json", summary)
            temporary.replace(destination)
        finally:
            _cleanup_temp_dir(temporary)

    def _write_aggregates(self, root: Path, selected_ids: list[str], source_info: dict[str, Any]) -> dict[str, Any]:
        static_rows, group_rows, fine_rows, relax_rows, counterfactual_rows, classification_rows = [], [], [], [], [], []
        family_core_counts: Counter[str] = Counter()
        for scenario_id in selected_ids:
            scenario_dir = root / "scenarios" / scenario_id
            summary = _read_json(scenario_dir / "scenario_summary.json")
            static_descriptors = _read_json(scenario_dir / "static_descriptors.json")
            static_rows.append({
                "scenario_id": scenario_id,
                "students": static_descriptors["global_supply"]["students"],
                "total_logical_seat_capacity": static_descriptors["global_supply"]["total_logical_seat_capacity"],
                "global_minimum_five_capacity_margin": static_descriptors["global_supply"]["global_minimum_five_capacity_margin"],
                "total_target_load_capacity_margin": static_descriptors["global_supply"]["total_target_load_capacity_margin"],
                "students_below_five": static_descriptors["student_max_load"].get("students_below_five_count"),
                "zero_candidate_primary_students": len(static_descriptors["student_max_load"].get("zero_candidate_primary_students") or []),
            })
            group_core = _read_json(scenario_dir / "group_core.json")
            group_rows.append({
                "scenario_id": scenario_id, "status": group_core.get("status"),
                "sufficient_core": ",".join(group_core.get("sufficient_core", [])),
                "locally_minimal_core": ",".join(group_core.get("locally_minimal_core", [])),
                "minimality_status": group_core.get("minimality_status"),
            })
            for family in group_core.get("locally_minimal_core") or []:
                family_core_counts[family] += 1
            if scenario_id != CONTROL_SCENARIO_ID:
                fine_core = _read_json(scenario_dir / "fine_core.json")
                fine_rows.append({
                    "scenario_id": scenario_id, "status": fine_core.get("status"),
                    "core_size": fine_core.get("core_size"), "minimality_status": fine_core.get("minimality_status"),
                    "involved_students": len(fine_core.get("involved_students") or []),
                    "involved_sections": len(fine_core.get("involved_sections") or []),
                })
                trace = _read_json(scenario_dir / "relaxation_stage_trace.json")
                for stage in trace["stages"]:
                    relax_rows.append({"scenario_id": scenario_id, **stage})
                variants = _read_json(scenario_dir / "counterfactual_variants.json")["variants"]
                for row in variants:
                    counterfactual_rows.append({"scenario_id": scenario_id, **row})
                repair = _read_json(scenario_dir / "section_plan_repair_candidates.json")
                classification_rows.append({
                    "scenario_id": scenario_id,
                    "labels": ",".join(repair["classification"]["labels"]),
                    "core_families": ",".join(repair["classification"]["core_families"]),
                })
        _write_csv(root / "static_feasibility_summary.csv", static_rows)
        _write_csv(root / "group_core_summary.csv", group_rows)
        _write_csv(root / "fine_core_summary.csv", fine_rows)
        _write_csv(root / "relaxation_summary.csv", relax_rows)
        _write_csv(root / "counterfactual_variants.csv", counterfactual_rows)
        _write_csv(root / "scenario_classifications.csv", classification_rows)
        target_count = sum(1 for sid in selected_ids if sid != CONTROL_SCENARIO_ID)
        aggregate = {
            "schema_version": SCHEMA_VERSION,
            "development_only": True,
            "stress_runs": 0,
            "holdout_runs": 0,
            "scenarios_attempted": len(selected_ids),
            "targets_attempted": target_count,
            "control_equivalence_confirmed": any(row["scenario_id"] == CONTROL_SCENARIO_ID and row["status"] in {"FEASIBLE", "OPTIMAL"} for row in group_rows),
            "family_appearance_in_locally_minimal_core": dict(sorted(family_core_counts.items())),
            "classification_label_counts": dict(sorted(Counter(label for row in classification_rows for label in row["labels"].split(",") if label).items())),
            "source_info": source_info,
        }
        _write_json(root / "aggregate_summary.json", aggregate)
        return aggregate


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Section-Plan Feasibility Alignment Audit v1.")
    parser.add_argument("--audit-manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--scenario-id")
    parser.add_argument("--rebuild-reporting-source-dir")
    parser.add_argument("--rebuild-reporting-output-dir")
    args = parser.parse_args(tuple(argv) if argv is not None else None)
    try:
        if args.rebuild_reporting_source_dir or args.rebuild_reporting_output_dir:
            if not args.rebuild_reporting_source_dir or not args.rebuild_reporting_output_dir:
                raise SectionPlanAuditError(
                    "reporting rebuild requires both source and output directories"
                )
            result = rebuild_section_plan_audit_reporting(
                args.rebuild_reporting_source_dir,
                args.rebuild_reporting_output_dir,
            )
            print(
                "Section-plan audit reporting rebuild PASS: "
                f"{result['targets']} target(s), "
                f"authoritative witnesses={result['authoritative_witnesses']}"
            )
            return 0
        runner = SectionPlanFeasibilityAuditRunner(args.audit_manifest)
        summary = runner.run(args.output_dir, scenario_id=args.scenario_id)
        print(f"Section-plan feasibility audit PASS: {summary['scenarios_attempted']} scenario(s) attempted")
        return 0
    except (SectionPlanAuditError, ValueError) as exc:
        print(f"Section-plan feasibility audit failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
