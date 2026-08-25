"""Development-only period-placement repair probe.

This module explores edits to copies of frozen section plans.  It deliberately
does not change the production section planner, requests, capacities, or CP-SAT
model.  The first executable slice is a candidate-universe and exact
student-level preview; full-model candidate validation is available as an
explicit function but is never run by the preview command.

The repository's canonical ``candidate_index`` is keyed by logical course and
section identity, not by period.  A placement edit therefore preserves that
index while changing a derived request/section period view.  The artifact
records both facts instead of pretending that a period move changes the
canonical edge count.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.allocation import (
    canonicalize_allocation_input,
    math_course_ids_from_catalog,
    run_constrained_first_baseline,
    run_fair_cp_sat_solver,
)
from src.allocation.input_models import CanonicalAllocationInput, LogicalRequest, LogicalSection
from src.benchmark_runner import _load_math_fallback_rules
from src.section_plan_feasibility_audit import (
    _load_scenario_context,
    load_section_plan_audit_manifest,
)


PERIODS = tuple(f"P{i}" for i in range(1, 8))
DOUBLE_PERIOD_PLACEMENTS = tuple((f"P{i}", f"P{i + 1}") for i in range(1, 7))
CONTROL_SCENARIO_ID = "normal_dev_reference_2026"
TARGET_SCENARIO_IDS = (
    "normal_dev_01",
    "normal_dev_03",
    "normal_dev_04",
    "normal_dev_05",
    "normal_dev_07",
    "normal_dev_09",
    "normal_dev_10",
)
SCENARIO_ORDER = (CONTROL_SCENARIO_ID, *TARGET_SCENARIO_IDS)
DEFAULT_MANIFEST = Path("data/scenarios/period_placement_repair_probe_v1.json")
DEFAULT_AUDIT_ROOT = Path(
    "../fair-course-allocation-artifacts/robustness-v1/"
    "section-plan-feasibility-audit-v1"
)
DEFAULT_OUTPUT = Path(
    "../fair-course-allocation-artifacts/robustness-v1/"
    "period-placement-repair-probe-v1"
)
SCHEMA_VERSION = 1


class PeriodPlacementProbeError(ValueError):
    """Raised when the frozen probe cannot proceed without ambiguity."""


@dataclass(frozen=True)
class AuthoritativeCoreInput:
    scenario_id: str
    student_id: str
    core_periods: tuple[str, ...]
    core_literals: tuple[dict[str, Any], ...]
    source_file: str
    source_hash: str
    evidence_type: str
    minimality_status: str


@dataclass(frozen=True)
class CandidateEdit:
    candidate_id: str
    edit_type: str
    logical_section_ids: tuple[str, ...]
    logical_course_ids: tuple[str, ...]
    original_placements: tuple[tuple[str, ...], ...]
    proposed_placements: tuple[tuple[str, ...], ...]
    valid_period_source: str
    occupancy_shape: tuple[int, ...]
    core_student: str
    core_period_relevance: tuple[str, ...]
    affected_candidate_edge_count: int
    affected_student_count: int
    section_capacity_unchanged: bool = True


@dataclass(frozen=True)
class ScenarioContext:
    scenario_id: str
    allocation_input: CanonicalAllocationInput
    students: pd.DataFrame
    requests: pd.DataFrame
    sections: pd.DataFrame
    catalog: pd.DataFrame
    authoritative_core: AuthoritativeCoreInput | None


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _jsonable(value: Any) -> Any:
    if hasattr(value, "__dataclass_fields__"):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = tuple(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_checksums(root: Path) -> str:
    checksum_path = root / "SHA256SUMS.txt"
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != checksum_path:
            entries.append(f"{_sha256_file(path)}  {path.relative_to(root)}")
    checksum_path.write_text("\n".join(entries) + "\n", encoding="utf-8")
    return _sha256_file(checksum_path)


def _manifest_hash(path: Path) -> str:
    return _sha256_file(path)


def load_period_placement_probe_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    path = Path(path)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PeriodPlacementProbeError(f"cannot read probe manifest: {path}: {exc}") from exc
    required = {
        "probe_name", "probe_version", "source_git_commit",
        "source_section_audit_manifest_hash", "source_raw_artifact_hash",
        "source_audited_artifact_hash", "solver_seed", "workers", "scenarios",
        "allowed_edit_types", "candidate_universe_definition", "tuning_allowed",
        "stress_execution_allowed", "holdout_execution_allowed",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise PeriodPlacementProbeError("probe manifest missing: " + ", ".join(missing))
    if int(payload["solver_seed"]) != 20260630 or int(payload["workers"]) != 1:
        raise PeriodPlacementProbeError("probe solver seed/workers are not frozen")
    if payload["tuning_allowed"] is not False:
        raise PeriodPlacementProbeError("period-placement probe must not tune candidates")
    if payload["stress_execution_allowed"] is not False or payload["holdout_execution_allowed"] is not False:
        raise PeriodPlacementProbeError("period-placement probe must forbid stress and holdout execution")
    if tuple(payload["scenarios"]) != SCENARIO_ORDER:
        raise PeriodPlacementProbeError("probe scenarios must contain exactly the frozen control and seven targets")
    if set(payload["allowed_edit_types"]) != {"single_section_move", "logical_section_swap"}:
        raise PeriodPlacementProbeError("probe edit types are not frozen")
    return payload


def _load_authoritative_core(
    audit_root: Path,
    scenario_id: str,
) -> AuthoritativeCoreInput | None:
    if scenario_id == CONTROL_SCENARIO_ID:
        return None
    path = audit_root / "scenarios" / scenario_id / "fine_core.json"
    if not path.is_file():
        raise PeriodPlacementProbeError(f"missing authoritative fine core: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    students = tuple(sorted(set(payload.get("involved_students") or [])))
    if len(students) != 1:
        raise PeriodPlacementProbeError(
            f"fine core must identify exactly one authoritative student: {scenario_id}"
        )
    literals = tuple(payload.get("sufficient_core") or [])
    if not literals:
        raise PeriodPlacementProbeError(f"fine core has no sufficient core literals: {scenario_id}")
    periods = tuple(sorted(set(payload.get("involved_periods") or [])))
    if any(period not in PERIODS for period in periods):
        raise PeriodPlacementProbeError(f"fine core has an invalid period: {scenario_id}")
    if scenario_id == "normal_dev_10" and students != ("G12_0536",):
        raise PeriodPlacementProbeError("normal_dev_10 must use G12_0536 as authoritative core student")
    return AuthoritativeCoreInput(
        scenario_id=scenario_id,
        student_id=students[0],
        core_periods=periods,
        core_literals=tuple(dict(item) for item in literals),
        source_file=str(path),
        source_hash=_sha256_file(path),
        evidence_type="audited_fine_sufficient_core",
        minimality_status=str(payload.get("minimality_status", "unknown")),
    )


def load_scenario_context(
    scenario_id: str,
    *,
    audit_manifest: dict[str, Any],
    audit_root: str | Path = DEFAULT_AUDIT_ROOT,
    config_dir: str | Path = "data/config",
) -> ScenarioContext:
    audit_root = Path(audit_root)
    config_dir = Path(config_dir)
    if scenario_id not in SCENARIO_ORDER:
        raise PeriodPlacementProbeError(f"scenario is not in the frozen probe: {scenario_id}")
    context = _load_scenario_context(audit_manifest, scenario_id, config_dir)
    source_root = Path(audit_manifest["source_normal_suite"]["artifact_dir"])
    scenario_root = source_root / "scenarios" / scenario_id
    generated = scenario_root / "generated"
    sections_root = scenario_root / "sections"
    students = pd.read_csv(generated / "students.csv", keep_default_na=False)
    requests = pd.read_csv(generated / "requests.csv", keep_default_na=False)
    sections = pd.read_csv(sections_root / "sections.csv", keep_default_na=False)
    catalog = pd.read_csv(config_dir / "course_catalog.csv", keep_default_na=False)
    return ScenarioContext(
        scenario_id=scenario_id,
        allocation_input=context["allocation_input"],
        students=students,
        requests=requests,
        sections=sections,
        catalog=catalog,
        authoritative_core=_load_authoritative_core(audit_root, scenario_id),
    )


def _section_placement(section: LogicalSection) -> tuple[str, ...]:
    return tuple(section.occupied_periods)


def legal_placements(section: LogicalSection) -> tuple[tuple[str, ...], ...]:
    """Return only placements allowed by the existing V1 period model."""
    if section.structure_type == "double_period":
        return DOUBLE_PERIOD_PLACEMENTS
    return tuple((period,) for period in PERIODS)


def _placement_shape(placement: tuple[str, ...]) -> tuple[int, ...]:
    values = tuple(int(period[1:]) for period in placement)
    if len(values) == 2:
        return (len(values), values[1] - values[0])
    return (len(values),)


def _placement_is_legal_for_original(
    original: tuple[str, ...],
    proposed: tuple[str, ...],
) -> bool:
    if any(period not in PERIODS for period in proposed):
        return False
    if len(original) != len(proposed):
        return False
    if len(proposed) == 2:
        return int(proposed[1][1:]) - int(proposed[0][1:]) == 1
    return len(proposed) == 1


def _compatible_for_swap(first: LogicalSection, second: LogicalSection) -> bool:
    return _placement_shape(_section_placement(first)) == _placement_shape(_section_placement(second))


def _requests_for_sections(
    allocation_input: CanonicalAllocationInput,
    section_ids: set[str],
) -> list[LogicalRequest]:
    return [
        request
        for request in allocation_input.logical_requests
        if request.request_type == "primary"
        and any(section_id in section_ids for section_id in allocation_input.candidate_index.get(request.request_key, ()))
    ]


def _affected_stats(
    allocation_input: CanonicalAllocationInput,
    section_ids: set[str],
) -> tuple[int, int]:
    requests = _requests_for_sections(allocation_input, section_ids)
    students = {request.student_id for request in requests}
    return len(requests), len(students)


def _candidate(
    *,
    scenario_id: str,
    edit_type: str,
    section_ids: tuple[str, ...],
    original: tuple[tuple[str, ...], ...],
    proposed: tuple[tuple[str, ...], ...],
    course_ids: tuple[str, ...],
    core: AuthoritativeCoreInput,
    allocation_input: CanonicalAllocationInput,
) -> CandidateEdit:
    affected_edges, affected_students = _affected_stats(allocation_input, set(section_ids))
    relevance = sorted({
        period
        for placement in (*original, *proposed)
        for period in placement
        if period in core.core_periods
    })
    token = "__".join(
        f"{section_id}:{','.join(placement)}"
        for section_id, placement in zip(section_ids, proposed)
    )
    return CandidateEdit(
        candidate_id=f"{edit_type}:{scenario_id}:{token}",
        edit_type=edit_type,
        logical_section_ids=section_ids,
        logical_course_ids=course_ids,
        original_placements=original,
        proposed_placements=proposed,
        valid_period_source=("P1-P7" if len(original[0]) == 1 else "P1-P7 consecutive double-period pairs"),
        occupancy_shape=tuple(_placement_shape(item) for item in original),
        core_student=core.student_id,
        core_period_relevance=tuple(relevance),
        affected_candidate_edge_count=affected_edges,
        affected_student_count=affected_students,
    )


def generate_candidate_universe(
    context: ScenarioContext,
) -> tuple[CandidateEdit, ...]:
    """Freeze all admissible moves/swaps touching the authoritative core courses."""
    if context.authoritative_core is None:
        return ()
    core = context.authoritative_core
    student = context.allocation_input.students_by_id[core.student_id]
    relevant_courses = {request.candidate_key for request in student.primary_requests}
    relevant = [
        section for section in context.allocation_input.logical_sections
        if section.logical_block_id in relevant_courses
    ]
    all_sections = list(context.allocation_input.logical_sections)
    candidates: list[CandidateEdit] = []
    for section in relevant:
        original = _section_placement(section)
        for proposed in legal_placements(section):
            if proposed == original:
                continue
            candidates.append(_candidate(
                scenario_id=context.scenario_id,
                edit_type="single_section_move",
                section_ids=(section.linked_section_group_id,),
                original=(original,), proposed=(proposed,),
                course_ids=(section.logical_block_id,), core=core,
                allocation_input=context.allocation_input,
            ))
    for index, first in enumerate(relevant):
        for second in all_sections:
            if first.linked_section_group_id >= second.linked_section_group_id:
                continue
            if not _compatible_for_swap(first, second):
                continue
            first_placement = _section_placement(first)
            second_placement = _section_placement(second)
            if first_placement == second_placement:
                continue
            candidates.append(_candidate(
                scenario_id=context.scenario_id,
                edit_type="logical_section_swap",
                section_ids=(first.linked_section_group_id, second.linked_section_group_id),
                original=(first_placement, second_placement),
                proposed=(second_placement, first_placement),
                course_ids=(first.logical_block_id, second.logical_block_id),
                core=core, allocation_input=context.allocation_input,
            ))
    return tuple(sorted(candidates, key=lambda item: item.candidate_id))


def _period_mask(periods: Iterable[str]) -> int:
    mask = 0
    for period in periods:
        mask |= 1 << (int(period[1:]) - 1)
    return mask


def _best_primary_assignment(
    allocation_input: CanonicalAllocationInput,
    student_id: str,
    *,
    objective: str,
) -> tuple[int, int, tuple[str, ...]]:
    student = allocation_input.students_by_id[student_id]
    requests = tuple(student.primary_requests)
    sections = allocation_input.logical_sections_by_id
    memo: dict[tuple[int, int, tuple[str, ...]], tuple[int, int, tuple[str, ...]]] = {}

    def better(
        left: tuple[int, int, tuple[str, ...]],
        right: tuple[int, int, tuple[str, ...]],
    ) -> tuple[int, int, tuple[str, ...]]:
        if objective == "units":
            left_key = (left[1], left[0], tuple(reversed(left[2])))
            right_key = (right[1], right[0], tuple(reversed(right[2])))
        else:
            left_key = (left[0], left[1], tuple(reversed(left[2])))
            right_key = (right[0], right[1], tuple(reversed(right[2])))
        return left if left_key >= right_key else right

    def visit(index: int, used_periods: int, identities: tuple[str, ...]) -> tuple[int, int, tuple[str, ...]]:
        key = (index, used_periods, identities)
        if key in memo:
            return memo[key]
        if index == len(requests):
            return (0, 0, ())
        request = requests[index]
        best = visit(index + 1, used_periods, identities)
        identity_set = set(identities)
        for section_id in allocation_input.candidate_index.get(request.request_key, ()):
            section = sections.get(section_id)
            if section is None or section.logical_block_id != request.candidate_key:
                continue
            occupied = _period_mask(section.occupied_periods)
            if occupied & used_periods or section.logical_block_id in identity_set:
                continue
            tail = visit(index + 1, used_periods | occupied, tuple(sorted((*identities, section.logical_block_id))))
            candidate = (tail[0] + 1, tail[1] + request.period_units, (request.candidate_key,) + tail[2])
            best = better(best, candidate)
        memo[key] = best
        return best

    return visit(0, 0, ())


def exact_student_level_analysis(
    allocation_input: CanonicalAllocationInput,
    student_id: str,
) -> dict[str, Any]:
    """Compute an exact no-capacity primary assignment bound by DP."""
    student = allocation_input.students_by_id[student_id]
    max_count, _, selected_by_count = _best_primary_assignment(allocation_input, student_id, objective="count")
    _, max_units, selected_by_units = _best_primary_assignment(allocation_input, student_id, objective="units")
    requests = tuple(student.primary_requests)
    conflicts: list[list[str]] = []
    for index, first in enumerate(requests):
        first_sections = [
            allocation_input.logical_sections_by_id[section_id]
            for section_id in allocation_input.candidate_index.get(first.request_key, ())
            if section_id in allocation_input.logical_sections_by_id
        ]
        if not first_sections:
            conflicts.append([first.candidate_key])
        for second in requests[index + 1:]:
            second_sections = [
                allocation_input.logical_sections_by_id[section_id]
                for section_id in allocation_input.candidate_index.get(second.request_key, ())
                if section_id in allocation_input.logical_sections_by_id
            ]
            compatible = any(
                not (_period_mask(a.occupied_periods) & _period_mask(b.occupied_periods))
                and first.candidate_key != second.candidate_key
                for a in first_sections for b in second_sections
            )
            if not compatible:
                conflicts.append([first.candidate_key, second.candidate_key])
    primary_count = len(requests)
    return {
        "student_id": student_id,
        "primary_request_count": primary_count,
        "target_period_units": student.target_period_units,
        "original_max_primary_assignments": max_count,
        "original_primary_unmet": primary_count - max_count,
        "original_max_primary_period_units": max_units,
        "original_max_schedule_gap": max(student.target_period_units - max_units, 0),
        "selected_by_count": list(selected_by_count),
        "selected_by_units": list(selected_by_units),
        "conflicting_primary_course_sets": conflicts,
        "candidate_free_periods": [
            period for period in PERIODS
            if not any(
                period in allocation_input.logical_sections_by_id[section_id].occupied_periods
                for request in requests
                for section_id in allocation_input.candidate_index.get(request.request_key, ())
            )
        ],
    }


def _replace_input_periods(
    allocation_input: CanonicalAllocationInput,
    placements: dict[str, tuple[str, ...]],
) -> CanonicalAllocationInput:
    sections = []
    for section in allocation_input.logical_sections:
        placement = placements.get(section.linked_section_group_id)
        if placement is None:
            sections.append(section)
            continue
        members = tuple(
            replace(member, period_1=placement[0], period_2=placement[1] if len(placement) == 2 else "")
            for member in section.member_sections
        )
        sections.append(replace(section, member_sections=members, occupied_periods=placement))
    by_id = {section.linked_section_group_id: section for section in sections}
    return replace(
        allocation_input,
        logical_sections=tuple(sections),
        logical_sections_by_id=by_id,
    )


def apply_candidate_to_input(
    allocation_input: CanonicalAllocationInput,
    candidate: CandidateEdit,
) -> CanonicalAllocationInput:
    """Apply an edit to an immutable in-memory canonical copy."""
    placements = dict(zip(candidate.logical_section_ids, candidate.proposed_placements))
    originals = dict(zip(candidate.logical_section_ids, candidate.original_placements))
    for section_id, proposed in placements.items():
        if not _placement_is_legal_for_original(originals[section_id], proposed):
            raise PeriodPlacementProbeError(f"invalid period placement for {section_id}: {proposed}")
    result = _replace_input_periods(allocation_input, placements)
    if len(result.logical_sections) != len(allocation_input.logical_sections):
        raise PeriodPlacementProbeError("period edit changed logical section count")
    for before, after in zip(allocation_input.logical_sections, result.logical_sections):
        if before.linked_section_group_id not in placements:
            if before != after:
                raise PeriodPlacementProbeError("period edit changed an unaffected logical section")
            continue
        if before.capacity != after.capacity or before.course_ids != after.course_ids or before.structure_type != after.structure_type:
            raise PeriodPlacementProbeError("period edit changed non-period logical-section metadata")
    return result


def apply_candidate_to_sections(
    sections: pd.DataFrame,
    candidate: CandidateEdit,
) -> pd.DataFrame:
    """Apply an edit to raw rows, preserving all non-period columns."""
    result = sections.copy(deep=True)
    before = sections.copy(deep=True)
    placement_by_group = dict(zip(candidate.logical_section_ids, candidate.proposed_placements))
    original_by_group = dict(zip(candidate.logical_section_ids, candidate.original_placements))
    for group_id, proposed in placement_by_group.items():
        if not _placement_is_legal_for_original(original_by_group[group_id], proposed):
            raise PeriodPlacementProbeError(f"invalid period placement for {group_id}: {proposed}")
    for index, row in result.iterrows():
        group_id = str(row["linked_section_group_id"])
        if group_id not in placement_by_group:
            continue
        placement = placement_by_group[group_id]
        result.at[index, "period_1"] = placement[0]
        result.at[index, "period_2"] = placement[1] if len(placement) == 2 else ""
    non_period = [column for column in result.columns if column not in {"period_1", "period_2"}]
    if not before[non_period].equals(result[non_period]):
        raise PeriodPlacementProbeError("period edit changed non-period raw section metadata")
    if len(before) != len(result) or set(before["section_id"]) != set(result["section_id"]):
        raise PeriodPlacementProbeError("period edit changed physical section rows")
    return result


def analyze_candidate(
    context: ScenarioContext,
    original_analysis: dict[str, Any],
    candidate: CandidateEdit,
) -> dict[str, Any]:
    edited = apply_candidate_to_input(context.allocation_input, candidate)
    analysis = exact_student_level_analysis(edited, candidate.core_student)
    target = analysis["target_period_units"]
    minimum_primary = max(0, analysis["primary_request_count"] - 1)
    minimum_five = min(5, analysis["primary_request_count"])
    meets_ordinary = analysis["original_max_primary_assignments"] >= minimum_primary
    meets_minimum_five = analysis["original_max_primary_assignments"] >= minimum_five
    meets_gap = analysis["original_max_primary_period_units"] >= max(target - 1, 0)
    promising = meets_ordinary and meets_minimum_five and meets_gap
    return {
        "scenario_id": context.scenario_id,
        "candidate_id": candidate.candidate_id,
        "edit_type": candidate.edit_type,
        "student_id": candidate.core_student,
        "original_max_primary_assignments": original_analysis["original_max_primary_assignments"],
        "edited_max_primary_assignments": analysis["original_max_primary_assignments"],
        "edited_primary_unmet": analysis["original_primary_unmet"],
        "edited_max_primary_period_units": analysis["original_max_primary_period_units"],
        "edited_schedule_gap": analysis["original_max_schedule_gap"],
        "ordinary_primary_unmet_at_most_one": meets_ordinary,
        "minimum_five_policy_reached": meets_minimum_five,
        "maximum_gap_one_policy_reached": meets_gap,
        "classification": "student_level_promising" if promising else "student_level_no_effect",
        "candidate_edge_membership_unchanged": True,
        "affected_candidate_edge_count": candidate.affected_candidate_edge_count,
        "capacity_unchanged": candidate.section_capacity_unchanged,
    }


def swap_disruption_metrics(candidate: CandidateEdit) -> dict[str, int]:
    """Report swap operation count separately from changed logical sections."""
    if candidate.edit_type != "logical_section_swap":
        return {"operation_count": 1, "changed_logical_section_count": len(candidate.logical_section_ids)}
    return {"operation_count": 1, "changed_logical_section_count": 2}


def validation_is_accepted(
    *,
    status: str,
    assignment_available: bool,
    response_hash: str | None,
    policy_pass: bool,
    consistency_issue_count: int,
) -> bool:
    """Apply the probe's fail-closed validated-repair contract."""
    return (
        status in {"FEASIBLE", "OPTIMAL"}
        and assignment_available
        and bool(response_hash)
        and policy_pass
        and consistency_issue_count == 0
    )


def minimum_edit_claim(
    *,
    zero_edit_proven_infeasible: bool,
    single_candidate_results: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Return a bounded claim; UNKNOWN candidates block a negative proof."""
    rows = list(single_candidate_results)
    if not zero_edit_proven_infeasible:
        return {"minimum_edit_count_within_frozen_admissible_universe": None, "proof_basis": "zero_edit_not_proven"}
    if any(row.get("status") == "UNKNOWN" for row in rows):
        return {"minimum_edit_count_within_frozen_admissible_universe": None, "proof_basis": "unknown_single_candidate"}
    if any(row.get("validated_repair") is True for row in rows):
        return {"minimum_edit_count_within_frozen_admissible_universe": 1, "proof_basis": "zero_edit_proof_and_validated_one_edit"}
    return {"minimum_edit_count_within_frozen_admissible_universe": None, "proof_basis": "all_single_candidates_completed_without_repair"}


def cost_gate_violations(
    scenario_rows: Iterable[dict[str, Any]],
    *,
    max_promising: int = 100,
    max_swaps: int = 100,
    max_invocations: int = 500,
    max_runtime_seconds: int = 18 * 60 * 60,
) -> list[str]:
    violations = []
    for row in scenario_rows:
        scenario_id = row["scenario_id"]
        if int(row["statically_promising_candidate_count"]) > max_promising:
            violations.append(f"{scenario_id}: promising candidates exceed {max_promising}")
        if int(row["swap_count"]) > max_swaps:
            violations.append(f"{scenario_id}: swaps exceed {max_swaps}")
        if int(row["estimated_maximum_solver_invocations"]) > max_invocations:
            violations.append(f"{scenario_id}: estimated solver invocations exceed {max_invocations}")
        if int(row["estimated_worst_case_runtime_seconds"]) > max_runtime_seconds:
            violations.append(f"{scenario_id}: estimated runtime exceeds {max_runtime_seconds}s")
    return violations


def _candidate_from_dict(payload: dict[str, Any]) -> CandidateEdit:
    return CandidateEdit(
        candidate_id=str(payload["candidate_id"]),
        edit_type=str(payload["edit_type"]),
        logical_section_ids=tuple(payload["logical_section_ids"]),
        logical_course_ids=tuple(payload["logical_course_ids"]),
        original_placements=tuple(tuple(item) for item in payload["original_placements"]),
        proposed_placements=tuple(tuple(item) for item in payload["proposed_placements"]),
        valid_period_source=str(payload["valid_period_source"]),
        occupancy_shape=tuple(tuple(item) for item in payload["occupancy_shape"]),
        core_student=str(payload["core_student"]),
        core_period_relevance=tuple(payload["core_period_relevance"]),
        affected_candidate_edge_count=int(payload["affected_candidate_edge_count"]),
        affected_student_count=int(payload["affected_student_count"]),
        section_capacity_unchanged=bool(payload.get("section_capacity_unchanged", True)),
    )


def run_formal_probe(
    *,
    preview_dir: str | Path = DEFAULT_OUTPUT,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    audit_root: str | Path = DEFAULT_AUDIT_ROOT,
    config_dir: str | Path = "data/config",
    candidate_runner: Any | None = None,
) -> dict[str, Any]:
    """Resume a formal probe only after the frozen cost gate passes.

    The default candidate runner calls the unchanged production model. A test
    caller may inject a deterministic runner, but this function never reduces
    the frozen candidate universe to fit a budget. Completed candidate records
    are skipped and each updated scenario checkpoint is written atomically.
    """
    preview_dir = Path(preview_dir)
    manifest = load_period_placement_probe_manifest(manifest_path)
    aggregate_path = preview_dir / "aggregate_summary.json"
    if not aggregate_path.is_file():
        raise PeriodPlacementProbeError("formal probe requires a completed cost preview")
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    violations = cost_gate_violations(aggregate.get("scenarios", []))
    if violations:
        raise PeriodPlacementProbeError(
            "cost gate blocks formal probe; review candidate scale first: " + "; ".join(violations)
        )
    audit_manifest = load_section_plan_audit_manifest("data/scenarios/section_plan_feasibility_audit_v1.json")
    runner = candidate_runner or validate_candidate_with_production_model
    total_runs = 0
    resumed = 0
    skipped = 0
    for scenario_id in SCENARIO_ORDER:
        scenario_root = preview_dir / "scenarios" / scenario_id
        universe_path = scenario_root / "candidate_universe.json"
        if not universe_path.is_file():
            raise PeriodPlacementProbeError(f"missing candidate checkpoint: {universe_path}")
        candidates = [_candidate_from_dict(item) for item in json.loads(universe_path.read_text(encoding="utf-8"))]
        checkpoint_path = scenario_root / "candidate_runs.json"
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8")) if checkpoint_path.is_file() else {}
        runs = list(checkpoint.get("runs", []))
        completed = {row.get("candidate_id") for row in runs if row.get("checkpoint_status") == "complete"}
        scenario_candidates = [candidate for candidate in candidates if candidate.edit_type == "single_section_move"]
        if not scenario_candidates and not completed:
            _write_json(checkpoint_path, {"status": "complete", "runs": [], "solver_runs": 0})
            continue
        context = load_scenario_context(
            scenario_id,
            audit_manifest=audit_manifest,
            audit_root=audit_root,
            config_dir=config_dir,
        )
        for candidate in scenario_candidates:
            if candidate.candidate_id in completed:
                skipped += 1
                continue
            result = runner(context, candidate)
            runs.append({"checkpoint_status": "complete", **result})
            total_runs += 1
            _write_json(checkpoint_path, {"status": "running", "runs": runs, "solver_runs": total_runs})
        _write_json(checkpoint_path, {"status": "complete", "runs": runs, "solver_runs": total_runs})
        resumed += len(completed)
    return {
        "status": "completed",
        "solver_runs": total_runs,
        "resumed_candidates": resumed,
        "skipped_complete_candidates": skipped,
        "stress_runs": 0,
        "holdout_runs": 0,
        "manifest": manifest["probe_name"],
    }


def build_cost_preview(context: ScenarioContext) -> dict[str, Any]:
    candidates = generate_candidate_universe(context)
    if context.authoritative_core is None:
        return {
            "scenario_id": context.scenario_id,
            "role": "feasible_control",
            "authoritative_student": None,
            "raw_candidate_count": 0,
            "statically_promising_candidate_count": 0,
            "single_move_count": 0,
            "swap_count": 0,
            "potential_pair_count": 0,
            "estimated_maximum_solver_invocations": 0,
            "estimated_worst_case_runtime_seconds": 0,
            "candidate_universe": [],
            "student_level_analysis": None,
        }
    core = context.authoritative_core
    original = exact_student_level_analysis(context.allocation_input, core.student_id)
    analyses = [analyze_candidate(context, original, candidate) for candidate in candidates]
    promising = [row for row in analyses if row["classification"] == "student_level_promising"]
    promising_ids = {row["candidate_id"] for row in promising}
    single_promising = [row for row in promising if row["edit_type"] == "single_section_move"]
    potential_pairs = 0
    for index, first in enumerate(single_promising):
        first_candidate = next(item for item in candidates if item.candidate_id == first["candidate_id"])
        for second in single_promising[index + 1:]:
            second_candidate = next(item for item in candidates if item.candidate_id == second["candidate_id"])
            if set(first_candidate.logical_section_ids).isdisjoint(second_candidate.logical_section_ids):
                potential_pairs += 1
    return {
        "scenario_id": context.scenario_id,
        "role": "infeasible_target",
        "authoritative_student": core.student_id,
        "core_periods": list(core.core_periods),
        "relevant_primary_courses": sorted({
            request.candidate_key
            for request in context.allocation_input.students_by_id[core.student_id].primary_requests
        }),
        "raw_candidate_count": len(candidates),
        "statically_promising_candidate_count": len(promising),
        "single_move_count": sum(item.edit_type == "single_section_move" for item in candidates),
        "swap_count": sum(item.edit_type == "logical_section_swap" for item in candidates),
        "potential_pair_count": potential_pairs,
        "estimated_maximum_solver_invocations": len(promising) + potential_pairs,
        "estimated_worst_case_runtime_seconds": (len(promising) + potential_pairs) * 120,
        "candidate_universe": [asdict(item) for item in candidates],
        "student_level_analysis": {"original": original, "candidates": analyses},
        "promising_candidate_ids": sorted(promising_ids),
    }


def materialize_candidate_canonical_input(
    context: ScenarioContext,
    candidate: CandidateEdit,
) -> CanonicalAllocationInput:
    """Rebuild candidate mappings from the edited raw section copy."""
    edited_sections = apply_candidate_to_sections(context.sections, candidate)
    return canonicalize_allocation_input(
        context.students.copy(deep=True),
        context.requests.copy(deep=True),
        edited_sections,
        context.catalog.copy(deep=True),
    )


def validate_candidate_with_production_model(
    context: ScenarioContext,
    candidate: CandidateEdit,
    *,
    solver_seed: int = 20260630,
    time_limit_seconds: int = 120,
) -> dict[str, Any]:
    """Run one explicit candidate through the unchanged production hard model.

    This function is intentionally not called by the cost-preview CLI.  A
    caller must opt into the expensive diagnostic operation after reviewing
    the frozen candidate counts.
    """
    allocation_input = materialize_candidate_canonical_input(context, candidate)
    catalog = context.catalog
    math_ids = math_course_ids_from_catalog(catalog)
    fallback_rules = _load_math_fallback_rules(Path("data/config"), catalog)
    hint = run_constrained_first_baseline(
        allocation_input,
        solver_seed,
        math_fallback_rules=fallback_rules,
        math_course_ids=math_ids,
    )
    result = run_fair_cp_sat_solver(
        allocation_input,
        seed=solver_seed,
        math_fallback_rules=fallback_rules,
        math_course_ids=math_ids,
        max_time_seconds_per_stage=float(time_limit_seconds),
        max_total_time_seconds=float(time_limit_seconds),
        num_search_workers=1,
        use_feasibility_bootstrap=False,
        use_constrained_first_hint=False,
        logical_schedule_completion_enabled=False,
        internal_feasibility_hint_strategy="constrained_first",
        internal_repair_time_seconds=float(time_limit_seconds),
        internal_repair_objective_strategy="hamming_to_constrained_first",
        stop_after_first_valid_solution=True,
    )
    status = result.status.value
    assignment_available = bool(result.assignments)
    response_hash = next(
        (diagnostic.response_proto_hash for diagnostic in reversed(result.stage_diagnostics)
         if diagnostic.response_proto_hash),
        None,
    )
    policy_pass = bool(result.policy_report and result.policy_report.summary.final_schedule_policy_pass)
    consistency_issue_count = len(result.consistency_issues)
    return {
        "candidate_id": candidate.candidate_id,
        "status": status,
        "response_hash": response_hash,
        "assignment_available": assignment_available,
        "policy_pass": policy_pass,
        "consistency_issue_count": consistency_issue_count,
        "hint_source": "constrained_first_internal",
        "validated_repair": validation_is_accepted(
            status=status,
            assignment_available=assignment_available,
            response_hash=response_hash,
            policy_pass=policy_pass,
            consistency_issue_count=consistency_issue_count,
        ),
        "hint_primary_assigned": sum(
            outcome.status.value == "assigned"
            for outcome in hint.request_outcomes
            if outcome.request_type == "primary"
        ),
    }


def _validate_source_artifacts(manifest: dict[str, Any], audit_root: Path) -> dict[str, Any]:
    raw_sums = audit_root / "SHA256SUMS.txt"
    audited_root = audit_root.with_name(audit_root.name + "-audited")
    audited_sums = audited_root / "SHA256SUMS.txt"
    if not raw_sums.is_file() or not audited_sums.is_file():
        raise PeriodPlacementProbeError("raw/audited section-plan artifacts are incomplete")
    raw_hash = _sha256_file(raw_sums)
    audited_hash = _sha256_file(audited_sums)
    if raw_hash != manifest["source_raw_artifact_hash"]:
        raise PeriodPlacementProbeError("raw audit artifact checksum manifest hash mismatch")
    if audited_hash != manifest["source_audited_artifact_hash"]:
        raise PeriodPlacementProbeError("audited artifact checksum manifest hash mismatch")
    run_manifest_path = audit_root / "run_manifest.json"
    if not run_manifest_path.is_file():
        raise PeriodPlacementProbeError("raw audit run manifest is missing")
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if run_manifest.get("audit_manifest_sha256") != manifest["source_section_audit_manifest_hash"]:
        raise PeriodPlacementProbeError("section-plan audit manifest lineage hash mismatch")
    return {
        "raw_sha256sums_sha256": raw_hash,
        "audited_sha256sums_sha256": audited_hash,
        "raw_artifact_read_only": True,
        "audited_artifact_read_only": True,
    }


def write_cost_preview(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    output_dir: str | Path = DEFAULT_OUTPUT,
    audit_root: str | Path = DEFAULT_AUDIT_ROOT,
    config_dir: str | Path = "data/config",
) -> dict[str, Any]:
    """Generate only candidate/static-analysis artifacts; never invokes CP-SAT."""
    manifest_path = Path(manifest_path)
    output_dir = Path(output_dir)
    audit_root = Path(audit_root)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise PeriodPlacementProbeError(f"probe output is non-empty; refusing to overwrite: {output_dir}")
    manifest = load_period_placement_probe_manifest(manifest_path)
    audit_manifest = load_section_plan_audit_manifest("data/scenarios/section_plan_feasibility_audit_v1.json")
    source_info = _validate_source_artifacts(manifest, audit_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "probe_manifest_snapshot.json", manifest)
    scenario_rows = []
    candidate_rows = []
    analysis_rows = []
    for scenario_id in SCENARIO_ORDER:
        context = load_scenario_context(
            scenario_id,
            audit_manifest=audit_manifest,
            audit_root=audit_root,
            config_dir=config_dir,
        )
        preview = build_cost_preview(context)
        scenario_rows.append({key: value for key, value in preview.items() if key not in {"candidate_universe", "student_level_analysis"}})
        for item in preview["candidate_universe"]:
            candidate_rows.append(item)
        analysis = preview.get("student_level_analysis")
        if analysis:
            analysis_rows.append({"scenario_id": scenario_id, "analysis_type": "original", **analysis["original"]})
            analysis_rows.extend({"scenario_id": scenario_id, "analysis_type": "candidate", **row} for row in analysis["candidates"])
        scenario_root = output_dir / "scenarios" / scenario_id
        _write_json(scenario_root / "authoritative_core_input.json", asdict(context.authoritative_core) if context.authoritative_core else {"scenario_id": scenario_id, "role": "feasible_control"})
        _write_json(scenario_root / "candidate_universe.json", preview["candidate_universe"])
        _write_json(scenario_root / "static_student_analysis.json", preview["student_level_analysis"] or {})
        _write_json(scenario_root / "candidate_runs.json", {"status": "not_started", "solver_runs": 0})
        _write_json(scenario_root / "validated_repairs.json", [])
        _write_json(scenario_root / "repair_summary.json", {
            "status": "cost_preview_only",
            "formal_validation_started": False,
            "minimum_claim": "unresolved_until_frozen_candidates_are_validated",
        })
    _write_csv(output_dir / "authoritative_core_inputs.csv", [
        {"scenario_id": row["scenario_id"], "authoritative_student": row.get("authoritative_student"), "core_periods": ";".join(row.get("core_periods", []))}
        for row in scenario_rows
    ])
    _write_csv(output_dir / "candidate_universe.csv", candidate_rows)
    _write_csv(output_dir / "student_level_analysis.csv", analysis_rows)
    _write_csv(output_dir / "candidate_validation_summary.csv", [])
    _write_csv(output_dir / "validated_repairs.csv", [])
    _write_csv(output_dir / "rejected_candidates.csv", [])
    _write_csv(output_dir / "unknown_candidates.csv", [])
    _write_csv(output_dir / "cross_scenario_templates.csv", [])
    aggregate = {
        "schema_version": SCHEMA_VERSION,
        "probe_name": manifest["probe_name"],
        "mode": "candidate_universe_cost_preview",
        "formal_solver_runs": 0,
        "stress_runs": 0,
        "holdout_runs": 0,
        "unexpected_correctness_failures": 0,
        "scenarios": scenario_rows,
        "candidate_counts": {
            "raw": sum(row["raw_candidate_count"] for row in scenario_rows),
            "promising": sum(row["statically_promising_candidate_count"] for row in scenario_rows),
            "potential_pairs": sum(row["potential_pair_count"] for row in scenario_rows),
        },
        "source_artifacts": source_info,
        "formal_probe_status": "not_started_pending_cost_gate",
    }
    _write_json(output_dir / "aggregate_summary.json", aggregate)
    _write_json(output_dir / "failures.json", {"failures": [], "unexpected_failure_count": 0})
    _write_json(output_dir / "provenance.json", {
        "source_git_commit": manifest["source_git_commit"],
        "source_section_audit_manifest_hash": manifest["source_section_audit_manifest_hash"],
        "source_raw_artifact_hash": manifest["source_raw_artifact_hash"],
        "source_audited_artifact_hash": manifest["source_audited_artifact_hash"],
        "no_new_solver_runs": True,
        "development_data": True,
        "stress_runs": 0,
        "holdout_runs": 0,
        "teacher_room_constraints_modeled": False,
    })
    _write_json(output_dir / "run_manifest.json", {
        "schema_version": SCHEMA_VERSION,
        "status": "cost_preview_completed",
        "completed_scenario_ids": list(SCENARIO_ORDER),
        "solver_runs": 0,
        "stress_runs": 0,
        "holdout_runs": 0,
    })
    checksum_hash = _write_checksums(output_dir)
    aggregate["sha256sums_sha256"] = checksum_hash
    _write_json(output_dir / "aggregate_summary.json", aggregate)
    _write_checksums(output_dir)
    return aggregate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preview frozen period-placement repair candidates.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--audit-root", default=str(DEFAULT_AUDIT_ROOT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--config-dir", default="data/config")
    args = parser.parse_args(argv)
    try:
        result = write_cost_preview(
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            audit_root=args.audit_root,
            config_dir=args.config_dir,
        )
    except PeriodPlacementProbeError as exc:
        print(f"Period-placement repair probe FAILED: {exc}")
        return 1
    print("Period-placement repair probe cost preview PASS")
    print(f"Scenarios: {len(result['scenarios'])}; formal solver runs: 0")
    for row in result["scenarios"]:
        print(
            f"{row['scenario_id']}: raw={row['raw_candidate_count']} "
            f"promising={row['statically_promising_candidate_count']} "
            f"moves={row['single_move_count']} swaps={row['swap_count']} "
            f"pairs={row['potential_pair_count']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
