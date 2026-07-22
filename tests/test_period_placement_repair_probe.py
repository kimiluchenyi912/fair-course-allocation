from __future__ import annotations

import json

import pandas as pd
import pytest

from src.allocation import canonicalize_allocation_input
from src.period_placement_repair_probe import (
    AuthoritativeCoreInput,
    CandidateEdit,
    ScenarioContext,
    apply_candidate_to_input,
    apply_candidate_to_sections,
    build_cost_preview,
    cost_gate_violations,
    exact_student_level_analysis,
    generate_candidate_universe,
    legal_placements,
    load_period_placement_probe_manifest,
    minimum_edit_claim,
    swap_disruption_metrics,
    validation_is_accepted,
)
from src.allocation.input_models import LogicalSection

from tests.test_cp_sat_solver import (
    base_sections,
    canonical,
    catalog,
    request_row,
    requests,
    section_row,
    sections,
    students,
)


def _context(
    *,
    core_student: str = "STU_1",
    section_rows: list[tuple] | None = None,
    request_courses: tuple[str, ...] = ("CORE_A", "CORE_B", "CORE_C"),
    target: int = 3,
) -> ScenarioContext:
    raw_students = students([(core_student, 12, target, False)])
    raw_requests = requests([request_row(core_student, course) for course in request_courses])
    raw_sections = sections(section_rows or [
        section_row("CORE_A_1", "CORE_A", "P1", capacity=10, group_id="CORE_A_1"),
        section_row("CORE_A_2", "CORE_A", "P2", capacity=10, group_id="CORE_A_2"),
        section_row("CORE_B_1", "CORE_B", "P1", capacity=10, group_id="CORE_B_1"),
        section_row("CORE_C_1", "CORE_C", "P2", capacity=10, group_id="CORE_C_1"),
    ])
    allocation_input = canonical(
        [(core_student, 12, target, False)],
        [request_row(core_student, course) for course in request_courses],
        section_rows or [
            section_row("CORE_A_1", "CORE_A", "P1", capacity=10, group_id="CORE_A_1"),
            section_row("CORE_A_2", "CORE_A", "P2", capacity=10, group_id="CORE_A_2"),
            section_row("CORE_B_1", "CORE_B", "P1", capacity=10, group_id="CORE_B_1"),
            section_row("CORE_C_1", "CORE_C", "P2", capacity=10, group_id="CORE_C_1"),
        ],
    )
    core = AuthoritativeCoreInput(
        scenario_id="fixture_target",
        student_id=core_student,
        core_periods=("P1", "P2"),
        core_literals=({"family": "ORDINARY_MAX_PRIMARY_UNMET", "id": core_student},),
        source_file="fixture/fine_core.json",
        source_hash="fixture",
        evidence_type="audited_fine_sufficient_core",
        minimality_status="unresolved_time_budget",
    )
    return ScenarioContext(
        scenario_id="fixture_target",
        allocation_input=allocation_input,
        students=raw_students,
        requests=raw_requests,
        sections=raw_sections,
        catalog=catalog(),
        authoritative_core=core,
    )


def _candidate(section_id: str, original: tuple[str, ...], proposed: tuple[str, ...], edit_type: str = "single_section_move") -> CandidateEdit:
    return CandidateEdit(
        candidate_id=f"fixture:{edit_type}:{section_id}",
        edit_type=edit_type,
        logical_section_ids=(section_id,),
        logical_course_ids=("CORE_A",),
        original_placements=(original,),
        proposed_placements=(proposed,),
        valid_period_source="fixture",
        occupancy_shape=((len(original),),),
        core_student="STU_1",
        core_period_relevance=tuple(original),
        affected_candidate_edge_count=1,
        affected_student_count=1,
    )


def test_probe_manifest_is_exactly_control_plus_seven_targets() -> None:
    payload = load_period_placement_probe_manifest()
    assert payload["scenarios"] == [
        "normal_dev_reference_2026", "normal_dev_01", "normal_dev_03",
        "normal_dev_04", "normal_dev_05", "normal_dev_07", "normal_dev_09", "normal_dev_10",
    ]
    assert payload["tuning_allowed"] is False
    assert payload["stress_execution_allowed"] is False
    assert payload["holdout_execution_allowed"] is False
    assert payload["solver_seed"] == 20260630
    assert payload["workers"] == 1


def test_probe_manifest_has_only_allowed_edit_types() -> None:
    payload = load_period_placement_probe_manifest()
    assert set(payload["allowed_edit_types"]) == {"single_section_move", "logical_section_swap"}


def test_legal_single_periods_cover_p1_to_p7() -> None:
    section = _context().allocation_input.logical_sections_by_id["CORE_A_1"]
    assert legal_placements(section) == tuple((f"P{i}",) for i in range(1, 8))


@pytest.mark.parametrize("period", [f"P{i}" for i in range(1, 8)])
def test_every_single_period_is_valid(period: str) -> None:
    section = _context().allocation_input.logical_sections_by_id["CORE_A_1"]
    assert (period,) in legal_placements(section)


def test_double_period_uses_consecutive_pairs_only() -> None:
    allocation_input = canonical([], [], base_sections())
    section = allocation_input.logical_sections_by_id["MATH23_1"]
    assert legal_placements(section) == tuple((f"P{i}", f"P{i + 1}") for i in range(1, 7))


@pytest.mark.parametrize("start", range(1, 7))
def test_each_double_period_pair_is_consecutive(start: int) -> None:
    allocation_input = canonical([], [], base_sections())
    section = allocation_input.logical_sections_by_id["MATH23_1"]
    assert (f"P{start}", f"P{start + 1}") in legal_placements(section)


def test_exact_dp_detects_period_conflict() -> None:
    conflict_context = _context(section_rows=[
        section_row("CORE_A_1", "CORE_A", "P1", capacity=10, group_id="CORE_A_1"),
        section_row("CORE_B_1", "CORE_B", "P1", capacity=10, group_id="CORE_B_1"),
        section_row("CORE_C_1", "CORE_C", "P2", capacity=10, group_id="CORE_C_1"),
    ])
    analysis = exact_student_level_analysis(conflict_context.allocation_input, "STU_1")
    assert analysis["original_max_primary_assignments"] == 2
    assert analysis["original_primary_unmet"] == 1
    assert ["CORE_A", "CORE_B"] in analysis["conflicting_primary_course_sets"]


def test_exact_dp_counts_period_units_not_request_rows() -> None:
    raw_requests = requests([request_row("STU_1", "MATH2_3_HA")])
    raw_sections = sections([section_row("MATH23_1", "MATH2_3_HA", "P5", "P6", capacity=10, group_id="MATH23_1")])
    result = canonical([("STU_1", 12, 2, False)], [request_row("STU_1", "MATH2_3_HA")], [section_row("MATH23_1", "MATH2_3_HA", "P5", "P6", capacity=10, group_id="MATH23_1")])
    analysis = exact_student_level_analysis(result, "STU_1")
    assert analysis["original_max_primary_assignments"] == 1
    assert analysis["original_max_primary_period_units"] == 2


def test_single_move_can_make_core_student_locally_promising() -> None:
    context = _context()
    original = exact_student_level_analysis(context.allocation_input, "STU_1")
    candidate = _candidate("CORE_B_1", ("P1",), ("P3",))
    from src.period_placement_repair_probe import analyze_candidate
    result = analyze_candidate(context, original, candidate)
    assert result["edited_max_primary_assignments"] == 3
    assert result["classification"] == "student_level_promising"


def test_generated_moves_touch_only_authoritative_primary_courses() -> None:
    context = _context()
    candidates = generate_candidate_universe(context)
    assert candidates
    assert all(set(candidate.logical_course_ids) <= {"CORE_A", "CORE_B", "CORE_C"} for candidate in candidates)


def test_generated_candidates_do_not_use_invalid_witness_student() -> None:
    context = _context(core_student="G12_0536")
    candidates = generate_candidate_universe(context)
    assert all(candidate.core_student == "G12_0536" for candidate in candidates)
    assert all(candidate.core_student != "G12_0105" for candidate in candidates)


def test_apply_move_preserves_unaffected_sections_and_capacity() -> None:
    context = _context()
    candidate = _candidate("CORE_A_1", ("P1",), ("P4",))
    edited = apply_candidate_to_input(context.allocation_input, candidate)
    assert edited.logical_sections_by_id["CORE_A_1"].occupied_periods == ("P4",)
    assert edited.logical_sections_by_id["CORE_A_1"].capacity == 10
    assert edited.logical_sections_by_id["CORE_B_1"] == context.allocation_input.logical_sections_by_id["CORE_B_1"]


def test_apply_move_preserves_raw_row_identity_and_non_period_metadata() -> None:
    context = _context()
    candidate = _candidate("CORE_A_1", ("P1",), ("P4",))
    edited = apply_candidate_to_sections(context.sections, candidate)
    assert set(edited["section_id"]) == set(context.sections["section_id"])
    assert edited.loc[edited["section_id"] == "CORE_A_1", "capacity"].item() == 10
    assert edited.loc[edited["section_id"] == "CORE_A_1", "period_1"].item() == "P4"


def test_invalid_period_is_rejected() -> None:
    with pytest.raises(ValueError, match="invalid period placement"):
        apply_candidate_to_input(_context().allocation_input, _candidate("CORE_A_1", ("P1",), ("P8",)))


def test_double_period_nonconsecutive_destination_is_rejected() -> None:
    candidate = CandidateEdit(
        candidate_id="bad-double",
        edit_type="single_section_move",
        logical_section_ids=("MATH23_1",),
        logical_course_ids=("MATH2_3_HA",),
        original_placements=(("P5", "P6"),),
        proposed_placements=(("P1", "P3"),),
        valid_period_source="fixture",
        occupancy_shape=((2, 1),),
        core_student="STU_1",
        core_period_relevance=("P5",),
        affected_candidate_edge_count=1,
        affected_student_count=1,
    )
    allocation_input = canonical([], [], base_sections())
    with pytest.raises(ValueError, match="invalid period placement"):
        apply_candidate_to_input(allocation_input, candidate)


def test_ha_move_is_atomic() -> None:
    allocation_input = canonical([], [], base_sections())
    candidate = CandidateEdit(
        candidate_id="ha-move",
        edit_type="single_section_move",
        logical_section_ids=("MATH23_1",),
        logical_course_ids=("MATH2_3_HA",),
        original_placements=(("P5", "P6"),),
        proposed_placements=(("P1", "P2"),),
        valid_period_source="fixture",
        occupancy_shape=((2, 1),),
        core_student="STU_1",
        core_period_relevance=("P5", "P6"),
        affected_candidate_edge_count=1,
        affected_student_count=1,
    )
    edited = apply_candidate_to_sections(sections(base_sections()), candidate)
    row = edited[edited["linked_section_group_id"] == "MATH23_1"].iloc[0]
    assert (row["period_1"], row["period_2"]) == ("P1", "P2")


def test_linked_rows_move_together() -> None:
    allocation_input = canonical([], [], base_sections())
    candidate = CandidateEdit(
        candidate_id="linked-move",
        edit_type="single_section_move",
        logical_section_ids=("GOV_1",),
        logical_course_ids=("GOV_ECON_REG",),
        original_placements=(("P7",),),
        proposed_placements=(("P2",),),
        valid_period_source="fixture",
        occupancy_shape=((1,),),
        core_student="STU_1",
        core_period_relevance=("P7",),
        affected_candidate_edge_count=1,
        affected_student_count=1,
    )
    edited = apply_candidate_to_sections(sections(base_sections()), candidate)
    rows = edited[edited["linked_section_group_id"] == "GOV_1"]
    assert len(rows) == 2
    assert set(rows["period_1"]) == {"P2"}
    assert set(rows["semester"]) == {"semester_1", "semester_2"}


def test_swap_reports_operation_and_changed_section_counts() -> None:
    candidate = CandidateEdit(
        candidate_id="swap",
        edit_type="logical_section_swap",
        logical_section_ids=("CORE_A_1", "CORE_B_1"),
        logical_course_ids=("CORE_A", "CORE_B"),
        original_placements=(("P1",), ("P3",)),
        proposed_placements=(("P3",), ("P1",)),
        valid_period_source="fixture",
        occupancy_shape=((1,), (1,)),
        core_student="STU_1",
        core_period_relevance=("P1", "P3"),
        affected_candidate_edge_count=2,
        affected_student_count=1,
    )
    assert swap_disruption_metrics(candidate) == {"operation_count": 1, "changed_logical_section_count": 2}


def test_incompatible_double_and_single_are_not_swap_candidates() -> None:
    allocation_input = canonical([], [], base_sections())
    from src.period_placement_repair_probe import _compatible_for_swap
    assert not _compatible_for_swap(
        allocation_input.logical_sections_by_id["MATH23_1"],
        allocation_input.logical_sections_by_id["CORE_A_1"],
    )


def test_candidate_membership_stays_identity_based_after_period_move() -> None:
    context = _context()
    candidate = _candidate("CORE_A_1", ("P1",), ("P4",))
    edited = apply_candidate_to_input(context.allocation_input, candidate)
    assert edited.candidate_index == context.allocation_input.candidate_index


def test_control_has_no_repair_candidates() -> None:
    raw_students = students()
    raw_requests = requests([])
    raw_sections = sections([])
    context = ScenarioContext(
        scenario_id="normal_dev_reference_2026",
        allocation_input=canonical([], [], []),
        students=raw_students,
        requests=raw_requests,
        sections=raw_sections,
        catalog=catalog(),
        authoritative_core=None,
    )
    preview = build_cost_preview(context)
    assert preview["role"] == "feasible_control"
    assert preview["raw_candidate_count"] == 0
    assert preview["estimated_maximum_solver_invocations"] == 0


def test_unknown_candidate_prevents_minimum_claim() -> None:
    claim = minimum_edit_claim(
        zero_edit_proven_infeasible=True,
        single_candidate_results=[{"status": "UNKNOWN", "validated_repair": False}],
    )
    assert claim["minimum_edit_count_within_frozen_admissible_universe"] is None
    assert claim["proof_basis"] == "unknown_single_candidate"


def test_validated_one_edit_claim_is_bounded_to_frozen_universe() -> None:
    claim = minimum_edit_claim(
        zero_edit_proven_infeasible=True,
        single_candidate_results=[{"status": "FEASIBLE", "validated_repair": True}],
    )
    assert claim["minimum_edit_count_within_frozen_admissible_universe"] == 1


def test_infeasible_is_not_accepted_without_assignment() -> None:
    assert not validation_is_accepted(
        status="INFEASIBLE", assignment_available=False, response_hash=None,
        policy_pass=False, consistency_issue_count=0,
    )


@pytest.mark.parametrize("status", ["UNKNOWN", "INFEASIBLE", "MODEL_INVALID"])
def test_nonincumbent_status_is_not_validated(status: str) -> None:
    assert not validation_is_accepted(
        status=status, assignment_available=False, response_hash=None,
        policy_pass=False, consistency_issue_count=0,
    )


def test_feasible_policy_failure_is_not_validated() -> None:
    assert not validation_is_accepted(
        status="FEASIBLE", assignment_available=True, response_hash="hash",
        policy_pass=False, consistency_issue_count=0,
    )


def test_feasible_consistency_failure_is_not_validated() -> None:
    assert not validation_is_accepted(
        status="FEASIBLE", assignment_available=True, response_hash="hash",
        policy_pass=True, consistency_issue_count=1,
    )


def test_feasible_complete_response_is_validated() -> None:
    assert validation_is_accepted(
        status="FEASIBLE", assignment_available=True, response_hash="hash",
        policy_pass=True, consistency_issue_count=0,
    )


def test_no_single_edit_repair_claim_requires_all_candidates_complete() -> None:
    claim = minimum_edit_claim(
        zero_edit_proven_infeasible=True,
        single_candidate_results=[
            {"status": "INFEASIBLE", "validated_repair": False},
            {"status": "FEASIBLE", "validated_repair": False},
        ],
    )
    assert claim["minimum_edit_count_within_frozen_admissible_universe"] is None
    assert claim["proof_basis"] == "all_single_candidates_completed_without_repair"


def test_zero_edit_proof_is_required_for_minimum_claim() -> None:
    claim = minimum_edit_claim(
        zero_edit_proven_infeasible=False,
        single_candidate_results=[{"status": "FEASIBLE", "validated_repair": True}],
    )
    assert claim["proof_basis"] == "zero_edit_not_proven"


def test_authoritative_core_has_traceable_evidence_source() -> None:
    context = _context()
    assert context.authoritative_core.evidence_type == "audited_fine_sufficient_core"
    assert context.authoritative_core.source_file.endswith("fine_core.json")


def test_candidate_preview_counts_moves_and_swaps_separately() -> None:
    preview = build_cost_preview(_context())
    assert preview["single_move_count"] > 0
    assert preview["swap_count"] > 0
    assert preview["raw_candidate_count"] == preview["single_move_count"] + preview["swap_count"]


def test_candidate_preview_has_no_solver_invocations() -> None:
    preview = build_cost_preview(_context())
    assert preview["estimated_maximum_solver_invocations"] >= 0


def test_cost_gate_blocks_large_candidate_universe() -> None:
    violations = cost_gate_violations([{
        "scenario_id": "fixture_target",
        "statically_promising_candidate_count": 101,
        "swap_count": 0,
        "estimated_maximum_solver_invocations": 101,
        "estimated_worst_case_runtime_seconds": 101,
    }])
    assert any("promising candidates" in item for item in violations)


def test_cost_gate_allows_small_fixture() -> None:
    assert cost_gate_violations([{
        "scenario_id": "fixture_target",
        "statically_promising_candidate_count": 1,
        "swap_count": 1,
        "estimated_maximum_solver_invocations": 2,
        "estimated_worst_case_runtime_seconds": 240,
    }]) == []


def test_candidate_capacity_flag_is_always_true() -> None:
    candidates = generate_candidate_universe(_context())
    assert candidates
    assert all(candidate.section_capacity_unchanged for candidate in candidates)


def test_candidate_course_identity_is_stable() -> None:
    context = _context()
    for candidate in generate_candidate_universe(context):
        edited = apply_candidate_to_input(context.allocation_input, candidate)
        for section_id in candidate.logical_section_ids:
            before = context.allocation_input.logical_sections_by_id[section_id]
            after = edited.logical_sections_by_id[section_id]
            assert before.course_ids == after.course_ids
            assert before.logical_block_id == after.logical_block_id


def test_candidate_section_count_is_stable() -> None:
    context = _context()
    candidate = generate_candidate_universe(context)[0]
    edited = apply_candidate_to_input(context.allocation_input, candidate)
    assert len(edited.logical_sections) == len(context.allocation_input.logical_sections)


def test_candidate_row_count_is_stable_for_linked_section() -> None:
    context = _context(section_rows=base_sections(), request_courses=())
    candidate = CandidateEdit(
        candidate_id="linked-row-count",
        edit_type="single_section_move",
        logical_section_ids=("GOV_1",), logical_course_ids=("GOV_ECON_REG",),
        original_placements=(("P7",),), proposed_placements=(("P2",),),
        valid_period_source="fixture", occupancy_shape=((1,),), core_student="STU_1",
        core_period_relevance=("P7",), affected_candidate_edge_count=0, affected_student_count=0,
    )
    edited = apply_candidate_to_sections(context.sections, candidate)
    assert len(edited) == len(context.sections)


def test_json_serializable_core_fixture(tmp_path) -> None:
    context = _context()
    path = tmp_path / "core.json"
    path.write_text(json.dumps({"student_id": context.authoritative_core.student_id}), encoding="utf-8")
    assert json.loads(path.read_text())["student_id"] == "STU_1"
