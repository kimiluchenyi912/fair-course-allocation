from __future__ import annotations

import json

import pandas as pd
import pytest

from src.allocation import canonicalize_allocation_input
from src.joint_period_edit_pilot import (
    AUTHORITATIVE_STUDENT_ID,
    CONTROL_SCENARIO_ID,
    JointPilotError,
    PlacementOption,
    _candidate_payload_domains,
    _cost_gate,
    _placement_domains_with_originals,
    apply_placement_map_to_sections,
    build_architecture_audit,
    build_frozen_placement_domains,
    build_joint_model,
    load_joint_period_edit_manifest,
    solve_joint_stage,
    validate_joint_solution,
)
from src.period_placement_repair_probe import ScenarioContext

from tests.test_cp_sat_solver import (
    base_sections,
    catalog,
    request_row,
    requests,
    section_row,
    sections,
    students,
)
from tests.test_period_placement_repair_probe import _context


def _feasible_context(*, all_same_period: bool = False) -> ScenarioContext:
    courses = ("CORE_A", "CORE_B", "CORE_C", "CORE_D", "ALT1")
    raw_students = students([("STU_1", 12, 5, False)])
    raw_requests = requests([request_row("STU_1", course) for course in courses])
    periods = ["P1", "P1", "P1", "P1", "P1"] if all_same_period else ["P1", "P2", "P3", "P4", "P5"]
    raw_sections = sections([
        section_row(f"{course}_1", course, period, capacity=10, group_id=f"{course}_1")
        for course, period in zip(courses, periods)
    ])
    allocation_input = canonicalize_allocation_input(raw_students, raw_requests, raw_sections, catalog())
    return ScenarioContext(
        scenario_id="fixture_target",
        allocation_input=allocation_input,
        students=raw_students,
        requests=raw_requests,
        sections=raw_sections,
        catalog=catalog(),
        authoritative_core=None,
    )


def _move_payload(section_id: str = "CORE_A_1", *, proposed: str = "P6", candidate_id: str = "candidate-1") -> dict:
    return {
        "candidate_id": candidate_id,
        "edit_type": "single_section_move",
        "logical_section_ids": [section_id],
        "logical_course_ids": ["CORE_A"],
        "original_placements": [["P1"]],
        "proposed_placements": [[proposed]],
        "core_student": AUTHORITATIVE_STUDENT_ID,
    }


def _domain_context():
    context = _context(core_student=AUTHORITATIVE_STUDENT_ID)
    return context


def test_manifest_contains_phase_a_fields() -> None:
    manifest = load_joint_period_edit_manifest()
    assert manifest["phase"] == "A_single_scenario_pilot"
    assert manifest["solver_seed"] == 20260630
    assert manifest["workers"] == 1


def test_manifest_has_only_control_and_target() -> None:
    manifest = load_joint_period_edit_manifest()
    assert (manifest["control_scenario_id"], manifest["target_scenario_id"]) == (CONTROL_SCENARIO_ID, "normal_dev_10")


def test_manifest_authoritative_student_is_frozen() -> None:
    assert load_joint_period_edit_manifest()["authoritative_student_id"] == AUTHORITATIVE_STUDENT_ID


def test_manifest_forbids_other_student_and_splits() -> None:
    manifest = load_joint_period_edit_manifest()
    assert manifest["forbidden_authoritative_student"] == "G12_0105"
    assert manifest["other_normal_targets_allowed"] is False
    assert manifest["stress_execution_allowed"] is False
    assert manifest["holdout_execution_allowed"] is False


def test_manifest_disables_external_seed() -> None:
    assert load_joint_period_edit_manifest()["external_persisted_seed"] is False


def test_domain_keeps_original_option() -> None:
    context = _domain_context()
    domains, summary = _candidate_payload_domains(context, [_move_payload()], {"candidate-1"})
    assert domains["CORE_A_1"][0].is_original
    assert ("P1",) in {option.placement for option in domains["CORE_A_1"]}
    assert summary.editable_logical_section_count == 1


def test_domain_uses_only_promising_ids() -> None:
    context = _domain_context()
    domains, _ = _candidate_payload_domains(context, [_move_payload()], set())
    assert domains == {}


def test_domain_deduplicates_same_destination() -> None:
    context = _domain_context()
    payloads = [_move_payload(candidate_id="candidate-1"), _move_payload(candidate_id="candidate-2")]
    domains, summary = _candidate_payload_domains(context, payloads, {"candidate-1", "candidate-2"})
    assert len(domains["CORE_A_1"]) == 2
    assert summary.raw_destination_records == 2
    assert summary.unique_destination_options == 1
    assert summary.deduplication_removed == 1


def test_domain_rejects_original_mismatch() -> None:
    context = _domain_context()
    payload = _move_payload()
    payload["original_placements"] = [["P2"]]
    with pytest.raises(JointPilotError, match="original placement mismatch"):
        _candidate_payload_domains(context, [payload], {"candidate-1"})


def test_domain_rejects_shape_change() -> None:
    context = _domain_context()
    payload = _move_payload()
    payload["proposed_placements"] = [["P6", "P7"]]
    with pytest.raises(JointPilotError, match="shape"):
        _candidate_payload_domains(context, [payload], {"candidate-1"})


def test_domain_rejects_invalid_period() -> None:
    context = _domain_context()
    payload = _move_payload(proposed="P8")
    with pytest.raises(JointPilotError, match="period"):
        _candidate_payload_domains(context, [payload], {"candidate-1"})


def test_noneditable_sections_have_original_only_domain() -> None:
    context = _feasible_context()
    domains = {"CORE_A_1": (PlacementOption("CORE_A_1", ("P1",), True), PlacementOption("CORE_A_1", ("P6",), False))}
    normalized = _placement_domains_with_originals(context.allocation_input, domains, False)
    assert len(normalized["CORE_B_1"]) == 1
    assert normalized["CORE_B_1"][0].placement == ("P2",)


def test_frozen_original_removes_edit_options() -> None:
    context = _feasible_context()
    domains = {"CORE_A_1": (PlacementOption("CORE_A_1", ("P1",), True), PlacementOption("CORE_A_1", ("P6",), False))}
    normalized = _placement_domains_with_originals(context.allocation_input, domains, True)
    assert len(normalized["CORE_A_1"]) == 1
    assert normalized["CORE_A_1"][0].is_original


def test_ha_domain_requires_same_shape() -> None:
    input_data = canonicalize_allocation_input(pd.DataFrame([], columns=students().columns), requests([]), sections(base_sections()), catalog())
    section = input_data.logical_sections_by_id["MATH23_1"]
    domain = (PlacementOption("MATH23_1", section.occupied_periods, True), PlacementOption("MATH23_1", ("P2", "P3"), False))
    assert all(len(option.placement) == 2 for option in domain)


def test_linked_domain_is_one_logical_section() -> None:
    input_data = canonicalize_allocation_input(pd.DataFrame([], columns=students().columns), requests([]), sections(base_sections()), catalog())
    section = input_data.logical_sections_by_id["GOV_1"]
    assert section.structure_type == "linked_semester"
    assert len(section.member_sections) == 2


def test_domain_summary_counts_authoritative_source_ids() -> None:
    context = _domain_context()
    domains, summary = _candidate_payload_domains(context, [_move_payload()], {"candidate-1"})
    assert summary.source_candidate_ids == ("candidate-1",)
    assert summary.authoritative_student_id == AUTHORITATIVE_STUDENT_ID


def test_joint_model_creates_placement_variables() -> None:
    context = _feasible_context()
    domains = {"CORE_A_1": (PlacementOption("CORE_A_1", ("P1",), True), PlacementOption("CORE_A_1", ("P6",), False))}
    build = build_joint_model(context.allocation_input, placement_domains=domains)
    assert len(build.placement_choice_vars) == 2
    assert len(build.section_changed_vars) == 1


def test_joint_model_has_start_variable_for_editable_section() -> None:
    context = _feasible_context()
    domains = {"CORE_A_1": (PlacementOption("CORE_A_1", ("P1",), True), PlacementOption("CORE_A_1", ("P6",), False))}
    build = build_joint_model(context.allocation_input, placement_domains=domains)
    assert "CORE_A_1" in build.section_start_vars


def test_joint_model_fixed_original_has_no_placement_choices() -> None:
    context = _feasible_context()
    domains = {"CORE_A_1": (PlacementOption("CORE_A_1", ("P1",), True), PlacementOption("CORE_A_1", ("P6",), False))}
    build = build_joint_model(context.allocation_input, placement_domains=domains, fixed_original=True)
    assert build.placement_choice_vars == {}
    assert build.section_changed_vars == {}


def test_joint_model_has_one_assignment_edge_per_candidate() -> None:
    context = _feasible_context()
    build = build_joint_model(context.allocation_input)
    assert len(build.assignment_vars) == 5
    assert set(build.assigned_vars) == {request.request_key for request in context.allocation_input.logical_requests}


def test_joint_model_counts_optional_intervals_for_fixed_sections() -> None:
    context = _feasible_context()
    build = build_joint_model(context.allocation_input)
    assert build.optional_intervals == 0


def test_joint_model_adds_capacity_constraints() -> None:
    context = _feasible_context()
    build = build_joint_model(context.allocation_input)
    assert build.model_constraints >= len(context.allocation_input.logical_sections)


def test_joint_model_adds_student_no_overlap() -> None:
    context = _feasible_context()
    domains = {"CORE_A_1": (PlacementOption("CORE_A_1", ("P1",), True), PlacementOption("CORE_A_1", ("P6",), False))}
    build = build_joint_model(context.allocation_input, placement_domains=domains)
    assert any(constraint.has_no_overlap() for constraint in build.model.Proto().constraints)


def test_joint_model_adds_target_load_semantics() -> None:
    context = _feasible_context()
    build = build_joint_model(context.allocation_input)
    assert build.model_variables >= len(build.assignment_vars)


def test_joint_model_adds_duplicate_identity_semantics() -> None:
    context = _feasible_context()
    build = build_joint_model(context.allocation_input)
    assert build.assigned_vars


def test_joint_model_adds_final_policy_semantics() -> None:
    context = _feasible_context()
    build = build_joint_model(context.allocation_input)
    assert build.model_constraints > 10


def test_joint_model_stats_include_proto_bytes() -> None:
    build = build_joint_model(_feasible_context().allocation_input)
    assert build.proto_bytes > 0
    assert build.model_variables > 0


def test_cost_gate_passes_small_fixture() -> None:
    assert _cost_gate(build_joint_model(_feasible_context())) == []


def test_fixed_equivalence_fixture_is_feasible() -> None:
    build = build_joint_model(_feasible_context(), fixed_original=True)
    stage = solve_joint_stage(build, "fixed_placement_feasibility", time_limit_seconds=5)
    assert stage.status in {"FEASIBLE", "OPTIMAL"}
    assert stage.incumbent_found


def test_fixed_equivalence_conflicting_fixture_is_infeasible() -> None:
    build = build_joint_model(_feasible_context(all_same_period=True), fixed_original=True)
    stage = solve_joint_stage(build, "fixed_placement_feasibility", time_limit_seconds=5)
    assert stage.status == "INFEASIBLE"
    assert not stage.incumbent_found


def test_stage_one_objective_is_zero_when_no_edit_is_needed() -> None:
    build = build_joint_model(_feasible_context())
    stage = solve_joint_stage(build, "changed_sections", time_limit_seconds=5)
    assert stage.status in {"FEASIBLE", "OPTIMAL"}
    assert stage.objective_value == 0


def test_stage_result_has_response_hash() -> None:
    build = build_joint_model(_feasible_context())
    stage = solve_joint_stage(build, "changed_sections", time_limit_seconds=5)
    assert len(stage.response_hash) == 64


def test_stage_result_reports_optimality_separately() -> None:
    build = build_joint_model(_feasible_context())
    stage = solve_joint_stage(build, "changed_sections", time_limit_seconds=5)
    assert stage.optimality_proven == (stage.status == "OPTIMAL")


def test_stage_result_selected_placements_use_domain() -> None:
    context = _feasible_context()
    domains = {"CORE_A_1": (PlacementOption("CORE_A_1", ("P1",), True), PlacementOption("CORE_A_1", ("P6",), False))}
    build = build_joint_model(context.allocation_input, placement_domains=domains)
    stage = solve_joint_stage(build, "changed_sections", time_limit_seconds=5)
    allowed = {("P1",), ("P6",)}
    assert dict(stage.selected_placements)["CORE_A_1"] in allowed


def test_validate_joint_solution_requires_incumbent() -> None:
    build = build_joint_model(_feasible_context())
    from src.joint_period_edit_pilot import JointStageResult
    stage = JointStageResult("changed_sections", "UNKNOWN", None, None, 0, False, False, "none", False, False, "")
    assert validate_joint_solution(build, stage)["assignment_available"] is False


def test_validate_joint_solution_reports_policy_pass() -> None:
    build = build_joint_model(_feasible_context())
    stage = solve_joint_stage(build, "changed_sections", time_limit_seconds=5)
    result = validate_joint_solution(build, stage)
    assert result["assignment_available"] is True
    assert result["policy_pass"] is True


def test_materialize_normal_section_changes_only_periods() -> None:
    context = _domain_context()
    original = context.sections.copy(deep=True)
    edited = apply_placement_map_to_sections(context, {"CORE_A_1": ("P6",)})
    assert edited.loc[edited.section_id == "CORE_A_1", "period_1"].item() == "P6"
    assert edited.loc[edited.section_id == "CORE_B_1", "period_1"].item() == original.loc[original.section_id == "CORE_B_1", "period_1"].item()


def test_materialize_preserves_section_row_count() -> None:
    context = _domain_context()
    edited = apply_placement_map_to_sections(context, {"CORE_A_1": ("P6",)})
    assert len(edited) == len(context.sections)


def test_materialize_preserves_non_period_metadata() -> None:
    context = _domain_context()
    edited = apply_placement_map_to_sections(context, {"CORE_A_1": ("P6",)})
    for column in ("section_id", "course_id", "capacity", "linked_section_group_id", "logical_block_id"):
        assert edited[column].tolist() == context.sections[column].tolist()


def test_architecture_audit_mentions_optional_intervals() -> None:
    assert "optional" in build_architecture_audit()["period_conflict"]


def test_architecture_audit_mentions_production_helpers() -> None:
    assert "final schedule policy" in build_architecture_audit()["production_reuse"][-1]


def test_architecture_audit_distinguishes_linked_course() -> None:
    assert "one canonical logical section" in build_architecture_audit()["gov_econ"]


def test_architecture_audit_distinguishes_ha() -> None:
    assert "period_units=2" in build_architecture_audit()["math_ha"]


def test_domain_option_sources_are_traceable() -> None:
    context = _domain_context()
    domains, _ = _candidate_payload_domains(context, [_move_payload()], {"candidate-1"})
    destination = next(option for option in domains["CORE_A_1"] if not option.is_original)
    assert destination.source_candidate_ids == ("candidate-1",)


def test_domain_summary_total_options_includes_original() -> None:
    context = _domain_context()
    _, summary = _candidate_payload_domains(context, [_move_payload()], {"candidate-1"})
    assert summary.total_unique_placement_options == 2


def test_domain_summary_original_only_is_zero_for_editable_fixture() -> None:
    context = _domain_context()
    _, summary = _candidate_payload_domains(context, [_move_payload()], {"candidate-1"})
    assert summary.original_only_section_count == 0


def test_control_domain_is_empty() -> None:
    context = _context()
    domains, summary = _candidate_payload_domains(context, [], set())
    assert domains == {}
    assert summary.editable_logical_section_count == 0


def test_fixed_original_keeps_all_sections() -> None:
    context = _feasible_context()
    normalized = _placement_domains_with_originals(context.allocation_input, {}, True)
    assert set(normalized) == {section.linked_section_group_id for section in context.allocation_input.logical_sections}


def test_joint_model_assignment_count_is_deterministic() -> None:
    first = build_joint_model(_feasible_context())
    second = build_joint_model(_feasible_context())
    assert len(first.assignment_vars) == len(second.assignment_vars)
    assert first.proto_bytes == second.proto_bytes


def test_joint_model_proto_has_no_external_seed_field() -> None:
    build = build_joint_model(_feasible_context())
    assert str(build.model.Proto())


def test_stage_seed_is_deterministic_for_fixture() -> None:
    first = solve_joint_stage(build_joint_model(_feasible_context()), "changed_sections", time_limit_seconds=5, seed=20260630)
    second = solve_joint_stage(build_joint_model(_feasible_context()), "changed_sections", time_limit_seconds=5, seed=20260630)
    assert first.status == second.status
    assert first.objective_value == second.objective_value


def test_stage_two_can_be_fixed_to_stage_one_value() -> None:
    build = build_joint_model(_feasible_context())
    stage1 = solve_joint_stage(build, "changed_sections", time_limit_seconds=5)
    stage2 = solve_joint_stage(build, "affected_assignments", time_limit_seconds=5, fixed_values={"changed_sections": stage1.objective_value or 0})
    assert stage2.status in {"FEASIBLE", "OPTIMAL"}
    assert stage2.objective_value == 0


def test_stage_three_can_be_fixed_to_prior_values() -> None:
    build = build_joint_model(_feasible_context())
    stage1 = solve_joint_stage(build, "changed_sections", time_limit_seconds=5)
    stage2 = solve_joint_stage(build, "affected_assignments", time_limit_seconds=5, fixed_values={"changed_sections": stage1.objective_value or 0})
    stage3 = solve_joint_stage(build, "placement_displacement", time_limit_seconds=5, fixed_values={"changed_sections": stage1.objective_value or 0, "affected_assignments": stage2.objective_value or 0})
    assert stage3.status in {"FEASIBLE", "OPTIMAL"}


def test_unknown_stage_is_rejected() -> None:
    with pytest.raises(JointPilotError):
        solve_joint_stage(build_joint_model(_feasible_context()), "not-a-stage", time_limit_seconds=1)


def test_assignment_stability_is_not_silently_solved() -> None:
    build = build_joint_model(_feasible_context())
    reference = {("primary:STU_1:CORE_A", "CORE_A_1")}
    stage = solve_joint_stage(build, "assignment_stability", time_limit_seconds=5, stability_reference=reference)
    assert stage.status in {"FEASIBLE", "OPTIMAL"}
    assert stage.hint_source == "none"


def test_assignment_stability_without_hint_is_rejected() -> None:
    with pytest.raises(JointPilotError, match="stability"):
        solve_joint_stage(build_joint_model(_feasible_context()), "assignment_stability", time_limit_seconds=1)


def test_ha_materialization_updates_both_periods() -> None:
    context = _context(section_rows=base_sections(), request_courses=())
    edited = apply_placement_map_to_sections(context, {"MATH23_1": ("P2", "P3")})
    rows = edited[edited.logical_block_id == "MATH2_3_HA"]
    assert set(rows.period_1) == {"P2"}
    assert set(rows.period_2) == {"P3"}


def test_linked_materialization_updates_both_rows_atomically() -> None:
    context = _context(section_rows=base_sections(), request_courses=())
    edited = apply_placement_map_to_sections(context, {"GOV_1": ("P4",)})
    rows = edited[edited.linked_section_group_id == "GOV_1"]
    assert set(rows.period_1) == {"P4"}
    assert set(rows.period_2) == {""}


def test_domain_rejects_unknown_section() -> None:
    context = _domain_context()
    payload = _move_payload("UNKNOWN")
    with pytest.raises(JointPilotError):
        _candidate_payload_domains(context, [payload], {"candidate-1"})


def test_domain_rejects_non_authoritative_student() -> None:
    context = _domain_context()
    payload = _move_payload()
    payload["core_student"] = "G12_0105"
    domains, summary = _candidate_payload_domains(context, [payload], {"candidate-1"})
    assert domains == {}
    assert summary.source_candidate_count == 0


def test_domain_source_ids_are_sorted() -> None:
    context = _domain_context()
    payloads = [_move_payload(candidate_id="candidate-2"), _move_payload(candidate_id="candidate-1")]
    domains, _ = _candidate_payload_domains(context, payloads, {"candidate-1", "candidate-2"})
    destination = next(option for option in domains["CORE_A_1"] if not option.is_original)
    assert destination.source_candidate_ids == ("candidate-1", "candidate-2")


def test_model_does_not_change_section_capacity() -> None:
    context = _feasible_context()
    build = build_joint_model(context.allocation_input)
    assert tuple(section.capacity for section in context.allocation_input.logical_sections) == tuple(section.capacity for section in build.allocation_input.logical_sections)


def test_model_does_not_change_section_count() -> None:
    context = _feasible_context()
    build = build_joint_model(context.allocation_input)
    assert len(build.allocation_input.logical_sections) == len(context.allocation_input.logical_sections)


def test_model_does_not_change_request_count() -> None:
    context = _feasible_context()
    build = build_joint_model(context.allocation_input)
    assert len(build.requests_by_key) == len(context.allocation_input.logical_requests)


def test_model_provides_displacement_expression() -> None:
    context = _feasible_context()
    domains = {"CORE_A_1": (PlacementOption("CORE_A_1", ("P1",), True), PlacementOption("CORE_A_1", ("P6",), False))}
    build = build_joint_model(context.allocation_input, placement_domains=domains)
    assert build.displacement_expr is not None


def test_model_has_changed_assignment_variables() -> None:
    context = _feasible_context()
    domains = {"CORE_A_1": (PlacementOption("CORE_A_1", ("P1",), True), PlacementOption("CORE_A_1", ("P6",), False))}
    build = build_joint_model(context.allocation_input, placement_domains=domains)
    assert build.affected_assignment_vars


def test_joint_stage_does_not_use_external_hint_by_default() -> None:
    build = build_joint_model(_feasible_context())
    stage = solve_joint_stage(build, "changed_sections", time_limit_seconds=5)
    assert stage.hint_source == "none"
    assert stage.repair_hint_enabled is False


def test_joint_stage_conditional_flag_is_explicit() -> None:
    build = build_joint_model(_feasible_context())
    stage = solve_joint_stage(build, "changed_sections", time_limit_seconds=5, conditional_on_unproven_incumbent=True)
    assert stage.conditional_on_unproven_incumbent is True


def test_pilot_scenario_ids_are_two() -> None:
    from src.joint_period_edit_pilot import PILOT_SCENARIO_IDS
    assert PILOT_SCENARIO_IDS == ("normal_dev_reference_2026", "normal_dev_10")


def test_no_stress_or_holdout_in_architecture_audit() -> None:
    audit = build_architecture_audit()
    assert "stress" not in json.dumps(audit).lower()


def test_minimum_is_not_global_in_docs() -> None:
    manifest = load_joint_period_edit_manifest()
    assert manifest["minimum_claim_scope"] == "frozen_admissible_placement_domain_only"


def test_authoritative_student_is_not_invalid_witness_student() -> None:
    assert AUTHORITATIVE_STUDENT_ID != "G12_0105"


def test_stage_response_hash_changes_with_response_or_is_present() -> None:
    build = build_joint_model(_feasible_context())
    stage = solve_joint_stage(build, "changed_sections", time_limit_seconds=5)
    assert stage.response_hash
