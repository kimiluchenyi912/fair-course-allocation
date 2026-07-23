from __future__ import annotations

from pathlib import Path

import pytest
from ortools.sat.python import cp_model

from src.joint_period_edit_pilot import PlacementOption, build_joint_model
from src.joint_stage1_model_size_reduction_audit import (
    ModelSizeAuditError,
    baseline_size_decomposition,
    cost_gate,
    enumerate_feasible_set,
    feasible_set_hash,
    load_model_size_audit_manifest,
    model_size,
    reduction_row,
    structural_invariance,
    _proto_sizes,
)
from tests.test_joint_period_edit_pilot import _feasible_context
from tests.test_cp_sat_solver import base_sections, catalog, request_row, requests, sections, students
from src.allocation import canonicalize_allocation_input


def _editable_builds():
    context = _feasible_context()
    domains = {
        "CORE_A_1": (
            PlacementOption("CORE_A_1", ("P1",), True),
            PlacementOption("CORE_A_1", ("P6",), False),
        ),
        "CORE_B_1": (
            PlacementOption("CORE_B_1", ("P2",), True),
            PlacementOption("CORE_B_1", ("P6",), False),
        ),
    }
    baseline = build_joint_model(context.allocation_input, placement_domains=domains)
    hybrid = build_joint_model(context.allocation_input, placement_domains=domains, occupancy_mode="hybrid_sparse_linear_occupancy")
    return baseline, hybrid


def test_manifest_is_frozen_to_control_and_one_target():
    manifest = load_model_size_audit_manifest()
    assert manifest["control_scenario_id"] == "normal_dev_reference_2026"
    assert manifest["size_target_scenario_id"] == "normal_dev_10"
    assert manifest["target_stage1_solve_allowed"] is False
    assert manifest["other_normal_targets_allowed"] is False
    assert manifest["stress_execution_allowed"] is False
    assert manifest["negative_execution_allowed"] is False
    assert manifest["holdout_execution_allowed"] is False
    assert manifest["frozen_editable_section_count"] == 312
    assert manifest["frozen_placement_option_count"] == 841


def test_unknown_formulation_fails_closed():
    with pytest.raises(Exception, match="unknown occupancy mode"):
        build_joint_model(_feasible_context().allocation_input, occupancy_mode="not-a-mode")


def test_hybrid_has_no_intervals_or_no_overlap():
    _, hybrid = _editable_builds()
    metrics = model_size(hybrid)
    assert hybrid.optional_intervals == 0
    assert metrics["interval_variables"] == 0
    assert metrics["no_overlap_constraints"] == 0
    assert metrics["q_occupancy_variables"] > 0
    assert metrics["w_conjunction_variables"] > 0


def test_assignment_universe_and_structural_hashes_are_equal():
    baseline, hybrid = _editable_builds()
    result = structural_invariance(baseline, hybrid)
    assert result["pass"] is True
    assert result["unexpected_mismatch"] is False
    assert result["candidate_edge_count"]["baseline"] == result["candidate_edge_count"]["hybrid"]


def test_q_is_exact_sum_of_occupying_placement_choices():
    _, hybrid = _editable_builds()
    proto = hybrid.model.Proto()
    q = hybrid.occupancy_vars[("CORE_A_1", "P1")]
    q_index = q.Index()
    matching = [constraint for constraint in proto.constraints if q_index in constraint.linear.vars]
    assert matching
    assert any(list(constraint.linear.vars).count(q_index) == 1 for constraint in matching)


def test_w_has_complete_three_constraint_linearization():
    _, hybrid = _editable_builds()
    proto = hybrid.model.Proto()
    w = next(iter(hybrid.occupancy_conjunction_vars.values()))
    w_index = w.Index()
    constraints = [constraint for constraint in proto.constraints if w_index in constraint.linear.vars]
    # w <= x, w <= q, w >= x + q - 1
    assert len(constraints) >= 3


def test_constant_zero_and_one_occupancy_are_sparse():
    _, hybrid = _editable_builds()
    metadata = hybrid.occupancy_metadata
    assert metadata["q_constant_zero_omitted"] > 0
    assert metadata["q_constant_one_omitted"] > 0
    assert metadata["q_without_consumers_omitted"] == 0


def test_single_contribution_period_omits_redundant_constraint():
    _, hybrid = _editable_builds()
    assert hybrid.occupancy_metadata["student_period_single_contribution_omitted"] > 0


def test_complete_feasible_projection_matches_baseline_and_hybrid():
    baseline, hybrid = _editable_builds()
    baseline_set = enumerate_feasible_set(baseline)
    hybrid_set = enumerate_feasible_set(hybrid)
    assert baseline_set == hybrid_set
    assert feasible_set_hash(baseline_set) == feasible_set_hash(hybrid_set)


def test_fixed_sections_use_linear_occupancy_without_q_or_w():
    context = _feasible_context()
    build = build_joint_model(context.allocation_input, fixed_original=True, occupancy_mode="hybrid_sparse_linear_occupancy")
    assert build.occupancy_vars == {}
    assert build.occupancy_conjunction_vars == {}
    assert build.optional_intervals == 0
    assert model_size(build)["no_overlap_constraints"] == 0


def test_linked_and_double_period_occupancy_are_atomic_and_exact():
    raw_students = students([("STU_GOV", 12, 1, False), ("STU_MATH", 12, 2, False)])
    raw_requests = requests([
        request_row("STU_GOV", "GOV_ECON_REG", "primary", "", "gov_econ_block", "GOV_ECON_REG"),
        request_row("STU_GOV", "GOV_ECON_REG", "primary", "", "gov_econ_block", "GOV_ECON_REG"),
        request_row("STU_MATH", "MATH2_3_HA"),
    ])
    input_data = canonicalize_allocation_input(raw_students, raw_requests, sections(base_sections()), catalog())
    domains = {
        "GOV_1": (PlacementOption("GOV_1", ("P7",), True), PlacementOption("GOV_1", ("P1",), False)),
        "MATH23_1": (PlacementOption("MATH23_1", ("P5", "P6"), True), PlacementOption("MATH23_1", ("P2", "P3"), False)),
    }
    baseline = build_joint_model(input_data, placement_domains=domains)
    hybrid = build_joint_model(input_data, placement_domains=domains, occupancy_mode="hybrid_sparse_linear_occupancy")
    assert baseline.placement_domains["GOV_1"][0].placement == ("P7",)
    assert hybrid.occupancy_vars["MATH23_1", "P2"] is not None
    assert hybrid.occupancy_vars["MATH23_1", "P3"] is not None
    assert enumerate_feasible_set(baseline) == enumerate_feasible_set(hybrid)


def test_size_decomposition_closes_interval_families():
    baseline, _ = _editable_builds()
    report = baseline_size_decomposition(baseline)
    assert sum(report["interval_families"].values()) == report["interval_variables"]
    assert report["total_constraints"] >= report["no_overlap_constraints"]


def test_cost_gate_uses_frozen_250mb_limit():
    assert "serialized_model_proto_exceeds_limit" in cost_gate({"total_variables": 0, "optional_intervals": 0, "serialized_binary_proto_bytes": 250000001, "build_time_seconds": 0})
    assert cost_gate({"total_variables": 0, "optional_intervals": 0, "serialized_binary_proto_bytes": 250000000, "build_time_seconds": 0}) == []


def test_proto_measurement_uses_binary_export_not_text_repr():
    _, hybrid = _editable_builds()
    sizes = _proto_sizes(hybrid.model)
    assert sizes["serialized_binary_proto_bytes"] > 0
    assert sizes["exported_binary_proto_file_bytes"] > 0
    assert sizes["binary_measurements_equal"] is True
    assert sizes["proto_text_bytes"] > 0
    metrics = model_size(hybrid)
    assert metrics["serialized_binary_proto_bytes"] == sizes["serialized_binary_proto_bytes"]
    assert metrics["proto_text_bytes"] == sizes["proto_text_bytes"]


def test_binary_and_text_fields_are_not_mixed_in_the_cost_gate():
    _, hybrid = _editable_builds()
    metrics = model_size(hybrid)
    assert metrics["cost_gate_measurement_field"] == "serialized_binary_proto_bytes"
    assert metrics["serialized_binary_proto_bytes"] != metrics["proto_text_bytes"]


def test_variable_family_counts_close():
    _, hybrid = _editable_builds()
    metrics = model_size(hybrid)
    assert sum(metrics["variable_families"].values()) == metrics["total_variables"]


def test_constraint_family_counts_close():
    baseline, hybrid = _editable_builds()
    for build in (baseline, hybrid):
        metrics = baseline_size_decomposition(build)
        assert sum(metrics["constraint_families"].values()) == metrics["total_constraints"]


def test_reduction_accounting_is_deterministic():
    row = reduction_row({"total_variables": 100, "optional_intervals": 40, "total_constraints": 80, "serialized_binary_proto_bytes": 1000, "build_time_seconds": 10}, {"total_variables": 70, "optional_intervals": 0, "total_constraints": 50, "serialized_binary_proto_bytes": 600, "build_time_seconds": 5})
    assert row["total_variables"]["reduction"] == 30
    assert row["optional_intervals"]["reduction_percent"] == 100


def test_hybrid_mode_is_explicit_not_default():
    context = _feasible_context()
    default = build_joint_model(context.allocation_input, use_optional_intervals_for_fixed=True)
    assert default.occupancy_mode == "full_optional_intervals"
    assert default.optional_intervals > 0


def test_no_target_solver_api_is_exposed_by_audit_module():
    from src import joint_stage1_model_size_reduction_audit as audit
    assert not hasattr(audit, "solve_target_stage1")
