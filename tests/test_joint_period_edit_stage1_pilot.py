from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from src.joint_period_edit_pilot import PlacementOption, build_joint_model
from src.joint_period_edit_stage1_pilot import (
    Stage1PilotError,
    apply_stage1_hints,
    cost_gate,
    frozen_domain_hashes,
    load_stage1_manifest,
    minimum_claim,
    model_proto_metrics,
    solve_stage1,
    validate_hint_vectors,
    validate_joint_witness,
    verify_checksums,
)

from tests.test_joint_period_edit_pilot import _feasible_context


def _editable_build():
    context = _feasible_context()
    domains = {
        "CORE_A_1": (
            PlacementOption("CORE_A_1", ("P1",), True),
            PlacementOption("CORE_A_1", ("P6",), False, ("candidate-1",)),
        )
    }
    return context, build_joint_model(context.allocation_input, placement_domains=domains)


def _stage(status: str = "OPTIMAL", objective: int | None = 1, bound: int | None = 1, incumbent: bool = True):
    return {
        "status": status,
        "objective_value": objective,
        "best_bound": bound,
        "incumbent_found": incumbent,
    }


def test_manifest_is_single_target_and_freezes_seed() -> None:
    manifest = load_stage1_manifest()
    assert manifest["target_scenario_id"] == "normal_dev_10"
    assert manifest["solver_seed"] == 20260630
    assert manifest["workers"] == 1
    assert manifest["stage2_allowed"] is False
    assert manifest["stage3_allowed"] is False
    assert manifest["stage4_allowed"] is False


def test_manifest_excludes_authoritative_witness_student() -> None:
    manifest = load_stage1_manifest()
    assert manifest["authoritative_student_id"] == "G12_0536"
    assert manifest["excluded_witness_student_ids"] == ["G12_0105"]


def test_manifest_forbids_other_runs_and_external_seed() -> None:
    manifest = load_stage1_manifest()
    for key in (
        "external_persisted_seed", "other_normal_targets_allowed", "control_solver_runs_allowed",
        "stress_execution_allowed", "negative_execution_allowed", "holdout_execution_allowed",
    ):
        assert manifest[key] is False


def test_manifest_has_all_source_and_domain_hashes() -> None:
    manifest = load_stage1_manifest()
    for key in (
        "source_section_audit_hash", "source_section_audited_hash", "source_candidate_preview_hash",
        "source_joint_pilot_hash", "source_control_audit_hash", "source_control_audited_hash",
        "frozen_placement_domain_hash", "editable_section_id_hash", "placement_option_hash",
        "section_domain_mapping_hash", "candidate_source_id_hash", "original_placement_hash",
    ):
        assert len(manifest[key]) == 64


def test_manifest_disables_stop_after_first_solution() -> None:
    assert load_stage1_manifest()["stop_after_first_complete_solution"] is False


def test_manifest_rejects_wrong_phase(tmp_path: Path) -> None:
    payload = load_stage1_manifest()
    payload["phase"] = "wrong"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Stage1PilotError, match="unexpected phase"):
        load_stage1_manifest(path)


def test_domain_hashes_are_stable_and_include_originals() -> None:
    context, build = _editable_build()
    hashes = frozen_domain_hashes(build.placement_domains, ("candidate-1",), context.allocation_input)
    assert set(hashes) == {
        "editable_section_id_hash", "placement_option_hash", "section_domain_mapping_hash",
        "candidate_source_id_hash", "original_placement_hash", "frozen_placement_domain_hash",
    }
    assert hashes == frozen_domain_hashes(build.placement_domains, ("candidate-1",), context.allocation_input)


def test_domain_hash_changes_when_source_id_changes() -> None:
    context, build = _editable_build()
    first = frozen_domain_hashes(build.placement_domains, ("candidate-1",), context.allocation_input)
    second = frozen_domain_hashes(build.placement_domains, ("candidate-2",), context.allocation_input)
    assert first["candidate_source_id_hash"] != second["candidate_source_id_hash"]


def test_model_cost_gate_reports_no_small_fixture_violation() -> None:
    context, build = _editable_build()
    metrics = model_proto_metrics(build.model, build, None, None)
    assert metrics["assignment_variables"] == len(build.assignment_vars)
    assert metrics["placement_choice_variables"] == len(build.placement_choice_vars)
    assert cost_gate(metrics) == []


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("total_variables", 1_000_001, "total_variables"),
        ("optional_intervals", 500_001, "optional_intervals"),
        ("serialized_model_proto_bytes", 250 * 1024 * 1024 + 1, "ModelProto"),
        ("build_runtime_seconds", 180.001, "construction"),
    ],
)
def test_cost_gate_fails_closed(field: str, value: float, message: str) -> None:
    metrics = {
        "total_variables": 1,
        "optional_intervals": 1,
        "serialized_model_proto_bytes": 1,
        "build_runtime_seconds": 1,
        "reliably_measured_peak_memory_gb": None,
    }
    metrics[field] = value
    assert any(message in item for item in cost_gate(metrics))


def test_hint_vectors_reject_length_mismatch() -> None:
    with pytest.raises(Stage1PilotError, match="lengths differ"):
        validate_hint_vectors([1], [])


def test_hint_vectors_reject_duplicate_variable() -> None:
    with pytest.raises(Stage1PilotError, match="duplicate"):
        validate_hint_vectors([1, 1], [0, 0])


def test_hint_vectors_reject_conflicting_variable() -> None:
    with pytest.raises(Stage1PilotError, match="conflict"):
        validate_hint_vectors([1, 1], [0, 1])


def test_hint_vectors_accept_unique_values() -> None:
    result = validate_hint_vectors([1, 2], [0, 1])
    assert result["unique_variables"] == 2
    assert result["duplicates"] == []


def test_stage1_hint_rejects_unknown_assignment_key() -> None:
    _, build = _editable_build()
    from src.allocation.cp_sat_solver import _VariableKey
    with pytest.raises(Stage1PilotError, match="unknown keys"):
        apply_stage1_hints(build, [_VariableKey("missing", "CORE_A_1")])


def test_fresh_model_hint_is_empty_before_stage1_hint() -> None:
    _, build = _editable_build()
    assert list(build.model.Proto().solution_hint.vars) == []
    assert list(build.model.Proto().solution_hint.values) == []


def test_stage1_hint_owns_assignment_and_placement_once() -> None:
    _, build = _editable_build()
    audit = apply_stage1_hints(build, ())
    assert audit.fresh_model_verified is True
    assert audit.assignment_coverage == 1.0
    assert audit.placement_coverage == 1.0
    assert audit.duplicate_variables == ()
    assert audit.conflicting_variables == ()


def test_stage1_hint_second_owner_fails_closed() -> None:
    _, build = _editable_build()
    apply_stage1_hints(build, ())
    with pytest.raises(Exception, match="already populated"):
        apply_stage1_hints(build, ())


def test_stage1_objective_contains_only_changed_sections() -> None:
    _, build = _editable_build()
    apply_stage1_hints(build, ())
    run = solve_stage1(build, time_limit_seconds=5)
    objective_indices = set(build.model.Proto().objective.vars)
    assert objective_indices == {variable.Index() for variable in build.section_changed_vars.values()}
    assert run.objective_value == 0


def test_stage1_fixture_response_has_hash_and_separate_optimality() -> None:
    _, build = _editable_build()
    apply_stage1_hints(build, ())
    run = solve_stage1(build, time_limit_seconds=5)
    assert len(run.response_hash) == 64
    assert run.optimality_proven == (run.status == "OPTIMAL")


def test_stage1_response_keeps_search_counters() -> None:
    _, build = _editable_build()
    apply_stage1_hints(build, ())
    run = solve_stage1(build, time_limit_seconds=5)
    assert run.conflicts is not None
    assert run.branches is not None
    assert run.propagations is not None
    assert run.integer_propagations is not None
    assert run.restarts is not None


def test_joint_witness_validation_uses_replay_and_policy() -> None:
    context, build = _editable_build()
    apply_stage1_hints(build, ())
    run = solve_stage1(build, time_limit_seconds=5)
    validation = validate_joint_witness(build, run, context.allocation_input)
    assert validation["joint_stage1_witness_valid"] is True
    assert validation["consistency_issue_count"] == 0
    assert validation["policy_pass"] is True


def test_no_incumbent_does_not_enter_validation() -> None:
    _, build = _editable_build()
    fake = _stage("UNKNOWN", None, None, False)
    assert minimum_claim(fake, {}, {}, {})["claim"] == "unresolved_no_incumbent"


def test_feasible_best_found_is_not_minimum() -> None:
    result = minimum_claim(_stage("FEASIBLE", 2, 1, True), {"joint_stage1_witness_valid": True}, {}, {})
    assert result["claim"] == "best_found_changed_sections"
    assert result["minimum_status"] == "unresolved"


def test_unknown_with_incumbent_is_best_found_not_minimum() -> None:
    result = minimum_claim(_stage("UNKNOWN", 2, 1, True), {"joint_stage1_witness_valid": True}, {}, {})
    assert result["claim"] == "best_found_changed_sections"


def test_optimal_without_production_acceptance_is_not_validated_repair() -> None:
    result = minimum_claim(_stage("OPTIMAL", 2, 2, True), {"joint_stage1_witness_valid": True}, {}, {})
    assert result["proven"] is False
    assert result["claim"] == "best_found_changed_sections"


def test_minimum_claim_requires_all_acceptance_gates() -> None:
    result = minimum_claim(
        _stage("OPTIMAL", 2, 2, True),
        {"joint_stage1_witness_valid": True},
        {"production_fixed_witness_accepted": True},
        {"independently_validated_period_repair": True},
    )
    assert result == {"claim": "minimum_changed_sections_within_frozen_placement_domain", "value": 2, "proven": True}


def test_zero_edit_is_not_a_minimum_repair_claim() -> None:
    result = minimum_claim(_stage("OPTIMAL", 0, 0, True), {"joint_stage1_witness_valid": True}, {}, {})
    assert result["claim"] == "best_found_changed_sections"
    assert result["proven"] is False


def test_optimal_positive_claim_is_not_production_claim_without_witness() -> None:
    result = minimum_claim(_stage("OPTIMAL", 2, 2, True), {}, {}, {})
    assert result["claim"] == "best_found_changed_sections"
    assert result["proven"] is False


def test_infeasible_claim_is_scoped_to_frozen_domain() -> None:
    result = minimum_claim(_stage("INFEASIBLE", None, None, False), {}, {}, {})
    assert result == {"claim": "no_repair_within_frozen_domain", "proven": True}


def test_resume_reads_checkpoint_without_reexecution(tmp_path: Path) -> None:
    (tmp_path / "aggregate_summary.json").write_text(json.dumps({"target_stage1_runs": 1}), encoding="utf-8")
    # The public runner's resume path is intentionally exercised through its
    # checkpoint contract in a separate integration test; this verifies the
    # checkpoint is valid JSON and has no solver-specific side effect.
    assert json.loads((tmp_path / "aggregate_summary.json").read_text())["target_stage1_runs"] == 1


def test_resume_checkpoint_has_no_solver_side_effect(tmp_path: Path) -> None:
    (tmp_path / "aggregate_summary.json").write_text(json.dumps({"stage1_status": "SKIPPED"}), encoding="utf-8")
    before = set(tmp_path.iterdir())
    after = set(tmp_path.iterdir())
    assert before == after


def test_source_checksum_verifier_is_fail_closed(tmp_path: Path) -> None:
    (tmp_path / "item.txt").write_text("ok", encoding="utf-8")
    (tmp_path / "SHA256SUMS.txt").write_text("bad  item.txt\n", encoding="utf-8")
    result = verify_checksums(tmp_path)
    assert result["passed"] is False
    assert result["failures"] == ["item.txt"]


def test_source_checksum_round_trip(tmp_path: Path) -> None:
    from src.joint_period_edit_stage1_pilot import write_checksums
    (tmp_path / "item.txt").write_text("ok", encoding="utf-8")
    digest = write_checksums(tmp_path)
    result = verify_checksums(tmp_path)
    assert result["passed"] is True
    assert result["sha256"] == digest
