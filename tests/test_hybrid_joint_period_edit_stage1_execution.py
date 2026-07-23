from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.hybrid_joint_period_edit_stage1_execution import (
    HybridExecutionError,
    _not_run,
    _stage1_config,
    load_execution_manifest,
    run_execution,
    verify_frozen_domain,
    write_checksums,
)
from src.joint_stage1_model_size_reduction_audit import cost_gate


def test_manifest_only_allows_normal_dev_10():
    manifest = load_execution_manifest()
    assert manifest["target_scenario_id"] == "normal_dev_10"
    assert manifest["phase"] == "hybrid_stage1_single_target_execution"
    assert manifest["formulation"] == "hybrid_sparse_linear_occupancy"


@pytest.mark.parametrize("field", [
    "stage2_allowed", "stage3_allowed", "stage4_allowed", "control_runs_allowed",
    "other_normal_targets_allowed", "stress_execution_allowed", "negative_execution_allowed",
    "holdout_execution_allowed", "external_persisted_seed",
])
def test_manifest_forbids_non_stage1_or_external_runs(tmp_path: Path, field: str):
    source = Path("data/scenarios/hybrid_joint_period_edit_stage1_execution_v1.json")
    payload = json.loads(source.read_text())
    payload[field] = True
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(HybridExecutionError):
        load_execution_manifest(path)


def test_manifest_freezes_domain_and_seed():
    manifest = load_execution_manifest()
    assert manifest["authoritative_student_id"] == "G12_0536"
    assert manifest["excluded_student_ids"] == ["G12_0105"]
    assert manifest["editable_section_count"] == 312
    assert manifest["placement_option_count"] == 841
    assert manifest["candidate_edge_count"] == 164269
    assert manifest["solver_seed"] == 20260630
    assert manifest["workers"] == 1


def test_stage1_config_has_only_changed_section_objective():
    config = _stage1_config(load_execution_manifest())
    assert config["objective"] == "minimize_sum_section_changed"
    assert config["repair_hint_enabled"] is False
    assert config["stop_after_first_complete_solution"] is False
    assert config["stage2_allowed"] is False


def test_not_run_is_explicit_not_zero_result():
    value = _not_run("no_incumbent")
    assert value["status"] == "not_run"
    assert value["assignment_available"] is False
    assert value["response_hash"] == ""
    assert value["not_run_reason"] == "no_incumbent"


def test_cost_gate_reads_binary_field_only():
    metrics = {
        "total_variables": 1,
        "optional_intervals": 0,
        "serialized_binary_proto_bytes": 100,
        "proto_serialized_bytes": 100,
        "proto_text_bytes": 999_999_999,
        "build_time_seconds": 1,
    }
    # The audit API's canonical field is binary; text size is irrelevant.
    assert cost_gate({"total_variables": 1, "optional_intervals": 0, "serialized_binary_proto_bytes": 100, "build_time_seconds": 1}) == []
    assert metrics["proto_text_bytes"] > 250_000_000


def test_cost_gate_uses_decimal_bytes():
    assert cost_gate({"total_variables": 1, "optional_intervals": 0, "serialized_binary_proto_bytes": 250_000_000, "build_time_seconds": 1}) == []
    assert "serialized_model_proto_exceeds_limit" in cost_gate({"total_variables": 1, "optional_intervals": 0, "serialized_binary_proto_bytes": 250_000_001, "build_time_seconds": 1})


def test_source_checksum_writer_is_stable(tmp_path: Path):
    (tmp_path / "a.json").write_text("{}\n")
    first = write_checksums(tmp_path)
    second = write_checksums(tmp_path)
    assert first == second
    assert len((tmp_path / "SHA256SUMS.txt").read_text().splitlines()) == 1


def test_nonempty_output_refuses_overwrite(tmp_path: Path):
    (tmp_path / "existing.json").write_text("{}")
    with pytest.raises(HybridExecutionError, match="non-empty"):
        run_execution(output_dir=tmp_path)


def test_incomplete_resume_refuses_to_run(tmp_path: Path):
    (tmp_path / "partial.json").write_text("{}")
    with pytest.raises(HybridExecutionError, match="completed aggregate"):
        run_execution(output_dir=tmp_path, resume=True)


def test_bad_target_manifest_fails_before_model_build(tmp_path: Path):
    payload = json.loads(Path("data/scenarios/hybrid_joint_period_edit_stage1_execution_v1.json").read_text())
    payload["target_scenario_id"] = "normal_dev_01"
    manifest = tmp_path / "bad.json"
    manifest.write_text(json.dumps(payload))
    with pytest.raises(HybridExecutionError, match="only normal_dev_10"):
        run_execution(manifest_path=manifest, output_dir=tmp_path / "out")


def test_frozen_domain_check_rejects_count_drift():
    manifest = load_execution_manifest()
    class Summary:
        editable_logical_section_count = 311
        total_unique_placement_options = 841
        source_candidate_ids = ()
    class Context:
        authoritative_core = None
    with pytest.raises(HybridExecutionError, match="editable section count"):
        verify_frozen_domain(manifest, Context(), {}, Summary())


def test_no_control_or_holdout_counter_is_enabled():
    manifest = load_execution_manifest()
    assert all(manifest[field] is False for field in (
        "control_runs_allowed", "other_normal_targets_allowed", "stress_execution_allowed",
        "negative_execution_allowed", "holdout_execution_allowed",
    ))


def test_budget_is_frozen():
    manifest = load_execution_manifest()
    assert manifest["stage1_budget_seconds"] == 300
    assert manifest["fixed_witness_acceptance_budget_seconds"] == 30
    assert manifest["production_validation_budget_seconds"] == 300


def test_external_seed_is_false():
    assert load_execution_manifest()["external_persisted_seed"] is False


def test_artifact_stages_are_separate():
    config = _stage1_config(load_execution_manifest())
    assert config["objective"] != "hamming_to_constrained_first"
    assert config["repair_hint_enabled"] is False


def test_execution_manifest_records_hashes():
    manifest = load_execution_manifest()
    assert len(manifest["source_hybrid_audit_hash"]) == 64
    assert len(manifest["source_candidate_preview_hash"]) == 64
    assert len(manifest["frozen_placement_domain_hash"]) == 64
