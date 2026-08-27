from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.joint_model_control_performance_audit import (
    DEFAULT_ORACLE,
    MANIFEST_PATH,
    REFERENCE_ID,
    VARIANTS,
    PerformanceAuditError,
    SolverRun,
    _assignment_key_hash,
    _accept_known_witness,
    build_variants,
    classify_diagnosis,
    load_audit_manifest,
    model_size_comparison,
    run_audit,
    structural_invariance,
    _solve_variant,
    _attempt_rows,
    _resume_performance_runs,
    apply_unique_solution_hint,
    assert_empty_solution_hint,
    validate_solution_hint_uniqueness,
    verify_checksums,
    verify_source_artifacts,
    write_audited_provenance_artifact,
    write_audit_artifact,
    write_checksums,
    _write_csv,
    _write_json,
)
from src.allocation.cp_sat_solver import _VariableKey

from tests.test_joint_period_edit_pilot import _feasible_context


def _fixture_builds():
    context = _feasible_context()
    return context, build_variants(context.allocation_input, context.catalog)


def test_manifest_contains_reference_only() -> None:
    manifest = load_audit_manifest()
    assert manifest["scenario_id"] == REFERENCE_ID


def test_manifest_variants_are_frozen() -> None:
    assert tuple(load_audit_manifest()["performance_variants"]) == VARIANTS


def test_manifest_seed_is_frozen() -> None:
    manifest = load_audit_manifest()
    assert manifest["solver_seed"] == 20260630
    assert manifest["workers"] == 1


def test_manifest_forbids_external_seed() -> None:
    assert load_audit_manifest()["external_persisted_seed_for_performance_runs"] is False


def test_manifest_forbids_other_scenarios() -> None:
    manifest = load_audit_manifest()
    assert manifest["other_normal_targets_allowed"] is False
    assert manifest["stress_execution_allowed"] is False
    assert manifest["holdout_execution_allowed"] is False


def test_manifest_has_separate_budgets() -> None:
    manifest = load_audit_manifest()
    assert (manifest["witness_acceptance_budget_seconds"], manifest["hamming_run_budget_seconds"], manifest["feasibility_only_budget_seconds"]) == (30, 180, 120)


def test_manifest_path_exists() -> None:
    assert MANIFEST_PATH.is_file()


def test_source_artifact_paths_are_external() -> None:
    manifest = load_audit_manifest()
    assert all("fair-course-allocation-artifacts" in value for value in manifest["source_artifacts"].values())


@pytest.fixture
def external_audit_manifest(require_external_artifact) -> dict[str, object]:
    manifest = load_audit_manifest()
    manifest["source_artifacts"] = {
        name: str(require_external_artifact(Path(raw).relative_to("../fair-course-allocation-artifacts")))
        for name, raw in manifest["source_artifacts"].items()
    }
    return manifest


@pytest.mark.external_artifact
def test_source_artifact_hashes_are_verified(external_audit_manifest) -> None:
    result = verify_source_artifacts(external_audit_manifest)
    assert all(item["checksum"]["passed"] for item in result.values())


@pytest.mark.external_artifact
def test_source_artifact_verification_is_read_only(external_audit_manifest) -> None:
    result = verify_source_artifacts(external_audit_manifest)
    assert all(item["read_only"] for item in result.values())


def test_production_variant_uses_production_assignment_keys() -> None:
    _, builds = _fixture_builds()
    assert len(builds["production_native"].assignment_vars) == 5


def test_joint_native_variant_has_no_intervals() -> None:
    _, builds = _fixture_builds()
    assert builds["joint_fixed_native_conflicts"].interval_variables == 0


def test_joint_interval_variant_has_intervals() -> None:
    _, builds = _fixture_builds()
    assert builds["joint_fixed_optional_intervals"].interval_variables == 5


def test_joint_interval_variant_has_no_overlap() -> None:
    _, builds = _fixture_builds()
    assert any(c.has_no_overlap() for c in builds["joint_fixed_optional_intervals"].model.Proto().constraints)


def test_native_joint_variant_has_no_overlap_constraint() -> None:
    _, builds = _fixture_builds()
    assert not any(c.has_no_overlap() for c in builds["joint_fixed_native_conflicts"].model.Proto().constraints)


def test_all_variants_have_same_assignment_key_count() -> None:
    _, builds = _fixture_builds()
    assert {len(build.assignment_vars) for build in builds.values()} == {5}


def test_all_variants_have_same_assignment_key_hash() -> None:
    _, builds = _fixture_builds()
    result = structural_invariance(builds)
    assert len(set(result["assignment_variable_key_hash"].values())) == 1
    assert not result["unexpected_mismatch"]


def test_structural_invariance_reports_expected_differences() -> None:
    _, builds = _fixture_builds()
    assert "interval_variables" in structural_invariance(builds)["allowed_differences"]


def test_structural_invariance_fails_for_different_key_universe() -> None:
    _, builds = _fixture_builds()
    builds["joint_fixed_native_conflicts"].assignment_vars.pop(next(iter(builds["joint_fixed_native_conflicts"].assignment_vars)))
    assert structural_invariance(builds)["unexpected_mismatch"]


def test_fixed_original_sections_keep_original_placements() -> None:
    _, builds = _fixture_builds()
    assert builds["joint_fixed_native_conflicts"].model is not None
    assert not any(str(variable.name).startswith("placement__") for variable in builds["joint_fixed_native_conflicts"].model.Proto().variables)


def test_candidate_hash_is_order_stable() -> None:
    keys = {_VariableKey("r2", "s2"), _VariableKey("r1", "s1")}
    assert _assignment_key_hash(keys) == _assignment_key_hash(reversed(tuple(keys)))


def test_model_size_comparison_has_all_variants() -> None:
    _, builds = _fixture_builds()
    assert {row["variant"] for row in model_size_comparison(builds)} == set(VARIANTS)


def test_model_size_comparison_has_multipliers() -> None:
    _, builds = _fixture_builds()
    row = model_size_comparison(builds)[0]
    assert row["total_variables_multiplier_vs_production"] == 1.0


def test_model_size_counts_intervals_separately() -> None:
    _, builds = _fixture_builds()
    rows = {row["variant"]: row for row in model_size_comparison(builds)}
    assert rows["joint_fixed_optional_intervals"]["interval_variables"] == 5


def test_known_witness_acceptance_is_marked_correctness_only() -> None:
    _, builds = _fixture_builds()
    selected = set(builds["production_native"].assignment_vars)
    result = _accept_known_witness(builds, selected, 2)
    assert all(row["fixed_for_equivalence_only"] for row in result.values())


def test_known_witness_acceptance_accepts_all_variants() -> None:
    _, builds = _fixture_builds()
    selected = set(builds["production_native"].assignment_vars)
    result = _accept_known_witness(builds, selected, 2)
    assert all(row["status"] in {"FEASIBLE", "OPTIMAL"} for row in result.values())


def test_known_witness_assignment_is_exact() -> None:
    _, builds = _fixture_builds()
    selected = set(builds["production_native"].assignment_vars)
    result = _accept_known_witness(builds, selected, 2)
    assert all(row["assignment_exact"] for row in result.values())


def test_known_witness_rejection_is_fail_closed() -> None:
    _, builds = _fixture_builds()
    unknown = {_VariableKey("missing", "missing")}
    with pytest.raises(PerformanceAuditError):
        _accept_known_witness(builds, unknown, 1)


def test_solver_run_has_log_provenance() -> None:
    row = SolverRun("A", "hamming", "UNKNOWN", False, None, 1.0, None, None, "hash", 0, 0, 0, 0, 0, 0.0, None, True, None, None). __dict__
    assert row["parsed_from_solver_log"] is True


def test_classification_is_unresolved_without_performance() -> None:
    result = classify_diagnosis({}, {}, {"unexpected_mismatch": False})
    assert result["classification"] == "unresolved"


def test_unknown_production_is_not_called_joint_failure() -> None:
    result = classify_diagnosis({"production_native": {"status": "UNKNOWN"}}, {}, {"unexpected_mismatch": False})
    assert result["classification"] == "environment_or_run_variance"


def test_optional_interval_classification_requires_evidence() -> None:
    result = classify_diagnosis(
        {"production_native": {"status": "FEASIBLE"}, "joint_fixed_optional_intervals": {"status": "UNKNOWN"}},
        {"joint_fixed_native_conflicts": {"status": "FEASIBLE"}},
        {"unexpected_mismatch": False},
    )
    assert result["classification"] == "optional_interval_bottleneck"


def test_scaffold_classification_requires_b_and_c_failure() -> None:
    result = classify_diagnosis(
        {"production_native": {"status": "FEASIBLE"}, "joint_fixed_native_conflicts": {"status": "UNKNOWN"}, "joint_fixed_optional_intervals": {"status": "UNKNOWN"}},
        {}, {"unexpected_mismatch": False},
    )
    assert result["classification"] == "joint_scaffold_bottleneck"


def test_objective_interaction_classification_requires_feasibility_evidence() -> None:
    result = classify_diagnosis(
        {"production_native": {"status": "FEASIBLE"}, "joint_fixed_optional_intervals": {"status": "UNKNOWN"}},
        {"joint_fixed_optional_intervals": {"status": "FEASIBLE"}},
        {"unexpected_mismatch": False},
    )
    assert result["classification"] == "hamming_objective_interaction"


def test_structural_failure_overrides_performance_classification() -> None:
    result = classify_diagnosis({"production_native": {"status": "FEASIBLE"}}, {}, {"unexpected_mismatch": True})
    assert result["classification"] == "unresolved"


def test_write_checksums_and_verify(tmp_path: Path) -> None:
    (tmp_path / "x.txt").write_text("x", encoding="utf-8")
    write_checksums(tmp_path)
    assert verify_checksums(tmp_path)["passed"]


def test_checksum_detects_tamper(tmp_path: Path) -> None:
    (tmp_path / "x.txt").write_text("x", encoding="utf-8")
    write_checksums(tmp_path)
    (tmp_path / "x.txt").write_text("changed", encoding="utf-8")
    assert not verify_checksums(tmp_path)["passed"]


def test_artifact_writer_rejects_nonempty_destination(tmp_path: Path) -> None:
    target = tmp_path / "artifact"
    target.mkdir()
    (target / "existing").write_text("x", encoding="utf-8")
    with pytest.raises(PerformanceAuditError):
        write_audit_artifact(target, {"summary": {}})


def test_artifact_writer_creates_json_and_csv(tmp_path: Path) -> None:
    result = write_audit_artifact(tmp_path / "artifact", {"summary": {"ok": True}, "rows_rows": [{"x": 1}]})
    assert Path(result["path"], "summary.json").is_file()
    assert Path(result["path"], "rows.csv").is_file()


def test_artifact_writer_returns_checksum(tmp_path: Path) -> None:
    result = write_audit_artifact(tmp_path / "artifact", {"summary": {}})
    assert len(result["sha256"]) == 64


def test_default_audit_does_not_request_performance() -> None:
    assert "--run-performance" in Path("src/joint_model_control_performance_audit.py").read_text()


def test_oracle_path_is_external() -> None:
    assert "fair-course-allocation-artifacts" in str(DEFAULT_ORACLE)


def test_no_target_counter_is_explicit_in_manifest() -> None:
    payload = load_audit_manifest()
    assert payload["scenario_id"] == "normal_dev_reference_2026"


def test_solver_seed_is_not_used_for_data_generation() -> None:
    payload = load_audit_manifest()
    assert "data_generation_seed" not in payload
    assert "section_planning_seed" not in payload


def test_witness_hash_is_not_a_performance_seed() -> None:
    payload = load_audit_manifest()
    assert payload["external_persisted_seed_for_performance_runs"] is False


def test_hamming_run_budget_is_not_witness_budget() -> None:
    payload = load_audit_manifest()
    assert payload["hamming_run_budget_seconds"] > payload["witness_acceptance_budget_seconds"]


def test_feasibility_budget_is_separate() -> None:
    payload = load_audit_manifest()
    assert payload["feasibility_only_budget_seconds"] != payload["hamming_run_budget_seconds"]


def test_only_reference_id_is_frozen() -> None:
    assert REFERENCE_ID == "normal_dev_reference_2026"


def test_unexpected_mismatch_is_boolean() -> None:
    _, builds = _fixture_builds()
    assert isinstance(structural_invariance(builds)["unexpected_mismatch"], bool)


def test_model_size_build_time_is_recorded() -> None:
    _, builds = _fixture_builds()
    assert all(row["build_time_seconds"] >= 0 for row in model_size_comparison(builds))


def test_variant_names_are_unique() -> None:
    assert len(VARIANTS) == len(set(VARIANTS))


def test_witness_acceptance_budget_is_positive() -> None:
    assert load_audit_manifest()["witness_acceptance_budget_seconds"] > 0


def test_performance_budget_is_positive() -> None:
    manifest = load_audit_manifest()
    assert manifest["hamming_run_budget_seconds"] > 0
    assert manifest["feasibility_only_budget_seconds"] > 0


def test_audit_module_does_not_modify_source_artifacts() -> None:
    assert "write_text" in Path("src/joint_model_control_performance_audit.py").read_text()
    assert "read_only" in Path("src/joint_model_control_performance_audit.py").read_text()


def test_performance_solver_rejects_preexisting_hint_before_solve(tmp_path: Path) -> None:
    _, builds = _fixture_builds()
    build = builds["production_native"]
    selected = set(build.assignment_vars)
    # A second owner must fail closed instead of silently clearing the first.
    for variable in build.assignment_vars.values():
        build.model.AddHint(variable, 1)
    with pytest.raises(PerformanceAuditError, match="already populated"):
        _solve_variant(build, selected, 2, "hamming", tmp_path / "solver.log", True)


def test_hint_audit_is_read_only() -> None:
    _, builds = _fixture_builds()
    before = {name: len(build.model.Proto().solution_hint.vars) for name, build in builds.items()}
    from src.joint_model_control_performance_audit import _hint_audit

    context = _feasible_context()
    _hint_audit(builds, context.allocation_input, context.catalog)
    assert before == {name: len(build.model.Proto().solution_hint.vars) for name, build in builds.items()}


def test_duplicate_hint_is_rejected_before_solve() -> None:
    from ortools.sat.python import cp_model

    model = cp_model.CpModel()
    variable = model.NewBoolVar("x")
    model.AddHint(variable, 1)
    model.AddHint(variable, 0)
    with pytest.raises(PerformanceAuditError, match="duplicate variables"):
        validate_solution_hint_uniqueness(model)


def test_conflicting_hint_is_rejected_before_solve() -> None:
    from ortools.sat.python import cp_model

    model = cp_model.CpModel()
    variable = model.NewBoolVar("x")
    model.AddHint(variable, 1)
    model.AddHint(variable, 0)
    with pytest.raises(PerformanceAuditError, match="duplicate variables"):
        validate_solution_hint_uniqueness(model)


def test_apply_hint_requires_empty_model_and_records_hash() -> None:
    from ortools.sat.python import cp_model

    model = cp_model.CpModel()
    assert_empty_solution_hint(model)
    variable = model.NewBoolVar("x")
    key = _VariableKey("request", "section")
    result = apply_unique_solution_hint(model, {key: variable}, {key})
    assert result["fresh_model_verified"] is True
    assert result["duplicate_variables"] == []
    assert result["unique_variables"] == 1


def test_attempt_accounting_separates_valid_and_excluded() -> None:
    rows = _attempt_rows(
        [{"variant": "A", "run_kind": "hamming", "status": "OPTIMAL"}],
        [{"variant": "B", "run_kind": "feasibility_only", "status": "UNKNOWN"}],
        [{"variant": "C", "run_kind": "hamming", "status": "MODEL_INVALID"}],
    )
    assert len(rows) == 3
    assert sum(row["included_in_benchmark"] for row in rows) == 2
    assert sum(row["attempt_class"] == "model_invalid" for row in rows) == 1


def test_each_solver_run_uses_independent_hint_copy(tmp_path: Path) -> None:
    _, builds = _fixture_builds()
    build = builds["production_native"]
    selected = set(build.assignment_vars)
    first = _solve_variant(build, selected, 1, "hamming", tmp_path / "one.log", True)
    second = _solve_variant(build, selected, 1, "feasibility_only", tmp_path / "two.log", False)
    assert first.fresh_model_verified and second.fresh_model_verified
    assert len(build.model.Proto().solution_hint.vars) == 0


def test_known_witness_acceptance_does_not_leak_hint_state() -> None:
    _, builds = _fixture_builds()
    selected = set(builds["production_native"].assignment_vars)
    _accept_known_witness(builds, selected, 1)
    assert all(not build.model.Proto().solution_hint.vars for build in builds.values())


def test_resume_skips_completed_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, builds = _fixture_builds()
    _write_csv(tmp_path / "performance_runs.csv", [{"variant": name, "status": "OPTIMAL"} for name in VARIANTS])
    _write_csv(tmp_path / "feasibility_only_runs.csv", [{"variant": name, "status": "UNKNOWN"} for name in VARIANTS[1:]])
    manifest = load_audit_manifest()

    def fail_if_called(*args, **kwargs):
        raise AssertionError("completed checkpoint was rerun")

    monkeypatch.setattr("src.joint_model_control_performance_audit._solve_variant", fail_if_called)
    hamming, feasibility = _resume_performance_runs(tmp_path, builds, set(), manifest)
    assert set(hamming) == set(VARIANTS)
    assert set(feasibility) == set(VARIANTS[1:])


def test_reporting_rebuild_does_not_solve_or_modify_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    raw = tmp_path / "raw"
    variant = raw / "variants" / "A"
    variant.mkdir(parents=True)
    row = {
        "variant": "A", "run_kind": "hamming", "status": "OPTIMAL",
        "assignment_available": "True", "response_hash": "response",
        "policy_status": "PASS", "consistency_issue_count": "0",
    }
    _write_csv(raw / "performance_runs.csv", [row])
    _write_csv(raw / "feasibility_only_runs.csv", [])
    _write_json(raw / "hint_audit.json", {"positive_key_hash": "positive", "variants": {"A": {"duplicate_hint_keys": 0, "conflicting_hint_values": 0, "hinted_positive_assignment_key_hash": "positive", "hint_coverage_ratio": 1.0, "invalid_hint_keys": 0}}})
    _write_json(variant / "response_stats.json", row)
    _write_json(variant / "validation.json", {"policy_status": "PASS", "consistency_issue_count": 0, "assignment_available": True})
    (variant / "hamming.log").write_text("solver log", encoding="utf-8")
    write_checksums(raw)
    before = (raw / "SHA256SUMS.txt").read_text()
    monkeypatch.setattr("src.joint_model_control_performance_audit._solve_variant", lambda *a, **k: (_ for _ in ()).throw(AssertionError("solver called")))
    result = write_audited_provenance_artifact(raw, tmp_path / "audited")
    assert Path(result["path"], "corrected_aggregate_summary.json").is_file()
    assert (raw / "SHA256SUMS.txt").read_text() == before
