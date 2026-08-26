from __future__ import annotations

import json
import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import src.cp_sat_robustness_runner as runner_module
from src.allocation import (
    CpSatAllocationResult,
    CpSatModelScope,
    CpSatModelStats,
    CpSatObjectiveValues,
    CpSatSolveStatus,
    CpSatStageDiagnostic,
    CpSatStageName,
)
from src.cp_sat_robustness_runner import (
    CpSatEvaluationCorrectnessError,
    CpSatEvaluationError,
    CpSatRobustnessRunner,
    _aggregate,
    _audit_status_semantics,
    _audited_readiness,
    _paired_rows,
    _stage_trace,
    _validate_objective_bounds,
    _validate_result,
    audit_existing_artifact,
    evaluation_manifest_hash,
    load_evaluation_manifest,
)


MANIFEST = Path("data/scenarios/cp_sat_development_evaluation_v1.json")


def _historical_manifest_for_persisted_artifact(tmp_path: Path, source: Path) -> Path:
    """Recreate the path-only pre-portability manifest for frozen artifact tests."""
    portable_root = "../fair-course-allocation-artifacts"
    snapshot = json.loads((source / "evaluation_manifest_snapshot.json").read_text(encoding="utf-8"))
    historical_root = str(Path(snapshot["source_normal_suite"]["artifact_dir"]).parents[1])
    path = tmp_path / "historical_cp_sat_development_evaluation_v1.json"
    path.write_text(
        MANIFEST.read_text(encoding="utf-8").replace(portable_root, historical_root),
        encoding="utf-8",
    )
    return path


def _unknown_result() -> CpSatAllocationResult:
    diagnostic = CpSatStageDiagnostic(
        stage_name=CpSatStageName.FEASIBILITY_BOOTSTRAP,
        model_scope=CpSatModelScope.BOOTSTRAP,
        status=CpSatSolveStatus.UNKNOWN,
        objective_value=None,
        best_objective_bound=None,
        wall_time_seconds=0.25,
        conflicts=2,
        branches=3,
        optimum_proven=False,
        response_proto_hash="",
        objective_descriptor_hash="bootstrap-objective",
    )
    return CpSatAllocationResult(
        algorithm_name="fair_cp_sat",
        seed=20260630,
        solve_status=CpSatSolveStatus.UNKNOWN,
        lexicographic_optimality_proven=False,
        stage_diagnostics=(diagnostic,),
        objective_values=CpSatObjectiveValues(),
        model_stats=CpSatModelStats(
            total_variables=10,
            total_constraints=4,
            build_time_seconds=0.0,
            solve_time_seconds=0.0,
        ),
    )


def _fingerprint() -> dict[str, object]:
    return {
        "students": 1,
        "logical_requests": 1,
        "logical_primaries": 1,
        "alternates": 0,
        "logical_sections": 1,
        "section_rows": 1,
        "candidate_edges": 1,
        "canonical_input_hash": "a" * 64,
    }


def _fake_input() -> SimpleNamespace:
    return SimpleNamespace(students=(SimpleNamespace(grade=9, target_period_units=5),))


def _audit_scenario() -> object:
    return CpSatRobustnessRunner(MANIFEST).select("normal", max_scenarios=1)[0]


def _audit_trace(*stages: tuple[str, str, list[dict[str, object]] | None]) -> list[dict[str, object]]:
    return [
        {
            "stage_name": name,
            "status": status,
            "skipped": False,
            "fixed_prior_objectives": fixed or [],
            "incumbent_found": status in {"FEASIBLE", "OPTIMAL"},
            "response_proto_hash": "response" if status in {"FEASIBLE", "OPTIMAL", "UNKNOWN", "INFEASIBLE"} else "",
        }
        for name, status, fixed in stages
    ]


def test_manifest_is_frozen_development_only_with_stable_hash() -> None:
    payload = load_evaluation_manifest(MANIFEST)
    assert evaluation_manifest_hash(MANIFEST) == evaluation_manifest_hash(MANIFEST)
    assert len(payload["scenario_groups"]["normal"]) == 12
    assert len(payload["scenario_groups"]["stress"]) == 12
    assert len(payload["scenario_groups"]["negative"]) == 3
    assert not payload["holdout_execution_allowed"]
    assert not payload["tuning_allowed"]
    assert not any("holdout" in item["scenario_id"] for item in payload["scenarios"])


def test_solver_configuration_is_exactly_frozen() -> None:
    config = load_evaluation_manifest(MANIFEST)["solver_configuration"]
    assert config["solver_seed"] == 20260630
    assert config["workers"] == 1
    assert (
        config["bootstrap_time_limit_seconds"],
        config["per_stage_time_limit_seconds"],
        config["total_time_limit_seconds"],
    ) == (30, 30, 300)
    assert config["initial_solution_artifact_dir"] is None
    assert config["external_persisted_seed"] is False


def test_scenario_selection_preserves_frozen_order_and_rejects_holdout() -> None:
    runner = CpSatRobustnessRunner(MANIFEST)
    assert [item.scenario_id for item in runner.select("normal", max_scenarios=2)] == [
        "normal_dev_reference_2026",
        "normal_dev_01",
    ]
    with pytest.raises(CpSatEvaluationError):
        runner.select("holdout")


def test_malformed_manifest_configuration_fails_closed(tmp_path) -> None:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload["solver_configuration"]["solver_seed"] = 2026
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CpSatEvaluationError, match="frozen CP-SAT configuration"):
        CpSatRobustnessRunner(path)


def test_dry_run_verifies_source_without_creating_output(tmp_path) -> None:
    runner = CpSatRobustnessRunner(MANIFEST)
    output = tmp_path / "dry-run-output"
    selected = runner.dry_run("normal", max_scenarios=1)
    assert selected == ["normal_dev_reference_2026"]
    assert not output.exists()


def test_source_verification_does_not_change_persisted_artifact(tmp_path) -> None:
    runner = CpSatRobustnessRunner(MANIFEST)
    source = Path(runner.manifest["source_normal_suite"]["artifact_dir"]) / "SHA256SUMS.txt"
    before = source.read_bytes()
    runner.verify_sources(runner.select("normal", max_scenarios=1))
    assert source.read_bytes() == before
    assert tmp_path.is_dir()


def test_nonempty_output_is_never_overwritten(tmp_path, monkeypatch) -> None:
    runner = CpSatRobustnessRunner(MANIFEST)
    monkeypatch.setattr(runner, "verify_sources", lambda scenarios: {})
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "marker.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(CpSatEvaluationError, match="non-empty"):
        runner.run(output, group="normal", max_scenarios=1)
    assert marker.read_text(encoding="utf-8") == "keep"


def test_resume_rejects_manifest_mismatch(tmp_path, monkeypatch) -> None:
    runner = CpSatRobustnessRunner(MANIFEST)
    monkeypatch.setattr(runner, "verify_sources", lambda scenarios: {})
    output = tmp_path / "resume"
    output.mkdir()
    (output / "run_manifest.json").write_text(
        json.dumps({
            "evaluation_manifest_sha256": "wrong",
            "source_git_commit": runner.manifest["source_git_commit"],
            "solver_configuration_hash": "wrong",
        }),
        encoding="utf-8",
    )
    with pytest.raises(CpSatEvaluationError, match="resume provenance mismatch"):
        runner.run(output, group="normal", max_scenarios=1, resume=True)


def test_unknown_scenario_is_recorded_without_assignment_or_fake_zero_metrics(tmp_path, monkeypatch) -> None:
    runner = CpSatRobustnessRunner(MANIFEST)
    scenario = runner.select("normal", max_scenarios=1)[0]
    fingerprint = _fingerprint()
    calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        runner_module,
        "_load_scenario_input",
        lambda *args: (_fake_input(), fingerprint, {"input_fingerprint": fingerprint}),
    )

    def fake_solver(*args, **kwargs):
        calls.append(kwargs)
        return _unknown_result()

    monkeypatch.setattr(runner_module, "run_fair_cp_sat_solver", fake_solver)
    output = tmp_path / "unknown"
    output.mkdir()
    runner._run_scenario(output, scenario)
    summary = json.loads(
        (output / "scenarios" / scenario.scenario_id / "scenario_summary.json").read_text(encoding="utf-8")
    )
    assert calls[0]["seed"] == 20260630
    assert summary["status"] == "completed_without_assignment"
    assert summary["result"]["status"] == "UNKNOWN"
    assert summary["result"]["primary_assigned"] is None
    assert not (output / "scenarios" / scenario.scenario_id / "solver" / "student_outcomes.csv").exists()


def test_formal_solver_entrypoint_is_called_without_persisted_seed(tmp_path, monkeypatch) -> None:
    runner = CpSatRobustnessRunner(MANIFEST)
    scenario = runner.select("normal", max_scenarios=1)[0]
    fingerprint = _fingerprint()
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        runner_module,
        "_load_scenario_input",
        lambda *args: (_fake_input(), fingerprint, {"input_fingerprint": fingerprint}),
    )

    def fake_solver(*args, **kwargs):
        calls.append(kwargs)
        return _unknown_result()

    monkeypatch.setattr(runner_module, "run_fair_cp_sat_solver", fake_solver)
    runner._run_scenario(tmp_path, scenario)
    assert calls and calls[0]["initial_solution_artifact_dir"] is None
    assert calls[0]["num_search_workers"] == 1
    assert calls[0]["bootstrap_time_seconds"] == 30
    assert calls[0]["max_time_seconds_per_stage"] == 30
    assert calls[0]["max_total_time_seconds"] == 300


def test_negative_certificate_is_checked_before_solver(tmp_path, monkeypatch) -> None:
    runner = CpSatRobustnessRunner(MANIFEST)
    scenario = runner.select("negative", max_scenarios=1)[0]
    fingerprint = _fingerprint()
    called = False
    monkeypatch.setattr(
        runner_module,
        "_load_scenario_input",
        lambda *args: (_fake_input(), fingerprint, {"input_fingerprint": fingerprint}),
    )
    monkeypatch.setattr(runner_module, "validate_certificate", lambda *args: (False, "invalid fixture"))

    def should_not_run(*args, **kwargs):
        nonlocal called
        called = True
        return _unknown_result()

    monkeypatch.setattr(runner_module, "run_fair_cp_sat_solver", should_not_run)
    with pytest.raises(CpSatEvaluationCorrectnessError, match="certificate invalid before solve"):
        runner._run_scenario(tmp_path, scenario)
    assert not called


def test_feasible_without_assignment_is_a_correctness_error() -> None:
    result = replace(_unknown_result(), solve_status=CpSatSolveStatus.FEASIBLE)
    scenario = CpSatRobustnessRunner(MANIFEST).select("normal", max_scenarios=1)[0]
    with pytest.raises(CpSatEvaluationCorrectnessError, match="has no final assignment"):
        _validate_result(result, {"final_schedule_policy_pass": None, "consistency_issue_count": None, "assignment_nonpublishable": True}, scenario)


def test_skipped_stage_has_no_reused_objective_or_response() -> None:
    first = _unknown_result().stage_diagnostics[0]
    skipped = replace(
        first,
        stage_name=CpSatStageName.LOGICAL_SCHEDULE_COMPLETION,
        model_scope=CpSatModelScope.ENRICHMENT,
        status=CpSatSolveStatus.SKIPPED,
        skipped=True,
        skip_reason="no_incumbent",
        objective_value=None,
        best_objective_bound=None,
        response_proto_hash="",
    )
    result = replace(_unknown_result(), stage_diagnostics=(first, skipped))
    trace = _stage_trace(result, "fixture")
    assert trace[1]["status"] == "SKIPPED"
    assert trace[1]["objective_value"] is None
    assert trace[1]["response_proto_hash"] == ""


def test_objective_bound_direction_and_logical_upper_bound_are_checked() -> None:
    _validate_objective_bounds(
        [{"stage_name": "logical_schedule_completion", "objective_value": 4, "best_objective_bound": 5, "objective_sense": "max"}],
        5,
    )
    with pytest.raises(CpSatEvaluationCorrectnessError, match="exceeds best bound"):
        _validate_objective_bounds(
            [{"stage_name": "logical_schedule_completion", "objective_value": 6, "best_objective_bound": 5, "objective_sense": "max"}],
            10,
        )
    with pytest.raises(CpSatEvaluationCorrectnessError, match="theoretical maximum"):
        _validate_objective_bounds(
            [{"stage_name": "logical_schedule_completion", "objective_value": 5, "best_objective_bound": 6, "objective_sense": "max"}],
            4,
        )


def test_aggregate_keeps_missing_metrics_null() -> None:
    summary = _aggregate(
        [{
            "status": "UNKNOWN",
            "final_assignment_available": False,
            "primary_satisfaction_rate": None,
            "logical_full_rate": None,
            "total_logical_gap": None,
            "runtime_seconds": 3.0,
            "objective_to_bound_gap": None,
        }],
        "fixture",
    )
    assert summary["status_counts"] == {"UNKNOWN": 1}
    assert summary["final_assignment_count"] == 0
    assert summary["primary_satisfaction_rate"]["count"] == 0
    assert summary["primary_satisfaction_rate"]["median"] is None


def test_cp_sat_vs_greedy_pairing_is_scenario_local() -> None:
    scenario = CpSatRobustnessRunner(MANIFEST).select("normal", max_scenarios=1)[0]
    metrics = {
        "status": "UNKNOWN",
        "final_assignment_available": False,
        "primary_assigned": None,
        "primary_satisfaction_rate": None,
        "logical_fully_scheduled_students": None,
        "logical_full_rate": None,
        "total_logical_gap": None,
        "gap_over_1_students": None,
        "below_five_students": None,
        "policy_violation_count": None,
    }
    rows = _paired_rows(metrics, {"seeded_random_greedy": {"primary_assigned": "1"}, "first_come_first_served_greedy": {"primary_assigned": "1"}, "grade_priority_greedy": {"primary_assigned": "1"}, "constrained_first_greedy": {"primary_assigned": "1"}}, scenario, 1)
    assert len(rows) == 4
    assert all(row["primary_assigned_delta"] is None for row in rows)


def test_missing_scenario_id_fails_closed() -> None:
    with pytest.raises(CpSatEvaluationError, match="not in development evaluation"):
        CpSatRobustnessRunner(MANIFEST).select(scenario_id="missing_scenario")


def test_manifest_stage_order_excludes_bootstrap_from_objectives() -> None:
    config = load_evaluation_manifest(MANIFEST)["solver_configuration"]
    assert config["stage_order"][0] == "feasibility_bootstrap"
    assert config["stage_order"][1] == "full_model_feasibility_incumbent"
    assert config["objective_order"][0] == "math_coverage"
    assert "feasibility_bootstrap" not in config["objective_order"]
    assert "full_model_feasibility_incumbent" not in config["objective_order"]


def test_unknown_has_no_final_stage_or_assignment_source() -> None:
    trace = _stage_trace(_unknown_result(), "fixture")
    assert not [row for row in trace if row["response_proto_hash"]]
    assert runner_module._final_stage(trace) is None


def test_negative_feasible_result_is_a_correctness_error() -> None:
    result = replace(_unknown_result(), solve_status=CpSatSolveStatus.FEASIBLE)
    scenario = CpSatRobustnessRunner(MANIFEST).select("negative", max_scenarios=1)[0]
    with pytest.raises(CpSatEvaluationCorrectnessError, match="FEASIBLE has no final assignment|negative scenario"):
        _validate_result(result, {"final_schedule_policy_pass": None, "consistency_issue_count": None, "assignment_nonpublishable": True}, scenario)


def test_negative_assignment_is_a_correctness_error_even_if_status_is_unknown() -> None:
    result = replace(
        _unknown_result(),
        student_outcomes=(object(),),
    )
    scenario = CpSatRobustnessRunner(MANIFEST).select("negative", max_scenarios=1)[0]
    with pytest.raises(CpSatEvaluationCorrectnessError, match="assignment exists"):
        _validate_result(result, {"final_schedule_policy_pass": True, "consistency_issue_count": 0, "assignment_nonpublishable": False}, scenario)


def test_ordinary_assignment_policy_failure_is_critical() -> None:
    result = replace(
        _unknown_result(),
        solve_status=CpSatSolveStatus.FEASIBLE,
        student_outcomes=(object(),),
    )
    scenario = CpSatRobustnessRunner(MANIFEST).select("normal", max_scenarios=1)[0]
    with pytest.raises(CpSatEvaluationCorrectnessError, match="policy or consistency"):
        _validate_result(result, {"final_schedule_policy_pass": False, "consistency_issue_count": 1, "assignment_nonpublishable": True}, scenario)


def test_minimization_bound_direction_is_checked() -> None:
    with pytest.raises(CpSatEvaluationCorrectnessError, match="best bound"):
        _validate_objective_bounds(
            [{"stage_name": "primary_unmet_count", "objective_value": 2, "best_objective_bound": 3, "objective_sense": "min"}],
            10,
        )


def test_grade_rows_keep_null_quality_for_no_assignment() -> None:
    rows = runner_module._grade_rows(_fake_input(), _unknown_result(), "fixture")
    assert rows == [{
        "scenario_id": "fixture",
        "grade": 9,
        "student_count": 1,
        "primary_satisfaction_rate": None,
        "logical_full_rate": None,
        "mean_gap": None,
        "gap_over_1_count": None,
        "below_five_count": None,
        "policy_violation_count": None,
    }]


def test_completed_summary_rejects_incomplete_checkpoint(tmp_path) -> None:
    path = tmp_path / "scenario_summary.json"
    path.write_text(json.dumps({"status": "running"}), encoding="utf-8")
    with pytest.raises(CpSatEvaluationError, match="not completed"):
        runner_module._read_completed_summary(path)


def test_resume_rejects_changed_scenario_selection(tmp_path, monkeypatch) -> None:
    runner = CpSatRobustnessRunner(MANIFEST)
    monkeypatch.setattr(runner, "verify_sources", lambda scenarios: {})
    output = tmp_path / "resume-selection"
    output.mkdir()
    (output / "run_manifest.json").write_text(json.dumps({
        "evaluation_manifest_sha256": runner.manifest_hash,
        "source_git_commit": runner.manifest["source_git_commit"],
        "solver_configuration_hash": runner_module._json_hash(runner.manifest["solver_configuration"]),
        "selected_scenario_ids": ["normal_dev_01"],
        "completed_scenario_ids": [],
        "holdout_runs": 0,
        "external_persisted_seed": False,
    }), encoding="utf-8")
    with pytest.raises(CpSatEvaluationError, match="scenario selection mismatch"):
        runner.run(output, group="normal", max_scenarios=1, resume=True)


def test_cli_dry_run_is_development_only(capsys) -> None:
    assert runner_module.main(["--dry-run", "--group", "normal", "--max-scenarios", "1"]) == 0
    assert "normal_dev_reference_2026" in capsys.readouterr().out


def test_full_model_infeasible_is_the_only_global_solver_proof() -> None:
    audit = _audit_status_semantics(
        _audit_scenario(),
        {"status": "INFEASIBLE", "final_assignment_available": False, "certificate_valid": None},
        _audit_trace(
            ("feasibility_bootstrap", "UNKNOWN", None),
            ("full_model_feasibility_incumbent", "INFEASIBLE", None),
        ),
    )
    assert audit["evaluation_outcome"] == "GLOBAL_INFEASIBLE"
    assert audit["global_infeasibility_proven"] is True
    assert audit["infeasibility_scope"] == "full_hard_model"


def test_later_math_infeasible_is_not_global_infeasible() -> None:
    audit = _audit_status_semantics(
        _audit_scenario(),
        {"status": "INFEASIBLE", "final_assignment_available": False, "certificate_valid": None},
        _audit_trace(
            ("feasibility_bootstrap", "UNKNOWN", None),
            ("full_model_feasibility_incumbent", "UNKNOWN", None),
            ("math_coverage", "INFEASIBLE", None),
        ),
    )
    assert audit["evaluation_outcome"] == "CORE_STAGE_INFEASIBLE"
    assert audit["global_infeasibility_proven"] is False
    assert audit["infeasibility_scope"] == "unknown"


def test_later_logical_infeasible_records_fixed_objective_scope() -> None:
    audit = _audit_status_semantics(
        _audit_scenario(),
        {"status": "INFEASIBLE", "final_assignment_available": False, "certificate_valid": None},
        _audit_trace(
            ("feasibility_bootstrap", "UNKNOWN", None),
            ("full_model_feasibility_incumbent", "UNKNOWN", None),
            ("math_coverage", "OPTIMAL", None),
            ("primary_unmet_count", "FEASIBLE", [{"stage_name": "math_coverage", "value": 5}]),
            ("logical_schedule_completion", "INFEASIBLE", [{"stage_name": "math_coverage", "value": 5}]),
        ),
    )
    assert audit["evaluation_outcome"] == "LEXICOGRAPHIC_STAGE_INFEASIBLE"
    assert audit["global_infeasibility_proven"] is False
    assert audit["infeasibility_scope"] == "fixed_objective_stage"
    assert audit["fixed_objective_infeasible_stage_count"] == 1


def test_unknown_without_assignment_is_explicitly_unknown() -> None:
    audit = _audit_status_semantics(
        _audit_scenario(),
        {"status": "UNKNOWN", "final_assignment_available": False, "certificate_valid": None},
        _audit_trace(
            ("feasibility_bootstrap", "UNKNOWN", None),
            ("full_model_feasibility_incumbent", "UNKNOWN", None),
        ),
    )
    assert audit["evaluation_outcome"] == "UNKNOWN_NO_FINAL_ASSIGNMENT"
    assert audit["infeasibility_scope"] == "unknown"
    assert audit["complete_incumbent_found"] is False


def test_negative_certificate_is_separate_from_solver_global_proof() -> None:
    audit = _audit_status_semantics(
        CpSatRobustnessRunner(MANIFEST).select("negative", max_scenarios=1)[0],
        {"status": "INFEASIBLE", "final_assignment_available": False, "certificate_valid": True},
        _audit_trace(("feasibility_bootstrap", "INFEASIBLE", None)),
    )
    assert audit["evaluation_outcome"] == "STRUCTURAL_CERTIFICATE_INFEASIBLE"
    assert audit["infeasibility_scope"] == "structural_certificate"
    assert audit["certificate_proof_valid"] is True
    assert audit["solver_global_infeasibility_proven"] is False


def test_audit_preserves_raw_result_and_terminal_stage_status() -> None:
    audit = _audit_status_semantics(
        _audit_scenario(),
        {"status": "INFEASIBLE", "final_assignment_available": False, "certificate_valid": None},
        _audit_trace(
            ("feasibility_bootstrap", "UNKNOWN", None),
            ("math_coverage", "UNKNOWN", None),
        ),
    )
    assert audit["raw_result_status"] == "INFEASIBLE"
    assert audit["raw_terminal_solver_status"] == "UNKNOWN"
    assert audit["terminal_stage"] == "math_coverage"


@pytest.mark.parametrize(
    ("publishable", "ready"),
    [(0, False), (5, False), (6, True), (7, True), (12, True)],
)
def test_holdout_readiness_uses_majority_of_normal_without_assignment(publishable: int, ready: bool) -> None:
    rows = [
        {
            "scenario_group": "normal",
            "publishable_assignment_available": index < publishable,
            "raw_result_status": "UNKNOWN",
        }
        for index in range(12)
    ] + [
        {"scenario_group": "stress", "publishable_assignment_available": True, "raw_result_status": "FEASIBLE"}
        for _ in range(12)
    ]
    result = _audited_readiness(rows, [], {"verified": True}, [])
    assert result["normal_scenarios_attempted"] == 12
    assert result["normal_publishable_assignments"] == publishable
    assert result["normal_no_assignment_count"] == 12 - publishable
    assert result["majority_normal_without_assignment"] is (publishable < 6)
    assert result["ready_for_holdout"] is ready


def test_audit_rebuild_is_read_only_and_does_not_invoke_solver(tmp_path, monkeypatch) -> None:
    source = Path("../fair-course-allocation-artifacts/robustness-v1/cp-sat-development-v1")
    if not (source / "SHA256SUMS.txt").is_file():
        pytest.skip("external Phase C artifact is not distributed with the repository")
    before = hashlib.sha256((source / "SHA256SUMS.txt").read_bytes()).hexdigest()
    monkeypatch.setattr(
        runner_module,
        "run_fair_cp_sat_solver",
        lambda *args, **kwargs: pytest.fail("status audit must not invoke CP-SAT"),
    )
    output = tmp_path / "audited"
    summary = audit_existing_artifact(
        source,
        output,
        manifest_path=_historical_manifest_for_persisted_artifact(tmp_path, source),
    )
    readiness = json.loads((output / "holdout_readiness_assessment.json").read_text())
    assert summary["all_status_semantics"]["scenarios"] == 27
    assert readiness["ready_for_holdout"] is False
    assert readiness["normal_publishable_assignments"] == 0
    assert hashlib.sha256((source / "SHA256SUMS.txt").read_bytes()).hexdigest() == before
    assert not (output / "scenarios").exists()
    assert (output / "status_semantics_audit.csv").is_file()


def test_audited_summary_contains_all_scenarios_and_no_holdouts(tmp_path) -> None:
    source = Path("../fair-course-allocation-artifacts/robustness-v1/cp-sat-development-v1")
    if not (source / "SHA256SUMS.txt").is_file():
        pytest.skip("external Phase C artifact is not distributed with the repository")
    output = tmp_path / "audited"
    audit_existing_artifact(
        source,
        output,
        manifest_path=_historical_manifest_for_persisted_artifact(tmp_path, source),
    )
    rows = runner_module.pd.read_csv(output / "status_semantics_audit.csv")
    assert len(rows) == 27
    assert set(rows["scenario_group"]) == {"normal", "stress", "negative"}
    assert not any("holdout" in scenario_id for scenario_id in rows["scenario_id"])
