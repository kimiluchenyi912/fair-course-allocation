from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from src.allocation import CpSatSolveStatus, run_fair_cp_sat_solver
from src.cp_sat_robustness_runner import (
    CpSatEvaluationError,
    _read_json,
    _result_metrics,
    _sha256_file,
    _verify_sha256_manifest,
    _write_checksums,
    _write_csv,
    _write_json,
)
import src.cp_sat_normal_evaluation_runner as normal_evaluation_runner_module
from src.cp_sat_normal_evaluation_runner import (
    STABLE_SCENARIO_ID,
    CpSatNormalEvaluationRunner,
    _audit_row_timing,
    _compute_aggregate_summary,
    _compute_success_gate,
    _greedy_diagnostics,
    _import_stable_reference,
    _infeasibility_scope_row,
    _is_publishable,
    _percentile,
    _stats,
    _repair_legacy_row,
    _validate_row_schema,
    load_normal_evaluation_manifest,
    rebuild_audited_normal_evaluation,
)

from tests.test_cp_sat_cold_start_recovery import _complete_fixture
from tests.test_cp_sat_solver import canonical, fallback_rules, math_ids, request_row, section_row


# ---------------------------------------------------------------------------
# 1. Manifest validation
# ---------------------------------------------------------------------------


def test_manifest_contains_only_the_12_frozen_normal_ids_in_order() -> None:
    payload = load_normal_evaluation_manifest()
    assert set(payload["scenario_groups"]) == {"normal"}
    assert payload["scenario_groups"]["normal"] == [
        "normal_dev_reference_2026", "normal_dev_01", "normal_dev_02", "normal_dev_03",
        "normal_dev_04", "normal_dev_05", "normal_dev_06", "normal_dev_07",
        "normal_dev_08", "normal_dev_09", "normal_dev_10", "normal_dev_11",
    ]
    assert len(payload["scenarios"]) == 12
    config = payload["solver_configuration"]
    assert config["solver_seed"] == 20260630
    assert config["workers"] == 1
    assert config["total_time_limit_seconds"] == 300
    assert config["internal_feasibility_hint_strategy"] == "constrained_first"
    assert config["internal_repair_objective_strategy"] == "hamming_to_constrained_first"
    assert config["stop_after_first_valid_solution"] is True
    assert config["external_persisted_seed"] is False
    assert config["initial_solution_artifact_dir"] is None
    assert payload["holdout_execution_allowed"] is False
    assert payload["stress_execution_allowed"] is False
    assert payload["tuning_allowed"] is True


def test_manifest_rejects_stress_or_holdout_scenario_ids(tmp_path) -> None:
    payload = load_normal_evaluation_manifest()
    payload["scenarios"][-1] = {**payload["scenarios"][-1], "scenario_id": "stress_bad"}
    payload["scenario_groups"]["normal"][-1] = "stress_bad"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CpSatEvaluationError):
        load_normal_evaluation_manifest(path)


def test_manifest_rejects_wrong_order(tmp_path) -> None:
    payload = load_normal_evaluation_manifest()
    normal = payload["scenario_groups"]["normal"]
    normal[0], normal[1] = normal[1], normal[0]
    path = tmp_path / "bad_order.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CpSatEvaluationError):
        load_normal_evaluation_manifest(path)


def test_manifest_rejects_stress_execution_allowed_true(tmp_path) -> None:
    payload = load_normal_evaluation_manifest()
    payload["stress_execution_allowed"] = True
    path = tmp_path / "bad_stress.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CpSatEvaluationError):
        load_normal_evaluation_manifest(path)


def test_manifest_solver_seed_workers_budget_are_frozen(tmp_path) -> None:
    for field, value in (("solver_seed", 1), ("workers", 4), ("total_time_limit_seconds", 60)):
        payload = load_normal_evaluation_manifest()
        payload["solver_configuration"][field] = value
        path = tmp_path / f"bad_{field}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(CpSatEvaluationError):
            load_normal_evaluation_manifest(path)


def test_hamming_objective_strategy_cannot_be_changed(tmp_path) -> None:
    payload = load_normal_evaluation_manifest()
    payload["solver_configuration"]["internal_repair_objective_strategy"] = "none"
    path = tmp_path / "bad_objective.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CpSatEvaluationError):
        load_normal_evaluation_manifest(path)


# ---------------------------------------------------------------------------
# 2. Stable reference import (against the real, already-completed artifact)
# ---------------------------------------------------------------------------


def _runner() -> CpSatNormalEvaluationRunner:
    return CpSatNormalEvaluationRunner()


@pytest.fixture
def stable_probe_artifact(require_external_artifact) -> Path:
    return require_external_artifact(
        "robustness-v1/cp-sat-cold-start-repair-probe-v1"
    )


@pytest.mark.external_artifact
def test_stable_reference_imports_when_configuration_matches(stable_probe_artifact: Path) -> None:
    runner = _runner()
    imported = _import_stable_reference(runner.manifest, runner.config_dir, stable_probe_artifact)
    row = imported["row"]
    assert row["result_origin"] == "imported_frozen_probe"
    assert row["solver_rerun"] is False
    assert row["status"] == "FEASIBLE"
    assert row["final_assignment_available"] is True
    assert row["final_schedule_policy_pass"] is True
    assert row["consistency_issue_count"] == 0
    assert row["source_response_hash"]
    assert row["source_artifact_path"] == str(stable_probe_artifact)


@pytest.mark.external_artifact
def test_stable_reference_import_does_not_trigger_a_solve(monkeypatch, stable_probe_artifact: Path) -> None:
    import src.cp_sat_normal_evaluation_runner as mod

    def _boom(*_args, **_kwargs):
        raise AssertionError("run_fair_cp_sat_solver must not be called while importing the stable reference")

    monkeypatch.setattr(mod, "run_fair_cp_sat_solver", _boom)
    runner = _runner()
    imported = mod._import_stable_reference(runner.manifest, runner.config_dir, stable_probe_artifact)
    assert imported["row"]["status"] == "FEASIBLE"


@pytest.mark.external_artifact
def test_stable_reference_import_fails_closed_on_solver_seed_mismatch(stable_probe_artifact: Path) -> None:
    runner = _runner()
    tampered = json.loads(json.dumps(runner.manifest))
    tampered["solver_configuration"]["solver_seed"] = 1
    with pytest.raises(CpSatEvaluationError, match="solver configuration mismatch"):
        _import_stable_reference(tampered, runner.config_dir, stable_probe_artifact)


def test_stable_reference_import_fails_closed_on_missing_artifact(tmp_path) -> None:
    runner = _runner()
    with pytest.raises(CpSatEvaluationError):
        _import_stable_reference(runner.manifest, runner.config_dir, tmp_path / "does-not-exist")


@pytest.mark.external_artifact
def test_stable_reference_import_reports_mandatory_fallback_candidate_count(stable_probe_artifact: Path) -> None:
    runner = _runner()
    imported = _import_stable_reference(runner.manifest, runner.config_dir, stable_probe_artifact)
    row = imported["row"]
    assert row["canonical_input_candidate_edges"] == 165481
    assert row["model_candidate_variables"] == 167120
    assert row["mandatory_fallback_candidate_variables"] == 1639


@pytest.mark.external_artifact
def test_stable_reference_import_does_not_modify_the_source_artifact(stable_probe_artifact: Path) -> None:
    runner = _runner()
    before = sorted((stable_probe_artifact / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines())
    _import_stable_reference(runner.manifest, runner.config_dir, stable_probe_artifact)
    after = sorted((stable_probe_artifact / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines())
    assert before == after
    # the checksum manifest itself must still verify cleanly
    _verify_sha256_manifest(stable_probe_artifact)


# ---------------------------------------------------------------------------
# 3. Greedy structural integrity + hint-is-not-final-result guarantees
# ---------------------------------------------------------------------------


def _small_allocation_input():
    courses = ["CORE_A", "CORE_B", "CORE_C", "CORE_D", "ALT1"]
    requests = [request_row("STU_1", course) for course in courses]
    sections = [
        section_row(f"SEC_{course}", course, f"P{index + 1}", capacity=10, group_id=f"{course}_1")
        for index, course in enumerate(courses)
    ]
    return canonical([("STU_1", 12, 5, False)], requests, sections)


def test_greedy_structural_integrity_is_clean_on_a_feasible_fixture() -> None:
    diagnostics = _greedy_diagnostics(_small_allocation_input(), math_ids(), fallback_rules())
    structural = diagnostics["structural_integrity"]
    assert structural["capacity_violations"] == 0
    assert structural["period_conflicts"] == 0
    assert structural["duplicate_logical_identity_issues"] == 0
    assert structural["invalid_candidate_edges"] == 0


def test_greedy_policy_failure_is_allowed_as_a_hint_but_not_fatal() -> None:
    # A high-demand single-section course with more requesters than seats
    # forces ordinary Greedy policy violations, which must be reportable as
    # hint diagnostics without raising -- the hint is never a gate.
    student_rows = [(f"STU_{i:03d}", 12, 1, False) for i in range(5)]
    requests = [request_row(student_id, "HIGH") for student_id, *_ in student_rows]
    sections = [section_row("SEC_HIGH", "HIGH", "P1", capacity=2, group_id="HIGH_1")]
    data = canonical(student_rows, requests, sections)
    diagnostics = _greedy_diagnostics(data, math_ids(), fallback_rules())
    assert diagnostics["policy_violation_count"] >= 0  # never raises; hint stays advisory


def test_hint_violations_do_not_propagate_to_final_solver_result() -> None:
    result = run_fair_cp_sat_solver(
        _complete_fixture(),
        seed=20260630,
        math_fallback_rules=fallback_rules(),
        math_course_ids=math_ids(),
        max_time_seconds_per_stage=2,
        bootstrap_time_seconds=0,
        max_total_time_seconds=5,
        num_search_workers=1,
        use_feasibility_bootstrap=False,
        use_constrained_first_hint=False,
        logical_schedule_completion_enabled=False,
        internal_feasibility_hint_strategy="constrained_first",
        internal_repair_time_seconds=2,
        internal_repair_objective_strategy="hamming_to_constrained_first",
        stop_after_first_valid_solution=True,
    )
    metrics = _result_metrics(result, _complete_fixture())
    # The final assignment comes only from the CP-SAT response; policy PASS
    # here is a property of that response, independent of any Greedy hint
    # policy violations.
    assert metrics["final_schedule_policy_pass"] is True
    assert result.assignments  # a real solver assignment exists
    assert result.model_stats.internal_hint_candidate_variables == result.model_stats.internal_hint_candidate_variables_hinted
    assert result.model_stats.internal_hint_candidate_coverage_rate == 1.0


# ---------------------------------------------------------------------------
# 4. UNKNOWN / INFEASIBLE handling
# ---------------------------------------------------------------------------


def test_infeasible_full_hard_model_keeps_infeasible_status_and_no_assignment() -> None:
    data = canonical(
        [("P1", 12, 1, True), ("P2", 12, 1, True)],
        [request_row("P1", "CORE_A"), request_row("P2", "CORE_A")],
        [section_row("SEC_CORE_A", "CORE_A", "P1", capacity=1, group_id="CORE_A_1")],
    )
    result = run_fair_cp_sat_solver(
        data,
        seed=20260630,
        math_fallback_rules=fallback_rules(),
        math_course_ids=math_ids(),
        max_time_seconds_per_stage=2,
        bootstrap_time_seconds=0,
        max_total_time_seconds=5,
        num_search_workers=1,
        use_feasibility_bootstrap=False,
        use_constrained_first_hint=False,
        logical_schedule_completion_enabled=False,
        internal_feasibility_hint_strategy="constrained_first",
        internal_repair_time_seconds=2,
        internal_repair_objective_strategy="hamming_to_constrained_first",
        stop_after_first_valid_solution=True,
    )
    assert result.solve_status == CpSatSolveStatus.INFEASIBLE
    assert result.assignments == ()
    metrics = _result_metrics(result, data)
    assert metrics["primary_assigned"] is None
    assert metrics["final_schedule_policy_pass"] is None
    assert metrics["assignment_nonpublishable"] is True


def test_unknown_metrics_are_null_via_result_metrics() -> None:
    # _result_metrics is the single source of truth this runner reuses for
    # quality fields; its no-outcome branch must report every quality field
    # as null rather than falling back to a Greedy value.
    data = _small_allocation_input()
    from src.allocation.cp_sat_models import CpSatAllocationResult, CpSatModelStats, CpSatSolveStatus as Status

    empty_result = CpSatAllocationResult(
        algorithm_name="cp_sat_full_feasibility",
        seed=1,
        solve_status=Status.UNKNOWN,
        lexicographic_optimality_proven=False,
        stage_diagnostics=(),
        objective_values=__import__("src.allocation.cp_sat_models", fromlist=["CpSatObjectiveValues"]).CpSatObjectiveValues(),
        model_stats=CpSatModelStats(total_variables=0, total_constraints=0, build_time_seconds=0.0, solve_time_seconds=0.0),
    )
    metrics = _result_metrics(empty_result, data)
    assert metrics["primary_assigned"] is None
    assert metrics["logical_full_rate"] is None
    assert metrics["final_schedule_policy_pass"] is None
    assert metrics["assignment_nonpublishable"] is True


# ---------------------------------------------------------------------------
# 5. Resume / provenance / SHA256
# ---------------------------------------------------------------------------


def test_resume_does_not_resolve_a_completed_scenario(tmp_path, monkeypatch) -> None:
    runner = _runner()
    source_info = {"synthetic": True}
    monkeypatch.setattr(runner, "verify_sources", lambda scenarios: source_info)
    monkeypatch.setattr(
        normal_evaluation_runner_module,
        "_verify_source_suite",
        lambda manifest, suite: source_info,
    )

    def _fail_solve(*_a, **_k):
        raise AssertionError("a completed scenario must not be re-solved on resume")

    monkeypatch.setattr(runner, "_solve_and_export_scenario", _fail_solve)
    monkeypatch.setattr(runner, "_import_scenario", _fail_solve)

    output = tmp_path / "phase_c"
    output.mkdir()
    from src.cp_sat_robustness_runner import _json_hash

    selected_ids = ["normal_dev_01"]
    run_manifest = {
        "schema_version": 1,
        "status": "running",
        "evaluation_manifest_sha256": runner.manifest_hash,
        "source_git_commit": runner.manifest["source_git_commit"],
        "solver_configuration_hash": _json_hash(runner.manifest["solver_configuration"]),
        "selected_scenario_ids": selected_ids,
        "completed_scenario_ids": ["normal_dev_01"],
        "failed_scenario_ids": [],
        "imported_scenario_ids": [],
        "solved_scenario_ids": ["normal_dev_01"],
        "holdout_runs": 0,
        "stress_runs": 0,
        "external_persisted_seed": False,
    }
    _write_json(output / "run_manifest.json", run_manifest)
    scenario_dir = output / "scenarios" / "normal_dev_01"
    scenario_dir.mkdir(parents=True)
    _write_json(scenario_dir / "scenario_summary.json", {
        "status": "completed_with_assignment",
        "evaluation_manifest_sha256": runner.manifest_hash,
        "scenario": {"scenario_id": "normal_dev_01"},
        "result": {
            "scenario_id": "normal_dev_01",
            "scenario_group": "normal",
            "result_origin": "solved",
            "solver_rerun": True,
            "status": "FEASIBLE",
            "final_assignment_available": True,
            "publishable_assignment_available": True,
            "publishable_recovery": True,
            "response_proto_hash": "deadbeef",
            "assignment_nonpublishable": False,
            "final_schedule_policy_pass": True,
            "consistency_issue_count": 0,
            "student_count": 1,
            "primary_assigned": 1,
            "primary_satisfaction_rate": 1.0,
            "logical_fully_scheduled_students": 1,
            "logical_full_rate": 1.0,
            "total_logical_gap": 0,
            "gap_over_1_students": 0,
            "below_five_students": 0,
            "policy_violation_count": 0,
            "end_to_end_scenario_runtime_seconds": 1.0,
            "time_to_first_solution_seconds": 0.5,
        },
    })
    _write_json(scenario_dir / "stage_trace.json", {"scenario_id": "normal_dev_01", "stages": []})

    summary = runner.run(output, scenario_id="normal_dev_01", resume=True)
    assert summary["attempted_count"] == 1


def test_resume_rejects_provenance_mismatch(tmp_path, monkeypatch) -> None:
    runner = _runner()
    monkeypatch.setattr(runner, "verify_sources", lambda scenarios: {"stub": True})
    output = tmp_path / "phase_c_bad"
    output.mkdir()
    _write_json(output / "run_manifest.json", {
        "evaluation_manifest_sha256": "not-the-real-hash",
        "source_git_commit": runner.manifest["source_git_commit"],
        "selected_scenario_ids": ["normal_dev_01"],
        "completed_scenario_ids": [],
    })
    with pytest.raises(CpSatEvaluationError, match="resume provenance mismatch"):
        runner.run(output, scenario_id="normal_dev_01", resume=True)


def test_sha256_generation_and_verification_round_trip(tmp_path) -> None:
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "a.json").write_text('{"x": 1}\n', encoding="utf-8")
    (root / "b.json").write_text('{"y": 2}\n', encoding="utf-8")
    _write_checksums(root)
    tree_hash, files, directories, total_bytes = _verify_sha256_manifest(root)
    assert files == 3  # a.json, b.json, SHA256SUMS.txt
    assert directories == 0
    assert total_bytes == sum((root / name).stat().st_size for name in ("a.json", "b.json", "SHA256SUMS.txt"))
    (root / "a.json").write_text('{"x": 999}\n', encoding="utf-8")
    with pytest.raises(CpSatEvaluationError, match="SHA256 mismatch"):
        _verify_sha256_manifest(root)


# ---------------------------------------------------------------------------
# 6. Aggregate / success-gate arithmetic (pure, synthetic rows)
# ---------------------------------------------------------------------------


def _row(scenario_id, *, status="FEASIBLE", publishable=True, **overrides):
    base = {
        "scenario_id": scenario_id,
        "scenario_group": "normal",
        "result_origin": "solved",
        "solver_rerun": True,
        "status": status,
        "final_assignment_available": publishable,
        "publishable_assignment_available": publishable,
        "publishable_recovery": publishable,
        "response_proto_hash": "deadbeef" if publishable else ("deadbeef" if status != "UNKNOWN" else ""),
        "assignment_nonpublishable": not publishable,
        "final_schedule_policy_pass": True if publishable else None,
        "consistency_issue_count": 0 if publishable else None,
        "time_to_first_solution_seconds": 10.0 if publishable else None,
        "solver_wall_time_seconds": 12.0,
        "end_to_end_scenario_runtime_seconds": 20.0,
        "primary_satisfaction_rate": 0.98 if publishable else None,
        "logical_full_rate": 0.9 if publishable else None,
        "total_logical_gap": 200 if publishable else None,
        "hamming_distance": 150 if publishable else None,
        "changed_students": 70 if publishable else None,
        "timing_diagnostic_valid": True,
        "student_count": 1,
        "primary_assigned": 1 if publishable else None,
        "logical_fully_scheduled_students": 1 if publishable else None,
        "gap_over_1_students": 0 if publishable else None,
        "below_five_students": 0 if publishable else None,
        "policy_violation_count": 0 if publishable else None,
    }
    base.update(overrides)
    return base


def test_percentile_and_stats_helpers() -> None:
    assert _percentile([], 0.9) is None
    assert _percentile([5.0], 0.9) == 5.0
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert _percentile(values, 0.0) == 1.0
    assert _percentile(values, 1.0) == 5.0
    stats = _stats([{"k": 1.0}, {"k": 2.0}, {"k": None}], "k")
    assert stats["count"] == 2
    assert stats["median"] == 1.5


def test_missing_assignment_scenarios_are_excluded_from_quality_median_but_denominator_is_reported() -> None:
    runner = _runner()
    rows = [_row(f"s{i}", publishable=True) for i in range(5)]
    rows += [_row(f"m{i}", status="UNKNOWN", publishable=False, primary_satisfaction_rate=None, logical_full_rate=None, total_logical_gap=None, hamming_distance=None, changed_students=None, time_to_first_solution_seconds=None, solver_wall_time_seconds=None) for i in range(3)]
    aggregate = runner._aggregate_summary(rows, [], {})
    assert aggregate["attempted_count"] == 8
    assert aggregate["publishable_assignment_count"] == 5
    assert aggregate["publishable_assignment_denominator"] == 8
    assert aggregate["primary_satisfaction_rate"]["count"] == 5
    assert aggregate["quality_metric_denominator"] == 5


def test_seven_of_twelve_publishable_is_a_pass_gate() -> None:
    runner = _runner()
    rows = [_row(STABLE_SCENARIO_ID, publishable=True)]
    rows += [_row(f"normal_dev_{i:02d}", publishable=True) for i in range(1, 7)]
    rows += [_row(f"normal_dev_{i:02d}", status="UNKNOWN", publishable=False) for i in range(7, 12)]
    assert len(rows) == 12
    gate = runner._success_gate(rows, [])
    assert gate["gate"] == "PASS"
    assert gate["publishable_count"] == 7
    assert gate["ready_for_stress_development"] is True
    assert gate["ready_for_holdout"] is False


def test_six_of_twelve_publishable_is_a_fail_gate() -> None:
    runner = _runner()
    rows = [_row(STABLE_SCENARIO_ID, publishable=True)]
    rows += [_row(f"normal_dev_{i:02d}", publishable=True) for i in range(1, 6)]
    rows += [_row(f"normal_dev_{i:02d}", status="UNKNOWN", publishable=False) for i in range(6, 12)]
    assert len(rows) == 12
    gate = runner._success_gate(rows, [])
    assert gate["gate"] == "FAIL"
    assert gate["publishable_count"] == 6
    assert gate["ready_for_stress_development"] is False
    assert gate["ready_for_holdout"] is False


def test_ready_for_holdout_is_always_false_even_on_pass() -> None:
    runner = _runner()
    rows = [_row(STABLE_SCENARIO_ID, publishable=True)]
    rows += [_row(f"normal_dev_{i:02d}", publishable=True) for i in range(1, 12)]
    gate = runner._success_gate(rows, [])
    assert gate["gate"] == "PASS"
    assert gate["ready_for_holdout"] is False


def test_critical_correctness_failure_forces_fail_gate() -> None:
    runner = _runner()
    rows = [_row(STABLE_SCENARIO_ID, publishable=True)]
    rows += [_row(f"normal_dev_{i:02d}", publishable=True) for i in range(1, 12)]
    failures = [{"scenario_id": "normal_dev_01", "failure_type": "critical_correctness_failure", "message": "x"}]
    gate = runner._success_gate(rows, failures)
    assert gate["gate"] == "FAIL"
    assert "critical_correctness_failures_present" in gate["blocking_reasons"]


# ---------------------------------------------------------------------------
# 7. Phase C Reporting Integrity Audit
# ---------------------------------------------------------------------------


@pytest.mark.external_artifact
def test_imported_stable_row_has_publishable_fields_true(stable_probe_artifact: Path) -> None:
    runner = _runner()
    imported = _import_stable_reference(runner.manifest, runner.config_dir, stable_probe_artifact)
    row = imported["row"]
    assert row["publishable_assignment_available"] is True
    assert row["publishable_recovery"] is True
    assert row["assignment_nonpublishable"] is False
    assert row["final_schedule_policy_pass"] is True
    assert row["consistency_issue_count"] == 0
    assert row["result_origin"] == "imported_frozen_probe"
    assert row["solver_rerun"] is False


@pytest.mark.external_artifact
def test_imported_and_solved_rows_share_the_required_field_set(stable_probe_artifact: Path) -> None:
    runner = _runner()
    imported = _import_stable_reference(runner.manifest, runner.config_dir, stable_probe_artifact)
    imported_row = imported["row"]

    result = run_fair_cp_sat_solver(
        _complete_fixture(),
        seed=20260630,
        math_fallback_rules=fallback_rules(),
        math_course_ids=math_ids(),
        max_time_seconds_per_stage=2,
        bootstrap_time_seconds=0,
        max_total_time_seconds=5,
        num_search_workers=1,
        use_feasibility_bootstrap=False,
        use_constrained_first_hint=False,
        logical_schedule_completion_enabled=False,
        internal_feasibility_hint_strategy="constrained_first",
        internal_repair_time_seconds=2,
        internal_repair_objective_strategy="hamming_to_constrained_first",
        stop_after_first_valid_solution=True,
    )
    metrics = _result_metrics(result, _complete_fixture())
    solved_row = {
        **metrics,
        "scenario_id": "normal_dev_01",
        "scenario_group": "normal",
        "result_origin": "solved",
        "solver_rerun": True,
        "status": result.solve_status.value,
        "final_assignment_available": bool(result.student_outcomes),
        "response_proto_hash": "deadbeef",
    }
    publishable = _is_publishable(
        solved_row["status"], solved_row["final_assignment_available"], solved_row["response_proto_hash"],
        metrics.get("final_schedule_policy_pass"), metrics.get("consistency_issue_count"),
    )
    solved_row["publishable_assignment_available"] = publishable
    solved_row["publishable_recovery"] = publishable
    _validate_row_schema(imported_row, "imported")
    _validate_row_schema(solved_row, "solved")
    missing_from_solved = set(_normal_evaluation_required_fields()) - set(solved_row)
    missing_from_imported = set(_normal_evaluation_required_fields()) - set(imported_row)
    assert not missing_from_solved
    assert not missing_from_imported
    for field in _normal_evaluation_required_fields():
        assert type(imported_row[field]) is type(solved_row[field]) or imported_row[field] is None or solved_row[field] is None


def _normal_evaluation_required_fields():
    return normal_evaluation_runner_module._REQUIRED_ROW_FIELDS


def test_missing_publishable_field_fails_closed_instead_of_defaulting() -> None:
    incomplete = {
        "scenario_id": "normal_dev_01", "scenario_group": "normal", "result_origin": "solved",
        "solver_rerun": True, "status": "FEASIBLE", "final_assignment_available": True,
        # publishable_assignment_available intentionally omitted
        "publishable_recovery": True, "final_schedule_policy_pass": True,
        "consistency_issue_count": 0, "assignment_nonpublishable": False,
        "response_proto_hash": "deadbeef",
    }
    with pytest.raises(CpSatEvaluationError, match="missing required fields"):
        _validate_row_schema(incomplete, "normal_dev_01")


def test_repair_legacy_row_derives_publishable_from_existing_fields() -> None:
    # Mirrors the real pre-fix raw artifact: the imported stable row lacked
    # publishable_assignment_available/publishable_recovery, but carried
    # every field needed to derive them correctly.
    legacy = {
        "scenario_id": "normal_dev_reference_2026", "status": "FEASIBLE",
        "final_assignment_available": True, "response_proto_hash": "deadbeef",
        "final_schedule_policy_pass": True, "consistency_issue_count": 0,
    }
    repaired = _repair_legacy_row(legacy)
    assert repaired["publishable_assignment_available"] is True
    assert repaired["publishable_recovery"] is True
    assert repaired["global_infeasibility_proven"] is False
    assert repaired["solver_global_infeasibility_proven"] is False
    # the original dict must not be mutated in place
    assert "publishable_assignment_available" not in legacy


def test_repair_legacy_row_fails_closed_when_source_fields_also_missing() -> None:
    legacy = {
        "scenario_id": "normal_dev_reference_2026", "status": "FEASIBLE",
        "final_assignment_available": True, "response_proto_hash": "deadbeef",
        # final_schedule_policy_pass and consistency_issue_count are themselves missing
    }
    with pytest.raises(CpSatEvaluationError, match="fail-closed, not defaulted"):
        _repair_legacy_row(legacy)


def test_strict_dominance_or_winner_wording_is_absent_from_source() -> None:
    source = Path(inspect.getfile(normal_evaluation_runner_module)).read_text(encoding="utf-8")
    lowered = source.lower()
    assert "strictly dominat" not in lowered
    assert "dominance" not in lowered
    assert '"winner"' not in lowered


def test_paired_comparison_uses_fixed_policy_compliance_tradeoff_label() -> None:
    runner = _runner()
    rows = [_row(STABLE_SCENARIO_ID, publishable=True, student_count=2630)]
    paired = runner._paired_vs_constrained_first(rows)
    for item in paired:
        assert item["comparison_interpretation"] == "policy_compliance_tradeoff"


def test_aggregate_and_gate_report_fixed_comparison_interpretation() -> None:
    runner = _runner()
    rows = [_row(STABLE_SCENARIO_ID, publishable=True)]
    rows += [_row(f"normal_dev_{i:02d}", publishable=True) for i in range(1, 12)]
    aggregate = runner._aggregate_summary(rows, [], {})
    gate = runner._success_gate(rows, [])
    assert aggregate["comparison_interpretation"] == "policy_compliance_tradeoff"
    assert gate["comparison_interpretation"] == "policy_compliance_tradeoff"


def test_timing_audit_flags_stage_wall_time_exceeding_configured_limit() -> None:
    row = {
        "time_to_first_solution_seconds": None,
        "solver_wall_time_seconds": 53.5,
        "end_to_end_scenario_runtime_seconds": 118.2,
    }
    anomalous_stage = {"wall_time_seconds": 1114.1, "effective_time_limit_seconds": 289.46}
    diagnostic = _audit_row_timing(row, anomalous_stage)
    assert diagnostic["timing_diagnostic_valid"] is False
    assert "stage_reported_wall_time_exceeds_configured_time_limit" in diagnostic["timing_anomaly_reasons"]
    # the raw anomalous value must still be preserved for audit, not discarded
    assert diagnostic["stage_reported_wall_time_seconds"] == 1114.1


def test_timing_audit_accepts_sane_timing() -> None:
    row = {
        "time_to_first_solution_seconds": 70.0,
        "solver_wall_time_seconds": 75.0,
        "end_to_end_scenario_runtime_seconds": 130.0,
    }
    sane_stage = {"wall_time_seconds": 75.5, "effective_time_limit_seconds": 289.0}
    diagnostic = _audit_row_timing(row, sane_stage)
    assert diagnostic["timing_diagnostic_valid"] is True
    assert diagnostic["timing_anomaly_reasons"] == []


def test_timing_audit_flags_negative_time_to_first_solution() -> None:
    row = {"time_to_first_solution_seconds": -5.0, "solver_wall_time_seconds": 10.0, "end_to_end_scenario_runtime_seconds": 20.0}
    diagnostic = _audit_row_timing(row, None)
    assert diagnostic["timing_diagnostic_valid"] is False
    assert "negative_time_to_first_solution" in diagnostic["timing_anomaly_reasons"]


def test_timing_audit_flags_first_solution_greater_than_solver_wall_time() -> None:
    row = {"time_to_first_solution_seconds": 200.0, "solver_wall_time_seconds": 50.0, "end_to_end_scenario_runtime_seconds": 300.0}
    diagnostic = _audit_row_timing(row, None)
    assert diagnostic["timing_diagnostic_valid"] is False
    assert "solver_wall_time_less_than_time_to_first_solution" in diagnostic["timing_anomaly_reasons"]


def test_timing_audit_flags_end_to_end_less_than_solver_wall_time() -> None:
    row = {"time_to_first_solution_seconds": None, "solver_wall_time_seconds": 500.0, "end_to_end_scenario_runtime_seconds": 10.0}
    diagnostic = _audit_row_timing(row, None)
    assert diagnostic["timing_diagnostic_valid"] is False
    assert "end_to_end_runtime_less_than_solver_wall_time" in diagnostic["timing_anomaly_reasons"]


def test_invalid_timing_scenario_is_excluded_from_aggregate_timing_stats() -> None:
    runner = _runner()
    valid_rows = [_row(f"normal_dev_{i:02d}", publishable=True) for i in range(1, 4)]
    invalid_row = _row("normal_dev_04", publishable=False, status="UNKNOWN", timing_diagnostic_valid=False,
                        solver_wall_time_seconds=1114.1, end_to_end_scenario_runtime_seconds=118.2)
    rows = [_row(STABLE_SCENARIO_ID, publishable=True)] + valid_rows + [invalid_row]
    aggregate = runner._aggregate_summary(rows, [], {})
    assert aggregate["timing_excluded_scenario_count"] == 1
    # the excluded row's anomalous wall time must not have polluted the max
    assert aggregate["solver_wall_time_seconds"]["max"] < 1000


def test_infeasibility_scope_confirmed_for_well_formed_infeasible_stage() -> None:
    row = {
        "scenario_id": "normal_dev_01", "status": "INFEASIBLE", "response_proto_hash": "deadbeef",
        "final_assignment_available": False, "global_infeasibility_proven": True,
        "solver_global_infeasibility_proven": True,
    }
    stage = {"stage_name": "internal_repair_feasibility", "fixed_prior_objectives": [], "conditional_on_unproven_incumbent": False}
    entry = _infeasibility_scope_row(row, stage)
    assert entry["scope_confirmed"] is True


def test_infeasibility_scope_rejects_fixed_objective_conditioned_stage() -> None:
    row = {
        "scenario_id": "normal_dev_01", "status": "INFEASIBLE", "response_proto_hash": "deadbeef",
        "final_assignment_available": False, "global_infeasibility_proven": True,
        "solver_global_infeasibility_proven": True,
    }
    stage = {"stage_name": "internal_repair_feasibility", "fixed_prior_objectives": [{"stage_name": "math_coverage", "value": 1}], "conditional_on_unproven_incumbent": False}
    entry = _infeasibility_scope_row(row, stage)
    assert entry["scope_confirmed"] is False


def test_unknown_status_is_never_reclassified_as_infeasible() -> None:
    row = {
        "scenario_id": "normal_dev_08", "status": "UNKNOWN", "response_proto_hash": "deadbeef",
        "final_assignment_available": False, "global_infeasibility_proven": False,
        "solver_global_infeasibility_proven": False,
    }
    stage = {"stage_name": "internal_repair_feasibility", "fixed_prior_objectives": [], "conditional_on_unproven_incumbent": False}
    entry = _infeasibility_scope_row(row, stage)
    # UNKNOWN must never satisfy the INFEASIBLE scope confirmation
    assert entry["scope_confirmed"] is False
    assert row["status"] != "INFEASIBLE"


# ---------------------------------------------------------------------------
# Audited rebuild: synthetic raw artifact, no solver invocation, determinism
# ---------------------------------------------------------------------------


def _build_synthetic_raw_artifact(root: Path) -> None:
    scenario_ids = [STABLE_SCENARIO_ID, "normal_dev_01", "normal_dev_02"]
    _write_json(root / "run_manifest.json", {
        "status": "completed",
        "selected_scenario_ids": scenario_ids,
        "source_info": {"artifact_dir": "synthetic"},
    })
    _write_json(root / "evaluation_manifest_snapshot.json", {
        "source_normal_suite": {"artifact_dir": str(root / "normal_suite_stub")},
    })
    _write_json(root / "failures.json", {"failures": [], "unexpected_failure_count": 0})

    rows = {
        STABLE_SCENARIO_ID: _row(STABLE_SCENARIO_ID, publishable=True, result_origin="imported_frozen_probe", solver_rerun=False, student_count=1),
        "normal_dev_01": _row("normal_dev_01", publishable=False, status="INFEASIBLE", student_count=1),
        "normal_dev_02": _row("normal_dev_02", publishable=True, student_count=1),
    }
    stages = {
        STABLE_SCENARIO_ID: {"stage_name": "internal_repair_feasibility", "wall_time_seconds": 75.75, "effective_time_limit_seconds": 287.69, "fixed_prior_objectives": [], "conditional_on_unproven_incumbent": False},
        "normal_dev_01": {"stage_name": "internal_repair_feasibility", "wall_time_seconds": 1.5, "effective_time_limit_seconds": 289.0, "fixed_prior_objectives": [], "conditional_on_unproven_incumbent": False},
        "normal_dev_02": {"stage_name": "internal_repair_feasibility", "wall_time_seconds": 1114.1, "effective_time_limit_seconds": 289.46, "fixed_prior_objectives": [], "conditional_on_unproven_incumbent": False},
    }
    rows["normal_dev_01"]["global_infeasibility_proven"] = True
    rows["normal_dev_01"]["solver_global_infeasibility_proven"] = True
    # Reproduce the real pre-fix defect: the imported stable row's raw
    # scenario_summary.json lacked these two fields entirely, so the rebuild
    # must repair them from the row's other already-persisted fields.
    del rows[STABLE_SCENARIO_ID]["publishable_assignment_available"]
    del rows[STABLE_SCENARIO_ID]["publishable_recovery"]
    for scenario_id in scenario_ids:
        scenario_dir = root / "scenarios" / scenario_id
        scenario_dir.mkdir(parents=True)
        _write_json(scenario_dir / "scenario_summary.json", {
            "status": "completed_with_assignment" if rows[scenario_id]["final_assignment_available"] else "completed_without_assignment",
            "result": rows[scenario_id],
        })
        _write_json(scenario_dir / "stage_trace.json", {"scenario_id": scenario_id, "stages": [stages[scenario_id]]})
    _write_checksums(root)


def test_audit_rebuild_does_not_invoke_the_solver(tmp_path, monkeypatch) -> None:
    def _boom(*_a, **_k):
        raise AssertionError("audited rebuild must never call run_fair_cp_sat_solver")
    monkeypatch.setattr(normal_evaluation_runner_module, "run_fair_cp_sat_solver", _boom)

    source = tmp_path / "raw"
    source.mkdir()
    _build_synthetic_raw_artifact(source)
    output = tmp_path / "audited"

    summary = rebuild_audited_normal_evaluation(source, output)
    assert summary["aggregate_summary"]["scenarios"] == 3


def test_audit_rebuild_writes_expected_files_and_preserves_source(tmp_path) -> None:
    source = tmp_path / "raw2"
    source.mkdir()
    _build_synthetic_raw_artifact(source)
    before_sums = (source / "SHA256SUMS.txt").read_text(encoding="utf-8")

    output = tmp_path / "audited2"
    rebuild_audited_normal_evaluation(source, output)

    for name in (
        "scenario_results.csv", "timing_results.csv", "paired_recovery_vs_constrained_first.csv",
        "aggregate_summary.json", "readiness_assessment.json", "timing_anomaly_audit.json",
        "infeasibility_scope_audit.json", "provenance.json", "SHA256SUMS.txt",
    ):
        assert (output / name).is_file(), name

    provenance = _read_json(output / "provenance.json")
    assert provenance["no_new_solver_runs"] is True
    assert provenance["normal_solver_runs_added"] == 0
    assert provenance["stress_runs"] == 0
    assert provenance["holdout_runs"] == 0
    assert provenance["source_artifact_path"] == str(source)

    timing_audit = _read_json(output / "timing_anomaly_audit.json")
    assert "normal_dev_02" in timing_audit["invalid_scenario_ids"]

    infeasibility_audit = _read_json(output / "infeasibility_scope_audit.json")
    assert infeasibility_audit["all_scopes_confirmed"] is True

    tree_hash, files, directories, total_bytes = _verify_sha256_manifest(output)
    assert files == 9  # the 9 required files, all checksummed and verified

    after_sums = (source / "SHA256SUMS.txt").read_text(encoding="utf-8")
    assert before_sums == after_sums
    _verify_sha256_manifest(source)  # source artifact must still verify cleanly


def test_audit_rebuild_refuses_to_overwrite_nonempty_output(tmp_path) -> None:
    source = tmp_path / "raw3"
    source.mkdir()
    _build_synthetic_raw_artifact(source)
    output = tmp_path / "audited3"
    output.mkdir()
    (output / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(CpSatEvaluationError, match="non-empty"):
        rebuild_audited_normal_evaluation(source, output)


def test_audit_rebuild_is_deterministic(tmp_path) -> None:
    source = tmp_path / "raw4"
    source.mkdir()
    _build_synthetic_raw_artifact(source)

    output_a = tmp_path / "audited_a"
    output_b = tmp_path / "audited_b"
    rebuild_audited_normal_evaluation(source, output_a)
    rebuild_audited_normal_evaluation(source, output_b)

    assert _read_json(output_a / "aggregate_summary.json") == _read_json(output_b / "aggregate_summary.json")
    assert _read_json(output_a / "readiness_assessment.json") == _read_json(output_b / "readiness_assessment.json")
    assert (output_a / "scenario_results.csv").read_text() == (output_b / "scenario_results.csv").read_text()
