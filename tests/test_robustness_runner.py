from __future__ import annotations

import copy
import json

import pandas as pd
import pytest

from src.allocation import canonicalize_allocation_input
from src.input_difficulty import build_input_difficulty
from src.robustness_runner import (
    ALGORITHM_LABELS,
    DEFAULT_SUITE_PATH,
    GREEDY_ALGORITHMS,
    RobustnessRunnerError,
    ScenarioManifestError,
    _difficulty_row,
    _failed_scenario_rows,
    _hash_input_files,
    _load_cached_scenario,
    build_aggregate_summary,
    build_paired_comparison,
    load_scenario_suite,
    main,
    run_robustness_suite,
    scenario_suite_hash,
    validate_scenario_suite,
)


def _small_input():
    students = pd.DataFrame(
        [("S1", 9, 2, "none", "false", "", "")],
        columns=[
            "student_id",
            "grade",
            "target_course_count",
            "unscheduled_preference",
            "priority_protected",
            "priority_reason",
            "priority_valid_school_year",
        ],
    )
    requests = pd.DataFrame(
        [
            ("S1", "CORE", "primary", "", "", ""),
            ("S1", "ELECTIVE", "primary", "", "", ""),
            ("S1", "ALT", "alternate", 1, "alternate", ""),
        ],
        columns=[
            "student_id",
            "course_id",
            "request_type",
            "request_rank",
            "request_group",
            "must_share_block_id",
        ],
    )
    sections = pd.DataFrame(
        [
            ("SEC_CORE", "CORE", "P1", "", "full_year", 1, "", "CORE_1", "CORE", ""),
            ("SEC_ELECTIVE", "ELECTIVE", "P2", "", "full_year", 1, "", "ELECTIVE_1", "ELECTIVE", ""),
            ("SEC_ALT", "ALT", "P3", "", "full_year", 1, "", "ALT_1", "ALT", ""),
        ],
        columns=[
            "section_id",
            "course_id",
            "period_1",
            "period_2",
            "semester",
            "capacity",
            "block_id",
            "linked_section_group_id",
            "logical_block_id",
            "semester_content",
        ],
    )
    catalog = pd.DataFrame(
        [("CORE", 1, "standard"), ("ELECTIVE", 1, "standard"), ("ALT", 1, "standard")],
        columns=["course_id", "periods_required", "schedule_structure"],
    )
    return canonicalize_allocation_input(students, requests, sections, catalog)


def _suite_payload():
    return json.loads(DEFAULT_SUITE_PATH.read_text(encoding="utf-8"))


def _fake_scenario_result(spec):
    rows = []
    for alias, label in ALGORITHM_LABELS.items():
        rows.append(
            {
                "scenario_id": spec.scenario_id,
                "scenario_family": spec.scenario_family,
                "split": spec.split,
                "row_type": "overall",
                "grade": "",
                "algorithm": alias,
                "algorithm_name": label,
                "algorithm_label": label,
                "completed": True,
                "status": "completed",
                "primary_assigned": 10 + len(alias),
                "primary_unmet": 1,
                "primary_satisfaction_rate": 0.9,
                "logical_full_students": 8,
                "logical_full_rate": 0.8,
                "target_logical_course_count": 20,
                "assigned_logical_course_count": 19,
                "total_logical_gap": 1,
                "gap_students": 2,
                "gap_over_1_students": 0,
                "below_five_students": 0,
                "ordinary_violations": 0,
                "protected_violations": 0,
                "high_demand_violations": 0,
                "policy_violation_count": 0,
                "final_schedule_policy_pass": True,
                "consistency_issue_count": 0,
                "section_over_capacity_count": 0,
                "period_conflict_count": 2,
                "duplicate_logical_course_rejection_count": 0,
                "total_alternates_assigned": 3,
                "runtime_seconds": 0.1,
            }
        )
    return {"scenario_rows": rows, "input_difficulty_row": {"scenario_id": spec.scenario_id, "students": 1}}


def test_current_suite_has_expected_development_and_holdout_counts() -> None:
    suite = load_scenario_suite()
    assert len(suite.scenarios) == 20
    assert sum(item.split == "development" for item in suite.scenarios) == 12
    assert sum(item.split == "holdout" for item in suite.scenarios) == 8


def test_reference_seed_roles_are_frozen_separately() -> None:
    reference = load_scenario_suite().scenarios[0]
    assert (reference.data_generation_seed, reference.section_planning_seed) == (2026, 2026)
    assert reference.algorithm_seed == 20260630
    assert reference.generation_scenario_id == "stable_year"


def test_all_development_algorithm_seeds_are_solver_seed() -> None:
    suite = load_scenario_suite()
    assert {item.algorithm_seed for item in suite.scenarios if item.split == "development"} == {20260630}


def test_suite_hash_is_deterministic() -> None:
    suite = load_scenario_suite()
    assert scenario_suite_hash(suite) == scenario_suite_hash(load_scenario_suite())


def test_dry_run_selects_development_in_manifest_order(tmp_path) -> None:
    result = run_robustness_suite(output_dir=tmp_path / "artifacts", dry_run=True, max_scenarios=3)
    assert result.scenario_ids == ("normal_dev_reference_2026", "normal_dev_01", "normal_dev_02")
    assert result.algorithms == GREEDY_ALGORITHMS
    assert not (tmp_path / "artifacts").exists()


def test_dry_run_selects_one_named_scenario(tmp_path) -> None:
    result = run_robustness_suite(
        output_dir=tmp_path / "artifacts", dry_run=True, scenario_id="normal_dev_04"
    )
    assert result.scenario_ids == ("normal_dev_04",)


def test_missing_named_scenario_fails_closed(tmp_path) -> None:
    with pytest.raises(RobustnessRunnerError, match="is not in split"):
        run_robustness_suite(output_dir=tmp_path / "artifacts", dry_run=True, scenario_id="missing")


def test_holdout_requires_explicit_confirmation(tmp_path) -> None:
    with pytest.raises(RobustnessRunnerError, match="requires confirm_holdout"):
        run_robustness_suite(output_dir=tmp_path / "artifacts", split="holdout", dry_run=True)


def test_holdout_confirmation_selects_holdout(tmp_path) -> None:
    result = run_robustness_suite(
        output_dir=tmp_path / "artifacts",
        split="holdout",
        dry_run=True,
        confirm_holdout_evaluation=True,
        max_scenarios=1,
    )
    assert result.scenario_ids == ("normal_holdout_01",)


def test_cp_sat_is_rejected_by_robustness_runner(tmp_path) -> None:
    with pytest.raises(RobustnessRunnerError, match="Greedy algorithms only"):
        run_robustness_suite(output_dir=tmp_path / "artifacts", dry_run=True, algorithms=("cp_sat",))


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda p: p["scenarios"].__setitem__(1, {**p["scenarios"][1], "scenario_id": p["scenarios"][0]["scenario_id"]}), "Duplicate scenario_id"),
        (lambda p: p["scenarios"].__setitem__(1, {**p["scenarios"][1], "data_generation_seed": 2026, "section_planning_seed": 2026}), "pair is not unique"),
        (lambda p: p["scenarios"].__setitem__(12, {**p["scenarios"][12], "tuning_allowed": True}), "cannot allow tuning"),
        (lambda p: p["scenarios"].__setitem__(1, {**p["scenarios"][1], "split": "stress"}), "split must be"),
        (lambda p: p["scenarios"][1].pop("notes"), "is missing"),
        (lambda p: p["scenarios"][0].__setitem__("expected_reference_fingerprint", None), "exactly one"),
        (lambda p: p["scenarios"].__setitem__(1, {**p["scenarios"][1], "expected_reference_fingerprint": p["scenarios"][0]["expected_reference_fingerprint"]}), "exactly one"),
        (lambda p: p["scenarios"].__setitem__(0, {**p["scenarios"][0], "expected_reference_fingerprint": {"students": 1}}), "fields must be exactly"),
    ],
)
def test_manifest_validation_rejects_malformed_suite(mutation, message) -> None:
    payload = _suite_payload()
    mutation(payload)
    with pytest.raises(ScenarioManifestError, match=message):
        validate_scenario_suite(payload)


def test_difficulty_uses_canonical_counts() -> None:
    difficulty = build_input_difficulty(_small_input())
    assert difficulty["scale"]["students"] == 1
    assert difficulty["scale"]["logical_requests"] == 3
    assert difficulty["scale"]["logical_primaries"] == 2
    assert difficulty["scale"]["alternates"] == 1
    assert difficulty["scale"]["logical_sections"] == 3
    assert difficulty["scale"]["section_rows"] == 3
    assert difficulty["scale"]["candidate_edges"] == 3
    assert len(difficulty["scale"]["canonical_input_hash"]) == 64


def test_difficulty_reports_candidate_flexibility_and_periods() -> None:
    difficulty = build_input_difficulty(_small_input())
    assert difficulty["request_flexibility"]["primary_candidate_sections"]["min"] == 1
    assert difficulty["period_candidate_structure"]["requests_with_candidates_in_only_one_period"] == 2


def test_difficulty_reports_capacity_ratio_and_shortfall() -> None:
    difficulty = build_input_difficulty(_small_input())
    demand = difficulty["demand_capacity"]
    assert demand["courses_with_primary_demand_over_capacity"] == 0
    assert demand["maximum_primary_demand_capacity_ratio"] == 1.0
    assert demand["total_capacity_only_primary_shortfall"] == 0
    assert "not a proof" in demand["capacity_only_shortfall_definition"]


def test_difficulty_is_json_serializable() -> None:
    json.dumps(build_input_difficulty(_small_input()), sort_keys=True)


def test_difficulty_row_flattens_selected_descriptors() -> None:
    difficulty = build_input_difficulty(_small_input())
    spec = load_scenario_suite().scenarios[0]
    row = _difficulty_row(spec, difficulty)
    assert row["scenario_id"] == spec.scenario_id
    assert row["candidate_edges"] == 3


def test_aggregate_summary_has_distribution_statistics() -> None:
    rows = _fake_scenario_result(load_scenario_suite().scenarios[0])["scenario_rows"]
    summary = build_aggregate_summary(rows)
    assert summary["algorithms"]["random"]["completed_count"] == 1
    assert summary["algorithms"]["random"]["metrics"]["primary_assigned"]["median"] == 16


def test_aggregate_summary_keeps_failed_rows_out_of_numeric_stats() -> None:
    rows = _fake_scenario_result(load_scenario_suite().scenarios[0])["scenario_rows"]
    failed = dict(rows[0], scenario_id="failed", completed=False, status="failed")
    summary = build_aggregate_summary([*rows, failed])
    random_summary = summary["algorithms"]["random"]
    assert random_summary["scenario_count"] == 2
    assert random_summary["completed_count"] == 1
    assert random_summary["failed_count"] == 1
    assert random_summary["metrics"]["primary_assigned"]["count"] == 1


def test_failed_scenario_rows_are_explicit_and_have_no_fake_metrics() -> None:
    spec = load_scenario_suite().scenarios[0]
    rows = _failed_scenario_rows(spec, "fixture failure", ("random", "constrained"))
    assert len(rows) == 2
    assert all(row["completed"] is False for row in rows)
    assert all(row["error"] == "fixture failure" for row in rows)
    assert all("primary_assigned" not in row for row in rows)


def test_paired_comparison_matches_by_scenario() -> None:
    rows = _fake_scenario_result(load_scenario_suite().scenarios[0])["scenario_rows"]
    paired = build_paired_comparison(rows)
    assert len(paired) == 3
    assert {row["baseline_algorithm"] for row in paired} == {"random", "fcfs", "grade_priority"}
    assert {row["comparison_algorithm"] for row in paired} == {"constrained"}
    constrained = next(row for row in paired if row["comparison_algorithm"] == "constrained")
    assert constrained["primary_assigned_delta"] == len("constrained") - len("random")


def test_input_hash_is_stable_and_includes_csv_names(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "a.csv").write_text("x\n1\n", encoding="utf-8")
    (second / "a.csv").write_text("x\n1\n", encoding="utf-8")
    assert _hash_input_files(first) == _hash_input_files(second)
    (second / "b.csv").write_text("x\n2\n", encoding="utf-8")
    assert _hash_input_files(first) != _hash_input_files(second)


def test_output_runner_writes_root_artifacts_without_running_real_algorithms(tmp_path, monkeypatch) -> None:
    config = tmp_path / "config"
    templates = tmp_path / "templates"
    config.mkdir()
    templates.mkdir()
    (config / "config.csv").write_text("key,value\na,b\n", encoding="utf-8")
    (templates / "template.csv").write_text("key,value\na,b\n", encoding="utf-8")
    monkeypatch.setattr("src.robustness_runner._run_scenario", lambda spec, **kwargs: _fake_scenario_result(spec))
    output = tmp_path / "artifacts"
    result = run_robustness_suite(
        output_dir=output,
        max_scenarios=1,
        config_dir=config,
        templates_dir=templates,
    )
    assert result.scenario_ids == ("normal_dev_reference_2026",)
    for filename in (
        "suite_manifest_snapshot.json",
        "run_manifest.json",
        "scenario_results.csv",
        "input_difficulty.csv",
        "aggregate_summary.json",
        "paired_algorithm_comparison.csv",
    ):
        assert (output / filename).is_file()


def test_output_directory_nonempty_requires_resume(tmp_path) -> None:
    output = tmp_path / "artifacts"
    output.mkdir()
    (output / "unrelated.txt").write_text("do not overwrite", encoding="utf-8")
    with pytest.raises(RobustnessRunnerError, match="non-empty"):
        run_robustness_suite(output_dir=output, dry_run=False, max_scenarios=1, config_dir=tmp_path, templates_dir=tmp_path)


def test_cached_scenario_requires_completed_status(tmp_path) -> None:
    scenario_dir = tmp_path / "scenario"
    scenario_dir.mkdir()
    spec = load_scenario_suite().scenarios[0]
    (scenario_dir / "scenario_result.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RobustnessRunnerError, match="provenance mismatch"):
        _load_cached_scenario(scenario_dir, spec, suite_hash="x", config_fingerprint="y", resume=True)


def test_cached_scenario_provenance_mismatch_fails_closed(tmp_path) -> None:
    scenario_dir = tmp_path / "scenario"
    scenario_dir.mkdir()
    spec = load_scenario_suite().scenarios[0]
    payload = {
        "status": "completed",
        "suite_hash": "wrong",
        "scenario_spec_hash": "wrong",
        "config_templates_fingerprint": "wrong",
        "scenario_rows": [],
        "input_difficulty_row": {},
    }
    (scenario_dir / "scenario_result.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RobustnessRunnerError, match="provenance mismatch"):
        _load_cached_scenario(scenario_dir, spec, suite_hash="x", config_fingerprint="y", resume=True)


def test_cached_scenario_can_be_reused(tmp_path) -> None:
    scenario_dir = tmp_path / "scenario"
    scenario_dir.mkdir()
    spec = load_scenario_suite().scenarios[0]
    suite_hash = scenario_suite_hash(load_scenario_suite())
    config_hash = "config"
    payload = {
        "status": "completed",
        "suite_hash": suite_hash,
        "scenario_spec_hash": __import__("src.robustness_runner", fromlist=["_sha256_json"])._sha256_json(spec.to_dict()),
        "config_templates_fingerprint": config_hash,
        "scenario_rows": [{"row_type": "overall"}],
        "input_difficulty_row": {"scenario_id": spec.scenario_id},
    }
    (scenario_dir / "scenario_result.json").write_text(json.dumps(payload), encoding="utf-8")
    assert _load_cached_scenario(scenario_dir, spec, suite_hash=suite_hash, config_fingerprint=config_hash, resume=True) == payload


def test_cli_dry_run_returns_zero(capsys) -> None:
    assert main(["--dry-run", "--max-scenarios", "1"]) == 0
    assert "Robustness runner PASS" in capsys.readouterr().out


def test_cli_holdout_without_confirmation_returns_one(capsys) -> None:
    assert main(["--dry-run", "--split", "holdout"]) == 1
    assert "requires confirm_holdout" in capsys.readouterr().out


def test_reference_fingerprint_has_all_canonical_fields() -> None:
    reference = load_scenario_suite().scenarios[0].expected_reference_fingerprint
    assert set(reference) == {
        "students",
        "logical_requests",
        "logical_primaries",
        "alternates",
        "logical_sections",
        "section_rows",
        "candidate_edges",
        "canonical_input_hash",
    }


def test_holdout_scenarios_are_not_tuning_allowed() -> None:
    assert all(not item.tuning_allowed for item in load_scenario_suite().scenarios if item.split == "holdout")


def test_disabled_scenario_is_not_selected(tmp_path) -> None:
    payload = _suite_payload()
    payload["scenarios"][1]["enabled"] = False
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    result = run_robustness_suite(suite_path=path, output_dir=tmp_path / "out", dry_run=True, max_scenarios=2)
    assert result.scenario_ids == ("normal_dev_reference_2026", "normal_dev_02")


def test_max_scenarios_must_be_positive(tmp_path) -> None:
    with pytest.raises(RobustnessRunnerError, match="must be positive"):
        run_robustness_suite(output_dir=tmp_path / "out", dry_run=True, max_scenarios=0)


def test_non_dry_run_requires_output_dir() -> None:
    with pytest.raises(RobustnessRunnerError, match="output_dir is required"):
        run_robustness_suite(dry_run=False, max_scenarios=1)


def test_algorithm_labels_are_real_baseline_names() -> None:
    assert set(ALGORITHM_LABELS) == set(GREEDY_ALGORITHMS)
    assert "cp_sat" not in ALGORITHM_LABELS


def test_suite_validation_returns_immutable_scenario_collection() -> None:
    suite = load_scenario_suite()
    assert isinstance(suite.scenarios, tuple)


def test_scenario_suite_hash_changes_when_seed_changes() -> None:
    payload = _suite_payload()
    first = validate_scenario_suite(payload)
    payload["scenarios"][1]["algorithm_seed"] += 1
    second = validate_scenario_suite(payload)
    assert scenario_suite_hash(first) != scenario_suite_hash(second)


def test_difficulty_scale_is_not_raw_csv_row_count() -> None:
    difficulty = build_input_difficulty(_small_input())
    assert difficulty["scale"]["logical_requests"] == 3
    assert difficulty["scale"]["logical_primaries"] == 2


def test_paired_comparison_has_no_significance_claim() -> None:
    rows = _fake_scenario_result(load_scenario_suite().scenarios[0])["scenario_rows"]
    assert all("p_value" not in row and "significant" not in row for row in build_paired_comparison(rows))


def test_aggregate_summary_documents_completed_row_semantics() -> None:
    summary = build_aggregate_summary([])
    assert "completed overall scenario rows" in summary["row_semantics"]


def test_manifest_copy_mutation_does_not_change_original_payload() -> None:
    original = _suite_payload()
    clone = copy.deepcopy(original)
    clone["scenarios"][1]["algorithm_seed"] += 1
    assert original["scenarios"][1]["algorithm_seed"] == 20260630
