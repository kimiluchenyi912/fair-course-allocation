from __future__ import annotations

import hashlib
import json
import random

import pandas as pd
import pytest

from src.benchmark_runner import (
    BenchmarkRunnerError,
    CpSatBenchmarkOptions,
    main,
    run_benchmark_suite,
)
from src.experiment_manifest import (
    CanonicalInputFingerprint,
    ExperimentSeeds,
    build_experiment_manifest,
)


SEEDS = ExperimentSeeds(data_generation_seed=2026, section_planning_seed=2026, solver_seed=20260630)


def _catalog() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("CORE_A", "Core A", "English", 1, "standard"),
            ("ALT1", "Alternate 1", "Electives", 1, "standard"),
            ("MATH2", "Math 2", "Mathematics", 1, "standard"),
        ],
        columns=["course_id", "course_name", "department", "periods_required", "schedule_structure"],
    )


def _students() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("STU_1", 11, 1, "none", "false", "", ""),
            ("STU_2", 11, 1, "none", "false", "", ""),
        ],
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


def _requests() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("STU_1", "CORE_A", "primary", "", "", ""),
            ("STU_1", "ALT1", "alternate", 1, "alternate", ""),
            ("STU_2", "CORE_A", "primary", "", "", ""),
            ("STU_2", "ALT1", "alternate", 1, "alternate", ""),
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


def _sections() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("SEC_CORE", "CORE_A", "P1", "", "full_year", 1, "", "CORE_A_1", "CORE_A", ""),
            ("SEC_ALT", "ALT1", "P2", "", "full_year", 2, "", "ALT1_1", "ALT1", ""),
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


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture(tmp_path, *, reverse_requests: bool = False):
    generated = tmp_path / "generated"
    planned = tmp_path / "sections"
    config = tmp_path / "config"
    generated.mkdir(parents=True)
    planned.mkdir(parents=True)
    config.mkdir(parents=True)
    students = _students()
    requests = _requests()
    if reverse_requests:
        requests = requests.iloc[::-1].reset_index(drop=True)
    students.to_csv(generated / "students.csv", index=False)
    requests.to_csv(generated / "requests.csv", index=False)
    _sections().to_csv(planned / "sections.csv", index=False)
    _catalog().to_csv(config / "course_catalog.csv", index=False)
    (generated / "generation_metadata.json").write_text(
        json.dumps(
            {
                "scenario_id": "stable_year",
                "seed": SEEDS.data_generation_seed,
                "total_students": 2,
                "primary_request_rows": 2,
                "alternate_request_rows": 2,
                "output_file_hashes": {
                    "students.csv": _sha256(generated / "students.csv"),
                    "requests.csv": _sha256(generated / "requests.csv"),
                },
            }
        ),
        encoding="utf-8",
    )
    (planned / "section_planning_metadata.json").write_text(
        json.dumps(
            {
                "scenario_id": "stable_year",
                "seed": SEEDS.section_planning_seed,
                "student_count": 2,
                "primary_request_rows": 2,
                "total_section_rows": 2,
                "total_logical_sections": 2,
                "total_primary_demand": 2,
                "input_file_hashes": {
                    "students.csv": _sha256(generated / "students.csv"),
                    "requests.csv": _sha256(generated / "requests.csv"),
                },
                "output_file_hashes": {
                    "sections.csv": _sha256(planned / "sections.csv"),
                },
            }
        ),
        encoding="utf-8",
    )
    return generated, planned, config


def _run(tmp_path, **kwargs):
    generated, planned, config = _write_fixture(tmp_path)
    return run_benchmark_suite(
        generated_input_dir=generated,
        sections_input_dir=planned,
        config_dir=config,
        seeds=SEEDS,
        **kwargs,
    )


def test_runner_records_three_seeds_separately(tmp_path) -> None:
    result = _run(tmp_path)

    summary = result.to_dict()
    assert summary["data_generation_seed"] == 2026
    assert summary["section_planning_seed"] == 2026
    assert summary["solver_seed"] == 20260630


def test_runner_computes_fingerprint_from_canonical_input(tmp_path) -> None:
    result = _run(tmp_path)

    fingerprint = result.manifest.fingerprint
    assert fingerprint.students == 2
    assert fingerprint.logical_requests == 4
    assert fingerprint.logical_primaries == 2
    assert fingerprint.alternates == 2
    assert fingerprint.logical_sections == 2
    assert fingerprint.section_rows == 2
    assert fingerprint.candidate_edges == 4


def test_fingerprint_mismatch_prevents_algorithm_execution(tmp_path, monkeypatch) -> None:
    generated, planned, config = _write_fixture(tmp_path)
    actual = build_experiment_manifest(generated, planned, config, scenario_id="stable_year", seeds=SEEDS).fingerprint
    expected = CanonicalInputFingerprint(**{**actual.__dict__, "students": 3})

    def fail_if_called(*args, **kwargs):
        raise AssertionError("algorithm should not run")

    monkeypatch.setattr("src.benchmark_runner.run_seeded_random_baseline", fail_if_called)

    with pytest.raises(BenchmarkRunnerError, match="Canonical fingerprint mismatch"):
        run_benchmark_suite(
            generated_input_dir=generated,
            sections_input_dir=planned,
            config_dir=config,
            seeds=SEEDS,
            expected_fingerprint=expected,
        )


def test_mismatch_message_names_all_seed_roles_and_paths(tmp_path) -> None:
    generated, planned, config = _write_fixture(tmp_path)
    actual = build_experiment_manifest(generated, planned, config, scenario_id="stable_year", seeds=SEEDS).fingerprint
    expected = CanonicalInputFingerprint(**{**actual.__dict__, "candidate_edges": 999})

    with pytest.raises(BenchmarkRunnerError) as error:
        run_benchmark_suite(
            generated_input_dir=generated,
            sections_input_dir=planned,
            config_dir=config,
            seeds=SEEDS,
            expected_fingerprint=expected,
        )

    message = str(error.value)
    assert "data_generation_seed" in message
    assert "section_planning_seed" in message
    assert "solver_seed" in message
    assert "generated_input_dir" in message
    assert "sections_input_dir" in message


def test_random_algorithm_result_is_summarized_correctly(tmp_path) -> None:
    result = _run(tmp_path, algorithms=("random",))
    row = result.results[0]

    assert row.algorithm_name == "seeded_random_greedy"
    assert row.status == "completed"
    assert row.primary_assigned == 1
    assert row.primary_unmet == 1
    assert row.primary_satisfaction_rate == 0.5
    assert row.alternate_rank1_assigned == 1
    assert row.fully_scheduled_students == 2
    assert row.ordinary_violations == 0
    assert row.section_over_capacity_count == 0


def test_constrained_algorithm_result_is_summarized_correctly(tmp_path) -> None:
    result = _run(tmp_path, algorithms=("constrained",))
    row = result.results[0]

    assert row.algorithm_name == "constrained_first_greedy"
    assert row.primary_assigned == 1
    assert row.primary_unmet == 1
    assert row.total_alternates_assigned == 1
    assert row.students_with_remaining_units == 0


def test_fcfs_algorithm_result_is_summarized_correctly(tmp_path) -> None:
    result = _run(tmp_path, algorithms=("fcfs",))
    row = result.results[0]

    assert row.algorithm_name == "first_come_first_served_greedy"
    assert row.status == "completed"
    assert row.primary_assigned == 1
    assert row.primary_unmet == 1
    assert row.alternate_rank1_assigned == 1


def test_fcfs_is_not_run_by_default(tmp_path, monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("FCFS should be opt-in")

    monkeypatch.setattr("src.benchmark_runner.run_fcfs_baseline", fail_if_called)

    result = _run(tmp_path)

    assert result.algorithms_run == ("random", "constrained")
    assert "fcfs" not in result.algorithms_run


def test_fcfs_can_be_selected_explicitly_via_cli(tmp_path) -> None:
    generated, planned, config = _write_fixture(tmp_path)
    output_json = tmp_path / "summary.json"

    assert main(
        [
            "--generated-input-dir",
            str(generated),
            "--sections-input-dir",
            str(planned),
            "--config-dir",
            str(config),
            "--data-seed",
            "2026",
            "--section-seed",
            "2026",
            "--solver-seed",
            "20260630",
            "--algorithms",
            "fcfs",
            "--output-json",
            str(output_json),
        ]
    ) == 0

    data = json.loads(output_json.read_text(encoding="utf-8"))
    assert data["algorithms_run"] == ["fcfs"]
    assert data["results"][0]["algorithm_name"] == "first_come_first_served_greedy"


def test_artifact_export_works_with_fcfs_algorithm(tmp_path) -> None:
    artifact_dir = tmp_path / "artifacts"

    _run(tmp_path, algorithms=("fcfs",), output_artifact_dir=artifact_dir)

    written = sorted(path.name for path in artifact_dir.iterdir())
    assert written == [
        "algorithm_summary.csv",
        "assignment_failure_summary.csv",
        "benchmark_manifest.json",
        "course_unmet_summary.csv",
        "section_utilization.csv",
        "student_schedule_gaps.csv",
    ]
    summary_rows = pd.read_csv(artifact_dir / "algorithm_summary.csv", keep_default_na=False)
    assert list(summary_rows["algorithm_name"]) == ["first_come_first_served_greedy"]


def test_grade_priority_algorithm_result_is_summarized_correctly(tmp_path) -> None:
    result = _run(tmp_path, algorithms=("grade_priority",))
    row = result.results[0]

    assert row.algorithm_name == "grade_priority_greedy"
    assert row.status == "completed"
    assert row.primary_assigned == 1
    assert row.primary_unmet == 1
    assert row.alternate_rank1_assigned == 1


def test_grade_priority_is_not_run_by_default(tmp_path, monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("grade_priority should be opt-in")

    monkeypatch.setattr("src.benchmark_runner.run_grade_priority_baseline", fail_if_called)

    result = _run(tmp_path)

    assert result.algorithms_run == ("random", "constrained")
    assert "grade_priority" not in result.algorithms_run


def test_grade_priority_can_be_selected_explicitly_via_cli(tmp_path) -> None:
    generated, planned, config = _write_fixture(tmp_path)
    output_json = tmp_path / "summary.json"

    assert main(
        [
            "--generated-input-dir",
            str(generated),
            "--sections-input-dir",
            str(planned),
            "--config-dir",
            str(config),
            "--data-seed",
            "2026",
            "--section-seed",
            "2026",
            "--solver-seed",
            "20260630",
            "--algorithms",
            "grade_priority",
            "--output-json",
            str(output_json),
        ]
    ) == 0

    data = json.loads(output_json.read_text(encoding="utf-8"))
    assert data["algorithms_run"] == ["grade_priority"]
    assert data["results"][0]["algorithm_name"] == "grade_priority_greedy"


def test_artifact_export_works_with_grade_priority_algorithm(tmp_path) -> None:
    artifact_dir = tmp_path / "artifacts"

    _run(tmp_path, algorithms=("grade_priority",), output_artifact_dir=artifact_dir)

    written = sorted(path.name for path in artifact_dir.iterdir())
    assert written == [
        "algorithm_summary.csv",
        "assignment_failure_summary.csv",
        "benchmark_manifest.json",
        "course_unmet_summary.csv",
        "section_utilization.csv",
        "student_schedule_gaps.csv",
    ]
    summary_rows = pd.read_csv(artifact_dir / "algorithm_summary.csv", keep_default_na=False)
    assert list(summary_rows["algorithm_name"]) == ["grade_priority_greedy"]


def test_cp_sat_is_not_run_by_default(tmp_path, monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("CP-SAT should be opt-in")

    monkeypatch.setattr("src.benchmark_runner.run_fair_cp_sat_solver", fail_if_called)

    result = _run(tmp_path)

    assert result.algorithms_run == ("random", "constrained")


def test_cp_sat_can_be_selected_explicitly_on_tiny_fixture(tmp_path) -> None:
    result = _run(
        tmp_path,
        algorithms=("cp_sat",),
        cp_sat_options=CpSatBenchmarkOptions(max_time_seconds_per_stage=2, max_total_time_seconds=5),
    )

    row = result.results[0]
    assert row.algorithm_name == "fair_cp_sat_solver_v1_2"
    assert row.solve_status in {"OPTIMAL", "FEASIBLE"}
    assert row.bootstrap_status is not None


def test_json_output_contains_seeds_fingerprint_algorithms_and_metrics(tmp_path) -> None:
    output = tmp_path / "summary.json"
    result = _run(tmp_path, output_json_path=output)

    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["runner_name"] == "benchmark_runner_v1"
    assert data["data_generation_seed"] == 2026
    assert data["canonical_fingerprint"] == result.to_dict()["canonical_fingerprint"]
    assert data["algorithms_run"] == ["random", "constrained"]
    assert data["results"][0]["primary_assigned"] == 1


def test_csv_output_contains_one_row_per_algorithm(tmp_path) -> None:
    output = tmp_path / "summary.csv"
    _run(tmp_path, output_csv_path=output)

    rows = pd.read_csv(output, keep_default_na=False)
    assert list(rows["algorithm_name"]) == ["seeded_random_greedy", "constrained_first_greedy"]
    assert list(rows["data_generation_seed"]) == [2026, 2026]


def test_output_metrics_are_deterministic_for_same_seed_and_input(tmp_path) -> None:
    first = _run(tmp_path / "first").to_csv_rows()
    second = _run(tmp_path / "second").to_csv_rows()

    assert _stable_metric_projection(first) == _stable_metric_projection(second)


def test_runner_does_not_mutate_canonical_input(tmp_path) -> None:
    generated, planned, config = _write_fixture(tmp_path)
    before = build_experiment_manifest(generated, planned, config, scenario_id="stable_year", seeds=SEEDS).fingerprint

    run_benchmark_suite(generated_input_dir=generated, sections_input_dir=planned, config_dir=config, seeds=SEEDS)

    after = build_experiment_manifest(generated, planned, config, scenario_id="stable_year", seeds=SEEDS).fingerprint
    assert after == before


def test_runner_does_not_depend_on_raw_request_row_order(tmp_path) -> None:
    generated, planned, config = _write_fixture(tmp_path / "normal")
    expected = build_experiment_manifest(generated, planned, config, scenario_id="stable_year", seeds=SEEDS).fingerprint
    reversed_generated, reversed_planned, reversed_config = _write_fixture(tmp_path / "reversed", reverse_requests=True)

    normal = run_benchmark_suite(
        generated_input_dir=generated,
        sections_input_dir=planned,
        config_dir=config,
        seeds=SEEDS,
        expected_fingerprint=expected,
        algorithms=("constrained",),
    )
    reversed_result = run_benchmark_suite(
        generated_input_dir=reversed_generated,
        sections_input_dir=reversed_planned,
        config_dir=reversed_config,
        seeds=SEEDS,
        expected_fingerprint=expected,
        algorithms=("constrained",),
    )

    assert normal.manifest.fingerprint.canonical_input_hash == reversed_result.manifest.fingerprint.canonical_input_hash
    assert _stable_metric_projection(normal.to_csv_rows()) == _stable_metric_projection(reversed_result.to_csv_rows())


def test_runner_does_not_pollute_global_random_state(tmp_path) -> None:
    random.seed(12345)
    before = random.getstate()

    _run(tmp_path)

    assert random.getstate() == before


def test_cli_defaults_to_random_and_constrained_and_writes_outputs(tmp_path) -> None:
    generated, planned, config = _write_fixture(tmp_path)
    output_json = tmp_path / "benchmark.json"
    output_csv = tmp_path / "benchmark.csv"

    assert main(
        [
            "--generated-input-dir",
            str(generated),
            "--sections-input-dir",
            str(planned),
            "--config-dir",
            str(config),
            "--data-seed",
            "2026",
            "--section-seed",
            "2026",
            "--solver-seed",
            "20260630",
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
        ]
    ) == 0
    assert json.loads(output_json.read_text(encoding="utf-8"))["algorithms_run"] == ["random", "constrained"]
    assert len(pd.read_csv(output_csv, keep_default_na=False)) == 2


def test_output_artifact_dir_writes_default_four_artifacts(tmp_path) -> None:
    artifact_dir = tmp_path / "artifacts"

    _run(tmp_path, output_artifact_dir=artifact_dir)

    written = sorted(path.name for path in artifact_dir.iterdir())
    assert written == [
        "algorithm_summary.csv",
        "assignment_failure_summary.csv",
        "benchmark_manifest.json",
        "course_unmet_summary.csv",
        "section_utilization.csv",
        "student_schedule_gaps.csv",
    ]

    manifest = json.loads((artifact_dir / "benchmark_manifest.json").read_text(encoding="utf-8"))
    assert manifest["runner_name"] == "benchmark_runner_v1"
    assert manifest["algorithms_run"] == ["random", "constrained"]
    assert manifest["manifest"]["students"] == 2
    assert manifest["artifact_files"] == list(
        [
            "algorithm_summary.csv",
            "course_unmet_summary.csv",
            "section_utilization.csv",
            "assignment_failure_summary.csv",
            "student_schedule_gaps.csv",
            "benchmark_manifest.json",
        ]
    )
    assert manifest["diagnostics_schema_version"] == "assignment_failure_diagnostics_v1"
    assert manifest["diagnostics_artifact_files"] == [
        "assignment_failure_summary.csv",
        "student_schedule_gaps.csv",
    ]

    summary_rows = pd.read_csv(artifact_dir / "algorithm_summary.csv", keep_default_na=False)
    assert list(summary_rows["algorithm_name"]) == ["seeded_random_greedy", "constrained_first_greedy"]

    course_rows = pd.read_csv(artifact_dir / "course_unmet_summary.csv", keep_default_na=False)
    assert set(course_rows["candidate_key"]) == {"CORE_A"}
    assert set(course_rows["primary_demand"]) == {2}

    section_rows = pd.read_csv(artifact_dir / "section_utilization.csv", keep_default_na=False)
    assert set(section_rows["linked_section_group_id"]) == {"CORE_A_1", "ALT1_1"}


def test_output_artifact_dir_without_flag_omits_large_tables(tmp_path) -> None:
    artifact_dir = tmp_path / "artifacts"

    _run(tmp_path, output_artifact_dir=artifact_dir)

    assert not (artifact_dir / "student_outcomes.csv").exists()
    assert not (artifact_dir / "request_outcomes.csv").exists()
    assert (artifact_dir / "assignment_failure_summary.csv").exists()
    assert (artifact_dir / "student_schedule_gaps.csv").exists()


def test_include_large_tables_writes_student_and_request_outcomes(tmp_path) -> None:
    artifact_dir = tmp_path / "artifacts"

    _run(tmp_path, output_artifact_dir=artifact_dir, include_large_tables=True)

    student_rows = pd.read_csv(artifact_dir / "student_outcomes.csv", keep_default_na=False)
    assert set(student_rows["student_id"]) == {"STU_1", "STU_2"}
    assert set(student_rows["algorithm_name"]) == {"seeded_random_greedy", "constrained_first_greedy"}
    assert {"target_course_count", "assigned_course_count", "schedule_gap_count", "assigned_alternate_count"} <= set(
        student_rows.columns
    )

    request_rows = pd.read_csv(artifact_dir / "request_outcomes.csv", keep_default_na=False)
    assert "candidate_attempts_count" in request_rows.columns
    assert {
        "candidate_rejections_count",
        "rejected_section_at_capacity_count",
        "rejected_period_conflict_count",
        "rejected_duplicate_logical_course_count",
        "rejected_student_load_limit_count",
        "rejected_other_count",
        "terminal_unmet_reason",
    } <= set(request_rows.columns)
    assert set(request_rows["request_type"]) == {"primary", "alternate"}

    manifest = json.loads((artifact_dir / "benchmark_manifest.json").read_text(encoding="utf-8"))
    assert manifest["artifact_files"] == [
        "algorithm_summary.csv",
        "course_unmet_summary.csv",
        "section_utilization.csv",
        "assignment_failure_summary.csv",
        "student_schedule_gaps.csv",
        "benchmark_manifest.json",
        "student_outcomes.csv",
        "request_outcomes.csv",
    ]


def test_artifact_tables_are_sorted_by_algorithm_then_key(tmp_path) -> None:
    artifact_dir = tmp_path / "artifacts"

    _run(tmp_path, output_artifact_dir=artifact_dir, include_large_tables=True)

    course_rows = pd.read_csv(artifact_dir / "course_unmet_summary.csv", keep_default_na=False)
    course_pairs = list(zip(course_rows["algorithm_name"], course_rows["candidate_key"]))
    assert course_pairs == sorted(course_pairs)

    section_rows = pd.read_csv(artifact_dir / "section_utilization.csv", keep_default_na=False)
    section_pairs = list(zip(section_rows["algorithm_name"], section_rows["linked_section_group_id"]))
    assert section_pairs == sorted(section_pairs)

    student_rows = pd.read_csv(artifact_dir / "student_outcomes.csv", keep_default_na=False)
    student_pairs = list(zip(student_rows["algorithm_name"], student_rows["student_id"]))
    assert student_pairs == sorted(student_pairs)

    request_rows = pd.read_csv(artifact_dir / "request_outcomes.csv", keep_default_na=False)
    request_pairs = list(zip(request_rows["algorithm_name"], request_rows["request_key"]))
    assert request_pairs == sorted(request_pairs)

    failure_rows = pd.read_csv(artifact_dir / "assignment_failure_summary.csv", keep_default_na=False)
    assert list(dict.fromkeys(failure_rows["algorithm_name"])) == [
        "seeded_random_greedy",
        "constrained_first_greedy",
    ]

    # algorithm_summary.csv tracks execution order, not alphabetical order,
    # so it is deliberately excluded from the sorted() comparison above.
    summary_rows = pd.read_csv(artifact_dir / "algorithm_summary.csv", keep_default_na=False)
    assert list(summary_rows["algorithm_name"]) == ["seeded_random_greedy", "constrained_first_greedy"]


def test_pre_existing_artifact_directory_untouched_on_fingerprint_mismatch(tmp_path, monkeypatch) -> None:
    generated, planned, config = _write_fixture(tmp_path)
    actual = build_experiment_manifest(generated, planned, config, scenario_id="stable_year", seeds=SEEDS).fingerprint
    expected = CanonicalInputFingerprint(**{**actual.__dict__, "students": 3})

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    old_manifest = artifact_dir / "benchmark_manifest.json"
    old_manifest.write_text('{"stale": true}', encoding="utf-8")
    unrelated_file = artifact_dir / "old_report.csv"
    unrelated_file.write_text("legacy,data\n1,2\n", encoding="utf-8")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("algorithm should not run")

    monkeypatch.setattr("src.benchmark_runner.run_seeded_random_baseline", fail_if_called)

    with pytest.raises(BenchmarkRunnerError, match="Canonical fingerprint mismatch"):
        run_benchmark_suite(
            generated_input_dir=generated,
            sections_input_dir=planned,
            config_dir=config,
            seeds=SEEDS,
            expected_fingerprint=expected,
            output_artifact_dir=artifact_dir,
        )

    assert old_manifest.read_text(encoding="utf-8") == '{"stale": true}'
    assert unrelated_file.read_text(encoding="utf-8") == "legacy,data\n1,2\n"
    assert sorted(path.name for path in artifact_dir.iterdir()) == ["benchmark_manifest.json", "old_report.csv"]


def test_fingerprint_mismatch_does_not_write_artifacts(tmp_path, monkeypatch) -> None:
    generated, planned, config = _write_fixture(tmp_path)
    actual = build_experiment_manifest(generated, planned, config, scenario_id="stable_year", seeds=SEEDS).fingerprint
    expected = CanonicalInputFingerprint(**{**actual.__dict__, "students": 3})
    artifact_dir = tmp_path / "artifacts"

    def fail_if_called(*args, **kwargs):
        raise AssertionError("algorithm should not run")

    monkeypatch.setattr("src.benchmark_runner.run_seeded_random_baseline", fail_if_called)

    with pytest.raises(BenchmarkRunnerError, match="Canonical fingerprint mismatch"):
        run_benchmark_suite(
            generated_input_dir=generated,
            sections_input_dir=planned,
            config_dir=config,
            seeds=SEEDS,
            expected_fingerprint=expected,
            output_artifact_dir=artifact_dir,
            include_large_tables=True,
        )

    assert not artifact_dir.exists()


def test_manifest_verification_failure_does_not_write_artifacts(tmp_path) -> None:
    generated, planned, config = _write_fixture(tmp_path)
    (generated / "students.csv").write_text(
        _students().iloc[:1].to_csv(index=False), encoding="utf-8"
    )
    artifact_dir = tmp_path / "artifacts"

    with pytest.raises(Exception):
        run_benchmark_suite(
            generated_input_dir=generated,
            sections_input_dir=planned,
            config_dir=config,
            seeds=SEEDS,
            output_artifact_dir=artifact_dir,
        )

    assert not artifact_dir.exists()


def test_artifact_export_does_not_trigger_cp_sat_by_default(tmp_path, monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("CP-SAT should be opt-in")

    monkeypatch.setattr("src.benchmark_runner.run_fair_cp_sat_solver", fail_if_called)

    result = _run(tmp_path, output_artifact_dir=tmp_path / "artifacts", include_large_tables=True)

    assert result.algorithms_run == ("random", "constrained")


def test_cli_output_artifact_dir_writes_default_artifacts_without_large_tables(tmp_path) -> None:
    generated, planned, config = _write_fixture(tmp_path)
    artifact_dir = tmp_path / "artifacts"

    assert main(
        [
            "--generated-input-dir",
            str(generated),
            "--sections-input-dir",
            str(planned),
            "--config-dir",
            str(config),
            "--data-seed",
            "2026",
            "--section-seed",
            "2026",
            "--solver-seed",
            "20260630",
            "--output-artifact-dir",
            str(artifact_dir),
        ]
    ) == 0

    assert (artifact_dir / "algorithm_summary.csv").exists()
    assert (artifact_dir / "benchmark_manifest.json").exists()
    assert not (artifact_dir / "student_outcomes.csv").exists()


def test_cli_include_large_tables_flag_writes_large_tables(tmp_path) -> None:
    generated, planned, config = _write_fixture(tmp_path)
    artifact_dir = tmp_path / "artifacts"

    assert main(
        [
            "--generated-input-dir",
            str(generated),
            "--sections-input-dir",
            str(planned),
            "--config-dir",
            str(config),
            "--data-seed",
            "2026",
            "--section-seed",
            "2026",
            "--solver-seed",
            "20260630",
            "--output-artifact-dir",
            str(artifact_dir),
            "--include-large-tables",
        ]
    ) == 0

    assert (artifact_dir / "student_outcomes.csv").exists()
    assert (artifact_dir / "request_outcomes.csv").exists()


def _stable_metric_projection(rows: tuple[dict, ...]) -> tuple[dict, ...]:
    ignored = {"runtime_seconds", "generated_input_dir", "sections_input_dir"}
    return tuple({key: value for key, value in row.items() if key not in ignored} for row in rows)
