from __future__ import annotations

import json

import pandas as pd
import pytest

from src.benchmark_visualizations import (
    VisualizationInputError,
    _top_unmet_courses,
    build_metrics_table,
    load_benchmark_artifacts,
    main,
    render_visualization_artifacts,
)


ALGORITHM_SUMMARY_COLUMNS = (
    "algorithm_name",
    "status",
    "students",
    "primary_assigned",
    "primary_unmet",
    "primary_satisfaction_rate",
    "total_alternates_assigned",
    "fully_scheduled_students",
    "ordinary_violations",
    "protected_violations",
)
COURSE_UNMET_COLUMNS = (
    "algorithm_name",
    "candidate_key",
    "primary_demand",
    "primary_assigned",
    "primary_unmet",
    "primary_unmet_rate",
)
STUDENT_OUTCOME_COLUMNS = (
    "algorithm_name",
    "student_id",
    "grade",
    "primary_unmet_count",
    "fully_scheduled",
    "priority_protected",
)

# Row order deliberately does NOT match algorithms_run order below, so tests
# can confirm manifest order wins over CSV row order.
ALGORITHM_SUMMARY_ROWS = [
    ("seeded_random_greedy", "completed", 10, 8, 2, 0.8, 3, 6, 1, 0),
    ("constrained_first_greedy", "completed", 10, 9, 1, 0.9, 1, 8, 0, 0),
]

COURSE_UNMET_ROWS = [
    ("seeded_random_greedy", "ART", 5, 3, 2, 0.4),
    ("seeded_random_greedy", "MATH", 4, 3, 1, 0.25),
    ("seeded_random_greedy", "ZEBRA_COURSE", 5, 3, 2, 0.4),
    ("seeded_random_greedy", "APPLE_COURSE", 5, 3, 2, 0.4),
    ("seeded_random_greedy", "PERFECT_COURSE", 4, 4, 0, 0.0),
    ("constrained_first_greedy", "ART", 5, 5, 0, 0.0),
]


def _student_outcome_rows() -> list[tuple]:
    rows = []
    for algorithm_name, unmet_pattern in (
        ("seeded_random_greedy", [1, 0, 1, 0, 1, 0, 0, 1, 0, 0]),
        ("constrained_first_greedy", [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]),
    ):
        grades = [9, 9, 9, 10, 10, 10, 11, 11, 12, 12]
        for index, (grade, unmet) in enumerate(zip(grades, unmet_pattern), start=1):
            student_id = f"STU_{index:02d}"
            rows.append(
                (
                    algorithm_name,
                    student_id,
                    grade,
                    unmet,
                    unmet == 0,
                    False,
                )
            )
    return rows


def _manifest_payload(algorithms_run: list[str]) -> dict:
    return {
        "algorithms_run": algorithms_run,
        "manifest": {
            "scenario_id": "unit_test_scenario",
            "data_generation_seed": 1,
            "section_planning_seed": 1,
            "solver_seed": 1,
            "students": 10,
            "logical_requests": 20,
            "logical_primaries": 15,
            "alternates": 5,
            "logical_sections": 6,
            "section_rows": 6,
            "canonical_input_hash": "fixture-hash",
        },
    }


def _write_fixture(
    tmp_path,
    *,
    algorithms_run: list[str] | None = None,
    algorithm_summary_rows: list[tuple] | None = None,
    course_unmet_rows: list[tuple] | None = None,
    student_outcome_rows: list[tuple] | None = None,
):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    manifest = _manifest_payload(algorithms_run if algorithms_run is not None else ["constrained", "random"])
    (artifact_dir / "benchmark_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    pd.DataFrame(
        algorithm_summary_rows if algorithm_summary_rows is not None else ALGORITHM_SUMMARY_ROWS,
        columns=ALGORITHM_SUMMARY_COLUMNS,
    ).to_csv(artifact_dir / "algorithm_summary.csv", index=False)

    pd.DataFrame(
        course_unmet_rows if course_unmet_rows is not None else COURSE_UNMET_ROWS,
        columns=COURSE_UNMET_COLUMNS,
    ).to_csv(artifact_dir / "course_unmet_summary.csv", index=False)

    pd.DataFrame(
        student_outcome_rows if student_outcome_rows is not None else _student_outcome_rows(),
        columns=STUDENT_OUTCOME_COLUMNS,
    ).to_csv(artifact_dir / "student_outcomes.csv", index=False)

    return artifact_dir


EXPECTED_CORE_FILES = {
    "algorithm_primary_outcomes.png",
    "algorithm_primary_satisfaction.png",
    "algorithm_fully_scheduled.png",
    "algorithm_policy_violations.png",
    "grade_primary_unmet_rate.png",
    "visualization_summary.md",
    "visualization_manifest.json",
}


def test_render_generates_all_core_files(tmp_path) -> None:
    artifact_dir = _write_fixture(tmp_path)
    output_dir = tmp_path / "viz"

    artifacts = load_benchmark_artifacts(artifact_dir)
    render_visualization_artifacts(artifacts, output_dir)

    written = {path.name for path in output_dir.iterdir()}
    assert EXPECTED_CORE_FILES <= written
    for name in EXPECTED_CORE_FILES:
        if name.endswith(".png"):
            assert (output_dir / name).stat().st_size > 0


def test_per_algorithm_top_unmet_course_image_count_matches_algorithms(tmp_path) -> None:
    artifact_dir = _write_fixture(tmp_path)
    output_dir = tmp_path / "viz"

    artifacts = load_benchmark_artifacts(artifact_dir)
    render_visualization_artifacts(artifacts, output_dir)

    course_images = sorted(p.name for p in output_dir.glob("top_unmet_courses_*.png"))
    assert course_images == [
        "top_unmet_courses_constrained_first_greedy.png",
        "top_unmet_courses_seeded_random_greedy.png",
    ]
    for name in course_images:
        assert (output_dir / name).stat().st_size > 0


def test_visualization_manifest_lists_generated_files_and_metadata(tmp_path) -> None:
    artifact_dir = _write_fixture(tmp_path)
    output_dir = tmp_path / "viz"

    artifacts = load_benchmark_artifacts(artifact_dir)
    generated = render_visualization_artifacts(artifacts, output_dir)

    manifest = json.loads((output_dir / "visualization_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "visualization_artifacts_v1"
    assert set(manifest["generated_files"]) == set(generated)
    assert set(manifest["generated_files"]) == {p.name for p in output_dir.iterdir()}
    assert manifest["algorithms_detected"] == ["constrained_first_greedy", "seeded_random_greedy"]
    assert manifest["top_unmet_courses_default_n"] == 15
    assert "grade_level_unmet_rate_definition" in manifest and manifest["grade_level_unmet_rate_definition"]
    assert manifest["grade_order"] == [9, 10, 11, 12]
    assert any("CP-SAT" in item for item in manifest["known_limitations"])
    assert any("priority_protected" in item for item in manifest["known_limitations"])
    assert any("rejection reason" in item for item in manifest["known_limitations"])
    for filename in ("algorithm_primary_outcomes.png", "grade_primary_unmet_rate.png"):
        assert filename in manifest["chart_data_sources"]


def test_markdown_summary_uses_fixture_numbers_not_stable_year(tmp_path) -> None:
    artifact_dir = _write_fixture(tmp_path)
    output_dir = tmp_path / "viz"

    artifacts = load_benchmark_artifacts(artifact_dir)
    render_visualization_artifacts(artifacts, output_dir)

    summary = (output_dir / "visualization_summary.md").read_text(encoding="utf-8")
    assert "Constrained First" in summary
    assert "Seeded Random" in summary
    assert "90.0%" in summary  # constrained_first_greedy primary_satisfaction_rate = 0.9
    assert "80.0%" in summary  # seeded_random_greedy primary_satisfaction_rate = 0.8
    # Real stable-year numbers must never leak into a fixture-driven summary.
    assert "16069" not in summary
    assert "17216" not in summary
    assert "2630" not in summary


def test_algorithm_order_follows_manifest_algorithms_run_not_csv_row_order(tmp_path) -> None:
    artifact_dir = _write_fixture(tmp_path, algorithms_run=["constrained", "random"])
    artifacts = load_benchmark_artifacts(artifact_dir)

    assert artifacts.algorithm_order == ("constrained_first_greedy", "seeded_random_greedy")

    table = build_metrics_table(artifacts)
    assert table["algorithm_name"].tolist() == ["constrained_first_greedy", "seeded_random_greedy"]


def test_algorithm_order_falls_back_to_csv_row_order_without_manifest_hint(tmp_path) -> None:
    artifact_dir = _write_fixture(tmp_path, algorithms_run=[])
    artifacts = load_benchmark_artifacts(artifact_dir)

    assert artifacts.algorithm_order == ("seeded_random_greedy", "constrained_first_greedy")


def test_top_unmet_courses_sorted_by_unmet_desc_then_candidate_key_asc(tmp_path) -> None:
    artifact_dir = _write_fixture(tmp_path)
    artifacts = load_benchmark_artifacts(artifact_dir)

    top = _top_unmet_courses(artifacts.course_unmet_summary, "seeded_random_greedy", top_n=15)

    assert top["candidate_key"].tolist() == ["APPLE_COURSE", "ART", "ZEBRA_COURSE", "MATH"]
    assert "PERFECT_COURSE" not in top["candidate_key"].tolist()


def test_missing_student_outcomes_csv_raises_clear_error(tmp_path) -> None:
    artifact_dir = _write_fixture(tmp_path)
    (artifact_dir / "student_outcomes.csv").unlink()

    with pytest.raises(VisualizationInputError, match="student_outcomes.csv"):
        load_benchmark_artifacts(artifact_dir)


def test_missing_required_column_raises_clear_error(tmp_path) -> None:
    artifact_dir = _write_fixture(tmp_path)
    frame = pd.read_csv(artifact_dir / "student_outcomes.csv")
    frame = frame.drop(columns=["grade"])
    frame.to_csv(artifact_dir / "student_outcomes.csv", index=False)

    with pytest.raises(VisualizationInputError, match="grade"):
        load_benchmark_artifacts(artifact_dir)


def test_invalid_satisfaction_rate_above_one_raises(tmp_path) -> None:
    rows = [
        ("seeded_random_greedy", "completed", 10, 8, 2, 1.5, 3, 6, 1, 0),
        ("constrained_first_greedy", "completed", 10, 9, 1, 0.9, 1, 8, 0, 0),
    ]
    artifact_dir = _write_fixture(tmp_path, algorithm_summary_rows=rows)

    with pytest.raises(VisualizationInputError, match="primary_satisfaction_rate"):
        load_benchmark_artifacts(artifact_dir)


def test_negative_primary_unmet_count_raises(tmp_path) -> None:
    rows = [
        ("seeded_random_greedy", "completed", 10, 8, -2, 0.8, 3, 6, 1, 0),
        ("constrained_first_greedy", "completed", 10, 9, 1, 0.9, 1, 8, 0, 0),
    ]
    artifact_dir = _write_fixture(tmp_path, algorithm_summary_rows=rows)

    with pytest.raises(VisualizationInputError, match="primary_unmet"):
        load_benchmark_artifacts(artifact_dir)


def test_nan_in_required_column_raises(tmp_path) -> None:
    artifact_dir = _write_fixture(tmp_path)
    frame = pd.read_csv(artifact_dir / "algorithm_summary.csv")
    frame.loc[0, "primary_assigned"] = float("nan")
    frame.to_csv(artifact_dir / "algorithm_summary.csv", index=False)

    with pytest.raises(VisualizationInputError, match="primary_assigned"):
        load_benchmark_artifacts(artifact_dir)


def test_no_cp_sat_output_when_not_in_input(tmp_path) -> None:
    artifact_dir = _write_fixture(tmp_path)
    output_dir = tmp_path / "viz"

    artifacts = load_benchmark_artifacts(artifact_dir)
    render_visualization_artifacts(artifacts, output_dir)

    course_images = {p.name for p in output_dir.glob("top_unmet_courses_*.png")}
    assert not any("cp_sat" in name.lower() for name in course_images)

    manifest = json.loads((output_dir / "visualization_manifest.json").read_text(encoding="utf-8"))
    assert not any("cp_sat" in name.lower() for name in manifest["algorithms_detected"])

    summary = (output_dir / "visualization_summary.md").read_text(encoding="utf-8")
    metrics_section = summary.split("## 3. Key metrics")[1].split("## 4. Observations")[0]
    assert "CP-SAT" not in metrics_section
    # The limitations section is allowed (in fact required) to state that
    # CP-SAT was not run, as an explicit disclaimer rather than a result row.
    assert "CP-SAT was not run" in summary


def test_protected_caveat_present_when_zero_protected_students(tmp_path) -> None:
    artifact_dir = _write_fixture(tmp_path)
    output_dir = tmp_path / "viz"

    artifacts = load_benchmark_artifacts(artifact_dir)
    render_visualization_artifacts(artifacts, output_dir)

    summary = (output_dir / "visualization_summary.md").read_text(encoding="utf-8")
    assert "zero** priority_protected students" in summary
    assert "cannot be used to validate protected-student policy behavior" in summary


def test_no_fabricated_rejection_reason_breakdown(tmp_path) -> None:
    artifact_dir = _write_fixture(tmp_path)
    output_dir = tmp_path / "viz"

    # request_outcomes.csv is intentionally absent from this fixture; the
    # loader must not require it, and no rejection-reason breakdown may be
    # invented anywhere in the outputs.
    assert not (artifact_dir / "request_outcomes.csv").exists()

    artifacts = load_benchmark_artifacts(artifact_dir)
    render_visualization_artifacts(artifacts, output_dir)

    summary = (output_dir / "visualization_summary.md").read_text(encoding="utf-8")
    manifest = json.loads((output_dir / "visualization_manifest.json").read_text(encoding="utf-8"))
    for forbidden in ("period_conflict", "capacity_rejections", "duplicate_course_rejections"):
        assert forbidden not in summary.lower()
        assert forbidden not in json.dumps(manifest).lower()


def test_repeated_run_removes_stale_files_from_output_dir(tmp_path) -> None:
    artifact_dir = _write_fixture(tmp_path)
    output_dir = tmp_path / "viz"

    artifacts = load_benchmark_artifacts(artifact_dir)
    render_visualization_artifacts(artifacts, output_dir)

    stale_file = output_dir / "stale_leftover_from_previous_run.png"
    stale_file.write_bytes(b"not a real chart")
    assert stale_file.exists()

    render_visualization_artifacts(artifacts, output_dir)

    assert not stale_file.exists()
    assert {p.name for p in output_dir.iterdir()} == set(
        json.loads((output_dir / "visualization_manifest.json").read_text(encoding="utf-8"))["generated_files"]
    )


def test_failed_validation_does_not_write_partial_output(tmp_path) -> None:
    rows = [
        ("seeded_random_greedy", "completed", 10, 8, -2, 0.8, 3, 6, 1, 0),
        ("constrained_first_greedy", "completed", 10, 9, 1, 0.9, 1, 8, 0, 0),
    ]
    artifact_dir = _write_fixture(tmp_path, algorithm_summary_rows=rows)
    output_dir = tmp_path / "viz"

    with pytest.raises(VisualizationInputError):
        load_benchmark_artifacts(artifact_dir)

    assert not output_dir.exists()


def test_cli_main_returns_nonzero_on_validation_failure(tmp_path) -> None:
    artifact_dir = _write_fixture(tmp_path)
    (artifact_dir / "course_unmet_summary.csv").unlink()
    output_dir = tmp_path / "viz"

    exit_code = main(["--artifact-dir", str(artifact_dir), "--output-dir", str(output_dir)])

    assert exit_code == 1
    assert not output_dir.exists()


def test_cli_main_succeeds_and_writes_output(tmp_path) -> None:
    artifact_dir = _write_fixture(tmp_path)
    output_dir = tmp_path / "viz"

    exit_code = main(["--artifact-dir", str(artifact_dir), "--output-dir", str(output_dir)])

    assert exit_code == 0
    assert (output_dir / "visualization_summary.md").exists()


# --- Output-directory safety -------------------------------------------------


def test_output_dir_equal_to_artifact_dir_is_rejected(tmp_path) -> None:
    artifact_dir = _write_fixture(tmp_path)
    before = (artifact_dir / "algorithm_summary.csv").read_text(encoding="utf-8")
    artifacts = load_benchmark_artifacts(artifact_dir)

    with pytest.raises(VisualizationInputError, match="same as, or an ancestor of"):
        render_visualization_artifacts(artifacts, artifact_dir)

    assert (artifact_dir / "algorithm_summary.csv").read_text(encoding="utf-8") == before
    assert (artifact_dir / "student_outcomes.csv").exists()
    assert (artifact_dir / "course_unmet_summary.csv").exists()
    assert (artifact_dir / "benchmark_manifest.json").exists()


def test_output_dir_ancestor_of_artifact_dir_is_rejected(tmp_path) -> None:
    parent_dir = tmp_path / "parent"
    parent_dir.mkdir()
    sentinel = parent_dir / "unrelated_sentinel.txt"
    sentinel.write_text("do not delete me", encoding="utf-8")
    artifact_dir = _write_fixture(parent_dir)  # writes to parent_dir / "artifacts"

    artifacts = load_benchmark_artifacts(artifact_dir)

    with pytest.raises(VisualizationInputError, match="same as, or an ancestor of"):
        render_visualization_artifacts(artifacts, parent_dir)

    assert sentinel.read_text(encoding="utf-8") == "do not delete me"
    assert (artifact_dir / "algorithm_summary.csv").exists()
    assert (artifact_dir / "student_outcomes.csv").exists()


def test_existing_nonempty_unknown_output_dir_is_rejected(tmp_path) -> None:
    artifact_dir = _write_fixture(tmp_path)
    output_dir = tmp_path / "viz"
    output_dir.mkdir()
    sentinel = output_dir / "my_important_file.txt"
    sentinel.write_text("please keep", encoding="utf-8")

    artifacts = load_benchmark_artifacts(artifact_dir)

    with pytest.raises(VisualizationInputError, match="not recognized as a Visualization Artifacts v1 output directory"):
        render_visualization_artifacts(artifacts, output_dir)

    assert sentinel.read_text(encoding="utf-8") == "please keep"
    assert {p.name for p in output_dir.iterdir()} == {"my_important_file.txt"}


def test_existing_empty_output_dir_succeeds(tmp_path) -> None:
    artifact_dir = _write_fixture(tmp_path)
    output_dir = tmp_path / "viz"
    output_dir.mkdir()

    artifacts = load_benchmark_artifacts(artifact_dir)
    generated = render_visualization_artifacts(artifacts, output_dir)

    assert set(generated) == {p.name for p in output_dir.iterdir()}
    assert (output_dir / "visualization_summary.md").exists()


def test_existing_recognized_v1_output_dir_can_be_safely_replaced(tmp_path) -> None:
    # Covers requirement 5: a directory this tool previously wrote (valid
    # visualization_manifest.json with the current schema_version) can be
    # rerun, stale leftovers are removed, and the fresh file set is exact.
    artifact_dir = _write_fixture(tmp_path)
    output_dir = tmp_path / "viz"
    artifacts = load_benchmark_artifacts(artifact_dir)
    render_visualization_artifacts(artifacts, output_dir)

    stale_file = output_dir / "stale_leftover_from_previous_run.png"
    stale_file.write_bytes(b"not a real chart")

    generated = render_visualization_artifacts(artifacts, output_dir)

    assert not stale_file.exists()
    assert set(generated) == {p.name for p in output_dir.iterdir()}


def test_existing_output_dir_with_corrupt_manifest_is_rejected(tmp_path) -> None:
    artifact_dir = _write_fixture(tmp_path)
    output_dir = tmp_path / "viz"
    output_dir.mkdir()
    (output_dir / "visualization_manifest.json").write_text("{not valid json", encoding="utf-8")
    sentinel = output_dir / "some_chart.png"
    sentinel.write_bytes(b"fake png bytes")

    artifacts = load_benchmark_artifacts(artifact_dir)

    with pytest.raises(VisualizationInputError, match="not recognized as a Visualization Artifacts v1 output directory"):
        render_visualization_artifacts(artifacts, output_dir)

    assert sentinel.exists()
    assert sentinel.read_bytes() == b"fake png bytes"
    assert (output_dir / "visualization_manifest.json").read_text(encoding="utf-8") == "{not valid json"


def test_existing_output_dir_with_schema_version_mismatch_is_rejected(tmp_path) -> None:
    artifact_dir = _write_fixture(tmp_path)
    output_dir = tmp_path / "viz"
    output_dir.mkdir()
    (output_dir / "visualization_manifest.json").write_text(
        json.dumps({"schema_version": "some_other_schema_v2"}), encoding="utf-8"
    )
    sentinel = output_dir / "some_chart.png"
    sentinel.write_bytes(b"fake png bytes")

    artifacts = load_benchmark_artifacts(artifact_dir)

    with pytest.raises(VisualizationInputError, match="schema_version"):
        render_visualization_artifacts(artifacts, output_dir)

    assert sentinel.exists()
    assert sentinel.read_bytes() == b"fake png bytes"


def test_staging_directory_cleaned_up_after_generation_failure(tmp_path, monkeypatch) -> None:
    artifact_dir = _write_fixture(tmp_path)
    output_dir = tmp_path / "viz"
    artifacts = load_benchmark_artifacts(artifact_dir)

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated rendering failure")

    monkeypatch.setattr("src.benchmark_visualizations.render_algorithm_policy_violations", _boom)

    with pytest.raises(RuntimeError, match="simulated rendering failure"):
        render_visualization_artifacts(artifacts, output_dir)

    assert not output_dir.exists()
    leftover_staging_dirs = list(output_dir.parent.glob(".benchmark_visualizations_staging_*"))
    assert leftover_staging_dirs == []
