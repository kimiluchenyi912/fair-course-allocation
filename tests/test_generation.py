from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.generation import generate_synthetic_dataset
from src.validation import validate_configuration


REPO_ROOT = Path(__file__).resolve().parents[1]
GOV_ECON_COURSES = {"GOV_ECON_REG", "GOV_APMACRO", "APGOV_ECON", "APGOV_APMACRO"}
GRADE10_MATH = {"MATH2", "MATH2_3_HA", "MATH3", "MATH3_H", "AP_CALC_AB", "AP_CALC_BC"}
GRADE11_MATH = {"MATH3", "MATH3_H", "INTRO_CALC", "AP_CALC_AB", "AP_CALC_BC", "AP_STATS", "CALC_D_LINALG"}
GRADE12_MATH = {"INTRO_CALC", "AP_CALC_AB", "AP_CALC_BC", "AP_STATS", "CALC_D_LINALG"}


@pytest.fixture(scope="module")
def generated_result(tmp_path_factory: pytest.TempPathFactory):
    config_dir, _ = copy_generation_inputs(tmp_path_factory.mktemp("generated_result"))
    return generate_synthetic_dataset(config_dir, "stable_year", 2026)


def copy_generation_inputs(base: Path) -> tuple[Path, Path]:
    config_dir = base / "config"
    templates_dir = base / "templates"
    shutil.copytree(REPO_ROOT / "data" / "config", config_dir)
    shutil.copytree(REPO_ROOT / "data" / "templates", templates_dir)
    return config_dir, templates_dir


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False)


def write_csv(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)


def issue_codes(report) -> set[str]:
    return {issue.code for issue in report.errors}


def course_by_id(catalog: pd.DataFrame) -> dict[str, pd.Series]:
    return {str(row["course_id"]): row for _, row in catalog.iterrows()}


def request_grades(students: pd.DataFrame, requests: pd.DataFrame) -> pd.DataFrame:
    return requests.merge(students[["student_id", "grade"]], on="student_id", how="left")


def primary_units_by_student(result) -> pd.Series:
    catalog = course_by_id(result.catalog)
    primary = result.requests[result.requests["request_type"] == "primary"].copy()
    primary = primary.drop_duplicates(["student_id", "course_id", "request_group", "must_share_block_id"])
    primary["period_units"] = primary["course_id"].map(lambda course_id: int(catalog[course_id]["periods_required"]))
    return primary.groupby("student_id")["period_units"].sum()


def test_current_config_generates_2630_students(generated_result) -> None:
    assert len(generated_result.students) == 2630


def test_grade_counts_are_correct(generated_result) -> None:
    counts = generated_result.students.groupby("grade").size().to_dict()

    assert counts == {9: 700, 10: 650, 11: 640, 12: 640}


def test_load_apportionment_matches_grade_profiles(generated_result) -> None:
    counts = generated_result.students.groupby(["grade", "target_course_count"]).size().to_dict()

    assert counts == {
        (9, 6): 70,
        (9, 7): 630,
        (10, 6): 130,
        (10, 7): 520,
        (11, 5): 32,
        (11, 6): 224,
        (11, 7): 384,
        (12, 5): 128,
        (12, 6): 301,
        (12, 7): 211,
    }


def test_same_seed_is_fully_deterministic(tmp_path: Path) -> None:
    config_dir, _ = copy_generation_inputs(tmp_path)

    first = generate_synthetic_dataset(config_dir, "stable_year", 2026)
    second = generate_synthetic_dataset(config_dir, "stable_year", 2026)

    pd.testing.assert_frame_equal(first.students, second.students)
    pd.testing.assert_frame_equal(first.requests, second.requests)
    pd.testing.assert_frame_equal(first.summary, second.summary)
    pd.testing.assert_frame_equal(first.catalog, second.catalog)
    assert first.metadata == second.metadata


def test_different_seed_changes_some_requests(tmp_path: Path) -> None:
    config_dir, _ = copy_generation_inputs(tmp_path)

    first = generate_synthetic_dataset(config_dir, "stable_year", 2026)
    second = generate_synthetic_dataset(config_dir, "stable_year", 2027)

    assert not first.requests.equals(second.requests)


def test_student_ids_are_unique(generated_result) -> None:
    assert generated_result.students["student_id"].is_unique


def test_all_request_course_references_are_valid(generated_result) -> None:
    course_ids = set(generated_result.catalog["course_id"])

    assert set(generated_result.requests["course_id"]).issubset(course_ids)


def test_primary_period_units_equal_student_target_load(generated_result) -> None:
    units = primary_units_by_student(generated_result)
    expected = generated_result.students.set_index("student_id")["target_course_count"]

    pd.testing.assert_series_equal(units.sort_index(), expected.sort_index(), check_names=False)


def test_alternates_have_unique_ranks_and_do_not_duplicate_primary(generated_result) -> None:
    requests = generated_result.requests
    primary = requests[requests["request_type"] == "primary"].groupby("student_id")["course_id"].apply(set)
    alternates = requests[requests["request_type"] == "alternate"]

    for student_id, group in alternates.groupby("student_id"):
        assert sorted(group["request_rank"].astype(int).tolist()) == [1, 2, 3]
        assert group["course_id"].is_unique
        assert set(group["course_id"]).isdisjoint(primary[student_id])


def test_grade10_math_is_not_below_math2(generated_result) -> None:
    requests = request_grades(generated_result.students, generated_result.requests)
    grade10_math = requests[
        (requests["grade"] == 10)
        & (requests["request_type"] == "primary")
        & (requests["course_id"].isin(GRADE10_MATH | {"MATH1", "INTRO_CALC", "AP_STATS", "CALC_D_LINALG"}))
    ]

    assert set(grade10_math["course_id"]).issubset(GRADE10_MATH)
    assert "INTRO_CALC" not in set(grade10_math["course_id"])


def test_grade11_math_is_not_below_math3(generated_result) -> None:
    requests = request_grades(generated_result.students, generated_result.requests)
    grade11_math = requests[
        (requests["grade"] == 11)
        & (requests["request_type"] == "primary")
        & (requests["course_id"].isin(GRADE11_MATH | {"MATH1", "MATH2", "MATH2_3_HA"}))
    ]

    assert set(grade11_math["course_id"]).issubset(GRADE11_MATH)


def test_grade12_does_not_generate_math3(generated_result) -> None:
    requests = request_grades(generated_result.students, generated_result.requests)
    grade12_math = requests[
        (requests["grade"] == 12)
        & (requests["request_type"] == "primary")
        & (requests["course_id"].isin(GRADE12_MATH | {"MATH1", "MATH2", "MATH2_3_HA", "MATH3", "MATH3_H"}))
    ]

    assert set(grade12_math["course_id"]).issubset(GRADE12_MATH)


def test_ap_euro_only_appears_in_grade12(generated_result) -> None:
    requests = request_grades(generated_result.students, generated_result.requests)
    grades = set(requests.loc[requests["course_id"] == "AP_EURO", "grade"])

    assert grades.issubset({12})


def test_ap_art_history_only_appears_in_grades_11_and_12(generated_result) -> None:
    requests = request_grades(generated_result.students, generated_result.requests)
    grades = set(requests.loc[requests["course_id"] == "AP_ART_HISTORY", "grade"])

    assert grades.issubset({11, 12})


def test_all_request_courses_match_student_grade(generated_result) -> None:
    catalog = course_by_id(generated_result.catalog)
    requests = request_grades(generated_result.students, generated_result.requests)

    for row in requests.itertuples(index=False):
        allowed = set(str(catalog[row.course_id]["eligible_grades"]).split(";"))
        assert str(row.grade) in allowed


def test_gov_econ_block_has_two_rows_but_one_period(generated_result) -> None:
    primary = request_grades(generated_result.students, generated_result.requests)
    gov = primary[(primary["request_type"] == "primary") & (primary["course_id"].isin(GOV_ECON_COURSES))]

    assert set(gov["grade"]) == {12}
    assert gov.groupby("student_id").size().eq(2).all()
    assert gov.groupby("student_id")["must_share_block_id"].nunique().eq(1).all()


def test_ap_physics_c_is_single_one_period_request(generated_result) -> None:
    catalog = course_by_id(generated_result.catalog)
    primary = generated_result.requests[generated_result.requests["request_type"] == "primary"]
    ap_physc = primary[primary["course_id"] == "AP_PHYSC"]

    assert int(catalog["AP_PHYSC"]["periods_required"]) == 1
    assert ap_physc.groupby("student_id").size().max() == 1


def test_math2_3_honors_accelerated_counts_as_two_periods(generated_result) -> None:
    catalog = course_by_id(generated_result.catalog)
    primary = generated_result.requests[generated_result.requests["request_type"] == "primary"]
    math23 = primary[primary["course_id"] == "MATH2_3_HA"]

    assert int(catalog["MATH2_3_HA"]["periods_required"]) == 2
    assert set(math23["request_type"]) == {"primary"}


def test_grade10_ap_calculus_targets_match_configuration(generated_result) -> None:
    primary = request_grades(generated_result.students, generated_result.requests)
    grade10 = primary[(primary["grade"] == 10) & (primary["request_type"] == "primary")]

    assert int((grade10["course_id"] == "AP_CALC_AB").sum()) == 6
    assert int((grade10["course_id"] == "AP_CALC_BC").sum()) == 20


def test_stable_seed_matches_configured_ap_stats_target(generated_result) -> None:
    primary = generated_result.requests[generated_result.requests["request_type"] == "primary"]

    assert int((primary["course_id"] == "AP_STATS").sum()) == 125


def test_stable_seed_matches_configured_calc_d_linalg_target(generated_result) -> None:
    primary = generated_result.requests[generated_result.requests["request_type"] == "primary"]

    assert int((primary["course_id"] == "CALC_D_LINALG").sum()) == 50


def test_generator_outputs_default_priority_fields(generated_result) -> None:
    students = generated_result.students

    assert set(students["priority_protected"]) == {"false"}
    assert set(students["priority_reason"]) == {""}
    assert set(students["priority_valid_school_year"]) == {""}


def test_cli_writes_all_expected_output_files(tmp_path: Path) -> None:
    config_dir, templates_dir = copy_generation_inputs(tmp_path)
    output_dir = tmp_path / "generated"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.generation",
            "--scenario",
            "stable_year",
            "--seed",
            "2026",
            "--config-dir",
            str(config_dir),
            "--templates-dir",
            str(templates_dir),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (output_dir / "students.csv").exists()
    assert (output_dir / "requests.csv").exists()
    assert (output_dir / "generation_summary.csv").exists()
    assert (output_dir / "generation_metadata.json").exists()


def test_invalid_generator_config_cli_fails_validation_without_partial_outputs(tmp_path: Path) -> None:
    config_dir, templates_dir = copy_generation_inputs(tmp_path)
    rules_path = config_dir / "grade_request_rules.csv"
    rules = read_csv(rules_path)
    rules.loc[rules["rule_key"] == "english_weights", "rule_value"] = "NO_SUCH_COURSE:1.0"
    write_csv(rules_path, rules)
    output_dir = tmp_path / "generated"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.generation",
            "--scenario",
            "stable_year",
            "--seed",
            "2026",
            "--config-dir",
            str(config_dir),
            "--templates-dir",
            str(templates_dir),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert "Validation FAIL" in result.stdout
    assert not output_dir.exists()


def test_generator_config_unknown_course_reference_fails(tmp_path: Path) -> None:
    config_dir, templates_dir = copy_generation_inputs(tmp_path)
    path = config_dir / "grade_request_rules.csv"
    rules = read_csv(path)
    rules.loc[rules["rule_key"] == "english_weights", "rule_value"] = "NO_SUCH_COURSE:1.0"
    write_csv(path, rules)

    report = validate_configuration(config_dir, templates_dir)

    assert "UNKNOWN_GENERATOR_RULE_COURSE" in issue_codes(report)


def test_negative_course_choice_weight_fails(tmp_path: Path) -> None:
    config_dir, templates_dir = copy_generation_inputs(tmp_path)
    path = config_dir / "course_choice_weights.csv"
    weights = read_csv(path)
    weights.loc[0, "weight"] = -1
    write_csv(path, weights)

    report = validate_configuration(config_dir, templates_dir)

    assert "INVALID_CHOICE_WEIGHT" in issue_codes(report)


def test_duplicate_grade_course_weight_fails(tmp_path: Path) -> None:
    config_dir, templates_dir = copy_generation_inputs(tmp_path)
    path = config_dir / "course_choice_weights.csv"
    weights = read_csv(path)
    course_row = weights[weights["scope_type"] == "course"].iloc[0]
    weights.loc[len(weights)] = course_row
    write_csv(path, weights)

    report = validate_configuration(config_dir, templates_dir)

    assert "DUPLICATE_COURSE_CHOICE_WEIGHT" in issue_codes(report)


def test_negative_fixed_target_fails(tmp_path: Path) -> None:
    config_dir, templates_dir = copy_generation_inputs(tmp_path)
    path = config_dir / "fixed_course_targets.csv"
    targets = read_csv(path)
    targets.loc[0, "target_count"] = -1
    write_csv(path, targets)

    report = validate_configuration(config_dir, templates_dir)

    assert "INVALID_FIXED_TARGET_COUNT" in issue_codes(report)


def test_noninteger_fixed_target_fails(tmp_path: Path) -> None:
    config_dir, templates_dir = copy_generation_inputs(tmp_path)
    path = config_dir / "fixed_course_targets.csv"
    targets = read_csv(path)
    targets["target_count"] = targets["target_count"].astype(str)
    targets.loc[0, "target_count"] = "6.5"
    write_csv(path, targets)

    report = validate_configuration(config_dir, templates_dir)

    assert "INVALID_FIXED_TARGET_COUNT" in issue_codes(report)


def test_fixed_target_course_must_apply_to_grade(tmp_path: Path) -> None:
    config_dir, templates_dir = copy_generation_inputs(tmp_path)
    path = config_dir / "fixed_course_targets.csv"
    targets = read_csv(path)
    targets.loc[0, "course_id"] = "AP_EURO"
    write_csv(path, targets)

    report = validate_configuration(config_dir, templates_dir)

    assert "GENERATOR_COURSE_GRADE_MISMATCH" in issue_codes(report)


def test_fixed_target_scenario_must_exist(tmp_path: Path) -> None:
    config_dir, templates_dir = copy_generation_inputs(tmp_path)
    path = config_dir / "fixed_course_targets.csv"
    targets = read_csv(path)
    targets.loc[0, "scenario_id"] = "NO_SUCH_SCENARIO"
    write_csv(path, targets)

    report = validate_configuration(config_dir, templates_dir)

    assert "UNKNOWN_FIXED_TARGET_SCENARIO" in issue_codes(report)
