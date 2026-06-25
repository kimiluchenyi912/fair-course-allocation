from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.validation import validate_configuration


REPO_ROOT = Path(__file__).resolve().parents[1]


def copy_validation_inputs(tmp_path: Path) -> tuple[Path, Path]:
    config_dir = tmp_path / "config"
    templates_dir = tmp_path / "templates"
    shutil.copytree(REPO_ROOT / "data" / "config", config_dir)
    shutil.copytree(REPO_ROOT / "data" / "templates", templates_dir)
    return config_dir, templates_dir


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False)


def write_csv(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)


def issue_codes(report) -> set[str]:
    return {issue.code for issue in report.errors}


def warning_codes(report) -> set[str]:
    return {issue.code for issue in report.warnings}


def test_current_real_configuration_passes(tmp_path: Path) -> None:
    config_dir, templates_dir = copy_validation_inputs(tmp_path)

    report = validate_configuration(config_dir, templates_dir)

    assert report.is_valid, report.to_text()
    assert report.errors == []
    assert report.warnings == []


def test_current_config_has_no_hard_max_section_policy(tmp_path: Path) -> None:
    config_dir, _ = copy_validation_inputs(tmp_path)
    capacity_rules = read_csv(config_dir / "capacity_rules.csv")
    catalog = read_csv(config_dir / "course_catalog.csv")

    assert set(capacity_rules["expansion_allowed"].map(lambda value: str(value).lower())) == {"true"}
    assert capacity_rules["default_max_sections"].map(str).str.strip().eq("").all()
    assert capacity_rules["expansion_threshold_ratio"].astype(float).eq(0.50).all()
    assert catalog["max_sections_override"].map(str).str.strip().eq("").all()


def test_capacity_rules_must_allow_uniform_expansion(tmp_path: Path) -> None:
    config_dir, templates_dir = copy_validation_inputs(tmp_path)
    path = config_dir / "capacity_rules.csv"
    capacity_rules = read_csv(path)
    capacity_rules["expansion_allowed"] = capacity_rules["expansion_allowed"].astype(str)
    capacity_rules.loc[capacity_rules["rule_id"] == "ap_csa", "expansion_allowed"] = "false"
    write_csv(path, capacity_rules)

    report = validate_configuration(config_dir, templates_dir)

    assert "EXPANSION_MUST_BE_UNIFORM" in issue_codes(report)


def test_grade_load_shares_must_sum_to_one(tmp_path: Path) -> None:
    config_dir, templates_dir = copy_validation_inputs(tmp_path)
    path = config_dir / "grade_profiles.csv"
    df = read_csv(path)
    df.loc[df["grade"] == 9, "share_7_classes"] = 0.80
    write_csv(path, df)

    report = validate_configuration(config_dir, templates_dir)

    assert not report.is_valid
    assert "LOAD_SHARE_SUM" in issue_codes(report)


def test_duplicate_course_id_is_reported(tmp_path: Path) -> None:
    config_dir, templates_dir = copy_validation_inputs(tmp_path)
    path = config_dir / "course_catalog.csv"
    df = read_csv(path)
    df.loc[1, "course_id"] = df.loc[0, "course_id"]
    write_csv(path, df)

    report = validate_configuration(config_dir, templates_dir)

    assert not report.is_valid
    assert "DUPLICATE_COURSE_ID" in issue_codes(report)


def test_invalid_free_period_is_reported(tmp_path: Path) -> None:
    config_dir, templates_dir = copy_validation_inputs(tmp_path)
    path = config_dir / "grade_profiles.csv"
    df = read_csv(path)
    df.loc[df["grade"] == 10, "allowed_free_periods"] = "P1;P3;P7"
    write_csv(path, df)

    report = validate_configuration(config_dir, templates_dir)

    assert not report.is_valid
    assert "INVALID_FREE_PERIOD" in issue_codes(report)


def test_capacity_bounds_must_be_consistent(tmp_path: Path) -> None:
    config_dir, templates_dir = copy_validation_inputs(tmp_path)
    path = config_dir / "capacity_rules.csv"
    df = read_csv(path)
    df.loc[df["rule_id"] == "normal_academic", "capacity_min"] = 50
    write_csv(path, df)

    report = validate_configuration(config_dir, templates_dir)

    assert not report.is_valid
    assert "CAPACITY_BOUNDS" in issue_codes(report)


def test_linked_block_must_reference_existing_course(tmp_path: Path) -> None:
    config_dir, templates_dir = copy_validation_inputs(tmp_path)
    path = config_dir / "linked_course_blocks.csv"
    df = read_csv(path)
    df.loc[df["block_template_id"] == "AP_PHYSC", "course_id"] = "NO_SUCH_COURSE"
    write_csv(path, df)

    report = validate_configuration(config_dir, templates_dir)

    assert not report.is_valid
    assert "UNKNOWN_BLOCK_COURSE" in issue_codes(report)


def test_malformed_semicolon_field_is_reported(tmp_path: Path) -> None:
    config_dir, templates_dir = copy_validation_inputs(tmp_path)
    path = config_dir / "course_catalog.csv"
    df = read_csv(path)
    df.loc[df["course_id"] == "MATH2", "eligible_grades"] = "9;;10"
    write_csv(path, df)

    report = validate_configuration(config_dir, templates_dir)

    assert not report.is_valid
    assert "MALFORMED_SEMICOLON_LIST" in issue_codes(report)


def test_ap_euro_must_be_grade_12_only(tmp_path: Path) -> None:
    config_dir, templates_dir = copy_validation_inputs(tmp_path)
    path = config_dir / "course_catalog.csv"
    df = read_csv(path)
    df.loc[df["course_id"] == "AP_EURO", "eligible_grades"] = "11;12"
    write_csv(path, df)

    report = validate_configuration(config_dir, templates_dir, strict_policy=True)

    assert not report.is_valid
    assert "BASELINE_COURSE_GRADES_MISMATCH" in issue_codes(report)


def test_ap_physics_c_structure_is_validated(tmp_path: Path) -> None:
    config_dir, templates_dir = copy_validation_inputs(tmp_path)
    catalog_path = config_dir / "course_catalog.csv"
    catalog = read_csv(catalog_path)
    catalog.loc[catalog["course_id"] == "AP_PHYSC", "periods_required"] = 2
    write_csv(catalog_path, catalog)

    blocks_path = config_dir / "linked_course_blocks.csv"
    blocks = read_csv(blocks_path)
    blocks.loc[blocks["block_template_id"] == "AP_PHYSC", "semester_2_content"] = "Optics"
    write_csv(blocks_path, blocks)

    report = validate_configuration(config_dir, templates_dir)
    codes = issue_codes(report)

    assert not report.is_valid
    assert "AP_PHYSC_STRUCTURE" in codes
    assert "AP_PHYSC_SEMESTERS" in codes


def test_template_invalid_references_are_reported(tmp_path: Path) -> None:
    config_dir, templates_dir = copy_validation_inputs(tmp_path)

    requests_path = templates_dir / "requests.csv"
    requests = read_csv(requests_path)
    requests.loc[0, "course_id"] = "NO_SUCH_COURSE"
    write_csv(requests_path, requests)

    assignments_path = templates_dir / "assignments.csv"
    assignments = read_csv(assignments_path)
    assignments.loc[0, "section_id"] = "NO_SUCH_SECTION"
    write_csv(assignments_path, assignments)

    unmet_path = templates_dir / "unmet_requests.csv"
    unmet = read_csv(unmet_path)
    unmet.loc[0, "student_id"] = "NO_SUCH_STUDENT"
    write_csv(unmet_path, unmet)

    report = validate_configuration(config_dir, templates_dir)
    codes = issue_codes(report)

    assert not report.is_valid
    assert "UNKNOWN_REQUEST_COURSE" in codes
    assert "UNKNOWN_ASSIGNMENT_SECTION" in codes
    assert "UNKNOWN_UNMET_STUDENT" in codes


def test_duplicate_alternate_rank_is_reported(tmp_path: Path) -> None:
    config_dir, templates_dir = copy_validation_inputs(tmp_path)
    path = templates_dir / "requests.csv"
    df = read_csv(path)
    df.loc[len(df)] = ["STU_0001", "AP_CSP", "alternate", 1, "elective_choice", ""]
    write_csv(path, df)

    report = validate_configuration(config_dir, templates_dir)

    assert not report.is_valid
    assert "DUPLICATE_ALTERNATE_RANK" in issue_codes(report)


def test_baseline_student_count_deviation_is_warning_by_default(tmp_path: Path) -> None:
    config_dir, templates_dir = copy_validation_inputs(tmp_path)
    path = config_dir / "grade_profiles.csv"
    df = read_csv(path)
    df.loc[df["grade"] == 9, "student_count"] = 701
    write_csv(path, df)

    report = validate_configuration(config_dir, templates_dir)

    assert report.is_valid, report.to_text()
    assert "BASELINE_GRADE_PROFILE_MISMATCH" in warning_codes(report)


def test_strict_policy_turns_baseline_deviation_into_error(tmp_path: Path) -> None:
    config_dir, templates_dir = copy_validation_inputs(tmp_path)
    path = config_dir / "grade_profiles.csv"
    df = read_csv(path)
    df.loc[df["grade"] == 9, "student_count"] = 701
    write_csv(path, df)

    report = validate_configuration(config_dir, templates_dir, strict_policy=True)

    assert not report.is_valid
    assert "BASELINE_GRADE_PROFILE_MISMATCH" in issue_codes(report)


def test_warning_cli_default_returns_success(tmp_path: Path) -> None:
    config_dir, templates_dir = copy_validation_inputs(tmp_path)
    path = config_dir / "grade_profiles.csv"
    df = read_csv(path)
    df.loc[df["grade"] == 9, "student_count"] = 701
    write_csv(path, df)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.validation",
            "--config-dir",
            str(config_dir),
            "--templates-dir",
            str(templates_dir),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Validation PASS: 0 error(s), 1 warning(s)" in result.stdout


def test_structure_error_fails_in_default_and_strict_modes(tmp_path: Path) -> None:
    config_dir, templates_dir = copy_validation_inputs(tmp_path)
    path = config_dir / "grade_profiles.csv"
    df = read_csv(path)
    df.loc[df["grade"] == 10, "allowed_free_periods"] = "P1;P3;P7"
    write_csv(path, df)

    default_report = validate_configuration(config_dir, templates_dir)
    strict_report = validate_configuration(config_dir, templates_dir, strict_policy=True)

    assert not default_report.is_valid
    assert not strict_report.is_valid
    assert "INVALID_FREE_PERIOD" in issue_codes(default_report)
    assert "INVALID_FREE_PERIOD" in issue_codes(strict_report)
