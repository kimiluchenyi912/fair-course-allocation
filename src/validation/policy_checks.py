from __future__ import annotations

import pandas as pd

from .constants import (
    BASELINE_CAPACITY_RULES,
    BASELINE_COURSE_GRADES,
    BASELINE_FIXED_TARGETS,
    BASELINE_GRADE_PROFILES,
    CONFIG_COLUMNS,
)
from .helpers import has_columns, line_number, parse_float, parse_int, text
from .models import ValidationReport


def validate_baseline_policy(
    config: dict[str, pd.DataFrame],
    report: ValidationReport,
    *,
    strict_policy: bool,
) -> None:
    _check_grade_profile_baseline(config.get("grade_profiles.csv"), report, strict_policy)
    _check_capacity_rule_baseline(config.get("capacity_rules.csv"), report, strict_policy)
    _check_course_grade_baseline(config.get("course_catalog.csv"), report, strict_policy)
    _check_fixed_target_baseline(config.get("fixed_course_targets.csv"), report, strict_policy)


def _check_grade_profile_baseline(
    df: pd.DataFrame | None,
    report: ValidationReport,
    strict_policy: bool,
) -> None:
    filename = "grade_profiles.csv"
    if not has_columns(df, CONFIG_COLUMNS[filename]):
        return
    for idx, row in df.iterrows():
        grade = parse_int(row["grade"])
        if grade not in BASELINE_GRADE_PROFILES:
            continue
        expected = BASELINE_GRADE_PROFILES[grade]
        actual = (
            parse_int(row["student_count"]),
            parse_float(row["share_5_classes"]),
            parse_float(row["share_6_classes"]),
            parse_float(row["share_7_classes"]),
        )
        if actual != expected:
            report.add_policy_issue(
                "BASELINE_GRADE_PROFILE_MISMATCH",
                filename,
                f"Current reference baseline is {expected}, found {actual}.",
                strict_policy,
                line_number(idx),
                str(grade),
            )


def _check_capacity_rule_baseline(
    df: pd.DataFrame | None,
    report: ValidationReport,
    strict_policy: bool,
) -> None:
    filename = "capacity_rules.csv"
    if not has_columns(df, CONFIG_COLUMNS[filename]):
        return
    indexed = df.set_index("rule_id", drop=False)
    for rule_id, expected_values in BASELINE_CAPACITY_RULES.items():
        if rule_id not in indexed.index:
            report.add_policy_issue(
                "BASELINE_CAPACITY_RULE_MISSING",
                filename,
                f"Current reference baseline includes rule '{rule_id}'.",
                strict_policy,
                identifier=rule_id,
            )
            continue
        row = indexed.loc[rule_id]
        for col, expected in expected_values.items():
            value = parse_float(row[col]) if col == "expansion_threshold_ratio" else parse_int(row[col])
            if value != expected:
                report.add_policy_issue(
                    "BASELINE_CAPACITY_RULE_MISMATCH",
                    filename,
                    f"{rule_id}.{col} baseline is {expected}, found {value}.",
                    strict_policy,
                    identifier=rule_id,
                )


def _check_course_grade_baseline(
    df: pd.DataFrame | None,
    report: ValidationReport,
    strict_policy: bool,
) -> None:
    filename = "course_catalog.csv"
    if not has_columns(df, CONFIG_COLUMNS[filename]):
        return
    indexed = df.set_index("course_id", drop=False)
    for course_id, expected_grades in BASELINE_COURSE_GRADES.items():
        if course_id not in indexed.index:
            report.add_policy_issue(
                "BASELINE_COURSE_MISSING",
                filename,
                f"Current reference baseline includes course '{course_id}'.",
                strict_policy,
                identifier=course_id,
            )
            continue
        actual = text(indexed.loc[course_id, "eligible_grades"]).split(";")
        if actual != expected_grades:
            report.add_policy_issue(
                "BASELINE_COURSE_GRADES_MISMATCH",
                filename,
                f"{course_id} baseline grades are {';'.join(expected_grades)}, found {';'.join(actual)}.",
                strict_policy,
                identifier=course_id,
            )


def _check_fixed_target_baseline(
    df: pd.DataFrame | None,
    report: ValidationReport,
    strict_policy: bool,
) -> None:
    filename = "fixed_course_targets.csv"
    if not has_columns(df, CONFIG_COLUMNS[filename]):
        return
    indexed = {
        (text(row["scenario_id"]), parse_int(row["grade"]), text(row["course_id"])): row
        for _, row in df.iterrows()
    }
    for key, expected in BASELINE_FIXED_TARGETS.items():
        row = indexed.get(key)
        identifier = f"{key[0]}:{key[1]}:{key[2]}"
        if row is None:
            report.add_policy_issue(
                "BASELINE_FIXED_TARGET_MISSING",
                filename,
                f"Current reference baseline includes fixed target {identifier}.",
                strict_policy,
                identifier=identifier,
            )
            continue
        actual = (
            parse_int(row["min_count"]),
            parse_int(row["target_count"]),
            parse_int(row["max_count"]),
        )
        if actual != expected:
            report.add_policy_issue(
                "BASELINE_FIXED_TARGET_MISMATCH",
                filename,
                f"{identifier} baseline min/target/max is {expected}, found {actual}.",
                strict_policy,
                identifier=identifier,
            )
