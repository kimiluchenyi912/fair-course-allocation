from __future__ import annotations

import math

import pandas as pd

from .constants import CONFIG_COLUMNS, VALID_GRADES, VALID_WEIGHT_SCOPES
from .helpers import has_columns, line_number, parse_float, parse_int, text
from .models import ValidationReport


WEIGHT_SUM_RULE_SUFFIXES = ("_weights",)
PROBABILITY_RULE_SUFFIXES = ("_probability",)


def validate_generation_config_tables(
    config: dict[str, pd.DataFrame],
    report: ValidationReport,
) -> None:
    catalog = config.get("course_catalog.csv")
    scenarios = config.get("demand_scenarios.csv")
    grade_profiles = config.get("grade_profiles.csv")

    course_ids = set(catalog["course_id"]) if has_columns(catalog, CONFIG_COLUMNS["course_catalog.csv"]) else set()
    scenario_ids = set(scenarios["scenario_id"]) if has_columns(scenarios, CONFIG_COLUMNS["demand_scenarios.csv"]) else set()
    grade_counts = _grade_counts(grade_profiles)
    course_lookup = _course_lookup(catalog)
    demand_tiers = set(catalog["demand_tier"]) if has_columns(catalog, CONFIG_COLUMNS["course_catalog.csv"]) else set()
    departments = set(catalog["department"]) if has_columns(catalog, CONFIG_COLUMNS["course_catalog.csv"]) else set()

    _validate_grade_request_rules(config.get("grade_request_rules.csv"), course_ids, course_lookup, report)
    _validate_course_choice_weights(
        config.get("course_choice_weights.csv"),
        course_lookup,
        scenario_ids,
        demand_tiers,
        departments,
        report,
    )
    _validate_fixed_course_targets(
        config.get("fixed_course_targets.csv"),
        course_lookup,
        scenario_ids,
        grade_counts,
        report,
    )


def _validate_grade_request_rules(
    df: pd.DataFrame | None,
    course_ids: set[str],
    course_lookup: dict[str, pd.Series],
    report: ValidationReport,
) -> None:
    filename = "grade_request_rules.csv"
    if not has_columns(df, CONFIG_COLUMNS[filename]):
        return

    duplicates = df[df.duplicated(["grade", "rule_key"], keep=False)]
    for idx, row in duplicates.iterrows():
        report.add_error(
            "DUPLICATE_GRADE_RULE",
            filename,
            "grade and rule_key combinations must be unique.",
            line_number(idx),
            f"{row['grade']}:{row['rule_key']}",
        )

    for idx, row in df.iterrows():
        line = line_number(idx)
        grade = text(row["grade"])
        rule_key = text(row["rule_key"])
        rule_value = text(row["rule_value"])
        if grade not in VALID_GRADES:
            report.add_error("INVALID_GENERATOR_RULE_GRADE", filename, "grade must be 9, 10, 11, or 12.", line, rule_key)
            continue

        if rule_key.endswith(PROBABILITY_RULE_SUFFIXES):
            value = parse_float(rule_value)
            if not _finite(value) or value < 0 or value > 1:
                report.add_error("INVALID_GENERATOR_PROBABILITY", filename, "Probability rules must be finite values from 0 to 1.", line, rule_key)
            continue

        if rule_key.endswith(WEIGHT_SUM_RULE_SUFFIXES):
            weights = _parse_weight_map(rule_value, filename, line, rule_key, report)
            if abs(sum(weights.values()) - 1.0) > 1e-4:
                report.add_error("GENERATOR_WEIGHT_SUM", filename, "Weight-map rule values must sum to 1.", line, rule_key)
            for course_id, weight in weights.items():
                if weight < 0:
                    report.add_error("NEGATIVE_GENERATOR_WEIGHT", filename, "Rule weights cannot be negative.", line, f"{rule_key}:{course_id}")
                if course_id.isdigit():
                    continue
                if course_id not in course_ids:
                    report.add_error("UNKNOWN_GENERATOR_RULE_COURSE", filename, "Rule references an unknown course_id.", line, course_id)
                    continue
                _validate_course_for_grade(course_lookup[course_id], grade, filename, line, course_id, report)
            continue

        value = parse_float(rule_value)
        if not _finite(value) or value < 0:
            report.add_error("INVALID_GENERATOR_RULE_VALUE", filename, "Numeric rule values must be finite and nonnegative.", line, rule_key)


def _validate_course_choice_weights(
    df: pd.DataFrame | None,
    course_lookup: dict[str, pd.Series],
    scenario_ids: set[str],
    demand_tiers: set[str],
    departments: set[str],
    report: ValidationReport,
) -> None:
    filename = "course_choice_weights.csv"
    if not has_columns(df, CONFIG_COLUMNS[filename]):
        return

    duplicates = df[df.duplicated(["scenario_id", "grade", "scope_type", "scope_id"], keep=False)]
    for idx, row in duplicates.iterrows():
        report.add_error(
            "DUPLICATE_COURSE_CHOICE_WEIGHT",
            filename,
            "scenario_id, grade, scope_type, and scope_id combinations must be unique.",
            line_number(idx),
            f"{row['scenario_id']}:{row['grade']}:{row['scope_type']}:{row['scope_id']}",
        )

    for idx, row in df.iterrows():
        line = line_number(idx)
        scenario_id = text(row["scenario_id"])
        grade = text(row["grade"])
        scope_type = text(row["scope_type"])
        scope_id = text(row["scope_id"])
        weight = parse_float(row["weight"])
        if scenario_id not in scenario_ids:
            report.add_error("UNKNOWN_WEIGHT_SCENARIO", filename, "scenario_id must reference demand_scenarios.csv.", line, scenario_id)
        if grade not in VALID_GRADES:
            report.add_error("INVALID_WEIGHT_GRADE", filename, "grade must be 9, 10, 11, or 12.", line, scope_id)
        if scope_type not in VALID_WEIGHT_SCOPES:
            report.add_error("INVALID_WEIGHT_SCOPE", filename, "scope_type must be demand_tier, department, or course.", line, scope_type)
        if not _finite(weight) or weight < 0:
            report.add_error("INVALID_CHOICE_WEIGHT", filename, "weight must be finite and nonnegative.", line, scope_id)

        if scope_type == "course":
            if scope_id not in course_lookup:
                report.add_error("UNKNOWN_WEIGHT_COURSE", filename, "course scope_id must reference course_catalog.course_id.", line, scope_id)
            elif grade in VALID_GRADES:
                _validate_course_for_grade(course_lookup[scope_id], grade, filename, line, scope_id, report)
        elif scope_type == "demand_tier" and scope_id not in demand_tiers:
            report.add_error("UNKNOWN_WEIGHT_DEMAND_TIER", filename, "demand_tier scope_id must exist in course_catalog.csv.", line, scope_id)
        elif scope_type == "department" and scope_id not in departments:
            report.add_error("UNKNOWN_WEIGHT_DEPARTMENT", filename, "department scope_id must exist in course_catalog.csv.", line, scope_id)


def _validate_fixed_course_targets(
    df: pd.DataFrame | None,
    course_lookup: dict[str, pd.Series],
    scenario_ids: set[str],
    grade_counts: dict[str, int],
    report: ValidationReport,
) -> None:
    filename = "fixed_course_targets.csv"
    if not has_columns(df, CONFIG_COLUMNS[filename]):
        return

    duplicates = df[df.duplicated(["scenario_id", "grade", "course_id"], keep=False)]
    for idx, row in duplicates.iterrows():
        report.add_error(
            "DUPLICATE_FIXED_TARGET",
            filename,
            "scenario_id, grade, and course_id combinations must be unique.",
            line_number(idx),
            f"{row['scenario_id']}:{row['grade']}:{row['course_id']}",
        )

    for idx, row in df.iterrows():
        line = line_number(idx)
        scenario_id = text(row["scenario_id"])
        grade = text(row["grade"])
        course_id = text(row["course_id"])
        target = parse_int(row["target_count"])
        min_count = parse_int(row["min_count"])
        max_count = parse_int(row["max_count"])

        if scenario_id not in scenario_ids:
            report.add_error("UNKNOWN_FIXED_TARGET_SCENARIO", filename, "scenario_id must reference demand_scenarios.csv.", line, scenario_id)
        if grade not in VALID_GRADES:
            report.add_error("INVALID_FIXED_TARGET_GRADE", filename, "grade must be 9, 10, 11, or 12.", line, course_id)
        if course_id not in course_lookup:
            report.add_error("UNKNOWN_FIXED_TARGET_COURSE", filename, "course_id must reference course_catalog.csv.", line, course_id)
        elif grade in VALID_GRADES:
            _validate_course_for_grade(course_lookup[course_id], grade, filename, line, course_id, report)

        for col, value in {"target_count": target, "min_count": min_count, "max_count": max_count}.items():
            if value is None or value < 0:
                report.add_error("INVALID_FIXED_TARGET_COUNT", filename, f"{col} must be a nonnegative integer.", line, course_id)
        if None not in (min_count, target, max_count) and not min_count <= target <= max_count:
            report.add_error("FIXED_TARGET_BOUNDS", filename, "Must satisfy min_count <= target_count <= max_count.", line, course_id)
        if target is not None and grade in grade_counts and target > grade_counts[grade]:
            report.add_error("FIXED_TARGET_EXCEEDS_GRADE", filename, "target_count cannot exceed the grade student_count.", line, course_id)


def _course_lookup(catalog: pd.DataFrame | None) -> dict[str, pd.Series]:
    if not has_columns(catalog, CONFIG_COLUMNS["course_catalog.csv"]):
        return {}
    return {text(row["course_id"]): row for _, row in catalog.iterrows()}


def _grade_counts(grade_profiles: pd.DataFrame | None) -> dict[str, int]:
    if not has_columns(grade_profiles, CONFIG_COLUMNS["grade_profiles.csv"]):
        return {}
    counts: dict[str, int] = {}
    for _, row in grade_profiles.iterrows():
        grade = text(row["grade"])
        count = parse_int(row["student_count"])
        if grade in VALID_GRADES and count is not None:
            counts[grade] = count
    return counts


def _validate_course_for_grade(
    course: pd.Series,
    grade: str,
    filename: str,
    line: int,
    identifier: str,
    report: ValidationReport,
) -> None:
    if grade not in text(course["eligible_grades"]).split(";"):
        report.add_error("GENERATOR_COURSE_GRADE_MISMATCH", filename, "Course is not eligible for the configured grade.", line, identifier)
    if text(course["occupies_school_period"]).lower() != "true" or parse_int(course["periods_required"]) == 0:
        report.add_error("GENERATOR_V1_EXCLUDED_COURSE", filename, "Generator cannot reference external, night, or zero-period V1-excluded courses.", line, identifier)


def _parse_weight_map(
    value: str,
    filename: str,
    line: int,
    identifier: str,
    report: ValidationReport,
) -> dict[str, float]:
    weights: dict[str, float] = {}
    for item in value.split(";"):
        if not item:
            report.add_error("MALFORMED_GENERATOR_WEIGHT_MAP", filename, "Weight-map entries cannot be blank.", line, identifier)
            continue
        if ":" not in item:
            report.add_error("MALFORMED_GENERATOR_WEIGHT_MAP", filename, "Weight-map entries must use key:weight.", line, identifier)
            continue
        key, raw_weight = item.split(":", 1)
        weight = parse_float(raw_weight)
        if not key or not _finite(weight):
            report.add_error("MALFORMED_GENERATOR_WEIGHT_MAP", filename, "Weight-map keys and weights must be finite and nonempty.", line, identifier)
            continue
        weights[key] = weight
    return weights


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(value)
