from __future__ import annotations

import pandas as pd

from .constants import (
    ALLOWED_FREE_PERIODS,
    CONFIG_COLUMNS,
    VALID_GRADES,
    VALID_PERIODS,
    VALID_STRUCTURES,
)
from .helpers import (
    has_columns,
    line_number,
    parse_bool,
    parse_float,
    parse_int,
    parse_optional_int,
    parse_semicolon_list,
    require_nonempty_unique,
    text,
)
from .models import ValidationReport


def validate_grade_profiles(df: pd.DataFrame | None, report: ValidationReport) -> None:
    filename = "grade_profiles.csv"
    if not has_columns(df, CONFIG_COLUMNS[filename]):
        return

    require_nonempty_unique(df, filename, "grade", "DUPLICATE_GRADE", report)
    seen_grades = {text(value) for value in df["grade"]}
    if seen_grades != VALID_GRADES:
        report.add_error(
            "GRADE_SET_MISMATCH",
            filename,
            "Grades must contain exactly 9, 10, 11, and 12 once each.",
        )

    for idx, row in df.iterrows():
        line = line_number(idx)
        grade = text(row["grade"])
        if grade not in VALID_GRADES:
            report.add_error("INVALID_GRADE", filename, "Grade must be 9, 10, 11, or 12.", line, grade)
            continue

        student_count = parse_int(row["student_count"])
        if student_count is None or student_count <= 0:
            report.add_error(
                "INVALID_STUDENT_COUNT",
                filename,
                "student_count must be a positive integer.",
                line,
                grade,
            )

        shares = [
            parse_float(row["share_5_classes"]),
            parse_float(row["share_6_classes"]),
            parse_float(row["share_7_classes"]),
        ]
        if any(value is None for value in shares):
            report.add_error("INVALID_LOAD_SHARE", filename, "Load shares must be numeric.", line, grade)
        elif any(value < 0 or value > 1 for value in shares if value is not None):
            report.add_error("LOAD_SHARE_OUT_OF_RANGE", filename, "Load shares must be between 0 and 1.", line, grade)
        elif abs(sum(shares) - 1.0) > 1e-6:
            report.add_error("LOAD_SHARE_SUM", filename, "5/6/7 class shares must sum to 1.", line, grade)

        free_periods = parse_semicolon_list(
            row["allowed_free_periods"],
            filename,
            "allowed_free_periods",
            line,
            report,
            allow_blank=False,
        )
        invalid_periods = [period for period in free_periods if period not in ALLOWED_FREE_PERIODS]
        if invalid_periods:
            report.add_error(
                "INVALID_FREE_PERIOD",
                filename,
                "Free periods may only use P1, P6, and P7 in V1.",
                line,
                ";".join(invalid_periods),
            )


def validate_capacity_rules(df: pd.DataFrame | None, report: ValidationReport) -> None:
    filename = "capacity_rules.csv"
    if not has_columns(df, CONFIG_COLUMNS[filename]):
        return

    require_nonempty_unique(df, filename, "rule_id", "DUPLICATE_CAPACITY_RULE_ID", report)
    require_nonempty_unique(df, filename, "course_category", "DUPLICATE_COURSE_CATEGORY_RULE", report)

    for idx, row in df.iterrows():
        line = line_number(idx)
        rule_id = text(row["rule_id"])
        default_capacity = parse_int(row["default_capacity"])
        capacity_min = parse_int(row["capacity_min"])
        capacity_max = parse_int(row["capacity_max"])
        min_sections = parse_int(row["default_min_sections"])
        max_sections = parse_optional_int(row["default_max_sections"])
        threshold = parse_float(row["expansion_threshold_ratio"])

        for col, value in {
            "default_capacity": default_capacity,
            "capacity_min": capacity_min,
            "capacity_max": capacity_max,
            "default_min_sections": min_sections,
        }.items():
            if value is None or value < 0:
                report.add_error("INVALID_NONNEGATIVE_INTEGER", filename, f"{col} must be a nonnegative integer.", line, rule_id)

        if max_sections is not None and max_sections < 0:
            report.add_error("INVALID_NONNEGATIVE_INTEGER", filename, "default_max_sections must be blank or nonnegative.", line, rule_id)
        if threshold is None or threshold < 0 or threshold > 1:
            report.add_error("INVALID_EXPANSION_THRESHOLD", filename, "expansion_threshold_ratio must be between 0 and 1.", line, rule_id)
        if None not in (capacity_min, default_capacity, capacity_max) and not capacity_min <= default_capacity <= capacity_max:
            report.add_error(
                "CAPACITY_BOUNDS",
                filename,
                "Must satisfy capacity_min <= default_capacity <= capacity_max.",
                line,
                rule_id,
            )
        if min_sections is not None and max_sections is not None and min_sections > max_sections:
            report.add_error(
                "SECTION_BOUNDS",
                filename,
                "Must satisfy default_min_sections <= default_max_sections.",
                line,
                rule_id,
            )


def validate_course_catalog(config: dict[str, pd.DataFrame], report: ValidationReport) -> None:
    filename = "course_catalog.csv"
    df = config.get(filename)
    rules = config.get("capacity_rules.csv")
    blocks = config.get("linked_course_blocks.csv")
    if not has_columns(df, CONFIG_COLUMNS[filename]):
        return

    require_nonempty_unique(df, filename, "course_id", "DUPLICATE_COURSE_ID", report)
    if any("prereq" in col.lower() for col in df.columns):
        report.add_error(
            "PREREQUISITE_FIELD_NOT_ALLOWED",
            filename,
            "Prerequisite eligibility is outside the V1 solver configuration.",
        )

    rule_categories = set(rules["course_category"]) if has_columns(rules, CONFIG_COLUMNS["capacity_rules.csv"]) else set()
    block_by_course = _blocks_by_course(blocks)

    for idx, row in df.iterrows():
        _validate_course_row(row, line_number(idx), rule_categories, block_by_course, report)

    indexed = df.set_index("course_id", drop=False)
    _validate_ap_physics_c(indexed, block_by_course, report)
    _validate_calc_d_linalg(indexed, report)
    _reject_external_dual_enrollment(indexed, report)


def validate_linked_course_blocks(config: dict[str, pd.DataFrame], report: ValidationReport) -> None:
    filename = "linked_course_blocks.csv"
    blocks = config.get(filename)
    courses = config.get("course_catalog.csv")
    if not has_columns(blocks, CONFIG_COLUMNS[filename]):
        return

    require_nonempty_unique(blocks, filename, "block_template_id", "DUPLICATE_BLOCK_ID", report)
    course_ids = set(courses["course_id"]) if has_columns(courses, CONFIG_COLUMNS["course_catalog.csv"]) else set()

    for idx, row in blocks.iterrows():
        line = line_number(idx)
        block_id = text(row["block_template_id"])
        if text(row["course_id"]) not in course_ids:
            report.add_error("UNKNOWN_BLOCK_COURSE", filename, "Linked block references an unknown course_id.", line, block_id)
        if text(row["period_sharing_rule"]) != "same_period":
            report.add_error("INVALID_PERIOD_SHARING_RULE", filename, "Linked semester blocks must share the same period.", line, block_id)

    required_gov = {
        "GOV_ECON_REG": "regular_regular",
        "GOV_APMACRO": "regular_ap",
        "APGOV_ECON": "ap_regular",
        "APGOV_APMACRO": "ap_ap",
    }
    indexed = blocks.set_index("block_template_id", drop=False)
    for block_id, level_mix in required_gov.items():
        if block_id not in indexed.index:
            report.add_error("MISSING_GOV_ECON_COMBO", filename, f"Missing Government/Economics block {block_id}.")
            continue
        row = indexed.loc[block_id]
        if text(row["period_sharing_rule"]) != "same_period" or text(row["allowed_level_mix"]) != level_mix:
            report.add_error(
                "GOV_ECON_BLOCK_RULE",
                filename,
                "Government/Economics blocks must share one period and allow the confirmed level mix.",
                identifier=block_id,
            )


def validate_demand_scenarios(config: dict[str, pd.DataFrame], report: ValidationReport) -> None:
    filename = "demand_scenarios.csv"
    df = config.get(filename)
    courses = config.get("course_catalog.csv")
    if not has_columns(df, CONFIG_COLUMNS[filename]):
        return

    require_nonempty_unique(df, filename, "scenario_id", "DUPLICATE_SCENARIO_ID", report)
    course_ids = set(courses["course_id"]) if has_columns(courses, CONFIG_COLUMNS["course_catalog.csv"]) else set()
    multiplier_cols = [
        "core_multiplier",
        "mainstream_multiplier",
        "popular_multiplier",
        "niche_multiplier",
        "fixed_limited_multiplier",
        "capacity_risk_course_multiplier",
    ]
    for idx, row in df.iterrows():
        line = line_number(idx)
        scenario_id = text(row["scenario_id"])
        if parse_int(row["random_seed"]) is None:
            report.add_error("INVALID_RANDOM_SEED", filename, "random_seed must be an integer.", line, scenario_id)
        for col in multiplier_cols:
            value = parse_float(row[col])
            if value is None or value < 0:
                report.add_error("INVALID_DEMAND_MULTIPLIER", filename, f"{col} must be a nonnegative number.", line, scenario_id)
        affected = parse_semicolon_list(row["affected_capacity_risk_courses"], filename, "affected_capacity_risk_courses", line, report)
        unknown = [course_id for course_id in affected if course_id not in course_ids]
        if unknown:
            report.add_error("UNKNOWN_DEMAND_COURSE", filename, "Demand scenario references unknown courses.", line, ";".join(unknown))

    if "stable_year" not in {text(value) for value in df["scenario_id"]}:
        report.add_error("MISSING_BASE_SCENARIO", filename, "stable_year baseline scenario must exist.")


def _validate_course_row(
    row: pd.Series,
    line: int,
    rule_categories: set[str],
    block_by_course: dict[str, pd.Series],
    report: ValidationReport,
) -> None:
    filename = "course_catalog.csv"
    course_id = text(row["course_id"])
    grades = parse_semicolon_list(row["eligible_grades"], filename, "eligible_grades", line, report, allow_blank=False)
    if [grade for grade in grades if grade not in VALID_GRADES]:
        report.add_error("INVALID_COURSE_GRADE", filename, "eligible_grades may only contain 9, 10, 11, and 12.", line, course_id)
    if text(row["course_category"]) not in rule_categories:
        report.add_error("UNKNOWN_COURSE_CATEGORY", filename, "course_category must reference capacity_rules.course_category.", line, course_id)

    periods = parse_int(row["periods_required"])
    occupies = parse_bool(row["occupies_school_period"])
    structure = text(row["schedule_structure"])
    if periods not in {0, 1, 2}:
        report.add_error("INVALID_PERIODS_REQUIRED", filename, "periods_required must be 0, 1, or 2.", line, course_id)
    if occupies is None:
        report.add_error("INVALID_BOOLEAN", filename, "occupies_school_period must be true or false.", line, course_id)
    if structure not in VALID_STRUCTURES:
        report.add_error("INVALID_SCHEDULE_STRUCTURE", filename, "Invalid schedule_structure.", line, course_id)
    if periods == 2 and structure != "double_period":
        report.add_error("DOUBLE_PERIOD_STRUCTURE", filename, "periods_required=2 requires schedule_structure=double_period.", line, course_id)
    if occupies is False and periods != 0:
        report.add_error("NON_SCHOOL_PERIOD_CONFLICT", filename, "occupies_school_period=false requires periods_required=0.", line, course_id)
    if occupies is False or periods == 0:
        report.add_error("ZERO_PERIOD_NOT_IN_V1", filename, "External, night, or zero-period courses are outside V1.", line, course_id)
    if parse_optional_int(row["capacity_override"]) is not None and parse_optional_int(row["capacity_override"]) <= 0:
        report.add_error("INVALID_CAPACITY_OVERRIDE", filename, "capacity_override must be blank or positive.", line, course_id)
    if parse_optional_int(row["max_sections_override"]) is not None and parse_optional_int(row["max_sections_override"]) < 0:
        report.add_error("INVALID_MAX_SECTIONS_OVERRIDE", filename, "max_sections_override must be blank or nonnegative.", line, course_id)
    if structure == "semester_block" and course_id not in block_by_course:
        report.add_error("MISSING_LINKED_BLOCK", filename, "semester_block course must have a linked_course_blocks row.", line, course_id)


def _blocks_by_course(blocks: pd.DataFrame | None) -> dict[str, pd.Series]:
    if not has_columns(blocks, CONFIG_COLUMNS["linked_course_blocks.csv"]):
        return {}
    return {text(row["course_id"]): row for _, row in blocks.iterrows()}


def _validate_ap_physics_c(
    indexed: pd.DataFrame,
    block_by_course: dict[str, pd.Series],
    report: ValidationReport,
) -> None:
    if "AP_PHYSC" not in indexed.index:
        report.add_error("MISSING_REQUIRED_COURSE", "course_catalog.csv", "Missing AP_PHYSC course.", identifier="AP_PHYSC")
        return
    row = indexed.loc["AP_PHYSC"]
    if parse_int(row["periods_required"]) != 1 or text(row["schedule_structure"]) != "semester_block":
        report.add_error("AP_PHYSC_STRUCTURE", "course_catalog.csv", "AP Physics C must be a one-period semester_block.", identifier="AP_PHYSC")
    block = block_by_course.get("AP_PHYSC")
    if block is None:
        report.add_error("AP_PHYSC_BLOCK_MISSING", "linked_course_blocks.csv", "AP Physics C must have a linked block.", identifier="AP_PHYSC")
    elif text(block["semester_1_content"]) != "Mechanics" or text(block["semester_2_content"]) != "Electricity and Magnetism":
        report.add_error(
            "AP_PHYSC_SEMESTERS",
            "linked_course_blocks.csv",
            "AP Physics C must use Mechanics then Electricity and Magnetism.",
            identifier="AP_PHYSC",
        )


def _validate_calc_d_linalg(indexed: pd.DataFrame, report: ValidationReport) -> None:
    if "CALC_D_LINALG" not in indexed.index:
        report.add_error("MISSING_REQUIRED_COURSE", "course_catalog.csv", "Missing CALC_D_LINALG course.", identifier="CALC_D_LINALG")
        return
    row = indexed.loc["CALC_D_LINALG"]
    if (
        text(row["course_category"]) != "dual_enrollment"
        or parse_int(row["periods_required"]) != 1
        or parse_bool(row["occupies_school_period"]) is not True
    ):
        report.add_error(
            "CALC_D_LINALG_STRUCTURE",
            "course_catalog.csv",
            "CALC_D_LINALG must be a TPHS daytime one-period dual-enrollment block.",
            identifier="CALC_D_LINALG",
        )


def _reject_external_dual_enrollment(indexed: pd.DataFrame, report: ValidationReport) -> None:
    for course_id, row in indexed.iterrows():
        haystack = " ".join([text(course_id), text(row["course_name"]), text(row["notes"])]).lower()
        if "external" in haystack or "zero-period" in haystack or "night" in haystack:
            report.add_error(
                "EXTERNAL_DUAL_ENROLLMENT_NOT_IN_V1",
                "course_catalog.csv",
                "External, night, or zero-period dual enrollment should not be configured for V1.",
                identifier=text(course_id),
            )
