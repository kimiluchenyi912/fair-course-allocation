from __future__ import annotations

import pandas as pd

from .constants import (
    CONFIG_COLUMNS,
    TEMPLATE_COLUMNS,
    VALID_PERIODS,
    VALID_PRIORITY_REASONS,
    VALID_REQUEST_TYPES,
    VALID_SEMESTERS,
    VALID_UNMET_REASON_CODES,
    VALID_UNSCHEDULED_PREFERENCES,
    VALID_GRADES,
    VALID_ASSIGNMENT_SOURCES,
)
from .helpers import has_columns, line_number, parse_bool, parse_float, parse_int, require_nonempty_unique, text
from .models import ValidationReport


def validate_templates(
    config: dict[str, pd.DataFrame],
    templates: dict[str, pd.DataFrame],
    report: ValidationReport,
) -> None:
    students = templates.get("students.csv")
    requests = templates.get("requests.csv")
    sections = templates.get("sections.csv")
    assignments = templates.get("assignments.csv")
    metrics = templates.get("metrics.csv")
    unmet = templates.get("unmet_requests.csv")
    outcomes = templates.get("student_outcomes.csv")
    catalog = config.get("course_catalog.csv")
    scenarios = config.get("demand_scenarios.csv")
    blocks = config.get("linked_course_blocks.csv")

    student_ids = _validate_students(students, report)
    course_ids = set(catalog["course_id"]) if has_columns(catalog, CONFIG_COLUMNS["course_catalog.csv"]) else set()
    block_ids = set(blocks["block_template_id"]) if has_columns(blocks, CONFIG_COLUMNS["linked_course_blocks.csv"]) else set()
    scenario_ids = set(scenarios["scenario_id"]) if has_columns(scenarios, CONFIG_COLUMNS["demand_scenarios.csv"]) else set()

    if has_columns(requests, TEMPLATE_COLUMNS["requests.csv"]):
        _validate_requests(requests, student_ids, course_ids, block_ids, report)

    section_ids = set()
    if has_columns(sections, TEMPLATE_COLUMNS["sections.csv"]):
        section_ids = _validate_sections(sections, course_ids, scenario_ids, report)

    if has_columns(assignments, TEMPLATE_COLUMNS["assignments.csv"]):
        _validate_assignments(assignments, sections, student_ids, section_ids, course_ids, report)
    if has_columns(unmet, TEMPLATE_COLUMNS["unmet_requests.csv"]):
        _validate_unmet_requests(unmet, student_ids, course_ids, report)
    if has_columns(outcomes, TEMPLATE_COLUMNS["student_outcomes.csv"]):
        _validate_student_outcomes(outcomes, student_ids, report)
    if has_columns(metrics, TEMPLATE_COLUMNS["metrics.csv"]):
        _validate_metrics(metrics, scenario_ids, report)


def _validate_students(df: pd.DataFrame | None, report: ValidationReport) -> set[str]:
    filename = "students.csv"
    if not has_columns(df, TEMPLATE_COLUMNS[filename]):
        return set()
    require_nonempty_unique(df, filename, "student_id", "DUPLICATE_STUDENT_ID", report)
    for idx, row in df.iterrows():
        line = line_number(idx)
        student_id = text(row["student_id"])
        if text(row["grade"]) not in VALID_GRADES:
            report.add_error("INVALID_TEMPLATE_STUDENT_GRADE", filename, "grade must be 9, 10, 11, or 12.", line, student_id)
        if parse_int(row["target_course_count"]) not in {5, 6, 7}:
            report.add_error("INVALID_TARGET_COURSE_COUNT", filename, "target_course_count must be 5, 6, or 7.", line, student_id)
        if text(row["unscheduled_preference"]) not in VALID_UNSCHEDULED_PREFERENCES:
            report.add_error("INVALID_UNSCHEDULED_PREFERENCE", filename, "Invalid unscheduled_preference.", line, student_id)
        protected = parse_bool(row["priority_protected"])
        if protected is None:
            report.add_error("INVALID_PRIORITY_PROTECTED", filename, "priority_protected must be true or false.", line, student_id)
        reason = text(row["priority_reason"])
        valid_year = text(row["priority_valid_school_year"])
        if reason not in VALID_PRIORITY_REASONS:
            report.add_error("INVALID_PRIORITY_REASON", filename, "priority_reason must be blank or prior_year_unmet_primary.", line, student_id)
        if protected is False and (reason or valid_year):
            report.add_error("INACTIVE_PRIORITY_FIELDS", filename, "Unprotected students should leave priority reason and valid year blank.", line, student_id)
        if protected is True and (not reason or not valid_year):
            report.add_error("MISSING_PRIORITY_FIELDS", filename, "Protected students need priority_reason and priority_valid_school_year.", line, student_id)
    return set(df["student_id"])


def _validate_requests(
    df: pd.DataFrame,
    student_ids: set[str],
    course_ids: set[str],
    block_ids: set[str],
    report: ValidationReport,
) -> None:
    filename = "requests.csv"
    for idx, row in df.iterrows():
        line = line_number(idx)
        student_id = text(row["student_id"])
        course_id = text(row["course_id"])
        request_type = text(row["request_type"])
        if student_id not in student_ids:
            report.add_error("UNKNOWN_REQUEST_STUDENT", filename, "Request references an unknown student_id.", line, student_id)
        if course_id not in course_ids:
            report.add_error("UNKNOWN_REQUEST_COURSE", filename, "Request references an unknown course_id.", line, course_id)
        if request_type not in VALID_REQUEST_TYPES:
            report.add_error("INVALID_REQUEST_TYPE", filename, "request_type must be primary or alternate.", line, student_id)
        rank = text(row["request_rank"])
        if request_type == "alternate":
            parsed_rank = parse_int(rank)
            if parsed_rank is None or parsed_rank <= 0:
                report.add_error("INVALID_ALTERNATE_RANK", filename, "Alternate request_rank must be a positive integer.", line, student_id)
        elif request_type == "primary" and rank:
            report.add_error("PRIMARY_RANK_NOT_BLANK", filename, "Primary request_rank should be blank.", line, student_id)
        block_id = text(row["must_share_block_id"])
        if block_id and block_id not in block_ids:
            report.add_error("UNKNOWN_REQUEST_BLOCK", filename, "must_share_block_id references an unknown linked block.", line, block_id)

    alternates = df[df["request_type"].astype(str) == "alternate"]
    duplicates = alternates[alternates.duplicated(["student_id", "request_rank"], keep=False)]
    for idx, row in duplicates.iterrows():
        report.add_error(
            "DUPLICATE_ALTERNATE_RANK",
            filename,
            "Alternate ranks must be unique within each student.",
            line_number(idx),
            f"{row['student_id']}:{row['request_rank']}",
        )


def _validate_sections(
    df: pd.DataFrame,
    course_ids: set[str],
    scenario_ids: set[str],
    report: ValidationReport,
) -> set[str]:
    filename = "sections.csv"
    require_nonempty_unique(df, filename, "section_id", "DUPLICATE_SECTION_ID", report)
    for idx, row in df.iterrows():
        line = line_number(idx)
        section_id = text(row["section_id"])
        if text(row["scenario_id"]) not in scenario_ids:
            report.add_error("UNKNOWN_SECTION_SCENARIO", filename, "Section references an unknown scenario_id.", line, section_id)
        if text(row["course_id"]) not in course_ids:
            report.add_error("UNKNOWN_SECTION_COURSE", filename, "Section references an unknown course_id.", line, section_id)
        for col in ["period_1", "period_2"]:
            period = text(row[col])
            if period and period not in VALID_PERIODS:
                report.add_error("INVALID_SECTION_PERIOD", filename, f"{col} must be P1 through P7 or blank.", line, section_id)
        if text(row["semester"]) not in VALID_SEMESTERS:
            report.add_error("INVALID_SECTION_SEMESTER", filename, "Invalid semester value.", line, section_id)
        capacity = parse_int(row["capacity"])
        if capacity is None or capacity <= 0:
            report.add_error("INVALID_SECTION_CAPACITY", filename, "Section capacity must be a positive integer.", line, section_id)
    return set(df["section_id"])


def _validate_assignments(
    assignments: pd.DataFrame,
    sections: pd.DataFrame | None,
    student_ids: set[str],
    section_ids: set[str],
    course_ids: set[str],
    report: ValidationReport,
) -> None:
    filename = "assignments.csv"
    section_course = {}
    if has_columns(sections, TEMPLATE_COLUMNS["sections.csv"]):
        section_course = dict(zip(sections["section_id"], sections["course_id"]))
    for idx, row in assignments.iterrows():
        line = line_number(idx)
        student_id = text(row["student_id"])
        section_id = text(row["section_id"])
        course_id = text(row["course_id"])
        if student_id not in student_ids:
            report.add_error("UNKNOWN_ASSIGNMENT_STUDENT", filename, "Assignment references an unknown student_id.", line, student_id)
        if section_id not in section_ids:
            report.add_error("UNKNOWN_ASSIGNMENT_SECTION", filename, "Assignment references an unknown section_id.", line, section_id)
        if course_id not in course_ids:
            report.add_error("UNKNOWN_ASSIGNMENT_COURSE", filename, "Assignment references an unknown course_id.", line, course_id)
        if section_id in section_course and course_id != text(section_course[section_id]):
            report.add_error("ASSIGNMENT_SECTION_COURSE_MISMATCH", filename, "Assignment course_id must match the referenced section.", line, section_id)
        if text(row["request_type"]) not in VALID_REQUEST_TYPES:
            report.add_error("INVALID_ASSIGNMENT_REQUEST_TYPE", filename, "request_type must be primary or alternate.", line, student_id)
        source = text(row["assignment_source"])
        if source not in VALID_ASSIGNMENT_SOURCES:
            report.add_error("INVALID_ASSIGNMENT_SOURCE", filename, "assignment_source must be primary or alternate.", line, student_id)
        replacement_course_id = text(row["replaced_primary_course_id"])
        replacement_block_id = text(row["replaced_primary_block_id"])
        if replacement_course_id and replacement_course_id not in course_ids:
            report.add_error("UNKNOWN_REPLACED_PRIMARY_COURSE", filename, "replaced_primary_course_id references an unknown course_id.", line, replacement_course_id)
        if source == "alternate" and not (replacement_course_id or replacement_block_id):
            report.add_error("MISSING_ALTERNATE_REPLACEMENT_TARGET", filename, "Alternate assignments must name the replaced primary course or block.", line, student_id)


def _validate_unmet_requests(
    df: pd.DataFrame,
    student_ids: set[str],
    course_ids: set[str],
    report: ValidationReport,
) -> None:
    filename = "unmet_requests.csv"
    for idx, row in df.iterrows():
        line = line_number(idx)
        student_id = text(row["student_id"])
        course_id = text(row["course_id"])
        replacement_course_id = text(row["replacement_course_id"])
        if student_id not in student_ids:
            report.add_error("UNKNOWN_UNMET_STUDENT", filename, "Unmet request references an unknown student_id.", line, student_id)
        if course_id not in course_ids:
            report.add_error("UNKNOWN_UNMET_COURSE", filename, "Unmet request references an unknown course_id.", line, course_id)
        if text(row["request_type"]) not in VALID_REQUEST_TYPES:
            report.add_error("INVALID_UNMET_REQUEST_TYPE", filename, "request_type must be primary or alternate.", line, student_id)
        if text(row["reason_code"]) not in VALID_UNMET_REASON_CODES:
            report.add_error("INVALID_UNMET_REASON_CODE", filename, "Invalid unmet reason_code.", line, student_id)
        candidate_sections = parse_int(row["candidate_sections"])
        if candidate_sections is None or candidate_sections < 0:
            report.add_error("INVALID_CANDIDATE_SECTIONS", filename, "candidate_sections must be a nonnegative integer.", line, student_id)
        if replacement_course_id and replacement_course_id not in course_ids:
            report.add_error("UNKNOWN_REPLACEMENT_COURSE", filename, "replacement_course_id references an unknown course_id.", line, replacement_course_id)
        replacement_rank = text(row["replacement_alternate_rank"])
        replacement_units = text(row["replacement_period_units"])
        if replacement_course_id:
            rank = parse_int(replacement_rank)
            units = parse_int(replacement_units)
            if rank is None or rank <= 0:
                report.add_error("INVALID_REPLACEMENT_ALTERNATE_RANK", filename, "replacement_alternate_rank must be a positive integer when replacement_course_id is set.", line, student_id)
            if units is None or units <= 0:
                report.add_error("INVALID_REPLACEMENT_PERIOD_UNITS", filename, "replacement_period_units must be positive when replacement_course_id is set.", line, student_id)
        elif replacement_rank or replacement_units:
            report.add_error("REPLACEMENT_FIELDS_WITHOUT_COURSE", filename, "Replacement rank/units require replacement_course_id.", line, student_id)
        if parse_bool(row["earns_next_year_priority"]) is None:
            report.add_error("INVALID_NEXT_YEAR_PRIORITY", filename, "earns_next_year_priority must be true or false.", line, student_id)


def _validate_student_outcomes(
    df: pd.DataFrame,
    student_ids: set[str],
    report: ValidationReport,
) -> None:
    filename = "student_outcomes.csv"
    require_nonempty_unique(df, filename, "student_id", "DUPLICATE_STUDENT_OUTCOME", report)
    for idx, row in df.iterrows():
        line = line_number(idx)
        student_id = text(row["student_id"])
        if student_id not in student_ids:
            report.add_error("UNKNOWN_OUTCOME_STUDENT", filename, "Outcome references an unknown student_id.", line, student_id)
        for col in ["primary_unmet_count", "alternate_assigned_count"]:
            value = parse_int(row[col])
            if value is None or value < 0:
                report.add_error("INVALID_OUTCOME_COUNT", filename, f"{col} must be a nonnegative integer.", line, student_id)
        for col in ["schedule_complete", "earns_next_year_priority"]:
            if parse_bool(row[col]) is None:
                report.add_error("INVALID_OUTCOME_BOOLEAN", filename, f"{col} must be true or false.", line, student_id)


def _validate_metrics(df: pd.DataFrame, scenario_ids: set[str], report: ValidationReport) -> None:
    filename = "metrics.csv"
    seen = set()
    for idx, row in df.iterrows():
        line = line_number(idx)
        key = (text(row["scenario_id"]), text(row["algorithm"]), text(row["random_seed"]))
        if key in seen:
            report.add_error("DUPLICATE_METRIC_ROW", filename, "Metrics rows must be unique by scenario, algorithm, and seed.", line, ":".join(key))
        seen.add(key)
        if key[0] not in scenario_ids:
            report.add_error("UNKNOWN_METRIC_SCENARIO", filename, "Metric references an unknown scenario_id.", line, key[0])
        if parse_int(row["random_seed"]) is None:
            report.add_error("INVALID_METRIC_RANDOM_SEED", filename, "random_seed must be an integer.", line, key[0])
        for col in ["complete_schedule_rate", "primary_fulfillment_rate", "alternate_use_rate", "assignment_churn_rate"]:
            value = parse_float(row[col])
            if value is None or value < 0 or value > 1:
                report.add_error("INVALID_METRIC_RATE", filename, f"{col} must be between 0 and 1.", line, key[0])
        for col in ["mean_unmet_primary", "max_unmet_primary", "solve_time_seconds"]:
            value = parse_float(row[col])
            if value is None or value < 0:
                report.add_error("INVALID_METRIC_NONNEGATIVE", filename, f"{col} must be nonnegative.", line, key[0])
