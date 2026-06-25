from __future__ import annotations

import pandas as pd

from .constants import CONFIG_COLUMNS, VALID_PERIODS
from .helpers import has_columns, line_number, parse_int, text
from .models import ValidationReport


def validate_section_planning_config(
    config: dict[str, pd.DataFrame],
    report: ValidationReport,
) -> None:
    catalog = config.get("course_catalog.csv")
    scenarios = config.get("demand_scenarios.csv")
    course_ids = set(catalog["course_id"]) if has_columns(catalog, CONFIG_COLUMNS["course_catalog.csv"]) else set()
    scenario_ids = set(scenarios["scenario_id"]) if has_columns(scenarios, CONFIG_COLUMNS["demand_scenarios.csv"]) else set()

    _validate_section_capacity_overrides(
        config.get("section_capacity_overrides.csv"),
        course_ids,
        scenario_ids,
        report,
    )
    _validate_section_planning_rules(config.get("section_planning_rules.csv"), report)


def _validate_section_capacity_overrides(
    df: pd.DataFrame | None,
    course_ids: set[str],
    scenario_ids: set[str],
    report: ValidationReport,
) -> None:
    filename = "section_capacity_overrides.csv"
    if not has_columns(df, CONFIG_COLUMNS[filename]):
        return

    duplicates = df[df.duplicated(["scenario_id", "course_id"], keep=False)]
    for idx, row in duplicates.iterrows():
        report.add_error(
            "DUPLICATE_SECTION_CAPACITY_OVERRIDE",
            filename,
            "scenario_id and course_id override combinations must be unique.",
            line_number(idx),
            f"{row['scenario_id']}:{row['course_id']}",
        )

    for idx, row in df.iterrows():
        line = line_number(idx)
        scenario_id = text(row["scenario_id"])
        course_id = text(row["course_id"])
        capacity = parse_int(row["capacity"])
        if scenario_id not in scenario_ids:
            report.add_error("UNKNOWN_SECTION_CAPACITY_SCENARIO", filename, "scenario_id must reference demand_scenarios.csv.", line, scenario_id)
        if course_id not in course_ids:
            report.add_error("UNKNOWN_SECTION_CAPACITY_COURSE", filename, "course_id must reference course_catalog.csv.", line, course_id)
        if capacity is None or capacity <= 0:
            report.add_error("INVALID_SECTION_CAPACITY_OVERRIDE", filename, "capacity must be a positive integer.", line, course_id)


def _validate_section_planning_rules(
    df: pd.DataFrame | None,
    report: ValidationReport,
) -> None:
    filename = "section_planning_rules.csv"
    if not has_columns(df, CONFIG_COLUMNS[filename]):
        return

    seen = set()
    for idx, row in df.iterrows():
        line = line_number(idx)
        rule_id = text(row["rule_id"])
        rule_value = text(row["rule_value"])
        if not rule_id:
            report.add_error("EMPTY_SECTION_PLANNING_RULE_ID", filename, "rule_id cannot be blank.", line)
        if rule_id in seen:
            report.add_error("DUPLICATE_SECTION_PLANNING_RULE", filename, "rule_id must be unique.", line, rule_id)
        seen.add(rule_id)
        if rule_id == "positive_demand_min_sections":
            value = parse_int(rule_value)
            if value is None or value < 1:
                report.add_error("INVALID_POSITIVE_DEMAND_MIN_SECTIONS", filename, "positive_demand_min_sections must be a positive integer.", line, rule_id)
        elif rule_id == "double_period_pairs":
            _validate_period_pairs(rule_value, filename, line, report)

    required = {"positive_demand_min_sections", "double_period_pairs"}
    missing = required - seen
    for rule_id in sorted(missing):
        report.add_error("MISSING_SECTION_PLANNING_RULE", filename, "Required section planning rule is missing.", identifier=rule_id)


def _validate_period_pairs(
    value: str,
    filename: str,
    line: int,
    report: ValidationReport,
) -> None:
    if not value:
        report.add_error("EMPTY_DOUBLE_PERIOD_PAIRS", filename, "double_period_pairs cannot be blank.", line)
        return
    for raw_pair in value.split(";"):
        parts = raw_pair.split("-")
        if len(parts) != 2 or parts[0] not in VALID_PERIODS or parts[1] not in VALID_PERIODS:
            report.add_error("INVALID_DOUBLE_PERIOD_PAIR", filename, "double_period_pairs must use values like P3-P4.", line, raw_pair)
            continue
        first = int(parts[0][1:])
        second = int(parts[1][1:])
        if second - first != 1:
            report.add_error("NONCONSECUTIVE_DOUBLE_PERIOD_PAIR", filename, "double_period_pairs must be consecutive.", line, raw_pair)
