from __future__ import annotations

from pathlib import Path

from .config_checks import (
    validate_capacity_rules,
    validate_course_catalog,
    validate_demand_scenarios,
    validate_grade_profiles,
    validate_linked_course_blocks,
)
from .constants import CONFIG_COLUMNS, TEMPLATE_COLUMNS
from .generation_config_checks import validate_generation_config_tables
from .io import load_tables, validate_generic_tables
from .models import ValidationReport
from .policy_checks import validate_baseline_policy
from .section_planning_config_checks import validate_section_planning_config
from .template_checks import validate_templates


def validate_configuration(
    config_dir: str | Path = "data/config",
    templates_dir: str | Path = "data/templates",
    *,
    strict_policy: bool = False,
) -> ValidationReport:
    report = ValidationReport()
    config_dir = Path(config_dir)
    templates_dir = Path(templates_dir)

    config = load_tables(config_dir, CONFIG_COLUMNS, report)
    templates = load_tables(templates_dir, TEMPLATE_COLUMNS, report)

    validate_generic_tables(config, CONFIG_COLUMNS, report)
    validate_generic_tables(templates, TEMPLATE_COLUMNS, report)

    validate_grade_profiles(config.get("grade_profiles.csv"), report)
    validate_capacity_rules(config.get("capacity_rules.csv"), report)
    validate_course_catalog(config, report)
    validate_linked_course_blocks(config, report)
    validate_demand_scenarios(config, report)
    validate_generation_config_tables(config, report)
    validate_section_planning_config(config, report)
    validate_templates(config, templates, report)
    validate_baseline_policy(config, report, strict_policy=strict_policy)

    return report
