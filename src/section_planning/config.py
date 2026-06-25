from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.validation import validate_configuration

from .models import SectionPlanningConfig, SectionPlanningError


def load_section_planning_config(
    config_dir: str | Path,
    templates_dir: str | Path,
    scenario_id: str,
) -> SectionPlanningConfig:
    report = validate_configuration(config_dir, templates_dir)
    if report.errors:
        raise SectionPlanningError(report.to_text())

    config_dir = Path(config_dir)
    demand_scenarios = _read(config_dir / "demand_scenarios.csv")
    if scenario_id not in set(demand_scenarios["scenario_id"]):
        raise SectionPlanningError(f"Unknown scenario_id '{scenario_id}'.")
    return SectionPlanningConfig(
        config_dir=str(config_dir),
        scenario_id=scenario_id,
        catalog=_read(config_dir / "course_catalog.csv"),
        capacity_rules=_read(config_dir / "capacity_rules.csv"),
        capacity_overrides=_read(config_dir / "section_capacity_overrides.csv"),
        planning_rules=_read(config_dir / "section_planning_rules.csv"),
        linked_blocks=_read(config_dir / "linked_course_blocks.csv"),
    )


def planning_rule(config: SectionPlanningConfig, rule_id: str) -> str:
    rows = config.planning_rules[config.planning_rules["rule_id"] == rule_id]
    if rows.empty:
        raise SectionPlanningError(f"Missing section planning rule '{rule_id}'.")
    return str(rows.iloc[0]["rule_value"])


def double_period_pairs(config: SectionPlanningConfig) -> list[tuple[str, str]]:
    pairs = []
    for item in planning_rule(config, "double_period_pairs").split(";"):
        first, second = item.split("-", 1)
        pairs.append((first, second))
    return pairs


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SectionPlanningError(f"Missing required file: {path}")
    return pd.read_csv(path, keep_default_na=False)
