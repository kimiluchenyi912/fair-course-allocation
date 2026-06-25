from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class SectionPlanningError(Exception):
    """Raised when section planning inputs or configuration are invalid."""


@dataclass
class SectionPlanningConfig:
    config_dir: str
    scenario_id: str
    catalog: pd.DataFrame
    capacity_rules: pd.DataFrame
    capacity_overrides: pd.DataFrame
    planning_rules: pd.DataFrame
    linked_blocks: pd.DataFrame


@dataclass
class SectionPlanningResult:
    sections: pd.DataFrame
    course_demand_summary: pd.DataFrame
    period_layout_summary: pd.DataFrame
    metadata: dict
    scenario_id: str
    seed: int
