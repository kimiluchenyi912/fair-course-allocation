from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class GenerationConfigError(Exception):
    """Raised when generation configuration cannot produce valid requests."""


@dataclass
class GenerationConfig:
    config_dir: str
    catalog: pd.DataFrame
    grade_profiles: pd.DataFrame
    demand_scenarios: pd.DataFrame
    linked_blocks: pd.DataFrame
    grade_rules: pd.DataFrame
    choice_weights: pd.DataFrame
    fixed_targets: pd.DataFrame
    scenario: pd.Series


@dataclass
class GenerationResult:
    students: pd.DataFrame
    requests: pd.DataFrame
    summary: pd.DataFrame
    catalog: pd.DataFrame
    metadata: dict
    seed: int
    scenario_id: str
