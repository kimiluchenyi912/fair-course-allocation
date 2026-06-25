from __future__ import annotations

import hashlib
from pathlib import Path
from random import Random

import pandas as pd

from .models import GenerationConfig, GenerationConfigError


REQUIRED_GENERATION_FILES = [
    "grade_request_rules.csv",
    "course_choice_weights.csv",
    "fixed_course_targets.csv",
]


def load_generation_config(config_dir: str | Path, scenario_id: str) -> GenerationConfig:
    config_dir = Path(config_dir)
    tables = {
        "catalog": _read_required(config_dir / "course_catalog.csv"),
        "grade_profiles": _read_required(config_dir / "grade_profiles.csv"),
        "demand_scenarios": _read_required(config_dir / "demand_scenarios.csv"),
        "linked_blocks": _read_required(config_dir / "linked_course_blocks.csv"),
        "grade_rules": _read_required(config_dir / "grade_request_rules.csv"),
        "choice_weights": _read_required(config_dir / "course_choice_weights.csv"),
        "fixed_targets": _read_required(config_dir / "fixed_course_targets.csv"),
    }
    scenarios = tables["demand_scenarios"]
    matches = scenarios[scenarios["scenario_id"] == scenario_id]
    if matches.empty:
        raise GenerationConfigError(f"Unknown scenario_id '{scenario_id}'.")
    config = GenerationConfig(config_dir=str(config_dir), scenario=matches.iloc[0], **tables)
    validate_generation_config(config)
    return config


def validate_generation_config(config: GenerationConfig) -> None:
    course_ids = set(config.catalog["course_id"])
    scenario_id = str(config.scenario["scenario_id"])
    problems: list[str] = []
    for filename in REQUIRED_GENERATION_FILES:
        path = Path(config.config_dir) / filename
        if not path.exists():
            problems.append(f"Missing generation config file: {filename}")

    for _, row in config.grade_rules.iterrows():
        value = str(row["rule_value"])
        if ":" not in value:
            continue
        for key in parse_weight_map(value):
            if key.isdigit():
                continue
            if key not in course_ids:
                problems.append(f"grade_request_rules.csv references unknown course_id '{key}'.")

    for _, row in config.fixed_targets.iterrows():
        if row["scenario_id"] != scenario_id:
            continue
        if row["course_id"] not in course_ids:
            problems.append(f"fixed_course_targets.csv references unknown course_id '{row['course_id']}'.")

    for _, row in config.choice_weights.iterrows():
        if row["scope_type"] == "course" and row["scope_id"] not in course_ids:
            problems.append(f"course_choice_weights.csv references unknown course_id '{row['scope_id']}'.")

    scenario_courses = [
        part
        for part in str(config.scenario["affected_capacity_risk_courses"]).split(";")
        if part
    ]
    for course_id in scenario_courses:
        if course_id not in course_ids:
            problems.append(f"demand_scenarios.csv references unknown course_id '{course_id}'.")

    if problems:
        raise GenerationConfigError("\n".join(problems))


def rule_value(config: GenerationConfig, grade: int, key: str) -> str:
    rows = config.grade_rules[
        (config.grade_rules["grade"].astype(int) == grade)
        & (config.grade_rules["rule_key"] == key)
    ]
    if rows.empty:
        raise GenerationConfigError(f"Missing grade rule for grade {grade}: {key}")
    return str(rows.iloc[0]["rule_value"])


def parse_weight_map(value: str) -> dict[str, float]:
    weights: dict[str, float] = {}
    for item in value.split(";"):
        if not item:
            continue
        key, raw_weight = item.split(":", 1)
        weights[key] = float(raw_weight)
    if not weights or sum(weights.values()) <= 0:
        raise GenerationConfigError(f"Weight map has no positive weight: {value}")
    return weights


def choose_weighted(rng: Random, weights: dict[str, float]) -> str:
    total = sum(max(0.0, weight) for weight in weights.values())
    if total <= 0:
        raise GenerationConfigError("Cannot choose from nonpositive weights.")
    threshold = rng.random() * total
    running = 0.0
    for key, weight in weights.items():
        running += max(0.0, weight)
        if running >= threshold:
            return key
    return next(reversed(weights))


def course_weight(config: GenerationConfig, course: pd.Series, seed: int, grade: int) -> float:
    tier = str(course["demand_tier"])
    department = str(course["department"])
    course_id = str(course["course_id"])
    scenario_id = str(config.scenario["scenario_id"])
    weight = _scope_weight(config.choice_weights, scenario_id, grade, "demand_tier", tier)
    weight *= _scope_weight(config.choice_weights, scenario_id, grade, "department", department)
    weight *= _scope_weight(config.choice_weights, scenario_id, grade, "course", course_id)
    weight *= _scenario_tier_multiplier(config, tier)
    if course_id in str(config.scenario["affected_capacity_risk_courses"]).split(";"):
        weight *= float(config.scenario["capacity_risk_course_multiplier"])
    return weight * _stable_noise(seed, str(config.scenario["scenario_id"]), course_id)


def eligible_courses(catalog: pd.DataFrame, grade: int) -> pd.DataFrame:
    needle = str(grade)
    return catalog[catalog["eligible_grades"].map(lambda value: needle in str(value).split(";"))]


def period_units(course: pd.Series) -> int:
    return int(course["periods_required"])


def catalog_by_id(catalog: pd.DataFrame) -> dict[str, pd.Series]:
    return {str(row["course_id"]): row for _, row in catalog.iterrows()}


def is_gov_econ(course_id: str) -> bool:
    return course_id in {"GOV_ECON_REG", "GOV_APMACRO", "APGOV_ECON", "APGOV_APMACRO"}


def is_excluded_from_v1(course: pd.Series) -> bool:
    return str(course["occupies_school_period"]).lower() != "true" or int(course["periods_required"]) <= 0


def _read_required(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise GenerationConfigError(f"Missing required config file: {path}")
    return pd.read_csv(path, keep_default_na=False)


def _scope_weight(
    weights: pd.DataFrame,
    scenario_id: str,
    grade: int,
    scope_type: str,
    scope_id: str,
) -> float:
    rows = weights[
        (weights["scenario_id"] == scenario_id)
        & (weights["grade"].astype(int) == grade)
        & (weights["scope_type"] == scope_type)
        & (weights["scope_id"] == scope_id)
    ]
    if rows.empty and scenario_id != "stable_year":
        rows = weights[
            (weights["scenario_id"] == "stable_year")
            & (weights["grade"].astype(int) == grade)
            & (weights["scope_type"] == scope_type)
            & (weights["scope_id"] == scope_id)
        ]
    if rows.empty:
        return 1.0
    return float(rows.iloc[0]["weight"])


def _scenario_tier_multiplier(config: GenerationConfig, tier: str) -> float:
    column = f"{tier}_multiplier"
    if column not in config.scenario.index:
        return 1.0
    return float(config.scenario[column])


def _stable_noise(seed: int, scenario_id: str, course_id: str) -> float:
    digest = hashlib.sha256(f"{seed}:{scenario_id}:{course_id}".encode("utf-8")).hexdigest()
    value = int(digest[:8], 16) / 0xFFFFFFFF
    return 0.85 + 0.30 * value
