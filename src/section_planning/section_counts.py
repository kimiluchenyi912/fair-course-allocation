from __future__ import annotations

import pandas as pd

from src.expansion_policy import waitlist_expansion_threshold

from .models import SectionPlanningConfig, SectionPlanningError


HIGH_DEMAND_FULL_COVERAGE_THRESHOLD = 120


def build_course_demand_summary(
    config: SectionPlanningConfig,
    demands: pd.Series,
) -> pd.DataFrame:
    capacity_rules = config.capacity_rules.set_index("course_category", drop=False)
    override_key = config.capacity_overrides[
        config.capacity_overrides["scenario_id"] == config.scenario_id
    ].set_index("course_id", drop=False)

    rows: list[dict] = []
    for course in config.catalog.itertuples(index=False):
        course_id = str(course.course_id)
        demand = int(demands.get(course_id, 0))
        capacity, source_rule, override_used = _capacity_for_course(course, capacity_rules, override_key)
        ratio = _expansion_ratio(course, capacity_rules)
        threshold = waitlist_expansion_threshold(capacity, ratio)
        existing_policy_sections = _waitlist_policy_section_count(demand, capacity, threshold)
        full_coverage_floor = _full_coverage_section_floor(demand, capacity)
        high_demand_triggered = demand > HIGH_DEMAND_FULL_COVERAGE_THRESHOLD
        planned_sections = max(existing_policy_sections, full_coverage_floor)
        planned_seats = planned_sections * capacity
        remaining_waitlist = max(demand - planned_seats, 0)
        uncovered_approved_demand = max(demand - planned_seats, 0)
        rows.append(
            {
                "scenario_id": config.scenario_id,
                "course_id": course_id,
                "logical_block_id": course_id,
                "primary_demand": demand,
                "section_capacity": capacity,
                "expansion_threshold": threshold,
                "existing_policy_sections": existing_policy_sections,
                "full_coverage_floor": full_coverage_floor,
                "high_demand_guarantee_triggered": str(high_demand_triggered).lower(),
                "planned_sections": planned_sections,
                "planned_seats": planned_seats,
                "final_planned_capacity": planned_seats,
                "remaining_waitlist": remaining_waitlist,
                "uncovered_approved_demand": uncovered_approved_demand,
                "source_capacity_rule": source_rule,
                "capacity_override_used": str(override_used).lower(),
            }
        )
    return pd.DataFrame(rows)


def _capacity_for_course(
    course,
    capacity_rules: pd.DataFrame,
    override_key: pd.DataFrame,
) -> tuple[int, str, bool]:
    course_id = str(course.course_id)
    if course_id in override_key.index:
        return int(override_key.loc[course_id, "capacity"]), "section_capacity_overrides.csv", True
    if str(course.capacity_override):
        return int(course.capacity_override), "course_catalog.capacity_override", True
    category = str(course.course_category)
    if category not in capacity_rules.index:
        raise SectionPlanningError(f"No capacity rule for course category '{category}'.")
    return int(capacity_rules.loc[category, "default_capacity"]), category, False


def _expansion_ratio(course, capacity_rules: pd.DataFrame) -> float:
    category = str(course.course_category)
    if category not in capacity_rules.index:
        raise SectionPlanningError(f"No capacity rule for course category '{category}'.")
    return float(capacity_rules.loc[category, "expansion_threshold_ratio"])


def _waitlist_policy_section_count(demand: int, capacity: int, threshold: int) -> int:
    if demand <= 0:
        return 0
    sections = 1
    while max(demand - sections * capacity, 0) >= threshold:
        sections += 1
    return sections


def _full_coverage_section_floor(demand: int, capacity: int) -> int:
    if demand <= HIGH_DEMAND_FULL_COVERAGE_THRESHOLD:
        return 0
    return -(-demand // capacity)
