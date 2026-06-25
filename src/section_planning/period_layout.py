from __future__ import annotations

import hashlib
from collections import defaultdict

import pandas as pd

from .config import double_period_pairs
from .demand import GOV_ECON_COURSES
from .models import SectionPlanningConfig


PERIODS = [f"P{i}" for i in range(1, 8)]


def build_sections_and_layout_summary(
    config: SectionPlanningConfig,
    course_summary: pd.DataFrame,
    conflict_weights: dict[tuple[str, str], int],
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    catalog = config.catalog.set_index("course_id", drop=False)
    linked_blocks = config.linked_blocks.set_index("course_id", drop=False)
    placed: list[dict] = []
    period_loads = {period: 0 for period in PERIODS}
    course_periods: dict[str, list[set[str]]] = defaultdict(list)

    planned = course_summary[course_summary["planned_sections"] > 0].copy()
    planned = planned.sort_values(["primary_demand", "course_id"], ascending=[False, True])
    for row in planned.itertuples(index=False):
        course = catalog.loc[row.course_id]
        for section_index in range(1, int(row.planned_sections) + 1):
            periods = _choose_periods(
                config,
                str(row.course_id),
                int(section_index),
                str(course.schedule_structure),
                conflict_weights,
                course_periods,
                period_loads,
                seed,
            )
            placed.append(
                {
                    "course_id": str(row.course_id),
                    "section_index": section_index,
                    "periods": periods,
                    "capacity": int(row.section_capacity),
                    "schedule_structure": str(course.schedule_structure),
                }
            )
            course_periods[str(row.course_id)].append(set(periods))
            for period in periods:
                period_loads[period] += 1

    sections = _expand_section_rows(config, placed, linked_blocks, seed)
    layout_summary = _period_layout_summary(sections)
    diagnostics = {
        "raw_period_overlap_score": _raw_period_overlap_score(course_periods, conflict_weights),
        "unavoidable_course_pair_conflict_score": _unavoidable_course_pair_conflict_score(course_periods, conflict_weights),
        "single_period_course_count": _single_period_course_count(course_periods),
        "single_period_multi_section_course_count": _single_period_multi_section_course_count(course_periods),
        "period_balance_warnings": _period_balance_warnings(layout_summary),
    }
    return sections, layout_summary, diagnostics


def _choose_periods(
    config: SectionPlanningConfig,
    course_id: str,
    section_index: int,
    schedule_structure: str,
    conflict_weights: dict[tuple[str, str], int],
    course_periods: dict[str, list[set[str]]],
    period_loads: dict[str, int],
    seed: int,
) -> tuple[str, ...]:
    candidates = double_period_pairs(config) if schedule_structure == "double_period" else [(period,) for period in PERIODS]
    scored = []
    for periods in candidates:
        period_set = set(periods)
        conflict = 0
        for other_course, other_period_sets in course_periods.items():
            pair = tuple(sorted((course_id, other_course)))
            weight = conflict_weights.get(pair, 0)
            if weight == 0:
                continue
            if any(period_set & other_periods for other_periods in other_period_sets):
                conflict += weight
        same_course_overlap = sum(1 for other_periods in course_periods.get(course_id, []) if period_set & other_periods)
        load = sum(period_loads[period] for period in periods)
        noise = _stable_noise(seed, course_id, section_index, "-".join(periods))
        scored.append(((same_course_overlap, load, conflict, noise), periods))
    scored.sort(key=lambda item: item[0])
    return tuple(scored[0][1])


def _expand_section_rows(
    config: SectionPlanningConfig,
    placed: list[dict],
    linked_blocks: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    rows: list[dict] = []
    for item in placed:
        course_id = item["course_id"]
        section_index = int(item["section_index"])
        period_1 = item["periods"][0]
        period_2 = item["periods"][1] if len(item["periods"]) > 1 else ""
        group_id = f"{course_id}_{section_index:02d}"
        if course_id in GOV_ECON_COURSES:
            first_content, second_content = _semester_contents(linked_blocks.loc[course_id], seed, group_id)
            rows.append(_section_row(config.scenario_id, group_id, course_id, section_index, period_1, "", "semester_1", item["capacity"], group_id, course_id, first_content))
            rows.append(_section_row(config.scenario_id, group_id, course_id, section_index, period_1, "", "semester_2", item["capacity"], group_id, course_id, second_content))
        else:
            semester = "full_year"
            semester_content = ""
            if item["schedule_structure"] == "semester_block":
                semester = "paired"
                if course_id in linked_blocks.index:
                    block = linked_blocks.loc[course_id]
                    semester_content = f"{block['semester_1_content']} / {block['semester_2_content']}"
            rows.append(_section_row(config.scenario_id, group_id, course_id, section_index, period_1, period_2, semester, item["capacity"], group_id, course_id, semester_content))
    columns = [
        "scenario_id",
        "section_id",
        "course_id",
        "period_1",
        "period_2",
        "semester",
        "capacity",
        "block_id",
        "linked_section_group_id",
        "logical_block_id",
        "semester_content",
        "planning_source",
        "teacher_resource_id",
        "room_resource_id",
    ]
    return pd.DataFrame(rows, columns=columns)


def _section_row(
    scenario_id: str,
    group_id: str,
    course_id: str,
    section_index: int,
    period_1: str,
    period_2: str,
    semester: str,
    capacity: int,
    block_id: str,
    logical_block_id: str,
    semester_content: str,
) -> dict:
    suffix = semester if semester in {"semester_1", "semester_2"} else "full_year"
    return {
        "scenario_id": scenario_id,
        "section_id": f"SEC_{group_id}_{suffix}",
        "course_id": course_id,
        "period_1": period_1,
        "period_2": period_2,
        "semester": semester,
        "capacity": capacity,
        "block_id": block_id if semester in {"semester_1", "semester_2", "paired"} else "",
        "linked_section_group_id": group_id,
        "logical_block_id": logical_block_id,
        "semester_content": semester_content,
        "planning_source": "section_planner",
        "teacher_resource_id": "",
        "room_resource_id": "",
    }


def _semester_contents(block: pd.Series, seed: int, group_id: str) -> tuple[str, str]:
    first = str(block["semester_1_content"])
    second = str(block["semester_2_content"])
    if _stable_noise(seed, group_id, "semester_order", "") >= 0.5:
        return second, first
    return first, second


def _period_layout_summary(sections: pd.DataFrame) -> pd.DataFrame:
    rows = []
    logical = _logical_sections(sections)
    for period in PERIODS:
        in_period = sections[(sections["period_1"] == period) | (sections["period_2"] == period)]
        logical_in_period = logical[logical["periods"].map(lambda periods: period in periods)]
        seats_in_period = logical[logical["seat_count_period"] == period]
        rows.append(
            {
                "period": period,
                "logical_section_count": int(len(logical_in_period)),
                "section_row_count": int(len(in_period)),
                "occupied_period_slot_count": int(len(logical_in_period)),
                "planned_seats": int(seats_in_period["capacity"].sum()) if not seats_in_period.empty else 0,
                "yearlong_logical_sections": int((logical_in_period["kind"] == "yearlong").sum()),
                "semester_rows": int(in_period["semester"].isin(["semester_1", "semester_2"]).sum()),
                "linked_logical_sections": int((logical_in_period["kind"] == "linked").sum()),
                "double_period_logical_sections": int((logical_in_period["kind"] == "double_period").sum()),
            }
        )
    return pd.DataFrame(rows)


def _logical_sections(sections: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for group_id, group in sections.groupby("linked_section_group_id", sort=False):
        periods = set(group["period_1"]) | {period for period in group["period_2"] if period}
        first = group.iloc[0]
        if group["semester"].isin(["semester_1", "semester_2"]).any():
            kind = "linked"
        elif any(group["period_2"] != ""):
            kind = "double_period"
        else:
            kind = "yearlong"
        rows.append(
            {
                "linked_section_group_id": group_id,
                "course_id": first["course_id"],
                "periods": periods,
                "seat_count_period": first["period_1"],
                "capacity": int(first["capacity"]),
                "kind": kind,
            }
        )
    return pd.DataFrame(rows)


def _raw_period_overlap_score(
    course_periods: dict[str, list[set[str]]],
    conflict_weights: dict[tuple[str, str], int],
) -> int:
    score = 0
    for (first, second), weight in conflict_weights.items():
        first_periods = course_periods.get(first, [])
        second_periods = course_periods.get(second, [])
        for first_section in first_periods:
            for second_section in second_periods:
                if first_section & second_section:
                    score += weight
    return int(score)


def _unavoidable_course_pair_conflict_score(
    course_periods: dict[str, list[set[str]]],
    conflict_weights: dict[tuple[str, str], int],
) -> int:
    score = 0
    for (first, second), weight in conflict_weights.items():
        first_periods = course_periods.get(first, [])
        second_periods = course_periods.get(second, [])
        if first_periods and second_periods and not any(
            first_section.isdisjoint(second_section)
            for first_section in first_periods
            for second_section in second_periods
        ):
            score += weight
    return int(score)


def _single_period_course_count(course_periods: dict[str, list[set[str]]]) -> int:
    return sum(1 for periods in course_periods.values() if len({period for section in periods for period in section}) == 1)


def _single_period_multi_section_course_count(course_periods: dict[str, list[set[str]]]) -> int:
    return sum(
        1
        for periods in course_periods.values()
        if len(periods) > 1 and len({period for section in periods for period in section}) == 1
    )


def _period_balance_warnings(layout_summary: pd.DataFrame) -> list[str]:
    counts = layout_summary["logical_section_count"]
    average = float(counts.mean())
    warnings = []
    if counts.max() > average * 1.25:
        warnings.append(
            f"max logical section count {int(counts.max())} exceeds 1.25x average {average:.2f}"
        )
    if counts.max() - counts.min() > 15:
        warnings.append(
            f"logical section count range {int(counts.max() - counts.min())} exceeds target 15"
        )
    return warnings


def _stable_noise(seed: int, *parts: object) -> float:
    digest = hashlib.sha256(":".join([str(seed), *map(str, parts)]).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF
