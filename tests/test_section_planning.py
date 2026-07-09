from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.generation import generate_synthetic_dataset
from src.section_planning import plan_sections
from src.section_planning.models import SectionPlanningConfig
from src.section_planning.section_counts import build_course_demand_summary
from src.validation import validate_configuration


REPO_ROOT = Path(__file__).resolve().parents[1]
VALID_PERIODS = {f"P{i}" for i in range(1, 8)}
GOV_ECON_COURSES = {"GOV_ECON_REG", "GOV_APMACRO", "APGOV_ECON", "APGOV_APMACRO"}


@pytest.fixture(scope="module")
def planned_result(tmp_path_factory: pytest.TempPathFactory):
    config_dir, _ = copy_inputs(tmp_path_factory.mktemp("section_plan"))
    generated = generate_synthetic_dataset(config_dir, "stable_year", 2026)
    planned = plan_sections(generated.students, generated.requests, config_dir, "stable_year", 2026)
    return generated, planned, config_dir


def copy_inputs(base: Path) -> tuple[Path, Path]:
    config_dir = base / "config"
    templates_dir = base / "templates"
    shutil.copytree(REPO_ROOT / "data" / "config", config_dir)
    shutil.copytree(REPO_ROOT / "data" / "templates", templates_dir)
    return config_dir, templates_dir


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False)


def write_csv(path: Path, df: pd.DataFrame) -> None:
    df.to_csv(path, index=False)


def expected_sections(demand: int, capacity: int, threshold: int) -> int:
    if demand <= 0:
        return 0
    sections = 1
    while max(demand - sections * capacity, 0) >= threshold:
        sections += 1
    return sections


def mini_section_config(
    course_id: str = "TEST_COURSE",
    category: str = "normal_academic",
    capacity: int = 40,
    course_name: str = "Test Course",
    department: str = "Electives",
) -> SectionPlanningConfig:
    return SectionPlanningConfig(
        config_dir="",
        scenario_id="stable_year",
        catalog=pd.DataFrame(
            [
                {
                    "course_id": course_id,
                    "course_name": course_name,
                    "department": department,
                    "course_category": category,
                    "capacity_override": "",
                }
            ]
        ),
        capacity_rules=pd.DataFrame(
            [
                {
                    "course_category": category,
                    "default_capacity": capacity,
                    "expansion_threshold_ratio": 0.50,
                }
            ]
        ),
        capacity_overrides=pd.DataFrame(columns=["scenario_id", "course_id", "capacity"]),
        planning_rules=pd.DataFrame(),
        linked_blocks=pd.DataFrame(),
    )


def summary_for_demand(
    demand: int,
    capacity: int = 40,
    course_id: str = "TEST_COURSE",
    course_name: str = "Test Course",
    department: str = "Electives",
) -> pd.Series:
    config = mini_section_config(
        course_id=course_id,
        capacity=capacity,
        course_name=course_name,
        department=department,
    )
    return build_course_demand_summary(config, pd.Series({course_id: demand})).iloc[0]


def synthetic_students(count: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "student_id": [f"STU_{index:03d}" for index in range(1, count + 1)],
            "grade": [12] * count,
            "target_course_count": [7] * count,
        }
    )


def synthetic_requests(
    count: int,
    course_id: str,
    rows_per_student: int = 1,
    request_group: str = "",
    block_id: str = "",
) -> pd.DataFrame:
    rows = []
    for student_id in synthetic_students(count)["student_id"]:
        for _ in range(rows_per_student):
            rows.append((student_id, course_id, "primary", "", request_group, block_id))
    return pd.DataFrame(
        rows,
        columns=[
            "student_id",
            "course_id",
            "request_type",
            "request_rank",
            "request_group",
            "must_share_block_id",
        ],
    )


def test_current_generated_data_can_be_planned(planned_result) -> None:
    _, planned, _ = planned_result

    assert not planned.sections.empty
    assert not planned.course_demand_summary.empty
    assert not planned.period_layout_summary.empty


def test_same_seed_section_plan_is_deterministic(tmp_path: Path) -> None:
    config_dir, _ = copy_inputs(tmp_path)
    generated = generate_synthetic_dataset(config_dir, "stable_year", 2026)

    first = plan_sections(generated.students, generated.requests, config_dir, "stable_year", 2026)
    second = plan_sections(generated.students, generated.requests, config_dir, "stable_year", 2026)

    pd.testing.assert_frame_equal(first.sections, second.sections)
    pd.testing.assert_frame_equal(first.course_demand_summary, second.course_demand_summary)
    pd.testing.assert_frame_equal(first.period_layout_summary, second.period_layout_summary)
    assert first.metadata == second.metadata


def test_equivalent_input_row_order_does_not_change_section_counts_or_layout(tmp_path: Path) -> None:
    config_dir, _ = copy_inputs(tmp_path)
    generated = generate_synthetic_dataset(config_dir, "stable_year", 2026)

    first = plan_sections(generated.students, generated.requests, config_dir, "stable_year", 2026)
    shuffled_students = generated.students.sample(frac=1, random_state=17).reset_index(drop=True)
    shuffled_requests = generated.requests.sample(frac=1, random_state=18).reset_index(drop=True)
    second = plan_sections(shuffled_students, shuffled_requests, config_dir, "stable_year", 2026)

    pd.testing.assert_frame_equal(first.course_demand_summary, second.course_demand_summary)
    pd.testing.assert_frame_equal(first.sections, second.sections)
    pd.testing.assert_frame_equal(first.period_layout_summary, second.period_layout_summary)


def test_different_seed_changes_some_period_layout(tmp_path: Path) -> None:
    config_dir, _ = copy_inputs(tmp_path)
    generated = generate_synthetic_dataset(config_dir, "stable_year", 2026)

    first = plan_sections(generated.students, generated.requests, config_dir, "stable_year", 2026)
    second = plan_sections(generated.students, generated.requests, config_dir, "stable_year", 2027)

    assert not first.sections[["period_1", "period_2"]].equals(second.sections[["period_1", "period_2"]])


def test_only_primary_requests_count_for_demand(tmp_path: Path) -> None:
    config_dir, _ = copy_inputs(tmp_path)
    generated = generate_synthetic_dataset(config_dir, "stable_year", 2026)
    zero_course = "AP_EURO"
    generated.requests = generated.requests[generated.requests["course_id"] != zero_course].copy()
    generated.requests.loc[len(generated.requests)] = [
        generated.students.iloc[0]["student_id"],
        zero_course,
        "alternate",
        1,
        "alternate",
        "",
    ]

    planned = plan_sections(generated.students, generated.requests, config_dir, "stable_year", 2026)
    row = planned.course_demand_summary.set_index("course_id").loc[zero_course]

    assert int(row["primary_demand"]) == 0
    assert int(row["planned_sections"]) == 0


def test_positive_demand_gets_at_least_one_section(planned_result) -> None:
    _, planned, _ = planned_result
    positive = planned.course_demand_summary[planned.course_demand_summary["primary_demand"] > 0]

    assert positive["planned_sections"].ge(1).all()


def test_zero_demand_gets_zero_sections(planned_result) -> None:
    _, planned, _ = planned_result
    zero = planned.course_demand_summary[planned.course_demand_summary["primary_demand"] == 0]

    assert zero["planned_sections"].eq(0).all()


@pytest.mark.parametrize(
    ("demand", "capacity", "threshold", "sections", "remaining"),
    [
        (19, 40, 20, 1, 0),
        (20, 40, 20, 1, 0),
        (59, 40, 20, 1, 19),
        (60, 40, 20, 2, 0),
        (37, 25, 13, 1, 12),
        (38, 25, 13, 2, 0),
        (74, 50, 25, 1, 24),
        (75, 50, 25, 2, 0),
    ],
)
def test_section_count_boundaries(demand: int, capacity: int, threshold: int, sections: int, remaining: int) -> None:
    planned = expected_sections(demand, capacity, threshold)

    assert planned == sections
    assert max(demand - planned * capacity, 0) == remaining


def test_capacity_40_demand_120_does_not_trigger_high_demand_floor() -> None:
    row = summary_for_demand(120, 40)

    assert int(row["existing_policy_sections"]) == 3
    assert int(row["full_coverage_floor"]) == 0
    assert str(row["high_demand_guarantee_triggered"]) == "false"
    assert int(row["planned_sections"]) == 3
    assert int(row["uncovered_approved_demand"]) == 0


@pytest.mark.parametrize(
    ("demand", "expected"),
    [
        (121, 4),
        (160, 4),
        (161, 5),
    ],
)
def test_capacity_40_demand_above_120_applies_full_coverage_floor(demand: int, expected: int) -> None:
    row = summary_for_demand(demand, 40)

    assert int(row["full_coverage_floor"]) == expected
    assert str(row["high_demand_guarantee_triggered"]) == "true"
    assert int(row["planned_sections"]) == expected
    assert int(row["final_planned_capacity"]) >= demand
    assert int(row["uncovered_approved_demand"]) == 0


def test_capacity_25_demand_121_uses_effective_capacity_for_full_coverage_floor() -> None:
    row = summary_for_demand(121, 25)

    assert int(row["section_capacity"]) == 25
    assert int(row["full_coverage_floor"]) == 5
    assert int(row["planned_sections"]) == 5
    assert int(row["final_planned_capacity"]) == 125
    assert int(row["uncovered_approved_demand"]) == 0


def test_demand_above_120_final_capacity_covers_logical_demand() -> None:
    row = summary_for_demand(277, 40)

    assert int(row["planned_seats"]) >= int(row["primary_demand"])
    assert int(row["uncovered_approved_demand"]) == 0


def test_demand_at_or_below_120_keeps_waitlist_policy_result() -> None:
    row = summary_for_demand(99, 40)

    assert int(row["existing_policy_sections"]) == expected_sections(99, 40, 20)
    assert int(row["full_coverage_floor"]) == 0
    assert int(row["planned_sections"]) == int(row["existing_policy_sections"])
    assert int(row["remaining_waitlist"]) == 19


def test_final_section_count_is_max_of_waitlist_policy_and_full_coverage_floor() -> None:
    row = summary_for_demand(121, 40)

    assert int(row["planned_sections"]) == max(
        int(row["existing_policy_sections"]),
        int(row["full_coverage_floor"]),
    )
    assert int(row["existing_policy_sections"]) == 3
    assert int(row["full_coverage_floor"]) == 4


def test_course_name_and_department_do_not_control_high_demand_trigger() -> None:
    core_named_low = summary_for_demand(
        120,
        40,
        course_id="REQUIRED_SOUNDING_CORE",
        course_name="Required Core Seminar",
        department="Graduation Requirements",
    )
    elective_high = summary_for_demand(
        121,
        40,
        course_id="POPULAR_ELECTIVE",
        course_name="Popular Elective",
        department="Visual and Performing Arts",
    )

    assert str(core_named_low["high_demand_guarantee_triggered"]) == "false"
    assert int(core_named_low["planned_sections"]) == 3
    assert str(elective_high["high_demand_guarantee_triggered"]) == "true"
    assert int(elective_high["planned_sections"]) == 4


def test_ap_stats_expected_count(planned_result) -> None:
    _, planned, _ = planned_result
    row = planned.course_demand_summary.set_index("course_id").loc["AP_STATS"]

    assert int(row["primary_demand"]) == 125
    assert int(row["section_capacity"]) == 40
    assert int(row["existing_policy_sections"]) == 3
    assert int(row["full_coverage_floor"]) == 4
    assert str(row["high_demand_guarantee_triggered"]) == "true"
    assert int(row["planned_sections"]) == 4
    assert int(row["planned_seats"]) == 160
    assert int(row["remaining_waitlist"]) == 0
    assert int(row["uncovered_approved_demand"]) == 0


def test_calc_d_linalg_uses_stable_year_capacity_override(planned_result) -> None:
    _, planned, _ = planned_result
    row = planned.course_demand_summary.set_index("course_id").loc["CALC_D_LINALG"]

    assert int(row["primary_demand"]) == 50
    assert int(row["section_capacity"]) == 45
    assert int(row["planned_sections"]) == 1
    assert int(row["planned_seats"]) == 45
    assert int(row["remaining_waitlist"]) == 5
    assert str(row["capacity_override_used"]).lower() == "true"


def test_ap_csa_uses_capacity_25_and_repeated_expansion(planned_result) -> None:
    _, planned, _ = planned_result
    row = planned.course_demand_summary.set_index("course_id").loc["AP_CSA"]
    demand = int(row["primary_demand"])

    assert int(row["section_capacity"]) == 25
    assert int(row["planned_sections"]) == expected_sections(demand, 25, 13)
    assert int(row["remaining_waitlist"]) < 13


def test_remaining_waitlist_is_below_threshold(planned_result) -> None:
    _, planned, _ = planned_result
    summary = planned.course_demand_summary

    assert (summary["remaining_waitlist"] < summary["expansion_threshold"]).all()


def test_no_hard_max_section_limits(planned_result) -> None:
    _, _, config_dir = planned_result
    catalog = read_csv(config_dir / "course_catalog.csv")
    rules = read_csv(config_dir / "capacity_rules.csv")

    assert catalog["max_sections_override"].map(str).str.strip().eq("").all()
    assert rules["default_max_sections"].map(str).str.strip().eq("").all()
    assert set(rules["expansion_allowed"].map(lambda value: str(value).lower())) == {"true"}


def test_gov_econ_demand_is_not_double_counted(planned_result) -> None:
    generated, planned, _ = planned_result
    primary = generated.requests[generated.requests["request_type"] == "primary"]
    summary = planned.course_demand_summary.set_index("course_id")

    for course_id in GOV_ECON_COURSES:
        request_rows = int((primary["course_id"] == course_id).sum())
        assert int(summary.loc[course_id, "primary_demand"]) * 2 == request_rows


def test_high_demand_gov_econ_raw_semester_rows_are_not_double_counted(tmp_path: Path) -> None:
    config_dir, _ = copy_inputs(tmp_path)
    planned = plan_sections(
        synthetic_students(121),
        synthetic_requests(121, "GOV_ECON_REG", rows_per_student=2, request_group="gov_econ_block", block_id="GOV_ECON_REG"),
        config_dir,
        "stable_year",
        2026,
    )
    row = planned.course_demand_summary.set_index("course_id").loc["GOV_ECON_REG"]

    assert int(row["primary_demand"]) == 121
    assert int(row["planned_sections"]) == 4
    assert int(row["planned_seats"]) == 160
    assert int(row["uncovered_approved_demand"]) == 0


def test_gov_econ_semester_rows_share_period_and_group(planned_result) -> None:
    _, planned, _ = planned_result
    gov = planned.sections[planned.sections["course_id"].isin(GOV_ECON_COURSES)]

    for _, group in gov.groupby("linked_section_group_id"):
        assert set(group["semester"]) == {"semester_1", "semester_2"}
        assert group["period_1"].nunique() == 1
        assert group["period_2"].eq("").all()
        assert group["capacity"].nunique() == 1


def test_high_demand_gov_econ_sections_expand_to_two_semester_rows(tmp_path: Path) -> None:
    config_dir, _ = copy_inputs(tmp_path)
    planned = plan_sections(
        synthetic_students(121),
        synthetic_requests(121, "GOV_ECON_REG", rows_per_student=2, request_group="gov_econ_block", block_id="GOV_ECON_REG"),
        config_dir,
        "stable_year",
        2026,
    )
    gov = planned.sections[planned.sections["course_id"] == "GOV_ECON_REG"]

    assert gov["linked_section_group_id"].nunique() == 4
    assert len(gov) == 8
    for _, group in gov.groupby("linked_section_group_id"):
        assert set(group["semester"]) == {"semester_1", "semester_2"}
        assert group["period_1"].nunique() == 1


def test_gov_econ_semester_content_is_complementary(planned_result) -> None:
    _, planned, _ = planned_result
    gov = planned.sections[planned.sections["course_id"].isin(GOV_ECON_COURSES)]

    for _, group in gov.groupby("linked_section_group_id"):
        assert group["semester_content"].nunique() == 2


def test_math2_3_sections_use_consecutive_double_periods(planned_result) -> None:
    _, planned, _ = planned_result
    math23 = planned.sections[planned.sections["course_id"] == "MATH2_3_HA"]

    assert not math23.empty
    for row in math23.itertuples(index=False):
        assert row.period_1 in VALID_PERIODS
        assert row.period_2 in VALID_PERIODS
        assert int(row.period_2[1:]) - int(row.period_1[1:]) == 1


def test_high_demand_math2_3_counts_logical_requests_once(tmp_path: Path) -> None:
    config_dir, _ = copy_inputs(tmp_path)
    planned = plan_sections(
        synthetic_students(121),
        synthetic_requests(121, "MATH2_3_HA"),
        config_dir,
        "stable_year",
        2026,
    )
    row = planned.course_demand_summary.set_index("course_id").loc["MATH2_3_HA"]

    assert int(row["primary_demand"]) == 121
    assert int(row["planned_sections"]) == 4
    assert int(row["planned_seats"]) == 160
    assert int(row["uncovered_approved_demand"]) == 0


def test_high_demand_math2_3_sections_remain_one_double_period_row(tmp_path: Path) -> None:
    config_dir, _ = copy_inputs(tmp_path)
    planned = plan_sections(
        synthetic_students(121),
        synthetic_requests(121, "MATH2_3_HA"),
        config_dir,
        "stable_year",
        2026,
    )
    math23 = planned.sections[planned.sections["course_id"] == "MATH2_3_HA"]

    assert len(math23) == 4
    assert math23["linked_section_group_id"].nunique() == 4
    assert math23["period_2"].ne("").all()


def test_ap_physics_c_sections_use_one_period(planned_result) -> None:
    _, planned, _ = planned_result
    ap_physc = planned.sections[planned.sections["course_id"] == "AP_PHYSC"]

    assert not ap_physc.empty
    assert ap_physc["period_1"].isin(VALID_PERIODS).all()
    assert ap_physc["period_2"].eq("").all()


def test_high_demand_ap_physics_c_sections_remain_single_period_rows(tmp_path: Path) -> None:
    config_dir, _ = copy_inputs(tmp_path)
    planned = plan_sections(
        synthetic_students(121),
        synthetic_requests(121, "AP_PHYSC"),
        config_dir,
        "stable_year",
        2026,
    )
    ap_physc = planned.sections[planned.sections["course_id"] == "AP_PHYSC"]

    assert len(ap_physc) == 4
    assert ap_physc["semester"].eq("paired").all()
    assert ap_physc["period_1"].isin(VALID_PERIODS).all()
    assert ap_physc["period_2"].eq("").all()


def test_section_ids_are_unique(planned_result) -> None:
    _, planned, _ = planned_result

    assert planned.sections["section_id"].is_unique


def test_all_section_course_references_are_valid(planned_result) -> None:
    _, planned, config_dir = planned_result
    course_ids = set(read_csv(config_dir / "course_catalog.csv")["course_id"])

    assert set(planned.sections["course_id"]).issubset(course_ids)


def test_all_section_periods_are_valid(planned_result) -> None:
    _, planned, _ = planned_result

    assert planned.sections["period_1"].isin(VALID_PERIODS).all()
    assert set(planned.sections.loc[planned.sections["period_2"] != "", "period_2"]).issubset(VALID_PERIODS)


def test_multi_section_courses_cover_multiple_periods(planned_result) -> None:
    _, planned, _ = planned_result
    eng9 = planned.sections[planned.sections["course_id"] == "ENG9"]

    assert eng9["period_1"].nunique() > 1


def test_three_section_single_period_course_covers_three_periods(planned_result) -> None:
    _, planned, _ = planned_result
    ap_chem = planned.sections[planned.sections["course_id"] == "AP_CHEM"]

    assert len(ap_chem) == 3
    assert ap_chem["period_1"].nunique() == 3


def test_five_section_single_period_course_covers_five_periods(planned_result) -> None:
    _, planned, _ = planned_result
    ap_csa = planned.sections[planned.sections["course_id"] == "AP_CSA"]

    assert len(ap_csa) == 5
    assert ap_csa["period_1"].nunique() == 5


def test_nine_section_single_period_course_covers_all_periods_evenly(planned_result) -> None:
    _, planned, _ = planned_result
    eng10 = planned.sections[planned.sections["course_id"] == "ENG10"]
    counts = eng10.groupby("period_1").size()

    assert len(eng10) == 9
    assert set(counts.index) == VALID_PERIODS
    assert counts.max() - counts.min() <= 1


def test_stable_year_has_no_multi_section_course_in_one_period(planned_result) -> None:
    _, planned, _ = planned_result
    summary = planned.course_demand_summary.set_index("course_id")
    offenders = []
    for course_id, row in summary.iterrows():
        if int(row["planned_sections"]) <= 1:
            continue
        periods = set(planned.sections.loc[planned.sections["course_id"] == course_id, "period_1"])
        if len(periods) == 1:
            offenders.append(course_id)

    assert offenders == []


def test_period_logical_occupancy_is_balanced(planned_result) -> None:
    _, planned, _ = planned_result
    counts = planned.period_layout_summary["logical_section_count"]

    assert counts.max() <= counts.mean() * 1.25
    assert counts.max() - counts.min() <= 15
    assert planned.metadata["period_balance_warnings"] == []


def test_period_summary_logical_counts_match_sections(planned_result) -> None:
    _, planned, _ = planned_result
    sections = planned.sections
    logical = sections.groupby("linked_section_group_id", sort=False).agg(
        period_1=("period_1", "first"),
        period_2=("period_2", "first"),
        capacity=("capacity", "first"),
    )

    for row in planned.period_layout_summary.itertuples(index=False):
        logical_count = 0
        occupied_slots = 0
        seats = 0
        for section in logical.itertuples(index=False):
            periods = {section.period_1}
            if section.period_2:
                periods.add(section.period_2)
            if row.period in periods:
                logical_count += 1
                occupied_slots += 1
            if row.period == section.period_1:
                seats += int(section.capacity)
        assert row.logical_section_count == logical_count
        assert row.occupied_period_slot_count == occupied_slots
        assert row.planned_seats == seats


def test_gov_econ_counts_once_in_period_summary(planned_result) -> None:
    _, planned, _ = planned_result
    gov_group = planned.sections[
        planned.sections["course_id"].isin(GOV_ECON_COURSES)
    ].groupby("linked_section_group_id").first().iloc[0]
    period_row = planned.period_layout_summary.set_index("period").loc[gov_group["period_1"]]
    rows_in_period = planned.sections[
        (planned.sections["linked_section_group_id"] == gov_group.name)
        & (planned.sections["period_1"] == gov_group["period_1"])
    ]

    assert len(rows_in_period) == 2
    assert period_row["section_row_count"] >= 2
    assert period_row["linked_logical_sections"] >= 1


def test_math2_3_counts_slots_in_both_periods_but_seats_once(planned_result) -> None:
    _, planned, _ = planned_result
    math_row = planned.sections[planned.sections["course_id"] == "MATH2_3_HA"].iloc[0]
    summary = planned.period_layout_summary.set_index("period")

    assert summary.loc[math_row["period_1"], "double_period_logical_sections"] >= 1
    assert summary.loc[math_row["period_2"], "double_period_logical_sections"] >= 1
    assert summary["planned_seats"].sum() == planned.metadata["total_planned_seats"]


def test_period_conflict_metric_is_computable(planned_result) -> None:
    _, planned, _ = planned_result

    assert isinstance(planned.metadata["raw_period_overlap_score"], int)
    assert isinstance(planned.metadata["unavoidable_course_pair_conflict_score"], int)
    assert planned.metadata["raw_period_overlap_score"] >= 0
    assert planned.metadata["unavoidable_course_pair_conflict_score"] >= 0


def test_total_planned_seats_matches_summary(planned_result) -> None:
    _, planned, _ = planned_result

    assert planned.metadata["total_planned_seats"] == int(planned.course_demand_summary["planned_seats"].sum())
    assert planned.metadata["total_planned_seats"] == int(planned.period_layout_summary["planned_seats"].sum())


def test_cli_outputs_all_files(tmp_path: Path) -> None:
    config_dir, templates_dir = copy_inputs(tmp_path)
    generated = generate_synthetic_dataset(config_dir, "stable_year", 2026)
    input_dir = tmp_path / "generated"
    input_dir.mkdir()
    generated.students.to_csv(input_dir / "students.csv", index=False)
    generated.requests.to_csv(input_dir / "requests.csv", index=False)
    output_dir = tmp_path / "sections"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.section_planning",
            "--input-dir",
            str(input_dir),
            "--scenario",
            "stable_year",
            "--seed",
            "2026",
            "--config-dir",
            str(config_dir),
            "--templates-dir",
            str(templates_dir),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert (output_dir / "sections.csv").exists()
    assert (output_dir / "course_demand_summary.csv").exists()
    assert (output_dir / "period_layout_summary.csv").exists()
    assert (output_dir / "section_planning_metadata.json").exists()


def test_cli_fails_for_invalid_input_without_partial_outputs(tmp_path: Path) -> None:
    config_dir, templates_dir = copy_inputs(tmp_path)
    generated = generate_synthetic_dataset(config_dir, "stable_year", 2026)
    input_dir = tmp_path / "generated"
    input_dir.mkdir()
    generated.students.to_csv(input_dir / "students.csv", index=False)
    bad_requests = generated.requests.copy()
    bad_requests.loc[0, "course_id"] = "NO_SUCH_COURSE"
    bad_requests.to_csv(input_dir / "requests.csv", index=False)
    output_dir = tmp_path / "sections"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.section_planning",
            "--input-dir",
            str(input_dir),
            "--scenario",
            "stable_year",
            "--seed",
            "2026",
            "--config-dir",
            str(config_dir),
            "--templates-dir",
            str(templates_dir),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert not output_dir.exists()


def test_cli_fails_for_invalid_config_without_partial_outputs(tmp_path: Path) -> None:
    config_dir, templates_dir = copy_inputs(tmp_path)
    overrides_path = config_dir / "section_capacity_overrides.csv"
    overrides = read_csv(overrides_path)
    overrides.loc[0, "course_id"] = "NO_SUCH_COURSE"
    write_csv(overrides_path, overrides)
    generated = generate_synthetic_dataset(REPO_ROOT / "data" / "config", "stable_year", 2026)
    input_dir = tmp_path / "generated"
    input_dir.mkdir()
    generated.students.to_csv(input_dir / "students.csv", index=False)
    generated.requests.to_csv(input_dir / "requests.csv", index=False)
    output_dir = tmp_path / "sections"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.section_planning",
            "--input-dir",
            str(input_dir),
            "--scenario",
            "stable_year",
            "--seed",
            "2026",
            "--config-dir",
            str(config_dir),
            "--templates-dir",
            str(templates_dir),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 1
    assert not output_dir.exists()


def test_section_planning_config_is_covered_by_validator(tmp_path: Path) -> None:
    config_dir, templates_dir = copy_inputs(tmp_path)
    path = config_dir / "section_capacity_overrides.csv"
    overrides = read_csv(path)
    overrides.loc[0, "capacity"] = 0
    write_csv(path, overrides)

    report = validate_configuration(config_dir, templates_dir)

    assert "INVALID_SECTION_CAPACITY_OVERRIDE" in {issue.code for issue in report.errors}


def test_planning_outputs_are_readable(tmp_path: Path) -> None:
    config_dir, templates_dir = copy_inputs(tmp_path)
    generated = generate_synthetic_dataset(config_dir, "stable_year", 2026)
    input_dir = tmp_path / "generated"
    input_dir.mkdir()
    generated.students.to_csv(input_dir / "students.csv", index=False)
    generated.requests.to_csv(input_dir / "requests.csv", index=False)
    output_dir = tmp_path / "sections"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.section_planning",
            "--input-dir",
            str(input_dir),
            "--scenario",
            "stable_year",
            "--seed",
            "2026",
            "--config-dir",
            str(config_dir),
            "--templates-dir",
            str(templates_dir),
            "--output-dir",
            str(output_dir),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert not pd.read_csv(output_dir / "sections.csv").empty
    assert not pd.read_csv(output_dir / "course_demand_summary.csv").empty
    assert not pd.read_csv(output_dir / "period_layout_summary.csv").empty
    with (output_dir / "section_planning_metadata.json").open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    assert metadata["planner_version"] == "v1"
    assert metadata["output_file_hashes"] == {
        "sections.csv": hashlib.sha256((output_dir / "sections.csv").read_bytes()).hexdigest(),
    }
