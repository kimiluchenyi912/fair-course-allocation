from __future__ import annotations

import pandas as pd
import pytest

from src.allocation import AllocationInputError, canonicalize_allocation_input


def catalog() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("NORMAL", 1, "standard"),
            ("MATH2_3_HA", 2, "double_period"),
            ("GOV_ECON_REG", 1, "semester_block"),
            ("AP_PHYSC", 1, "semester_block"),
            ("ALT1", 1, "standard"),
            ("ALT2", 1, "standard"),
            ("ALT3", 1, "standard"),
        ],
        columns=["course_id", "periods_required", "schedule_structure"],
    )


def students() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("STU_1", 12, 7, "none", "false", "", ""),
        ],
        columns=[
            "student_id",
            "grade",
            "target_course_count",
            "unscheduled_preference",
            "priority_protected",
            "priority_reason",
            "priority_valid_school_year",
        ],
    )


def requests() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("STU_1", "ALT3", "alternate", 3, "alternate", ""),
            ("STU_1", "GOV_ECON_REG", "primary", "", "gov_econ_block", "GOV_ECON_REG"),
            ("STU_1", "NORMAL", "primary", "", "", ""),
            ("STU_1", "ALT1", "alternate", 1, "alternate", ""),
            ("STU_1", "MATH2_3_HA", "primary", "", "", ""),
            ("STU_1", "GOV_ECON_REG", "primary", "", "gov_econ_block", "GOV_ECON_REG"),
            ("STU_1", "ALT2", "alternate", 2, "alternate", ""),
        ],
        columns=[
            "student_id",
            "course_id",
            "request_type",
            "request_rank",
            "request_group",
            "must_share_block_id",
        ],
    )


def section_row(
    section_id: str,
    course_id: str,
    period_1: str,
    period_2: str = "",
    semester: str = "full_year",
    capacity: int = 40,
    block_id: str = "",
    group_id: str = "",
    logical_block_id: str = "",
    semester_content: str = "",
) -> tuple:
    return (
        section_id,
        course_id,
        period_1,
        period_2,
        semester,
        capacity,
        block_id,
        group_id,
        logical_block_id,
        semester_content,
    )


def sections() -> pd.DataFrame:
    return pd.DataFrame(
        [
            section_row("SEC_NORMAL_02", "NORMAL", "P5", group_id="NORMAL_02", logical_block_id="NORMAL"),
            section_row("SEC_GOV_01_S2", "GOV_ECON_REG", "P3", semester="semester_2", capacity=35, block_id="GOV_01", group_id="GOV_01", logical_block_id="GOV_ECON_REG", semester_content="Economics"),
            section_row("SEC_ALT2_01", "ALT2", "P6", group_id="ALT2_01", logical_block_id="ALT2"),
            section_row("SEC_MATH_01", "MATH2_3_HA", "P1", "P2", group_id="MATH_01", logical_block_id="MATH2_3_HA"),
            section_row("SEC_GOV_01_S1", "GOV_ECON_REG", "P3", semester="semester_1", capacity=35, block_id="GOV_01", group_id="GOV_01", logical_block_id="GOV_ECON_REG", semester_content="Government"),
            section_row("SEC_AP_PHYSC_01", "AP_PHYSC", "P4", semester="paired", block_id="AP_PHYSC_01", group_id="AP_PHYSC_01", logical_block_id="AP_PHYSC", semester_content="Mechanics / Electricity and Magnetism"),
            section_row("SEC_ALT1_01", "ALT1", "P5", group_id="ALT1_01", logical_block_id="ALT1"),
            section_row("SEC_NORMAL_01", "NORMAL", "P1", group_id="NORMAL_01", logical_block_id="NORMAL"),
            section_row("SEC_ALT3_01", "ALT3", "P7", group_id="ALT3_01", logical_block_id="ALT3"),
        ],
        columns=[
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
        ],
    )


def canonical_input(
    students_df: pd.DataFrame | None = None,
    requests_df: pd.DataFrame | None = None,
    sections_df: pd.DataFrame | None = None,
):
    return canonicalize_allocation_input(
        students_df if students_df is not None else students(),
        requests_df if requests_df is not None else requests(),
        sections_df if sections_df is not None else sections(),
        catalog(),
    )


def issue_codes(error: AllocationInputError) -> set[str]:
    return {issue.code for issue in error.issues}


def test_normal_single_period_section_is_canonicalized() -> None:
    result = canonical_input()

    section = result.logical_sections_by_id["NORMAL_01"]

    assert section.structure_type == "normal"
    assert section.occupied_periods == ("P1",)
    assert section.period_units == 1
    assert section.capacity == 40


def test_math_double_period_section_is_canonicalized() -> None:
    result = canonical_input()

    section = result.logical_sections_by_id["MATH_01"]

    assert len(section.member_sections) == 1
    assert section.structure_type == "double_period"
    assert section.occupied_periods == ("P1", "P2")
    assert section.period_units == 2


def test_gov_econ_rows_become_one_logical_section() -> None:
    result = canonical_input()

    section = result.logical_sections_by_id["GOV_01"]

    assert tuple(member.semester for member in section.member_sections) == ("semester_1", "semester_2")
    assert section.structure_type == "linked_semester"
    assert section.occupied_periods == ("P3",)
    assert section.period_units == 1
    assert section.capacity == 35


def test_ap_physics_c_stays_one_paired_content_section() -> None:
    result = canonical_input()

    section = result.logical_sections_by_id["AP_PHYSC_01"]

    assert len(section.member_sections) == 1
    assert section.structure_type == "paired_content"
    assert section.occupied_periods == ("P4",)
    assert section.period_units == 1
    assert section.member_sections[0].semester_content == "Mechanics / Electricity and Magnetism"


def test_target_course_count_maps_to_target_period_units() -> None:
    result = canonical_input()

    assert result.students_by_id["STU_1"].target_period_units == 7


def test_primary_rows_group_into_logical_requests() -> None:
    result = canonical_input()

    primary = result.students_by_id["STU_1"].primary_requests
    gov = next(request for request in primary if request.candidate_key == "GOV_ECON_REG")

    assert [request.candidate_key for request in primary] == ["GOV_ECON_REG", "MATH2_3_HA", "NORMAL"]
    assert len(gov.source_rows) == 2
    assert gov.period_units == 1


def test_global_alternates_preserve_rank_one_through_three() -> None:
    result = canonical_input()

    alternates = result.students_by_id["STU_1"].alternate_requests

    assert [request.request_rank for request in alternates] == [1, 2, 3]
    assert [request.course_ids for request in alternates] == [("ALT1",), ("ALT2",), ("ALT3",)]


def test_alternate_candidate_index_maps_to_matching_logical_section() -> None:
    result = canonical_input()

    assert result.candidate_index["alternate:STU_1:1:ALT1"] == ("ALT1_01",)


def test_gov_econ_request_matches_complete_linked_section() -> None:
    result = canonical_input()

    candidates = result.candidate_index["primary:STU_1:GOV_ECON_REG"]
    section = result.logical_sections_by_id[candidates[0]]

    assert candidates == ("GOV_01",)
    assert len(section.member_sections) == 2


def test_candidate_lists_are_stably_sorted() -> None:
    result = canonical_input()

    assert result.candidate_index["primary:STU_1:NORMAL"] == ("NORMAL_01", "NORMAL_02")


def test_row_order_does_not_change_canonical_result() -> None:
    first = canonical_input()
    second = canonical_input(
        students().sample(frac=1, random_state=4).reset_index(drop=True),
        requests().sample(frac=1, random_state=5).reset_index(drop=True),
        sections().sample(frac=1, random_state=6).reset_index(drop=True),
    )

    assert first == second


def test_blank_request_block_fields_read_as_na_are_treated_as_empty() -> None:
    na_requests = requests().copy()
    na_requests.loc[na_requests["must_share_block_id"] == "", "must_share_block_id"] = pd.NA

    result = canonical_input(requests_df=na_requests)

    assert result.requests_by_key["primary:STU_1:NORMAL"].candidate_key == "NORMAL"
    assert result.requests_by_key["alternate:STU_1:1:ALT1"].candidate_key == "ALT1"


def test_duplicate_logical_primary_is_rejected() -> None:
    bad_requests = pd.concat([requests(), requests().iloc[[2]]], ignore_index=True)

    with pytest.raises(AllocationInputError) as raised:
        canonical_input(requests_df=bad_requests)

    assert "DUPLICATE_LOGICAL_PRIMARY" in issue_codes(raised.value)


def test_duplicate_alternate_logical_request_is_rejected() -> None:
    duplicate = requests().iloc[[3]].copy()
    duplicate.loc[:, "request_rank"] = 2
    bad_requests = pd.concat([requests(), duplicate], ignore_index=True)

    with pytest.raises(AllocationInputError) as raised:
        canonical_input(requests_df=bad_requests)

    assert "DUPLICATE_ALTERNATE_LOGICAL_REQUEST" in issue_codes(raised.value)


def test_duplicate_alternate_rank_is_rejected() -> None:
    bad_requests = requests().copy()
    bad_requests.loc[bad_requests["course_id"] == "ALT2", "request_rank"] = 1

    with pytest.raises(AllocationInputError) as raised:
        canonical_input(requests_df=bad_requests)

    assert "DUPLICATE_ALTERNATE_RANK" in issue_codes(raised.value)


def test_gov_econ_missing_semester_row_is_rejected() -> None:
    bad_sections = sections()[sections()["section_id"] != "SEC_GOV_01_S2"].copy()

    with pytest.raises(AllocationInputError) as raised:
        canonical_input(sections_df=bad_sections)

    assert "INVALID_LINKED_SEMESTER_STRUCTURE" in issue_codes(raised.value)


def test_gov_econ_mismatched_period_is_rejected() -> None:
    bad_sections = sections().copy()
    bad_sections.loc[bad_sections["section_id"] == "SEC_GOV_01_S2", "period_1"] = "P4"

    with pytest.raises(AllocationInputError) as raised:
        canonical_input(sections_df=bad_sections)

    assert "INVALID_LINKED_SEMESTER_STRUCTURE" in issue_codes(raised.value)


def test_gov_econ_mismatched_capacity_is_rejected() -> None:
    bad_sections = sections().copy()
    bad_sections.loc[bad_sections["section_id"] == "SEC_GOV_01_S2", "capacity"] = 40

    with pytest.raises(AllocationInputError) as raised:
        canonical_input(sections_df=bad_sections)

    assert "INCONSISTENT_LOGICAL_SECTION_CAPACITY" in issue_codes(raised.value)


def test_nonconsecutive_math_periods_are_rejected() -> None:
    bad_sections = sections().copy()
    bad_sections.loc[bad_sections["section_id"] == "SEC_MATH_01", "period_2"] = "P3"

    with pytest.raises(AllocationInputError) as raised:
        canonical_input(sections_df=bad_sections)

    assert "NONCONSECUTIVE_DOUBLE_PERIOD" in issue_codes(raised.value)


def test_math_missing_second_period_is_rejected() -> None:
    bad_sections = sections().copy()
    bad_sections.loc[bad_sections["section_id"] == "SEC_MATH_01", "period_2"] = ""

    with pytest.raises(AllocationInputError) as raised:
        canonical_input(sections_df=bad_sections)

    assert "INVALID_DOUBLE_PERIOD_STRUCTURE" in issue_codes(raised.value)


def test_unknown_request_student_and_course_are_rejected() -> None:
    bad_requests = requests().copy()
    bad_requests.loc[0, "student_id"] = "UNKNOWN_STUDENT"
    bad_requests.loc[1, "course_id"] = "UNKNOWN_COURSE"

    with pytest.raises(AllocationInputError) as raised:
        canonical_input(requests_df=bad_requests)

    assert {"UNKNOWN_REQUEST_STUDENT", "UNKNOWN_REQUEST_COURSE"}.issubset(issue_codes(raised.value))


def test_invalid_section_capacity_is_rejected() -> None:
    bad_sections = sections().copy()
    bad_sections.loc[bad_sections["section_id"] == "SEC_NORMAL_01", "capacity"] = 0

    with pytest.raises(AllocationInputError) as raised:
        canonical_input(sections_df=bad_sections)

    assert "INVALID_SECTION_CAPACITY" in issue_codes(raised.value)
