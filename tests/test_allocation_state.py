from __future__ import annotations

import pandas as pd
import pytest

from src.allocation import (
    AllocationState,
    AssignmentRecord,
    AssignmentRejectionReason,
    canonicalize_allocation_input,
)


def catalog() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("NORMAL", 1, "standard"),
            ("ALT1", 1, "standard"),
            ("ALT2", 1, "standard"),
            ("ALT3", 1, "standard"),
            ("MATH2_3_HA", 2, "double_period"),
            ("GOV_ECON_REG", 1, "semester_block"),
            ("AP_PHYSC", 1, "semester_block"),
        ],
        columns=["course_id", "periods_required", "schedule_structure"],
    )


def students(target_1: int = 7) -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("STU_1", 12, target_1, "none", "false", "", ""),
            ("STU_2", 12, 7, "none", "false", "", ""),
            ("STU_3", 12, 7, "none", "false", "", ""),
            ("STU_PROTECTED", 12, 7, "none", "true", "prior_year_unmet_primary", "2026-2027"),
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
    rows = []
    for student_id in ("STU_1", "STU_2", "STU_3", "STU_PROTECTED"):
        rows.extend(
            [
                (student_id, "NORMAL", "primary", "", "", ""),
                (student_id, "MATH2_3_HA", "primary", "", "", ""),
                (student_id, "GOV_ECON_REG", "primary", "", "gov_econ_block", "GOV_ECON_REG"),
                (student_id, "GOV_ECON_REG", "primary", "", "gov_econ_block", "GOV_ECON_REG"),
                (student_id, "AP_PHYSC", "primary", "", "", ""),
                (student_id, "ALT1", "alternate", 1, "alternate", ""),
                (student_id, "NORMAL", "alternate", 2, "alternate", ""),
                (student_id, "ALT2", "alternate", 3, "alternate", ""),
            ]
        )
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


def sections(include_mismatch: bool = False) -> pd.DataFrame:
    rows = [
        section_row("SEC_NORMAL_A", "NORMAL", "P1", capacity=2, group_id="NORMAL_A", logical_block_id="NORMAL"),
        section_row("SEC_NORMAL_B", "NORMAL", "P2", group_id="NORMAL_B", logical_block_id="NORMAL"),
        section_row("SEC_ALT1_A", "ALT1", "P2", group_id="ALT1_A", logical_block_id="ALT1"),
        section_row("SEC_ALT2_A", "ALT2", "P5", group_id="ALT2_A", logical_block_id="ALT2"),
        section_row("SEC_ALT3_A", "ALT3", "P5", group_id="ALT3_A", logical_block_id="ALT3"),
        section_row("SEC_MATH_A", "MATH2_3_HA", "P5", "P6", group_id="MATH_A", logical_block_id="MATH2_3_HA"),
        section_row("SEC_GOV_A_S1", "GOV_ECON_REG", "P7", semester="semester_1", block_id="GOV_A", group_id="GOV_A", logical_block_id="GOV_ECON_REG", semester_content="Government"),
        section_row("SEC_GOV_A_S2", "GOV_ECON_REG", "P7", semester="semester_2", block_id="GOV_A", group_id="GOV_A", logical_block_id="GOV_ECON_REG", semester_content="Economics"),
        section_row("SEC_AP_PHYSC_A", "AP_PHYSC", "P4", semester="paired", block_id="AP_PHYSC_A", group_id="AP_PHYSC_A", logical_block_id="AP_PHYSC", semester_content="Mechanics / Electricity and Magnetism"),
    ]
    if include_mismatch:
        rows.append(
            section_row(
                "SEC_MISMATCH_A",
                "MATH2_3_HA",
                "P3",
                "P4",
                group_id="MISMATCH_A",
                logical_block_id="NORMAL",
            )
        )
    return pd.DataFrame(
        rows,
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


def canonical(target_1: int = 7, include_mismatch: bool = False):
    return canonicalize_allocation_input(
        students(target_1),
        requests(),
        sections(include_mismatch),
        catalog(),
    )


def state(target_1: int = 7, include_mismatch: bool = False) -> AllocationState:
    return AllocationState(canonical(target_1, include_mismatch))


def key(student_id: str, course_or_rank: str) -> str:
    if course_or_rank in {"ALT1", "ALT2", "ALT3", "NORMAL_ALT"}:
        rank = {"ALT1": 1, "NORMAL_ALT": 2, "ALT2": 3, "ALT3": 3}[course_or_rank]
        candidate = "NORMAL" if course_or_rank == "NORMAL_ALT" else course_or_rank
        return f"alternate:{student_id}:{rank}:{candidate}"
    return f"primary:{student_id}:{course_or_rank}"


def signature(allocation_state: AllocationState) -> tuple:
    return (
        allocation_state.all_assignments(),
        allocation_state.student_assignments("STU_1"),
        allocation_state.student_occupied_periods("STU_1"),
        allocation_state.student_used_period_units("STU_1"),
        allocation_state.section_roster("NORMAL_A"),
        allocation_state.section_assigned_count("NORMAL_A"),
        allocation_state.section_remaining_capacity("NORMAL_A"),
        allocation_state.validate_internal_consistency(),
    )


def assert_reasons(result, *reasons: AssignmentRejectionReason) -> None:
    assert result.allowed is False
    assert result.reasons == reasons


def assign_or_fail(
    allocation_state: AllocationState,
    student_id: str,
    request_key: str,
    section_id: str,
) -> AssignmentRecord:
    result = allocation_state.try_assign(student_id, request_key, section_id)
    assert result.allowed
    assert result.assignment is not None
    return result.assignment


def test_normal_assignment_updates_student_and_section_state() -> None:
    allocation_state = state()
    assignment = assign_or_fail(allocation_state, "STU_1", key("STU_1", "NORMAL"), "NORMAL_A")

    assert assignment.assignment_key == "STU_1|primary:STU_1:NORMAL|NORMAL_A"
    assert assignment.member_section_ids == ("SEC_NORMAL_A",)
    assert allocation_state.student_assignments("STU_1") == (assignment,)
    assert allocation_state.student_occupied_periods("STU_1") == frozenset({"P1"})
    assert allocation_state.student_used_period_units("STU_1") == 1
    assert allocation_state.student_remaining_period_units("STU_1") == 6
    assert allocation_state.section_roster("NORMAL_A") == ("STU_1",)
    assert allocation_state.section_assigned_count("NORMAL_A") == 1
    assert allocation_state.section_remaining_capacity("NORMAL_A") == 1
    assert allocation_state.is_request_assigned(key("STU_1", "NORMAL"))
    assert allocation_state.get_assignment(assignment.assignment_key) == assignment
    assert allocation_state.validate_internal_consistency() == ()


def test_math_double_period_assignment_uses_two_periods_two_units_and_one_seat() -> None:
    allocation_state = state()
    assignment = assign_or_fail(allocation_state, "STU_1", key("STU_1", "MATH2_3_HA"), "MATH_A")

    assert assignment.occupied_periods == ("P5", "P6")
    assert assignment.period_units == 2
    assert assignment.structure_type == "double_period"
    assert allocation_state.student_occupied_periods("STU_1") == frozenset({"P5", "P6"})
    assert allocation_state.student_used_period_units("STU_1") == 2
    assert allocation_state.section_assigned_count("MATH_A") == 1


def test_gov_econ_assignment_uses_one_period_one_unit_one_seat_and_keeps_member_rows() -> None:
    allocation_state = state()
    assignment = assign_or_fail(allocation_state, "STU_1", key("STU_1", "GOV_ECON_REG"), "GOV_A")

    assert assignment.member_section_ids == ("SEC_GOV_A_S1", "SEC_GOV_A_S2")
    assert assignment.occupied_periods == ("P7",)
    assert assignment.period_units == 1
    assert assignment.structure_type == "linked_semester"
    assert allocation_state.section_assigned_count("GOV_A") == 1


def test_ap_physics_c_assignment_uses_one_period_one_unit_one_seat() -> None:
    allocation_state = state()
    assignment = assign_or_fail(allocation_state, "STU_1", key("STU_1", "AP_PHYSC"), "AP_PHYSC_A")

    assert assignment.member_section_ids == ("SEC_AP_PHYSC_A",)
    assert assignment.occupied_periods == ("P4",)
    assert assignment.period_units == 1
    assert assignment.structure_type == "paired_content"
    assert allocation_state.section_assigned_count("AP_PHYSC_A") == 1


def test_two_students_can_enter_same_section_until_capacity() -> None:
    allocation_state = state()
    assign_or_fail(allocation_state, "STU_1", key("STU_1", "NORMAL"), "NORMAL_A")
    assign_or_fail(allocation_state, "STU_2", key("STU_2", "NORMAL"), "NORMAL_A")

    assert allocation_state.section_roster("NORMAL_A") == ("STU_1", "STU_2")
    assert allocation_state.section_assigned_count("NORMAL_A") == 2
    assert allocation_state.section_remaining_capacity("NORMAL_A") == 0


def test_primary_and_alternate_use_same_assignment_engine() -> None:
    allocation_state = state()
    primary = assign_or_fail(allocation_state, "STU_1", key("STU_1", "NORMAL"), "NORMAL_A")
    alternate = assign_or_fail(allocation_state, "STU_1", key("STU_1", "ALT1"), "ALT1_A")

    assert primary.request_type == "primary"
    assert alternate.request_type == "alternate"
    assert alternate.alternate_rank == 1
    assert allocation_state.student_used_period_units("STU_1") == 2


def test_alternate_rank_is_preserved_in_assignment_record() -> None:
    allocation_state = state()
    assignment = assign_or_fail(allocation_state, "STU_1", key("STU_1", "ALT2"), "ALT2_A")

    assert assignment.request_type == "alternate"
    assert assignment.alternate_rank == 3


def test_all_assignments_are_sorted_deterministically() -> None:
    allocation_state = state()
    second = assign_or_fail(allocation_state, "STU_2", key("STU_2", "NORMAL"), "NORMAL_A")
    first = assign_or_fail(allocation_state, "STU_1", key("STU_1", "ALT1"), "ALT1_A")

    assert allocation_state.all_assignments() == tuple(sorted((first, second), key=lambda item: item.assignment_key))


def test_unknown_student_is_rejected() -> None:
    result = state().check_assignment("UNKNOWN", key("STU_1", "NORMAL"), "NORMAL_A")

    assert_reasons(result, AssignmentRejectionReason.UNKNOWN_STUDENT)


def test_unknown_request_is_rejected() -> None:
    result = state().check_assignment("STU_1", "primary:STU_1:UNKNOWN", "NORMAL_A")

    assert_reasons(result, AssignmentRejectionReason.UNKNOWN_REQUEST)


def test_unknown_section_is_rejected() -> None:
    result = state().check_assignment("STU_1", key("STU_1", "NORMAL"), "UNKNOWN_SECTION")

    assert_reasons(result, AssignmentRejectionReason.UNKNOWN_SECTION)


def test_request_not_owned_by_student_is_rejected() -> None:
    result = state().check_assignment("STU_2", key("STU_1", "NORMAL"), "NORMAL_A")

    assert_reasons(result, AssignmentRejectionReason.REQUEST_NOT_OWNED_BY_STUDENT)


def test_section_not_candidate_for_request_is_rejected() -> None:
    result = state().check_assignment("STU_1", key("STU_1", "NORMAL"), "ALT1_A")

    assert_reasons(result, AssignmentRejectionReason.SECTION_NOT_CANDIDATE_FOR_REQUEST)


def test_request_already_assigned_is_rejected() -> None:
    allocation_state = state()
    assign_or_fail(allocation_state, "STU_1", key("STU_1", "NORMAL"), "NORMAL_A")

    result = allocation_state.check_assignment("STU_1", key("STU_1", "NORMAL"), "NORMAL_B")

    assert_reasons(
        result,
        AssignmentRejectionReason.REQUEST_ALREADY_ASSIGNED,
        AssignmentRejectionReason.DUPLICATE_LOGICAL_COURSE_OR_BLOCK,
    )


def test_primary_and_alternate_same_logical_identity_is_rejected() -> None:
    allocation_state = state()
    assign_or_fail(allocation_state, "STU_1", key("STU_1", "NORMAL"), "NORMAL_A")

    result = allocation_state.check_assignment("STU_1", key("STU_1", "NORMAL_ALT"), "NORMAL_B")

    assert_reasons(result, AssignmentRejectionReason.DUPLICATE_LOGICAL_COURSE_OR_BLOCK)


def test_period_conflict_is_rejected() -> None:
    allocation_state = state()
    assign_or_fail(allocation_state, "STU_1", key("STU_1", "ALT1"), "ALT1_A")

    result = allocation_state.check_assignment("STU_1", key("STU_1", "NORMAL"), "NORMAL_B")

    assert_reasons(result, AssignmentRejectionReason.PERIOD_CONFLICT)


def test_math_double_period_conflict_on_either_period_rejects_whole_assignment() -> None:
    allocation_state = state()
    assign_or_fail(allocation_state, "STU_1", key("STU_1", "ALT2"), "ALT2_A")
    before = signature(allocation_state)

    result = allocation_state.try_assign("STU_1", key("STU_1", "MATH2_3_HA"), "MATH_A")

    assert_reasons(result, AssignmentRejectionReason.PERIOD_CONFLICT)
    assert signature(allocation_state) == before


def test_target_load_exceeded_is_rejected() -> None:
    allocation_state = state(target_1=1)
    assign_or_fail(allocation_state, "STU_1", key("STU_1", "NORMAL"), "NORMAL_A")

    result = allocation_state.check_assignment("STU_1", key("STU_1", "ALT1"), "ALT1_A")

    assert_reasons(result, AssignmentRejectionReason.TARGET_LOAD_EXCEEDED)


def test_one_remaining_unit_rejects_two_unit_math_request() -> None:
    allocation_state = state(target_1=2)
    assign_or_fail(allocation_state, "STU_1", key("STU_1", "NORMAL"), "NORMAL_A")

    result = allocation_state.check_assignment("STU_1", key("STU_1", "MATH2_3_HA"), "MATH_A")

    assert_reasons(result, AssignmentRejectionReason.TARGET_LOAD_EXCEEDED)


def test_section_full_is_rejected() -> None:
    allocation_state = state()
    assign_or_fail(allocation_state, "STU_1", key("STU_1", "NORMAL"), "NORMAL_A")
    assign_or_fail(allocation_state, "STU_2", key("STU_2", "NORMAL"), "NORMAL_A")

    result = allocation_state.check_assignment("STU_3", key("STU_3", "NORMAL"), "NORMAL_A")

    assert_reasons(result, AssignmentRejectionReason.SECTION_FULL)


def test_request_and_section_period_unit_mismatch_is_rejected() -> None:
    result = state(include_mismatch=True).check_assignment("STU_1", key("STU_1", "NORMAL"), "MISMATCH_A")

    assert_reasons(result, AssignmentRejectionReason.PERIOD_UNIT_MISMATCH)


@pytest.mark.parametrize(
    ("student_id", "request_key", "section_id"),
    [
        ("UNKNOWN", key("STU_1", "NORMAL"), "NORMAL_A"),
        ("STU_1", "primary:STU_1:UNKNOWN", "NORMAL_A"),
        ("STU_1", key("STU_1", "NORMAL"), "UNKNOWN_SECTION"),
        ("STU_2", key("STU_1", "NORMAL"), "NORMAL_A"),
        ("STU_1", key("STU_1", "NORMAL"), "ALT1_A"),
    ],
)
def test_rejections_do_not_mutate_state(student_id: str, request_key: str, section_id: str) -> None:
    allocation_state = state()
    before = signature(allocation_state)

    result = allocation_state.try_assign(student_id, request_key, section_id)

    assert result.allowed is False
    assert signature(allocation_state) == before


def test_check_assignment_does_not_mutate_state() -> None:
    allocation_state = state()
    before = signature(allocation_state)

    result = allocation_state.check_assignment("STU_1", key("STU_1", "NORMAL"), "NORMAL_A")

    assert result.allowed
    assert signature(allocation_state) == before


def test_unknown_query_ids_raise_key_error_consistently() -> None:
    allocation_state = state()

    with pytest.raises(KeyError):
        allocation_state.student_assignments("UNKNOWN")
    with pytest.raises(KeyError):
        allocation_state.section_roster("UNKNOWN")
    with pytest.raises(KeyError):
        allocation_state.is_request_assigned("UNKNOWN")
    with pytest.raises(KeyError):
        allocation_state.get_assignment("UNKNOWN")


def test_normal_unassign_restores_state() -> None:
    allocation_state = state()
    before = signature(allocation_state)
    assignment = assign_or_fail(allocation_state, "STU_1", key("STU_1", "NORMAL"), "NORMAL_A")

    result = allocation_state.unassign(assignment.assignment_key)

    assert result.allowed
    assert result.assignment == assignment
    assert signature(allocation_state) == before


def test_math_unassign_releases_two_periods_and_two_units() -> None:
    allocation_state = state()
    assignment = assign_or_fail(allocation_state, "STU_1", key("STU_1", "MATH2_3_HA"), "MATH_A")

    allocation_state.unassign(assignment.assignment_key)

    assert allocation_state.student_occupied_periods("STU_1") == frozenset()
    assert allocation_state.student_used_period_units("STU_1") == 0
    assert allocation_state.section_assigned_count("MATH_A") == 0


def test_gov_econ_unassign_restores_only_one_roster_seat() -> None:
    allocation_state = state()
    assignment = assign_or_fail(allocation_state, "STU_1", key("STU_1", "GOV_ECON_REG"), "GOV_A")

    allocation_state.unassign(assignment.assignment_key)

    assert allocation_state.student_used_period_units("STU_1") == 0
    assert allocation_state.section_assigned_count("GOV_A") == 0
    assert allocation_state.section_roster("GOV_A") == ()


def test_unassign_from_full_section_restores_capacity() -> None:
    allocation_state = state()
    assign_or_fail(allocation_state, "STU_1", key("STU_1", "NORMAL"), "NORMAL_A")
    assignment = assign_or_fail(allocation_state, "STU_2", key("STU_2", "NORMAL"), "NORMAL_A")
    assert allocation_state.section_remaining_capacity("NORMAL_A") == 0

    allocation_state.unassign(assignment.assignment_key)

    assert allocation_state.section_remaining_capacity("NORMAL_A") == 1
    assert allocation_state.section_roster("NORMAL_A") == ("STU_1",)


def test_unassign_one_assignment_does_not_break_other_assignments_for_same_student() -> None:
    allocation_state = state()
    normal = assign_or_fail(allocation_state, "STU_1", key("STU_1", "NORMAL"), "NORMAL_A")
    alternate = assign_or_fail(allocation_state, "STU_1", key("STU_1", "ALT1"), "ALT1_A")

    allocation_state.unassign(normal.assignment_key)

    assert allocation_state.student_assignments("STU_1") == (alternate,)
    assert allocation_state.student_occupied_periods("STU_1") == frozenset({"P2"})
    assert allocation_state.student_used_period_units("STU_1") == 1


def test_unassign_missing_assignment_returns_structured_failure() -> None:
    result = state().unassign("missing")

    assert_reasons(result, AssignmentRejectionReason.UNKNOWN_ASSIGNMENT)


def test_assign_unassign_then_assign_again_has_consistent_semantics() -> None:
    allocation_state = state()
    first = assign_or_fail(allocation_state, "STU_1", key("STU_1", "NORMAL"), "NORMAL_A")
    allocation_state.unassign(first.assignment_key)

    second = assign_or_fail(allocation_state, "STU_1", key("STU_1", "NORMAL"), "NORMAL_A")

    assert second == first
    assert allocation_state.validate_internal_consistency() == ()


def test_protected_student_does_not_receive_special_state_feasibility() -> None:
    allocation_state = state()
    regular = allocation_state.check_assignment("STU_1", key("STU_1", "NORMAL"), "NORMAL_A")
    protected = allocation_state.check_assignment("STU_PROTECTED", key("STU_PROTECTED", "NORMAL"), "NORMAL_A")

    assert regular.allowed
    assert regular.reasons == ()
    assert protected.allowed
    assert protected.reasons == ()


def test_state_layer_does_not_check_ordinary_student_primary_unmet_limit() -> None:
    allocation_state = state()
    result = allocation_state.check_assignment("STU_1", key("STU_1", "ALT1"), "ALT1_A")

    assert result.allowed
    assert result.reasons == ()


def test_state_layer_does_not_report_global_infeasible() -> None:
    result = state(target_1=1).validate_internal_consistency()

    assert result == ()
