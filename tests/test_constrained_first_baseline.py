from __future__ import annotations

import random

import pandas as pd

from src.allocation import (
    AlternateRequestStatus,
    AllocationState,
    AssignmentRejectionReason,
    MandatoryFallbackStatus,
    MathFallbackRule,
    PrimaryRequestStatus,
    build_ordering_context,
    build_student_difficulty_profiles,
    candidate_section_priority,
    canonicalize_allocation_input,
    evaluate_math_policy,
    primary_request_priority,
    run_constrained_first_baseline,
    run_seeded_random_baseline,
)


def catalog() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("CORE", 1, "standard"),
            ("SCARCE", 1, "standard"),
            ("FLEX", 1, "standard"),
            ("NEXT", 1, "standard"),
            ("PRESS", 1, "standard"),
            ("HIGH", 1, "standard"),
            ("MATH1", 1, "standard"),
            ("MATH2", 1, "standard"),
            ("MATH2_3_HA", 2, "double_period"),
            ("DOUBLE", 2, "double_period"),
            ("ALT1", 1, "standard"),
            ("ALT2", 1, "standard"),
            ("GOV_ECON_REG", 1, "semester_block"),
        ],
        columns=["course_id", "periods_required", "schedule_structure"],
    )


def students(rows: list[tuple[str, int, int, bool]] | None = None) -> pd.DataFrame:
    rows = rows or [("STU_1", 12, 2, False)]
    return pd.DataFrame(
        [
            (
                student_id,
                grade,
                target,
                "none",
                str(protected).lower(),
                "prior_year_unmet_primary" if protected else "",
                "2026-2027" if protected else "",
            )
            for student_id, grade, target, protected in rows
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


def request_row(
    student_id: str,
    course_id: str,
    request_type: str = "primary",
    rank: int | str = "",
    group: str = "",
    block_id: str = "",
) -> tuple:
    return (student_id, course_id, request_type, rank, group, block_id)


def requests(rows: list[tuple]) -> pd.DataFrame:
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
        group_id or section_id,
        logical_block_id or course_id,
        semester_content,
    )


def sections(rows: list[tuple] | None = None) -> pd.DataFrame:
    rows = rows or base_sections()
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


def base_sections() -> list[tuple]:
    return [
        section_row("SEC_CORE_1", "CORE", "P1", capacity=10, group_id="CORE_1"),
        section_row("SEC_SCARCE_1", "SCARCE", "P1", capacity=10, group_id="SCARCE_1"),
        section_row("SEC_FLEX_1", "FLEX", "P1", capacity=10, group_id="FLEX_1"),
        section_row("SEC_FLEX_2", "FLEX", "P2", capacity=10, group_id="FLEX_2"),
        section_row("SEC_NEXT_1", "NEXT", "P1", capacity=10, group_id="NEXT_1"),
        section_row("SEC_PRESS_1", "PRESS", "P1", capacity=200, group_id="PRESS_1"),
        section_row("SEC_HIGH_1", "HIGH", "P3", capacity=200, group_id="HIGH_1"),
        section_row("SEC_MATH1_1", "MATH1", "P2", capacity=10, group_id="MATH1_1"),
        section_row("SEC_MATH2_1", "MATH2", "P4", capacity=10, group_id="MATH2_1"),
        section_row("SEC_MATH23_1", "MATH2_3_HA", "P5", "P6", capacity=10, group_id="MATH23_1"),
        section_row("SEC_DOUBLE_1", "DOUBLE", "P6", "P7", capacity=10, group_id="DOUBLE_1"),
        section_row("SEC_ALT1_1", "ALT1", "P6", capacity=10, group_id="ALT1_1"),
        section_row("SEC_ALT2_1", "ALT2", "P7", capacity=10, group_id="ALT2_1"),
        section_row(
            "SEC_GOV_1_S1",
            "GOV_ECON_REG",
            "P7",
            semester="semester_1",
            capacity=10,
            block_id="GOV_1",
            group_id="GOV_1",
            logical_block_id="GOV_ECON_REG",
            semester_content="Government",
        ),
        section_row(
            "SEC_GOV_1_S2",
            "GOV_ECON_REG",
            "P7",
            semester="semester_2",
            capacity=10,
            block_id="GOV_1",
            group_id="GOV_1",
            logical_block_id="GOV_ECON_REG",
            semester_content="Economics",
        ),
    ]


def canonical(
    student_rows: list[tuple[str, int, int, bool]] | None = None,
    request_rows: list[tuple] | None = None,
    section_rows: list[tuple] | None = None,
):
    return canonicalize_allocation_input(
        students(student_rows),
        requests(request_rows or [request_row("STU_1", "CORE")]),
        sections(section_rows),
        catalog(),
    )


def fallback_rules() -> tuple[MathFallbackRule, ...]:
    return (MathFallbackRule("MATH2_3_HA", "MATH2", "mandatory_fallback", True, "test"),)


def math_ids() -> tuple[str, ...]:
    return ("MATH1", "MATH2", "MATH2_3_HA")


def run_cf(input_data, seed: int = 20260630):
    return run_constrained_first_baseline(
        input_data,
        seed,
        math_fallback_rules=fallback_rules(),
        math_course_ids=math_ids(),
    )


def context(input_data, seed: int = 20260630):
    return build_ordering_context(input_data, seed, fallback_rules(), math_ids(), ())


def outcome(result, request_key: str):
    return next(item for item in result.request_outcomes if item.request_key == request_key)


def student_outcome(result, student_id: str):
    return next(item for item in result.student_outcomes if item.student_id == student_id)


def key(student_id: str, course_id: str) -> str:
    return f"primary:{student_id}:{course_id}"


def alt_key(student_id: str, rank: int, course_id: str) -> str:
    return f"alternate:{student_id}:{rank}:{course_id}"


def profile_map(input_data, seed: int = 20260630):
    return {
        profile.student_id: profile
        for profile in build_student_difficulty_profiles(input_data, seed, fallback_rules(), math_ids())
    }


def test_protected_student_orders_before_equal_regular_student() -> None:
    data = canonical(
        [("REG", 12, 1, False), ("PRO", 12, 1, True)],
        [request_row("REG", "CORE"), request_row("PRO", "CORE")],
    )

    assert run_cf(data).student_processing_order[0] == "PRO"


def test_unique_math_primary_risk_student_orders_early() -> None:
    data = canonical(
        [("MATH", 12, 1, False), ("CORE_STU", 12, 1, False)],
        [request_row("MATH", "MATH1"), request_row("CORE_STU", "CORE")],
    )

    assert run_cf(data).student_processing_order[0] == "MATH"


def test_high_demand_primary_count_orders_student_early() -> None:
    student_rows = [("HIGH_STU", 12, 1, False), ("CORE_STU", 12, 1, False)]
    student_rows.extend((f"FILL_{index:03d}", 12, 1, False) for index in range(121))
    request_rows = [request_row("HIGH_STU", "HIGH"), request_row("CORE_STU", "CORE")]
    request_rows.extend(request_row(f"FILL_{index:03d}", "HIGH") for index in range(121))
    data = canonical(student_rows, request_rows)

    assert profile_map(data)["HIGH_STU"].high_demand_primary_count == 1
    assert run_cf(data).student_processing_order.index("HIGH_STU") < run_cf(data).student_processing_order.index("CORE_STU")


def test_scarce_candidate_student_orders_before_flexible_student() -> None:
    data = canonical(
        [("SCARCE_STU", 12, 1, False), ("FLEX_STU", 12, 1, False)],
        [request_row("SCARCE_STU", "SCARCE"), request_row("FLEX_STU", "FLEX")],
    )

    assert run_cf(data).student_processing_order[0] == "SCARCE_STU"


def test_double_period_burden_orders_student_early_when_other_features_match() -> None:
    data = canonical(
        [("DOUBLE_STU", 12, 2, False), ("CORE_STU", 12, 2, False)],
        [request_row("DOUBLE_STU", "DOUBLE"), request_row("CORE_STU", "CORE")],
    )

    assert run_cf(data).student_processing_order[0] == "DOUBLE_STU"


def test_same_seed_student_order_is_reproducible() -> None:
    data = canonical(
        [("A", 12, 1, False), ("B", 12, 1, False)],
        [request_row("A", "CORE"), request_row("B", "CORE")],
    )

    assert run_cf(data, seed=77) == run_cf(data, seed=77)


def test_seeded_tie_break_does_not_override_clear_protected_priority() -> None:
    data = canonical(
        [("REG", 12, 1, False), ("PRO", 12, 1, True)],
        [request_row("REG", "CORE"), request_row("PRO", "CORE")],
    )

    assert run_cf(data, seed=1).student_processing_order[0] == "PRO"
    assert run_cf(data, seed=999).student_processing_order[0] == "PRO"


def test_currently_feasible_candidate_count_orders_primary_requests() -> None:
    data = canonical(
        [("STU_1", 12, 2, False)],
        [request_row("STU_1", "SCARCE"), request_row("STU_1", "FLEX")],
    )
    state = AllocationState(data)
    ctx = context(data)
    scarce = data.requests_by_key[key("STU_1", "SCARCE")]
    flex = data.requests_by_key[key("STU_1", "FLEX")]
    remaining = [scarce, flex]

    assert primary_request_priority(state, data, scarce, remaining, ctx) < primary_request_priority(state, data, flex, remaining, ctx)


def test_zero_feasible_primary_still_gets_unmet_outcome() -> None:
    data = canonical(
        [("STU_1", 12, 1, False)],
        [request_row("STU_1", "SCARCE")],
        [row for row in base_sections() if row[1] != "SCARCE"],
    )

    result = run_cf(data)

    assert outcome(result, key("STU_1", "SCARCE")).status == PrimaryRequestStatus.UNMET_NO_CANDIDATES


def test_math_critical_request_wins_equal_request_priority() -> None:
    data = canonical(
        [("STU_1", 12, 2, False)],
        [request_row("STU_1", "MATH1"), request_row("STU_1", "CORE")],
    )
    state = AllocationState(data)
    ctx = context(data)
    math = data.requests_by_key[key("STU_1", "MATH1")]
    core = data.requests_by_key[key("STU_1", "CORE")]
    remaining = [math, core]

    assert primary_request_priority(state, data, math, remaining, ctx) < primary_request_priority(state, data, core, remaining, ctx)


def test_high_demand_request_wins_equal_request_priority() -> None:
    student_rows = [("STU_1", 12, 2, False)]
    student_rows.extend((f"FILL_{index:03d}", 12, 1, False) for index in range(121))
    request_rows = [request_row("STU_1", "HIGH"), request_row("STU_1", "CORE")]
    request_rows.extend(request_row(f"FILL_{index:03d}", "HIGH") for index in range(121))
    data = canonical(student_rows, request_rows)

    assert outcome(run_cf(data), key("STU_1", "HIGH")).candidate_attempts


def test_double_period_request_wins_equal_request_priority() -> None:
    data = canonical(
        [("STU_1", 12, 3, False)],
        [request_row("STU_1", "DOUBLE"), request_row("STU_1", "CORE")],
    )
    state = AllocationState(data)
    ctx = context(data)
    double = data.requests_by_key[key("STU_1", "DOUBLE")]
    core = data.requests_by_key[key("STU_1", "CORE")]
    remaining = [double, core]

    assert primary_request_priority(state, data, double, remaining, ctx) < primary_request_priority(state, data, core, remaining, ctx)


def test_request_order_recomputes_after_assignment_changes_state() -> None:
    data = canonical(
        [("STU_1", 12, 2, False)],
        [request_row("STU_1", "SCARCE"), request_row("STU_1", "FLEX")],
    )

    result = run_cf(data)

    assert outcome(result, key("STU_1", "SCARCE")).status == PrimaryRequestStatus.ASSIGNED
    assert outcome(result, key("STU_1", "FLEX")).assigned_linked_section_group_id == "FLEX_2"


def test_request_ordering_is_independent_of_raw_request_row_order() -> None:
    student_rows = [("STU_1", 12, 2, False)]
    first = canonical(student_rows, [request_row("STU_1", "SCARCE"), request_row("STU_1", "FLEX")])
    second = canonical(student_rows, [request_row("STU_1", "FLEX"), request_row("STU_1", "SCARCE")])

    assert run_cf(first) == run_cf(second)


def test_candidate_remaining_capacity_ratio_orders_sections() -> None:
    data = canonical(
        [("FILL", 12, 1, False), ("STU_1", 12, 1, False)],
        [request_row("FILL", "FLEX"), request_row("STU_1", "FLEX")],
        [
            section_row("SEC_FLEX_A", "FLEX", "P1", capacity=2, group_id="FLEX_A"),
            section_row("SEC_FLEX_B", "FLEX", "P2", capacity=4, group_id="FLEX_B"),
        ],
    )
    state = AllocationState(data)
    state.try_assign("FILL", key("FILL", "FLEX"), "FLEX_A")
    req = data.requests_by_key[key("STU_1", "FLEX")]
    ctx = context(data)

    assert candidate_section_priority(state, data, req, "FLEX_B", (), ctx) < candidate_section_priority(state, data, req, "FLEX_A", (), ctx)


def test_candidate_period_pressure_orders_less_contested_period() -> None:
    student_rows = [("STU_1", 12, 1, False)]
    student_rows.extend((f"PRESS_{index}", 12, 1, False) for index in range(3))
    request_rows = [request_row("STU_1", "FLEX")]
    request_rows.extend(request_row(f"PRESS_{index}", "PRESS") for index in range(3))
    data = canonical(student_rows, request_rows)
    state = AllocationState(data)
    req = data.requests_by_key[key("STU_1", "FLEX")]
    ctx = context(data)

    assert candidate_section_priority(state, data, req, "FLEX_2", (), ctx) < candidate_section_priority(state, data, req, "FLEX_1", (), ctx)


def test_candidate_future_option_preservation_orders_section() -> None:
    data = canonical(
        [("STU_1", 12, 2, False)],
        [request_row("STU_1", "FLEX"), request_row("STU_1", "NEXT")],
    )
    state = AllocationState(data)
    req = data.requests_by_key[key("STU_1", "FLEX")]
    remaining = list(data.students_by_id["STU_1"].primary_requests)
    ctx = context(data)

    assert candidate_section_priority(state, data, req, "FLEX_2", remaining, ctx) < candidate_section_priority(state, data, req, "FLEX_1", remaining, ctx)


def test_math2_3_candidate_pressure_combines_both_periods() -> None:
    section_rows = base_sections() + [
        section_row("SEC_MATH23_2", "MATH2_3_HA", "P3", "P4", capacity=10, group_id="MATH23_2"),
    ]
    student_rows = [("STU_1", 12, 2, False), ("P5", 12, 1, False), ("P6", 12, 1, False)]
    request_rows = [
        request_row("STU_1", "MATH2_3_HA"),
        request_row("P5", "ALT1"),
        request_row("P6", "DOUBLE"),
    ]
    data = canonical(student_rows, request_rows, section_rows)
    state = AllocationState(data)
    req = data.requests_by_key[key("STU_1", "MATH2_3_HA")]
    ctx = context(data)

    assert candidate_section_priority(state, data, req, "MATH23_2", (), ctx) < candidate_section_priority(state, data, req, "MATH23_1", (), ctx)


def test_gov_econ_period_pressure_counts_logical_section_once() -> None:
    data = canonical([("STU_1", 12, 1, False)], [
        request_row("STU_1", "GOV_ECON_REG", "primary", "", "gov_econ_block", "GOV_ECON_REG"),
        request_row("STU_1", "GOV_ECON_REG", "primary", "", "gov_econ_block", "GOV_ECON_REG"),
    ])
    ctx = context(data)

    assert ctx.period_pressure["P7"] == 1


def test_candidate_ordering_same_seed_is_reproducible() -> None:
    data = canonical([("STU_1", 12, 1, False)], [request_row("STU_1", "FLEX")])

    assert run_cf(data, seed=3) == run_cf(data, seed=3)


def test_assignment_still_records_try_assign_rejections() -> None:
    data = canonical([("STU_1", 12, 1, False)], [request_row("STU_1", "CORE"), request_row("STU_1", "SCARCE")])
    result = run_cf(data)

    assert AssignmentRejectionReason.PERIOD_CONFLICT in outcome(result, key("STU_1", "CORE")).candidate_attempts[0].rejection_reasons


def test_constrained_first_satisfies_fixture_that_random_seed_can_block() -> None:
    data = canonical(
        [("STU_1", 12, 2, False)],
        [request_row("STU_1", "SCARCE"), request_row("STU_1", "FLEX")],
    )

    random_result = run_seeded_random_baseline(data, seed=0)
    constrained = run_cf(data, seed=0)

    assert student_outcome(random_result, "STU_1").primary_unmet_count >= 1
    assert student_outcome(constrained, "STU_1").primary_unmet_count == 0


def test_least_constraining_candidate_avoids_future_conflict_fixture() -> None:
    data = canonical(
        [("STU_1", 12, 2, False)],
        [request_row("STU_1", "FLEX"), request_row("STU_1", "NEXT")],
    )

    result = run_cf(data)

    assert outcome(result, key("STU_1", "FLEX")).assigned_linked_section_group_id == "FLEX_2"
    assert student_outcome(result, "STU_1").primary_unmet_count == 0


def test_math2_3_priority_can_reduce_math_primary_loss_fixture() -> None:
    data = canonical(
        [("STU_1", 12, 2, False)],
        [request_row("STU_1", "MATH2_3_HA"), request_row("STU_1", "ALT1")],
    )

    constrained = run_cf(data)

    assert outcome(constrained, key("STU_1", "MATH2_3_HA")).status == PrimaryRequestStatus.ASSIGNED


def test_high_demand_priority_reduces_high_demand_violation_fixture() -> None:
    student_rows = [("STU_1", 12, 1, False)]
    student_rows.extend((f"FILL_{index:03d}", 12, 1, False) for index in range(121))
    request_rows = [request_row("STU_1", "HIGH"), request_row("STU_1", "CORE")]
    request_rows.extend(request_row(f"FILL_{index:03d}", "HIGH") for index in range(121))
    data = canonical(student_rows, request_rows)
    result = run_cf(data)

    assert outcome(result, key("STU_1", "HIGH")).status == PrimaryRequestStatus.ASSIGNED


def test_fallback_remains_between_primary_and_alternates() -> None:
    data = canonical(
        [("STU_1", 12, 1, False)],
        [request_row("STU_1", "MATH2_3_HA"), request_row("STU_1", "ALT1", "alternate", 1, "alternate")],
        [row for row in base_sections() if row[1] != "MATH2_3_HA"],
    )
    result = run_cf(data)

    assert result.mandatory_fallback_outcomes[0].status == MandatoryFallbackStatus.ASSIGNED
    assert outcome(result, alt_key("STU_1", 1, "ALT1")).status == AlternateRequestStatus.NOT_NEEDED


def test_fallback_success_does_not_change_primary_satisfaction() -> None:
    data = canonical(
        [("STU_1", 12, 1, False)],
        [request_row("STU_1", "MATH2_3_HA")],
        [row for row in base_sections() if row[1] != "MATH2_3_HA"],
    )
    result = run_cf(data)
    student = student_outcome(result, "STU_1")

    assert result.mandatory_fallback_outcomes[0].status == MandatoryFallbackStatus.ASSIGNED
    assert student.primary_assigned_count == 0
    assert student.mandatory_fallback_assigned_count == 1


def test_alternates_remain_strict_rank_order() -> None:
    data = canonical(
        [("STU_1", 12, 1, False)],
        [
            request_row("STU_1", "SCARCE"),
            request_row("STU_1", "ALT1", "alternate", 1, "alternate"),
            request_row("STU_1", "ALT2", "alternate", 2, "alternate"),
        ],
        [row for row in base_sections() if row[1] == "ALT1" or row[1] == "ALT2"],
    )
    result = run_cf(data)

    assert outcome(result, alt_key("STU_1", 1, "ALT1")).status == AlternateRequestStatus.ASSIGNED
    assert outcome(result, alt_key("STU_1", 2, "ALT2")).status == AlternateRequestStatus.NOT_NEEDED


def test_student_outcome_and_policy_report_use_random_baseline_counting_semantics() -> None:
    data = canonical([("STU_1", 12, 1, False)], [request_row("STU_1", "SCARCE"), request_row("STU_1", "FLEX")])
    result = run_cf(data)

    assert student_outcome(result, "STU_1").primary_request_count == 2
    assert result.policy_report.ordinary_violation_student_ids == ()


def test_random_baseline_regression_fixture_is_unchanged_by_constrained_first_module() -> None:
    data = canonical(
        [("STU_1", 12, 2, False)],
        [request_row("STU_1", "SCARCE"), request_row("STU_1", "FLEX")],
    )

    first = run_seeded_random_baseline(data, seed=0)
    second = run_seeded_random_baseline(data, seed=0)

    assert first == second
    assert student_outcome(first, "STU_1").primary_unmet_count == 1


def test_constrained_first_same_seed_full_result_equal() -> None:
    data = canonical(
        [("STU_1", 12, 2, False), ("STU_2", 12, 1, False)],
        [request_row("STU_1", "SCARCE"), request_row("STU_1", "FLEX"), request_row("STU_2", "CORE")],
    )

    assert run_cf(data, seed=42) == run_cf(data, seed=42)


def test_equivalent_raw_row_order_does_not_change_result() -> None:
    student_df = students([("STU_1", 12, 2, False), ("STU_2", 12, 1, False)])
    request_df = requests([request_row("STU_1", "SCARCE"), request_row("STU_1", "FLEX"), request_row("STU_2", "CORE")])
    section_df = sections()
    first = canonicalize_allocation_input(student_df, request_df, section_df, catalog())
    second = canonicalize_allocation_input(
        student_df.sample(frac=1, random_state=1).reset_index(drop=True),
        request_df.sample(frac=1, random_state=2).reset_index(drop=True),
        section_df.sample(frac=1, random_state=3).reset_index(drop=True),
        catalog().sample(frac=1, random_state=4).reset_index(drop=True),
    )

    assert run_cf(first) == run_cf(second)


def test_capacity_and_consistency_remain_valid() -> None:
    data = canonical(
        [("STU_1", 12, 2, False), ("STU_2", 12, 2, False)],
        [request_row("STU_1", "SCARCE"), request_row("STU_1", "FLEX"), request_row("STU_2", "FLEX")],
    )
    result = run_cf(data)

    assert result.consistency_issues == ()
    assert all(row.assigned_count <= row.capacity for row in result.section_roster_summary)


def test_request_and_student_outcome_counts_are_complete() -> None:
    data = canonical(
        [("STU_1", 12, 2, False), ("STU_2", 12, 1, False)],
        [request_row("STU_1", "SCARCE"), request_row("STU_1", "FLEX"), request_row("STU_2", "CORE")],
    )
    result = run_cf(data)

    assert len(result.request_outcomes) == len(data.logical_requests)
    assert len(result.student_outcomes) == len(data.students)


def test_constrained_first_does_not_modify_canonical_input() -> None:
    data = canonical()
    before = data

    run_cf(data)

    assert data == before


def test_constrained_first_does_not_pollute_global_random_state() -> None:
    random.seed(99)
    before = random.getstate()

    run_cf(canonical(), seed=3)

    assert random.getstate() == before


def test_result_models_do_not_contain_global_infeasible_semantics() -> None:
    result = run_cf(canonical([("STU_1", 12, 1, False)], [request_row("STU_1", "SCARCE"), request_row("STU_1", "FLEX")]))

    assert "infeasible" not in repr(result).lower()


def test_math_policy_reads_constrained_first_fallback_result() -> None:
    data = canonical(
        [("STU_1", 12, 1, False)],
        [request_row("STU_1", "MATH2_3_HA")],
        [row for row in base_sections() if row[1] != "MATH2_3_HA"],
    )
    result = run_cf(data)
    report = evaluate_math_policy(data, result, math_ids(), fallback_rules())

    assert report.coverage_satisfied_by_mandatory_fallback_student_ids == ("STU_1",)
