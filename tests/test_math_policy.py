from __future__ import annotations

import pandas as pd
import pytest

from src.allocation import (
    MathCoverageStatus,
    MathFallbackConfigError,
    MathPolicyViolationType,
    canonicalize_allocation_input,
    evaluate_math_policy,
    math_course_ids_from_catalog,
    parse_math_fallback_rules,
    run_seeded_random_baseline,
)


def catalog() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("MATH1", "Integrated Math 1", "Mathematics", 1, "standard"),
            ("MATH2", "Integrated Math 2", "Mathematics", 1, "standard"),
            ("MATH2_3_HA", "Integrated Math 2/3 Honors Accelerated", "Mathematics", 2, "double_period"),
            ("AP_STATS", "AP Statistics", "Mathematics", 1, "standard"),
            ("CALC_D_LINALG", "Calc D + Linear Algebra", "Mathematics", 1, "semester_block"),
            ("ENGLISH", "English", "English", 1, "standard"),
            ("SCIENCE", "Science", "Science", 1, "standard"),
        ],
        columns=["course_id", "course_name", "department", "periods_required", "schedule_structure"],
    )


def students(rows: list[tuple[str, int]] | None = None) -> pd.DataFrame:
    rows = rows or [("STU_1", 2)]
    return pd.DataFrame(
        [(student_id, 10, target, "none", "false", "", "") for student_id, target in rows],
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


def request_row(student_id: str, course_id: str, request_type: str = "primary", rank: int | str = "") -> tuple:
    return (student_id, course_id, request_type, rank, "alternate" if request_type == "alternate" else "", "")


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


def section_row(course_id: str, period: str, group_id: str | None = None, period_2: str = "", semester: str = "full_year") -> tuple:
    group_id = group_id or f"{course_id}_1"
    return (
        f"SEC_{group_id}",
        course_id,
        period,
        period_2,
        semester,
        40,
        group_id if semester == "paired" else "",
        group_id,
        course_id,
        "",
    )


def sections(course_ids: list[str] | None = None) -> pd.DataFrame:
    course_ids = course_ids if course_ids is not None else ["MATH1", "MATH2", "MATH2_3_HA", "AP_STATS", "CALC_D_LINALG", "ENGLISH"]
    rows = []
    for index, course_id in enumerate(course_ids, start=1):
        if course_id == "MATH2_3_HA":
            rows.append(section_row(course_id, "P3", "MATH23_1", "P4"))
        elif course_id == "CALC_D_LINALG":
            rows.append(section_row(course_id, f"P{index}", "CALCD_1", "", "paired"))
        else:
            rows.append(section_row(course_id, f"P{index}", f"{course_id}_1"))
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


def fallback_df(rows: list[tuple] | None = None) -> pd.DataFrame:
    rows = rows or [("MATH2_3_HA", "MATH2", "mandatory_fallback", "true", "test", "test mapping")]
    return pd.DataFrame(
        rows,
        columns=["source_course_id", "fallback_course_id", "policy_type", "enabled", "source_type", "notes"],
    )


def canonical(rows: list[tuple], section_course_ids: list[str] | None = None, student_rows: list[tuple[str, int]] | None = None):
    return canonicalize_allocation_input(
        students(student_rows),
        requests(rows),
        sections(section_course_ids),
        catalog()[["course_id", "periods_required", "schedule_structure"]],
    )


def report_for(rows: list[tuple], section_course_ids: list[str] | None = None, student_rows: list[tuple[str, int]] | None = None):
    data = canonical(rows, section_course_ids, student_rows)
    baseline = run_seeded_random_baseline(data, seed=7)
    rules = parse_math_fallback_rules(catalog(), fallback_df())
    return evaluate_math_policy(data, baseline, math_course_ids_from_catalog(catalog()), rules), data, baseline


def report_for_completed_fallback(
    rows: list[tuple],
    section_course_ids: list[str] | None = None,
    student_rows: list[tuple[str, int]] | None = None,
):
    data = canonical(rows, section_course_ids, student_rows)
    rules = parse_math_fallback_rules(catalog(), fallback_df())
    baseline = run_seeded_random_baseline(
        data,
        seed=7,
        math_fallback_rules=rules,
        math_course_ids=math_course_ids_from_catalog(catalog()),
    )
    return evaluate_math_policy(data, baseline, math_course_ids_from_catalog(catalog()), rules), data, baseline


def student(report, student_id: str):
    return next(item for item in report.student_outcomes if item.student_id == student_id)


def test_department_mathematics_courses_are_identified_as_math() -> None:
    assert "MATH1" in math_course_ids_from_catalog(catalog())


def test_non_mathematics_course_is_not_identified_as_math() -> None:
    assert "ENGLISH" not in math_course_ids_from_catalog(catalog())


def test_ap_stats_is_math_without_title_substring_matching() -> None:
    assert "AP_STATS" in math_course_ids_from_catalog(catalog())


def test_calc_d_linalg_is_math() -> None:
    assert "CALC_D_LINALG" in math_course_ids_from_catalog(catalog())


def test_math2_3_to_math2_fallback_config_loads() -> None:
    rules = parse_math_fallback_rules(catalog(), fallback_df())

    assert rules[0].source_course_id == "MATH2_3_HA"
    assert rules[0].fallback_course_id == "MATH2"
    assert rules[0].enabled is True


def test_unknown_source_is_rejected() -> None:
    with pytest.raises(MathFallbackConfigError):
        parse_math_fallback_rules(catalog(), fallback_df([("UNKNOWN", "MATH2", "mandatory_fallback", "true", "test", "")]))


def test_unknown_fallback_is_rejected() -> None:
    with pytest.raises(MathFallbackConfigError):
        parse_math_fallback_rules(catalog(), fallback_df([("MATH2_3_HA", "UNKNOWN", "mandatory_fallback", "true", "test", "")]))


def test_non_math_source_or_fallback_is_rejected() -> None:
    with pytest.raises(MathFallbackConfigError):
        parse_math_fallback_rules(catalog(), fallback_df([("ENGLISH", "MATH2", "mandatory_fallback", "true", "test", "")]))
    with pytest.raises(MathFallbackConfigError):
        parse_math_fallback_rules(catalog(), fallback_df([("MATH2_3_HA", "ENGLISH", "mandatory_fallback", "true", "test", "")]))


def test_blank_source_or_fallback_is_rejected() -> None:
    with pytest.raises(MathFallbackConfigError):
        parse_math_fallback_rules(catalog(), fallback_df([("", "MATH2", "mandatory_fallback", "true", "test", "")]))
    with pytest.raises(MathFallbackConfigError):
        parse_math_fallback_rules(catalog(), fallback_df([("MATH2_3_HA", "", "mandatory_fallback", "true", "test", "")]))


def test_conflicting_duplicate_mapping_is_rejected() -> None:
    with pytest.raises(MathFallbackConfigError):
        parse_math_fallback_rules(
            catalog(),
            fallback_df(
                [
                    ("MATH2_3_HA", "MATH2", "mandatory_fallback", "true", "test", ""),
                    ("MATH2_3_HA", "MATH1", "mandatory_fallback", "true", "test", ""),
                ]
            ),
        )


def test_fallback_config_order_is_stable() -> None:
    rules = parse_math_fallback_rules(
        catalog(),
        fallback_df(
            [
                ("MATH2_3_HA", "MATH2", "mandatory_fallback", "true", "test", ""),
                ("AP_STATS", "MATH2", "mandatory_fallback", "false", "test", ""),
            ]
        ),
    )

    assert [rule.source_course_id for rule in rules] == ["AP_STATS", "MATH2_3_HA"]


def test_only_regular_single_math_primary_assigned_is_satisfied() -> None:
    report, _, _ = report_for([request_row("STU_1", "MATH1")])

    assert student(report, "STU_1").coverage_status == MathCoverageStatus.SATISFIED_BY_PRIMARY


def test_only_regular_single_math_primary_unmet_is_single_math_violation() -> None:
    report, _, _ = report_for([request_row("STU_1", "MATH1")], section_course_ids=["ENGLISH"])
    outcome = student(report, "STU_1")

    assert outcome.coverage_status == MathCoverageStatus.VIOLATED_SINGLE_MATH_REQUIRED
    assert outcome.violation_type == MathPolicyViolationType.SINGLE_MATH_REQUIRED
    assert report.single_math_required_violation_student_ids == ("STU_1",)


def test_only_math2_3_primary_assigned_is_satisfied() -> None:
    report, _, _ = report_for([request_row("STU_1", "MATH2_3_HA")])

    assert student(report, "STU_1").coverage_status == MathCoverageStatus.SATISFIED_BY_PRIMARY


def test_only_math2_3_primary_unmet_is_pending_mandatory_fallback_not_direct_violation() -> None:
    report, _, _ = report_for([request_row("STU_1", "MATH2_3_HA")], section_course_ids=["MATH2"])
    outcome = student(report, "STU_1")

    assert outcome.coverage_status == MathCoverageStatus.PENDING_MANDATORY_FALLBACK
    assert outcome.fallback_source_course_id == "MATH2_3_HA"
    assert outcome.fallback_target_course_id == "MATH2"
    assert report.direct_violation_student_ids == ()
    assert report.fallback_pending_student_ids == ("STU_1",)
    assert report.policy_currently_satisfied is False


def test_completed_math2_3_fallback_assignment_satisfies_math_coverage() -> None:
    report, _, baseline = report_for_completed_fallback([request_row("STU_1", "MATH2_3_HA")], section_course_ids=["MATH2"])
    outcome = student(report, "STU_1")

    assert outcome.coverage_status == MathCoverageStatus.SATISFIED_BY_MANDATORY_FALLBACK
    assert outcome.fallback_assignment_key is not None
    assert report.coverage_satisfied_by_mandatory_fallback_student_ids == ("STU_1",)
    assert report.coverage_pending_fallback_student_ids == ()
    assert baseline.student_outcomes[0].primary_unmet_count == 1


def test_completed_math2_3_fallback_failure_is_final_math_coverage_violation() -> None:
    report, _, _ = report_for_completed_fallback([request_row("STU_1", "MATH2_3_HA")], section_course_ids=["ENGLISH"])
    outcome = student(report, "STU_1")

    assert outcome.coverage_status == MathCoverageStatus.VIOLATED_MANDATORY_FALLBACK_FAILED
    assert outcome.violation_type == MathPolicyViolationType.MANDATORY_FALLBACK_FAILED
    assert report.fallback_failed_student_ids == ("STU_1",)
    assert report.current_math_coverage_violation_student_ids == ("STU_1",)
    assert report.direct_violation_student_ids == ()


def test_math_alternate_assignment_does_not_clear_single_math_primary_violation() -> None:
    report, _, _ = report_for(
        [request_row("STU_1", "MATH1"), request_row("STU_1", "MATH2", "alternate", 1)],
        section_course_ids=["MATH2"],
    )

    assert student(report, "STU_1").coverage_status == MathCoverageStatus.VIOLATED_SINGLE_MATH_REQUIRED


def test_non_math_primary_is_no_math_required() -> None:
    report, _, _ = report_for([request_row("STU_1", "ENGLISH")])

    assert student(report, "STU_1").coverage_status == MathCoverageStatus.NO_MATH_REQUIRED


def test_two_math_primaries_at_least_one_assigned_is_satisfied() -> None:
    report, _, _ = report_for([request_row("STU_1", "MATH1"), request_row("STU_1", "MATH2")], section_course_ids=["MATH1"])

    assert student(report, "STU_1").coverage_status == MathCoverageStatus.SATISFIED_BY_PRIMARY


def test_two_math_primaries_all_unmet_is_multiple_math_violation() -> None:
    report, _, _ = report_for([request_row("STU_1", "MATH1"), request_row("STU_1", "MATH2")], section_course_ids=["ENGLISH"])
    outcome = student(report, "STU_1")

    assert outcome.coverage_status == MathCoverageStatus.VIOLATED_MULTIPLE_MATH_AT_LEAST_ONE
    assert outcome.violation_type == MathPolicyViolationType.MULTIPLE_MATH_AT_LEAST_ONE
    assert report.multiple_math_at_least_one_violation_student_ids == ("STU_1",)


def test_multiple_math_counts_are_accurate() -> None:
    report, _, _ = report_for([request_row("STU_1", "MATH1"), request_row("STU_1", "MATH2")], section_course_ids=["ENGLISH"])

    assert student(report, "STU_1").math_primary_count == 2
    assert report.multiple_math_student_ids == ("STU_1",)


def test_input_request_order_does_not_change_report() -> None:
    first, _, _ = report_for([request_row("STU_1", "MATH1"), request_row("STU_1", "MATH2")], section_course_ids=["MATH1"])
    second, _, _ = report_for([request_row("STU_1", "MATH2"), request_row("STU_1", "MATH1")], section_course_ids=["MATH1"])

    assert first == second


def test_no_math_primary_student_is_not_violation() -> None:
    report, _, _ = report_for([request_row("STU_1", "ENGLISH")])

    assert report.no_math_primary_student_ids == ("STU_1",)
    assert report.direct_violation_student_ids == ()


def test_every_student_has_one_math_policy_outcome() -> None:
    report, _, _ = report_for(
        [request_row("STU_1", "MATH1"), request_row("STU_2", "ENGLISH")],
        student_rows=[("STU_1", 1), ("STU_2", 1)],
    )

    assert [item.student_id for item in report.student_outcomes] == ["STU_1", "STU_2"]


def test_satisfied_pending_and_violated_sets_are_disjoint_and_complete() -> None:
    report, _, _ = report_for(
        [
            request_row("STU_1", "MATH1"),
            request_row("STU_2", "MATH2_3_HA"),
            request_row("STU_3", "MATH2"),
            request_row("STU_4", "ENGLISH"),
        ],
        section_course_ids=["MATH1", "ENGLISH"],
        student_rows=[("STU_1", 1), ("STU_2", 2), ("STU_3", 1), ("STU_4", 1)],
    )
    grouped = [
        set(report.coverage_satisfied_student_ids),
        set(report.coverage_pending_fallback_student_ids),
        set(report.current_math_coverage_violation_student_ids),
        set(report.no_math_primary_student_ids),
    ]

    assert sum(len(group) for group in grouped) == 4
    assert set.union(*grouped) == {"STU_1", "STU_2", "STU_3", "STU_4"}
    assert all(first.isdisjoint(second) for index, first in enumerate(grouped) for second in grouped[index + 1 :])


def test_report_order_is_deterministic() -> None:
    report, _, _ = report_for(
        [request_row("STU_2", "ENGLISH"), request_row("STU_1", "MATH1")],
        student_rows=[("STU_2", 1), ("STU_1", 1)],
    )

    assert [item.student_id for item in report.student_outcomes] == ["STU_1", "STU_2"]


def test_evaluator_does_not_modify_inputs_and_repeated_evaluation_is_equal() -> None:
    data = canonical([request_row("STU_1", "MATH1")])
    baseline = run_seeded_random_baseline(data, seed=7)
    before_data = data
    before_baseline = baseline
    rules = parse_math_fallback_rules(catalog(), fallback_df())

    first = evaluate_math_policy(data, baseline, math_course_ids_from_catalog(catalog()), rules)
    second = evaluate_math_policy(data, baseline, math_course_ids_from_catalog(catalog()), rules)

    assert first == second
    assert data == before_data
    assert baseline == before_baseline
