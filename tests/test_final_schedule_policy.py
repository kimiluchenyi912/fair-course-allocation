from __future__ import annotations

import csv
from types import SimpleNamespace

import pytest

from src.final_schedule_policy import (
    BELOW_MINIMUM_COURSE_COUNT,
    ORDINARY_PRIMARY_UNMET_OVER_LIMIT,
    PROTECTED_PRIMARY_UNMET,
    SCHEDULE_GAP_OVER_LIMIT,
    evaluate_final_schedule_policy,
    load_policy_reports_from_artifacts,
    main,
    summary_row,
    violation_row,
)


def _student(
    student_id: str,
    *,
    target: int,
    assigned: int,
    primary_unmet: int,
    protected: bool = False,
    grade: int = 11,
    alternates: int = 0,
    assigned_period_units: int | None = None,
):
    return SimpleNamespace(
        student_id=student_id,
        grade=grade,
        target_period_units=target,
        assigned_period_units=assigned if assigned_period_units is None else assigned_period_units,
        remaining_period_units=max(target - (assigned if assigned_period_units is None else assigned_period_units), 0),
        assignment_keys=tuple(f"{student_id}_ASSIGN_{index}" for index in range(assigned)),
        primary_unmet_count=primary_unmet,
        alternate_assigned_count=alternates,
        priority_protected=protected,
    )


@pytest.mark.parametrize(
    "student",
    [
        pytest.param(_student("ORD_7_TO_6", target=7, assigned=6, primary_unmet=1), id="ordinary_7_to_6"),
        pytest.param(_student("ORD_6_TO_5", target=6, assigned=5, primary_unmet=1), id="ordinary_6_to_5"),
        pytest.param(_student("ORD_5_TO_5", target=5, assigned=5, primary_unmet=0), id="ordinary_5_to_5"),
        pytest.param(
            _student("ALT_FILL", target=7, assigned=7, primary_unmet=1, alternates=1),
            id="alternate_fills_primary_gap",
        ),
        pytest.param(
            _student("PROTECTED_OK", target=6, assigned=5, primary_unmet=0, protected=True),
            id="protected_no_primary_unmet",
        ),
    ],
)
def test_final_schedule_policy_pass_scenarios(student) -> None:
    report = evaluate_final_schedule_policy("alg", (student,))

    assert report.summary.final_schedule_policy_pass is True
    assert report.summary.violating_student_count == 0
    assert report.violations == ()


@pytest.mark.parametrize(
    ("student", "reasons"),
    [
        pytest.param(
            _student("ORD_PRIMARY_2", target=7, assigned=7, primary_unmet=2, alternates=2),
            (ORDINARY_PRIMARY_UNMET_OVER_LIMIT,),
            id="ordinary_two_primary_unmet_even_if_full",
        ),
        pytest.param(
            _student("PROTECTED_PRIMARY", target=6, assigned=6, primary_unmet=1, protected=True),
            (PROTECTED_PRIMARY_UNMET,),
            id="protected_primary_unmet",
        ),
        pytest.param(
            _student("GAP_7_TO_5", target=7, assigned=5, primary_unmet=1),
            (SCHEDULE_GAP_OVER_LIMIT,),
            id="seven_to_five",
        ),
        pytest.param(
            _student("GAP_6_TO_4", target=6, assigned=4, primary_unmet=1),
            (BELOW_MINIMUM_COURSE_COUNT, SCHEDULE_GAP_OVER_LIMIT),
            id="six_to_four",
        ),
        pytest.param(
            _student("GAP_5_TO_3", target=5, assigned=3, primary_unmet=1),
            (BELOW_MINIMUM_COURSE_COUNT, SCHEDULE_GAP_OVER_LIMIT),
            id="five_to_three",
        ),
        pytest.param(
            _student("MIN_5_TO_4", target=5, assigned=4, primary_unmet=1),
            (BELOW_MINIMUM_COURSE_COUNT,),
            id="five_to_four",
        ),
        pytest.param(
            _student("GAP_ASSIGNED_5", target=7, assigned=5, primary_unmet=1),
            (SCHEDULE_GAP_OVER_LIMIT,),
            id="assigned_at_least_five_gap_two",
        ),
        pytest.param(
            _student("MIN_ASSIGNED_4", target=5, assigned=4, primary_unmet=0),
            (BELOW_MINIMUM_COURSE_COUNT,),
            id="gap_one_but_below_minimum",
        ),
        pytest.param(
            _student("MULTI", target=7, assigned=4, primary_unmet=2),
            (BELOW_MINIMUM_COURSE_COUNT, SCHEDULE_GAP_OVER_LIMIT, ORDINARY_PRIMARY_UNMET_OVER_LIMIT),
            id="multiple_reasons",
        ),
    ],
)
def test_final_schedule_policy_fail_scenarios(student, reasons) -> None:
    report = evaluate_final_schedule_policy("alg", (student,))

    assert report.summary.final_schedule_policy_pass is False
    assert report.summary.violating_student_count == 1
    assert report.violations[0].violation_reasons == reasons


def test_violation_counts_are_deduped_but_rule_counts_may_overlap() -> None:
    report = evaluate_final_schedule_policy(
        "alg",
        (
            _student("A", target=7, assigned=4, primary_unmet=2),
            _student("B", target=5, assigned=4, primary_unmet=0),
        ),
    )

    assert report.summary.violating_student_count == 2
    assert report.summary.below_minimum_course_count == 2
    assert report.summary.schedule_gap_over_limit_count == 1
    assert report.summary.ordinary_primary_unmet_violation_count == 1


def test_logical_course_count_uses_assignment_count_not_period_units() -> None:
    report = evaluate_final_schedule_policy(
        "alg",
        (_student("DOUBLE_PERIOD", target=5, assigned=4, primary_unmet=0, assigned_period_units=5),),
    )

    assert report.summary.final_schedule_policy_pass is False
    assert report.violations[0].assigned_logical_course_count == 4
    assert report.violations[0].violation_reasons == (BELOW_MINIMUM_COURSE_COUNT,)


def test_summary_and_violation_rows_have_stable_schema_and_json_reasons() -> None:
    report = evaluate_final_schedule_policy("alg", (_student("MULTI", target=7, assigned=4, primary_unmet=2),))

    assert summary_row(report) == {
        "algorithm_name": "alg",
        "final_schedule_policy_pass": False,
        "violating_student_count": 1,
        "protected_primary_unmet_violation_count": 0,
        "ordinary_primary_unmet_violation_count": 1,
        "schedule_gap_over_limit_count": 1,
        "below_minimum_course_count": 1,
        "minimum_assigned_course_count": 4,
        "maximum_schedule_gap_count": 3,
        "maximum_primary_unmet_count": 2,
    }
    assert violation_row(report.violations[0])["violation_reasons"] == (
        '["below_minimum_course_count","schedule_gap_over_limit","ordinary_primary_unmet_over_limit"]'
    )


def test_violation_sort_order_uses_algorithm_severity_gap_primary_grade_student() -> None:
    report = evaluate_final_schedule_policy(
        "alg",
        (
            _student("Z", target=7, assigned=5, primary_unmet=1, grade=12),
            _student("A", target=5, assigned=4, primary_unmet=0, grade=9),
            _student("P", target=6, assigned=6, primary_unmet=1, protected=True, grade=10),
        ),
    )

    assert [item.student_id for item in report.violations] == ["A", "Z", "P"]


def test_reports_are_deterministic_and_do_not_leak_between_algorithms() -> None:
    pass_report = evaluate_final_schedule_policy("pass_alg", (_student("OK", target=6, assigned=5, primary_unmet=1),))
    fail_report = evaluate_final_schedule_policy("fail_alg", (_student("BAD", target=5, assigned=4, primary_unmet=0),))

    assert pass_report == evaluate_final_schedule_policy("pass_alg", (_student("OK", target=6, assigned=5, primary_unmet=1),))
    assert pass_report.summary.final_schedule_policy_pass is True
    assert fail_report.summary.final_schedule_policy_pass is False
    assert fail_report.summary.violating_student_count == 1


def test_load_policy_reports_from_artifacts_reads_all_algorithms(tmp_path) -> None:
    artifact_dir = _write_student_outcomes_csv(
        tmp_path,
        [
            _csv_student("pass_alg", "OK", target=6, assignment_count=5, primary_unmet=1),
            _csv_student("fail_alg", "BAD", target=5, assignment_count=4, primary_unmet=0),
        ],
    )

    reports = load_policy_reports_from_artifacts(artifact_dir)

    assert [report.summary.algorithm_name for report in reports] == ["pass_alg", "fail_alg"]
    assert [report.summary.final_schedule_policy_pass for report in reports] == [True, False]


def test_artifact_loader_uses_logical_count_columns_not_pipe_delimited_assignment_keys(tmp_path) -> None:
    artifact_dir = _write_student_outcomes_csv(
        tmp_path,
        [
            {
                **_csv_student("alg", "PIPE_KEYS", target=5, assignment_count=4, primary_unmet=0),
                "assignment_keys": "STU|primary:STU:COURSE|SEC_A|STU|primary:STU:OTHER|SEC_B",
            }
        ],
    )

    report = load_policy_reports_from_artifacts(artifact_dir)[0]

    assert report.summary.final_schedule_policy_pass is False
    assert report.summary.below_minimum_course_count == 1
    assert report.violations[0].assigned_logical_course_count == 4


def test_policy_gate_cli_pass_fail_missing_algorithm_and_all_mode(tmp_path, capsys) -> None:
    artifact_dir = _write_student_outcomes_csv(
        tmp_path,
        [
            _csv_student("pass_alg", "OK", target=6, assignment_count=5, primary_unmet=1),
            _csv_student("fail_alg", "BAD", target=5, assignment_count=4, primary_unmet=0),
        ],
    )

    assert main(["--artifact-dir", str(artifact_dir), "--algorithm", "pass_alg"]) == 0
    assert "pass_alg: PASS" in capsys.readouterr().out
    assert main(["--artifact-dir", str(artifact_dir), "--algorithm", "fail_alg"]) == 1
    assert "fail_alg: FAIL" in capsys.readouterr().out
    assert main(["--artifact-dir", str(artifact_dir)]) == 1
    assert "Traceback" not in capsys.readouterr().out
    assert main(["--artifact-dir", str(artifact_dir), "--algorithm", "missing_alg"]) != 0
    assert "Algorithm not found" in capsys.readouterr().out


def test_policy_gate_cli_reports_missing_file_and_invalid_counts_without_traceback(tmp_path, capsys) -> None:
    assert main(["--artifact-dir", str(tmp_path)]) != 0
    missing_output = capsys.readouterr().out
    assert "Missing required artifact file" in missing_output
    assert "Traceback" not in missing_output

    artifact_dir = _write_student_outcomes_csv(
        tmp_path / "bad",
        [_csv_student("alg", "BAD", target=5, assignment_count=4, primary_unmet=0)],
    )
    path = artifact_dir / "student_outcomes.csv"
    text = path.read_text(encoding="utf-8").replace(",5,", ",not_an_int,", 1)
    path.write_text(text, encoding="utf-8")

    assert main(["--artifact-dir", str(artifact_dir)]) != 0
    invalid_output = capsys.readouterr().out
    assert "Invalid integer" in invalid_output
    assert "Traceback" not in invalid_output


def _csv_student(
    algorithm_name: str,
    student_id: str,
    *,
    target: int,
    assignment_count: int,
    primary_unmet: int,
    protected: bool = False,
):
    return {
        "algorithm_name": algorithm_name,
        "student_id": student_id,
        "grade": 11,
        "target_logical_course_count": target,
        "assigned_logical_course_count": assignment_count,
        "target_period_units": target,
        "assignment_keys": "|".join(f"{student_id}_{index}" for index in range(assignment_count)),
        "primary_unmet_count": primary_unmet,
        "alternate_assigned_count": 0,
        "priority_protected": str(protected).lower(),
    }


def _write_student_outcomes_csv(tmp_path, rows):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir(parents=True)
    with (artifact_dir / "student_outcomes.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "algorithm_name",
                "student_id",
                "grade",
                "target_logical_course_count",
                "assigned_logical_course_count",
                "target_period_units",
                "assignment_keys",
                "primary_unmet_count",
                "alternate_assigned_count",
                "priority_protected",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return artifact_dir
