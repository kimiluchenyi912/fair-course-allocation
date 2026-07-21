from __future__ import annotations

import json

import pytest

from src.allocation import CpSatSolveStatus, CpSatStageName, run_fair_cp_sat_solver
from src.cp_sat_robustness_runner import (
    CpSatEvaluationError,
    CpSatColdStartRecoveryRunner,
    load_recovery_evaluation_manifest,
)

from tests.test_cp_sat_solver import (
    canonical,
    fallback_rules,
    math_ids,
    request_row,
    section_row,
)


def _complete_fixture():
    courses = ["CORE_A", "CORE_B", "CORE_C", "CORE_D", "ALT1"]
    requests = [request_row("STU_1", course) for course in courses]
    sections = [
        section_row(f"SEC_{course}", course, f"P{index + 1}", capacity=10, group_id=f"{course}_1")
        for index, course in enumerate(courses)
    ]
    return canonical([("STU_1", 12, 5, False)], requests, sections)


def test_recovery_manifest_is_normal_only_and_explicit() -> None:
    payload = load_recovery_evaluation_manifest()
    assert set(payload["scenario_groups"]) == {"normal"}
    assert len(payload["scenarios"]) == 12
    config = payload["solver_configuration"]
    assert config["internal_feasibility_hint_strategy"] == "constrained_first"
    assert config["use_constrained_first_hint"] is False
    assert config["external_persisted_seed"] is False
    assert config["initial_solution_artifact_dir"] is None


def test_recovery_manifest_rejects_holdout_or_wrong_strategy(tmp_path) -> None:
    payload = load_recovery_evaluation_manifest()
    payload["scenarios"].append({**payload["scenarios"][0], "scenario_id": "holdout_bad"})
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(CpSatEvaluationError):
        load_recovery_evaluation_manifest(path)


def test_internal_repair_finds_full_policy_valid_incumbent() -> None:
    result = run_fair_cp_sat_solver(
        _complete_fixture(),
        seed=20260630,
        math_fallback_rules=fallback_rules(),
        math_course_ids=math_ids(),
        max_time_seconds_per_stage=1,
        bootstrap_time_seconds=1,
        internal_repair_time_seconds=1,
        max_total_time_seconds=5,
        num_search_workers=1,
        use_constrained_first_hint=False,
        internal_feasibility_hint_strategy="constrained_first",
    )
    assert result.model_stats.internal_repair_status in {CpSatSolveStatus.FEASIBLE, CpSatSolveStatus.OPTIMAL}
    assert result.model_stats.internal_repair_incumbent_found is True
    assert result.policy_report is not None
    assert result.model_stats.post_solve_policy_gate_pass is True
    assert result.consistency_issues == ()


def test_internal_repair_is_the_only_stage_with_repair_hint() -> None:
    result = run_fair_cp_sat_solver(
        _complete_fixture(),
        seed=20260630,
        math_fallback_rules=fallback_rules(),
        math_course_ids=math_ids(),
        max_time_seconds_per_stage=1,
        bootstrap_time_seconds=1,
        internal_repair_time_seconds=1,
        max_total_time_seconds=5,
        num_search_workers=1,
        use_constrained_first_hint=False,
        internal_feasibility_hint_strategy="constrained_first",
    )
    repair = [item for item in result.stage_diagnostics if item.stage_name == CpSatStageName.INTERNAL_REPAIR_FEASIBILITY]
    later = [item for item in result.stage_diagnostics if item.stage_name != CpSatStageName.INTERNAL_REPAIR_FEASIBILITY]
    assert repair and all(item.repair_hint_enabled for item in repair)
    assert all(not item.repair_hint_enabled for item in later)


def test_internal_hint_covers_all_candidate_variables_and_auxiliaries() -> None:
    result = run_fair_cp_sat_solver(
        _complete_fixture(),
        seed=20260630,
        math_fallback_rules=fallback_rules(),
        math_course_ids=math_ids(),
        max_time_seconds_per_stage=1,
        bootstrap_time_seconds=1,
        internal_repair_time_seconds=1,
        max_total_time_seconds=5,
        num_search_workers=1,
        use_constrained_first_hint=False,
        internal_feasibility_hint_strategy="constrained_first",
    )
    stats = result.model_stats
    assert stats.internal_hint_candidate_variables == stats.internal_hint_candidate_variables_hinted
    assert stats.internal_hint_candidate_coverage_rate == 1.0
    assert stats.internal_hint_duplicate_keys == 0
    assert stats.internal_hint_out_of_domain_keys == 0
    assert stats.internal_hint_auxiliary_variables_hinted >= 0


def test_hint_application_does_not_change_model_structure() -> None:
    result = run_fair_cp_sat_solver(
        _complete_fixture(),
        seed=20260630,
        math_fallback_rules=fallback_rules(),
        math_course_ids=math_ids(),
        max_time_seconds_per_stage=1,
        bootstrap_time_seconds=1,
        internal_repair_time_seconds=1,
        max_total_time_seconds=5,
        num_search_workers=1,
        use_constrained_first_hint=False,
        internal_feasibility_hint_strategy="constrained_first",
    )
    assert result.model_stats.model_invariance_equal is True
    assert result.model_stats.model_invariance_before_hint_hash
    assert result.model_stats.model_invariance_before_hint_hash == result.model_stats.model_invariance_after_hint_hash


def test_default_solver_does_not_enable_internal_recovery() -> None:
    result = run_fair_cp_sat_solver(
        _complete_fixture(),
        seed=20260630,
        math_fallback_rules=fallback_rules(),
        math_course_ids=math_ids(),
        max_time_seconds_per_stage=1,
        use_constrained_first_hint=False,
        enforce_final_schedule_hard_constraints=False,
    )
    assert result.model_stats.internal_feasibility_hint_strategy == "none"
    assert result.model_stats.internal_repair_status is None


def test_recovery_requires_final_schedule_hard_constraints() -> None:
    with pytest.raises(ValueError, match="final schedule hard constraints"):
        run_fair_cp_sat_solver(
            _complete_fixture(),
            seed=1,
            max_time_seconds_per_stage=1,
            enforce_final_schedule_hard_constraints=False,
            internal_feasibility_hint_strategy="constrained_first",
        )


def test_unknown_with_validated_incumbent_is_distinct_status() -> None:
    assert CpSatSolveStatus.UNKNOWN_WITH_VALIDATED_INCUMBENT.value == "UNKNOWN_WITH_VALIDATED_INCUMBENT"


def test_recovery_runner_fails_closed_on_nonempty_output(tmp_path) -> None:
    runner = CpSatColdStartRecoveryRunner()
    output = tmp_path / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(CpSatEvaluationError, match="non-empty"):
        runner.run(output)
    assert (output / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_recovery_runner_selects_manifest_order() -> None:
    runner = CpSatColdStartRecoveryRunner()
    selected = runner.select()
    assert selected[0].scenario_id == "normal_dev_reference_2026"
    assert len(selected) == 12
