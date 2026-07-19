from __future__ import annotations

import json

import pytest

from src.stress_robustness_runner import (
    DEFAULT_STRESS_SUITE_PATH,
    StressManifestError,
    StressRunnerError,
    load_stress_scenario_suite,
    run_stress_robustness_suite,
    stress_suite_hash,
    validate_stress_scenario_suite,
)


def _payload():
    return json.loads(DEFAULT_STRESS_SUITE_PATH.read_text(encoding="utf-8"))


def test_stress_manifest_has_fifteen_development_and_eight_holdout_scenarios() -> None:
    suite = load_stress_scenario_suite()
    assert len(suite.scenarios) == 23
    assert sum(item.split == "development" for item in suite.scenarios) == 15
    assert sum(item.split == "holdout" for item in suite.scenarios) == 8
    assert sum(item.expected_feasibility == "structurally_infeasible" for item in suite.scenarios if item.split == "development") == 3


def test_stress_suite_hash_is_deterministic() -> None:
    assert stress_suite_hash(load_stress_scenario_suite()) == stress_suite_hash(load_stress_scenario_suite())


def test_stress_dry_run_selects_development_in_order(tmp_path) -> None:
    result = run_stress_robustness_suite(output_dir=tmp_path / "out", dry_run=True, max_scenarios=2)
    assert result.scenario_ids == ("stress_dev_enrollment_surge_05", "stress_dev_enrollment_surge_10")
    assert result.algorithms == ("random", "fcfs", "grade_priority", "constrained")
    assert not (tmp_path / "out").exists()


def test_stress_holdout_requires_explicit_confirmation(tmp_path) -> None:
    with pytest.raises(StressRunnerError, match="requires confirm_holdout"):
        run_stress_robustness_suite(output_dir=tmp_path / "out", split="holdout", dry_run=True)


def test_stress_rejects_cp_sat(tmp_path) -> None:
    with pytest.raises(StressRunnerError, match="four Greedy"):
        run_stress_robustness_suite(output_dir=tmp_path / "out", algorithms=("cp_sat",), dry_run=True)


@pytest.mark.parametrize("field", ["scenario_id", "base_scenario_id", "transform_order"])
def test_stress_manifest_missing_field_fails_closed(field) -> None:
    payload = _payload()
    payload["scenarios"][0].pop(field)
    with pytest.raises(StressManifestError, match="missing"):
        validate_stress_scenario_suite(payload)


def test_stress_manifest_rejects_structural_transform_as_ordinary() -> None:
    payload = _payload()
    payload["scenarios"][0]["transforms"] = [{"type": "global_capacity_deficit"}]
    payload["scenarios"][0]["transform_order"] = ["global_capacity_deficit"]
    with pytest.raises(StressManifestError, match="ordinary stress"):
        validate_stress_scenario_suite(payload)


def test_stress_manifest_does_not_allow_holdout_tuning() -> None:
    payload = _payload()
    payload["scenarios"][-1]["tuning_allowed"] = True
    with pytest.raises(StressManifestError, match="holdout"):
        validate_stress_scenario_suite(payload)
