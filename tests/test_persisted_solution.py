from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

import src.allocation.cp_sat_solver as solver_module
from src.allocation import CpSatSolveStatus, run_fair_cp_sat_solver
from src.allocation.persisted_solution import (
    PersistedSolutionArtifactError,
    load_persisted_solution_seed,
)
from src.experiment_manifest import canonical_input_fingerprint
from tests.test_cp_sat_solver import canonical, request_row


REQUIRED = (
    "benchmark_manifest.json",
    "algorithm_summary.csv",
    "student_outcomes.csv",
    "request_outcomes.csv",
    "final_schedule_policy_summary.csv",
    "final_schedule_policy_violations.csv",
    "artifact_recovery_provenance.json",
)


def _artifact(tmp_path: Path, data, *, status: str = "FEASIBLE", assigned_requests: tuple[str, ...] | None = None) -> Path:
    root = tmp_path / "artifact"
    root.mkdir(parents=True)
    fingerprint = canonical_input_fingerprint(data)
    assigned_requests = assigned_requests or tuple(request.request_key for request in data.logical_requests)
    request_rows = []
    selected_by_student: dict[str, int] = {}
    for request in data.logical_requests:
        assigned = request.request_key in assigned_requests
        group = data.candidate_index[request.request_key][0] if assigned else ""
        request_rows.append(
            {
                "request_key": request.request_key,
                "student_id": request.student_id,
                "request_type": request.request_type,
                "candidate_key": request.candidate_key,
                "status": "assigned" if assigned else "unassigned_all_candidates_rejected",
                "assignment_key": f"{request.student_id}|{request.request_key}|{group}" if assigned else "",
                "assigned_linked_section_group_id": group,
            }
        )
        if assigned:
            selected_by_student[request.student_id] = selected_by_student.get(request.student_id, 0) + 1
    pd.DataFrame(request_rows).to_csv(root / "request_outcomes.csv", index=False)
    pd.DataFrame(
        [
            {
                "student_id": student.student_id,
                "assigned_logical_course_count": selected_by_student.get(student.student_id, 0),
            }
            for student in data.students
        ]
    ).to_csv(root / "student_outcomes.csv", index=False)
    pd.DataFrame(
        [{
            "algorithm_name": "fair_cp_sat_solver_v1_2",
            "status": status,
            "solve_status": status,
            "ordinary_violations": 0,
            "protected_violations": 0,
            "high_demand_violations": 0,
            "section_over_capacity_count": 0,
            "consistency_issue_count": 0,
        }]
    ).to_csv(root / "algorithm_summary.csv", index=False)
    pd.DataFrame(
        [{
            "algorithm_name": "fair_cp_sat_solver_v1_2",
            "final_schedule_policy_pass": True,
            "violating_student_count": 0,
        }]
    ).to_csv(root / "final_schedule_policy_summary.csv", index=False)
    (root / "final_schedule_policy_violations.csv").write_text(
        "algorithm_name,student_id\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": 1,
        "manifest": {
            **{key: value for key, value in fingerprint.__dict__.items()},
            "data_generation_seed": 2026,
            "section_planning_seed": 2026,
            "solver_seed": 20260630,
        },
    }
    (root / "benchmark_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    provenance = {
        "status": status,
        "solve_status": status,
        "algorithm_name": "fair_cp_sat_solver_v1_2",
        "source_git_commit": "source-commit",
        "data_generation_seed": 2026,
        "section_planning_seed": 2026,
        "solver_seed": 20260630,
        "final_schedule_policy_pass": True,
        "violating_student_count": 0,
        "section_over_capacity_count": 0,
        "consistency_issue_count": 0,
        "fingerprint": {**{key: value for key, value in fingerprint.__dict__.items()}, "data_generation_seed": 2026, "section_planning_seed": 2026, "solver_seed": 20260630},
    }
    (root / "artifact_recovery_provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    _refresh_hashes(root)
    return root


def _refresh_hashes(root: Path) -> None:
    lines = []
    for name in sorted(REQUIRED):
        digest = hashlib.sha256((root / name).read_bytes()).hexdigest()
        lines.append(f"{digest}  {name}")
    (root / "SHA256SUMS.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _data(requests: list[tuple]):
    return canonical([("STU_1", 12, len(requests), False)], requests)


@pytest.mark.parametrize("status", ["FEASIBLE", "OPTIMAL"])
def test_valid_persisted_statuses_are_accepted(tmp_path, status) -> None:
    data = _data([request_row("STU_1", "CORE_A")])
    seed = load_persisted_solution_seed(_artifact(tmp_path, data, status=status), data)
    assert seed.source_status == status
    assert seed.source_policy_pass is True
    assert len(seed.selected_assignments) == 1


def test_missing_file_is_rejected(tmp_path) -> None:
    data = _data([request_row("STU_1", "CORE_A")])
    root = _artifact(tmp_path, data)
    (root / "request_outcomes.csv").unlink()
    with pytest.raises(PersistedSolutionArtifactError, match="SHA256 mismatch|missing"):
        load_persisted_solution_seed(root, data)


def test_hash_mismatch_is_rejected_before_csv_use(tmp_path) -> None:
    data = _data([request_row("STU_1", "CORE_A")])
    root = _artifact(tmp_path, data)
    (root / "request_outcomes.csv").write_text("corrupt\n", encoding="utf-8")
    with pytest.raises(PersistedSolutionArtifactError, match="SHA256 mismatch"):
        load_persisted_solution_seed(root, data)


def test_fingerprint_mismatch_is_rejected(tmp_path) -> None:
    data = _data([request_row("STU_1", "CORE_A")])
    root = _artifact(tmp_path, data)
    payload = json.loads((root / "benchmark_manifest.json").read_text())
    payload["manifest"]["students"] = 999
    (root / "benchmark_manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    _refresh_hashes(root)
    with pytest.raises(PersistedSolutionArtifactError, match="fingerprint mismatch"):
        load_persisted_solution_seed(root, data)


def test_student_universe_mismatch_is_rejected(tmp_path) -> None:
    data = _data([request_row("STU_1", "CORE_A")])
    root = _artifact(tmp_path, data)
    frame = pd.read_csv(root / "student_outcomes.csv")
    frame.loc[0, "student_id"] = "OTHER"
    frame.to_csv(root / "student_outcomes.csv", index=False)
    _refresh_hashes(root)
    with pytest.raises(PersistedSolutionArtifactError, match="student universe"):
        load_persisted_solution_seed(root, data)


def test_request_universe_mismatch_is_rejected(tmp_path) -> None:
    data = _data([request_row("STU_1", "CORE_A")])
    root = _artifact(tmp_path, data)
    frame = pd.read_csv(root / "request_outcomes.csv")
    frame.loc[0, "request_key"] = "primary:STU_1:UNKNOWN"
    frame.to_csv(root / "request_outcomes.csv", index=False)
    _refresh_hashes(root)
    with pytest.raises(PersistedSolutionArtifactError, match="logical request universe"):
        load_persisted_solution_seed(root, data)


def test_unknown_section_assignment_is_rejected(tmp_path) -> None:
    data = _data([request_row("STU_1", "CORE_A")])
    root = _artifact(tmp_path, data)
    frame = pd.read_csv(root / "request_outcomes.csv")
    frame.loc[0, "assigned_linked_section_group_id"] = "UNKNOWN_SECTION"
    frame.to_csv(root / "request_outcomes.csv", index=False)
    _refresh_hashes(root)
    with pytest.raises(PersistedSolutionArtifactError, match="not a candidate"):
        load_persisted_solution_seed(root, data)


@pytest.mark.parametrize(
    ("field", "message"),
    [("ordinary_violations", "violations"), ("protected_violations", "violations"), ("high_demand_violations", "violations"), ("section_over_capacity_count", "section_over_capacity"), ("consistency_issue_count", "consistency")],
)
def test_source_policy_capacity_and_consistency_fail_closed(tmp_path, field, message) -> None:
    data = _data([request_row("STU_1", "CORE_A")])
    root = _artifact(tmp_path, data)
    frame = pd.read_csv(root / "algorithm_summary.csv")
    frame.loc[0, field] = 1
    frame.to_csv(root / "algorithm_summary.csv", index=False)
    _refresh_hashes(root)
    with pytest.raises(PersistedSolutionArtifactError, match=message):
        load_persisted_solution_seed(root, data)


def test_unknown_source_status_and_policy_fail(tmp_path) -> None:
    data = _data([request_row("STU_1", "CORE_A")])
    root = _artifact(tmp_path, data, status="UNKNOWN")
    with pytest.raises(PersistedSolutionArtifactError, match="FEASIBLE or OPTIMAL"):
        load_persisted_solution_seed(root, data)
    root = _artifact(tmp_path / "policy", data)
    frame = pd.read_csv(root / "final_schedule_policy_summary.csv")
    frame.loc[0, "final_schedule_policy_pass"] = False
    frame.to_csv(root / "final_schedule_policy_summary.csv", index=False)
    _refresh_hashes(root)
    with pytest.raises(PersistedSolutionArtifactError, match="policy"):
        load_persisted_solution_seed(root, data)


def test_duplicate_assignment_keys_are_rejected(tmp_path) -> None:
    data = _data([request_row("STU_1", "CORE_A"), request_row("STU_1", "CORE_C")])
    root = _artifact(tmp_path, data)
    frame = pd.read_csv(root / "request_outcomes.csv")
    frame.loc[1, "assignment_key"] = frame.loc[0, "assignment_key"]
    frame.to_csv(root / "request_outcomes.csv", index=False)
    _refresh_hashes(root)
    with pytest.raises(PersistedSolutionArtifactError, match="duplicate persisted assignment"):
        load_persisted_solution_seed(root, data)


def test_fake_assignment_on_unassigned_request_is_rejected(tmp_path) -> None:
    data = _data([request_row("STU_1", "CORE_A"), request_row("STU_1", "CORE_C")])
    root = _artifact(tmp_path, data, assigned_requests=("primary:STU_1:CORE_A",))
    frame = pd.read_csv(root / "request_outcomes.csv")
    frame.loc[1, "assigned_linked_section_group_id"] = "CORE_C_1"
    frame.to_csv(root / "request_outcomes.csv", index=False)
    _refresh_hashes(root)
    with pytest.raises(PersistedSolutionArtifactError, match="fake assignment"):
        load_persisted_solution_seed(root, data)


def test_assigned_logical_count_must_match_request_outcomes(tmp_path) -> None:
    data = _data([request_row("STU_1", "CORE_A")])
    root = _artifact(tmp_path, data)
    frame = pd.read_csv(root / "student_outcomes.csv")
    frame.loc[0, "assigned_logical_course_count"] = 0
    frame.to_csv(root / "student_outcomes.csv", index=False)
    _refresh_hashes(root)
    with pytest.raises(PersistedSolutionArtifactError, match="assigned logical count"):
        load_persisted_solution_seed(root, data)


def test_full_model_seed_is_complete_and_exposed_as_hint_metadata(tmp_path) -> None:
    requests = [
        request_row("STU_1", "CORE_A"),
        request_row("STU_1", "CORE_C"),
        request_row("STU_1", "CORE_D"),
        request_row("STU_1", "ALT1"),
        request_row("STU_1", "ALT2"),
    ]
    data = _data(requests)
    root = _artifact(tmp_path, data)
    result = run_fair_cp_sat_solver(
        data,
        seed=20260630,
        max_time_seconds_per_stage=1,
        use_feasibility_bootstrap=False,
        use_constrained_first_hint=False,
        logical_schedule_completion_enabled=False,
        initial_solution_artifact_dir=root,
    )
    assert result.solve_status in {CpSatSolveStatus.FEASIBLE, CpSatSolveStatus.OPTIMAL}
    stats = result.model_stats
    assert stats.initial_solution_seed_enabled is True
    assert stats.initial_solution_seed_role == "full_model_initial_hint"
    assert stats.initial_solution_seed_source_status == "FEASIBLE"
    assert stats.initial_solution_seed_hint_coverage == 1.0
    assert stats.initial_solution_seed_unknown_keys == 0
    assert stats.initial_solution_seed_duplicate_keys == 0
    assert stats.hint_source == "persisted_feasible_seed"
    assert stats.hint_coverage_rate == 1.0


def test_no_artifact_argument_does_not_load_external_seed(monkeypatch) -> None:
    data = _data([request_row("STU_1", "CORE_A")])
    monkeypatch.setattr(solver_module, "load_persisted_solution_seed", lambda *_args: pytest.fail("unexpected artifact read"))
    result = run_fair_cp_sat_solver(
        data,
        seed=1,
        max_time_seconds_per_stage=0.2,
        use_feasibility_bootstrap=False,
        use_constrained_first_hint=False,
        logical_schedule_completion_enabled=False,
        enforce_final_schedule_hard_constraints=False,
    )
    assert result.solve_status in {CpSatSolveStatus.FEASIBLE, CpSatSolveStatus.OPTIMAL}


def test_stage_diagnostics_bind_metrics_to_current_response_and_objective() -> None:
    data = _data([request_row("STU_1", "CORE_A"), request_row("STU_1", "CORE_C")])
    result = run_fair_cp_sat_solver(
        data,
        seed=20260630,
        max_time_seconds_per_stage=1,
        use_feasibility_bootstrap=False,
        use_constrained_first_hint=False,
        enforce_final_schedule_hard_constraints=False,
    )

    solved = tuple(item for item in result.stage_diagnostics if not item.skipped)
    assert solved
    assert all(item.response_proto_hash for item in solved)
    assert all(item.objective_descriptor_hash for item in solved)
    logical = next(
        item
        for item in solved
        if item.stage_name.value == "logical_schedule_completion"
    )
    assert logical.objective_value is not None
    assert logical.best_objective_bound is not None
    assert logical.objective_value <= logical.best_objective_bound
