from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import formal_remaining_k2_batch as formal
from src import supplemental_k2_unresolved_pairs as supplemental


MANIFEST = Path("data/scenarios/supplemental_k2_unresolved_pairs_v1.json")
EXPECTED_PAIR_IDS = [pair["pair_id"] for pair in supplemental.EXPECTED_PAIRS]


def load_inputs() -> tuple[dict[str, object], str, dict[str, object], list[dict[str, object]]]:
    manifest, manifest_hash = supplemental.load_manifest(MANIFEST)
    source = supplemental.verify_source_formal_artifact(manifest)
    runs = supplemental.planned_runs(source, manifest)
    return manifest, manifest_hash, source, runs


def approved_manifest(tmp_path: Path) -> Path:
    payload = json.loads(MANIFEST.read_text(encoding="utf-8"))
    payload["extended_budget_policy"] = {
        "approval_reference": "test-only-fake-solver-approval",
        "approval_status": "approved",
        "approved": True,
        "per_pair_time_limit_seconds": 1,
    }
    path = tmp_path / "approved-manifest.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def fake_result(
    run: dict[str, object],
    manifest: dict[str, object],
    *,
    status: str = "INFEASIBLE",
    incumbent: bool = False,
) -> dict[str, object]:
    raw_hash = formal.json_hash(
        {
            "fake_raw_solver_response": True,
            "pair_id": run["pair_id"],
            "status": status,
            "incumbent": incumbent,
        }
    )
    return {
        "status": status,
        "incumbent_found": incumbent,
        "assignment_available": incumbent,
        "response_hash": raw_hash,
        "response_hash_verified": True,
        "response_stats": {
            "status": status,
            "incumbent_found": incumbent,
            "assignment_available": incumbent,
            "response_hash": raw_hash,
        },
        "solver_log": [f"fake status: {status}\n"],
        "solver_config": supplemental.supplemental_run_config(run, manifest),
        "destination_domains": run["frozen_non_original_destination_domains"],
        "model_fingerprint": {
            "sha256": formal.json_hash({"pair": run["pair_id"]}),
            "binary_proto_bytes": 1,
            "total_variables": 1,
            "total_constraints": 1,
            "full_model_persisted": False,
        },
        "validation": {"validated": False, "reason": "fake result"},
        "hint_audit": {
            "hint_used": False,
            "assignment_hint_used": False,
            "objective_used": False,
            "candidate_pruning": False,
            "full_domain_preserved": True,
        },
    }


def test_source_formal_artifact_identity_and_hashes() -> None:
    manifest, _, source, _ = load_inputs()
    assert manifest["source_formal_artifact_sha256sums_hash"] == supplemental.SOURCE_SHA256SUMS_HASH
    assert source["verification"] == {
        "passed": True,
        "file_count": 22,
        "checksum_entry_count": 21,
        "sha256sums_sha256": supplemental.SOURCE_SHA256SUMS_HASH,
    }
    assert source["source_manifest_hash"] == supplemental.SOURCE_MANIFEST_HASH
    assert source["source_ordering_hash"] == supplemental.SOURCE_ORDERING_HASH


def test_exact_unresolved_pair_set_is_orders_one_and_two() -> None:
    manifest, _, source, runs = load_inputs()
    assert manifest["supplemental_pairs"] == list(supplemental.EXPECTED_PAIRS)
    assert [run["pair_id"] for run in runs] == EXPECTED_PAIR_IDS
    assert [run["original_formal_order"] for run in runs] == [1, 2]
    assert len(source["pairs"]) == 2


def test_original_classifications_are_preserved() -> None:
    _, _, source, runs = load_inputs()
    assert [run["original_classification"] for run in runs] == [
        "artifact_failure",
        "unresolved_unknown_no_incumbent",
    ]
    assert source["original_run_preserved"] is True
    assert source["original_result_not_overwritten"] is True
    assert source["original_invocation_not_reclassified"] is True


def test_orders_three_through_twelve_are_not_supplemental() -> None:
    _, _, source, runs = load_inputs()
    assert source["not_yet_run_formal_orders"] == list(range(3, 13))
    assert not ({run["original_formal_order"] for run in runs} & set(range(3, 13)))


def test_manifest_freezes_solver_semantics_and_disables_run_b() -> None:
    manifest, _, _, _ = load_inputs()
    assert manifest["hint"] is False
    assert manifest["assignment_hint"] is False
    assert manifest["objective"] is False
    assert manifest["candidate_pruning"] is False
    assert manifest["run_b_allowed"] is False
    assert manifest["no_automatic_rerun"] is True
    assert manifest["seed"] == 20260630
    assert manifest["workers"] == 1


def test_exact_frozen_destination_domains_are_bound() -> None:
    manifest, _, _, runs = load_inputs()
    assert [run["frozen_non_original_destination_domains"] for run in runs] == [
        pair["frozen_non_original_destination_domains"]
        for pair in manifest["supplemental_pairs"]
    ]
    assert runs[0]["frozen_non_original_destination_domains"]["INTERMEDIATE_ACTING_01"] == [
        ["P3"],
        ["P7"],
    ]


def test_budget_policy_is_pending_explicit_approval() -> None:
    manifest, _, _, runs = load_inputs()
    assert manifest["extended_budget_policy"] == {
        "approval_reference": None,
        "approval_status": "pending_explicit_approval",
        "approved": False,
        "per_pair_time_limit_seconds": None,
    }
    assert all(
        supplemental.supplemental_run_config(run, manifest)["time_limit_seconds"] is None
        for run in runs
    )


def test_dry_run_has_zero_solver_invocations(tmp_path: Path) -> None:
    calls = 0

    def forbidden(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        raise AssertionError("solver called")

    result = supplemental.run_experiment(
        manifest_path=MANIFEST,
        output_dir=tmp_path / "dry",
        dry_run=True,
        solver_runner=forbidden,
    )
    assert calls == 0
    assert result["solver_invocations"] == 0
    assert result["supplemental_pair_count"] == 2


def test_dry_run_writes_no_solver_or_response_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "dry"
    supplemental.run_experiment(manifest_path=MANIFEST, output_dir=output, dry_run=True)
    files = [path.name for path in output.rglob("*") if path.is_file()]
    assert "solver.log" not in files
    assert "response_stats.json" not in files
    assert "model.pb" not in files
    assert not (output / "runs").exists()


def test_explicit_mode_is_required(tmp_path: Path) -> None:
    with pytest.raises(supplemental.SupplementalK2Error, match="exactly one explicit mode"):
        supplemental.run_experiment(manifest_path=MANIFEST, output_dir=tmp_path / "x")


def test_execute_fails_closed_while_budget_is_unapproved(tmp_path: Path) -> None:
    output = tmp_path / "execute"
    with pytest.raises(supplemental.SupplementalK2Error, match="not explicitly approved"):
        supplemental.run_experiment(
            manifest_path=MANIFEST,
            output_dir=output,
            execute=True,
            solver_runner=lambda *args: pytest.fail("solver must not run"),
        )
    assert not output.exists()


def test_dry_run_does_not_mutate_formal_artifact(tmp_path: Path) -> None:
    source = supplemental.DEFAULT_SOURCE_ARTIFACT
    before = formal.sha256_file(source / "SHA256SUMS.txt")
    supplemental.run_experiment(
        manifest_path=MANIFEST, output_dir=tmp_path / "dry", dry_run=True
    )
    assert formal.sha256_file(source / "SHA256SUMS.txt") == before
    assert formal.verify_checksums(source)["passed"] is True


def test_source_formal_artifact_cannot_be_output() -> None:
    with pytest.raises(supplemental.SupplementalK2Error, match="must be separate"):
        supplemental.run_experiment(
            manifest_path=MANIFEST,
            output_dir=supplemental.DEFAULT_SOURCE_ARTIFACT,
            dry_run=True,
        )


def test_dry_run_provenance_is_independent_and_preserves_original(tmp_path: Path) -> None:
    output = tmp_path / "dry"
    supplemental.run_experiment(manifest_path=MANIFEST, output_dir=output, dry_run=True)
    provenance = formal.read_json(output / "provenance.json")
    assert provenance["supplemental_experiment"] is True
    assert provenance["source_formal_artifact_read_only"] is True
    assert provenance["original_result_not_overwritten"] is True
    assert provenance["response_hash_namespace"] == supplemental.RESPONSE_HASH_NAMESPACE
    assert provenance["finalization_is_separate"] is True


def test_independent_response_hash_namespace(tmp_path: Path) -> None:
    output = tmp_path / "run"
    manifest_path = approved_manifest(tmp_path)
    supplemental.run_experiment(
        manifest_path=manifest_path,
        output_dir=output,
        execute=True,
        max_new_solver_runs=1,
        solver_runner=lambda run, manifest: fake_result(run, manifest),
    )
    run_dir = output / "runs" / EXPECTED_PAIR_IDS[0]
    result = formal.read_json(run_dir / "run_result.json")
    stats = formal.read_json(run_dir / "response_stats.json")
    assert result["response_hash_namespace"] == supplemental.RESPONSE_HASH_NAMESPACE
    assert result["response_hash"] != result["source_response_hash"]
    assert stats["source_response_hash_verified"] is True
    assert stats["response_hash"] == result["response_hash"]


def test_supplemental_result_never_overwrites_original_result(tmp_path: Path) -> None:
    output = tmp_path / "run"
    manifest_path = approved_manifest(tmp_path)
    supplemental.run_experiment(
        manifest_path=manifest_path,
        output_dir=output,
        execute=True,
        max_new_solver_runs=1,
        solver_runner=lambda run, manifest: fake_result(run, manifest),
    )
    result = formal.read_json(output / "runs" / EXPECTED_PAIR_IDS[0] / "run_result.json")
    assert result["original_classification"] == "artifact_failure"
    assert result["original_result_not_overwritten"] is True
    assert result["result_classification"] == "supplemental_fixed_pair_infeasible"


def test_resume_deduplicates_completed_infeasible_pair(tmp_path: Path) -> None:
    output = tmp_path / "run"
    manifest_path = approved_manifest(tmp_path)
    calls: list[str] = []

    def fake(run: dict[str, object], manifest: dict[str, object]) -> dict[str, object]:
        calls.append(str(run["pair_id"]))
        return fake_result(run, manifest)

    supplemental.run_experiment(
        manifest_path=manifest_path,
        output_dir=output,
        execute=True,
        max_new_solver_runs=1,
        solver_runner=fake,
    )
    supplemental.run_experiment(
        manifest_path=manifest_path,
        output_dir=output,
        execute=True,
        resume=True,
        max_new_solver_runs=1,
        solver_runner=fake,
    )
    assert calls == EXPECTED_PAIR_IDS


def test_artifact_failure_fails_closed_and_cannot_rerun(tmp_path: Path) -> None:
    output = tmp_path / "run"
    manifest_path = approved_manifest(tmp_path)
    with pytest.raises(supplemental.SupplementalK2Error, match="failed closed"):
        supplemental.run_experiment(
            manifest_path=manifest_path,
            output_dir=output,
            execute=True,
            solver_runner=lambda *args: {"status": "INFEASIBLE"},
        )
    checkpoint = formal.read_json(output / "checkpoint.json")
    assert checkpoint["pair_states"][0]["state"] == "artifact_failure"
    with pytest.raises(supplemental.SupplementalK2Error, match="automatic rerun"):
        supplemental.run_experiment(
            manifest_path=manifest_path,
            output_dir=output,
            execute=True,
            resume=True,
            solver_runner=lambda *args: pytest.fail("must not rerun"),
        )


def test_unknown_without_incumbent_stops_and_remains_unresolved(tmp_path: Path) -> None:
    output = tmp_path / "run"
    manifest_path = approved_manifest(tmp_path)
    calls = 0

    def fake(run: dict[str, object], manifest: dict[str, object]) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return fake_result(run, manifest, status="UNKNOWN")

    result = supplemental.run_experiment(
        manifest_path=manifest_path,
        output_dir=output,
        execute=True,
        solver_runner=fake,
    )
    assert calls == 1
    assert result["stop_reason"] == "supplemental_unresolved_unknown_no_incumbent"
    assert result["global_k2_status"] == "unresolved"


def test_unknown_terminal_result_is_not_rerun_on_resume(tmp_path: Path) -> None:
    output = tmp_path / "run"
    manifest_path = approved_manifest(tmp_path)
    supplemental.run_experiment(
        manifest_path=manifest_path,
        output_dir=output,
        execute=True,
        solver_runner=lambda run, manifest: fake_result(run, manifest, status="UNKNOWN"),
    )
    result = supplemental.run_experiment(
        manifest_path=manifest_path,
        output_dir=output,
        execute=True,
        resume=True,
        solver_runner=lambda *args: pytest.fail("terminal result must not rerun"),
    )
    assert result["new_solver_invocations_this_call"] == 0


def test_incumbent_stops_without_finalization(tmp_path: Path) -> None:
    manifest_path = approved_manifest(tmp_path)
    result = supplemental.run_experiment(
        manifest_path=manifest_path,
        output_dir=tmp_path / "run",
        execute=True,
        solver_runner=lambda run, manifest: fake_result(
            run, manifest, status="FEASIBLE", incumbent=True
        ),
    )
    assert result["solver_invocations"] == 1
    assert result["stop_reason"] == "supplemental_incumbent_pending_validation"
    assert result["finalization_performed"] is False


def test_supplemental_infeasible_does_not_mutate_global_k2(tmp_path: Path) -> None:
    manifest_path = approved_manifest(tmp_path)
    result = supplemental.run_experiment(
        manifest_path=manifest_path,
        output_dir=tmp_path / "run",
        execute=True,
        solver_runner=lambda run, manifest: fake_result(run, manifest),
    )
    assert result["pair_result_counts"]["supplemental_fixed_pair_infeasible"] == 2
    assert result["global_k2_status"] == "unresolved"
    assert result["proven_lower_bound"] == 2
    assert result["exact_minimum_claim"] is None
    assert result["supplemental_evidence_applied_to_global_proof"] is False


def test_finalization_is_always_separate(tmp_path: Path) -> None:
    result = supplemental.run_experiment(
        manifest_path=MANIFEST, output_dir=tmp_path / "dry", dry_run=True
    )
    assert result["additional_independent_evidence_only"] is True
    assert result["finalization_performed"] is False
    assert result["supplemental_evidence_applied_to_global_proof"] is False


def test_source_hash_drift_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    manifest, _ = supplemental.load_manifest(MANIFEST)
    monkeypatch.setattr(
        formal,
        "verify_checksums",
        lambda path: {
            "passed": True,
            "file_count": 22,
            "checksum_entry_count": 21,
            "sha256sums_sha256": "0" * 64,
        },
    )
    with pytest.raises(supplemental.SupplementalK2Error, match="identity drift"):
        supplemental.verify_source_formal_artifact(manifest)


def test_all_real_solver_entrypoints_are_forbidden_during_dry_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("real solver entrypoint called")

    monkeypatch.setattr(supplemental, "_real_solver_runner", forbidden)
    monkeypatch.setattr(formal, "_real_solver_runner", forbidden)
    result = supplemental.run_experiment(
        manifest_path=MANIFEST,
        output_dir=tmp_path / "dry",
        dry_run=True,
        solver_runner=forbidden,
    )
    assert result["solver_invocations"] == 0


def test_formal_supplemental_artifact_is_not_created_by_dry_run(tmp_path: Path) -> None:
    output = tmp_path / "dry"
    supplemental.run_experiment(manifest_path=MANIFEST, output_dir=output, dry_run=True)
    assert output.is_dir()
    assert output.resolve() != supplemental.SUPPLEMENTAL_OUTPUT.resolve()


def test_current_repo_manifest_cannot_execute_even_with_fake_solver(tmp_path: Path) -> None:
    calls = 0

    def fake(*args: object, **kwargs: object) -> object:
        nonlocal calls
        calls += 1
        return {}

    with pytest.raises(supplemental.SupplementalK2Error, match="not explicitly approved"):
        supplemental.run_experiment(
            manifest_path=MANIFEST,
            output_dir=tmp_path / "run",
            execute=True,
            max_new_solver_runs=2,
            solver_runner=fake,
        )
    assert calls == 0
