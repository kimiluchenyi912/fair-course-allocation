from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from src import formal_remaining_k2_batch as batch


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = REPO_ROOT / "data/scenarios/formal_remaining_k2_batch_v1.json"
EXPECTED_PAIR_IDS = [
    "AP_3D_ART_DESIGN_01__INTERMEDIATE_ACTING_01",
    "AP_3D_ART_DESIGN_01__FOOTBALL_01",
    "INTERMEDIATE_ACTING_01__SOCIAL_JUSTICE_01",
    "AP_JAPANESE_LANG_01__INTERMEDIATE_ACTING_01",
    "FOOTBALL_01__SOCIAL_JUSTICE_01",
    "CREATIVE_WRITING_01__INTERMEDIATE_ACTING_01",
    "AP_JAPANESE_LANG_01__FOOTBALL_01",
    "CREATIVE_WRITING_01__FOOTBALL_01",
    "AP_3D_ART_DESIGN_01__CHINESE4_01",
    "CHINESE4_01__SOCIAL_JUSTICE_01",
    "AP_JAPANESE_LANG_01__CHINESE4_01",
    "CHINESE4_01__CREATIVE_WRITING_01",
]


def load_inputs() -> tuple[dict[str, object], str, dict[str, object], list[dict[str, object]]]:
    manifest, manifest_hash = batch.load_manifest(MANIFEST)
    source = batch.verify_formal_input(manifest)
    runs = batch.planned_runs(source, manifest)
    return manifest, manifest_hash, source, runs


def fake_result(
    run: dict[str, object],
    manifest: dict[str, object],
    *,
    status: str = "INFEASIBLE",
    incumbent: bool = False,
    full_model_bytes: bytes | None = None,
    save_reason: str | None = None,
) -> dict[str, object]:
    response_hash = f"response-{run['formal_order']}-{status}"
    result: dict[str, object] = {
        "status": status,
        "incumbent_found": incumbent,
        "assignment_available": incumbent,
        "response_hash": response_hash,
        "response_hash_verified": True,
        "response_stats": {"status": status, "response_hash": response_hash},
        "solver_log": (f"status: {status}\n",),
        "solver_config": batch.frozen_run_config(run, manifest),
        "destination_domains": {
            str(run["section_id_a"]): [
                [f"P{index + 1}"] for index in range(int(run["destination_domain_size_a"]))
            ],
            str(run["section_id_b"]): [
                [f"P{index + 1}"] for index in range(int(run["destination_domain_size_b"]))
            ],
        },
        "model_fingerprint": {"sha256": f"model-{run['formal_order']}"},
        "validation": {"validated": incumbent},
    }
    if full_model_bytes is not None:
        result["full_model_bytes"] = full_model_bytes
        result["full_model_save_reason"] = save_reason
    return result


def copy_source_and_manifest(tmp_path: Path) -> tuple[Path, Path]:
    manifest, _ = batch.load_manifest(MANIFEST)
    source = tmp_path / "source"
    shutil.copytree(Path(manifest["formal_input_artifact"]), source)
    manifest["formal_input_artifact"] = str(source)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return source, manifest_path


def test_formal_input_artifact_identity_and_hashes() -> None:
    manifest, _, source, _ = load_inputs()
    assert source["verification"] == {
        "passed": True,
        "file_count": 11,
        "checksum_entry_count": 10,
        "sha256sums_sha256": batch.SOURCE_SHA256SUMS_HASH,
    }
    assert source["result_hash"] == batch.RESULT_HASH
    assert source["proof_hash"] == batch.PROOF_HASH
    assert Path(manifest["formal_input_artifact"]).name == "all-student-k2-blocker-safe-screen-v1"


def test_exact_twelve_pair_order_is_authoritative() -> None:
    _, _, source, runs = load_inputs()
    assert source["pair_count"] == 12
    assert [run["pair_id"] for run in runs] == EXPECTED_PAIR_IDS


def test_ordering_hash_guard_matches_formal_artifact() -> None:
    _, _, source, _ = load_inputs()
    assert source["ordering_hash"] == batch.ORDERING_HASH


@pytest.mark.parametrize("field", ["result_hash", "source_proof_hash"])
def test_result_and_proof_hash_guards_fail_closed(tmp_path: Path, field: str) -> None:
    _, manifest_path = copy_source_and_manifest(tmp_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload[field] = "0" * 64
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(batch.FormalK2BatchError, match="frozen manifest drift"):
        batch.load_manifest(manifest_path)


def test_source_checksum_hash_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    source, manifest_path = copy_source_and_manifest(tmp_path)
    monkeypatch.setattr(batch, "DEFAULT_SOURCE_ARTIFACT", source)
    provenance = json.loads((source / "provenance.json").read_text(encoding="utf-8"))
    provenance["drift"] = True
    (source / "provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    batch.write_checksums(source)
    manifest, _ = batch.load_manifest(manifest_path)
    with pytest.raises(batch.FormalK2BatchError, match="identity drift"):
        batch.verify_formal_input(manifest)


def test_pair_order_drift_fails_closed_after_identity_check(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    source, manifest_path = copy_source_and_manifest(tmp_path)
    monkeypatch.setattr(batch, "DEFAULT_SOURCE_ARTIFACT", source)
    rows = batch.read_csv(source / "blocker_screen_survivors.csv")
    rows[0], rows[1] = rows[1], rows[0]
    batch.write_csv(source / "blocker_screen_survivors.csv", rows)
    monkeypatch.setattr(
        batch,
        "verify_checksums",
        lambda root: {
            "passed": True,
            "file_count": 11,
            "checksum_entry_count": 10,
            "sha256sums_sha256": batch.SOURCE_SHA256SUMS_HASH,
        },
    )
    manifest, _ = batch.load_manifest(manifest_path)
    with pytest.raises(batch.FormalK2BatchError, match="count/order drift"):
        batch.verify_formal_input(manifest)


def test_deterministic_three_by_four_batch_partition() -> None:
    _, _, _, runs = load_inputs()
    assert [sum(run["batch_index"] == index for run in runs) for index in (1, 2, 3)] == [4, 4, 4]
    assert [run["batch_position"] for run in runs] == [1, 2, 3, 4] * 3


def test_manifest_freezes_run_a_protocol_and_budget() -> None:
    manifest, _, _, _ = load_inputs()
    assert manifest["global_invocation_budget"] == 12
    assert manifest["run_b_allowed"] is False
    assert manifest["no_automatic_rerun"] is True
    assert manifest["hint"] is False
    assert manifest["objective"] is False
    assert manifest["candidate_pruning"] is False
    assert manifest["full_model_proto_default"] is False


def test_dry_run_never_calls_solver(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(batch, "_real_solver_runner", lambda *args: pytest.fail("solver called"))
    result = batch.run_batch(manifest_path=MANIFEST, output_dir=tmp_path / "dry", dry_run=True)
    assert result["solver_invocations"] == 0
    assert result["new_feasibility_results"] == 0


def test_dry_run_writes_plan_without_solver_artifacts(tmp_path: Path) -> None:
    output = tmp_path / "dry"
    result = batch.run_batch(manifest_path=MANIFEST, output_dir=output, dry_run=True)
    assert result["planned_pair_count"] == 12
    assert result["batch_count"] == 3
    files = {path.name for path in output.rglob("*") if path.is_file()}
    assert {
        "planned_pairs.csv",
        "checkpoint.json",
        "aggregate_summary.json",
        "compact_evidence_plan.json",
        "SHA256SUMS.txt",
    } <= files
    assert "solver.log" not in files
    assert "model.pb" not in files
    assert not (output / "runs").exists()


def test_dry_run_refuses_formal_artifact_path() -> None:
    with pytest.raises(batch.FormalK2BatchError, match="dry-run may not write"):
        batch.run_batch(manifest_path=MANIFEST, output_dir=batch.FORMAL_OUTPUT, dry_run=True)


def test_explicit_mode_is_required(tmp_path: Path) -> None:
    with pytest.raises(batch.FormalK2BatchError, match="exactly one explicit mode"):
        batch.run_batch(manifest_path=MANIFEST, output_dir=tmp_path)


def test_max_new_solver_runs_counts_only_new_calls(tmp_path: Path) -> None:
    calls: list[str] = []

    def fake(run: dict[str, object], manifest: dict[str, object]) -> dict[str, object]:
        calls.append(str(run["pair_id"]))
        return fake_result(run, manifest)

    output = tmp_path / "run"
    first = batch.run_batch(
        manifest_path=MANIFEST,
        output_dir=output,
        execute=True,
        max_new_solver_runs=2,
        solver_runner=fake,
    )
    second = batch.run_batch(
        manifest_path=MANIFEST,
        output_dir=output,
        execute=True,
        resume=True,
        max_new_solver_runs=1,
        solver_runner=fake,
    )
    assert first["new_solver_invocations_this_call"] == 2
    assert second["new_solver_invocations_this_call"] == 1
    assert second["solver_invocations"] == 3
    assert calls == EXPECTED_PAIR_IDS[:3]


def test_completed_pair_and_batch_are_not_repeated(tmp_path: Path) -> None:
    calls: list[str] = []

    def fake(run: dict[str, object], manifest: dict[str, object]) -> dict[str, object]:
        calls.append(str(run["pair_id"]))
        return fake_result(run, manifest)

    output = tmp_path / "run"
    batch.run_batch(
        manifest_path=MANIFEST,
        output_dir=output,
        execute=True,
        max_new_solver_runs=4,
        solver_runner=fake,
    )
    checkpoint = batch.read_json(output / "checkpoint.json")
    assert checkpoint["completed_batch_indices"] == [1]
    batch.run_batch(
        manifest_path=MANIFEST,
        output_dir=output,
        execute=True,
        resume=True,
        max_new_solver_runs=1,
        solver_runner=fake,
    )
    assert calls == EXPECTED_PAIR_IDS[:5]


@pytest.mark.parametrize("state", ["running", "artifact_failure"])
def test_interrupted_or_artifact_failure_checkpoint_never_reruns(tmp_path: Path, state: str) -> None:
    output = tmp_path / "dry"
    batch.run_batch(manifest_path=MANIFEST, output_dir=output, dry_run=True)
    checkpoint = batch.read_json(output / "checkpoint.json")
    checkpoint["pair_states"][0]["state"] = state
    if state == "artifact_failure":
        checkpoint["pair_states"][0]["result_classification"] = "artifact_failure"
    batch.write_json(output / "checkpoint.json", checkpoint)
    batch.write_checksums(output)
    with pytest.raises(batch.FormalK2BatchError, match="interrupted run|artifact failure"):
        batch.run_batch(
            manifest_path=MANIFEST,
            output_dir=output,
            execute=True,
            resume=True,
            solver_runner=lambda *args: pytest.fail("must not rerun"),
        )


def test_manifest_hash_drift_stops_resume(tmp_path: Path) -> None:
    manifest_copy = tmp_path / "manifest.json"
    manifest_copy.write_text(MANIFEST.read_text(encoding="utf-8"), encoding="utf-8")
    output = tmp_path / "dry"
    batch.run_batch(manifest_path=manifest_copy, output_dir=output, dry_run=True)
    payload = json.loads(manifest_copy.read_text(encoding="utf-8"))
    payload["reporting_note"] = "hash drift"
    manifest_copy.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(batch.FormalK2BatchError, match="manifest_hash"):
        batch.run_batch(manifest_path=manifest_copy, output_dir=output, dry_run=True, resume=True)


@pytest.mark.parametrize(
    ("status", "incumbent", "classification"),
    [
        ("INFEASIBLE", False, "fixed_pair_infeasible"),
        ("UNKNOWN", False, "unresolved_unknown_no_incumbent"),
        ("FEASIBLE", True, "incumbent_pending_validation"),
        ("MODEL_INVALID", False, "model_invalid"),
    ],
)
def test_status_stopping_rules(
    tmp_path: Path, status: str, incumbent: bool, classification: str
) -> None:
    calls = 0

    def fake(run: dict[str, object], manifest: dict[str, object]) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return fake_result(run, manifest, status=status, incumbent=incumbent)

    result = batch.run_batch(
        manifest_path=MANIFEST,
        output_dir=tmp_path / status,
        execute=True,
        max_new_solver_runs=12,
        solver_runner=fake,
    )
    expected_calls = 12 if status == "INFEASIBLE" else 1
    assert calls == expected_calls
    assert result["pair_result_counts"][classification] == expected_calls
    assert result["global_k2_status"] == "unresolved"
    assert result["exact_minimum_claim"] is None


def test_global_invocation_budget_is_twelve(tmp_path: Path) -> None:
    result = batch.run_batch(
        manifest_path=MANIFEST,
        output_dir=tmp_path / "all",
        execute=True,
        solver_runner=lambda run, manifest: fake_result(run, manifest),
    )
    assert result["solver_invocations"] == 12
    assert result["stop_reason"] == "all_pairs_completed"


def test_compact_evidence_omits_model_by_default(tmp_path: Path) -> None:
    output = tmp_path / "run"
    batch.run_batch(
        manifest_path=MANIFEST,
        output_dir=output,
        execute=True,
        max_new_solver_runs=1,
        solver_runner=lambda run, manifest: fake_result(run, manifest),
    )
    run_dir = output / "runs" / EXPECTED_PAIR_IDS[0]
    assert not (run_dir / "model.pb").exists()
    evidence = batch.read_json(run_dir / "compact_evidence.json")
    assert evidence["full_model_saved"] is False
    assert evidence["source_hashes"]["sha256sums"] == batch.SOURCE_SHA256SUMS_HASH
    assert evidence["manifest_hash"] == batch.sha256_file(MANIFEST)
    assert (run_dir / "provenance.json").is_file()


def test_model_proto_is_saved_only_for_allowed_anomaly(tmp_path: Path) -> None:
    output = tmp_path / "run"
    batch.run_batch(
        manifest_path=MANIFEST,
        output_dir=output,
        execute=True,
        max_new_solver_runs=1,
        solver_runner=lambda run, manifest: fake_result(
            run,
            manifest,
            status="MODEL_INVALID",
            full_model_bytes=b"debug-model",
            save_reason="model_invalid",
        ),
    )
    run_dir = output / "runs" / EXPECTED_PAIR_IDS[0]
    assert (run_dir / "model.pb").read_bytes() == b"debug-model"
    evidence = batch.read_json(run_dir / "compact_evidence.json")
    assert evidence["full_model_saved"] is True
    assert evidence["full_model_save_reason"] == "model_invalid"


def test_unapproved_model_proto_reason_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(batch.FormalK2BatchError, match="run failed closed"):
        batch.run_batch(
            manifest_path=MANIFEST,
            output_dir=tmp_path / "run",
            execute=True,
            max_new_solver_runs=1,
            solver_runner=lambda run, manifest: fake_result(
                run,
                manifest,
                full_model_bytes=b"model",
                save_reason="ordinary_run",
            ),
        )


def test_response_fingerprint_drift_stops_resume(tmp_path: Path) -> None:
    output = tmp_path / "run"
    batch.run_batch(
        manifest_path=MANIFEST,
        output_dir=output,
        execute=True,
        max_new_solver_runs=1,
        solver_runner=lambda run, manifest: fake_result(run, manifest),
    )
    result_path = output / "runs" / EXPECTED_PAIR_IDS[0] / "run_result.json"
    result = batch.read_json(result_path)
    result["response_hash"] = "drift"
    batch.write_json(result_path, result)
    batch.write_checksums(output)
    with pytest.raises(batch.FormalK2BatchError, match="response fingerprint drift"):
        batch.run_batch(
            manifest_path=MANIFEST,
            output_dir=output,
            execute=True,
            resume=True,
            solver_runner=lambda *args: pytest.fail("must not rerun"),
        )


def test_solver_config_fingerprint_drift_stops_resume(tmp_path: Path) -> None:
    output = tmp_path / "run"
    batch.run_batch(
        manifest_path=MANIFEST,
        output_dir=output,
        execute=True,
        max_new_solver_runs=1,
        solver_runner=lambda run, manifest: fake_result(run, manifest),
    )
    config_path = output / "runs" / EXPECTED_PAIR_IDS[0] / "solver_config.json"
    config = batch.read_json(config_path)
    config["workers"] = 2
    batch.write_json(config_path, config)
    batch.write_checksums(output)
    with pytest.raises(batch.FormalK2BatchError, match="solver config fingerprint drift"):
        batch.run_batch(
            manifest_path=MANIFEST,
            output_dir=output,
            execute=True,
            resume=True,
            solver_runner=lambda *args: pytest.fail("must not rerun"),
        )


def test_artifact_write_failure_records_fail_closed_state(tmp_path: Path) -> None:
    output = tmp_path / "run"
    calls = 0

    def incomplete(run: dict[str, object], manifest: dict[str, object]) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {"status": "INFEASIBLE"}

    with pytest.raises(batch.FormalK2BatchError, match="run failed closed"):
        batch.run_batch(
            manifest_path=MANIFEST,
            output_dir=output,
            execute=True,
            max_new_solver_runs=4,
            solver_runner=incomplete,
        )
    assert calls == 1
    checkpoint = batch.read_json(output / "checkpoint.json")
    assert checkpoint["pair_states"][0]["state"] == "artifact_failure"
    assert checkpoint["total_solver_invocations"] == 1


def test_no_run_b_surface_exists() -> None:
    manifest, _, _, _ = load_inputs()
    assert manifest["run_b_allowed"] is False
    assert "guided" not in Path(batch.__file__).read_text(encoding="utf-8")


def test_partial_batch_keeps_global_claim_boundary(tmp_path: Path) -> None:
    output = tmp_path / "run"
    result = batch.run_batch(
        manifest_path=MANIFEST,
        output_dir=output,
        execute=True,
        max_new_solver_runs=2,
        solver_runner=lambda run, manifest: fake_result(run, manifest),
    )
    assert result["global_k2_status"] == "unresolved"
    assert result["proven_lower_bound"] == 2
    assert result["exact_minimum_claim"] is None
    assert result["pair_result_counts"]["planned_not_run"] == 10


def test_all_solver_entrypoints_can_be_forbidden_during_dry_run(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("real solver entrypoint called")

    monkeypatch.setattr(batch, "_real_solver_runner", forbidden)
    result = batch.run_batch(
        manifest_path=MANIFEST,
        output_dir=tmp_path / "dry",
        dry_run=True,
        solver_runner=forbidden,
    )
    assert result["solver_invocations"] == 0
