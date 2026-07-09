from __future__ import annotations

import hashlib
import json
from dataclasses import FrozenInstanceError

import pandas as pd
import pytest

from src.allocation import canonicalize_allocation_input
from src.experiment_manifest import (
    STABLE_YEAR_BENCHMARK_SEEDS,
    ExperimentManifestError,
    ExperimentSeeds,
    build_experiment_manifest,
    canonical_input_fingerprint,
    load_experiment_manifest,
    verify_experiment_manifest,
    write_experiment_manifest,
)


def _catalog() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("NORMAL", 1, "standard"),
            ("GOV_ECON_REG", 1, "semester_block"),
            ("ALT", 1, "standard"),
        ],
        columns=["course_id", "periods_required", "schedule_structure"],
    )


def _students() -> pd.DataFrame:
    return pd.DataFrame(
        [("STU_1", 12, 2, "afternoon", "false", "", "")],
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


def _requests() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("STU_1", "NORMAL", "primary", "", "", ""),
            ("STU_1", "GOV_ECON_REG", "primary", "", "gov_econ_block", "GOV_ECON_REG"),
            ("STU_1", "GOV_ECON_REG", "primary", "", "gov_econ_block", "GOV_ECON_REG"),
            ("STU_1", "ALT", "alternate", 1, "alternate", ""),
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


def _sections() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("SEC_NORMAL", "NORMAL", "P1", "", "full_year", 40, "", "NORMAL_1", "NORMAL", ""),
            ("SEC_GOV_S1", "GOV_ECON_REG", "P2", "", "semester_1", 40, "GOV_1", "GOV_1", "GOV_ECON_REG", "Government"),
            ("SEC_GOV_S2", "GOV_ECON_REG", "P2", "", "semester_2", 40, "GOV_1", "GOV_1", "GOV_ECON_REG", "Economics"),
            ("SEC_ALT", "ALT", "P3", "", "full_year", 40, "", "ALT_1", "ALT", ""),
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


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_inputs(tmp_path, *, data_seed: int = 2026, section_seed: int = 2026):
    generated = tmp_path / "generated"
    planned = tmp_path / "sections"
    config = tmp_path / "config"
    generated.mkdir()
    planned.mkdir()
    config.mkdir()
    _students().to_csv(generated / "students.csv", index=False)
    _requests().to_csv(generated / "requests.csv", index=False)
    _sections().to_csv(planned / "sections.csv", index=False)
    _catalog().to_csv(config / "course_catalog.csv", index=False)
    (config / "other.csv").write_text("key,value\nexample,1\n", encoding="utf-8")
    (generated / "generation_metadata.json").write_text(
        json.dumps(
            {
                "scenario_id": "stable_year",
                "seed": data_seed,
                "total_students": 1,
                "primary_request_rows": 3,
                "alternate_request_rows": 1,
                "output_file_hashes": {
                    "students.csv": _sha256(generated / "students.csv"),
                    "requests.csv": _sha256(generated / "requests.csv"),
                },
            }
        ),
        encoding="utf-8",
    )
    (planned / "section_planning_metadata.json").write_text(
        json.dumps(
            {
                "scenario_id": "stable_year",
                "seed": section_seed,
                "student_count": 1,
                "primary_request_rows": 3,
                "total_section_rows": 4,
                "total_logical_sections": 3,
                "total_primary_demand": 2,
                "input_file_hashes": {
                    "students.csv": _sha256(generated / "students.csv"),
                    "requests.csv": _sha256(generated / "requests.csv"),
                },
                "output_file_hashes": {
                    "sections.csv": _sha256(planned / "sections.csv"),
                },
            }
        ),
        encoding="utf-8",
    )
    return generated, planned, config


def test_experiment_seeds_are_explicit_and_immutable() -> None:
    assert STABLE_YEAR_BENCHMARK_SEEDS == ExperimentSeeds(2026, 2026, 20260630)
    with pytest.raises(FrozenInstanceError):
        STABLE_YEAR_BENCHMARK_SEEDS.solver_seed = 1


def test_manifest_counts_are_derived_from_canonical_input(tmp_path) -> None:
    generated, planned, config = _write_inputs(tmp_path)

    manifest = build_experiment_manifest(
        generated,
        planned,
        config,
        scenario_id="stable_year",
        seeds=STABLE_YEAR_BENCHMARK_SEEDS,
    )

    assert manifest.seeds == ExperimentSeeds(2026, 2026, 20260630)
    assert manifest.fingerprint.students == 1
    assert manifest.fingerprint.logical_requests == 3
    assert manifest.fingerprint.logical_primaries == 2
    assert manifest.fingerprint.alternates == 1
    assert manifest.fingerprint.logical_sections == 3
    assert manifest.fingerprint.section_rows == 4
    assert manifest.fingerprint.candidate_edges == 3


def test_canonical_fingerprint_is_independent_of_input_row_order() -> None:
    first = canonicalize_allocation_input(_students(), _requests(), _sections(), _catalog())
    second = canonicalize_allocation_input(
        _students(),
        _requests().sample(frac=1, random_state=7).reset_index(drop=True),
        _sections().sample(frac=1, random_state=8).reset_index(drop=True),
        _catalog().sample(frac=1, random_state=9).reset_index(drop=True),
    )

    assert canonical_input_fingerprint(first) == canonical_input_fingerprint(second)


def test_solver_seed_changes_manifest_identity_not_input_fingerprint(tmp_path) -> None:
    generated, planned, config = _write_inputs(tmp_path)
    first = build_experiment_manifest(
        generated,
        planned,
        config,
        scenario_id="stable_year",
        seeds=ExperimentSeeds(2026, 2026, 1),
    )
    second = build_experiment_manifest(
        generated,
        planned,
        config,
        scenario_id="stable_year",
        seeds=ExperimentSeeds(2026, 2026, 2),
    )

    assert first.fingerprint == second.fingerprint
    assert first.to_dict() != second.to_dict()


def test_generation_seed_mismatch_fails_closed(tmp_path) -> None:
    generated, planned, config = _write_inputs(tmp_path, data_seed=20260630)

    with pytest.raises(ExperimentManifestError, match="generation metadata data seed"):
        build_experiment_manifest(
            generated,
            planned,
            config,
            scenario_id="stable_year",
            seeds=STABLE_YEAR_BENCHMARK_SEEDS,
        )


def test_section_planner_lineage_hash_mismatch_fails_closed(tmp_path) -> None:
    generated, planned, config = _write_inputs(tmp_path)
    metadata_path = planned / "section_planning_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["input_file_hashes"]["requests.csv"] = "wrong"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ExperimentManifestError, match="section metadata input hash for requests.csv"):
        build_experiment_manifest(
            generated,
            planned,
            config,
            scenario_id="stable_year",
            seeds=STABLE_YEAR_BENCHMARK_SEEDS,
        )


def test_generation_output_hash_and_metadata_counts_are_required(tmp_path) -> None:
    generated, planned, config = _write_inputs(tmp_path)
    metadata_path = generated / "generation_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["output_file_hashes"]["requests.csv"] = "wrong"
    metadata["total_students"] = 999
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ExperimentManifestError) as error:
        build_experiment_manifest(
            generated,
            planned,
            config,
            scenario_id="stable_year",
            seeds=STABLE_YEAR_BENCHMARK_SEEDS,
        )

    assert any("generation metadata output hash for requests.csv" in issue for issue in error.value.issues)
    assert any("generation metadata total_students" in issue for issue in error.value.issues)


def test_section_output_hash_is_bound_to_section_planning_seed(tmp_path) -> None:
    generated, planned, config = _write_inputs(tmp_path)
    sections = pd.read_csv(planned / "sections.csv", keep_default_na=False)
    sections.loc[sections["section_id"] == "SEC_NORMAL", "period_1"] = "P4"
    sections.to_csv(planned / "sections.csv", index=False)

    with pytest.raises(ExperimentManifestError, match="section metadata output hash for sections.csv"):
        build_experiment_manifest(
            generated,
            planned,
            config,
            scenario_id="stable_year",
            seeds=STABLE_YEAR_BENCHMARK_SEEDS,
        )


def test_section_metadata_canonical_count_mismatch_fails_closed(tmp_path) -> None:
    generated, planned, config = _write_inputs(tmp_path)
    metadata_path = planned / "section_planning_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["total_logical_sections"] = 999
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ExperimentManifestError, match="section metadata total_logical_sections"):
        build_experiment_manifest(
            generated,
            planned,
            config,
            scenario_id="stable_year",
            seeds=STABLE_YEAR_BENCHMARK_SEEDS,
        )


def test_manifest_round_trip_and_verification_detect_input_changes(tmp_path) -> None:
    generated, planned, config = _write_inputs(tmp_path)
    manifest = build_experiment_manifest(
        generated,
        planned,
        config,
        scenario_id="stable_year",
        seeds=STABLE_YEAR_BENCHMARK_SEEDS,
    )
    manifest_path = tmp_path / "experiment_manifest.json"
    write_experiment_manifest(manifest, manifest_path)

    loaded = load_experiment_manifest(manifest_path)
    canonical = verify_experiment_manifest(loaded, config_dir=config)
    assert canonical_input_fingerprint(canonical) == manifest.fingerprint

    sections = pd.read_csv(planned / "sections.csv", keep_default_na=False)
    sections.loc[sections["section_id"] == "SEC_NORMAL", "period_1"] = "P4"
    sections.to_csv(planned / "sections.csv", index=False)
    with pytest.raises(
        ExperimentManifestError,
        match="section metadata output hash|canonical_input_hash|section_input_hash",
    ):
        verify_experiment_manifest(loaded, config_dir=config)
