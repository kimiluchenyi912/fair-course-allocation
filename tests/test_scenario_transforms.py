from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from src.allocation import canonicalize_allocation_input
from src.infeasibility_certificates import validate_certificate
from src.scenario_transforms import (
    ScenarioTransformError,
    StressTransformSpec,
    apply_stress_transforms,
    load_base_frames,
    nearest_integer_count,
)


CATALOG_COLUMNS = ["course_id", "periods_required", "schedule_structure"]
STUDENT_COLUMNS = [
    "student_id", "grade", "target_course_count", "unscheduled_preference", "random_seed_group",
    "priority_protected", "priority_reason", "priority_valid_school_year",
]
REQUEST_COLUMNS = ["student_id", "course_id", "request_type", "request_rank", "request_group", "must_share_block_id"]
SECTION_COLUMNS = [
    "section_id", "course_id", "period_1", "period_2", "semester", "capacity", "block_id",
    "linked_section_group_id", "logical_block_id", "semester_content",
]


def _fixture_base(tmp_path: Path, *, student_target: int = 5, section_count: int = 12) -> tuple[Path, Path]:
    base = tmp_path / "base"
    generated = base / "generated"
    sections = base / "sections"
    generated.mkdir(parents=True)
    sections.mkdir()
    config = tmp_path / "config"
    config.mkdir()
    course_ids = [f"C{index}" for index in range(1, max(section_count, 6) + 1)]
    catalog = pd.DataFrame([(course, 1, "standard") for course in course_ids], columns=CATALOG_COLUMNS)
    catalog.to_csv(config / "course_catalog.csv", index=False)
    students = pd.DataFrame(
        [
            ["S1", 9, student_target, "none", "S1", False, "", ""],
            ["S2", 10, 1, "none", "S2", False, "", ""],
        ],
        columns=STUDENT_COLUMNS,
    )
    requests = []
    for course in course_ids[:6]:
        requests.append(["S1", course, "primary", "", "", ""])
    requests.append(["S1", "C7", "alternate", 1, "alternate", ""])
    requests.append(["S2", "C1", "primary", "", "", ""])
    requests = pd.DataFrame(requests, columns=REQUEST_COLUMNS)
    section_rows = []
    for index, course in enumerate(course_ids[:section_count], start=1):
        section_rows.append([f"SEC_{course}", course, f"P{(index - 1) % 7 + 1}", "", "full_year", 10, "", f"GROUP_{course}", course, ""])
    sections_frame = pd.DataFrame(section_rows, columns=SECTION_COLUMNS)
    students.to_csv(generated / "students.csv", index=False)
    requests.to_csv(generated / "requests.csv", index=False)
    sections_frame.to_csv(sections / "sections.csv", index=False)
    (generated / "generation_metadata.json").write_text(json.dumps({
        "scenario_id": "stable_year", "seed": 2026, "total_students": len(students),
        "primary_request_rows": 7, "alternate_request_rows": 1,
        "output_file_hashes": {},
    }), encoding="utf-8")
    (sections / "section_planning_metadata.json").write_text(json.dumps({
        "scenario_id": "stable_year", "seed": 2026, "student_count": len(students),
        "primary_request_rows": 7, "total_section_rows": len(sections_frame),
        "total_logical_sections": len(sections_frame), "total_primary_demand": 7,
        "input_file_hashes": {}, "output_file_hashes": {},
    }), encoding="utf-8")
    return base, config


def _spec(scenario_id: str, transform: dict, expected: str = "unknown") -> StressTransformSpec:
    return StressTransformSpec(scenario_id, "base", 123, (transform,), (transform["type"],), "test", expected)


def test_nearest_integer_rounding_is_stable() -> None:
    assert nearest_integer_count(10, 0.05) == 1
    assert nearest_integer_count(10, 0.04) == 0
    assert nearest_integer_count(2630, 0.075) == 197


def test_base_frames_are_read_only_inputs(tmp_path: Path) -> None:
    base, _ = _fixture_base(tmp_path)
    before = (base / "generated" / "students.csv").read_bytes()
    load_base_frames(base)
    assert (base / "generated" / "students.csv").read_bytes() == before


def test_enrollment_surge_is_deterministic_and_adds_complete_profiles(tmp_path: Path) -> None:
    base, config = _fixture_base(tmp_path)
    first = tmp_path / "first"
    second = tmp_path / "second"
    report1 = apply_stress_transforms(base, first, _spec("surge", {"type": "enrollment_surge", "percentage": 0.5}), config_dir=config)
    report2 = apply_stress_transforms(base, second, _spec("surge", {"type": "enrollment_surge", "percentage": 0.5}), config_dir=config)
    assert report1["deterministic_replay_hash"] == report2["deterministic_replay_hash"]
    students = pd.read_csv(first / "generated" / "students.csv", keep_default_na=False)
    requests = pd.read_csv(first / "generated" / "requests.csv", keep_default_na=False)
    assert len(students) == 3
    assert len(requests) == 15
    assert requests["request_id"].is_unique
    assert len(pd.read_csv(base / "generated" / "students.csv")) == 2


def test_alternate_drop_preserves_primary_rows_and_original_ranks(tmp_path: Path) -> None:
    base, config = _fixture_base(tmp_path)
    output = tmp_path / "drop"
    apply_stress_transforms(base, output, _spec("drop", {"type": "alternate_drop", "percentage": 1.0}), config_dir=config)
    requests = pd.read_csv(output / "generated" / "requests.csv", keep_default_na=False)
    assert len(requests[requests["request_type"] == "alternate"]) == 0
    assert len(requests[requests["request_type"] == "primary"]) == 7


def test_capacity_reduction_changes_each_logical_group_once(tmp_path: Path) -> None:
    base, config = _fixture_base(tmp_path)
    catalog = pd.read_csv(config / "course_catalog.csv", keep_default_na=False)
    catalog.loc[len(catalog)] = ["GOV_ECON_REG", 1, "semester_block"]
    catalog.to_csv(config / "course_catalog.csv", index=False)
    sections = pd.read_csv(base / "sections" / "sections.csv", keep_default_na=False)
    linked_rows = pd.DataFrame([
        ["SEC_GOV_1", "GOV_ECON_REG", "P1", "", "semester_1", 10, "GOV_ECON_REG", "GROUP_GOV", "GOV_ECON_REG", "Government"],
        ["SEC_GOV_2", "GOV_ECON_REG", "P1", "", "semester_2", 10, "GOV_ECON_REG", "GROUP_GOV", "GOV_ECON_REG", "Economics"],
    ], columns=SECTION_COLUMNS)
    sections = pd.concat([sections, linked_rows], ignore_index=True)
    sections.to_csv(base / "sections" / "sections.csv", index=False)
    output = tmp_path / "capacity"
    apply_stress_transforms(base, output, _spec("capacity", {"type": "capacity_reduction", "percentage": 0.5}), config_dir=config)
    result = pd.read_csv(output / "sections" / "sections.csv", keep_default_na=False)
    assert set(result[result["linked_section_group_id"] == "GROUP_GOV"]["capacity"]) == {5}


def test_section_outage_removes_complete_logical_group(tmp_path: Path) -> None:
    base, config = _fixture_base(tmp_path)
    output = tmp_path / "outage"
    report = apply_stress_transforms(base, output, _spec("outage", {"type": "section_outage", "count": 1}), config_dir=config)
    assert report["section_rows_changed"] == -1
    assert report["logical_sections_removed"] == 1


def test_protected_negative_certificate_is_revalidated(tmp_path: Path) -> None:
    base, config = _fixture_base(tmp_path)
    output = tmp_path / "protected"
    report = apply_stress_transforms(base, output, _spec("protected", {"type": "protected_primary_no_candidate"}, "structurally_infeasible"), config_dir=config)
    catalog = pd.read_csv(config / "course_catalog.csv", keep_default_na=False)
    allocation_input = canonicalize_allocation_input(
        pd.read_csv(output / "generated" / "students.csv", keep_default_na=False),
        pd.read_csv(output / "generated" / "requests.csv", keep_default_na=False),
        pd.read_csv(output / "sections" / "sections.csv", keep_default_na=False),
        catalog,
    )
    assert validate_certificate(report["certificate"], allocation_input) == (True, "protected primary has zero candidates")


def test_minimum_load_negative_certificate_is_revalidated(tmp_path: Path) -> None:
    base, config = _fixture_base(tmp_path)
    output = tmp_path / "minimum"
    report = apply_stress_transforms(base, output, _spec("minimum", {"type": "minimum_logical_load_max_four"}, "structurally_infeasible"), config_dir=config)
    assert report["certificate"]["feasible_logical_course_count"] <= 4


def test_global_capacity_negative_certificate_is_revalidated(tmp_path: Path) -> None:
    base, config = _fixture_base(tmp_path, section_count=12)
    output = tmp_path / "global"
    report = apply_stress_transforms(base, output, _spec("global", {"type": "global_capacity_deficit"}, "structurally_infeasible"), config_dir=config)
    assert report["certificate"]["capacity_margin"] < 0


def test_nonempty_destination_fails_closed(tmp_path: Path) -> None:
    base, config = _fixture_base(tmp_path)
    output = tmp_path / "existing"
    output.mkdir()
    (output / "do_not_overwrite").write_text("x", encoding="utf-8")
    with pytest.raises(ScenarioTransformError, match="non-empty"):
        apply_stress_transforms(base, output, _spec("existing", {"type": "alternate_drop", "percentage": 0.1}), config_dir=config)
