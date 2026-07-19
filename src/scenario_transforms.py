"""Deterministic, schema-aware transforms for synthetic stress inputs."""

from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.allocation import canonicalize_allocation_input
from src.experiment_manifest import canonical_input_fingerprint
from src.infeasibility_certificates import (
    build_global_capacity_certificate,
    build_minimum_load_certificate,
    build_protected_primary_certificate,
    validate_certificate,
)
from src.section_planning.demand import logical_block_id_from_request


TRANSFORM_SCHEMA_VERSION = 1
REQUEST_ID_COLUMN = "request_id"
STUDENT_COLUMNS = (
    "student_id",
    "grade",
    "target_course_count",
    "unscheduled_preference",
    "random_seed_group",
    "priority_protected",
    "priority_reason",
    "priority_valid_school_year",
)


class ScenarioTransformError(ValueError):
    """Raised when a stress transform cannot be applied without ambiguity."""


@dataclass(frozen=True)
class StressTransformSpec:
    scenario_id: str
    base_scenario_id: str
    transform_seed: int
    transforms: tuple[dict[str, Any], ...]
    transform_order: tuple[str, ...]
    scenario_family: str
    expected_feasibility: str


def nearest_integer_count(total: int, percentage: float) -> int:
    if total < 0 or not 0 <= percentage <= 1:
        raise ScenarioTransformError("count rounding requires total >= 0 and percentage in [0, 1]")
    return math.floor(total * percentage + 0.5)


def load_base_frames(base_scenario_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = Path(base_scenario_dir)
    generated = base / "generated"
    sections = base / "sections"
    required = (generated / "students.csv", generated / "requests.csv", sections / "sections.csv")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ScenarioTransformError("Base scenario is missing: " + ", ".join(missing))
    return (
        pd.read_csv(required[0], keep_default_na=False),
        pd.read_csv(required[1], keep_default_na=False),
        pd.read_csv(required[2], keep_default_na=False),
    )


def apply_stress_transforms(
    base_scenario_dir: str | Path,
    output_dir: str | Path,
    spec: StressTransformSpec,
    *,
    config_dir: str | Path = "data/config",
    source_git_commit: str = "",
) -> dict[str, Any]:
    """Transform a persistent Phase A scenario into a new atomic scenario dir."""

    destination = Path(output_dir)
    if destination.exists() and any(destination.iterdir()):
        raise ScenarioTransformError(f"Scenario output is non-empty; refusing to overwrite: {destination}")
    students, requests, sections = load_base_frames(base_scenario_dir)
    catalog = pd.read_csv(Path(config_dir) / "course_catalog.csv", keep_default_na=False)
    _require_columns(students, STUDENT_COLUMNS, "students")
    _require_columns(requests, ("student_id", "course_id", "request_type", "request_rank", "request_group", "must_share_block_id"), "requests")
    _require_columns(sections, ("section_id", "course_id", "period_1", "period_2", "semester", "capacity", "linked_section_group_id", "logical_block_id"), "sections")
    requests = _with_request_ids(requests)
    before_input = _canonicalize(students, requests, sections, catalog)
    before_fingerprint = canonical_input_fingerprint(before_input)
    current_students, current_requests, current_sections = students.copy(deep=True), requests.copy(deep=True), sections.copy(deep=True)
    stage_reports: list[dict[str, Any]] = []
    certificate: dict[str, Any] | None = None
    for stage_index, transform in enumerate(spec.transforms):
        if not isinstance(transform, dict) or transform.get("type") != spec.transform_order[stage_index]:
            raise ScenarioTransformError(f"Transform order mismatch for {spec.scenario_id}")
        transform_type = str(transform["type"])
        stage_seed = spec.transform_seed + stage_index
        if transform_type == "enrollment_surge":
            current_students, current_requests, report = _enrollment_surge(current_students, current_requests, transform, stage_seed)
        elif transform_type == "popular_course_surge":
            current_students, current_requests, report = _popular_course_surge(current_students, current_requests, current_sections, catalog, transform, stage_seed)
        elif transform_type == "alternate_drop":
            current_requests, report = _alternate_drop(current_requests, transform, stage_seed)
        elif transform_type == "capacity_reduction":
            current_sections, report = _capacity_reduction(current_sections, transform)
        elif transform_type == "section_outage":
            current_sections, report = _section_outage(current_requests, current_sections, catalog, transform)
        elif transform_type == "protected_primary_no_candidate":
            current_students, current_sections, selected_request_key, report = _protected_no_candidate(
                current_students, current_requests, current_sections, catalog
            )
            report["selected_primary_request_key"] = selected_request_key
        elif transform_type == "minimum_logical_load_max_four":
            current_requests, selected_student_id, report = _minimum_load_max_four(current_students, current_requests, current_sections, catalog)
            report["selected_student_id"] = selected_student_id
        elif transform_type == "global_capacity_deficit":
            current_sections, report = _global_capacity_deficit(current_students, current_requests, current_sections, catalog)
        else:
            raise ScenarioTransformError(f"Unsupported transform type: {transform_type}")
        report.update({"transform_type": transform_type, "stage_index": stage_index, "seed": stage_seed})
        stage_reports.append(report)

    after_input = _canonicalize(current_students, current_requests, current_sections, catalog)
    after_fingerprint = canonical_input_fingerprint(after_input)
    if spec.expected_feasibility == "structurally_infeasible":
        certificate = _build_certificate(spec, after_input, stage_reports)
        valid, reason = validate_certificate(certificate, after_input)
        if not valid:
            raise ScenarioTransformError(f"Invalid infeasibility certificate: {reason}")

    report = _build_transformation_report(
        spec,
        before_fingerprint,
        after_fingerprint,
        students,
        requests,
        sections,
        current_students,
        current_requests,
        current_sections,
        stage_reports,
        source_git_commit,
        certificate,
    )
    _atomic_write_scenario(
        Path(base_scenario_dir),
        destination,
        current_students,
        current_requests,
        current_sections,
        after_input,
        report,
        certificate,
    )
    return report


def _enrollment_surge(students: pd.DataFrame, requests: pd.DataFrame, transform: dict[str, Any], seed: int):
    percentage = _percentage(transform)
    count = nearest_integer_count(len(students), percentage)
    rng = random.Random(seed)
    selected = sorted(rng.sample(sorted(students["student_id"].astype(str)), count)) if count else []
    clones: list[pd.Series] = []
    request_clones: list[pd.DataFrame] = []
    created_student_ids: list[str] = []
    created_request_ids: list[str] = []
    for index, student_id in enumerate(selected, start=1):
        clone_id = f"{student_id}__SURGE_{index:04d}"
        row = students[students["student_id"].astype(str) == student_id].iloc[0].copy()
        row["student_id"] = clone_id
        clones.append(row)
        created_student_ids.append(clone_id)
        source = requests[requests["student_id"].astype(str) == student_id].copy()
        source["student_id"] = clone_id
        source[REQUEST_ID_COLUMN] = [f"{value}__{clone_id}" for value in source[REQUEST_ID_COLUMN]]
        created_request_ids.extend(source[REQUEST_ID_COLUMN].astype(str).tolist())
        request_clones.append(source)
    new_students = pd.concat([students, pd.DataFrame(clones, columns=students.columns)], ignore_index=True)
    new_requests = pd.concat([requests, *request_clones], ignore_index=True) if request_clones else requests.copy()
    return new_students, new_requests, {"selected_student_ids": selected, "created_student_ids": created_student_ids, "created_request_ids": created_request_ids, "students_added": len(clones), "requests_added": sum(len(frame) for frame in request_clones), "percentage": percentage, "rounding_rule": "floor(n * percentage + 0.5)"}


def _popular_course_surge(students, requests, sections, catalog, transform, seed):
    percentage = _percentage(transform)
    current = _canonicalize(students, requests, sections, catalog)
    demand: dict[str, int] = {}
    for request in current.logical_requests:
        if request.request_type == "primary":
            demand[request.candidate_key] = demand.get(request.candidate_key, 0) + 1
    capacity: dict[str, int] = {}
    for section in current.logical_sections:
        capacity[section.logical_block_id] = capacity.get(section.logical_block_id, 0) + section.capacity
    keys = sorted(set(demand) | set(capacity))
    top_count = max(1, math.ceil(len(keys) * 0.1)) if keys else 0
    ranked = sorted(keys, key=lambda key: (-_pressure(demand.get(key, 0), capacity.get(key, 0)), key))
    top_keys = set(ranked[:top_count])
    candidate_students = sorted(
        {
            request.student_id
            for request in current.logical_requests
            if request.request_type == "primary" and request.candidate_key in top_keys
        }
    )
    count = nearest_integer_count(len(candidate_students), percentage)
    selected = sorted(random.Random(seed).sample(candidate_students, count)) if count else []
    selected_rows = students[students["student_id"].astype(str).isin(selected)]
    clones: list[pd.Series] = []
    request_clones: list[pd.DataFrame] = []
    created_student_ids: list[str] = []
    created_request_ids: list[str] = []
    for index, (_, student) in enumerate(selected_rows.sort_values("student_id").iterrows(), start=1):
        source_id = str(student["student_id"])
        clone_id = f"{source_id}__POPULAR_{index:04d}"
        row = student.copy()
        row["student_id"] = clone_id
        clones.append(row)
        created_student_ids.append(clone_id)
        source = requests[requests["student_id"].astype(str) == source_id].copy()
        source["student_id"] = clone_id
        source[REQUEST_ID_COLUMN] = [f"{value}__{clone_id}" for value in source[REQUEST_ID_COLUMN]]
        created_request_ids.extend(source[REQUEST_ID_COLUMN].astype(str).tolist())
        request_clones.append(source)
    new_students = pd.concat([students, pd.DataFrame(clones, columns=students.columns)], ignore_index=True)
    new_requests = pd.concat([requests, *request_clones], ignore_index=True) if request_clones else requests.copy()
    return new_students, new_requests, {"selected_student_ids": selected, "created_student_ids": created_student_ids, "created_request_ids": created_request_ids, "selected_logical_course_ids": sorted(top_keys), "top_decile_count": top_count, "students_added": len(clones), "requests_added": sum(len(frame) for frame in request_clones), "percentage": percentage, "rounding_rule": "floor(n * percentage + 0.5)"}


def _alternate_drop(requests, transform, seed):
    percentage = _percentage(transform)
    alternate = requests[requests["request_type"].astype(str) == "alternate"]
    count = nearest_integer_count(len(alternate), percentage)
    selected = sorted(random.Random(seed).sample(sorted(alternate[REQUEST_ID_COLUMN].astype(str)), count)) if count else []
    before_by_student = alternate.groupby("student_id").size().astype(int).to_dict()
    kept = requests[~requests[REQUEST_ID_COLUMN].astype(str).isin(selected)].copy()
    after_by_student = kept[kept["request_type"].astype(str) == "alternate"].groupby("student_id").size().astype(int).to_dict()
    per_student = {
        str(student): {"before": int(before_by_student.get(student, 0)), "after": int(after_by_student.get(student, 0))}
        for student in sorted(set(before_by_student) | set(after_by_student))
    }
    return kept, {"selected_request_ids": selected, "removed_request_ids": selected, "alternate_requests_removed": count, "per_student_alternate_counts": per_student, "percentage": percentage, "rounding_rule": "floor(n * percentage + 0.5)"}


def _capacity_reduction(sections, transform):
    percentage = _percentage(transform)
    updated = sections.copy()
    changed: list[dict[str, Any]] = []
    for group_id, group in sections.groupby("linked_section_group_id", sort=True):
        old = int(group["capacity"].iloc[0])
        new = max(1, math.floor(old * (1 - percentage)))
        if new != old:
            mask = updated["linked_section_group_id"] == group_id
            updated.loc[mask, "capacity"] = new
        changed.append({"linked_section_group_id": str(group_id), "old_capacity": old, "new_capacity": new, "capacity_removed": old - new})
    return updated, {"capacity_changes": changed, "capacity_removed": sum(row["capacity_removed"] for row in changed), "logical_sections_changed": sum(row["old_capacity"] != row["new_capacity"] for row in changed), "percentage": percentage}


def _section_outage(requests, sections, catalog, transform):
    count = _positive_int(transform, "count")
    primary = requests[requests["request_type"].astype(str) == "primary"].copy()
    primary["candidate_key"] = primary.apply(logical_block_id_from_request, axis=1)
    demand = primary[["student_id", "candidate_key"]].drop_duplicates().groupby("candidate_key").size().to_dict()
    capacities = sections.groupby("logical_block_id", sort=True)["capacity"].first().to_dict()
    group_rows = []
    for group_id, group in sections.groupby("linked_section_group_id", sort=True):
        key = str(group["logical_block_id"].iloc[0])
        group_rows.append((-_pressure(demand.get(key, 0), int(capacities.get(key, 0))), key, str(group_id)))
    selected = [group_id for _, _, group_id in sorted(group_rows)[:count]]
    selected_details = [
        {
            "linked_section_group_id": group_id,
            "periods": sorted({str(period) for period in sections.loc[sections["linked_section_group_id"].astype(str) == group_id, "period_1"]}),
            "capacity": int(sections.loc[sections["linked_section_group_id"].astype(str) == group_id, "capacity"].iloc[0]),
        }
        for group_id in selected
    ]
    updated = sections[~sections["linked_section_group_id"].astype(str).isin(selected)].copy()
    return updated, {"selected_logical_section_ids": selected, "selected_logical_sections": selected_details, "logical_sections_removed": len(selected), "section_rows_removed": len(sections) - len(updated)}


def _protected_no_candidate(students, requests, sections, catalog):
    current = _canonicalize(students, requests, sections, catalog)
    protected = sorted(student.student_id for student in current.students if student.priority_protected)
    priority_before = False
    if protected:
        student_id = protected[0]
        priority_before = True
    else:
        student_id = sorted(str(value) for value in students["student_id"])[0]
        students = students.copy()
        mask = students["student_id"].astype(str) == student_id
        priority_before = bool(students.loc[mask, "priority_protected"].iloc[0])
        students.loc[mask, "priority_protected"] = True
        students.loc[mask, "priority_reason"] = "prior_year_unmet_primary"
        students.loc[mask, "priority_valid_school_year"] = "2026-2027"
        current = _canonicalize(students, requests, sections, catalog)
    primary = sorted(current.students_by_id[student_id].primary_requests, key=lambda item: item.request_key)
    if not primary:
        raise ScenarioTransformError(f"No primary request for protected student {student_id}")
    request = primary[0]
    key = request.candidate_key
    mask = sections["logical_block_id"].astype(str) == key
    updated = sections[~mask].copy()
    if len(updated) == len(sections):
        raise ScenarioTransformError(f"Protected request candidate key has no sections: {key}")
    return students, updated, request.request_key, {"selected_student_ids": [student_id], "selected_request_ids": [request.request_key], "selected_logical_course_ids": [key], "priority_before": priority_before, "priority_after": True, "logical_sections_removed": int(mask.sum() and sections.loc[mask, "linked_section_group_id"].nunique()), "section_rows_removed": int(mask.sum())}


def _minimum_load_max_four(students, requests, sections, catalog):
    current = _canonicalize(students, requests, sections, catalog)
    candidates = []
    for student in current.students:
        if student.target_period_units < 5:
            continue
        keys = sorted({request.candidate_key for request in student.primary_requests})
        if len(keys) > 4:
            candidates.append((student.student_id, keys))
    if not candidates:
        raise ScenarioTransformError("Could not find a target>=5 student with more than four primary choices")
    student_id, keys = candidates[0]
    keep = set(keys[:4])
    student_requests = requests[requests["student_id"].astype(str) == student_id].copy()
    logical_keys = student_requests.apply(logical_block_id_from_request, axis=1)
    keep_mask = (student_requests["request_type"].astype(str) == "primary") & logical_keys.isin(keep)
    retained = student_requests[keep_mask]
    removed_request_ids = student_requests.loc[~student_requests.index.isin(retained.index), REQUEST_ID_COLUMN].astype(str).tolist()
    other = requests[requests["student_id"].astype(str) != student_id]
    updated = pd.concat([other, retained], ignore_index=True)
    return updated, student_id, {"selected_student_ids": [student_id], "selected_logical_course_ids": sorted(keep), "removed_request_ids": removed_request_ids, "requests_removed": int(len(student_requests) - len(retained)), "unique_primary_choices_after": len(keep)}


def _global_capacity_deficit(students, requests, sections, catalog):
    current = _canonicalize(students, requests, sections, catalog)
    required = 5 * len(current.students)
    total = sum(section.capacity for section in current.logical_sections)
    if total < required:
        return sections.copy(), {"capacity_removed": 0, "required_minimum_capacity": required, "total_logical_capacity_before": total}
    need = total - (required - 1)
    updated = sections.copy()
    changes = []
    for section in sorted(current.logical_sections, key=lambda item: (-item.capacity, item.linked_section_group_id)):
        if need <= 0:
            break
        reduction = min(max(section.capacity - 1, 0), need)
        new = section.capacity - reduction
        mask = updated["linked_section_group_id"].astype(str) == section.linked_section_group_id
        updated.loc[mask, "capacity"] = new
        changes.append({"linked_section_group_id": section.linked_section_group_id, "old_capacity": section.capacity, "new_capacity": new, "capacity_removed": reduction})
        need -= reduction
    removed_groups: list[str] = []
    if need > 0:
        removable = sorted(section.linked_section_group_id for section in current.logical_sections)
        while need > 0 and len(removable) - len(removed_groups) > 1:
            group_id = removable[len(removed_groups)]
            group_capacity = int(updated.loc[updated["linked_section_group_id"].astype(str) == group_id, "capacity"].iloc[0])
            updated = updated[updated["linked_section_group_id"].astype(str) != group_id].copy()
            removed_groups.append(group_id)
            need -= group_capacity
        if need > 0:
            raise ScenarioTransformError("Could not create a strict global capacity deficit while keeping one positive section")
    removed_capacity = sum(row["capacity_removed"] for row in changes) + sum(
        int(sections.loc[sections["linked_section_group_id"].astype(str) == group_id, "capacity"].iloc[0])
        for group_id in removed_groups
    )
    return updated, {"capacity_changes": changes, "capacity_removed": removed_capacity, "required_minimum_capacity": required, "total_logical_capacity_before": total, "total_logical_capacity_target": required - 1, "logical_sections_removed": len(removed_groups), "selected_logical_section_ids": removed_groups, "section_rows_removed": len(sections) - len(updated)}


def _build_certificate(spec, allocation_input, stage_reports):
    if spec.expected_feasibility != "structurally_infeasible":
        return None
    if spec.transforms[0]["type"] == "protected_primary_no_candidate":
        selected = next(row for row in stage_reports if row["transform_type"] == "protected_primary_no_candidate")
        return build_protected_primary_certificate(allocation_input, selected["selected_student_ids"][0], selected["selected_primary_request_key"])
    if spec.transforms[0]["type"] == "minimum_logical_load_max_four":
        selected = next(row for row in stage_reports if row["transform_type"] == "minimum_logical_load_max_four")
        return build_minimum_load_certificate(allocation_input, selected["selected_student_id"])
    if spec.transforms[0]["type"] == "global_capacity_deficit":
        return build_global_capacity_certificate(allocation_input)
    raise ScenarioTransformError(f"No certificate builder for {spec.scenario_id}")


def _build_transformation_report(spec, before_fp, after_fp, before_students, before_requests, before_sections, students, requests, sections, stage_reports, source_git_commit, certificate):
    created_students = sorted({item for stage in stage_reports for item in stage.get("created_student_ids", [])})
    created_requests = sorted({item for stage in stage_reports for item in stage.get("created_request_ids", [])})
    return {
        "schema_version": TRANSFORM_SCHEMA_VERSION,
        "scenario_id": spec.scenario_id,
        "base_scenario_id": spec.base_scenario_id,
        "base_fingerprint": _fingerprint_dict(before_fp),
        "transformed_fingerprint": _fingerprint_dict(after_fp),
        "transform_schema_version": TRANSFORM_SCHEMA_VERSION,
        "transform_order": list(spec.transform_order),
        "transforms": list(spec.transforms),
        "transform_seed": spec.transform_seed,
        "source_git_commit": source_git_commit,
        "files_read": ["generated/students.csv", "generated/requests.csv", "sections/sections.csv"],
        "files_written": ["generated/students.csv", "generated/requests.csv", "sections/sections.csv", "transformation_report.json"] + (["infeasibility_certificate.json"] if certificate is not None else []),
        "rows_before": {"students": len(before_students), "requests": len(before_requests), "section_rows": len(before_sections)},
        "rows_after": {"students": len(students), "requests": len(requests), "section_rows": len(sections)},
        "students_added": sum(stage.get("students_added", 0) for stage in stage_reports),
        "requests_added": sum(stage.get("requests_added", 0) for stage in stage_reports),
        "alternate_requests_removed": sum(stage.get("alternate_requests_removed", 0) for stage in stage_reports),
        "logical_sections_changed": after_fp.logical_sections - before_fp.logical_sections,
        "section_rows_changed": len(sections) - len(before_sections),
        "capacity_removed": sum(row.get("capacity_removed", 0) for stage in stage_reports for row in stage.get("capacity_changes", [])),
        "logical_sections_removed": sum(stage.get("logical_sections_removed", 0) for stage in stage_reports),
        "selected_student_ids": sorted({item for stage in stage_reports for item in stage.get("selected_student_ids", [])}),
        "selected_request_ids": sorted({item for stage in stage_reports for item in stage.get("selected_request_ids", [])}),
        "selected_logical_course_ids": sorted({item for stage in stage_reports for item in stage.get("selected_logical_course_ids", [])}),
        "created_student_ids": created_students,
        "created_request_ids": created_requests,
        "rounding_rule": "floor(n * percentage + 0.5) for percentage transforms; integer capacities use floor with minimum one",
        "stage_reports": stage_reports,
        "certificate": certificate,
        "validation_status": "passed",
        "deterministic_replay_hash": _dataframe_hashes(students, requests, sections),
    }


def _atomic_write_scenario(base_dir, destination, students, requests, sections, allocation_input, report, certificate):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        generated = temp / "generated"
        section_dir = temp / "sections"
        generated.mkdir()
        section_dir.mkdir()
        base_generated = Path(base_dir) / "generated"
        base_sections = Path(base_dir) / "sections"
        students.to_csv(generated / "students.csv", index=False)
        requests.to_csv(generated / "requests.csv", index=False)
        sections.to_csv(section_dir / "sections.csv", index=False)
        for name in ("generation_summary.csv",):
            source = base_generated / name
            if source.is_file():
                shutil.copy2(source, generated / name)
        for name in ("course_demand_summary.csv", "period_layout_summary.csv"):
            source = base_sections / name
            if source.is_file():
                shutil.copy2(source, section_dir / name)
        generation_metadata = _read_json(base_generated / "generation_metadata.json")
        generation_metadata.update({
            "total_students": len(students),
            "primary_request_rows": int((requests["request_type"] == "primary").sum()),
            "alternate_request_rows": int((requests["request_type"] == "alternate").sum()),
            "output_file_hashes": {"students.csv": _sha256_file(generated / "students.csv"), "requests.csv": _sha256_file(generated / "requests.csv")},
        })
        _write_json(generated / "generation_metadata.json", generation_metadata)
        section_metadata = _read_json(base_sections / "section_planning_metadata.json")
        section_metadata.update({
            "student_count": len(students),
            "primary_request_rows": int((requests["request_type"] == "primary").sum()),
            "total_section_rows": len(sections),
            "total_logical_sections": len(allocation_input.logical_sections),
            "total_primary_demand": len(tuple(request for request in allocation_input.logical_requests if request.request_type == "primary")),
            "total_planned_seats": sum(section.capacity for section in allocation_input.logical_sections),
            "total_remaining_waitlist": _remaining_waitlist(allocation_input),
            "output_file_hashes": {"sections.csv": _sha256_file(section_dir / "sections.csv")},
            "input_file_hashes": {"students.csv": _sha256_file(generated / "students.csv"), "requests.csv": _sha256_file(generated / "requests.csv")},
        })
        _write_json(section_dir / "section_planning_metadata.json", section_metadata)
        _write_json(temp / "transformation_report.json", report)
        if certificate is not None:
            _write_json(temp / "infeasibility_certificate.json", certificate)
        if destination.exists():
            destination.rmdir()
        temp.replace(destination)
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def _canonicalize(students, requests, sections, catalog):
    return canonicalize_allocation_input(students, requests, sections, catalog)


def _remaining_waitlist(allocation_input):
    demand = {}
    for request in allocation_input.logical_requests:
        if request.request_type == "primary":
            demand[request.candidate_key] = demand.get(request.candidate_key, 0) + 1
    capacity = {}
    for section in allocation_input.logical_sections:
        capacity[section.logical_block_id] = capacity.get(section.logical_block_id, 0) + section.capacity
    return sum(max(demand.get(key, 0) - capacity.get(key, 0), 0) for key in demand)


def _with_request_ids(requests):
    result = requests.copy()
    if REQUEST_ID_COLUMN not in result.columns:
        result.insert(0, REQUEST_ID_COLUMN, [f"{row.request_type}:{row.student_id}:{index:06d}:{row.course_id}" for index, row in enumerate(result.itertuples(index=False), start=1)])
    ids = result[REQUEST_ID_COLUMN].astype(str)
    if ids.duplicated().any() or not ids.all():
        raise ScenarioTransformError("request_id values must be non-empty and unique")
    return result


def _require_columns(frame, columns, name):
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ScenarioTransformError(f"{name} missing columns: {missing}")


def _percentage(transform):
    value = transform.get("percentage")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
        raise ScenarioTransformError("percentage must be in [0, 1]")
    return float(value)


def _positive_int(transform, field):
    value = transform.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ScenarioTransformError(f"{field} must be a positive integer")
    return value


def _pressure(demand, capacity):
    if capacity == 0:
        return float("inf") if demand else 0.0
    return demand / capacity


def _fingerprint_dict(fingerprint):
    return {key: getattr(fingerprint, key) for key in ("students", "logical_requests", "logical_primaries", "alternates", "logical_sections", "section_rows", "candidate_edges", "canonical_input_hash")}


def _dataframe_hashes(students, requests, sections):
    return _sha256_bytes(b"".join(_dataframe_bytes(frame) for frame in (students, requests, sections)))


def _dataframe_bytes(frame):
    return frame.to_csv(index=False, lineterminator="\n").encode("utf-8")


def _read_json(path):
    if not path.is_file():
        raise ScenarioTransformError(f"Missing metadata file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScenarioTransformError(f"Invalid metadata file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ScenarioTransformError(f"Metadata must be an object: {path}")
    return payload


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()
