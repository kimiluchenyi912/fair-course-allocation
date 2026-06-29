from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from .config import load_section_planning_config
from .demand import conflict_graph, course_demands, validate_inputs
from .models import SectionPlanningResult, SectionPlanningError
from .period_layout import build_sections_and_layout_summary
from .section_counts import build_course_demand_summary


PLANNER_VERSION = "v1"
RULE_VERSION = "uniform_waitlist_expansion_with_high_demand_floor_v1"
UNMODELED_CONSTRAINTS = [
    "teacher availability by period",
    "teacher daily section load",
    "room inventory and room type",
    "science lab availability",
    "teacher-course qualifications",
]


def plan_sections(
    students: pd.DataFrame,
    requests: pd.DataFrame,
    config_dir: str | Path,
    scenario_id: str,
    seed: int,
    templates_dir: str | Path = "data/templates",
) -> SectionPlanningResult:
    config = load_section_planning_config(config_dir, templates_dir, scenario_id)
    validate_inputs(students, requests, config.catalog)
    demands = course_demands(config.catalog, requests)
    conflicts = conflict_graph(requests)
    course_summary = build_course_demand_summary(config, demands)
    sections, layout_summary, diagnostics = build_sections_and_layout_summary(
        config,
        course_summary,
        conflicts,
        seed,
    )
    metadata = _metadata(
        students,
        requests,
        sections,
        course_summary,
        diagnostics,
        scenario_id,
        seed,
    )
    return SectionPlanningResult(
        sections=sections,
        course_demand_summary=course_summary,
        period_layout_summary=layout_summary,
        metadata=metadata,
        scenario_id=scenario_id,
        seed=seed,
    )


def plan_sections_from_files(
    input_dir: str | Path,
    config_dir: str | Path,
    scenario_id: str,
    seed: int,
    templates_dir: str | Path = "data/templates",
) -> SectionPlanningResult:
    input_dir = Path(input_dir)
    students_path = input_dir / "students.csv"
    requests_path = input_dir / "requests.csv"
    if not students_path.exists() or not requests_path.exists():
        raise SectionPlanningError("input_dir must contain students.csv and requests.csv.")
    students = pd.read_csv(students_path, keep_default_na=False)
    requests = pd.read_csv(requests_path, keep_default_na=False)
    result = plan_sections(students, requests, config_dir, scenario_id, seed, templates_dir)
    result.metadata["input_file_hashes"] = {
        "students.csv": _file_hash(students_path),
        "requests.csv": _file_hash(requests_path),
    }
    return result


def write_result_atomic(result: SectionPlanningResult, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        result.sections.to_csv(temp_dir / "sections.csv", index=False)
        result.course_demand_summary.to_csv(temp_dir / "course_demand_summary.csv", index=False)
        result.period_layout_summary.to_csv(temp_dir / "period_layout_summary.csv", index=False)
        with (temp_dir / "section_planning_metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(result.metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")
        if output_dir.exists():
            shutil.rmtree(output_dir)
        temp_dir.replace(output_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise


def _metadata(
    students: pd.DataFrame,
    requests: pd.DataFrame,
    sections: pd.DataFrame,
    course_summary: pd.DataFrame,
    diagnostics: dict,
    scenario_id: str,
    seed: int,
) -> dict:
    primary = requests[requests["request_type"] == "primary"]
    return {
        "planner_version": PLANNER_VERSION,
        "rule_version": RULE_VERSION,
        "scenario_id": scenario_id,
        "seed": int(seed),
        "student_count": int(len(students)),
        "primary_request_rows": int(len(primary)),
        "total_section_rows": int(len(sections)),
        "total_logical_sections": int(course_summary["planned_sections"].sum()),
        "total_primary_demand": int(course_summary["primary_demand"].sum()),
        "total_planned_seats": int(course_summary["planned_seats"].sum()),
        "total_remaining_waitlist": int(course_summary["remaining_waitlist"].sum()),
        "raw_period_overlap_score": int(diagnostics["raw_period_overlap_score"]),
        "unavoidable_course_pair_conflict_score": int(diagnostics["unavoidable_course_pair_conflict_score"]),
        "single_period_course_count": int(diagnostics["single_period_course_count"]),
        "single_period_multi_section_course_count": int(diagnostics["single_period_multi_section_course_count"]),
        "period_balance_warnings": diagnostics["period_balance_warnings"],
        "unmodeled_constraints": UNMODELED_CONSTRAINTS,
    }


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()
