from __future__ import annotations

from pathlib import Path
from random import Random

import pandas as pd

from src.validation import validate_configuration

from .apportionment import largest_remainder
from .models import GenerationResult
from .request_generator import RequestBuilder
from .rules import load_generation_config
from .summary import build_metadata, build_summary


def generate_synthetic_dataset(
    config_dir: str | Path,
    scenario_id: str,
    seed: int,
    templates_dir: str | Path = "data/templates",
) -> GenerationResult:
    validation_report = validate_configuration(config_dir, templates_dir)
    if validation_report.errors:
        raise ValueError(validation_report.to_text())

    config = load_generation_config(config_dir, scenario_id)
    rng = Random(seed)
    students = generate_students(config.grade_profiles, rng)
    builder = RequestBuilder(config, seed)
    requests = builder.generate_requests(students)
    summary = build_summary(students, requests, config.catalog)
    metadata = build_metadata(students, requests, summary, scenario_id, seed)
    return GenerationResult(
        students=students,
        requests=requests,
        summary=summary,
        catalog=config.catalog,
        metadata=metadata,
        seed=seed,
        scenario_id=scenario_id,
    )


def generate_students(grade_profiles: pd.DataFrame, rng: Random) -> pd.DataFrame:
    rows: list[dict] = []
    for profile in grade_profiles.sort_values("grade").itertuples(index=False):
        grade = int(profile.grade)
        count = int(profile.student_count)
        load_counts = largest_remainder(
            count,
            {
                5: float(profile.share_5_classes),
                6: float(profile.share_6_classes),
                7: float(profile.share_7_classes),
            },
        )
        loads = [load for load, load_count in sorted(load_counts.items()) for _ in range(load_count)]
        rng.shuffle(loads)
        for index, target_load in enumerate(loads, start=1):
            rows.append(
                {
                    "student_id": f"G{grade:02d}_{index:04d}",
                    "grade": grade,
                    "target_course_count": target_load,
                    "unscheduled_preference": _unscheduled_preference(target_load),
                    "random_seed_group": f"S{index % 10}",
                    "priority_protected": "false",
                    "priority_reason": "",
                    "priority_valid_school_year": "",
                }
            )
    return pd.DataFrame(rows)


def _unscheduled_preference(target_load: int) -> str:
    if target_load == 7:
        return "none"
    if target_load == 6:
        return "either"
    return "afternoon"
