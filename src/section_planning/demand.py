from __future__ import annotations

from itertools import combinations

import pandas as pd

from src.validation.helpers import text

from .models import SectionPlanningError


GOV_ECON_COURSES = {"GOV_ECON_REG", "GOV_APMACRO", "APGOV_ECON", "APGOV_APMACRO"}


def validate_inputs(students: pd.DataFrame, requests: pd.DataFrame, catalog: pd.DataFrame) -> None:
    required_student_cols = {"student_id", "grade", "target_course_count"}
    required_request_cols = {"student_id", "course_id", "request_type", "must_share_block_id"}
    if not required_student_cols.issubset(students.columns):
        missing = sorted(required_student_cols - set(students.columns))
        raise SectionPlanningError(f"students.csv missing columns: {missing}")
    if not required_request_cols.issubset(requests.columns):
        missing = sorted(required_request_cols - set(requests.columns))
        raise SectionPlanningError(f"requests.csv missing columns: {missing}")
    unknown_students = set(requests["student_id"]) - set(students["student_id"])
    if unknown_students:
        raise SectionPlanningError(f"requests.csv references unknown students: {sorted(unknown_students)[:5]}")
    unknown_courses = set(requests["course_id"]) - set(catalog["course_id"])
    if unknown_courses:
        raise SectionPlanningError(f"requests.csv references unknown courses: {sorted(unknown_courses)[:5]}")


def logical_primary_requests(requests: pd.DataFrame) -> pd.DataFrame:
    primary = requests[requests["request_type"] == "primary"].copy()
    primary["logical_block_id"] = primary.apply(logical_block_id_from_request, axis=1)
    return primary.drop_duplicates(["student_id", "logical_block_id"])


def course_demands(catalog: pd.DataFrame, requests: pd.DataFrame) -> pd.Series:
    logical = logical_primary_requests(requests)
    counts = logical.groupby("course_id")["student_id"].nunique()
    return catalog.set_index("course_id").index.to_series().map(counts).fillna(0).astype(int)


def conflict_graph(requests: pd.DataFrame) -> dict[tuple[str, str], int]:
    logical = logical_primary_requests(requests)
    weights: dict[tuple[str, str], int] = {}
    for _, group in logical.groupby("student_id"):
        courses = sorted(set(group["logical_block_id"]))
        for first, second in combinations(courses, 2):
            key = tuple(sorted((first, second)))
            weights[key] = weights.get(key, 0) + 1
    return weights


def logical_block_id_from_request(row: pd.Series) -> str:
    """Return the shared logical key used for planner and allocation inputs."""
    course_id = text(row["course_id"])
    block_id = text(row.get("must_share_block_id", ""))
    if course_id in GOV_ECON_COURSES:
        return block_id or course_id
    return block_id or course_id
