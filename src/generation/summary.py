from __future__ import annotations

import pandas as pd


RISK_COURSES = [
    "COMPUTER_PROGRAMMING",
    "AP_CSP",
    "AP_CSA",
    "AP_STATS",
    "AP_PHYSC",
    "CALC_D_LINALG",
]


def build_summary(students: pd.DataFrame, requests: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    rows.append({"metric": "total_students", "group": "all", "value": len(students)})
    for grade, count in students.groupby("grade").size().sort_index().items():
        rows.append({"metric": "students_by_grade", "group": str(grade), "value": int(count)})
    for (grade, load), count in students.groupby(["grade", "target_course_count"]).size().sort_index().items():
        rows.append({"metric": "target_load_by_grade", "group": f"{grade}:{load}", "value": int(count)})
    for request_type, count in requests.groupby("request_type").size().sort_index().items():
        rows.append({"metric": "request_rows", "group": request_type, "value": int(count)})

    primary = requests[requests["request_type"] == "primary"]
    for course_id in RISK_COURSES:
        rows.append(
            {
                "metric": "risk_course_primary_rows",
                "group": course_id,
                "value": int((primary["course_id"] == course_id).sum()),
            }
        )
    return pd.DataFrame(rows, columns=["metric", "group", "value"])


def build_metadata(
    students: pd.DataFrame,
    requests: pd.DataFrame,
    summary: pd.DataFrame,
    scenario_id: str,
    seed: int,
) -> dict:
    return {
        "scenario_id": scenario_id,
        "seed": seed,
        "total_students": int(len(students)),
        "primary_request_rows": int((requests["request_type"] == "primary").sum()),
        "alternate_request_rows": int((requests["request_type"] == "alternate").sum()),
        "summary_rows": int(len(summary)),
    }
