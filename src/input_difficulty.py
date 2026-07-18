from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from src.allocation.input_models import CanonicalAllocationInput
from src.experiment_manifest import canonical_input_fingerprint


def build_input_difficulty(allocation_input: CanonicalAllocationInput) -> dict[str, Any]:
    """Describe a canonical fixed-section input without running an allocator."""
    fingerprint = canonical_input_fingerprint(allocation_input)
    primary = tuple(request for request in allocation_input.logical_requests if request.request_type == "primary")
    alternates = tuple(request for request in allocation_input.logical_requests if request.request_type == "alternate")
    candidates = [len(allocation_input.candidate_index.get(request.request_key, ())) for request in primary]
    primary_by_student: dict[str, list[Any]] = defaultdict(list)
    all_by_student: dict[str, list[Any]] = defaultdict(list)
    for request in primary:
        primary_by_student[request.student_id].append(request)
        all_by_student[request.student_id].append(request)
    for request in alternates:
        all_by_student[request.student_id].append(request)

    course_rows = _course_rows(allocation_input, primary, alternates)
    ratios = [row["primary_demand_capacity_ratio"] for row in course_rows if row["primary_demand_capacity_ratio"] is not None]
    over_demand = [row for row in course_rows if row["primary_demand"] > row["planned_capacity"]]
    primary_period_counts = {
        student_id: len(
            {
                period
                for request in requests
                for section_id in allocation_input.candidate_index.get(request.request_key, ())
                for period in allocation_input.logical_sections_by_id[section_id].occupied_periods
            }
        )
        for student_id, requests in primary_by_student.items()
    }

    return {
        "schema_version": 1,
        "scale": {
            "students": fingerprint.students,
            "logical_requests": fingerprint.logical_requests,
            "logical_primaries": fingerprint.logical_primaries,
            "alternates": fingerprint.alternates,
            "logical_sections": fingerprint.logical_sections,
            "section_rows": fingerprint.section_rows,
            "candidate_edges": fingerprint.candidate_edges,
            "canonical_input_hash": fingerprint.canonical_input_hash,
        },
        "student_load": {
            "target_logical_course_distribution": _counter_dict(
                Counter(student.target_period_units for student in allocation_input.students)
            ),
            "target_period_unit_distribution": _counter_dict(
                Counter(student.target_period_units for student in allocation_input.students)
            ),
            "grade_distribution": _counter_dict(Counter(student.grade for student in allocation_input.students)),
            "protected_student_count": sum(student.priority_protected for student in allocation_input.students),
            "students_with_5_targets": sum(student.target_period_units == 5 for student in allocation_input.students),
            "students_with_6_targets": sum(student.target_period_units == 6 for student in allocation_input.students),
            "students_with_7_targets": sum(student.target_period_units == 7 for student in allocation_input.students),
        },
        "request_flexibility": {
            "primary_candidate_sections": _numeric_summary(candidates),
            "primaries_with_zero_candidates": sum(value == 0 for value in candidates),
            "primaries_with_one_candidate": sum(value == 1 for value in candidates),
            "students_with_any_zero_candidate_primary": sum(
                any(not allocation_input.candidate_index.get(request.request_key, ()) for request in requests)
                for requests in primary_by_student.values()
            ),
            "alternate_count_distribution": _counter_dict(
                Counter(sum(request.request_type == "alternate" for request in all_by_student[student.student_id]) for student in allocation_input.students)
            ),
        },
        "demand_capacity": {
            "courses": course_rows,
            "courses_with_primary_demand_over_capacity": len(over_demand),
            "requests_exposed_to_over_demand_courses": sum(row["primary_demand"] for row in over_demand),
            "maximum_primary_demand_capacity_ratio": max(ratios) if ratios else None,
            "median_primary_demand_capacity_ratio": _percentile(ratios, 0.5),
            "p90_primary_demand_capacity_ratio": _percentile(ratios, 0.9),
            "total_capacity_only_primary_shortfall": sum(
                max(row["primary_demand"] - row["planned_capacity"], 0) for row in course_rows
            ),
            "capacity_only_shortfall_definition": (
                "A per-logical-course total-capacity lower bound only; it ignores period conflicts, "
                "shared sections, and combinations, and is not a proof of globally unmet demand."
            ),
        },
        "period_candidate_structure": {
            "requests_with_candidates_in_only_one_period": sum(
                _distinct_candidate_periods(allocation_input, request) == 1 for request in primary
            ),
            "requests_with_candidates_across_multiple_periods": sum(
                _distinct_candidate_periods(allocation_input, request) > 1 for request in primary
            ),
            "average_distinct_candidate_periods": round(
                sum(_distinct_candidate_periods(allocation_input, request) for request in primary) / len(primary), 6
            ) if primary else 0.0,
            "students_with_highly_concentrated_primary_candidate_periods": sum(
                value <= 1 for value in primary_period_counts.values()
            ),
            "limitation": (
                "Period concentration counts distinct occupied candidate periods and does not prove that "
                "a complete student schedule is feasible."
            ),
        },
    }


def _course_rows(allocation_input: CanonicalAllocationInput, primary: tuple[Any, ...], alternates: tuple[Any, ...]) -> list[dict[str, Any]]:
    primary_by_course: Counter[str] = Counter(request.candidate_key for request in primary)
    total_by_course: Counter[str] = Counter(request.candidate_key for request in primary + alternates)
    course_ids: dict[str, tuple[str, ...]] = {}
    for request in primary + alternates:
        course_ids.setdefault(request.candidate_key, request.course_ids)
    capacities: Counter[str] = Counter()
    for section in allocation_input.logical_sections:
        capacities[section.logical_block_id] += section.capacity
    rows = []
    for candidate_key in sorted(set(course_ids) | set(capacities)):
        capacity = int(capacities[candidate_key])
        demand = int(primary_by_course[candidate_key])
        rows.append(
            {
                "logical_course_key": candidate_key,
                "course_ids": list(course_ids.get(candidate_key, (candidate_key,))),
                "primary_demand": demand,
                "total_demand": int(total_by_course[candidate_key]),
                "planned_capacity": capacity,
                "primary_demand_capacity_ratio": round(demand / capacity, 6) if capacity else None,
            }
        )
    return rows


def _distinct_candidate_periods(allocation_input: CanonicalAllocationInput, request: Any) -> int:
    return len(
        {
            period
            for section_id in allocation_input.candidate_index.get(request.request_key, ())
            for period in allocation_input.logical_sections_by_id[section_id].occupied_periods
        }
    )


def _numeric_summary(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"min": 0, "median": 0, "p10": 0, "p90": 0, "max": 0}
    return {
        "min": min(values),
        "median": _percentile(values, 0.5),
        "p10": _percentile(values, 0.1),
        "p90": _percentile(values, 0.9),
        "max": max(values),
    }


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower), 6)


def _counter_dict(counter: Counter[Any]) -> dict[str, int]:
    return {str(key): int(counter[key]) for key in sorted(counter, key=lambda value: str(value))}
