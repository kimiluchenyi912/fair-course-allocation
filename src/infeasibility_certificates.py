"""Fail-closed certificates for deliberately structural stress scenarios.

These certificates are diagnostics for synthetic negative controls.  They do
not run an allocator and do not claim that an ordinary stress scenario is
infeasible.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.allocation.input_models import CanonicalAllocationInput


CERTIFICATE_SCHEMA_VERSION = 1


def build_protected_primary_certificate(
    allocation_input: CanonicalAllocationInput,
    student_id: str,
    request_key: str,
) -> dict[str, Any]:
    student = allocation_input.students_by_id.get(student_id)
    request = allocation_input.requests_by_key.get(request_key)
    if student is None or request is None or request.request_type != "primary":
        raise ValueError("Protected certificate requires an existing primary request and student.")
    candidates = tuple(allocation_input.candidate_index.get(request_key, ()))
    if not student.priority_protected or request.student_id != student_id or candidates:
        raise ValueError("Protected primary certificate conditions are not satisfied.")
    return {
        "schema_version": CERTIFICATE_SCHEMA_VERSION,
        "certificate_type": "protected_primary_no_candidate",
        "expected_feasibility": "structurally_infeasible",
        "student_id": student_id,
        "primary_request_key": request_key,
        "candidate_logical_section_ids": list(candidates),
        "candidate_count": len(candidates),
        "target_period_units": student.target_period_units,
        "required_protected_primary_unmet": 0,
        "proof": "A protected primary has zero legal logical-section candidates.",
        "valid": True,
    }


def build_minimum_load_certificate(
    allocation_input: CanonicalAllocationInput,
    student_id: str,
) -> dict[str, Any]:
    student = allocation_input.students_by_id.get(student_id)
    if student is None:
        raise ValueError(f"Unknown certificate student: {student_id}")
    primary = [request for request in student.primary_requests]
    feasible_keys = sorted(
        {
            request.candidate_key
            for request in primary
            if allocation_input.candidate_index.get(request.request_key, ())
        }
    )
    if student.target_period_units < 5 or len(feasible_keys) > 4:
        raise ValueError("Minimum-load certificate conditions are not satisfied.")
    return {
        "schema_version": CERTIFICATE_SCHEMA_VERSION,
        "certificate_type": "minimum_logical_load_max_four",
        "expected_feasibility": "structurally_infeasible",
        "student_id": student_id,
        "target_period_units": student.target_period_units,
        "required_minimum_logical_courses": 5,
        "feasible_logical_course_keys": feasible_keys,
        "feasible_logical_course_count": len(feasible_keys),
        "proof": "The target requires at least five logical courses but at most four have candidates.",
        "valid": True,
    }


def build_global_capacity_certificate(
    allocation_input: CanonicalAllocationInput,
) -> dict[str, Any]:
    student_count = len(allocation_input.students)
    required_capacity = 5 * student_count
    total_capacity = sum(section.capacity for section in allocation_input.logical_sections)
    margin = total_capacity - required_capacity
    if margin >= 0:
        raise ValueError("Global capacity certificate requires a negative capacity margin.")
    return {
        "schema_version": CERTIFICATE_SCHEMA_VERSION,
        "certificate_type": "global_capacity_deficit",
        "expected_feasibility": "structurally_infeasible",
        "student_count": student_count,
        "required_minimum_capacity": required_capacity,
        "total_logical_seat_capacity": total_capacity,
        "capacity_margin": margin,
        "counting_method": "one capacity per logical section; linked semester rows are deduplicated; periods are ignored",
        "proof": "Total logical seat capacity is below five seats per student.",
        "valid": True,
    }


def validate_certificate(
    certificate: Mapping[str, Any],
    allocation_input: CanonicalAllocationInput,
) -> tuple[bool, str]:
    """Recheck a certificate against the transformed canonical input."""

    if certificate.get("schema_version") != CERTIFICATE_SCHEMA_VERSION:
        return False, "unsupported certificate schema"
    kind = certificate.get("certificate_type")
    if certificate.get("expected_feasibility") != "structurally_infeasible":
        return False, "certificate does not declare structural infeasibility"
    if kind == "protected_primary_no_candidate":
        student = allocation_input.students_by_id.get(str(certificate.get("student_id", "")))
        request_key = str(certificate.get("primary_request_key", ""))
        if student is None or not student.priority_protected:
            return False, "protected student is missing or not protected"
        request = allocation_input.requests_by_key.get(request_key)
        if request is None or request.request_type != "primary" or request.student_id != student.student_id:
            return False, "protected primary request is missing"
        if allocation_input.candidate_index.get(request_key, ()):
            return False, "protected primary still has candidates"
        return True, "protected primary has zero candidates"
    if kind == "minimum_logical_load_max_four":
        student = allocation_input.students_by_id.get(str(certificate.get("student_id", "")))
        if student is None:
            return False, "minimum-load student is missing"
        feasible_keys = {
            request.candidate_key
            for request in student.primary_requests
            if allocation_input.candidate_index.get(request.request_key, ())
        }
        if student.target_period_units < 5 or len(feasible_keys) > 4:
            return False, "minimum-load proof no longer holds"
        return True, "minimum-load proof holds"
    if kind == "global_capacity_deficit":
        required = 5 * len(allocation_input.students)
        actual = sum(section.capacity for section in allocation_input.logical_sections)
        if actual >= required:
            return False, "global logical capacity is no longer below the minimum"
        return True, "global logical capacity proof holds"
    return False, f"unknown certificate type: {kind}"
