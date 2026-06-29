"""Canonical fixed-section allocation inputs and local assignment state.

This package intentionally stops at local fixed-section assignment mechanics.
It does not choose students, fill alternates, evaluate final outcomes, or run
global optimization.
"""

from .assignment_models import (
    AssignmentFeasibility,
    AssignmentRecord,
    AssignmentRejectionReason,
    AssignmentResult,
    StateConsistencyIssue,
)
from .input_adapter import canonicalize_allocation_input
from .input_models import (
    AllocationInputError,
    AllocationInputIssue,
    CanonicalAllocationInput,
    CanonicalStudent,
    CourseMetadata,
    LogicalRequest,
    LogicalSection,
    SectionMember,
    SourceRequestRow,
)
from .state import AllocationState

__all__ = [
    "AllocationState",
    "AssignmentFeasibility",
    "AllocationInputError",
    "AllocationInputIssue",
    "AssignmentRecord",
    "AssignmentRejectionReason",
    "AssignmentResult",
    "CanonicalAllocationInput",
    "CanonicalStudent",
    "CourseMetadata",
    "LogicalRequest",
    "LogicalSection",
    "SectionMember",
    "SourceRequestRow",
    "StateConsistencyIssue",
    "canonicalize_allocation_input",
]
