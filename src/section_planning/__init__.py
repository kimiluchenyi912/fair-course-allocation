"""Synthetic section count and period layout planning."""

from .models import SectionPlanningResult
from .runner import plan_sections

__all__ = ["SectionPlanningResult", "plan_sections"]
