"""Synthetic student and request generation."""

from .models import GenerationResult
from .student_generator import generate_synthetic_dataset

__all__ = ["GenerationResult", "generate_synthetic_dataset"]
