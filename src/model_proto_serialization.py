"""Compatibility helpers for deterministic CP-SAT model serialization."""

from __future__ import annotations

import tempfile
from pathlib import Path

from ortools.sat import cp_model_pb2
from ortools.sat.python import cp_model


def deterministic_model_proto_bytes(
    model: cp_model.CpModel,
    *,
    export_path: str | Path | None = None,
) -> bytes:
    """Return deterministic protobuf bytes without relying on the pybind proto API."""
    temporary_path: Path | None = None
    if export_path is None:
        handle = tempfile.NamedTemporaryFile(prefix="cp_model_", suffix=".pb", delete=False)
        handle.close()
        temporary_path = Path(handle.name)
        export_path = temporary_path
    path = Path(export_path)
    try:
        if not model.ExportToFile(str(path)):
            raise ValueError(f"OR-Tools failed to export ModelProto: {path}")
        parsed = cp_model_pb2.CpModelProto()
        parsed.ParseFromString(path.read_bytes())
        return parsed.SerializeToString(deterministic=True)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
