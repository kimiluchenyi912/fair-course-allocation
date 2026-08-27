from __future__ import annotations

import os
from pathlib import Path
from typing import Callable

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EXTERNAL_ARTIFACT_ROOT = REPO_ROOT.parent / "fair-course-allocation-artifacts"


def resolve_external_artifact_root() -> Path:
    configured = os.environ.get("FCA_ARTIFACT_ROOT")
    return Path(configured).expanduser() if configured else DEFAULT_EXTERNAL_ARTIFACT_ROOT


def require_external_artifact_path(root: Path, relative_path: str | Path = ".") -> Path:
    if not root.exists():
        pytest.skip(
            "clean clone does not include external experiment artifacts "
            f"(missing root: {root})"
        )
    if not root.is_dir():
        pytest.fail(f"external artifact root is not a directory: {root}")
    target = root / relative_path
    if not target.exists():
        pytest.fail(f"external artifact root exists but required artifact is missing: {target}")
    return target


RequireExternalArtifact = Callable[[str | Path], Path]
