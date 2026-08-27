from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tests.artifact_test_contract import (
    RequireExternalArtifact,
    require_external_artifact_path,
    resolve_external_artifact_root,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "external_artifact: verifies a separately retained historical experiment artifact",
    )


@pytest.fixture
def external_artifact_root() -> Path:
    return resolve_external_artifact_root()


@pytest.fixture
def require_external_artifact(
    external_artifact_root: Path,
) -> RequireExternalArtifact:
    return lambda relative_path=".": require_external_artifact_path(
        external_artifact_root, relative_path
    )
