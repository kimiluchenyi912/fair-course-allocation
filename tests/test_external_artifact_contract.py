from __future__ import annotations

from pathlib import Path

import pytest

from tests.artifact_test_contract import require_external_artifact_path


def test_missing_external_artifact_root_skips_only_the_requesting_test(tmp_path: Path) -> None:
    with pytest.raises(pytest.skip.Exception, match="clean clone"):
        require_external_artifact_path(tmp_path / "not-present", "historical/run")


def test_existing_external_artifact_is_returned(tmp_path: Path) -> None:
    artifact = tmp_path / "historical" / "run"
    artifact.mkdir(parents=True)
    assert require_external_artifact_path(tmp_path, "historical/run") == artifact


def test_existing_root_with_missing_artifact_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(pytest.fail.Exception, match="required artifact is missing"):
        require_external_artifact_path(tmp_path, "historical/run")


def test_existing_non_directory_root_fails_closed(tmp_path: Path) -> None:
    root = tmp_path / "artifact-root"
    root.write_text("not a directory", encoding="utf-8")
    with pytest.raises(pytest.fail.Exception, match="not a directory"):
        require_external_artifact_path(root)


def test_ordinary_unit_test_does_not_depend_on_external_artifacts() -> None:
    assert sum((1, 2, 3)) == 6
