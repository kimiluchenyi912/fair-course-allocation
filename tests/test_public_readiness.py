from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {"", ".csv", ".gitignore", ".json", ".md", ".py", ".txt", ".yml", ".yaml"}


def repository_text_files() -> list[Path]:
    files = [
        REPO_ROOT / name
        for name in ("AGENTS.md", "DECISIONS.md", "LICENSE", "PLAN.md", "PROGRESS.md", "README.md", "requirements.txt")
    ]
    for directory in (".github", "data", "docs", "src", "tests"):
        files.extend(
            path
            for path in (REPO_ROOT / directory).rglob("*")
            if path.is_file()
            and path.suffix in TEXT_SUFFIXES
            and "__pycache__" not in path.parts
            and "generated" not in path.parts
        )
    return sorted(set(files))


def test_tracked_content_has_no_developer_home_path() -> None:
    forbidden = "/Users/" + "klu/"
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in repository_text_files()
        if forbidden in path.read_text(encoding="utf-8")
    ]
    assert offenders == []


def test_scenario_manifests_are_parseable_and_portable() -> None:
    for path in sorted((REPO_ROOT / "data" / "scenarios").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        serialized = json.dumps(payload)
        assert "/Users/" not in serialized


def test_public_readme_has_required_release_context() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for heading in (
        "## Current evidence",
        "## Quickstart",
        "## Architecture",
        "## Data and limitations",
        "## License",
    ):
        assert heading in readme
    assert "2,630 synthetic students" in readme
    assert "Python 3.12" in readme
    assert "synthetic assumptions inspired" in readme
    assert "[MIT License](LICENSE)" in readme


def test_school_specific_inputs_have_a_clear_synthetic_disclaimer() -> None:
    config_notice = (REPO_ROOT / "data" / "config" / "README.md").read_text(encoding="utf-8")
    specification = (REPO_ROOT / "docs" / "SIMULATION_SPEC.md").read_text(encoding="utf-8")
    assert "synthetic assumptions inspired by a U.S. public" in config_notice
    assert "not official" in config_notice
    assert "public-source status has not been independently verified" in config_notice
    assert "synthetic assumptions inspired by a U.S." in specification


def test_tracked_content_has_no_school_identifiers() -> None:
    forbidden = (
        "T" + "PHS",
        "SDU" + "HSD",
        "Torrey " + "Pines",
        "San " + "Dieguito",
        "sdu" + "hsd.net",
    )
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in repository_text_files()
        if any(value.lower() in path.read_text(encoding="utf-8").lower() for value in forbidden)
    ]
    assert offenders == []


def test_infeasibility_claims_are_scoped_to_the_frozen_model() -> None:
    ambiguous = "globally " + "infeasible"
    ambiguous_key = "globally_" + "infeasible"
    offenders = [
        str(path.relative_to(REPO_ROOT))
        for path in repository_text_files()
        if ambiguous.lower() in path.read_text(encoding="utf-8").lower()
        or ambiguous_key in path.read_text(encoding="utf-8").lower()
    ]
    assert offenders == []


def test_ci_runs_full_tests_and_both_validators_on_python_312() -> None:
    workflow = (REPO_ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert 'python-version: "3.12"' in workflow
    assert "python -m pytest -q" in workflow
    assert "python -m src.validation\n" in workflow
    assert "python -m src.validation --strict-policy" in workflow
