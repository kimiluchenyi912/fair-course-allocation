from __future__ import annotations

import json
from pathlib import Path

import pytest
from ortools.sat.python import cp_model

import src.all_student_k2_blocker_safe_screen as screen


REPO_ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(
    not all(
        (path / "SHA256SUMS.txt").is_file()
        for path in (
            screen.DEFAULT_SOURCE_SUITE,
            screen.DEFAULT_SOURCE_SCREEN,
            screen.DEFAULT_OUTPUT,
        )
    ),
    reason="external frozen K2 artifacts are not distributed with the repository",
)


def survivor_row(section_a: str, section_b: str) -> dict[str, str]:
    return {
        "pair_id": f"{section_a}__{section_b}",
        "section_id_a": section_a,
        "section_id_b": section_b,
        "course_id_a": section_a.removesuffix("_01"),
        "course_id_b": section_b.removesuffix("_01"),
        "final_class": "core_screen_survivor",
        "core_screen_survivor": "True",
    }


def counted_row(
    classification: str,
    *,
    tested: bool,
) -> dict[str, object]:
    return {
        "classification": classification,
        "previously_tested_selected_pair": tested,
    }


def formal_counts() -> dict[str, object]:
    rows = [
        *(counted_row("blocker_safe_excluded", tested=True) for _ in range(6)),
        *(counted_row("blocker_safe_excluded", tested=False) for _ in range(1219)),
        *(counted_row("blocker_screen_survivor", tested=False) for _ in range(12)),
    ]
    return screen.build_counts(rows)


def test_authoritative_g12_0105_blocker_proof_facts() -> None:
    proof, input_hashes = screen.verify_blocker_proof(repo_root=REPO_ROOT)

    assert proof["proof_verified"] is True
    assert all(fact["passed"] is True for fact in proof["facts"].values())
    assert proof["facts"]["student_is_ordinary"]["priority_protected"] == "false"
    assert proof["facts"]["ordinary_primary_unmet_limit_is_one"]["observed_limit"] == 1
    assert proof["facts"]["logical_primary_count_is_seven"]["logical_primary_count"] == 7
    assert proof["facts"]["each_blocker_primary_has_one_expected_candidate_section"][
        "candidate_sections"
    ] == {
        "CHINESE4": ["CHINESE4_01"],
        "FOOTBALL": ["FOOTBALL_01"],
        "INTERMEDIATE_ACTING": ["INTERMEDIATE_ACTING_01"],
    }
    assert proof["facts"]["all_blocker_sections_are_originally_p1"][
        "original_placements"
    ] == {
        "CHINESE4": ["P1"],
        "FOOTBALL": ["P1"],
        "INTERMEDIATE_ACTING": ["P1"],
    }
    assert input_hashes["source_suite_sha256sums"]["passed"] is True


def test_blocker_set_is_exact() -> None:
    assert screen.BLOCKER_SECTION_IDS == {
        "CHINESE4_01",
        "FOOTBALL_01",
        "INTERMEDIATE_ACTING_01",
    }


def test_disjoint_pair_is_safely_excluded() -> None:
    rows, diagnostics = screen.classify_static_survivors(
        [survivor_row("A_01", "B_01")],
        set(),
    )
    assert rows[0]["classification"] == "blocker_safe_excluded"
    assert diagnostics["classification_closure"] is True


@pytest.mark.parametrize("blocker", sorted(screen.BLOCKER_SECTION_IDS))
def test_pair_with_any_one_blocker_survives_without_containing_all_three(
    blocker: str,
) -> None:
    pair = (blocker, "OTHER_01")
    rows, _ = screen.classify_static_survivors(
        [survivor_row(*pair)],
        set(),
    )
    assert not screen.BLOCKER_SECTION_IDS.issubset(pair)
    assert rows[0]["classification"] == "blocker_screen_survivor"


def test_duplicate_and_closure_guards_fail_closed() -> None:
    duplicate = survivor_row("A_01", "B_01")
    with pytest.raises(screen.BlockerSafeScreenError, match="screening closure failed"):
        screen.classify_static_survivors([duplicate, duplicate], set())


def test_tested_untested_and_full_universe_counts_close_without_overlap() -> None:
    counts = formal_counts()
    assert counts == {
        "all_static_survivors": {
            "input_count": 1237,
            "blocker_safe_excluded": 1225,
            "blocker_screen_survivor": 12,
        },
        "previously_tested_selected_static_survivors": {
            "input_count": 6,
            "blocker_safe_excluded": 6,
            "blocker_screen_survivor": 0,
        },
        "untested_static_survivors": {
            "input_count": 1231,
            "blocker_safe_excluded": 1219,
            "blocker_screen_survivor": 12,
        },
    }
    closure = screen.build_universe_closure(counts)
    assert closure == {
        "original_static_necessary_condition_failed": 47278,
        "previously_proven_infeasible_unique_pair": 1,
        "new_all_student_blocker_safe_excluded": 1225,
        "remaining_not_excluded_by_current_safe_necessary_conditions": 12,
        "total_unique_pairs": 48516,
        "expected_total_unique_pairs": 48516,
        "closure_passed": True,
    }


def test_claim_scope_and_formal_projection_hash_comparison() -> None:
    counts = formal_counts()
    universe = screen.build_universe_closure(counts)
    same = screen.build_aggregate_summary(
        counts=counts,
        diagnostics={},
        universe=universe,
        formal_ordering_hash=screen.INFORMAL_PROJECTION_ORDERING_HASH,
    )
    different = screen.build_aggregate_summary(
        counts=counts,
        diagnostics={},
        universe=universe,
        formal_ordering_hash="different",
    )
    assert same["formal_and_informal_ordering_hash_match"] is True
    assert different["formal_and_informal_ordering_hash_match"] is False
    assert same["remaining_pair_claim"] == (
        "not excluded by current safe necessary conditions"
    )
    assert same["global_k2_status"] == "unresolved"
    assert same["exact_minimum_claim"] is None
    assert same["new_feasibility_results"] == 0


def test_real_solver_entrypoints_are_never_called(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("real solver invocation is forbidden")

    monkeypatch.setattr(cp_model.CpSolver, "solve", forbidden)
    monkeypatch.setattr(cp_model.CpSolver, "Solve", forbidden)

    proof, _ = screen.verify_blocker_proof(repo_root=REPO_ROOT)
    rows, _ = screen.classify_static_survivors(
        [survivor_row("A_01", "B_01")],
        set(),
    )
    assert proof["proof_verified"] is True
    assert rows[0]["classification"] == "blocker_safe_excluded"


def test_formal_artifact_checksum_and_claim_contract() -> None:
    verification = screen.verify_sha256sums(screen.DEFAULT_OUTPUT)
    summary = json.loads(
        (screen.DEFAULT_OUTPUT / "aggregate_summary.json").read_text(encoding="utf-8")
    )
    survivors = screen.read_csv(
        screen.DEFAULT_OUTPUT / "blocker_screen_survivors.csv"
    )
    files = {path.name for path in screen.DEFAULT_OUTPUT.iterdir() if path.is_file()}

    assert verification["file_count"] == 11
    assert verification["checksum_entry_count"] == 10
    assert files == {
        "SHA256SUMS.txt",
        "aggregate_summary.json",
        "all_static_survivor_classifications.csv",
        "blocker_definition.json",
        "blocker_safe_excluded_pairs.csv",
        "blocker_screen_survivors.csv",
        "failures.json",
        "input_hashes.json",
        "manifest.json",
        "proof_audit.json",
        "provenance.json",
    }
    assert not {"solver.log", "model.pb", "CpSolverResponse"} & files
    assert len(survivors) == 12
    assert all(row["feasibility_claim"] == "none" for row in survivors)
    assert summary["real_solver_invocations"] == 0
    assert summary["source_screening_reruns"] == 0
    assert summary["global_k2_remains_unresolved"] is True
