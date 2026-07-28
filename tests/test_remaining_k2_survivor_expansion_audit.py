from __future__ import annotations

import pytest
from ortools.sat.python import cp_model

import src.remaining_k2_survivor_expansion_audit as audit


def pair(pair_id: str, feasible: int = 5, affected: int = 10) -> audit.SurvivorPair:
    return audit.SurvivorPair(
        pair_id=pair_id,
        core_feasible_placement_combinations=feasible,
        affected_student_union_count=affected,
        changed_candidate_period_relationships=1,
        total_absolute_period_displacement=3,
    )


def proposal(
    classification: audit.RuleClass,
    *,
    proof_complete: bool,
    proof_sketch: str,
) -> audit.RuleProposal:
    return audit.RuleProposal(
        name="rule",
        classification=classification,
        mathematical_definition="x >= 1",
        affected_scope="one fixed pair",
        necessity_reason="every production solution must satisfy x >= 1",
        proof_sketch=proof_sketch,
        required_inputs=("x",),
        computational_complexity="O(1)",
        implementation_difficulty="low",
        false_exclusion_risk="none if implemented as proved",
        expected_survivor_reduction=audit.Estimate(None, "pairs", "unknown until cached-row evaluation"),
        required_data_exposed=True,
        proof_complete=proof_complete,
    )


def test_frozen_state_keeps_k2_unresolved() -> None:
    audit.validate_frozen_state()
    assert audit.REMAINING_UNTESTED_SURVIVORS == 1231
    assert audit.GLOBAL_K2_REMAINS_UNRESOLVED is True
    assert audit.PROVEN_LOWER_BOUND == 2
    assert audit.EXACT_MINIMUM_CLAIM is None


def test_evidence_classification_schema() -> None:
    finding = audit.EvidenceFinding(
        name="presolve",
        classification="directly_evidenced",
        finding="the raw log records an empty domain",
        sources=("solver.log",),
    )
    assert finding.classification == "directly_evidenced"
    with pytest.raises(ValueError, match="unsupported evidence classification"):
        audit.EvidenceFinding("bad", "guess", "claim", ("source",))  # type: ignore[arg-type]


def test_safe_exclusion_requires_a_complete_proof() -> None:
    safe = proposal(
        "safe_static_exclusion",
        proof_complete=True,
        proof_sketch="a production solution would contradict x >= 1",
    )
    assert safe.proof_complete is True
    with pytest.raises(ValueError, match="complete proof"):
        proposal("safe_static_exclusion", proof_complete=False, proof_sketch="")


def test_unproved_rule_cannot_be_marked_safe() -> None:
    heuristic = proposal("ranking_heuristic", proof_complete=False, proof_sketch="")
    assert heuristic.classification == "ranking_heuristic"
    with pytest.raises(ValueError, match="belongs in safe_static_exclusion"):
        proposal("ranking_heuristic", proof_complete=True, proof_sketch="proof")


def test_batch_order_and_hash_are_deterministic() -> None:
    pairs = (pair("B", feasible=5, affected=4), pair("A", feasible=10, affected=8), pair("C", feasible=10, affected=3))
    expected = ("C", "A", "B")
    assert tuple(item.pair_id for item in audit.deterministic_survivor_order(pairs)) == expected
    assert tuple(item.pair_id for item in audit.deterministic_survivor_order(reversed(pairs))) == expected
    assert audit.ordering_hash(pairs) == audit.ordering_hash(tuple(reversed(pairs)))


def test_completed_pair_is_never_repeated() -> None:
    pairs = (pair("A"), pair("B"), pair("C"))
    remaining = audit.remaining_after_completed(pairs, ("B", "B"))
    assert tuple(item.pair_id for item in remaining) == ("A", "C")
    with pytest.raises(ValueError, match="unknown completed"):
        audit.remaining_after_completed(pairs, ("D",))


@pytest.mark.parametrize("blocker", sorted(audit.BLOCKER_SECTION_IDS))
def test_any_one_blocker_section_satisfies_intersection_rule(blocker: str) -> None:
    fixed_pair = (blocker, "OTHER_01")
    assert not audit.BLOCKER_SECTION_IDS.issubset(fixed_pair)
    assert audit.passes_required_section_intersection(fixed_pair, audit.BLOCKER_SECTION_IDS)
    assert not audit.safely_excluded_by_blocker_rule(fixed_pair)


def test_pair_disjoint_from_blocker_set_is_safely_excluded() -> None:
    fixed_pair = ("A_01", "B_01")
    assert not audit.passes_required_section_intersection(fixed_pair, audit.BLOCKER_SECTION_IDS)
    assert audit.safely_excluded_by_blocker_rule(fixed_pair)


def test_blocker_rule_has_complete_proof_fields() -> None:
    rule = audit.g12_0105_blocker_rule()
    assert rule.classification == "safe_static_exclusion"
    assert rule.mathematical_definition == "pair_section_ids ∩ blocker_section_ids ≠ ∅"
    assert rule.proof_complete is True
    assert rule.proof_sketch
    assert rule.necessity_reason
    assert rule.required_inputs


def test_survivor_reduction_counts_are_only_an_informal_projection() -> None:
    assert audit.REMAINING_UNTESTED_SURVIVORS == 1231
    assert audit.PROJECTED_EXCLUDED_SURVIVORS == 1219
    assert audit.PROJECTED_REMAINING_SURVIVORS == 12
    assert audit.PROJECTED_REDUCTION_PERCENT == 99.0
    assert audit.PROJECTION_LABELS == {
        "exploratory_projection",
        "informal_projection",
        "not_a_formal_screening_artifact_result",
    }
    assert audit.PROJECTION_IS_FORMAL_SCREENING_RESULT is False
    assert audit.PROJECTED_ORDERING_HASH == (
        "f1246ec8582d6925e26fcc9ee53583a7ac275d6063781b2f59865af788573927"
    )


def test_storage_and_runtime_values_are_explicit_estimates() -> None:
    storage = audit.storage_projections(
        pair_count=1231,
        model_bytes=101_734_178,
        compact_bytes_per_pair=6_440,
        anomaly_rate=0.01,
    )
    runtime = audit.runtime_projection(pair_count=1231, model_build_seconds=11.1867, solver_seconds=3.1269)
    assert all(item.bytes.is_estimate and item.file_count.is_estimate for item in storage)
    assert runtime.is_estimate is True
    with pytest.raises(ValueError, match="explicitly labeled"):
        audit.Estimate(1, "bytes", "basis", is_estimate=False)


def test_analysis_contracts_make_no_real_solver_invocation(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("real solver invocation is forbidden in this audit")

    monkeypatch.setattr(cp_model.CpSolver, "solve", forbidden)
    monkeypatch.setattr(cp_model.CpSolver, "Solve", forbidden)

    audit.validate_frozen_state()
    audit.deterministic_survivor_order((pair("A"), pair("B")))
    audit.storage_projections(pair_count=12, model_bytes=101_734_178, compact_bytes_per_pair=6_440)
    assert audit.REAL_SOLVER_INVOCATIONS == 0
