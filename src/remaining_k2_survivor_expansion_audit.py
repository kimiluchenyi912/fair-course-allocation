"""Pure analysis contracts for the Remaining K=2 Survivor Expansion Audit.

This module deliberately has no solver, artifact writer, or command-line
entry point.  It only makes the audit's proof, ordering, resume, and estimate
rules executable in unit tests.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Iterable, Literal, Sequence


EvidenceClass = Literal["directly_evidenced", "strongly_inferred", "unknown"]
RuleClass = Literal["safe_static_exclusion", "ranking_heuristic"]

TOTAL_UNIQUE_SECTION_PAIRS = 48_516
STATIC_CORE_SCREEN_SURVIVORS = 1_237
TESTED_PORTFOLIO_PAIRS = 6
REMAINING_UNTESTED_SURVIVORS = 1_231
SPECIFICALLY_EXCLUDED_UNIQUE_PAIRS = 7
GLOBAL_K2_REMAINS_UNRESOLVED = True
PROVEN_LOWER_BOUND = 2
EXACT_MINIMUM_CLAIM: None = None
REAL_SOLVER_INVOCATIONS = 0
BLOCKER_SECTION_IDS = frozenset(
    {"CHINESE4_01", "FOOTBALL_01", "INTERMEDIATE_ACTING_01"}
)
PROJECTED_EXCLUDED_SURVIVORS = 1_219
PROJECTED_REMAINING_SURVIVORS = 12
PROJECTED_REDUCTION_PERCENT = 99.0
PROJECTION_LABELS = frozenset(
    {
        "exploratory_projection",
        "informal_projection",
        "not_a_formal_screening_artifact_result",
    }
)
PROJECTION_IS_FORMAL_SCREENING_RESULT = False
PROJECTED_ORDERING_HASH = (
    "f1246ec8582d6925e26fcc9ee53583a7ac275d6063781b2f59865af788573927"
)


@dataclass(frozen=True)
class Estimate:
    value: float | int | None
    unit: str
    basis: str
    is_estimate: bool = True

    def __post_init__(self) -> None:
        if self.is_estimate is not True:
            raise ValueError("audit projections must remain explicitly labeled as estimates")
        if not self.unit or not self.basis:
            raise ValueError("estimate unit and basis are required")


@dataclass(frozen=True)
class EvidenceFinding:
    name: str
    classification: EvidenceClass
    finding: str
    sources: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.classification not in {"directly_evidenced", "strongly_inferred", "unknown"}:
            raise ValueError(f"unsupported evidence classification: {self.classification}")
        if not self.name or not self.finding or not self.sources:
            raise ValueError("evidence findings require a name, finding, and source")


@dataclass(frozen=True)
class RuleProposal:
    name: str
    classification: RuleClass
    mathematical_definition: str
    affected_scope: str
    necessity_reason: str
    proof_sketch: str
    required_inputs: tuple[str, ...]
    computational_complexity: str
    implementation_difficulty: str
    false_exclusion_risk: str
    expected_survivor_reduction: Estimate
    required_data_exposed: bool
    proof_complete: bool

    def __post_init__(self) -> None:
        if self.classification not in {"safe_static_exclusion", "ranking_heuristic"}:
            raise ValueError(f"unsupported rule classification: {self.classification}")
        required_text = (
            self.name,
            self.mathematical_definition,
            self.affected_scope,
            self.necessity_reason,
            self.computational_complexity,
            self.implementation_difficulty,
            self.false_exclusion_risk,
        )
        if not all(required_text) or not self.required_inputs:
            raise ValueError("rule proposal metadata is incomplete")
        if self.classification == "safe_static_exclusion":
            if not self.proof_complete or not self.proof_sketch:
                raise ValueError("safe_static_exclusion requires a complete proof")
        elif self.proof_complete:
            raise ValueError("a proved necessary rule belongs in safe_static_exclusion")


@dataclass(frozen=True)
class SurvivorPair:
    pair_id: str
    core_feasible_placement_combinations: int
    affected_student_union_count: int
    changed_candidate_period_relationships: int
    total_absolute_period_displacement: int

    def ranking_key(self) -> tuple[int, int, int, int, str]:
        return (
            -self.core_feasible_placement_combinations,
            self.affected_student_union_count,
            -self.changed_candidate_period_relationships,
            self.total_absolute_period_displacement,
            self.pair_id,
        )


def deterministic_survivor_order(pairs: Iterable[SurvivorPair]) -> tuple[SurvivorPair, ...]:
    values = tuple(pairs)
    ids = [pair.pair_id for pair in values]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate survivor pair_id")
    return tuple(sorted(values, key=SurvivorPair.ranking_key))


def ordering_hash(pairs: Sequence[SurvivorPair]) -> str:
    ordered = deterministic_survivor_order(pairs)
    payload = json.dumps(
        [asdict(pair) for pair in ordered],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def remaining_after_completed(
    pairs: Sequence[SurvivorPair],
    completed_pair_ids: Iterable[str],
) -> tuple[SurvivorPair, ...]:
    ordered = deterministic_survivor_order(pairs)
    known = {pair.pair_id for pair in ordered}
    completed = set(completed_pair_ids)
    unknown = completed - known
    if unknown:
        raise ValueError(f"checkpoint contains unknown completed pairs: {sorted(unknown)}")
    return tuple(pair for pair in ordered if pair.pair_id not in completed)


def passes_required_section_intersection(
    section_ids: tuple[str, str],
    required_sections: frozenset[str],
) -> bool:
    """Return the proved prerequisite for a pair to repair a fixed conflict."""
    return bool(set(section_ids) & required_sections)


def safely_excluded_by_blocker_rule(section_ids: tuple[str, str]) -> bool:
    """Exclude exactly pairs that do not intersect the proved blocker set."""
    return not passes_required_section_intersection(section_ids, BLOCKER_SECTION_IDS)


def g12_0105_blocker_rule() -> RuleProposal:
    return RuleProposal(
        name="g12_0105_three_p1_repair_section_intersection",
        classification="safe_static_exclusion",
        mathematical_definition="pair_section_ids ∩ blocker_section_ids ≠ ∅",
        affected_scope="frozen normal_dev_10 fixed-pair universe only",
        necessity_reason=(
            "without moving a blocker section, at most one of three P1-only "
            "primaries can be assigned although an ordinary student may miss at most one"
        ),
        proof_sketch=(
            "G12_0105 has seven primaries and needs at least six; the four other "
            "primaries plus at most one of the three P1-only primaries total at most five"
        ),
        required_inputs=(
            "ordinary student status",
            "primary request universe",
            "single candidate section per blocker primary",
            "original P1 section periods",
            "at-most-one logical course per student-period",
        ),
        computational_complexity="O(1) per pair after indexed preflight",
        implementation_difficulty="low",
        false_exclusion_risk="none when all frozen proof preconditions are verified",
        expected_survivor_reduction=Estimate(
            PROJECTED_EXCLUDED_SURVIVORS,
            "pairs",
            "informal cached-row projection; not a formal screening artifact result",
        ),
        required_data_exposed=True,
        proof_complete=True,
    )


@dataclass(frozen=True)
class StorageProjection:
    strategy: Literal["A", "B", "C"]
    bytes: Estimate
    file_count: Estimate


def storage_projections(
    *,
    pair_count: int,
    model_bytes: int,
    compact_bytes_per_pair: int,
    anomaly_rate: float = 0.0,
    delta_bytes_per_pair: int = 1_024,
) -> tuple[StorageProjection, ...]:
    if pair_count < 0 or model_bytes < 0 or compact_bytes_per_pair < 0:
        raise ValueError("storage inputs cannot be negative")
    if not 0.0 <= anomaly_rate <= 1.0:
        raise ValueError("anomaly_rate must be between zero and one")
    anomaly_models = math.ceil(pair_count * anomaly_rate)
    basis = f"{pair_count} pairs; observed model and compact-evidence sizes"
    return (
        StorageProjection(
            "A",
            Estimate(pair_count * (model_bytes + compact_bytes_per_pair), "bytes", basis),
            Estimate(pair_count * 7, "files", "seven observed files per run"),
        ),
        StorageProjection(
            "B",
            Estimate(
                model_bytes + pair_count * (compact_bytes_per_pair + delta_bytes_per_pair),
                "bytes",
                basis + "; one canonical base model plus deterministic deltas",
            ),
            Estimate(1 + pair_count * 7, "files", "one base plus compact evidence and deltas"),
        ),
        StorageProjection(
            "C",
            Estimate(
                pair_count * compact_bytes_per_pair + anomaly_models * model_bytes,
                "bytes",
                basis + f"; anomaly_rate={anomaly_rate:.4f}",
            ),
            Estimate(
                pair_count * 6 + anomaly_models,
                "files",
                "six compact files per run plus anomaly model files",
            ),
        ),
    )


def runtime_projection(
    *,
    pair_count: int,
    model_build_seconds: float,
    solver_seconds: float,
) -> Estimate:
    if pair_count < 0 or model_build_seconds < 0 or solver_seconds < 0:
        raise ValueError("runtime inputs cannot be negative")
    return Estimate(
        pair_count * (model_build_seconds + solver_seconds),
        "seconds",
        "pair count multiplied by observed/audited per-pair build and median solver times",
    )


@dataclass(frozen=True)
class BatchProtocol:
    batch_size: int = 12
    per_pair_time_limit_seconds: float = 75.0
    global_invocation_budget: int = 12
    checkpoint_interval_pairs: int = 4
    solver_seed: int = 20_260_630
    workers: int = 1

    def __post_init__(self) -> None:
        if min(self.batch_size, self.global_invocation_budget, self.checkpoint_interval_pairs, self.workers) <= 0:
            raise ValueError("batch protocol counts must be positive")
        if self.batch_size > self.global_invocation_budget:
            raise ValueError("batch size cannot exceed the global invocation budget")
        if self.per_pair_time_limit_seconds <= 0:
            raise ValueError("per-pair time limit must be positive")


def validate_frozen_state() -> None:
    if STATIC_CORE_SCREEN_SURVIVORS - TESTED_PORTFOLIO_PAIRS != REMAINING_UNTESTED_SURVIVORS:
        raise ValueError("remaining survivor count drifted")
    if SPECIFICALLY_EXCLUDED_UNIQUE_PAIRS != TESTED_PORTFOLIO_PAIRS + 1:
        raise ValueError("specific exclusion count drifted")
    if not GLOBAL_K2_REMAINS_UNRESOLVED or EXACT_MINIMUM_CLAIM is not None:
        raise ValueError("claim scope drifted")
    if PROJECTED_EXCLUDED_SURVIVORS + PROJECTED_REMAINING_SURVIVORS != REMAINING_UNTESTED_SURVIVORS:
        raise ValueError("projection count drifted")
    if PROJECTION_IS_FORMAL_SCREENING_RESULT:
        raise ValueError("projection was mislabeled as a formal screening result")
