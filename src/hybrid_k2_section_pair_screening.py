"""Hybrid K=2 Section-Pair Screening Audit v1.

This development-only runner screens the full frozen 312-section pair
universe for ``normal_dev_10`` before trying any fixed-pair solver runs. The
static screen is a necessary condition for the authoritative core student
only: if G12_0536 cannot satisfy the final-schedule student-local hard rules
even after a pair's complete non-original destination combinations are tried,
that pair cannot be a global two-section repair. Survivors are not global
feasibility evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ortools.sat.python import cp_model
from ortools.sat import cp_model_pb2

from src.allocation import canonicalize_allocation_input, math_course_ids_from_catalog, run_constrained_first_baseline
from src.allocation.cp_sat_solver import _VariableKey
from src.allocation.random_baseline import _build_mandatory_fallback_plans
from src.allocation.state import AllocationState
from src.benchmark_runner import _load_math_fallback_rules
from src.final_schedule_policy import evaluate_final_schedule_policy
from src.hybrid_stage1_incumbent_bootstrap import (
    SearchResult,
    apply_bootstrap_hints,
    hamming_expression,
    solve_bootstrap,
    validate_bootstrap_witness,
)
from src.joint_period_edit_pilot import (
    AUTHORITATIVE_STUDENT_ID,
    PlacementOption,
    _section_placement,
    _student_outcomes_for_solution,
    apply_placement_map_to_sections,
    build_frozen_placement_domains,
    build_joint_model,
)
from src.joint_period_edit_stage1_pilot import (
    frozen_domain_hashes,
    independent_production_validation,
    production_fixed_witness_acceptance,
    verify_checksums,
)
from src.period_placement_repair_probe import (
    DEFAULT_AUDIT_ROOT,
    DEFAULT_OUTPUT as DEFAULT_PREVIEW_OUTPUT,
    PERIODS,
    CandidateEdit,
    _candidate_from_dict,
    _requests_for_sections,
    _sha256_file,
    exact_student_level_analysis,
    load_scenario_context,
)
from src.section_plan_feasibility_audit import load_section_plan_audit_manifest


TARGET_SCENARIO_ID = "normal_dev_10"
EXCLUDED_STUDENT_ID = "G12_0105"
SOLVER_SEED = 20260630
WORKERS = 1
EXPECTED_EDITABLE_SECTION_COUNT = 312
EXPECTED_PLACEMENT_OPTION_COUNT = 841
EXPECTED_CANDIDATE_EDGE_COUNT = 164269
EXPECTED_UNIQUE_PAIR_COUNT = 48516
PORTFOLIO_SIZE_MAX = 6
FIXED_PAIR_FEASIBILITY_BUDGET_SECONDS = 75.0
FIXED_PAIR_GUIDED_BUDGET_SECONDS = 75.0
FIXED_WITNESS_BUDGET_SECONDS = 30.0
PRODUCTION_BUDGET_SECONDS = 300.0
RECOVERED_MODEL_PROTO_PROVENANCE = {
    "model_proto_origin": "reporting_only_rebuild_after_solver",
    "model_proto_is_original_solved_proto": False,
    "original_solved_proto_persisted": False,
    "model_proto_fingerprint_match_to_solved_model": "unverified",
    "model_rebuild_used_same_frozen_inputs_and_builder": True,
    "model_rebuild_solver_invocations": 0,
}
RECONSTRUCTED_SOLVER_CONFIG_PROVENANCE = {
    "solver_config_origin": "reconstructed_from_invoked_command_candidate_and_retained_evidence",
    "solver_config_original_pre_solve_file_persisted": False,
    "status_evidence_source": "raw_solver_log_final_summary",
    "runtime_evidence_source": "raw_solver_log_and_terminal_transcript",
}

DEFAULT_MANIFEST = Path("data/scenarios/hybrid_k2_section_pair_screening_v1.json")
DEFAULT_OUTPUT = Path(
    "/Users/klu/Projects/fair-course-allocation-artifacts/robustness-v1/"
    "hybrid-k2-section-pair-screening-v1"
)
ROBUSTNESS_ROOT = Path("/Users/klu/Projects/fair-course-allocation-artifacts/robustness-v1")
DEFAULT_K2_DIAGNOSTIC = ROBUSTNESS_ROOT / "hybrid-k2-search-bottleneck-diagnostic-v1"
DEFAULT_BOOTSTRAP = ROBUSTNESS_ROOT / "hybrid-stage1-incumbent-bootstrap-v1"
DEFAULT_SIZE_AUDIT = ROBUSTNESS_ROOT / "joint-stage1-model-size-reduction-audit-v1"
DEFAULT_SECTION_AUDITED = ROBUSTNESS_ROOT / "section-plan-feasibility-audit-v1-audited"
DEFAULT_CONTROL_AUDITED = ROBUSTNESS_ROOT / "joint-model-control-performance-audit-v1-audited"

EXPECTED_SOURCE_ARTIFACT_HASHES: dict[str, tuple[Path, str]] = {
    "hybrid_k2_search_bottleneck_diagnostic": (
        DEFAULT_K2_DIAGNOSTIC,
        "9a4bdfc3d46477541e7f5efb46fc301f13b2864bac6169210f594b6f8312c20f",
    ),
    "hybrid_stage1_incumbent_bootstrap": (
        DEFAULT_BOOTSTRAP,
        "528a13614477c0403c12626696e2f0cd394beb69bb5818316ea680ad0810773a",
    ),
    "joint_stage1_model_size_reduction_audit": (
        DEFAULT_SIZE_AUDIT,
        "5a5775ca7bff4054b034ab336d9ebc49fa6c11f9cc0e58cff94be8a34dbd3f80",
    ),
    "period_placement_repair_probe": (
        DEFAULT_PREVIEW_OUTPUT,
        "c43e00a74bbe513064b4d40839ad648b000f08757499f91ae0952d9e542ea6e2",
    ),
    "section_plan_feasibility_audit": (
        DEFAULT_AUDIT_ROOT,
        "1e9e899918d5f873e4a25bd4689153638b11dca3af113916212e5d3458369e18",
    ),
    "section_plan_feasibility_audit_audited": (
        DEFAULT_SECTION_AUDITED,
        "6a74f985b801606f7fb9c4323ea6853f8313a1e99db7e92388aad2a8f85dc015",
    ),
    "joint_model_control_performance_audit_audited": (
        DEFAULT_CONTROL_AUDITED,
        "f5eb6f8020180fbfa9fe706c1e99c0c8b40f6e592f9b1fc551d78e85fdbceae5",
    ),
}

PREVIOUSLY_EXCLUDED_PAIR = ("AP_3D_ART_DESIGN_01", "SOCIAL_JUSTICE_01")


class ScreeningError(ValueError):
    """Raised when the K=2 screening protocol must fail closed."""


@dataclass(frozen=True)
class CoreCandidate:
    request_key: str
    candidate_key: str
    period_units: int
    section_id: str
    logical_identity: str
    occupied_periods: tuple[str, ...]


@dataclass(frozen=True)
class CoreRequest:
    request_key: str
    candidate_key: str
    period_units: int
    candidates: tuple[CoreCandidate, ...]


@dataclass(frozen=True)
class CoreProfile:
    student_id: str
    target_period_units: int
    primary_requests: tuple[CoreRequest, ...]

    @property
    def core_candidate_section_ids(self) -> tuple[str, ...]:
        return tuple(sorted({candidate.section_id for request in self.primary_requests for candidate in request.candidates}))


@dataclass(frozen=True)
class CoreEvaluation:
    student_id: str
    primary_request_count: int
    target_period_units: int
    max_primary_assignments: int
    primary_unmet: int
    max_primary_period_units: int
    schedule_gap: int
    max_logical_gap: int
    selected_by_count: tuple[str, ...]
    selected_by_units: tuple[str, ...]
    student_local_feasible: bool
    fallback_used_for_primary: bool = False


@dataclass(frozen=True)
class SectionEffectSignature:
    logical_section_id: str
    logical_course_id: str
    original_placement: tuple[str, ...]
    non_original_placements: tuple[tuple[str, ...], ...]
    candidate_for_core_student: bool
    core_effect: str
    distinct_core_effect_signature_count: int
    effect_signature_hash: str


@dataclass(frozen=True)
class PairScreeningResult:
    pair_id: str
    section_id_a: str
    section_id_b: str
    course_id_a: str
    course_id_b: str
    final_class: str
    previously_proven_infeasible: bool
    core_neutral_pair: bool
    core_necessary_condition_failed: bool
    core_screen_survivor: bool
    invalid_domain_pair: bool
    screening_error: bool
    total_placement_combinations: int
    core_feasible_placement_combinations: int
    best_core_primary_unmet: int | None
    best_core_schedule_gap: int | None
    best_logical_assigned: int | None
    canonical_destinations: tuple[tuple[str, ...], tuple[str, ...]] | None
    canonical_sort_key: tuple[Any, ...] | None
    pair_sort_key: tuple[Any, ...] | None
    affected_student_union_count: int
    changed_candidate_period_relationships: int
    total_absolute_period_displacement: int
    evaluator_result_hash: str
    error: str = ""


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScreeningError(f"cannot read JSON: {path}: {exc}") from exc


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    values = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if not values:
        tmp.write_text("", encoding="utf-8")
        tmp.replace(path)
        return
    fields = tuple(dict.fromkeys(key for row in values for key in row))
    with tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(values)
    tmp.replace(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _json_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _hash_rows(rows: Iterable[Any]) -> str:
    return _json_hash(list(rows))


def write_checksums(root: Path) -> str:
    checksum = root / "SHA256SUMS.txt"
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != checksum:
            lines.append(f"{_sha256_file(path)}  {path.relative_to(root)}")
    checksum.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return _sha256_file(checksum)


def load_screening_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    payload = _read_json(Path(path))
    required = {
        "experiment_name",
        "experiment_version",
        "phase",
        "source_git_commit",
        "target_scenario_id",
        "authoritative_student_id",
        "excluded_student_ids",
        "editable_section_count",
        "placement_option_count",
        "candidate_edge_count",
        "expected_unique_pair_count",
        "source_k2_diagnostic_hash",
        "source_bootstrap_hash",
        "frozen_placement_domain_hash",
        "previously_excluded_unique_pair_count",
        "previously_excluded_unique_pairs",
        "selected_pair_portfolio_size_max",
        "fixed_pair_feasibility_budget_seconds",
        "fixed_pair_guided_budget_seconds",
        "production_fixed_witness_budget_seconds",
        "production_validation_budget_seconds",
        "solver_seed",
        "workers",
        "external_persisted_seed",
        "stop_after_first_valid_incumbent",
        "global_k2_allowed",
        "k1_allowed",
        "k3_allowed",
        "other_normal_targets_allowed",
        "stress_execution_allowed",
        "negative_execution_allowed",
        "holdout_execution_allowed",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise ScreeningError("screening manifest missing: " + ", ".join(missing))
    if payload["experiment_name"] != "hybrid_k2_section_pair_screening":
        raise ScreeningError("unexpected experiment_name")
    if payload["phase"] != "k2_unique_section_pair_screening":
        raise ScreeningError("unexpected screening phase")
    if payload["target_scenario_id"] != TARGET_SCENARIO_ID:
        raise ScreeningError("only normal_dev_10 is allowed")
    if payload["authoritative_student_id"] != AUTHORITATIVE_STUDENT_ID:
        raise ScreeningError("authoritative student must be G12_0536")
    if payload["excluded_student_ids"] != [EXCLUDED_STUDENT_ID]:
        raise ScreeningError("G12_0105 must be the only excluded student")
    if (int(payload["editable_section_count"]), int(payload["placement_option_count"]), int(payload["candidate_edge_count"])) != (
        EXPECTED_EDITABLE_SECTION_COUNT,
        EXPECTED_PLACEMENT_OPTION_COUNT,
        EXPECTED_CANDIDATE_EDGE_COUNT,
    ):
        raise ScreeningError("frozen domain counts drifted in manifest")
    if int(payload["expected_unique_pair_count"]) != EXPECTED_UNIQUE_PAIR_COUNT:
        raise ScreeningError("expected unique pair count must be 48516")
    if int(payload["solver_seed"]) != SOLVER_SEED or int(payload["workers"]) != WORKERS:
        raise ScreeningError("solver seed/workers drift")
    for field in (
        "external_persisted_seed",
        "global_k2_allowed",
        "k1_allowed",
        "k3_allowed",
        "other_normal_targets_allowed",
        "stress_execution_allowed",
        "negative_execution_allowed",
        "holdout_execution_allowed",
    ):
        if payload[field] is not False:
            raise ScreeningError(f"{field} must be false")
    if payload["stop_after_first_valid_incumbent"] is not True:
        raise ScreeningError("screening protocol must stop after the first valid incumbent")
    if int(payload["selected_pair_portfolio_size_max"]) != PORTFOLIO_SIZE_MAX:
        raise ScreeningError("portfolio size max drifted")
    for field, expected in (
        ("fixed_pair_feasibility_budget_seconds", FIXED_PAIR_FEASIBILITY_BUDGET_SECONDS),
        ("fixed_pair_guided_budget_seconds", FIXED_PAIR_GUIDED_BUDGET_SECONDS),
        ("production_fixed_witness_budget_seconds", FIXED_WITNESS_BUDGET_SECONDS),
        ("production_validation_budget_seconds", PRODUCTION_BUDGET_SECONDS),
    ):
        if float(payload[field]) != expected:
            raise ScreeningError(f"{field} is not frozen at {expected}")
    return payload


def verify_source_artifacts(manifest: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    manifest_fields = {
        "hybrid_k2_search_bottleneck_diagnostic": "source_k2_diagnostic_hash",
        "hybrid_stage1_incumbent_bootstrap": "source_bootstrap_hash",
        "joint_stage1_model_size_reduction_audit": "source_hybrid_model_size_hash",
        "period_placement_repair_probe": "source_candidate_preview_hash",
        "section_plan_feasibility_audit": "source_section_audit_hash",
        "section_plan_feasibility_audit_audited": "source_section_audited_hash",
        "joint_model_control_performance_audit_audited": "source_control_audited_hash",
    }
    for name, (root, expected_hash) in EXPECTED_SOURCE_ARTIFACT_HASHES.items():
        check = verify_checksums(root)
        if not check["passed"] or check["sha256"] != expected_hash:
            raise ScreeningError(f"source artifact verification failed: {name}")
        field = manifest_fields[name]
        if str(manifest[field]) != expected_hash:
            raise ScreeningError(f"manifest {field} does not match verified source artifact")
        result[name] = check | {"read_only": True}
    k2_aggregate = _read_json(DEFAULT_K2_DIAGNOSTIC / "aggregate_summary.json")
    if k2_aggregate.get("result_classification") != "unresolved_no_incumbent":
        raise ScreeningError("source K=2 diagnostic must remain unresolved")
    correction = _read_json(DEFAULT_K2_DIAGNOSTIC / "execution_history_correction.json")
    if correction.get("protocol_deviation") is not True:
        raise ScreeningError("source K=2 diagnostic execution-history correction missing")
    return result


def production_candidate_edge_count(allocation_input: Any, math_fallback_rules: tuple[Any, ...]) -> int:
    raw_edges = sum(len(candidates) for candidates in allocation_input.candidate_index.values())
    fallback_plans = _build_mandatory_fallback_plans(allocation_input, math_fallback_rules)
    fallback_edges = sum(len(plan.candidates) for plan in fallback_plans)
    return raw_edges + fallback_edges


def candidate_membership_hash(allocation_input: Any, math_fallback_rules: tuple[Any, ...]) -> str:
    membership = {key: list(value) for key, value in sorted(allocation_input.candidate_index.items())}
    for plan in _build_mandatory_fallback_plans(allocation_input, math_fallback_rules):
        membership[plan.fallback_request.request_key] = list(plan.candidates)
    return _json_hash(membership)


def section_capacity_hash(allocation_input: Any) -> str:
    return _hash_rows(
        [section.linked_section_group_id, section.capacity]
        for section in sorted(allocation_input.logical_sections, key=lambda item: item.linked_section_group_id)
    )


def logical_identity_hash(allocation_input: Any) -> str:
    return _hash_rows(
        [section.linked_section_group_id, section.logical_block_id]
        for section in sorted(allocation_input.logical_sections, key=lambda item: item.linked_section_group_id)
    )


def load_target_context_and_domains(
    *,
    preview_dir: str | Path = DEFAULT_PREVIEW_OUTPUT,
    audit_root: str | Path = DEFAULT_AUDIT_ROOT,
    config_dir: str | Path = "data/config",
) -> tuple[Any, dict[str, tuple[PlacementOption, ...]], Any]:
    audit_manifest = load_section_plan_audit_manifest("data/scenarios/section_plan_feasibility_audit_v1.json")
    context = load_scenario_context(
        TARGET_SCENARIO_ID,
        audit_manifest=audit_manifest,
        audit_root=Path(audit_root),
        config_dir=config_dir,
    )
    domains, summary = build_frozen_placement_domains(context, preview_dir)
    return context, domains, summary


def structural_revalidation(
    manifest: Mapping[str, Any],
    context: Any,
    domains: Mapping[str, tuple[PlacementOption, ...]],
    domain_summary: Any,
    *,
    config_dir: str | Path = "data/config",
) -> dict[str, Any]:
    rules = _load_math_fallback_rules(Path(config_dir), context.catalog)
    hashes = frozen_domain_hashes(domains, domain_summary.source_candidate_ids, context.allocation_input)
    hashes["section_capacity_hash"] = section_capacity_hash(context.allocation_input)
    hashes["logical_identity_hash"] = logical_identity_hash(context.allocation_input)
    hashes["candidate_membership_hash"] = candidate_membership_hash(context.allocation_input, rules)
    expected = {
        "editable_section_id_hash": manifest["editable_section_id_hash"],
        "placement_option_hash": manifest["placement_option_hash"],
        "frozen_placement_domain_hash": manifest["frozen_placement_domain_hash"],
        "original_placement_hash": manifest["original_placement_hash"],
        "section_capacity_hash": manifest["section_capacity_hash"],
        "logical_identity_hash": manifest["logical_identity_hash"],
        "candidate_membership_hash": manifest["candidate_membership_hash"],
    }
    failures = [key for key, value in expected.items() if hashes.get(key) != value]
    pair_count = len(domains) * (len(domains) - 1) // 2
    production_edges = production_candidate_edge_count(context.allocation_input, rules)
    count_failures = []
    if domain_summary.editable_logical_section_count != int(manifest["editable_section_count"]):
        count_failures.append("editable_section_count")
    if domain_summary.total_unique_placement_options != int(manifest["placement_option_count"]):
        count_failures.append("placement_option_count")
    if production_edges != int(manifest["candidate_edge_count"]):
        count_failures.append("candidate_edge_count")
    if pair_count != int(manifest["expected_unique_pair_count"]):
        count_failures.append("expected_unique_pair_count")
    if failures or count_failures:
        raise ScreeningError(f"structural revalidation drift: hashes={failures}; counts={count_failures}")
    return {
        "editable_section_count": domain_summary.editable_logical_section_count,
        "placement_option_count": domain_summary.total_unique_placement_options,
        "raw_candidate_edge_count": sum(len(value) for value in context.allocation_input.candidate_index.values()),
        "candidate_edge_count": production_edges,
        "unique_unordered_pair_count": pair_count,
        "expected_pair_formula": "312 * 311 / 2",
        "candidate_pruning": False,
        "preview_external_placement": False,
        "hashes": hashes,
        "passed": True,
    }


def build_core_profile(allocation_input: Any, student_id: str = AUTHORITATIVE_STUDENT_ID) -> CoreProfile:
    student = allocation_input.students_by_id[student_id]
    requests = []
    for request in student.primary_requests:
        candidates = []
        for section_id in allocation_input.candidate_index.get(request.request_key, ()):
            section = allocation_input.logical_sections_by_id.get(section_id)
            if section is None or section.logical_block_id != request.candidate_key:
                continue
            candidates.append(
                CoreCandidate(
                    request_key=request.request_key,
                    candidate_key=request.candidate_key,
                    period_units=int(request.period_units),
                    section_id=section_id,
                    logical_identity=section.logical_block_id,
                    occupied_periods=tuple(section.occupied_periods),
                )
            )
        requests.append(
            CoreRequest(
                request_key=request.request_key,
                candidate_key=request.candidate_key,
                period_units=int(request.period_units),
                candidates=tuple(candidates),
            )
        )
    return CoreProfile(
        student_id=student.student_id,
        target_period_units=int(student.target_period_units),
        primary_requests=tuple(requests),
    )


def _period_mask(periods: Iterable[str]) -> int:
    mask = 0
    for period in periods:
        if period not in PERIODS:
            raise ScreeningError(f"invalid period in placement: {period}")
        mask |= 1 << (int(period[1:]) - 1)
    return mask


def _best_core_assignment(
    profile: CoreProfile,
    placement_overrides: Mapping[str, tuple[str, ...]],
    *,
    objective: str,
) -> tuple[int, int, tuple[str, ...]]:
    memo: dict[tuple[int, int, tuple[str, ...]], tuple[int, int, tuple[str, ...]]] = {}

    def better(
        left: tuple[int, int, tuple[str, ...]],
        right: tuple[int, int, tuple[str, ...]],
    ) -> tuple[int, int, tuple[str, ...]]:
        if objective == "units":
            left_key = (left[1], left[0], tuple(reversed(left[2])))
            right_key = (right[1], right[0], tuple(reversed(right[2])))
        else:
            left_key = (left[0], left[1], tuple(reversed(left[2])))
            right_key = (right[0], right[1], tuple(reversed(right[2])))
        return left if left_key >= right_key else right

    def visit(index: int, used_periods: int, identities: tuple[str, ...]) -> tuple[int, int, tuple[str, ...]]:
        key = (index, used_periods, identities)
        if key in memo:
            return memo[key]
        if index == len(profile.primary_requests):
            return (0, 0, ())
        request = profile.primary_requests[index]
        best = visit(index + 1, used_periods, identities)
        identity_set = set(identities)
        for candidate in request.candidates:
            periods = tuple(placement_overrides.get(candidate.section_id, candidate.occupied_periods))
            occupied = _period_mask(periods)
            if occupied & used_periods or candidate.logical_identity in identity_set:
                continue
            tail = visit(index + 1, used_periods | occupied, tuple(sorted((*identities, candidate.logical_identity))))
            current = (
                tail[0] + 1,
                tail[1] + request.period_units,
                (request.candidate_key,) + tail[2],
            )
            best = better(best, current)
        memo[key] = best
        return best

    return visit(0, 0, ())


def evaluate_core_student(
    profile: CoreProfile,
    placement_overrides: Mapping[str, tuple[str, ...]] | None = None,
) -> CoreEvaluation:
    overrides = placement_overrides or {}
    max_count, _, selected_by_count = _best_core_assignment(profile, overrides, objective="count")
    _, max_units, selected_by_units = _best_core_assignment(profile, overrides, objective="units")
    primary_count = len(profile.primary_requests)
    primary_unmet = primary_count - max_count
    schedule_gap = max(profile.target_period_units - max_units, 0)
    max_logical_gap = primary_count - max_count
    feasible = primary_unmet <= 1 and schedule_gap <= 1 and max_count >= 5
    return CoreEvaluation(
        student_id=profile.student_id,
        primary_request_count=primary_count,
        target_period_units=profile.target_period_units,
        max_primary_assignments=max_count,
        primary_unmet=primary_unmet,
        max_primary_period_units=max_units,
        schedule_gap=schedule_gap,
        max_logical_gap=max_logical_gap,
        selected_by_count=tuple(selected_by_count),
        selected_by_units=tuple(selected_by_units),
        student_local_feasible=feasible,
    )


def zero_edit_core_student_verification(context: Any) -> dict[str, Any]:
    profile = build_core_profile(context.allocation_input, AUTHORITATIVE_STUDENT_ID)
    evaluation = evaluate_core_student(profile)
    legacy = exact_student_level_analysis(context.allocation_input, AUTHORITATIVE_STUDENT_ID)
    if evaluation.student_local_feasible:
        raise ScreeningError("zero-edit core student model is feasible; screening basis is invalid")
    return {
        "zero_edit_core_student_feasible": False,
        "authoritative_student_id": AUTHORITATIVE_STUDENT_ID,
        "request_universe": [request.candidate_key for request in profile.primary_requests],
        "fallback_satisfies_original_primary": False,
        "linked_gov_econ_logical_identity_checked": True,
        "ha_double_period_occupancy_checked": True,
        "duplicate_logical_identity_checked": True,
        "period_conflict_checked": True,
        "minimum_five_checked": True,
        "maximum_gap_one_checked": True,
        "new_evaluator": asdict(evaluation),
        "legacy_exact_student_level_analysis": legacy,
    }


def non_original_placements(options: Sequence[PlacementOption]) -> tuple[tuple[str, ...], ...]:
    return tuple(option.placement for option in options if not option.is_original)


def section_effect_signature(
    section_id: str,
    domains: Mapping[str, tuple[PlacementOption, ...]],
    profile: CoreProfile,
    allocation_input: Any | None = None,
) -> SectionEffectSignature:
    if section_id not in domains:
        raise ScreeningError(f"missing placement domain for section: {section_id}")
    options = domains[section_id]
    original = next((option.placement for option in options if option.is_original), None)
    if original is None:
        raise ScreeningError(f"placement domain lacks original option: {section_id}")
    non_original = non_original_placements(options)
    core_sections = set(profile.core_candidate_section_ids)
    relevant = section_id in core_sections
    signatures = []
    for placement in non_original:
        if relevant and placement != original:
            signatures.append((section_id, placement))
        else:
            signatures.append(())
    course_id = ""
    if allocation_input is not None and section_id in allocation_input.logical_sections_by_id:
        course_id = allocation_input.logical_sections_by_id[section_id].logical_block_id
    return SectionEffectSignature(
        logical_section_id=section_id,
        logical_course_id=course_id or section_id,
        original_placement=tuple(original),
        non_original_placements=tuple(non_original),
        candidate_for_core_student=relevant,
        core_effect="relevant" if relevant else "neutral",
        distinct_core_effect_signature_count=len(set(signatures)),
        effect_signature_hash=_json_hash(signatures),
    )


def enumerate_unique_pairs(section_ids: Sequence[str]) -> tuple[tuple[str, str], ...]:
    return tuple(combinations(sorted(section_ids), 2))


def core_mapping_key(profile: CoreProfile, overrides: Mapping[str, tuple[str, ...]]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    relevant = set(profile.core_candidate_section_ids)
    return tuple(sorted((section_id, tuple(placement)) for section_id, placement in overrides.items() if section_id in relevant))


def _placement_start(placement: tuple[str, ...]) -> int:
    return int(placement[0][1:])


def _pair_id(first: str, second: str) -> str:
    return f"{first}__{second}"


def _pair_course_ids(allocation_input: Any, first: str, second: str) -> tuple[str, str]:
    sections = allocation_input.logical_sections_by_id
    return (sections[first].logical_block_id, sections[second].logical_block_id)


def _affected_student_count(allocation_input: Any, section_ids: set[str]) -> int:
    return len({request.student_id for request in _requests_for_sections(allocation_input, section_ids)})


def section_to_primary_student_sets(allocation_input: Any) -> dict[str, set[str]]:
    result: dict[str, set[str]] = defaultdict(set)
    for request in allocation_input.logical_requests:
        if request.request_type != "primary":
            continue
        for section_id in allocation_input.candidate_index.get(request.request_key, ()):
            result[section_id].add(request.student_id)
    return result


def affected_student_count_for_pair(section_students: Mapping[str, set[str]], first: str, second: str) -> int:
    return len(set(section_students.get(first, set())) | set(section_students.get(second, set())))


def _relationship_change_count(profile: CoreProfile, overrides: Mapping[str, tuple[str, ...]]) -> int:
    candidates = {candidate.section_id: candidate.occupied_periods for request in profile.primary_requests for candidate in request.candidates}
    return sum(1 for section_id, placement in overrides.items() if section_id in candidates and tuple(placement) != tuple(candidates[section_id]))


def _destination_displacement(domains: Mapping[str, tuple[PlacementOption, ...]], overrides: Mapping[str, tuple[str, ...]]) -> int:
    total = 0
    for section_id, placement in overrides.items():
        original = next(option.placement for option in domains[section_id] if option.is_original)
        total += abs(_placement_start(original) - _placement_start(tuple(placement)))
    return total


def screen_pair(
    first: str,
    second: str,
    *,
    profile: CoreProfile,
    domains: Mapping[str, tuple[PlacementOption, ...]],
    allocation_input: Any,
    evaluation_cache: dict[tuple[tuple[str, tuple[str, ...]], ...], CoreEvaluation],
    section_students: Mapping[str, set[str]] | None = None,
    previously_excluded_pairs: set[tuple[str, str]] | None = None,
) -> PairScreeningResult:
    pair = tuple(sorted((first, second)))
    pair_id = _pair_id(*pair)
    previous = pair in (previously_excluded_pairs or set())
    try:
        first_options = non_original_placements(domains[first])
        second_options = non_original_placements(domains[second])
        course_a, course_b = _pair_course_ids(allocation_input, *pair)
        if not first_options or not second_options:
            return PairScreeningResult(
                pair_id,
                pair[0],
                pair[1],
                course_a,
                course_b,
                "invalid_domain_pair",
                previous,
                False,
                False,
                False,
                True,
                False,
                0,
                0,
                None,
                None,
                None,
                None,
                None,
                None,
                0,
                0,
                0,
                "",
                "section has no non-original placement option",
            )
        total = 0
        feasible_count = 0
        affected_count = (
            affected_student_count_for_pair(section_students, *pair)
            if section_students is not None
            else _affected_student_count(allocation_input, set(pair))
        )
        best_any: tuple[Any, CoreEvaluation, tuple[tuple[str, ...], tuple[str, ...]], dict[str, tuple[str, ...]]] | None = None
        best_feasible: tuple[Any, CoreEvaluation, tuple[tuple[str, ...], tuple[str, ...]], dict[str, tuple[str, ...]]] | None = None
        evaluated_hash_inputs = []
        for first_placement, second_placement in product(first_options, second_options):
            total += 1
            overrides = {pair[0]: tuple(first_placement), pair[1]: tuple(second_placement)}
            key = core_mapping_key(profile, overrides)
            if key not in evaluation_cache:
                evaluation_cache[key] = evaluate_core_student(profile, dict(key))
            evaluation = evaluation_cache[key]
            relationship_changes = _relationship_change_count(profile, overrides)
            displacement = _destination_displacement(domains, overrides)
            sort_key = (
                evaluation.primary_unmet,
                evaluation.schedule_gap,
                -evaluation.max_primary_assignments,
                affected_count,
                relationship_changes,
                displacement,
                tuple((sid, overrides[sid]) for sid in pair),
            )
            item = (sort_key, evaluation, (tuple(first_placement), tuple(second_placement)), overrides)
            if best_any is None or sort_key < best_any[0]:
                best_any = item
            if evaluation.student_local_feasible:
                feasible_count += 1
                if best_feasible is None or sort_key < best_feasible[0]:
                    best_feasible = item
            evaluated_hash_inputs.append((first_placement, second_placement, asdict(evaluation)))
        chosen = best_feasible or best_any
        if chosen is None:
            raise ScreeningError(f"no placements evaluated for {pair_id}")
        chosen_key, chosen_eval, chosen_destinations, chosen_overrides = chosen
        survivor = feasible_count > 0
        failed = feasible_count == 0
        neutral = not any(section_id in profile.core_candidate_section_ids for section_id in pair)
        final_class = (
            "previously_proven_infeasible"
            if previous
            else "core_screen_survivor"
            if survivor
            else "core_necessary_condition_failed"
        )
        pair_sort_key = (
            chosen_eval.primary_unmet,
            chosen_eval.schedule_gap,
            -feasible_count,
            affected_count,
            _relationship_change_count(profile, chosen_overrides),
            _destination_displacement(domains, chosen_overrides),
            pair_id,
        )
        return PairScreeningResult(
            pair_id=pair_id,
            section_id_a=pair[0],
            section_id_b=pair[1],
            course_id_a=course_a,
            course_id_b=course_b,
            final_class=final_class,
            previously_proven_infeasible=previous,
            core_neutral_pair=neutral,
            core_necessary_condition_failed=failed,
            core_screen_survivor=survivor,
            invalid_domain_pair=False,
            screening_error=False,
            total_placement_combinations=total,
            core_feasible_placement_combinations=feasible_count,
            best_core_primary_unmet=chosen_eval.primary_unmet,
            best_core_schedule_gap=chosen_eval.schedule_gap,
            best_logical_assigned=chosen_eval.max_primary_assignments,
            canonical_destinations=chosen_destinations,
            canonical_sort_key=chosen_key,
            pair_sort_key=pair_sort_key,
            affected_student_union_count=affected_count,
            changed_candidate_period_relationships=_relationship_change_count(profile, chosen_overrides),
            total_absolute_period_displacement=_destination_displacement(domains, chosen_overrides),
            evaluator_result_hash=_json_hash(evaluated_hash_inputs),
        )
    except Exception as exc:
        course_a, course_b = ("", "")
        if hasattr(allocation_input, "logical_sections_by_id") and first in allocation_input.logical_sections_by_id and second in allocation_input.logical_sections_by_id:
            course_a, course_b = _pair_course_ids(allocation_input, *pair)
        return PairScreeningResult(
            pair_id,
            pair[0],
            pair[1],
            course_a,
            course_b,
            "screening_error",
            previous,
            False,
            False,
            False,
            False,
            True,
            0,
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            0,
            0,
            0,
            "",
            str(exc),
        )


def pair_result_to_row(result: PairScreeningResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["canonical_destinations"] = json.dumps(result.canonical_destinations, sort_keys=True)
    payload["canonical_sort_key"] = json.dumps(result.canonical_sort_key, sort_keys=True, default=str)
    payload["pair_sort_key"] = json.dumps(result.pair_sort_key, sort_keys=True, default=str)
    return payload


def run_static_pair_screening(
    *,
    profile: CoreProfile,
    domains: Mapping[str, tuple[PlacementOption, ...]],
    allocation_input: Any,
    previously_excluded_pairs: set[tuple[str, str]],
) -> tuple[list[PairScreeningResult], dict[str, Any]]:
    started = time.perf_counter()
    pairs = enumerate_unique_pairs(tuple(domains))
    if len(pairs) != EXPECTED_UNIQUE_PAIR_COUNT:
        raise ScreeningError(f"unique pair count drift: {len(pairs)}")
    evaluation_cache: dict[tuple[tuple[str, tuple[str, ...]], ...], CoreEvaluation] = {}
    section_students = section_to_primary_student_sets(allocation_input)
    results: list[PairScreeningResult] = []
    total_combinations = 0
    for first, second in pairs:
        result = screen_pair(
            first,
            second,
            profile=profile,
            domains=domains,
            allocation_input=allocation_input,
            evaluation_cache=evaluation_cache,
            section_students=section_students,
            previously_excluded_pairs=previously_excluded_pairs,
        )
        total_combinations += result.total_placement_combinations
        results.append(result)
    final_classes = Counter(result.final_class for result in results)
    closed = sum(final_classes.values()) == EXPECTED_UNIQUE_PAIR_COUNT
    errors = sum(1 for result in results if result.screening_error)
    invalid = sum(1 for result in results if result.invalid_domain_pair)
    if not closed or errors or invalid:
        raise ScreeningError("pair screening did not close cleanly")
    summary = {
        "total_unique_pairs": len(results),
        "raw_placement_combination_count": total_combinations,
        "previously_proven_infeasible_pairs": sum(1 for result in results if result.previously_proven_infeasible),
        "core_neutral_pair_count": sum(1 for result in results if result.core_neutral_pair),
        "core_necessary_condition_failed_count": sum(1 for result in results if result.core_necessary_condition_failed),
        "core_screen_survivor_count": sum(1 for result in results if result.core_screen_survivor and not result.previously_proven_infeasible),
        "invalid_domain_pair_count": invalid,
        "screening_error_count": errors,
        "unclassified_pair_count": sum(1 for result in results if result.final_class == "unclassified"),
        "final_class_counts": dict(sorted(final_classes.items())),
        "class_count_closure": closed,
        "evaluator_cache_misses": len(evaluation_cache),
        "evaluator_cache_hits": total_combinations - len(evaluation_cache),
        "unique_effect_signature_count": len(evaluation_cache),
        "screening_runtime_seconds": round(time.perf_counter() - started, 6),
        "screening_hash": _json_hash([pair_result_to_row(result) for result in results]),
        "student_level_necessary_condition_only": True,
        "survivor_proves_global_feasible": False,
    }
    return results, summary


def section_effect_signatures(
    domains: Mapping[str, tuple[PlacementOption, ...]],
    profile: CoreProfile,
    allocation_input: Any,
) -> list[SectionEffectSignature]:
    return [
        section_effect_signature(section_id, domains, profile, allocation_input)
        for section_id in sorted(domains)
    ]


def section_effect_row(signature: SectionEffectSignature) -> dict[str, Any]:
    return {
        "logical_section_id": signature.logical_section_id,
        "logical_course_id": signature.logical_course_id,
        "original_placement": ";".join(signature.original_placement),
        "non_original_placements": json.dumps(signature.non_original_placements),
        "candidate_for_core_student": signature.candidate_for_core_student,
        "core_effect": signature.core_effect,
        "distinct_core_effect_signature_count": signature.distinct_core_effect_signature_count,
        "effect_signature_hash": signature.effect_signature_hash,
    }


def survivor_sort_key(result: PairScreeningResult) -> tuple[Any, ...]:
    if result.pair_sort_key is None:
        return (999, 999, 0, 999999, 999999, 999999, result.pair_id)
    return result.pair_sort_key


def candidate_from_pair_result(result: PairScreeningResult, domains: Mapping[str, tuple[PlacementOption, ...]]) -> CandidateEdit:
    if result.canonical_destinations is None:
        raise ScreeningError(f"survivor has no canonical destinations: {result.pair_id}")
    original = tuple(
        next(option.placement for option in domains[section_id] if option.is_original)
        for section_id in (result.section_id_a, result.section_id_b)
    )
    return CandidateEdit(
        candidate_id=f"k2_section_pair:{result.pair_id}:{_json_hash(result.canonical_destinations)[:12]}",
        edit_type="k2_section_pair_full_destination_domain",
        logical_section_ids=(result.section_id_a, result.section_id_b),
        logical_course_ids=(result.course_id_a, result.course_id_b),
        original_placements=original,
        proposed_placements=tuple(result.canonical_destinations),
        valid_period_source="frozen full non-original destination domain",
        occupancy_shape=tuple(tuple(len(periods) for periods in (placement,)) for placement in original),
        core_student=AUTHORITATIVE_STUDENT_ID,
        core_period_relevance=tuple(sorted({period for placement in (*original, *result.canonical_destinations) for period in placement})),
        affected_candidate_edge_count=int(result.changed_candidate_period_relationships),
        affected_student_count=int(result.affected_student_union_count),
    )


def select_pair_portfolio(
    results: Sequence[PairScreeningResult],
    domains: Mapping[str, tuple[PlacementOption, ...]],
    *,
    max_size: int = PORTFOLIO_SIZE_MAX,
) -> tuple[tuple[CandidateEdit, ...], dict[str, Any]]:
    survivors = sorted(
        (
            result for result in results
            if result.core_screen_survivor and not result.previously_proven_infeasible and not result.invalid_domain_pair and not result.screening_error
        ),
        key=survivor_sort_key,
    )
    relaxation_steps = []

    def choose(course_pair_cap: int, section_cap: int) -> list[PairScreeningResult]:
        selected: list[PairScreeningResult] = []
        course_pair_counts: Counter[tuple[str, str]] = Counter()
        section_counts: Counter[str] = Counter()
        for result in survivors:
            course_pair = tuple(sorted((result.course_id_a, result.course_id_b)))
            if course_pair_counts[course_pair] >= course_pair_cap:
                continue
            if section_counts[result.section_id_a] >= section_cap or section_counts[result.section_id_b] >= section_cap:
                continue
            selected.append(result)
            course_pair_counts[course_pair] += 1
            section_counts[result.section_id_a] += 1
            section_counts[result.section_id_b] += 1
            if len(selected) == max_size:
                break
        return selected

    selected = choose(1, 2)
    if len(selected) < max_size:
        relaxation_steps.append("same_course_pair_cap_relaxed_to_2")
        selected = choose(2, 2)
    if len(selected) < max_size:
        relaxation_steps.append("section_participation_cap_relaxed_to_3")
        selected = choose(2, 3)
    candidates = tuple(candidate_from_pair_result(result, domains) for result in selected)
    section_histogram = Counter(section_id for candidate in candidates for section_id in candidate.logical_section_ids)
    course_pairs = [tuple(sorted(candidate.logical_course_ids)) for candidate in candidates]
    audit = {
        "survivor_count": len(survivors),
        "portfolio_count": len(candidates),
        "portfolio_hash": _json_hash([asdict(candidate) for candidate in candidates]),
        "unique_section_id_pair_count": len({tuple(sorted(candidate.logical_section_ids)) for candidate in candidates}),
        "unique_course_pair_count": len(set(course_pairs)),
        "section_participation_histogram": dict(sorted(section_histogram.items())),
        "relaxation_steps": relaxation_steps,
        "previously_excluded_pair_in_portfolio": any(tuple(sorted(candidate.logical_section_ids)) == PREVIOUSLY_EXCLUDED_PAIR for candidate in candidates),
        "portfolio_frozen_before_solver": True,
    }
    if len({tuple(sorted(candidate.logical_section_ids)) for candidate in candidates}) != len(candidates):
        raise ScreeningError("portfolio contains duplicate section-ID pairs")
    if audit["previously_excluded_pair_in_portfolio"]:
        raise ScreeningError("previously excluded pair entered the portfolio")
    return candidates, audit


def portfolio_payload(candidates: Sequence[CandidateEdit], audit: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "count": len(candidates),
        "portfolio_hash": audit["portfolio_hash"],
        "candidates": [asdict(candidate) for candidate in candidates],
        "diversity_audit": dict(audit),
        "canonical_destination_hints_are_search_guidance_only": True,
    }


class _FirstSolutionCallback(cp_model.CpSolverSolutionCallback):
    def __init__(self, build: Any) -> None:
        super().__init__()
        self.build = build
        self.count = 0
        self.first_time: float | None = None
        self.assignments: tuple[tuple[str, str], ...] = ()
        self.placements: tuple[tuple[str, tuple[str, ...]], ...] = ()

    def on_solution_callback(self) -> None:
        self.count += 1
        if self.count != 1:
            return
        self.first_time = float(self.WallTime())
        self.assignments = _selected_assignments_from_values(self.build, self)
        self.placements = _selected_placements_from_values(self.build, self)


def _selected_assignments_from_values(build: Any, values: Any) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (key.request_key, key.section_id)
            for key, variable in build.assignment_vars.items()
            if values.BooleanValue(variable)
        )
    )


def _selected_placements_from_values(build: Any, values: Any) -> tuple[tuple[str, tuple[str, ...]], ...]:
    rows = []
    for section in build.allocation_input.logical_sections:
        section_id = section.linked_section_group_id
        options = build.placement_domains[section_id]
        if len(options) == 1:
            rows.append((section_id, options[0].placement))
        else:
            rows.append(
                (
                    section_id,
                    next(
                        option.placement
                        for option in options
                        if values.BooleanValue(build.placement_choice_vars[(section_id, option.placement)])
                    ),
                )
            )
    return tuple(rows)


def add_fixed_pair_constraints(build: Any, section_ids: tuple[str, str]) -> None:
    if len(set(section_ids)) != 2:
        raise ScreeningError("fixed pair must contain two distinct section IDs")
    missing = sorted(set(section_ids) - set(build.section_changed_vars))
    if missing:
        raise ScreeningError(f"fixed pair section missing from changed variables: {missing}")
    for section_id, variable in build.section_changed_vars.items():
        build.model.Add(variable == int(section_id in section_ids))


def solve_fixed_pair_no_hint(
    build: Any,
    *,
    run_id: str,
    seed: int = SOLVER_SEED,
    time_limit_seconds: float = FIXED_PAIR_FEASIBILITY_BUDGET_SECONDS,
) -> SearchResult:
    started = time.perf_counter()
    build.model.ClearObjective()
    proto = build.model.Proto()
    if getattr(proto, "objective", None) and proto.objective.vars:
        raise ScreeningError("Run A objective was not cleared before solve")
    if getattr(proto, "solution_hint", None) and proto.solution_hint.vars:
        raise ScreeningError("Run A solution hint must be empty before solve")
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_seconds)
    solver.parameters.num_search_workers = WORKERS
    solver.parameters.random_seed = int(seed)
    solver.parameters.stop_after_first_solution = True
    solver.parameters.log_search_progress = True
    logs: list[str] = []
    solver.log_callback = logs.append
    callback = _FirstSolutionCallback(build)
    status_code = solver.Solve(build.model, callback)
    status = solver.StatusName(status_code)
    response = solver.ResponseProto()
    available = status in {"FEASIBLE", "OPTIMAL"} or callback.count > 0
    if status in {"FEASIBLE", "OPTIMAL"}:
        assignments = _selected_assignments_from_values(build, solver)
        placements = _selected_placements_from_values(build, solver)
    else:
        assignments = callback.assignments
        placements = callback.placements
    return SearchResult(
        run_id=run_id,
        k=2,
        hint_id="none",
        status=status,
        assignment_available=available,
        incumbent_found=available,
        solution_count=callback.count,
        first_solution_time_seconds=callback.first_time,
        objective_value=None,
        best_bound=None,
        optimality_proven=status == "OPTIMAL",
        wall_time_seconds=float(solver.WallTime()),
        end_to_end_runtime_seconds=time.perf_counter() - started,
        deterministic_time_seconds=float(getattr(response, "deterministic_time", 0.0)),
        conflicts=int(getattr(response, "num_conflicts", solver.NumConflicts())),
        branches=int(getattr(response, "num_branches", solver.NumBranches())),
        propagations=int(getattr(response, "num_binary_propagations", 0)),
        integer_propagations=int(getattr(response, "num_integer_propagations", 0)),
        restarts=int(getattr(response, "num_restarts", 0)),
        response_hash=hashlib.sha256(str(response).encode("utf-8")).hexdigest(),
        selected_assignments=assignments,
        selected_placements=placements,
        solver_log=tuple(logs),
    )


def build_fixed_pair_model(
    allocation_input: Any,
    domains: Mapping[str, tuple[PlacementOption, ...]],
    candidate: CandidateEdit,
    *,
    math_fallback_rules: tuple[Any, ...],
    math_course_ids: tuple[str, ...],
) -> Any:
    build = build_joint_model(
        allocation_input,
        placement_domains=domains,
        math_fallback_rules=math_fallback_rules,
        math_course_ids=math_course_ids,
        occupancy_mode="hybrid_sparse_linear_occupancy",
    )
    add_fixed_pair_constraints(build, candidate.logical_section_ids)
    return build


def fixed_pair_feasibility_run(
    allocation_input: Any,
    domains: Mapping[str, tuple[PlacementOption, ...]],
    candidate: CandidateEdit,
    *,
    math_fallback_rules: tuple[Any, ...],
    math_course_ids: tuple[str, ...],
    seed: int = SOLVER_SEED,
    time_limit_seconds: float = FIXED_PAIR_FEASIBILITY_BUDGET_SECONDS,
) -> tuple[Any, dict[str, Any], SearchResult]:
    build = build_fixed_pair_model(
        allocation_input,
        domains,
        candidate,
        math_fallback_rules=math_fallback_rules,
        math_course_ids=math_course_ids,
    )
    hint = {
        "hint_used": False,
        "objective_used": False,
        "fixed_section_ids": list(candidate.logical_section_ids),
        "full_domain_preserved": True,
        "candidate_pruning": False,
        "model_restriction": "fixed_two_section_ids_changed_all_others_original_destinations_free",
        "feasible_region_same_as_guided_run": True,
    }
    result = solve_fixed_pair_no_hint(
        build,
        run_id=f"{candidate.candidate_id}:feasibility",
        seed=seed,
        time_limit_seconds=time_limit_seconds,
    )
    return build, hint, result


def fixed_pair_guided_run(
    context: Any,
    domains: Mapping[str, tuple[PlacementOption, ...]],
    candidate: CandidateEdit,
    *,
    config_dir: Path,
    math_fallback_rules: tuple[Any, ...],
    math_course_ids: tuple[str, ...],
    seed: int = SOLVER_SEED,
    time_limit_seconds: float = FIXED_PAIR_GUIDED_BUDGET_SECONDS,
) -> tuple[Any, dict[str, Any], SearchResult]:
    placement_map = dict(zip(candidate.logical_section_ids, candidate.proposed_placements))
    edited_sections = apply_placement_map_to_sections(context, placement_map)
    edited_input = canonicalize_allocation_input(
        context.students.copy(deep=True),
        context.requests.copy(deep=True),
        edited_sections,
        context.catalog.copy(deep=True),
    )
    baseline = run_constrained_first_baseline(
        edited_input,
        seed,
        math_fallback_rules=math_fallback_rules,
        math_course_ids=math_course_ids,
    )
    assignment_hint = tuple(
        _VariableKey(assignment.request_key, assignment.linked_section_group_id)
        for assignment in baseline.assignments
    )
    build = build_fixed_pair_model(
        context.allocation_input,
        domains,
        candidate,
        math_fallback_rules=math_fallback_rules,
        math_course_ids=math_course_ids,
    )
    build.model.Minimize(hamming_expression(build, assignment_hint))
    hint = apply_bootstrap_hints(build, candidate, assignment_hint)
    hint.update(
        {
            "hint_used": True,
            "objective_used": True,
            "assignment_hint_source": "edited_plan_constrained_first",
            "hamming_objective": "unweighted_assignment_distance_to_edited_plan_constrained_first",
            "fixed_section_ids": list(candidate.logical_section_ids),
            "full_domain_preserved": True,
            "candidate_pruning": False,
            "feasible_region_same_as_feasibility_run": True,
        }
    )
    result = solve_bootstrap(
        build,
        run_id=f"{candidate.candidate_id}:guided",
        k=2,
        hint_id=candidate.candidate_id,
        seed=seed,
        time_limit_seconds=time_limit_seconds,
    )
    return build, hint, result


def _model_size(build: Any, *, export_path: Path | None = None) -> dict[str, Any]:
    proto = build.model.Proto()
    proto_text_bytes = len(str(proto).encode("utf-8"))
    temp_path: Path | None = None
    if export_path is None:
        handle = tempfile.NamedTemporaryFile(prefix="hybrid_k2_model_", suffix=".pb", delete=False)
        handle.close()
        temp_path = Path(handle.name)
        export_path = temp_path
    export_path.parent.mkdir(parents=True, exist_ok=True)
    if not build.model.ExportToFile(str(export_path)):
        raise ScreeningError(f"failed to export model proto: {export_path}")
    exported_bytes = export_path.read_bytes()
    parsed = cp_model_pb2.CpModelProto()
    parsed.ParseFromString(exported_bytes)
    serialized_bytes = parsed.SerializeToString(deterministic=True)
    if temp_path is not None:
        temp_path.unlink(missing_ok=True)
    return {
        "total_variables": len(proto.variables),
        "total_constraints": len(proto.constraints),
        "binary_proto_bytes": len(serialized_bytes),
        "serialized_binary_proto_bytes": len(serialized_bytes),
        "exported_binary_proto_file_bytes": len(exported_bytes),
        "binary_measurements_equal": len(serialized_bytes) == len(exported_bytes),
        "proto_text_bytes": proto_text_bytes,
        "proto_measurement_method": "ExportToFile_pb_and_cp_model_pb2_deterministic_SerializeToString",
        "assignment_variables": len(build.assignment_vars),
        "placement_choice_variables": len(build.placement_choice_vars),
        "changed_section_variables": len(build.section_changed_vars),
        "occupancy_mode": getattr(build, "occupancy_mode", ""),
    }


def _response_payload(result: SearchResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["solver_log"] = None
    payload["response_hash_verified"] = bool(result.response_hash)
    if result.response_hash == "unavailable_artifact_write_failure_after_solver":
        payload["response_hash"] = None
        payload["response_hash_available"] = False
        payload["response_hash_verified"] = False
        payload["response_hash_unavailable_reason"] = (
            "post_solve_artifact_write_failure_before_structured_response_persistence"
        )
    payload["runtime_seconds"] = result.end_to_end_runtime_seconds
    return payload


def _raw_solver_log_final_status(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    status: str | None = None
    in_summary = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "CpSolverResponse summary:":
            in_summary = True
            continue
        if in_summary and stripped.startswith("status:"):
            status = stripped.split(":", 1)[1].strip()
    return status


def _write_solver_run(
    output: Path,
    pair_id: str,
    run_name: str,
    *,
    solver_config: Mapping[str, Any],
    hint_audit: Mapping[str, Any],
    model_size: Mapping[str, Any],
    response: Mapping[str, Any],
    validation: Mapping[str, Any],
    solver_log: Sequence[str],
) -> None:
    run_dir = output / "runs" / pair_id / run_name
    _write_json(run_dir / "solver_config.json", dict(solver_config))
    _write_json(run_dir / "hint_audit.json", dict(hint_audit))
    _write_json(run_dir / "model_size.json", dict(model_size))
    _write_json(run_dir / "response_stats.json", dict(response))
    (run_dir / "solver.log").write_text("\n".join(solver_log), encoding="utf-8")
    _write_json(run_dir / "validation.json", dict(validation))


@dataclass(frozen=True)
class DiagnosticOutcome:
    pair_id: str
    run_rows: tuple[dict[str, Any], ...]
    incumbent_source: dict[str, Any] | None
    newly_proven_infeasible: bool
    unresolved: bool
    correctness_failure: bool


def diagnostic_run_row(pair_id: str, run_name: str, result: SearchResult) -> dict[str, Any]:
    return {
        "pair_id": pair_id,
        "run_name": run_name,
        "status": result.status,
        "incumbent_found": result.incumbent_found,
        "assignment_available": result.assignment_available,
        "runtime_seconds": result.end_to_end_runtime_seconds,
        "wall_time_seconds": result.wall_time_seconds,
        "branches": result.branches,
        "conflicts": result.conflicts,
        "response_hash": result.response_hash,
    }


def run_fixed_pair_protocol(
    *,
    pair_index: int,
    candidate: CandidateEdit,
    context: Any,
    domains: Mapping[str, tuple[PlacementOption, ...]],
    math_fallback_rules: tuple[Any, ...],
    math_course_ids: tuple[str, ...],
    config_dir: Path,
    output: Path,
    allow_guided_run: bool = True,
) -> DiagnosticOutcome:
    pair_id = f"portfolio_pair_{pair_index + 1}"
    run_rows: list[dict[str, Any]] = []
    build_a, hint_a, result_a = fixed_pair_feasibility_run(
        context.allocation_input,
        domains,
        candidate,
        math_fallback_rules=math_fallback_rules,
        math_course_ids=math_course_ids,
        seed=SOLVER_SEED,
        time_limit_seconds=FIXED_PAIR_FEASIBILITY_BUDGET_SECONDS,
    )
    validation_a = {"validated": False, "reason": "validated only if this run produced an accepted incumbent"}
    response_a = _response_payload(result_a)
    _write_solver_run(
        output,
        pair_id,
        "feasibility",
        solver_config={
            "seed": SOLVER_SEED,
            "workers": WORKERS,
            "max_time_in_seconds": FIXED_PAIR_FEASIBILITY_BUDGET_SECONDS,
            "objective": "none",
            "hint": "none",
            "stop_after_first_solution": True,
            "external_persisted_seed": False,
            "fixed_section_ids": list(candidate.logical_section_ids),
        },
        hint_audit=hint_a,
        model_size=_model_size(build_a, export_path=output / "runs" / pair_id / "feasibility" / "model.pb"),
        response=response_a,
        validation=validation_a,
        solver_log=result_a.solver_log,
    )
    run_rows.append(diagnostic_run_row(pair_id, "feasibility", result_a))
    if result_a.status == "MODEL_INVALID":
        return DiagnosticOutcome(pair_id, tuple(run_rows), None, False, False, True)
    if result_a.incumbent_found:
        witness = validate_bootstrap_witness(context, build_a, result_a, config_dir=config_dir, k=2)
        witness["changed_logical_section_count_must_equal_2"] = int(witness.get("changed_logical_section_count", -1)) == 2
        return DiagnosticOutcome(
            pair_id,
            tuple(run_rows),
            {
                "run_name": "feasibility",
                "candidate": asdict(candidate),
                "placement_map": dict(result_a.selected_placements),
                "assignments": result_a.selected_assignments,
                "result": result_a,
                "build": build_a,
                "witness": witness,
            },
            False,
            False,
            False,
        )
    if result_a.status == "INFEASIBLE":
        return DiagnosticOutcome(pair_id, tuple(run_rows), None, True, False, False)
    if result_a.status != "UNKNOWN":
        return DiagnosticOutcome(pair_id, tuple(run_rows), None, False, True, False)
    if not allow_guided_run:
        return DiagnosticOutcome(pair_id, tuple(run_rows), None, False, True, False)

    build_b, hint_b, result_b = fixed_pair_guided_run(
        context,
        domains,
        candidate,
        config_dir=config_dir,
        math_fallback_rules=math_fallback_rules,
        math_course_ids=math_course_ids,
        seed=SOLVER_SEED,
        time_limit_seconds=FIXED_PAIR_GUIDED_BUDGET_SECONDS,
    )
    validation_b = {"validated": False, "reason": "validated only if this run produced an accepted incumbent"}
    response_b = _response_payload(result_b)
    _write_solver_run(
        output,
        pair_id,
        "guided",
        solver_config={
            "seed": SOLVER_SEED,
            "workers": WORKERS,
            "max_time_in_seconds": FIXED_PAIR_GUIDED_BUDGET_SECONDS,
            "objective": "hamming_to_edited_plan_constrained_first",
            "hint": "canonical_destination_plus_edited_plan_constrained_first",
            "stop_after_first_solution": True,
            "external_persisted_seed": False,
            "fixed_section_ids": list(candidate.logical_section_ids),
        },
        hint_audit=hint_b,
        model_size=_model_size(build_b, export_path=output / "runs" / pair_id / "guided" / "model.pb"),
        response=response_b,
        validation=validation_b,
        solver_log=result_b.solver_log,
    )
    run_rows.append(diagnostic_run_row(pair_id, "guided", result_b))
    if result_b.status == "MODEL_INVALID":
        return DiagnosticOutcome(pair_id, tuple(run_rows), None, False, False, True)
    if result_b.incumbent_found:
        witness = validate_bootstrap_witness(context, build_b, result_b, config_dir=config_dir, k=2)
        witness["changed_logical_section_count_must_equal_2"] = int(witness.get("changed_logical_section_count", -1)) == 2
        return DiagnosticOutcome(
            pair_id,
            tuple(run_rows),
            {
                "run_name": "guided",
                "candidate": asdict(candidate),
                "placement_map": dict(result_b.selected_placements),
                "assignments": result_b.selected_assignments,
                "result": result_b,
                "build": build_b,
                "witness": witness,
            },
            False,
            False,
            False,
        )
    if result_b.status == "INFEASIBLE":
        return DiagnosticOutcome(pair_id, tuple(run_rows), None, True, False, False)
    return DiagnosticOutcome(pair_id, tuple(run_rows), None, False, True, False)


def _minimum_claim(
    *,
    previous_k1_proof_verified: bool,
    witness_valid: bool,
    fixed_witness_accepted: bool,
    production_validated: bool,
) -> dict[str, Any]:
    if previous_k1_proof_verified and witness_valid and fixed_witness_accepted and production_validated:
        return {
            "claim": "minimum_changed_sections_within_frozen_placement_domain",
            "value": 2,
            "proven": True,
            "scope": "frozen_312_section_841_option_domain_only",
        }
    if witness_valid:
        return {"claim": "candidate_repair_found_pending_full_validation", "proven": False}
    return {"claim": "global_k2_unresolved_no_validated_incumbent", "proven": False, "lower_bound_remains": 2}


def _previous_k1_proof_verified() -> bool:
    aggregate = _read_json(DEFAULT_BOOTSTRAP / "aggregate_summary.json")
    return aggregate.get("stop_reason") == "k1_01_infeasible_global_cap_proof"


def _write_static_artifacts(
    output: Path,
    *,
    manifest: Mapping[str, Any],
    source_verification: Mapping[str, Any],
    structural: Mapping[str, Any],
    zero_edit: Mapping[str, Any],
    effects: Sequence[SectionEffectSignature],
    results: Sequence[PairScreeningResult],
    screening_summary: Mapping[str, Any],
    portfolio: Sequence[CandidateEdit],
    portfolio_audit: Mapping[str, Any],
) -> None:
    _write_json(output / "screening_manifest_snapshot.json", dict(manifest))
    _write_json(output / "source_artifact_verification.json", dict(source_verification))
    _write_json(output / "structural_revalidation.json", dict(structural))
    _write_json(output / "zero_edit_core_student_verification.json", dict(zero_edit))
    _write_csv(output / "section_effect_signatures.csv", [section_effect_row(item) for item in effects])
    _write_json(output / "pair_screening_summary.json", dict(screening_summary))
    _write_json(output / "screening_cache_stats.json", {
        "evaluator_cache_hits": screening_summary["evaluator_cache_hits"],
        "evaluator_cache_misses": screening_summary["evaluator_cache_misses"],
        "unique_effect_signature_count": screening_summary["unique_effect_signature_count"],
        "screening_runtime_seconds": screening_summary["screening_runtime_seconds"],
    })
    _write_csv(output / "all_pair_screening.csv", [pair_result_to_row(item) for item in results])
    survivor_rows = [pair_result_to_row(item) for item in results if item.core_screen_survivor and not item.previously_proven_infeasible]
    _write_csv(output / "survivor_pairs.csv", survivor_rows)
    _write_json(output / "selected_pair_portfolio.json", portfolio_payload(portfolio, portfolio_audit))
    _write_json(output / "portfolio_diversity_audit.json", dict(portfolio_audit))


def run_screening_audit(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    output_dir: str | Path = DEFAULT_OUTPUT,
    preview_dir: str | Path = DEFAULT_PREVIEW_OUTPUT,
    audit_root: str | Path = DEFAULT_AUDIT_ROOT,
    config_dir: str | Path = "data/config",
    resume: bool = False,
    screening_only: bool = False,
    max_new_solver_runs: int | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    resume_from_aggregate = False
    prior_static_only_sha = None
    existing_aggregate: dict[str, Any] | None = None
    if output.exists() and any(output.iterdir()):
        if not resume:
            raise ScreeningError(f"screening output is non-empty; refusing overwrite: {output}")
        aggregate = output / "aggregate_summary.json"
        if aggregate.is_file():
            existing_aggregate = _read_json(aggregate)
            prior_static_only_sha = existing_aggregate.get("sha256sums_hash")
            if max_new_solver_runs is None or max_new_solver_runs <= 0:
                return existing_aggregate | {"resumed": True, "solver_reexecuted": False}
            resume_from_aggregate = True
        checkpoint = output / "checkpoint.json"
        if not aggregate.is_file() and not checkpoint.is_file():
            raise ScreeningError("resume requested without checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    manifest = load_screening_manifest(manifest_path)
    context, domains, domain_summary = load_target_context_and_domains(
        preview_dir=preview_dir,
        audit_root=audit_root,
        config_dir=config_dir,
    )
    if resume_from_aggregate:
        source_verification = _read_json(output / "source_artifact_verification.json")
        structural = _read_json(output / "structural_revalidation.json")
        zero_edit = _read_json(output / "zero_edit_core_student_verification.json")
        screening_summary = _read_json(output / "pair_screening_summary.json")
        portfolio_payload_existing = _read_json(output / "selected_pair_portfolio.json")
        portfolio = tuple(_candidate_from_dict(item) for item in portfolio_payload_existing["candidates"])
        portfolio_audit = _read_json(output / "portfolio_diversity_audit.json")
    else:
        source_verification = verify_source_artifacts(manifest)
        structural = structural_revalidation(manifest, context, domains, domain_summary, config_dir=config_dir)
        zero_edit = zero_edit_core_student_verification(context)
        profile = build_core_profile(context.allocation_input, AUTHORITATIVE_STUDENT_ID)
        effects = section_effect_signatures(domains, profile, context.allocation_input)
        previous_pairs = {tuple(sorted(pair)) for pair in manifest["previously_excluded_unique_pairs"]}
        results, screening_summary = run_static_pair_screening(
            profile=profile,
            domains=domains,
            allocation_input=context.allocation_input,
            previously_excluded_pairs=previous_pairs,
        )
        portfolio, portfolio_audit = select_pair_portfolio(results, domains, max_size=int(manifest["selected_pair_portfolio_size_max"]))
        _write_static_artifacts(
            output,
            manifest=manifest,
            source_verification=source_verification,
            structural=structural,
            zero_edit=zero_edit,
            effects=effects,
            results=results,
            screening_summary=screening_summary,
            portfolio=portfolio,
            portfolio_audit=portfolio_audit,
        )

    diagnostic_rows: list[dict[str, Any]] = [dict(row) for row in _read_csv(output / "diagnostic_runs.csv")]
    failures: list[str] = []
    discovered_witness: dict[str, Any] = {"status": "not_run", "not_run_reason": "no_incumbent_found"}
    acceptance: dict[str, Any] = {"status": "not_run", "not_run_reason": "no_valid_joint_witness"}
    production: dict[str, Any] = {"status": "not_run", "not_run_reason": "fixed_witness_acceptance_not_passed"}
    accepted_source: dict[str, Any] | None = None
    newly_excluded: list[str] = []
    unresolved_pairs: list[str] = []
    counters = {
        "fixed_pair_feasibility_runs": 0,
        "fixed_pair_guided_runs": 0,
        "total_solver_invocations": 0,
        "global_k2_reruns": 0,
        "k1_runs": 0,
        "k3_runs": 0,
        "production_fixed_witness_acceptance_runs": 0,
        "production_validation_runs": 0,
        "control_runs": 0,
        "other_normal_target_runs": 0,
        "stress_runs": 0,
        "negative_runs": 0,
        "holdout_runs": 0,
        "stage2_runs": 0,
        "stage3_runs": 0,
        "stage4_runs": 0,
    }
    if existing_aggregate is not None:
        counters.update({key: int(value) for key, value in existing_aggregate.get("solver_counters", {}).items() if key in counters})

    _write_csv(output / "diagnostic_runs.csv", diagnostic_rows)
    _write_json(output / "discovered_witness.json", discovered_witness)
    _write_json(output / "production_fixed_witness_acceptance.json", acceptance)
    _write_json(output / "production_validation.json", production)
    _write_json(output / "failures.json", {"failures": failures, "unexpected_failure_count": 0})
    _write_json(output / "checkpoint.json", {
        "schema_version": 1,
        "screening_complete": True,
        "portfolio_frozen": True,
        "diagnostic_rows": diagnostic_rows,
        "counters": counters,
        "accepted": False,
        "validated": False,
    })

    new_solver_runs = 0
    if not screening_only:
        rules = _load_math_fallback_rules(Path(config_dir), context.catalog)
        math_ids = math_course_ids_from_catalog(context.catalog)
        completed = {(row.get("pair_id"), row.get("run_name")) for row in diagnostic_rows}
        for index, candidate in enumerate(portfolio):
            pair_id = f"portfolio_pair_{index + 1}"
            if (pair_id, "feasibility") in completed:
                continue
            if max_new_solver_runs is not None and new_solver_runs >= max_new_solver_runs:
                break
            outcome = run_fixed_pair_protocol(
                pair_index=index,
                candidate=candidate,
                context=context,
                domains=domains,
                math_fallback_rules=rules,
                math_course_ids=math_ids,
                config_dir=Path(config_dir),
                output=output,
                allow_guided_run=max_new_solver_runs is None,
            )
            diagnostic_rows.extend(outcome.run_rows)
            new_solver_runs += len(outcome.run_rows)
            counters["fixed_pair_feasibility_runs"] += sum(1 for row in outcome.run_rows if row["run_name"] == "feasibility")
            counters["fixed_pair_guided_runs"] += sum(1 for row in outcome.run_rows if row["run_name"] == "guided")
            counters["total_solver_invocations"] = counters["fixed_pair_feasibility_runs"] + counters["fixed_pair_guided_runs"]
            if outcome.correctness_failure:
                failures.append(f"{outcome.pair_id}:correctness_failure")
                break
            if outcome.incumbent_source is not None:
                accepted_source = outcome.incumbent_source
                discovered_witness = {
                    "status": "found",
                    "pair_id": outcome.pair_id,
                    "run_name": outcome.incumbent_source["run_name"],
                    "candidate": outcome.incumbent_source["candidate"],
                    "placement_map": {key: list(value) for key, value in outcome.incumbent_source["placement_map"].items()},
                    "selected_assignment_count": len(outcome.incumbent_source["assignments"]),
                    "response_hash": outcome.incumbent_source["result"].response_hash,
                    "joint_witness": outcome.incumbent_source["witness"],
                }
                break
            if outcome.newly_proven_infeasible:
                newly_excluded.append(outcome.pair_id)
            if outcome.unresolved:
                unresolved_pairs.append(outcome.pair_id)
            _write_csv(output / "diagnostic_runs.csv", diagnostic_rows)
            _write_json(output / "checkpoint.json", {
                "schema_version": 1,
                "screening_complete": True,
                "portfolio_frozen": True,
                "diagnostic_rows": diagnostic_rows,
                "counters": counters,
                "accepted": accepted_source is not None,
                "validated": False,
                "max_new_solver_runs": max_new_solver_runs,
                "new_solver_runs_this_invocation": new_solver_runs,
            })
            if max_new_solver_runs is not None and new_solver_runs >= max_new_solver_runs:
                break

    witness_valid = bool(
        discovered_witness.get("status") == "found"
        and discovered_witness.get("joint_witness", {}).get("joint_bootstrap_witness_valid")
        and discovered_witness.get("joint_witness", {}).get("changed_logical_section_count_must_equal_2")
    )
    downstream_validation_allowed = max_new_solver_runs is None
    if downstream_validation_allowed and accepted_source is not None and witness_valid:
        placement_map = accepted_source["placement_map"]
        assignments = accepted_source["assignments"]
        acceptance = production_fixed_witness_acceptance(
            context,
            placement_map,
            assignments,
            config_dir=Path(config_dir),
            seed=SOLVER_SEED,
            time_limit_seconds=FIXED_WITNESS_BUDGET_SECONDS,
        )
        counters["production_fixed_witness_acceptance_runs"] = 1
        acceptance["production_fixed_witness_accepted"] = bool(
            acceptance.get("status") in {"FEASIBLE", "OPTIMAL"}
            and acceptance.get("assignment_exact")
            and acceptance.get("policy_pass")
            and acceptance.get("consistency_issue_count") == 0
            and acceptance.get("response_hash")
        )
        if acceptance["production_fixed_witness_accepted"]:
            production = independent_production_validation(
                context,
                placement_map,
                config_dir=Path(config_dir),
                seed=SOLVER_SEED,
                time_limit_seconds=PRODUCTION_BUDGET_SECONDS,
            )
            counters["production_validation_runs"] = 1

    validated = bool(production.get("independently_validated_period_repair"))
    previous_k1 = _previous_k1_proof_verified()
    claim = _minimum_claim(
        previous_k1_proof_verified=previous_k1,
        witness_valid=witness_valid,
        fixed_witness_accepted=bool(acceptance.get("production_fixed_witness_accepted")),
        production_validated=validated,
    )
    result_classification = (
        "validated_exactly_two_section_repair"
        if validated
        else "incumbent_found_pending_or_failed_validation"
        if witness_valid
        else "unresolved_no_incumbent"
    )
    _write_csv(output / "diagnostic_runs.csv", diagnostic_rows)
    _write_json(output / "discovered_witness.json", discovered_witness)
    _write_json(output / "production_fixed_witness_acceptance.json", acceptance)
    _write_json(output / "production_validation.json", production)
    _write_json(output / "failures.json", {"failures": failures, "unexpected_failure_count": len(failures)})
    _write_json(output / "provenance.json", {
        "source_git_commit": manifest["source_git_commit"],
        "prior_static_only_sha256sums_hash": prior_static_only_sha,
        "exploratory_dry_runs": 1,
        "accepted_formal_static_screening_runs": 1,
        "total_static_screening_executions": 2,
        "new_solver_runs_this_invocation": new_solver_runs,
        "external_persisted_seed": False,
        "global_k2_reruns": 0,
        "k1_runs": 0,
        "k3_runs": 0,
        "control_runs": 0,
        "other_normal_target_runs": 0,
        "stress_runs": 0,
        "negative_runs": 0,
        "holdout_runs": 0,
        "stage2_runs": 0,
        "stage3_runs": 0,
        "stage4_runs": 0,
        "protocol_deviations": [],
        **counters,
    })
    aggregate = {
        "experiment_name": manifest["experiment_name"],
        "experiment_version": manifest["experiment_version"],
        "phase": manifest["phase"],
        "target_scenario_id": TARGET_SCENARIO_ID,
        "authoritative_student_id": AUTHORITATIVE_STUDENT_ID,
        "screening": screening_summary,
        "portfolio_count": len(portfolio),
        "portfolio_hash": portfolio_audit["portfolio_hash"],
        "newly_proven_infeasible_unique_pairs": newly_excluded,
        "unknown_pairs": unresolved_pairs,
        "untested_pair_count": max(len(portfolio) - len({row["pair_id"] for row in diagnostic_rows}), 0),
        "accepted": accepted_source is not None,
        "validated": validated,
        "minimum_claim": claim,
        "result_classification": result_classification,
        "previous_k1_proof_verified": previous_k1,
        "solver_counters": counters,
        "execution_counts": {
            "exploratory_dry_runs": 1,
            "accepted_formal_static_screening_runs": 1,
            "total_static_screening_executions": 2,
            "total_solver_invocations": counters["total_solver_invocations"],
            "new_solver_runs_this_invocation": new_solver_runs,
        },
        "global_k2_remains_unresolved": not validated,
        "stress_runs": 0,
        "negative_runs": 0,
        "holdout_runs": 0,
        "failures": failures,
    }
    _write_json(output / "aggregate_summary.json", aggregate)
    _write_json(output / "checkpoint.json", {
        "schema_version": 1,
        "screening_complete": True,
        "portfolio_frozen": True,
        "diagnostic_rows": diagnostic_rows,
        "counters": counters,
        "accepted": accepted_source is not None,
        "validated": validated,
        "aggregate_written": True,
        "max_new_solver_runs": max_new_solver_runs,
        "new_solver_runs_this_invocation": new_solver_runs,
    })
    checksum_hash = write_checksums(output)
    aggregate["sha256sums_hash"] = checksum_hash
    _write_json(output / "aggregate_summary.json", aggregate)
    checksum_hash = write_checksums(output)
    aggregate["sha256sums_hash"] = checksum_hash
    _write_json(output / "aggregate_summary.json", aggregate)
    write_checksums(output)
    return aggregate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Hybrid K=2 section-pair screening audit.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--screening-only", action="store_true")
    parser.add_argument("--max-new-solver-runs", type=int, default=None)
    args = parser.parse_args(argv)
    try:
        result = run_screening_audit(
            manifest_path=args.manifest,
            output_dir=args.output_dir,
            resume=args.resume,
            screening_only=args.screening_only,
            max_new_solver_runs=args.max_new_solver_runs,
        )
    except ScreeningError as exc:
        print(f"Hybrid K=2 section-pair screening FAIL: {exc}")
        return 1
    print("Hybrid K=2 section-pair screening PASS")
    print(json.dumps({
        "result_classification": result.get("result_classification"),
        "total_unique_pairs": result.get("screening", {}).get("total_unique_pairs"),
        "core_screen_survivor_count": result.get("screening", {}).get("core_screen_survivor_count"),
        "portfolio_count": result.get("portfolio_count"),
        "solver_counters": result.get("solver_counters"),
        "minimum_claim": result.get("minimum_claim"),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
