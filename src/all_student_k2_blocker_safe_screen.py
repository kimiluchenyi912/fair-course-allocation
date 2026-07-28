"""Formal all-student K=2 blocker necessary-condition screen.

This module reads frozen inputs and an accepted static-screen artifact. It
contains no model builder, solver call, or feasibility search.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = 1
TARGET_SCENARIO_ID = "normal_dev_10"
AUTHORITATIVE_STUDENT_ID = "G12_0105"
BLOCKER_COURSE_TO_SECTION = {
    "CHINESE4": "CHINESE4_01",
    "FOOTBALL": "FOOTBALL_01",
    "INTERMEDIATE_ACTING": "INTERMEDIATE_ACTING_01",
}
BLOCKER_SECTION_IDS = frozenset(BLOCKER_COURSE_TO_SECTION.values())
TOTAL_UNIQUE_PAIRS = 48_516
ORIGINAL_STATIC_EXCLUSIONS = 47_278
PREVIOUSLY_PROVEN_INFEASIBLE = 1
INPUT_STATIC_SURVIVORS = 1_237
TESTED_SELECTED_SURVIVORS = 6
INFORMAL_PROJECTION_ORDERING_HASH = (
    "f1246ec8582d6925e26fcc9ee53583a7ac275d6063781b2f59865af788573927"
)
DEFAULT_SOURCE_SUITE = Path(
    "/Users/klu/Projects/fair-course-allocation-artifacts/robustness-v1/"
    "normal-development-v1"
)
DEFAULT_SOURCE_SCREEN = Path(
    "/Users/klu/Projects/fair-course-allocation-artifacts/robustness-v1/"
    "hybrid-k2-section-pair-screening-v1"
)
DEFAULT_OUTPUT = Path(
    "/Users/klu/Projects/fair-course-allocation-artifacts/robustness-v1/"
    "all-student-k2-blocker-safe-screen-v1"
)


class BlockerSafeScreenError(ValueError):
    """Raised before accepting an incomplete or ambiguous safe screen."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_sha256sums(root: Path) -> dict[str, Any]:
    checksum_path = root / "SHA256SUMS.txt"
    if not checksum_path.is_file():
        raise BlockerSafeScreenError(f"missing checksum file: {checksum_path}")
    failures: list[str] = []
    seen: set[str] = set()
    entries = 0
    for line_number, line in enumerate(checksum_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            continue
        try:
            expected, relative_text = line.split("  ", 1)
        except ValueError as exc:
            raise BlockerSafeScreenError(f"malformed checksum line {line_number}") from exc
        relative = Path(relative_text)
        if relative.is_absolute() or ".." in relative.parts or relative_text in seen:
            raise BlockerSafeScreenError(f"unsafe or duplicate checksum path: {relative_text}")
        seen.add(relative_text)
        entries += 1
        target = root / relative
        if not target.is_file() or sha256_file(target) != expected:
            failures.append(relative_text)
    files = sorted(path for path in root.rglob("*") if path.is_file())
    uncovered = sorted(
        str(path.relative_to(root))
        for path in files
        if path != checksum_path and str(path.relative_to(root)) not in seen
    )
    if failures or uncovered or len(files) != entries + 1:
        raise BlockerSafeScreenError(
            f"checksum verification failed: failures={failures}, uncovered={uncovered}"
        )
    return {
        "root": str(root),
        "file_count": len(files),
        "checksum_entry_count": entries,
        "sha256sums_sha256": sha256_file(checksum_path),
        "passed": True,
    }


def _function_source(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    matches = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    ]
    if len(matches) != 1:
        raise BlockerSafeScreenError(f"expected one {name} definition in {path}")
    return ast.unparse(matches[0])


def _integer_constant(path: Path, name: str) -> int:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, int):
                return value
    raise BlockerSafeScreenError(f"integer constant {name} not found in {path}")


def _proof_input_paths(repo_root: Path, source_suite: Path) -> dict[str, Path]:
    scenario = source_suite / "scenarios" / TARGET_SCENARIO_ID
    return {
        "students_csv": scenario / "generated" / "students.csv",
        "requests_csv": scenario / "generated" / "requests.csv",
        "sections_csv": scenario / "sections" / "sections.csv",
        "canonical_input_fingerprint": scenario / "input_fingerprint.json",
        "course_catalog": repo_root / "data" / "config" / "course_catalog.csv",
        "input_adapter_source": repo_root / "src" / "allocation" / "input_adapter.py",
        "final_schedule_policy_source": repo_root / "src" / "final_schedule_policy.py",
        "production_solver_source": repo_root / "src" / "allocation" / "cp_sat_solver.py",
        "fixed_pair_screen_source": repo_root / "src" / "hybrid_k2_section_pair_screening.py",
    }


def verify_blocker_proof(
    *,
    repo_root: Path,
    source_suite: Path = DEFAULT_SOURCE_SUITE,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_verification = verify_sha256sums(source_suite)
    paths = _proof_input_paths(repo_root, source_suite)
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise BlockerSafeScreenError(f"missing proof inputs: {missing}")

    students = read_csv(paths["students_csv"])
    requests = read_csv(paths["requests_csv"])
    sections = read_csv(paths["sections_csv"])
    student_rows = [row for row in students if row["student_id"] == AUTHORITATIVE_STUDENT_ID]
    primary_rows = [
        row
        for row in requests
        if row["student_id"] == AUTHORITATIVE_STUDENT_ID and row["request_type"] == "primary"
    ]
    logical_primary_ids = sorted(
        {row["must_share_block_id"] or row["course_id"] for row in primary_rows}
    )
    candidate_mapping_source = _function_source(
        paths["input_adapter_source"], "_build_candidate_index"
    )
    period_constraint_source = _function_source(
        paths["production_solver_source"], "_add_student_period_constraints"
    )
    fairness_constraint_source = _function_source(
        paths["production_solver_source"], "_add_fairness_hard_constraints"
    )
    fixed_pair_constraint_source = _function_source(
        paths["fixed_pair_screen_source"], "add_fixed_pair_constraints"
    )
    ordinary_unmet_limit = _integer_constant(
        paths["final_schedule_policy_source"],
        "MAXIMUM_ORDINARY_PRIMARY_UNMET_COUNT",
    )

    blocker_candidates: dict[str, list[str]] = {}
    blocker_placements: dict[str, list[str]] = {}
    for course_id in BLOCKER_COURSE_TO_SECTION:
        matching = [row for row in sections if row["logical_block_id"] == course_id]
        blocker_candidates[course_id] = sorted(
            {row["linked_section_group_id"] for row in matching}
        )
        blocker_placements[course_id] = sorted(
            {
                period
                for row in matching
                for period in (row["period_1"], row["period_2"])
                if period
            }
        )

    facts = {
        "student_exists_exactly_once": {
            "passed": len(student_rows) == 1,
            "observed_count": len(student_rows),
        },
        "student_is_ordinary": {
            "passed": len(student_rows) == 1
            and student_rows[0]["priority_protected"].strip().lower() == "false",
            "priority_protected": student_rows[0]["priority_protected"] if student_rows else None,
        },
        "ordinary_primary_unmet_limit_is_one": {
            "passed": ordinary_unmet_limit == 1
            and "MAXIMUM_ORDINARY_PRIMARY_UNMET_COUNT" in fairness_constraint_source,
            "observed_limit": ordinary_unmet_limit,
        },
        "logical_primary_count_is_seven": {
            "passed": len(logical_primary_ids) == 7,
            "logical_primary_count": len(logical_primary_ids),
            "logical_primary_ids": logical_primary_ids,
        },
        "blocker_requests_are_primary": {
            "passed": set(BLOCKER_COURSE_TO_SECTION).issubset(logical_primary_ids),
            "blocker_course_ids": sorted(BLOCKER_COURSE_TO_SECTION),
        },
        "each_blocker_primary_has_one_expected_candidate_section": {
            "passed": all(
                blocker_candidates[course_id] == [section_id]
                for course_id, section_id in BLOCKER_COURSE_TO_SECTION.items()
            )
            and "section.logical_block_id" in candidate_mapping_source
            and "request.candidate_key" in candidate_mapping_source,
            "candidate_sections": blocker_candidates,
        },
        "all_blocker_sections_are_originally_p1": {
            "passed": all(
                blocker_placements[course_id] == ["P1"]
                for course_id in BLOCKER_COURSE_TO_SECTION
            ),
            "original_placements": blocker_placements,
        },
        "student_period_assignment_limit_is_one": {
            "passed": "by_student_period" in period_constraint_source
            and "<= 1" in period_constraint_source,
            "production_function": "_add_student_period_constraints",
        },
        "fixed_pair_leaves_nonselected_sections_original": {
            "passed": "variable == int(section_id in section_ids)" in fixed_pair_constraint_source,
            "production_function": "add_fixed_pair_constraints",
        },
    }
    failures = [name for name, fact in facts.items() if fact["passed"] is not True]
    if failures:
        raise BlockerSafeScreenError(f"independent blocker proof failed: {failures}")

    input_hashes = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in sorted(paths.items())
    }
    input_hashes["source_suite_sha256sums"] = source_verification
    proof = {
        "schema_version": SCHEMA_VERSION,
        "scenario_id": TARGET_SCENARIO_ID,
        "student_id": AUTHORITATIVE_STUDENT_ID,
        "facts": facts,
        "blocker_section_ids": sorted(BLOCKER_SECTION_IDS),
        "derived_bounds": {
            "blocker_primary_count": 3,
            "maximum_blocker_primaries_satisfied_when_pair_is_disjoint": 1,
            "minimum_blocker_primaries_unmet_when_pair_is_disjoint": 2,
            "maximum_ordinary_primary_unmet": ordinary_unmet_limit,
        },
        "safe_necessary_condition": (
            "pair_section_ids ∩ blocker_section_ids ≠ ∅"
        ),
        "proof_conclusion": (
            "A fixed K=2 pair disjoint from the blocker set leaves all three "
            "single-candidate primaries at P1; at most one can be assigned, "
            "so at least two are unmet and the production hard policy is infeasible."
        ),
        "proof_verified": True,
    }
    proof["proof_hash"] = json_hash(proof)
    return proof, input_hashes


def _bool(row: Mapping[str, str], name: str) -> bool:
    value = row.get(name, "").strip().lower()
    if value not in {"true", "false"}:
        raise BlockerSafeScreenError(f"invalid boolean {name}={row.get(name)!r}")
    return value == "true"


def classify_static_survivors(
    rows: Sequence[Mapping[str, str]],
    tested_pair_ids: set[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    classifications: list[dict[str, Any]] = []
    pair_ids: set[str] = set()
    invalid_pair_count = 0
    duplicate_pair_count = 0
    screening_error_count = 0
    for source in rows:
        try:
            pair_id = source["pair_id"]
            section_ids = (source["section_id_a"], source["section_id_b"])
            if pair_id in pair_ids:
                duplicate_pair_count += 1
                continue
            pair_ids.add(pair_id)
            if (
                len(set(section_ids)) != 2
                or pair_id != f"{section_ids[0]}__{section_ids[1]}"
                or source["final_class"] != "core_screen_survivor"
                or not _bool(source, "core_screen_survivor")
            ):
                invalid_pair_count += 1
                continue
            intersects = bool(set(section_ids) & BLOCKER_SECTION_IDS)
            classifications.append(
                {
                    "pair_id": pair_id,
                    "section_id_a": section_ids[0],
                    "section_id_b": section_ids[1],
                    "course_id_a": source["course_id_a"],
                    "course_id_b": source["course_id_b"],
                    "classification": (
                        "blocker_screen_survivor"
                        if intersects
                        else "blocker_safe_excluded"
                    ),
                    "intersects_blocker_set": intersects,
                    "previously_tested_selected_pair": pair_id in tested_pair_ids,
                    "claim_scope": "safe_necessary_condition_screening_only",
                }
            )
        except (KeyError, BlockerSafeScreenError):
            screening_error_count += 1
    classifications.sort(key=lambda row: row["pair_id"])
    diagnostics = {
        "input_static_survivor_count": len(rows),
        "classification_count": len(classifications),
        "classification_closure": len(classifications) == len(rows),
        "duplicate_pair_count": duplicate_pair_count,
        "invalid_pair_count": invalid_pair_count,
        "screening_error_count": screening_error_count,
        "deterministic_ordering": "pair_id_ascending",
        "classification_ordering_hash": json_hash(
            [row["pair_id"] for row in classifications]
        ),
        "result_hash": json_hash(
            [
                [row["pair_id"], row["classification"]]
                for row in classifications
            ]
        ),
    }
    if (
        diagnostics["classification_closure"] is not True
        or duplicate_pair_count
        or invalid_pair_count
        or screening_error_count
    ):
        raise BlockerSafeScreenError(f"screening closure failed: {diagnostics}")
    return classifications, diagnostics


def _count_classes(rows: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    values = list(rows)
    return {
        "input_count": len(values),
        "blocker_safe_excluded": sum(
            row["classification"] == "blocker_safe_excluded" for row in values
        ),
        "blocker_screen_survivor": sum(
            row["classification"] == "blocker_screen_survivor" for row in values
        ),
    }


def build_counts(
    classifications: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    tested = [row for row in classifications if row["previously_tested_selected_pair"]]
    untested = [row for row in classifications if not row["previously_tested_selected_pair"]]
    counts = {
        "all_static_survivors": _count_classes(classifications),
        "previously_tested_selected_static_survivors": _count_classes(tested),
        "untested_static_survivors": _count_classes(untested),
    }
    if (
        counts["all_static_survivors"]["input_count"] != INPUT_STATIC_SURVIVORS
        or counts["previously_tested_selected_static_survivors"]["input_count"]
        != TESTED_SELECTED_SURVIVORS
        or counts["untested_static_survivors"]["input_count"]
        != INPUT_STATIC_SURVIVORS - TESTED_SELECTED_SURVIVORS
        or counts["all_static_survivors"]["input_count"]
        != counts["previously_tested_selected_static_survivors"]["input_count"]
        + counts["untested_static_survivors"]["input_count"]
    ):
        raise BlockerSafeScreenError(f"tested/untested count closure failed: {counts}")
    return counts


def build_universe_closure(counts: Mapping[str, Any]) -> dict[str, Any]:
    new_exclusions = counts["all_static_survivors"]["blocker_safe_excluded"]
    remaining = counts["all_static_survivors"]["blocker_screen_survivor"]
    total = (
        ORIGINAL_STATIC_EXCLUSIONS
        + PREVIOUSLY_PROVEN_INFEASIBLE
        + new_exclusions
        + remaining
    )
    closure = {
        "original_static_necessary_condition_failed": ORIGINAL_STATIC_EXCLUSIONS,
        "previously_proven_infeasible_unique_pair": PREVIOUSLY_PROVEN_INFEASIBLE,
        "new_all_student_blocker_safe_excluded": new_exclusions,
        "remaining_not_excluded_by_current_safe_necessary_conditions": remaining,
        "total_unique_pairs": total,
        "expected_total_unique_pairs": TOTAL_UNIQUE_PAIRS,
        "closure_passed": total == TOTAL_UNIQUE_PAIRS,
    }
    if closure["closure_passed"] is not True:
        raise BlockerSafeScreenError(f"full universe closure failed: {closure}")
    return closure


def build_aggregate_summary(
    *,
    counts: Mapping[str, Any],
    diagnostics: Mapping[str, Any],
    universe: Mapping[str, Any],
    formal_ordering_hash: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_name": "all_student_k2_blocker_safe_screen",
        "experiment_version": "v1",
        "result_type": "safe_necessary_condition_screening_only",
        "proof_verified": True,
        "counts": counts,
        "screening_diagnostics": diagnostics,
        "universe_closure": universe,
        "formal_remaining_pair_ordering_hash": formal_ordering_hash,
        "informal_projection_ordering_hash": INFORMAL_PROJECTION_ORDERING_HASH,
        "formal_and_informal_ordering_hash_match": (
            formal_ordering_hash == INFORMAL_PROJECTION_ORDERING_HASH
        ),
        "global_k2_status": "unresolved",
        "global_k2_remains_unresolved": True,
        "proven_lower_bound": 2,
        "exact_minimum_claim": None,
        "remaining_pair_claim": (
            "not excluded by current safe necessary conditions"
        ),
        "real_solver_invocations": 0,
        "source_screening_reruns": 0,
        "new_feasibility_results": 0,
        "failures": [],
    }


def _parse_json_list(value: str, field: str) -> list[Any]:
    try:
        result = json.loads(value)
    except json.JSONDecodeError as exc:
        raise BlockerSafeScreenError(f"invalid JSON in {field}") from exc
    if not isinstance(result, list):
        raise BlockerSafeScreenError(f"{field} must contain a JSON list")
    return result


def build_remaining_pairs(
    source_rows: Sequence[Mapping[str, str]],
    classifications: Sequence[Mapping[str, Any]],
    section_effect_rows: Sequence[Mapping[str, str]],
) -> tuple[list[dict[str, Any]], str]:
    source_by_id = {row["pair_id"]: row for row in source_rows}
    effects = {row["logical_section_id"]: row for row in section_effect_rows}
    survivors = [
        row for row in classifications if row["classification"] == "blocker_screen_survivor"
    ]
    ranked = sorted(
        survivors,
        key=lambda row: (
            -int(source_by_id[row["pair_id"]]["core_feasible_placement_combinations"]),
            int(source_by_id[row["pair_id"]]["affected_student_union_count"]),
            -int(source_by_id[row["pair_id"]]["changed_candidate_period_relationships"]),
            int(source_by_id[row["pair_id"]]["total_absolute_period_displacement"]),
            row["pair_id"],
        ),
    )
    ranking_records = [
        {
            "pair_id": row["pair_id"],
            "core_feasible_placement_combinations": int(
                source_by_id[row["pair_id"]]["core_feasible_placement_combinations"]
            ),
            "affected_student_union_count": int(
                source_by_id[row["pair_id"]]["affected_student_union_count"]
            ),
            "changed_candidate_period_relationships": int(
                source_by_id[row["pair_id"]]["changed_candidate_period_relationships"]
            ),
            "total_absolute_period_displacement": int(
                source_by_id[row["pair_id"]]["total_absolute_period_displacement"]
            ),
        }
        for row in ranked
    ]
    ordering_hash = hashlib.sha256(
        json.dumps(
            ranking_records,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    output: list[dict[str, Any]] = []
    for index, row in enumerate(ranked, start=1):
        source = source_by_id[row["pair_id"]]
        section_ids = (row["section_id_a"], row["section_id_b"])
        effect_rows = [effects.get(section_id) for section_id in section_ids]
        if any(effect is None for effect in effect_rows):
            raise BlockerSafeScreenError(f"missing section effect for {row['pair_id']}")
        domains = [
            _parse_json_list(effect["non_original_placements"], "non_original_placements")
            for effect in effect_rows
            if effect is not None
        ]
        domain_sizes = [len(domain) for domain in domains]
        total_combinations = int(source["total_placement_combinations"])
        if domain_sizes[0] * domain_sizes[1] != total_combinations:
            raise BlockerSafeScreenError(
                f"placement combination drift for {row['pair_id']}"
            )
        output.append(
            {
                "formal_order": index,
                "pair_id": row["pair_id"],
                "section_id_a": section_ids[0],
                "section_id_b": section_ids[1],
                "course_id_a": row["course_id_a"],
                "course_id_b": row["course_id_b"],
                "original_placement_a": effect_rows[0]["original_placement"],
                "original_placement_b": effect_rows[1]["original_placement"],
                "destination_domain_size_a": domain_sizes[0],
                "destination_domain_size_b": domain_sizes[1],
                "total_placement_combinations": total_combinations,
                "core_feasible_placement_combinations": int(
                    source["core_feasible_placement_combinations"]
                ),
                "affected_student_union_count": int(
                    source["affected_student_union_count"]
                ),
                "changed_candidate_period_relationships": int(
                    source["changed_candidate_period_relationships"]
                ),
                "total_absolute_period_displacement": int(
                    source["total_absolute_period_displacement"]
                ),
                "classification": "blocker_screen_survivor",
                "feasibility_claim": "none",
            }
        )
    return output, ordering_hash


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise BlockerSafeScreenError(f"refusing to write empty CSV: {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_checksums(root: Path) -> str:
    checksum = root / "SHA256SUMS.txt"
    lines = [
        f"{sha256_file(path)}  {path.relative_to(root)}"
        for path in sorted(root.iterdir())
        if path.is_file() and path != checksum
    ]
    checksum.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return sha256_file(checksum)


def _git_head(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def run_formal_screen(
    *,
    repo_root: Path,
    source_suite: Path = DEFAULT_SOURCE_SUITE,
    source_screen: Path = DEFAULT_SOURCE_SCREEN,
    output_dir: Path = DEFAULT_OUTPUT,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise BlockerSafeScreenError(
            f"output is non-empty; refusing to overwrite: {output_dir}"
        )
    proof, proof_input_hashes = verify_blocker_proof(
        repo_root=repo_root,
        source_suite=source_suite,
    )
    screen_verification = verify_sha256sums(source_screen)
    source_rows = read_csv(source_screen / "survivor_pairs.csv")
    section_effect_rows = read_csv(source_screen / "section_effect_signatures.csv")
    portfolio = read_json(source_screen / "selected_pair_portfolio.json")
    aggregate = read_json(source_screen / "aggregate_summary.json")
    tested_pair_ids = {
        "__".join(candidate["logical_section_ids"])
        for candidate in portfolio["candidates"]
    }
    if (
        len(source_rows) != INPUT_STATIC_SURVIVORS
        or aggregate["screening"]["core_screen_survivor_count"] != INPUT_STATIC_SURVIVORS
        or aggregate["screening"]["core_necessary_condition_failed_count"]
        != ORIGINAL_STATIC_EXCLUSIONS
        or aggregate["screening"]["previously_proven_infeasible_pairs"]
        != PREVIOUSLY_PROVEN_INFEASIBLE
        or len(tested_pair_ids) != TESTED_SELECTED_SURVIVORS
    ):
        raise BlockerSafeScreenError("formal source screening counts drifted")

    classifications, diagnostics = classify_static_survivors(
        source_rows, tested_pair_ids
    )
    counts = build_counts(classifications)
    universe = build_universe_closure(counts)
    remaining_pairs, formal_ordering_hash = build_remaining_pairs(
        source_rows,
        classifications,
        section_effect_rows,
    )
    excluded = [
        row for row in classifications if row["classification"] == "blocker_safe_excluded"
    ]
    if len(remaining_pairs) != counts["all_static_survivors"]["blocker_screen_survivor"]:
        raise BlockerSafeScreenError("remaining-pair detail count drifted")

    input_hashes = proof_input_hashes | {
        "source_screen_sha256sums": screen_verification,
        "survivor_pairs_csv": {
            "path": str(source_screen / "survivor_pairs.csv"),
            "sha256": sha256_file(source_screen / "survivor_pairs.csv"),
        },
        "section_effect_signatures_csv": {
            "path": str(source_screen / "section_effect_signatures.csv"),
            "sha256": sha256_file(source_screen / "section_effect_signatures.csv"),
        },
        "selected_pair_portfolio_json": {
            "path": str(source_screen / "selected_pair_portfolio.json"),
            "sha256": sha256_file(source_screen / "selected_pair_portfolio.json"),
        },
    }
    blocker_definition = {
        "blocker_course_to_section": BLOCKER_COURSE_TO_SECTION,
        "blocker_section_ids": sorted(BLOCKER_SECTION_IDS),
        "safe_necessary_condition": (
            "pair_section_ids ∩ blocker_section_ids ≠ ∅"
        ),
        "safe_exclusion_condition": (
            "pair_section_ids ∩ blocker_section_ids = ∅"
        ),
        "screening_scope": (
            "frozen normal_dev_10 K=2 fixed-pair production domain and hard policy"
        ),
    }
    summary = build_aggregate_summary(
        counts=counts,
        diagnostics=diagnostics,
        universe=universe,
        formal_ordering_hash=formal_ordering_hash,
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "experiment_name": summary["experiment_name"],
        "experiment_version": "v1",
        "scenario_id": TARGET_SCENARIO_ID,
        "source_git_commit": _git_head(repo_root),
        "source_suite": str(source_suite),
        "source_screen_artifact": str(source_screen),
        "output_dir": str(output_dir),
        "accepted_formal_static_safe_screen_executions": 1,
        "exploratory_projection_executions": 1,
        "original_48516_pair_screening_reruns": 0,
        "real_solver_invocations": 0,
    }
    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_git_commit": manifest["source_git_commit"],
        "implementation_file": str(Path(__file__).resolve()),
        "implementation_sha256": sha256_file(Path(__file__).resolve()),
        "source_artifacts_read_only": True,
        "six_pair_portfolio_artifact_modified": False,
        "solver_logs_modified": False,
        "formal_execution": "accepted_static_safe_screen",
        "result_is_feasibility_result": False,
        "real_solver_invocations": 0,
        "source_screening_reruns": 0,
    }

    parent = output_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = parent / f".{output_dir.name}.tmp-{os.getpid()}"
    if temporary.exists():
        raise BlockerSafeScreenError(f"temporary output already exists: {temporary}")
    temporary.mkdir()
    try:
        _write_json(temporary / "manifest.json", manifest)
        _write_json(temporary / "proof_audit.json", proof)
        _write_json(temporary / "blocker_definition.json", blocker_definition)
        _write_json(temporary / "input_hashes.json", input_hashes)
        _write_csv(
            temporary / "all_static_survivor_classifications.csv",
            classifications,
        )
        _write_csv(temporary / "blocker_safe_excluded_pairs.csv", excluded)
        _write_csv(
            temporary / "blocker_screen_survivors.csv",
            remaining_pairs,
        )
        _write_json(temporary / "aggregate_summary.json", summary)
        _write_json(temporary / "provenance.json", provenance)
        _write_json(
            temporary / "failures.json",
            {"failures": [], "failure_count": 0},
        )
        checksum_hash = _write_checksums(temporary)
        if output_dir.exists():
            output_dir.rmdir()
        temporary.replace(output_dir)
    except Exception:
        if temporary.exists():
            for path in sorted(temporary.iterdir(), reverse=True):
                path.unlink()
            temporary.rmdir()
        raise
    return summary | {
        "artifact_dir": str(output_dir),
        "artifact_file_count": 11,
        "artifact_checksum_entry_count": 10,
        "artifact_sha256sums_sha256": checksum_hash,
        "remaining_pairs": remaining_pairs,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the formal all-student K=2 blocker safe screen."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--source-suite", type=Path, default=DEFAULT_SOURCE_SUITE)
    parser.add_argument("--source-screen", type=Path, default=DEFAULT_SOURCE_SCREEN)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        result = run_formal_screen(
            repo_root=args.repo_root.resolve(),
            source_suite=args.source_suite.resolve(),
            source_screen=args.source_screen.resolve(),
            output_dir=args.output_dir.resolve(),
        )
    except BlockerSafeScreenError as exc:
        print(f"All-student K=2 blocker safe screen failed: {exc}")
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
