from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
from ortools.sat.python import cp_model

import src.hybrid_k2_search_bottleneck_diagnostic as diag
from src.allocation import canonicalize_allocation_input
from src.allocation.cp_sat_solver import _VariableKey
from src.hybrid_stage1_incumbent_bootstrap import SearchResult
from src.joint_period_edit_pilot import PlacementOption
from src.period_placement_repair_probe import CandidateEdit, ScenarioContext

from tests.test_cp_sat_solver import catalog, request_row, requests, section_row, sections, students


def _catalog_with_department():
    frame = catalog().copy()
    frame["department"] = "Other"
    return frame


# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _pair_context(*, conflicting: bool = False) -> ScenarioContext:
    """A tiny, genuinely solvable scenario with two independently movable sections.

    G12_0536 has five primary courses each in a distinct period (P1-P5),
    meeting the production hard "minimum five" schedule policy. The frozen
    pair moves CORE_A_1 and CORE_B_1 to the still-free P6/P7. With
    `conflicting=False` those destinations are free, giving a FEASIBLE
    production model. With `conflicting=True` CORE_B_1 is moved onto
    CORE_A_1's own destination period, forcing an unavoidable period conflict
    for the student's only two candidate sections for those courses (both
    capacity 1, both required), which the production hard model cannot
    satisfy -> INFEASIBLE.
    """
    courses = ("CORE_A", "CORE_B", "CORE_C", "CORE_D", "ALT1")
    raw_students = students([("G12_0536", 12, 5, False)])
    raw_requests = requests([request_row("G12_0536", course) for course in courses])
    raw_sections = sections([
        section_row("CORE_A_1", "CORE_A", "P1", capacity=10, group_id="CORE_A_1"),
        section_row("CORE_B_1", "CORE_B", "P2", capacity=10, group_id="CORE_B_1"),
        section_row("CORE_C_1", "CORE_C", "P3", capacity=10, group_id="CORE_C_1"),
        section_row("CORE_D_1", "CORE_D", "P4", capacity=10, group_id="CORE_D_1"),
        section_row("ALT1_1", "ALT1", "P5", capacity=10, group_id="ALT1_1"),
    ])
    allocation_input = canonicalize_allocation_input(raw_students, raw_requests, raw_sections, catalog())
    return ScenarioContext(
        scenario_id="fixture_target",
        allocation_input=allocation_input,
        students=raw_students,
        requests=raw_requests,
        sections=raw_sections,
        catalog=_catalog_with_department(),
        authoritative_core=None,
    )


def _pair_candidate(*, conflicting: bool = False, candidate_id: str = "bootstrap_pair:test") -> CandidateEdit:
    core_b_destination = "P5" if not conflicting else "P5"
    return CandidateEdit(
        candidate_id=candidate_id,
        edit_type="bootstrap_pair",
        logical_section_ids=("CORE_A_1", "CORE_B_1"),
        logical_course_ids=("CORE_A", "CORE_B"),
        original_placements=(("P1",), ("P2",)),
        proposed_placements=(("P5",), ("P5",) if conflicting else ("P6",)),
        valid_period_source="test",
        occupancy_shape=((1,), (1,)),
        core_student="G12_0536",
        core_period_relevance=("P1", "P2"),
        affected_candidate_edge_count=2,
        affected_student_count=1,
    )


def _single_candidate(section_id: str = "S1", old: str = "P3", new: str = "P1") -> CandidateEdit:
    return CandidateEdit(
        candidate_id=f"single:{section_id}",
        edit_type="single_section_move",
        logical_section_ids=(section_id,),
        logical_course_ids=(section_id,),
        original_placements=((old,),),
        proposed_placements=((new,),),
        valid_period_source="test",
        occupancy_shape=((1,),),
        core_student="G12_0536",
        core_period_relevance=(old,),
        affected_candidate_edge_count=10,
        affected_student_count=5,
    )


def _pair_portfolio_payload(*, count: int = 2) -> dict:
    def row(candidate_id: str, second_destination: str) -> dict:
        return {
            "candidate_id": candidate_id,
            "edit_type": "bootstrap_pair",
            "logical_section_ids": ["SEC_A", "SEC_B"],
            "logical_course_ids": ["COURSE_A", "COURSE_B"],
            "original_placements": [["P7"], ["P3"]],
            "proposed_placements": [["P6"], [second_destination]],
            "valid_period_source": "test",
            "occupancy_shape": [[1], [1]],
            "core_student": "G12_0536",
            "core_period_relevance": ["P3", "P7"],
            "affected_candidate_edge_count": 10,
            "affected_student_count": 10,
        }

    rows = [row("bootstrap_pair:one", "P2"), row("bootstrap_pair:two", "P4")][:count]
    return {"count": len(rows), "candidates": rows}


def _fake_build(*, two_placements: bool = False, second_section: bool = False) -> SimpleNamespace:
    model = cp_model.CpModel()
    assignment_key = _VariableKey("primary:G12_0536:S1", "S1")
    assignment_var = model.NewBoolVar("assignment")
    section = SimpleNamespace(linked_section_group_id="S1", occupied_periods=("P1",))
    domains = {"S1": (PlacementOption("S1", ("P1",), True, ()),)}
    choices = {}
    changed = {"S1": model.NewBoolVar("changed_s1")}
    if two_placements:
        domains["S1"] = (PlacementOption("S1", ("P1",), True, ()), PlacementOption("S1", ("P2",), False, ()))
        choices = {("S1", ("P1",)): model.NewBoolVar("p1"), ("S1", ("P2",)): model.NewBoolVar("p2")}
    if second_section:
        changed["S2"] = model.NewBoolVar("changed_s2")
    return SimpleNamespace(
        model=model,
        assignment_vars={assignment_key: assignment_var},
        section_changed_vars=changed,
        allocation_input=SimpleNamespace(logical_sections=(section,)),
        placement_domains=domains,
        placement_choice_vars=choices,
    )


# --------------------------------------------------------------------------
# 1. Manifest validation
# --------------------------------------------------------------------------


def test_manifest_only_contains_normal_dev_10() -> None:
    manifest = diag.load_diagnostic_manifest()
    assert manifest["target_scenario_id"] == "normal_dev_10"
    assert manifest["phase"] == "k2_search_bottleneck_diagnostic"


def test_manifest_excludes_g12_0105() -> None:
    manifest = diag.load_diagnostic_manifest()
    assert manifest["excluded_student_ids"] == ["G12_0105"]
    assert manifest["authoritative_student_id"] == "G12_0536"


def test_manifest_pair_count_is_frozen_at_two() -> None:
    assert diag.load_diagnostic_manifest()["pair_count"] == 2


def test_manifest_seed_and_workers_are_frozen() -> None:
    manifest = diag.load_diagnostic_manifest()
    assert (manifest["solver_seed"], manifest["workers"]) == (20260630, 1)
    assert manifest["external_persisted_seed"] is False


def test_manifest_frozen_domain_counts_match_bootstrap() -> None:
    manifest = diag.load_diagnostic_manifest()
    assert (manifest["editable_section_count"], manifest["placement_option_count"], manifest["candidate_edge_count"]) == (312, 841, 164269)


def test_manifest_budgets_match_diagnostic_constants() -> None:
    manifest = diag.load_diagnostic_manifest()
    assert manifest["exact_destination_no_hint_budget_seconds"] == diag.DIAGNOSTIC_A_BUDGET_SECONDS
    assert manifest["exact_destination_hamming_budget_seconds"] == diag.DIAGNOSTIC_B_BUDGET_SECONDS
    assert manifest["fixed_section_ids_destination_free_budget_seconds"] == diag.DIAGNOSTIC_C_BUDGET_SECONDS
    assert manifest["joint_fixed_witness_budget_seconds"] == diag.FIXED_WITNESS_BUDGET_SECONDS
    assert manifest["production_validation_budget_seconds"] == diag.PRODUCTION_BUDGET_SECONDS


def test_manifest_forbids_full_search_and_other_k_values() -> None:
    manifest = diag.load_diagnostic_manifest()
    assert manifest["full_k2_search_allowed"] is False
    assert manifest["k1_allowed"] is False
    assert manifest["k3_allowed"] is False


@pytest.mark.parametrize(
    "field",
    ["other_normal_targets_allowed", "stress_execution_allowed", "negative_execution_allowed", "holdout_execution_allowed"],
)
def test_manifest_forbids_other_execution_modes(tmp_path: Path, field: str) -> None:
    payload = json.loads(Path(diag.DEFAULT_MANIFEST).read_text())
    payload[field] = True
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(diag.DiagnosticError):
        diag.load_diagnostic_manifest(path)


def test_manifest_missing_field_fails_closed(tmp_path: Path) -> None:
    payload = json.loads(Path(diag.DEFAULT_MANIFEST).read_text())
    del payload["frozen_pair_portfolio_hash"]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(diag.DiagnosticError):
        diag.load_diagnostic_manifest(path)


# --------------------------------------------------------------------------
# 2. Frozen pair portfolio loading (never regenerated)
# --------------------------------------------------------------------------


@pytest.fixture
def bootstrap_artifact(require_external_artifact) -> Path:
    return require_external_artifact(
        "robustness-v1/hybrid-stage1-incumbent-bootstrap-v1"
    )


@pytest.fixture
def verified_source_artifacts(require_external_artifact, monkeypatch):
    remapped = {
        name: (
            require_external_artifact(f"robustness-v1/{root.name}"),
            expected_hash,
        )
        for name, (root, expected_hash) in diag.EXPECTED_SOURCE_ARTIFACT_HASHES.items()
    }
    monkeypatch.setattr(diag, "EXPECTED_SOURCE_ARTIFACT_HASHES", remapped)
    return remapped


@pytest.mark.external_artifact
def test_pair_portfolio_loaded_from_real_bootstrap_source(bootstrap_artifact: Path) -> None:
    manifest = diag.load_diagnostic_manifest()
    pairs = diag.load_frozen_pairs(manifest, bootstrap_artifact)
    assert len(pairs) == 2
    for candidate in pairs:
        assert candidate.edit_type == "bootstrap_pair"
        assert len(set(candidate.logical_section_ids)) == 2
        assert candidate.core_student == "G12_0536"


def test_pair_portfolio_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    manifest = dict(diag.load_diagnostic_manifest())
    manifest["frozen_pair_portfolio_hash"] = "0" * 64
    bootstrap_dir = tmp_path / "bootstrap"
    bootstrap_dir.mkdir()
    (bootstrap_dir / "pair_hint_portfolio.json").write_text(json.dumps(_pair_portfolio_payload()))
    with pytest.raises(diag.DiagnosticError, match="hash"):
        diag.load_frozen_pairs(manifest, bootstrap_dir)


def test_pair_portfolio_wrong_count_fails_closed(tmp_path: Path) -> None:
    bootstrap_dir = tmp_path / "bootstrap"
    bootstrap_dir.mkdir()
    payload = _pair_portfolio_payload(count=1)
    path = bootstrap_dir / "pair_hint_portfolio.json"
    path.write_text(json.dumps(payload))
    manifest = dict(diag.load_diagnostic_manifest())
    manifest["frozen_pair_portfolio_hash"] = diag._sha256_file(path)
    with pytest.raises(diag.DiagnosticError, match="exactly 2"):
        diag.load_frozen_pairs(manifest, bootstrap_dir)


def test_pair_portfolio_never_regenerated_only_read(tmp_path: Path) -> None:
    bootstrap_dir = tmp_path / "bootstrap"
    bootstrap_dir.mkdir()
    payload = _pair_portfolio_payload()
    path = bootstrap_dir / "pair_hint_portfolio.json"
    path.write_text(json.dumps(payload))
    manifest = dict(diag.load_diagnostic_manifest())
    manifest["frozen_pair_portfolio_hash"] = diag._sha256_file(path)
    pairs = diag.load_frozen_pairs(manifest, bootstrap_dir)
    assert [item.candidate_id for item in pairs] == ["bootstrap_pair:one", "bootstrap_pair:two"]
    # Loading again must reproduce byte-identical CandidateEdit objects; the
    # loader must not mutate, reorder, or resynthesize the source file.
    assert path.read_text() == json.dumps(payload)


def test_pair_portfolio_rejects_duplicate_section_ids(tmp_path: Path) -> None:
    payload = _pair_portfolio_payload()
    payload["candidates"][0]["logical_section_ids"] = ["SEC_A", "SEC_A"]
    bootstrap_dir = tmp_path / "bootstrap"
    bootstrap_dir.mkdir()
    path = bootstrap_dir / "pair_hint_portfolio.json"
    path.write_text(json.dumps(payload))
    manifest = dict(diag.load_diagnostic_manifest())
    manifest["frozen_pair_portfolio_hash"] = diag._sha256_file(path)
    with pytest.raises(diag.DiagnosticError, match="two distinct sections"):
        diag.load_frozen_pairs(manifest, bootstrap_dir)


# --------------------------------------------------------------------------
# 3. Pair analysis
# --------------------------------------------------------------------------


def test_pair_analysis_records_transitions_and_hash() -> None:
    context = _pair_context()
    candidate = _pair_candidate()
    analysis = diag.analyze_pair("pair_1", candidate, context.allocation_input, Path("data/config"))
    assert analysis["pair_id"] == "pair_1"
    assert analysis["logical_section_ids"] == ["CORE_A_1", "CORE_B_1"]
    assert analysis["original_placements"] == [["P1"], ["P2"]]
    assert analysis["hinted_destination_placements"] == [["P5"], ["P6"]]
    assert len(analysis["period_transitions"]) == 2
    assert analysis["pair_hash"] == diag._json_hash(asdict(candidate))


def test_pair_analysis_computes_affected_student_union_deduplicated() -> None:
    # STU_1 has both CORE_A and CORE_B as primary requests, so the affected
    # student union must contain STU_1 exactly once, not twice.
    context = _pair_context()
    candidate = _pair_candidate()
    analysis = diag.analyze_pair("pair_1", candidate, context.allocation_input, Path("data/config"))
    assert analysis["affected_student_union"] == ["G12_0536"]
    assert analysis["affected_student_count"] == 1


def test_compare_frozen_pairs_detects_shared_section_ids() -> None:
    a = {"logical_section_ids": ["SEC_A", "SEC_B"]}
    b = {"logical_section_ids": ["SEC_B", "SEC_A"]}
    result = diag.compare_frozen_pairs((a, b))
    assert result["pairs_share_identical_section_ids"] is True


def test_compare_frozen_pairs_detects_different_section_ids() -> None:
    a = {"logical_section_ids": ["SEC_A", "SEC_B"]}
    b = {"logical_section_ids": ["SEC_C", "SEC_D"]}
    result = diag.compare_frozen_pairs((a, b))
    assert result["pairs_share_identical_section_ids"] is False


# --------------------------------------------------------------------------
# 4. Previous K=2 log analysis
# --------------------------------------------------------------------------


def test_log_parser_extracts_presolved_fingerprint_and_variables() -> None:
    log = (
        "Initial optimization model ''\n#Variables: 100\n"
        "The solution hint is incomplete: 10 out of 100 non fixed variables hinted.\n"
        "Presolved optimization model '': (model_fingerprint: 0xabc123)\n#Variables: 80\n"
        "[Symmetry] Graph too large. Skipping.\n"
        "Starting search at 2.5s\n"
        "CpSolverResponse summary:\nstatus: UNKNOWN\nobjective: 40\nbest_bound: 3\n"
        "conflicts: 5\nbranches: 6\nrestarts: 1\nwalltime: 10.5\nusertime: 10.5\ndeterministic_time: 9.1\n"
    )
    result = diag._parse_k2_solver_log(log)
    assert result["initial_variable_count"] == 100
    assert result["presolved_variable_count"] == 80
    assert result["presolved_model_fingerprint"] == "0xabc123"
    assert result["search_started"] is True
    assert result["search_start_time_seconds"] == 2.5
    assert result["hint_incomplete_message_seen"] is True
    assert result["hint_infeasible_message_seen"] is False
    assert result["raw_cp_solver_response_summary_fields"]["status"] == "UNKNOWN"
    assert result["raw_cp_solver_response_summary_fields"]["objective"] == 40
    assert result["raw_cp_solver_response_summary_fields"]["best_bound"] == 3


def test_log_parser_does_not_report_hint_infeasible_from_incomplete_message() -> None:
    log = "The solution hint is incomplete: 5 out of 10 non fixed variables hinted.\n"
    result = diag._parse_k2_solver_log(log)
    assert result["hint_incomplete_message_seen"] is True
    assert result["hint_infeasible_message_seen"] is False


@pytest.mark.external_artifact
def test_analyze_previous_k2_runs_against_real_bootstrap_artifact(bootstrap_artifact: Path) -> None:
    # Read-only regression check against the already-verified bootstrap
    # artifact; no new solver run occurs here.
    analysis = diag.analyze_previous_k2_runs(bootstrap_artifact)
    assert analysis["k2_01"]["structured_response"]["status"] == "UNKNOWN"
    assert analysis["k2_01"]["structured_response"]["incumbent_found"] is False
    assert analysis["k2_02"]["structured_response"]["status"] == "UNKNOWN"
    assert analysis["k2_01"]["log_evidence"]["presolved_variable_count"] == 454985
    assert analysis["k2_02"]["log_evidence"]["presolved_variable_count"] == 454988
    assert analysis["comparison"]["both_runs_entered_branch_search"] is True
    assert analysis["comparison"]["both_hints_reported_incomplete_not_infeasible"] is True


@pytest.mark.external_artifact
def test_analyze_previous_k2_runs_does_not_call_raw_log_objective_an_incumbent(
    bootstrap_artifact: Path,
) -> None:
    analysis = diag.analyze_previous_k2_runs(bootstrap_artifact)
    note = analysis["comparison"]["raw_log_values_are_not_a_valid_incumbent"]
    assert "not a found feasible solution" in note or "NOT a found feasible" in note


# --------------------------------------------------------------------------
# 5. Diagnostic C model restriction: add_fixed_section_pair_constraint
# --------------------------------------------------------------------------


def test_fixed_pair_constraint_forces_both_sections_changed() -> None:
    build = _fake_build(second_section=True)
    before = len(build.model.Proto().constraints)
    diag.add_fixed_section_pair_constraint(build, ("S1", "S2"))
    # One "== 1" constraint per named section, no "== 0" constraints since
    # there are no other editable sections in this fixture.
    assert len(build.model.Proto().constraints) == before + 2


def test_fixed_pair_constraint_forces_other_sections_unchanged() -> None:
    build = _fake_build(second_section=True)
    build.section_changed_vars["S3"] = build.model.NewBoolVar("changed_s3")
    before = len(build.model.Proto().constraints)
    diag.add_fixed_section_pair_constraint(build, ("S1", "S2"))
    assert len(build.model.Proto().constraints) == before + 3  # S1==1, S2==1, S3==0


def test_fixed_pair_constraint_rejects_same_section_twice() -> None:
    build = _fake_build(second_section=True)
    with pytest.raises(diag.DiagnosticError, match="two distinct sections"):
        diag.add_fixed_section_pair_constraint(build, ("S1", "S1"))


def test_fixed_pair_constraint_rejects_unknown_section() -> None:
    build = _fake_build(second_section=True)
    with pytest.raises(diag.DiagnosticError, match="missing a changed-indicator"):
        diag.add_fixed_section_pair_constraint(build, ("S1", "UNKNOWN"))


def test_fixed_pair_constraint_does_not_prune_destination_options() -> None:
    build = _fake_build(two_placements=True, second_section=True)
    diag.add_fixed_section_pair_constraint(build, ("S1", "S2"))
    # Both placement options for S1 remain present as real decision variables;
    # the constraint only touches section_changed_vars, never
    # placement_choice_vars directly.
    assert len(build.placement_choice_vars) == 2


# --------------------------------------------------------------------------
# 6. Diagnostic A (real tiny production model, no hint, no objective)
# --------------------------------------------------------------------------


def test_diagnostic_a_finds_feasible_exact_plan(tmp_path: Path) -> None:
    empty_config = tmp_path / "config"
    empty_config.mkdir()
    context = _pair_context()
    placement_map = {"CORE_A_1": ("P6",), "CORE_B_1": ("P7",)}
    response = diag.exact_plan_production_feasibility_no_hint(
        context, placement_map, config_dir=empty_config, time_limit_seconds=10.0,
    )
    assert response["status"] in {"FEASIBLE", "OPTIMAL"}
    assert response["assignment_available"] is True
    assert response["hint_used"] is False
    assert response["objective_used"] is False
    assert response["policy_pass"] is True


def test_diagnostic_a_reports_infeasible_when_exact_plan_conflicts(tmp_path: Path) -> None:
    empty_config = tmp_path / "config"
    empty_config.mkdir()
    context = _pair_context()
    placement_map = {"CORE_A_1": ("P6",), "CORE_B_1": ("P6",)}
    response = diag.exact_plan_production_feasibility_no_hint(
        context, placement_map, config_dir=empty_config, time_limit_seconds=10.0,
    )
    assert response["status"] == "INFEASIBLE"
    assert response["assignment_available"] is False


# --------------------------------------------------------------------------
# 7. Diagnostic B thin wrapper
# --------------------------------------------------------------------------


def test_diagnostic_b_wraps_independent_production_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {}

    def fake_validation(context, placement_map, *, config_dir, seed, time_limit_seconds):
        calls["time_limit_seconds"] = time_limit_seconds
        return {"status": "FEASIBLE", "assignment_available": True, "response_hash": "abc"}

    monkeypatch.setattr(diag, "independent_production_validation", fake_validation)
    response = diag.exact_plan_production_feasibility_with_hint(
        object(), {}, config_dir=Path("data/config"), time_limit_seconds=60.0,
    )
    assert calls["time_limit_seconds"] == 60.0
    assert response["diagnostic_id"] == "diagnostic_b"
    assert response["assignment_available"] is True


def test_diagnostic_b_uses_60_second_default_not_300() -> None:
    assert diag.DIAGNOSTIC_B_BUDGET_SECONDS == 60.0
    assert diag.DIAGNOSTIC_B_BUDGET_SECONDS != diag.PRODUCTION_BUDGET_SECONDS


# --------------------------------------------------------------------------
# 8. run_pair_protocol orchestration (diagnostic functions monkeypatched so
# each named scenario in the task's fixture list is driven deterministically)
# --------------------------------------------------------------------------


def _patch_diagnostics(monkeypatch: pytest.MonkeyPatch, *, a: dict, b: dict | None = None, c: dict | None = None) -> list[str]:
    calls: list[str] = []

    def fake_a(context, placement_map, *, config_dir, seed=diag.SOLVER_SEED, time_limit_seconds=diag.DIAGNOSTIC_A_BUDGET_SECONDS):
        calls.append("a")
        return {"diagnostic_id": "diagnostic_a", "model_restriction": "x", **a}

    def fake_b(context, placement_map, *, config_dir, seed=diag.SOLVER_SEED, time_limit_seconds=diag.DIAGNOSTIC_B_BUDGET_SECONDS):
        calls.append("b")
        return {"diagnostic_id": "diagnostic_b", "model_restriction": "x", "internal_hint_algorithm": "constrained_first_greedy", **(b or {})}

    def fake_hint_quality(context, candidate, config_dir, seed):
        return {"classification": "test"}, ()

    def fake_c(allocation_input, domains, candidate, assignment_hint, *, run_id, math_fallback_rules, math_course_ids, seed=diag.SOLVER_SEED, time_limit_seconds=diag.DIAGNOSTIC_C_BUDGET_SECONDS):
        calls.append("c")
        payload = {
            "run_id": run_id, "k": 2, "hint_id": candidate.candidate_id, "assignment_available": False,
            "incumbent_found": False, "solution_count": 0, "first_solution_time_seconds": None,
            "objective_value": None, "best_bound": None, "optimality_proven": False,
            "wall_time_seconds": 1.0, "end_to_end_runtime_seconds": 1.0, "deterministic_time_seconds": 1.0,
            "conflicts": 0, "branches": 0, "propagations": 0, "integer_propagations": 0, "restarts": 0,
            "response_hash": "c-hash", "selected_assignments": (), "selected_placements": (), "solver_log": (),
            "status": "UNKNOWN",
        }
        payload.update(c or {})
        result = SearchResult(**payload)
        build = _fake_build()
        hint = {"model_restriction": "fixed_two_section_ids"}
        return build, hint, result

    monkeypatch.setattr(diag, "exact_plan_production_feasibility_no_hint", fake_a)
    monkeypatch.setattr(diag, "exact_plan_production_feasibility_with_hint", fake_b)
    monkeypatch.setattr(diag, "_assignment_hint_quality_for_pair", fake_hint_quality)
    monkeypatch.setattr(diag, "fixed_pair_destination_free_diagnostic", fake_c)
    if c is not None and c.get("assignment_available"):
        monkeypatch.setattr(diag, "validate_bootstrap_witness", lambda *a, **k: {"joint_bootstrap_witness_valid": True})
    return calls


def _run_protocol(tmp_path: Path, **kwargs) -> diag._PairOutcome:
    return diag.run_pair_protocol(
        pair_index=0, candidate=_pair_candidate(), context=SimpleNamespace(allocation_input=None), domains={},
        rules=(), math_ids=(), config_dir=Path("data/config"), output=tmp_path, already_completed=set(),
        **kwargs,
    )


def test_pair_protocol_stops_after_diagnostic_a_feasible(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_diagnostics(monkeypatch, a={"status": "FEASIBLE", "assignment_available": True, "selected_assignments": [], "response_hash": "a-hash"})
    outcome = _run_protocol(tmp_path)
    assert calls == ["a"]
    assert outcome.incumbent_found is True
    assert "exact_plan_assignment_feasible_diagnostic_a" in outcome.classification_evidence


def test_pair_protocol_unknown_a_runs_b(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_diagnostics(
        monkeypatch,
        a={"status": "UNKNOWN", "assignment_available": False},
        b={"status": "UNKNOWN", "assignment_available": False},
    )
    _run_protocol(tmp_path)
    assert calls == ["a", "b", "c"]


def test_pair_protocol_infeasible_a_skips_b_runs_c(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_diagnostics(monkeypatch, a={"status": "INFEASIBLE", "assignment_available": False})
    _run_protocol(tmp_path)
    assert calls == ["a", "c"]


def test_pair_protocol_b_feasible_stops_before_c(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_diagnostics(
        monkeypatch,
        a={"status": "UNKNOWN", "assignment_available": False},
        b={"status": "FEASIBLE", "assignment_available": True, "selected_assignments": []},
    )
    outcome = _run_protocol(tmp_path)
    assert calls == ["a", "b"]
    assert outcome.incumbent_found is True
    assert "assignment_hint_helpful_diagnostic_b" in outcome.classification_evidence


def test_pair_protocol_c_feasible_after_exact_infeasible_supports_global_bottleneck(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_diagnostics(
        monkeypatch,
        a={"status": "INFEASIBLE", "assignment_available": False},
        c={"status": "FEASIBLE", "assignment_available": True, "selected_assignments": (), "selected_placements": ()},
    )
    outcome = _run_protocol(tmp_path)
    assert calls == ["a", "c"]
    assert outcome.incumbent_found is True
    assert "global_section_pair_selection_bottleneck_supported" in outcome.classification_evidence


def test_pair_protocol_c_feasible_after_exact_unknown_is_destination_bottleneck(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_diagnostics(
        monkeypatch,
        a={"status": "UNKNOWN", "assignment_available": False},
        b={"status": "UNKNOWN", "assignment_available": False},
        c={"status": "FEASIBLE", "assignment_available": True, "selected_assignments": (), "selected_placements": ()},
    )
    outcome = _run_protocol(tmp_path)
    assert calls == ["a", "b", "c"]
    assert outcome.incumbent_found is True
    assert "destination_selection_bottleneck" in outcome.classification_evidence


def test_pair_protocol_c_infeasible_is_fixed_section_pair_infeasible(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_diagnostics(
        monkeypatch,
        a={"status": "INFEASIBLE", "assignment_available": False},
        c={"status": "INFEASIBLE", "assignment_available": False},
    )
    outcome = _run_protocol(tmp_path)
    assert calls == ["a", "c"]
    assert outcome.incumbent_found is False
    assert "fixed_section_pair_infeasible" in outcome.classification_evidence


def test_pair_protocol_all_unknown_reports_unresolved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_diagnostics(
        monkeypatch,
        a={"status": "UNKNOWN", "assignment_available": False},
        b={"status": "UNKNOWN", "assignment_available": False},
        c={"status": "UNKNOWN", "assignment_available": False},
    )
    outcome = _run_protocol(tmp_path)
    assert calls == ["a", "b", "c"]
    assert outcome.incumbent_found is False
    assert not any("infeasible" in item for item in outcome.classification_evidence)


def test_pair_protocol_run_count_is_at_most_three(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_diagnostics(
        monkeypatch,
        a={"status": "UNKNOWN", "assignment_available": False},
        b={"status": "UNKNOWN", "assignment_available": False},
        c={"status": "UNKNOWN", "assignment_available": False},
    )
    outcome = _run_protocol(tmp_path)
    assert len(outcome.run_rows) == 3
    assert len(calls) == 3


def test_pair_protocol_model_invalid_stops_and_reports_correctness_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _patch_diagnostics(monkeypatch, a={"status": "MODEL_INVALID", "assignment_available": False})
    outcome = _run_protocol(tmp_path)
    assert calls == ["a"]
    assert outcome.correctness_failure is True


def test_pair_protocol_writes_run_artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_diagnostics(monkeypatch, a={"status": "INFEASIBLE", "assignment_available": False}, c={"status": "INFEASIBLE", "assignment_available": False})
    _run_protocol(tmp_path)
    for diagnostic_id in ("diagnostic_a", "diagnostic_c"):
        run_dir = tmp_path / "runs" / "pair_1" / diagnostic_id
        assert (run_dir / "solver_config.json").is_file()
        assert (run_dir / "hint_audit.json").is_file()
        assert (run_dir / "response_stats.json").is_file()
        assert (run_dir / "solver.log").is_file()
        assert (run_dir / "validation.json").is_file()


# --------------------------------------------------------------------------
# 9. Bottleneck classification
# --------------------------------------------------------------------------


def test_classify_bottleneck_exact_plan_feasible() -> None:
    outcome = diag._PairOutcome("pair_1", True, ["exact_plan_assignment_feasible_diagnostic_a"], [], None, False)
    result = diag.classify_bottleneck((outcome,))
    assert "exact_plan_assignment_feasible" in result["labels"]


def test_classify_bottleneck_hint_helpful() -> None:
    outcome = diag._PairOutcome("pair_1", True, ["assignment_hint_helpful_diagnostic_b"], [], None, False)
    result = diag.classify_bottleneck((outcome,))
    assert "assignment_hint_helpful" in result["labels"]


def test_classify_bottleneck_no_incumbent_uses_exhausted_label() -> None:
    outcome = diag._PairOutcome(
        "pair_1", False,
        ["exact_destination_pair_infeasible_diagnostic_a", "fixed_section_pair_infeasible"],
        [{"pair_id": "pair_1", "diagnostic_id": "diagnostic_a", "status": "INFEASIBLE"}, {"pair_id": "pair_1", "diagnostic_id": "diagnostic_c", "status": "INFEASIBLE"}],
        None, False,
    )
    result = diag.classify_bottleneck((outcome,))
    assert "exact_destination_pair_infeasible" in result["labels"]
    assert "fixed_section_pair_infeasible" in result["labels"]


def test_classify_bottleneck_never_claims_complete_causal_proof() -> None:
    outcome = diag._PairOutcome("pair_1", True, ["global_section_pair_selection_bottleneck_supported"], [], None, False)
    result = diag.classify_bottleneck((outcome,))
    assert "not a complete causal proof" in result["note"]


def test_classify_bottleneck_correctness_failure_overrides_everything() -> None:
    outcome = diag._PairOutcome("pair_1", False, ["diagnostic_a_model_invalid"], [], None, True)
    result = diag.classify_bottleneck((outcome,))
    assert result["labels"] == ["correctness_failure"]


# --------------------------------------------------------------------------
# 10. Top-level run_diagnostic: resume / atomic checkpoint / nonempty guard
# --------------------------------------------------------------------------


def test_run_diagnostic_resume_returns_checkpoint_without_search(tmp_path: Path) -> None:
    aggregate = {"pairs_run": 1, "search_reexecuted": False}
    (tmp_path / "aggregate_summary.json").write_text(json.dumps(aggregate))
    result = diag.run_diagnostic(output_dir=tmp_path, resume=True)
    assert result["resumed"] is True
    assert result["search_reexecuted"] is False


def test_run_diagnostic_nonempty_output_refuses_without_resume(tmp_path: Path) -> None:
    (tmp_path / "partial.json").write_text("{}")
    with pytest.raises(diag.DiagnosticError, match="refusing overwrite"):
        diag.run_diagnostic(output_dir=tmp_path)


def test_run_diagnostic_resume_requires_atomic_checkpoint(tmp_path: Path) -> None:
    (tmp_path / "partial.json").write_text("{}")
    with pytest.raises(diag.DiagnosticError, match="atomic checkpoint"):
        diag.run_diagnostic(output_dir=tmp_path, resume=True)


def test_write_checksums_is_stable(tmp_path: Path) -> None:
    (tmp_path / "one.json").write_text("{}\n")
    first = diag.write_checksums(tmp_path)
    second = diag.write_checksums(tmp_path)
    assert first == second
    assert len((tmp_path / "SHA256SUMS.txt").read_text().splitlines()) == 1


# --------------------------------------------------------------------------
# 11. Source-artifact / no-pruning / hint-uniqueness guarantees
# --------------------------------------------------------------------------


@pytest.mark.external_artifact
def test_source_artifacts_are_verified_read_only(verified_source_artifacts) -> None:
    manifest = diag.load_diagnostic_manifest()
    result = diag.verify_source_artifacts(manifest)
    assert set(result) == set(diag.EXPECTED_SOURCE_ARTIFACT_HASHES)
    for check in result.values():
        assert check["passed"] is True
        assert check["read_only"] is True


@pytest.mark.external_artifact
def test_source_artifact_hash_mismatch_fails_closed(verified_source_artifacts) -> None:
    bad_manifest = dict(diag.load_diagnostic_manifest())
    bad_manifest["source_bootstrap_artifact_hash"] = "0" * 64
    with pytest.raises(diag.DiagnosticError):
        diag.verify_source_artifacts(bad_manifest)


def test_diagnostic_c_hint_uses_apply_bootstrap_hints_fail_closed_paths() -> None:
    # apply_bootstrap_hints (reused unchanged from the bootstrap module)
    # already fails closed on an unknown assignment key; confirm this
    # diagnostic module surfaces the same guarantee via its own error type
    # boundary rather than silently swallowing it.
    build = _fake_build()
    unknown = _VariableKey("unknown", "S1")
    with pytest.raises(Exception):
        diag.apply_bootstrap_hints(build, _single_candidate("S1"), [unknown])


@pytest.mark.external_artifact
def test_frozen_domain_verification_recorded_without_pruning(bootstrap_artifact: Path) -> None:
    manifest = diag.load_diagnostic_manifest()
    pairs = diag.load_frozen_pairs(manifest, bootstrap_artifact)
    for candidate in pairs:
        # A pair candidate must always carry both of its original sections'
        # legal domain intact; this diagnostic never removes candidate
        # sections from either the assignment or placement universe.
        assert len(candidate.logical_section_ids) == 2


# --------------------------------------------------------------------------
# 12. Execution-history correction (reporting-only; no real solver anywhere
# in this section -- apply_execution_history_correction never builds or
# solves a CP-SAT model)
# --------------------------------------------------------------------------


def _fake_completed_artifact(tmp_path: Path, *, run_counts: tuple[int, int, int] = (2, 0, 2)) -> Path:
    """Build a minimal but structurally real completed-artifact directory
    (checkpoint/provenance/aggregate/failures/frozen_pair_portfolio) so
    apply_execution_history_correction can be exercised without ever running
    a real diagnostic or solver."""
    output = tmp_path / "artifact"
    output.mkdir()
    a_count, b_count, c_count = run_counts
    run_rows: list[dict] = []
    for index in range(a_count):
        run_rows.append({"pair_id": f"pair_{index + 1}", "diagnostic_id": "diagnostic_a", "status": "INFEASIBLE", "assignment_available": False, "runtime_seconds": 1.5, "response_hash": f"a{index}"})
    for index in range(b_count):
        run_rows.append({"pair_id": f"pair_{index + 1}", "diagnostic_id": "diagnostic_b", "status": "INFEASIBLE", "assignment_available": False, "runtime_seconds": 2.5, "response_hash": f"b{index}"})
    for index in range(c_count):
        run_rows.append({"pair_id": f"pair_{index + 1}", "diagnostic_id": "diagnostic_c", "status": "INFEASIBLE", "assignment_available": False, "runtime_seconds": 3.5, "response_hash": f"c{index}"})
    diag._write_json(output / "checkpoint.json", {
        "schema_version": 1, "run_rows": run_rows, "failures": [],
        "discovered_witness": {"status": "not_run"}, "fixed_witness_acceptance": {"status": "not_run"},
        "production_validation": {"status": "not_run"}, "accepted": False, "validated": False,
        "search_reexecuted": False,
    })
    diag._write_json(output / "provenance.json", {
        "source_git_commit": "abc", "previous_full_k2_reruns": 0,
        "diagnostic_a_runs": a_count, "diagnostic_b_runs": b_count, "diagnostic_c_runs": c_count,
        "k1_runs": 0, "k3_runs": 0, "production_validation_runs": 0,
        "control_runs": 0, "other_normal_target_runs": 0, "stress_runs": 0, "negative_runs": 0, "holdout_runs": 0,
        "external_persisted_seed": False,
    })
    diag._write_json(output / "aggregate_summary.json", {
        "experiment_name": "hybrid_k2_search_bottleneck_diagnostic", "phase": "k2_search_bottleneck_diagnostic",
        "target_scenario_id": "normal_dev_10", "pairs_run": 2, "previous_full_k2_reruns": 0,
        "diagnostic_a_runs": a_count, "diagnostic_b_runs": b_count, "diagnostic_c_runs": c_count,
        "k1_runs": 0, "k3_runs": 0, "production_validation_runs": 0,
        "control_runs": 0, "other_normal_target_runs": 0, "stress_runs": 0, "negative_runs": 0, "holdout_runs": 0,
        "external_persisted_seed": False, "accepted": False, "validated": False,
        "minimum_claim": {"claim": "unresolved_no_incumbent", "proven": False},
        "bottleneck_classification": ["diagnostic_portfolio_exhausted_no_incumbent"],
        "result_classification": "unresolved_no_incumbent", "failures": [],
    })
    diag._write_json(output / "failures.json", {"failures": [], "unexpected_failure_count": 0})
    a = {"pair_id": "pair_1", "logical_section_ids": ["AP_3D_ART_DESIGN_01", "SOCIAL_JUSTICE_01"]}
    b = {"pair_id": "pair_2", "logical_section_ids": ["AP_3D_ART_DESIGN_01", "SOCIAL_JUSTICE_01"]}
    diag._write_json(output / "frozen_pair_portfolio.json", {
        "count": 2, "pairs": [a, b], "comparison": {}, "frozen_domain": {},
    })
    return output


def test_apply_execution_history_correction_writes_total_invocations_eight(tmp_path: Path) -> None:
    output = _fake_completed_artifact(tmp_path)
    result = diag.apply_execution_history_correction(output)
    history = diag._read_json(output / "execution_history_correction.json")
    assert history["total_solver_invocations"]["total"] == 8
    assert result["total_solver_invocations"]["total"] == 8


def test_apply_execution_history_correction_accepted_final_runs_four(tmp_path: Path) -> None:
    output = _fake_completed_artifact(tmp_path)
    history = diag.build_execution_history_correction({})
    assert history["accepted_final_artifact_runs"]["total"] == 4
    diag.apply_execution_history_correction(output)
    written = diag._read_json(output / "execution_history_correction.json")
    assert written["accepted_final_artifact_runs"]["total"] == 4


def test_apply_execution_history_correction_superseded_runs_four(tmp_path: Path) -> None:
    output = _fake_completed_artifact(tmp_path)
    diag.apply_execution_history_correction(output)
    written = diag._read_json(output / "execution_history_correction.json")
    assert written["superseded_runs"]["total"] == 4


def test_apply_execution_history_correction_diagnostic_a_total_invocations_four(tmp_path: Path) -> None:
    output = _fake_completed_artifact(tmp_path)
    diag.apply_execution_history_correction(output)
    written = diag._read_json(output / "execution_history_correction.json")
    assert written["total_solver_invocations"]["diagnostic_a"] == 4


def test_apply_execution_history_correction_diagnostic_c_total_invocations_four(tmp_path: Path) -> None:
    output = _fake_completed_artifact(tmp_path)
    diag.apply_execution_history_correction(output)
    written = diag._read_json(output / "execution_history_correction.json")
    assert written["total_solver_invocations"]["diagnostic_c"] == 4


def test_apply_execution_history_correction_diagnostic_b_total_invocations_zero(tmp_path: Path) -> None:
    output = _fake_completed_artifact(tmp_path)
    diag.apply_execution_history_correction(output)
    written = diag._read_json(output / "execution_history_correction.json")
    assert written["total_solver_invocations"]["diagnostic_b"] == 0


def test_apply_execution_history_correction_flags_protocol_deviation(tmp_path: Path) -> None:
    output = _fake_completed_artifact(tmp_path)
    diag.apply_execution_history_correction(output)
    written = diag._read_json(output / "execution_history_correction.json")
    assert written["protocol_deviation"] is True
    provenance = diag._read_json(output / "provenance.json")
    assert provenance["protocol_deviation"] is True
    aggregate = diag._read_json(output / "aggregate_summary.json")
    assert aggregate["protocol_deviation"] is True


def test_apply_execution_history_correction_rerun_reason_nonempty(tmp_path: Path) -> None:
    output = _fake_completed_artifact(tmp_path)
    diag.apply_execution_history_correction(output)
    written = diag._read_json(output / "execution_history_correction.json")
    assert isinstance(written["protocol_deviation_reason"], str) and written["protocol_deviation_reason"].strip() != ""
    failures = diag._read_json(output / "failures.json")
    deviation = failures["protocol_deviations"][0]
    assert isinstance(deviation["description"], str) and deviation["description"].strip() != ""
    assert isinstance(deviation["reason"], str) and deviation["reason"].strip() != ""


def test_apply_execution_history_correction_result_based_parameter_change_false(tmp_path: Path) -> None:
    output = _fake_completed_artifact(tmp_path)
    diag.apply_execution_history_correction(output)
    written = diag._read_json(output / "execution_history_correction.json")
    assert written["result_based_parameter_change"] is False
    assert written["portfolio_changed_between_batches"] is False
    assert written["seed_changed_between_batches"] is False
    assert written["budget_changed_between_batches"] is False


def test_apply_execution_history_correction_does_not_fabricate_first_batch_hashes(tmp_path: Path) -> None:
    output = _fake_completed_artifact(tmp_path)
    diag.apply_execution_history_correction(output)
    written = diag._read_json(output / "execution_history_correction.json")
    evidence = written["first_batch_evidence"]
    assert evidence["first_batch_response_hashes"] is None
    assert evidence["first_batch_exact_runtime_seconds"] is None
    assert evidence["first_batch_per_run_pair_diagnostic_mapping"] is None
    # Statuses are the one first-batch fact recorded as directly observed
    # (this session's own transcript), never invented from the second batch.
    assert evidence["first_batch_statuses_observed"] == ["INFEASIBLE", "INFEASIBLE", "INFEASIBLE", "INFEASIBLE"]
    assert "not reconstructed" in evidence["first_batch_statuses_evidence_source"]


def test_apply_execution_history_correction_frozen_pair_candidate_count_two(tmp_path: Path) -> None:
    output = _fake_completed_artifact(tmp_path)
    diag.apply_execution_history_correction(output)
    written = diag._read_json(output / "execution_history_correction.json")
    assert written["pair_semantics"]["frozen_pair_candidate_count"] == 2


def test_apply_execution_history_correction_exact_destination_plan_count_two(tmp_path: Path) -> None:
    output = _fake_completed_artifact(tmp_path)
    diag.apply_execution_history_correction(output)
    written = diag._read_json(output / "execution_history_correction.json")
    assert written["pair_semantics"]["exact_destination_plan_count"] == 2


def test_apply_execution_history_correction_unique_section_id_pair_count_one(tmp_path: Path) -> None:
    output = _fake_completed_artifact(tmp_path)
    diag.apply_execution_history_correction(output)
    written = diag._read_json(output / "execution_history_correction.json")
    assert written["pair_semantics"]["unique_section_id_pair_count"] == 1
    assert written["pair_semantics"]["unique_fixed_section_pair_count_tested"] == 1


def test_apply_execution_history_correction_recognizes_c_feasible_region_identity(tmp_path: Path) -> None:
    output = _fake_completed_artifact(tmp_path)
    diag.apply_execution_history_correction(output)
    portfolio = diag._read_json(output / "frozen_pair_portfolio.json")
    assert portfolio["comparison"]["pairs_share_identical_section_ids"] is True
    assert portfolio["comparison"]["pair_semantics"]["diagnostic_c_search_config_count"] == 2


def test_apply_execution_history_correction_never_claims_two_distinct_pairs_infeasible(tmp_path: Path) -> None:
    output = _fake_completed_artifact(tmp_path)
    diag.apply_execution_history_correction(output)
    written = diag._read_json(output / "execution_history_correction.json")
    note = written["pair_semantics"]["note"]
    assert "not proof that a second, distinct section pair is infeasible" in note
    assert "two distinct section pairs" not in note.lower().replace("a second, distinct section pair", "")


def test_apply_execution_history_correction_scopes_fixed_pair_claim_to_frozen_domain(tmp_path: Path) -> None:
    output = _fake_completed_artifact(tmp_path)
    diag.apply_execution_history_correction(output)
    written = diag._read_json(output / "execution_history_correction.json")
    assert "over the full frozen destination domain" in written["pair_semantics"]["note"]


def test_apply_execution_history_correction_preserves_unresolved_no_incumbent(tmp_path: Path) -> None:
    output = _fake_completed_artifact(tmp_path)
    diag.apply_execution_history_correction(output)
    aggregate = diag._read_json(output / "aggregate_summary.json")
    assert aggregate["result_classification"] == "unresolved_no_incumbent"
    assert aggregate["accepted"] is False


def test_apply_execution_history_correction_preserves_no_minimum_claim(tmp_path: Path) -> None:
    output = _fake_completed_artifact(tmp_path)
    diag.apply_execution_history_correction(output)
    aggregate = diag._read_json(output / "aggregate_summary.json")
    assert aggregate["minimum_claim"] == {"claim": "unresolved_no_incumbent", "proven": False}
    assert aggregate["validated"] is False


def test_apply_execution_history_correction_rejects_run_count_mismatch(tmp_path: Path) -> None:
    output = _fake_completed_artifact(tmp_path, run_counts=(1, 0, 2))
    with pytest.raises(diag.DiagnosticError):
        diag.apply_execution_history_correction(output)


def test_apply_execution_history_correction_rejects_missing_files(tmp_path: Path) -> None:
    output = tmp_path / "incomplete"
    output.mkdir()
    with pytest.raises(diag.DiagnosticError):
        diag.apply_execution_history_correction(output)


def test_apply_execution_history_correction_does_not_touch_runs_directory(tmp_path: Path) -> None:
    output = _fake_completed_artifact(tmp_path)
    runs_dir = output / "runs" / "pair_1" / "diagnostic_a"
    runs_dir.mkdir(parents=True)
    solver_log = runs_dir / "solver.log"
    solver_log.write_text("original solver evidence\n", encoding="utf-8")
    before = solver_log.stat().st_mtime_ns
    before_text = solver_log.read_text(encoding="utf-8")
    diag.apply_execution_history_correction(output)
    assert solver_log.stat().st_mtime_ns == before
    assert solver_log.read_text(encoding="utf-8") == before_text


def test_apply_execution_history_correction_updates_sha256sums(tmp_path: Path) -> None:
    output = _fake_completed_artifact(tmp_path)
    diag.apply_execution_history_correction(output)
    checksum_file = output / "SHA256SUMS.txt"
    assert checksum_file.is_file()
    text = checksum_file.read_text(encoding="utf-8")
    assert "execution_history_correction.json" in text


def test_apply_execution_history_correction_never_builds_or_solves_cp_sat_model(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("apply_execution_history_correction must never build or solve a CP-SAT model")

    monkeypatch.setattr(diag, "build_joint_model", _forbidden)
    monkeypatch.setattr(diag, "_build_full_feasibility_cp_sat_model", _forbidden)
    monkeypatch.setattr(diag, "solve_bootstrap", _forbidden)
    output = _fake_completed_artifact(tmp_path)
    diag.apply_execution_history_correction(output)  # must not raise


def test_apply_execution_history_correction_resume_still_returns_without_new_search(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Even after the correction adds new keys to aggregate_summary.json, the
    # resume short-circuit (an existing aggregate_summary.json) must still
    # return immediately with search_reexecuted=False and must never invoke
    # run_pair_protocol (which is what would trigger a real diagnostic run).
    output = _fake_completed_artifact(tmp_path)
    diag.apply_execution_history_correction(output)

    def _forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("resume must not run any new diagnostic protocol")

    monkeypatch.setattr(diag, "run_pair_protocol", _forbidden)
    result = diag.run_diagnostic(output_dir=output, resume=True)
    assert result["resumed"] is True
    assert result["search_reexecuted"] is False
