from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from ortools.sat.python import cp_model

import src.hybrid_stage1_incumbent_bootstrap as bootstrap
from src.allocation.cp_sat_solver import _VariableKey
from src.joint_period_edit_pilot import PlacementOption
from src.period_placement_repair_probe import CandidateEdit


def candidate(
    candidate_id: str,
    section_id: str,
    old: str = "P3",
    new: str = "P1",
    course: str | None = None,
) -> CandidateEdit:
    return CandidateEdit(
        candidate_id=candidate_id,
        edit_type="single_section_move",
        logical_section_ids=(section_id,),
        logical_course_ids=(course or section_id,),
        original_placements=((old,),),
        proposed_placements=((new,),),
        valid_period_source="test",
        occupancy_shape=((1,),),
        core_student="G12_0536",
        core_period_relevance=(old,),
        affected_candidate_edge_count=10,
        affected_student_count=5,
    )


def promising(candidate_id: str, *, unmet: int = 1, gap: int = 1, **extra: object) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "classification": "student_level_promising",
        "edited_primary_unmet": unmet,
        "edited_schedule_gap": gap,
        "ordinary_primary_unmet_at_most_one": True,
        "minimum_five_policy_reached": True,
        "maximum_gap_one_policy_reached": True,
        **extra,
    }


def test_manifest_only_allows_normal_dev_10() -> None:
    manifest = bootstrap.load_bootstrap_manifest()
    assert manifest["target_scenario_id"] == "normal_dev_10"
    assert manifest["phase"] == "single_target_incumbent_bootstrap"


def test_manifest_authoritative_student_and_exclusion_are_frozen() -> None:
    manifest = bootstrap.load_bootstrap_manifest()
    assert manifest["authoritative_student_id"] == "G12_0536"
    assert manifest["excluded_student_ids"] == ["G12_0105"]


def test_manifest_counts_and_seed_are_frozen() -> None:
    manifest = bootstrap.load_bootstrap_manifest()
    assert (manifest["editable_section_count"], manifest["placement_option_count"], manifest["candidate_edge_count"]) == (312, 841, 164269)
    assert (manifest["solver_seed"], manifest["workers"]) == (20260630, 1)


def test_manifest_portfolio_budgets_are_frozen() -> None:
    manifest = bootstrap.load_bootstrap_manifest()
    assert (manifest["k1_hint_portfolio_size_max"], manifest["k2_hint_portfolio_size_max"]) == (3, 2)
    assert (manifest["k1_run_budget_seconds"], manifest["k2_run_budget_seconds"]) == (180, 180)


def test_manifest_requires_first_complete_solution() -> None:
    assert bootstrap.load_bootstrap_manifest()["stop_after_first_complete_solution"] is True


@pytest.mark.parametrize(
    "field",
    [
        "stage2_allowed", "stage3_allowed", "stage4_allowed", "control_runs_allowed",
        "other_normal_targets_allowed", "stress_execution_allowed",
        "negative_execution_allowed", "holdout_execution_allowed",
    ],
)
def test_manifest_forbids_other_runs(tmp_path: Path, field: str) -> None:
    payload = json.loads(Path(bootstrap.DEFAULT_MANIFEST).read_text())
    payload[field] = True
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(bootstrap.BootstrapError):
        bootstrap.load_bootstrap_manifest(path)


def test_manifest_forbids_external_seed(tmp_path: Path) -> None:
    payload = json.loads(Path(bootstrap.DEFAULT_MANIFEST).read_text())
    payload["external_persisted_seed"] = True
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(bootstrap.BootstrapError):
        bootstrap.load_bootstrap_manifest(path)


def test_previous_log_audit_separates_structured_response_and_log_evidence(tmp_path: Path) -> None:
    log = """Initial optimization model ''\n#Variables: 100\nThe solution hint is incomplete: 10 out of 100 non fixed variables hinted.\nPresolved optimization model ''\n#Variables: 80\n#kAtMostOne: 7\n#kLinearN: 3\n[Symmetry] Graph too large. Skipping.\nStarting search at 2.0s\n'best_solutions':      0\nCpSolverResponse summary:\nstatus: UNKNOWN\n"""
    log_path = tmp_path / "solver.log"
    log_path.write_text(log)
    response = tmp_path / "response.json"
    response.write_text(json.dumps({"status": "UNKNOWN", "incumbent_found": False, "assignment_available": False, "solution_count": 0, "objective_value": None, "best_bound": None}))
    hint = tmp_path / "hint.json"
    hint.write_text(json.dumps({"fresh_model_verified": True}))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"solver_seed": 20260630}))
    audit = bootstrap.audit_previous_stage1_log(log_path, response, hint, config)
    assert audit["structured_response"]["status"] == "UNKNOWN"
    assert audit["log_evidence"]["hint_incomplete_message_seen"] is True
    assert audit["log_evidence"]["presolved_variable_count"] == 80
    assert audit["log_evidence"]["presolved_constraint_count"] == 10
    assert audit["log_evidence"]["search_started"] is True
    assert audit["log_evidence"]["best_solutions_added"] == 0


def test_previous_log_audit_does_not_infer_hint_infeasibility(tmp_path: Path) -> None:
    paths = []
    for name, value in (("log", "solution hint is incomplete"), ("response", {"status": "UNKNOWN"}), ("hint", {}), ("config", {})):
        path = tmp_path / name
        path.write_text(value if isinstance(value, str) else json.dumps(value))
        paths.append(path)
    audit = bootstrap.audit_previous_stage1_log(*paths)
    assert audit["log_evidence"]["hint_infeasible_message_seen"] is False
    assert audit["inference"]["hint_infeasibility_proven"] is False


def test_previous_log_audit_keeps_structured_stats(tmp_path: Path) -> None:
    log = tmp_path / "log"
    log.write_text("")
    response = tmp_path / "response"
    response.write_text(json.dumps({"status": "UNKNOWN", "branches": 12, "conflicts": 3, "deterministic_time_seconds": 4.5}))
    hint = tmp_path / "hint"
    hint.write_text("{}")
    config = tmp_path / "config"
    config.write_text("{}")
    audit = bootstrap.audit_previous_stage1_log(log, response, hint, config)
    assert audit["structured_response"]["branches"] == 12
    assert audit["structured_response"]["deterministic_time_seconds"] == 4.5


def test_single_portfolio_filters_to_exact_promising_moves() -> None:
    good = candidate("good", "S1")
    bad = candidate("bad", "S2")
    result = bootstrap.select_single_edit_portfolio(
        [good, bad],
        {"good": promising("good"), "bad": promising("bad", unmet=2)},
        max_size=3,
    )
    assert [item.candidate.candidate_id for item in result] == ["good"]


def test_single_portfolio_is_deterministically_sorted() -> None:
    first = candidate("z", "S1", new="P2")
    second = candidate("a", "S2", new="P2")
    rows = {item.candidate_id: promising(item.candidate_id) for item in (first, second)}
    selected = bootstrap.select_single_edit_portfolio([first, second], rows, max_size=3)
    assert [item.candidate.candidate_id for item in selected] == ["a", "z"]


def test_single_portfolio_respects_maximum_three() -> None:
    candidates = [candidate(f"S{i}", f"S{i}", new=f"P{i % 6 + 1}") for i in range(1, 6)]
    rows = {item.candidate_id: promising(item.candidate_id) for item in candidates}
    assert len(bootstrap.select_single_edit_portfolio(candidates, rows, max_size=3)) == 3


def test_single_portfolio_deduplicates_same_section_destination() -> None:
    first = candidate("first", "S1")
    second = candidate("second", "S1")
    rows = {"first": promising("first"), "second": promising("second")}
    selected = bootstrap.select_single_edit_portfolio([first, second], rows, max_size=3)
    assert [item.candidate.candidate_id for item in selected] == ["first"]


def test_single_portfolio_deduplicates_same_course_transition() -> None:
    first = candidate("first", "S1", course="COURSE")
    second = candidate("second", "S2", course="COURSE")
    rows = {"first": promising("first"), "second": promising("second")}
    selected = bootstrap.select_single_edit_portfolio([first, second], rows, max_size=3)
    assert [item.candidate.candidate_id for item in selected] == ["first"]


def test_candidate_sort_key_contains_all_frozen_tiebreaks() -> None:
    item = candidate("candidate", "S1", old="P7", new="P1")
    key = bootstrap.candidate_sort_key(item, promising("candidate", affected_student_count=4, changed_candidate_period_relationships=6))
    assert key == (1, 1, 4, 6, 6, "candidate")


def test_pair_portfolio_uses_distinct_sections(monkeypatch: pytest.MonkeyPatch) -> None:
    first = candidate("first", "S1")
    second = candidate("second", "S2", old="P4", new="P2")
    singles = bootstrap.select_single_edit_portfolio([first, second], {"first": promising("first"), "second": promising("second")}, max_size=20)
    monkeypatch.setattr(bootstrap, "apply_candidate_to_input", lambda value, item: value)
    monkeypatch.setattr(bootstrap, "exact_student_level_analysis", lambda value, student: {"primary_request_count": 6, "original_primary_unmet": 1, "original_max_primary_assignments": 5, "original_max_schedule_gap": 1})
    pairs = bootstrap.build_pair_hint_portfolio(singles, object(), {}, max_size=2)
    assert pairs and set(pairs[0].candidate.logical_section_ids) == {"S1", "S2"}


def test_pair_portfolio_is_capped_at_two(monkeypatch: pytest.MonkeyPatch) -> None:
    items = [candidate(f"c{i}", f"S{i}", new=f"P{i % 6 + 1}") for i in range(1, 5)]
    singles = bootstrap.select_single_edit_portfolio(items, {item.candidate_id: promising(item.candidate_id) for item in items}, max_size=20)
    monkeypatch.setattr(bootstrap, "apply_candidate_to_input", lambda value, item: value)
    monkeypatch.setattr(bootstrap, "exact_student_level_analysis", lambda value, student: {"primary_request_count": 6, "original_primary_unmet": 1, "original_max_primary_assignments": 5, "original_max_schedule_gap": 1})
    assert len(bootstrap.build_pair_hint_portfolio(singles, object(), {}, max_size=2)) == 2


def test_pair_portfolio_is_hint_generation_not_domain_pruning(monkeypatch: pytest.MonkeyPatch) -> None:
    item = candidate("c1", "S1")
    singles = bootstrap.select_single_edit_portfolio([item], {"c1": promising("c1")}, max_size=20)
    monkeypatch.setattr(bootstrap, "apply_candidate_to_input", lambda value, item: value)
    monkeypatch.setattr(bootstrap, "exact_student_level_analysis", lambda value, student: {"primary_request_count": 6, "original_primary_unmet": 1, "original_max_primary_assignments": 5, "original_max_schedule_gap": 1})
    assert bootstrap.build_pair_hint_portfolio(singles, object(), {}, max_size=2) == ()


def test_candidate_payload_records_portfolio_metadata() -> None:
    item = bootstrap.select_single_edit_portfolio([candidate("c1", "S1")], {"c1": promising("c1")}, max_size=1)[0]
    payload = bootstrap.candidate_payload(item)
    assert payload["candidate_id"] == "c1"
    assert "portfolio_sort_key" in payload
    assert payload["source_classification"] == "student_level_promising"


def fake_build(*, two_placements: bool = False) -> SimpleNamespace:
    model = cp_model.CpModel()
    assignment_key = _VariableKey("primary:G12_0536:S1", "S1")
    assignment_var = model.NewBoolVar("assignment")
    section = SimpleNamespace(linked_section_group_id="S1", occupied_periods=("P1",))
    domains = {"S1": (PlacementOption("S1", ("P1",), True, ()),)}
    choices = {}
    if two_placements:
        domains["S1"] = (PlacementOption("S1", ("P1",), True, ()), PlacementOption("S1", ("P2",), False, ()))
        choices = {("S1", ("P1",)): model.NewBoolVar("p1"), ("S1", ("P2",)): model.NewBoolVar("p2")}
    return SimpleNamespace(
        model=model,
        assignment_vars={assignment_key: assignment_var},
        section_changed_vars={"S1": model.NewBoolVar("changed")},
        allocation_input=SimpleNamespace(logical_sections=(section,)),
        placement_domains=domains,
        placement_choice_vars=choices,
    )


def test_k1_cap_is_added() -> None:
    build = fake_build()
    before = len(build.model.Proto().constraints)
    bootstrap.add_change_cap(build, 1)
    assert len(build.model.Proto().constraints) == before + 1


def test_k2_cap_is_added() -> None:
    build = fake_build()
    bootstrap.add_change_cap(build, 2)
    assert len(build.model.Proto().constraints) == 1


def test_invalid_cap_fails_closed() -> None:
    with pytest.raises(bootstrap.BootstrapError):
        bootstrap.add_change_cap(fake_build(), 3)


def test_hamming_expression_rewards_selected_assignment() -> None:
    build = fake_build()
    key = next(iter(build.assignment_vars))
    expression = bootstrap.hamming_expression(build, [key])
    build.model.Minimize(expression)
    assert build.model.Proto().objective.vars


def test_hamming_expression_penalizes_unselected_assignment() -> None:
    build = fake_build()
    expression = bootstrap.hamming_expression(build, [])
    build.model.Minimize(expression)
    assert list(build.model.Proto().objective.coeffs) == [1]


def test_change_cap_is_the_explicit_bounded_feasibility_restriction() -> None:
    build = fake_build()
    before = len(build.model.Proto().constraints)
    bootstrap.add_change_cap(build, 1)
    assert len(build.model.Proto().constraints) == before + 1
    assert "sole bootstrap feasibility restriction" in bootstrap.add_change_cap.__doc__


def test_bootstrap_hints_cover_assignment_edges() -> None:
    build = fake_build(two_placements=True)
    item = candidate("c1", "S1", old="P1", new="P2")
    key = next(iter(build.assignment_vars))
    audit = bootstrap.apply_bootstrap_hints(build, item, [key])
    assert audit["assignment_coverage"] == 1.0
    assert audit["placement_coverage"] == 1.0
    assert audit["assignment_positive"] == 1


def test_bootstrap_hints_select_edit_placement() -> None:
    build = fake_build(two_placements=True)
    item = candidate("c1", "S1", old="P1", new="P2")
    audit = bootstrap.apply_bootstrap_hints(build, item, [])
    assert audit["placement_positive"] == 1
    assert audit["fresh_model_verified"] is True


def test_bootstrap_hints_fail_on_unknown_assignment() -> None:
    build = fake_build()
    unknown = _VariableKey("unknown", "S1")
    with pytest.raises(bootstrap.BootstrapError):
        bootstrap.apply_bootstrap_hints(build, candidate("c1", "S1"), [unknown])


def test_bootstrap_hints_fail_on_existing_hint() -> None:
    build = fake_build()
    variable = next(iter(build.assignment_vars.values()))
    build.model.AddHint(variable, 0)
    with pytest.raises(Exception):
        bootstrap.apply_bootstrap_hints(build, candidate("c1", "S1"), [])


def test_solver_stops_after_first_complete_solution() -> None:
    build = fake_build()
    build.model.Add(next(iter(build.assignment_vars.values())) == 1)
    build.model.Minimize(0)
    result = bootstrap.solve_bootstrap(build, run_id="k1_01", k=1, hint_id="c1", time_limit_seconds=1)
    assert result.status in {"FEASIBLE", "OPTIMAL"}
    assert result.assignment_available is True
    assert result.solution_count >= 1


@pytest.mark.parametrize(
    ("k1", "k2", "accepted", "validated", "claim"),
    [
        ([{"k": 1, "status": "FEASIBLE", "witness_valid": True}], [], True, True, "minimum_changed_sections_within_frozen_placement_domain"),
        ([{"k": 1, "status": "INFEASIBLE"}], [{"k": 2, "status": "FEASIBLE", "witness_valid": True}], True, True, "minimum_changed_sections_within_frozen_placement_domain"),
        ([{"k": 1, "status": "UNKNOWN"}], [{"k": 2, "status": "FEASIBLE", "witness_valid": True}], True, True, "validated_repair_with_at_most_2_changes"),
        ([{"k": 1, "status": "UNKNOWN"}], [{"k": 2, "status": "UNKNOWN"}], False, False, "unresolved_no_incumbent"),
    ],
)
def test_claim_rules(k1: list[dict[str, object]], k2: list[dict[str, object]], accepted: bool, validated: bool, claim: str) -> None:
    assert bootstrap._claim(k1, k2, accepted, validated)["claim"] == claim


def test_infeasible_cap_two_claim_requires_both_caps() -> None:
    result = bootstrap._claim([{"k": 1, "status": "INFEASIBLE"}], [{"k": 2, "status": "INFEASIBLE"}], False, False)
    assert result == {"claim": "no_repair_within_cap_2_and_frozen_domain", "proven": True}


def test_artifact_checksums_are_stable(tmp_path: Path) -> None:
    (tmp_path / "one.json").write_text("{}\n")
    first = bootstrap.write_checksums(tmp_path)
    second = bootstrap.write_checksums(tmp_path)
    assert first == second
    assert len((tmp_path / "SHA256SUMS.txt").read_text().splitlines()) == 1


def test_resume_returns_checkpoint_without_search(tmp_path: Path) -> None:
    aggregate = {"k1_runs": 1, "search_reexecuted": False}
    (tmp_path / "aggregate_summary.json").write_text(json.dumps(aggregate))
    result = bootstrap.run_bootstrap(output_dir=tmp_path, resume=True)
    assert result["resumed"] is True
    assert result["search_reexecuted"] is False


def test_nonempty_output_refuses_non_resume(tmp_path: Path) -> None:
    (tmp_path / "partial.json").write_text("{}")
    with pytest.raises(bootstrap.BootstrapError):
        bootstrap.run_bootstrap(output_dir=tmp_path)


def test_resume_requires_atomic_checkpoint(tmp_path: Path) -> None:
    (tmp_path / "partial.json").write_text("{}")
    with pytest.raises(bootstrap.BootstrapError, match="atomic checkpoint"):
        bootstrap.run_bootstrap(output_dir=tmp_path, resume=True)


def test_manifest_has_all_source_hashes() -> None:
    manifest = bootstrap.load_bootstrap_manifest()
    for field in ("source_previous_stage1_hash", "source_hybrid_audit_hash", "source_candidate_preview_hash", "source_section_audit_hash", "source_section_audited_hash", "source_control_audit_hash", "source_control_audited_hash"):
        assert len(manifest[field]) == 64


def test_stage1_protocol_does_not_enable_other_scenarios() -> None:
    manifest = bootstrap.load_bootstrap_manifest()
    assert all(manifest[field] is False for field in ("control_runs_allowed", "other_normal_targets_allowed", "stress_execution_allowed", "negative_execution_allowed", "holdout_execution_allowed"))
