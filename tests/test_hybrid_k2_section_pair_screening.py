from __future__ import annotations

import json
import hashlib
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from ortools.sat import cp_model_pb2

import src.hybrid_k2_section_pair_screening as k2
from src.hybrid_stage1_incumbent_bootstrap import SearchResult
from src.joint_period_edit_pilot import PlacementOption
from src.period_placement_repair_probe import CandidateEdit


class FakeVar:
    def __init__(self, name: str) -> None:
        self.name = name

    def __eq__(self, other: object) -> tuple[str, str, object]:  # type: ignore[override]
        return ("eq", self.name, other)


class FakeModel:
    def __init__(self) -> None:
        self.constraints: list[object] = []
        self.objective: object | None = "existing"
        self.cleared = False

    def Add(self, constraint: object) -> None:
        self.constraints.append(constraint)

    def ClearObjective(self) -> None:
        self.cleared = True
        self.objective = None

    def Minimize(self, objective: object) -> None:
        self.objective = objective

    def Proto(self) -> SimpleNamespace:
        return SimpleNamespace(
            variables=(),
            constraints=tuple(self.constraints),
            __str__=lambda: "fake-proto",
        )

    def ExportToFile(self, path: str) -> bool:
        proto = cp_model_pb2.CpModelProto()
        Path(path).write_bytes(proto.SerializeToString(deterministic=True))
        return True


def search_result(status: str, *, incumbent: bool = False) -> SearchResult:
    return SearchResult(
        run_id=f"run:{status}",
        k=2,
        hint_id="hint",
        status=status,
        assignment_available=incumbent,
        incumbent_found=incumbent,
        solution_count=int(incumbent),
        first_solution_time_seconds=0.1 if incumbent else None,
        objective_value=None,
        best_bound=None,
        optimality_proven=status == "OPTIMAL",
        wall_time_seconds=1.0,
        end_to_end_runtime_seconds=1.0,
        deterministic_time_seconds=0.0,
        conflicts=0,
        branches=0,
        propagations=0,
        integer_propagations=0,
        restarts=0,
        response_hash=f"hash-{status}-{incumbent}",
        selected_assignments=(("R1", "A"),) if incumbent else (),
        selected_placements=(("A", ("P2",)), ("B", ("P5",))) if incumbent else (),
        solver_log=(),
    )


def placement(section_id: str, periods: tuple[str, ...], *, original: bool = False) -> PlacementOption:
    return PlacementOption(section_id=section_id, placement=periods, is_original=original)


def domains() -> dict[str, tuple[PlacementOption, ...]]:
    return {
        "A": (placement("A", ("P1",), original=True), placement("A", ("P2",)), placement("A", ("P3",))),
        "B": (placement("B", ("P4",), original=True), placement("B", ("P5",)), placement("B", ("P6",))),
        "C": (placement("C", ("P2",), original=True), placement("C", ("P3",))),
        "D": (placement("D", ("P3",), original=True), placement("D", ("P4",))),
        "E": (placement("E", ("P4",), original=True), placement("E", ("P5",))),
        "F": (placement("F", ("P5",), original=True), placement("F", ("P6",))),
    }


def core_profile(*candidate_sections: str) -> k2.CoreProfile:
    candidates = tuple(
        k2.CoreCandidate(
            request_key=f"REQ_{section_id}",
            candidate_key=f"REQ_{section_id}:{section_id}",
            period_units=1,
            section_id=section_id,
            logical_identity=f"COURSE_{section_id}",
            occupied_periods=("P1",),
        )
        for section_id in candidate_sections
    )
    return k2.CoreProfile(
        student_id=k2.AUTHORITATIVE_STUDENT_ID,
        target_period_units=max(1, len(candidates)),
        primary_requests=(
            k2.CoreRequest(
                request_key="REQ",
                candidate_key="REQ",
                period_units=1,
                candidates=candidates,
            ),
        ),
    )


def evaluation(*, feasible: bool, primary_unmet: int = 1, schedule_gap: int = 1) -> k2.CoreEvaluation:
    return k2.CoreEvaluation(
        student_id=k2.AUTHORITATIVE_STUDENT_ID,
        primary_request_count=1,
        target_period_units=1,
        max_primary_assignments=0 if primary_unmet else 1,
        primary_unmet=primary_unmet,
        max_primary_period_units=0 if schedule_gap else 1,
        schedule_gap=schedule_gap,
        max_logical_gap=schedule_gap,
        selected_by_count=(),
        selected_by_units=(),
        student_local_feasible=feasible,
    )


def fake_input(section_ids: tuple[str, ...] = ("A", "B", "C", "D", "E", "F")) -> SimpleNamespace:
    return SimpleNamespace(
        logical_sections_by_id={
            section_id: SimpleNamespace(logical_block_id=f"COURSE_{section_id}")
            for section_id in section_ids
        },
        logical_requests=(),
        candidate_index={},
    )


def candidate(section_ids: tuple[str, str] = ("A", "B")) -> CandidateEdit:
    ds = domains()
    return CandidateEdit(
        candidate_id="candidate:A_B",
        edit_type="k2_section_pair_full_destination_domain",
        logical_section_ids=section_ids,
        logical_course_ids=tuple(f"COURSE_{section_id}" for section_id in section_ids),
        original_placements=tuple(ds[section_id][0].placement for section_id in section_ids),
        proposed_placements=tuple(ds[section_id][1].placement for section_id in section_ids),
        valid_period_source="test",
        occupancy_shape=(1, 1),
        core_student=k2.AUTHORITATIVE_STUDENT_ID,
        core_period_relevance=("P1", "P2", "P4", "P5"),
        affected_candidate_edge_count=2,
        affected_student_count=3,
    )


@pytest.fixture(autouse=True)
def forbid_real_solver_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: object, **kwargs: object) -> object:
        raise AssertionError("real solver path must not run in K=2 screening unit tests")

    monkeypatch.setattr(k2.cp_model, "CpSolver", forbidden)
    monkeypatch.setattr(k2, "production_fixed_witness_acceptance", forbidden)
    monkeypatch.setattr(k2, "independent_production_validation", forbidden)


def test_manifest_loads_and_allows_only_normal_dev_10() -> None:
    manifest = k2.load_screening_manifest()

    assert manifest["target_scenario_id"] == "normal_dev_10"
    assert manifest["authoritative_student_id"] == "G12_0536"
    assert manifest["excluded_student_ids"] == ["G12_0105"]
    assert manifest["other_normal_targets_allowed"] is False


def test_manifest_rejects_other_target(tmp_path: Path) -> None:
    payload = json.loads(k2.DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    payload["target_scenario_id"] = "normal_dev_09"
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(k2.ScreeningError, match="only normal_dev_10"):
        k2.load_screening_manifest(manifest)


def test_frozen_count_constants_match_manifest() -> None:
    manifest = k2.load_screening_manifest()

    assert manifest["editable_section_count"] == k2.EXPECTED_EDITABLE_SECTION_COUNT == 312
    assert manifest["placement_option_count"] == k2.EXPECTED_PLACEMENT_OPTION_COUNT == 841
    assert manifest["candidate_edge_count"] == k2.EXPECTED_CANDIDATE_EDGE_COUNT == 164269
    assert manifest["expected_unique_pair_count"] == k2.EXPECTED_UNIQUE_PAIR_COUNT == 48516


def test_enumerate_unique_pairs_is_unordered_deduplicated_and_deterministic() -> None:
    pairs = k2.enumerate_unique_pairs(("C", "A", "B"))

    assert pairs == (("A", "B"), ("A", "C"), ("B", "C"))
    assert len(pairs) == len(set(pairs))
    assert all(first < second for first, second in pairs)
    assert k2.enumerate_unique_pairs(("B", "C", "A")) == k2.enumerate_unique_pairs(("C", "B", "A"))


def test_section_effect_signature_neutral_and_relevant_are_deterministic() -> None:
    profile = core_profile("A")
    ds = domains()

    relevant_a = k2.section_effect_signature("A", ds, profile, fake_input())
    relevant_b = k2.section_effect_signature("A", ds, profile, fake_input())
    neutral = k2.section_effect_signature("B", ds, profile, fake_input())

    assert relevant_a.core_effect == "relevant"
    assert relevant_a.candidate_for_core_student is True
    assert relevant_a.distinct_core_effect_signature_count == 2
    assert relevant_a.effect_signature_hash == relevant_b.effect_signature_hash
    assert neutral.core_effect == "neutral"
    assert neutral.candidate_for_core_student is False
    assert neutral.distinct_core_effect_signature_count == 1


def test_screen_pair_checks_all_combinations_and_uses_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[tuple[str, tuple[str, ...]], ...]] = []

    def fake_evaluate(profile: k2.CoreProfile, overrides: dict[str, tuple[str, ...]] | None = None) -> k2.CoreEvaluation:
        calls.append(tuple(sorted((overrides or {}).items())))
        return evaluation(feasible=False)

    monkeypatch.setattr(k2, "evaluate_core_student", fake_evaluate)
    result = k2.screen_pair(
        "A",
        "B",
        profile=core_profile("A"),
        domains=domains(),
        allocation_input=fake_input(),
        evaluation_cache={},
    )

    assert result.total_placement_combinations == 4
    assert result.core_feasible_placement_combinations == 0
    assert result.final_class == "core_necessary_condition_failed"
    assert len(calls) == 2


def test_screen_pair_survivor_uses_best_feasible_canonical_destination(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_evaluate(profile: k2.CoreProfile, overrides: dict[str, tuple[str, ...]] | None = None) -> k2.CoreEvaluation:
        placement_a = (overrides or {}).get("A")
        if placement_a == ("P3",):
            return evaluation(feasible=True, primary_unmet=0, schedule_gap=0)
        return evaluation(feasible=False, primary_unmet=1, schedule_gap=1)

    monkeypatch.setattr(k2, "evaluate_core_student", fake_evaluate)
    result = k2.screen_pair(
        "A",
        "B",
        profile=core_profile("A"),
        domains=domains(),
        allocation_input=fake_input(),
        evaluation_cache={},
    )

    assert result.final_class == "core_screen_survivor"
    assert result.core_necessary_condition_failed is False
    assert result.core_screen_survivor is True
    assert result.canonical_destinations is not None
    assert result.canonical_destinations[0] == ("P3",)


def test_previously_excluded_pair_is_not_reported_as_survivor(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(k2, "evaluate_core_student", lambda profile, overrides=None: evaluation(feasible=True, primary_unmet=0, schedule_gap=0))

    result = k2.screen_pair(
        "A",
        "B",
        profile=core_profile("A"),
        domains=domains(),
        allocation_input=fake_input(),
        evaluation_cache={},
        previously_excluded_pairs={("A", "B")},
    )

    assert result.previously_proven_infeasible is True
    assert result.final_class == "previously_proven_infeasible"
    assert result.core_screen_survivor is True


def survivor(
    first: str,
    second: str,
    *,
    course_a: str | None = None,
    course_b: str | None = None,
    sort_tail: int = 0,
    previous: bool = False,
) -> k2.PairScreeningResult:
    return k2.PairScreeningResult(
        pair_id=f"{first}__{second}",
        section_id_a=first,
        section_id_b=second,
        course_id_a=course_a or f"COURSE_{first}",
        course_id_b=course_b or f"COURSE_{second}",
        final_class="previously_proven_infeasible" if previous else "core_screen_survivor",
        previously_proven_infeasible=previous,
        core_neutral_pair=False,
        core_necessary_condition_failed=False,
        core_screen_survivor=True,
        invalid_domain_pair=False,
        screening_error=False,
        total_placement_combinations=1,
        core_feasible_placement_combinations=1,
        best_core_primary_unmet=0,
        best_core_schedule_gap=0,
        best_logical_assigned=1,
        canonical_destinations=(("P2",), ("P5",)),
        canonical_sort_key=(sort_tail,),
        pair_sort_key=(0, 0, -1, 1, 1, sort_tail, f"{first}__{second}"),
        affected_student_union_count=1,
        changed_candidate_period_relationships=1,
        total_absolute_period_displacement=sort_tail,
        evaluator_result_hash="hash",
    )


def test_survivor_sort_key_is_deterministic() -> None:
    result = survivor("B", "C", sort_tail=9)

    assert k2.survivor_sort_key(result) == k2.survivor_sort_key(result)
    assert k2.survivor_sort_key(replace(result, pair_sort_key=None))[-1] == result.pair_id


def test_candidate_from_pair_result_uses_canonical_destinations() -> None:
    result = survivor("A", "B")
    edit = k2.candidate_from_pair_result(result, domains())

    assert edit.logical_section_ids == ("A", "B")
    assert edit.proposed_placements == result.canonical_destinations
    assert edit.original_placements == (("P1",), ("P4",))


def test_select_pair_portfolio_enforces_unique_pairs_and_excludes_previous() -> None:
    results = [
        survivor("A", "B", previous=True),
        survivor("A", "C", sort_tail=1),
        survivor("D", "E", sort_tail=2),
    ]

    selected, audit = k2.select_pair_portfolio(results, domains(), max_size=2)

    assert [edit.logical_section_ids for edit in selected] == [("A", "C"), ("D", "E")]
    assert audit["unique_section_id_pair_count"] == 2
    assert audit["previously_excluded_pair_in_portfolio"] is False


def test_select_pair_portfolio_records_diversity_relaxation() -> None:
    results = [
        survivor("A", "B", course_a="X", course_b="Y", sort_tail=1),
        survivor("C", "D", course_a="X", course_b="Y", sort_tail=2),
    ]

    selected, audit = k2.select_pair_portfolio(results, domains(), max_size=2)

    assert len(selected) == 2
    assert audit["relaxation_steps"] == ["same_course_pair_cap_relaxed_to_2"]


def test_select_pair_portfolio_respects_section_participation_cap() -> None:
    results = [
        survivor("A", "B", sort_tail=1),
        survivor("A", "C", sort_tail=2),
        survivor("A", "D", sort_tail=3),
    ]

    selected, audit = k2.select_pair_portfolio(results, domains(), max_size=3)

    assert len(selected) == 3
    assert audit["section_participation_histogram"]["A"] == 3
    assert "section_participation_cap_relaxed_to_3" in audit["relaxation_steps"]


def fake_build() -> SimpleNamespace:
    return SimpleNamespace(
        model=FakeModel(),
        section_changed_vars={"A": FakeVar("A"), "B": FakeVar("B"), "C": FakeVar("C")},
        assignment_vars={},
        placement_choice_vars={},
        allocation_input=SimpleNamespace(logical_sections=()),
        placement_domains={},
        occupancy_mode="hybrid_sparse_linear_occupancy",
    )


def test_add_fixed_pair_constraints_forces_two_changed_sections() -> None:
    build = fake_build()

    k2.add_fixed_pair_constraints(build, ("A", "B"))

    assert build.model.constraints == [("eq", "A", 1), ("eq", "B", 1), ("eq", "C", 0)]


def test_model_size_does_not_require_bytesize_and_records_binary_measurements(tmp_path: Path) -> None:
    build = fake_build()

    size = k2._model_size(build, export_path=tmp_path / "model.pb")

    assert size["serialized_binary_proto_bytes"] == size["exported_binary_proto_file_bytes"]
    assert size["binary_measurements_equal"] is True
    assert size["binary_proto_bytes"] == size["serialized_binary_proto_bytes"]
    assert size["proto_text_bytes"] >= size["serialized_binary_proto_bytes"]
    assert size["proto_measurement_method"] == "ExportToFile_pb_and_cp_model_pb2_deterministic_SerializeToString"


def test_recovery_provenance_constants_do_not_claim_original_solved_proto() -> None:
    assert k2.RECOVERED_MODEL_PROTO_PROVENANCE == {
        "model_proto_origin": "reporting_only_rebuild_after_solver",
        "model_proto_is_original_solved_proto": False,
        "original_solved_proto_persisted": False,
        "model_proto_fingerprint_match_to_solved_model": "unverified",
        "model_rebuild_used_same_frozen_inputs_and_builder": True,
        "model_rebuild_solver_invocations": 0,
    }
    assert k2.RECONSTRUCTED_SOLVER_CONFIG_PROVENANCE == {
        "solver_config_origin": "reconstructed_from_invoked_command_candidate_and_retained_evidence",
        "solver_config_original_pre_solve_file_persisted": False,
        "status_evidence_source": "raw_solver_log_final_summary",
        "runtime_evidence_source": "raw_solver_log_and_terminal_transcript",
    }


def test_response_payload_keeps_unavailable_response_hash_null() -> None:
    payload = k2._response_payload(search_result("INFEASIBLE"))
    assert payload["response_hash"].startswith("hash-INFEASIBLE")
    assert payload["response_hash_verified"] is True

    missing = search_result("INFEASIBLE")
    missing = replace(missing, response_hash="unavailable_artifact_write_failure_after_solver")
    payload = k2._response_payload(missing)

    assert payload["response_hash"] is None
    assert payload["response_hash_available"] is False
    assert payload["response_hash_verified"] is False
    assert payload["response_hash_unavailable_reason"] == "post_solve_artifact_write_failure_before_structured_response_persistence"


def test_raw_solver_log_final_status_requires_summary_status(tmp_path: Path) -> None:
    log = tmp_path / "solver.log"
    log.write_text("CpSolverResponse summary:\nstatus: INFEASIBLE\n", encoding="utf-8")
    assert k2._raw_solver_log_final_status(log) == "INFEASIBLE"

    log.write_text("INFEASIBLE appeared earlier but no final summary\n", encoding="utf-8")
    assert k2._raw_solver_log_final_status(log) is None


def test_fixed_pair_run_a_protocol_has_no_hint_or_objective(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, object] = {}

    def fake_build_fixed_pair_model(*args: object, **kwargs: object) -> SimpleNamespace:
        return fake_build()

    def fake_solve(build: SimpleNamespace, **kwargs: object) -> SearchResult:
        build.model.ClearObjective()
        observed["objective"] = build.model.objective
        observed["kwargs"] = kwargs
        return search_result("UNKNOWN")

    monkeypatch.setattr(k2, "build_fixed_pair_model", fake_build_fixed_pair_model)
    monkeypatch.setattr(k2, "solve_fixed_pair_no_hint", fake_solve)

    build, hint, result = k2.fixed_pair_feasibility_run(
        fake_input(),
        domains(),
        candidate(),
        math_fallback_rules=(),
        math_course_ids=(),
    )

    assert build.model.objective is None
    assert observed["objective"] is None
    assert hint["hint_used"] is False
    assert hint["objective_used"] is False
    assert hint["full_domain_preserved"] is True
    assert hint["candidate_pruning"] is False
    assert result.status == "UNKNOWN"


def test_fixed_pair_run_b_protocol_uses_hints_without_domain_pruning(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, object] = {}

    class FakeBaseline:
        assignments = [SimpleNamespace(request_key="REQ1", linked_section_group_id="A")]

    context = SimpleNamespace(
        students=SimpleNamespace(copy=lambda deep=True: "students"),
        requests=SimpleNamespace(copy=lambda deep=True: "requests"),
        sections="sections",
        catalog=SimpleNamespace(copy=lambda deep=True: "catalog"),
        allocation_input=fake_input(),
    )

    monkeypatch.setattr(k2, "apply_placement_map_to_sections", lambda ctx, placement_map: ("edited", placement_map))
    monkeypatch.setattr(k2, "canonicalize_allocation_input", lambda *args: "edited_input")
    monkeypatch.setattr(k2, "run_constrained_first_baseline", lambda *args, **kwargs: FakeBaseline())
    monkeypatch.setattr(k2, "build_fixed_pair_model", lambda *args, **kwargs: fake_build())
    monkeypatch.setattr(k2, "hamming_expression", lambda build, assignment_hint: ("hamming", tuple(assignment_hint)))

    def fake_apply_hints(build: SimpleNamespace, edit: CandidateEdit, assignment_hint: object) -> dict[str, object]:
        observed["assignment_hint"] = tuple(assignment_hint)
        return {"placement_hint_positive_count": 2}

    def fake_solve(build: SimpleNamespace, **kwargs: object) -> SearchResult:
        observed["objective"] = build.model.objective
        observed["kwargs"] = kwargs
        return search_result("UNKNOWN")

    monkeypatch.setattr(k2, "apply_bootstrap_hints", fake_apply_hints)
    monkeypatch.setattr(k2, "solve_bootstrap", fake_solve)

    build, hint, result = k2.fixed_pair_guided_run(
        context,
        domains(),
        candidate(),
        config_dir=Path("data/config"),
        math_fallback_rules=(),
        math_course_ids=(),
    )

    assert build.model.objective[0] == "hamming"
    assert hint["hint_used"] is True
    assert hint["objective_used"] is True
    assert hint["full_domain_preserved"] is True
    assert hint["candidate_pruning"] is False
    assert hint["feasible_region_same_as_feasibility_run"] is True
    assert result.status == "UNKNOWN"
    assert observed["assignment_hint"]


def test_protocol_skips_b_after_a_infeasible(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(k2, "fixed_pair_feasibility_run", lambda *args, **kwargs: (fake_build(), {}, search_result("INFEASIBLE")))
    monkeypatch.setattr(k2, "fixed_pair_guided_run", lambda *args, **kwargs: pytest.fail("Run B should not execute after A INFEASIBLE"))
    monkeypatch.setattr(k2, "_write_solver_run", lambda *args, **kwargs: None)

    outcome = k2.run_fixed_pair_protocol(
        pair_index=0,
        candidate=candidate(),
        context=SimpleNamespace(allocation_input=fake_input()),
        domains=domains(),
        math_fallback_rules=(),
        math_course_ids=(),
        config_dir=Path("data/config"),
        output=tmp_path,
    )

    assert outcome.newly_proven_infeasible is True
    assert outcome.unresolved is False
    assert [row["status"] for row in outcome.run_rows] == ["INFEASIBLE"]


def test_protocol_runs_b_after_a_unknown(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(k2, "fixed_pair_feasibility_run", lambda *args, **kwargs: (fake_build(), {}, search_result("UNKNOWN")))
    monkeypatch.setattr(k2, "fixed_pair_guided_run", lambda *args, **kwargs: (fake_build(), {}, search_result("UNKNOWN")))
    monkeypatch.setattr(k2, "_write_solver_run", lambda *args, **kwargs: None)

    outcome = k2.run_fixed_pair_protocol(
        pair_index=0,
        candidate=candidate(),
        context=SimpleNamespace(allocation_input=fake_input()),
        domains=domains(),
        math_fallback_rules=(),
        math_course_ids=(),
        config_dir=Path("data/config"),
        output=tmp_path,
    )

    assert outcome.newly_proven_infeasible is False
    assert outcome.unresolved is True
    assert [row["status"] for row in outcome.run_rows] == ["UNKNOWN", "UNKNOWN"]


def test_protocol_stops_after_incumbent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(k2, "fixed_pair_feasibility_run", lambda *args, **kwargs: (fake_build(), {}, search_result("FEASIBLE", incumbent=True)))
    monkeypatch.setattr(k2, "fixed_pair_guided_run", lambda *args, **kwargs: pytest.fail("Run B should not execute after A incumbent"))
    monkeypatch.setattr(k2, "_write_solver_run", lambda *args, **kwargs: None)
    monkeypatch.setattr(k2, "validate_bootstrap_witness", lambda *args, **kwargs: {"joint_bootstrap_witness_valid": True, "changed_logical_section_count": 2})

    outcome = k2.run_fixed_pair_protocol(
        pair_index=0,
        candidate=candidate(),
        context=SimpleNamespace(allocation_input=fake_input()),
        domains=domains(),
        math_fallback_rules=(),
        math_course_ids=(),
        config_dir=Path("data/config"),
        output=tmp_path,
    )

    assert outcome.incumbent_source is not None
    assert outcome.incumbent_source["run_name"] == "feasibility"
    assert outcome.newly_proven_infeasible is False


def test_unknown_status_is_not_reported_as_infeasible() -> None:
    row = k2.diagnostic_run_row("pair", "feasibility", search_result("UNKNOWN"))

    assert row["status"] == "UNKNOWN"
    assert row["incumbent_found"] is False


def test_resume_completed_run_does_not_reexecute(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output = tmp_path / "artifact"
    output.mkdir()
    (output / "aggregate_summary.json").write_text('{"result_classification": "done"}', encoding="utf-8")
    monkeypatch.setattr(k2, "verify_source_artifacts", lambda *args, **kwargs: pytest.fail("resume should return aggregate before source verification"))

    result = k2.run_screening_audit(output_dir=output, resume=True)

    assert result["result_classification"] == "done"
    assert result["resumed"] is True
    assert result["solver_reexecuted"] is False


def test_nonempty_artifact_refuses_overwrite_without_resume(tmp_path: Path) -> None:
    output = tmp_path / "artifact"
    output.mkdir()
    (output / "existing.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(k2.ScreeningError, match="refusing overwrite"):
        k2.run_screening_audit(output_dir=output, resume=False)


def test_screening_only_writes_static_outputs_without_solver(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(k2, "verify_source_artifacts", lambda manifest: {"sources": "verified"})
    monkeypatch.setattr(k2, "load_target_context_and_domains", lambda **kwargs: (SimpleNamespace(allocation_input=fake_input()), domains(), {"ok": True}))
    monkeypatch.setattr(k2, "structural_revalidation", lambda *args, **kwargs: {"structural": "ok"})
    monkeypatch.setattr(k2, "zero_edit_core_student_verification", lambda context: {"zero": "ok"})
    monkeypatch.setattr(k2, "build_core_profile", lambda *args, **kwargs: core_profile("A"))
    monkeypatch.setattr(k2, "section_effect_signatures", lambda *args, **kwargs: [])
    monkeypatch.setattr(k2, "run_static_pair_screening", lambda **kwargs: ([survivor("A", "B")], {"evaluator_cache_hits": 0, "evaluator_cache_misses": 1, "unique_effect_signature_count": 1, "screening_runtime_seconds": 0.0, "total_unique_pairs": 1, "core_screen_survivor_count": 1}))
    monkeypatch.setattr(k2, "select_pair_portfolio", lambda *args, **kwargs: ((candidate(),), {"portfolio_hash": "hash", "previously_excluded_pair_in_portfolio": False}))
    monkeypatch.setattr(k2, "run_fixed_pair_protocol", lambda *args, **kwargs: pytest.fail("screening_only must not invoke fixed-pair solver protocol"))
    monkeypatch.setattr(k2, "_previous_k1_proof_verified", lambda: True)

    result = k2.run_screening_audit(output_dir=tmp_path / "out", screening_only=True)

    assert result["solver_counters"]["total_solver_invocations"] == 0
    assert result["result_classification"] == "unresolved_no_incumbent"


def write_static_artifact_stub(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "aggregate_summary.json").write_text(
        json.dumps(
            {
                "sha256sums_hash": "static-sha",
                "solver_counters": {
                    "fixed_pair_feasibility_runs": 0,
                    "fixed_pair_guided_runs": 0,
                    "total_solver_invocations": 0,
                },
            }
        ),
        encoding="utf-8",
    )
    (root / "checkpoint.json").write_text(
        json.dumps({"screening_complete": True, "aggregate_written": True}),
        encoding="utf-8",
    )
    (root / "source_artifact_verification.json").write_text("{}", encoding="utf-8")
    (root / "structural_revalidation.json").write_text("{}", encoding="utf-8")
    (root / "zero_edit_core_student_verification.json").write_text("{}", encoding="utf-8")
    (root / "pair_screening_summary.json").write_text(
        json.dumps(
            {
                "evaluator_cache_hits": 0,
                "evaluator_cache_misses": 1,
                "unique_effect_signature_count": 1,
                "screening_runtime_seconds": 0.0,
                "total_unique_pairs": 1,
                "core_screen_survivor_count": 1,
                "class_count_closure": True,
                "survivor_proves_global_feasible": False,
            }
        ),
        encoding="utf-8",
    )
    portfolio = k2.portfolio_payload((candidate(), candidate(("C", "D"))), {"portfolio_hash": "hash"})
    for item in portfolio["candidates"]:
        item["occupancy_shape"] = [[1], [1]]
    (root / "selected_pair_portfolio.json").write_text(json.dumps(portfolio), encoding="utf-8")
    (root / "portfolio_diversity_audit.json").write_text(json.dumps({"portfolio_hash": "hash"}), encoding="utf-8")
    (root / "diagnostic_runs.csv").write_text("", encoding="utf-8")


def stub_resume_dependencies(monkeypatch: pytest.MonkeyPatch, statuses: list[str]) -> list[str]:
    calls: list[str] = []
    monkeypatch.setattr(k2, "load_target_context_and_domains", lambda **kwargs: (SimpleNamespace(allocation_input=fake_input(), catalog="catalog"), domains(), {"ok": True}))
    monkeypatch.setattr(k2, "_load_math_fallback_rules", lambda *args, **kwargs: ())
    monkeypatch.setattr(k2, "math_course_ids_from_catalog", lambda catalog: ())
    monkeypatch.setattr(k2, "_previous_k1_proof_verified", lambda: True)
    monkeypatch.setattr(k2, "_write_solver_run", lambda *args, **kwargs: None)

    def fake_protocol(**kwargs: object) -> k2.DiagnosticOutcome:
        call_id = f"portfolio_pair_{kwargs['pair_index'] + 1}"
        calls.append(call_id)
        status = statuses.pop(0)
        return k2.DiagnosticOutcome(
            pair_id=call_id,
            run_rows=(k2.diagnostic_run_row(call_id, "feasibility", search_result(status)),),
            incumbent_source=None,
            newly_proven_infeasible=status == "INFEASIBLE",
            unresolved=status != "INFEASIBLE",
            correctness_failure=False,
        )

    monkeypatch.setattr(k2, "run_fixed_pair_protocol", fake_protocol)
    return calls


def test_max_new_solver_runs_one_executes_only_first_pair_run_a(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    write_static_artifact_stub(tmp_path)
    calls = stub_resume_dependencies(monkeypatch, ["UNKNOWN", "UNKNOWN"])

    result = k2.run_screening_audit(output_dir=tmp_path, resume=True, max_new_solver_runs=1)
    rows = k2._read_csv(tmp_path / "diagnostic_runs.csv")

    assert calls == ["portfolio_pair_1"]
    assert [(row["pair_id"], row["run_name"], row["status"]) for row in rows] == [("portfolio_pair_1", "feasibility", "UNKNOWN")]
    assert result["solver_counters"]["fixed_pair_feasibility_runs"] == 1
    assert result["solver_counters"]["fixed_pair_guided_runs"] == 0
    assert result["execution_counts"]["new_solver_runs_this_invocation"] == 1
    assert result["failures"] == []


def test_max_new_solver_runs_does_not_execute_second_pair(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    write_static_artifact_stub(tmp_path)
    calls = stub_resume_dependencies(monkeypatch, ["INFEASIBLE", "UNKNOWN"])

    result = k2.run_screening_audit(output_dir=tmp_path, resume=True, max_new_solver_runs=1)

    assert calls == ["portfolio_pair_1"]
    assert result["newly_proven_infeasible_unique_pairs"] == ["portfolio_pair_1"]
    assert result["solver_counters"]["total_solver_invocations"] == 1


def test_resume_does_not_repeat_completed_run_a(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    write_static_artifact_stub(tmp_path)
    (tmp_path / "diagnostic_runs.csv").write_text(
        "pair_id,run_name,status,incumbent_found,assignment_available,runtime_seconds,wall_time_seconds,branches,conflicts,response_hash\n"
        "portfolio_pair_1,feasibility,UNKNOWN,False,False,1.0,1.0,0,0,hash\n",
        encoding="utf-8",
    )
    calls = stub_resume_dependencies(monkeypatch, ["UNKNOWN"])

    result = k2.run_screening_audit(output_dir=tmp_path, resume=True, max_new_solver_runs=1)

    assert calls == ["portfolio_pair_2"]
    assert result["solver_counters"]["fixed_pair_feasibility_runs"] == 1
    assert result["execution_counts"]["new_solver_runs_this_invocation"] == 1


def test_default_resume_with_aggregate_keeps_existing_behavior(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    write_static_artifact_stub(tmp_path)
    monkeypatch.setattr(k2, "load_target_context_and_domains", lambda **kwargs: pytest.fail("default resume should return aggregate without reloading context"))

    result = k2.run_screening_audit(output_dir=tmp_path, resume=True)

    assert result["resumed"] is True
    assert result["solver_reexecuted"] is False


def test_protocol_deviation_is_explicitly_recorded() -> None:
    row = k2.diagnostic_run_row("pair", "feasibility", search_result("MODEL_INVALID"))

    assert row["status"] == "MODEL_INVALID"
    assert row["incumbent_found"] is False


def test_candidate_from_pair_result_requires_canonical_destination() -> None:
    result = replace(survivor("A", "B"), canonical_destinations=None)

    with pytest.raises(k2.ScreeningError, match="no canonical destinations"):
        k2.candidate_from_pair_result(result, domains())


def test_invalid_pair_domain_is_classified_without_solver() -> None:
    broken = domains()
    broken["A"] = (placement("A", ("P1",), original=True),)

    result = k2.screen_pair(
        "A",
        "B",
        profile=core_profile("A"),
        domains=broken,
        allocation_input=fake_input(),
        evaluation_cache={},
    )

    assert result.invalid_domain_pair is True
    assert result.final_class == "invalid_domain_pair"


FORMAL_ARTIFACT = Path(
    "/Users/klu/Projects/fair-course-allocation-artifacts/robustness-v1/"
    "hybrid-k2-section-pair-screening-v1"
)
DRY_RUN_ARTIFACT = Path("/tmp/hybrid-k2-section-pair-screening-dryrun-20260724-105148")
EXPECTED_PORTFOLIO_HASH = "ef83de1d2dfecaa6f55b8d074156466d96f73c5334be61d2aba856819445fd67"


def read_artifact_json(root: Path, name: str) -> dict[str, object]:
    if not root.is_dir():
        pytest.skip(f"artifact is not available: {root}")
    return json.loads((root / name).read_text(encoding="utf-8"))


def test_formal_artifact_matches_static_dry_run_counts() -> None:
    formal = read_artifact_json(FORMAL_ARTIFACT, "pair_screening_summary.json")
    dry = read_artifact_json(DRY_RUN_ARTIFACT, "pair_screening_summary.json")
    frozen_keys = (
        "total_unique_pairs",
        "raw_placement_combination_count",
        "core_necessary_condition_failed_count",
        "core_screen_survivor_count",
        "previously_proven_infeasible_pairs",
        "screening_error_count",
        "unclassified_pair_count",
        "class_count_closure",
    )

    assert {key: formal[key] for key in frozen_keys} == {key: dry[key] for key in frozen_keys}
    assert formal["total_unique_pairs"] == 48516
    assert formal["raw_placement_combination_count"] == 139415
    assert formal["core_necessary_condition_failed_count"] == 47278
    assert formal["core_screen_survivor_count"] == 1237


def test_formal_artifact_portfolio_hash_and_diversity() -> None:
    portfolio = read_artifact_json(FORMAL_ARTIFACT, "selected_pair_portfolio.json")
    audit = read_artifact_json(FORMAL_ARTIFACT, "portfolio_diversity_audit.json")

    assert portfolio["count"] == 6
    assert portfolio["portfolio_hash"] == EXPECTED_PORTFOLIO_HASH
    assert audit["portfolio_hash"] == EXPECTED_PORTFOLIO_HASH
    assert audit["unique_section_id_pair_count"] == 6
    assert audit["unique_course_pair_count"] == 6
    assert audit["relaxation_steps"] == [
        "same_course_pair_cap_relaxed_to_2",
        "section_participation_cap_relaxed_to_3",
    ]


def test_formal_artifact_class_closure_and_run_a_solver_counters() -> None:
    aggregate = read_artifact_json(FORMAL_ARTIFACT, "aggregate_summary.json")
    screening = aggregate["screening"]
    counters = aggregate["solver_counters"]

    assert screening["class_count_closure"] is True
    assert screening["survivor_proves_global_feasible"] is False
    assert aggregate["global_k2_remains_unresolved"] is True
    assert aggregate["minimum_claim"]["proven"] is False
    assert counters["fixed_pair_feasibility_runs"] == 3
    assert counters["fixed_pair_guided_runs"] == 0
    assert counters["total_solver_invocations"] == 3
    assert counters["production_fixed_witness_acceptance_runs"] == 0
    assert counters["production_validation_runs"] == 0
    assert counters["global_k2_reruns"] == 0
    assert counters["k1_runs"] == 0
    assert counters["k3_runs"] == 0


def test_formal_artifact_execution_counts_include_three_run_a_attempts() -> None:
    aggregate = read_artifact_json(FORMAL_ARTIFACT, "aggregate_summary.json")
    provenance = read_artifact_json(FORMAL_ARTIFACT, "provenance.json")

    assert aggregate["execution_counts"] == {
        "exploratory_dry_runs": 1,
        "accepted_formal_static_screening_runs": 1,
        "total_static_screening_executions": 2,
        "total_solver_invocations": 3,
        "new_solver_runs_this_invocation": 1,
    }
    assert provenance["exploratory_dry_runs"] == 1
    assert provenance["accepted_formal_static_screening_runs"] == 1
    assert provenance["total_static_screening_executions"] == 2
    assert provenance["total_solver_invocations"] == 3
    assert provenance["fixed_pair_feasibility_runs"] == 3
    assert provenance["fixed_pair_guided_runs"] == 0
    assert provenance["production_validation_runs"] == 0


def test_formal_artifact_marks_model_proto_as_reporting_only_rebuild() -> None:
    model_size = read_artifact_json(FORMAL_ARTIFACT, "runs/portfolio_pair_1/feasibility/model_size.json")
    response = read_artifact_json(FORMAL_ARTIFACT, "runs/portfolio_pair_1/feasibility/response_stats.json")
    aggregate = read_artifact_json(FORMAL_ARTIFACT, "aggregate_summary.json")

    for payload in (model_size, response, aggregate):
        assert payload["model_proto_origin"] == "reporting_only_rebuild_after_solver"
        assert payload["model_proto_is_original_solved_proto"] is False
        assert payload["original_solved_proto_persisted"] is False
        assert payload["model_proto_fingerprint_match_to_solved_model"] == "unverified"
        assert payload["model_rebuild_used_same_frozen_inputs_and_builder"] is True
        assert payload["model_rebuild_solver_invocations"] == 0
    assert model_size["binary_proto_bytes"] == 101734174
    assert model_size["proto_text_bytes"] == 292100197


def test_formal_artifact_records_reconstructed_solver_config_evidence() -> None:
    solver_config = read_artifact_json(FORMAL_ARTIFACT, "runs/portfolio_pair_1/feasibility/solver_config.json")
    response = read_artifact_json(FORMAL_ARTIFACT, "runs/portfolio_pair_1/feasibility/response_stats.json")
    validation = read_artifact_json(FORMAL_ARTIFACT, "runs/portfolio_pair_1/feasibility/validation.json")

    for payload in (solver_config, response, validation):
        assert payload["solver_config_origin"] == "reconstructed_from_invoked_command_candidate_and_retained_evidence"
        assert payload["solver_config_original_pre_solve_file_persisted"] is False
        assert payload["status_evidence_source"] == "raw_solver_log_final_summary"
        assert payload["runtime_evidence_source"] == "raw_solver_log_and_terminal_transcript"
        assert payload["response_hash"] is None
        assert payload["response_hash_verified"] is False
    assert response["response_hash_available"] is False
    assert response["status_evidence_status"] == "INFEASIBLE"


def test_formal_artifact_pair_two_run_a_is_infeasible_without_hints() -> None:
    solver_config = read_artifact_json(FORMAL_ARTIFACT, "runs/portfolio_pair_2/feasibility/solver_config.json")
    hint_audit = read_artifact_json(FORMAL_ARTIFACT, "runs/portfolio_pair_2/feasibility/hint_audit.json")
    response = read_artifact_json(FORMAL_ARTIFACT, "runs/portfolio_pair_2/feasibility/response_stats.json")

    assert solver_config["fixed_section_ids"] == ["AP_JAPANESE_LANG_01", "SOCIAL_JUSTICE_01"]
    assert solver_config["seed"] == 20260630
    assert solver_config["workers"] == 1
    assert solver_config["max_time_in_seconds"] == 75.0
    assert solver_config["objective"] == "none"
    assert solver_config["hint"] == "none"
    assert solver_config["stop_after_first_solution"] is True
    assert hint_audit["hint_used"] is False
    assert hint_audit["objective_used"] is False
    assert hint_audit["candidate_pruning"] is False
    assert hint_audit["full_domain_preserved"] is True
    assert response["status"] == "INFEASIBLE"
    assert response["incumbent_found"] is False
    assert response["assignment_available"] is False
    assert response["branches"] == 0
    assert response["conflicts"] == 0


def test_formal_artifact_pair_three_run_a_is_infeasible_without_hints() -> None:
    solver_config = read_artifact_json(FORMAL_ARTIFACT, "runs/portfolio_pair_3/feasibility/solver_config.json")
    hint_audit = read_artifact_json(FORMAL_ARTIFACT, "runs/portfolio_pair_3/feasibility/hint_audit.json")
    response = read_artifact_json(FORMAL_ARTIFACT, "runs/portfolio_pair_3/feasibility/response_stats.json")

    assert solver_config["fixed_section_ids"] == ["AP_JAPANESE_LANG_01", "CREATIVE_WRITING_01"]
    assert solver_config["seed"] == 20260630
    assert solver_config["workers"] == 1
    assert solver_config["max_time_in_seconds"] == 75.0
    assert solver_config["objective"] == "none"
    assert solver_config["hint"] == "none"
    assert solver_config["stop_after_first_solution"] is True
    assert hint_audit["hint_used"] is False
    assert hint_audit["objective_used"] is False
    assert hint_audit["candidate_pruning"] is False
    assert hint_audit["full_domain_preserved"] is True
    assert response["status"] == "INFEASIBLE"
    assert response["incumbent_found"] is False
    assert response["assignment_available"] is False
    assert response["branches"] == 0
    assert response["conflicts"] == 0


def test_formal_artifact_scoped_infeasible_conclusion_keeps_global_k2_unresolved() -> None:
    aggregate = read_artifact_json(FORMAL_ARTIFACT, "aggregate_summary.json")
    validation = read_artifact_json(FORMAL_ARTIFACT, "runs/portfolio_pair_1/feasibility/validation.json")

    for payload in (aggregate, validation):
        assert payload["fixed_section_pair_infeasible"] is True
        assert payload["fixed_section_pair_infeasible_scope"]["scenario_id"] == "normal_dev_10"
        assert payload["fixed_section_pair_infeasible_scope"]["logical_section_ids"] == [
            "AP_3D_ART_DESIGN_01",
            "CREATIVE_WRITING_01",
        ]
        assert payload["fixed_section_pair_infeasible_scope"]["both_selected_sections_forced_changed"] is True
        assert payload["fixed_section_pair_infeasible_scope"]["destination_domain"] == (
            "full_frozen_non_original_destination_domains"
        )
    assert aggregate["global_k2_remains_unresolved"] is True
    assert aggregate["lower_bound_remains"] == 2
    assert aggregate["minimum_claim"]["proven"] is False
    assert aggregate["exact_minimum_claim"] is False
    assert aggregate["repair_witness_found"] is False
    assert aggregate["production_validation_runs"] == 0


def test_formal_artifact_checksum_file_is_current() -> None:
    if not FORMAL_ARTIFACT.is_dir():
        pytest.skip(f"artifact is not available: {FORMAL_ARTIFACT}")
    checksum_file = FORMAL_ARTIFACT / "SHA256SUMS.txt"
    entries = [line for line in checksum_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    assert len(entries) == 40
    assert any(line.endswith("runs/portfolio_pair_1/feasibility/model.pb") for line in entries)
    assert any(line.endswith("runs/portfolio_pair_2/feasibility/model.pb") for line in entries)
    assert any(line.endswith("runs/portfolio_pair_3/feasibility/model.pb") for line in entries)
    for line in entries:
        expected, relative = line.split("  ", 1)
        actual = hashlib.sha256((FORMAL_ARTIFACT / relative).read_bytes()).hexdigest()
        assert actual == expected
