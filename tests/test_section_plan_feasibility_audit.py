from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from src.cp_sat_robustness_runner import CpSatEvaluationError
import src.section_plan_feasibility_audit as audit
from src.section_plan_feasibility_audit import (
    CONTROL_SCENARIO_ID,
    FAMILY_NAMES,
    SectionPlanAuditError,
    build_repair_candidates,
    build_static_descriptors,
    fine_grained_core,
    group_level_core,
    load_section_plan_audit_manifest,
    rebuild_section_plan_audit_reporting,
    run_counterfactual_variants,
    run_relaxation_with_fallback_layer,
    validate_relaxation_witness,
    verify_diagnostic_model_equivalence,
    _build_diagnostic_model,
    _constrained_first_full_hint_seed,
)

from tests.test_cp_sat_solver import canonical, fallback_rules, math_ids, request_row, section_row


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _feasible_fixture():
    courses = ["CORE_A", "CORE_B", "CORE_C", "CORE_D", "ALT1"]
    requests = [request_row("STU_1", course) for course in courses]
    sections = [
        section_row(f"SEC_{course}", course, f"P{index + 1}", capacity=10, group_id=f"{course}_1")
        for index, course in enumerate(courses)
    ]
    return canonical([("STU_1", 12, 5, False)], requests, sections)


def _capacity_and_minimum_five_infeasible_fixture():
    courses = ["CORE_A", "CORE_B", "CORE_C", "CORE_D", "ALT1"]
    students = [(f"STU_{i}", 12, 5, False) for i in range(5)]
    requests = [request_row(student_id, course) for student_id, *_ in students for course in courses]
    sections = [
        section_row(f"SEC_{course}", course, f"P{index + 1}", capacity=1, group_id=f"{course}_1")
        for index, course in enumerate(courses)
    ]
    return canonical(students, requests, sections)


def _minimum_five_only_infeasible_fixture():
    # Ample capacity, but only 4 catalog courses exist so a target of 5
    # logical courses can never be reached regardless of section supply.
    courses = ["CORE_A", "CORE_B", "CORE_C", "CORE_D"]
    requests = [request_row("STU_1", course) for course in courses]
    sections = [
        section_row(f"SEC_{course}", course, f"P{index + 1}", capacity=10, group_id=f"{course}_1")
        for index, course in enumerate(courses)
    ]
    return canonical([("STU_1", 12, 5, False)], requests, sections)


def _period_structure_infeasible_fixture():
    # Two courses forced onto the same single period leave no way to take
    # both, even though total capacity is ample.
    requests = [request_row("STU_1", "CORE_A"), request_row("STU_1", "CORE_B")]
    sections = [
        section_row("SEC_A", "CORE_A", "P1", capacity=10, group_id="A"),
        section_row("SEC_B", "CORE_B", "P1", capacity=10, group_id="B"),
    ]
    return canonical([("STU_1", 12, 2, False)], requests, sections)


def _multi_family_infeasible_fixture():
    return _capacity_and_minimum_five_infeasible_fixture()


def _write_reporting_fixture(root: Path) -> None:
    scenario_ids = [
        "normal_dev_01", "normal_dev_03", "normal_dev_04", "normal_dev_05",
        "normal_dev_07", "normal_dev_09", "normal_dev_10",
    ]
    for scenario_id in scenario_ids:
        scenario = root / "scenarios" / scenario_id
        scenario.mkdir(parents=True)
        core_student = f"{scenario_id}_CORE"
        witness_student = "G12_0105" if scenario_id == "normal_dev_10" else f"{scenario_id}_WITNESS"
        labels = ["period_supply_misalignment"]
        if scenario_id == "normal_dev_09":
            labels.append("primary_protection_interaction")
        for name, payload in {
            "group_core.json": {
                "sufficient_core": ["ORDINARY_MAX_PRIMARY_UNMET", "STUDENT_PERIOD_CONFLICT"],
                "locally_minimal_core": ["ORDINARY_MAX_PRIMARY_UNMET", "STUDENT_PERIOD_CONFLICT"],
            },
            "fine_core.json": {
                "involved_students": [core_student],
                "minimality_status": "unresolved_time_budget",
            },
            "relaxation_witness.json": {
                "valid": False,
                "issues": [f"ordinary_extra_unmet_mismatch:{witness_student}"],
            },
            "relaxation_stage_trace.json": {
                "stages": [
                    {"stage_name": "stage_1", "status": "OPTIMAL", "optimality_proven": True},
                    {"stage_name": "stage_2", "status": "UNKNOWN", "optimality_proven": False},
                ]
            },
            "counterfactual_variants.json": {
                "variants": [
                    {"variant": "period_conflict_only", "status": "OPTIMAL"},
                    {"variant": "capacity_only", "status": "INFEASIBLE"},
                ]
            },
            "section_plan_repair_candidates.json": {
                "classification": {"labels": labels}
            },
        }.items():
            (scenario / name).write_text(json.dumps(payload), encoding="utf-8")
    (root / "scenario_classifications.csv").write_text(
        "scenario_id,labels\n" + "\n".join(f"{sid},period_supply_misalignment" for sid in scenario_ids) + "\n",
        encoding="utf-8",
    )
    (root / "relaxation_summary.csv").write_text(
        "scenario_id,stage_name,status,objective_value\n"
        + "\n".join(f"{sid},stage_1,OPTIMAL,1" for sid in scenario_ids) + "\n",
        encoding="utf-8",
    )
    (root / "run_manifest.json").write_text(json.dumps({"source_git_commit": "test"}), encoding="utf-8")
    sums = []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.name != "SHA256SUMS.txt"):
        sums.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.relative_to(root)}")
    (root / "SHA256SUMS.txt").write_text("\n".join(sums) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# 1. Manifest / provenance
# ---------------------------------------------------------------------------


def test_manifest_has_one_control_and_seven_targets() -> None:
    payload = load_section_plan_audit_manifest()
    ids = [item["scenario_id"] for item in payload["scenarios"]]
    assert ids[0] == CONTROL_SCENARIO_ID
    assert len(ids) == 8
    roles = {item["scenario_id"]: item["role"] for item in payload["scenarios"]}
    assert roles[CONTROL_SCENARIO_ID] == "feasible_control"
    assert all(role == "infeasible_target" for sid, role in roles.items() if sid != CONTROL_SCENARIO_ID)


def test_manifest_rejects_stress_or_holdout(tmp_path) -> None:
    payload = load_section_plan_audit_manifest()
    payload["scenarios"][-1] = {**payload["scenarios"][-1], "scenario_id": "stress_bad"}
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SectionPlanAuditError):
        load_section_plan_audit_manifest(path)


def test_manifest_forbids_holdout_and_stress_execution() -> None:
    payload = load_section_plan_audit_manifest()
    assert payload["holdout_execution_allowed"] is False
    assert payload["stress_execution_allowed"] is False


def test_manifest_solver_seed_and_workers_are_frozen(tmp_path) -> None:
    payload = load_section_plan_audit_manifest()
    payload["solver_seed"] = 1
    path = tmp_path / "bad_seed.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SectionPlanAuditError):
        load_section_plan_audit_manifest(path)


def test_manifest_time_budgets_are_present() -> None:
    payload = load_section_plan_audit_manifest()
    budgets = payload["time_budgets_seconds"]
    assert budgets["group_core"] == 60
    assert budgets["fine_core"] == 120
    assert budgets["counterfactual_variant"] == 30
    assert budgets["relaxation_stage_1"] == 120
    assert budgets["relaxation_stage_2"] == 120
    assert budgets["relaxation_stage_3"] == 60


# ---------------------------------------------------------------------------
# 2. Static descriptors
# ---------------------------------------------------------------------------


def test_global_supply_capacity_is_deduplicated_by_logical_section() -> None:
    data = _feasible_fixture()
    descriptors = build_static_descriptors(data, time_limit_seconds=5, seed=20260630)
    supply = descriptors["global_supply"]
    # 5 logical sections of capacity 10 each -> 50, not double-counted by
    # any underlying member-section rows.
    assert supply["total_logical_seat_capacity"] == sum(section.capacity for section in data.logical_sections)
    assert supply["logical_sections"] == len(data.logical_sections)


def test_course_demand_descriptor_reports_ratios_and_shortfall() -> None:
    data = _capacity_and_minimum_five_infeasible_fixture()
    descriptors = build_static_descriptors(data, time_limit_seconds=5, seed=20260630)
    demand = descriptors["course_demand"]
    assert demand["total_course_level_capacity_only_shortfall"] > 0
    assert set(demand["primary_demand_over_capacity_courses"]) == {"CORE_A", "CORE_B", "CORE_C", "CORE_D", "ALT1"}
    assert "note" in demand


def test_student_max_load_finds_below_five_students_when_only_four_courses_exist() -> None:
    data = _minimum_five_only_infeasible_fixture()
    descriptors = build_static_descriptors(data, time_limit_seconds=5, seed=20260630)
    max_load = descriptors["student_max_load"]
    assert max_load["solve_status"] in {"OPTIMAL", "FEASIBLE"}
    assert "STU_1" in max_load["students_below_five"]


def test_zero_candidate_primary_requests_are_reported() -> None:
    data = _feasible_fixture()
    descriptors = build_static_descriptors(data, time_limit_seconds=5, seed=20260630)
    # every primary request in this fixture has a candidate, so none should
    # be flagged -- this asserts the field is populated (not silently 0/None
    # due to a bug) rather than presuming a specific student is affected.
    assert descriptors["student_max_load"]["zero_candidate_primary_students"] == []


def test_period_concentration_descriptor_reports_supply_and_demand_by_period() -> None:
    data = _period_structure_infeasible_fixture()
    descriptors = build_static_descriptors(data, time_limit_seconds=5, seed=20260630)
    concentration = descriptors["period_concentration"]
    assert concentration["supply_by_period"]["P1"] == 20
    assert concentration["primary_candidate_demand_by_period"]["P1"] == 2


# ---------------------------------------------------------------------------
# 3. Diagnostic model equivalence + assumption/core mechanics
# ---------------------------------------------------------------------------


def test_diagnostic_model_reproduces_production_feasible_status() -> None:
    build = _build_diagnostic_model(_feasible_fixture(), (), math_ids())
    equivalence = verify_diagnostic_model_equivalence(build, time_limit_seconds=10, seed=20260630)
    assert equivalence["status"] in {"FEASIBLE", "OPTIMAL"}


def test_diagnostic_model_reproduces_production_infeasible_status() -> None:
    build = _build_diagnostic_model(_capacity_and_minimum_five_infeasible_fixture(), (), math_ids())
    equivalence = verify_diagnostic_model_equivalence(build, time_limit_seconds=10, seed=20260630)
    assert equivalence["status"] == "INFEASIBLE"


def test_group_level_core_is_empty_for_a_feasible_fixture() -> None:
    build = _build_diagnostic_model(_feasible_fixture(), (), math_ids())
    core = group_level_core(build, time_limit_seconds=10, seed=20260630)
    assert core["status"] in {"FEASIBLE", "OPTIMAL"}
    assert core["sufficient_core"] == []
    assert core["locally_minimal_core"] == []


def test_group_level_core_returns_sufficient_and_minimal_core_for_infeasible_fixture() -> None:
    build = _build_diagnostic_model(_capacity_and_minimum_five_infeasible_fixture(), (), math_ids())
    core = group_level_core(build, time_limit_seconds=15, seed=20260630)
    assert core["status"] == "INFEASIBLE"
    assert set(core["sufficient_core"]) >= {"SECTION_CAPACITY", "MINIMUM_FIVE_LOGICAL"}
    assert core["minimality_status"] == "locally_minimal"


def test_sufficient_core_is_never_labeled_minimum() -> None:
    build = _build_diagnostic_model(_capacity_and_minimum_five_infeasible_fixture(), (), math_ids())
    core = group_level_core(build, time_limit_seconds=15, seed=20260630)
    assert "minimum" not in core["minimality_status"]
    assert core["minimality_status"] in {"locally_minimal", "unresolved_time_budget", "not_applicable", "unresolved_no_infeasibility_proof"}


def test_deletion_filtering_reduces_an_over_complete_core() -> None:
    # The sufficient core starts as "all 7 families assumed"; deletion
    # filtering must shrink it to a strict subset for a fixture where only
    # two families are jointly necessary (capacity + minimum-five).
    build = _build_diagnostic_model(_capacity_and_minimum_five_infeasible_fixture(), (), math_ids())
    core = group_level_core(build, time_limit_seconds=15, seed=20260630)
    assert core["status"] == "INFEASIBLE"
    assert set(core["locally_minimal_core"]) <= set(FAMILY_NAMES)
    assert len(core["locally_minimal_core"]) < len(FAMILY_NAMES)
    assert "PROTECTED_PRIMARY" not in core["locally_minimal_core"]
    assert "STUDENT_PERIOD_CONFLICT" not in core["locally_minimal_core"]


def test_fine_grained_core_ids_are_traceable_to_students_and_sections() -> None:
    data = _capacity_and_minimum_five_infeasible_fixture()
    fine = fine_grained_core(
        data, (), math_ids(),
        target_families=("SECTION_CAPACITY", "MINIMUM_FIVE_LOGICAL"),
        time_limit_seconds=15, seed=20260630,
    )
    assert fine["status"] == "INFEASIBLE"
    assert fine["core_size"] > 0
    assert fine["involved_students"]
    assert fine["involved_sections"]


def test_fine_core_with_no_target_families_is_not_applicable() -> None:
    result = fine_grained_core(_feasible_fixture(), (), math_ids(), target_families=(), time_limit_seconds=5, seed=20260630)
    assert result["status"] == "not_applicable"
    assert result["core_size"] == 0


# ---------------------------------------------------------------------------
# 4. Relaxation model
# ---------------------------------------------------------------------------


def _relaxation_hint(data):
    return _constrained_first_full_hint_seed(data, fallback_rules(), math_ids(), 20260630).keys


def test_relaxation_stage_1_minimizes_relaxed_instance_count_and_finds_a_witness() -> None:
    data = _capacity_and_minimum_five_infeasible_fixture()
    result = run_relaxation_with_fallback_layer(
        data, (), math_ids(), stage1_seconds=15, stage2_seconds=15, stage3_seconds=10,
        seed=20260630, hint_keys=_relaxation_hint(data),
    )
    assert result["layer"] == 1
    assert result["stages"][0]["status"] in {"FEASIBLE", "OPTIMAL"}
    assert result["stages"][0]["objective_value"] is not None


def test_relaxation_feasible_is_never_reported_as_optimal_unless_proven() -> None:
    data = _capacity_and_minimum_five_infeasible_fixture()
    result = run_relaxation_with_fallback_layer(
        data, (), math_ids(), stage1_seconds=15, stage2_seconds=15, stage3_seconds=10,
        seed=20260630, hint_keys=_relaxation_hint(data),
    )
    for stage in result["stages"]:
        if stage["status"] == "FEASIBLE":
            assert stage["optimality_proven"] is False
        if stage["optimality_proven"]:
            assert stage["status"] == "OPTIMAL"


def test_relaxation_lexicographic_stages_are_fixed_in_sequence() -> None:
    data = _capacity_and_minimum_five_infeasible_fixture()
    result = run_relaxation_with_fallback_layer(
        data, (), math_ids(), stage1_seconds=15, stage2_seconds=15, stage3_seconds=10,
        seed=20260630, hint_keys=_relaxation_hint(data),
    )
    names = [stage["stage_name"] for stage in result["stages"]]
    assert names[:3] == [
        "stage_1_minimize_relaxed_instance_count",
        "stage_2_minimize_total_slack_magnitude",
        "stage_3_minimize_hamming_to_constrained_first",
    ]


def test_witness_validation_confirms_slack_accounting_and_no_duplicate_identity() -> None:
    data = _capacity_and_minimum_five_infeasible_fixture()
    result = run_relaxation_with_fallback_layer(
        data, (), math_ids(), stage1_seconds=15, stage2_seconds=15, stage3_seconds=10,
        seed=20260630, hint_keys=_relaxation_hint(data),
    )
    witness = validate_relaxation_witness(result["build"], result["final_solver"], data)
    assert witness["valid"] is True
    assert witness["no_duplicate_logical_identity"] is True
    assert witness["capacity_overflow_closure"] is True
    assert witness["student_policy_slack_closure"] is True
    assert witness["response_hash_present"] is True


def test_diagnostic_witness_dict_never_marks_itself_publishable() -> None:
    data = _capacity_and_minimum_five_infeasible_fixture()
    result = run_relaxation_with_fallback_layer(
        data, (), math_ids(), stage1_seconds=15, stage2_seconds=15, stage3_seconds=10,
        seed=20260630, hint_keys=_relaxation_hint(data),
    )
    witness = validate_relaxation_witness(result["build"], result["final_solver"], data)
    assert "publishable" not in witness
    assert "production" not in json.dumps(witness).lower()


# ---------------------------------------------------------------------------
# 5. Counterfactual variants
# ---------------------------------------------------------------------------


def test_counterfactual_variants_only_change_the_targeted_family() -> None:
    # In the capacity+minimum-five fixture, relaxing capacity alone (while
    # keeping minimum-five and every other family hard) restores
    # feasibility -- unlimited seats let every student reach the minimum.
    # Relaxing period-conflict alone does nothing, since this fixture has
    # no period conflicts to begin with.
    build = _build_diagnostic_model(_capacity_and_minimum_five_infeasible_fixture(), (), math_ids())
    rows = run_counterfactual_variants(build, time_limit_seconds=10, seed=20260630)
    by_variant = {row["variant"]: row for row in rows}
    assert by_variant["capacity_only"]["assignment_found"] is True
    assert by_variant["period_conflict_only"]["assignment_found"] is False


def test_counterfactual_variant_schema_is_stable() -> None:
    build = _build_diagnostic_model(_feasible_fixture(), (), math_ids())
    rows = run_counterfactual_variants(build, time_limit_seconds=10, seed=20260630)
    assert len(rows) == 8
    for row in rows:
        assert set(row) == {"variant", "relaxed_families", "status", "runtime_seconds", "assignment_found"}


def test_counterfactual_unknown_status_is_not_treated_as_ineffective(monkeypatch) -> None:
    build = _build_diagnostic_model(_feasible_fixture(), (), math_ids())
    original_solve = audit._solve_with_assumptions

    def _force_unknown(build_arg, literals, *, time_limit_seconds, seed):
        solver, _status = original_solve(build_arg, literals, time_limit_seconds=0.0001, seed=seed)
        return solver, "UNKNOWN"

    monkeypatch.setattr(audit, "_solve_with_assumptions", _force_unknown)
    rows = run_counterfactual_variants(build, time_limit_seconds=10, seed=20260630)
    for row in rows:
        assert row["status"] == "UNKNOWN"
        assert row["assignment_found"] is False  # UNKNOWN is reported as-is, not silently coerced


# ---------------------------------------------------------------------------
# 6. Classification / repair candidates
# ---------------------------------------------------------------------------


def test_classification_has_evidence_for_every_label() -> None:
    data = _capacity_and_minimum_five_infeasible_fixture()
    build = _build_diagnostic_model(data, (), math_ids())
    group_core = group_level_core(build, time_limit_seconds=15, seed=20260630)
    families = tuple(group_core["locally_minimal_core"])
    fine_core = fine_grained_core(data, (), math_ids(), target_families=families, time_limit_seconds=15, seed=20260630)
    relaxation = run_relaxation_with_fallback_layer(
        data, (), math_ids(), stage1_seconds=15, stage2_seconds=15, stage3_seconds=10,
        seed=20260630, hint_keys=_relaxation_hint(data),
    )
    repair = build_repair_candidates("fixture_scenario", data, group_core, fine_core, relaxation)
    assert repair["classification"]["labels"]
    for label in repair["classification"]["labels"]:
        assert label in repair["classification"]["evidence"] or label == "unresolved_multi_family_interaction"


def test_missing_core_or_witness_reports_not_applicable_rather_than_zero() -> None:
    empty_fine = fine_grained_core(_feasible_fixture(), (), math_ids(), target_families=(), time_limit_seconds=5, seed=20260630)
    assert empty_fine["status"] == "not_applicable"
    assert empty_fine["core_size"] == 0
    assert empty_fine["minimality_status"] == "not_applicable"


# ---------------------------------------------------------------------------
# 7. Reporting-only rebuild
# ---------------------------------------------------------------------------


def test_reporting_rebuild_separates_core_and_counterfactual_semantics(tmp_path, monkeypatch) -> None:
    source = tmp_path / "raw"
    audited = tmp_path / "audited"
    _write_reporting_fixture(source)
    source_checksums_before = (source / "SHA256SUMS.txt").read_bytes()

    def _solver_must_not_run(*args, **kwargs):
        raise AssertionError("reporting rebuild must not invoke solver code")

    monkeypatch.setattr(audit, "run_counterfactual_variants", _solver_must_not_run)
    result = rebuild_section_plan_audit_reporting(source, audited)

    aggregate = json.loads((audited / "aggregate_summary.json").read_text())
    rows = {row["scenario_id"]: row for row in csv.DictReader((audited / "scenario_classifications.csv").open())}
    assert result["authoritative_witnesses"] == 0
    assert aggregate["group_core_counts"]["STUDENT_PERIOD_CONFLICT"] == 7
    assert aggregate["counterfactuals"]["student_period_conflict_restored_feasibility"]["restored"] == 7
    assert aggregate["interpretation"]["unsat_core"] != aggregate["interpretation"]["relaxation_counterfactual"]
    assert rows["normal_dev_09"]["primary_classification"] == "period_supply_misalignment"
    assert rows["normal_dev_09"]["secondary_classification"] == "low_confidence_signal"
    assert rows["normal_dev_10"]["witness_authoritative"] == "False"
    assert "G12_0105" not in rows["normal_dev_10"]["fine_core_student_ids"]
    assert rows["normal_dev_10"]["authoritative_repair_recommendations"] == "[]"
    assert source_checksums_before == (source / "SHA256SUMS.txt").read_bytes()
    assert (audited / "SHA256SUMS.txt").is_file()


def test_reporting_rebuild_refuses_nonempty_destination(tmp_path) -> None:
    source = tmp_path / "raw"
    audited = tmp_path / "audited"
    _write_reporting_fixture(source)
    audited.mkdir()
    (audited / "keep.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(SectionPlanAuditError, match="non-empty"):
        rebuild_section_plan_audit_reporting(source, audited)

    assert (audited / "keep.txt").read_text(encoding="utf-8") == "keep"


def test_invalid_witness_cannot_create_authoritative_protection_classification() -> None:
    quality = audit._witness_quality(
        {"stages": [{"status": "OPTIMAL", "optimality_proven": True}, {"status": "UNKNOWN"}]},
        {"valid": False},
    )
    classification = audit._classify_scenario(
        {
            "locally_minimal_core": ["ORDINARY_MAX_PRIMARY_UNMET", "STUDENT_PERIOD_CONFLICT"]
        },
        {},
        [],
        {"students_requiring_minimum_five_slack": [], "students_requiring_max_gap_slack": [],
         "protected_or_high_demand_conflicts": ["G12_0105"]},
        [],
        witness_authoritative=quality["witness_authoritative"],
    )
    assert quality["witness_use"] == "diagnostic_only"
    assert classification["labels"] == ["period_supply_misalignment"]
    assert "primary_protection_interaction" not in classification["labels"]
