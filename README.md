# Fair Course Allocation

## CP-SAT cold-start recovery

Run the development-only internal constrained-first recovery gate. It uses
the persisted normal inputs, evaluates the stable reference first, and writes
artifacts outside the repository:

```bash
python -m src.cp_sat_robustness_runner \
  --recovery-manifest data/scenarios/cp_sat_cold_start_recovery_v1.json \
  --recovery-output-dir /Users/klu/Projects/fair-course-allocation-artifacts/robustness-v1/cp-sat-cold-start-recovery-v1
```

The internal Greedy result is a hint only. The recovery runner does not run
stress or holdout scenarios after a failed stable-reference gate.

## Configuration validation

Run the Version 1 configuration and template validator:

```bash
python -m src.validation
```

The validator checks `data/config/` and `data/templates/` before synthetic
request generation or allocation algorithms are run.

To treat current TPHS baseline deviations as errors instead of warnings:

```bash
python -m src.validation --strict-policy
```

## Synthetic request generation

Generate a deterministic Version 1 synthetic student/request dataset:

```bash
python -m src.generation --scenario stable_year --seed 2026 --output-dir data/generated/stable_2026
```

The generator writes `students.csv`, `requests.csv`,
`generation_summary.csv`, and `generation_metadata.json`. Generated datasets
under `data/generated/` are ignored by Git except for `.gitkeep`.

## Synthetic section planning

Plan section counts and deterministic period layouts from generated students
and requests:

```bash
python -m src.section_planning \
  --input-dir data/generated/stable_2026 \
  --scenario stable_year \
  --seed 2026 \
  --output-dir data/generated/stable_2026_sections
```

The planner writes `sections.csv`, `course_demand_summary.csv`,
`period_layout_summary.csv`, and `section_planning_metadata.json`. It does not
assign students to sections.

## Reproducible benchmark manifest

The stable-year comparison checkpoint uses data seed `2026`, section-planning
seed `2026`, and solver seed `20260630`.

```python
from src.experiment_manifest import (
    STABLE_YEAR_BENCHMARK_SEEDS,
    build_experiment_manifest,
    load_experiment_manifest,
    verify_experiment_manifest,
    write_experiment_manifest,
)

manifest_path = "data/generated/stable_2026_manifest.json"
manifest = build_experiment_manifest(
    "data/generated/stable_2026",
    "data/generated/stable_2026_sections",
    "data/config",
    scenario_id="stable_year",
    seeds=STABLE_YEAR_BENCHMARK_SEEDS,
    repo_root=".",  # Omit to leave git_commit empty.
)
write_experiment_manifest(manifest, manifest_path)
canonical_input = verify_experiment_manifest(
    load_experiment_manifest(manifest_path),
    config_dir="data/config",
    repo_root=".",
)
# Only now run the benchmark/solver with manifest.seeds.solver_seed.
```

The manifest validates the data and section-planning seeds against generation
and planner metadata, including their output hashes and the planner's hashes of
its upstream files. It records the solver seed, input paths, canonical-input
counts, canonical/file/configuration hashes, and optionally the Git commit. Call
`verify_experiment_manifest` before every benchmark or solver run. If it raises
`ExperimentManifestError`, stop; do not run the benchmark or solver.

## Benchmark Runner v1

Run the manifest-guarded benchmark runner on already generated inputs:

```bash
python -m src.benchmark_runner \
  --generated-input-dir data/generated/stable_2026 \
  --sections-input-dir data/generated/stable_2026_sections \
  --data-seed 2026 \
  --section-seed 2026 \
  --solver-seed 20260630 \
  --output-json /tmp/fca_benchmark_summary.json
```

The default runner executes seeded random greedy and constrained-first greedy.
CP-SAT is opt-in because it is more expensive; include `--algorithms
random,constrained,cp_sat` to request it explicitly. Every benchmark summary
records the data-generation seed, section-planning seed, solver seed, and
canonical input fingerprint. A fingerprint mismatch invalidates comparison with
older benchmark results.

For an explicit, validated full-model search hint, add
`--cp-sat-initial-solution-artifact-dir <path>` while selecting `cp_sat`.
The artifact is never discovered automatically. Its hashes, fingerprint,
request mappings, source policy status, and local replay are checked before the
normal CP-SAT model runs; it is not a hard constraint or a final assignment.

## Scenario robustness benchmark v1

Run the frozen normal-year development suite with the four Greedy baselines.
Keep generated inputs and benchmark artifacts outside the repository:

```bash
python -m src.robustness_runner \
  --split development \
  --output-dir /tmp/fca_robustness_v1_development
```

Use `--max-scenarios 1` for a smoke run or `--dry-run` to inspect selected
scenario IDs without generating data. The suite records separate
`data_generation_seed`, `section_planning_seed`, and `algorithm_seed` values,
canonical input fingerprints, input difficulty descriptors, per-scenario
results, aggregate distributions, and paired algorithm deltas. The runner
rejects CP-SAT selection; this phase is Greedy-only and does not change any
generator, section-planning, capacity, or allocation semantics.

The manifest contains 12 development scenarios and 8 holdout scenarios.
Holdout evaluation requires `--split holdout --confirm-holdout-evaluation`.
Development results are for tuning and robustness measurement only; they are
not a generalization proof. Use `--resume` only when cached scenario
provenance, suite hash, configuration fingerprint, and scenario specification
all match.

## Scenario Robustness Benchmark v1 Phase B

Run the development-only stress suite against persistent Phase A artifacts:

```bash
python -m src.stress_robustness_runner \
  --split development \
  --output-dir /Users/klu/Projects/fair-course-allocation-artifacts/robustness-v1/stress-development-v1
```

Phase B contains 12 ordinary stress scenarios and 3 deliberately structural
negative controls in development, plus 8 holdout definitions that are not
run by default. It runs the four Greedy baselines only. Transforms are
deterministic and schema-aware, and write before/after canonical fingerprints,
transformation reports, certificates, paired normal/stress metrics, aggregate
summaries, and SHA-256 artifact manifests. A structural negative's expected
policy failure is diagnostic output, not a runner crash or publishable
allocation. In the negative summary, `policy_fail_count=4` means four Greedy
result rows failed policy, not four individual violations. This phase does not
run CP-SAT or holdout scenarios and does not change generator,
section-planning, capacity, period-layout, or policy semantics.

## Scenario Robustness Benchmark v1 Phase C: CP-SAT development evaluation

Phase C uses only the persisted Phase A and Phase B development artifacts. It
runs the production CP-SAT entry point with the frozen configuration in
`data/scenarios/cp_sat_development_evaluation_v1.json`: solver seed
`20260630`, one worker, 30-second bootstrap, 30 seconds per stage, 300
seconds total, and no external persisted seed.

```bash
python -m src.cp_sat_robustness_runner \
  --group all \
  --output-dir /Users/klu/Projects/fair-course-allocation-artifacts/robustness-v1/cp-sat-development-v1
```

Use `--dry-run` to inspect development IDs or `--verify-only` to validate
source hashes and canonical fingerprints without solving. Phase C does not
regenerate inputs, rerun Greedy, or run normal/stress holdouts. `UNKNOWN` is
not `INFEASIBLE`, and `FEASIBLE` is not `OPTIMAL`; development results
are not a final test or proof of generalization. See
`docs/CP_SAT_ROBUSTNESS_EVALUATION.md` for the output contract.

To audit an existing Phase C artifact without a solver rerun, write a separate
audited summary directory:

```bash
python -m src.cp_sat_robustness_runner \
  --audit-source-dir /path/to/cp-sat-development-v1 \
  --audit-output-dir /path/to/cp-sat-development-v1-audited
```

The audit distinguishes full-model infeasibility proof, fixed-objective-stage
infeasibility, bootstrap/core-stage results without a global proof, and
`UNKNOWN` without a final assignment. Holdout readiness requires usable normal
development assignments; a vacuous policy pass with zero assignments is not
enough.

## Cold-start feasibility recovery Phase B

Run the one-scenario distance-guided repair probe against the persisted stable
reference. It writes all artifacts outside the repository and refuses to
overwrite a non-empty output directory:

```bash
python -m src.cp_sat_repair_probe \
  --manifest data/scenarios/cp_sat_cold_start_repair_probe_v1.json \
  --output-dir /Users/klu/Projects/fair-course-allocation-artifacts/robustness-v1/cp-sat-cold-start-repair-probe-v1
```

The probe uses data seed `2026`, section seed `2026`, solver seed `20260630`,
one worker, and a 300-second budget. It runs only the full hard model's
`internal_repair_feasibility` stage. The constrained-first assignment is an
internal hint; the unweighted Hamming objective applies only to candidate
assignment variables, and the hint is not a hard constraint. `FEASIBLE` is not
`OPTIMAL`, and `UNKNOWN` is not `INFEASIBLE`. The final assignment, when
available, must come from the repair solver response and pass the final policy
and consistency checks. The probe is development diagnostics only; it does not
run stress or holdout scenarios or prove generalization.

## Cold-start feasibility recovery Phase C

Run the frozen 12-normal-scenario evaluation. The stable reference is
imported from the completed Phase B probe artifact (never re-solved); the
remaining 11 scenarios are each solved once under the identical frozen
configuration:

```bash
python -m src.cp_sat_normal_evaluation_runner \
  --evaluation-manifest data/scenarios/cp_sat_cold_start_normal_evaluation_v1.json \
  --output-dir /Users/klu/Projects/fair-course-allocation-artifacts/robustness-v1/cp-sat-cold-start-normal-development-v1
```

To rebuild a corrected reporting layer from an existing raw Phase C artifact
without re-solving anything:

```bash
python -m src.cp_sat_normal_evaluation_runner \
  --audit-source-dir /path/to/cp-sat-cold-start-normal-development-v1 \
  --audit-output-dir /path/to/cp-sat-cold-start-normal-development-v1-audited
```

Current frozen result: 3 FEASIBLE, 7 INFEASIBLE, 2 UNKNOWN, 3/12 publishable,
success gate FAIL, `ready_for_stress_development`/`ready_for_holdout` both
false. See `docs/CP_SAT_ROBUSTNESS_EVALUATION.md`.

## Section-plan feasibility alignment audit

Run the read-only diagnostic slice that explains why the 7 INFEASIBLE normal
scenarios above are globally infeasible, using the stable reference as a
control. It never modifies the production section planner, generator, hard
constraints, objective, or policy, and never re-solves the Phase C evaluation:

```bash
python -m src.section_plan_feasibility_audit \
  --audit-manifest data/scenarios/section_plan_feasibility_audit_v1.json \
  --output-dir /Users/klu/Projects/fair-course-allocation-artifacts/robustness-v1/section-plan-feasibility-audit-v1
```

It reports, per scenario: static feasibility descriptors (capacity, course
demand, exact student max-load matching, period concentration), an
OR-Tools assumption-literal unsatisfiable core (group-level and
fine-grained, with deletion-filtering to a locally minimal core), a fixed
three-stage lexicographic slack-relaxation witness, and eight
single/multi-family counterfactual checks. Diagnostic witnesses are never
publishable assignments or repaired section plans; a sufficient
unsatisfiable core is never reported as a proven minimum unless the solver
proves it. A counterfactual is not a smaller unsat core: it only tests whether
removing a family restores feasibility. Invalid or unproven witnesses remain
diagnostic-only and cannot support repair recommendations or root-cause
students. See `docs/SECTION_PLAN_FEASIBILITY_AUDIT.md`.

To rebuild corrected reporting from an existing raw audit artifact without
running a solver:

```bash
python -m src.section_plan_feasibility_audit \
  --rebuild-reporting-source-dir /Users/klu/Projects/fair-course-allocation-artifacts/robustness-v1/section-plan-feasibility-audit-v1 \
  --rebuild-reporting-output-dir /Users/klu/Projects/fair-course-allocation-artifacts/robustness-v1/section-plan-feasibility-audit-v1-audited
```

## Period-placement repair probe

The development-only period-placement probe generates a frozen candidate
universe and exact core-student preview on copies of the seven audited target
section plans. It performs no solver runs and refuses to overwrite a non-empty
artifact directory:

```bash
.venv/bin/python -m src.period_placement_repair_probe \
  --output-dir /Users/klu/Projects/fair-course-allocation-artifacts/robustness-v1/period-placement-repair-probe-v1
```

It uses only authoritative fine-core evidence, keeps capacities and policy
hard, and does not claim a global minimum or teacher/room feasibility. Formal
candidate validation is blocked until the generated cost preview is reviewed;
stress, negative, and holdout scenarios are not included.

## Joint period-edit feasibility pilot

The Phase A pilot jointly models frozen section-placement choices and student
assignment for only the feasible control and `normal_dev_10`. It uses the
frozen promising placement domain, keeps production hard policies and section
counts unchanged, and distinguishes a diagnostic joint witness from an
independently validated production repair:

```bash
.venv/bin/python -m src.joint_period_edit_pilot \
  --output-dir /Users/klu/Projects/fair-course-allocation-artifacts/robustness-v1/joint-period-edit-pilot-v1
```

The pilot excludes `G12_0105`, does not use an external persisted seed, and
does not run the other six targets, stress, negative, or holdout scenarios.
Minimum claims, if any, are bounded to the frozen placement domain and are
not global schedule-planning claims. See
`docs/JOINT_PERIOD_EDIT_FEASIBILITY.md`.
