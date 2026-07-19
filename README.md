# Fair Course Allocation

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
