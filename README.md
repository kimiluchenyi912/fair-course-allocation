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
