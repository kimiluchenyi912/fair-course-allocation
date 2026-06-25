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
