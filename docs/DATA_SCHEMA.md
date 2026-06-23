# Scheduling Data Schema v1

## Design principles

- Course eligibility has already been approved by counselors.
- Requests, sections, and assignments are separate tables.
- Semester blocks and double-period courses are represented explicitly.
- Configurations carry a confidence/source label.
- All random generation uses a saved seed.

## Configuration tables

### `grade_profiles.csv`

Controls school size and target schedule loads.

| Column | Meaning |
|---|---|
| grade | 9, 10, 11, or 12 |
| student_count | synthetic students in the grade |
| share_5_classes | probability of requesting five scheduled classes |
| share_6_classes | probability of requesting six scheduled classes |
| share_7_classes | probability of requesting seven scheduled classes |
| allowed_free_periods | semicolon-separated allowed free periods, currently P1;P6;P7 |
| source_type | student estimate, public source, or model assumption |
| notes | clarification |

Shares must sum to 1 within each grade.

### `capacity_rules.csv`

Default section behavior by course category.

| Column | Meaning |
|---|---|
| rule_id | unique rule |
| course_category | category receiving the rule |
| default_capacity | default seats per section |
| capacity_min | allowed lower bound |
| capacity_max | allowed upper/overload bound |
| expansion_threshold_ratio | waitlist ratio that can trigger another section |
| default_min_sections | default lower section bound |
| default_max_sections | default upper section bound; blank means course-specific |
| expansion_allowed | whether new sections may be added |
| source_type | estimate or model assumption |

### `course_catalog.csv`

Seed catalog of confirmed structural and bottleneck courses. It is deliberately incomplete; later course rows may be added without changing the schema.

Important columns:

- `eligible_grades`: semicolon-separated grades used by the synthetic generator;
- `periods_required`: 0, 1, or 2;
- `occupies_school_period`: false for external/zero-period arrangements;
- `schedule_structure`: standard, double_period, or semester_block;
- `capacity_override`: blank uses category rule;
- `max_sections_override`: prevents unrealistic section growth;
- `demand_tier`: core, mainstream, popular, niche, or fixed_limited;
- `protected_core`: core requests should be served by opening adequate sections;
- `known_capacity_risk`: identifies initial bottleneck calibration courses;
- `source_type` and `confidence` distinguish observation from assumption.

### `linked_course_blocks.csv`

Defines courses that share one period across semesters or have changing semester content.

Examples:

- four permitted Government/Economics combinations;
- AP Physics C Mechanics/E&M;
- Calculus D/Linear Algebra.

## Input tables

### `students.csv`

One row per student.

| Column | Meaning |
|---|---|
| student_id | anonymous ID |
| grade | current grade |
| target_course_count | 5, 6, or 7 |
| unscheduled_preference | morning, afternoon, either, or none |
| random_seed_group | reproducibility group |

No sensitive identifying data is required.

### `requests.csv`

One row per requested course.

| Column | Meaning |
|---|---|
| student_id | requesting student |
| course_id | requested course |
| request_type | primary or alternate |
| request_rank | rank among alternates; primary may be blank |
| request_group | optional group for mutually exclusive choices |
| must_share_block_id | optional semester-block requirement |

Primary courses are not assumed to be ranked unless the real process later supplies rankings.

### `sections.csv`

Concrete offerings used by the fixed-section solver.

| Column | Meaning |
|---|---|
| section_id | unique section |
| course_id | offered course |
| period_1 | first occupied period |
| period_2 | second period for double-period courses; otherwise blank |
| semester | full_year, semester_1, semester_2, or paired |
| capacity | seat limit |
| block_id | linked semester block if relevant |
| teacher_resource_id | optional future staffing constraint |
| room_resource_id | optional future room constraint |

## Output tables

### `assignments.csv`

| Column | Meaning |
|---|---|
| student_id | student |
| section_id | assigned section |
| course_id | assigned course |
| request_type | primary or alternate |
| period_1 | assigned period |
| period_2 | second period if used |
| assignment_reason | solver, alternate substitution, or manual override |

### `unmet_requests.csv`

| Column | Meaning |
|---|---|
| student_id | student |
| course_id | unmet request |
| request_type | primary or alternate |
| reason_code | capacity, period_conflict, section_not_opened, load_limit, or other |
| candidate_sections | number of possible sections before conflict filtering |

### `metrics.csv`

One row per simulation/algorithm/seed combination.

Core fields:

- scenario_id;
- algorithm;
- random_seed;
- complete_schedule_rate;
- primary_fulfillment_rate;
- alternate_use_rate;
- mean_unmet_primary;
- max_unmet_primary;
- assignment_churn_rate;
- solve_time_seconds.

## Legacy sample data

The files in `data/sample/` belong to the earlier prerequisite-centered toy
model. They are retained for reference but are not authoritative inputs for
the new scheduling simulation.

New development should use `data/config/` for configuration and
`data/templates/` for table shapes.

## Validation rules

1. Grade load shares sum to 1.
2. Every course and section ID is unique.
3. Every section references a known course.
4. Every request references a known student and course.
5. `periods_required = 2` requires `schedule_structure = double_period`.
6. `occupies_school_period = false` requires `periods_required = 0`.
7. Section capacity is positive and does not exceed configured overload limits unless explicitly overridden.
8. Alternate ranks are positive and unique within each student.
9. A student's free periods may only fall in P1, P6, or P7.
10. Government/Economics blocks occupy one shared period and use one approved regular/AP combination.
