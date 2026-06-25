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
| expansion_threshold_ratio | waitlist ratio that can trigger another section; V1 baseline is 0.50 for every category |
| default_min_sections | advisory initial lower section hint |
| default_max_sections | deprecated/advisory only; not a hard cap for V1 section planning |
| expansion_allowed | whether the category participates in the uniform waitlist expansion policy; V1 baseline is true for every category |
| source_type | estimate or model assumption |

### `course_catalog.csv`

Seed catalog of confirmed structural and bottleneck courses. It is deliberately incomplete; later course rows may be added without changing the schema.

Important columns:

- `eligible_grades`: semicolon-separated grades used by the synthetic generator;
- `periods_required`: 0, 1, or 2;
- `occupies_school_period`: false for external/zero-period arrangements;
- `schedule_structure`: standard, double_period, or semester_block;
- `capacity_override`: blank uses category rule;
- `max_sections_override`: deprecated/advisory only; must not be treated as a hard cap in V1;
- `demand_tier`: core, mainstream, popular, niche, or fixed_limited;
- `protected_core`: core requests should be served by opening adequate sections;
- `known_capacity_risk`: identifies initial bottleneck calibration courses;
- `source_type` and `confidence` distinguish observation from assumption.

### `grade_request_rules.csv`

Grade-level synthetic request rules. These rules select core/request patterns
before elective fill and are not used by the solver.

| Column | Meaning |
|---|---|
| grade | 9, 10, 11, or 12 |
| rule_key | named rule such as english_weights, math_weights, or language_probability |
| rule_value | probability value or semicolon-separated `course_id:weight` map |
| source_type | student estimate, public source, or model assumption |
| notes | clarification |

### `course_choice_weights.csv`

Synthetic elective demand weights for unknown exact demand. Weights may be
defined by scenario, grade, demand tier, department, or specific course. They
should shape relative choice probability only; they are not precise enrollment
counts.

| Column | Meaning |
|---|---|
| scenario_id | scenario receiving the weight; currently stable_year baseline rows |
| grade | 9, 10, 11, or 12 |
| scope_type | demand_tier, department, or course |
| scope_id | matching tier, department name, or course_id |
| weight | nonnegative relative multiplier |
| source_type | student estimate or model assumption |
| notes | clarification |

### `fixed_course_targets.csv`

Small set of configured target counts for known special cases. These should be
used sparingly when a confirmed estimate is more important than generic weights.

| Column | Meaning |
|---|---|
| scenario_id | scenario receiving the fixed target |
| grade | grade receiving the target |
| course_id | targeted course |
| target_count | deterministic synthetic target |
| min_count | acceptable lower bound for future validation |
| max_count | acceptable upper bound for future validation |
| source_type | student estimate or model assumption |
| notes | clarification |

### `linked_course_blocks.csv`

Defines courses that share one period across semesters or have changing semester content.

Examples:

- four permitted Government/Economics combinations;
- AP Physics C Mechanics/E&M;
- Calculus D/Linear Algebra.

### `section_capacity_overrides.csv`

Scenario-specific capacity overrides used by the section planner.

| Column | Meaning |
|---|---|
| scenario_id | scenario receiving the override |
| course_id | course receiving the override |
| capacity | positive integer section capacity |
| source_type | student estimate or model assumption |
| notes | clarification |

The stable-year configuration uses this table to model `CALC_D_LINALG` with
capacity 45 while keeping its normal configured capacity at 40.

### `section_planning_rules.csv`

Small set of planner model assumptions.

| Column | Meaning |
|---|---|
| rule_id | unique rule name |
| rule_value | scalar value or semicolon-separated structured value |
| source_type | estimate or model assumption |
| notes | clarification |

Current V1 rules include positive-demand minimum sections and the allowed
consecutive pairs for double-period Math 2/3 sections.

### Waitlist expansion rule

Future section planning uses one expansion rule for all courses: if the
remaining waitlist for a course reaches `ceil(default_capacity ×
expansion_threshold_ratio)`, the planner may consider adding another section.
With the current 0.50 baseline this means:

- capacity 40: 19 does not trigger, 20 triggers;
- capacity 25: 12 does not trigger, 13 triggers;
- capacity 50: 24 does not trigger, 25 triggers.

The rule can be applied repeatedly after each added section. V1 does not use
course-specific hard maximum section counts for AP CSA, AP Statistics, AP
Physics C, Calc D + Linear Algebra, or niche electives.

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
| priority_protected | true if this student has one-year primary-request protection |
| priority_reason | blank or prior_year_unmet_primary |
| priority_valid_school_year | school year in which the protection applies |

No sensitive identifying data is required.

The first-year synthetic generator defaults all students to
`priority_protected=false` and leaves priority reason/year blank because it has
no prior-year assignment results.

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
| scenario_id | scenario used to plan the section |
| section_id | unique section |
| course_id | offered course |
| period_1 | first occupied period |
| period_2 | second period for double-period courses; otherwise blank |
| semester | full_year, semester_1, semester_2, or paired |
| capacity | seat limit |
| block_id | linked semester block if relevant |
| linked_section_group_id | shared group ID for paired semester rows or logical section |
| logical_block_id | logical course/block used for request counting |
| semester_content | content label for semester-block rows |
| planning_source | source of this section row, such as section_planner |
| teacher_resource_id | optional future staffing constraint |
| room_resource_id | optional future room constraint |

Section planning rows are not student assignments. A planned section only says
that a course is offered in a period with a capacity.

## Output tables

### `assignments.csv`

| Column | Meaning |
|---|---|
| student_id | student |
| section_id | assigned section |
| course_id | assigned course |
| request_type | primary or alternate |
| assignment_source | primary or alternate |
| replaced_primary_course_id | primary course replaced by this alternate, if any |
| replaced_primary_block_id | linked/logical block replaced by this alternate, if any |
| period_1 | assigned period |
| period_2 | second period if used |
| assignment_reason | solver, alternate substitution, or manual override |

### `unmet_requests.csv`

| Column | Meaning |
|---|---|
| student_id | student |
| course_id | unmet primary or alternate request |
| logical_block_id | logical course/block counted for primary unmet limits |
| request_type | primary or alternate |
| reason_code | capacity, period_conflict, section_not_opened, load_limit, or other |
| candidate_sections | number of possible sections before conflict filtering |
| replacement_course_id | alternate course assigned as replacement, if any |
| replacement_alternate_rank | alternate rank used as replacement, if any |
| replacement_period_units | period units supplied by this replacement row |
| earns_next_year_priority | whether this unmet primary earns next-year protection |

Two-period primaries may use multiple replacement rows. For example, one
unmet double-period course can be represented by two rows that share the same
logical block and use two different one-period alternate courses.

### `student_outcomes.csv`

One row per student after a solver run.

| Column | Meaning |
|---|---|
| student_id | student |
| primary_unmet_count | count of unmet primary logical courses/blocks |
| alternate_assigned_count | number of alternates assigned |
| schedule_complete | true if assigned period units meet the target load |
| earns_next_year_priority | true if this year creates a one-year priority tag |

A student who receives alternates for all missing period units can still have
`primary_unmet_count > 0` while `schedule_complete=true`.

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

### `generation_summary.csv`

Summary rows emitted by the synthetic generator.

| Column | Meaning |
|---|---|
| metric | metric name |
| group | grade, target-load bucket, request type, or tracked course |
| value | integer metric value |

### `generation_metadata.json`

Run metadata emitted by the synthetic generator, including scenario ID, random
seed, total students, request row counts, and summary row count.

### `course_demand_summary.csv`

Summary emitted by the section planner.

| Column | Meaning |
|---|---|
| scenario_id | scenario ID |
| course_id | course or logical block ID |
| logical_block_id | logical course/block ID |
| primary_demand | primary logical requests counted for section planning |
| section_capacity | capacity used for this course |
| expansion_threshold | integer waitlist threshold for another section |
| planned_sections | logical sections planned |
| planned_seats | planned_sections × section_capacity |
| remaining_waitlist | demand left below the expansion threshold |
| source_capacity_rule | rule or override source |
| capacity_override_used | true if course/scenario capacity override was used |

### `period_layout_summary.csv`

Summary emitted by the section planner.

| Column | Meaning |
|---|---|
| period | P1 through P7 |
| logical_section_count | logical section groups touching this period |
| section_row_count | CSV section rows touching this period |
| occupied_period_slot_count | logical occupied slots in this period |
| planned_seats | capacity counted once per logical section |
| yearlong_logical_sections | yearlong logical sections touching this period |
| semester_rows | semester rows touching this period |
| linked_logical_sections | linked semester logical sections touching this period |
| double_period_logical_sections | double-period logical sections touching this period |

### `section_planning_metadata.json`

Run metadata emitted by the section planner, including input file hashes,
planner/rule versions, total sections, total planned seats, total remaining
waitlist, period balance warnings, conflict diagnostics, and unmodeled
real-world constraints.

Conflict diagnostics include:

- `raw_period_overlap_score`: weighted overlap of co-requested course pairs
  across planned section periods;
- `unavoidable_course_pair_conflict_score`: weighted co-request pairs for which
  no non-overlapping section option exists.

These are layout diagnostics only, not student assignment results.

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

## Future fairness interfaces

The V1 generator prepares fields for, but does not implement, one-year fairness
protection. Future solvers should distinguish:

- `primary unmet`: a counselor-approved primary logical course/block was not assigned;
- `schedule incomplete`: even primary plus alternate assignments do not reach the target period units.

Ordinary students should have at most one unmet primary logical course/block in
a school year. Protected students should have zero unmet primary logical
courses/blocks. Logical course/block counting is not the same as request-row
counting:

- a regular yearlong course counts as one logical course;
- AP Physics C counts as one logical course;
- Math 2/3 Honors Accelerated counts as one logical course, even though it uses two periods;
- Grade 12 Government/Economics counts as one linked block, even though it has two semester request rows.

If a primary course is involuntarily unmet, the solver should use ranked
alternates where possible to keep the schedule complete. Receiving alternates
does not erase the primary unmet event.

## Multi-year priority flow

1. First-year synthetic generation sets no priority tags.
2. A solver run emits `unmet_requests.csv` and `student_outcomes.csv`.
3. The next-year data-prep layer reads involuntary primary unmet outcomes.
4. Eligible students receive `priority_protected=true`,
   `priority_reason=prior_year_unmet_primary`, and a one-year
   `priority_valid_school_year`.
5. After that protected year is scheduled, the tag expires and does not
   accumulate permanently.

If fixed sections and periods cannot satisfy every protected student's primary
requests, a future solver must report infeasibility or section-change
diagnostics rather than silently relaxing protection.
