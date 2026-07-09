# Fair Course Allocation Project Plan

## 1. Project Goal

Build a fair and explainable system that allocates students to existing
high-school course sections.

The system should balance:

- complete student schedules,
- approved primary course requests,
- ranked alternate requests,
- course capacity,
- schedule conflicts,
- and fairness between students.

The solver starts after counselors have approved course requests. It does
not decide prerequisites, honors placement, AP eligibility, or graduation
eligibility.

## 2. Version 1 Scope

Version 1 assumes that the school has already decided:

- which courses will be offered,
- how many sections each course has,
- the period or periods occupied by each section,
- the capacity of each section,
- and which student requests have been approved.

The system will decide which students are assigned to which concrete
sections.

## 3. Out of Scope for Version 1

Version 1 will not:

- create the school's master schedule,
- assign teachers,
- assign classrooms,
- decide teacher workloads,
- determine student eligibility for requested courses,
- evaluate graduation eligibility,
- recommend courses to students,
- or automatically determine how many sections the school should offer.

Synthetic section-count and period-layout planning is available before the
fixed-section allocator. The planner does not assign students to sections; it
prepares fixed section inputs for baselines and the solver.

## 4. Inputs

The system will use:

- student information,
- counselor-approved primary requests,
- counselor-approved alternate requests,
- course catalog configuration,
- section information,
- linked semester-block definitions,
- capacity rules,
- grade-level load profiles,
- demand scenarios for synthetic data generation,
- and optional submission times or lottery keys for baseline comparison.

Only synthetic or anonymized data will be used.

The old files in `data/sample/` are legacy toy data from the earlier
prerequisite-centered model. They are useful for historical reference but
are no longer authoritative inputs for the new scheduling model.

## 5. Hard Constraints

Every valid fixed-section allocation must satisfy:

1. A section cannot exceed its capacity.
2. A student cannot take two sections in the same occupied period.
3. A student cannot take multiple sections for the same request group.
4. A student cannot receive more scheduled courses than their
   `target_course_count`.
5. Free periods must be allowed only by the configured unscheduled-period
   rules. V1 permits free periods only in Period 1, Period 6, and Period 7.
6. Double-period and semester-block courses must consume the periods defined
   by their section or linked block.
7. The system must clearly report incomplete schedules and infeasible
   allocation problems.
8. If a primary request is unmet, the solver must use ranked alternates where
   possible to keep the schedule complete.
9. Ordinary students may have at most one unmet logical primary course/block.
10. One-year protected students with a prior-year involuntary unmet primary
    must have zero unmet logical primaries, or the solver must report
    infeasibility diagnostics.
11. In the CP-SAT solver, every logical primary request for a high-demand
    course with approved logical primary demand greater than 120 must be
    satisfied. Demand of exactly 120 does not trigger this hard guarantee.

`target_course_count` is both the student's desired load and an upper bound.
An allocation remains valid when a student receives fewer courses, but the
student must be reported as having an incomplete schedule.

## 6. Optimization Priorities

The fixed-section CP-SAT solver treats protected students, ordinary max-one
unmet primary, and high-demand demand-greater-than-120 guarantees as hard
policies. It does not add a hard single-math guarantee and does not create
sections when math capacity is short.

The fair algorithm will optimize soft goals lexicographically in this order:

1. minimize math coverage violations;
2. minimize unmet primary requests, then unmet primary period units;
3. maximize rank-1 alternates, then rank-2 alternates, then rank-3 alternates;
4. maximize complete target schedules;
5. minimize total remaining period units;
6. apply a deterministic seeded tie-break only after substantive goals are
   fixed.

Students with equal priority should be treated using a reproducible lottery.
Solver status must distinguish OPTIMAL, FEASIBLE, INFEASIBLE, MODEL_INVALID,
and UNKNOWN. FEASIBLE is not reported as OPTIMAL, and UNKNOWN is not reported
as INFEASIBLE.

The CP-SAT implementation uses a Core/Enrichment decomposition. The Core model
contains primary requests, mandatory math fallback options, math coverage, and
hard policy constraints, and solves math coverage, primary unmet count, and
primary unmet period units as separate stages. The
Enrichment model then fixes those Core incumbent values and optimizes
alternates, complete schedules, remaining units, and the seeded tie-break. If a
stage returns FEASIBLE but not OPTIMAL, the solver may continue lower-priority
stages conditionally, but the result remains FEASIBLE and is not reported as a
global lexicographic optimum.

CP-SAT v1.2 adds a primary-only feasibility bootstrap before the Core model.
The bootstrap enforces the same fixed-section hard policies for primary
assignments, but omits fallback, alternate, math-coverage, complete-schedule,
remaining-unit, and tie-break variables. Its purpose is to find a first
hard-feasible primary incumbent quickly and to provide a Core hint. It is not
part of the lexicographic objective vector, and a bootstrap feasible result is
not reported as primary optimality.

The solver may receive `use_feasibility_bootstrap`, `bootstrap_time_seconds`,
and `max_total_time_seconds` keyword-only parameters. The global budget, when
provided, caps each stage by the remaining total time. Exhausted lower-priority
stages must be marked as skipped, not as OPTIMAL, and skipped stages must not
invent objective values.

CP-SAT warm starts may use constrained-first partial hints and stage-to-stage
incumbent hints. These hints are search guidance only; they do not relax or
replace hard constraints.

Logical primary counts are not simple request-row counts. A Grade 12
Government/Economics linked block counts once, and Math 2/3 Honors Accelerated
counts once even though it consumes two periods.

## 7. Baseline Algorithms

The fair algorithm will be compared with:

1. Seeded random greedy allocation
2. Constrained-first greedy allocation
3. First-come, first-served
4. Grade-priority allocation

All algorithms must use the same input data and hard constraints. Greedy
baselines may use policy fields for ordering, but they do not backtrack,
repair, displace students, or prove global infeasibility.

Benchmark Runner v1 defaults to the first two implemented baselines: seeded
random greedy and constrained-first greedy. CP-SAT comparison is supported only
when explicitly requested because stable-year CP-SAT runs are more expensive.
Every benchmark report must bind results to separate data-generation,
section-planning, and solver seeds plus the canonical allocation-input
fingerprint.

## 8. Evaluation Metrics

The project will measure:

- complete-schedule rate overall and by grade,
- primary-request fulfillment rate,
- alternate-use rate,
- number of students missing 1, 2, or more requested courses,
- maximum and average student loss,
- waitlist size by course,
- section utilization,
- fairness by grade and request-load group,
- assignment churn across nearby random-demand scenarios,
- variability across random seeds,
- and algorithm runtime.

## 9. Data And Simulation Configuration

The new authoritative structure is documented in:

- `docs/SIMULATION_SPEC.md`
- `docs/DATA_SCHEMA.md`
- `data/config/`
- `data/templates/`

Configuration files define grade profiles, capacity rules, course catalog
seed rows, linked semester blocks, and demand scenarios. Template files define
the shape of student, request, section, assignment, and metrics tables.

## 10. Implementation Order

The next stages are:

1. Configuration validator
2. Synthetic request generator
3. Synthetic section-count and period-layout planner
4. Baseline allocation algorithms
5. Fixed-section CP-SAT solver
6. Fairness and schedule-completeness metrics
7. Section-planning diagnostics and scenario comparison
8. Lightweight website only after the model and metrics are tested

## 11. Version 1 Completion Criteria

Version 1 is complete when:

- all hard constraints have automated tests,
- configuration validation catches malformed inputs,
- synthetic students and approved requests can be generated reproducibly,
- baselines produce valid allocations,
- the CP-SAT solver produces valid fixed-section allocations,
- all algorithms can be compared on the same dataset,
- fairness and satisfaction metrics are calculated,
- incomplete and infeasible cases are clearly reported,
- and results can be explained using small hand-verifiable examples.
