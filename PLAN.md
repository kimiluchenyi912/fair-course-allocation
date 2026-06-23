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

Section planning and the 50% waitlist expansion analysis are planned as a
later stage after the fixed-section allocator and metrics are tested.

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

`target_course_count` is both the student's desired load and an upper bound.
An allocation remains valid when a student receives fewer courses, but the
student must be reported as having an incomplete schedule.

## 6. Optimization Priorities

The fair algorithm will optimize goals in this order:

1. Maximize the number of students receiving a complete target schedule.
2. Maximize fulfilled primary requests.
3. Minimize use of alternates.
4. Protect the worst-off students and distribute unavoidable unmet requests
   fairly.
5. Minimize total unmet requests.
6. Balance section utilization only after the higher priorities.

Students with equal priority should be treated using a reproducible lottery.

## 7. Baseline Algorithms

The fair algorithm will be compared with:

1. Random lottery
2. First-come, first-served
3. Grade-priority allocation

All algorithms must use the same input data and hard constraints.

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
3. Baseline allocation algorithms
4. Fixed-section CP-SAT solver
5. Fairness and schedule-completeness metrics
6. Section planning and 50% waitlist expansion analysis
7. Lightweight website only after the model and metrics are tested

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
