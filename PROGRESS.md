# Progress

## Completed

- Set up the Python environment and Git repository.
- Created the initial project documentation and sample CSV files.
- Audited the original data model with Codex.
- Clarified that counselor-approved course requests are the solver input.
- Removed prerequisite eligibility checking from the core Version 1 scope.
- Defined the basic Grade 9 course-request structure.
- Decided to model Grades 9–12 together using transparent synthetic data.
- Completed the Grade 9–12 course-structure review for the first simulation.
- Completed whole-school size, target course-load, and capacity assumptions.
- Adopted the new request, section, assignment, and metrics data structure.
- Added initial configuration and template CSVs for the new simulation model.
- Implemented the Version 1 configuration validator and pytest coverage.
- Split structural/model validation errors from current TPHS baseline policy
  warnings, with an optional strict-policy mode.
- Implemented the Version 1 synthetic student and course-request generator,
  including deterministic load apportionment, grade rules, elective weights,
  fixed Grade 10 AP Calculus targets, and generation pytest coverage.
- Added stable-year fixed demand targets for AP Statistics and Calc D + Linear
  Algebra, brought generator-specific config into validation, and defined
  one-year priority protection / alternate replacement data interfaces.
- Implemented the Version 1 synthetic section-count and period-layout planner,
  including uniform 50% waitlist expansion, stable-year Calc D capacity
  override, Math 2/3 consecutive-period layout, Gov/Econ linked semester rows,
  planner diagnostics, CLI output, and pytest coverage.

## Current Direction

Build a scheduling system that assigns counselor-approved requests to fixed
course sections while respecting capacity, period conflicts, section
structures, and allowed free-period positions.

The system will prioritize:

1. Complete student schedules
2. Primary requests over alternates
3. Fair distribution of unavoidable unmet requests
4. Reasonable section balance after higher-priority goals

The old `data/sample/` files remain as legacy toy data and are not the
authoritative input model for new development.

## Next Step

Review section-planning diagnostics and request probability assumptions, then
build baseline allocation checks before implementing the CP-SAT fixed-section
solver.
