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
- Added Scenario Robustness Benchmark Suite v1 Phase B. The development-only
  stress layer applies deterministic schema-aware enrollment, popular-course,
  alternate, capacity, and logical-section transforms to persistent Phase A
  inputs, runs the four Greedy baselines, and writes paired diagnostics and
  SHA-256 guarded artifacts outside Git.
- Added fail-closed structural negative certificates for protected primary
  no-candidate, minimum logical load with at most four choices, and global
  logical capacity deficit scenarios.
- Added the frozen Phase C CP-SAT development evaluation manifest and runner.
  It consumes the persisted normal/stress development artifacts, verifies
  canonical fingerprints and source SHA-256 manifests, calls the formal CP-SAT
  solver with its frozen seed/budget configuration, records stage traces and
  null-safe metrics, supports fail-closed resume, and writes evaluation
  artifacts outside Git. Holdout scenarios remain unrun.
- Added the Phase C status-semantics audit. Existing raw traces can be
  classified without CP-SAT reruns, distinguishing full-model proof,
  fixed-objective-stage infeasibility, bootstrap/core-stage outcomes, UNKNOWN,
  and structural certificates. Holdout readiness now fails closed for a
  majority of normal scenarios without publishable assignments; the current
  cold-start result is 0/12 and is not evidence of global model infeasibility.

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

Review the Phase C CP-SAT development artifact and its status audit before
explicitly approving any holdout evaluation. Do not treat UNKNOWN as
INFEASIBLE, FEASIBLE as OPTIMAL, lexicographic-stage or bootstrap
INFEASIBLE as a full-model proof, ordinary stress results as a generalization
claim, or structural negative certificates as evidence about ordinary-year
feasibility.
