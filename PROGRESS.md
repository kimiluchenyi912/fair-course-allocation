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
- Added the distance-guided CP-SAT cold-start repair probe (Phase B): a single
  stable-reference development probe using a constrained-first internal hint
  and an unweighted Hamming-distance objective over the unchanged full hard
  model, with no external persisted seed, legacy bootstrap, or later
  lexicographic stage.
- Added the frozen 12-normal Cold-Start Feasibility Recovery evaluation
  (Phase C): imports the stable reference from the completed probe artifact
  without re-solving it, and solves the remaining 11 normal development
  scenarios once each under the identical frozen configuration. Result:
  3 FEASIBLE, 7 INFEASIBLE, 2 UNKNOWN, 3/12 publishable, success gate FAIL,
  `ready_for_stress_development`/`ready_for_holdout` both false.
- Ran a Reporting Integrity Audit over that Phase C artifact (no re-solve):
  fixed a missing `publishable_assignment_available` field on the imported
  stable row, replaced ambiguous comparison language with a fixed
  `policy_compliance_tradeoff` classification, and traced an anomalous
  `normal_dev_11` stage-level wall-time reading to OR-Tools' native
  `CpSolver.wall_time` property (unrelated to this codebase's own, sane,
  perf_counter-based timing) -- flagged and excluded from aggregate solver
  timing while the raw value is preserved for audit.
- Added the Section-Plan Feasibility Alignment Audit v1: an independent,
  read-only diagnostic CP-SAT model (reusing the production canonical input,
  candidate index, mandatory-fallback injection, and policy thresholds) that
  explains the 7 Phase C INFEASIBLE scenarios via OR-Tools assumption-literal
  unsatisfiable cores (group-level and fine-grained, with deletion-filtering
  to a locally minimal core), a fixed three-stage lexicographic
  slack-relaxation witness, and eight single/multi-family counterfactual
  checks. Diagnostic witnesses are never publishable and no production code,
  section plan, or policy was changed. See
  `docs/SECTION_PLAN_FEASIBILITY_AUDIT.md`.

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

Review the Section-Plan Feasibility Alignment Audit v1 findings (unsatisfiable
cores, relaxation witnesses, counterfactual checks, and evidence-backed
classification) for the 7 INFEASIBLE normal development scenarios before
deciding whether the next slice is a section-capacity rebalance, a
period-aware section-planning change, or a policy reconsideration. Do not
treat UNKNOWN as INFEASIBLE, FEASIBLE as OPTIMAL, lexicographic-stage or
bootstrap INFEASIBLE as a full-model proof, a diagnostic relaxation witness as
publishable, a sufficient unsatisfiable core as proven minimum, ordinary
stress results as a generalization claim, or structural negative certificates
as evidence about ordinary-year feasibility.

The Section-Plan audit distinguishes an unsat core (constraints jointly
sufficient for infeasibility) from a relaxation counterfactual (removing a
family is sufficient to restore feasibility). Invalid or unproven relaxation
witnesses retain raw diagnostic values only; they do not supply authoritative
repair amounts or root-cause student IDs. The current audited conclusion is
strong evidence of period-supply misalignment under the frozen section plans,
not proof of a minimum section move.

- Began the Core-Targeted Minimum Period-Placement Repair Probe v1. Added a
  frozen control/target manifest and an independent candidate-universe runner.
  The preview uses only audited fine-core students, excludes invalid witness
  students, preserves section identity/capacity/HA/linked semantics, and uses
  an exact no-capacity student-level dynamic program. Formal CP-SAT candidate
  validation is not run until its explicit cost gate is reviewed.
