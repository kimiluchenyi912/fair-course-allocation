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

## Joint Period-Edit Feasibility Pilot v1

Added the Phase A single-scenario joint diagnostic model. It is limited to the
feasible control `normal_dev_reference_2026` and pilot target `normal_dev_10`,
uses only the frozen statically promising placement domain for authoritative
student `G12_0536`, and excludes `G12_0105`. Placement choices and assignment
intervals share one CP-SAT model while production candidate, capacity,
identity, fallback, fairness, minimum-five, and maximum-gap semantics remain
unchanged. The control and target zero-edit equivalence checks run before the
cost gate; any joint witness still requires independent validation by the
unchanged production solver. This is not a seven-scenario evaluation and does
not run stress, negative, or holdout scenarios.

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

- Added the Joint Model Control-Equivalence Performance Audit v1 scaffold. It
  compares production-native, fixed-native-joint, and fixed-optional-interval
  formulations on `normal_dev_reference_2026` only. Structural invariance and
  known-witness acceptance are fail-closed gates; the witness is correctness
  evidence only, not a performance seed. Expensive A/B/C runs are explicit
  opt-in, with no target, stress, negative, or holdout execution.

- Audited the Joint Model control benchmark hint provenance. The reporting
  audit is now read-only, solver runs own one fresh model copy each, duplicate
  or conflicting hint variables fail closed before search, and valid versus
  excluded solver attempts are represented separately. The Phase A gate uses
  structural and known-witness correctness checks rather than requiring a
  control feasibility-only incumbent.

- The provenance audit found and isolated a legacy per-variant checkpoint
  collision between Hamming and feasibility-only outputs. The affected B/C
  Hamming rows were excluded and rerun with the exact frozen settings; the raw
  artifact remains unchanged and the audited sibling records both replacements
  and the five pre-search duplicate-hint attempts.

- Added the single-target Joint Period-Edit Stage 1 Pilot v1 orchestration.
  It validates source hashes and a frozen 312-section/841-option placement
  domain, applies one constrained-first hint owner to a fresh model, and
  minimizes changed logical sections for `normal_dev_10` only. It records
  model cost gates, raw solver response/log data, joint witness validation,
  fixed-witness production acceptance, and independent production cold-start
  validation without changing production planner or CP-SAT semantics. Stage
  2--4, control, other normal targets, stress, negative, and holdout runs are
  disabled.

- Added the Joint Stage 1 Hybrid Occupancy Model-Size Reduction Audit v1.
  The original full optional-interval formulation remains the default; the
  audit-only hybrid mode uses sparse exact q/w linear occupancy channels,
  preserves the frozen 312-section/841-option domain, proves small-fixture
  equivalence, and accepts the stable control witness. The target model is
  built for cost accounting only; Stage 1 Solve and all other scenario runs
  remain disabled.

- Corrected the hybrid audit's proto accounting. The earlier Stage 1 cost-gate
  report is retained as historical provenance, but its serialized-byte label
  was text-format length. The audit now records binary export bytes separately
  from text bytes and both frozen baseline/hybrid binary measurements pass the
  250,000,000-byte gate. Stage 1 remains unrun.

- Added the single-target Hybrid Joint Period-Edit Stage 1 Execution v1. The
  one permitted `normal_dev_10` run used the frozen hybrid model, seed
  `20260630`, one worker, and 300 seconds. It returned `UNKNOWN` without an
  incumbent, so no witness, acceptance, production validation, or minimum
  claim was produced; no retry or other scenario run was made.

- Added the Hybrid Stage 1 Incumbent Bootstrap Audit v1. It uses a frozen
  complete 312-section/841-option domain, deterministic bounded K=1/K=2
  candidate portfolios, fresh constrained-first hint generation, and atomic
  per-run checkpoints. Hints and Hamming distance guide search but do not
  restrict the model. The slice remains limited to `normal_dev_10`; any
  incumbent requires unchanged production acceptance and independent cold-start
  validation, and `UNKNOWN` is unresolved rather than infeasible.

- Added the Hybrid K=2 Search Bottleneck Diagnostic Audit v1. It reuses the
  bootstrap's two frozen K=2 pair candidates without regenerating them and
  runs up to three placement-fixing ablations per pair (exact destinations
  with no hint, exact destinations with a coherent hint and Hamming
  objective, and fixed section IDs with destinations left free) to separate
  section-pair-selection, destination-selection, and assignment-feasibility
  explanations for the bootstrap's unresolved K=2 searches. It does not
  rerun the full cap-2 portfolio and does not run K=1 or K=3. See
  `docs/HYBRID_K2_SEARCH_BOTTLENECK_DIAGNOSTIC.md` for the protocol and
  `aggregate_summary.json`/`bottleneck_classification.json` in the artifact
  for the actual run's result.
- Corrected the diagnostic's artifact reporting after discovering its first
  completed batch (4 solver runs) had been superseded and rerun to fix a
  `provenance.json` finalization bug, without disclosing that history. Added
  `execution_history_correction.json` (total invocations = 8: 4x Diagnostic
  A, 0x Diagnostic B, 4x Diagnostic C, vs. 4 accepted-final-artifact runs) and
  corrected pair-count semantics (2 frozen pair candidates and 2 exact
  destination plans, but only 1 unique fixed section-ID pair, since both
  pairs move the same two sections). Applied via
  `--apply-execution-history-correction`, a reporting-only path that never
  builds or solves a CP-SAT model and never touches `runs/**` solver
  evidence; no new diagnostic solver runs were executed for this correction.

- Added the Hybrid K=2 Section-Pair Static Screening v1 runner and manifest.
  The accepted formal artifact enumerates all 48,516 unordered frozen section
  pairs for `normal_dev_10`, evaluates 139,415 placement combinations against
  the G12_0536 student-local necessary condition, and selects a six-pair
  portfolio. It matches the exploratory dry run exactly on frozen counts:
  47,278 necessary-condition failures, 1,237 survivors, one previously proven
  infeasible pair, and portfolio hash
  `ef83de1d2dfecaa6f55b8d074156466d96f73c5334be61d2aba856819445fd67`. The
  selected portfolio has now been exhausted with six fixed-pair Run A attempts,
  all returning `INFEASIBLE`, zero Run B attempts, zero incumbents, and zero
  production acceptance or validation runs. Those six scoped exclusions plus
  the prior evidence exclude seven specific unique section-ID pairs; 1,231
  static survivors remain untested. Survivors are not global or production
  feasibility evidence; global K2 remains unresolved and no exact minimum claim
  is made.
