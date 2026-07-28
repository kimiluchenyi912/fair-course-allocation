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
12. In the CP-SAT solver, any returned final FEASIBLE or OPTIMAL schedule must
    pass Final Schedule Policy Gate v1: at least five assigned logical courses,
    at most one logical schedule gap, ordinary students with at most one unmet
    logical primary, and protected students with zero unmet logical primaries.

`target_course_count` is both the student's desired load and an upper bound.
Greedy baseline allocations can remain useful comparison artifacts when a
student receives fewer courses, but a CP-SAT final schedule is not publishable
unless it satisfies the final schedule policy gate.

## 6. Optimization Priorities

The fixed-section CP-SAT solver treats protected students, ordinary max-one
unmet primary, high-demand demand-greater-than-120 guarantees, minimum assigned
logical course count, and maximum logical schedule gap as hard policies. It
does not add a hard single-math guarantee and does not create sections when
math capacity is short.

The fair algorithm will optimize soft goals lexicographically in this order:

1. minimize math coverage violations;
2. minimize unmet primary requests, then unmet primary period units;
3. maximize assigned logical courses, equivalently minimizing logical schedule
   gaps when the Final Schedule Policy Gate v1 maximum-gap hard constraint is
   enabled;
4. maximize rank-1 alternates, then rank-2 alternates, then rank-3 alternates;
5. maximize complete target schedules under the legacy period-unit metric;
6. minimize total remaining period units;
7. apply a deterministic seeded tie-break only after substantive goals are
   fixed.

Students with equal priority should be treated using a reproducible lottery.
Solver status must distinguish OPTIMAL, FEASIBLE, INFEASIBLE, MODEL_INVALID,
and UNKNOWN. FEASIBLE is not reported as OPTIMAL, and UNKNOWN is not reported
as INFEASIBLE.

The CP-SAT implementation uses a Core/Enrichment decomposition. The Core model
contains primary requests, mandatory math fallback options, math coverage, and
hard policy constraints, and solves math coverage, primary unmet count, and
primary unmet period units as separate stages. The Enrichment model then fixes
those Core incumbent values and optimizes logical schedule completion,
alternates, legacy period-unit complete schedules, remaining units, and the
seeded tie-break. If a stage returns FEASIBLE but not OPTIMAL, the solver may
continue lower-priority stages conditionally, but the result remains FEASIBLE
and is not reported as a global lexicographic optimum.

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

When Final Schedule Policy Gate v1 is enabled, the solver also runs a
full-model feasibility-incumbent stage. Its optional deterministic
`constrained_first_greedy_full` seed supplies a complete hint for candidate and
derived Boolean variables, including explicit zero values for unselected
candidates. An explicitly supplied `--cp-sat-initial-solution-artifact-dir`
may provide a separately validated FEASIBLE/OPTIMAL artifact as the
`persisted_feasible_seed` candidate. The artifact is checked fail-closed by
SHA256, canonical fingerprint, request/student universes, candidate mappings,
and independent `AllocationState` replay. Neither seed is a hard constraint,
final schedule, feasibility proof, or optimality certificate; the current
solver response remains authoritative. A logical-schedule-completion stage can
be disabled for controlled diagnostics without changing hard policies or Core
stages.

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

Benchmark artifacts distinguish legacy period-unit fullness from logical-course
fullness. `fully_scheduled_students` is the historical period-unit metric
based on assigned period units matching the target. Logical schedule
publishability uses `logical_fully_scheduled_students`,
`students_with_logical_schedule_gap`, and `total_logical_schedule_gap`.
For CP-SAT, the logical completion objective and the exported student-level
logical fields are checked against the actual final solver response; the
objective bound is bounded by the total configured target logical-course count.

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

Scenario Robustness Benchmark v1 is now available as a separate Greedy-only
measurement layer. It uses the frozen manifest in
`data/scenarios/normal_year_robustness_v1.json`, keeps data-generation,
section-planning, and algorithm seeds distinct, and writes generated inputs
and results to an external output directory. The development split contains
the stable reference plus eleven independent normal-year seeds; the eight
holdout scenarios require explicit confirmation and must not be used for
ordinary tuning. This layer does not alter generator semantics, section
planning, capacities, period layout, baseline algorithms, or CP-SAT behavior.
Its summaries describe observed development variation and are not a
generalization proof. This Phase A suite has no stress scenarios, capacity
shocks, multi-scenario CP-SAT runs, or holdout results; those belong to later
explicitly scoped phases.

Scenario Robustness Benchmark v1 Phase B adds a separate development-only
stress layer: 12 ordinary stress scenarios plus 3 structural negative
controls. Deterministic transforms operate on persistent Phase A inputs and
carry fail-closed protected-no-candidate, minimum-load, or global-capacity
certificates. The four Greedy baselines run; CP-SAT and holdout evaluation
remain explicitly excluded. Results are diagnostics and paired development
comparisons, not a generalization claim, and transformed inputs remain
outside Git. In a negative summary, `policy_fail_count=4` means that all four
Greedy result rows failed policy; it is not a count of four individual
violations.

### Scenario Robustness Benchmark v1 Phase C

Phase C freezes the current production CP-SAT configuration for a development-
only evaluation. It consumes the persisted Phase A and Phase B artifacts and
does not regenerate inputs, rerun Greedy, reapply transforms, or alter the
solver model, hard policies, objective order, stage order, seeds, workers, or
time budgets. The evaluation manifest includes 12 normal development scenarios,
12 ordinary stress scenarios, and 3 structural negative controls. The 16
normal/stress holdout scenarios are absent from the manifest and remain unrun.

The runner uses solver seed `20260630`, one worker, 30-second bootstrap and
per-stage budgets, a 300-second total budget, and no external persisted seed.
It preserves `UNKNOWN` versus `INFEASIBLE` and `FEASIBLE` versus
`OPTIMAL`, requires every exported assignment to pass the final schedule
policy and consistency replay, and validates negative certificates before
solving. A development result is measurement/debugging evidence, not a final
test or a generalization claim. Stress scenarios are controlled perturbations,
not a complete model of real-world disruptions. The Phase C artifact is written
outside Git and supports fail-closed resume.

The Phase C status audit can rebuild summaries from existing raw stage traces
without invoking CP-SAT. Only the full-model feasibility stage can establish a
global hard-model infeasibility proof. Bootstrap/core-stage `INFEASIBLE` results
remain scoped and non-global; later fixed-objective stages are reported as
lexicographic-stage infeasibility. Structural certificates and solver proofs
are separate. Holdout readiness is blocked when a majority of normal
development scenarios have no publishable assignment, including the current
0/12 cold-start result.

### Cold-Start Feasibility Recovery v1 Phase A

The recovery experiment is a development-only extension of the same CP-SAT
runner. It uses `internal_feasibility_hint_strategy=constrained_first` to
build an internal constrained-first assignment with solver seed `20260630`,
then supplies that assignment as a complete candidate hint to the unchanged
full hard model. The greedy assignment is never exported as a solver result,
never becomes a constraint, and may contain policy violations that are
recorded as hint diagnostics.

The explicit `internal_repair_feasibility` stage is the only stage that enables
`repair_hint`. A validated FEASIBLE/OPTIMAL solver response may seed the
existing lexicographic stages; UNKNOWN without an incumbent falls back to the
legacy bootstrap/full-feasibility path. Model structure is checked before and
after hint application with `solution_hint` removed. The recovery manifest
contains only the 12 normal development scenarios, runs the stable reference
first, and stops without tuning if that reference has no publishable
assignment. It does not run stress or holdout scenarios and is not a
generalization claim.

### Cold-Start Feasibility Recovery v1 Phase B

Phase B is a single stable-reference repair probe over the unchanged full hard
model. It uses the constrained-first Greedy assignment only as an internal
hint and adds an explicit, unweighted Hamming-distance objective over the
candidate assignment variables. The hint is never fixed as a hard constraint,
and no external persisted seed, legacy bootstrap, later lexicographic stage,
stress scenario, or holdout scenario is used. The probe keeps data and section
seeds at `2026`, uses solver seed `20260630`, one worker, and a 300-second
budget.

The probe records model-invariance hashes, hint coverage, response provenance,
assignment changes, and final policy/consistency validation in an artifact
directory outside Git. `FEASIBLE` means a validated incumbent without an
optimality proof; `UNKNOWN` is not `INFEASIBLE`. This is a development
diagnostic and does not establish generalization or change generator, section
planning, capacity, period-layout, or policy semantics.

### Cold-Start Feasibility Recovery v1 Phase C — Frozen 12-Normal Development Evaluation

Phase C applies the same distance-guided repair method to all 12 normal
development scenarios. The stable reference (`normal_dev_reference_2026`) is
never re-solved: its result is imported from the completed Phase B probe
artifact once solver configuration and input provenance are verified to
match. The remaining 11 scenarios are each solved once under the identical
frozen configuration (seed `20260630`, one worker, 300-second budget,
constrained-first internal hint, unweighted Hamming objective, stop after
first solver solution). Result: 3 FEASIBLE, 7 INFEASIBLE, 2 UNKNOWN;
3/12 publishable; success gate FAIL; `ready_for_stress_development` and
`ready_for_holdout` both `false`. A follow-up Reporting Integrity Audit
rebuilt a corrected reporting layer from the same raw artifact (no re-solve):
it fixed a missing `publishable_assignment_available` field on the imported
stable row, replaced ambiguous "dominance"/"winner" language with a fixed
`policy_compliance_tradeoff` classification, and traced an anomalous
`normal_dev_11` stage-level wall-time reading (1114.1s, exceeding its own
289s configured budget) to OR-Tools' native `CpSolver.wall_time` property —
confirmed unrelated to this codebase's own perf_counter-based timing, which
stayed sane throughout; the anomaly is flagged and excluded from aggregate
solver-timing statistics while the raw value is preserved for audit.

### Section-Plan Feasibility Alignment Audit v1

A read-only diagnostic slice explaining *why* the 7 Phase C scenarios above
are globally INFEASIBLE, without changing the production section planner,
generator, CP-SAT hard constraints, objective, or Final Schedule Policy. It
builds an independent diagnostic CP-SAT model that reuses the production
canonical input, candidate index, mandatory-fallback injection, and policy
threshold constants; the seven diagnosable hard-policy families
(`SECTION_CAPACITY`, `STUDENT_PERIOD_CONFLICT`, `PROTECTED_PRIMARY`,
`ORDINARY_MAX_PRIMARY_UNMET`, `HIGH_DEMAND_PRIMARY`, `MINIMUM_FIVE_LOGICAL`,
`MAXIMUM_LOGICAL_GAP_ONE`) are each gated by an OR-Tools assumption literal so
`SufficientAssumptionsForInfeasibility` plus deletion-filtering can produce a
group-level and fine-grained (per-section/per-student) unsatisfiable core.
A separate controlled-relaxation model adds non-negative slack to the same
seven families and solves a fixed three-stage lexicographic objective
(minimize relaxed-instance count, then total slack magnitude, then Hamming
distance to the Constrained First hint) to produce a diagnostic witness —
never a publishable assignment. Eight fixed single/multi-family counterfactual
checks answer whether relaxing one rule family alone would restore
feasibility. A counterfactual is not an unsat core: cores are jointly
sufficient for infeasibility, while counterfactuals test sufficient feasibility
after removing a family. Invalid or unproven witnesses remain diagnostic-only
and cannot supply authoritative repair amounts or root-cause students. See
`docs/SECTION_PLAN_FEASIBILITY_AUDIT.md` for the full method, output contract,
and per-scenario findings.

### Joint Period-Edit Feasibility Pilot v1

The next Phase A diagnostic pilot jointly chooses placements from the frozen
promising-candidate domain and assigns students using production-equivalent
hard semantics. It runs only `normal_dev_reference_2026` and
`normal_dev_10`; the other six targets, stress, negative, and holdout inputs
remain unrun. The control fixed-placement model must be feasible and the
target zero-edit model must remain INFEASIBLE (or stop on UNKNOWN) before any
repair search begins.

The pilot does not modify the section planner, request data, capacities,
policy configuration, or production CP-SAT model. HA and Gov/Econ linked
sections remain atomic. A joint-model witness is diagnostic only and requires
an independent production-model validation before it can be called a
validated repair. Minimum wording, if ever supported, is limited to the
frozen placement domain and cannot describe a global or real-world minimum.

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

### Core-Targeted Minimum Period-Placement Repair Probe v1

The next development diagnostic freezes one feasible control and seven audited
normal targets, then generates period-only edits from the authoritative fine
core. The first step is an exact student-level candidate preview; it does not
run CP-SAT. Candidate validation, when explicitly authorized after the cost
gate, will reuse the unchanged production hard model on an in-memory section
plan copy. No capacity, request, policy, HA, or linked-course semantics may
change. Any minimum statement is limited to the frozen admissible candidate
universe; teacher and room realism remain outside the modeled scope.

### Joint Model Control-Equivalence Performance Audit v1

This reference-control-only audit compares the production-native model with a
fixed-placement joint model using native period conflicts and a fixed-placement
joint model using optional intervals and `NoOverlap`. Structural invariance and
a known policy-compliant witness must pass before cold-start performance runs.
The witness is fixed for correctness acceptance only, never as a performance
hint. Runs use only the internal Constrained First hint, seed `20260630`, one
worker, and frozen budgets. Targets, stress, negative, and holdout scenarios
are forbidden; `UNKNOWN` is not infeasibility.

The performance runner has one hint owner. Reporting does not mutate model
protos; every Hamming or feasibility-only run uses an independent clean proto,
checks that its solution hint is empty, applies the internal Constrained First
hint once, and validates unique variable indices before `Solve`. A duplicate or
conflicting hint is a pre-search model wiring error and fails closed. Attempt
provenance must retain excluded `MODEL_INVALID` attempts separately from valid
benchmark rows. The Phase A correctness gate is structural invariance,
known-witness acceptance, hint audit, and source-hash validation; a frozen
control feasibility-only `UNKNOWN` is not an equivalence failure or a reason to
block later target repair search.

Per-run response and validation artifacts must use the run kind in their
identity. If an older checkpoint cannot bind a response, log, and validation
without ambiguity, that result is marked provenance-unverified and excluded;
only the same frozen variant may be rerun to replace it. The raw artifact is
never rewritten during this audit.

### Joint Period-Edit Stage 1 Pilot v1

The next slice is restricted to `normal_dev_10` and the frozen promising
placement domain from the candidate preview. It runs one fresh optional-
interval joint model and minimizes changed logical sections only. Control,
other normal targets, stress, negative, holdout, and Stage 2--4 runs are
forbidden. A Stage 1 incumbent is a diagnostic joint witness until the
unchanged production model accepts the exact assignment on an edited-plan
copy and an independent production cold-start validation passes. Any minimum
claim is scoped to the frozen domain and requires an `OPTIMAL` Stage 1 plus
both production gates; `FEASIBLE` and `UNKNOWN` remain best-found or
unresolved results.

### Joint Stage 1 Hybrid Occupancy Model-Size Reduction Audit v1

This audit adds an explicit `hybrid_sparse_linear_occupancy` mode beside the
unchanged `full_optional_intervals` mode. It uses sparse exact q/w occupancy
channels for actual periods, including linked semester and double-period
logical sections, without pruning candidates or changing sections. Small
exhaustive fixtures, structural hashes, and a fresh reference-control witness
acceptance are required evidence. The `normal_dev_10` model is built for size
accounting only; Stage 1 Solve and all other scenario runs are disabled. The
frozen 250,000,000-byte cost gate is not raised.

The earlier `4a97c02` Stage 1 report is retained as historical provenance: its
serialized-byte label used text-format proto length and Stage 1 never ran.
The hybrid audit uses actual binary `.pb` export/serialization bytes instead;
the corrected baseline and hybrid values both pass the frozen gate.

### Hybrid Joint Period-Edit Stage 1 Execution v1

This execution slice permits exactly one Stage 1 solve for `normal_dev_10`
using the audited hybrid occupancy formulation, seed `20260630`, one worker,
and a 300-second budget. Stage 2--4, control, other normal, stress, negative,
and holdout runs are disabled. `UNKNOWN` without an incumbent is unresolved
and does not trigger witness acceptance or production validation. A valid
witness would require both fixed-witness production acceptance and an
independent production cold-start validation before any repair or minimum
claim.

### Hybrid Stage 1 Incumbent Bootstrap Audit v1

The incumbent bootstrap is a bounded, single-target follow-up for
`normal_dev_10`. It uses only the frozen 312-section/841-option placement
domain and at most three K=1 followed by at most two K=2 candidate-guided
searches. A fresh edited-plan constrained-first assignment is a hint; the
full joint model retains the complete domain and adds the corresponding
change-count cap. The hints and Hamming objective do not restrict the feasible
region; the explicit cap is the sole deliberate feasibility restriction in
each bounded search model. It does not prune placement options or candidate
edges and does not change production hard-policy semantics.

`UNKNOWN` remains unresolved, while `INFEASIBLE` is scoped to the frozen full
domain and cap. Any incumbent requires joint replay, fixed-witness production
acceptance, and independent production cold-start validation. The runner uses
atomic checkpoints so `--resume` skips completed candidates. Control, other
normal, stress, negative, holdout, and Stage 2--4 runs remain disabled; no
section planning, capacity, period layout, or production policy is changed.

### Hybrid K=2 Search Bottleneck Diagnostic Audit v1

This diagnostic reuses, without regenerating, the bootstrap's two frozen K=2
pair candidates for `normal_dev_10` and separates why both bootstrap K=2
searches returned `UNKNOWN` into four candidate explanations: global
section-pair selection, destination-placement selection, fixed-plan
assignment feasibility, and hint/Hamming search guidance. For each pair it
runs, in a fixed order and only as needed, Diagnostic A (exact hinted
destinations, full production hard model, no hint, no objective, 60s),
Diagnostic B (same exact plan, edited-plan Constrained First hint plus
Hamming objective, 60s, only if A is `UNKNOWN`), and Diagnostic C (the full
hybrid joint model with the pair's two section IDs forced changed and the
other 310 editable sections forced original, destinations otherwise free,
120s, only while no incumbent exists yet). At most six new solver runs total
across both pairs; any incumbent stops all remaining runs. It does not rerun
the full 312-section cap-2 portfolio, does not mine new pair candidates, and
does not run K=1 or K=3.

`INFEASIBLE` from Diagnostic A or B is scoped to the one exact plan tested,
never generalized to the section pair. `INFEASIBLE` from Diagnostic C is
scoped to the one fixed section-ID pair tested across its full destination
domain, never to other K=2 pairs. `UNKNOWN` is never described as
`INFEASIBLE`. Any minimum-changed-sections claim still requires the
pre-existing K=1 infeasibility proof plus full joint/production validation;
if no incumbent is found the previously proven lower bound of 2 stands
unchanged and K=2 itself is not declared infeasible.

### Hybrid K=2 Section-Pair Static Screening v1

This static-only pass enumerates all 48,516 unordered pairs in the frozen
312-section `normal_dev_10` domain and checks whether each pair can satisfy
the authoritative G12_0536 student-local necessary condition across its full
non-original destination-product domain. It is deliberately not a solver run:
fixed-pair Run A/B, production fixed-witness acceptance, independent
production validation, global K=2, K=1, and K=3 are all disabled.

The accepted formal static artifact matches the exploratory dry run: 139,415
placement combinations, 47,278 necessary-condition failures, 1,237 survivors,
one previously proven infeasible pair, and a six-pair portfolio with hash
`ef83de1d2dfecaa6f55b8d074156466d96f73c5334be61d2aba856819445fd67`.
All six selected portfolio pairs have now completed fixed-pair Run A and
returned `INFEASIBLE`, with zero Run B attempts, zero incumbents, zero
production acceptance, and zero production validation. Those six results plus
the one previously proven pair give seven specifically excluded unique
section-ID pairs, leaving 1,231 untested static survivors. Survivors are only
G12_0536 necessary-condition survivors; they do not prove global or production
feasibility. The same-course-pair cap was relaxed to 2, but the final
portfolio still contains six unique course pairs. The section participation
cap was relaxed to 3 and that relaxation was used. The lower bound remains 2,
global K=2 remains unresolved, and no exact minimum claim is made.

### Remaining K=2 Survivor Expansion Audit v1

The next-step audit is investigation and protocol design only. Read-only
inspection of the six fixed-pair models found a proof-backed blind spot in the
G12_0536-only screen: ordinary student G12_0105 has three single-section
primaries fixed at P1, so any feasible pair must intersect the blocker set
`{CHINESE4_01, FOOTBALL_01, INTERMEDIATE_ACTING_01}` in at least one section.
Applying that necessary condition to cached survivor rows projects, but does
not formally record, a reduction from 1,231 untested survivors to 12.

The audit defines evidence classifications, safe-exclusion proof requirements,
ranking-only heuristics, deterministic ordering and resume contracts, storage
options, and objective stopping rules. It runs no solver, reruns no static
screening, modifies no formal artifact, and creates no new feasibility result.
Until a separately authorized implementation and execution are completed,
global K2 remains unresolved, the lower bound remains 2, and no exact minimum
or repair witness is claimed. See
`docs/REMAINING_K2_SURVIVOR_EXPANSION_AUDIT.md`.

### All-Student K=2 Blocker Safe Screen v1

The independently verified follow-up applied the G12_0105 blocker-set
intersection rule to all 1,237 accepted static survivors without rerunning the
original 48,516-pair screen or invoking a solver. It formally safe-excluded
1,225 pairs and retained 12 pairs that are only not excluded by current safe
necessary conditions. The previously tested six pairs are included once in
the 1,225 safe exclusions; among the 1,231 previously untested survivors,
1,219 are excluded and 12 remain.

The full universe closes as 47,278 original static exclusions, one previously
proven unique pair, 1,225 new blocker-safe exclusions, and 12 remaining pairs.
Global K2 remains unresolved, the lower bound remains 2, and no exact minimum
or feasibility claim is made for the remaining pairs. See
`docs/ALL_STUDENT_K2_BLOCKER_SAFE_SCREEN.md`.
