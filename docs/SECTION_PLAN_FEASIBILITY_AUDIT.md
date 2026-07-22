# Section-Plan Feasibility Alignment Audit v1

## Purpose

The frozen 12-normal Cold-Start Feasibility Recovery evaluation (Phase C)
proved 7 of 12 normal development scenarios globally INFEASIBLE under the
unchanged production full hard model. This audit is a **development-only,
read-only diagnostic slice** that explains *why*, without changing the
production section planner, generator, CP-SAT hard constraints, objective, or
Final Schedule Policy. It answers:

1. which constraint families jointly cause the infeasibility;
2. whether it is global capacity, course-level capacity, period-supply
   structure, student minimum-load/max-gap policy, or primary protection
   policy;
3. how much relaxation each scenario needs to become feasible (best-found,
   not proven minimum, unless the solver proves optimality);
4. which logical courses, sections, periods, and students are the
   bottleneck;
5. an evidence-backed classification to inform whether the next slice should
   change section planning or reconsider a policy threshold.

This audit does not modify the production section plan, generator, hard
constraints, objective, or policy. It does not raise the 3/12 recovery rate.
No diagnostic relaxation witness is ever a publishable assignment.

## Scope

- Control: `normal_dev_reference_2026` (must remain FEASIBLE/OPTIMAL in the
  diagnostic model; INFEASIBLE here is a correctness failure that stops the
  audit immediately).
- Targets: the 7 scenarios Phase C proved globally INFEASIBLE --
  `normal_dev_01, 03, 04, 05, 07, 09, 10`.
- No stress, negative, or holdout scenario is included.
- Frozen solver seed `20260630`, one worker.

## Architecture

The audit never re-implements candidate generation, section planning, or the
production assignment-variable semantics. It reuses, unmodified:

- `CanonicalAllocationInput` (students, logical requests, logical sections,
  candidate index) -- the same data structure the production solver consumes.
  `LogicalSection.capacity`/`.occupied_periods` are already deduplicated at
  canonicalization time, so Math 2/3 HA double-period rows and linked Gov/Econ
  rows are never double-counted.
- `_build_mandatory_fallback_plans` / `_convert_fallback_plans` for the
  mandatory math-fallback candidate injection.
- `_add_duplicate_identity_constraints`, `_add_student_target_constraints`,
  `_add_fallback_constraints`, `_add_math_coverage_constraints` -- called
  verbatim for the families that are never relaxed in this audit.
- `_validate_candidates_for_request`, `_safe_name`, `_VariableKey` -- the
  exact production candidate-validity checks and variable-key type.
- `_constrained_first_full_hint_seed` and `_hamming_distance_expression` --
  the exact production Constrained First hint and Hamming-distance objective
  used by the repair probe/evaluation, reused for the relaxation witness's
  Stage 3 tie-break.
- The production policy threshold constants: `HIGH_DEMAND_PRIMARY_THRESHOLD`,
  `MINIMUM_ASSIGNED_LOGICAL_COURSE_COUNT`, `MAXIMUM_SCHEDULE_GAP_COUNT`,
  `MAXIMUM_ORDINARY_PRIMARY_UNMET_COUNT`, `MAXIMUM_PROTECTED_PRIMARY_UNMET_COUNT`.

The production constraint-builder functions for `SECTION_CAPACITY`,
`STUDENT_PERIOD_CONFLICT`, and the three primary-policy/two load-policy
families are monolithic (each adds several families' constraints in one
unconditional pass with no reification hook), so they cannot be selectively
gated without modifying production code, which this audit does not do.
Instead, the audit rebuilds those seven families' constraints itself, using
the identical linear-inequality form and the same imported constants and
`allocation_input`/`assigned_vars` data, each gated by an OR-Tools assumption
boolean (`model.Add(...).OnlyEnforceIf(family_or_fine_literal)`). No
candidate-generation or section-planning judgment is reimplemented -- only a
one- or two-line threshold inequality per family, per production formula.

### Constraint family taxonomy

**Never relaxed** (semantic/structural, not a policy choice):
`ASSIGNMENT_CANDIDATE_VALIDITY`, `DUPLICATE_LOGICAL_IDENTITY`,
`STUDENT_TARGET_LOAD_CAP`, `MANDATORY_FALLBACK_SEMANTICS`,
`MATH_COVERAGE_SOFT_POLICY` (already soft in production -- has its own
violation indicator and can never by itself force hard infeasibility, so it
is excluded from the diagnosable set).

**Diagnosable** (hard-policy families this audit measures):
`SECTION_CAPACITY`, `STUDENT_PERIOD_CONFLICT`, `PROTECTED_PRIMARY`,
`ORDINARY_MAX_PRIMARY_UNMET`, `HIGH_DEMAND_PRIMARY`, `MINIMUM_FIVE_LOGICAL`,
`MAXIMUM_LOGICAL_GAP_ONE`.

### Model-equivalence check

Before any core/relaxation work, the diagnostic model is solved with every
diagnosable family's assumption fixed true. This **must** reproduce the
production ground truth exactly: FEASIBLE/OPTIMAL for the control,
INFEASIBLE for every target. Any mismatch raises
`SectionPlanAuditCorrectnessFailure` and stops the audit immediately -- it is
treated as a production-model-disagreement bug, not a diagnostic result.

## Method

### 1. Static feasibility descriptors (no solver)

- **Global supply**: total logical seat capacity (deduplicated), minimum
  required assignments (5 x students), target-load capacity margin, capacity
  and section count by period.
- **Course-level demand**: primary/total demand vs. logical capacity per
  course, over-capacity course lists, top-20 pressure courses. Explicitly
  documented as *not* a global-infeasibility proof (ignores alternates and
  period combinations).
- **Student max-load matching**: an *exact* CP-SAT computation (not a greedy
  approximation) that drops section capacity but keeps candidate
  availability, period-occupancy semantics (including multi-period
  HA/linked sections), and duplicate-identity, then maximizes total assigned
  count in one combined solve. Because students never compete for capacity
  in this relaxed model, the joint maximum is also each student's individual
  maximum. Reports students whose maximum-possible load is below five or
  below target-minus-one, and zero/one-candidate primary requests.
- **Period concentration**: seat supply, primary candidate demand, and
  capacity-to-demand pressure by period; students whose primary candidates
  concentrate in one period.

### 2. Group-level unsatisfiable core

All seven family assumption literals are passed to
`model.AddAssumptions(...)`; if INFEASIBLE, `SufficientAssumptionsForInfeasibility`
returns a sufficient subset, then deletion-filtering (drop one family at a
time, re-solve with the rest still assumed, keep the drop only if still
INFEASIBLE) reduces it to a locally minimal core. A sufficient core is never
labeled minimum; the locally minimal core is never labeled the global minimum
-- both are explicit fields, with `minimality_status` recording whether
filtering completed, is unresolved due to the time budget, or is not
applicable (feasible / no infeasibility proof).

### 3. Fine-grained core

For the families in the group-level core, a **second** diagnostic model is
built with per-instance assumption literals for just those families (one per
section for `SECTION_CAPACITY`, one per (student, period) pair for
`STUDENT_PERIOD_CONFLICT`, one per student or request for the policy
families) while every other family keeps its single group literal. The same
sufficient-core-then-deletion-filtering procedure runs at this finer
granularity, producing IDs traceable to specific students, requests, logical
courses, sections, and periods.

### 4. Controlled relaxation witness

A separate model replaces each of the seven families' hard constraint with a
non-negative integer slack (or boolean indicator for the two mandatory-style
families): `capacity_overflow[section] >= assigned - capacity`,
`extra_primary_unmet[student] >= primary_unmet - 1`, boolean
`protected_unmet`/`high_demand_unmet` indicators, `load_shortfall[student] >=
5 - assigned`, `excess_gap[student] >= target - assigned - 1`. Student period
conflicts stay hard in this first layer. A fixed, unweighted three-stage
lexicographic objective is solved once per stage, with the prior stage's
optimal value fixed as a hard constraint before the next stage runs:

1. minimize the **count** of constraint instances with any nonzero
   relaxation;
2. fix that count, minimize the **total slack magnitude**;
3. fix that magnitude, minimize the **Hamming distance** to the Constrained
   First hint (a stability tie-break only -- reused verbatim from
   production).

If Stage 1 is FEASIBLE but not OPTIMAL, every later reference to "minimum
relaxation" is instead reported as "best found relaxation." If layer 1 is
still INFEASIBLE (not expected for this model, since assigning nothing
trivially satisfies every family once capacity/policy are slacked and only
period-conflict remains hard), a second, separately reported layer adds
`period_overlap_slack[student, period]` and repeats Stage 1 only.

### 5. Witness validation

Before any relaxation witness is used for interpretation, the audit verifies
capacity-overflow accounting closes exactly against the raw assignment
per section, student-policy slack accounting closes exactly per student, no
duplicate logical identity exists, and a response hash is present. A witness
is never labeled publishable, policy-compliant, or a production assignment --
it is a **diagnostic relaxation witness** only. If validation is false, Stage 2
has no incumbent, or Stage 1 optimality is unproven, its slack magnitudes and
student IDs are retained as raw diagnostics only and are excluded from
authoritative repair recommendations and root-cause student evidence.

### 6. Counterfactual variants

Eight fixed variants re-solve the group-core-gated model with specific
families' literals fixed false (relaxed) and the rest fixed true: capacity
only; period-conflict only; all three primary-policy families; minimum-five
only; maximum-gap only; minimum-five + maximum-gap; all five student-policy
families; capacity + period-conflict. Each records status, runtime, and
whether an assignment was found. `UNKNOWN` is reported as `UNKNOWN`, never
silently treated as "this relaxation doesn't help." A counterfactual is not
an unsatisfiable core: the core is a set of constraints jointly sufficient for
infeasibility, while the counterfactual is sufficient feasibility after
removing a family. These findings answer different questions and do not prove
a minimum core or a minimum section repair.

### 7. Repair-candidate interpretation and classification

For each target scenario, `section_plan_repair_candidates.json` reports raw
diagnostic sections needing added seats in the best-found witness (and whether they are
Math 2/3 HA or linked Gov/Econ structures), course-level period-supply issues
from the static descriptors, and the students requiring each kind of policy
slack. A scenario is classified into one or more of
`global_capacity_deficit`, `course_capacity_bottleneck`,
`period_supply_misalignment`, `minimum_load_policy_interaction`,
`maximum_gap_policy_interaction`, `primary_protection_interaction`,
`linked_or_ha_structure`, or (only when the evidence does not cleanly
attribute a cause) `unresolved_multi_family_interaction`. Every authoritative
label carries core/counterfactual evidence. An interaction supported only by
an invalid or unproven witness is downgraded to `low_confidence_signal` and is
not a formal secondary classification.

## Time budgets (frozen, no parameter sweep)

| Stage | Budget |
|---|---|
| Group-level core | 60s |
| Fine-grained core | 120s |
| Each counterfactual variant (x8) | 30s |
| Relaxation Stage 1 | 120s |
| Relaxation Stage 2 | 120s |
| Relaxation Stage 3 | 60s |

## Output artifact

`/Users/klu/Projects/fair-course-allocation-artifacts/robustness-v1/section-plan-feasibility-audit-v1/`
(outside the repository, never committed):

- `audit_manifest_snapshot.json`, `run_manifest.json`, `failures.json`,
  `SHA256SUMS.txt`
- `static_feasibility_summary.csv`, `group_core_summary.csv`,
  `fine_core_summary.csv`, `relaxation_summary.csv`,
  `counterfactual_variants.csv`, `scenario_classifications.csv`,
  `aggregate_summary.json`
- `scenarios/<scenario_id>/static_descriptors.json`, `group_core.json`,
  `fine_core.json`, `counterfactual_variants.json`, and, for INFEASIBLE
  targets only, `relaxation_stage_trace.json`, `relaxation_witness.json`,
  `section_plan_repair_candidates.json`

When reporting semantics need correction, the existing raw artifact is not
overwritten. Use the reporting-only rebuild to create a separate audited
directory; it verifies the raw SHA256 manifest and writes corrected aggregate,
classification, relaxation, evidence-quality, provenance, and checksum files
without invoking CP-SAT.

## Limitations

- Development diagnostic only; not run against stress, negative, or holdout
  scenarios.
- A relaxed diagnostic witness is never publishable and is not a repaired
  section plan.
- A sufficient unsatisfiable core is not proven globally minimum; only
  locally minimal under deletion-filtering (or `unresolved_time_budget` /
  `unresolved_no_infeasibility_proof` when filtering could not complete).
- An unsat core and a relaxation counterfactual answer different questions;
  a counterfactual is not a smaller core or proof of a singleton core.
- A `FEASIBLE` (not `OPTIMAL`) relaxation stage result is "best found," not
  proven minimum relaxation.
- Invalid or unproven witnesses are diagnostic-only; their slack magnitude and
  extra student IDs are not authoritative repair or root-cause evidence.
- Findings describe the current frozen section plan and hard policy
  thresholds; they are not a claim that the affected students' requests are
  unsatisfiable under every possible section plan.
- Deciding whether to change section planning or reconsider a policy
  threshold is a separate, later decision this audit does not make.
