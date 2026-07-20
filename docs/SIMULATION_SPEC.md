# Whole-School Scheduling Simulation Specification v1

## 1. Purpose

Build a synthetic, Torrey Pines–inspired high-school scheduling environment that can test whether different allocation algorithms produce complete, fair, and stable student schedules.

The simulation is not presented as an exact copy of TPHS. It combines:

- public course information;
- student-informed estimates;
- explicit model assumptions;
- randomized yearly demand.

Anonymous school data should be able to replace the synthetic inputs without changing the solver.

## 2. Solver boundary

The solver starts **after counselors approve course requests**.

The solver does not:

- check prerequisites;
- decide whether a student belongs in honors or AP;
- recommend courses;
- evaluate graduation eligibility before requests are submitted.

The solver does:

- assign approved requests to concrete sections and periods;
- respect section capacity and schedule conflicts;
- preserve primary requests when possible;
- use ranked alternates when necessary;
- minimize incomplete schedules;
- distribute unavoidable losses fairly.

## 3. School population

| Grade | Approximate students |
|---|---:|
| 9 | 700 |
| 10 | 650 |
| 11 | 640 |
| 12 | 640 |
| **Total** | **2,630** |

These are simulation parameters, not claims of exact current enrollment.

## 4. Target number of scheduled classes

| Grade | 7 classes | 6 classes | 5 classes |
|---|---:|---:|---:|
| 9 | 90% | 10% | 0% |
| 10 | 80% | 20% | 0% |
| 11 | 60% | 35% | 5% |
| 12 | 33% | 47% | 20% |

`target_course_count` is both the student's desired load and the maximum number
of normal-period course slots to fill. Greedy comparison artifacts may be
underfilled and reported for analysis, but a CP-SAT final FEASIBLE or OPTIMAL
schedule must satisfy Final Schedule Policy Gate v1 before it can be described
as publishable.

## 5. Period structure and unscheduled time

The school day has seven periods: P1–P7.

Unscheduled periods are restricted to the edges of the day:

- morning: P1;
- afternoon: P6 and/or P7.

The solver must not create an interior free period in P2–P5 merely to make optimization easier.

Exact allowed patterns should remain configurable. V1 uses the hard rule that all free periods must be a subset of `{P1, P6, P7}`.

## 6. Grade-level request-generation rules

These rules guide synthetic request generation only. They are not eligibility checks inside the solver.

### Grade 9

Typical structure:

- English 9 or English 9 Honors;
- one mathematics course;
- Biology;
- PE or an approved athletic/PE arrangement;
- usually one world language;
- remaining slots are electives.

About 90% request seven classes. A known capacity-risk elective is Computer Programming.

Special ELD, foundational, functional, readiness, and other support pathways are excluded from V1 unless later added as a separate scenario.

### Grade 10

Core distributions:

- English 10 vs English 10 Honors: approximately 50/50;
- World History vs AP World History: approximately 50/50;
- Chemistry vs Honors Chemistry: approximately 1/3 vs 2/3;
- mathematics normally begins at Integrated Math 2 or above.

Math details:

- Integrated Math 2/3 Honors Accelerated requires two periods;
- AP Calculus AB: approximately 4–8 Grade 10 students, base value 6;
- AP Calculus BC: approximately 20 Grade 10 students;
- Introduction to Calculus is not normally generated for Grade 10.

Known capacity risks:

- AP CSP, especially among students without prior Computer Programming;
- AP CSA for any Grade 10 applicant.

### Grade 11

Typical structure:

- English 11 or AP English Language;
- U.S. History or AP U.S. History;
- mathematics normally begins at Integrated Math 3 or above;
- science, language, PE, and electives vary by student;
- AP Seminar is available;
- AP Art History is available to Grades 11–12.

Removed or corrected rules:

- AP Humanities is no longer offered;
- AP European History is not generated for Grade 11;
- world languages have no honors versions.

Special course structures:

- AP Physics C occupies one period for the full year, with Mechanics in one semester and E&M in the other;
- Calculus D + Linear Algebra is a dual-enrollment, school-scheduled block that occupies one period and is capacity constrained.

Known capacity risks:

- Calculus D + Linear Algebra: normal capacity about 40, sometimes overloaded to 45, with roughly five students still unable to enroll in the observed year;
- AP Statistics: historically about three sections of 40, with roughly five students unable to enroll in the observed year;
- AP Physics C: observed capacity pressure, exact waitlist unknown.

### Grade 12

Typical required structure:

- English 12 or AP English Literature;
- one shared-period Government/Economics block;
- other mathematics, science, language, PE, and electives are optional.

Government/Economics rules:

- Government and Economics occupy one period across the year;
- semester order varies by section;
- regular and AP levels may be mixed:
  - Government + Economics;
  - Government + AP Macroeconomics;
  - AP Government + Economics;
  - AP Government + AP Macroeconomics.

Math begins at Introduction to Calculus or above in the normal Grade 12 generator. Math 3 and Math 3 Honors are not generated for ordinary Grade 12 students.

Grade-specific electives:

- AP Research is available to Grade 12;
- AP European History is Grade 12 only;
- AP Art History is available to Grades 11–12.

## 7. Special scheduling representations

### Double-period course

Example: Integrated Math 2/3 Honors Accelerated.

The student receives one course request, but the assigned section consumes two periods. Whether the periods must be consecutive remains configurable until verified.

Math policy evaluation classifies math courses using
`course_catalog.department = Mathematics`. If `MATH2_3_HA` is the student's
only math primary and is unmet, the seeded random greedy baseline runs an
explicit mandatory fallback attempt to `MATH2` after all primary requests and
before ordinary ranked alternates. The fallback does not erase the original
Math 2/3 primary unmet and does not increase primary satisfaction. If assigned,
it satisfies math coverage and consumes one period unit; if it fails, the
baseline returns a reported math coverage violation rather than treating the
case as globally infeasible.

### Sequential semester block sharing one period

Examples:

- Government/Economics;
- AP Physics C content sequence;
- Calculus D/Linear Algebra dual-enrollment sequence.

The block occupies one period for the full year while the course content changes by semester.

### Zero-period or non-school-period credit

Some athletic credit or external dual-enrollment arrangements may satisfy a requirement without consuming P1–P7. These must be represented explicitly with `occupies_school_period = false` rather than treated as ordinary classes.

## 8. Capacity defaults

| Category | Default capacity |
|---|---:|
| Normal academic section | 40 |
| Most AP sections | 40 |
| AP Computer Science A | 25 |
| PE section | 50 base, configurable 40–60 |
| Calculus D + Linear Algebra | 40 normal, overload up to 45 |

These are default simulation values. Individual courses may override them.

## 9. Section planning policy

The section planner converts synthetic primary request demand into section
counts and period layout. It does not decide which students receive which
section.

V1 uses the model assumption that any course with positive primary logical
demand opens at least one section before the waitlist expansion rule is applied.
Courses with zero primary demand open zero sections.

### Uniform expansion rule

For any course with standard section capacity `C`, after currently planned
sections are filled, an additional section may open when the remaining waitlist
reaches `ceil(0.5 × C)`, subject to staffing, rooms, period layout, complete
schedule feasibility, and fairness constraints.

Illustration for a 40-seat course:

- 19 remaining students: no new section;
- 20 or more remaining students: consider one additional section.

Illustration for AP CSA with a 25-seat course capacity:

- 12 remaining students: no new section;
- 13 or more remaining students: consider one additional section.

Illustration for a 50-seat PE course:

- 24 remaining students: no new section;
- 25 or more remaining students: consider one additional section.

The rule applies uniformly to core courses, AP courses, CS courses, arts, PE,
and niche electives. It may be applied repeatedly: after adding a section, if
the remaining waitlist still reaches the same threshold, another section may be
considered.

### High-demand full-capacity floor

When approved logical primary demand is greater than 120, the section planner
also applies a full-capacity floor. The final logical section count must be at
least `ceil(logical_primary_demand / effective_section_capacity)`, using the
same effective capacity that will appear on the generated section rows.

The final section count is the larger of the uniform 50% waitlist result and
this high-demand full-capacity floor. Demand of exactly 120 does not trigger
the floor; demand of 121 does. This policy is based only on approved logical
primary demand, not course name, department, or a separate required-course list.

This is a section-planning rule, not a student-assignment rule. It does not
guarantee that every
student can be scheduled. For courses at or below 120 demand, remaining unmet
demand may come from threshold rounding. For courses above 120 demand, planned
logical capacity should cover all approved logical primary demand, but later
student-level allocation can still fail because of period-layout conflicts,
full-schedule feasibility, or fairness constraints.

### Niche electives

Low-demand niche electives often result in one section, and sometimes two, as
a normal consequence of low synthetic demand and the 50% waitlist threshold.
This is descriptive, not a hard maximum.

### Period layout

The V1 section planner places planned sections into P1–P7 using a deterministic
heuristic seeded by scenario and `section_planning_seed`. The heuristic uses primary
co-request conflicts to reduce obvious period conflicts, spreads multiple
sections of the same course when feasible, and balances section counts across
periods. It is not a proof of student-level schedule feasibility.

Math 2/3 Honors Accelerated is modeled as a consecutive double-period section
using the configurable pairs P1-P2 through P6-P7. This is a V1 model
assumption until more precise master-schedule information is available.

Government/Economics linked blocks generate two semester rows that share the
same linked section group and period. Semester order is assigned by the
planner seed and is not determined by student requests.

### Not modeled in section planning

The current planner does not model teacher availability, teacher load, room
inventory, room type, lab availability, or which teachers can teach which
courses.

## 10. Baseline allocation algorithms

Baseline allocation algorithms use the same canonical students, approved
requests, fixed sections, capacities, period layout, mandatory math fallback
configuration, and policy report definitions. They differ only in ordering
policy.

The seeded random greedy baseline processes students and primary requests with
seeded random order, then runs mandatory math fallback, then ranked alternates.

The constrained-first greedy baseline uses the same three phases:

1. primary requests;
2. mandatory math fallback;
3. ranked alternates.

It is still greedy. It does not backtrack, swap, displace students, repair a
previous assignment, regenerate sections, or prove global optimality. It only
changes ordering:

- protected status, math coverage risk, high-demand primaries, scarce
  candidates, candidate flexibility, double-period burden, target load, and a
  seeded tie-break determine initial student order;
- each student's remaining primary requests are dynamically reordered by
  current feasible candidate count, math/high-demand/double-period priority,
  static candidate count, period units, and seeded tie-break;
- candidate sections are ordered by current remaining capacity ratio, period
  pressure, current-student future-option preservation, remaining seats, and
  seeded tie-break.

All actual feasibility decisions still go through the same fixed-section
assignment state. Fairness fields influence constrained-first ordering only;
they are not hard constraints in the baseline. A baseline failure to assign a
request is reported as an outcome, not as global infeasibility.

Benchmark Runner v1 defaults to seeded random greedy and constrained-first
greedy only. First-come-first-served, grade-priority, and CP-SAT comparisons
remain broader evaluation targets, but CP-SAT is opt-in because it is
substantially more expensive than the greedy baselines.

## 11. Fixed-section CP-SAT allocation

The fair CP-SAT allocator starts from canonical students, approved logical
requests, fixed logical sections, fixed capacities, fixed occupied periods,
and the configured mandatory math fallback rules. It does not change section
counts, capacities, period layout, course eligibility, or generated requests.

The CP-SAT hard policies are:

- every selected assignment must use a request-specific canonical candidate
  logical section;
- each logical request can be assigned at most once;
- section capacity, student period conflicts, target period units, and
  duplicate logical course/block identity must match `AllocationState`
  semantics;
- protected students must have zero unmet logical primaries;
- ordinary non-protected students may have at most one unmet logical primary;
- logical primary requests for courses with approved logical primary demand
  greater than 120 must be assigned.
- any returned final FEASIBLE or OPTIMAL solution must also satisfy Final
  Schedule Policy Gate v1: at least five assigned logical courses and at most
  one logical schedule gap for every student.

The high-demand hard policy uses only logical primary demand. Demand of 120 or
less does not trigger a full-assignment guarantee. Courses at or below that
threshold can still leave students unassigned because the section planner uses
the uniform 50% waitlist expansion policy rather than a full-capacity floor.

Mathematics is identified from `course_catalog.department == "Mathematics"` in
configuration helpers, not from course-name strings. A student with one math
primary receives very high soft priority, and students with multiple math
primaries receive high soft priority for at least one math assignment. This is
not a hard constraint. If fixed capacity and periods leave a single-math
student without math coverage, the solver reports a math coverage violation;
it does not create a section, increase capacity, or call the model globally
infeasible on that basis alone.

The configured mandatory fallback currently maps Math 2/3 Honors Accelerated
to Math 2. Fallback assignment is synthetic: it does not change the original
primary request outcome, but it consumes real target units, period slots, and
section capacity, and it can satisfy math coverage. Ranked alternates are also
modeled globally and can fill remaining target units without changing primary
satisfaction statistics.

Soft goals are solved lexicographically:

1. minimize math coverage violations;
2. minimize primary unmet count, then primary unmet period units;
3. maximize assigned logical courses;
4. maximize rank-1 alternates;
5. maximize rank-2 alternates;
6. maximize rank-3 alternates;
7. maximize fully scheduled students under the legacy period-unit metric;
8. minimize total remaining period units;
9. apply a deterministic seeded assignment tie-break.

The implementation separates this into two CP-SAT model scopes:

- the Core model contains primary requests, mandatory math fallback options,
  math coverage variables, and all hard constraints needed for primary
  allocation quality;
- the Enrichment model contains the full request set, including ranked
  alternates and complete-schedule variables, and fixes the Core incumbent
  values before optimizing lower-priority goals. Final schedule minimum-course
  and maximum-gap constraints live in this full Enrichment model because they
  may require ranked alternates or mandatory fallback assignments.

Core primary quality is solved in three distinct stages: math coverage
violations, primary unmet count, and primary unmet period units. The solver
does not collapse count and units into one large-weight objective.

After the Core incumbent values are fixed, Enrichment first optimizes logical
schedule completion by maximizing assigned logical-course count across primary,
mandatory fallback, and ranked alternate assignments. This expression counts
Math 2/3 Honors Accelerated and linked semester groups once and does not use
period units. With Final Schedule Policy Gate v1 enabled, the hard maximum
logical gap is one, so maximizing assigned logical courses is equivalent to
minimizing total logical schedule gap and the number of students with a
logical gap, conditional on the higher-priority incumbent values already fixed.
It is intentionally placed above alternate-rank preference quality so the
solver does not choose a lower-rank complete logical schedule merely to protect
rank labels, but it cannot reduce primary satisfaction because primary
objective values are fixed first.

CP-SAT v1.2 adds a primary-only feasibility bootstrap before the Core model.
The bootstrap creates only primary request-section assignment variables plus
minimal helper expressions needed for hard constraints. It enforces canonical
candidate validity, section capacity, period conflicts, target-load upper
bounds, duplicate logical identities, protected all-primary policy, ordinary
max-one-primary-unmet policy, and the demand-greater-than-120 high-demand
primary guarantee. It does not create fallback variables, alternate variables,
math coverage variables, complete-schedule variables, remaining-unit variables,
or any soft objective.

Bootstrap status is diagnostic, not lexicographic optimality. If the no-objective
bootstrap returns a CP-SAT `OPTIMAL` status, that means the hard-feasibility
model was fully solved and a feasible assignment was found; it does not mean
the primary allocation is globally optimal. Bootstrap values are not included
in the solver objective vector and do not update the highest globally proven
optimization stage.

Omitted bootstrap variables are safe because they can be extended with
`fallback=0` and `alternate=0`; math coverage and complete-schedule indicators
are then derived or optimized later by Core and Enrichment. A bootstrap
`INFEASIBLE` result can therefore stop the solve only when the canonical
candidate data are valid and all hard policies above are present. A
`MODEL_INVALID` result means the internal model or canonical candidate data are
malformed, not that the school policy problem is infeasible. If bootstrap ends
`UNKNOWN` without an incumbent, the solver falls back to the Core stages.

The solver may use solution hints to speed search. The default warm start uses
a constrained-first partial hint plus stage-to-stage incumbent hints. When the
Final Schedule Policy Gate is enabled, its full-model feasibility stage also
maps the unchanged constrained-first assignment to a complete 0/1 vector for
candidate and deterministically derived Boolean variables; unselected
candidates receive explicit zero hints. An explicit
`--cp-sat-initial-solution-artifact-dir <path>` opts into a persisted
`persisted_feasible_seed`. The loader verifies SHA256 sums, source
FEASIBLE/OPTIMAL status, policy/capacity/consistency summaries, the exact
canonical fingerprint and request/student universes, then independently
replays assignments through `AllocationState`. Hints are not constraints:
CP-SAT may repair, ignore, or improve them, and the formal hard constraints
and objective stages remain the source of truth. Metadata records the artifact
hashes, source identity, coverage, unknown/duplicate mappings, and stages that
selected the candidate. The final exported assignment always comes from the
current solver response, never directly from the persisted artifact.

The logical schedule completion stage has a controlled disabled mode for
incumbent diagnostics. Disabling it omits only that enrichment objective stage;
it does not change hard constraints, Core objectives, or the meaning of
FEASIBLE, OPTIMAL, UNKNOWN, or INFEASIBLE.

The solver also supports an optional global time budget. Each stage receives
the smaller of its per-stage limit and the remaining global budget. When the
budget is exhausted, lower-priority stages are marked as skipped with explicit
diagnostics rather than being reported as `OPTIMAL`. If a valid incumbent has
already been found, the result can remain `FEASIBLE`; if no incumbent exists,
the result is `UNKNOWN`.

Every stage records CP-SAT status, objective value, best bound, runtime,
conflicts, branches, and whether optimality was proven. A FEASIBLE incumbent
is not reported as OPTIMAL, and UNKNOWN is not reported as INFEASIBLE. When a
stage has a FEASIBLE incumbent but not a proof of optimality, later stages may
continue conditionally with that incumbent value fixed. Such lower-priority
improvements are useful diagnostics, but they do not prove a global
lexicographic optimum.

After CP-SAT returns selected variables, the solution is replayed into a fresh
`AllocationState` with the same supplemental fallback requests. Replay must
pass all local feasibility checks and internal consistency validation. A
solution that cannot replay is treated as an internal model error, not as a
student allocation result.

After replay, any CP-SAT FEASIBLE or OPTIMAL final solution is checked with the
same Final Schedule Policy Gate v1 evaluator used by benchmark artifacts. If
the model claims a final solution but the evaluator fails it, that is an
internal model/evaluator consistency error rather than a normal infeasible
allocation instance.

The logical completion objective value must match the raw logical-course counts
from the actual final CP-SAT `ResponseProto` solution after replay. The model
uses one bounded logical-assigned counter per student, so the objective and its
best bound satisfy `objective <= best_bound <= sum(target_logical_course_count)`.
The exported student-level logical fields are the metric authority; raw
assignment-row counts are not a substitute because linked and double-period
requests can have different row/unit representations. Under the final hard
max-gap policy, the evaluator also checks that logical-full students plus
logical-gap students equals total students, that no student exceeds its target,
and that total logical gap equals the replayed target-minus-assigned total. A
mismatch is treated as an internal solver/evaluator consistency error.

Benchmark reports preserve the historical `fully_scheduled_students` field as
a period-unit metric: it counts students whose assigned period units match the
target period units. That is not always the same as receiving the target number
of logical courses because a double-period logical course such as Math 2/3
Honors Accelerated can fill two period units while counting as one course.
Logical-course completeness is reported separately with
`logical_fully_scheduled_students`, `students_with_logical_schedule_gap`, and
`total_logical_schedule_gap`. Final Schedule Policy Gate v1 uses the logical
course metrics.

The CP-SAT allocator still does not implement dynamic section planning,
local repair, swap/displacement, beam search, or website behavior.

## 12. Known bottleneck courses

The initial calibration set includes:

- Grade 9 Computer Programming;
- Grade 10 AP CSP;
- AP CSA;
- Calculus D + Linear Algebra;
- AP Statistics;
- AP Physics C.

These courses should be represented as realistic stress points rather than guaranteed failures every year.

## 13. Modeling uncertainty and yearly change

Unknown elective demand is not filled with fake precision. Each course receives:

- a demand tier: core, mainstream, popular, niche, or fixed-limited;
- a base participation rate or expected demand;
- a yearly random multiplier;
- a trend multiplier for increasingly competitive course selection;
- initial planning hints and the uniform waitlist expansion policy.

Recommended test scenarios:

1. **Stable year** — demand close to baseline.
2. **Competitive-growth year** — increased AP, CS, advanced math, and advanced science demand.
3. **Stress year** — several popular courses rise simultaneously.
4. **Randomized robustness runs** — repeat many seeds and compare outcomes.

Randomness must be reproducible through separately recorded generation,
section-planning, and solver seeds.

Stable-year benchmark reports must record data-generation, section-planning,
and allocation-solver seeds separately, along with the canonical allocation
input fingerprint. The current comparison checkpoint uses
`data_seed=2026`, `section_planning_seed=2026`, and
`solver_seed=20260630`; the solver seed must not be reused to regenerate the
benchmark input.

Before a benchmark or solver run, load and verify its experiment manifest.
Verification must confirm the data and section-planning seeds against
generation and planner metadata, confirm each stage's recorded output hashes,
and confirm the planner-recorded hashes of its upstream files. Manifest counts
must all come from canonical allocation input. The manifest records the solver
seed, generated and section input paths, canonical counts,
canonical-input/file/configuration hashes, and an optional Git commit. If
verification fails, do not run the benchmark or solver.

### Scenario Robustness Benchmark v1

`data/scenarios/normal_year_robustness_v1.json` is a frozen manifest for the
first multi-seed normal-year measurement suite. It contains one stable
development reference, eleven additional development scenarios, and eight
holdout scenarios. Each entry records three separate roles:

- `data_generation_seed` controls synthetic students and approved requests;
- `section_planning_seed` controls fixed section counts and period layout;
- `algorithm_seed` controls the allocation baseline tie-break.

The reference uses data seed `2026`, section seed `2026`, and algorithm seed
`20260630`. The solver/allocation seed must never be substituted for either
input seed. The manifest also requires unique data/section seed pairs,
canonical reference counts, and a canonical-input hash.

`python -m src.robustness_runner` runs the selected development scenarios with
seeded random Greedy, first-come-first-served Greedy, grade-priority Greedy,
and constrained-first Greedy. CP-SAT is intentionally rejected by this runner
and remains an explicit, separate benchmark choice. It writes scenario inputs,
input difficulty descriptors, benchmark artifacts, aggregate distributions,
and paired comparisons outside the repository when given an external output
directory. Cached scenarios are reused only after fail-closed provenance
checks. Holdout execution requires explicit confirmation and is not a tuning
default.

Difficulty descriptors are calculated from canonical allocation input, not raw
CSV row counts. They describe student load, candidate flexibility,
course-level demand/capacity ratios, and period candidate concentration. A
capacity-only shortfall is a per-logical-course lower bound; it is not a proof
of globally unmet demand. Development results support reproducible comparison
and tuning audits, not claims of generalization to unseen schools or years.
This Phase A suite is normal-year only: it does not include stress scenarios
or capacity shocks. A later Phase B may add those scenario families, and a
later Phase C may add multi-scenario CP-SAT and explicitly confirmed holdout
evaluation.

### Scenario Robustness Benchmark v1 Phase B

Phase B is a development-only stress diagnostic. Its frozen manifest contains
12 ordinary stress scenarios and 3 artificial structural-infeasibility
controls, plus 8 holdout definitions that are not run by default. Ordinary
scenarios have `expected_feasibility=unknown`; structural controls use explicit
certificates and are not evidence about ordinary-year feasibility.

The transform layer reads a persistent normal-year scenario and writes a new
scenario directory atomically. It never edits the base artifact. Enrollment
and popular-course clones copy complete student/request profiles. Alternate
ranks are preserved. Capacity changes are applied once per logical section,
so linked Gov/Econ semester rows are not double counted. The optional
`request_id` column added to transformed CSVs is provenance for cloned rows;
canonical request identity remains defined by the existing student/course,
request-group, and linked-block semantics.

Every transformed scenario records before/after canonical fingerprints, row
and logical-section changes, selected IDs, parameters, transform order, a
deterministic replay hash, and validation status. Structural certificates are
rechecked against the transformed canonical input. The global capacity
certificate counts one capacity per logical section, deduplicates linked
semester rows, and ignores periods as a strict upper-bound proof.

The Phase B runner uses the four Greedy baselines only. It writes ordinary
stress aggregates separately from negative-control summaries, and pairs each
ordinary scenario with its already-persisted normal result. A policy failure
in a structural negative is expected diagnostic output, not a runner crash;
`policy_fail_count=4` means four Greedy result rows failed policy, not four
individual violations. Unexpected schema, provenance, or integrity failures
are runner failures. No
CP-SAT multiscenario evidence, holdout evidence, or generalization claim is
made in Phase B.

### Phase C: Frozen CP-SAT development evaluation

Phase C is a separate measurement layer over the persisted Phase A normal and
Phase B stress development artifacts. Its frozen manifest is
`data/scenarios/cp_sat_development_evaluation_v1.json`. The manifest contains
12 normal development scenarios, 12 ordinary stress scenarios, and 3
structural negative controls. It contains no holdout scenario IDs. Normal and
stress holdouts remain unviewed, unrun, and unavailable for tuning.

The runner calls the formal `run_fair_cp_sat_solver` entry point directly. It
does not regenerate students, re-run section planning, reapply transforms, run
Greedy again, or copy CP-SAT model-building logic. Its configuration is frozen:
solver seed `20260630`, one worker, 30-second bootstrap, 30 seconds per stage,
300 seconds total, and no external persisted seed. The production stage order,
objective order, per-stage trace, response hashes, status, bounds, and policy
replay are exported. `UNKNOWN` means no incumbent was returned within budget;
it is not `INFEASIBLE`. `FEASIBLE` means an incumbent without a proof of
optimality; it is not `OPTIMAL`.

For ordinary scenarios, a final assignment is publishable only if it comes from
the current CP-SAT response and passes Final Schedule Policy Gate v1 and
consistency checks. Missing assignments have null quality metrics. Negative
scenarios validate their structural certificate before solving and must never
produce a publishable assignment. This phase reports development behavior and
controlled stress degradation only; it is not a final test, does not prove
generalization, and does not claim that stress transforms cover all real-world
perturbations. Capacity-only shortfall and negative certificates are diagnostics
under their stated proof assumptions, not global optimization certificates for
ordinary scenarios.

Phase C status reporting keeps the raw OR-Tools terminal status and the stage
that produced it. Only `full_model_feasibility_incumbent=INFEASIBLE` sets
`solver_global_infeasibility_proven`; bootstrap/core or later fixed-objective
stage failures do not. A later stage with fixed prior objectives is reported
as `LEXICOGRAPHIC_STAGE_INFEASIBLE`, while structural negative certificates
are recorded separately from solver proof. Missing assignments retain null
quality metrics. Holdout readiness is blocked when a majority of normal
development scenarios lack publishable assignments, so a vacuous policy pass
cannot make a 0/12 development result ready for holdout.

## 14. Solver priorities

Use lexicographic optimization:

1. minimize math coverage violations;
2. minimize unmet primary requests, then unmet primary period units;
3. maximize rank-1, rank-2, and rank-3 alternates in that order;
4. maximize the number of students receiving a complete target schedule;
5. minimize remaining period units;
6. apply deterministic seeded tie-breaks.

The solver must never silently violate hard constraints.

## 15. Required evaluation metrics

Report at minimum:

- complete-schedule rate overall and by grade;
- primary-request fulfillment rate;
- alternate-use rate;
- number of students missing 1, 2, or more requested courses;
- maximum and average student loss;
- waitlist size by course;
- section utilization;
- fairness by grade and request-load group;
- assignment churn across nearby random-demand scenarios;
- variability across random seeds.

Compare against:

- random lottery;
- first-come-first-served;
- grade-priority allocation;
- the fairness-optimized solver.

The full evaluation may include the broader algorithm set above. Benchmark
Runner v1 defaults to seeded random greedy and constrained-first greedy only;
CP-SAT and fairness-solver comparisons must be explicitly requested and marked
as opt-in benchmark runs.

## 14. V1 implementation order

1. Replace the old prerequisite-centered schema with the new scheduling schema.
2. Validate all configuration files.
3. Generate synthetic students and approved primary/alternate requests.
4. Plan synthetic section counts and period layouts.
5. Implement simple baselines.
6. Implement the fixed-section CP-SAT allocator.
7. Add fairness and robustness metrics.
8. Add section-planning diagnostics and scenario comparisons.
9. Build a lightweight website only after the solver is tested.
