# Remaining K=2 Survivor Expansion Audit v1

## Scope and frozen state

This is an investigation and protocol-design audit. It does not run CP-SAT,
rerun static screening, execute a new section pair, or write to the formal
artifact. The formal six-pair portfolio remains exhausted with no incumbent:

- total unique section pairs: `48,516`
- original G12_0536 static survivors: `1,237`
- selected portfolio pairs tested by Run A: `6`
- six-pair result: `6 INFEASIBLE`, `0` Run B, `0` incumbents
- total formal solver invocations: `6`
- previously proven infeasible unique pair: `1`
- specifically excluded unique pairs: `7`
- formally untested static survivors: `1,231`
- global K2: unresolved
- proven lower bound: `2`
- exact minimum claim: none
- repair witness: none

Nothing below changes those formal counts. A cached-row reduction is explicitly
labeled a projection until a separately approved static-screen implementation
and review persist it.

## Evidence method

Every finding uses one of three classifications:

- `directly_evidenced`: stated by a frozen input, persisted model/log, artifact
  field, checksum, or production builder line.
- `strongly_inferred`: supported by several direct facts but not isolated by an
  unsatisfiable core or an equivalent proof.
- `unknown`: current evidence cannot distinguish the cause.

The roughly three-to-four-second runtimes and zero branch/conflict counters are
not used alone to identify a causal constraint.

Read-only evidence came from the formal screening artifact, its six
`runs/portfolio_pair_*/feasibility` directories, `survivor_pairs.csv`,
`section_effect_signatures.csv`, frozen `normal_dev_10` student/request/section
inputs, the prior section-plan audit, and these model builders:

- `src/hybrid_k2_section_pair_screening.py`
- `src/joint_period_edit_pilot.py`
- `src/allocation/cp_sat_solver.py`
- `src/final_schedule_policy.py`

The artifact had 62 files, 61 checksum entries, and a verified
`SHA256SUMS.txt` hash of
`1c29f43b6abf6a50876c4b3f907b5b1f30ca12eb120d315bc4677341b3aef239`
at the start of this audit.

## Six-run comparison

### Pair and affected-request comparison

`affected primary` is the persisted portfolio's primary-request student union.
`primary overlap` is the number of students with primary candidate edges to
both selected sections. Full candidate exposure includes optional alternates
and is larger; it is not a mandatory enrollment count.

| pair | selected sections | affected primary | primary overlap | full candidate union | G12_0105 exposure |
| --- | --- | ---: | ---: | ---: | --- |
| 1 | AP_3D_ART_DESIGN_01 + CREATIVE_WRITING_01 | 67 | 1 | 191 | Creative Writing rank-3 alternate only |
| 2 | AP_JAPANESE_LANG_01 + SOCIAL_JUSTICE_01 | 68 | 2 | 142 | none |
| 3 | AP_JAPANESE_LANG_01 + CREATIVE_WRITING_01 | 87 | 2 | 209 | Creative Writing rank-3 alternate only |
| 4 | AP_3D_ART_DESIGN_01 + AP_JAPANESE_LANG_01 | 50 | 1 | 107 | none |
| 5 | CREATIVE_WRITING_01 + SOCIAL_JUSTICE_01 | 86 | 1 | 226 | Creative Writing rank-3 alternate only |
| 6 | AP_3D_ART_DESIGN_01 + ROCK_N_ROLL_HISTORY_01 | 23 | 0 | 57 | none |

All six pairs affect the original G12_0536 screening logic: pairs 1-5 move two
G12_0536 primary candidates and pair 6 moves one. None moves a G12_0105 primary
section.

### Placement and capacity comparison

| section | original | frozen non-original destinations | primary demand | all candidate edges | capacity |
| --- | --- | --- | ---: | ---: | ---: |
| AP_3D_ART_DESIGN_01 | P7 | P1, P2, P4, P5, P6 | 15 | 45 | 40 |
| AP_JAPANESE_LANG_01 | P7 | P1, P2, P4, P5, P6 | 36 | 64 | 40 |
| CREATIVE_WRITING_01 | P3 | P1, P2, P4, P5, P6 | 53 | 147 | 40 |
| SOCIAL_JUSTICE_01 | P3 | P1, P2, P4, P5, P6 | 34 | 80 | 40 |
| ROCK_N_ROLL_HISTORY_01 | P4 | P3, P7 | 8 | 12 | 40 |

All five are one-period logical sections. Candidate edges exceeding capacity
does not prove a capacity contradiction: alternates are optional, and ordinary
students may leave one primary unmet. None of these five courses crosses the
strict high-demand threshold of primary demand greater than 120.

### Persisted model and presolve comparison

The six models each contain 617,170 variables and 1,352,984 constraints. Their
binary sizes are 101,734,174 bytes for recovered pair 1 and 101,734,178 bytes
for pairs 2-6. Each log has a distinct model fingerprint, but every log records:

```text
INFEASIBLE: 'var #168708 as empty domain after intersecting with [1]'
status: INFEASIBLE
conflicts: 0
branches: 0
```

Read-only protobuf inspection maps variable 168708 in all six persisted models
to:

```text
assignment__primary_G12_0105_CHINESE4__CHINESE4_01
```

That variable participates in:

- CHINESE4_01 capacity `<= 40`;
- G12_0105 period-occupancy channels;
- G12_0105 target-course upper bound;
- G12_0105 ordinary-primary lower bound of six assigned primaries out of seven;
- G12_0105 minimum-five and target-minus-one lower bounds;
- the changed-section/affected-assignment channel.

The presolve rule application counts vary modestly by pair. Because presolve
stops on the empty domain, the logs do not emit a final presolved model or a
unique removed-variable/removed-constraint count. Rule application counts are
not counts of unique causal constraints.

## Classified findings

### Directly evidenced

1. All six Run A models use the same 164,269 assignment edges, 841 placement
   choices, unchanged capacities, no hint, no objective, no candidate pruning,
   one worker, seed `20260630`, and a 75-second limit.
2. Each pair forces its two selected logical sections changed and every other
   editable section original. The builder gives every editable section a
   placement one-hot and a changed/original equivalence.
3. All six finish in presolve with the same variable-name manifestation,
   G12_0105's CHINESE4 primary assignment, but with different model
   fingerprints.
4. G12_0105 is ordinary (`priority_protected=false`) with target 7 and seven
   logical primaries. Its CHINESE4, FOOTBALL, and INTERMEDIATE_ACTING primaries
   each have one section and all three sections are at P1.
5. Student-period constraints permit at most one assigned course in P1.
   Ordinary-primary policy requires at least six of seven primaries, so at most
   one primary may be unmet.
6. No selected portfolio pair moves CHINESE4_01, FOOTBALL_01, or
   INTERMEDIATE_ACTING_01. Three pairs move only G12_0105's rank-3 Creative
   Writing alternate; three have no G12_0105 candidate edge.
7. Therefore at least two of those three P1 primaries remain unmet in every
   destination combination of every selected pair. This directly contradicts
   the ordinary max-one-primary-unmet hard rule. Alternates cannot satisfy that
   primary count.
8. G12_0105 also requests high-demand ANATOMY_PHYSIOLOGY (171 primaries),
   AP_LIT (240), and AP_PHYSICS1 (130), all of which are hard-required by the
   greater-than-120 rule. Those requirements make the schedule tighter but are
   not needed for the three-P1 contradiction proof.
9. Linked Government/AP Macro rows are one logical block, and HA double-period
   occupancy uses exact occupied periods in the builder. None of the five
   selected changed sections is linked or double-period.

### Strongly inferred

1. The common variable-168708 presolve message is consistent with propagation
   of the proved G12_0105 P1/ordinary-primary contradiction.
2. The original G12_0536-only static screen missed this condition because its
   manifest and preceding candidate line explicitly excluded G12_0105 from
   evidence, candidate generation, and objectives while the production hard
   model still contains every G12_0105 constraint.
3. An all-student local feasibility screen should eliminate substantially more
   pairs than a G12_0536-only screen. The concrete P1 rule alone projects a
   99.0% reduction on cached untested survivors.

These are not claims that one named protobuf constraint was the first or only
presolve cause.

### Unknown

1. The raw logs do not identify the exact propagation chain that emptied
   variable 168708.
2. The evidence does not show whether section capacity, high-demand
   requirements, minimum-five, or maximum-gap constraints also participate in
   the final presolve contradiction.
3. There is no persisted unsatisfiable core for any Run A model.
4. The logs do not establish how many unique variables or constraints presolve
   removed before failure.
5. Six deliberately selected pairs are too biased and too small a sample to
   estimate the infeasible rate of the remaining survivors.
6. No current evidence establishes feasibility for any of the projected 12
   pairs or infeasibility of global K2.

## Safe static exclusions

Only a proved necessary condition may use `safe_static_exclusion`. An
implementation must fail closed if its input universe, linked identity,
placement domain, request count, or proof preconditions drift.

### G12_0105 three-P1 repair-section intersection

- **name:** `g12_0105_three_p1_repair_section_intersection`
- **classification:** `safe_static_exclusion`
- **mathematical definition:** let
  \(R=\{\text{CHINESE4_01},\text{FOOTBALL_01},
  \text{INTERMEDIATE_ACTING_01}\}\). A fixed pair \(P\) must satisfy
  \(P\cap R\ne\varnothing\).
- **affected scope:** frozen normal_dev_10 fixed-pair universe only.
- **why necessary:** if the pair moves none of R, all three corresponding
  primaries stay at P1; at most one can be assigned, forcing at least two
  primary misses where at most one is allowed.
- **proof sketch:** restrict any purported production solution to G12_0105.
  The P1 conflict gives \(\sum_{r\in R}x_r\le1\). The ordinary hard rule over
  seven primaries gives \(\sum x_r\ge6\). Even if the other four primaries are
  assigned, the upper bound is \(4+1=5\), a contradiction.
- **required inputs:** student protection flag and target, logical primary
  universe, candidate sections, frozen original periods, selected pair IDs.
- **complexity:** O(1) per pair after one indexed preflight.
- **implementation difficulty:** low.
- **false-exclusion risk:** none if the single-section and fixed-original
  preconditions are checked; otherwise fail closed.
- **expected survivor reduction:** **estimate/cached projection**, 1,219 of
  1,231 untested survivors (99.0%), leaving 12. This is not yet a formal
  screening result.
- **data exposure:** all inputs already exist in canonical allocation input,
  placement domains, and cached survivor rows.

The projected 12 consist of each of the three repair sections paired with one
of AP_3D_ART_DESIGN_01, AP_JAPANESE_LANG_01, CREATIVE_WRITING_01, or
SOCIAL_JUSTICE_01. Each repair section has only non-P1 destination choices:
CHINESE4_01 can move to P7; FOOTBALL_01 and INTERMEDIATE_ACTING_01 can move to
P3 or P7.

### All-student local hard-policy feasibility

- **name:** `all_student_local_exact_feasibility`
- **classification:** `safe_static_exclusion`
- **mathematical definition:** for every destination combination \(d\), solve
  each student's exact local request-selection problem with candidate
  membership, occupied periods, duplicate logical identities, linked/HA
  occupancy, protected/ordinary primary limits, high-demand requirements,
  minimum five, target-minus-one, and target upper bound, while relaxing all
  cross-student capacity constraints. Exclude the pair if every \(d\) has at
  least one locally infeasible student.
- **affected scope:** every fixed pair and full frozen non-original destination
  product.
- **why necessary:** any global production assignment restricts to a feasible
  schedule for every individual student.
- **proof sketch:** the local model removes capacity coupling and is therefore
  a relaxation. Infeasibility of a relaxation for one student disproves a
  global assignment for that destination.
- **required inputs:** canonical requests/candidates, all student policy flags,
  linked identities, period-unit shapes, frozen domains, thresholds.
- **complexity:** O(pair destinations × students × local DP state); with seven
  periods and small request counts, bitmask DP is bounded and deterministic.
- **implementation difficulty:** medium.
- **false-exclusion risk:** none after exhaustive fixtures prove equivalence of
  request grouping, linked blocks, HA occupancy, fallback, and policy counts.
- **expected survivor reduction:** at least the P1 projection above if that
  proof is encoded; additional reduction is unknown.
- **data exposure:** current code exposes most data, but its existing evaluator
  handles only G12_0536 primaries and must not be generalized without adding
  alternates, fallback, protection, and high-demand semantics.

### Mandatory-request section-capacity Hall bound

- **name:** `mandatory_request_section_capacity_hall_bound`
- **classification:** `safe_static_exclusion`
- **mathematical definition:** for any set of hard-mandatory logical requests
  \(Q\), require
  \(|Q|\le\sum_{j\in N(Q)}capacity_j\), where \(N(Q)\) is the union of eligible
  logical sections under the frozen candidate index. Exclude only when a
  violated cut is accompanied by the exact request/section witness.
- **affected scope:** protected primaries and greater-than-120 high-demand
  primaries; optionally other obligations only after a proved lower-quota
  transformation.
- **why necessary:** every mandatory request consumes one seat in one eligible
  section.
- **proof sketch:** direct Hall/capacity cut on any production assignment.
- **required inputs:** mandatory request set, candidate index, logical section
  capacities, linked identities.
- **complexity:** polynomial max-flow for the basic bipartite formulation.
- **implementation difficulty:** medium.
- **false-exclusion risk:** none for exact mandatory requests; ordinary
  alternatives and fallback must remain relaxed unless their transformation
  is separately proved.
- **expected survivor reduction:** unknown and possibly zero because placement
  does not change capacity or candidate membership.
- **data exposure:** yes.

### Destination period-demand lower bound

- **name:** `destination_period_demand_lower_bound`
- **classification:** `safe_static_exclusion`
- **mathematical definition:** for each destination combination and period set
  \(T\), compute a proved lower bound \(L(T)\) on mandatory seat-period units
  forced into \(T\); require \(L(T)\) not exceed the corresponding
  section-capacity and per-student period resource bound.
- **affected scope:** destination-specific protected/high-demand obligations,
  linked blocks, and HA double-period units.
- **why necessary:** production assignments cannot create seat-period or
  student-period resources.
- **proof sketch:** sum the exact lower-bound obligations over a resource cut.
  Pair exclusion requires every destination combination to have a violated
  cut.
- **required inputs:** destination placements, candidate membership,
  capacities, mandatory requests, linked/HA occupied-period units.
- **complexity:** O(destination combinations × 2^7 × flow-bound cost).
- **implementation difficulty:** high.
- **false-exclusion risk:** none only if \(L(T)\) is a lower bound; heuristic
  demand allocation must not be used.
- **expected survivor reduction:** unknown.
- **data exposure:** yes, but no current helper supplies the proved aggregate
  lower bound.

### Linked/HA/domain structural validity

- **name:** `linked_ha_non_original_domain_validity`
- **classification:** `safe_static_exclusion`
- **mathematical definition:** both sections must have at least one legal
  non-original logical placement; linked semester rows must remain atomic and
  HA options must retain their required consecutive two-period shape.
- **affected scope:** pairs containing linked or double-period sections.
- **why necessary:** illegal or empty destination domains cannot satisfy the
  fixed-pair changed-section constraints.
- **proof sketch:** placement one-hot plus changed=1 requires one legal
  non-original choice for each selected logical section.
- **required inputs:** logical structure type, linked group, original placement,
  frozen destination domain.
- **complexity:** O(domain size) per pair.
- **implementation difficulty:** low.
- **false-exclusion risk:** none with logical-section rather than physical-row
  identities.
- **expected survivor reduction:** **estimate 0** on the current artifact,
  because structural revalidation already passed and invalid-domain count is
  zero.
- **data exposure:** already implemented by domain validation.

## Ranking heuristics

These rules are `ranking_heuristic`, never safe exclusions:

1. Rank by descending count of destination combinations that pass the proved
   all-student local relaxation.
2. Then rank by descending G12_0536 core-feasible placement combinations.
3. Prefer larger minimum mandatory-course capacity slack and lower predicted
   period saturation.
4. Prefer smaller affected-primary-student union as a disruption/cost proxy.
5. Prefer more changed core period relationships, then lower canonical period
   displacement.
6. Diversify repeated section and course participation across a batch.
7. Give diagnostic priority to pairs that change a section named in a repeated
   presolve witness, but never exclude a pair for lacking that name.
8. Treat linked/HA complexity as an audit-cost penalty, not as evidence of
   infeasibility.

For the cached 12-pair projection, a deterministic fallback tuple using
existing fields is:

```text
(
  -core_feasible_placement_combinations,
  affected_student_union_count,
  -changed_candidate_period_relationships,
  total_absolute_period_displacement,
  pair_id,
)
```

The canonical JSON ordering projection has SHA256
`f1246ec8582d6925e26fcc9ee53583a7ac275d6063781b2f59865af788573927`.
This hash is projection output, not a formal frozen batch artifact. Any future
all-student metric changes the ordering schema and therefore must produce a new
ordering hash before execution.

## Compact batch Run A protocol

This protocol is designed only; it is not executed here.

### Preconditions

1. Independently verify the formal safe-screen implementation and unit-test the
   P1 proof and all-student evaluator.
2. Reverify every frozen source artifact hash, canonical input fingerprint,
   section capacity hash, logical identity hash, candidate membership hash,
   domain hash, seed, worker count, and OR-Tools version.
3. Build the complete candidate table from the cached 1,231 survivors without
   altering the formal artifact.
4. Persist the ordered candidate records, ordering schema version, and ordering
   hash in a new artifact root.
5. Require an explicit authorization for solver execution.

### Frozen execution contract

- deterministic batch size: 12 pairs;
- per-pair limit: 75 seconds;
- global invocation budget: 12;
- checkpoint interval: every four completed pairs and at batch end;
- seed: `20260630`;
- workers: 1;
- one fresh solver per pair;
- same frozen production model and source hashes;
- both selected sections forced changed;
- every other editable section forced original;
- complete frozen non-original destination domains;
- no hint, objective, or candidate pruning;
- no Run B and no automatic retry;
- every conclusion scoped to the tested fixed pair.

If the proof-backed screen is not approved or leaves more than 32 survivors,
do not silently fall back to 1,231 runs. Return to protocol review.

### Checkpoint and resume

Each pair has states `planned`, `started`, or `completed`. A completed record is
keyed by pair ID plus ordering hash, source/input hashes, model/config
fingerprint, seed/workers/budget, raw-log hash, and response hash. Resume:

1. verifies the checkpoint and periodic SHA manifest;
2. rejects duplicate or unknown pair IDs;
3. subtracts completed IDs from the immutable ordered list;
4. never reruns a completed pair;
5. never automatically reruns a `started` but incomplete pair;
6. requires human classification of an interrupted run before proceeding.

Atomic replacement is required for checkpoint and manifest writes.

### Per-run evidence

Retain:

- raw solver log;
- structured status and full response hash;
- pair delta and full solver configuration;
- canonical input/source hashes;
- base-model, delta, and reconstructed model fingerprints;
- model variable/constraint counts;
- build, solve, artifact-write, and end-to-end timings;
- explicit hint/objective/pruning audits;
- artifact-write status;
- scoped conclusion.

Hash each file once when finalized. Every four pairs, write a SHA256 manifest
over the new immutable entries and hash that manifest. Do not repeatedly
rehash all earlier model files during ordinary checkpoints.

### Fail-closed and anomaly handling

- `INFEASIBLE`: mark only that fixed pair excluded.
- `FEASIBLE` or `OPTIMAL` with an incumbent: stop the whole batch; preserve
  evidence and request separately authorized witness replay/production
  validation.
- `UNKNOWN`: retain as unresolved, stop the batch for review, no Run B.
- `MODEL_INVALID`: stop immediately.
- source/config/model fingerprint drift: stop before Solve.
- missing response hash, raw-log final status mismatch, duplicate completion,
  checksum failure, or artifact-write failure: stop immediately.
- post-solve artifact failure: retain the raw log and mark recovery-only; do
  not reconstruct an original response hash and do not rerun automatically.
- build time over 22.4 seconds (2× the audited estimate), solve wall time over
  its configured limit plus serialization tolerance, nonzero anomaly rate, or
  unexpected model size/count drift: escalate before the next pair.

Partial results never support a global K2 statement.

## Storage and cost analysis

Every value in this section is an **estimate** unless explicitly called an
observed measurement.

Observed inputs:

- six solver wall times: 2.826-3.775 seconds;
- observed median solver wall time: 3.1269 seconds;
- model size: approximately 101.7 MB each;
- mean compact evidence size excluding model.pb: approximately 6.44 KB per
  pair (pair 1's recovery record is larger);
- related audited hybrid model build: 11.1867 seconds. The six Run A artifacts
  did not persist build time, so their exact construction runtime is unknown;
- read-only checksum measurement: six model files hashed in 2.53 seconds,
  approximately 241 MB/s on this machine.

### Strategy comparison for all 1,231 current survivors

| strategy | reproducibility/auditability | disk estimate | checksum estimate | recovery/implementation risk |
| --- | --- | ---: | ---: | --- |
| A: full model.pb per pair | strongest byte-level evidence; simplest replay | 125.24 GB / 116.64 GiB; 8,617 files | about 8.65 minutes for one full model pass at observed throughput | low logic risk; high storage/write cost |
| B: canonical base model + deterministic pair delta/config/fingerprint | strong if reconstruction is byte/fingerprint verified | 0.111 GB; about 8,618 unbundled files | one 101.7 MB base plus compact deltas | medium risk; depends on deterministic builder, versions, and exact delta application |
| C: compact normal evidence; full model only for anomaly | adequate routine audit, weaker ordinary byte replay | 7.93 MB at 0% anomaly; 1.33 GB at 1%; 6.32 GB at 5% | compact normally; anomaly models hashed once | lowest storage, highest dependence on later deterministic reconstruction |

Estimated all-1,231 runtime:

- model construction: 3.83 hours;
- solver at observed median: 1.07 hours;
- build plus median solve: 4.89 hours sequential;
- solver-only worst case at the frozen 75-second limit: 25.65 hours;
- artifact serialization/write time: unknown and must be measured separately.

Projected post-proof 12-pair estimates:

- full model storage under A: 1.22 GB;
- build time: 2.24 minutes;
- solver time at observed median: 0.63 minutes;
- combined build plus median solve: 2.86 minutes;
- solver-only 75-second worst case: 15 minutes;
- one full model checksum pass: about 5.1 seconds;
- files under the current seven-file layout: 84.

### Recommendation

Use strategy B, with strategy-A full model capture for any anomaly. Persist one
canonical no-pair-fixed base model, a deterministic pair delta that fixes all
312 changed-section variables, complete config/source fingerprints, raw logs,
and response hashes. Before execution, reconstruct all five normally persisted
historical pair models (pairs 2-6) and require exact model fingerprints and
variable/constraint counts; pair 1 remains recovery-only and cannot be used as
an original-proto byte oracle. If reconstruction is not exact, use strategy A
for the projected 12 rather than weakening auditability.

## Decision and stopping rules

1. **Strengthen static screening first** when a proof is complete and cached
   projection removes at least 10% or 100 survivors with no fixture failure.
   The P1 rule projects 99.0%/1,219, so it clears this threshold.
2. **Bulk Run A is worthwhile** only after at most 32 survivors remain. Above
   32, run no solver and improve the proof-backed screen or ranking first.
3. **First batch:** the projected 12 only, after independent proof and ordering
   review. No automatic second batch.
4. **After each four-pair checkpoint and batch end, inspect:** status mix,
   incumbent rate, infeasible rate, unknown/model-invalid rate, repeated
   presolve signature rate, median/p95 build and solve time, model/config hash
   drift, artifact failure rate, cumulative disk, and cumulative invocation
   budget.
5. **Expand only if:** survivors remain, anomaly rate is zero, hashes are
   stable, median build is <=16.8 seconds (1.5× estimate), median solve is
   <=10 seconds, p95 solve is <=20 seconds, and a new explicit invocation
   budget is approved.
6. **Redesign ranking/static analysis** after eight consecutive
   `INFEASIBLE` results with the same presolve variable/signature, or after a
   complete 12-pair batch with zero incumbents.
7. **Stop immediately** on any incumbent, UNKNOWN, MODEL_INVALID, checksum
   failure, response/log mismatch, artifact failure, or fingerprint drift.
8. **K3 transition:** only after the 12 projected pairs are exhausted or K2
   closure is explicitly abandoned as a product decision. K3 may seek a
   witness but does not retroactively prove K2 infeasible; the lower bound
   remains 2 unless K2 is closed by independently accepted evidence.
9. **Cost-bounded repair transition:** prefer it when the product needs a
   useful repair rather than exact cardinality, especially after a zero-
   incumbent 12-pair batch or when K3 storage/build projections exceed an
   approved budget.
10. **Project-value stop:** stop pursuing global closure when there is no
    product requirement for an exact minimum and the project already
    demonstrates reproducible screening, scoped solver evidence, fairness
    constraints, claim discipline, and an auditable next-step protocol.

## Audit conclusion

The six failures have a common, provable blind spot: the portfolio repairs
G12_0536 but never moves any of the three P1-only primary sections that make
ordinary student G12_0105 locally infeasible. This supplies a safe static
condition and a projected reduction from 1,231 to 12, but it is not written
into the formal artifact and no new feasibility result is claimed here.

Global K2 remains unresolved, the proven lower bound remains 2, no exact
minimum is known, and no repair witness exists. This audit ends at design; it
does not implement or start a batch solver runner.
