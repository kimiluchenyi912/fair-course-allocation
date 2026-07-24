# Hybrid K=2 Search Bottleneck Diagnostic Audit v1

This development-only diagnostic reuses, without regenerating, the two frozen
K=2 pair candidates already produced by the Hybrid Stage 1 Incumbent
Bootstrap Audit v1 for `normal_dev_10`. Both pairs move the same two logical
sections (`AP_3D_ART_DESIGN_01`, `SOCIAL_JUSTICE_01`); only the hinted
destination for `SOCIAL_JUSTICE_01` differs (`P2` vs `P4`). It does not rerun
the full 312-section cap-2 portfolio search, does not mine new pair
candidates, and does not run K=1 or K=3.

## Purpose

The bootstrap's two K=2 searches both returned `UNKNOWN` with no incumbent
after their 180-second budgets. That result alone does not say *why* the
search did not resolve: it could be that global section-pair selection over
the full frozen domain is too large a search space, that the specific
destination choice within an otherwise-correct section pair is the hard part,
that a fixed edited plan has no feasible student assignment at all, or that
the assignment hint and Hamming objective simply did not guide search well
enough. This diagnostic separates those four possibilities with three
placement-fixing ablations run against each of the two already-frozen pairs.

## Diagnostic A: exact destinations, no hint, feasibility only

For a pair, both target sections are physically fixed to their hinted
destination placements in an edited copy of the input (no CP-SAT placement
decision variables at all). The full unchanged production hard-constraint
model (`_build_full_feasibility_cp_sat_model`, the same ENRICHMENT-scope
model production allocation uses) is solved with no assignment hint, no
objective, and a 60-second budget, stopping at the first solution. `INFEASIBLE`
here is a proof that this *exact* pair-plus-exact-destinations combination has
no valid student assignment — it says nothing about other destinations for
the same two sections. `UNKNOWN` with no incumbent means the exact-plan
feasibility question itself was not resolved in budget, so Diagnostic B runs
next.

## Diagnostic B: exact destinations, coherent hint + Hamming

Only runs when Diagnostic A returns `UNKNOWN` with no incumbent. Same exact
edited plan, same production hard model and feasible region as Diagnostic A;
the only difference is a freshly regenerated edited-plan Constrained First
assignment hint and an unweighted Hamming-to-that-hint objective, still with a
60-second budget. Hints and the Hamming objective are search guidance only —
they never restrict CP-SAT's feasible region. If B finds a solution where A
could not, that is evidence the hint mattered for this exact plan; if B is
also `INFEASIBLE`, that is the same exact-plan infeasibility proof as A would
have given, just reached via a different search path.

## Diagnostic C: fixed section IDs, destinations free

Only runs when neither A nor B has found an incumbent (independent of whether
they returned `INFEASIBLE` or `UNKNOWN`). This is the one ablation that keeps
the full hybrid joint model (`build_joint_model`, the same model the
bootstrap's K=1/K=2 searches used, with the complete 312-section/841-option
placement domain). It deliberately restricts the search: the pair's two named
logical sections have their `section_changed` indicator fixed to `1` (they
must move), and every other of the 310 editable sections has its
`section_changed` indicator fixed to `0` (must stay at its original
placement, which the model's own `changed + placement_choice[original] == 1`
wiring already enforces without any additional placement fixing). Neither
target section's own destination options are pruned — the solver remains free
to choose any of their non-original placements. The same frozen pair
placement hint, a fresh edited-plan Constrained First assignment hint, and an
unweighted Hamming objective are applied, with a 120-second budget. This
placement fixing is Diagnostic C's sole deliberate feasibility restriction; it
does not change production hard-policy semantics, prune destination options,
or touch any other section's domain.

**Diagnostic A/B versus Diagnostic C is the exact-destination-versus-
fixed-section-ID distinction that matters for interpreting this artifact.** A
and B ask "does this literal edited plan work?"; C asks "does *some* plan that
moves exactly these two sections work, even if not the hinted destinations?"
An `INFEASIBLE` from A/B does not imply anything about C, and a feasible C
result after an infeasible or unresolved A/B is evidence — not proof — that
the destination choice, not the section-pair choice, was the harder part of
the original K=2 search.

## Run budget and stopping rules

At most six new diagnostic solver runs total across both pairs (A, B, C for
each of two pairs), capped at two runs per diagnostic type. The protocol runs
Pair 1's A → (B if A is `UNKNOWN`) → (C if no incumbent yet) in that fixed
order, then Pair 2 only if Pair 1 finished with no incumbent and no
correctness failure. Any run that finds a valid, fully-validated incumbent
stops all remaining diagnostic runs immediately; the protocol does not keep
searching once a witness is found. No `sum(section_changed) <= K` full cap-2
search, no K=1, and no K=3 model is ever built by this module.

## Evidence discipline

`INFEASIBLE` from Diagnostic A or B is scoped strictly to the one exact plan
tested; it is never generalized to "this section pair cannot be repaired."
`INFEASIBLE` from Diagnostic C is scoped strictly to the one fixed section-ID
pair tested with its full destination domain; it excludes only that pair, not
other K=2 pairs outside this frozen portfolio of two. `UNKNOWN` is never
described as `INFEASIBLE`. The previous K=2 solver logs are read with
structured-response fields (from `response_stats.json`), parsed log evidence
(regex facts from `solver.log`, explicitly labeled `evidence_source:
parsed_solver_log`), and inference kept in separate sub-objects; the raw
`solver.log` `CpSolverResponse summary` block prints numeric `objective`/
`best_bound` values even when `status: UNKNOWN` and `solution_count: 0` --
those numbers are recorded for completeness but are never reported as a found
incumbent's objective.

## Claim scope

Any `minimum_changed_sections_within_frozen_placement_domain = 2` claim
requires: the pre-existing K=1 `INFEASIBLE` proof and its source hash, exactly
two changed logical sections, a production-hard-policy-valid assignment,
Final Schedule Policy PASS, zero consistency issues, and both the joint/
fixed-plan witness acceptance and the independent production cold-start
validation passing, with response hashes recorded at every stage. This
diagnostic never claims a globally minimum, unique minimum, or real-world
(teacher/room-feasible) minimum repair. If no incumbent is found by either
pair's exhausted protocol, the overall result is `unresolved_no_incumbent`,
the previously proven K=1 lower bound of 2 stands unchanged, and K=2 itself is
never described as proven infeasible (that would require the full global
cap-2 model to return `INFEASIBLE`, which this diagnostic does not attempt).

## Scope restrictions

This module never runs the control scenario, other normal-development
targets, stress, negative, or holdout scenarios, and never runs Stage 2--4.
It does not change the production section planner, hard-policy semantics, the
Final Schedule Policy, student requests, section counts, capacities, or period
layout, and it performs no candidate pruning. `G12_0105` is excluded, matching
every other slice in this line of work. It does not model teacher, room, or
department constraints, so no result here is a real-world schedulable repair
claim independent of those unmodeled resources.

## Running

Run only against the already-verified bootstrap artifact; the manifest and
this module both fail closed if that artifact's checksum has drifted:

```bash
.venv/bin/python -m src.hybrid_k2_search_bottleneck_diagnostic \
  --output-dir /Users/klu/Projects/fair-course-allocation-artifacts/robustness-v1/hybrid-k2-search-bottleneck-diagnostic-v1
```

The command refuses to overwrite a non-empty, unrecognized output directory
and supports `--resume` for a directory this module already wrote (recognized
by an `aggregate_summary.json` or atomic `checkpoint.json`).

## Result

The completed run produced four solver diagnostics (two per pair; Diagnostic
B never ran because both pairs' Diagnostic A already returned `INFEASIBLE`,
not `UNKNOWN`):

| pair | diagnostic | status | wall time |
|---|---|---|---|
| pair_1 | A | `INFEASIBLE` | ~1.5s |
| pair_1 | C | `INFEASIBLE` | ~3.3s (0 branches; proven during initial copy) |
| pair_2 | A | `INFEASIBLE` | ~1.4s |
| pair_2 | C | `INFEASIBLE` | ~3.6s (0 branches; proven during initial copy) |

Both pairs move the same two logical sections
(`AP_3D_ART_DESIGN_01`, `SOCIAL_JUSTICE_01`). Diagnostic A shows each exact
hinted destination plan is infeasible for the full production model. Diagnostic
C then shows that even with `SOCIAL_JUSTICE_01`'s and
`AP_3D_ART_DESIGN_01`'s destinations left completely free within their legal
domains -- not fixed to the bootstrap's hinted periods -- and every other
section held at its original placement, there is still no valid student
assignment: CP-SAT proved this during its initial constraint copy, before any
branching. `bottleneck_classification.json` records
`exact_destination_pair_infeasible` and `fixed_section_pair_infeasible` for
both pairs, and the overall result is `diagnostic_portfolio_exhausted_no_incumbent`
/ `unresolved_no_incumbent` -- no incumbent was found, so
`hybrid_fixed_witness_acceptance.json` and `production_validation.json`
correctly recorded `not_run`.

This is real evidence, scoped exactly as designed: this specific
bootstrap-mined section pair is not repairable by any 2-section move within
the frozen domain, regardless of destination. It does **not** mean K=2 is
infeasible in general -- only these two logical sections, as a pair, are
excluded; the frozen domain contains 312 editable sections and this
diagnostic tested exactly one section-ID pair (in two hint variants). The
previously proven K=1 lower bound of 2 stands unchanged, and no
`minimum_changed_sections_within_frozen_placement_domain` claim is made.

## Execution history correction

The accepted final artifact's `provenance.json`/`aggregate_summary.json`
originally reported four diagnostic solver runs (2x Diagnostic A + 2x
Diagnostic C) with no indication that this was one of two executed batches.
The real history: a first batch of four runs completed successfully with the
same four `INFEASIBLE` statuses recorded above, but was superseded and rerun
after a `provenance.json` finalization bug was found (a stale, pre-search
all-zero snapshot was never overwritten with the real end-of-run counters);
the original artifact directory was deleted before this correction, so its
per-run response hashes, exact runtimes, and pair/diagnostic-level mapping
are recorded as unavailable rather than reconstructed. Total actual solver
invocations across both batches: **8** (4x Diagnostic A, 0x Diagnostic B, 4x
Diagnostic C). The accepted final artifact still reflects exactly 4 runs,
consistent with this protocol's per-run-type cap of two; the "at most six new
solver runs" budget described above applies to one complete execution of the
protocol, not to the cumulative count across a bug-fix rerun. No portfolio,
model restriction, seed, or budget changed between batches, and neither batch
produced an incumbent, so the rerun did not introduce incumbent-selection
bias -- see `execution_history_correction.json` in the artifact (applied via
`--apply-execution-history-correction`, a reporting-only code path that never
builds or solves a CP-SAT model) for the full, machine-readable record.

Separately, because both frozen pairs move the identical section-ID pair
(`AP_3D_ART_DESIGN_01`, `SOCIAL_JUSTICE_01`), this diagnostic tested exactly
**one** unique fixed section-ID pair, not two: two exact-destination plans and
two Diagnostic C search-guidance configurations were evaluated against that
one pair. Either Diagnostic C `INFEASIBLE` result alone already proves this
pair infeasible across its full frozen destination domain; the second is a
supporting reproduction under a different hint/Hamming configuration, not
independent evidence that a second, distinct section pair is infeasible.
`frozen_pair_portfolio.json`'s `comparison.pair_semantics` records this
count explicitly.
