# Joint Period-Edit Stage 1 Pilot v1

This is a development diagnostic for one scenario: `normal_dev_10`. The
placement domain is frozen from the previously reviewed promising candidate
preview: 312 editable logical sections and 841 placement options, including
each original placement. The domain is hash-checked before model construction;
there is no dynamic pruning or result-driven candidate addition.

Stage 1 uses the existing joint optional-interval formulation and minimizes
the number of changed logical sections. It keeps production candidate edges,
capacities, duplicate logical identity, fallback, fairness, minimum-five,
maximum-gap, linked Gov/Econ, and double-period semantics. It does not add,
remove, split, merge, or resize sections.

The run is deliberately narrow: control, other normal targets, Stage 2--4,
stress, negative, and holdout scenarios are not run. It uses one worker,
solver seed `20260630`, no external persisted seed, and a 300-second Stage 1
budget. The constrained-first assignment and original placements are search
hints only. A fresh model is required, its hint must be empty, and duplicate
or conflicting hint indices fail closed.

## Result semantics

- `OPTIMAL` with a positive objective can support
  `minimum_changed_sections_within_frozen_placement_domain`, subject to all
  later gates.
- `FEASIBLE` supports only `best_found_changed_sections`.
- `UNKNOWN` is unresolved; an incumbent observed by a callback is still not a
  proof of minimum.
- `INFEASIBLE` is scoped only to this frozen placement domain.
- An objective of zero conflicts with the known `normal_dev_10` zero-edit
  infeasibility evidence and is recorded as a correctness failure.

A joint witness is internally replayed first. If valid, the unchanged
production model accepts the edited plan with all assignment variables fixed
for 30 seconds, without hints or a Hamming objective. Only a successful exact
acceptance allows one independent production cold-start validation using the
existing constrained-first hint and distance-guided repair. Even then, the
result is not a global or real-world minimum, and no teacher, room, or
department feasibility claim is made.

Artifacts are written outside the repository. The runner refuses to overwrite
a non-empty directory and `--resume` returns a completed checkpoint without
re-running Stage 1.

The follow-up hybrid occupancy audit compares the unchanged optional-interval
encoding with exact sparse q/w occupancy channels. It checks fixture feasible
sets and a known control witness, then only builds the target model. It does
not prune placements, produce repairs, or establish a minimum.

The execution follow-up permits one hybrid Stage 1 invocation for
`normal_dev_10`. `UNKNOWN` without an incumbent is unresolved, and acceptance
and production validation are skipped in that case.
