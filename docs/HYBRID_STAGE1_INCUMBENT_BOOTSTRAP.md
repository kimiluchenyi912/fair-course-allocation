# Hybrid Stage 1 Incumbent Bootstrap Audit v1

This development-only audit searches the frozen `normal_dev_10` joint
period-edit domain for a usable incumbent. It uses the authoritative core
student `G12_0536`; `G12_0105` remains excluded. The placement domain remains
the complete 312-section/841-option audited domain, with no result-driven
pruning.

The search uses at most three deterministic K=1 portfolios and, when the K=1
protocol does not stop, at most two K=2 portfolios. Each portfolio first
creates a fresh edited-plan constrained-first hint. The joint model then adds
only `sum(section_changed) <= K` and minimizes unweighted assignment Hamming
distance to that hint. The hints and Hamming objective do not restrict the feasible region. The explicit change cap is the sole deliberate feasibility
restriction in each bounded search model: it excludes solutions with more than
K changed sections without pruning the frozen placement options, candidate
edges, or production hard-policy semantics.

`INFEASIBLE` is a proof only for the corresponding frozen full-domain cap;
`UNKNOWN` is never treated as infeasible. A solver incumbent is still only a
candidate witness. It must pass joint replay, fixed-witness production
acceptance, and one independent production cold-start validation before a
validated repair claim is recorded. Minimum wording additionally requires a
validated K=1 or a validated K=2 result after a K=1 infeasibility proof.

The runner is deliberately limited to this target. It does not run control,
other normal, stress, negative, holdout, or Stage 2--4 scenarios. It does not
change the production planner, section count, capacities, period layout,
requests, or hard policies, and it does not model teacher, room, or department
constraints. No external persisted seed is used.

Each completed candidate is written to an atomic `checkpoint.json` and
`search_runs.csv`. `--resume` requires that checkpoint, skips recorded run
IDs, and never reruns a completed candidate; a completed aggregate is returned
without search. Artifacts are external to the repository.

Run only when the frozen experiment has been approved:

```bash
.venv/bin/python -m src.hybrid_stage1_incumbent_bootstrap \
  --output-dir /Users/klu/Projects/fair-course-allocation-artifacts/robustness-v1/hybrid-stage1-incumbent-bootstrap-v1
```

This is a bounded diagnostic, not a section planner, allocation result, global
minimum proof, or generalization evaluation.

The formal overall result is recorded as
`result_classification=unresolved_no_incumbent`. The K=1 result is a scoped
cap-1 infeasibility result, while both K=2 searches are `UNKNOWN` without an
incumbent; therefore this is not a no-repair proof.

The follow-up Hybrid K=2 Search Bottleneck Diagnostic Audit v1 (see
`docs/HYBRID_K2_SEARCH_BOTTLENECK_DIAGNOSTIC.md`) reuses these same two frozen
K=2 pair candidates, unmodified, to distinguish exact-destination
infeasibility from fixed-section-ID infeasibility for each pair. It does not
rerun this bootstrap's K=1/K=2 portfolio search and does not mine new
candidates.
