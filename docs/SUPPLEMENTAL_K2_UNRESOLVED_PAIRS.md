# Supplemental K=2 Unresolved-Pair Protocol v1

## Purpose

This protocol creates a separate experiment for the two formal K=2 pairs that
do not have a verified scoped feasibility conclusion:

1. Formal Order 1, `AP_3D_ART_DESIGN_01 + INTERMEDIATE_ACTING_01`, whose
   original classification is `artifact_failure`.
2. Formal Order 2, `AP_3D_ART_DESIGN_01 + FOOTBALL_01`, whose original result
   is `UNKNOWN` without an incumbent.

The supplemental experiment is not artifact recovery and is not a rerun of
the original formal batch. The original checkpoint, classifications, response
evidence, and invocation accounting remain permanent and read-only.

Formal Orders 3--12 are not supplemental unresolved pairs. They are simply
not yet run and require a separate future formal execution decision.

## Frozen source and model semantics

The manifest binds the checksummed formal artifact at:

```text
/Users/klu/Projects/fair-course-allocation-artifacts/robustness-v1/
formal-remaining-k2-batch-v1
```

It verifies the artifact SHA256SUMS hash, formal manifest hash, ordering hash,
pair-row hash, exact pair states, and Order 2 response reference. Source drift
fails closed.

Any future approved supplemental run must use the same production hard model,
complete frozen non-original destination domains, both selected sections
forced changed, and every other editable section fixed at its original
placement. It uses no placement hint, assignment hint, objective, candidate
pruning, Run B, or automatic retry. The seed is `20260630`, workers are `1`,
and the solver stops after the first solution.

## Budget gate

No extended supplemental budget has been approved. The checked-in manifest
therefore records:

```json
{
  "approval_status": "pending_explicit_approval",
  "approved": false,
  "per_pair_time_limit_seconds": null,
  "approval_reference": null
}
```

`--execute` fails before creating an output directory or calling a solver.
Execution requires a separately reviewed manifest revision with an explicit
approval reference and a positive integer per-pair time limit.

## Evidence and proof boundary

Supplemental results use their own checkpoint, artifact root, classifications,
provenance, and namespaced response hash. The original evidence is referenced,
never overwritten.

Future classifications are:

- `supplemental_fixed_pair_infeasible`;
- `supplemental_incumbent_pending_validation`;
- `supplemental_unresolved_unknown_no_incumbent`;
- `supplemental_model_invalid`;
- `supplemental_artifact_failure`.

A verified supplemental `INFEASIBLE` result could later be considered by a
separate finalizer as additional independent scoped evidence. It would not be
a recovered original result. `UNKNOWN` leaves the proof gap open.

This runner never applies supplemental evidence to the global proof. Global
K=2 remains unresolved, the proven lower bound remains 2, and there is no exact
minimum claim.

## Solver-free dry-run

```bash
python -m src.supplemental_k2_unresolved_pairs \
  --manifest data/scenarios/supplemental_k2_unresolved_pairs_v1.json \
  --output-dir /tmp/supplemental-k2-unresolved-pairs-dryrun \
  --dry-run
```

The dry-run verifies the formal source and writes only a two-pair plan,
checkpoint, source audit, summary, provenance, failures file, and checksums. It
does not write a solver log, response artifact, or model protobuf.
