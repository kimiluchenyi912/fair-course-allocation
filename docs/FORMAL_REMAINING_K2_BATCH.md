# Formal Remaining 12-Pair K=2 Batch Runner v1

## Scope

This slice implements and dry-runs a fail-closed Run A batch protocol for the
12 section pairs retained by the accepted All-Student K=2 Blocker Safe Screen.
It does not execute CP-SAT, test a pair, create the formal batch artifact, or
produce a feasibility result.

The 12 pairs are only **not excluded by current safe necessary conditions**.
They are not known feasible. Global K=2 remains unresolved, the proven lower
bound remains 2, and the exact minimum claim remains none.

## Authoritative input

The runner reads the pair universe and order only from:

```text
/Users/klu/Projects/fair-course-allocation-artifacts/robustness-v1/
all-student-k2-blocker-safe-screen-v1
```

It verifies all 10 checksum entries and freezes:

- artifact files: `11`
- `SHA256SUMS.txt` hash:
  `9205ad79ff3248ac578362ca113cf0e1110a4bd677859b96c73eb5122416db95`
- safe-screen result hash:
  `244c39b79d3b61c379d386aec203fb2bd73395f1af44ebc575c2f9d500169af5`
- blocker proof hash:
  `853223911427d9decccff42496022f0030c7cf0eba9e4fcc1184e6eec68f31b1`
- pair count: `12`
- formal ordering hash:
  `f1246ec8582d6925e26fcc9ee53583a7ac275d6063781b2f59865af788573927`

The runner does not accept a CLI pair list and does not reconstruct the list
from the original 1,237 survivors or the six-pair portfolio artifact.

## Frozen plan

The 12 ordered pairs are partitioned deterministically into three batches of
four. Future real runs require the explicit `--execute` mode. Omitting both
`--dry-run` and `--execute` fails before model construction.

Every future pair run is one production Run A with:

- a fresh production model and solver;
- both selected sections forced changed;
- all other editable sections fixed at their original placements;
- both selected sections' complete frozen non-original destination domains;
- no candidate pruning, placement hint, assignment hint, or objective;
- seed `20260630`, one worker, and a 75-second limit;
- feasibility only and stop after first solution;
- no Run B and no automatic retry.

The global invocation budget is 12. `--max-new-solver-runs` limits only new
calls made by the current invocation and cannot raise that global limit.

## Checkpoint and stopping semantics

The checkpoint binds the manifest hash, source checksum hash, ordering hash,
pair-row hash, pair config fingerprints, response hashes, total invocations,
and completed batches. Completed pairs and batches are skipped on resume.

`running` is never treated as completed. An interrupted `running` state or an
`artifact_failure` requires manual review and cannot be automatically rerun.
Manifest, source, ordering, config, or response drift stops fail-closed.

Pair classifications are limited to:

- `fixed_pair_infeasible`: scoped Run A `INFEASIBLE`; continue;
- `incumbent_pending_validation`: any incumbent; stop the batch;
- `unresolved_unknown_no_incumbent`: `UNKNOWN` without incumbent; stop;
- `model_invalid`: stop;
- `artifact_failure`: stop;
- `planned_not_run`: no solver call occurred.

Partial runs always retain `global_k2_status = unresolved`,
`proven_lower_bound = 2`, and `exact_minimum_claim = null`. This runner does
not publish global K=2 infeasibility even if all 12 scoped runs later return
`INFEASIBLE`; that decision belongs to an independent finalizer.

## Compact evidence

An ordinary future run stores pair/order identity, source and manifest hashes,
complete destination domains, placement-combination count, model and config
fingerprints, solver configuration, response status/statistics/hash, raw log,
validation, run result, checkpoint, and provenance.

Full `model.pb` persistence is disabled by default. It is permitted only for
`MODEL_INVALID`, artifact/debug recovery, configuration or fingerprint
mismatch, unexpected status, or an explicit reproducibility escalation. The
evidence record always states `full_model_saved` and its reason.

## Dry-run

The accepted implementation dry-run writes only a temporary plan artifact:

```bash
python -m src.formal_remaining_k2_batch \
  --manifest data/scenarios/formal_remaining_k2_batch_v1.json \
  --output-dir /tmp/formal-remaining-k2-batch-dryrun-<timestamp> \
  --dry-run
```

It verifies the source, writes 12 planned records, three four-pair batches,
checkpoint and compact-evidence plans, and records zero solver invocations. It
does not write `solver.log`, `CpSolverResponse`, or `model.pb` and does not
touch the future formal artifact directory.
