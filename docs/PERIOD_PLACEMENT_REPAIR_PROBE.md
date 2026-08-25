# Core-Targeted Minimum Period-Placement Repair Probe v1

This is a development-only diagnostic probe. It operates on in-memory copies
of frozen section plans for one feasible control and seven normal development
targets. It does not modify the production section planner, student requests,
capacities, CP-SAT constraints, Final Schedule Policy, or persisted audit
artifacts.

## Evidence boundary

Candidate generation uses only the audited fine sufficient core. Invalid
relaxation witnesses are diagnostic-only: their students, slack values, and
repair values are excluded from authoritative candidate generation. In
particular, `normal_dev_10` uses only `G12_0536`; `G12_0105` is excluded.

The student-level pass is an exact dynamic program over request identity and
occupied periods, with capacities ignored. A promising edit is not a
full-model repair; it only means that the authoritative core student passes
the local primary, minimum-five, and maximum-gap-one tests under the edited
period plan.

## Edit semantics

The frozen universe contains single logical-section moves and compatible
logical-section swaps. P1-P7 are the only modeled legal single periods.
Double-period sections can use only consecutive pairs P1-P2 through P6-P7.
Gov/Econ rows and other multi-row logical sections move atomically; row count,
section identity, course identity, capacity, and semester structure remain
unchanged. A swap reports both operation count and changed logical section
count.

The current repository has no teacher, room, department, or explicit
immutable-section constraints. Those constraints are outside the modeled
scope and are not claimed to be validated by this probe.

The canonical request-section membership is identity-based and remains the
same after a period move. Period-derived candidate views are rebuilt from the
edited logical sections. The artifact records this distinction instead of
reporting a false change to `candidate_index` edge membership.

## Minimum claims

The probe starts from the audited zero-edit infeasibility proof for the frozen
section plan under the current hard model. A
validated one-edit result may therefore be reported as
`minimum_edit_count_within_frozen_admissible_universe = 1`. It is not a global
minimum section-plan repair and is not unique. UNKNOWN candidates prevent a
no-single-edit conclusion. A FEASIBLE response is valid only when it includes
the solver response assignment, passes Final Schedule Policy, and has zero
consistency issues. UNKNOWN is not INFEASIBLE.

## Cost gate and artifacts

Run the candidate preview first:

```bash
.venv/bin/python -m src.period_placement_repair_probe \
  --output-dir $FCA_ARTIFACT_ROOT/robustness-v1/period-placement-repair-probe-v1
```

The preview writes candidate and static-analysis artifacts only and performs
zero solver runs. Formal candidate validation is intentionally blocked until
the per-scenario candidate counts and estimated runtime are reviewed. A
non-empty output directory is never overwritten. No stress, negative, or
holdout scenarios are included in v1.
