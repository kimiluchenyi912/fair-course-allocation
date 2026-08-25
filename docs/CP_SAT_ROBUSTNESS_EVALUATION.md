# CP-SAT Robustness Evaluation Phase C

Phase C is a frozen, development-only evaluation of the production fixed-section
CP-SAT pipeline. It consumes the persisted Phase A normal-year and Phase B stress
artifacts; it does not regenerate students, re-plan sections, rerun Greedy, or
apply stress transforms again.

The frozen evaluation manifest is
`data/scenarios/cp_sat_development_evaluation_v1.json`. It contains 12 normal
development scenarios, 12 ordinary stress development scenarios, and 3
structural negative controls. It contains no holdout IDs. Normal and stress
holdouts remain frozen and unrun.

## Frozen Solver Configuration

- solver seed: `20260630`
- workers: `1`
- bootstrap budget: `30` seconds
- per-stage budget: `30` seconds
- total budget: `300` seconds
- external persisted seed: disabled

The runner calls `run_fair_cp_sat_solver` directly with this configuration. It
does not duplicate model construction or turn Greedy results into constraints.
The production stage order and objective order are recorded in the manifest and
checked before execution.

`FEASIBLE` means that an incumbent was found but optimality was not proven;
`OPTIMAL` means the solver proved the stage result; `UNKNOWN` means that no
usable incumbent was returned within the available budget. `UNKNOWN` is not
`INFEASIBLE`, and `FEASIBLE` is not `OPTIMAL`.

For ordinary scenarios, an assignment is exported only when it comes from the
current CP-SAT response and passes the final schedule policy and consistency
checks. `UNKNOWN` and `INFEASIBLE` retain null assignment metrics. Structural
negative controls validate their certificate before solving; a feasible or
publishable negative result is a critical correctness conflict.

## Commands

Inspect selected development scenarios without solving:

```bash
python -m src.cp_sat_robustness_runner --dry-run --group all --max-scenarios 2
```

Verify source artifact hashes and canonical fingerprints without solving:

```bash
python -m src.cp_sat_robustness_runner --verify-only --group all
```

Run the frozen development evaluation into the external artifact directory:

```bash
python -m src.cp_sat_robustness_runner \
  --group all \
  --output-dir $FCA_ARTIFACT_ROOT/robustness-v1/cp-sat-development-v1
```

The output includes scenario summaries, stage traces, normal/stress/negative
result tables, grade subgroup metrics, CP-SAT versus existing Greedy pairs,
normal-versus-stress pairs, failure records, a holdout-readiness assessment,
and a SHA-256 manifest. A non-empty output directory is never overwritten;
`--resume` requires the same evaluation manifest, source commit, solver
configuration, and scenario selection.

## Status Semantics Audit

Existing raw Phase C results can be audited without invoking CP-SAT:

```bash
python -m src.cp_sat_robustness_runner \
  --audit-source-dir $FCA_ARTIFACT_ROOT/robustness-v1/cp-sat-development-v1 \
  --audit-output-dir $FCA_ARTIFACT_ROOT/robustness-v1/cp-sat-development-v1-audited
```

The audit preserves raw terminal statuses and stage traces, then adds an
explicit outcome and proof scope. Only an `INFEASIBLE` result from the
`full_model_feasibility_incumbent` stage is a full hard-model infeasibility
proof. An `INFEASIBLE` bootstrap or core stage is not such a proof; a later
stage with fixed prior objectives is reported as
`LEXICOGRAPHIC_STAGE_INFEASIBLE`. Structural negative certificates are kept
separate from `solver_global_infeasibility_proven`.

Holdout readiness is false when a majority of normal development scenarios have
no publishable assignment. In particular, zero normal assignments cannot pass
the readiness gate merely because there are no policy violations to count.
The audited artifact records the assignment rate, blocking reasons, source
artifact hashes, and `no_new_solver_runs=true`. It contains summaries only and
does not copy or rewrite raw solver responses.

## Interpretation Boundary

Development results are measurement and debugging evidence, not a final test
set or a generalization claim. Ordinary stress scenarios have expected
feasibility `unknown`; stress transforms do not represent every real-world
school disruption. The result of a capacity-only or structural certificate is
not a global optimum certificate for ordinary scenarios. Holdout results must
remain unseen and unrun until a separately approved one-time evaluation.
