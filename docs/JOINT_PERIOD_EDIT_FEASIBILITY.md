# Joint Period-Edit Feasibility Pilot v1

This Phase A slice is a single-scenario development diagnostic. It runs only
the feasible control `normal_dev_reference_2026` and pilot target
`normal_dev_10`. The other six period-audit targets, stress scenarios,
negative scenarios, and holdout scenarios are not run.

The independent joint model chooses placements from a frozen universe taken
from the statically promising candidates in the period-placement preview and
assigns students in the same logical-request universe. It does not add,
delete, resize, or otherwise plan sections, change requests, or relax any
production hard policy. The authoritative fine-core student is `G12_0536`;
`G12_0105` is excluded from evidence, candidate generation, and objectives.

Normal sections occupy one period, Math 2/3 Honors Accelerated remains one
logical section occupying two consecutive periods, and a Government/Economics
linked block remains one logical section with two physical semester rows.
The joint model uses placement-choice variables and assignment-dependent
optional intervals with per-student `NoOverlap`; capacity, candidate validity,
target load, duplicate identity, fallback semantics, fairness hard policies,
minimum-five, and maximum-gap helpers retain production semantics.

The fixed-original control check must be feasible. The target zero-edit check
must remain `INFEASIBLE` or stop on `UNKNOWN`; a feasible mismatch is a
correctness failure. A cost gate is evaluated before joint repair search.

The lexicographic diagnostic stages are changed logical sections, actual
assigned edges in changed sections, placement displacement, and an internal
assignment-stability stage. A `FEASIBLE` stage is not `OPTIMAL`; an
`UNKNOWN` stage does not support a minimum claim. Any minimum wording is
bounded to the frozen admissible placement domain and requires a zero-edit
proof, an optimal Stage 1, and independent production validation. No global
minimum or real-world schedule claim is made.

A joint witness is diagnostic only. When one exists, an independent copy of
the edited plan is passed to the unchanged production CP-SAT model with an
internal constrained-first hint and no external persisted seed. Only a
successful production response with a response hash, assignment, Final
Schedule Policy pass, and zero consistency issues is an independently
validated period repair.

The pilot does not model teacher, room, department, or master-schedule
realism. It writes artifacts outside the repository and keeps atomic stage
checkpoints; those artifacts are not repository inputs.

## Control-equivalence performance audit

`src.joint_model_control_performance_audit` is a separate, control-only audit.
It compares the production-native model, a fixed-placement joint model using
native period conflicts, and the fixed-placement optional-interval model. A
known stable assignment is first fixed in fresh models and checked for exact
acceptance, policy pass, and consistency. This witness is not a performance
hint. Performance runs are cold-start runs with only the internal Constrained
First hint and frozen solver settings.

The default command does not run the expensive performance variants. The
explicit `--run-performance` mode writes raw OR-Tools logs and structured
statistics, but still runs only the reference control. It never runs
`normal_dev_10`, stress, negative, or holdout scenarios. A result of `UNKNOWN`
does not prove infeasibility, and any diagnosis remains a performance
observation rather than a repair or correctness claim.

The control-equivalence correctness gate is structural invariance,
known-witness acceptance, hint audit, and source-hash verification. It does
not require the B/C feasibility-only probes to find an incumbent within their
frozen time budget. The performance runner has one hint owner: its reporting
audit is read-only, each run uses a fresh model copy, and duplicate or
conflicting hint indices fail closed before `Solve`. A `MODEL_INVALID` attempt
rejected before search is retained in attempt provenance but excluded from
performance aggregation; it is not reported as solver performance failure.

If a legacy checkpoint cannot bind a response, solver log, and validation to a
single run because per-variant files were reused, that checkpoint is marked
provenance-unverified and excluded. A replacement must use the same seed,
worker count, objective, and budget; the raw checkpoint is not rewritten.
