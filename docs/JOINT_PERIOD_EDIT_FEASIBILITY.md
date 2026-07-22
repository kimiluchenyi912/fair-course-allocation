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
