# Hybrid Joint Period-Edit Stage 1 Execution v1

This development slice runs only `normal_dev_10`, using the audited
`hybrid_sparse_linear_occupancy` model, the frozen 312-section/841-option
domain, seed `20260630`, one worker, and a 300-second Stage 1 budget. It does
not run control, other normal, stress, negative, holdout, or Stage 2--4
scenarios, and it does not use an external persisted seed.

The only objective is `sum(section_changed)`. Hints are owned by the fresh
execution model: the complete assignment universe receives the internal
Constrained First suggestion and every editable placement receives its
original-placement value. Duplicate or conflicting hint indices fail closed.

`OPTIMAL` proves only a frozen-domain Stage 1 objective. `FEASIBLE` or
`UNKNOWN` with an incumbent is best-found and minimum-unresolved. `UNKNOWN`
without an incumbent produces no witness. A valid joint witness must pass
replay, capacity, period, identity, policy, and consistency checks before the
single fixed-witness production acceptance; only that acceptance permits one
independent production cold-start validation. No result is globally minimum or
a real-world repair claim.
