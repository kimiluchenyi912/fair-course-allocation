# Joint Stage 1 Hybrid Occupancy Model-Size Reduction Audit v1

The existing `full_optional_intervals` builder remains the default. The audit
adds `hybrid_sparse_linear_occupancy` with the same assignment keys, candidate
membership, placement variables, section identities, and production policy
helpers.

For each logical section and period, `q` is the exact sum of placement choices
that occupy that period. Constant zero and one channels are omitted. A
variable channel uses `w = x AND q` with all three linearization inequalities.
Fixed sections and constant occupancy contribute `x` directly. The conflict
constraint is the exact student-period sum, including linked Government /
Economics and Math 2/3 occupancy.

Evidence is layered: structural invariance, exhaustive small-fixture
feasible-set comparison, and fresh known-witness acceptance on
`normal_dev_reference_2026`. Acceptance is a correctness check only, not the
sole proof of global equivalence. The frozen placement domain is not pruned
and the 250,000,000-byte cost gate is not raised.

The `normal_dev_10` hybrid model is built only for size accounting. Stage 1
Solve, production repair validation, teacher/room constraints, minimum-repair
claims, other normal scenarios, stress, negative, and holdout runs are out of
scope. An exact model that still exceeds the gate is reported as such.

## Measurement history

Commit `4a97c02` recorded the earlier Stage 1 cost-gate stop using text-format
proto bytes under a serialized-binary label. Stage 1 was not run, so no solver
result was contaminated. This audit preserves that raw artifact as historical
provenance and corrects the measurement with binary `.pb` export plus a parsed
protobuf `SerializeToString` round trip. The previous classification was
caused by text-format proto bytes being reported as serialized binary bytes;
the corrected binary measurements show both baseline and hybrid pass the
frozen 250,000,000-byte gate.
