# All-Student K=2 Blocker Safe Screen v1

## Scope

This formal static pass applies one independently verified necessary condition
to the 1,237 survivors in the accepted Hybrid K=2 Section-Pair Screening
artifact. It does not rerun the original 48,516-pair screen and does not build
or solve a CP-SAT model.

The accepted artifact is:

```text
/Users/klu/Projects/fair-course-allocation-artifacts/robustness-v1/
all-student-k2-blocker-safe-screen-v1
```

It records zero real solver invocations, zero source-screening reruns, and zero
new feasibility results.

## Independently verified proof

The authoritative frozen `normal_dev_10` inputs and current production
hard-policy source establish:

1. G12_0105 exists exactly once and is ordinary
   (`priority_protected=false`).
2. The ordinary primary-unmet upper bound is one.
3. G12_0105 has seven logical primaries.
4. Three of those primaries have exactly one candidate section:
   `CHINESE4 → CHINESE4_01`, `FOOTBALL → FOOTBALL_01`, and
   `INTERMEDIATE_ACTING → INTERMEDIATE_ACTING_01`.
5. All three candidate sections have original placement P1.
6. The production student-period hard constraint permits at most one logical
   course for a student in one period.
7. The fixed-pair restriction changes exactly the two selected sections and
   fixes every nonselected editable section at its original placement.

Let

```text
blocker_section_ids = {
  CHINESE4_01,
  FOOTBALL_01,
  INTERMEDIATE_ACTING_01
}
```

A fixed pair must satisfy:

```text
pair_section_ids ∩ blocker_section_ids ≠ ∅
```

If the pair is disjoint, all three blocker primaries stay at P1. At most one
can be assigned, so at least two are unmet. That contradicts the ordinary
max-one-primary-unmet hard policy. Disjoint pairs are therefore proved
infeasible within the frozen fixed-pair production domain.

The structured proof hash is
`853223911427d9decccff42496022f0030c7cf0eba9e4fcc1184e6eec68f31b1`.
Key frozen input hashes are:

| input | SHA256 |
| --- | --- |
| students.csv | `215294e132dd221dab2e154f44da81f58e6e1a8780c6c51323182bce7f8f8f45` |
| requests.csv | `cf3c680ecad30d386e1b66ac44304193ca16bc229216434c52b207a02e16aeb9` |
| sections.csv | `84f2b8e7ef01000982a00dadb171c6525cf7c6e2dd64633d2b04599760e4428b` |
| canonical input fingerprint | `337512af8bc16fd9972bf7563a5e960556e10b439043ee2b40ebec81bd6e4195` |
| source normal-suite SHA256SUMS | `9e24470e55e85f005b35c347ceaeaa15082b1da84c58af16e5fe75704be73b6d` |
| source static-screen SHA256SUMS | `1c29f43b6abf6a50876c4b3f907b5b1f30ca12eb120d315bc4677341b3aef239` |
| survivor_pairs.csv | `6deeaea054e59faeec4edb60e30b3fd203299243331ff0e5f917e8b82a3fffa0` |

All additional source-code and source-table hashes are frozen in
`input_hashes.json`.

## Formal screening result

Every source survivor is deterministically assigned exactly one class:

- `blocker_safe_excluded`: the pair is disjoint from the blocker set;
- `blocker_screen_survivor`: the pair intersects the blocker set.

| population | input | safe excluded | screen survivor |
| --- | ---: | ---: | ---: |
| all static survivors | 1,237 | 1,225 | 12 |
| previously tested selected survivors | 6 | 6 | 0 |
| previously untested survivors | 1,231 | 1,219 | 12 |

The classification has zero duplicates, zero invalid pairs, zero screening
errors, and complete 1,237-row closure. The classification ordering hash is
`1edea0cd455f179e2576e2163609ccb81e5452f3cef17457f57967b802d0ecd7`;
the result hash is
`244c39b79d3b61c379d386aec203fb2bd73395f1af44ebc575c2f9d500169af5`.

The full pair universe closes without double counting:

```text
47,278 original static exclusions
     + 1 previously proven unique pair
 + 1,225 new all-student blocker-safe exclusions
    + 12 not excluded by current safe necessary conditions
 = 48,516 unique pairs
```

## Frozen remaining pairs

The remaining pairs are ordered by descending core-feasible placement
combinations, ascending affected-student count, descending changed core-period
relationships, ascending displacement, then pair ID.

| order | section pair | course pair | original placements | destination sizes | combinations | affected students |
| ---: | --- | --- | --- | --- | ---: | ---: |
| 1 | AP_3D_ART_DESIGN_01 + INTERMEDIATE_ACTING_01 | AP_3D_ART_DESIGN + INTERMEDIATE_ACTING | P7 + P1 | 5 × 2 | 10 | 36 |
| 2 | AP_3D_ART_DESIGN_01 + FOOTBALL_01 | AP_3D_ART_DESIGN + FOOTBALL | P7 + P1 | 5 × 2 | 10 | 55 |
| 3 | INTERMEDIATE_ACTING_01 + SOCIAL_JUSTICE_01 | INTERMEDIATE_ACTING + SOCIAL_JUSTICE | P1 + P3 | 2 × 5 | 10 | 56 |
| 4 | AP_JAPANESE_LANG_01 + INTERMEDIATE_ACTING_01 | AP_JAPANESE_LANG + INTERMEDIATE_ACTING | P7 + P1 | 5 × 2 | 10 | 57 |
| 5 | FOOTBALL_01 + SOCIAL_JUSTICE_01 | FOOTBALL + SOCIAL_JUSTICE | P1 + P3 | 2 × 5 | 10 | 74 |
| 6 | CREATIVE_WRITING_01 + INTERMEDIATE_ACTING_01 | CREATIVE_WRITING + INTERMEDIATE_ACTING | P3 + P1 | 5 × 2 | 10 | 75 |
| 7 | AP_JAPANESE_LANG_01 + FOOTBALL_01 | AP_JAPANESE_LANG + FOOTBALL | P7 + P1 | 5 × 2 | 10 | 76 |
| 8 | CREATIVE_WRITING_01 + FOOTBALL_01 | CREATIVE_WRITING + FOOTBALL | P3 + P1 | 5 × 2 | 10 | 92 |
| 9 | AP_3D_ART_DESIGN_01 + CHINESE4_01 | AP_3D_ART_DESIGN + CHINESE4 | P7 + P1 | 5 × 1 | 5 | 60 |
| 10 | CHINESE4_01 + SOCIAL_JUSTICE_01 | CHINESE4 + SOCIAL_JUSTICE | P1 + P3 | 1 × 5 | 5 | 79 |
| 11 | AP_JAPANESE_LANG_01 + CHINESE4_01 | AP_JAPANESE_LANG + CHINESE4 | P7 + P1 | 5 × 1 | 5 | 81 |
| 12 | CHINESE4_01 + CREATIVE_WRITING_01 | CHINESE4 + CREATIVE_WRITING | P1 + P3 | 1 × 5 | 5 | 98 |

The formal ordering hash was recomputed from the accepted inputs:

```text
f1246ec8582d6925e26fcc9ee53583a7ac275d6063781b2f59865af788573927
```

It matches the earlier informal projection hash. The equality is a verified
comparison, not reuse of the projected value.

## Claim boundary

The 1,225 excluded static survivors are formally proved infeasible within the
current frozen fixed-pair production domain and hard-policy rules. The 12
remaining pairs are only **not excluded by current safe necessary
conditions**. They are not known feasible, and no 12-pair solver batch has
run.

Global K=2 remains unresolved, the proven lower bound remains 2, and the exact
minimum claim remains none.

The artifact contains 11 files and 10 checksum entries. Its
`SHA256SUMS.txt` hash is
`9205ad79ff3248ac578362ca113cf0e1110a4bd677859b96c73eb5122416db95`.
It contains no solver log, model protobuf, or solver response.

## Formal batch handoff

`docs/FORMAL_REMAINING_K2_BATCH.md` consumes this artifact as the sole source
of the 12-pair universe and order. Its runner verifies this artifact's complete
checksum set, result hash, blocker proof hash, pair count, and ordering hash
before planning three four-pair batches. The current implementation has only
completed a solver-free temporary dry-run; none of the 12 pairs has been
executed and this artifact remains unchanged.
