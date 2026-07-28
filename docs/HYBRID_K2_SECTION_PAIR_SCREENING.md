# Hybrid K=2 Section-Pair Static Screening v1

This development-only slice screens the full frozen `normal_dev_10` K=2
section-pair universe before any fixed-pair solver runs. It uses the frozen
312 editable sections and 841 placement options from the prior hybrid
artifacts, and it evaluates only the authoritative `G12_0536` student-local
necessary condition.

## Command

```bash
.venv/bin/python -m src.hybrid_k2_section_pair_screening \
  --manifest data/scenarios/hybrid_k2_section_pair_screening_v1.json \
  --output-dir /Users/klu/Projects/fair-course-allocation-artifacts/robustness-v1/hybrid-k2-section-pair-screening-v1 \
  --screening-only
```

`--screening-only` writes the static artifact and deliberately skips fixed-pair
Run A, fixed-pair Run B, production fixed-witness acceptance, and independent
production validation.

## Scope

The screen enumerates every unordered pair in the frozen section domain:

- editable sections: `312`
- placement options: `841`
- unordered section pairs: `48,516`
- placement combinations evaluated: `139,415`

For each pair, every full non-original destination combination is checked
against the G12_0536 local request/period/logical-gap rules. The check is a
necessary condition only. A pair that fails this screen cannot repair the
authoritative student under this frozen local model. A pair that survives has
not been proven globally feasible and has not been proven production feasible.

## Accepted Static Result

The accepted formal artifact matches the exploratory dry run on the frozen
counts:

- necessary-condition failures: `47,278`
- core-screen survivors: `1,237`
- previously proven infeasible pairs: `1`
- screening errors: `0`
- unclassified pairs: `0`
- class-count closure: `48,516`
- selected portfolio pairs: `6`
- portfolio hash:
  `ef83de1d2dfecaa6f55b8d074156466d96f73c5334be61d2aba856819445fd67`

The selected portfolio has six unique section-ID pairs and six unique course
pairs. The same-course-pair cap was relaxed to 2, but the final result still
contains six unique course pairs. The section-participation cap was relaxed to
3 and that relaxation was used.

## Safety Counters

The selected six-pair portfolio has now been exhausted with fixed-pair Run A.
Run A forces each selected pair's two section IDs to change, keeps every other
editable section fixed to its original placement, preserves each selected
section's full frozen non-original destination domain, and uses no placement
hint, assignment hint, candidate pruning, or objective.

- fixed-pair Run A: `6`
- fixed-pair Run B: `0`
- total solver invocations: `6`
- production fixed-witness acceptance: `0`
- production validation: `0`
- global K2/K1/K3: `0`
- other normal, stress, negative, holdout: `0`

The artifact provenance records one exploratory dry run, one accepted formal
static screening run, two total static screening executions, and six total
solver invocations.

## Fixed-Pair Run A Results

| pair | section IDs | Run A status | incumbent | response-hash evidence | Run B needed | scoped conclusion |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `AP_3D_ART_DESIGN_01` + `CREATIVE_WRITING_01` | `INFEASIBLE` | no | recovery-only log evidence; response hash unavailable | no | fixed pair infeasible across full frozen destination domain |
| 2 | `AP_JAPANESE_LANG_01` + `SOCIAL_JUSTICE_01` | `INFEASIBLE` | no | verified persisted response hash | no | fixed pair infeasible across full frozen destination domain |
| 3 | `AP_JAPANESE_LANG_01` + `CREATIVE_WRITING_01` | `INFEASIBLE` | no | verified persisted response hash | no | fixed pair infeasible across full frozen destination domain |
| 4 | `AP_3D_ART_DESIGN_01` + `AP_JAPANESE_LANG_01` | `INFEASIBLE` | no | verified persisted response hash | no | fixed pair infeasible across full frozen destination domain |
| 5 | `CREATIVE_WRITING_01` + `SOCIAL_JUSTICE_01` | `INFEASIBLE` | no | verified persisted response hash | no | fixed pair infeasible across full frozen destination domain |
| 6 | `AP_3D_ART_DESIGN_01` + `ROCK_N_ROLL_HISTORY_01` | `INFEASIBLE` | no | verified persisted response hash | no | fixed pair infeasible across full frozen destination domain |

The selected portfolio is complete and exhausted with `exhausted_no_incumbent`:
six Run A attempts, six `INFEASIBLE` statuses, zero incumbents, and zero Run B
attempts. Run B was not required because each Run A already proved its fixed
pair infeasible across that pair's full frozen destination domain.

## Interpretation

The 1,237 survivors are only G12_0536 necessary-condition survivors. Six
selected survivors have now been fixed-pair tested, leaving 1,231 untested
static survivors. Together with the one previously proven infeasible pair, the
artifact has seven specifically excluded unique section-ID pairs.

Among the four sections `AP_3D_ART_DESIGN_01`, `AP_JAPANESE_LANG_01`,
`CREATIVE_WRITING_01`, and `SOCIAL_JUSTICE_01`, all six two-section
combinations have been excluded: five by selected Run A and one by prior
evidence. This is a local fact about those four sections only. It is not a
global K2 infeasibility proof, not a repair witness, not global-feasibility
evidence, and not a publication-ready assignment. Global K2 remains unresolved.
The previous K=1 lower bound of 2 stands unchanged, and no exact minimum claim
is made. The next phase is not decided here and should not automatically
continue solver runs.

## Read-only expansion design

The separate
`docs/REMAINING_K2_SURVIVOR_EXPANSION_AUDIT.md` reviews the six persisted Run A
models and designs a stronger proof-backed screen plus a bounded future batch
protocol. It runs no solver and does not change this artifact. Its cached-row
projection estimates that the necessary condition
`pair_section_ids ∩ {CHINESE4_01, FOOTBALL_01, INTERMEDIATE_ACTING_01} ≠ ∅`
would reduce the 1,231 untested survivors to 12. This exploratory, informal
projection is not a formal screening artifact result; the formal count remains
1,231 until a separately approved implementation persists a reviewed result.
Global K2 remains unresolved and no exact minimum claim is made.
