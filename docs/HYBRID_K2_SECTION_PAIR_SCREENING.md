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

All solver and downstream validation counters are zero in this slice:

- fixed-pair Run A: `0`
- fixed-pair Run B: `0`
- total solver invocations: `0`
- production fixed-witness acceptance: `0`
- production validation: `0`
- global K2/K1/K3: `0`
- other normal, stress, negative, holdout: `0`

The artifact provenance records one exploratory dry run, one accepted formal
static screening run, two total static screening executions, and zero total
solver invocations.

## Interpretation

The 1,237 survivors are only G12_0536 necessary-condition survivors. They are
not repair witnesses, not global-feasibility evidence, and not publication-ready
assignments. Global K2 remains unresolved. The previous K=1 lower bound of 2
stands unchanged, and no exact minimum claim is made.
