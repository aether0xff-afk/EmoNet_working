# EmoNet v5.7 Result — Residual-Driven Fast Dynamics

검증일: 2026-08-09 KST

## Version boundary

- v5.6.1 diagnostic: frozen
- v5.7 residual-driven fast dynamics: this result
- branch: `feature/v5.7-residual-fast-dynamics`
- PR: #16
- workflow run: `31291957451`
- artifact: `v5.7-residual-fast-dynamics-benchmark`
- artifact id: `9031698453`

The recurrent topology/equations were unchanged from v5.0. Slow EMA memory was unchanged from v5.6. The only fast-channel change was:

```text
old: fast <- embedding_t
new: fast <- embedding_t - slow_(t-1)
```

## Validation

Before the benchmark:

- v5.6 state/reset invariants passed;
- v5.6.1 identity-disjoint task construction passed;
- v5.7 confirmed identical recurrent/input weights to raw v5.0 for the same seed;
- residual was verified to use the previous slow state before slow update;
- slow memory was verified identical to v5.6;
- fast reset was verified not to alter slow context.

## Results

Mean across five recurrent seeds:

### Semantic diagnostic

```text
slow semantic       = 0.800
residual-fast semantic = 0.660
raw-fast semantic      = 0.595
```

Slow semantic retention remains intact, as required.

### Identity-disjoint structural temporal task

| Representation | Accuracy |
| --- | ---: |
| residual-driven recurrent fast | 0.5625 |
| old raw-input recurrent fast | 0.5650 |
| slow EMA | 0.5125 |
| residual-fast after fast reset | 0.5200 |
| direct residual-change baseline | **1.000** |
| relational task-validity baseline | **1.000** |

Gaps:

```text
residual-fast - raw-fast = -0.0025
residual-fast - slow     = +0.0500
residual-fast - reset    = +0.0425
residual-fast - direct residual = -0.4375
```

Seed structural residual-fast accuracy:

```text
seed 7   = 0.5875
seed 13  = 0.5125
seed 21  = 0.6125
seed 42  = 0.6000
seed 100 = 0.5000
```

## Predeclared gate

```text
residual_fast_structural_at_least_0_70 = false
residual_fast_beats_raw_fast_by_0_10 = false
residual_fast_beats_slow_by_0_12 = false
fast_reset_reduces_structural_by_0_12 = false
relational_validity_at_least_0_95 = true
slow_semantic_remains_at_least_0_78 = true
all_primary_gates = false
```

**v5.7 fails.**

## Important diagnostic observation

The direct label-free residual-change baseline reaches `1.000` on the same identity-disjoint structural task.

Therefore the desired temporal pattern is not absent from the residual signal itself. The gap appears after or at the interface between residual-driven recurrent dynamics and the downstream linear readout.

However, it is not yet valid to conclude that recurrence destroys the information.

The direct residual baseline uses nonlinear quantities such as L2 norms of residual vectors and residual-to-residual changes. A coordinate-wise linear probe on the raw recurrent trace cannot directly compute such rotationally invariant energy features.

Two possibilities remain:

1. **information loss:** the recurrent dynamics truly discard the useful change structure;
2. **nonlinear encoding/readout mismatch:** the structure remains in the trace but is encoded in activation energy/trajectory geometry rather than a globally aligned linear coordinate direction.

## Next version

**v5.7.1 — Fast Trace Geometry Diagnostic** should change no state generator.

On the frozen identity-disjoint task compare:

- raw residual-fast trace + linear probe;
- per-tick trace activation energy (`||h_t||`);
- per-tick trace-change energy (`||h_t - h_(t-1)||`);
- combined energy trajectory;
- the same energy features from old raw-input fast;
- current residual norm;
- full causal residual-change baseline.

All energy features are deterministic, label-free functions of the frozen trace.

If energy-trajectory readout becomes strong while raw linear coordinates remain weak, the fast dynamics contain useful identity-invariant temporal information but require a nonlinear/invariant readout. If energy features also remain weak, the recurrent transform itself is the bottleneck.
