# EmoNet v5.7.1 Result — Fast Trace Geometry Diagnostic

검증일: 2026-08-09 KST

## Version boundary

- v5.7 residual-driven state generator: frozen and unchanged
- branch: `feature/v5.7.1-fast-trace-geometry-diagnostic`
- PR: #17
- successful workflow run: `31295857065`
- job: `93200718248`
- artifact: `v5.7.1-fast-trace-geometry`
- artifact id: `9032929835`

Two earlier workflow attempts failed before the scientific benchmark because of Python package/test import setup. No benchmark result was produced by those failed attempts. The successful run passed all frozen-state and geometry-invariance tests before running the diagnostic.

## Validation

Before the benchmark:

- frozen v5.7 residual-state tests: `4 passed`;
- v5.7.1 geometry invariants: `3 passed`;
- geometry features were verified invariant to neuron permutation;
- the state generator was unchanged from v5.7;
- train/test token identities in the structural task remained disjoint.

## Mean accuracy across five recurrent seeds

| Readout | Accuracy |
| --- | ---: |
| raw recurrent coordinates | 0.5625 |
| activation-energy trajectory | 0.5425 |
| state-change-energy trajectory | 0.5225 |
| activation + change energy trajectory | 0.5325 |
| population moments | **0.5750** |
| full trace geometry | 0.5425 |
| current residual vector | 0.5125 |
| current residual norm | 0.5375 |
| full causal residual-change sequence | **1.0000** |
| relational task-validity baseline | **1.0000** |
| full geometry after fast reset | 0.5000 |

Best trace-geometry readout:

```text
population_moments = 0.5750
advantage over raw coordinates = +0.0125
advantage over reset = +0.0750
```

The direct residual-change representation exceeds the best recurrent-trace geometry by `+0.4250`.

## Diagnosis

**The v5.7 failure is not explained by a simple linear-coordinate readout mismatch.**

Permutation-invariant nonlinear summaries of the recurrent trace do not recover the structural information that is perfectly available in the causal residual-change sequence.

The useful signal is also not present in the single current residual vector or its norm. Therefore the discriminative information lives in the **temporal pattern of residual changes across events**.

The current fixed random leaky-tanh recurrence does not preserve/expose that identity-invariant temporal structure robustly in the final fast trace.

This remains a mechanistic temporal-memory result only. It is not evidence for emotion semantics.

## Next version

`v5.8` should change the fast dynamics, not the readout.

The cleanest next mechanism is **activity-dependent neural adaptation / fatigue**:

- keep slow EMA semantic memory unchanged;
- keep residual input `embedding_t - slow_(t-1)` unchanged;
- keep fixed seeded topology and no task/emotion labels;
- add a fast adaptation state that suppresses recently active neurons;
- reset recurrent and adaptation state together at episode boundaries;
- compare against v5.7 residual recurrence, adaptation-only/simple baselines, slow EMA, direct residual-change, and reset/ablation controls.

The purpose is to test whether a biologically plausible fast adaptation mechanism can encode repetition/novelty structure without hand-defining emotion axes.
