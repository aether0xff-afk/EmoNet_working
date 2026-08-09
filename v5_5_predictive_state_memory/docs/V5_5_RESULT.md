# EmoNet v5.5 Result — Predictive State Memory

검증일: 2026-08-09 KST

## Version boundary

- v5.4.1 contrastive abstraction diagnostic: frozen
- v5.5 predictive-state development: this result
- branch: `feature/v5.5-predictive-state-memory`
- PR: #13
- successful benchmark run: `31291278280`
- artifact: `v5.5-predictive-state-memory-benchmark`
- artifact id: `9031485869`

A prior workflow attempt failed before any benchmark execution because sparse checkout omitted an inherited helper directory. The scientific protocol, fixture, gates, architecture, objective, and hyperparameters were not changed. The successful rerun added only the missing checkout dependency.

## Leakage/protocol verification

Before the benchmark:

```text
frozen recurrent core tests: 4 passed
v5.5 leakage/protocol tests: 4 passed
```

The evaluated recurrent trace contains history and the same current event only. The future consequence is never passed into the evaluated trace.

Core training API accepts only:

- event-sequence embeddings
- future-event identity
- candidate future embeddings

No semantic-state, positive/negative, usable/blocked, emotion, valence, arousal, or downstream probe label enters core training.

## Result

Mean across five recurrent seeds:

| Metric | Result |
| --- | ---: |
| held-out future consequence top-1 | **0.095** |
| predictive recurrent semantic macro | **0.525** |
| reset trace semantic macro | 0.500 |
| wrong/opposite trace semantic macro | 0.475 |
| v5.0 random recurrent macro | **0.595** |
| historical v5.4 contrastive-past-memory macro | **0.630** |
| EMA embedding memory macro | **0.800** |

Gaps:

```text
predictive - random = -0.070
predictive - v5.4   = -0.105
predictive - reset  = +0.025
predictive - wrong  = +0.050
predictive - EMA    = -0.275
```

No recurrent seed reached the preregistered `0.68` semantic macro threshold.

## Seed-wise result

| Seed | Held-out future top-1 | Predictive semantic | Random | EMA |
| ---: | ---: | ---: | ---: | ---: |
| 7 | 0.175 | 0.550 | 0.575 | 0.800 |
| 13 | 0.025 | 0.475 | 0.575 | 0.800 |
| 21 | 0.100 | 0.525 | 0.625 | 0.800 |
| 42 | 0.075 | 0.575 | 0.475 | 0.800 |
| 100 | 0.100 | 0.500 | 0.725 | 0.800 |

## Training behavior

At epoch 150, train future top-1 across seeds was approximately:

```text
seed 7   = 0.125
seed 13  = 0.458
seed 21  = 0.475
seed 42  = 0.492
seed 100 = 0.442
mean     ≈ 0.398
```

The training objective therefore learned substantial training-set future identity information for most seeds, while held-out future-paraphrase retrieval remained only `0.095`.

This is consistent with poor abstraction/generalization rather than a simple no-learning failure.

## Predeclared gate

Every primary gate failed:

```text
heldout_future_retrieval_at_least_0_30 = false
predictive_semantic_macro_at_least_0_72 = false
predictive_beats_random_by_0_10 = false
predictive_beats_historical_v5_4_by_0_08 = false
predictive_beats_reset_by_0_15 = false
predictive_beats_wrong_by_0_15 = false
at_least_4_of_5_seeds_at_or_above_0_68 = false
all_primary_gates = false
```

## Interpretation

v5.5 fails clearly.

Predicting the exact identity of a future consequence does not solve the abstraction problem identified in v5.4.1. It largely replaces one instance-identification objective (remember the exact past event) with another (identify the exact future sentence).

The recurrent state becomes worse, not better, as a held-out semantic-state representation.

The broader pattern across versions is now important:

```text
v5.0 recurrent dynamics:
  strong temporal-order memory

v5.1–v5.4 recurrent semantic state:
  weak / inconsistent

v5.2 cosine reconstruction:
  broad semantic similarity retained,
  decisive semantic distinctions lost

v5.3/v5.4 contrastive past identity:
  exact event/domain memory improves,
  stable state abstraction remains weak

v5.5 future identity prediction:
  training identity improves,
  held-out abstraction collapses

EMA embedding memory:
  consistently strong semantic retention
```

## Design consequence

The next step should not keep asking one fast recurrent state to perform every kind of memory.

A more defensible architecture is a **dual-timescale state**:

1. fast recurrent dynamics for temporal order / transient trajectory;
2. slow persistent semantic memory that deliberately resists overwrite.

This is also closer to the original EmoNet motivation of a time-varying internal process with memory, rather than treating one homogeneous recurrent vector as both trajectory and long-term context.

## Next version

**v5.6 — Dual-Timescale State** should first be an architecture/control experiment, not another complex learning objective.

Start with:

```text
frozen semantic embedding
        ├──> fast recurrent state / trace
        └──> slow EMA-like persistent state

internal state = fast dynamics + slow memory
```

Mandatory tests:

- temporal-order benchmark
- fresh semantic-context benchmark
- EMA alone
- fast recurrent alone
- fast + slow dual state
- slow-state reset
- fast-state reset

The key question is not whether dual state trivially contains more information, but whether the fast component provides measurable information **beyond** the slow EMA baseline while the slow component prevents semantic information loss.
