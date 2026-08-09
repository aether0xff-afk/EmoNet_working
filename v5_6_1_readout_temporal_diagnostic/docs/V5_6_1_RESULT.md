# EmoNet v5.6.1 Result — Readout & Temporal-Control Diagnostic

검증일: 2026-08-09 KST

## Version boundary

- v5.6 dual-timescale state: frozen failed result
- v5.6.1: diagnostic only
- state generators changed from v5.6: **no**
- branch: `feature/v5.6.1-readout-temporal-diagnostic`
- PR: #15
- workflow run: `31291723730`
- job: `93189961555`
- artifact: `v5.6.1-readout-temporal-diagnostic`
- artifact id: `9031628591`

## Engineering validation

Before the diagnostic benchmark:

- v5.6 fast/slow reset invariants passed;
- deterministic random-projection tests passed;
- structural task same-multiset tests passed;
- all 240 pair-specific A/B token identities were unique;
- train/test token identities were disjoint.

## A. Semantic integration diagnostic

Mean across five recurrent seeds:

| Readout | Semantic accuracy |
| --- | ---: |
| fast only | 0.595 |
| slow EMA only | **0.800** |
| raw concatenation | 0.675 |
| separate fast/slow score fusion | 0.660 |
| equal-dim projection 8+8 | 0.630 |
| equal-dim projection 16+16 | 0.650 |
| equal-dim projection 32+32 | **0.685** |
| equal-dim projection 64+64 | 0.645 |

Best balanced projected result:

```text
32 dimensions per block → 0.685
```

This remains far below slow-only `0.800`.

Automated diagnosis:

```text
fast_block_adds_semantic_nuisance_without_complementarity
```

### Interpretation

The v5.6 semantic failure is not explained merely by the raw dimensionality imbalance (`2048 fast` vs `384 slow`). Equal-dimensional label-free projections do not recover the slow semantic signal, and even fitting separate probes before score fusion reaches only `0.660`.

On the current semantic task, the fast random recurrent block therefore provides little complementary information and mostly introduces nuisance for a joint semantic readout.

The correct architecture conclusion is not to force fast and slow into one undifferentiated vector. They should remain semantically distinct state channels unless a later mechanism demonstrates useful integration.

## B. Identity-disjoint temporal-structure diagnostic

New task:

```text
class 0: A B A B
class 1: A A B B
```

For every pair:

- the two classes use exactly the same multiset `2A + 2B`;
- current input is identical;
- A/B identities are unique to the pair;
- train and test A/B identities are disjoint;
- 80 train pairs / 40 held-out test pairs;
- 120 total pairs.

A label-free relational benchmark-validating feature built from pairwise cosine similarities among the four transient events reaches:

```text
1.000
```

Therefore the task itself has a perfectly recoverable identity-invariant structural signal.

Mean across five recurrent seeds:

| Representation | Accuracy |
| --- | ---: |
| fast recurrent only | **0.565** |
| slow EMA only | 0.5125 |
| raw dual | 0.550 |
| projected dual 8+8 | 0.550 |
| projected dual 16+16 | **0.5725** |
| projected dual 32+32 | 0.5525 |
| projected dual 64+64 | 0.5475 |
| raw dual after fast reset | 0.5075 |
| raw dual after slow reset | 0.5075 |
| relational structure baseline | **1.000** |

### Interpretation

The harder benchmark successfully removes the previous EMA shortcut: slow-only falls to chance (`0.5125`).

The frozen random recurrent fast state rises only modestly above chance (`0.565`). This is a weak signal, not robust evidence of an identity-invariant temporal abstraction. The balanced dual variants remain similarly weak.

Thus v5.6.1 rules out the optimistic interpretation that the old fast recurrence already contains a strong general temporal-structure representation that the previous benchmark merely failed to expose.

## Combined conclusion

v5.6.1 narrows the architecture problem substantially:

```text
slow EMA:
  strong semantic retention                    ✓ 0.800
  identity-invariant structural sequence task  ~ chance

fast random recurrent:
  semantic retention                           weak 0.595
  identity-invariant structural sequence task  weak 0.565

naive / balanced dual fusion:
  does not solve either weakness robustly
```

The dual-timescale *idea* remains plausible, but the current fast channel is not justified.

## Design consequence

The next fast channel should not receive the full semantic embedding redundantly. That makes it compete with slow memory on information the slow channel already preserves better.

A cleaner role is to drive fast dynamics with **change relative to the slow state**:

```text
slow_t ≈ persistent context
residual_t = embedding_t - slow_(t-1)
fast dynamics ← residual_t
```

This is label-free and naturally converts repeated vs changed events into a shared signal independent of event identity.

For the structural benchmark:

```text
ABAB = repeated switching
AABB = repeated persistence then switching
```

so residual-driven dynamics have a principled reason to encode the temporal pattern without needing a globally fixed A/B identity.

## Next version

**v5.7 — Residual-Driven Fast Dynamics** should:

- preserve the slow EMA state unchanged;
- preserve the v5.0 recurrent equations/topology unchanged;
- change only the fast input from raw embedding to `embedding - previous slow state`;
- keep fast and slow as separate channels rather than forcing raw concatenation;
- compare residual-fast vs old raw-fast vs slow-only on the identity-disjoint structural task;
- verify slow semantic retention remains unchanged;
- use fast-reset controls.

v5.7 remains development-only.
