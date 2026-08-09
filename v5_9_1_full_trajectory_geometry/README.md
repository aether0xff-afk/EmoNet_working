# EmoNet v5.9.1 — Full Trajectory Geometry Diagnostic

## Version boundary

- base: frozen v5.9 encoder-free result
- branch: `feature/v5.9.1-full-trajectory-geometry`
- **v5.7/v5.8 dynamics are unchanged**
- **v5.9 vector worlds, tasks, seeds, and train/test splits are unchanged**
- no language encoder, no emotion/semantic labels, no hyperparameter tuning
- diagnostic only

v5.9 established that the trace produced during one common final observation is at chance in the encoder-free vector worlds. It did **not** test the original v3.1 trajectory-first hypothesis because all history-event traces were generated and then discarded after updating recurrent state.

v5.9.1 tests the stronger and more authentic question:

> Does the full event-by-event neural trajectory contain identity-invariant temporal relations even when those relations are not compressed into the final recurrent state?

## Frozen environment

Exactly the v5.9 setup is reused:

```text
input dimension: 384
vector worlds:   101, 211, 307
recurrent seeds: 7, 13, 21, 42, 100
pairs/task/world: 120
train pairs: 0..79
test pairs: 80..119
```

Each pair has fresh orthonormal A/B/C vectors. Event directions are disjoint across train/test pairs and independently regenerated in each vector world.

Tasks:

```text
alternation       ABAB vs AABB
palindrome        ABBA vs AABB
repeat_gap        ABCA vs AABC
repeat_position   ABCA vs ABAC
```

## Primary protocol

Primary evaluation remains **leave-one-vector-world-out**.

For each held-out world, task, and recurrent seed:

1. fit a ridge probe on train pairs from the other two worlds;
2. evaluate on test pairs from the held-out world;
3. record the same within-world train/test score as a diagnostic.

No held-out world contributes to probe fitting.

## Trajectory readouts

For the four transient history events only, preserve the complete tick-by-tick trace instead of discarding it.

### 1. Event-trace self-similarity — primary trajectory readout

Flatten each event trace (`ticks × neurons`) and compute the six position-wise cosine similarities:

```text
sim(1,2), sim(1,3), sim(1,4), sim(2,3), sim(2,4), sim(3,4)
```

This is label-free and neuron-coordinate permutation invariant when the same network is used for all four events.

### 2. Event-final-state similarity

Take the final neural state of each transient event and compute the same six pairwise cosine similarities.

### 3. Event-mean-state similarity

Mean each transient trace over ticks, then compute the six pairwise cosine similarities.

### 4. Full raw episode trajectory

Concatenate all event traces from prefix, four transient events, suffix, and common current. This can reveal identity-specific trajectory memory but is **not** sufficient by itself for an abstraction claim.

### 5. Frozen v5.9 current-only raw trace

Recompute the common-current-only trace as the compressed-state baseline.

### 6. Raw input relational matrix

The six pairwise dot products of the four transient input vectors. This is the task-validity upper bound and should remain `1.0`.

All readouts are computed separately for frozen v5.7 and frozen v5.8.

## Causal / structure controls

For every pair, both class trajectories are generated from the same event-vector multiset.

For a probe trained on the real class trajectory geometry, also evaluate the **opposite-class history feature from the same pair** while keeping the original label. A genuine positional trajectory code should therefore flip/collapse under this control.

The frozen current-only result should remain near chance; otherwise v5.9 would not reproduce.

## Predeclared diagnostic gates

A trajectory readout qualifies as an **identity-invariant trajectory-geometry candidate** only if all hold under leave-one-vector-world-out:

```text
macro accuracy >= 0.85
all 4 tasks >= 0.80
opposite-history accuracy <= 0.20
leave-one-world-out drop from within-world <= 0.05
raw input relational validity >= 0.99
current-only raw accuracy <= 0.60
```

These gates are evaluated separately for:

- v5.7 event-trace self-similarity;
- v5.8 event-trace self-similarity.

Final-state similarity, mean-state similarity, and full raw episode trajectory are localization diagnostics rather than substitutes for the primary trajectory-similarity gate.

## Interpretation matrix

### A. Trace self-similarity passes; current-only remains chance

Conclusion allowed:

> The frozen dynamics preserve identity-invariant temporal relations in full trajectory geometry, but do not compress them into a persistent final state.

This would support the authentic v3.1 **trajectory-first representation** direction.

It would **not** prove that recurrence creates information beyond the input relational structure. Because the raw input relational matrix is an explicit upper-bound baseline, a successful trace-similarity result may show preservation/re-expression rather than emergent abstraction.

### B. Full raw episode works but self-similarity fails

Conclusion:

> The episode trajectory contains identity/world-specific information, but no renderer/identity-invariant relational geometry has been established.

### C. Self-similarity also fails

Conclusion:

> The current fast dynamics do not robustly preserve the tested relational structure even in their full trajectory; an architectural change is required before returning to semantic/emotion tests.

## Claim boundary

This version contains no language and no affect labels. No result may be used as evidence of emotion semantics, consciousness, or behavioral superiority.
