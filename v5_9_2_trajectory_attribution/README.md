# EmoNet v5.9.2 — Trajectory Attribution / Input-Copy Audit

## Version boundary

- base: frozen v5.9.1 full-trajectory result
- no state-generator changes
- no language encoder
- frozen v5.9 vector worlds, tasks, splits, recurrent seeds, v5.7 weights, and v5.8 adaptation parameters
- diagnostic only; no tuning

v5.9.1 established a sharp result:

```text
raw input relational matrix          1.0000
full event-trace self-similarity    ~0.9996
common-current compressed trace     ~0.50
```

This supports trajectory-level relation preservation but does **not** establish a contribution from recurrence or adaptation. A simpler explanation is that each neural event response approximately preserves the equality/difference geometry already present in the inputs.

v5.9.2 locates where the near-perfect relational geometry first appears.

## Frozen benchmark

Exactly reuse v5.9/v5.9.1:

```text
vector worlds:   101, 211, 307
recurrent seeds: 7, 13, 21, 42, 100
input dimension: 384
pairs/task/world: 120
train pairs: 0..79
test pairs: 80..119
```

Tasks:

```text
alternation       ABAB vs AABB
palindrome        ABBA vs AABB
repeat_gap        ABCA vs AABC
repeat_position   ABCA vs ABAC
```

Primary evaluation remains leave-one-vector-world-out.

## Attribution ladder

For the same four transient positions, compute the same six pairwise cosine relations at successive stages.

### Stage 0 — raw input relation

```text
x_t
```

Six pairwise dot products of the original orthonormal event vectors.

### Stage 1 — sequential residual-input relation

```text
r_t = x_t - slow_(t-1)
```

Use the actual residual vectors emitted by the frozen v5.7 sequential run.

This tests whether the slow EMA transformation itself already preserves/creates the easily decoded structure.

### Stage 2 — fixed neural input-drive relation

```text
d_t = W_in r_t
```

Use the same frozen input matrix as v5.7/v5.8 and compute the six pairwise drive cosines.

### Stage 3 — isolated residual-response relation

Feed each **sequentially observed residual vector** into the same frozen fast dynamics, but reset fast recurrent/adaptation state to zero before every transient event.

This preserves:

- residual input;
- W_in;
- within-event leaky dynamics;

but removes:

- event-to-event recurrent carry;
- event-to-event adaptation carry.

Compute event-trace self-similarity from the four isolated responses.

### Stage 4 — isolated raw-input response

Feed each raw A/B/C input vector independently into the frozen fast dynamics from zero fast state. This is the strongest simple input-copy baseline.

### Stage 5 — sequential trajectory relation

Frozen v5.9.1 primary feature: event-trace self-similarity from the actual sequential history.

Compute separately for v5.7 and v5.8.

### Stage 6 — recurrent modulation geometry

For each sample:

```text
delta_geometry = sequential_trace_similarity - isolated_residual_trace_similarity
```

A leave-one-world-out probe on this six-dimensional delta asks whether recurrence changes trajectory geometry in a class-structured, identity-invariant way even when it is not necessary for solving the task.

## Additional paired metrics

Record across held-out samples:

- cosine agreement between sequential and isolated-residual six-dimensional geometry;
- mean L2 distance between those geometry vectors;
- v5.7 vs v5.8 sequential-geometry distance;
- current-only compressed-state accuracy from v5.9.1 as a frozen localization reference.

## Predeclared interpretation

### Input-geometry preservation is sufficient

Diagnose **copy_like_preservation** if all hold for v5.7:

```text
raw input relational accuracy >= 0.99
residual relational accuracy >= 0.95
input-drive relational accuracy >= 0.95
isolated-residual trace accuracy >= 0.95
sequential trace accuracy >= 0.95
|sequential - isolated-residual| <= 0.03
mean paired geometry cosine >= 0.95
```

This means recurrence is not required for the v5.9.1 relational success.

### Recurrent contribution is essential

Diagnose **recurrent_essential** only if:

```text
sequential trace accuracy >= 0.90
isolated-residual trace accuracy <= 0.75
sequential advantage >= 0.15
```

### Recurrent modulation exists but is not necessary

Diagnose **recurrent_modulation_detectable** if:

```text
delta-geometry accuracy >= 0.70
```

while isolated-residual already remains >=0.90.

These diagnoses are evaluated separately for v5.7 and v5.8 where applicable.

## Claim boundary

Even a strong recurrent modulation result would concern encoder-free temporal geometry only. No emotion/affect claim is permitted.

If copy-like preservation is confirmed, the correct next scientific target is not to improve this benchmark score. The next benchmark must hold current-window input relations fixed while varying **earlier latent history**, so that useful state cannot be obtained by re-reading the visible trajectory itself.
