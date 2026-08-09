# EmoNet v5.9 — Encoder-Free Mechanistic Temporal Benchmark

## Version boundary

- base: frozen v5.8.2 renderer audit
- **no language encoder**
- frozen neural mechanisms:
  - v5.7 residual-driven fixed recurrence
  - v5.8 adaptive residual recurrence (`adaptation_decay=0.995`, `adaptation_strength=0.20`)
- no emotion labels, semantic labels, language templates, or task labels enter state generation
- no hyperparameter tuning in this version

This version isolates the neural dynamics from the language-renderer confound found in v5.8.2.

## Controlled vector environment

Input dimension stays `384` to match the previous MiniLM embedding width.

For every pair, A/B/C are freshly generated orthonormal unit vectors from a seeded Gaussian matrix. Therefore:

- repeated event = exactly the same vector;
- distinct A/B/C events = orthogonal within a pair;
- train/test pairs use independent event directions;
- event directions contain no class information;
- vector norms are fixed;
- no text is embedded.

Three independently generated **vector worlds** are used:

```text
101, 211, 307
```

Fixed neutral prefix, suffix, and current vectors are generated once per world and are identical across classes/pairs in that world.

Recurrent seeds remain:

```text
7, 13, 21, 42, 100
```

## Temporal suite

Exactly four latent tasks:

```text
alternation       ABAB vs AABB
palindrome        ABBA vs AABB
repeat_gap        ABCA vs AABC
repeat_position   ABCA vs ABAC
```

Pair count per task/world: `120`.

```text
train pair IDs = 0..79
test pair IDs  = 80..119
```

Every competing class uses the same event multiset.

## Primary generalization protocol

The main result is **leave-one-vector-world-out**.

For each held-out vector world:

1. fit one ridge probe using train pairs from the other two worlds;
2. evaluate on test pairs from the held-out world;
3. repeat for each recurrent seed and task.

Thus a probe cannot rely on the particular A/B/C directions used by the test vector world.

Within-world train/test accuracy is recorded only as a diagnostic.

## Readouts

Frozen v5.7:

- full raw current-event fast trace
- population moments

Frozen v5.8:

- full raw current-event fast trace
- population moments
- raw adaptation-state vector
- adaptation-state population moments
- adaptation-only raw trace (recurrent matrix removed)

Causal controls:

- fast reset before common current
- opposite-class history using the same pair vectors

Task-validity baseline:

- pairwise dot products among the four transient event vectors

## Predeclared interpretation gates

A mechanism qualifies as an **encoder-free temporal abstraction candidate** only if all hold under leave-one-world-out evaluation:

```text
macro accuracy >= 0.70
fast-reset drop >= 0.15
opposite-history accuracy <= 0.35
at least 3/4 tasks >= 0.65
relational validity >= 0.95
```

These gates are evaluated separately for v5.7 raw and v5.8 adaptive raw.

Adaptation is considered to add meaningful value only if:

```text
v5.8 raw - v5.7 raw >= 0.03
```

Recurrence is considered justified over adaptation alone only if:

```text
v5.8 raw - adaptation-only raw >= 0.03
```

The adaptation-state readouts are localization diagnostics and do not replace the primary fast-trace gate.

## Claim boundary

A pass would establish only that the frozen dynamics encode identity-invariant temporal structure in a controlled vector environment. It would not yet establish language robustness, emotion semantics, consciousness, or downstream behavioral value.
