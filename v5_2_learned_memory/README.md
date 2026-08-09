# EmoNet v5.2 — Learned Memory Core

## Version boundary

- v5.0: fixed-random temporal-memory baseline — frozen
- v5.1: first natural-language semantic-context attempt — frozen failed result
- v5.1.1: failure diagnostic — frozen
- v5.1.2: domain-conditioned calibration — frozen failed calibration
- **v5.2: first change to the recurrent core**

v5.2 lives only in this directory plus its dedicated workflow. Older version directories are not modified.

## Why v5.2 exists

v5.1.2 showed that even domains whose frozen semantic-event embedding was 87.5% linearly decodable fell to roughly 57.5–67.5% after passing through the fixed-random v5.0 recurrent substrate. The random reservoir is therefore useful as a temporal sanity baseline but too lossy as a semantic state model.

## Core-learning rule

v5.2 still receives **no emotion labels and no benchmark task labels**.

The recurrent dynamics are trained only with a self-supervised delayed-memory objective:

```text
event_1 → event_2 → ... → event_t
                     ↓
                recurrent state h_t
                     ↓
       predict embeddings of earlier events
       at lags 1, 2, and 3
```

For each lag `k`, a separate linear memory head attempts to reconstruct the frozen embedding of `event_(t-k)` from the current recurrent state.

This gives the dynamics a reason to preserve information through intervening events without telling it what `usable`, `blocked`, `positive`, `negative`, or any emotion category means.

## Architecture

The learned substrate deliberately stays close to v5.0:

```text
frozen sentence embedding
        ↓
trainable input projection
        ↓
leaky tanh recurrent dynamics
        ↓
tick-by-tick trace
```

No GRU/LSTM gates are introduced in this version. Recurrent weights are stabilized during training so v5.2 tests whether **the same basic recurrent form becomes useful when given a label-free memory objective**.

## Development benchmark

v5.2 reuses the already-seen v5.1 fixture for development only.

Core training uses only the v5.1 **train split** and only event embeddings. The `usable/blocked` labels are hidden from the core optimizer.

After training, the core is frozen. Domain-conditioned linear probes are then fit exactly as diagnostic readouts on the train split and evaluated on the held-out paraphrase split.

## Baselines

The same evaluation compares:

- `v5.0_random_recurrent`
- `v5.2_learned_recurrent`
- `ema_embedding_memory`

The EMA baseline is important: if a simple exponentially weighted average of past embeddings matches or beats v5.2, the learned recurrent dynamics have not yet justified their complexity.

## Controls

- real learned trace
- reset-before-current trace
- opposite-arm/wrong trace

## Predeclared development gates

v5.2 development succeeds only if:

1. lag-3 reconstruction generalizes to held-out paraphrases;
2. learned recurrent semantic macro accuracy is at least `0.70`;
3. learned recurrent improves by at least `+0.10` over the frozen v5.0 random recurrent baseline;
4. learned recurrent beats reset by at least `+0.15`;
5. learned recurrent beats wrong/opposite trace by at least `+0.15`;
6. the EMA baseline is reported and not hidden even if it wins.

Because the fixture has already been used for calibration, passing v5.2 is **not confirmatory semantic evidence**. A later fresh version must freeze this architecture and test on untouched data.
