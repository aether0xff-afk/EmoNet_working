# EmoNet v5.3 — Contrastive Delayed Memory

## Version boundary

- v5.0: fixed-random temporal baseline — frozen
- v5.1–v5.1.2: semantic benchmark diagnostics — frozen
- v5.2: cosine delayed-reconstruction core — frozen failed result
- v5.2.1: readout diagnostic — frozen
- **v5.3 changes only the self-supervised memory objective**

The recurrent architecture is imported unchanged from v5.2.

## Why the objective changes

v5.2 achieved held-out lag-3 embedding cosine around `0.64`, yet its reconstructed embedding retained only about `58.5%` semantic-state accuracy versus `80%` in the original frozen embedding.

Cosine reconstruction therefore preserved broad similarity while losing small but decision-relevant semantic differences.

## Label-free contrastive objective

For each recurrent state and lag `k ∈ {1,2,3}`:

1. the memory head predicts a normalized embedding;
2. every **unique training event embedding** is a candidate;
3. the true event from `k` events ago is the positive candidate;
4. all other event embeddings are negatives;
5. cross-entropy over cosine similarity / temperature is minimized.

```text
state_t
  ↓
memory head k
  ↓
predicted embedding
  ↓
cosine against all unique training events
  ↓
identify the exact event at t-k
```

No emotion label, valence/arousal label, or `usable/blocked` benchmark label is provided to the optimizer.

Near-paraphrases and opposite-state sentences are naturally hard negatives because their frozen embeddings are similar but correspond to different event identities.

## Frozen architecture

v5.3 reuses `LearnedLeakyRecurrentCore` from v5.2 without modification:

- hidden = 128
- event ticks = 16
- stimulation ticks = 6
- leaky tanh recurrence
- lag 1/2/3 linear memory heads

Thus v5.2 vs v5.3 isolates **objective choice** rather than architecture choice.

## Development benchmark

The already-seen v5.1 fixture remains development-only.

Compare:

- v5.0 random recurrent trace
- v5.2 cosine-memory trace (historical frozen result)
- v5.3 contrastive-memory trace
- EMA embedding memory
- reset trace
- opposite-arm/wrong trace

Also report held-out lag-3 event retrieval against the unique held-out event vocabulary.

## Predeclared gates

### Semantic-memory gate

- held-out lag-3 retrieval top-1 >= `0.20`
- contrastive trace semantic macro >= `0.70`
- contrastive trace improves >= `+0.10` over v5.0 random recurrent
- contrastive trace improves >= `+0.10` over v5.2 cosine-memory trace
- contrastive trace beats reset >= `+0.15`
- contrastive trace beats wrong trace >= `+0.15`

### Complexity check

EMA remains a mandatory baseline. The result must explicitly report whether the learned recurrent trace reaches or exceeds simple EMA memory. A semantic-memory pass does not automatically imply that the neural dynamics are justified over EMA.

## Claim boundary

This fixture has already influenced development decisions. Even if every gate passes, v5.3 is **development evidence only**. The protocol must then be frozen and evaluated on a new untouched fixture in a later version.
