# EmoNet v5.8.2 — Benchmark Equivalence / Renderer Sensitivity Diagnostic

## Version boundary

- base: frozen v5.8.1 result
- **no state-generator changes**
- frozen models compared:
  - v5.7 residual-driven fixed recurrence
  - v5.8 adaptive residual recurrence (`decay=0.995`, `strength=0.20`)
- diagnostic only; no tuning

## Motivation

The earlier v5.7.1 `ABAB vs AABB` benchmark gave v5.7 raw accuracy around 0.5625. The logically equivalent v5.8 `alternation` benchmark gives frozen v5.7 raw around 0.6275.

Both use:

```text
class 0: A B A B
class 1: A A B B
```

with the same 2A+2B event multiset, same suffix, and same current observation. The main differences are textual renderer details such as the prefix wording and pseudo-token number formatting.

Before interpreting any adaptation effect, this version tests whether the raw neural code is robust to those superficial renderer changes.

## Frozen renderer definitions

### OLD renderer

Exactly reproduces v5.6.1/v5.7.1 `structural_pair()`:

- `Structural sequence case {pair_id:03d} begins with a neutral start marker.`
- pseudo-token numeric field `{pair_id:03d}`

### NEW renderer

Exactly reproduces v5.8 `build_case('alternation', pair_id)`:

- `Temporal structure case alternation-{pair_id:03d} begins with a neutral marker.`
- pseudo-token numeric field `{pair_id:05d}`

Both use the same logical A/B sequence and same suffix/current.

## Primary transfer matrix

For each frozen model and each recurrent seed:

```text
OLD train -> OLD test
NEW train -> NEW test
OLD train -> NEW test
NEW train -> OLD test
```

Train pair IDs: `0..79`.
Test pair IDs: `80..119`.
No event identity crosses the train/test boundary.

## Paired renderer diagnostics

For the same pair ID and logical class, measure:

- cosine similarity of OLD vs NEW event-A embeddings;
- cosine similarity of OLD vs NEW event-B embeddings;
- cosine similarity of OLD vs NEW prefix embeddings;
- normalized distance between OLD vs NEW raw current-event traces;
- prediction agreement when the same trained probe sees OLD and NEW versions of the same held-out latent sequence.

## Controls

- relational structure baseline for OLD and NEW must both remain >= 0.95;
- text bag/event multiset remains identical between classes within each renderer;
- opposite-history control remains available;
- reset control remains available;
- v5.7 and v5.8 use the same encoder and evaluation splits.

## Diagnostic interpretation

### Renderer-robust code

Supported only if cross-render transfer is close to within-render performance:

```text
mean cross-render accuracy >= 0.70
mean cross-render drop from within-render <= 0.10
```

### Renderer-sensitive code

Indicated if both within-render conditions are substantially above chance but cross-render transfer falls by > 0.15.

### Fixture difficulty drift

Indicated if OLD/NEW within-render accuracy differs materially while relational validity stays perfect and paired renderer embedding/trace diagnostics show nontrivial representation shifts.

No outcome in v5.8.2 can retroactively make v5.8 pass, and no emotion/affect claim is permitted.
