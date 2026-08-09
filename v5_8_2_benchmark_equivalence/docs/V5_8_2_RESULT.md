# EmoNet v5.8.2 Result — Benchmark Equivalence / Renderer Sensitivity

검증일: 2026-08-09 KST

## Version / run

- branch: `feature/v5.8.2-benchmark-equivalence-diagnostic`
- PR: #20
- workflow run: `31296930686`
- job: `93203460219`
- artifact: `v5.8.2-benchmark-equivalence`
- artifact id: `9033285691`
- artifact digest: `sha256:88cada6672a79b200241d9c0d62e24433a5e882d2730d0ace47ec0a70aba5474`

No state generator changed. Frozen v5.7 and v5.8 were evaluated under one common script.

## Renderer validity

Both OLD and NEW renderers implement the same latent task:

```text
class 0 = A B A B
class 1 = A A B B
```

Held-out relational validity:

```text
OLD relational accuracy = 1.000
NEW relational accuracy = 1.000
```

Paired OLD/NEW input-embedding cosine:

```text
event A = 0.9760
event B = 0.9765
prefix  = 0.7610
```

Thus both fixtures are structurally valid, and individual A/B event renderings are very similar in the frozen semantic embedding space.

## Frozen v5.7

```text
OLD train -> OLD test = 0.5625
NEW train -> NEW test = 0.6275
OLD train -> NEW test = 0.5275
NEW train -> OLD test = 0.5725

within-render mean = 0.5950
cross-render mean  = 0.5500
cross drop         = 0.0450
```

Controls:

```text
OLD reset    = 0.5200
NEW reset    = 0.5050
OLD opposite = 0.4375
NEW opposite = 0.3725
```

Mean paired OLD/NEW raw-trace normalized distance: `0.08337`.

## Frozen v5.8 adaptation

```text
OLD train -> OLD test = 0.5775
NEW train -> NEW test = 0.7075
OLD train -> NEW test = 0.5600
NEW train -> OLD test = 0.5875

within-render mean = 0.6425
cross-render mean  = 0.57375
cross drop         = 0.06875
```

Controls:

```text
OLD reset    = 0.5100
NEW reset    = 0.5075
OLD opposite = 0.4225
NEW opposite = 0.2925
```

Mean paired OLD/NEW raw-trace normalized distance: `0.08230`.

## Interpretation

The earlier OLD and NEW scores are exactly reproduced under one evaluator. Therefore the drift is not caused by using different evaluation scripts.

The neural code does contain history-order information within a renderer, as shown by above-chance within-render accuracy and reset/opposite-history interventions. However, a probe learned on one textual renderer transfers poorly to the logically identical other renderer: cross-render means are only `0.55` (v5.7) and `0.57375` (v5.8).

The code therefore **does not yet qualify as a renderer-invariant abstract representation of `ABAB vs AABB`**.

The surprising v5.8 raw performance should be described as a stable, history-causal, neuron-coordinate-specific code **within the tested rendering distribution**, not as general relational temporal abstraction.

The fact that A/B embeddings themselves have cosine ≈0.976 across renderers while trace/probe transfer still fails shows that relatively small input-space changes can rotate/shift the recurrent coordinate code enough to impair downstream linear transfer.

## Next version

Before changing the neural architecture again, remove the language-renderer confound.

Create **v5.9 — Encoder-Free Mechanistic Temporal Benchmark**:

- replace sentence text with controlled random unit-vector events;
- keep event identity train/test disjoint;
- vary vector-world seeds independently from recurrent seeds;
- use fixed neutral prefix/suffix/current vectors;
- test the same relational temporal suite;
- compare frozen v5.7 recurrence, frozen v5.8 adaptation, adaptation state, adaptation-only, reset, opposite-history, and direct relational controls;
- no language model, semantic embedding model, or emotion labels.

This will answer whether the dynamics themselves can form identity-invariant temporal structure, independently of language rendering.

No emotion/affect claim is supported by v5.8.2.
