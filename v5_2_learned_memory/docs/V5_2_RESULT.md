# EmoNet v5.2 Result — Label-Free Learned Memory Core

검증일: 2026-08-09 KST

## Version boundary

- v5.0 fixed-random temporal baseline: frozen
- v5.1 / v5.1.1 / v5.1.2 semantic diagnostics: frozen
- v5.2 learned-memory core: this result
- branch: `feature/v5.2-learned-memory-core`
- PR: #8
- workflow run: `31290183461`
- artifact: `v5.2-learned-memory-benchmark`
- artifact id: `9031131979`

The v5.2 core received no emotion labels and no `usable/blocked` task labels. Its only training objective was delayed reconstruction of frozen event embeddings at lags 1, 2, and 3.

## Self-supervised training result

Across five seeds:

```text
lag-3 train cosine = 0.7309
lag-3 test cosine  = 0.6413
```

Training loss fell from roughly `1.0` at initialization to roughly `0.09` for every seed. Held-out lag-3 cosine was stable (`~0.639–0.645`) across seeds.

Therefore the self-supervised delayed-memory objective itself trained successfully and generalized at the embedding-reconstruction level.

## Semantic downstream result

Macro accuracy across five domains and five recurrent seeds:

| Representation | Accuracy |
| --- | ---: |
| v5.0 frozen random recurrent | **0.580** |
| v5.2 learned recurrent trace | **0.560** |
| simple EMA embedding memory | **0.750** |
| learned trace after reset | 0.500 |
| learned wrong/opposite trace | 0.440 |

Gaps:

```text
learned - random = -0.020
learned - EMA    = -0.190
learned - reset  = +0.060
learned - wrong  = +0.120
```

Per-domain mean:

| Domain | Learned | Random v5.0 | EMA |
| --- | ---: | ---: | ---: |
| access | 0.575 | 0.575 | 0.750 |
| authorization | 0.575 | 0.675 | 0.875 |
| device | 0.625 | 0.625 | 0.750 |
| resource | 0.575 | 0.525 | 0.750 |
| schedule | 0.450 | 0.500 | 0.625 |

## Acceptance

```text
lag3_test_cosine_above_0_40 = true
learned_semantic_macro_at_least_0_70 = false
learned_beats_random_by_0_10 = false
learned_beats_reset_by_0_15 = false
learned_beats_wrong_by_0_15 = false
all_primary_gates = false
```

v5.2 therefore fails its scientific development gate.

## Important interpretation

The negative result is informative because training clearly succeeded on its own objective.

The learned core can reconstruct a broad approximation of a three-events-old frozen embedding, yet this does not translate into improved held-out semantic-state decoding from the raw trace. A simple EMA of event embeddings substantially outperforms both the learned and random recurrent cores.

This separates two concepts that must not be conflated:

> reconstructing general semantic similarity of a past event is not the same as preserving the task-relevant distinctions of that event in a useful internal state geometry.

Possible explanations that remain open:

1. the delayed cosine-reconstruction objective preserves broad meaning but washes out fine distinctions such as negation / permitted-vs-blocked state;
2. the information exists in the hidden state but the small per-domain downstream probe cannot efficiently recover the linear direction learned by the memory head;
3. the recurrent state geometry is unnecessarily difficult compared with simple additive/EMA memory.

## Next version

**v5.2.1** is a readout diagnostic, not a new core.

It will freeze the v5.2 training protocol and test the output of the already-trained lag-3 memory head itself:

- direct semantic-event embedding baseline;
- lag-3 reconstructed embedding;
- raw recurrent trace;
- EMA memory.

If the reconstructed embedding preserves the semantic-state distinction while the raw trace probe does not, the bottleneck is downstream readout/sample efficiency. If reconstructed embedding also fails, the delayed cosine objective is preserving the wrong information and the next core objective must change.
