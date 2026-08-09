# EmoNet v5.2.1 Result — Memory Readout Diagnostic

검증일: 2026-08-09 KST

## Version boundary

- v5.2 learned-memory core: frozen failed result
- v5.2.1 readout diagnostic: this result
- core architecture changed: **no**
- training objective changed: **no**
- optimizer / epochs changed: **no**
- benchmark fixture changed: **no**
- branch: `feature/v5.2.1-memory-readout-diagnostic`
- PR: #9
- workflow run: `31290350349`
- artifact: `v5.2.1-memory-readout-diagnostic`
- artifact id: `9031179833`

## Result

The v5.2 delayed-memory training result reproduced exactly:

```text
lag-3 train cosine = 0.7309
lag-3 test cosine  = 0.6413
```

Macro held-out semantic-state accuracy:

| Readout | Accuracy |
| --- | ---: |
| original semantic-event embedding | **0.800** |
| semantic geometry transferred directly to lag-3 reconstruction | 0.560 |
| probe trained natively on lag-3 reconstruction | 0.585 |
| raw recurrent trace probe | 0.560 |
| EMA embedding memory | **0.750** |

Diagnostic gaps:

```text
reconstruction-native - raw trace = +0.025
reconstruction-native - geometry-transfer = +0.025
input semantic - reconstruction-native = +0.215
EMA - reconstruction-native = +0.165
```

Automated diagnosis:

```text
cosine_objective_information_bottleneck
```

## Interpretation

This rules against the hypothesis that v5.2 mainly failed because a small downstream probe could not discover a useful direction hidden in the raw trace.

Even after using the memory head that was explicitly trained to reconstruct the three-events-old embedding, semantic-state accuracy remains only `0.585`. Applying the semantic direction learned on the original frozen embeddings directly to the reconstructed embeddings gives only `0.560`.

Therefore the relatively high held-out cosine (`0.641`) is not preserving the fine semantic distinction needed by the benchmark. It is mostly preserving broad embedding similarity while losing information such as the distinction between permitted/blocked, available/unavailable, or operational/failed statements.

The key lesson is:

> high cosine reconstruction of a sentence embedding is not sufficient as a memory objective when small but decision-relevant semantic differences occupy only a small part of the embedding geometry.

## Next version

**v5.3** should keep the v5.2 recurrent architecture fixed and change only the label-free objective.

The proposed objective is delayed **contrastive event retrieval**:

- from the current recurrent state, predict the identity/embedding of the event at lag 1, 2, or 3;
- use all unique training-event embeddings as candidate negatives;
- optimize InfoNCE / cross-entropy over cosine similarity;
- no emotion labels and no `usable/blocked` labels;
- naturally similar opposite-state sentences become hard negatives, forcing the state to preserve the details that v5.2 cosine reconstruction averaged away.

v5.3 remains development on the seen fixture. A fresh untouched fixture is still required before any confirmatory semantic-memory claim.
