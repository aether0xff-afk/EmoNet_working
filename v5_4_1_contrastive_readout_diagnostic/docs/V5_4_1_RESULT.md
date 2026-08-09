# EmoNet v5.4.1 Result — Contrastive Memory Readout Diagnostic

검증일: 2026-08-09 KST

## Version boundary

- v5.4 fresh confirmatory test: frozen failed result
- v5.4.1: diagnostic only
- core architecture changed from v5.4: **no**
- self-supervised objective changed: **no**
- optimizer / epochs / seeds changed: **no**
- fresh fixture changed: **no**
- branch: `feature/v5.4.1-contrastive-readout-diagnostic`
- PR: #12
- workflow run: `31290961066`
- artifact: `v5.4.1-contrastive-readout-diagnostic`
- artifact id: `9031375252`

This version cannot retroactively turn v5.4 into a confirmatory pass.

## Main readout result

Macro held-out semantic-state accuracy:

| Readout | Accuracy |
| --- | ---: |
| original semantic-event embedding | **0.800** |
| semantic geometry transferred to lag-3 memory-head output | 0.630 |
| memory-head native semantic probe | 0.625 |
| raw recurrent trace | 0.630 |
| EMA embedding memory | **0.800** |

Diagnostic gaps:

```text
memory-head native - raw trace      = -0.005
memory-head native - geometry transfer = -0.005
semantic input - memory-head native = +0.175
EMA - memory-head native            = +0.175
```

## Retrieval decomposition

| Retrieval diagnostic | Accuracy |
| --- | ---: |
| all-event exact lag-3 top-1 | 0.365 |
| semantic-candidate exact top-1 | **0.380** |
| semantic-candidate polarity accuracy | **0.675** |
| semantic-candidate domain accuracy | **1.000** |

The memory head identifies the semantic domain perfectly on this fixture and can often retrieve the exact delayed semantic sentence, but the higher-level positive/negative state shared across paraphrases and domains is much less stable.

## Diagnosis

Automated diagnosis:

```text
instance_identity_without_stable_state_abstraction
```

This diagnosis is supported by three observations.

### 1. It is not mainly a raw-trace readout problem

If useful semantic state existed in the trained memory head but the raw trace probe simply failed to discover it, `memory_head_native` should substantially exceed `raw_trace`.

Instead:

```text
memory_head native = 0.625
raw trace          = 0.630
```

They are effectively identical.

### 2. It is not mainly a coordinate-system / geometry-transfer problem

A native probe trained directly on the memory-head output does not outperform a probe whose semantic direction was learned from the original frozen embeddings:

```text
native head probe        = 0.625
semantic geometry transfer = 0.630
```

So the failure is not explained by the correct abstraction merely being rotated into a new coordinate system.

### 3. The objective strongly preserves identity/domain but only weakly preserves abstract state

The clearest pattern is:

```text
domain identity   = 1.000
exact sentence ID = 0.380
state polarity    = 0.675
```

The contrastive delayed-event objective rewards remembering which specific event occurred. It therefore encourages instance-level discrimination. That is sufficient for strong event retrieval, but it does not force different paraphrases with the same future implications to organize into a shared latent state.

## Scientific interpretation

The current evidence supports:

> v5.3/v5.4 contrastive training learns a useful delayed event-identity memory, but exact past-event identity is not the same thing as a stable predictive semantic state.

This explains why v5.3 looked promising on the development fixture but v5.4 did not confirm strongly across new semantic domains.

The next objective should therefore not ask the system to remember the past for its own sake. It should ask the internal state to retain whatever aspects of history are useful for predicting **future consequences**.

## Next version

**v5.5 — Predictive State Memory** should be a new development version.

Principle:

```text
past events + same current situation
             ↓
        recurrent state
             ↓
 predict what happens next
```

Core training remains label-free:

- no emotion labels
- no `usable/blocked` labels
- no downstream probe labels
- only the actually observed next-event identity/embedding is used as the self-supervised target

The motivation is that two differently worded histories with the same consequences should be encouraged to form similar useful internal states, while histories leading to different consequences should remain distinguishable.

A simple EMA memory remains mandatory as a baseline. v5.5 is development only; if it succeeds, a later untouched fixture must be used for confirmation.
