# EmoNet v5.3 Result — Contrastive Delayed Memory

검증일: 2026-08-09 KST

## Version boundary

- v5.0 fixed-random temporal baseline: frozen
- v5.1–v5.1.2 semantic diagnostics/calibration: frozen
- v5.2 cosine delayed-memory core: frozen failed result
- v5.2.1 readout diagnostic: frozen
- v5.3 contrastive delayed-memory: this result
- branch: `feature/v5.3-contrastive-memory`
- PR: #10
- workflow run: `31290569475`
- artifact: `v5.3-contrastive-memory-benchmark`
- artifact id: `9031243811`

v5.3 reused the v5.2 recurrent architecture unchanged. Only the label-free self-supervised objective changed from cosine reconstruction to exact delayed-event retrieval with contrastive cross-entropy over unique training-event embeddings.

No emotion labels or `usable/blocked` downstream labels were provided to the core optimizer.

## Main result

Mean across five recurrent seeds:

| Metric | Result |
| --- | ---: |
| held-out lag-3 exact event retrieval top-1 | **0.425** |
| held-out lag-3 cosine | 0.373 |
| v5.3 contrastive trace semantic macro | **0.725** |
| reset trace semantic macro | 0.500 |
| wrong/opposite trace semantic macro | 0.275 |
| v5.0 random recurrent macro | 0.580 |
| v5.2 cosine-memory historical macro | 0.560 |
| EMA embedding memory macro | **0.750** |

The test event vocabulary contained 75 unique events, so exact retrieval is a substantially harder requirement than broad cosine similarity.

## Gaps

```text
v5.3 - v5.0 random = +0.145
v5.3 - v5.2 cosine = +0.165
v5.3 - reset       = +0.225
v5.3 - wrong       = +0.450
v5.3 - EMA         = -0.025
```

## Seed-wise result

| Seed | Retrieval top-1 | Semantic macro | Random | EMA |
| ---: | ---: | ---: | ---: | ---: |
| 7 | 0.400 | 0.700 | 0.550 | 0.750 |
| 13 | 0.525 | 0.750 | 0.625 | 0.750 |
| 21 | 0.300 | 0.600 | 0.575 | 0.750 |
| 42 | 0.450 | 0.750 | 0.575 | 0.750 |
| 100 | 0.450 | 0.825 | 0.575 | 0.750 |

The improvement is therefore not produced by a single lucky topology, although seed sensitivity remains material and seed 21 is notably weaker.

## Domain observations

The strongest v5.3 domains are authorization and device; schedule remains comparatively difficult. The per-domain result also shows that improvement is not uniform across seeds/domains, so the mean result should not be interpreted as solved semantic memory.

## Predeclared gate

```text
heldout_lag3_retrieval_top1_at_least_0_20 = true
contrastive_semantic_macro_at_least_0_70 = true
contrastive_beats_random_by_0_10 = true
contrastive_beats_v5_2_cosine_by_0_10 = true
contrastive_beats_reset_by_0_15 = true
contrastive_beats_wrong_by_0_15 = true
semantic_memory_gate = true
```

**v5.3 passes the predeclared development semantic-memory gate.**

## Complexity check

The required simple-memory baseline still matters:

```text
EMA embedding memory = 0.750
v5.3 recurrent trace = 0.725
EMA advantage        = +0.025
```

Therefore v5.3 does **not** establish that recurrent neural dynamics are superior to a simple exponentially weighted embedding memory.

The defensible interpretation is narrower:

> changing the label-free objective from broad cosine reconstruction to contrastive delayed-event retrieval makes the unchanged recurrent architecture preserve substantially more held-out semantic-state information than both the frozen random reservoir and the failed cosine-memory version.

This supports the importance of the learning objective, but not yet the necessity or superiority of the neural recurrent substrate itself.

## Why v5.3 is not confirmatory evidence

The v5.1 fixture had already been inspected repeatedly during v5.1–v5.2.1 development. Although v5.3 gates were declared before its run, architecture/objective selection was informed by results from the same fixture family.

Therefore v5.3 is development evidence only.

## Next version

**v5.4 should be a frozen-protocol confirmatory test on a completely fresh semantic fixture.**

Before seeing v5.4 results, freeze:

- sentence encoder
- recurrent architecture
- contrastive objective
- temperature
- epochs / optimizer
- recurrent seeds
- probe type
- baselines
- acceptance gates

v5.4 must not tune any of these after inspecting its fresh test result.

If v5.4 reproduces the semantic-memory result, the next research question becomes whether the recurrent trace provides information beyond simple EMA memory and eventually whether that extra state is affect-relevant.
