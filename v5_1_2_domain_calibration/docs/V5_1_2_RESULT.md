# EmoNet v5.1.2 Result — Domain-Conditioned Semantic Calibration

검증일: 2026-08-09 KST

## Version boundary

- v5.0 temporal-memory baseline: frozen
- v5.1 semantic-context attempt: frozen failed result
- v5.1.1 failure diagnostic: frozen
- v5.1.2 domain-conditioned calibration: this result
- branch: `feature/v5.1.2-domain-calibration`
- PR: #7
- workflow run: `31289981323`
- artifact: `v5.1.2-domain-calibration`
- artifact id: `9031070122`

This version reused the already-seen v5.1 fixture and is calibration only, never confirmatory evidence.

## Input adequacy after domain conditioning

Semantic-event embedding accuracy:

| Domain | Accuracy |
| --- | ---: |
| access | 0.875 |
| authorization | 0.875 |
| device | 0.875 |
| resource | 0.750 |
| schedule | 0.625 |
| **macro** | **0.800** |

History-bag macro accuracy: `0.750`.

Domain conditioning improves input adequacy over the global v5.1.1 semantic-event result (`0.675`), but the schedule domain remains weak and the predefined calibration gate (`macro >= 0.85`, every domain >= 0.75) is not met.

## Trace results

Macro accuracy across five recurrent seeds:

| Trace view | Real | Reset | Wrong/opposite |
| --- | ---: | ---: | ---: |
| final state | 0.570 | 0.500 | 0.430 |
| summary features | 0.570 | 0.500 | 0.430 |
| raw flattened trace | **0.580** | **0.500** | **0.420** |

Raw trace gaps:

```text
real - reset = +0.080
real - wrong = +0.160
input semantic macro - real = +0.220
```

## Per-domain raw-trace mean

| Domain | Input semantic | Raw trace |
| --- | ---: | ---: |
| access | 0.875 | 0.575 |
| authorization | 0.875 | 0.675 |
| device | 0.875 | 0.625 |
| resource | 0.750 | 0.525 |
| schedule | 0.625 | 0.500 |

The strongest diagnostic evidence comes from access, authorization, and device: their frozen semantic input representations are each `0.875` decodable on held-out paraphrases, while the frozen random recurrent trace falls to `0.575`, `0.675`, and `0.625` respectively.

## Acceptance

```text
input_semantic_macro_above_0_85 = false
every_domain_input_at_least_0_75 = false
raw_trace_macro_above_0_65 = false
raw_trace_beats_reset_by_0_10 = false
raw_trace_beats_wrong_by_0_10 = true
calibration_pass = false
```

## Interpretation

v5.1.2 is a calibration failure, but it narrows the bottleneck substantially.

The result is no longer explained only by a globally weak cross-domain probe. In three domains where the semantic input itself reaches `0.875`, the v5.0 fixed random recurrent core still loses a large fraction of held-out semantic decodability.

Together with v5.1.1:

- gap0 -> gap2 caused only about a `0.05` raw-trace drop;
- raw vs summary at gap2 differed by only about `0.015`;
- domain-conditioned input can reach `0.875`;
- corresponding recurrent traces remain around `0.575–0.675`.

The strongest current diagnosis is therefore:

> the fixed random recurrent v5.0 substrate is a valid temporal-memory sanity baseline, but it is not a sufficiently information-preserving semantic state model.

This is not yet evidence that a learned EmoNet core will solve the problem.

## Next version

**v5.2** should change the core for the first time since v5.0.

The proposed change is a label-free learned memory objective:

- keep the frozen semantic input encoder;
- train recurrent dynamics without emotion/task labels;
- require its state after intervening neutral events to retain/reconstruct or contrastively identify the earlier event embedding;
- compare directly against the frozen v5.0 random recurrent baseline;
- keep reset/wrong controls;
- treat the existing fixture as development/calibration only.

A later untouched fixture must be used for confirmatory semantic-context evidence after v5.2 is frozen.
