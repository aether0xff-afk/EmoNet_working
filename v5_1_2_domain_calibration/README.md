# EmoNet v5.1.2 — Domain-Conditioned Semantic Calibration

## Version boundary

- v5.0: temporal-memory baseline — frozen
- v5.1: global semantic-context attempt — frozen failed result
- v5.1.1: failure-source diagnostic — frozen
- v5.1.2: protocol calibration only

This version reuses the same v5.1 fixture and frozen v5.0 core. It does not modify either.

## Why this version exists

v5.1.1 found that a single global `usable vs blocked` linear axis across five semantic domains was only 67.5% decodable from the semantic-event embedding itself. That makes the original global probe an inadequate judge of recurrent retention.

v5.1.2 asks the narrower diagnostic question:

> Within a fixed semantic domain, can a held-out paraphrase probe decode the latent world state from the input embedding and from the recurrent trace?

Separate probes are fit for:

- access
- resource
- device
- schedule
- authorization

The final score is the macro-average across domains.

## Conditions

For each domain:

- semantic event embedding
- history bag embedding
- final recurrent state
- v5.0 summary features
- raw flattened current-event trace
- reset trace
- opposite-arm/wrong trace

The original v5.1 spacing (two neutral events between semantic event and current input) is preserved.

## Claim boundary

This is calibration on an already-seen fixture. Even a strong result is not confirmatory evidence. A later version must freeze the chosen protocol and evaluate it on a fresh untouched semantic fixture.
