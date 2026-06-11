# EmoNet v7 SNN Core Experiment Summary — 2026-06-07

## Scope

This report summarizes the validated plumbing and optimizer smoke experiments for the clean v7 rebuild. It does **not** claim that emotional meaning has emerged.

## CI status

The `v7-snn-tests` workflow passed on commit `0c003a2605cd77257e6a25e248d85a06fd402e85`.

Validated steps:

- Python compile check
- unit tests
- decay smoke experiment
- hash wiring smoke experiment
- multilingual input plumbing experiment
- five-seed multilingual selectivity suite
- offline internal-thought metadata ablation suite
- self-supervised optimizer smoke experiment
- artifact upload

## Decay heartbeat

| Metric | Value |
|---|---:|
| Total ticks | 152 |
| Peak active ratio | 0.078125 |
| Final active ratio | 0.0 |
| Post-input nonzero ticks | 4 |
| Last nonzero tick | 11 |
| Post-input spike ratio | 0.166667 |
| Maximum absolute membrane value | 4.956134 |
| Contains NaN | false |

The SNN retains a short post-input trace and then decays to zero without numerical instability.

## Five-seed multilingual text-only selectivity suite

The default EventEncoder now injects text content only. Structural metadata remains available for controlled ablations but is disabled by default.

| Relation | Text embedding distance mean | Input current distance mean | Trace latent distance mean |
|---|---:|---:|---:|
| Identical | ~0.0000 | ~0.0000 | ~0.0000 |
| Lexical overlap but different context | 0.1652 | 0.0802 | 0.0770 |
| Paraphrase | 0.2904 | 0.1507 | 0.1167 |
| Interpretation contrast | 0.4984 | 0.2526 | 0.1794 |
| Unrelated | 1.0630 | 0.4916 | 0.1998 |

The text-only default preserves considerably more input separability than the previous metadata-concatenated default. The pipeline remains deterministic for identical text and preserves nonzero separation for distinct text.

## Offline internal-thought metadata ablation

Mean trace latent cosine distance across the three thought-content contrasts:

| Ablation | Mean trace latent distance |
|---|---:|
| Text only | 0.1234 |
| Text + event kind | 0.0636 |
| Text + event kind + speaker | 0.0344 |

Untrained metadata embeddings reduced text-content separability when concatenated into the same MLP input. Therefore, metadata injection is disabled by default. Metadata remains logged for future experiments with separate encoders, restricted scales, or learned gates.

## Self-supervised optimizer smoke

| Metric | Value |
|---|---:|
| Steps | 20 |
| Initial total loss | 0.962929 |
| Final total loss | 0.067229 |
| Relative reduction | 93.02% |

This validates the optimization path:

```text
EventEncoder
→ differentiable SNN window
→ TraceEncoder
→ NextEventPredictor
→ loss
→ gradient
→ optimizer step
```

The smoke fixture uses a deterministic hash encoder and tiny example pairs. It validates trainability, not emotional semantics or real-world generalization.

## Current conclusion

The minimum SNN core is stable, reproducible, observable, and differentiable. Text content can be transformed into SNN traces without complete collapse, and the model can optimize a simple next-event objective.

## Claims that remain unsupported

- The internal trace represents human emotion.
- Emotional semantics have emerged without labels.
- Internal thoughts regulate emotion-like state in a human-like way.
- The LM Studio-generated thought loop works end-to-end on the user's local server.
- STDP, rewiring, or self-organizing clusters have been validated.

## Next blocking input

To validate the generated-thought loop, provide:

```text
LM Studio server base URL
loaded chat model identifier
```
