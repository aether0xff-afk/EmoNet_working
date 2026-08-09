# EmoNet v5.1 Result — Semantic Context Memory

검증일: 2026-08-09 KST

## Version boundary

- version: **v5.1**
- branch: `feature/v5.1-semantic-context`
- PR: #5
- base: v5.0 (`feature/v5-clean-trace-rebuild`)
- v5.0 core modification: **none**
- workflow: `v5.1-semantic-context`
- verified run: `31289556597`
- artifact: `v5.1-semantic-context-probe`
- artifact id: `9030958118`

## Protocol

Frozen input encoder:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Frozen recurrent core:

```text
v5.0 FixedRecurrentDynamics
```

Dataset:

- 5 semantic domains
- 60 train pairs
- 20 held-out paraphrase test pairs
- 120 train samples / seed
- 40 test samples / seed
- recurrent seeds: `7, 13, 21, 42, 100`

Each pair keeps the current text and final history event identical while changing an earlier semantic world-state statement.

No emotion / valence / arousal labels were used.

## Mean accuracy

| Condition | Accuracy |
| --- | ---: |
| current text only | 0.500 |
| last event only | 0.500 |
| history bag embedding | **0.775** |
| full history embedding | **0.725** |
| trace only real | 0.555 |
| text + real trace | 0.555 |
| text + temporal shuffle | 0.500 |
| text + wrong trace | 0.445 |
| text + reset trace | 0.500 |

Real-trace seed results:

```text
seed 7   = 0.500
seed 13  = 0.550
seed 21  = 0.550
seed 42  = 0.625
seed 100 = 0.550
```

## Acceptance

The scientific acceptance gate **failed**.

```text
encoder_full_history_above_0_80 = false
real_trace_above_0_65_mean = false
every_seed_real_above_0_55 = false
real_beats_reset_by_0_15 = false
real_beats_wrong_by_0_15 = false
all_primary_gates = false
```

CI itself succeeded. A green workflow means only that the protocol executed correctly; it does not mean the hypothesis passed.

## Interpretation

v5.1 does **not** establish semantic context memory.

There is only a weak directional signal:

```text
real trace  = 0.555
reset       = 0.500
wrong trace = 0.445
```

This is too small to support the intended claim.

At the same time, the semantic input baseline is not perfect:

```text
history bag  = 0.775
full history = 0.725
```

Therefore v5.1 alone cannot cleanly distinguish among:

1. semantic input representation weakness,
2. recurrent-state information loss / memory decay,
3. trace-summary information loss,
4. linear-probe limitation.

## Domain diagnostic

Approximate mean accuracy across seeds:

| Domain | history bag | full history | real trace |
| --- | ---: | ---: | ---: |
| access | 0.750 | 0.750 | 0.525 |
| authorization | 0.750 | 0.625 | 0.625 |
| device | 0.750 | 0.750 | 0.525 |
| resource | 1.000 | 0.875 | 0.575 |
| schedule | 0.625 | 0.625 | 0.525 |

The large input-to-trace drop suggests that recurrent retention or trace readout is a likely bottleneck, but this is only a diagnostic inference.

## Next version

**v5.1.1** will not change the scientific claim. It is a diagnostic/calibration version designed to locate the failure source by varying:

- semantic-event-to-current gap: 0 / 1 / 2 neutral events
- raw flattened trace vs final state vs v5.0 summary features
- semantic-event embedding ceiling
- reset / wrong controls

v5.1 remains frozen as the first failed semantic-context attempt.
