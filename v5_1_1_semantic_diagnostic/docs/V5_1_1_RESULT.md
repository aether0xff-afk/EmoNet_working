# EmoNet v5.1.1 Result — Semantic Failure Diagnostic

검증일: 2026-08-09 KST

## Version boundary

- v5.0 temporal-memory baseline: frozen
- v5.1 semantic-context attempt: frozen failed result
- v5.1.1 diagnostic: this document
- branch: `feature/v5.1.1-semantic-diagnostic`
- PR: #6
- workflow run: `31289831024`
- artifact: `v5.1.1-semantic-failure-diagnostic`
- artifact id: `9031031885`

## Input adequacy

```text
semantic event embedding = 0.675
history bag embedding     = 0.775
```

The globally shared `usable vs blocked` linear axis is therefore not strongly linearly decodable from the frozen MiniLM input representation itself.

## Gap diagnostic

Mean real accuracy across five recurrent seeds:

| Gap | Final state | v5.0 summary | Raw flattened trace |
| --- | ---: | ---: | ---: |
| 0 neutral events | 0.635 | 0.610 | 0.620 |
| 1 neutral event | 0.570 | 0.625 | 0.595 |
| 2 neutral events | 0.570 | 0.555 | 0.570 |

Raw-trace gap0 -> gap2 drop:

```text
0.620 -> 0.570 = -0.050
```

The v5.1 spacing therefore causes only a modest additional drop. Large recurrent memory decay is not sufficient to explain the v5.1 result.

## Trace readout diagnostic

At the original v5.1 gap2 spacing:

```text
raw flattened trace = 0.570
summary features     = 0.555
final state          = 0.570
```

Raw minus summary:

```text
+0.015
```

The handcrafted summary is therefore not the primary bottleneck either.

## Controls

Reset accuracy is exactly `0.500` for all gaps and all trace views.

Wrong/opposite-arm traces generally fall below chance, for example:

```text
gap0 raw wrong = 0.380
gap2 raw wrong = 0.430
```

This is consistent with a weak real semantic signal being present, but the signal is not strong enough for the intended v5.1 claim.

## Diagnosis

Automated diagnosis:

```text
input_or_fixture_adequacy
```

The strongest current interpretation is:

> v5.1 asked one linear probe to learn a shared abstract `usable/blocked` direction across five different semantic domains, but the frozen input representation itself supports that axis only weakly on held-out paraphrases. Consequently the experiment was not an adequate clean test of recurrent semantic retention.

This does **not** prove the recurrent core is good. It only shows that input/fixture adequacy must be repaired before the recurrent trace can be judged fairly.

## Next version

v5.1.2 will be a benchmark/probe calibration version.

It will:

1. evaluate semantic decodability separately within each domain;
2. keep held-out paraphrase templates;
3. compare input embedding vs real trace vs reset/wrong trace within the same domain;
4. report macro-average across domains;
5. remain diagnostic, not confirmatory.

If this establishes an adequate protocol, a later fresh version must use a new untouched fixture for confirmatory semantic-context evidence.
