# EmoNet v5.8.1 Result — Distributed Temporal Code Diagnostic

검증일: 2026-08-09 KST

## Version / run

- branch: `feature/v5.8.1-distributed-code-diagnostic`
- PR: #19
- workflow run: `31296507497`
- job: `93202344422`
- artifact: `v5.8.1-distributed-code-diagnostic`
- artifact id: `9033156333`
- artifact digest: `sha256:59a72f008786739c0194cf42172bad70c41589c51cc137032bb0305bbc459e61`

The v5.8 state generator and its selected adaptation parameters were frozen:

```text
adaptation_decay    = 0.995
adaptation_strength = 0.20
slow_decay          = 0.80
```

All inherited and diagnostic-construction tests passed before the benchmark.

## Mean held-out results

| Readout / intervention | Accuracy |
| --- | ---: |
| adaptive full raw trace | **0.86375** |
| frozen v5.7 full raw trace | **0.831875** |
| adaptation-only raw trace | 0.783125 |
| final fast state | 0.698125 |
| mean fast state | 0.746875 |
| adaptation-state vector | **0.898125** |
| adaptation-state population moments | 0.535625 |
| adaptive fast-reset raw trace | 0.515625 |
| opposite-class history raw trace | **0.13625** |
| test-only neuron permutation | 0.49625 |
| joint train+test neuron permutation | **0.86375** |

Adaptive raw by task:

```text
alternation       0.7075
palindrome        0.9375
repeat_gap        0.9275
repeat_position   0.8825
```

Frozen v5.7 raw by task:

```text
alternation       0.6275
palindrome        0.9450
repeat_gap        0.9000
repeat_position   0.8550
```

## What is supported

The high raw-coordinate signal is real and history-causal on this development suite:

- resetting fast state collapses `0.86375 -> 0.515625`;
- feeding the opposite class history with the same pair identities/current collapses accuracy to `0.13625`;
- a test-only neuron permutation collapses to `0.49625`;
- applying the same neuron permutation to train and test preserves exactly `0.86375`.

Therefore the probe uses a **stable neuron-coordinate-specific distributed code**, not merely global activation magnitude.

The adaptation state itself is even more decodable (`0.898125`), while four global adaptation moments are near chance (`0.535625`). This also supports a distributed neuron-specific adaptation pattern rather than a scalar fatigue magnitude explanation.

The full current-event trajectory (`0.86375`) is more informative than the mean state (`0.746875`) or final state (`0.698125`).

## What is NOT supported

The preregistered condition that adaptation should beat frozen v5.7 raw by at least 10 percentage points failed:

```text
adaptive - v5.7 raw = +0.031875
```

Thus **v5.8.1 does not establish activity-dependent adaptation as the main source of the distributed temporal code**.

On the exact four-task suite, frozen v5.7 recurrence already carries substantial raw-coordinate temporal information (`0.831875`). Adaptation contributes modestly on average and its contribution varies by task:

- alternation: +0.0800
- palindrome: -0.0075
- repeat_gap: +0.0275
- repeat_position: +0.0275

This also reveals a benchmark-comparability issue: the earlier v5.7.1 identity-disjoint `ABAB vs AABB` fixture yielded raw accuracy around `0.5625`, whereas the new alternation fixture gives frozen v5.7 `0.6275` under a very similar logical pattern. The logical sequence is the same, but wrapper/pseudo-token rendering differs.

## Next version

Before any confirmatory adaptation claim, create **v5.8.2 benchmark-equivalence diagnostic**.

It must freeze both v5.7 and v5.8 dynamics and compare old/new alternation fixtures under one evaluator, including:

1. within-old and within-new held-out accuracy;
2. old-train -> new-test transfer;
3. new-train -> old-test transfer;
4. wrapper-swapped fixtures using the same underlying pseudo-event identities;
5. embedding-similarity diagnostics for old/new rendered events;
6. opposite-history/reset controls under both renderers.

The goal is to determine whether the observed score drift reflects real temporal-code generalization or sensitivity to superficial language rendering.

No emotion/affect claim is supported by this result.
