# EmoNet v5.9 Result — Encoder-Free Mechanistic Temporal Benchmark

검증일: 2026-08-09 KST

## Version / run

- branch: `feature/v5.9-encoder-free-mechanistic-temporal`
- PR: #21
- workflow run: `31297232597`
- job: `93204226781`
- artifact: `v5.9-encoder-free-temporal`
- artifact id: `9033417636`
- artifact digest: `sha256:a1d587d62abdba9d846b32010185652ae3f9895621d660a92f49a90651d9e89d`

No language model or semantic encoder was used. The only input source was deterministic lookup of 384-dimensional controlled vectors.

All tests passed before the benchmark:

- frozen v5.7 tests: 4 passed
- frozen v5.8 tests: 5 passed
- vector-world construction tests: 5 passed

Within every pair, A/B/C vectors were unit length and orthogonal; train/test event identities were disjoint.

## Primary leave-one-vector-world-out result

Three independent vector worlds (`101, 211, 307`) and five recurrent seeds (`7, 13, 21, 42, 100`) were evaluated across four temporal tasks.

Mean accuracy:

```text
v5.7 raw final-current trace       0.505625
v5.7 population moments           0.508542
v5.8 raw final-current trace       0.503125
v5.8 population moments           0.510833
v5.8 adaptation-state vector      0.532917
v5.8 adaptation-state moments     0.536250
adaptation-only raw               0.503125
relational validity baseline      1.000000
```

Causal controls were also at chance:

```text
v5.7 reset     0.497500
v5.7 opposite  0.494375
v5.8 reset     0.499792
v5.8 opposite  0.496875
```

Per-task raw v5.7:

```text
alternation       0.5025
palindrome        0.5000
repeat_gap        0.5083
repeat_position   0.5117
```

Per-task raw v5.8:

```text
alternation       0.5050
palindrome        0.5000
repeat_gap        0.5100
repeat_position   0.4975
```

Within-vector-world train/test was also near chance:

```text
v5.7 raw = 0.511875
v5.8 raw = 0.515000
```

Therefore failure is not caused merely by leave-one-world transfer difficulty.

## Interpretation

The frozen v5.7/v5.8 **final current-event trace does not encode identity-invariant relational temporal structure** in this controlled environment.

This strengthens the v5.8.2 conclusion: the strong raw-coordinate results in language-rendered development fixtures were not evidence for a general abstract `ABAB/AABB` representation.

Adaptation does not rescue the encoder-free final-state representation:

```text
v5.8 - v5.7 = -0.0025
v5.8 - adaptation-only ≈ 0.0
```

## Important scope correction

v5.9 tests a **compressed final-state-memory question**:

> after the history has been consumed, can the raw trace generated during one common final observation reveal the relational structure of the earlier sequence?

It does **not** directly test the original v3.1 trace hypothesis, where the relevant representation is the **full time-evolving trajectory itself**.

The benchmark currently discards the event-by-event history traces after they update recurrent state. Therefore v5.9 failure must not be generalized to:

> the full neural trajectory contains no relational temporal information.

That stronger question remains untested.

## Next version

Create **v5.9.1 — Full Trajectory Geometry Diagnostic** with the same frozen encoder-free vector worlds and dynamics.

Record all event traces instead of discarding history and test:

1. full raw episode trajectory;
2. event-trace cosine-similarity matrix;
3. event-final-state similarity matrix;
4. event-mean-state similarity matrix;
5. common-current trace only (frozen v5.9 baseline);
6. raw input-vector relational matrix (task-validity upper bound).

Use the same leave-one-vector-world-out protocol.

If trajectory self-similarity succeeds while final state remains at chance, the correct conclusion is that the current network represents temporal relations **in trajectory geometry but does not compress them into persistent final state**.

If full trajectory geometry also fails, then the fast dynamics themselves require architectural change.

No emotion/affect claim is supported by v5.9.
