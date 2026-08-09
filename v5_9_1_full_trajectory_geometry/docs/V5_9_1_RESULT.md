# EmoNet v5.9.1 Result — Full Trajectory Geometry Diagnostic

검증일: 2026-08-09 KST

## Version / run

- branch: `feature/v5.9.1-full-trajectory-geometry`
- PR: #22
- workflow run: `31297763595`
- job: `93205522774`
- artifact: `v5.9.1-full-trajectory-geometry`
- artifact id: `9033552536`
- artifact digest: `sha256:e4ae7095d5e30fb56cba1766c1731c542802d79a95a2d8a45fc6ac23f73bed7d`

No language encoder, semantic label, affect label, or state-generator change was introduced.

All inherited and v5.9.1 validation tests passed before the benchmark:

```text
frozen v5.7 tests              4 passed
frozen v5.8 tests              5 passed
frozen v5.9 vector-world tests 5 passed
v5.9.1 trajectory tests        4 passed
```

The pre-run computational amendment replacing the non-primary 14,336-dimensional full-episode ridge readout with a fixed 256-dimensional signed feature hash was committed before the first benchmark result. Primary trajectory features and gates were unchanged.

## Primary leave-one-vector-world-out result

Mean accuracy across:

```text
3 vector worlds × 5 recurrent seeds × 4 tasks
```

### Frozen v5.7

```text
event-trace self-similarity   0.999583
final-state similarity        0.996458
mean-state similarity         0.998333
hashed full episode           0.523125
common-current raw trace      0.505625
opposite-history self-sim     0.000417
```

### Frozen v5.8

```text
event-trace self-similarity   0.999583
final-state similarity        0.996250
mean-state similarity         0.998333
hashed full episode           0.528750
common-current raw trace      0.503125
opposite-history self-sim     0.000417
```

### Input relational upper bound

```text
raw input relational matrix   1.000000
```

## Per-task primary trajectory accuracy

v5.7:

```text
alternation       1.000000
palindrome        1.000000
repeat_gap        0.998333
repeat_position   1.000000
```

v5.8 is numerically identical on the primary readout.

Within-vector-world primary accuracy is also `0.999583` for both mechanisms, so leave-one-world-out causes essentially no loss.

## Preregistered gates

Every primary trajectory-similarity gate passed for both frozen v5.7 and frozen v5.8:

```text
macro >= 0.85                         PASS
all four tasks >= 0.80                PASS
opposite-history <= 0.20              PASS
leave-world drop <= 0.05              PASS
input relational validity >= 0.99     PASS
current-only raw <= 0.60              PASS
```

## Main result

The v5.9 failure and v5.9.1 success localize the information loss very sharply:

```text
raw input relation
        ↓
full event-by-event neural trajectory   ~100% decodable
        ↓
common final observation / compressed state   ~50% chance
```

Therefore the frozen dynamics **do preserve/re-express identity-invariant temporal relations in the geometry of the full trajectory**, while failing to compress those relations into the persistent final recurrent state tested by v5.9.

This directly supports the authentic v3.1 trajectory-first direction much more strongly than the later final-state experiments did.

## Important limitation: this is not yet a dynamics-contribution result

The primary neural trajectory score (`0.999583`) is almost identical to the raw input relational upper bound (`1.0`). v5.7 and v5.8 are also identical on the primary readout despite their different adaptation mechanism.

Therefore v5.9.1 **does not establish that recurrent dynamics create relational information beyond the input sequence**.

A simpler explanation remains viable:

> each event trace may preserve the equality/difference geometry already present in the event inputs, and the six pairwise trace similarities may therefore act as a near-isometric re-expression of the raw input relational matrix.

The result establishes trajectory-level preservation, not emergent abstraction.

The near-chance 256D hashed full-episode readout does not contradict the primary result: arbitrary coordinate hashing destroys the explicit pairwise relational operation that makes the trajectory geometry identity-invariant.

## Next version

Create **v5.9.2 — Trajectory Attribution / Input-Copy Audit** with all state generators and vector worlds frozen.

For the same four transient events, compare under the same leave-one-world-out protocol:

1. raw input-vector relational matrix;
2. actual residual-input relational matrix (`x_t - slow_(t-1)`);
3. fixed neural input-drive relational matrix (`W_in @ residual_t`);
4. full event-trace self-similarity;
5. event-final-state similarity;
6. event-isolated fast responses with history/recurrent carry removed;
7. sequential minus isolated trajectory geometry;
8. v5.7 versus v5.8.

The next question is:

> Does the neural dynamics contribute anything measurable beyond a coordinate transformation of the current/residual inputs?

If isolated/no-recurrence responses already reproduce the ~100% trajectory geometry, the current trajectory result is primarily **input-geometry preservation**.

If sequential dynamics materially outperform isolated responses on fresh worlds, then a genuine recurrent contribution is present and should be isolated in a later confirmatory version.

No emotion/affect claim is supported by v5.9.1.
