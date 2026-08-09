# EmoNet v5.9.2 Result — Trajectory Attribution / Input-Copy Audit

검증일: 2026-08-09 KST

## Version / run

- branch: `feature/v5.9.2-trajectory-attribution-audit`
- PR: #23
- formal workflow run: `31298459516`
- job: `93207232806`
- artifact: `v5.9.2-trajectory-attribution`
- artifact id: `9033774813`
- artifact digest: `sha256:6fbff763fab206e28d11a1b25da4243eb1f1b4b5aeb078e8f03b9ab7d4ebe247`

The earlier slow implementation run was excluded from scientific interpretation. Before the formal run, the isolated-response simulator was optimized to reuse the exact same seeded dynamics objects and reset only their state between events. Weight identity was explicitly checked; the protocol and features did not change.

All inherited and attribution-construction tests passed before the formal benchmark.

## Attribution ladder — mean leave-one-vector-world-out accuracy

```text
raw input relational matrix          1.000000
sequential residual relational       1.000000
fixed W_in drive relational          1.000000

v5.7 isolated residual response      1.000000
v5.7 isolated raw-input response     1.000000
v5.7 sequential trajectory           0.999583
v5.7 sequential - isolated delta     0.971250
v5.7 common-current compressed trace 0.505625

v5.8 isolated residual response      1.000000
v5.8 isolated raw-input response     1.000000
v5.8 sequential trajectory           0.999583
v5.8 sequential - isolated delta     0.971458
v5.8 common-current compressed trace 0.503125
```

Opposite-history control:

```text
v5.7 sequential opposite = 0.000417
v5.8 sequential opposite = 0.000417
```

Paired sequential-vs-isolated geometry:

```text
v5.7 cosine = 0.951206
v5.7 L2     = 0.401879

v5.8 cosine = 0.950419
v5.8 L2     = 0.406221
```

Mean v5.7-vs-v5.8 sequential-geometry L2 distance:

```text
0.042759
```

## Formal diagnosis

The runner classified both frozen mechanisms as:

```text
v5.7 copy_like_preservation           TRUE
v5.8 copy_like_preservation           TRUE
v5.7 recurrent_essential              FALSE
v5.8 recurrent_essential              FALSE
v5.7 recurrent_modulation_detectable  TRUE
v5.8 recurrent_modulation_detectable  TRUE
```

## Main conclusion

The near-perfect v5.9.1 trajectory result **does not require event-to-event recurrence or adaptation**.

The relevant relational structure is already perfectly decodable at every earlier stage:

```text
raw event vectors
    -> residual vectors
    -> W_in neural drives
    -> isolated per-event neural responses
    -> sequential neural trajectory
```

Even when the fast recurrent/adaptation state is reset before every transient event, pairwise neural-trace geometry still solves the four tasks at `1.0`. Therefore the v5.9.1 ~99.96% score is primarily **input-geometry preservation / re-expression**, not evidence that recurrence creates the relation.

This is the correct interpretation of the v3.1 trajectory success on this benchmark.

## But recurrence is not doing nothing

Although recurrence is unnecessary for solving the task, it produces a highly structured and identity-invariant deformation of trajectory geometry.

The six-dimensional feature

```text
sequential_trace_geometry - isolated_trace_geometry
```

alone predicts the temporal class at:

```text
v5.7 = 0.971250
v5.8 = 0.971458
```

Sequential and isolated geometry are similar but not identical (`cosine ≈ 0.95`, `L2 ≈ 0.40`). Thus recurrence contributes a reproducible class-structured modulation.

However this modulation is **redundant on the current benchmark**, because the visible four-event input relations already solve the class perfectly without memory.

Adaptation again adds little to the relational geometry: v5.7 and v5.8 sequential trajectories are very close (`mean L2 ≈ 0.0428`).

## Scientific consequence

The next benchmark must make **input-copy solutions impossible by construction**.

The visible trajectory presented to the evaluator must be exactly identical across the two classes. Only an earlier hidden prior history may differ.

Then:

```text
visible input relation = identical across labels -> chance by construction
```

and any label information found in the subsequent visible neural trajectory must have arrived through persistent internal state.

## Next version

Create **v5.10 — Hidden Prior / Same Visible Window Causal State Benchmark**.

Recommended structure:

```text
hidden prior class 0: P Q P Q
hidden prior class 1: P P Q Q

(optional neutral delay)

visible window, identical in both classes: A B C D
common final observation
```

All P/Q/A/B/C/D vectors are fresh controlled vectors with train/test identity separation and independent vector worlds.

The probe may read only the neural trajectory during the identical visible window. It may not read hidden-prior traces or vectors.

Required controls:

- intact state;
- fast reset before visible window (slow memory preserved);
- slow reset before visible window (fast recurrent/adaptation state preserved);
- both reset;
- event-isolated visible responses;
- opposite hidden prior with same visible window;
- visible input relational baseline, which must remain at chance;
- compressed final-state readout alongside full visible trajectory.

Use multiple hidden-to-visible delays to measure persistence.

This is the first clean test of whether EmoNet internal state contributes information **beyond the currently visible input trajectory**.

No emotion/affect claim is supported by v5.9.2.
