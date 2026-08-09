# EmoNet v5.10 — Hidden Prior / Same Visible Window Causal State Benchmark

## Version boundary

- base: frozen v5.9.2 attribution result
- branch: `feature/v5.10-hidden-prior-visible-window`
- no language encoder
- frozen v5.7/v5.8 dynamics and hyperparameters
- no tuning in this version
- new benchmark question, so v5.10 is a separate version rather than a v5.9 diagnostic patch

v5.9.2 showed that the near-perfect v5.9.1 visible trajectory-relational task was mostly input-geometry preservation: isolated per-event responses already solved it at `1.0`. Recurrence nevertheless produced a structured class-dependent deformation (`delta geometry ≈0.971`), but that deformation was redundant because the visible input sequence itself revealed the class.

v5.10 removes that loophole completely.

## Core question

> Can an earlier hidden relational history alter the later neural trajectory even when the entire visible input window is exactly identical across the two labels?

The downstream probe is forbidden from reading hidden-prior vectors or hidden-prior traces. It may read only neural state/trajectory during the later identical visible window.

Therefore a pure input-copy solution is chance by construction.

## Controlled vector worlds

Use three independent vector worlds:

```text
101, 211, 307
```

and frozen recurrent seeds:

```text
7, 13, 21, 42, 100
```

Input dimension remains `384`.

For every pair independently generate **nine orthonormal unit vectors**:

```text
P, Q, R       hidden-prior events
A, B, C, D    visible-window events
N             neutral delay event
Z             common final event
```

All vectors are fresh across pair IDs and independently regenerated in each vector world. Train/test pair identities never overlap.

Pair count per task/world:

```text
80
train pair IDs = 0..49
test pair IDs  = 50..79
```

## Hidden-prior tasks

### Primary — norm-matched repeat position

```text
label 0 hidden prior: P Q R P
label 1 hidden prior: Q P P R
```

Both labels contain exactly the same multiset `{P,P,Q,R}`.

For orthonormal P/Q/R and EMA decay `0.80`, the squared norm of the slow EMA state after the four hidden events is identical for these two sequences. The pair-specific delay vector N is orthogonal to P/Q/R, so after any common number of N delay events the slow-state norms remain identical.

This blocks a trivial scalar slow-memory-magnitude solution.

### Diagnostic — easy alternation

```text
label 0 hidden prior: P Q P Q
label 1 hidden prior: P P Q Q
```

Same event multiset, but slow EMA norm is not constrained to match. This task is recorded only as a calibration/diagnostic and cannot substitute for the primary norm-matched task.

## Hidden-to-visible delay

Evaluate independently at:

```text
0 neutral events
1 neutral N event
3 repeated neutral N events
```

The neutral delay sequence is identical across labels.

## Visible window — exactly identical across labels

After hidden prior and delay, both labels receive:

```text
A B C D
```

followed by the same final event:

```text
Z
```

Within a pair, **the exact same A/B/C/D/Z vectors** are used for both labels.

The probe never receives P/Q/R/N features directly.

## Frozen models

Evaluate separately:

- v5.7 residual-driven fixed recurrence + slow EMA;
- v5.8 residual-driven adaptive recurrence + same slow EMA.

Frozen parameters:

```text
slow EMA decay        = 0.80
v5.8 adaptation decay = 0.995
v5.8 strength         = 0.20
```

## Readouts

### Primary state-carry readout

**Full raw neural trajectory during visible A/B/C/D only.**

Concatenate the four visible event traces. Hidden-prior traces and delay traces are excluded.

A linear probe is fit using train pairs from two vector worlds and evaluated on test pairs from the held-out third world.

### Coordinate-invariant visible trajectory geometry

Six pairwise cosine similarities among the four visible event traces.

This is a stronger abstraction diagnostic, but it does not replace the primary raw visible-trajectory gate.

### Localization diagnostics

Immediately before A:

- raw fast recurrent state;
- raw slow EMA state;
- slow-state norm;
- for v5.8, raw adaptation state.

After A/B/C/D, on common final Z:

- raw Z-event trace (compressed-state persistence reference).

## Causal interventions

Starting from the exact state immediately after hidden prior + delay:

### Intact

Preserve fast + slow (+ adaptation for v5.8), then run A/B/C/D/Z.

### Fast reset

Reset fast recurrent state (and adaptation for v5.8) immediately before A, but preserve slow EMA state.

This isolates the **slow-memory path** into the visible window.

### Slow reset

Reset slow EMA immediately before A, but preserve fast recurrent/adaptation state.

This isolates the **fast-state path**.

### Both reset

Reset fast and slow state immediately before A.

Because A/B/C/D are identical across labels, this condition must be chance.

### Opposite hidden prior

For each held-out pair, apply the probe trained on intact trajectories to the intact visible trajectory generated from the opposite label's hidden prior while retaining the original label.

A causal history code should flip/collapse.

## Input-copy / shortcut baselines

### Visible input relational baseline

The six pairwise dot products among A/B/C/D. Because A/B/C/D are the exact same vectors for both labels, this feature is identical across labels and must be chance.

### Hidden relational validity

The six pairwise dot products among P/Q/R/P or Q/P/P/R. This feature is **not available to the visible-window probe**; it only confirms that the hidden task itself is structurally valid and should be near 1.0.

### Slow-state norm shortcut check

On the primary norm-matched task, pre-visible slow-state norm must remain at chance under leave-one-world-out evaluation.

## Primary protocol

Primary evaluation is leave-one-vector-world-out.

For each:

```text
model × recurrent seed × hidden task × delay × held-out world
```

fit a ridge probe on train pairs from the other two worlds and evaluate on test pairs from the held-out world.

No held-out world contributes to fitting.

## Predeclared primary gate

Evaluate this gate on **primary norm-matched hidden prior only**, using intact raw visible A/B/C/D trajectory.

A model qualifies as a `causal_state_carry_candidate` only if all hold:

```text
macro accuracy across delays >= 0.70
at least 2/3 delay conditions >= 0.65
both-reset accuracy <= 0.55
opposite-hidden-prior accuracy <= 0.30
visible-input relational baseline <= 0.55
hidden relational validity >= 0.99
slow-state-norm baseline <= 0.55
```

The coordinate-invariant visible self-similarity readout qualifies separately if:

```text
macro >= 0.65
both-reset <= 0.55
opposite-hidden-prior <= 0.35
```

## State-path localization

These are diagnostic classifications, not pass-gate substitutions.

For a model that passes intact raw visible trajectory:

```text
slow-path evidence:
    fast-reset (slow preserved) >= 0.65

fast-path evidence:
    slow-reset (fast preserved) >= 0.65

both paths independently useful:
    both of the above >= 0.65
```

If only the slow path survives, the result should be attributed primarily to the EMA memory rather than recurrent neural state.

If only the fast path survives, recurrence/adaptation carries the hidden prior independently of slow EMA.

## Adaptation comparison

v5.8 adaptation is considered to add meaningful hidden-state value only if:

```text
v5.8 intact raw macro - v5.7 intact raw macro >= 0.03
```

## Claim boundary

A successful v5.10 result would establish that internal state changes a later identical-input neural trajectory and therefore contributes information beyond the currently visible input sequence in this controlled vector environment.

It would **not** yet establish emotion semantics, language robustness, consciousness, or downstream behavioral usefulness.
