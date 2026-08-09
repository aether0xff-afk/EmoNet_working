# EmoNet v5.8 — Activity-Dependent Adaptive Fast Dynamics

## Version boundary

- base: frozen v5.7.1 fast-trace diagnosis
- branch: `feature/v5.8-adaptive-fast-dynamics`
- previous versions/results remain frozen
- **slow EMA memory is unchanged** (`decay=0.80`)
- **residual input is unchanged**: `r_t = embedding_t - slow_(t-1)`
- no emotion labels, valence/arousal labels, task labels, or semantic-state labels enter state generation

This is a **development version**, not confirmatory evidence.

## Why change the fast dynamics?

v5.7.1 ruled out a simple raw-coordinate readout explanation:

- raw recurrent trace: `0.5625`
- best permutation-invariant trace geometry: `0.5750`
- direct causal residual-change sequence: `1.0000`

The useful temporal structure is present before recurrence but is not robustly preserved/exposed by the current fixed random leaky-tanh fast dynamics.

## Mechanism

v5.8 adds only one biologically motivated computation: **activity-dependent adaptation / fatigue**.

For fast neural state `h` and nonnegative adaptation state `a`:

```text
pre_t = W_rec h_(t-1) + W_in r_t - beta * a_(t-1)
candidate_t = tanh(pre_t)
h_t = (1-rate) h_(t-1) + rate * candidate_t
a_t = decay * a_(t-1) + (1-decay) * |h_t|
```

Recently active neurons therefore respond less strongly for a while. Because different residual directions excite different fixed random neuron subsets, repeated/returning events can produce stimulus-specific adaptation without assigning any emotion meaning to neurons.

`reset_fast()` resets **both** `h` and `a`. Slow memory is reset separately.

## Fairness

For a given seed, v5.8 reuses exactly the same seeded recurrent and input matrices as v5.7. Setting `beta=0` must reproduce the v5.7 fast dynamics numerically.

Adaptation hyperparameters are selected using **training identities only** with a dedicated calibration seed that is not one of the five evaluation seeds. Held-out test identities are not used for selection.

Candidate grid, fixed before test:

```text
adaptation decay:    [0.970, 0.985, 0.995]
adaptation strength: [0.20, 0.50, 0.80]
```

Evaluation seeds remain:

```text
7, 13, 21, 42, 100
```

## Identity-disjoint temporal suite

Each task uses pair-specific pseudo-event identities. Train and test identities never overlap. Within each pair the competing classes use the same event multiset whenever possible, so identity or bag-of-events shortcuts are minimized.

1. **alternation** — `ABAB` vs `AABB`
2. **palindrome** — `ABBA` vs `AABB`
3. **repeat_gap** — `ABCA` vs `AABC`
4. **repeat_position** — `ABCA` vs `ABAC`

The current observation after the sequence is identical across classes.

A deterministic relational baseline must validate that each task is structurally solvable on held-out identities.

## Mandatory comparisons

- v5.7 residual-driven recurrent fast
- v5.8 adaptive recurrent fast
- v5.8 with `beta=0` ablation
- adaptation-only dynamics with recurrent matrix removed
- slow EMA
- fast reset before the common current observation
- direct causal residual-change baseline
- relational structure-validity baseline

## Predeclared development gates

All primary gates must pass:

```text
temporal-suite macro >= 0.70
adaptive recurrent >= v5.7 residual recurrent + 0.10
adaptive recurrent >= slow EMA + 0.12
fast reset reduces adaptive score by >= 0.10
at least 3/4 tasks >= 0.65
relational validity macro >= 0.95
beta=0 reproduces v5.7 within numerical tolerance
```

Complexity is a separate question:

```text
adaptive recurrent > adaptation-only by >= 0.03
```

If adaptation-only matches or beats the recurrent system, the result does **not** justify recurrence.

## Claim boundary

A pass would only show that activity-dependent adaptation is a better label-free temporal-state mechanism on identity-disjoint structural sequences. It would not establish emotion semantics, consciousness, or superiority over general sequence models.
