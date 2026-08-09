# EmoNet v5.8.1 — Distributed Temporal Code Diagnostic

## Version boundary

- base: frozen v5.8 result
- state generator, adaptation parameters, encoder, slow memory, temporal suite, and evaluation seeds are unchanged
- **diagnostic only**; no new architecture or tuning

Fixed v5.8 parameters:

```text
adaptation_decay    = 0.995
adaptation_strength = 0.20
slow_decay          = 0.80
seeds               = 7, 13, 21, 42, 100
```

## Why this diagnostic exists

v5.8 failed its preregistered population-moment primary gate (`0.61375`) but unexpectedly produced `0.86375` held-out accuracy from the same adaptive traces in raw neuron coordinates.

v5.8.1 asks whether this is a genuine stable distributed neural code caused by activity-dependent adaptation.

## Readouts / interventions

All use the exact v5.8 state generator.

- adaptive full raw current-event trace
- frozen v5.7 full raw trace
- adaptation-only full raw trace (recurrent matrix removed)
- adaptive fast-reset raw trace
- opposite-class history with the same pair identities/current observation
- final fast state only
- mean fast state over the current event
- raw adaptation-state vector
- adaptation-state population moments
- joint fixed neuron permutation applied to train and test
- test-only neuron permutation

## Expected diagnostic pattern

Evidence for a stable distributed adaptation code requires:

```text
adaptive raw >= 0.80
adaptive raw - v5.7 raw >= 0.10
adaptive raw - adaptation-only raw >= 0.03
adaptive raw - fast-reset raw >= 0.15
opposite-history accuracy <= 0.35
joint train+test neuron permutation changes accuracy by <= 0.01
test-only neuron permutation reduces accuracy by >= 0.15
```

The final-state/adaptation-state readouts are localization diagnostics, not pass gates.

A successful diagnostic still does not make v5.8 a preregistered pass. It only justifies a **fresh confirmatory version** with new temporal patterns and new event identities.

No emotion/affect claim is permitted.
