# EmoNet v5.8 Result — Activity-Dependent Adaptive Fast Dynamics

검증일: 2026-08-09 KST

## Version / run

- branch: `feature/v5.8-adaptive-fast-dynamics`
- PR: #18
- workflow run: `31296139203`
- job: `93201428590`
- artifact: `v5.8-adaptive-fast-dynamics-benchmark`
- artifact id: `9033043696`
- artifact digest: `sha256:703262b8e3f899af64ac7168033df9fa564f2722a5745d3b33935a1b0b3c06b6`

All inherited and v5.8 protocol tests passed before the benchmark:

- frozen v5.7 tests: 4 passed
- frozen v5.7.1 geometry tests: 3 passed
- v5.8 boundary/isolation tests: 5 passed
- `beta=0` maximum trace difference from v5.7: exactly `0.0`

## Train-only adaptation selection

No held-out test identity was used in selection.

Selected on calibration seed `31415`:

```text
adaptation_decay    = 0.995
adaptation_strength = 0.20
validation macro    = 0.64375
```

The candidate was selected using the preregistered permutation-invariant population-moment readout, **not** raw neuron coordinates.

## Preregistered primary result

Primary readout: per-tick population moments from the common-current fast trace.

```text
adaptive recurrent      0.61375
frozen v5.7             0.643125
slow EMA                0.80000
adaptation-only         0.564375
fast reset              0.50125
direct residual history 1.00000
relational validity     1.00000
```

Per-task primary adaptive accuracy:

```text
alternation       0.590
palindrome        0.605
repeat_gap        0.630
repeat_position   0.630
```

Only reset sensitivity, relational validity, and beta=0 reproduction gates passed. Therefore:

**v5.8 FAILS its preregistered primary development gate.**

This result must not be retroactively reclassified as a pass.

## Unexpected secondary result

The same adaptive fast traces were also recorded in raw neuron coordinates before the result was known.

Mean held-out raw-coordinate accuracy:

```text
adaptive raw trace = 0.86375
```

Per-task raw-coordinate means across five seeds:

```text
alternation       0.7075
palindrome        0.9375
repeat_gap        0.9275
repeat_position   0.8825
```

Per-seed macro raw-coordinate accuracy:

```text
seed 7    0.821875
seed 13   0.900000
seed 21   0.862500
seed 42   0.853125
seed 100  0.881250
```

This is qualitatively different from v5.7.1, where raw coordinates and all tested permutation-invariant geometry stayed near chance.

## Interpretation boundary

The unexpected raw result is **diagnostic evidence, not a v5.8 pass**.

A plausible interpretation is that activity-dependent adaptation creates a distributed neuron-specific temporal code that is destroyed when the trace is reduced to population mean/std/energy statistics.

However v5.8 did not include the controls needed to establish this interpretation:

- raw v5.7 comparison on the exact four-task suite;
- raw adaptation-only comparison;
- raw fast-reset intervention;
- direct adaptation-state readout;
- neuron-coordinate permutation controls;
- terminal-state vs full-current-trajectory readouts.

These are delegated to a new frozen-state diagnostic version, v5.8.1.

No affect/emotion claim is supported by this result.
