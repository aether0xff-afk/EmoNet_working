# EmoNet v5.7.1 — Fast Trace Geometry Diagnostic

## Version boundary

- base: frozen `feature/v5.7-residual-fast-dynamics`
- this version is diagnostic only
- **v5.7 residual-driven state generation is unchanged**
- no learning objective, topology, slow-memory rule, or task fixture is changed

## Question

v5.7 showed that the label-free residual-change signal solves the identity-disjoint temporal structure task, while the flattened recurrent trace is near chance.

This diagnostic asks:

> Did the recurrent dynamics destroy the useful change signal, or is that signal still present in permutation-invariant trace geometry that a linear probe on raw neuron coordinates does not expose?

## Readouts

All readouts are deterministic and label-free before the downstream ridge probe.

1. raw neuron coordinates — frozen v5.7 baseline
2. activation energy trajectory — per tick RMS activation
3. state-change energy trajectory — per tick RMS `h_t - h_{t-1}`
4. population moments — per tick mean/std/mean-absolute/RMS
5. full geometry — concatenation of activation energy, change energy, and population moments
6. current residual vector
7. current residual norm
8. full causal residual-change baseline from v5.7
9. relational structure validity baseline

## Interpretation

- If trace geometry is strong while raw coordinates are weak, v5.7 mainly has a **readout / coordinate-system problem**.
- If both raw coordinates and geometry remain weak while direct residual-change is strong, the **current recurrent dynamics fail to preserve the useful invariant structure**.

No result from this version can retroactively turn v5.7 into a pass.
