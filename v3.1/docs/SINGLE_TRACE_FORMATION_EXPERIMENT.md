# Single-Sentence TRACE Formation Experiment

## Goal

Observe **how one neural TRACE forms**, rather than only comparing the final
TRACE with other samples.

The experiment follows one sentence through EmoNet and records the internal
sequence:

```text
sentence
  -> stimulus vector
  -> tick 0 active neurons
  -> tick 1 propagation / fired edges
  -> tick 2 changed activation strengths
  -> ...
  -> dominant route
  -> persistent fatigue / rewiring changes
  -> final TRACE summary
```

## Script

```text
v3.1/scripts/inspect_single_trace_formation.py
```

The script reuses the existing v3.1 runtime. It does not alter the neural
dynamics.

## Default run

From the repository root:

```powershell
python -u v3.1/scripts/inspect_single_trace_formation.py `
  --input v3.1/outputs/targeted_records_trace_normalized.csv `
  --row-index 0 `
  --config v3.1/configs/final_dynamics_v1.json `
  --seed 42 `
  --output-json v3.1/outputs/single_trace_formation/sample_000.json `
  --output-md v3.1/outputs/single_trace_formation/sample_000.md
```

## Outputs

### JSON

`sample_000.json` contains machine-readable data for every recorded tick:

- active neuron IDs
- newly activated/deactivated neuron IDs
- dominant neuron and its `K`
- top active neurons and `stim_vec`
- largest observed `K` changes
- fired edges
- dominant route
- edge additions/removals after the sentence
- largest fatigue increases

### Markdown

`sample_000.md` is the same run rendered as a human-readable chronological
report. This is the easiest file to read first.

## Interpretation

The experiment answers questions such as:

- Which neurons become active first?
- At which tick does a stable dominant route appear?
- Which edges actually carry activity?
- Which neurons increase or decrease activation strength most strongly?
- What persistent changes remain after the sentence?

It does **not yet** answer the stronger causal question:

> Exactly how much of neuron 83's activation was caused by memory, intrinsic
> alignment, inhibition, fatigue, or parent input?

The current core stores the resulting node state and fired edges, but does not
log every pre-threshold contribution as separate additive terms. Therefore the
first experiment is deliberately observational.

## Next causal experiment

After identifying a candidate divergence point or high-contribution neuron,
repeat the same sentence while perturbing one component at a time, for example:

1. disable one candidate neuron;
2. remove one candidate edge;
3. reset memory contribution;
4. set fatigue contribution to zero;
5. compare the resulting route and TRACE distance with the unmodified run.

This turns an observed formation path into a causal test.
