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

## Example input

The repository includes one clean example sentence:

```text
친구가 내 발표를 공개적으로 비웃어서 화가 났다.
```

at:

```text
v3.1/experiments/single_trace_example.csv
```

Its appraisal labels are kept only for interpretation/reporting. Run with
`--stim-source text` so those labels are not used to construct the input vector.

## Script

```text
v3.1/scripts/inspect_single_trace_formation.py
```

The script reuses the existing v3.1 runtime and does not change the neural
dynamics.

## Run

From the repository root in PowerShell:

```powershell
python -u v3.1/scripts/inspect_single_trace_formation.py `
  --input v3.1/experiments/single_trace_example.csv `
  --row-index 0 `
  --config v3.1/configs/final_dynamics_v1.json `
  --seed 42 `
  --stim-source text `
  --output-json v3.1/outputs/single_trace_formation/sample_000.json `
  --output-md v3.1/outputs/single_trace_formation/sample_000.md
```

## What is recorded

For every recorded tick the JSON/Markdown report stores:

- active neuron IDs
- newly activated/deactivated neurons
- dominant neuron and its `K`
- top neuron `K` and stimulus state
- largest observed tick-to-tick `K` changes
- edges that fired
- dominant route
- edge additions/removals after the sentence
- largest fatigue increases

This lets us read the TRACE as a chronological process instead of only a final
vector.

## Important limitation

`delta_K` is an **observed state change**, not an exact causal decomposition.
The current core does not separately log every contribution from memory,
intrinsic alignment, inhibition, fatigue and parent input before thresholding.

Therefore this first experiment answers:

> What happened, in what order, while this TRACE formed?

It does not yet fully answer:

> Exactly which internal term caused each neuron to fire?

## Next causal experiment

After locating a candidate divergence point or high-contribution neuron, repeat
the same input while perturbing one component at a time:

1. disable one candidate neuron;
2. remove one candidate edge;
3. reset memory contribution;
4. set fatigue contribution to zero;
5. compare route and TRACE distance with the original run.

That converts an observed formation path into a causal test.
