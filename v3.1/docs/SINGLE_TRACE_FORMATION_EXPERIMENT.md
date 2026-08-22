# Single-Sentence TRACE Formation Experiment

## Goal

Observe **how one neural TRACE forms**, rather than only comparing the final
TRACE with other samples.

The experiment follows one sentence through EmoNet and records the internal
sequence:

```text
sentence
  -> stimulus vector
  -> tick-by-tick active neurons
  -> fired edges / K changes
  -> observed max-K route
  -> persistent fatigue / rewiring changes
  -> final TRACE summary
```

## Example input

The repository includes one clean, label-free example sentence:

```text
친구가 내 발표를 공개적으로 비웃어서 화가 났다.
```

at:

```text
v3.1/experiments/single_trace_example.csv
```

The CSV contains only `record_id,text`. It deliberately contains no valence,
arousal, target, control or other appraisal labels.

## Script

```text
v3.1/scripts/inspect_single_trace_formation.py
```

The script reuses the existing v3.1 runtime and does not change the neural
dynamics.

## Lightweight smoke run

The GitHub Actions smoke test uses `--stim-source proxy` so the experiment is
self-contained and can exercise the neural-dynamics/TRACE observability path
without downloading or fitting the learned text encoder.

From the repository root in PowerShell:

```powershell
python -u v3.1/scripts/inspect_single_trace_formation.py `
  --input v3.1/experiments/single_trace_example.csv `
  --row-index 0 `
  --config v3.1/configs/final_dynamics_v1.json `
  --seed 42 `
  --stim-source proxy `
  --output-json v3.1/outputs/single_trace_formation/sample_000.json `
  --output-md v3.1/outputs/single_trace_formation/sample_000.md
```

This smoke run proves that we can observe the internal formation process. It is
**not** evidence that the sentence semantics alone produced a particular
emotion TRACE. A semantic experiment should repeat the same inspector with the
learned text stimulus encoder (or another frozen semantic encoder).

## What is recorded

For every recorded tick the JSON report stores:

- active neuron IDs
- newly activated/deactivated neurons
- max-K (observed dominant) neuron and its `K`
- top neuron `K` and local stimulus state
- comparable tick-to-tick `K` changes
- all fired edges
- an observed max-K route reconstructed from TickRecord
- the existing exporter dominant route for comparison
- edge additions/removals after the sentence
- largest fatigue increases

The Markdown report renders a compact subset so a person can read the process
without dumping thousands of fired edges per tick.

## Important limitations found by the first run

1. `delta_K` is an **observed state change**, not an exact causal decomposition.
   The current core does not separately log every contribution from memory,
   intrinsic alignment, inhibition, fatigue and parent input before thresholding.
2. In the first smoke run, the existing `dominant_branch_ids()` helper returned
   only `-1`. The raw TickRecord still contained clear max-K neurons, so the
   inspector now reports both the exporter route and a direct observed max-K
   route. This is a useful diagnostic finding, not something to hide.
3. The smoke run uses `proxy` stimulus generation. Use a frozen semantic text
   encoder before making claims about the meaning of a particular sentence.

Therefore this first experiment answers:

> What happened, in what order, while this TRACE formed?

It does not yet fully answer:

> Exactly which internal term caused each neuron to fire, and did the sentence's
> semantic meaning specifically cause that route?

## Next causal experiment

After locating a candidate divergence point or high-contribution neuron, repeat
a matched pair from the same initial seed and perturb one component at a time:

1. disable one candidate neuron;
2. remove one candidate edge;
3. reset memory contribution;
4. set fatigue contribution to zero;
5. compare route and TRACE distance with the original run.

That converts an observed formation path into a causal test.
