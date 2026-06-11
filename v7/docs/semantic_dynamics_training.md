# Persistent Semantic Dynamics Training

## Purpose

Move beyond single-event plumbing checks and train the SNN on short ordered episodes. The SNN state is preserved across transitions inside one episode and reset only when the next episode begins.

This stage evaluates trainability and validation behavior. It does not establish emotional semantics.

## Episode flow

```text
episode start
→ initialize SNN state
→ current event embedding
→ EventEncoder
→ differentiable SNN window
→ TraceEncoder
→ NextEventPredictor
→ next event embedding objective
→ preserve state for next transition
episode end
→ reset state for next episode
```

## Starter fixture

```text
fixtures/semantic_training_episodes.yaml
```

The fixture contains train and validation episodes across reply delays, study planning, debugging, schedule pressure, and local-server troubleshooting.

## Train with LM Studio embeddings

```powershell
python experiments/train_semantic_dynamics.py `
  --encoder lmstudio `
  --base-url http://127.0.0.1:1234 `
  --embedding-model text-embedding-nomic-embed-text-v1.5 `
  --epochs 30 `
  --device auto `
  --output runs/semantic_dynamics_lmstudio
```

## Device policy

```text
--device cpu      always run on CPU
--device cuda     use CUDA when available; otherwise fall back to CPU
--device cuda:0   use a specific CUDA device when available; otherwise fall back to CPU
--device auto     use CUDA when available; otherwise fall back to CPU
```

Add `--no-cuda-fallback` when a CUDA run must fail instead of silently using CPU. `summary.json` and `run_log.jsonl` record `requested_device`, `resolved_device`, and `used_device_fallback`.

## 2026-06-11 CUDA smoke record

Remote host `DESKTOP-MMLRCFK` was checked over SSH with an RTX 4090.

```text
GPU: NVIDIA GeForce RTX 4090
Driver: 591.86
Memory after tiny smoke: 700 MiB / 24564 MiB
Python env: C:/Users/remote/miniconda3/envs/picasso-gpu/python.exe
PyTorch: 2.11.0+cu128
```

Strict CUDA command:

```powershell
python experiments/train_semantic_dynamics.py `
  --encoder hash `
  --epochs 1 `
  --num-neurons 16 `
  --event-ticks 4 `
  --stimulation-ticks 2 `
  --device cuda `
  --no-cuda-fallback `
  --output runs/codex_aet18_cuda_strict_smoke `
  --quiet
```

Observed summary:

```text
requested_device: cuda
resolved_device: cuda
used_device_fallback: false
best_validation_total: 1.0425889492034912
```

Matched CPU command on the same host and seed produced:

```text
requested_device: cpu
resolved_device: cpu
used_device_fallback: false
best_validation_total: 1.042582909266154
```

The one-epoch hash-encoder CPU/GPU validation delta was approximately `0.00000604`. Treat this as a smoke-level device-path check only, not a numerical reproducibility claim for full training.

## Output files

```text
run_log.jsonl
embedding_cache.json
history.csv
summary.json
best_checkpoint.pt
```

## Compare persistent state against reset baseline

```powershell
python experiments/run_state_persistence_ablation.py `
  --encoder lmstudio `
  --base-url http://127.0.0.1:1234 `
  --embedding-model text-embedding-nomic-embed-text-v1.5 `
  --epochs 30 `
  --device auto `
  --output runs/state_persistence_ablation_lmstudio
```

The comparison output is saved to:

```text
runs/state_persistence_ablation_lmstudio/comparison.json
```

## Repeat the baseline across seeds

```powershell
python experiments/run_state_persistence_multiseed.py `
  --encoder lmstudio `
  --base-url http://127.0.0.1:1234 `
  --embedding-model text-embedding-nomic-embed-text-v1.5 `
  --epochs 30 `
  --seeds 7 13 21 42 100 `
  --device auto `
  --output runs/state_persistence_multiseed_lmstudio
```

Multi-seed outputs:

```text
run_log.jsonl
by_seed.csv
summary.json
seed_7/comparison.json
seed_13/comparison.json
...
```

## Evaluate ambiguity-controlled context dependence

The starter fixture does not force the model to use history. Use the controlled fixture below to test whether prior context matters when the current event text is identical.

```text
fixtures/context_dependence_episodes.yaml
```

Run both state policies and evaluate their context margins:

```powershell
python experiments/run_context_dependence_ablation.py `
  --encoder lmstudio `
  --base-url http://127.0.0.1:1234 `
  --embedding-model text-embedding-nomic-embed-text-v1.5 `
  --epochs 30 `
  --device auto `
  --output runs/context_dependence_ablation_lmstudio
```

Repeat the controlled context comparison across seeds:

```powershell
python experiments/run_context_dependence_multiseed.py `
  --encoder lmstudio `
  --base-url http://127.0.0.1:1234 `
  --embedding-model text-embedding-nomic-embed-text-v1.5 `
  --epochs 30 `
  --seeds 7 13 21 42 100 `
  --device auto `
  --output runs/context_dependence_multiseed_lmstudio
```

Important metrics:

```text
trained_prediction_distance_mean
trained_latent_distance_mean
trained_context_margin_mean
persistent_minus_reset_trained_context_margin
positive_context_margin_rate
context_margin_mean
context_margin_std
```

A positive persistent-minus-reset context margin suggests that preserved state helps distinguish identical current text under different prior contexts.

## Interpretation rule

```text
persistent best validation loss
<
reset_each_transition best validation loss
```

is evidence that preserved state contributes to the next-event objective on the tested fixture. It is not evidence that the state represents emotion.

Use the multi-seed win rate, mean reset-minus-persistent delta, and ambiguity-controlled context margin before making a structural claim.
