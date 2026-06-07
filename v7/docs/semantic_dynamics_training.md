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
  --output runs/semantic_dynamics_lmstudio
```

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
  --output runs/context_dependence_ablation_lmstudio
```

Important metrics:

```text
trained_prediction_distance_mean
trained_latent_distance_mean
trained_context_margin_mean
persistent_minus_reset_trained_context_margin
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
