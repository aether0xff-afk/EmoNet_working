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

## Interpretation rule

```text
persistent best validation loss
<
reset_each_transition best validation loss
```

is evidence that preserved state contributes to the next-event objective on this fixture. It is not evidence that the state represents emotion.

Use the multi-seed win rate and the mean reset-minus-persistent delta before making a structural claim.
