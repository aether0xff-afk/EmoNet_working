# LM Studio Thought Feedback Suite

## Purpose

Run repeated generated-thought feedback experiments against the local LM Studio server and summarize trace variability by controlled prompt condition.

This suite measures plumbing behavior. It does not establish emotional semantics.

## Models used

```text
chat model:
gemma-4-26b-a4b-it-qat

embedding model:
text-embedding-nomic-embed-text-v1.5
```

## Pull the latest branch

```powershell
git pull
```

## Run one LM Studio-only feedback check

```powershell
python experiments/run_lmstudio_thought_feedback.py `
  --base-url http://127.0.0.1:1234 `
  --chat-model gemma-4-26b-a4b-it-qat `
  --embedding-source lmstudio `
  --embedding-model text-embedding-nomic-embed-text-v1.5 `
  --output runs/lmstudio_thought_feedback_local_embeddings
```

## Run repeated controlled conditions

```powershell
python experiments/run_lmstudio_thought_feedback_suite.py `
  --base-url http://127.0.0.1:1234 `
  --chat-model gemma-4-26b-a4b-it-qat `
  --embedding-model text-embedding-nomic-embed-text-v1.5 `
  --runs-per-condition 3 `
  --output runs/lmstudio_thought_feedback_suite
```

## Output files

```text
runs.csv
runs.jsonl
summary.csv
metadata.json
```

## Controlled conditions

```text
open
reassurance
negative_interpretation
uncertainty
```

The conditions are prompt interventions. Differences between them must not be interpreted as spontaneously emerged emotional categories.
