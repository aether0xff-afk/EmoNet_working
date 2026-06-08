# Context Objective Benchmark

## Purpose

Strengthen and measure context-sensitive memory before adding self-organizing clusters or rewiring.

The benchmark compares four models on ambiguity-controlled episodes where the current event text is identical but prior context and the correct next event differ.

## Compared models

```text
snn_next_only
→ persistent SNN trained with the original next-event objective

snn_context_contrastive
→ persistent SNN trained with next-event objective + context-ranking objective

context_free_mlp
→ current text embedding only; cannot use prior context

gru_context_contrastive
→ standard GRU recurrent baseline trained with context-ranking objective
```

## Shuffled-history evaluation

For recurrent models, validation is evaluated twice.

```text
real history
→ use the correct prior context

shuffled history
→ swap prior contexts inside each contrast pair
```

Important metric:

```text
real_minus_shuffled_context_margin
```

A positive value suggests that the model relies on the correct prior context rather than residual state noise alone.

## Context-ranking objective

The benchmark adds a pairwise ranking objective.

```text
same current event
+
different prior context
→ prefer own contextual target
→ reject opposite contextual target
```

Current experimental hyperparameters:

```text
context_weight = 1.0
context_margin = 0.05
```

These are exposed as experiment arguments. They are not treated as biological constants or final architecture choices.

## Local LM Studio run

```powershell
python experiments/run_context_objective_benchmark.py `
  --encoder lmstudio `
  --base-url http://127.0.0.1:1234 `
  --embedding-model text-embedding-nomic-embed-text-v1.5 `
  --epochs 30 `
  --seeds 7 13 21 42 100 `
  --output runs/context_objective_benchmark_lmstudio
```

## Output files

```text
run_log.jsonl
embedding_cache.json
by_seed_model.csv
summary_by_model.csv
metadata.json
seed_7/<model_type>/history.csv
seed_7/<model_type>/best_checkpoint.pt
seed_7/<model_type>/summary.json
...
```

## Interpretation order

1. `snn_context_contrastive` context margin should exceed `snn_next_only`.
2. `snn_context_contrastive` real-history margin should exceed its shuffled-history margin.
3. `context_free_mlp` should remain near zero context margin because its current text is identical across pairs.
4. Compare SNN and GRU. If GRU performs similarly, context memory is validated but no SNN-specific advantage has been established.
5. Only after stable multi-seed results should cluster formation and rewiring experiments begin.

## Scope boundary

This benchmark tests context-sensitive prediction. It does not establish emotional semantics, human-like appraisal, or biological fidelity.
