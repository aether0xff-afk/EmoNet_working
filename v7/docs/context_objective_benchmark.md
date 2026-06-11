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

## Validation-best checkpoint rule

Use the checked entrypoint below. It reloads the checkpoint with the lowest validation total before calculating the final real-history and shuffled-history metrics.

```text
training epochs
→ save best_checkpoint.pt when validation improves
→ reload best_checkpoint.pt
→ evaluate real history
→ evaluate shuffled history
```

The original runner is retained for implementation reference. New local experiments and CI use the checked runner.

## Local LM Studio run

```powershell
python experiments/run_context_objective_benchmark_checked.py `
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

## 2026-06-11 AET-25 LM Studio Multi-Seed Run

Execution:

```text
code commit: 51cf9b7a465a5d40549656a9884fbf9e106e688b
output: runs/context_objective_benchmark_lmstudio_aet25_committed
encoder: lmstudio
embedding model: text-embedding-nomic-embed-text-v1.5
base URL: https://desktop-mmlrcfk.tail93ffc6.ts.net
device: cpu
epochs: 30
seeds: 7, 13, 21, 42, 100
fixture: fixtures/context_dependence_episodes.yaml
```

Command:

```powershell
python experiments/run_context_objective_benchmark_checked.py `
  --encoder lmstudio `
  --base-url $env:EMONET_LMSTUDIO_BASE_URL `
  --embedding-model text-embedding-nomic-embed-text-v1.5 `
  --epochs 30 `
  --seeds 7 13 21 42 100 `
  --output runs/context_objective_benchmark_lmstudio_aet25_committed `
  --quiet
```

Summary:

| Model | Real context margin mean | Shuffled context margin mean | Real minus shuffled mean | Best validation total mean |
| --- | ---: | ---: | ---: | ---: |
| `snn_context_contrastive` | 0.010292 | -0.010292 | 0.020584 | 0.098093 |
| `snn_next_only` | 0.000060 | -0.000060 | 0.000121 | 0.057802 |
| `gru_context_contrastive` | 0.009593 | -0.009593 | 0.019186 | 0.090149 |
| `context_free_mlp` | 0.000000 | n/a | n/a | 0.101677 |

Interpretation:

- The contrastive SNN used prior context on this controlled fixture: its
  real-minus-shuffled context margin was positive across the five-seed run and
  far above the next-event-only SNN.
- The context-free MLP stayed at zero context margin, as expected when current
  text is identical inside each contrast pair.
- The GRU recurrent baseline remained competitive with the contrastive SNN.
  This supports the narrower claim that the fixture can reward context memory;
  it does not establish an SNN-specific advantage.
- These results remain context-objective evidence only. They do not establish
  emotional semantics, subjective feeling, or validated emotion labels.
