# Trace Meaning and Response-Influence Evaluation

Status date: 2026-06-11

This document defines the evaluation framework for deciding whether raw traces
are useful internal representations rather than random logs. It also defines
how semantic reports may be tested as response-conditioning inputs without
treating them as ground-truth emotion labels.

## Evaluation Questions

1. Does the same input and state policy produce structurally consistent traces?
2. Do similar or context-related inputs produce closer traces than contrastive
   inputs?
3. Does prior context change traces when the current event text is identical?
4. Does resetting, shuffling, masking, or perturbing trace/history information
   degrade downstream predictions?
5. Do self-organized or rewired structures explain trace variation better than
   fixed or context-free baselines?
6. Does a neutral semantic report change generated responses in measurable,
   bounded ways?

## Core Quantitative Metrics

Trace consistency:

- same-context trace distance: cosine and euclidean
- repeated-run latent distance under fixed seed
- spike-rate and active-edge density variance
- final-state distance: spike, membrane, adaptation

Context dependence:

- real versus shuffled trace distance
- real versus reset trace distance
- trace context gap: `real_vs_shuffled - same_context`
- context retrieval accuracy from shuffled traces
- persistent-minus-reset context margin

Semantic readability:

- targeted-axis MAE from evaluation-only probes
- targeted direction accuracy
- pair-order accuracy
- current-text leakage gap: trace probe versus text-only probe
- shuffled-history and reset-history MAE degradation

Representation ablation:

- raw pooled trace versus final state versus latent `z`
- history delta versus full representation
- SNN trace versus GRU hidden state versus context-free MLP
- memory-threshold trace versus baseline adaptive SNN trace

Response influence:

- response delta between no-report and report-conditioned generation
- report-masked response delta
- report-shuffled response degradation
- human/LLM judge dimensions: content fit, uncertainty calibration,
  overconfidence, specificity, naturalness
- cost and latency

## Qualitative Review Items

For each selected example, record:

- input event and prior context,
- expected controlled contrast,
- trace/report summary,
- generated response with report,
- generated response without report,
- whether report use improved specificity or only added decorative wording,
- whether the response over-claimed emotion labels,
- whether the response stayed faithful to the original user text.

## Existing v7 Entry Points

Context structure:

- `experiments/run_trace_context_structure_benchmark.py`
- `experiments/summarize_trace_context_structure_benchmark.py`

Semantic alignment:

- `experiments/run_trace_semantic_alignment_benchmark.py`
- `experiments/summarize_trace_semantic_alignment_benchmark.py`

Representation ablation:

- `experiments/run_trace_semantic_representation_ablation.py`
- `experiments/summarize_trace_semantic_representation_ablation.py`

History-reconstructive SNN:

- `experiments/run_history_reconstructive_snn_benchmark.py`
- `experiments/summarize_history_reconstructive_snn_benchmark.py`

Thought/report feedback plumbing:

- `experiments/run_internal_thought_ablation.py`
- `experiments/run_lmstudio_thought_feedback_suite.py`

Rewiring and community structure:

- `experiments/run_activity_guided_rewiring_pipeline.py`
- `experiments/visualize_activity_guided_rewiring_clusters.py`

## Ablation Table Draft

| Ablation | Control | Treatment | Required win condition | Failure interpretation |
| --- | --- | --- | --- | --- |
| Same input repeat | Same seed, same state policy | Repeated extraction | Near-zero trace/latent distance | Non-determinism or unstable logging path |
| Context use | Reset each transition | Persistent state | Persistent context margin and trace gap improve | State is not carrying useful context |
| History shuffle | Real prior context | Swapped prior context | Shuffled history degrades MAE/margin | Model may be using current text leakage |
| Representation mode | Current-text embedding | Raw trace, final state, latent `z` | Trace modes beat text-only on controlled context fixture | Trace is not adding information beyond text |
| SNN baseline | GRU context baseline | SNN context model | SNN is competitive or reveals different tradeoff | Context memory exists but not SNN-specific |
| Rewiring | No rewiring | Activity-guided rewiring | No material semantic regression; community diagnostics improve | Rewiring is destructive or cosmetic |
| Report influence | No semantic report | Neutral report included | Response changes are faithful, calibrated, and useful | Report is ignored or causes over-interpretation |
| Report perturbation | Correct report | Shuffled/masked report | Perturbation measurably changes response quality | Report has no causal influence on output |

## Minimum Dataset and Sampling Strategy

Start with controlled fixtures before open-ended data:

- `fixtures/context_dependence_episodes.yaml`: identical current text with
  different prior context and target.
- `fixtures/semantic_alignment_episodes.yaml`: coarse axis labels used only for
  evaluation probes.
- `fixtures/internal_thoughts.yaml`: controlled injected thought conditions.

Minimum split rule:

- train probes on train episodes only,
- evaluate on validation episodes only,
- never let the same episode ID appear in both probe train and validation rows,
- report seeds explicitly,
- report encoder backend and device policy.

Sampling for qualitative review:

- at least one high-confidence success per controlled relation,
- at least one failure per relation,
- at least one case where current text is identical but prior context changes,
- at least one case where report conditioning changes response wording,
- at least one case where report conditioning should not change the response.

## Response-Conditioning Protocol

Semantic reports are intermediate representations. They are not labels. A
response experiment should compare:

```text
direct response
direct response + neutral report
direct response + masked report
direct response + shuffled report from a different episode
```

Report text should remain neutral:

```text
active ratio
trace persistence
peak/final spike count
latent signature
context margin or trace gap summary
```

Avoid report text such as:

```text
the user is angry
the model feels abandoned
this neuron cluster means fear
```

## Decision Thresholds

Do not claim trace meaning from one metric. A trace milestone should require at
least:

- positive context gap on controlled pairs,
- shuffled/reset degradation,
- probe generalization without episode leakage,
- text-only baseline comparison,
- multi-seed stability,
- qualitative examples that do not rely on emotion-label storytelling.

Do not claim response influence unless report perturbation changes response
quality or behavior in the expected direction.

## Interpretation Boundary

Passing this framework supports the narrow claim that v7 traces carry
controlled-context information and can influence response behavior under a
specified protocol. It does not prove subjective emotion, human-like appraisal,
stable neuron meanings, or biological fidelity.
