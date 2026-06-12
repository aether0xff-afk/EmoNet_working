# EmoNet v7 Implementation Spec and Decision Log

Status date: 2026-06-12

This document is the working contract for the v7 rebuild. It describes what is
implemented, what the current experiments are allowed to claim, and which design
questions are still open.

## Purpose

EmoNet v7 is a conservative rebuild around a sparse recurrent spiking neural
network. The current research question is:

```text
Can ordered text events produce persistent, trainable neural dynamics that carry
context across an episode without assigning predefined emotion labels?
```

The current implementation supports that question through short ordered
episodes, frozen text embeddings, recurrent SNN state, trace latents,
self-supervised next-event training, context-dependence checks, and controlled
rewiring ablations.

## Claim Boundary

Supported claims:

- The v7 pipeline can encode text events into SNN currents, run adaptive sparse
  recurrent dynamics, and compress raw traces into latent vectors.
- Persistent episode training is implemented with validation-best checkpointing.
- CPU and CUDA experiment entrypoints share an explicit device policy.
- Context-dependence and rewiring experiments exist as controlled ablations.
- Internal-thought feedback can be tested through neutral SNN state reports and
  local LLM-generated thoughts.

Unsupported claims:

- The system has not demonstrated validated emotional meaning.
- Neuron groups are not predefined or verified emotion clusters.
- Rewiring is not a final biological rule.
- Current fixture results should not be treated as broad generalization.
- CPU/GPU smoke checks are device-path checks, not full numerical equivalence
  guarantees.

## Architecture

```text
Event
-> text embedding backend
-> EventEncoder
-> event current
-> AdaptiveSparseRSNN or NeuronMemoryThresholdRSNN
-> tick traces / differentiable window
-> TraceEncoder
-> latent z
-> objective head, state report, or experiment metric
```

Core modules:

- `src/emonet_v7/schemas.py` defines the event contract.
- `src/emonet_v7/event_encoder.py` maps frozen embeddings and metadata into
  neuron currents.
- `src/emonet_v7/adaptive_rsnn.py` implements the baseline ALIF-style sparse
  recurrent SNN.
- `src/emonet_v7/training_window.py` keeps training windows differentiable.
- `src/emonet_v7/trace_encoder.py` compresses trace sequences into latent
  vectors.
- `src/emonet_v7/self_supervised.py` defines the next-event training objective.
- `src/emonet_v7/context_objective.py` defines context-ranking checks and
  comparison baselines.
- `src/emonet_v7/memory_threshold_rsnn.py` isolates neuron-local memory
  threshold experiments from the baseline SNN.
- `src/emonet_v7/activity_guided_rewiring.py` implements the current controlled
  rewiring ablation.
- `src/emonet_v7/thought_module.py` and `src/emonet_v7/state_bridge.py` bridge
  neutral state reports into local LLM thought experiments.

## Event Schema

The canonical event object is:

```text
event_id: str
kind: user_message | internal_thought | module_message | elapsed_time
text: str
speaker_id: str
elapsed_seconds: float
```

The schema intentionally keeps the event neutral. It records source and timing
metadata but does not add emotion labels, appraisal axes, valence fields, or
cluster IDs.

## Neural Substrate

The baseline substrate is `AdaptiveSparseRSNN`:

- ALIF-style membrane, spike, adaptation, and threshold state.
- Deterministic sparse directed recurrent mask without self-loops.
- Trainable input and recurrent weights.
- Per-tick recurrent input from previous spikes plus external event current.
- Detached tick traces for inspection and differentiable windows for training.

The memory-threshold variant is `NeuronMemoryThresholdRSNN`:

- It keeps fast spiking state separate from neuron-local accumulation and
  persistent memory strength.
- Memory consolidation happens at event boundaries, not every tick.
- It remains an ablation substrate and downstream rewiring testbed, not the
  primary substrate contract.

## Trace and State Reports

Raw trace snapshots record:

```text
tick
membrane
spike
adaptation
threshold
active_edges
```

Memory-threshold traces additionally record:

```text
accumulation
memory_strength
```

Neutral LLM-facing reports are produced by `build_neutral_state_report` and
include:

```text
active_ratio
trace_persistence
peak_spike_count
final_spike_count
latent_signature
```

These reports are deliberately descriptive and numeric. They do not name user
emotions.

## Training Objectives

The persistent semantic dynamics trainer runs ordered episodes:

```text
initialize state
for each transition:
  current event embedding
  EventEncoder current
  differentiable SNN window
  TraceEncoder latent
  predict next event embedding
  preserve state
reset state at episode boundary
```

The base objective combines:

- cosine next-event embedding distance
- firing-rate regularization
- inactive-neuron regularization
- membrane stability penalty

Validation-best checkpointing is the default interpretation point. See
`docs/semantic_dynamics_training.md` for commands, output files, and the
2026-06-11 CUDA smoke record.

Generated checkpoints, caches, logs, figures, and run directories are not the
durable evidence layer. The repository keeps reviewed summaries and decision
records instead. See `docs/result_artifact_policy.md` for the artifact
promotion policy.

## Context Evaluation

Context tests use paired episodes where the current event text is identical but
prior context and the correct next event differ. Important checks:

- persistent state versus reset-each-transition
- context-ranking objective versus next-event-only objective
- real-history margin versus shuffled-history margin
- SNN comparison against a context-free MLP and a GRU baseline

A positive context margin supports the narrower claim that prior state helped on
the controlled fixture. It does not establish emotional semantics.

The broader trace-meaning and response-influence evaluation framework is defined
in `docs/trace_meaning_and_response_evaluation.md`.

The current fixture hierarchy is fixed in
`docs/benchmark_fixture_policy.md`: `semantic_alignment_episodes.yaml` is the
primary long-run regression fixture, `context_dependence_episodes.yaml` is the
fast CI/context guardrail, and `response_conditioning_cases.yaml` is a secondary
response-surface fixture.

## Thought Module Protocol

The thought module protocol is intentionally minimal:

```text
user event + neutral numeric state report -> one short internal thought
```

The prompt tells the local model not to answer the user, not to assert emotion
labels, and not to overstate confidence. Generated thoughts can be injected back
as `internal_thought` events for ablations.

The multi-module discussion contract is defined in
`docs/adaptive_thought_module_protocol.md`. The first integration target is a
two-module private-state loop with natural-language module messages, fixed
budget/termination rules, and no central emotion-label aggregator.

## Cluster and Rewiring Experiments

Current rewiring is activity-guided:

- collect train-episode neuron-memory profiles
- compute positive neuron-neuron coactivity
- discover functional communities by spectral features and modularity search
- remove weak inter-community edges first
- add high-coactivity intra-community edges
- preserve the directed edge budget
- reset optimizer moments for changed edges only

This is a structural-plasticity ablation. It is useful for testing whether
activity-derived communities improve a controlled objective, but it is not yet a
final learning rule or biological model.

The current experiment design and visualization handoff live in
`docs/activity_guided_rewiring_experiment_design.md`.

As of the 2026-06-12 AET-29 pipeline run, activity-guided rewiring found a
semantic-preserving configuration but did not establish adjacency-community
evidence. The rule remains a controlled ablation and search heuristic.

## Memory-Threshold Substrate Decision

Decision date: 2026-06-12

Decision: **hold `NeuronMemoryThresholdRSNN` as an ablation substrate; do not
promote it to the primary substrate yet.**

Evidence used:

- Semantic benchmark output:
  `runs/memory_threshold_semantic_benchmark_lmstudio/decision_report.json`
- Parameter sweep output:
  `runs/memory_threshold_parameter_sweep_lmstudio/decision_report.json`
- Context-structure output:
  `runs/memory_threshold_context_structure_best_lmstudio/decision_report.json`
- Emergent-cluster diagnostic:
  `runs/memory_threshold_emergent_cluster_best_lmstudio/decision_report.json`
- CUDA smoke output:
  `runs/aet28_memory_threshold_cuda_smoke/`

The positive evidence is real but narrow:

- Best memory-threshold model:
  `snn_memory_feedback`
- Mean real targeted MAE:
  `0.27069973498582844`
- Previous contrastive SNN mean real targeted MAE:
  `0.2921911489218474`
- GRU contrastive mean real targeted MAE:
  `0.27535309784114365`
- Previous SNN minus best memory model MAE:
  `0.021491413936018944`
- GRU minus best memory model MAE:
  `0.004653362855315202`
- Shuffled-history degradation:
  `0.058600530028343145`
- Reset-history degradation:
  `0.22351464480161665`
- Memory strength mean absolute value:
  `0.297730765491724`, below the non-saturation threshold used by the
  summarizer.

The parameter sweep selected:

```text
feedback_0.050__threshold_0.500__accumulation_decay_0.850
```

This selected configuration passed the semantic direction, pair-order,
shuffled/reset degradation, seed-stability, and non-saturation checks. Its mean
real targeted MAE was `0.26881304606795314`, improving on the previous
contrastive SNN by `0.023378102853894245` and on the GRU by
`0.006540051773190503`.

The context-structure check also supports keeping it as a serious candidate:

- Trace context gap:
  `0.02976742759346958`
- Trace reset gap:
  `0.055465207248926104`
- Same-context repeat distance:
  approximately `0`
- Context retrieval accuracy:
  `1.0`
- Linear probe accuracy:
  `0.575` versus `0.5` chance
- Context and reset gaps were positive for all five seeds.

CUDA stability was checked on `DESKTOP-MMLRCFK` with
`C:/Users/remote/miniconda3/envs/picasso-gpu/python.exe`, PyTorch
`2.11.0+cu128`, and an NVIDIA RTX 4090. A short hash-encoder smoke command ran
with `--device cuda`, `--epochs 3`, and `--seeds 42` and produced
`by_seed_model.csv`, `summary_by_model.csv`, `metadata.json`, and
`run_log.jsonl`. The runner uses `torch.device(args.device)` directly, so this
path does not silently fall back to CPU.

Blocking evidence against promotion:

- Emergent-cluster diagnostic verdict:
  `community_evidence_not_established`
- Trained minus weight-shuffled null modularity:
  `-0.0028722506016492398`
- Response coherence gap:
  `-0.007053542811061739`
- Trained minus null response coherence gap:
  `-0.008771147352330939`
- Functional community evidence was false, and trained topology did not beat the
  shuffled-weight null for most seeds.

Rationale:

The memory-threshold variant improves the controlled semantic/context metrics
and avoids obvious memory saturation on the selected fixture. That justifies
keeping it as the main ablation substrate for AET-29 rewiring work. It does not
yet justify replacing `AdaptiveSparseRSNN` as the primary substrate, because the
extra memory state and consolidation logic increase the core contract
complexity, while the structural/community evidence needed for a substrate
promotion is not established.

Promotion conditions:

- Reproduce the semantic and context-structure advantage on the primary
  regression fixture chosen in AET-30.
- Show non-saturated memory behavior across a broader parameter range and
  longer runs.
- Show degradation under shuffled/reset history while preserving exact-repeat
  stability.
- Establish structural or functional community evidence against appropriate
  initialization, shuffled-weight, and label-permutation nulls.
- Demonstrate CUDA stability on the long-run benchmark, not only a short smoke.
- Keep the claim boundary: this would still be an emotion-related dynamics
  substrate candidate, not evidence that the system feels emotions.

## Legacy Migration Boundary

Pre-v7 work is treated as motivation, diagnostic history, and integration
reference. It is not treated as direct proof for the v7 SNN. The migration
review lives in `docs/legacy_experiment_migration_review.md` and classifies
v1-v6 materials into:

- keep
- keep after redesign
- re-run under v7 fixtures
- discard from v7 core scope

The main carried-forward lessons are branch/trace collapse diagnostics,
late-ignition and saturation checks, neutral trace-report framing, response
surface softening risk, and the rule that LLMs are expression layers rather than
the emotion engine.

## Repo Hygiene

Packaging is defined by `pyproject.toml` and the source tree under
`src/emonet_v7`. Generated `*.egg-info/` metadata is not part of the canonical
source contract and should remain untracked.

## Device Policy

Experiment entrypoints use `resolve_device`:

```text
--device auto    CUDA when available, otherwise CPU
--device cpu     always CPU
--device cuda    CUDA when available, otherwise CPU fallback
--device cuda:N  specific CUDA device when available, otherwise CPU fallback
```

Use `--no-cuda-fallback` when a CUDA run must fail instead of silently falling
back. Summaries record `requested_device`, `resolved_device`, and
`used_device_fallback`.

## Open Decisions

- Whether a future rewiring rule can establish community evidence; the current
  activity-guided rule remains an ablation after AET-29.
- Which metrics justify closing a context-memory milestone: validation loss,
  context margin, shuffled-history delta, multi-seed win rate, or a combined
  threshold.
- How to represent long-term memory without introducing predefined emotion
  axes.

## Change Log

- 2026-06-07: v7 heartbeat, text wiring, fixture suites, and internal-thought
  scaffold documented.
- 2026-06-11: persistent semantic dynamics training and CPU/CUDA device policy
  verified through local tests and a strict remote CUDA smoke run.
- 2026-06-11: this implementation spec consolidates architecture, schema,
  training, context evaluation, thought-module, and rewiring decisions for
  Linear issue AET-13.
- 2026-06-11: legacy v1-v6 experiments classified for migration in
  `docs/legacy_experiment_migration_review.md` for Linear issue AET-12.
- 2026-06-11: activity-guided rewiring experiment design and visualization
  handoff documented for Linear issue AET-9.
- 2026-06-11: adaptive thought module lifecycle and two-module discussion
  protocol documented for Linear issue AET-10.
- 2026-06-11: trace meaning and response-influence evaluation framework
  documented for Linear issue AET-11.
- 2026-06-11: generated package metadata removed from tracked source and
  ignored for Linear issue AET-19.
- 2026-06-12: memory-threshold substrate promotion was held; it remains an
  ablation substrate pending broader fixture, long-run CUDA, and community
  evidence for Linear issue AET-28.
- 2026-06-12: activity-guided rewiring found a semantic-preserving regime but
  did not establish rewired community evidence for Linear issue AET-29.
- 2026-06-12: benchmark fixture hierarchy fixed for Linear issue AET-30:
  `semantic_alignment_episodes.yaml` is primary long-run regression,
  `context_dependence_episodes.yaml` is CI/context guardrail, and
  `response_conditioning_cases.yaml` is secondary exploratory response work.
