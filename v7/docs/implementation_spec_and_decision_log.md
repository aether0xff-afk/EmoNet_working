# EmoNet v7 Implementation Spec and Decision Log

Status date: 2026-06-11

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
- It exists for memory and rewiring ablations, not as a replacement for the
  baseline contract yet.

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

## Context Evaluation

Context tests use paired episodes where the current event text is identical but
prior context and the correct next event differ. Important checks:

- persistent state versus reset-each-transition
- context-ranking objective versus next-event-only objective
- real-history margin versus shuffled-history margin
- SNN comparison against a context-free MLP and a GRU baseline

A positive context margin supports the narrower claim that prior state helped on
the controlled fixture. It does not establish emotional semantics.

## Thought Module Protocol

The thought module protocol is intentionally minimal:

```text
user event + neutral numeric state report -> one short internal thought
```

The prompt tells the local model not to answer the user, not to assert emotion
labels, and not to overstate confidence. Generated thoughts can be injected back
as `internal_thought` events for ablations.

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

- Whether the memory-threshold substrate should graduate from ablation module to
  the primary substrate.
- Which context fixture should become the main regression benchmark.
- Whether rewiring should run during training, between training phases, or only
  as an offline ablation.
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
