# Legacy Experiment Migration Review

Status date: 2026-06-11

This document classifies pre-v7 EmoNet work for v7 migration. The goal is to
avoid re-importing old assumptions while preserving useful evidence, schemas,
and failure modes.

## Summary Decision

v7 should not port the old v1-v6 runtime code directly. The older lines used
heuristic affect vectors, dominant-branch extraction, style regressors, and
character-runtime adapters that answer different questions from the current v7
SNN rebuild.

What should migrate:

- the distinction between internal trace and final language expression
- branch-collapse and late-ignition diagnostics as cautionary benchmark patterns
- neutral trace-to-report ideas, without emotion labels
- runtime event concepts such as user message versus elapsed-time/no-reply event
- evaluation warnings around style softening and biased supervision

What should not migrate:

- fixed four-axis emotion control as a v7 substrate assumption
- predefined emotion clusters
- direct `z -> style -> prompt` as the main proof path
- GPT-interpreted trajectory reports as ground truth
- v5/v6 character artifacts as v7 core dependencies

## Migration Table

| Source | Relevant content | v7 decision | Reason |
| --- | --- | --- | --- |
| `v1/emotion_z_pipeline.py` | Text to stimulus vector, dynamics history, GRU history encoder, latent `z` | Keep concept only | Useful early sketch, but monolithic and centered on handcrafted affect/stimulus targets. v7 now uses event schemas, frozen embeddings, SNN currents, trace windows, and self-supervised objectives. |
| `v2/emonet/` | Modular MVP with trait EMA, clustering, rewiring, branch tracking, dominant path, `z -> s`, prompt generator | Reuse as vocabulary/reference only | The modules name many later concerns, but the fixed control vector and prompt path would over-constrain v7. |
| `v3/emonet/core.py` | Node dynamics, neuron roles, branch log, dominant branch extraction, style axes | Do not port directly | It is a heuristic branch-dynamics model, not the v7 ALIF-style SNN substrate. Preserve diagnostic ideas and failure cases. |
| `v3/BRANCH_COLLAPSE_MITIGATION_2026-04-06.md` | Branch length collapse analysis and mitigation plan | Keep as regression warning | The core lesson is to measure whether trace/path structure collapses into one-tick artifacts before making semantic claims. |
| `v3/BRANCH_DEPTH_RESEARCH_2026-04-08.md` | Full-export branch distributions, delayed ignition, hard-cap saturation | Keep metrics pattern | `len1_ratio`, first-active tick, active-window, max-tick saturation, and path coverage are useful analogs for future v7 trace diagnostics. |
| `v4/RESEARCH_SUMMARY_2026-04-10.md` | Branch recovery, calibration-backed config, episode interpretation, generation comparison | Keep evidence categories only | The result categories are valuable, but the calibrated config and GPT-episode interpretations are not v7 proof. |
| `v4/RESEARCH_GAPS_2026-04-09.md` | Surface softening, target bias, `z -> s` weakness, fragile positive/anticipatory states | Keep as evaluation warnings | These are directly relevant when v7 later maps neutral trace reports to language or response behavior. |
| `v5/` | Character-chat MVP, LLM as expression layer, trace plus character/session context | Keep separation principle | v7 should preserve that the LLM is a surface/expression layer, not the emotion engine. Do not depend on v5 runtime artifacts. |
| `v6/` | Ruca/Rookie runtime, no-reply event model, response gate, v5 adapter | Revisit after v7 report schema stabilizes | Useful for future integration, but current v7 core should remain independent. |
| `encoder-LLM-testing/` | LLM emotion-label benchmark materials; low direct label accuracy in saved summary | Keep as anti-label caution | Supports the decision to avoid treating direct emotion labels as primary ground truth. |
| `encoder-ML testing/` | ML encoder and style-regression benchmark materials | Re-run only if a new v7-compatible target is defined | Old style targets are too biased toward safe/cooperative language for current claims. |
| `attack_experiments/` | Decoder inversion/chosen plaintext/backprop input recovery materials | Out of v7 core scope | Security experiments may matter later but are not part of current substrate/training milestones. |
| root `src/` and `docs/architecture.md` | Minecraft RL MVP | Out of EmoNet v7 scope | Different project surface; do not mix into v7 planning. |

## Reusable Data and Schema Ideas

Reusable ideas:

- event source and elapsed-time separation from v6
- trace profile summaries from v5/v6, but rewritten as neutral v7 reports
- branch/trace diagnostics from v3/v4 as quantitative health checks
- human/automatic evaluation categories from v4, after relabeling them around
  context memory and trace influence instead of emotion-label correctness

Do not reuse as-is:

- `stim_vec` as dopamine/serotonin/norepinephrine/melatonin ground truth
- fixed 32 style axes as the main v7 target
- `dominant_branch_len` as the only path metric
- style tags or generated response scores as proof of internal emotional state

## Duplicate Implementation Avoidance List

When adding v7 work, avoid recreating these old paths:

- a second monolithic `emotion_z_pipeline.py`
- a second prompt generator that treats emotion/style axes as the central output
- a second v5/v6 adapter that imports old runtime artifacts into v7 core
- a second branch optimizer over heuristic node dynamics
- a second LLM-label benchmark that treats labels as ground truth without a
  context-control fixture

Prefer extending current v7 modules:

- use `schemas.py` for event shape
- use `adaptive_rsnn.py` or `memory_threshold_rsnn.py` for substrate work
- use `training_window.py` and `trace_encoder.py` for differentiable trace paths
- use `context_objective.py` for controlled context evaluation
- use `state_bridge.py` for neutral report generation
- use `activity_guided_rewiring.py` for rewiring ablations

## Evidence That Can Inform v7 Metrics

The most useful historical metrics are diagnostic rather than proof metrics:

- branch or trace collapse rate
- first active tick / delayed ignition
- active-window length
- saturation against max ticks
- path or active-edge coverage
- style softening or response-surface drift
- target distribution bias
- baseline comparison against text-only or context-free models

For v7, these should be translated into SNN-native measurements such as spike
rate, active-edge density, latent distance, context margin, real-minus-shuffled
margin, and persistent-minus-reset deltas.

## Current Claim Boundary

Old results may be cited as motivation or failure analysis. They should not be
used as evidence that the current v7 SNN has emotional semantics.

The only current v7 evidence should come from v7 fixtures, v7 experiment
entrypoints, v7 summaries, and fresh runs that record their device, seed, and
checkpoint policy.
