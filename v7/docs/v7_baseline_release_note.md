# EmoNet v7 Baseline Release Note

Baseline date: 2026-06-11

This note freezes the current v7 baseline after the merge of
`feature/v7-snn-rebuild` into `main`.

## Source Reference

- Baseline merge commit: `89a4345 Merge feature/v7-snn-rebuild`
- Repo hygiene follow-up: `7b027a1 chore(v7): stop tracking generated package metadata`
- Package: `emonet-v7` version `0.2.0`
- Supported Python: `>=3.11,<3.13`

The generated `*.egg-info/` package metadata is intentionally untracked. The
canonical source contract is `pyproject.toml` plus `src/emonet_v7`.

## Implemented Modules

Core runtime:

- `schemas.py`: neutral event schema.
- `event_encoder.py`: frozen text embedding to SNN current projection.
- `adaptive_rsnn.py`: baseline adaptive sparse recurrent spiking network.
- `training_window.py`: differentiable SNN training window.
- `trace_encoder.py`: trace sequence to latent projection.
- `self_supervised.py`: next-event prediction objective.
- `context_objective.py`: context-ranking and context-free comparison helpers.
- `device.py`: explicit CPU/CUDA device resolution policy.
- `run_logger.py`: run metadata and JSONL logging.

Experimental and evaluation modules:

- `thought_module.py` and `state_bridge.py`: neutral state reports and
  internal-thought feedback plumbing.
- `memory_threshold_rsnn.py`, `memory_threshold_trace_encoder.py`, and
  `memory_threshold_bundle.py`: memory-threshold ablation substrate.
- `activity_guided_rewiring.py`: controlled activity-guided rewiring ablation.
- `lmstudio_client.py`, `text_encoder.py`, and `embedding_cache.py`: local
  embedding and LM Studio integration helpers.
- `selectivity.py` and `metrics.py`: fixture-level measurement utilities.

## Runnable Smoke Commands

Install in a Python 3.11 or 3.12 environment:

```powershell
cd v7
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

Run the deterministic heartbeat:

```powershell
python experiments/run_decay.py --config configs/v7_0_default.yaml --output runs/decay_seed42
```

Run the hash wiring smoke:

```powershell
python experiments/run_selectivity.py --encoder hash --output runs/selectivity_hash_seed42
```

Run the persistent semantic dynamics hash smoke:

```powershell
python experiments/train_semantic_dynamics.py `
  --encoder hash `
  --epochs 1 `
  --num-neurons 16 `
  --event-ticks 4 `
  --stimulation-ticks 2 `
  --device auto `
  --output runs/semantic_dynamics_hash_smoke `
  --quiet
```

Run the unit test suite:

```powershell
py -3.11 -m pytest -q
```

## CUDA Smoke Record

The CUDA path was checked on 2026-06-11 over SSH on remote host
`DESKTOP-MMLRCFK` with an RTX 4090, PyTorch `2.11.0+cu128`, and strict CUDA
fallback disabled.

Record:

- Strict CUDA run resolved to `cuda`.
- `used_device_fallback` was `false`.
- Matched one-epoch hash CPU/GPU validation delta was approximately
  `0.00000604`.

See `docs/semantic_dynamics_training.md#2026-06-11-cuda-smoke-record` for the
exact command and observed summary.

## Verified Test Range

Local verification after AET-19:

```text
py -3.11 -m pip install --no-deps -e .
py -3.11 -m pytest -q
34 passed
git diff --check
```

The full dependency install path should be run in a clean Python 3.11 or 3.12
environment. During AET-19 verification, the global Python 3.11 environment had
an existing `torch` file locked by another process, so dependency installation
was not used as the final verification signal.

## What This Baseline Can Claim

- Text events can be encoded into SNN currents.
- The adaptive sparse recurrent SNN runs and records observable traces.
- Trace latents can feed a self-supervised next-event objective.
- Persistent state, reset-state, context-dependence, memory-threshold, and
  rewiring ablations have runnable entrypoints.
- CPU and CUDA training entrypoints share an explicit device policy.
- Neutral internal-thought feedback plumbing exists for local LLM experiments.

## What This Baseline Must Not Claim

- It does not demonstrate validated emotional meaning.
- It does not prove that neuron groups are emotion clusters.
- It does not validate biological fidelity.
- It does not establish broad real-world generalization.
- It does not prove that rewiring is the final substrate rule.
- It does not show that generated internal thoughts reliably improve responses.
- CPU/GPU smoke checks are device-path checks, not full numerical equivalence
  guarantees.

## Why Follow-Up Work Continues

The baseline is stable enough to serve as a reproducible starting point, but the
next project is needed to answer higher-risk questions:

- whether two private thought modules can exchange state safely and usefully;
- whether neutral trace reports measurably influence generated responses;
- whether LM Studio multi-seed runs hold up beyond fixture smoke tests;
- whether long CUDA runs match local CPU behavior well enough for future
  experimentation;
- whether memory-threshold dynamics or activity-guided rewiring should graduate
  beyond controlled ablations.

Those items are tracked in the Linear project `EmoNet v7 후속 작업`.
