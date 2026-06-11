# EmoNet v7.0 SNN Core

This directory is a clean rebuild of EmoNet v7 around an adaptive sparse
recurrent spiking neural network.

## Current implementation map

- Adaptive sparse recurrent SNN heartbeat with raw tick trace collection.
- Text event schema, frozen embedding adapters, EventEncoder, and TraceEncoder.
- Fixture-based selectivity and multi-seed wiring checks.
- Neutral internal-thought feedback scaffold for local LLM experiments.
- Persistent semantic dynamics training with validation-best checkpoints.
- Context-dependence ablations for identical current text under different prior
  context.
- Memory-threshold and activity-guided rewiring ablations.
- Explicit CPU/CUDA device selection for training entrypoints.

The working architecture and decision record lives in
`docs/implementation_spec_and_decision_log.md`.
The frozen baseline release note lives in
`docs/v7_baseline_release_note.md`.

## Install

```powershell
cd v7
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
```

For multilingual embeddings:

```powershell
pip install -e .[text]
```

For later LM Studio calls:

```powershell
pip install -e .[llm]
```

For all optional dependencies:

```powershell
pip install -e .[all]
```

## Run heartbeat

```powershell
python experiments/run_decay.py --config configs/v7_0_default.yaml --output runs/decay_seed42
```

## Run Milestone 2 offline smoke test

```powershell
python experiments/run_selectivity.py --encoder hash --output runs/selectivity_hash_seed42
```

Hash mode verifies deterministic wiring only. It is not a semantic experiment.

## Run multilingual input plumbing check

```powershell
python experiments/run_selectivity.py `
  --encoder sentence-transformer `
  --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 `
  --output runs/selectivity_multilingual_seed42
```

## Run the fixture-based multi-seed suite

```powershell
python experiments/run_selectivity_suite.py `
  --encoder sentence-transformer `
  --output runs/selectivity_suite
```

## Run the offline internal-thought metadata ablation

```powershell
python experiments/run_internal_thought_ablation.py `
  --encoder sentence-transformer `
  --output runs/internal_thought_ablation
```

## Check LM Studio connectivity and exposed model identifiers

```powershell
python experiments/check_lmstudio.py `
  --base-url http://localhost:1234
```

## Run one LM Studio-generated internal-thought feedback experiment

```powershell
python experiments/run_lmstudio_thought_feedback.py `
  --base-url http://localhost:1234 `
  --chat-model <loaded-model-identifier> `
  --output runs/lmstudio_thought_feedback
```

Provide the local server base address and loaded model identifier before running this step.

## Run persistent semantic dynamics training

```powershell
python experiments/train_semantic_dynamics.py `
  --encoder lmstudio `
  --base-url http://127.0.0.1:1234 `
  --embedding-model text-embedding-nomic-embed-text-v1.5 `
  --epochs 30 `
  --device auto `
  --output runs/semantic_dynamics_lmstudio
```

Use `--no-cuda-fallback` with `--device cuda` when a CUDA run must fail instead
of silently using CPU. See `docs/semantic_dynamics_training.md` for the training
contract, output files, context ablations, and the 2026-06-11 CUDA smoke record.

## Scope boundary

The current code still does not establish validated emotional meaning,
predefined emotion axes, predefined emotion clusters, biological fidelity, or
broad real-world generalization. Context and rewiring results should be read as
controlled fixture ablations until broader benchmarks exist.
