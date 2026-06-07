# EmoNet v7.0 SNN Core

This directory is a clean rebuild of EmoNet v7 around an adaptive sparse recurrent spiking neural network.

## Implemented milestones

### Milestone 1: SNN heartbeat

- ALIF-style adaptive neurons
- fixed sparse recurrent mask
- event window simulation
- raw tick trace collection
- decay experiment with CSV and PNG outputs

### Milestone 2: text event wiring

- Event schema for user messages and future internal thoughts
- trainable EventEncoder from frozen text embeddings to SNN currents
- GRU TraceEncoder that compresses raw traces into latent z
- offline deterministic hash encoder for wiring smoke tests
- multilingual sentence-transformers adapter for semantic input plumbing checks
- selectivity script with pairwise embedding, current, and trace distances

### Milestone 2.1: fixture-based multi-seed checks

- YAML fixtures for repeated, paraphrased, contrastive, and unrelated sentences
- five-seed selectivity suite with mean, standard deviation, minimum, and maximum summaries
- uploaded GitHub Actions artifacts for reproducible inspection

### Milestone 3 scaffold: internal-thought feedback

- EventEncoder metadata ablation flags
- offline injected-thought suite with reassurance, negative interpretation, and uncertainty conditions
- neutral SNN state report without emotion labels
- ThoughtModule prompt builder
- LM Studio-generated internal-thought runner
- fake-client tests that do not require a local LLM server

The EventEncoder and TraceEncoder are not trained yet. Distinct trace distances at this stage show that the pipeline is wired, not that meaningful emotional dynamics have emerged.

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

## Run one LM Studio-generated internal-thought feedback experiment

```powershell
python experiments/run_lmstudio_thought_feedback.py `
  --base-url http://localhost:1234 `
  --chat-model <loaded-model-identifier> `
  --output runs/lmstudio_thought_feedback
```

Provide the local server base address and loaded model identifier before running this step.

## Scope boundary

The current code still excludes trained semantic dynamics, validated emotional meaning, STDP, rewiring, predefined emotion axes, and predefined emotion clusters. The LM Studio runner is implemented but has not yet been verified against the user's local server.
