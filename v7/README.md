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
- LM Studio client boundary for later local model calls
- selectivity script with pairwise embedding and trace distances

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
pip install sentence-transformers
```

For later LM Studio calls:

```powershell
pip install openai
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

## LM Studio boundary

The later internal-thought feedback loop will use LM Studio. Provide the local server base address and loaded model identifier when that milestone starts.

## Scope boundary

Milestone 2 still excludes trained semantic dynamics, LLM-generated internal thoughts, STDP, rewiring, predefined emotion axes, and predefined emotion clusters. Those features are added only after the base text-to-trace plumbing is reviewed.
