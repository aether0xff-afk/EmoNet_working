# EmoNet v7.0 SNN Core

This directory is a clean rebuild of EmoNet v7 around an adaptive sparse recurrent spiking neural network.

## Current milestone

Milestone 1: SNN heartbeat

- ALIF-style adaptive neurons
- fixed sparse recurrent mask
- event window simulation
- raw tick trace collection
- decay experiment with CSV and PNG outputs
- unit tests for mask generation, adaptation, decay, and deterministic seeds

## Run

```powershell
cd v7
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]
python experiments/run_decay.py --config configs/v7_0_default.yaml --output runs/decay_seed42
pytest
```

## Scope boundary

The first milestone intentionally excludes text embeddings, LLM calls, STDP, rewiring, predefined emotion axes, and predefined emotion clusters. Those features are added only after the base dynamics are stable.
