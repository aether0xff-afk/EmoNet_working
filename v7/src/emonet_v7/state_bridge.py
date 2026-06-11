"""Neutral summaries that bridge SNN traces into language-model prompts."""

from __future__ import annotations

from typing import Any

import torch

from .adaptive_rsnn import TickTrace


def build_neutral_state_report(
    *,
    traces: list[TickTrace],
    latent_z: torch.Tensor,
    stimulation_ticks: int,
    signature_size: int = 8,
) -> dict[str, Any]:
    """Summarize trace dynamics without assigning emotion labels."""

    if not traces:
        raise ValueError("traces must not be empty")
    spike_counts = [int(trace.spike.sum()) for trace in traces]
    total_spikes = sum(spike_counts)
    post_input_spikes = sum(spike_counts[stimulation_ticks:])
    neuron_count = traces[0].spike.shape[-1]
    tick_count = len(traces)
    latent = latent_z.detach().cpu().flatten()
    return {
        "active_ratio": float(total_spikes / max(1, neuron_count * tick_count)),
        "trace_persistence": float(post_input_spikes / total_spikes) if total_spikes else 0.0,
        "peak_spike_count": max(spike_counts),
        "final_spike_count": spike_counts[-1],
        "latent_signature": [round(float(value), 4) for value in latent[:signature_size]],
    }
