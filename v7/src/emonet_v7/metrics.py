"""Trace metrics used by Milestone 1 experiments."""

from __future__ import annotations

import numpy as np

from .adaptive_rsnn import TickTrace


def tick_rows(traces: list[TickTrace]) -> list[dict[str, float | int]]:
    """Convert trace snapshots into CSV-friendly rows."""

    rows: list[dict[str, float | int]] = []
    for trace in traces:
        spike = trace.spike.numpy()
        membrane = trace.membrane.numpy()
        adaptation = trace.adaptation.numpy()
        threshold = trace.threshold.numpy()
        rows.append(
            {
                "tick": trace.tick,
                "active_neuron_count": int(spike.sum()),
                "active_ratio": float(spike.mean()),
                "membrane_mean": float(membrane.mean()),
                "membrane_std": float(membrane.std()),
                "membrane_abs_max": float(np.abs(membrane).max()),
                "adaptation_mean": float(adaptation.mean()),
                "threshold_mean": float(threshold.mean()),
                "active_edge_count": int(trace.active_edges.numpy().sum()),
            }
        )
    return rows
