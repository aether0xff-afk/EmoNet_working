#!/usr/bin/env python3
"""Fast runner for analyze_trace_mechanism.

Uses the exact same analysis, conditions, features and report code, but stops at
run_until_converged because branch extraction/z encoding are not used by this
mechanism experiment.
"""

from __future__ import annotations

import numpy as np

import analyze_trace_mechanism as base


def run_sample_fast(row, condition, base_dynamics, seed):
    max_ticks = condition.max_ticks or 64
    args = base.make_model_args(seed, max_ticks, base_dynamics)
    model = base.exporter.build_model(args)
    base.apply_condition(model, condition)
    stim = base.stimulus(row)

    # forward() would additionally prune branches, solve top-k paths and encode
    # z. None of those products enter this analysis, so run only the neural
    # dynamics whose raw TickRecords are the object being measured.
    model.reset()
    model.run_until_converged(stim)

    matrix = base.activation_matrix(model)
    feature = base.temporal_feature(matrix)
    raw_log = list(model.state.branch_log)
    dominant = []
    for record in raw_log:
        states = getattr(record, "node_states", {}) or {}
        if states:
            node_id, state = max(states.items(), key=lambda item: float(item[1].K))
            dominant.append((int(node_id), float(state.K)))
        else:
            dominant.append((-1, 0.0))
    summary = {
        "record_id": row.get("record_id", ""),
        "ticks": len(raw_log),
        "termination": str(model.last_termination_reason),
        "mean_density": float(np.mean([len(getattr(r, "active_nodes", []) or []) / 256.0 for r in raw_log])) if raw_log else 0.0,
        "dominant_route": [node for node, _ in dominant],
        "dominant_k": [value for _, value in dominant],
    }
    return feature, summary, model


base.run_sample = run_sample_fast


if __name__ == "__main__":
    base.run(base.parse_args())
