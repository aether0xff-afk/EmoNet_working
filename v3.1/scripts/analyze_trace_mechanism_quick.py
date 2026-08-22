#!/usr/bin/env python3
"""Balanced quick check for the v3.1 TRACE mechanism experiment."""

from __future__ import annotations

import analyze_trace_mechanism as base
from analyze_trace_mechanism_fast import run_sample_fast


base.run_sample = run_sample_fast
_KEEP = {"baseline", "single_tick", "no_recurrence", "no_alignment", "no_memory", "no_inhibition"}
base.CONDITIONS = [condition for condition in base.CONDITIONS if condition.name in _KEEP]
_original_read_rows = base.read_rows


def balanced_read_rows(path):
    rows = _original_read_rows(path)
    buckets = {}
    for row in rows:
        label = str(row.get("valence", "")).strip().lower() or "unknown"
        buckets.setdefault(label, []).append(row)
    selected = []
    for label in ("negative", "mixed", "positive"):
        selected.extend(buckets.get(label, [])[:12])
    if not selected:
        selected = rows[:36]
    return selected


base.read_rows = balanced_read_rows


if __name__ == "__main__":
    args = base.parse_args()
    args.limit = 0
    args.output_dir = args.output_dir.parent / "trace_mechanism_quick"
    base.run(args)
