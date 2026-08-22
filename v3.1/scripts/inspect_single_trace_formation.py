#!/usr/bin/env python3
"""Inspect how one v3.1 neural TRACE forms tick by tick.

This is an observational diagnostic. It reuses the existing EmoNet runtime and
exports the internal sequence already recorded by TickRecord: active neurons,
node K/stimulus states, fired edges, dominant route, fatigue and rewiring.
It intentionally does not claim an exact additive causal decomposition of K,
because the current core does not log every pre-threshold contribution.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import export_neural_activation_traces as exporter  # noqa: E402


DEFAULT_DYNAMICS = {
    "k_threshold_base": 0.72,
    "k_remem_base": 0.95,
    "k_decay": 0.99,
    "refractory_ticks": 1,
    "input_topk": 2,
    "input_signal_clip": 1.50,
    "intrinsic_alignment_gain": 0.24,
    "fatigue_gain": 0.30,
    "fatigue_threshold_gain": 0.18,
    "fatigue_k_leak": 0.08,
    "inhibitory_suppression_gain": 0.18,
    "density_control_start_tick": 0,
    "density_target_high": 1.0,
    "density_soft_k_leak_gain": 0.0,
    "density_hard_cap": 1.0,
    "density_pruned_fatigue_gain": 0.0,
    "ne_thresh_reduce_gain": 0.25,
    "ne_remem_reduce_gain": 0.25,
    "activity_churn_eps": 0.02,
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_dynamics(path: Path | None) -> dict[str, Any]:
    params = dict(DEFAULT_DYNAMICS)
    if path is not None:
        payload = json.loads(path.read_text(encoding="utf-8"))
        params.update(payload.get("dynamics", {}))
    return params


def make_model_args(args: argparse.Namespace, dynamics: dict[str, Any]) -> SimpleNamespace:
    values: dict[str, Any] = {
        "n_neurons": args.n_neurons,
        "seed": args.seed,
        "z_encoder_mode": "stat",
        "stim_source": args.stim_source,
        "max_ticks": args.max_ticks,
        "min_ticks_before_converged": 6,
        "convergence_patience": 4,
        "progress_every": 0,
        **DEFAULT_DYNAMICS,
    }
    values.update(dynamics)
    return SimpleNamespace(**values)


def edge_set(model: Any) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for neuron in model.state.neurons:
        src = int(neuron.neuron_id)
        for dst in neuron.out_neighbors:
            edges.add((src, int(dst)))
    return edges


def fatigue_vector(model: Any) -> np.ndarray:
    return np.asarray([float(n.fatigue) for n in model.state.neurons], dtype=np.float32)


def json_vec(value: Any, decimals: int = 5) -> list[float]:
    arr = exporter.to_numpy(value).reshape(-1)
    return [round(float(x), decimals) for x in arr]


def branch_log_for(outputs: dict[str, Any]) -> list[Any]:
    # Match the exporter: prefer the pruned log used to construct activation.
    return list(outputs.get("pruned_branch_log") or outputs.get("branch_log") or [])


def top_node_states(record: Any, limit: int) -> list[dict[str, Any]]:
    states = getattr(record, "node_states", {}) or {}
    ranked = sorted(states.items(), key=lambda item: float(item[1].K), reverse=True)
    return [
        {
            "neuron_id": int(node_id),
            "K": round(float(state.K), 6),
            "stim_vec": json_vec(state.stim_vec),
        }
        for node_id, state in ranked[:limit]
    ]


def build_tick_report(outputs: dict[str, Any], top_n: int) -> list[dict[str, Any]]:
    ticks: list[dict[str, Any]] = []
    previous_k: dict[int, float] = {}
    previous_active: set[int] = set()

    for record in branch_log_for(outputs):
        active = [int(x) for x in (getattr(record, "active_nodes", []) or [])]
        active_set = set(active)
        ranked = top_node_states(record, top_n)
        states = getattr(record, "node_states", {}) or {}

        # Only compute a K delta when the same node was observed in the previous
        # TickRecord. A missing node does not imply that its true previous K was 0.
        changes: list[dict[str, Any]] = []
        first_observed: list[int] = []
        for node_id, state in states.items():
            node_id = int(node_id)
            current = float(state.K)
            if node_id not in previous_k:
                first_observed.append(node_id)
                continue
            before = previous_k[node_id]
            delta = current - before
            if abs(delta) > 1e-9:
                changes.append(
                    {
                        "neuron_id": node_id,
                        "previous_K": round(before, 6),
                        "current_K": round(current, 6),
                        "delta_K": round(delta, 6),
                    }
                )
        changes.sort(key=lambda x: abs(float(x["delta_K"])), reverse=True)

        fired = getattr(record, "edges_fired", []) or []
        ticks.append(
            {
                "tick": int(getattr(record, "tick", len(ticks))),
                "active_count": len(active),
                "active_nodes": active,
                "newly_active_nodes": sorted(active_set - previous_active),
                "deactivated_nodes": sorted(previous_active - active_set),
                "first_observed_state_nodes": sorted(first_observed),
                "dominant_node": ranked[0] if ranked else None,
                "top_nodes": ranked,
                "largest_K_changes": changes[:top_n],
                "edges_fired": [[int(src), int(dst)] for src, dst in fired],
            }
        )

        previous_k = {int(node_id): float(state.K) for node_id, state in states.items()}
        previous_active = active_set

    return ticks


def clip_list(values: list[Any], limit: int = 24) -> str:
    if len(values) <= limit:
        return str(values)
    return f"{values[:limit]} ... (+{len(values) - limit} more)"


def markdown_report(report: dict[str, Any]) -> str:
    row = report["input"]
    observed_route = " -> ".join(map(str, report["run"]["observed_dominant_route"])) or "(none)"
    exporter_route = report["run"]["exporter_dominant_route"]
    lines = [
        "# Single-Sentence TRACE Formation Report",
        "",
        "## Input",
        "",
        f"- record_id: `{row.get('record_id', '')}`",
        f"- text: {row.get('text', '')}",
        "",
        "## Run summary",
        "",
        f"- stimulus mode: `{report['settings']['stim_source']}`",
        f"- stimulus vector: `{report['run']['stim_vec']}`",
        f"- ticks_run: `{report['run']['ticks_run']}`",
        f"- termination: `{report['run']['termination_reason']}`",
        f"- observed max-K route: `{observed_route}`",
        f"- exporter dominant route: `{exporter_route}`",
        f"- exporter route valid: `{report['run']['exporter_route_valid']}`",
        f"- added edges after sentence: `{len(report['persistent_changes']['added_edges'])}`",
        f"- removed edges after sentence: `{len(report['persistent_changes']['removed_edges'])}`",
        f"- neurons with fatigue increase: `{report['persistent_changes']['fatigue_changed_count']}`",
        "",
        "> The observed max-K route is reconstructed directly from TickRecord because",
        "> the existing dominant-branch helper may return `-1` when it cannot identify",
        "> a branch. `delta_K` is observational, not exact causal attribution.",
        "",
        "## Tick-by-tick formation",
        "",
    ]

    for tick in report["ticks"]:
        dom = tick["dominant_node"] or {}
        top_nodes = [
            (node["neuron_id"], node["K"])
            for node in tick["top_nodes"][:5]
        ]
        lines.extend(
            [
                f"### Tick {tick['tick']}",
                "",
                f"- active count: `{tick['active_count']}`",
                f"- newly active: `{clip_list(tick['newly_active_nodes'])}`",
                f"- deactivated: `{clip_list(tick['deactivated_nodes'])}`",
                f"- dominant: neuron `{dom.get('neuron_id', '')}`; K=`{dom.get('K', '')}`",
                f"- top 5 neurons (id, K): `{top_nodes}`",
                f"- fired edges: `{len(tick['edges_fired'])}` total; first edges `{clip_list(tick['edges_fired'], 12)}`",
                "- largest comparable K changes:",
            ]
        )
        if tick["largest_K_changes"]:
            for change in tick["largest_K_changes"][:8]:
                lines.append(
                    f"  - neuron {change['neuron_id']}: {change['previous_K']} -> "
                    f"{change['current_K']} (delta {change['delta_K']:+.6f})"
                )
        else:
            lines.append("  - none (first observed tick or no comparable change)")
        lines.append("")

    lines.extend(
        [
            "## Persistent state changes",
            "",
            f"- added edges: `{len(report['persistent_changes']['added_edges'])}`",
            f"- removed edges: `{len(report['persistent_changes']['removed_edges'])}`",
            f"- first removed edges: `{clip_list(report['persistent_changes']['removed_edges'], 24)}`",
            "- largest fatigue increases:",
        ]
    )
    for item in report["persistent_changes"]["top_fatigue_increases"]:
        lines.append(f"  - neuron {item['neuron_id']}: +{item['delta']:.6f}")

    lines.extend(
        [
            "",
            "## What to look for next",
            "",
            "Run a matched comparison sentence from the same initial seed. The first tick",
            "where its max-K neuron, fired-edge set, or surviving active set diverges is a",
            "candidate TRACE-formation divergence point. That point can then be tested",
            "causally by ablating the neuron/edge or changing one dynamics component.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    rows = read_rows(args.input)
    if not rows:
        raise ValueError(f"no rows in {args.input}")
    if not 0 <= args.row_index < len(rows):
        raise IndexError(f"row-index must be within 0..{len(rows) - 1}")

    row = rows[args.row_index]
    model_args = make_model_args(args, load_dynamics(args.config))
    model = exporter.build_model(model_args)
    model_input = exporter.model_input_for_row(row, model_args)

    before_edges = edge_set(model)
    before_fatigue = fatigue_vector(model)
    outputs = model.forward(model_input)
    after_edges = edge_set(model)
    after_fatigue = fatigue_vector(model)

    ticks = build_tick_report(outputs, args.top_nodes)
    fatigue_delta = after_fatigue - before_fatigue
    ranked_fatigue = np.argsort(fatigue_delta)[::-1]
    top_fatigue = [
        {"neuron_id": int(i), "delta": round(float(fatigue_delta[i]), 6)}
        for i in ranked_fatigue[: args.top_nodes]
        if float(fatigue_delta[i]) > 1e-9
    ]

    exporter_route = [int(x) for x in exporter.dominant_branch_ids(outputs)]
    observed_route = [
        int(tick["dominant_node"]["neuron_id"])
        for tick in ticks
        if tick["dominant_node"] is not None
    ]

    report = {
        "input": dict(row),
        "settings": {
            "seed": args.seed,
            "n_neurons": args.n_neurons,
            "max_ticks": args.max_ticks,
            "stim_source": args.stim_source,
            "config": str(args.config) if args.config else None,
        },
        "run": {
            "ticks_run": int(outputs.get("ticks_run", len(ticks))),
            "termination_reason": str(outputs.get("termination_reason", "")),
            "stim_vec": json_vec(outputs.get("stim_vec", model_input)),
            "observed_dominant_route": observed_route,
            "exporter_dominant_route": exporter_route,
            "exporter_route_valid": bool(exporter_route) and any(x >= 0 for x in exporter_route),
            "final_z": json_vec(outputs.get("z", np.zeros((0,), dtype=np.float32))),
        },
        "ticks": ticks,
        "persistent_changes": {
            "added_edges": [[int(a), int(b)] for a, b in sorted(after_edges - before_edges)],
            "removed_edges": [[int(a), int(b)] for a, b in sorted(before_edges - after_edges)],
            "fatigue_changed_count": int(np.count_nonzero(fatigue_delta > 1e-9)),
            "top_fatigue_increases": top_fatigue,
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown_report(report), encoding="utf-8")
    return args.output_json, args.output_md


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("v3.1/outputs/targeted_records_trace_normalized.csv"))
    parser.add_argument("--config", type=Path, default=Path("v3.1/configs/final_dynamics_v1.json"))
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--n-neurons", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stim-source", choices=["auto", "text", "proxy"], default="auto")
    parser.add_argument("--max-ticks", type=int, default=64)
    parser.add_argument("--top-nodes", type=int, default=12)
    parser.add_argument("--output-json", type=Path, default=Path("v3.1/outputs/single_trace_formation/sample_000.json"))
    parser.add_argument("--output-md", type=Path, default=Path("v3.1/outputs/single_trace_formation/sample_000.md"))
    return parser.parse_args()


def main() -> None:
    json_path, md_path = run(parse_args())
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
