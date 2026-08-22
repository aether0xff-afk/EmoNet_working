#!/usr/bin/env python3
"""Inspect how one v3.1 neural TRACE forms over time.

This diagnostic intentionally uses the existing EmoNet runtime without changing
its dynamics. It runs exactly one input row, then exports a human-readable
per-tick report showing:

- which neurons were active,
- which neuron was dominant,
- which edges fired,
- how activation strength K changed,
- how the dominant route evolved,
- which neurons accumulated fatigue,
- and a compact summary of the final TRACE.

The script does *not* claim a causal decomposition of K into exact additive
terms, because the current runtime does not log every internal contribution
before thresholding. Instead it exposes the complete sequence that is already
recorded by TickRecord, plus before/after persistent state changes. This makes
one sentence inspectable end-to-end and provides a clean base for later causal
instrumentation.
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
REPO_ROOT = Path(__file__).resolve().parents[2]
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
    return SimpleNamespace(
        n_neurons=args.n_neurons,
        seed=args.seed,
        z_encoder_mode="stat",
        stim_source=args.stim_source,
        max_ticks=args.max_ticks,
        min_ticks_before_converged=6,
        convergence_patience=4,
        progress_every=0,
        **DEFAULT_DYNAMICS,
        **dynamics,
    )


def fatigue_vector(model: Any) -> np.ndarray:
    return np.asarray(
        [float(neuron.fatigue) for neuron in model.state.neurons],
        dtype=np.float32,
    )


def edge_set(model: Any) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for neuron in model.state.neurons:
        src = int(neuron.neuron_id)
        for dst in neuron.out_neighbors:
            edges.add((src, int(dst)))
    return edges


def _jsonable_vec(value: Any, decimals: int = 5) -> list[float]:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    return [round(float(x), decimals) for x in arr]


def _top_node_states(record: Any, limit: int) -> list[dict[str, Any]]:
    ranked = sorted(
        record.node_states.items(),
        key=lambda item: float(item[1].K),
        reverse=True,
    )
    result: list[dict[str, Any]] = []
    for neuron_id, state in ranked[:limit]:
        result.append(
            {
                "neuron_id": int(neuron_id),
                "K": round(float(state.K), 6),
                "stim_vec": _jsonable_vec(state.stim_vec),
            }
        )
    return result


def build_tick_report(outputs: dict[str, Any], *, top_nodes: int) -> list[dict[str, Any]]:
    branch_log = list(outputs.get("branch_log") or [])
    ticks: list[dict[str, Any]] = []

    previous_k: dict[int, float] = {}
    previous_active: set[int] = set()

    for record in branch_log:
        active = [int(node_id) for node_id in record.active_nodes]
        active_set = set(active)
        ranked = _top_node_states(record, top_nodes)
        dominant = ranked[0] if ranked else None

        k_delta: list[dict[str, Any]] = []
        for node_id, state in record.node_states.items():
            current = float(state.K)
            before = previous_k.get(int(node_id), 0.0)
            delta = current - before
            if abs(delta) > 1e-9:
                k_delta.append(
                    {
                        "neuron_id": int(node_id),
                        "previous_K": round(before, 6),
                        "current_K": round(current, 6),
                        "delta_K": round(delta, 6),
                    }
                )
        k_delta.sort(key=lambda item: abs(float(item["delta_K"])), reverse=True)

        ticks.append(
            {
                "tick": int(record.tick),
                "active_count": len(active),
                "active_nodes": active,
                "newly_active_nodes": sorted(active_set - previous_active),
                "deactivated_nodes": sorted(previous_active - active_set),
                "dominant_node": dominant,
                "top_nodes": ranked,
                "largest_K_changes": k_delta[:top_nodes],
                "edges_fired": [
                    [int(src), int(dst)] for src, dst in record.edges_fired
                ],
            }
        )

        previous_k = {
            int(node_id): float(state.K)
            for node_id, state in record.node_states.items()
        }
        previous_active = active_set

    return ticks


def make_markdown(report: dict[str, Any]) -> str:
    row = report["input"]
    lines = [
        "# Single-Sentence TRACE Formation Report",
        "",
        "## Input",
        "",
        f"- record_id: `{row.get('record_id', '')}`",
        f"- text: {row.get('text', '')}",
        f"- valence/arousal: `{row.get('valence', '')}` / `{row.get('arousal', '')}`",
        f"- target/control: `{row.get('target', '')}` / `{row.get('control_state', '')}`",
        f"- social/action: `{row.get('social_orientation', '')}` / `{row.get('action_tendency_class', '')}`",
        "",
        "## What this report shows",
        "",
        "This report follows one sentence through EmoNet tick by tick. It shows the",
        "active neurons, dominant neuron, fired edges, activation-strength changes,",
        "and persistent fatigue/rewiring changes that remain after the sample.",
        "",
        "> Important: `delta_K` is an observed state change, not an exact additive",
        "> causal attribution. The current core does not separately log every internal",
        "> contribution before thresholding.",
        "",
        "## Run summary",
        "",
        f"- ticks_run: `{report['run']['ticks_run']}`",
        f"- dominant_route: `{' -> '.join(map(str, report['run']['dominant_route']))}`",
        f"- added_edges: `{len(report['persistent_changes']['added_edges'])}`",
        f"- removed_edges: `{len(report['persistent_changes']['removed_edges'])}`",
        f"- neurons_with_fatigue_increase: `{report['persistent_changes']['fatigue_changed_count']}`",
        "",
        "## Tick-by-tick formation",
        "",
    ]

    for tick in report["ticks"]:
        dominant = tick.get("dominant_node") or {}
        lines.extend(
            [
                f"### Tick {tick['tick']}",
                "",
                f"- active neurons: `{tick['active_count']}`",
                f"- newly active: `{tick['newly_active_nodes']}`",
                f"- deactivated: `{tick['deactivated_nodes']}`",
                f"- dominant: neuron `{dominant.get('neuron_id', '')}` with K=`{dominant.get('K', '')}`",
                f"- fired edges: `{tick['edges_fired']}`",
                "- largest K changes:",
            ]
        )
        for change in tick["largest_K_changes"]:
            lines.append(
                "  - neuron "
                f"{change['neuron_id']}: {change['previous_K']} -> "
                f"{change['current_K']} (delta {change['delta_K']:+.6f})"
            )
        lines.append("")

    lines.extend(
        [
            "## Persistent state changes after this sentence",
            "",
            f"- added edges: `{report['persistent_changes']['added_edges']}`",
            f"- removed edges: `{report['persistent_changes']['removed_edges']}`",
            "- largest fatigue increases:",
        ]
    )
    for item in report["persistent_changes"]["top_fatigue_increases"]:
        lines.append(
            f"  - neuron {item['neuron_id']}: +{item['delta']:.6f}"
        )

    lines.extend(
        [
            "",
            "## How to read it",
            "",
            "Look for the first tick where two otherwise similar inputs would diverge:",
            "a different neuron becomes dominant, a different edge fires, or a different",
            "set of neurons survives. That is the natural candidate point for a later",
            "causal ablation/perturbation experiment.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> tuple[Path, Path]:
    rows = read_rows(args.input)
    if not rows:
        raise ValueError(f"no rows in {args.input}")
    if args.row_index < 0 or args.row_index >= len(rows):
        raise IndexError(
            f"row-index {args.row_index} outside [0, {len(rows) - 1}]"
        )

    row = rows[args.row_index]
    dynamics = load_dynamics(args.config)
    model_args = make_model_args(args, dynamics)
    model = exporter.build_model(model_args)

    before_edges = edge_set(model)
    before_fatigue = fatigue_vector(model)

    outputs = model.forward(exporter.model_input_for_row(row, model_args))

    after_edges = edge_set(model)
    after_fatigue = fatigue_vector(model)
    fatigue_delta = after_fatigue - before_fatigue

    dominant_route = [int(x) for x in exporter.dominant_branch_ids(outputs)]
    ticks = build_tick_report(outputs, top_nodes=args.top_nodes)

    top_fatigue_ids = np.argsort(fatigue_delta)[::-1]
    top_fatigue = [
        {
            "neuron_id": int(idx),
            "delta": round(float(fatigue_delta[idx]), 6),
        }
        for idx in top_fatigue_ids[: args.top_nodes]
        if float(fatigue_delta[idx]) > 1e-9
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
            "dominant_route": dominant_route,
            "stim_vec": _jsonable_vec(
                outputs.get("stim_vec", exporter.model_input_for_row(row, model_args))
            ),
            "final_z": _jsonable_vec(outputs.get("z", [])),
        },
        "ticks": ticks,
        "persistent_changes": {
            "added_edges": [
                [int(src), int(dst)] for src, dst in sorted(after_edges - before_edges)
            ],
            "removed_edges": [
                [int(src), int(dst)] for src, dst in sorted(before_edges - after_edges)
            ],
            "fatigue_changed_count": int(np.count_nonzero(fatigue_delta > 1e-9)),
            "top_fatigue_increases": top_fatigue,
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    markdown = make_markdown(report)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown, encoding="utf-8")
    return args.output_json, args.output_md


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("v3.1/outputs/targeted_records_trace_normalized.csv"),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("v3.1/configs/final_dynamics_v1.json"),
    )
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--n-neurons", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--stim-source",
        choices=["auto", "text", "proxy"],
        default="auto",
    )
    parser.add_argument("--max-ticks", type=int, default=64)
    parser.add_argument("--top-nodes", type=int, default=12)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("v3.1/outputs/single_trace_formation/sample_000.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("v3.1/outputs/single_trace_formation/sample_000.md"),
    )
    return parser.parse_args()


def main() -> None:
    json_path, md_path = run(parse_args())
    print(
        json.dumps(
            {"json": str(json_path), "markdown": str(md_path)},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
