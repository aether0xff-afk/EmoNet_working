#!/usr/bin/env python3
"""Render one-sample neural state change visualization for v3.1.

The figure shows the state that can persist inside one EmoNet runtime:
edge rewiring and fatigue accumulation, alongside the tick-level activation
trace for the selected sample.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import export_neural_activation_traces as exporter  # noqa: E402


def configure_fonts() -> None:
    candidates = [
        Path("C:/Windows/Fonts/malgun.ttf"),
        Path("C:/Windows/Fonts/NanumGothic.ttf"),
        Path("C:/Windows/Fonts/gulim.ttc"),
    ]
    for candidate in candidates:
        if candidate.exists():
            font_manager.fontManager.addfont(str(candidate))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(candidate)).get_name()
            break
    plt.rcParams["axes.unicode_minus"] = False


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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_dynamics(path: Path | None) -> dict[str, Any]:
    params = dict(DEFAULT_DYNAMICS)
    if path is None:
        return params
    payload = json.loads(path.read_text(encoding="utf-8"))
    params.update(payload.get("dynamics", {}))
    return params


def make_model_args(args: argparse.Namespace, dynamics: dict[str, Any]) -> SimpleNamespace:
    values = {
        "n_neurons": args.n_neurons,
        "seed": args.seed,
        "z_encoder_mode": "stat",
        "stim_source": args.stim_source,
        "max_ticks": args.max_ticks,
        "min_ticks_before_converged": 6,
        "convergence_patience": 4,
        "progress_every": 0,
        **DEFAULT_DYNAMICS,
        **dynamics,
    }
    return SimpleNamespace(**values)


def edge_set(model: Any) -> set[tuple[int, int]]:
    edges: set[tuple[int, int]] = set()
    for neuron in model.state.neurons:
        src = int(neuron.neuron_id)
        for dst in neuron.out_neighbors:
            edges.add((src, int(dst)))
    return edges


def fatigue_vector(model: Any) -> np.ndarray:
    return np.asarray([float(neuron.fatigue) for neuron in model.state.neurons], dtype=np.float32)


def truncate_text(text: str, max_len: int = 110) -> str:
    text = " ".join(str(text).split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def render_figure(
    *,
    row: dict[str, str],
    outputs: dict[str, Any],
    activation: np.ndarray,
    before_edges: set[tuple[int, int]],
    after_edges: set[tuple[int, int]],
    before_fatigue: np.ndarray,
    after_fatigue: np.ndarray,
    output: Path,
) -> None:
    added = sorted(after_edges - before_edges)
    removed = sorted(before_edges - after_edges)
    fatigue_delta = after_fatigue - before_fatigue
    dominant_ids = exporter.dominant_branch_ids(outputs)
    active_counts = exporter.active_count_series(outputs)

    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    grid = fig.add_gridspec(3, 2, height_ratios=[1.25, 1.0, 0.85])

    title = (
        f"v3.1 one-sample neural state change | record={row.get('record_id', '')} | "
        f"ticks={outputs.get('ticks_run')} | added_edges={len(added)} | "
        f"removed_edges={len(removed)} | fatigue_changed={int(np.count_nonzero(fatigue_delta > 1e-6))}"
    )
    fig.suptitle(title, fontsize=15, fontweight="bold")

    ax0 = fig.add_subplot(grid[0, 0])
    im = ax0.imshow(activation.T, aspect="auto", interpolation="nearest", cmap="magma")
    ax0.set_title("Tick-by-neuron activation K")
    ax0.set_xlabel("tick")
    ax0.set_ylabel("neuron id")
    fig.colorbar(im, ax=ax0, fraction=0.025, pad=0.02)

    ax1 = fig.add_subplot(grid[0, 1])
    ax1.plot(active_counts, color="#1f77b4", linewidth=2.0, label="active count")
    if dominant_ids:
        ax1b = ax1.twinx()
        ax1b.plot(dominant_ids, color="#d62728", linewidth=1.5, alpha=0.8, label="dominant node id")
        ax1b.set_ylabel("dominant node id")
    ax1.set_title("Activation breadth and dominant route")
    ax1.set_xlabel("tick")
    ax1.set_ylabel("active neurons")
    ax1.grid(alpha=0.25)

    ax2 = fig.add_subplot(grid[1, 0])
    top = np.argsort(fatigue_delta)[-40:][::-1]
    ax2.bar(np.arange(len(top)), fatigue_delta[top], color="#9467bd")
    ax2.set_title("Top fatigue increases after this sample")
    ax2.set_xlabel("ranked neuron")
    ax2.set_ylabel("fatigue delta")
    ax2.set_xticks(np.arange(0, len(top), 5))
    ax2.set_xticklabels([str(int(top[i])) for i in range(0, len(top), 5)], rotation=45)
    ax2.grid(axis="y", alpha=0.25)

    ax3 = fig.add_subplot(grid[1, 1])
    if added:
        ax3.scatter([dst for _, dst in added], [src for src, _ in added], s=16, c="#2ca02c", label="added", alpha=0.75)
    if removed:
        ax3.scatter([dst for _, dst in removed], [src for src, _ in removed], s=22, marker="x", c="#d62728", label="removed", alpha=0.8)
    ax3.set_xlim(-1, activation.shape[1] + 1)
    ax3.set_ylim(-1, activation.shape[1] + 1)
    ax3.set_title("Synaptic rewiring during the sample")
    ax3.set_xlabel("destination neuron")
    ax3.set_ylabel("source neuron")
    ax3.grid(alpha=0.2)
    ax3.legend(loc="upper right")

    ax4 = fig.add_subplot(grid[2, :])
    ax4.axis("off")
    labels = [
        ("text", truncate_text(row.get("text", ""))),
        ("valence/arousal", f"{row.get('valence', '')} / {row.get('arousal', '')}"),
        ("target/control", f"{row.get('target', '')} / {row.get('control_state', '')}"),
        ("social/action", f"{row.get('social_orientation', '')} / {row.get('action_tendency_class', '')}"),
        ("edges", f"before={len(before_edges)}, after={len(after_edges)}, added={len(added)}, removed={len(removed)}"),
        ("fatigue", f"mean_delta={float(fatigue_delta.mean()):.4f}, max_delta={float(fatigue_delta.max()):.4f}"),
        ("activation", f"density={float(np.count_nonzero(activation) / max(activation.size, 1)):.4f}, max_K={float(activation.max()):.4f}"),
        ("dominant_branch_first_24", " ".join(str(node_id) for node_id in dominant_ids[:24])),
    ]
    table = ax4.table(
        cellText=[[key, value] for key, value in labels],
        colLabels=["field", "value"],
        loc="center",
        cellLoc="left",
        colWidths=[0.18, 0.78],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def run(args: argparse.Namespace) -> Path:
    configure_fonts()
    rows = read_csv(args.input)
    if not rows:
        raise ValueError(f"no rows in {args.input}")
    row = rows[args.row_index]

    dynamics = load_dynamics(args.config)
    model_args = make_model_args(args, dynamics)
    model = exporter.build_model(model_args)

    before_edges = edge_set(model)
    before_fatigue = fatigue_vector(model)
    outputs = model.forward(exporter.model_input_for_row(row, model_args))
    after_edges = edge_set(model)
    after_fatigue = fatigue_vector(model)
    activation = exporter.tick_activation_matrix(outputs, model_args.n_neurons)

    render_figure(
        row=row,
        outputs=outputs,
        activation=activation,
        before_edges=before_edges,
        after_edges=after_edges,
        before_fatigue=before_fatigue,
        after_fatigue=after_fatigue,
        output=args.output,
    )
    return args.output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("v3.1/outputs/targeted_records_trace_normalized.csv"))
    parser.add_argument("--config", type=Path, default=Path("v3.1/configs/final_dynamics_v1.json"))
    parser.add_argument("--output", type=Path, default=Path("v3.1/outputs/visualizations/sample_000_neural_state_change.png"))
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--n-neurons", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stim-source", choices=["auto", "text", "proxy"], default="auto")
    parser.add_argument("--max-ticks", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    output = run(parse_args())
    print(json.dumps({"output": str(output)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
