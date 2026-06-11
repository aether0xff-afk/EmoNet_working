"""Render activity-guided rewiring community diagnostics.

The visualizer consumes outputs from
``run_activity_guided_rewiring_emergent_cluster_benchmark.py``. It does not
re-run training or reload checkpoints, so it can be used on copied result
folders.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METRIC_COLUMNS = [
    "trained_minus_initial_modularity",
    "trained_minus_null_modularity",
    "response_coherence_gap",
    "trained_minus_null_response_coherence_gap",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="runs/activity_guided_rewiring_pipeline_lmstudio/rewired_cluster")
    parser.add_argument("--output")
    return parser.parse_args()


def seed_directories(input_dir: Path) -> list[Path]:
    seeds = [path for path in input_dir.glob("seed_*") if path.is_dir()]
    return sorted(seeds, key=lambda path: int(path.name.split("_", 1)[1]))


def read_seed(seed_dir: Path) -> dict[str, Any]:
    communities_path = seed_dir / "neuron_communities.csv"
    diagnostic_path = seed_dir / "cluster_diagnostic.json"
    if not communities_path.exists():
        raise FileNotFoundError(f"community assignment file not found: {communities_path}")
    if not diagnostic_path.exists():
        raise FileNotFoundError(f"cluster diagnostic file not found: {diagnostic_path}")
    communities = pd.read_csv(communities_path)
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    summary = dict(diagnostic.get("summary", {}))
    return {
        "seed": int(summary.get("seed", seed_dir.name.split("_", 1)[1])),
        "communities": communities,
        "summary": summary,
    }


def render_seed_assignment(seed_data: dict[str, Any], output_dir: Path) -> Path:
    seed = int(seed_data["seed"])
    communities = seed_data["communities"].sort_values("neuron")
    labels = communities["community"].to_numpy(dtype=int)
    unique_labels = sorted(set(labels.tolist()))
    remap = {label: index for index, label in enumerate(unique_labels)}
    compact = np.array([remap[label] for label in labels], dtype=int).reshape(1, -1)

    fig_width = max(8.0, min(18.0, len(labels) / 10.0))
    fig, ax = plt.subplots(figsize=(fig_width, 2.2), constrained_layout=True)
    ax.imshow(compact, aspect="auto", cmap="tab20")
    ax.set_title(f"Seed {seed} neuron community assignment")
    ax.set_xlabel("Neuron index")
    ax.set_yticks([])
    ax.set_xticks(np.linspace(0, len(labels) - 1, num=min(9, len(labels)), dtype=int))
    path = output_dir / f"seed_{seed}_community_assignment.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def render_size_summary(seed_rows: list[dict[str, Any]], output_dir: Path) -> Path:
    rows: list[dict[str, Any]] = []
    for seed_data in seed_rows:
        counts = seed_data["communities"]["community"].value_counts().sort_index()
        for community, size in counts.items():
            rows.append({"seed": int(seed_data["seed"]), "community": int(community), "size": int(size)})
    frame = pd.DataFrame(rows)
    pivot = frame.pivot_table(index="seed", columns="community", values="size", fill_value=0, aggfunc="sum")

    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    bottom = np.zeros(len(pivot), dtype=float)
    x = np.arange(len(pivot))
    for community in pivot.columns:
        values = pivot[community].to_numpy(dtype=float)
        ax.bar(x, values, bottom=bottom, label=f"C{community}")
        bottom += values
    ax.set_title("Community sizes by seed")
    ax.set_xlabel("Seed")
    ax.set_ylabel("Neuron count")
    ax.set_xticks(x, [str(seed) for seed in pivot.index])
    ax.legend(title="Community", ncols=4, fontsize=8)
    path = output_dir / "community_sizes_by_seed.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def render_metric_summary(seed_rows: list[dict[str, Any]], output_dir: Path) -> Path:
    rows = []
    for seed_data in seed_rows:
        summary = seed_data["summary"]
        rows.append({"seed": int(seed_data["seed"]), **{column: float(summary.get(column, 0.0)) for column in METRIC_COLUMNS}})
    frame = pd.DataFrame(rows)
    means = frame[METRIC_COLUMNS].mean()
    stds = frame[METRIC_COLUMNS].std(ddof=0)

    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    x = np.arange(len(METRIC_COLUMNS))
    ax.bar(x, means.to_numpy(dtype=float), yerr=stds.to_numpy(dtype=float), capsize=4, color="#4C78A8")
    ax.axhline(0.0, color="#333333", linewidth=0.8)
    ax.set_title("Rewiring cluster diagnostic metrics")
    ax.set_ylabel("Mean across seeds")
    ax.set_xticks(x, [column.replace("_", "\n") for column in METRIC_COLUMNS], fontsize=8)
    path = output_dir / "rewiring_cluster_metrics.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def visualize(input_dir: Path, output_dir: Path | None = None) -> dict[str, Any]:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir) if output_dir else input_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = seed_directories(input_dir)
    if not seeds:
        raise FileNotFoundError(f"no seed_* result directories found in {input_dir}")
    seed_rows = [read_seed(seed_dir) for seed_dir in seeds]
    files = [render_seed_assignment(seed_data, output_dir) for seed_data in seed_rows]
    files.append(render_size_summary(seed_rows, output_dir))
    files.append(render_metric_summary(seed_rows, output_dir))
    manifest = {
        "input": str(input_dir),
        "output": str(output_dir),
        "seed_count": len(seed_rows),
        "files": [str(path) for path in files],
        "interpretation_boundary": (
            "These figures visualize discovered adjacency communities and summary diagnostics only. "
            "They do not establish stable neuron roles, emotional ground truth, or biological fidelity."
        ),
    }
    (output_dir / "visualization_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    args = parse_args()
    manifest = visualize(Path(args.input), Path(args.output) if args.output else None)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
