from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


def load_visualizer():
    path = Path("experiments/visualize_activity_guided_rewiring_clusters.py")
    spec = importlib.util.spec_from_file_location("rewiring_visualizer", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_seed(root: Path, seed: int, communities: list[int]) -> None:
    seed_dir = root / f"seed_{seed}"
    seed_dir.mkdir(parents=True)
    pd.DataFrame({"neuron": list(range(len(communities))), "community": communities}).to_csv(
        seed_dir / "neuron_communities.csv",
        index=False,
    )
    summary = {
        "seed": seed,
        "trained_minus_initial_modularity": 0.1,
        "trained_minus_null_modularity": 0.2,
        "response_coherence_gap": 0.3,
        "trained_minus_null_response_coherence_gap": 0.4,
    }
    (seed_dir / "cluster_diagnostic.json").write_text(json.dumps({"summary": summary}), encoding="utf-8")


def test_rewiring_visualizer_writes_manifest_and_figures(tmp_path: Path) -> None:
    module = load_visualizer()
    input_dir = tmp_path / "cluster"
    write_seed(input_dir, 7, [0, 0, 1, 1])
    write_seed(input_dir, 13, [0, 1, 1, 2])

    manifest = module.visualize(input_dir)

    assert manifest["seed_count"] == 2
    expected = [
        input_dir / "figures" / "seed_7_community_assignment.png",
        input_dir / "figures" / "seed_13_community_assignment.png",
        input_dir / "figures" / "community_sizes_by_seed.png",
        input_dir / "figures" / "rewiring_cluster_metrics.png",
        input_dir / "figures" / "visualization_manifest.json",
    ]
    for path in expected:
        assert path.exists()
