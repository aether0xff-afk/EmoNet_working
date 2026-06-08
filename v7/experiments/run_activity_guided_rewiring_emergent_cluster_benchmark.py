"""Re-run adjacency-community diagnostics on activity-guided rewired checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

import run_context_objective_benchmark as base
import run_memory_threshold_semantic_benchmark as memory
import run_trace_semantic_alignment_benchmark as semantic
from run_memory_threshold_emergent_cluster_benchmark import (
    INTERPRETATION_BOUNDARY,
    axis_selectivity,
    community_sizes,
    discover_communities,
    extract_memory_profiles,
    modularity,
    response_coherence,
    shuffled_weight_adjacency,
    unique_episode_steps,
    weighted_adjacency,
)


MODEL_TYPE = "snn_memory_feedback_rewired"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="runs/activity_guided_rewiring_semantic_benchmark_lmstudio")
    parser.add_argument("--output", default="runs/activity_guided_rewiring_emergent_cluster_lmstudio")
    parser.add_argument("--fixture", default="fixtures/semantic_alignment_episodes.yaml")
    parser.add_argument("--feedback-strength", type=float, default=0.05)
    parser.add_argument("--memory-threshold", type=float, default=0.50)
    parser.add_argument("--accumulation-decay", type=float, default=0.85)
    parser.add_argument("--memory-decay", type=float, default=0.98)
    parser.add_argument("--encoder", choices=["hash", "lmstudio"], default="hash")
    parser.add_argument("--base-url")
    parser.add_argument("--embedding-model", default="text-embedding-nomic-embed-text-v1.5")
    parser.add_argument("--num-neurons", type=int, default=128)
    parser.add_argument("--event-ticks", type=int, default=16)
    parser.add_argument("--stimulation-ticks", type=int, default=6)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 13, 21, 42, 100])
    parser.add_argument("--min-clusters", type=int, default=2)
    parser.add_argument("--max-clusters", type=int, default=8)
    parser.add_argument("--null-permutations", type=int, default=64)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def checkpoint_path(input_dir: Path, seed: int) -> Path:
    return input_dir / f"seed_{seed}" / MODEL_TYPE / "best_checkpoint.pt"


def main() -> None:
    args = parse_args()
    if not args.seeds:
        raise ValueError("--seeds must not be empty")
    if args.null_permutations <= 0:
        raise ValueError("--null-permutations must be positive")
    if args.min_clusters < 2 or args.max_clusters < args.min_clusters:
        raise ValueError("invalid cluster-count range")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    logger = base.RunLogger(output_dir=output, verbose=not args.quiet)
    logger.section("activity-guided rewiring emergent-cluster diagnostic")
    logger.log("config", "Rewired adjacency community 진단 설정을 불러왔다.", **vars(args))

    device = torch.device(args.device)
    episodes = base.load_episodes(args.fixture)
    episode_by_id = {episode.episode_id: episode for episode in episodes}
    semantic_labels = semantic.load_semantic_labels(args.fixture)
    train_pairs = base.load_contrast_pairs(args.fixture, split="train")
    validation_pairs = base.load_contrast_pairs(args.fixture, split="validation")
    base.validate_contrast_pairs(episodes, train_pairs + validation_pairs)
    episode_steps = unique_episode_steps(train_pairs + validation_pairs)
    text_encoder = base.build_text_encoder(args, output)
    logger.log("embedding.ready", "Embedding encoder와 캐시가 준비됐다.", output_dim=text_encoder.output_dim)

    rows: list[dict[str, Any]] = []
    input_dir = Path(args.input)
    for seed in args.seeds:
        rng = np.random.default_rng(seed)
        model = memory.build_model(
            text_dim=text_encoder.output_dim,
            num_neurons=args.num_neurons,
            seed=seed,
            feedback_strength=args.feedback_strength,
            args=args,
            device=device,
        )
        initial_adjacency = weighted_adjacency(model)
        initial_labels, initial_detection = discover_communities(
            initial_adjacency,
            min_clusters=args.min_clusters,
            max_clusters=args.max_clusters,
            seed=seed,
        )
        path = checkpoint_path(input_dir, seed)
        if not path.exists():
            raise FileNotFoundError(f"best checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        memory.load_state_dict_for(model, checkpoint)
        trained_adjacency = weighted_adjacency(model)
        trained_labels, trained_detection = discover_communities(
            trained_adjacency,
            min_clusters=args.min_clusters,
            max_clusters=args.max_clusters,
            seed=seed,
        )
        profiles = extract_memory_profiles(
            model=model,
            episode_steps=episode_steps,
            episode_by_id=episode_by_id,
            semantic_labels=semantic_labels,
            text_encoder=text_encoder,
            args=args,
            device=device,
        )
        trained_coherence = response_coherence(profiles["response_by_neuron"], trained_labels)
        initial_modularity = modularity(initial_adjacency, initial_labels)
        trained_modularity = modularity(trained_adjacency, trained_labels)

        null_modularity: list[float] = []
        null_coherence_gap: list[float] = []
        for permutation in range(args.null_permutations):
            null_adjacency = shuffled_weight_adjacency(trained_adjacency, rng)
            null_labels, _ = discover_communities(
                null_adjacency,
                min_clusters=args.min_clusters,
                max_clusters=args.max_clusters,
                seed=seed + permutation + 1,
            )
            null_modularity.append(modularity(null_adjacency, null_labels))
            shuffled_labels = trained_labels.copy()
            rng.shuffle(shuffled_labels)
            null_coherence_gap.append(
                response_coherence(profiles["response_by_neuron"], shuffled_labels)["response_coherence_gap"]
            )

        selectivity = axis_selectivity(profiles["response_by_neuron"], profiles["episode_labels"], trained_labels)
        rewiring_history = checkpoint.get("rewiring_history", [])
        row = {
            "seed": seed,
            "rewiring_event_count": len(rewiring_history),
            "rewired_edges_total": int(sum(item["rewired_edge_count"] for item in rewiring_history)),
            "initial_modularity": initial_modularity,
            "trained_modularity": trained_modularity,
            "trained_minus_initial_modularity": trained_modularity - initial_modularity,
            "null_modularity_mean": float(np.mean(null_modularity)),
            "null_modularity_std": float(np.std(null_modularity)),
            "trained_minus_null_modularity": trained_modularity - float(np.mean(null_modularity)),
            "selected_cluster_count": int(len(set(trained_labels.tolist()))),
            "selected_community_sizes": community_sizes(trained_labels),
            **trained_coherence,
            "null_response_coherence_gap_mean": float(np.mean(null_coherence_gap)),
            "null_response_coherence_gap_std": float(np.std(null_coherence_gap)),
            "trained_minus_null_response_coherence_gap": trained_coherence["response_coherence_gap"] - float(np.mean(null_coherence_gap)),
            "community_dominant_axis_mean_abs_correlation": selectivity["community_dominant_axis_mean_abs_correlation"],
            "neuron_axis_mean_abs_correlation": selectivity["neuron_axis_mean_abs_correlation"],
        }
        rows.append(row)
        seed_output = output / f"seed_{seed}"
        seed_output.mkdir(parents=True, exist_ok=True)
        (seed_output / "cluster_diagnostic.json").write_text(
            json.dumps(
                {
                    "summary": row,
                    "initial_detection": initial_detection,
                    "trained_detection": trained_detection,
                    "axis_selectivity": selectivity,
                    "rewiring_history": rewiring_history,
                    "interpretation_boundary": INTERPRETATION_BOUNDARY,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        pd.DataFrame({"neuron": np.arange(args.num_neurons), "community": trained_labels}).to_csv(
            seed_output / "neuron_communities.csv",
            index=False,
            encoding="utf-8-sig",
        )
        logger.log("seed.done", "Rewired adjacency community 진단을 마쳤다.", **row)

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "by_seed_cluster.csv", index=False, encoding="utf-8-sig")
    numeric_columns = [column for column in frame.columns if column not in {"seed", "selected_community_sizes"}]
    summary = frame[numeric_columns].agg(["mean", "std", "min", "max"]).T.reset_index().rename(columns={"index": "metric"})
    summary.to_csv(output / "summary_metrics.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "source_rewiring_benchmark": str(input_dir),
        "fixture": args.fixture,
        "seeds": args.seeds,
        "community_detection": "weighted undirected adjacency from absolute recurrent weights; normalized spectral clustering; k selected by maximum weighted modularity",
        "nulls": "edge-weight shuffle for modularity; size-preserving neuron-label permutation for functional response-coherence gap",
        "semantic_labels_used_for_training": False,
        "semantic_labels_used_for_rewiring": False,
        "semantic_labels_used_for_community_discovery": False,
        "interpretation_boundary": INTERPRETATION_BOUNDARY,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.log("benchmark.done", "Rewired emergent-cluster 진단을 마쳤다.", files=["run_log.jsonl", "embedding_cache.json", "by_seed_cluster.csv", "summary_metrics.csv", "metadata.json"])
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
