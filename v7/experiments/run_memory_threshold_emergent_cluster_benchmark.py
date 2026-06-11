"""Diagnose adjacency-based community structure in the selected memory-threshold SNN.

This is a conservative first cluster benchmark. The sparse connectivity mask is
fixed at initialization, while recurrent weights are trained. Communities are
discovered from the trained weighted adjacency only; semantic labels are used
only after community discovery for evaluation summaries.

The benchmark asks:
1. Does trained weighted adjacency show more modular organization than the same
   model at initialization and edge-weight-shuffled null graphs?
2. Do neurons inside discovered communities show more coherent neuron-local
   memory responses than size-matched random community assignments?
3. Do communities exhibit non-uniform coarse semantic response profiles?
"""

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
from run_memory_threshold_context_structure_benchmark import checkpoint_path

try:
    from sklearn.cluster import KMeans
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("scikit-learn is required for spectral community detection") from exc


AXES = ("valence", "arousal", "certainty", "social_distance")
DEFAULT_CONFIG_KEY = "feedback_0.050__threshold_0.500__accumulation_decay_0.850"
INTERPRETATION_BOUNDARY = (
    "This diagnostic tests weighted-adjacency communities and within-community memory-response coherence under a controlled fixture. "
    "Because the sparse mask is fixed at initialization, it does not establish fully self-organized topology, stable neuron roles, emotional ground truth, or biological fidelity."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="runs/memory_threshold_parameter_sweep_lmstudio")
    parser.add_argument("--output", default="runs/memory_threshold_emergent_cluster_best_lmstudio")
    parser.add_argument("--fixture", default="fixtures/semantic_alignment_episodes.yaml")
    parser.add_argument("--config-key", default=DEFAULT_CONFIG_KEY)
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


def weighted_adjacency(model) -> np.ndarray:
    weight = (model.snn.recurrent_weight * model.snn.recurrent_mask).detach().cpu().numpy()
    adjacency = np.abs(weight)
    adjacency = (adjacency + adjacency.T) / 2.0
    np.fill_diagonal(adjacency, 0.0)
    return adjacency


def modularity(adjacency: np.ndarray, labels: np.ndarray) -> float:
    total_weight_twice = float(adjacency.sum())
    if total_weight_twice <= 0.0:
        return 0.0
    degree = adjacency.sum(axis=1)
    expected = np.outer(degree, degree) / total_weight_twice
    same = labels[:, None] == labels[None, :]
    return float(((adjacency - expected) * same).sum() / total_weight_twice)


def spectral_labels(adjacency: np.ndarray, *, cluster_count: int, seed: int) -> np.ndarray:
    if cluster_count < 2 or cluster_count >= adjacency.shape[0]:
        raise ValueError("cluster_count must remain between 2 and num_neurons - 1")
    degree = adjacency.sum(axis=1)
    safe_degree = np.where(degree > 1e-12, degree, 1.0)
    inv_sqrt = np.diag(1.0 / np.sqrt(safe_degree))
    normalized = inv_sqrt @ adjacency @ inv_sqrt
    eigenvalues, eigenvectors = np.linalg.eigh(normalized)
    features = eigenvectors[:, -cluster_count:]
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.where(norms > 1e-12, norms, 1.0)
    return KMeans(n_clusters=cluster_count, random_state=seed, n_init=20).fit_predict(features)


def discover_communities(adjacency: np.ndarray, *, min_clusters: int, max_clusters: int, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    max_allowed = min(max_clusters, adjacency.shape[0] - 1)
    candidates: list[dict[str, Any]] = []
    best_labels = None
    best_modularity = float("-inf")
    for cluster_count in range(min_clusters, max_allowed + 1):
        labels = spectral_labels(adjacency, cluster_count=cluster_count, seed=seed)
        score = modularity(adjacency, labels)
        sizes = [int((labels == value).sum()) for value in sorted(set(labels.tolist()))]
        row = {"cluster_count": cluster_count, "modularity": score, "community_sizes": sizes}
        candidates.append(row)
        if score > best_modularity:
            best_modularity = score
            best_labels = labels
    if best_labels is None:
        raise RuntimeError("community discovery produced no candidate partition")
    return best_labels, {"selected_modularity": best_modularity, "candidates": candidates}


def shuffled_weight_adjacency(adjacency: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    upper = np.triu_indices_from(adjacency, k=1)
    weights = adjacency[upper].copy()
    rng.shuffle(weights)
    shuffled = np.zeros_like(adjacency)
    shuffled[upper] = weights
    shuffled[(upper[1], upper[0])] = weights
    return shuffled


def community_sizes(labels: np.ndarray) -> list[int]:
    return [int((labels == value).sum()) for value in sorted(set(labels.tolist()))]


def safe_corr(left: np.ndarray, right: np.ndarray) -> float:
    if float(np.std(left)) <= 1e-12 or float(np.std(right)) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def response_coherence(response_by_neuron: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    within: list[float] = []
    between: list[float] = []
    neuron_count = response_by_neuron.shape[0]
    for left in range(neuron_count):
        for right in range(left + 1, neuron_count):
            corr = safe_corr(response_by_neuron[left], response_by_neuron[right])
            if labels[left] == labels[right]:
                within.append(corr)
            else:
                between.append(corr)
    within_mean = float(np.mean(within)) if within else 0.0
    between_mean = float(np.mean(between)) if between else 0.0
    return {
        "within_community_response_correlation": within_mean,
        "between_community_response_correlation": between_mean,
        "response_coherence_gap": within_mean - between_mean,
    }


def permuted_labels(labels: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    shuffled = labels.copy()
    rng.shuffle(shuffled)
    return shuffled


def axis_selectivity(response_by_neuron: np.ndarray, episode_labels: np.ndarray, community_labels: np.ndarray) -> dict[str, Any]:
    neuron_axis_corr = np.zeros((response_by_neuron.shape[0], len(AXES)), dtype=np.float64)
    for neuron in range(response_by_neuron.shape[0]):
        for axis_index in range(len(AXES)):
            neuron_axis_corr[neuron, axis_index] = safe_corr(response_by_neuron[neuron], episode_labels[:, axis_index])
    communities: list[dict[str, Any]] = []
    for community in sorted(set(community_labels.tolist())):
        selected = neuron_axis_corr[community_labels == community]
        mean_abs = np.abs(selected).mean(axis=0)
        signed = selected.mean(axis=0)
        communities.append(
            {
                "community": int(community),
                "size": int(selected.shape[0]),
                "mean_abs_axis_correlation": {axis: float(mean_abs[index]) for index, axis in enumerate(AXES)},
                "mean_signed_axis_correlation": {axis: float(signed[index]) for index, axis in enumerate(AXES)},
                "dominant_axis": AXES[int(np.argmax(mean_abs))],
                "dominant_axis_mean_abs_correlation": float(mean_abs.max()),
            }
        )
    return {
        "communities": communities,
        "community_dominant_axis_mean_abs_correlation": float(np.mean([row["dominant_axis_mean_abs_correlation"] for row in communities])),
        "neuron_axis_mean_abs_correlation": float(np.abs(neuron_axis_corr).mean()),
    }


def unique_episode_steps(pairs) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for pair in pairs:
        mapping[pair.left_episode_id] = pair.step_index
        mapping[pair.right_episode_id] = pair.step_index
    return mapping


def extract_memory_profiles(*, model, episode_steps: dict[str, int], episode_by_id, semantic_labels, text_encoder, args, device):
    episode_ids = sorted(episode_steps)
    memory_rows: list[np.ndarray] = []
    label_rows: list[np.ndarray] = []
    with torch.no_grad():
        memory.set_mode(model, training=False)
        for episode_id in episode_ids:
            episode = episode_by_id[episode_id]
            output = memory.run_sequence(
                model=model,
                episode=episode,
                step_index=episode_steps[episode_id],
                condition="real_history",
                swapped_history_source=None,
                text_encoder=text_encoder,
                args=args,
                device=device,
            )
            memory_rows.append(output.final_memory_strength.detach().cpu().reshape(-1).numpy())
            label_rows.append(semantic_labels[episode_id])
    response_by_episode = np.stack(memory_rows)
    return {
        "episode_ids": episode_ids,
        "response_by_neuron": response_by_episode.T,
        "episode_labels": np.stack(label_rows),
    }


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
    logger.section("memory-threshold emergent-cluster diagnostic")
    logger.log("config", "Adjacency 기반 community 진단 설정을 불러왔다.", **vars(args))

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
        model = memory.build_model(text_dim=text_encoder.output_dim, num_neurons=args.num_neurons, seed=seed, feedback_strength=args.feedback_strength, args=args, device=device)
        initial_adjacency = weighted_adjacency(model)
        initial_labels, initial_detection = discover_communities(initial_adjacency, min_clusters=args.min_clusters, max_clusters=args.max_clusters, seed=seed)
        path = checkpoint_path(input_dir, args.config_key, seed)
        if not path.exists():
            raise FileNotFoundError(f"best checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        memory.load_state_dict_for(model, checkpoint)
        trained_adjacency = weighted_adjacency(model)
        trained_labels, trained_detection = discover_communities(trained_adjacency, min_clusters=args.min_clusters, max_clusters=args.max_clusters, seed=seed)
        profiles = extract_memory_profiles(model=model, episode_steps=episode_steps, episode_by_id=episode_by_id, semantic_labels=semantic_labels, text_encoder=text_encoder, args=args, device=device)
        trained_coherence = response_coherence(profiles["response_by_neuron"], trained_labels)
        initial_modularity = modularity(initial_adjacency, initial_labels)
        trained_modularity = modularity(trained_adjacency, trained_labels)

        null_modularity: list[float] = []
        null_coherence_gap: list[float] = []
        for permutation in range(args.null_permutations):
            null_adjacency = shuffled_weight_adjacency(trained_adjacency, rng)
            null_labels, _ = discover_communities(null_adjacency, min_clusters=args.min_clusters, max_clusters=args.max_clusters, seed=seed + permutation + 1)
            null_modularity.append(modularity(null_adjacency, null_labels))
            null_coherence_gap.append(response_coherence(profiles["response_by_neuron"], permuted_labels(trained_labels, rng))["response_coherence_gap"])

        selectivity = axis_selectivity(profiles["response_by_neuron"], profiles["episode_labels"], trained_labels)
        row = {
            "seed": seed,
            "config_key": args.config_key,
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
                    "interpretation_boundary": INTERPRETATION_BOUNDARY,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        pd.DataFrame({"neuron": np.arange(args.num_neurons), "community": trained_labels}).to_csv(seed_output / "neuron_communities.csv", index=False, encoding="utf-8-sig")
        logger.log("seed.done", "Adjacency community 진단을 마쳤다.", **row)

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "by_seed_cluster.csv", index=False, encoding="utf-8-sig")
    numeric_columns = [column for column in frame.columns if column not in {"seed", "config_key", "selected_community_sizes"}]
    summary = frame[numeric_columns].agg(["mean", "std", "min", "max"]).T.reset_index().rename(columns={"index": "metric"})
    summary.to_csv(output / "summary_metrics.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "source_sweep": str(input_dir),
        "config_key": args.config_key,
        "feedback_strength": args.feedback_strength,
        "memory_threshold": args.memory_threshold,
        "accumulation_decay": args.accumulation_decay,
        "memory_decay": args.memory_decay,
        "fixture": args.fixture,
        "seeds": args.seeds,
        "community_detection": "weighted undirected adjacency from absolute recurrent weights; normalized spectral clustering; k selected by maximum weighted modularity",
        "nulls": "edge-weight shuffle for modularity; size-preserving neuron-label permutation for functional response-coherence gap",
        "semantic_labels_used_for_training": False,
        "semantic_labels_used_for_community_discovery": False,
        "interpretation_boundary": INTERPRETATION_BOUNDARY,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.log("benchmark.done", "Memory-threshold emergent-cluster 진단을 마쳤다.", files=["run_log.jsonl", "embedding_cache.json", "by_seed_cluster.csv", "summary_metrics.csv", "metadata.json"])
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
