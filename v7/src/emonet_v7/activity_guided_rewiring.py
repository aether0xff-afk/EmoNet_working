"""Activity-guided structural rewiring for memory-threshold RSNN ablations.

The utility changes sparse adjacency without semantic labels. It discovers
functional community candidates from train-episode neuron-memory profiles,
removes weak inter-community edges, and adds high-coactivity intra-community
edges while preserving the directed edge budget.

This is an experimental structural-plasticity ablation, not the final EmoNet
rewiring rule.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import torch
from sklearn.cluster import KMeans

from .memory_threshold_rsnn import NeuronMemoryThresholdRSNN


@dataclass
class RewiringReport:
    """Diagnostics for one adjacency rewiring operation."""

    seed: int
    requested_fraction: float
    edge_count_before: int
    edge_count_after: int
    rewired_edge_count: int
    cluster_count: int
    community_sizes: list[int]
    mean_removed_weight_abs: float
    mean_added_similarity: float
    functional_modularity: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def safe_correlation_matrix(response_by_episode: np.ndarray) -> np.ndarray:
    """Return a finite positive neuron-neuron coactivity matrix."""

    if response_by_episode.ndim != 2:
        raise ValueError("response_by_episode must have shape [episodes, neurons]")
    if response_by_episode.shape[0] < 2:
        raise ValueError("at least two episode profiles are required")
    corr = np.corrcoef(response_by_episode, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.clip(corr, 0.0, 1.0)
    np.fill_diagonal(corr, 0.0)
    return corr


def weighted_modularity(adjacency: np.ndarray, labels: np.ndarray) -> float:
    """Compute undirected weighted modularity for one partition."""

    total_weight_twice = float(adjacency.sum())
    if total_weight_twice <= 0.0:
        return 0.0
    degree = adjacency.sum(axis=1)
    expected = np.outer(degree, degree) / total_weight_twice
    same = labels[:, None] == labels[None, :]
    return float(((adjacency - expected) * same).sum() / total_weight_twice)


def spectral_labels(adjacency: np.ndarray, *, cluster_count: int, seed: int) -> np.ndarray:
    """Find spectral-clustering labels without semantic supervision."""

    neuron_count = adjacency.shape[0]
    if cluster_count < 2 or cluster_count >= neuron_count:
        raise ValueError("cluster_count must remain between 2 and num_neurons - 1")
    degree = adjacency.sum(axis=1)
    safe_degree = np.where(degree > 1e-12, degree, 1.0)
    inv_sqrt = np.diag(1.0 / np.sqrt(safe_degree))
    normalized = inv_sqrt @ adjacency @ inv_sqrt
    _, eigenvectors = np.linalg.eigh(normalized)
    features = eigenvectors[:, -cluster_count:]
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.where(norms > 1e-12, norms, 1.0)
    return KMeans(n_clusters=cluster_count, random_state=seed, n_init=20).fit_predict(features)


def discover_functional_communities(
    similarity: np.ndarray,
    *,
    min_clusters: int,
    max_clusters: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    """Choose k by maximum modularity in the coactivity graph."""

    max_allowed = min(max_clusters, similarity.shape[0] - 1)
    best_labels = None
    best_score = float("-inf")
    for cluster_count in range(min_clusters, max_allowed + 1):
        labels = spectral_labels(similarity, cluster_count=cluster_count, seed=seed)
        score = weighted_modularity(similarity, labels)
        if score > best_score:
            best_score = score
            best_labels = labels
    if best_labels is None:
        raise RuntimeError("functional community discovery produced no partition")
    return best_labels, best_score


def _candidate_existing_edges(mask: np.ndarray, weight_abs: np.ndarray, labels: np.ndarray) -> list[tuple[float, int, int]]:
    """Rank removable edges: weak inter-community edges first."""

    candidates: list[tuple[float, int, int]] = []
    target_indices, source_indices = np.nonzero(mask)
    for target, source in zip(target_indices.tolist(), source_indices.tolist(), strict=True):
        inter_penalty = 0.0 if labels[target] != labels[source] else 1.0
        score = inter_penalty + float(weight_abs[target, source])
        candidates.append((score, target, source))
    candidates.sort(key=lambda item: item[0])
    return candidates


def _candidate_new_edges(mask: np.ndarray, similarity: np.ndarray, labels: np.ndarray) -> list[tuple[float, int, int]]:
    """Rank new directed edges: high-coactivity intra-community pairs first."""

    neuron_count = mask.shape[0]
    candidates: list[tuple[float, int, int]] = []
    for target in range(neuron_count):
        for source in range(neuron_count):
            if target == source or mask[target, source]:
                continue
            if labels[target] != labels[source]:
                continue
            candidates.append((float(similarity[target, source]), target, source))
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates


def reset_optimizer_state_for_changed_edges(
    *,
    optimizer: torch.optim.Optimizer,
    parameter: torch.nn.Parameter,
    changed_mask: torch.Tensor,
) -> None:
    """Clear Adam-style moments only for structurally modified edge entries."""

    state = optimizer.state.get(parameter)
    if not state:
        return
    mask = changed_mask.to(device=parameter.device, dtype=torch.bool)
    for value in state.values():
        if torch.is_tensor(value) and value.shape == parameter.shape:
            value.masked_fill_(mask, 0.0)


def rewire_from_memory_profiles(
    *,
    snn: NeuronMemoryThresholdRSNN,
    response_by_episode: np.ndarray,
    fraction: float,
    seed: int,
    min_clusters: int = 2,
    max_clusters: int = 8,
    new_weight_scale: float = 0.05,
) -> tuple[RewiringReport, torch.Tensor]:
    """Rewire a fixed fraction of edges while preserving edge count.

    Returns the report and a boolean mask of structurally changed weight entries
    so the caller can clear optimizer moments only for those entries.
    """

    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must remain in [0, 1]")
    if new_weight_scale <= 0.0:
        raise ValueError("new_weight_scale must be positive")
    similarity = safe_correlation_matrix(response_by_episode)
    labels, functional_modularity = discover_functional_communities(
        similarity,
        min_clusters=min_clusters,
        max_clusters=max_clusters,
        seed=seed,
    )
    mask = snn.recurrent_mask.detach().cpu().numpy().astype(bool)
    weight = snn.recurrent_weight.detach().cpu().numpy()
    weight_abs = np.abs(weight)
    edge_count_before = int(mask.sum())
    requested = int(round(edge_count_before * fraction))
    if fraction > 0.0:
        requested = max(1, requested)
    removable = _candidate_existing_edges(mask, weight_abs, labels)
    addable = _candidate_new_edges(mask, similarity, labels)
    rewired_count = min(requested, len(removable), len(addable))
    removed = removable[:rewired_count]
    added = addable[:rewired_count]
    changed_mask = torch.zeros_like(snn.recurrent_mask, dtype=torch.bool)

    with torch.no_grad():
        for _, target, source in removed:
            snn.recurrent_mask[target, source] = 0.0
            snn.recurrent_weight[target, source] = 0.0
            changed_mask[target, source] = True
        for similarity_value, target, source in added:
            snn.recurrent_mask[target, source] = 1.0
            latent_sign = torch.sign(snn.recurrent_weight[target, source])
            if float(latent_sign) == 0.0:
                latent_sign = torch.tensor(1.0, device=snn.recurrent_weight.device)
            scaled = new_weight_scale * (0.5 + 0.5 * similarity_value)
            snn.recurrent_weight[target, source] = latent_sign * scaled
            changed_mask[target, source] = True
        snn.recurrent_mask.fill_diagonal_(0.0)

    edge_count_after = int(snn.recurrent_mask.detach().sum().item())
    if edge_count_after != edge_count_before:
        raise RuntimeError("rewiring must preserve the directed edge budget")
    sizes = [int((labels == community).sum()) for community in sorted(set(labels.tolist()))]
    report = RewiringReport(
        seed=seed,
        requested_fraction=float(fraction),
        edge_count_before=edge_count_before,
        edge_count_after=edge_count_after,
        rewired_edge_count=rewired_count,
        cluster_count=len(sizes),
        community_sizes=sizes,
        mean_removed_weight_abs=float(np.mean([weight_abs[target, source] for _, target, source in removed])) if removed else 0.0,
        mean_added_similarity=float(np.mean([score for score, _, _ in added])) if added else 0.0,
        functional_modularity=float(functional_modularity),
    )
    return report, changed_mask
