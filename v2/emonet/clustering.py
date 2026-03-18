from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import networkx as nx
import torch

from .config import AppConfig


@dataclass
class ClusterStats:
    cluster_ids: torch.Tensor
    cluster_sizes: List[int]
    mean_degree: torch.Tensor
    mean_activity: torch.Tensor
    var_activity: torch.Tensor
    mean_memory: torch.Tensor
    modularity: float


class StructuralClusterManager:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.cluster_ids: torch.Tensor | None = None
        self.last_recluster_episode: int = -10**9
        self.episode_index: int = 0

    def _build_graph(self, adjacency: torch.Tensor, weights: torch.Tensor) -> nx.Graph:
        n = adjacency.shape[0]
        g = nx.Graph()
        g.add_nodes_from(range(n))
        sym = (weights.abs() + weights.abs().T) / 2.0
        mask = adjacency.bool() & (sym > self.config.cluster.edge_threshold)
        idx = mask.nonzero(as_tuple=False)
        for i, j in idx.tolist():
            if i >= j:
                continue
            g.add_edge(i, j, weight=float(sym[i, j].item()))
        return g

    def initialize_from_graph(self, adjacency: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        graph = self._build_graph(adjacency, weights)
        n = adjacency.shape[0]
        if graph.number_of_edges() == 0:
            self.cluster_ids = torch.arange(n, dtype=torch.long) // max(1, self.config.cluster.min_cluster_size)
            return self.cluster_ids
        components = [sorted(comp) for comp in nx.connected_components(graph)]
        components.sort(key=len, reverse=True)
        cluster_ids = torch.full((n,), -1, dtype=torch.long)
        cid = 0
        for comp in components:
            if len(comp) < self.config.cluster.min_cluster_size and cid > 0:
                cluster_ids[comp] = cid - 1
                continue
            if len(comp) > self.config.cluster.max_cluster_size:
                chunk = self.config.cluster.max_cluster_size
                for s in range(0, len(comp), chunk):
                    part = comp[s : s + chunk]
                    cluster_ids[part] = cid
                    cid += 1
            else:
                cluster_ids[comp] = cid
                cid += 1
        unassigned = (cluster_ids < 0).nonzero(as_tuple=False).flatten().tolist()
        for idx in unassigned:
            cluster_ids[idx] = 0
        self.cluster_ids = cluster_ids
        return cluster_ids

    def maybe_recluster(self, adjacency: torch.Tensor, weights: torch.Tensor, edge_change_ratio: float, modularity: float) -> bool:
        self.episode_index += 1
        if self.cluster_ids is None:
            self.initialize_from_graph(adjacency, weights)
            self.last_recluster_episode = self.episode_index
            return True
        enough_gap = (self.episode_index - self.last_recluster_episode) >= self.config.cluster.recluster_min_gap
        if enough_gap and (edge_change_ratio >= self.config.cluster.recluster_edge_change_ratio or modularity < 0.15):
            self.initialize_from_graph(adjacency, weights)
            self.last_recluster_episode = self.episode_index
            return True
        return False

    def compute_cluster_stats(self, adjacency: torch.Tensor, weights: torch.Tensor, activity: torch.Tensor, memory: torch.Tensor, calculate_modularity: bool = False) -> ClusterStats:
        if self.cluster_ids is None:
            self.initialize_from_graph(adjacency, weights)
        assert self.cluster_ids is not None
        cluster_ids = self.cluster_ids.to(activity.device)
        num_clusters = int(cluster_ids.max().item()) + 1
        cluster_sizes: List[int] = []
        mean_degree, mean_activity, var_activity, mean_memory = [], [], [], []
        degrees = adjacency.sum(dim=1).float().to(activity.device)
        for cid in range(num_clusters):
            mask = cluster_ids == cid
            if mask.sum() == 0:
                cluster_sizes.append(0)
                mean_degree.append(torch.tensor(0.0, device=activity.device))
                mean_activity.append(torch.tensor(0.0, device=activity.device))
                var_activity.append(torch.tensor(0.0, device=activity.device))
                mean_memory.append(torch.tensor(0.0, device=activity.device))
                continue
            cluster_sizes.append(int(mask.sum().item()))
            deg = degrees[mask]
            act = activity[mask]
            mem = memory[mask]
            mean_degree.append(deg.mean())
            mean_activity.append(act.mean())
            var_activity.append(act.var(unbiased=False) if act.numel() > 1 else torch.tensor(0.0, device=activity.device))
            mean_memory.append(mem.abs().mean())
        modularity = 0.0
        if calculate_modularity:
            n = adjacency.shape[0]
            same = cluster_ids.unsqueeze(0) == cluster_ids.unsqueeze(1)
            sym_w = (weights.abs() + weights.abs().T) / 2.0
            internal = sym_w[same].mean().item() if same.any() else 0.0
            external = sym_w[~same].mean().item() if (~same).any() else 0.0
            modularity = float(max(0.0, internal - external))
        return ClusterStats(
            cluster_ids=cluster_ids,
            cluster_sizes=cluster_sizes,
            mean_degree=torch.stack(mean_degree),
            mean_activity=torch.stack(mean_activity),
            var_activity=torch.stack(var_activity),
            mean_memory=torch.stack(mean_memory),
            modularity=modularity,
        )
