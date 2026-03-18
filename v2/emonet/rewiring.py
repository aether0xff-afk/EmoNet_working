from __future__ import annotations

from typing import Dict, List, Tuple

import torch

from .clustering import ClusterStats
from .config import AppConfig


class ClusterRewirer:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.cooldowns: Dict[int, int] = {}

    def step_cooldowns(self) -> None:
        for cid in list(self.cooldowns):
            self.cooldowns[cid] -= 1
            if self.cooldowns[cid] <= 0:
                del self.cooldowns[cid]

    def compute_homeostasis(self, stats: ClusterStats) -> torch.Tensor:
        cfg = self.config.cluster
        hs = (
            cfg.alpha * (stats.mean_degree - cfg.target_degree).abs() / (cfg.target_degree + 1e-6)
            + cfg.beta * (stats.mean_activity - cfg.target_activity).abs()
            + cfg.gamma * stats.var_activity
            + cfg.delta * stats.mean_memory
        )
        return hs

    def maybe_rewire(
        self,
        adjacency: torch.Tensor,
        weights: torch.Tensor,
        activity: torch.Tensor,
        stats: ClusterStats,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[int], int]:
        cfg = self.config.cluster
        cluster_ids = stats.cluster_ids.to(adjacency.device)
        hs = self.compute_homeostasis(stats)
        triggered: List[int] = []
        edge_updates = 0
        self.step_cooldowns()
        for cid, score in enumerate(hs.tolist()):
            if cid in self.cooldowns and score < cfg.emergency_threshold:
                continue
            if score <= cfg.tau_rewire:
                continue
            triggered.append(cid)
            mask = (cluster_ids == cid)
            nodes = mask.nonzero(as_tuple=False).flatten()
            if nodes.numel() < 2:
                continue
            sub_adj = adjacency[nodes][:, nodes]
            sub_w = weights[nodes][:, nodes]
            edges_count = int(sub_adj.sum().item())
            budget = min(max(1, int(cfg.edge_budget_ratio * max(edges_count, 1))), cfg.max_edge_updates)
            local_updates = 0

            # 1) prune weakest intra-cluster edges
            weak_mask = (sub_adj > 0) & (sub_w.abs() < cfg.tau_prune)
            weak_idx = weak_mask.nonzero(as_tuple=False)
            prune_n = min(len(weak_idx), budget // 2)
            if prune_n > 0:
                selected = weak_idx[:prune_n]
                for i_local, j_local in selected.tolist():
                    i, j = int(nodes[i_local]), int(nodes[j_local])
                    adjacency[i, j] = 0
                    weights[i, j] = 0.0
                    edge_updates += 1
                    local_updates += 1

            # 2) add coherent intra-cluster edges by activity proximity
            if local_updates < budget:
                local_activity = activity[nodes]
                sim = 1.0 - (local_activity.unsqueeze(0) - local_activity.unsqueeze(1)).abs()
                add_candidates = ((sub_adj == 0) & (sim > cfg.tau_corr)).nonzero(as_tuple=False)
                add_n = min(len(add_candidates), max(0, budget - local_updates))
                for i_local, j_local in add_candidates[:add_n].tolist():
                    if i_local == j_local:
                        continue
                    i, j = int(nodes[i_local]), int(nodes[j_local])
                    adjacency[i, j] = 1
                    sign = 1.0 if weights[i].mean().item() >= 0 else -1.0
                    weights[i, j] = sign * float(sim[i_local, j_local].item()) * 0.1
                    edge_updates += 1
                    local_updates += 1

            # 3) weaken excessive inter-cluster edges
            if local_updates < budget:
                outside = (cluster_ids != cid).nonzero(as_tuple=False).flatten()
                for i in nodes.tolist():
                    inter_vals = weights[i, outside].abs()
                    if inter_vals.numel() == 0:
                        continue
                    max_val, arg = inter_vals.max(dim=0)
                    if float(max_val.item()) > 0.25 and local_updates < budget:
                        j = int(outside[int(arg.item())])
                        weights[i, j] *= 0.5
                        edge_updates += 1
                        local_updates += 1

            self.cooldowns[cid] = cfg.cooldown
        return adjacency, weights, triggered, edge_updates
