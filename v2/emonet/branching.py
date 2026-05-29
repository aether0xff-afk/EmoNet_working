from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch

from .config import AppConfig
from .utils import cosine_similarity_list, longest_common_prefix, safe_mean


@dataclass
class BranchPath:
    neuron_path: List[int]
    cluster_path: List[int]
    edge_weights: List[float]
    step_ids: List[int]
    flow_strength: float = 0.0
    activation_support: float = 0.0
    memory_contrib: float = 0.0
    cluster_coherence: float = 0.0
    transition_richness: float = 0.0
    score: float = 0.0
    persistence: int = 1
    alive: bool = True

    @property
    def root(self) -> int:
        return self.neuron_path[0]

    @property
    def terminal(self) -> int:
        return self.neuron_path[-1]


class BranchTracker:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.paths: List[BranchPath] = []
        self.total_spawned = 0
        self.pruned_last_step = 0
        self.merged_last_step = 0

    def reset(self) -> None:
        self.paths = []
        self.total_spawned = 0
        self.pruned_last_step = 0
        self.merged_last_step = 0

    def spawn_roots(self, spikes: torch.Tensor, potentials: torch.Tensor, cluster_ids: torch.Tensor, t: int) -> None:
        mask = (spikes > 0) & (potentials > self.config.branch.tau_root)
        roots = mask.nonzero(as_tuple=False).flatten().tolist()
        existing_roots = {p.root for p in self.paths if len(p.neuron_path) == 1 and p.step_ids[-1] == t}
        for root in roots:
            if root in existing_roots:
                continue
            self.paths.append(
                BranchPath(
                    neuron_path=[root],
                    cluster_path=[int(cluster_ids[root].item())],
                    edge_weights=[],
                    step_ids=[t],
                )
            )
            self.total_spawned += 1

    def expand_paths(self, adjacency: torch.Tensor, weights: torch.Tensor, drives: torch.Tensor, cluster_ids: torch.Tensor, t: int) -> None:
        new_paths: List[BranchPath] = []
        for path in self.paths:
            if not path.alive or len(path.neuron_path) >= self.config.branch.l_max:
                new_paths.append(path)
                continue
            last = path.terminal
            outgoing = (adjacency[last] > 0).nonzero(as_tuple=False).flatten().tolist()
            candidates = []
            for j in outgoing:
                if j in path.neuron_path:
                    continue
                if abs(float(weights[last, j].item())) <= self.config.branch.tau_edge:
                    continue
                if float(drives[j].item()) <= self.config.branch.tau_flow:
                    continue
                candidates.append(j)
            if not candidates:
                path.persistence += 1
                new_paths.append(path)
                continue
            for j in candidates:
                new_path = BranchPath(
                    neuron_path=path.neuron_path + [j],
                    cluster_path=path.cluster_path + [int(cluster_ids[j].item())],
                    edge_weights=path.edge_weights + [float(weights[last, j].item())],
                    step_ids=path.step_ids + [t],
                    persistence=path.persistence + 1,
                )
                new_paths.append(new_path)
                self.total_spawned += 1
        self.paths = new_paths

    def _transition_richness(self, cluster_path: Sequence[int]) -> float:
        if len(cluster_path) <= 1:
            return 0.0
        transitions = {(cluster_path[i], cluster_path[i + 1]) for i in range(len(cluster_path) - 1)}
        return len(transitions) / max(1, len(cluster_path) - 1)

    def score_paths(self, activity: torch.Tensor, memory: torch.Tensor) -> None:
        cfg = self.config.branch
        for path in self.paths:
            if len(path.neuron_path) <= 1:
                flow = 0.0
            else:
                flow = safe_mean(abs(w) for w in path.edge_weights)
            nodes = torch.tensor(path.neuron_path, dtype=torch.long, device=activity.device)
            act = float(activity[nodes].mean().item()) if nodes.numel() else 0.0
            mem = float(memory[nodes].abs().mean().item()) if nodes.numel() else 0.0
            coherence = 0.0
            if len(path.cluster_path) > 1:
                same = sum(1 for i in range(len(path.cluster_path) - 1) if path.cluster_path[i] == path.cluster_path[i + 1])
                coherence = same / max(1, len(path.cluster_path) - 1)
            trans = self._transition_richness(path.cluster_path)
            length_pen = max(0, len(path.neuron_path) - 1)
            score = (
                cfg.lambda_f * flow
                + cfg.lambda_a * act
                + cfg.lambda_m * mem
                + cfg.lambda_c * coherence
                + cfg.lambda_t * trans
                - cfg.lambda_l * length_pen
            )
            path.flow_strength = flow
            path.activation_support = act
            path.memory_contrib = mem
            path.cluster_coherence = coherence
            path.transition_richness = trans
            path.score = float(score)

    def prune_paths(self) -> None:
        cfg = self.config.branch
        self.pruned_last_step = 0
        kept: List[BranchPath] = []
        by_root: Dict[int, List[BranchPath]] = {}
        for path in self.paths:
            if path.score < cfg.tau_branch_min:
                self.pruned_last_step += 1
                continue
            if len(path.neuron_path) > 1 and all(abs(w) < cfg.tau_edge for w in path.edge_weights[-2:]):
                self.pruned_last_step += 1
                continue
            by_root.setdefault(path.root, []).append(path)
        for root, group in by_root.items():
            group = sorted(group, key=lambda p: p.score, reverse=True)[: cfg.per_root_topk]
            kept.extend(group)
        kept = sorted(kept, key=lambda p: p.score, reverse=True)[: cfg.global_topk]
        self.pruned_last_step += max(0, len(self.paths) - len(kept) - self.pruned_last_step)
        self.paths = kept

    def _merge_pair(self, a: BranchPath, b: BranchPath) -> BranchPath:
        prefix_len = longest_common_prefix(a.neuron_path, b.neuron_path)
        base = a if a.score >= b.score else b
        prefix_nodes = base.neuron_path[:prefix_len]
        prefix_clusters = base.cluster_path[:prefix_len]
        prefix_steps = base.step_ids[:prefix_len]
        prefix_weights = base.edge_weights[: max(0, prefix_len - 1)]
        suffix_nodes = base.neuron_path[prefix_len:]
        suffix_clusters = base.cluster_path[prefix_len:]
        suffix_steps = base.step_ids[prefix_len:]
        suffix_weights = base.edge_weights[prefix_len - 1 :] if prefix_len > 0 else base.edge_weights[:]
        merged = BranchPath(
            neuron_path=prefix_nodes + suffix_nodes,
            cluster_path=prefix_clusters + suffix_clusters,
            edge_weights=prefix_weights + suffix_weights,
            step_ids=prefix_steps + suffix_steps,
            persistence=max(a.persistence, b.persistence),
        )
        merged.score = max(a.score, b.score) + self.config.branch.merge_eta * min(a.score, b.score)
        merged.flow_strength = max(a.flow_strength, b.flow_strength)
        merged.activation_support = max(a.activation_support, b.activation_support)
        merged.memory_contrib = max(a.memory_contrib, b.memory_contrib)
        merged.cluster_coherence = max(a.cluster_coherence, b.cluster_coherence)
        merged.transition_richness = max(a.transition_richness, b.transition_richness)
        return merged

    def merge_paths(self) -> None:
        cfg = self.config.branch
        self.merged_last_step = 0
        merged: List[BranchPath] = []
        used = [False] * len(self.paths)
        for i, path_i in enumerate(self.paths):
            if used[i]:
                continue
            cur = path_i
            for j in range(i + 1, len(self.paths)):
                if used[j]:
                    continue
                path_j = self.paths[j]
                prefix = longest_common_prefix(cur.neuron_path, path_j.neuron_path)
                prefix_ratio = prefix / max(1, min(len(cur.neuron_path), len(path_j.neuron_path)))
                node_sim = cosine_similarity_list(cur.neuron_path, path_j.neuron_path)
                cluster_sim = cosine_similarity_list(cur.cluster_path, path_j.cluster_path)
                if prefix_ratio > cfg.tau_prefix or node_sim > cfg.tau_node or cluster_sim > cfg.tau_cluster:
                    cur = self._merge_pair(cur, path_j)
                    used[j] = True
                    self.merged_last_step += 1
            used[i] = True
            merged.append(cur)
        self.paths = merged

    def finalize_dominant(self, activity: torch.Tensor) -> Optional[BranchPath]:
        if not self.paths:
            return None
        cfg = self.config.branch
        best = None
        best_score = -1e9
        for path in self.paths:
            terminal_activation = float(activity[path.terminal].item()) if path.neuron_path else 0.0
            cluster_quality = 0.5 * path.cluster_coherence + 0.5 * path.transition_richness
            score = (
                cfg.mu1 * path.score
                + cfg.mu2 * path.memory_contrib
                + cfg.mu3 * float(path.persistence)
                + cfg.mu4 * terminal_activation
                + cfg.mu5 * cluster_quality
            )
            if score > best_score:
                best_score = score
                best = path
        return best

    def stats_vector(self) -> torch.Tensor:
        scores = sorted([p.score for p in self.paths], reverse=True)
        top1 = scores[0] if scores else 0.0
        top3 = safe_mean(scores[:3], 0.0)
        return torch.tensor(
            [
                float(len(self.paths)),
                float(self.pruned_last_step),
                float(self.merged_last_step),
                float(top1),
                float(top3),
            ],
            dtype=torch.float32,
        )
