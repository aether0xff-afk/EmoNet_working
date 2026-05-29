from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn

from .branching import BranchPath, BranchTracker
from .clustering import StructuralClusterManager
from .config import AppConfig, default_config
from .dynamics import EmotionalDynamicsNet
from .encoders import ControlEncoder
from .history import HistoryBuffer
from .history_encoder import HistoryEncoder
from .prompt_generator import PromptGenerator
from .rewiring import ClusterRewirer
from .tone_regressor import ToneRegressor
from .traits import TraitState


@dataclass
class InferenceOutput:
    h_t: torch.Tensor
    z: torch.Tensor
    s: torch.Tensor
    prompt: dict
    dominant_branch: Optional[BranchPath]
    history: torch.Tensor
    cluster_ids: torch.Tensor


class EmotionArchitecture(nn.Module):
    def __init__(self, config: AppConfig | None = None) -> None:
        super().__init__()
        self.config = config or default_config()
        self.control_encoder = ControlEncoder(self.config)
        self.dynamics = EmotionalDynamicsNet(self.config)
        self.cluster_manager = StructuralClusterManager(self.config)
        self.rewirer = ClusterRewirer(self.config)
        self.branch_tracker = BranchTracker(self.config)
        self.history_buffer = HistoryBuffer(self.config.latent.history_dim)
        self.history_encoder = HistoryEncoder(self.config, latent_dim=self.config.latent.default_latent_dim)
        self.tone_regressor = ToneRegressor(self.config, latent_dim=self.config.latent.default_latent_dim)
        self.prompt_generator = PromptGenerator(self.config)
        self.trait_state = TraitState.zeros(self.config.trait_dim, device=self.dynamics.theta_base.device)
        self.edge_change_accumulator = 0.0
        self.cluster_manager.initialize_from_graph(self.dynamics.adjacency.detach().cpu(), self.dynamics.effective_weight_matrix().detach().cpu())

    def _build_history_vector(
        self,
        cluster_stats,
        branch_stats: torch.Tensor,
        rewired_clusters: list[int],
        dynamics_out: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        spikes = dynamics_out["spikes"]
        activity = dynamics_out["activity"]
        reaction = dynamics_out["reaction"]
        mem_total = dynamics_out["memory_total"]
        e_mask = self.dynamics.neuron_types == 0
        i_mask = self.dynamics.neuron_types == 1
        m_mask = self.dynamics.neuron_types == 2
        def mean_mask(mask: torch.Tensor) -> float:
            return float(spikes[mask].mean().item()) if mask.any() else 0.0
        cluster_feats = cluster_stats.mean_activity.float()[:20]
        cluster_feats = torch.nn.functional.pad(cluster_feats, (0, max(0, 20 - cluster_feats.numel())))[:20]
        stress = self.rewirer.compute_homeostasis(cluster_stats).float()[:8]
        stress = torch.nn.functional.pad(stress, (0, max(0, 8 - stress.numel())))[:8]
        vec = torch.tensor(
            [
                float(spikes.mean().item()),
                mean_mask(e_mask),
                mean_mask(i_mask),
                mean_mask(m_mask),
                float(activity.mean().item()),
                float(reaction.mean().item()),
                float(mem_total.mean().item()),
                float(mem_total.abs().mean().item()),
                float(len(rewired_clusters) > 0),
                float(len(rewired_clusters)),
            ],
            dtype=torch.float32,
            device=spikes.device,
        )
        combined = torch.cat([vec, cluster_feats.to(spikes.device), stress.to(spikes.device), branch_stats.to(spikes.device)], dim=0)
        combined = torch.nn.functional.pad(combined, (0, max(0, self.config.latent.history_dim - combined.numel())))
        return combined[: self.config.latent.history_dim]

    def _run_episode(self, text: str, latent_dim: int | None = None) -> InferenceOutput:
        if latent_dim is not None and latent_dim != self.history_encoder.latent_dim:
            self.history_encoder = HistoryEncoder(self.config, latent_dim=latent_dim).to(self.dynamics.theta_base.device)
            self.tone_regressor = ToneRegressor(self.config, latent_dim=latent_dim).to(self.dynamics.theta_base.device)
        p = self.trait_state.p.to(self.dynamics.theta_base.device)
        h_t = self.control_encoder([text], p).squeeze(0)
        self.dynamics.reset_episode_state()
        self.branch_tracker.reset()
        self.history_buffer.reset()

        total_edge_updates = 0
        for t in range(self.config.dynamics.t_max):
            out = self.dynamics.step(h_t, p)
            stats = self.cluster_manager.compute_cluster_stats(
                self.dynamics.adjacency.detach().cpu(),
                self.dynamics.effective_weight_matrix().detach().cpu(),
                out["activity"].detach().cpu(),
                out["memory_total"].detach().cpu(),
                calculate_modularity=False,
            )
            self.branch_tracker.spawn_roots(out["spikes"], out["potential"], stats.cluster_ids.to(out["spikes"].device), t)
            self.branch_tracker.expand_paths(out["adjacency"], out["weights"], out["drive"], stats.cluster_ids.to(out["spikes"].device), t)
            self.branch_tracker.score_paths(out["activity"], out["memory_total"])
            self.branch_tracker.prune_paths()
            self.branch_tracker.merge_paths()
            adjacency_new, weight_new, rewired, edge_updates = self.rewirer.maybe_rewire(
                self.dynamics.adjacency,
                self.dynamics.weight.data,
                out["activity"],
                stats,
            )
            self.dynamics.adjacency.copy_(adjacency_new)
            self.dynamics.weight.data.copy_(weight_new)
            total_edge_updates += edge_updates
            history_vec = self._build_history_vector(stats, self.branch_tracker.stats_vector(), rewired, out)
            self.history_buffer.add_timestep(history_vec.detach().cpu())

        final_stats = self.cluster_manager.compute_cluster_stats(
            self.dynamics.adjacency.detach().cpu(),
            self.dynamics.effective_weight_matrix().detach().cpu(),
            out["activity"].detach().cpu(),
            out["memory_total"].detach().cpu(),
            calculate_modularity=True,
        )
        edge_ratio = total_edge_updates / max(1, int(self.dynamics.adjacency.sum().item()))
        self.cluster_manager.maybe_recluster(
            self.dynamics.adjacency.detach().cpu(),
            self.dynamics.effective_weight_matrix().detach().cpu(),
            edge_ratio,
            final_stats.modularity,
        )
        dominant = self.branch_tracker.finalize_dominant(out["activity"])
        history = self.history_buffer.stacked(device=self.dynamics.theta_base.device)
        z = self.history_encoder(history, dominant).squeeze(0)
        s = self.tone_regressor(z, p).squeeze(0)
        prompt = self.prompt_generator.generate_constraints(s)
        return InferenceOutput(
            h_t=h_t,
            z=z,
            s=s,
            prompt=prompt,
            dominant_branch=dominant,
            history=history,
            cluster_ids=final_stats.cluster_ids,
        )

    @torch.no_grad()
    def infer(self, text: str, latent_dim: int | None = None) -> InferenceOutput:
        out = self._run_episode(text, latent_dim=latent_dim)
        return InferenceOutput(
            h_t=out.h_t.detach().cpu(),
            z=out.z.detach().cpu(),
            s=out.s.detach().cpu(),
            prompt=out.prompt,
            dominant_branch=out.dominant_branch,
            history=out.history.detach().cpu(),
            cluster_ids=out.cluster_ids.detach().cpu(),
        )

    def forward(self, text: str) -> Dict[str, torch.Tensor | dict | BranchPath | None]:
        out = self._run_episode(text)
        return {
            "h_t": out.h_t,
            "z": out.z,
            "s": out.s,
            "prompt": out.prompt,
            "dominant_branch": out.dominant_branch,
            "history": out.history,
            "cluster_ids": out.cluster_ids,
        }
