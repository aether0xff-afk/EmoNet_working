from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import nn

from .config import AppConfig
from .traits import MemoryState


@dataclass
class DynamicsState:
    potential: torch.Tensor
    spikes: torch.Tensor
    reaction: torch.Tensor
    activity: torch.Tensor
    theta_base: torch.Tensor
    theta: torch.Tensor
    gain_base: torch.Tensor
    gain: torch.Tensor
    memory: MemoryState


class EmotionalDynamicsNet(nn.Module):
    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        n = config.dynamics.num_neurons
        self.n = n
        self.control_proj = nn.Parameter(torch.randn(n, config.control_dim) * 0.15)
        self.trait_proj = nn.Parameter(torch.randn(n, config.trait_dim) * 0.12)
        self.w_hr = nn.Parameter(torch.randn(n, config.control_dim) * 0.12)
        self.w_rr = nn.Parameter(torch.randn(n, n) * 0.03)
        self.w_er = nn.Parameter(torch.randn(n, n) * 0.03)
        self.w_pr = nn.Parameter(torch.randn(n, config.trait_dim) * 0.08)
        self.b_r = nn.Parameter(torch.zeros(n))
        self.w_he = nn.Parameter(torch.randn(n, config.control_dim) * 0.12)
        self.w_re = nn.Parameter(torch.randn(n, n) * 0.03)
        self.w_ee = nn.Parameter(torch.randn(n, n) * 0.03)
        self.w_pe = nn.Parameter(torch.randn(n, config.trait_dim) * 0.08)
        self.b_e = nn.Parameter(torch.zeros(n))
        self.w_hp = nn.Parameter(torch.randn(n, config.control_dim) * 0.05)
        self.w_rp = nn.Parameter(torch.randn(n, n) * 0.01)
        self.w_pp = nn.Parameter(torch.randn(n, config.trait_dim) * 0.05)
        self.b_p = nn.Parameter(torch.zeros(n))
        self.alpha = nn.Parameter(torch.full((n,), config.dynamics.potential_decay))
        self.theta_base = nn.Parameter(torch.ones(n) * 0.4)
        self.gain_base = nn.Parameter(torch.ones(n))
        self.mod_theta = nn.Parameter(torch.randn(n) * 0.05)
        self.mod_gain = nn.Parameter(torch.randn(n) * 0.05)

        num_exc = int(round(n * config.dynamics.excitatory_ratio))
        num_inh = int(round(n * config.dynamics.inhibitory_ratio))
        neuron_types = torch.zeros(n, dtype=torch.long)
        neuron_types[num_exc : num_exc + num_inh] = 1
        neuron_types[num_exc + num_inh :] = 2
        self.register_buffer("neuron_types", neuron_types)

        adjacency = (torch.rand(n, n) < config.dynamics.initial_connect_prob).float()
        adjacency.fill_diagonal_(0.0)
        self.register_buffer("adjacency", adjacency)
        self.weight = nn.Parameter(torch.randn(n, n) * config.dynamics.weight_scale)
        self.state: Optional[DynamicsState] = None
        self.reset_episode_state()

    def effective_weight_matrix(self) -> torch.Tensor:
        src_types = self.neuron_types.unsqueeze(1)
        w = self.weight
        eff = torch.where(src_types == 0, w.abs(), w)
        eff = torch.where(src_types == 1, -w.abs(), eff)
        eff = torch.where(src_types == 2, 0.3 * torch.tanh(w), eff)
        return self.adjacency * eff

    def reset_episode_state(self) -> None:
        device = self.theta_base.device
        if self.state is None:
            memory = MemoryState.zeros(self.n, device=device)
        else:
            memory = self.state.memory
            memory.reset_episode()
        z = torch.zeros(self.n, device=device)
        self.state = DynamicsState(
            potential=z.clone(),
            spikes=z.clone(),
            reaction=z.clone(),
            activity=z.clone(),
            theta_base=self.theta_base.detach().clone(),
            theta=self.theta_base.detach().clone(),
            gain_base=self.gain_base.detach().clone(),
            gain=self.gain_base.detach().clone(),
            memory=memory,
        )

    def _modulatory_summary(self, spikes: torch.Tensor) -> torch.Tensor:
        mask = self.neuron_types == 2
        if mask.sum() == 0:
            return torch.tensor(0.0, device=spikes.device)
        return spikes[mask].mean()

    def step(self, h_t: torch.Tensor, p: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.state is None:
            self.reset_episode_state()
        assert self.state is not None
        h_t = h_t.flatten().to(self.theta_base.device)
        p = p.flatten().to(self.theta_base.device)
        w_eff = self.effective_weight_matrix()
        reaction_delta = (
            self.w_hr @ h_t
            + self.w_rr @ self.state.reaction
            + self.w_er @ self.state.memory.episode
            + self.w_pr @ p
            + self.b_r
        )
        reaction = torch.tanh(reaction_delta)
        episode_delta = (
            self.w_he @ h_t
            + self.w_re @ reaction
            + self.w_ee @ self.state.memory.episode
            + self.w_pe @ p
            + self.b_e
        )
        episode_memory = self.state.memory.update_episode(episode_delta)
        persistent_delta = self.w_hp @ h_t + self.w_rp @ reaction + self.w_pp @ p + self.b_p
        persistent_memory = self.state.memory.update_persistent(
            persistent_delta,
            rate=self.config.dynamics.persistent_update_rate,
            decay=1.0 - self.config.dynamics.persistent_update_rate * 0.5,
        )
        drive = (
            w_eff.T @ self.state.spikes
            + self.control_proj @ h_t
            + self.trait_proj @ p
            + self.config.dynamics.reaction_scale * reaction
            + self.config.dynamics.episode_memory_scale * episode_memory
            + self.config.dynamics.persistent_memory_scale * persistent_memory
        )
        mod_summary = self._modulatory_summary(self.state.spikes)
        theta = self.theta_base + self.mod_theta * mod_summary
        gain = torch.relu(self.gain_base + self.mod_gain * mod_summary) + 0.1
        potential = self.alpha.clamp(0.1, 0.99) * self.state.potential + gain * drive - theta
        spikes = (potential > 0).float()
        activity = self.config.dynamics.activity_decay * self.state.activity + (1.0 - self.config.dynamics.activity_decay) * spikes
        self.state = DynamicsState(
            potential=potential,
            spikes=spikes,
            reaction=reaction,
            activity=activity,
            theta_base=self.state.theta_base,
            theta=theta,
            gain_base=self.state.gain_base,
            gain=gain,
            memory=self.state.memory,
        )
        return {
            "potential": potential,
            "spikes": spikes,
            "reaction": reaction,
            "episode_memory": episode_memory,
            "persistent_memory": persistent_memory,
            "memory_total": self.state.memory.total,
            "drive": drive,
            "theta": theta,
            "gain": gain,
            "weights": w_eff,
            "adjacency": self.adjacency,
            "activity": activity,
        }
