from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LearnedCoreConfig:
    hidden_dim: int = 128
    event_ticks: int = 16
    stimulation_ticks: int = 6
    update_rate: float = 0.35
    input_scale: float = 0.80
    recurrent_scale: float = 0.92
    max_lag: int = 3

    def validate(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.event_ticks <= 0:
            raise ValueError("event_ticks must be positive")
        if not 0 < self.stimulation_ticks <= self.event_ticks:
            raise ValueError("stimulation_ticks must be in [1, event_ticks]")
        if not 0.0 < self.update_rate <= 1.0:
            raise ValueError("update_rate must be in (0, 1]")
        if self.max_lag <= 0:
            raise ValueError("max_lag must be positive")


class LearnedLeakyRecurrentCore(nn.Module):
    """Trainable v5.0-like recurrent substrate.

    The core has no task or emotion head. `memory_heads` exist only for the
    self-supervised delayed-event objective and may be ignored after training.
    """

    def __init__(self, input_dim: int, config: LearnedCoreConfig, seed: int) -> None:
        super().__init__()
        config.validate()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        self.input_dim = int(input_dim)
        self.config = config
        self.seed = int(seed)

        # Keep initialization deterministic without changing caller RNG state.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed)

        self.input_weight = nn.Parameter(
            torch.randn(
                config.hidden_dim,
                self.input_dim,
                generator=generator,
                dtype=torch.float32,
            )
            * (config.input_scale / np.sqrt(max(1, self.input_dim)))
        )

        recurrent = torch.randn(
            config.hidden_dim,
            config.hidden_dim,
            generator=generator,
            dtype=torch.float32,
        )
        # Orthogonalize to provide a stable full-rank starting substrate.
        q, _ = torch.linalg.qr(recurrent)
        self.recurrent_weight = nn.Parameter(q * config.recurrent_scale)
        self.bias = nn.Parameter(torch.zeros(config.hidden_dim, dtype=torch.float32))

        self.memory_heads = nn.ModuleList(
            [nn.Linear(config.hidden_dim, self.input_dim) for _ in range(config.max_lag)]
        )
        for index, head in enumerate(self.memory_heads):
            head_generator = torch.Generator(device="cpu")
            head_generator.manual_seed(self.seed + 1000 + index)
            with torch.no_grad():
                head.weight.copy_(
                    torch.randn(
                        head.weight.shape,
                        generator=head_generator,
                        dtype=head.weight.dtype,
                    )
                    * (0.02 / np.sqrt(max(1, config.hidden_dim)))
                )
                head.bias.zero_()

    def zero_state(self, batch_size: int, device: torch.device | None = None) -> torch.Tensor:
        target_device = device or self.input_weight.device
        return torch.zeros(batch_size, self.config.hidden_dim, device=target_device)

    def step(self, state: torch.Tensor, event_drive: torch.Tensor | None) -> torch.Tensor:
        if event_drive is None:
            drive = torch.zeros(
                state.shape[0],
                self.config.hidden_dim,
                dtype=state.dtype,
                device=state.device,
            )
        else:
            drive = event_drive @ self.input_weight.T
        candidate = torch.tanh(state @ self.recurrent_weight.T + drive + self.bias)
        rate = self.config.update_rate
        return (1.0 - rate) * state + rate * candidate

    def run_event(
        self,
        state: torch.Tensor,
        embedding: torch.Tensor,
        *,
        return_trace: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if embedding.ndim != 2 or embedding.shape[1] != self.input_dim:
            raise ValueError("embedding must have shape [batch, input_dim]")
        trace: list[torch.Tensor] = []
        for tick in range(self.config.event_ticks):
            drive = embedding if tick < self.config.stimulation_ticks else None
            state = self.step(state, drive)
            if return_trace:
                trace.append(state)
        stacked = torch.stack(trace, dim=1) if return_trace else None
        return state, stacked

    def run_sequence(
        self,
        embeddings: torch.Tensor,
        *,
        return_event_traces: bool = False,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor] | None]:
        """Run [batch, events, input_dim] through the recurrent core."""

        if embeddings.ndim != 3 or embeddings.shape[2] != self.input_dim:
            raise ValueError("embeddings must have shape [batch, events, input_dim]")
        state = self.zero_state(embeddings.shape[0], embeddings.device)
        states: list[torch.Tensor] = []
        traces: list[torch.Tensor] = []
        for event_index in range(embeddings.shape[1]):
            state, trace = self.run_event(
                state,
                embeddings[:, event_index],
                return_trace=return_event_traces,
            )
            states.append(state)
            if return_event_traces:
                assert trace is not None
                traces.append(trace)
        return states, traces if return_event_traces else None

    def delayed_memory_loss(self, embeddings: torch.Tensor) -> tuple[torch.Tensor, dict[int, float]]:
        states, _ = self.run_sequence(embeddings, return_event_traces=False)
        losses: list[torch.Tensor] = []
        lag_losses: dict[int, list[torch.Tensor]] = {
            lag: [] for lag in range(1, self.config.max_lag + 1)
        }

        for event_index, state in enumerate(states):
            for lag in range(1, self.config.max_lag + 1):
                target_index = event_index - lag
                if target_index < 0:
                    continue
                prediction = F.normalize(self.memory_heads[lag - 1](state), dim=-1)
                target = F.normalize(embeddings[:, target_index].detach(), dim=-1)
                cosine_loss = 1.0 - (prediction * target).sum(dim=-1)
                term = cosine_loss.mean()
                losses.append(term)
                lag_losses[lag].append(term.detach())

        if not losses:
            raise RuntimeError("sequence is too short for delayed-memory training")
        total = torch.stack(losses).mean()
        diagnostics = {
            lag: float(torch.stack(values).mean().cpu()) if values else float("nan")
            for lag, values in lag_losses.items()
        }
        return total, diagnostics

    @torch.no_grad()
    def lag_cosine_at_final(self, embeddings: torch.Tensor, lag: int) -> float:
        if not 1 <= lag <= self.config.max_lag:
            raise ValueError("lag outside configured range")
        states, _ = self.run_sequence(embeddings, return_event_traces=False)
        target_index = embeddings.shape[1] - 1 - lag
        if target_index < 0:
            raise ValueError("sequence too short for requested lag")
        prediction = F.normalize(self.memory_heads[lag - 1](states[-1]), dim=-1)
        target = F.normalize(embeddings[:, target_index], dim=-1)
        return float((prediction * target).sum(dim=-1).mean().cpu())

    @torch.no_grad()
    def final_event_trace(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Return the final event trace [batch, ticks, hidden]."""

        if embeddings.ndim != 3:
            raise ValueError("embeddings must be [batch, events, input_dim]")
        state = self.zero_state(embeddings.shape[0], embeddings.device)
        for event_index in range(embeddings.shape[1] - 1):
            state, _ = self.run_event(
                state,
                embeddings[:, event_index],
                return_trace=False,
            )
        _, trace = self.run_event(
            state,
            embeddings[:, -1],
            return_trace=True,
        )
        assert trace is not None
        return trace

    @torch.no_grad()
    def reset_final_event_trace(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Erase history immediately before the final event."""

        state = self.zero_state(embeddings.shape[0], embeddings.device)
        _, trace = self.run_event(
            state,
            embeddings[:, -1],
            return_trace=True,
        )
        assert trace is not None
        return trace

    @torch.no_grad()
    def stabilize_recurrent(self, max_spectral_norm: float = 0.98) -> None:
        """Keep training from turning the recurrent substrate into an unstable amplifier."""

        norm = torch.linalg.matrix_norm(self.recurrent_weight, ord=2)
        if float(norm) > max_spectral_norm:
            self.recurrent_weight.mul_(max_spectral_norm / norm)
