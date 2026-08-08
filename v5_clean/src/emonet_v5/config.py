from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DynamicsConfig:
    """Configuration for the fixed recurrent baseline substrate.

    v5-clean intentionally starts without rewiring, affect axes, or learned
    emotion supervision. The only persistent quantity across events is the
    recurrent neural state itself.
    """

    num_neurons: int = 128
    recurrent_density: float = 0.08
    spectral_radius: float = 0.92
    update_rate: float = 0.35
    input_scale: float = 0.80
    event_ticks: int = 16
    stimulation_ticks: int = 6
    seed: int = 42

    def validate(self) -> None:
        if self.num_neurons <= 0:
            raise ValueError("num_neurons must be positive")
        if not 0.0 < self.recurrent_density <= 1.0:
            raise ValueError("recurrent_density must be in (0, 1]")
        if not 0.0 < self.spectral_radius < 1.5:
            raise ValueError("spectral_radius must be in (0, 1.5)")
        if not 0.0 < self.update_rate <= 1.0:
            raise ValueError("update_rate must be in (0, 1]")
        if self.input_scale <= 0.0:
            raise ValueError("input_scale must be positive")
        if self.event_ticks <= 0:
            raise ValueError("event_ticks must be positive")
        if not 0 < self.stimulation_ticks <= self.event_ticks:
            raise ValueError("stimulation_ticks must be in [1, event_ticks]")


@dataclass(frozen=True)
class ExperimentConfig:
    """Protocol-level settings kept separate from dynamics settings."""

    control_seed: int = 2026
    distance_epsilon: float = 1e-8
