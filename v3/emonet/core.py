from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional, Sequence
import math
import re

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None


TORCH_AVAILABLE = torch is not None

STIM_DIM = 4
BRANCH_FEATURE_DIM = 6
STYLE_AXES = (
    "verbosity",
    "sentence_length",
    "pace",
    "fragmentation",
    "repetition",
    "rhythmicity",
    "directness",
    "explicitness",
    "specificity",
    "abstraction",
    "certainty",
    "logicality",
    "warmth",
    "distance",
    "politeness",
    "formality",
    "cooperativeness",
    "dominance",
    "calmness",
    "tension",
    "positivity",
    "heaviness",
    "urgency",
    "emotional_openness",
    "softness",
    "sharpness",
    "playfulness",
    "seriousness",
    "metaphoricity",
    "plainness",
    "initiative",
    "reflectiveness",
)


def clamp_stim_vec(stim_vec: np.ndarray) -> np.ndarray:
    return np.clip(stim_vec.astype(np.float32, copy=False), 0.0, 1.0)


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right) + 1e-8)
    if denom <= 0.0:
        return 0.0
    return float(np.dot(left, right) / denom)


@dataclass(slots=True)
class MemoryItem:
    stim_vec: np.ndarray
    K_snapshot: float
    input_text: str
    created_tick: int
    strength: float


@dataclass(slots=True)
class NeuronState:
    neuron_id: int
    neuron_type: Literal["inhibitory", "excitatory", "modulatory"]
    K: float = 0.0
    stim_vec: np.ndarray = field(default_factory=lambda: np.zeros(STIM_DIM, dtype=np.float32))
    k_threshold: float = 1.0
    k_remem: float = 1.2
    refractory_left: int = 0
    dropped_out: bool = False
    out_neighbors: set[int] = field(default_factory=set)
    in_neighbors: set[int] = field(default_factory=set)
    memories: list[MemoryItem] = field(default_factory=list)


@dataclass(slots=True)
class NodeStepState:
    K: float
    stim_vec: np.ndarray


@dataclass(slots=True)
class TickRecord:
    tick: int
    active_nodes: list[int]
    node_states: dict[int, NodeStepState]
    edges_fired: list[tuple[int, int]]


@dataclass(slots=True)
class EmoNetState:
    neurons: list[NeuronState]
    tick: int
    prev_K: np.ndarray
    curr_K: np.ndarray
    global_threshold_shift: float
    global_remem_shift: float
    alive_mask: np.ndarray
    branch_log: list[TickRecord]


@dataclass(slots=True)
class BranchStep:
    tick: int
    node_id: int
    K: float
    stim_vec: np.ndarray


@dataclass(slots=True)
class BranchPath:
    score: float
    steps: list[BranchStep]


@dataclass(slots=True)
class DominantBranchStep:
    tick: int
    stim_vec: np.ndarray
    K: float


@dataclass(slots=True)
class EmoNetConfig:
    n_neurons: int = 256
    ratio_inhib: float = 0.45
    ratio_excit: float = 0.45
    ratio_mod: float = 0.10

    n_inhibitory: int = 115
    n_excitatory: int = 115
    n_modulatory: int = 26

    initial_out_degree: int = 5
    target_in_degree: int = 5
    max_ticks: int = 32
    delta_k_eps: float = 1e-3

    k_threshold_base: float = 1.0
    k_remem_base: float = 1.2
    k_decay: float = 0.95
    refractory_ticks: int = 3

    memory_decay: float = 0.97
    memory_delete_threshold: float = 0.05
    memory_sim_gain: float = 0.10
    memory_stim_mix: float = 0.20
    memory_k_mix: float = 0.10
    max_memory_per_neuron: int = 64

    max_out_degree: int = 12
    min_out_degree: int = 1
    dopa_rewire_gain: float = 0.30
    sero_prune_gain: float = 0.30

    mela_dropout_gain: float = 0.30
    ne_thresh_reduce_gain: float = 0.25
    ne_remem_reduce_gain: float = 0.25
    global_recovery_rate: float = 0.10

    z_dim: int = 64
    s_dim: int = 32
    topk_branches: int = 4

    n_layers: int = 2
    n_heads: int = 4
    d_model: int = 64
    ff_dim: int = 128
    dropout: float = 0.10

    seed: Optional[int] = None

    def __post_init__(self) -> None:
        expected_total = self.n_inhibitory + self.n_excitatory + self.n_modulatory
        if expected_total != self.n_neurons:
            raise ValueError(f"Neuron counts must sum to n_neurons, got {expected_total} != {self.n_neurons}")
        if self.s_dim != len(STYLE_AXES):
            raise ValueError(f"s_dim must match STYLE_AXES ({len(STYLE_AXES)})")


class StimEncoder:
    POSITIVE_WORDS = {
        "good",
        "great",
        "happy",
        "love",
        "joy",
        "excited",
        "win",
        "success",
        "fun",
        "energy",
    }
    CALM_WORDS = {
        "calm",
        "steady",
        "safe",
        "okay",
        "gentle",
        "peace",
        "settle",
        "stable",
        "restored",
        "comfortable",
    }
    ALERT_WORDS = {
        "urgent",
        "now",
        "must",
        "immediately",
        "warning",
        "risk",
        "issue",
        "critical",
        "asap",
        "alert",
    }
    REST_WORDS = {
        "sleep",
        "night",
        "dream",
        "rest",
        "tired",
        "quiet",
        "late",
        "dark",
        "slow",
        "bed",
    }

    WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)

    def encode(self, text: str | Sequence[float] | np.ndarray) -> np.ndarray:
        if isinstance(text, np.ndarray):
            return self._validate_stim_vec(text)
        if isinstance(text, (list, tuple)):
            return self._validate_stim_vec(np.asarray(text, dtype=np.float32))
        if not isinstance(text, str):
            raise TypeError("StimEncoder expects a string or a 4D stim vector")

        lowered = text.lower()
        tokens = self.WORD_RE.findall(lowered)
        token_count = len(tokens)
        letters = sum(ch.isalpha() for ch in text)
        uppercase = sum(ch.isupper() for ch in text)
        uppercase_ratio = uppercase / max(1, letters)

        exclamations = min(text.count("!") / 3.0, 1.0)
        questions = min(text.count("?") / 3.0, 1.0)
        positive_hits = sum(token in self.POSITIVE_WORDS for token in tokens)
        calm_hits = sum(token in self.CALM_WORDS for token in tokens)
        alert_hits = sum(token in self.ALERT_WORDS for token in tokens)
        rest_hits = sum(token in self.REST_WORDS for token in tokens)

        token_scale = min(token_count / 24.0, 1.0)
        positive_scale = min(positive_hits / 3.0, 1.0)
        calm_scale = min(calm_hits / 3.0, 1.0)
        alert_scale = min(alert_hits / 3.0, 1.0)
        rest_scale = min(rest_hits / 3.0, 1.0)

        dopamine = 0.20 + 0.35 * positive_scale + 0.20 * exclamations + 0.10 * token_scale
        serotonin = 0.20 + 0.45 * calm_scale + 0.05 * (1.0 - exclamations) + 0.05 * (1.0 - questions)
        norepinephrine = 0.15 + 0.45 * alert_scale + 0.15 * questions + 0.15 * uppercase_ratio + 0.10 * exclamations
        melatonin = 0.10 + 0.50 * rest_scale + 0.10 * max(0.0, 0.25 - token_scale)

        stim_vec = np.asarray([dopamine, serotonin, norepinephrine, melatonin], dtype=np.float32)
        return clamp_stim_vec(stim_vec)

    @staticmethod
    def _validate_stim_vec(stim_vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(stim_vec, dtype=np.float32).reshape(-1)
        if vec.shape != (STIM_DIM,):
            raise ValueError(f"stim_vec must have shape ({STIM_DIM},), got {vec.shape}")
        return clamp_stim_vec(vec.copy())


class EmoNetGraph:
    def __init__(self, config: EmoNetConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng

    def build_graph(self) -> list[NeuronState]:
        neurons = self._build_neurons()
        for src in range(self.config.n_neurons):
            targets = self._sample_targets(src, neurons, self.config.initial_out_degree)
            for dst in targets:
                self.add_edge(neurons, src, dst)
        self._rebalance_in_degree(neurons)
        return neurons

    def add_edge(self, neurons: list[NeuronState], src: int, dst: int) -> bool:
        if src == dst:
            return False
        source = neurons[src]
        target = neurons[dst]
        if dst in source.out_neighbors:
            return False
        source.out_neighbors.add(dst)
        target.in_neighbors.add(src)
        return True

    def remove_edge(self, neurons: list[NeuronState], src: int, dst: int) -> bool:
        source = neurons[src]
        target = neurons[dst]
        if dst not in source.out_neighbors:
            return False
        source.out_neighbors.remove(dst)
        target.in_neighbors.discard(src)
        return True

    def _build_neurons(self) -> list[NeuronState]:
        neuron_types = (
            ["inhibitory"] * self.config.n_inhibitory
            + ["excitatory"] * self.config.n_excitatory
            + ["modulatory"] * self.config.n_modulatory
        )
        self.rng.shuffle(neuron_types)

        neurons: list[NeuronState] = []
        for neuron_id, neuron_type in enumerate(neuron_types):
            neurons.append(
                NeuronState(
                    neuron_id=neuron_id,
                    neuron_type=neuron_type,
                    K=0.0,
                    stim_vec=np.zeros(STIM_DIM, dtype=np.float32),
                    k_threshold=self.config.k_threshold_base,
                    k_remem=self.config.k_remem_base,
                )
            )
        return neurons

    def _sample_targets(self, src: int, neurons: list[NeuronState], degree: int) -> list[int]:
        candidates = [node_id for node_id in range(len(neurons)) if node_id != src]
        if degree > len(candidates):
            degree = len(candidates)
        targets = self.rng.choice(candidates, size=degree, replace=False)
        return [int(target) for target in np.atleast_1d(targets)]

    def _rebalance_in_degree(self, neurons: list[NeuronState]) -> None:
        underfull = {idx for idx, neuron in enumerate(neurons) if len(neuron.in_neighbors) < self.config.target_in_degree}
        attempts = 0
        max_attempts = self.config.n_neurons * 20

        while underfull and attempts < max_attempts:
            dst = int(self.rng.choice(list(underfull)))
            donor_candidates = [
                idx
                for idx, neuron in enumerate(neurons)
                if dst not in neuron.out_neighbors
                and idx != dst
                and len(neuron.out_neighbors) >= self.config.initial_out_degree
            ]
            if not donor_candidates:
                underfull.discard(dst)
                attempts += 1
                continue

            src = int(self.rng.choice(donor_candidates))
            removable_targets = [
                target
                for target in neurons[src].out_neighbors
                if len(neurons[target].in_neighbors) > self.config.target_in_degree
            ]
            if not removable_targets:
                attempts += 1
                continue

            old_dst = int(self.rng.choice(removable_targets))
            self.remove_edge(neurons, src, old_dst)
            self.add_edge(neurons, src, dst)

            if len(neurons[dst].in_neighbors) >= self.config.target_in_degree:
                underfull.discard(dst)
            if len(neurons[old_dst].in_neighbors) < self.config.target_in_degree:
                underfull.add(old_dst)
            attempts += 1


class BranchExtractor:
    def prune_to_survivors(self, branch_log: list[TickRecord]) -> list[TickRecord]:
        if not branch_log:
            return []

        final_index = self._find_last_non_empty_record_index(branch_log)
        if final_index is None:
            return []

        pruned: list[Optional[TickRecord]] = [None] * (final_index + 1)
        survivor_nodes = set(branch_log[final_index].active_nodes)

        final_record = branch_log[final_index]
        pruned[final_index] = TickRecord(
            tick=final_record.tick,
            active_nodes=sorted(survivor_nodes),
            node_states={node_id: final_record.node_states[node_id] for node_id in sorted(survivor_nodes)},
            edges_fired=[],
        )

        for idx in range(final_index - 1, -1, -1):
            record = branch_log[idx]
            kept_edges = [(src, dst) for src, dst in record.edges_fired if dst in survivor_nodes]
            surviving_here = {src for src, _ in kept_edges}
            pruned[idx] = TickRecord(
                tick=record.tick,
                active_nodes=sorted(surviving_here),
                node_states={node_id: record.node_states[node_id] for node_id in sorted(surviving_here)},
                edges_fired=kept_edges,
            )
            survivor_nodes = surviving_here

        return [record for record in pruned if record is not None and record.active_nodes]

    def extract_topk_branches(self, pruned_branch_log: list[TickRecord], topk: int) -> list[BranchPath]:
        if not pruned_branch_log:
            return []

        topk_paths: dict[tuple[int, int], list[BranchPath]] = {}

        for index, record in enumerate(pruned_branch_log):
            prev_edges = pruned_branch_log[index - 1].edges_fired if index > 0 else []
            parents_by_node: dict[int, list[int]] = {}
            for src, dst in prev_edges:
                parents_by_node.setdefault(dst, []).append(src)

            for node_id in record.active_nodes:
                state = record.node_states[node_id]
                step = BranchStep(
                    tick=record.tick,
                    node_id=node_id,
                    K=state.K,
                    stim_vec=state.stim_vec.copy(),
                )
                candidates: list[BranchPath] = []
                parent_nodes = parents_by_node.get(node_id, [])
                if parent_nodes:
                    prev_tick = pruned_branch_log[index - 1].tick
                    for parent_id in parent_nodes:
                        for parent_path in topk_paths.get((prev_tick, parent_id), []):
                            candidates.append(
                                BranchPath(
                                    score=parent_path.score + step.K,
                                    steps=parent_path.steps + [step],
                                )
                            )
                else:
                    candidates.append(BranchPath(score=step.K, steps=[step]))

                candidates.sort(key=lambda path: path.score, reverse=True)
                topk_paths[(record.tick, node_id)] = candidates[:topk]

        final_record = pruned_branch_log[-1]
        completed_paths: list[BranchPath] = []
        for node_id in final_record.active_nodes:
            completed_paths.extend(topk_paths.get((final_record.tick, node_id), []))
        completed_paths.sort(key=lambda path: path.score, reverse=True)
        return completed_paths[:topk]

    def build_dominant_branch(
        self,
        topk_paths: list[BranchPath],
        fallback_stim_vec: np.ndarray,
        branch_log: list[TickRecord],
        topk: int,
    ) -> list[DominantBranchStep]:
        selected_paths = topk_paths[:topk]
        if not selected_paths:
            return self._fallback_branch(fallback_stim_vec, branch_log)

        total_score = sum(path.score for path in selected_paths) + 1e-8
        global_weights = [path.score / total_score for path in selected_paths]

        steps_by_tick: dict[int, list[tuple[float, BranchStep]]] = {}
        for weight, path in zip(global_weights, selected_paths, strict=False):
            for step in path.steps:
                steps_by_tick.setdefault(step.tick, []).append((weight, step))

        dominant_branch: list[DominantBranchStep] = []
        for tick in sorted(steps_by_tick):
            contributions = steps_by_tick[tick]
            present_weight_sum = sum(weight for weight, _ in contributions) + 1e-8
            stim_acc = np.zeros(STIM_DIM, dtype=np.float32)
            k_acc = 0.0
            for weight, step in contributions:
                normalized_weight = weight / present_weight_sum
                stim_acc += normalized_weight * step.stim_vec
                k_acc += normalized_weight * step.K
            dominant_branch.append(
                DominantBranchStep(
                    tick=tick,
                    stim_vec=clamp_stim_vec(stim_acc),
                    K=float(k_acc),
                )
            )
        return dominant_branch

    @staticmethod
    def _find_last_non_empty_record_index(branch_log: list[TickRecord]) -> Optional[int]:
        for index in range(len(branch_log) - 1, -1, -1):
            if branch_log[index].active_nodes:
                return index
        return None

    @staticmethod
    def _fallback_branch(fallback_stim_vec: np.ndarray, branch_log: list[TickRecord]) -> list[DominantBranchStep]:
        best_state: Optional[DominantBranchStep] = None
        for record in branch_log:
            for state in record.node_states.values():
                if best_state is None or state.K > best_state.K:
                    best_state = DominantBranchStep(
                        tick=record.tick,
                        stim_vec=state.stim_vec.copy(),
                        K=state.K,
                    )
        if best_state is not None:
            return [best_state]
        return [DominantBranchStep(tick=0, stim_vec=fallback_stim_vec.copy(), K=0.0)]


if TORCH_AVAILABLE:
    class DominantBranchEncoder(nn.Module):
        def __init__(self, config: EmoNetConfig):
            super().__init__()
            self.input_proj = nn.Linear(BRANCH_FEATURE_DIM, config.d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.n_heads,
                dim_feedforward=config.ff_dim,
                dropout=config.dropout,
                batch_first=True,
                activation="relu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
            self.to_z = nn.Linear(config.d_model, config.z_dim)

        def forward(self, branch_tensor: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            squeeze_batch = branch_tensor.dim() == 2
            if squeeze_batch:
                branch_tensor = branch_tensor.unsqueeze(0)

            key_padding_mask = None
            if attention_mask is not None:
                if attention_mask.dim() == 1:
                    attention_mask = attention_mask.unsqueeze(0)
                key_padding_mask = ~attention_mask.bool()

            hidden = self.encoder(self.input_proj(branch_tensor), src_key_padding_mask=key_padding_mask)
            if attention_mask is not None:
                mask = attention_mask.float().unsqueeze(-1)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
            else:
                pooled = hidden.mean(dim=1)

            z = self.to_z(pooled)
            return z.squeeze(0) if squeeze_batch else z


    class ZtoSRegressor(nn.Module):
        def __init__(self, config: EmoNetConfig):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(config.z_dim, 128),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, config.s_dim),
                nn.Sigmoid(),
            )

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            return self.net(z)
else:
    class DominantBranchEncoder:
        def __init__(self, config: EmoNetConfig):
            self.config = config

        def forward(self, branch_tensor: Any, attention_mask: Any = None) -> Any:
            raise RuntimeError("torch is required to encode dominant branches into z")


    class ZtoSRegressor:
        def __init__(self, config: EmoNetConfig):
            self.config = config

        def forward(self, z: Any) -> Any:
            raise RuntimeError("torch is required to regress s from z")


class EmoNet:
    def __init__(self, config: Optional[EmoNetConfig] = None):
        self.config = config or EmoNetConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.graph = EmoNetGraph(self.config, self.rng)
        self.stim_encoder = StimEncoder()
        self.branch_extractor = BranchExtractor()

        self.z_encoder = DominantBranchEncoder(self.config) if TORCH_AVAILABLE else None
        self.z_to_s_regressor = ZtoSRegressor(self.config) if TORCH_AVAILABLE else None

        self.last_base_stim_vec = np.zeros(STIM_DIM, dtype=np.float32)
        self.pending_signals: dict[int, list[float]] = {}
        self.pruned_branch_log: list[TickRecord] = []
        self.topk_branches: list[BranchPath] = []
        self.dominant_branch: list[DominantBranchStep] = []
        self._last_delta_k = math.inf

        self.build_graph()
        self.reset()

    def reset(self) -> None:
        for neuron in self.state.neurons:
            neuron.K = 0.0
            neuron.stim_vec = np.zeros(STIM_DIM, dtype=np.float32)
            neuron.k_threshold = self.config.k_threshold_base
            neuron.k_remem = self.config.k_remem_base
            neuron.refractory_left = 0
            neuron.dropped_out = False
            neuron.memories.clear()

        self.state.tick = 0
        self.state.prev_K = np.zeros(self.config.n_neurons, dtype=np.float32)
        self.state.curr_K = np.zeros(self.config.n_neurons, dtype=np.float32)
        self.state.global_threshold_shift = 0.0
        self.state.global_remem_shift = 0.0
        self.state.alive_mask = np.ones(self.config.n_neurons, dtype=bool)
        self.state.branch_log = []

        self.last_base_stim_vec = np.zeros(STIM_DIM, dtype=np.float32)
        self.pending_signals = {}
        self.pruned_branch_log = []
        self.topk_branches = []
        self.dominant_branch = []
        self._last_delta_k = math.inf

    def build_graph(self) -> None:
        neurons = self.graph.build_graph()
        self.state = EmoNetState(
            neurons=neurons,
            tick=0,
            prev_K=np.zeros(self.config.n_neurons, dtype=np.float32),
            curr_K=np.zeros(self.config.n_neurons, dtype=np.float32),
            global_threshold_shift=0.0,
            global_remem_shift=0.0,
            alive_mask=np.ones(self.config.n_neurons, dtype=bool),
            branch_log=[],
        )

    def text_to_stim_vec(self, text: str | Sequence[float] | np.ndarray) -> np.ndarray:
        return self.stim_encoder.encode(text)

    def run_tick(self, base_stim_vec: np.ndarray, text: str) -> TickRecord:
        self._restore_awake_neurons()

        self.state.global_threshold_shift *= 1.0 - self.config.global_recovery_rate
        self.state.global_remem_shift *= 1.0 - self.config.global_recovery_rate

        input_strengths = np.zeros(self.config.n_neurons, dtype=np.float32)
        for node_id, signals in self.pending_signals.items():
            if signals:
                input_strengths[node_id] = float(np.mean(signals))

        previous_k = self.state.curr_K.copy()
        effective_threshold = self.config.k_threshold_base - self.state.global_threshold_shift
        effective_remem = self.config.k_remem_base - self.state.global_remem_shift

        active_candidates: list[int] = []
        for neuron in self.state.neurons:
            neuron.stim_vec = base_stim_vec.copy()
            neuron.k_threshold = effective_threshold
            neuron.k_remem = effective_remem

            neuron.K *= self.config.k_decay
            if neuron.refractory_left > 0 or neuron.dropped_out:
                neuron.K = max(0.0, neuron.K)
                continue

            neuron.K += float(input_strengths[neuron.neuron_id])
            neuron.K += 0.3 * float(neuron.stim_vec[0]) + 0.3 * float(neuron.stim_vec[2])
            neuron.K -= 0.3 * float(neuron.stim_vec[1]) + 0.3 * float(neuron.stim_vec[3])
            neuron.K = max(0.0, neuron.K)

            if neuron.K > effective_threshold:
                active_candidates.append(neuron.neuron_id)

        for node_id in active_candidates:
            neuron = self.state.neurons[node_id]
            self._apply_memory_sequence(neuron, text, effective_remem)
            self._apply_type_effect(neuron)

        self._apply_modulatory_effects(active_candidates)

        final_active_nodes = [
            node_id
            for node_id in active_candidates
            if not self.state.neurons[node_id].dropped_out
        ]

        for node_id in final_active_nodes:
            self._apply_rewiring(self.state.neurons[node_id])

        next_pending_signals: dict[int, list[float]] = {}
        edges_fired: list[tuple[int, int]] = []
        for node_id in final_active_nodes:
            neuron = self.state.neurons[node_id]
            fire_value = float(neuron.K)
            for dst in sorted(neuron.out_neighbors):
                next_pending_signals.setdefault(dst, []).append(fire_value)
                edges_fired.append((node_id, dst))
            neuron.refractory_left = self.config.refractory_ticks

        record = TickRecord(
            tick=self.state.tick,
            active_nodes=sorted(final_active_nodes),
            node_states={
                node_id: NodeStepState(
                    K=float(self.state.neurons[node_id].K),
                    stim_vec=self.state.neurons[node_id].stim_vec.copy(),
                )
                for node_id in sorted(final_active_nodes)
            },
            edges_fired=edges_fired,
        )
        self.state.branch_log.append(record)

        for neuron in self.state.neurons:
            if neuron.refractory_left > 0:
                neuron.refractory_left = max(0, neuron.refractory_left - 1)

        self.pending_signals = next_pending_signals
        self.state.prev_K = previous_k
        self.state.curr_K = np.asarray([neuron.K for neuron in self.state.neurons], dtype=np.float32)
        self.state.alive_mask = np.asarray([not neuron.dropped_out for neuron in self.state.neurons], dtype=bool)
        self._last_delta_k = float(np.mean(np.abs(self.state.curr_K - self.state.prev_K)))
        self.state.tick += 1

        return record

    def run_until_converged(self, text: str | Sequence[float] | np.ndarray) -> np.ndarray:
        base_stim_vec = self.text_to_stim_vec(text)
        self.last_base_stim_vec = base_stim_vec.copy()
        input_text = text if isinstance(text, str) else repr(list(np.asarray(base_stim_vec, dtype=float)))

        while self.state.tick < self.config.max_ticks:
            self.run_tick(base_stim_vec, input_text)
            if self._last_delta_k < self.config.delta_k_eps:
                break

        return base_stim_vec

    def prune_to_survivors(self) -> list[TickRecord]:
        self.pruned_branch_log = self.branch_extractor.prune_to_survivors(self.state.branch_log)
        return self.pruned_branch_log

    def extract_topk_branches(self) -> list[BranchPath]:
        source_log = self.pruned_branch_log or self.prune_to_survivors()
        self.topk_branches = self.branch_extractor.extract_topk_branches(source_log, self.config.topk_branches)
        return self.topk_branches

    def build_dominant_branch(self) -> list[DominantBranchStep]:
        topk_paths = self.topk_branches or self.extract_topk_branches()
        self.dominant_branch = self.branch_extractor.build_dominant_branch(
            topk_paths=topk_paths,
            fallback_stim_vec=self.last_base_stim_vec,
            branch_log=self.pruned_branch_log or self.state.branch_log,
            topk=self.config.topk_branches,
        )
        return self.dominant_branch

    def dominant_branch_to_tensor(
        self, dominant_branch: Optional[list[DominantBranchStep]] = None
    ) -> np.ndarray | Any:
        branch = dominant_branch or self.dominant_branch or self.build_dominant_branch()
        features: list[np.ndarray] = []
        for step in branch:
            tick_norm = float(step.tick) / float(self.config.max_ticks)
            features.append(
                np.concatenate(
                    [
                        step.stim_vec.astype(np.float32, copy=False),
                        np.asarray([step.K, tick_norm], dtype=np.float32),
                    ]
                )
            )

        if not features:
            features.append(np.zeros(BRANCH_FEATURE_DIM, dtype=np.float32))

        branch_tensor = np.stack(features, axis=0)
        if TORCH_AVAILABLE:
            return torch.from_numpy(branch_tensor)
        return branch_tensor

    def encode_z(self, branch_tensor: Optional[Any] = None) -> Any:
        if not TORCH_AVAILABLE or self.z_encoder is None:
            raise RuntimeError("torch is required for z encoding")

        tensor = branch_tensor if branch_tensor is not None else self.dominant_branch_to_tensor()
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor, dtype=torch.float32)
        return self.z_encoder(tensor.float())

    def regress_s(self, z: Optional[Any] = None) -> Any:
        if not TORCH_AVAILABLE or self.z_to_s_regressor is None:
            raise RuntimeError("torch is required for style regression")

        latent = z if z is not None else self.encode_z()
        if not isinstance(latent, torch.Tensor):
            latent = torch.as_tensor(latent, dtype=torch.float32)
        return self.z_to_s_regressor(latent.float())

    def forward(self, text: str | Sequence[float] | np.ndarray) -> dict[str, Any]:
        self.reset()
        base_stim_vec = self.run_until_converged(text)
        pruned_branch_log = self.prune_to_survivors()
        dominant_branch = self.build_dominant_branch()
        branch_tensor = self.dominant_branch_to_tensor(dominant_branch)
        z = self.encode_z(branch_tensor)
        s = self.regress_s(z)
        return {
            "stim_vec": base_stim_vec,
            "pruned_branch_log": pruned_branch_log,
            "dominant_branch": dominant_branch,
            "z": z,
            "s": s,
        }

    def format_style_prompt(self, s: Sequence[float] | np.ndarray | Any) -> str:
        if TORCH_AVAILABLE and isinstance(s, torch.Tensor):
            values = s.detach().cpu().numpy().reshape(-1)
        else:
            values = np.asarray(s, dtype=np.float32).reshape(-1)
        if values.shape != (self.config.s_dim,):
            raise ValueError(f"s must have shape ({self.config.s_dim},), got {values.shape}")

        lines = ["[STYLE_VECTOR]"]
        for axis_name, value in zip(STYLE_AXES, values, strict=False):
            lines.append(f"{axis_name}={float(value):.4f}")
        lines.extend(
            [
                "",
                "[INSTRUCTION]",
                "위 스타일 벡터를 말투 제어값으로 사용하라.",
                "내용 자체는 사용자 요청에 맞게 답하되,",
                "문체, 어조, 리듬, 거리감, 직설성은 위 벡터를 따르라.",
            ]
        )
        return "\n".join(lines)

    def _apply_memory_sequence(self, neuron: NeuronState, text: str, effective_remem: float) -> None:
        if neuron.K > effective_remem:
            neuron.memories.append(
                MemoryItem(
                    stim_vec=neuron.stim_vec.copy(),
                    K_snapshot=float(neuron.K),
                    input_text=text,
                    created_tick=self.state.tick,
                    strength=float(neuron.K),
                )
            )
            if len(neuron.memories) > self.config.max_memory_per_neuron:
                weakest_index = min(range(len(neuron.memories)), key=lambda idx: neuron.memories[idx].strength)
                del neuron.memories[weakest_index]

        retained_memories: list[MemoryItem] = []
        for memory in neuron.memories:
            memory.strength *= self.config.memory_decay
            if memory.strength >= self.config.memory_delete_threshold:
                retained_memories.append(memory)
        neuron.memories = retained_memories

        for memory in neuron.memories:
            similarity = cosine_similarity(neuron.stim_vec, memory.stim_vec)
            memory.strength += self.config.memory_sim_gain * similarity

        if not neuron.memories:
            return

        strengths = np.asarray([memory.strength for memory in neuron.memories], dtype=np.float32)
        weights = strengths / float(strengths.sum() + 1e-8)

        memory_stim = np.zeros(STIM_DIM, dtype=np.float32)
        memory_k = 0.0
        for weight, memory in zip(weights, neuron.memories, strict=False):
            memory_stim += float(weight) * memory.stim_vec
            memory_k += float(weight) * memory.K_snapshot

        neuron.stim_vec = clamp_stim_vec(
            (1.0 - self.config.memory_stim_mix) * neuron.stim_vec + self.config.memory_stim_mix * memory_stim
        )
        neuron.K = max(0.0, neuron.K + self.config.memory_k_mix * memory_k)

    def _apply_type_effect(self, neuron: NeuronState) -> None:
        mean_value = float(np.mean(neuron.stim_vec))
        centered = mean_value - neuron.stim_vec
        effect_strength = min(0.5, 0.1 * float(neuron.K))

        if neuron.neuron_type == "inhibitory":
            neuron.stim_vec = neuron.stim_vec + effect_strength * centered
        elif neuron.neuron_type == "excitatory":
            neuron.stim_vec = neuron.stim_vec + effect_strength * (-centered)
        neuron.stim_vec = clamp_stim_vec(neuron.stim_vec)

    def _apply_modulatory_effects(self, active_candidates: list[int]) -> None:
        modulatory_ids = [
            node_id
            for node_id in active_candidates
            if self.state.neurons[node_id].neuron_type == "modulatory"
        ]
        if not modulatory_ids:
            return

        mod_vec = np.mean(
            [self.state.neurons[node_id].stim_vec for node_id in modulatory_ids],
            axis=0,
            dtype=np.float32,
        )
        mod_vec = clamp_stim_vec(mod_vec)

        drop_probability = self.config.mela_dropout_gain * float(mod_vec[3])
        awake_nodes = [neuron for neuron in self.state.neurons if not neuron.dropped_out]
        for neuron in awake_nodes:
            if self.rng.random() < drop_probability:
                neuron.dropped_out = True

        self.state.global_threshold_shift += self.config.ne_thresh_reduce_gain * float(mod_vec[2])
        self.state.global_remem_shift += self.config.ne_remem_reduce_gain * float(mod_vec[2])

    def _apply_rewiring(self, neuron: NeuronState) -> None:
        prune_probability = self.config.sero_prune_gain * float(neuron.stim_vec[1])
        outgoing = list(neuron.out_neighbors)
        self.rng.shuffle(outgoing)

        for dst in outgoing:
            if len(neuron.out_neighbors) <= self.config.min_out_degree:
                break
            if self.rng.random() < prune_probability:
                self.graph.remove_edge(self.state.neurons, neuron.neuron_id, dst)

        add_attempts = int(math.floor(self.config.dopa_rewire_gain * float(neuron.stim_vec[0]) * 4.0))
        for _ in range(add_attempts):
            if len(neuron.out_neighbors) >= self.config.max_out_degree:
                break

            candidates = [
                node_id
                for node_id in range(self.config.n_neurons)
                if node_id != neuron.neuron_id and node_id not in neuron.out_neighbors
            ]
            if not candidates:
                break
            dst = int(self.rng.choice(candidates))
            self.graph.add_edge(self.state.neurons, neuron.neuron_id, dst)

    def _restore_awake_neurons(self) -> None:
        for neuron in self.state.neurons:
            neuron.dropped_out = False
