from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

STYLE_NAMES: List[str] = [
    "positive_valence", "negative_valence", "arousal", "calmness", "dominance", "submissiveness",
    "warmth", "empathy", "reassurance", "hostility", "social_distance", "protectiveness",
    "assertiveness", "deference", "cooperativeness", "confrontationality", "guidance", "permission_seeking",
    "certainty", "hedging", "confidence", "tentativeness", "reflectiveness", "spontaneity",
    "verbosity", "directness", "formality", "concreteness", "abstraction", "figurativeness",
    "humor_playfulness", "emphasis_intensity",
]

TRAIT_NAMES: List[str] = [
    "threat_sensitivity",
    "social_sensitivity",
    "recovery_speed",
    "impulsive_reactivity",
    "emotional_inertia",
    "baseline_warmth",
    "baseline_directness",
    "baseline_formality",
]

@dataclass
class TextEncoderConfig:
    vocab_size: int = 8192
    embed_dim: int = 192
    hidden_dim: int = 256
    dropout: float = 0.1
    max_tokens: int = 96

@dataclass
class DynamicsConfig:
    num_neurons: int = 192
    excitatory_ratio: float = 0.60
    inhibitory_ratio: float = 0.25
    modulatory_ratio: float = 0.15
    t_max: int = 4
    potential_decay: float = 0.85
    activity_decay: float = 0.95
    persistent_update_rate: float = 0.01
    reaction_scale: float = 0.20
    episode_memory_scale: float = 0.25
    persistent_memory_scale: float = 0.12
    initial_connect_prob: float = 0.06
    weight_scale: float = 0.15

@dataclass
class ClusterConfig:
    min_cluster_size: int = 8
    max_cluster_size: int = 40
    recluster_min_gap: int = 20
    recluster_edge_change_ratio: float = 0.12
    edge_threshold: float = 0.04
    target_degree: float = 10.0
    target_activity: float = 0.18
    alpha: float = 0.35
    beta: float = 0.30
    gamma: float = 0.15
    delta: float = 0.20
    tau_rewire: float = 0.42
    tau_prune: float = 0.05
    tau_corr: float = 0.55
    cooldown: int = 4
    emergency_threshold: float = 0.80
    max_edge_updates: int = 12
    edge_budget_ratio: float = 0.02

@dataclass
class BranchConfig:
    l_max: int = 6
    tau_root: float = 0.60
    tau_edge: float = 0.05
    tau_flow: float = 0.10
    tau_branch_min: float = 0.18
    tau_prefix: float = 0.70
    tau_node: float = 0.60
    tau_cluster: float = 0.75
    per_root_topk: int = 4
    global_topk: int = 32
    lambda_f: float = 0.30
    lambda_a: float = 0.22
    lambda_m: float = 0.20
    lambda_c: float = 0.15
    lambda_t: float = 0.08
    lambda_l: float = 0.05
    mu1: float = 0.35
    mu2: float = 0.20
    mu3: float = 0.20
    mu4: float = 0.15
    mu5: float = 0.10
    merge_eta: float = 0.30

@dataclass
class LatentConfig:
    history_dim: int = 64
    path_model_dim: int = 128
    path_transformer_layers: int = 2
    path_transformer_heads: int = 4
    global_history_hidden: int = 128
    latent_dim_candidates: Tuple[int, int, int] = (32, 64, 128)
    default_latent_dim: int = 64
    max_cluster_embeddings: int = 64
    max_path_steps: int = 32

@dataclass
class StyleConfig:
    num_styles: int = 32
    safety_hostility_clip: float = 0.20
    safety_confront_clip: float = 0.25
    scorer_hidden_dim: int = 256

@dataclass
class TrainingConfig:
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    device: str = "cpu"

@dataclass
class AppConfig:
    text: TextEncoderConfig = field(default_factory=TextEncoderConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    branch: BranchConfig = field(default_factory=BranchConfig)
    latent: LatentConfig = field(default_factory=LatentConfig)
    style: StyleConfig = field(default_factory=StyleConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    trait_dim: int = 8
    control_dim: int = 4


def default_config() -> AppConfig:
    return AppConfig()
