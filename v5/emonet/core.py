from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Sequence
import math
import re

import numpy as np

try:
    import joblib
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
except ImportError:
    joblib = None
    TruncatedSVD = None
    TfidfVectorizer = None
    Ridge = None
    Pipeline = None

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None

from .paths import default_benchmark_csv, default_stim_dataset_csv, project_root


TORCH_AVAILABLE = torch is not None
SKLEARN_AVAILABLE = TfidfVectorizer is not None

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
    intrinsic_bias: np.ndarray = field(default_factory=lambda: np.zeros(STIM_DIM, dtype=np.float32))
    k_threshold: float = 1.0
    k_remem: float = 1.2
    refractory_left: int = 0
    recent_activity: float = 0.0
    fatigue: float = 0.0
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
    max_ticks: int = 128
    min_ticks_before_converged: int = 6
    delta_k_eps: float = 1e-3
    convergence_patience: int = 6
    activity_count_delta_eps: float = 2.0
    edge_count_delta_eps: float = 12.0
    activity_churn_eps: float = 0.02

    k_threshold_base: float = 0.72
    k_remem_base: float = 0.95
    k_decay: float = 0.99
    refractory_ticks: int = 1
    input_topk: int = 2
    input_signal_clip: float = 1.50

    memory_decay: float = 0.985
    memory_delete_threshold: float = 0.05
    memory_sim_gain: float = 0.10
    memory_stim_mix: float = 0.25
    memory_k_mix: float = 0.35
    memory_k_snapshot_log_gain: float = 0.75
    memory_k_snapshot_cap: float = 3.0
    max_memory_per_neuron: int = 64
    state_self_stim_mix: float = 0.55
    state_parent_stim_mix: float = 0.25
    state_base_stim_mix: float = 0.15
    state_bias_stim_mix: float = 0.05
    recent_activity_decay: float = 0.80
    hysteresis_threshold_gain: float = 0.12
    hysteresis_remem_gain: float = 0.08
    hysteresis_k_bonus: float = 0.08
    intrinsic_alignment_gain: float = 0.24
    intrinsic_alignment_salience_floor: float = 0.20
    fatigue_decay: float = 0.90
    fatigue_gain: float = 0.30
    fatigue_threshold_gain: float = 0.18
    fatigue_k_leak: float = 0.08
    fire_output_log_gain: float = 0.75
    inhibitory_suppression_gain: float = 0.18
    max_active_fraction_per_tick: float = 1.0
    target_active_fraction: float = 0.18
    homeostatic_threshold_gain: float = 1.20
    homeostatic_k_leak_gain: float = 0.80
    homeostatic_fire_gain: float = 5.00
    sensory_drive_decay_ticks: float = 6.0

    max_out_degree: int = 12
    min_out_degree: int = 1
    dopa_rewire_gain: float = 0.80
    sero_prune_gain: float = 0.04

    mela_dropout_gain: float = 0.04
    ne_thresh_reduce_gain: float = 0.25
    ne_remem_reduce_gain: float = 0.25
    global_recovery_rate: float = 0.10

    z_dim: int = 64
    s_dim: int = 32
    topk_branches: int = 4
    branch_end_window: int = 6
    branch_length_bonus: float = 0.35
    branch_k_feature_log_scale: float = 128.0
    z_encoder_mode: Literal["stat", "transformer"] = "stat"
    z_encoder_path: Path = field(default_factory=lambda: _project_root() / "artifacts" / "dominant_branch_encoder.pt")
    load_z_encoder_checkpoint: bool = True

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
        if self.s_dim <= 0:
            raise ValueError("s_dim must be positive")
        if self.min_ticks_before_converged < 0:
            raise ValueError("min_ticks_before_converged must be non-negative")
        if self.convergence_patience < 0:
            raise ValueError("convergence_patience must be non-negative")
        if self.branch_end_window <= 0:
            raise ValueError("branch_end_window must be positive")
        if self.input_topk <= 0:
            raise ValueError("input_topk must be positive")
        if self.input_signal_clip <= 0.0:
            raise ValueError("input_signal_clip must be positive")
        if not 0.0 <= self.recent_activity_decay <= 1.0:
            raise ValueError("recent_activity_decay must be in [0, 1]")
        for field_name in (
            "state_self_stim_mix",
            "state_parent_stim_mix",
            "state_base_stim_mix",
            "state_bias_stim_mix",
            "hysteresis_threshold_gain",
            "hysteresis_remem_gain",
            "hysteresis_k_bonus",
            "intrinsic_alignment_gain",
            "intrinsic_alignment_salience_floor",
            "memory_k_snapshot_log_gain",
            "memory_k_snapshot_cap",
            "fatigue_decay",
            "fatigue_gain",
            "fatigue_threshold_gain",
            "fatigue_k_leak",
            "fire_output_log_gain",
            "inhibitory_suppression_gain",
            "activity_count_delta_eps",
            "edge_count_delta_eps",
            "activity_churn_eps",
        ):
            if getattr(self, field_name) < 0.0:
                raise ValueError(f"{field_name} must be non-negative")
        if self.fatigue_decay > 1.0:
            raise ValueError("fatigue_decay must be in [0, 1]")


def _project_root() -> Path:
    return project_root()


@dataclass(slots=True)
class StimEncoderConfig:
    dataset_csv: Path = field(default_factory=default_stim_dataset_csv)
    benchmark_csv: Path = field(default_factory=default_benchmark_csv)
    model_cache_path: Path = field(
        default_factory=lambda: _project_root() / "artifacts" / "ridge_stim_encoder.joblib"
    )
    prefer_model: str = "Ridge"
    ridge_alpha: float = 2.0
    random_state: int = 42
    force_refit: bool = False
    max_samples: Optional[int] = None


@dataclass(slots=True)
class ZSDecoderConfig:
    model_path: Path = field(default_factory=lambda: _project_root() / "artifacts" / "z_to_s_decoder.npz")
    ridge_alpha: float = 1.0


class LinearZtoSDecoder:
    def __init__(
        self,
        config: Optional[ZSDecoderConfig] = None,
        z_dim: int = 64,
        s_dim: int = len(STYLE_AXES),
    ):
        self.config = config or ZSDecoderConfig()
        self.z_dim = int(z_dim)
        self.s_dim = int(s_dim)
        self.x_mean: Optional[np.ndarray] = None
        self.x_scale: Optional[np.ndarray] = None
        self.y_bias: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None

    @property
    def fitted(self) -> bool:
        return (
            self.x_mean is not None
            and self.x_scale is not None
            and self.y_bias is not None
            and self.weights is not None
        )

    def fit(self, z: np.ndarray, s: np.ndarray) -> "LinearZtoSDecoder":
        z_arr = np.asarray(z, dtype=np.float32)
        s_arr = np.asarray(s, dtype=np.float32)
        if z_arr.ndim != 2:
            raise ValueError(f"z must be 2D [n_samples, z_dim], got {z_arr.shape}")
        if s_arr.ndim != 2:
            raise ValueError(f"s must be 2D [n_samples, s_dim], got {s_arr.shape}")
        if z_arr.shape[0] != s_arr.shape[0]:
            raise ValueError("z and s must have the same number of rows")
        if z_arr.shape[1] != self.z_dim:
            raise ValueError(f"expected z_dim={self.z_dim}, got {z_arr.shape[1]}")
        if s_arr.shape[1] != self.s_dim:
            raise ValueError(f"expected s_dim={self.s_dim}, got {s_arr.shape[1]}")
        if z_arr.shape[0] < 2:
            raise ValueError("at least 2 rows are required to fit z->s decoder")

        self.x_mean = z_arr.mean(axis=0, dtype=np.float64).astype(np.float32)
        x_scale = z_arr.std(axis=0, dtype=np.float64).astype(np.float32)
        x_scale = np.where(x_scale < 1e-6, 1.0, x_scale).astype(np.float32)
        self.x_scale = x_scale
        self.y_bias = s_arr.mean(axis=0, dtype=np.float64).astype(np.float32)

        z_norm = (z_arr - self.x_mean) / self.x_scale
        centered_targets = s_arr - self.y_bias

        gram = z_norm.T @ z_norm
        ridge = float(self.config.ridge_alpha) * np.eye(self.z_dim, dtype=np.float32)
        rhs = z_norm.T @ centered_targets
        self.weights = np.linalg.solve(gram + ridge, rhs).astype(np.float32)
        return self

    def predict(self, z: np.ndarray) -> np.ndarray:
        if not self.fitted or self.x_mean is None or self.x_scale is None or self.y_bias is None or self.weights is None:
            raise RuntimeError("z->s decoder is not fitted")
        z_arr = np.asarray(z, dtype=np.float32)
        squeeze_batch = z_arr.ndim == 1
        if squeeze_batch:
            z_arr = z_arr.reshape(1, -1)
        if z_arr.ndim != 2 or z_arr.shape[1] != self.z_dim:
            raise ValueError(f"z must have shape [n_samples, {self.z_dim}]")
        z_norm = (z_arr - self.x_mean) / self.x_scale
        pred = z_norm @ self.weights + self.y_bias
        pred = np.clip(pred, 0.0, 1.0).astype(np.float32)
        return pred[0] if squeeze_batch else pred

    def mean_absolute_error(self, z: np.ndarray, s: np.ndarray) -> float:
        pred = self.predict(z)
        target = np.asarray(s, dtype=np.float32)
        if pred.shape != target.shape:
            raise ValueError(f"prediction and target shapes must match, got {pred.shape} vs {target.shape}")
        return float(np.mean(np.abs(pred - target)))

    def save(self, path: Optional[Path] = None) -> Path:
        if not self.fitted or self.x_mean is None or self.x_scale is None or self.y_bias is None or self.weights is None:
            raise RuntimeError("z->s decoder is not fitted")
        target_path = Path(path) if path is not None else self.config.model_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with target_path.open("wb") as handle:
            np.savez(
                handle,
                z_dim=np.asarray(self.z_dim, dtype=np.int32),
                s_dim=np.asarray(self.s_dim, dtype=np.int32),
                ridge_alpha=np.asarray(self.config.ridge_alpha, dtype=np.float32),
                x_mean=self.x_mean,
                x_scale=self.x_scale,
                y_bias=self.y_bias,
                weights=self.weights,
            )
        return target_path

    @classmethod
    def load(cls, path: Path) -> "LinearZtoSDecoder":
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"z->s decoder model not found: {model_path}")
        with np.load(model_path, allow_pickle=False) as payload:
            decoder = cls(
                config=ZSDecoderConfig(
                    model_path=model_path,
                    ridge_alpha=float(payload["ridge_alpha"]),
                ),
                z_dim=int(payload["z_dim"]),
                s_dim=int(payload["s_dim"]),
            )
            decoder.x_mean = np.asarray(payload["x_mean"], dtype=np.float32)
            decoder.x_scale = np.asarray(payload["x_scale"], dtype=np.float32)
            decoder.y_bias = np.asarray(payload["y_bias"], dtype=np.float32)
            decoder.weights = np.asarray(payload["weights"], dtype=np.float32)
        return decoder


class SafeTruncatedSVD:
    def __init__(self, n_components: int = 300, random_state: int = 42):
        self.n_components = int(n_components)
        self.random_state = int(random_state)
        self._svd: Optional[TruncatedSVD] = None

    def fit(self, X, y=None):
        if TruncatedSVD is None:
            raise RuntimeError("scikit-learn is required for SVD-backed stimulus encoding")
        n_features = int(X.shape[1])
        safe = min(self.n_components, max(1, n_features - 1))
        self._svd = TruncatedSVD(n_components=safe, random_state=self.random_state)
        self._svd.fit(X)
        return self

    def transform(self, X):
        if self._svd is None:
            raise RuntimeError("SafeTruncatedSVD is not fitted")
        return self._svd.transform(X)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)

# 힌트를 주면 연구 목적이랑 철학에 맞지 않잖아!!!!
class StimEncoder:
    POSITIVE_HINTS = {
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
        "좋",
        "기쁘",
        "행복",
        "신나",
        "즐겁",
        "감사",
        "고맙",
        "다행",
        "웃",
        "설렌",
    }
    GAIN_HINTS = {
        "success",
        "win",
        "reward",
        "progress",
        "achieve",
        "해냈",
        "성공",
        "합격",
        "좋아졌",
        "기회",
        "보상",
        "늘었",
        "이겼",
    }
    AGENCY_HINTS = {
        "decide",
        "control",
        "can do",
        "handle",
        "choose",
        "결정",
        "통제",
        "할 수",
        "직접",
        "해결",
        "주도",
        "선택",
    }
    CALM_HINTS = {
        "calm",
        "steady",
        "safe",
        "okay",
        "gentle",
        "peace",
        "settle",
        "stable",
        "편안",
        "차분",
        "안정",
        "괜찮",
        "평온",
        "진정",
    }
    SAFETY_HINTS = {
        "support",
        "trust",
        "together",
        "warm",
        "safe",
        "도와",
        "믿",
        "함께",
        "응원",
        "위로",
        "안전",
        "다정",
    }
    THREAT_HINTS = {
        "urgent",
        "risk",
        "warning",
        "critical",
        "danger",
        "must",
        "불안",
        "위험",
        "압박",
        "긴장",
        "무섭",
        "화가",
        "짜증",
        "분노",
        "억울",
        "답답",
    }
    ALERT_HINTS = {
        "alert",
        "asap",
        "immediately",
        "issue",
        "deadline",
        "지금",
        "당장",
        "빨리",
        "즉시",
        "문제",
        "경고",
        "불길",
    }
    FATIGUE_HINTS = {
        "tired",
        "burnout",
        "sleep",
        "rest",
        "slow",
        "피곤",
        "지쳤",
        "힘들",
        "번아웃",
        "소진",
        "잠",
        "졸",
        "쉬고",
        "무기력",
    }
    REST_HINTS = {
        "night",
        "bed",
        "dream",
        "quiet",
        "late",
        "밤",
        "잠",
        "휴식",
        "쉬고",
        "누워",
        "졸리",
        "새벽",
    }

    WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)

    def __init__(self, config: Optional[StimEncoderConfig] = None):
        self.config = config or StimEncoderConfig()
        self.pipeline: Optional[Pipeline] = None
        self.vector_name = "char_tfidf"
        self.vector_kind = "char"
        self.use_svd = False
        self.svd_dim = 300
        self._loaded = False

    def fit(self) -> None:
        if not SKLEARN_AVAILABLE or joblib is None:
            raise RuntimeError("scikit-learn and joblib are required for ridge stimulus encoding")

        self._choose_vector_setup()
        df = self._load_dataset()
        targets = self._resolve_training_targets(df)

        self.pipeline = self._build_pipeline()
        self.pipeline.fit(df["text"].astype(str).to_numpy(), targets)

        cache_path = self.config.model_cache_path
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "pipeline": self.pipeline,
                "vector_name": self.vector_name,
                "vector_kind": self.vector_kind,
                "use_svd": self.use_svd,
                "svd_dim": self.svd_dim,
                "config": self.config,
            },
            cache_path,
        )
        self._loaded = True

    def ensure_fitted(self) -> None:
        if self.pipeline is not None:
            return
        if not SKLEARN_AVAILABLE or joblib is None:
            raise RuntimeError("scikit-learn and joblib are required for ridge stimulus encoding")

        cache_path = self.config.model_cache_path
        if cache_path.exists() and not self.config.force_refit:
            artifact = joblib.load(cache_path)
            self.pipeline = artifact["pipeline"]
            self.vector_name = artifact.get("vector_name", self.vector_name)
            self.vector_kind = artifact.get("vector_kind", self.vector_kind)
            self.use_svd = artifact.get("use_svd", self.use_svd)
            self.svd_dim = artifact.get("svd_dim", self.svd_dim)
            self._loaded = True
            return
        self.fit()

    def encode(self, text: str | Sequence[float] | np.ndarray) -> np.ndarray:
        if isinstance(text, np.ndarray):
            return self._validate_stim_vec(text)
        if isinstance(text, (list, tuple)):
            return self._validate_stim_vec(np.asarray(text, dtype=np.float32))
        if not isinstance(text, str):
            raise TypeError("StimEncoder expects a string or a 4D stim vector")

        self.ensure_fitted()
        if self.pipeline is None:
            raise RuntimeError("Stimulus encoder pipeline is not fitted")

        predicted = np.asarray(self.pipeline.predict([text])[0], dtype=np.float32).reshape(STIM_DIM)
        return clamp_stim_vec(predicted)

    def _choose_vector_setup(self) -> None:
        benchmark_csv = self.config.benchmark_csv
        if benchmark_csv.exists():
            import pandas as pd

            df = pd.read_csv(benchmark_csv)
            if "status" in df.columns:
                df = df[df["status"] == "ok"].copy()
            if len(df) > 0 and "model" in df.columns:
                preferred = df[df["model"] == self.config.prefer_model].copy()
                if len(preferred) > 0:
                    df = preferred
            if len(df) > 0 and "RMSE(mean)" in df.columns:
                df = df.sort_values(["RMSE(mean)", "MAE(mean)"], ascending=[True, True]).reset_index(drop=True)
                vector_name = str(df.iloc[0].get("vector", self.vector_name))
                self.vector_name = vector_name
                self.vector_kind = "char" if "char" in vector_name.lower() else "word"
                self.use_svd = "svd" in vector_name.lower()
                if self.use_svd:
                    digits = "".join(ch for ch in vector_name if ch.isdigit())
                    if digits:
                        self.svd_dim = int(digits)

    def _build_pipeline(self) -> Pipeline:
        if TfidfVectorizer is None or Ridge is None or Pipeline is None:
            raise RuntimeError("scikit-learn is required for ridge stimulus encoding")

        steps: list[tuple[str, Any]] = [("tfidf", self._make_vectorizer(self.vector_kind))]
        if self.use_svd:
            steps.append(("svd", SafeTruncatedSVD(self.svd_dim, self.config.random_state)))
        steps.append(("model", Ridge(alpha=self.config.ridge_alpha, random_state=self.config.random_state)))
        return Pipeline(steps)

    def _load_dataset(self):
        import pandas as pd

        dataset_csv = self.config.dataset_csv
        if not dataset_csv.exists():
            raise FileNotFoundError(f"stimulus training dataset not found: {dataset_csv}")
        df = pd.read_csv(dataset_csv)
        if "text" not in df.columns:
            raise ValueError(f"'text' column not found in {dataset_csv}")
        stim_columns = ("dopamine", "serotonin", "norepinephrine", "melatonin")
        has_proxy_target = "y" in df.columns
        has_stim_targets = all(column in df.columns for column in stim_columns)
        if not has_proxy_target and not has_stim_targets:
            columns = ", ".join(stim_columns)
            raise ValueError(
                f"dataset must contain 'y' or direct stim columns ({columns}) in {dataset_csv}"
            )
        if self.config.max_samples is not None and self.config.max_samples > 0 and len(df) > self.config.max_samples:
            df = df.sample(n=self.config.max_samples, random_state=self.config.random_state).reset_index(drop=True)
        return df

    @classmethod
    def _resolve_training_targets(cls, df) -> np.ndarray:
        stim_columns = ("dopamine", "serotonin", "norepinephrine", "melatonin")
        if all(column in df.columns for column in stim_columns):
            stim_targets = df.loc[:, stim_columns].astype(float).to_numpy(dtype=np.float32, copy=True)
            return np.clip(stim_targets, 0.0, 1.0).astype(np.float32, copy=False)
        return cls._build_proxy_targets(df["text"].astype(str).tolist(), df["y"].astype(float).to_numpy())

    @classmethod
    def _make_vectorizer(cls, kind: str) -> TfidfVectorizer:
        if TfidfVectorizer is None:
            raise RuntimeError("scikit-learn is required for ridge stimulus encoding")
        if kind == "word":
            return TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.95, dtype=np.float32)
        if kind == "char":
            return TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, max_df=0.95, dtype=np.float32)
        raise ValueError("vector_kind must be 'word' or 'char'")

    @classmethod
    def _build_proxy_targets(cls, texts: list[str], base_scores: np.ndarray) -> np.ndarray:
        stim_targets = np.zeros((len(texts), STIM_DIM), dtype=np.float32)
        for idx, (text, base_score) in enumerate(zip(texts, base_scores, strict=False)):
            lowered = str(text).lower()
            token_count = len(cls.WORD_RE.findall(lowered))
            punctuation = min((lowered.count("!") + lowered.count("?")) / 4.0, 1.0)
            pressure = min(token_count / 40.0, 1.0)

            positive = cls._hint_fraction(lowered, cls.POSITIVE_HINTS)
            gain = cls._hint_fraction(lowered, cls.GAIN_HINTS)
            agency = cls._hint_fraction(lowered, cls.AGENCY_HINTS)
            calm = cls._hint_fraction(lowered, cls.CALM_HINTS)
            safety = cls._hint_fraction(lowered, cls.SAFETY_HINTS)
            threat = cls._hint_fraction(lowered, cls.THREAT_HINTS)
            alert = cls._hint_fraction(lowered, cls.ALERT_HINTS)
            fatigue = cls._hint_fraction(lowered, cls.FATIGUE_HINTS)
            rest = cls._hint_fraction(lowered, cls.REST_HINTS)

            score = float(np.clip(base_score, 0.0, 1.0))
            dopamine = 0.35 * score + 0.25 * gain + 0.20 * agency + 0.15 * positive + 0.05 * punctuation
            serotonin = 0.20 + 0.25 * score + 0.25 * calm + 0.25 * safety - 0.10 * threat
            norepinephrine = 0.20 + 0.35 * (1.0 - score) + 0.20 * threat + 0.15 * alert + 0.10 * punctuation
            melatonin = 0.10 + 0.30 * fatigue + 0.25 * rest + 0.20 * (1.0 - score) + 0.10 * pressure

            stim_targets[idx] = clamp_stim_vec(
                np.asarray([dopamine, serotonin, norepinephrine, melatonin], dtype=np.float32)
            )
        return stim_targets

    @staticmethod
    def _hint_fraction(text: str, hints: set[str]) -> float:
        hits = sum(1 for token in hints if token in text)
        return min(hits / 3.0, 1.0)

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

    def _sample_intrinsic_bias(self, neuron_type: Literal["inhibitory", "excitatory", "modulatory"]) -> np.ndarray:
        bias_centers = {
            "inhibitory": np.asarray([0.18, 0.58, 0.24, 0.38], dtype=np.float32),
            "excitatory": np.asarray([0.58, 0.18, 0.52, 0.18], dtype=np.float32),
            "modulatory": np.asarray([0.36, 0.34, 0.44, 0.28], dtype=np.float32),
        }
        jitter = self.rng.normal(loc=0.0, scale=0.08, size=STIM_DIM).astype(np.float32)
        return clamp_stim_vec(bias_centers[neuron_type] + jitter)

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
                    intrinsic_bias=self._sample_intrinsic_bias(neuron_type),
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
                step_k = 0.0 if state.K is None else float(state.K)
                step = BranchStep(
                    tick=record.tick,
                    node_id=node_id,
                    K=step_k,
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

    def extract_topk_branches_with_strategy(
        self,
        branch_log: list[TickRecord],
        topk: int,
        *,
        end_window: int = 1,
        length_bonus: float = 0.0,
    ) -> list[BranchPath]:
        if not branch_log:
            return []

        topk_paths: dict[tuple[int, int], list[BranchPath]] = {}

        for index, record in enumerate(branch_log):
            prev_edges = branch_log[index - 1].edges_fired if index > 0 else []
            parents_by_node: dict[int, list[int]] = {}
            for src, dst in prev_edges:
                parents_by_node.setdefault(dst, []).append(src)

            for node_id in record.active_nodes:
                state = record.node_states[node_id]
                step_k = 0.0 if state.K is None else float(state.K)
                step = BranchStep(
                    tick=record.tick,
                    node_id=node_id,
                    K=step_k,
                    stim_vec=state.stim_vec.copy(),
                )
                candidates: list[BranchPath] = []
                parent_nodes = parents_by_node.get(node_id, [])
                if parent_nodes:
                    prev_tick = branch_log[index - 1].tick
                    for parent_id in parent_nodes:
                        for parent_path in topk_paths.get((prev_tick, parent_id), []):
                            candidates.append(
                                BranchPath(
                                    score=parent_path.score + step.K + length_bonus,
                                    steps=parent_path.steps + [step],
                                )
                            )
                else:
                    candidates.append(BranchPath(score=step.K + length_bonus, steps=[step]))

                candidates.sort(key=lambda path: (path.score, len(path.steps), path.steps[-1].tick), reverse=True)
                topk_paths[(record.tick, node_id)] = candidates[:topk]

        effective_window = max(1, end_window)
        completed_paths: list[BranchPath] = []
        for record in branch_log[-effective_window:]:
            for node_id in record.active_nodes:
                completed_paths.extend(topk_paths.get((record.tick, node_id), []))
        completed_paths.sort(key=lambda path: (path.score, len(path.steps), path.steps[-1].tick), reverse=True)
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
        best_path = selected_paths[0]
        return [
            DominantBranchStep(
                tick=step.tick,
                stim_vec=clamp_stim_vec(step.stim_vec.copy()),
                K=float(step.K),
            )
            for step in best_path.steps
        ]

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
                state_k = 0.0 if state.K is None else float(state.K)
                best_k = -math.inf if best_state is None else float(best_state.K)
                if state_k > best_k:
                    best_state = DominantBranchStep(
                        tick=record.tick,
                        stim_vec=state.stim_vec.copy(),
                        K=state_k,
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


class NumpyBranchEncoder:
    def __init__(self, config: EmoNetConfig):
        self.config = config
        self.feature_dim = 8 * BRANCH_FEATURE_DIM + 3
        rng = np.random.default_rng(config.seed)
        scale = 1.0 / math.sqrt(self.feature_dim)
        self.projection = rng.normal(0.0, scale, size=(self.feature_dim, config.z_dim)).astype(np.float32)
        self.bias = rng.normal(0.0, 0.05, size=(config.z_dim,)).astype(np.float32)

    def encode(self, branch_tensor: np.ndarray) -> np.ndarray:
        features = self._summarize_branch(branch_tensor)
        latent = np.tanh(features @ self.projection + self.bias)
        return latent.astype(np.float32)

    def _summarize_branch(self, branch_tensor: np.ndarray) -> np.ndarray:
        arr = np.asarray(branch_tensor, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != BRANCH_FEATURE_DIM:
            raise ValueError(f"branch tensor must have shape [seq_len, {BRANCH_FEATURE_DIM}]")

        seq_len = arr.shape[0]
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        min_value = arr.min(axis=0)
        max_value = arr.max(axis=0)
        first = arr[0]
        last = arr[-1]
        delta = last - first

        if seq_len > 1:
            ticks = np.linspace(0.0, 1.0, seq_len, dtype=np.float32)
            centered_ticks = ticks - ticks.mean()
            denom = float(np.dot(centered_ticks, centered_ticks) + 1e-8)
            centered = arr - mean
            slopes = (centered_ticks[:, None] * centered).sum(axis=0) / denom
        else:
            slopes = np.zeros(BRANCH_FEATURE_DIM, dtype=np.float32)

        extras = np.asarray(
            [
                float(seq_len) / float(self.config.max_ticks),
                float(mean[4]) if BRANCH_FEATURE_DIM > 4 else 0.0,
                float(last[4]) if BRANCH_FEATURE_DIM > 4 else 0.0,
            ],
            dtype=np.float32,
        )
        return np.concatenate([mean, std, min_value, max_value, first, last, delta, slopes, extras], axis=0)


class EmoNet:
    def __init__(
        self,
        config: Optional[EmoNetConfig] = None,
        stim_encoder_config: Optional[StimEncoderConfig] = None,
        stim_encoder: Optional[StimEncoder] = None,
    ):
        self.config = config or EmoNetConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.graph = EmoNetGraph(self.config, self.rng)
        self.stim_encoder = stim_encoder or StimEncoder(stim_encoder_config)
        self.branch_extractor = BranchExtractor()

        self.z_encoder = DominantBranchEncoder(self.config) if TORCH_AVAILABLE else None
        self.numpy_branch_encoder = NumpyBranchEncoder(self.config)
        self.z_to_s_regressor = ZtoSRegressor(self.config) if TORCH_AVAILABLE else None
        self.use_torch_z_encoder = self.config.z_encoder_mode == "transformer" and TORCH_AVAILABLE
        if (
            self.use_torch_z_encoder
            and self.z_encoder is not None
            and self.config.load_z_encoder_checkpoint
            and Path(self.config.z_encoder_path).exists()
        ):
            self.load_z_encoder(self.config.z_encoder_path)

        self.last_base_stim_vec = np.zeros(STIM_DIM, dtype=np.float32)
        self.pending_signals: dict[int, list[tuple[float, np.ndarray]]] = {}
        self.pruned_branch_log: list[TickRecord] = []
        self.topk_branches: list[BranchPath] = []
        self.dominant_branch: list[DominantBranchStep] = []
        self._last_delta_k = math.inf
        self.last_termination_reason: str = "not_started"

        self.build_graph()
        self.reset()

    def reset(self) -> None:
        for neuron in self.state.neurons:
            neuron.K = 0.0
            neuron.stim_vec = np.zeros(STIM_DIM, dtype=np.float32)
            neuron.k_threshold = self.config.k_threshold_base
            neuron.k_remem = self.config.k_remem_base
            neuron.refractory_left = 0
            neuron.recent_activity = 0.0
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
        self.last_termination_reason = "not_started"

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

    def _aggregate_signal_bundle(
        self,
        signals: Sequence[tuple[float, np.ndarray] | float],
    ) -> tuple[float, np.ndarray]:
        parsed: list[tuple[float, np.ndarray]] = []
        for item in signals:
            if isinstance(item, tuple):
                strength, stim_vec = item
                stim_arr = np.asarray(stim_vec, dtype=np.float32).reshape(STIM_DIM)
            else:
                strength = item
                stim_arr = np.zeros(STIM_DIM, dtype=np.float32)
            parsed.append((max(0.0, float(strength)), clamp_stim_vec(stim_arr)))

        if not parsed:
            return 0.0, np.zeros(STIM_DIM, dtype=np.float32)

        parsed.sort(key=lambda pair: pair[0], reverse=True)
        topk_items = parsed[: self.config.input_topk]
        strength_sum = float(sum(strength for strength, _ in topk_items))
        clipped_strength = min(self.config.input_signal_clip, strength_sum)
        if strength_sum <= 1e-8:
            return clipped_strength, np.zeros(STIM_DIM, dtype=np.float32)

        weights = np.asarray([strength for strength, _ in topk_items], dtype=np.float32)
        weights = weights / float(weights.sum() + 1e-8)
        blended_stim = np.zeros(STIM_DIM, dtype=np.float32)
        for weight, (_, stim_vec) in zip(weights, topk_items, strict=False):
            blended_stim += float(weight) * stim_vec
        return clipped_strength, clamp_stim_vec(blended_stim)

    def _aggregate_pending_inputs(self) -> tuple[np.ndarray, np.ndarray]:
        input_strengths = np.zeros(self.config.n_neurons, dtype=np.float32)
        input_stimuli = np.zeros((self.config.n_neurons, STIM_DIM), dtype=np.float32)
        for node_id, signals in self.pending_signals.items():
            if not signals:
                continue
            strength, stim_vec = self._aggregate_signal_bundle(signals)
            input_strengths[node_id] = float(strength)
            input_stimuli[node_id] = stim_vec
        return input_strengths, input_stimuli

    def _compose_neuron_stimulus(
        self,
        neuron: NeuronState,
        base_stim_vec: np.ndarray,
        parent_stim_vec: np.ndarray,
    ) -> np.ndarray:
        components = (
            (self.config.state_self_stim_mix, neuron.stim_vec),
            (self.config.state_parent_stim_mix, parent_stim_vec),
            (self.config.state_base_stim_mix, base_stim_vec),
            (self.config.state_bias_stim_mix, neuron.intrinsic_bias),
        )
        total_weight = sum(max(0.0, float(weight)) for weight, _ in components)
        if total_weight <= 1e-8:
            return clamp_stim_vec(base_stim_vec.copy())

        blended = np.zeros(STIM_DIM, dtype=np.float32)
        for weight, stim_vec in components:
            normalized_weight = max(0.0, float(weight)) / total_weight
            blended += normalized_weight * stim_vec.astype(np.float32, copy=False)
        return clamp_stim_vec(blended)

    def _compute_local_activation_params(
        self,
        neuron: NeuronState,
        effective_threshold: float,
        effective_remem: float,
    ) -> tuple[float, float]:
        threshold = max(
            0.0,
            float(effective_threshold)
            - self.config.hysteresis_threshold_gain * float(neuron.recent_activity)
            + self.config.fatigue_threshold_gain * float(neuron.fatigue),
        )
        remem = max(
            0.0,
            float(effective_remem)
            - self.config.hysteresis_remem_gain * float(neuron.recent_activity)
            + 0.5 * self.config.fatigue_threshold_gain * float(neuron.fatigue),
        )
        return threshold, remem

    def _compute_intrinsic_alignment_drive(
        self,
        neuron: NeuronState,
        base_stim_vec: np.ndarray,
        *,
        sensory_drive: float = 1.0,
    ) -> float:
        if self.config.intrinsic_alignment_gain <= 0.0:
            return 0.0
        base_stim = clamp_stim_vec(np.asarray(base_stim_vec, dtype=np.float32).reshape(STIM_DIM))
        dopamine = float(base_stim[0])
        serotonin = float(base_stim[1])
        norepinephrine = float(base_stim[2])
        melatonin = float(base_stim[3])
        peak_salience = float(np.max(base_stim))
        mixed_load = norepinephrine + 0.5 * melatonin + 0.2 * dopamine - 0.3 * serotonin
        activation_salience = max(peak_salience, mixed_load)
        salience_floor = float(self.config.intrinsic_alignment_salience_floor)
        if salience_floor > 0.0:
            salience_gate = (activation_salience - salience_floor) / max(1e-6, 1.0 - salience_floor)
            salience_gate = float(np.clip(salience_gate, 0.0, 1.0))
        else:
            salience_gate = max(0.0, activation_salience)
        if salience_gate <= 0.0:
            return 0.0
        alignment = max(0.0, cosine_similarity(base_stim, neuron.intrinsic_bias))
        fatigue_scale = 1.0 + float(neuron.fatigue)
        return (
            float(self.config.intrinsic_alignment_gain)
            * salience_gate
            * float(sensory_drive)
            * alignment
            / fatigue_scale
        )

    def _compress_memory_k_snapshot(self, value: float) -> float:
        raw_k = max(0.0, float(value))
        log_gain = float(self.config.memory_k_snapshot_log_gain)
        if log_gain > 0.0:
            raw_k = math.log1p(log_gain * raw_k) / log_gain
        cap = float(self.config.memory_k_snapshot_cap)
        if cap > 0.0:
            raw_k = min(cap, raw_k)
        return raw_k

    def _compute_fire_value(self, neuron: NeuronState) -> float:
        raw_k = max(0.0, float(neuron.K))
        if raw_k <= 0.0:
            return 0.0
        if self.config.fire_output_log_gain > 0.0:
            raw_k = math.log1p(self.config.fire_output_log_gain * raw_k) / self.config.fire_output_log_gain
        return raw_k / (1.0 + float(neuron.fatigue))

    def _apply_lateral_inhibition(self, active_candidates: list[int]) -> list[int]:
        if self.config.inhibitory_suppression_gain <= 0.0 or not active_candidates:
            return active_candidates

        suppression_by_node = np.zeros(self.config.n_neurons, dtype=np.float32)
        for node_id in active_candidates:
            neuron = self.state.neurons[node_id]
            if neuron.neuron_type != "inhibitory":
                continue
            suppression_value = self.config.inhibitory_suppression_gain * self._compute_fire_value(neuron)
            if suppression_value <= 0.0:
                continue
            for dst in neuron.out_neighbors:
                suppression_by_node[dst] += float(suppression_value)

        if not np.any(suppression_by_node > 0.0):
            return active_candidates

        surviving_candidates: list[int] = []
        for node_id in active_candidates:
            neuron = self.state.neurons[node_id]
            suppression = float(suppression_by_node[node_id])
            if suppression > 0.0:
                neuron.K = max(0.0, neuron.K - suppression)
            if neuron.K > neuron.k_threshold:
                surviving_candidates.append(node_id)
        return surviving_candidates

    def _cap_active_candidates(self, active_candidates: list[int]) -> list[int]:
        max_fraction = float(self.config.max_active_fraction_per_tick)
        if max_fraction <= 0.0 or max_fraction >= 1.0 or not active_candidates:
            return active_candidates
        max_active = max(1, int(round(float(self.config.n_neurons) * max_fraction)))
        if len(active_candidates) <= max_active:
            return active_candidates
        ranked = sorted(
            active_candidates,
            key=lambda node_id: (
                float(self.state.neurons[node_id].K - self.state.neurons[node_id].k_threshold),
                float(self.state.neurons[node_id].K),
            ),
            reverse=True,
        )
        return sorted(ranked[:max_active])

    @staticmethod
    def _compute_activity_churn(prev_nodes: Sequence[int], curr_nodes: Sequence[int]) -> float:
        prev_set = set(int(node_id) for node_id in prev_nodes)
        curr_set = set(int(node_id) for node_id in curr_nodes)
        union = prev_set | curr_set
        if not union:
            return 0.0
        return 1.0 - float(len(prev_set & curr_set)) / float(len(union))

    def _tick_is_stable(self, prev_record: TickRecord, curr_record: TickRecord) -> bool:
        active_delta = abs(len(curr_record.active_nodes) - len(prev_record.active_nodes))
        edge_delta = abs(len(curr_record.edges_fired) - len(prev_record.edges_fired))
        activity_churn = self._compute_activity_churn(prev_record.active_nodes, curr_record.active_nodes)
        return (
            active_delta <= self.config.activity_count_delta_eps
            and edge_delta <= self.config.edge_count_delta_eps
            and activity_churn <= self.config.activity_churn_eps
        )

    def run_tick(self, base_stim_vec: np.ndarray, text: str) -> TickRecord:
        self._restore_awake_neurons()

        self.state.global_threshold_shift *= 1.0 - self.config.global_recovery_rate
        self.state.global_remem_shift *= 1.0 - self.config.global_recovery_rate

        input_strengths, input_stimuli = self._aggregate_pending_inputs()

        previous_k = self.state.curr_K.copy()
        prev_active_ratio = 0.0
        if self.state.branch_log:
            prev_active_ratio = len(self.state.branch_log[-1].active_nodes) / max(1.0, float(self.config.n_neurons))
        activity_pressure = max(0.0, prev_active_ratio - float(self.config.target_active_fraction))
        fire_scale = max(0.05, 1.0 - float(self.config.homeostatic_fire_gain) * activity_pressure)
        effective_threshold = (
            self.config.k_threshold_base
            - self.state.global_threshold_shift
            + float(self.config.homeostatic_threshold_gain) * activity_pressure
        )
        effective_remem = self.config.k_remem_base - self.state.global_remem_shift
        sensory_decay_ticks = max(0.0, float(self.config.sensory_drive_decay_ticks))
        if sensory_decay_ticks <= 0.0:
            sensory_drive = 1.0
        else:
            sensory_drive = math.exp(-float(self.state.tick) / sensory_decay_ticks)
        homeostatic_k_leak = float(self.config.homeostatic_k_leak_gain) * activity_pressure

        active_candidates: list[int] = []
        for neuron in self.state.neurons:
            neuron.recent_activity *= self.config.recent_activity_decay
            neuron.fatigue *= self.config.fatigue_decay
            neuron.stim_vec = self._compose_neuron_stimulus(
                neuron,
                base_stim_vec=base_stim_vec,
                parent_stim_vec=input_stimuli[neuron.neuron_id],
            )
            neuron.k_threshold, neuron.k_remem = self._compute_local_activation_params(
                neuron,
                effective_threshold=effective_threshold,
                effective_remem=effective_remem,
            )

            neuron.K *= self.config.k_decay
            if homeostatic_k_leak > 0.0:
                neuron.K = max(0.0, neuron.K - homeostatic_k_leak)
            neuron.K = max(0.0, neuron.K - self.config.fatigue_k_leak * float(neuron.fatigue))
            if neuron.refractory_left > 0 or neuron.dropped_out:
                neuron.K = max(0.0, neuron.K)
                continue

            neuron.K += float(input_strengths[neuron.neuron_id])
            neuron.K += self.config.hysteresis_k_bonus * float(neuron.recent_activity)
            neuron.K += self._compute_intrinsic_alignment_drive(
                neuron,
                base_stim_vec,
                sensory_drive=sensory_drive,
            )
            neuron.K += sensory_drive * (0.3 * float(neuron.stim_vec[0]) + 0.3 * float(neuron.stim_vec[2]))
            neuron.K -= sensory_drive * (0.3 * float(neuron.stim_vec[1]) + 0.3 * float(neuron.stim_vec[3]))
            neuron.K = max(0.0, neuron.K)

            if neuron.K > neuron.k_threshold:
                active_candidates.append(neuron.neuron_id)

        for node_id in active_candidates:
            neuron = self.state.neurons[node_id]
            self._apply_memory_sequence(neuron, text, neuron.k_remem)
            self._apply_type_effect(neuron)

        active_candidates = self._apply_lateral_inhibition(active_candidates)
        active_candidates = self._cap_active_candidates(active_candidates)
        self._apply_modulatory_effects(active_candidates)

        final_active_nodes = [
            node_id
            for node_id in active_candidates
            if not self.state.neurons[node_id].dropped_out
        ]

        for node_id in final_active_nodes:
            self._apply_rewiring(self.state.neurons[node_id])

        next_pending_signals: dict[int, list[tuple[float, np.ndarray]]] = {}
        edges_fired: list[tuple[int, int]] = []
        for node_id in final_active_nodes:
            neuron = self.state.neurons[node_id]
            fire_value = fire_scale * self._compute_fire_value(neuron)
            for dst in sorted(neuron.out_neighbors):
                next_pending_signals.setdefault(dst, []).append((fire_value, neuron.stim_vec.copy()))
                edges_fired.append((node_id, dst))
            neuron.refractory_left = self.config.refractory_ticks
            neuron.recent_activity = min(2.0, neuron.recent_activity + 1.0)
            neuron.fatigue = min(4.0, neuron.fatigue + self.config.fatigue_gain)

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
        stability_streak = 0
        self.last_termination_reason = "max_ticks"

        while self.state.tick < self.config.max_ticks:
            self.run_tick(base_stim_vec, input_text)
            if self.state.tick < self.config.min_ticks_before_converged:
                continue
            if self._last_delta_k < self.config.delta_k_eps:
                self.last_termination_reason = "delta_k"
                break
            if len(self.state.branch_log) >= 2:
                if self._tick_is_stable(self.state.branch_log[-2], self.state.branch_log[-1]):
                    stability_streak += 1
                else:
                    stability_streak = 0
            if self.config.convergence_patience > 0 and stability_streak >= self.config.convergence_patience:
                self.last_termination_reason = "stable_convergence"
                break

        return base_stim_vec

    def prune_to_survivors(self) -> list[TickRecord]:
        self.pruned_branch_log = self.branch_extractor.prune_to_survivors(self.state.branch_log)
        return self.pruned_branch_log

    def extract_topk_branches(self) -> list[BranchPath]:
        source_log = self.state.branch_log
        if not source_log:
            self.prune_to_survivors()
            source_log = self.state.branch_log
        self.topk_branches = self.branch_extractor.extract_topk_branches_with_strategy(
            source_log,
            self.config.topk_branches,
            end_window=self.config.branch_end_window,
            length_bonus=self.config.branch_length_bonus,
        )
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
    ) -> np.ndarray:
        branch = dominant_branch or self.dominant_branch or self.build_dominant_branch()
        features: list[np.ndarray] = []
        for step in branch:
            tick_norm = float(step.tick) / float(self.config.max_ticks)
            k_scale = max(1.0, float(self.config.branch_k_feature_log_scale))
            k_feature = math.log1p(max(0.0, float(step.K))) / math.log1p(k_scale)
            k_feature = float(np.clip(k_feature, 0.0, 1.0))
            features.append(
                np.concatenate(
                    [
                        step.stim_vec.astype(np.float32, copy=False),
                        np.asarray([k_feature, tick_norm], dtype=np.float32),
                    ]
                )
            )

        if not features:
            features.append(np.zeros(BRANCH_FEATURE_DIM, dtype=np.float32))

        return np.stack(features, axis=0).astype(np.float32, copy=False)

    def encode_z(self, branch_tensor: Optional[Any] = None) -> Any:
        tensor = branch_tensor if branch_tensor is not None else self.dominant_branch_to_tensor()
        if self.use_torch_z_encoder:
            if self.z_encoder is None or not TORCH_AVAILABLE:
                raise RuntimeError("torch transformer encoder is not available")
            if not isinstance(tensor, torch.Tensor):
                tensor = torch.as_tensor(tensor, dtype=torch.float32)
            return self.z_encoder(tensor.float())

        if TORCH_AVAILABLE and isinstance(tensor, torch.Tensor):
            tensor = tensor.detach().cpu().numpy()
        return self.numpy_branch_encoder.encode(np.asarray(tensor, dtype=np.float32))

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
        return {
            "stim_vec": base_stim_vec,
            "pruned_branch_log": pruned_branch_log,
            "dominant_branch": dominant_branch,
            "branch_tensor": branch_tensor,
            "z": z,
            "ticks_run": int(self.state.tick),
            "termination_reason": str(self.last_termination_reason),
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

    def save_z_encoder(self, path: Optional[Path] = None) -> Path:
        if not TORCH_AVAILABLE or self.z_encoder is None:
            raise RuntimeError("torch transformer encoder is not available")
        target_path = Path(path) if path is not None else Path(self.config.z_encoder_path)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": self.z_encoder.state_dict(),
            "z_dim": int(self.config.z_dim),
            "d_model": int(self.config.d_model),
            "n_layers": int(self.config.n_layers),
            "n_heads": int(self.config.n_heads),
            "ff_dim": int(self.config.ff_dim),
            "dropout": float(self.config.dropout),
        }
        torch.save(payload, target_path)
        return target_path

    def load_z_encoder(self, path: Path, strict: bool = True) -> Path:
        if not TORCH_AVAILABLE or self.z_encoder is None:
            raise RuntimeError("torch transformer encoder is not available")
        model_path = Path(path)
        if not model_path.exists():
            raise FileNotFoundError(f"z encoder checkpoint not found: {model_path}")
        payload = torch.load(model_path, map_location="cpu")
        state_dict = payload["state_dict"] if isinstance(payload, dict) and "state_dict" in payload else payload
        self.z_encoder.load_state_dict(state_dict, strict=strict)
        return model_path

    def _apply_memory_sequence(self, neuron: NeuronState, text: str, effective_remem: float) -> None:
        if neuron.K > effective_remem:
            k_snapshot = self._compress_memory_k_snapshot(neuron.K)
            neuron.memories.append(
                MemoryItem(
                    stim_vec=neuron.stim_vec.copy(),
                    K_snapshot=float(k_snapshot),
                    input_text=text,
                    created_tick=self.state.tick,
                    strength=float(k_snapshot),
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
