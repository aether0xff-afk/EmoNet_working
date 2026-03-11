#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Emotion Z Pipeline
텍스트 -> 자극 인코더 -> 감정 동역학 네트워크 -> 히스토리 H -> GRU 기반 히스토리 인코더 -> z

핵심 목표:
- 지금 단계에서는 z까지만 구현
- 출력 말투/프롬프트는 아직 구현하지 않음
- 히스토리 인코더는 self-supervised summary/future prediction으로 학습

실행 예시:
    python emotion_z_pipeline.py train \
        --dataset_csv /mnt/data/dataset_for_regression.csv \
        --benchmark_csv /mnt/data/benchmark_results_20260305_180830.csv \
        --model_out /mnt/data/emotion_z_pipeline.pt \
        --history_train_samples 512 \
        --epochs 3

    python emotion_z_pipeline.py infer \
        --model_path /mnt/data/emotion_z_pipeline.pt \
        --text "왜 이렇게 일이 많지 너무 지친다"
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset


# =========================
# 공통 유틸
# =========================


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def clamp_nonnegative(vec: Sequence[float]) -> List[float]:
    return [max(0.0, float(v)) for v in vec]


def normalize(vec: Sequence[float], eps: float = 1e-8) -> List[float]:
    arr = clamp_nonnegative(vec)
    s = float(sum(arr))
    if s < eps:
        return [1.0 / max(1, len(arr)) for _ in arr]
    return [float(v / s) for v in arr]


def mean_vec(vec: Sequence[float]) -> float:
    return float(sum(vec) / max(1, len(vec)))


def cosine_similarity(a: Sequence[float], b: Sequence[float], eps: float = 1e-8) -> float:
    aa = float(sum(float(x) * float(x) for x in a))
    bb = float(sum(float(y) * float(y) for y in b))
    if aa < eps or bb < eps:
        return 0.0
    ab = float(sum(float(x) * float(y) for x, y in zip(a, b)))
    return ab / math.sqrt(aa * bb)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# =========================
# 텍스트 회귀 / 자극 인코더
# =========================


class SafeTruncatedSVD:
    def __init__(self, n_components: int = 300, random_state: int = 42):
        self.n_components = int(n_components)
        self.random_state = int(random_state)
        self._svd: Optional[TruncatedSVD] = None

    def fit(self, X, y=None):
        n_features = int(X.shape[1])
        safe = min(self.n_components, max(1, n_features - 1))
        self._svd = TruncatedSVD(n_components=safe, random_state=self.random_state)
        self._svd.fit(X)
        return self

    def transform(self, X):
        if self._svd is None:
            raise RuntimeError("SafeTruncatedSVD is not fitted.")
        return self._svd.transform(X)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)


@dataclass
class StimulusModelConfig:
    vector_name: str = "char_tfidf"
    vector_kind: str = "char"  # word or char
    use_svd: bool = False
    svd_dim: int = 300
    ridge_alpha: float = 2.0
    random_state: int = 42


class BestRidgeStimulusEncoder:
    APPRAISAL_NAMES = [
        "pleasantness",
        "goal_gain",
        "goal_loss",
        "novelty",
        "threat",
        "social_safety",
        "agency",
        "fatigue_pressure",
    ]

    HORMONE_NAMES = ["dopamine", "serotonin", "norepinephrine", "melatonin"]

    POSITIVE_HINTS = {"고마", "감사", "행복", "좋", "기쁘", "다행", "성공", "축하", "신나", "웃"}
    NEGATIVE_HINTS = {"화", "짜증", "불안", "걱정", "무섭", "지쳤", "슬프", "외롭", "실패", "억울", "답답"}
    THREAT_HINTS = {"위험", "압박", "무섭", "불안", "위협", "긴장", "떨", "불확실", "망하", "큰일"}
    SAFETY_HINTS = {"괜찮", "편안", "안전", "믿", "도와", "함께", "응원", "수용", "다정"}
    FATIGUE_HINTS = {"피곤", "지쳤", "졸", "힘들", "번아웃", "소진", "과제", "바쁘", "부담"}
    AGENCY_HINTS = {"해냈", "할 수", "결정", "주도", "통제", "선택", "직접", "이겼"}
    LOSS_HINTS = {"잃", "실패", "망했", "못", "박탈", "빼앗", "끝났", "없다"}
    GAIN_HINTS = {"얻", "성공", "합격", "좋아졌", "늘었", "보상", "해냈", "기회"}

    def __init__(self, config: Optional[StimulusModelConfig] = None):
        self.config = config or StimulusModelConfig()
        self.pipeline: Optional[Pipeline] = None
        self.label_map: Optional[pd.DataFrame] = None

    @staticmethod
    def _make_vectorizer(kind: str) -> TfidfVectorizer:
        if kind == "word":
            return TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, max_df=0.95, dtype=np.float32)
        if kind == "char":
            return TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, max_df=0.95, dtype=np.float32)
        raise ValueError("vector_kind must be 'word' or 'char'")

    @classmethod
    def choose_from_benchmark(
        cls,
        benchmark_csv: Optional[Path],
        prefer_model: str = "Ridge",
        random_state: int = 42,
    ) -> StimulusModelConfig:
        default = StimulusModelConfig(random_state=random_state)
        if benchmark_csv is None or not benchmark_csv.exists():
            return default

        df = pd.read_csv(benchmark_csv)
        if "status" in df.columns:
            df = df[df["status"] == "ok"].copy()
        if len(df) == 0:
            return default

        if prefer_model and "model" in df.columns:
            prefer = df[df["model"] == prefer_model].copy()
            if len(prefer) > 0:
                df = prefer

        if "RMSE(mean)" in df.columns and "MAE(mean)" in df.columns:
            df = df.sort_values(["RMSE(mean)", "MAE(mean)"], ascending=[True, True]).reset_index(drop=True)
        best = df.iloc[0]
        vector_name = str(best.get("vector", "char_tfidf"))
        use_svd = "svd" in vector_name.lower()
        vector_kind = "char" if "char" in vector_name.lower() else "word"
        svd_dim = 300
        if use_svd:
            digits = "".join(ch for ch in vector_name if ch.isdigit())
            if digits:
                svd_dim = int(digits)
        return StimulusModelConfig(
            vector_name=vector_name,
            vector_kind=vector_kind,
            use_svd=use_svd,
            svd_dim=svd_dim,
            ridge_alpha=2.0,
            random_state=random_state,
        )

    def _build_pipeline(self) -> Pipeline:
        steps: List[Tuple[str, Any]] = [("tfidf", self._make_vectorizer(self.config.vector_kind))]
        if self.config.use_svd:
            steps.append(("svd", SafeTruncatedSVD(self.config.svd_dim, self.config.random_state)))
        steps.append(("model", Ridge(alpha=self.config.ridge_alpha, random_state=self.config.random_state)))
        return Pipeline(steps)

    def fit(self, dataset_csv: Path, label_map_csv: Optional[Path] = None, max_samples: Optional[int] = None) -> None:
        df = pd.read_csv(dataset_csv)
        for col in ["text", "y"]:
            if col not in df.columns:
                raise ValueError(f"'{col}' column not found in {dataset_csv}")
        if max_samples is not None and max_samples > 0 and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=self.config.random_state).reset_index(drop=True)
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(df["text"].astype(str).to_numpy(), df["y"].astype(float).to_numpy())
        if label_map_csv is not None and label_map_csv.exists():
            self.label_map = pd.read_csv(label_map_csv)

    def predict_score(self, text: str) -> float:
        if self.pipeline is None:
            raise RuntimeError("Stimulus encoder is not fitted.")
        score = float(self.pipeline.predict([str(text)])[0])
        return clamp(score, 0.0, 1.0)

    @staticmethod
    def _hint_count(text: str, hints: Iterable[str]) -> float:
        s = str(text)
        return float(sum(1 for token in hints if token in s))

    def score_to_appraisal(self, text: str, score: float) -> Dict[str, float]:
        s = str(text)
        pos = self._hint_count(s, self.POSITIVE_HINTS)
        neg = self._hint_count(s, self.NEGATIVE_HINTS)
        threat_hint = self._hint_count(s, self.THREAT_HINTS)
        safety_hint = self._hint_count(s, self.SAFETY_HINTS)
        fatigue_hint = self._hint_count(s, self.FATIGUE_HINTS)
        agency_hint = self._hint_count(s, self.AGENCY_HINTS)
        loss_hint = self._hint_count(s, self.LOSS_HINTS)
        gain_hint = self._hint_count(s, self.GAIN_HINTS)
        punctuation = float(s.count("!") + s.count("?") + s.count("…") + s.count("..."))
        length_pressure = min(3.0, len(s) / 80.0)

        pleasantness = clamp(0.55 * score + 0.08 * pos - 0.10 * neg, 0.0, 1.0)
        goal_gain = clamp(0.45 * score + 0.10 * gain_hint + 0.05 * pos, 0.0, 1.0)
        goal_loss = clamp(0.55 * (1.0 - score) + 0.10 * loss_hint + 0.06 * neg, 0.0, 1.0)
        novelty = clamp(0.18 + 0.08 * punctuation + 0.03 * len(set(s)) / max(1, len(s)), 0.0, 1.0)
        threat = clamp(0.35 * goal_loss + 0.18 * threat_hint + 0.06 * punctuation, 0.0, 1.0)
        social_safety = clamp(0.25 + 0.30 * pleasantness + 0.10 * safety_hint - 0.12 * threat, 0.0, 1.0)
        agency = clamp(0.25 + 0.18 * pleasantness + 0.12 * agency_hint - 0.10 * goal_loss, 0.0, 1.0)
        fatigue_pressure = clamp(0.18 + 0.16 * fatigue_hint + 0.10 * length_pressure + 0.08 * neg, 0.0, 1.0)

        values = [
            pleasantness,
            goal_gain,
            goal_loss,
            novelty,
            threat,
            social_safety,
            agency,
            fatigue_pressure,
        ]
        return {name: float(v) for name, v in zip(self.APPRAISAL_NAMES, values)}

    @staticmethod
    def appraisal_to_hormone(appraisal: Dict[str, float]) -> List[float]:
        pleasantness = appraisal["pleasantness"]
        goal_gain = appraisal["goal_gain"]
        goal_loss = appraisal["goal_loss"]
        novelty = appraisal["novelty"]
        threat = appraisal["threat"]
        social_safety = appraisal["social_safety"]
        agency = appraisal["agency"]
        fatigue_pressure = appraisal["fatigue_pressure"]

        dopamine = clamp(0.40 * pleasantness + 0.32 * goal_gain + 0.20 * agency + 0.08 * novelty, 0.0, 1.0)
        serotonin = clamp(0.45 * social_safety + 0.25 * pleasantness + 0.20 * (1.0 - threat) + 0.10 * (1.0 - fatigue_pressure), 0.0, 1.0)
        norepinephrine = clamp(0.50 * threat + 0.20 * novelty + 0.15 * goal_loss + 0.15 * (1.0 - social_safety), 0.0, 1.0)
        melatonin = clamp(0.60 * fatigue_pressure + 0.20 * (1.0 - agency) + 0.20 * (1.0 - pleasantness), 0.0, 1.0)
        return [dopamine, serotonin, norepinephrine, melatonin]

    def encode_text(self, text: str) -> Dict[str, Any]:
        score = self.predict_score(text)
        appraisal = self.score_to_appraisal(text, score)
        hormone = self.appraisal_to_hormone(appraisal)
        return {
            "score": score,
            "appraisal": appraisal,
            "u": [appraisal[name] for name in self.APPRAISAL_NAMES],
            "h": hormone,
        }


# =========================
# 감정 동역학 네트워크
# =========================


@dataclass
class MemoryTrace:
    emo_vec: List[float]
    timestep: int
    membrane_potential: float


@dataclass
class NeuronState:
    neuron_id: int
    neuron_type: str
    v_th: float
    v_mem: float
    gain: float = 1.0
    memory: List[MemoryTrace] = field(default_factory=list)
    recent_spikes: List[int] = field(default_factory=list)
    last_rewire_timestep: int = -10**9
    suppressed_until: int = -1
    preferred_emo: List[float] = field(default_factory=list)
    total_fires: int = 0


@dataclass
class DynamicsConfig:
    seed: int = 42
    n_exc: int = 16
    n_inh: int = 8
    n_mod: int = 4
    connect_prob: float = 0.18
    steps: int = 24
    epsilon: float = 1e-3
    stable_steps_required: int = 3
    min_steps: int = 6

    lambda_decay: float = 0.10
    w_now: float = 0.70
    w_mem: float = 0.30
    memory_prune_threshold: float = 0.05
    max_memory_size: int = 100
    age_prune_steps: int = 20

    target_rate: float = 0.20
    eta_homeo: float = 0.01
    homeo_window: int = 20

    rewiring_cooldown: int = 5
    k_remove: float = 0.20
    k_new: float = 0.10
    max_new_synapses: int = 2
    max_out_degree: int = 20

    k_drop: float = 0.05
    max_dropout: float = 0.03
    k_ne: float = 0.05
    delta_mem: float = 0.05


class EmotionDynamicsNet:
    def __init__(self, config: Optional[DynamicsConfig] = None):
        self.config = config or DynamicsConfig()
        self.rng = random.Random(self.config.seed)
        self.n_total = self.config.n_exc + self.config.n_inh + self.config.n_mod
        self.neurons: List[NeuronState] = []
        self.weights: List[List[float]] = []
        self._reset_network()

    def _reset_network(self) -> None:
        self.neurons = []
        idx = 0
        for _ in range(self.config.n_exc):
            self.neurons.append(self._make_neuron(idx, "exc"))
            idx += 1
        for _ in range(self.config.n_inh):
            self.neurons.append(self._make_neuron(idx, "inh"))
            idx += 1
        for _ in range(self.config.n_mod):
            self.neurons.append(self._make_neuron(idx, "mod"))
            idx += 1

        self.weights = [[0.0 for _ in range(self.n_total)] for _ in range(self.n_total)]
        for i in range(self.n_total):
            for j in range(self.n_total):
                if i == j:
                    continue
                if self.rng.random() < self.config.connect_prob:
                    self.weights[i][j] = self.rng.uniform(0.1, 1.0)

    def _make_neuron(self, neuron_id: int, neuron_type: str) -> NeuronState:
        base_emo = normalize(np.random.default_rng(self.config.seed + neuron_id).random(4).tolist())
        return NeuronState(
            neuron_id=neuron_id,
            neuron_type=neuron_type,
            v_th=self.rng.uniform(0.55, 0.95),
            v_mem=self.rng.uniform(0.45, 0.80),
            preferred_emo=base_emo,
        )

    def clone_fresh(self) -> "EmotionDynamicsNet":
        return EmotionDynamicsNet(DynamicsConfig(**asdict(self.config)))

    def out_degree(self, i: int) -> int:
        return sum(1 for w in self.weights[i] if w > 0.0)

    def _prune_memory(self, neuron: NeuronState, timestep: int) -> None:
        kept: List[MemoryTrace] = []
        for mem in neuron.memory:
            delta_t = timestep - mem.timestep
            strength = mem.membrane_potential * math.exp(-self.config.lambda_decay * delta_t)
            if delta_t <= self.config.age_prune_steps and strength >= self.config.memory_prune_threshold:
                kept.append(mem)
        neuron.memory = kept[-self.config.max_memory_size :]

    def _memory_sum(self, neuron: NeuronState, timestep: int) -> List[float]:
        total = [0.0, 0.0, 0.0, 0.0]
        for mem in neuron.memory:
            delta_t = timestep - mem.timestep
            strength = mem.membrane_potential * math.exp(-self.config.lambda_decay * delta_t)
            for i, val in enumerate(mem.emo_vec):
                total[i] += float(val) * strength
        return total

    def _combine_emotion(self, current: List[float], memory_sum: List[float]) -> List[float]:
        mixed = [
            self.config.w_now * float(a) + self.config.w_mem * float(b)
            for a, b in zip(current, memory_sum)
        ]
        return normalize(mixed)

    @staticmethod
    def _flatten_emotion(E: List[float], serotonin: float) -> List[float]:
        m = mean_vec(E)
        alpha = min(0.8, 0.6 * serotonin)
        return normalize([e + alpha * (m - e) for e in E])

    @staticmethod
    def _sharpen_emotion(E: List[float], dopamine: float) -> List[float]:
        m = mean_vec(E)
        beta = min(1.0, 0.8 * dopamine)
        return normalize([max(0.0, e + beta * (e - m)) for e in E])

    @staticmethod
    def _dominant_emotion(E: List[float]) -> bool:
        total = float(sum(E))
        if total <= 1e-8:
            return False
        e_max = max(E)
        mean = total / len(E)
        return (e_max / total >= 0.5) and (e_max >= 1.8 * mean)

    def _can_rewire(self, neuron: NeuronState, timestep: int) -> bool:
        return (timestep - neuron.last_rewire_timestep) >= self.config.rewiring_cooldown

    def _try_rewire_exc(self, idx: int, dopamine: float, timestep: int) -> int:
        neuron = self.neurons[idx]
        if not self._can_rewire(neuron, timestep):
            return 0
        num_new = int(self.config.max_new_synapses * self.config.k_new * dopamine)
        added = 0
        for _ in range(max(0, num_new)):
            if self.out_degree(idx) >= self.config.max_out_degree:
                break
            target = self.rng.randrange(self.n_total)
            if target != idx and self.weights[idx][target] == 0.0:
                self.weights[idx][target] = self.rng.uniform(0.1, 1.0)
                added += 1
        if added > 0:
            neuron.last_rewire_timestep = timestep
        return added

    def _try_rewire_inh(self, idx: int, serotonin: float, timestep: int) -> int:
        neuron = self.neurons[idx]
        if not self._can_rewire(neuron, timestep):
            return 0
        current_targets = [j for j, w in enumerate(self.weights[idx]) if w > 0.0]
        if not current_targets:
            return 0
        num_remove = int(len(current_targets) * self.config.k_remove * serotonin)
        if num_remove <= 0:
            return 0
        targets = self.rng.sample(current_targets, min(num_remove, len(current_targets)))
        for j in targets:
            self.weights[idx][j] = 0.0
        neuron.last_rewire_timestep = timestep
        return len(targets)

    def _apply_modulation(self, E: List[float], timestep: int) -> int:
        dopamine, serotonin, norepinephrine, melatonin = E
        drop_ratio = min(self.config.max_dropout, self.config.k_drop * melatonin)
        num_drop = int(self.n_total * drop_ratio)
        suppressed = 0
        if num_drop > 0:
            candidates = [n for n in self.neurons if n.neuron_type != "mod"]
            chosen = self.rng.sample(candidates, min(num_drop, len(candidates)))
            for neuron in chosen:
                neuron.suppressed_until = timestep
                suppressed += 1

        delta_th = self.config.k_ne * norepinephrine
        for neuron in self.neurons:
            neuron.v_th = clamp(neuron.v_th - delta_th, 0.10, 5.0)
            if self._dominant_emotion(E):
                neuron.v_mem = clamp(neuron.v_mem - self.config.delta_mem, 0.10, 5.0)
        return suppressed

    def _compute_membrane_potential(
        self,
        idx: int,
        prev_outputs: List[float],
        E: List[float],
        u: List[float],
        h: List[float],
    ) -> float:
        neuron = self.neurons[idx]
        incoming = 0.0
        for j in range(self.n_total):
            incoming += self.weights[j][idx] * prev_outputs[j]
        emo_drive = cosine_similarity(E, neuron.preferred_emo)
        stim_drive = 0.45 * mean_vec(u) + 0.25 * mean_vec(h)
        type_bias = 0.0
        if neuron.neuron_type == "exc":
            type_bias = 0.08 * h[0]
        elif neuron.neuron_type == "inh":
            type_bias = 0.08 * h[1]
        else:
            type_bias = 0.08 * (h[2] - h[3])
        v_in = neuron.gain * (0.60 * incoming + 0.25 * emo_drive + 0.15 * stim_drive + type_bias)
        return float(v_in)

    def _update_homeostasis(self, neuron: NeuronState) -> None:
        recent = neuron.recent_spikes[-self.config.homeo_window :]
        if len(recent) == 0:
            return
        rate = float(sum(recent) / len(recent))
        if rate > self.config.target_rate:
            neuron.v_th += self.config.eta_homeo
        elif rate < self.config.target_rate:
            neuron.v_th -= self.config.eta_homeo
        neuron.v_th = clamp(neuron.v_th, 0.10, 5.0)

    def run(self, u: List[float], h: List[float]) -> Dict[str, Any]:
        self._reset_network()
        E = normalize(
            [
                0.30 * u[1] + 0.20 * u[0],
                0.25 * u[5] + 0.15 * (1.0 - u[4]),
                0.35 * u[4] + 0.15 * u[3],
                0.35 * u[7] + 0.10 * (1.0 - u[6]),
            ]
        )
        prev_outputs = [0.0 for _ in range(self.n_total)]
        history_rows: List[List[float]] = []
        log_rows: List[Dict[str, float]] = []
        stable_count = 0
        prev_summary = None

        for t in range(self.config.steps):
            suppressed = self._apply_modulation(E, t)
            outputs: List[float] = []
            per_neuron_emotions: List[List[float]] = []
            rewires_add = 0
            rewires_remove = 0
            exc_spikes = 0
            inh_spikes = 0
            mod_spikes = 0

            for i, neuron in enumerate(self.neurons):
                self._prune_memory(neuron, t)
                if t <= neuron.suppressed_until:
                    neuron.recent_spikes.append(0)
                    outputs.append(0.0)
                    per_neuron_emotions.append(E[:])
                    self._update_homeostasis(neuron)
                    continue

                v_in = self._compute_membrane_potential(i, prev_outputs, E, u, h)
                fired = float(v_in > neuron.v_th)
                outputs.append(fired)
                neuron.recent_spikes.append(int(fired))
                neuron.recent_spikes = neuron.recent_spikes[-self.config.homeo_window :]

                E_local = E[:]
                if fired > 0.0:
                    neuron.total_fires += 1
                    mem_sum = self._memory_sum(neuron, t)
                    E_local = self._combine_emotion(E, mem_sum)
                    if v_in > neuron.v_mem:
                        neuron.memory.append(MemoryTrace(E_local[:], t, v_in))
                        neuron.memory = neuron.memory[-self.config.max_memory_size :]

                    dopamine, serotonin, norepinephrine, melatonin = E_local
                    if neuron.neuron_type == "exc":
                        exc_spikes += 1
                        E_local = self._sharpen_emotion(E_local, dopamine)
                        rewires_add += self._try_rewire_exc(i, dopamine, t)
                    elif neuron.neuron_type == "inh":
                        inh_spikes += 1
                        E_local = self._flatten_emotion(E_local, serotonin)
                        rewires_remove += self._try_rewire_inh(i, serotonin, t)
                    else:
                        mod_spikes += 1
                per_neuron_emotions.append(E_local)
                self._update_homeostasis(neuron)

            E_next = [0.0, 0.0, 0.0, 0.0]
            for e_local in per_neuron_emotions:
                for j in range(4):
                    E_next[j] += e_local[j]
            E_next = normalize([v / max(1, len(per_neuron_emotions)) for v in E_next])

            firing_rate = float(sum(outputs) / max(1, self.n_total))
            avg_threshold = float(sum(n.v_th for n in self.neurons) / self.n_total)
            avg_memory_size = float(sum(len(n.memory) for n in self.neurons) / self.n_total)
            avg_gain = float(sum(n.gain for n in self.neurons) / self.n_total)
            conflict = float(np.std(E_next))
            exc_ratio = float(exc_spikes / max(1, self.config.n_exc))
            inh_ratio = float(inh_spikes / max(1, self.config.n_inh))
            mod_ratio = float(mod_spikes / max(1, self.config.n_mod))
            rewires_add_ratio = float(rewires_add / max(1, self.n_total))
            rewires_remove_ratio = float(rewires_remove / max(1, self.n_total))
            suppressed_ratio = float(suppressed / max(1, self.n_total))
            step_energy = float(sum(abs(a - b) for a, b in zip(E_next, E)))

            row = [
                *E_next,
                firing_rate,
                avg_threshold,
                avg_memory_size,
                rewires_add_ratio,
                rewires_remove_ratio,
                suppressed_ratio,
                conflict,
                exc_ratio,
                inh_ratio,
                mod_ratio,
                avg_gain,
                step_energy,
            ]
            history_rows.append(row)
            log_rows.append(
                {
                    "timestep": float(t),
                    "firing_rate": firing_rate,
                    "avg_threshold": avg_threshold,
                    "avg_memory_size": avg_memory_size,
                    "rewires_add_ratio": rewires_add_ratio,
                    "rewires_remove_ratio": rewires_remove_ratio,
                    "suppressed_ratio": suppressed_ratio,
                    "conflict": conflict,
                    "step_energy": step_energy,
                }
            )

            current_summary = np.array(row, dtype=np.float32)
            if prev_summary is not None:
                delta = float(np.mean(np.abs(current_summary - prev_summary)))
                if delta < self.config.epsilon or firing_rate < 0.01:
                    stable_count += 1
                else:
                    stable_count = 0
            prev_summary = current_summary
            prev_outputs = outputs
            E = E_next

            if t + 1 >= self.config.min_steps and stable_count >= self.config.stable_steps_required:
                break

        H = np.asarray(history_rows, dtype=np.float32)
        return {"H": H, "logs": log_rows, "final_emo": E[:], "d_h": H.shape[1] if H.size else 0}


# =========================
# H -> z 데이터셋
# =========================


class PrefixHistoryDataset(Dataset):
    def __init__(self, items: List[Dict[str, np.ndarray]]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        return self.items[idx]


class PrefixDatasetBuilder:
    def __init__(self, future_horizon: int = 3, min_prefix_len: int = 4, prefix_stride: int = 2):
        self.future_horizon = int(future_horizon)
        self.min_prefix_len = int(min_prefix_len)
        self.prefix_stride = int(prefix_stride)

    @staticmethod
    def make_summary_target(H: np.ndarray) -> np.ndarray:
        final_emo = H[-1, 0:4]
        mean_firing = np.array([float(np.mean(H[:, 4]))], dtype=np.float32)
        mean_threshold = np.array([float(np.mean(H[:, 5]))], dtype=np.float32)
        mean_memory = np.array([float(np.mean(H[:, 6]))], dtype=np.float32)
        mean_conflict = np.array([float(np.mean(H[:, 10]))], dtype=np.float32)
        mean_energy = np.array([float(np.mean(H[:, 15]))], dtype=np.float32)
        seq_len_norm = np.array([float(min(1.0, len(H) / 30.0))], dtype=np.float32)
        return np.concatenate(
            [final_emo, mean_firing, mean_threshold, mean_memory, mean_conflict, mean_energy, seq_len_norm],
            axis=0,
        ).astype(np.float32)

    def build(self, histories: Sequence[np.ndarray]) -> PrefixHistoryDataset:
        items: List[Dict[str, np.ndarray]] = []
        for H in histories:
            if len(H) < 2:
                continue
            summary_target = self.make_summary_target(H)
            start = min(self.min_prefix_len, max(1, len(H) - 1))
            added = 0
            for prefix_end in range(start, len(H), self.prefix_stride):
                prefix = H[:prefix_end].astype(np.float32)
                future = H[prefix_end : prefix_end + self.future_horizon]
                if len(future) == 0:
                    continue
                future_target = future.mean(axis=0).astype(np.float32)
                items.append(
                    {
                        "prefix": prefix,
                        "length": np.array([len(prefix)], dtype=np.int64),
                        "summary_target": summary_target,
                        "future_target": future_target,
                    }
                )
                added += 1
            if added == 0:
                prefix = H[:-1].astype(np.float32)
                future_target = H[-1].astype(np.float32)
                items.append(
                    {
                        "prefix": prefix,
                        "length": np.array([len(prefix)], dtype=np.int64),
                        "summary_target": summary_target,
                        "future_target": future_target,
                    }
                )
        return PrefixHistoryDataset(items)


# =========================
# H -> z 모델
# =========================


class HistoryEncoderModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, z_dim: int = 16, summary_dim: int = 10):
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.z_dim = int(z_dim)
        self.summary_dim = int(summary_dim)

        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.to_z = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, z_dim),
        )
        self.summary_head = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, summary_dim),
        )
        self.future_head = nn.Sequential(
            nn.Linear(z_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def encode(self, padded_seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = pack_padded_sequence(padded_seq, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.gru(packed)
        h_last = h_n[-1]
        z = self.to_z(h_last)
        return z

    def forward(self, padded_seq: torch.Tensor, lengths: torch.Tensor) -> Dict[str, torch.Tensor]:
        z = self.encode(padded_seq, lengths)
        summary_hat = self.summary_head(z)
        future_hat = self.future_head(z)
        return {"z": z, "summary_hat": summary_hat, "future_hat": future_hat}


@dataclass
class HistoryEncoderTrainConfig:
    z_dim: int = 16
    hidden_dim: int = 64
    batch_size: int = 32
    lr: float = 1e-3
    epochs: int = 5
    alpha_future: float = 0.5
    device: str = "cpu"


class HistoryEncoderTrainer:
    def __init__(self, config: Optional[HistoryEncoderTrainConfig] = None):
        self.config = config or HistoryEncoderTrainConfig()
        self.model: Optional[HistoryEncoderModel] = None
        self.input_dim: Optional[int] = None
        self.summary_dim: Optional[int] = None
        self.train_history: List[Dict[str, float]] = []

    @staticmethod
    def _collate(batch: Sequence[Dict[str, np.ndarray]]) -> Dict[str, torch.Tensor]:
        lengths = torch.tensor([int(item["length"][0]) for item in batch], dtype=torch.long)
        input_dim = batch[0]["prefix"].shape[1]
        max_len = int(max(lengths).item())
        padded = torch.zeros((len(batch), max_len, input_dim), dtype=torch.float32)
        summary_target = torch.tensor(np.stack([item["summary_target"] for item in batch]), dtype=torch.float32)
        future_target = torch.tensor(np.stack([item["future_target"] for item in batch]), dtype=torch.float32)
        for i, item in enumerate(batch):
            seq = torch.tensor(item["prefix"], dtype=torch.float32)
            padded[i, : seq.shape[0]] = seq
        return {"prefix": padded, "lengths": lengths, "summary_target": summary_target, "future_target": future_target}

    def fit(self, dataset: PrefixHistoryDataset) -> None:
        if len(dataset) == 0:
            raise ValueError("History dataset is empty.")
        sample = dataset[0]
        self.input_dim = int(sample["prefix"].shape[1])
        self.summary_dim = int(sample["summary_target"].shape[0])
        self.model = HistoryEncoderModel(
            input_dim=self.input_dim,
            hidden_dim=self.config.hidden_dim,
            z_dim=self.config.z_dim,
            summary_dim=self.summary_dim,
        )
        device = torch.device(self.config.device)
        self.model.to(device)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True, collate_fn=self._collate)
        optim = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)
        mse = nn.MSELoss()

        self.train_history = []
        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            total_loss = 0.0
            total_summary = 0.0
            total_future = 0.0
            n_batches = 0
            for batch in loader:
                prefix = batch["prefix"].to(device)
                lengths = batch["lengths"].to(device)
                summary_target = batch["summary_target"].to(device)
                future_target = batch["future_target"].to(device)

                out = self.model(prefix, lengths)
                loss_summary = mse(out["summary_hat"], summary_target)
                loss_future = mse(out["future_hat"], future_target)
                loss = loss_summary + self.config.alpha_future * loss_future

                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optim.step()

                total_loss += float(loss.item())
                total_summary += float(loss_summary.item())
                total_future += float(loss_future.item())
                n_batches += 1

            row = {
                "epoch": float(epoch),
                "loss": total_loss / max(1, n_batches),
                "summary_loss": total_summary / max(1, n_batches),
                "future_loss": total_future / max(1, n_batches),
            }
            self.train_history.append(row)
            print(
                f"[HistoryEncoder] epoch {epoch}/{self.config.epochs} "
                f"loss={row['loss']:.4f} summary={row['summary_loss']:.4f} future={row['future_loss']:.4f}"
            )

    def encode(self, H: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("History encoder is not fitted.")
        device = torch.device(self.config.device)
        self.model.eval()
        with torch.no_grad():
            seq = torch.tensor(H[None, :, :], dtype=torch.float32, device=device)
            lengths = torch.tensor([H.shape[0]], dtype=torch.long, device=device)
            z = self.model.encode(seq, lengths).cpu().numpy()[0]
        return z.astype(np.float32)


# =========================
# 전체 파이프라인
# =========================


@dataclass
class PipelineTrainConfig:
    seed: int = 42
    stimulus_max_samples: int = 30000
    history_train_samples: int = 512
    future_horizon: int = 3
    min_prefix_len: int = 4
    prefix_stride: int = 2
    z_dim: int = 16
    hidden_dim: int = 64
    batch_size: int = 32
    lr: float = 1e-3
    epochs: int = 5
    alpha_future: float = 0.5
    device: str = "cpu"


class EmotionZPipeline:
    def __init__(self, seed: int = 42):
        self.seed = int(seed)
        seed_everything(self.seed)
        self.stimulus_encoder: Optional[BestRidgeStimulusEncoder] = None
        self.dynamics_config = DynamicsConfig(seed=self.seed)
        self.history_trainer: Optional[HistoryEncoderTrainer] = None
        self.metadata: Dict[str, Any] = {}

    def fit(
        self,
        dataset_csv: Path,
        benchmark_csv: Optional[Path],
        label_map_csv: Optional[Path],
        config: PipelineTrainConfig,
    ) -> Dict[str, Any]:
        seed_everything(config.seed)
        stim_cfg = BestRidgeStimulusEncoder.choose_from_benchmark(benchmark_csv, prefer_model="Ridge", random_state=config.seed)
        self.stimulus_encoder = BestRidgeStimulusEncoder(stim_cfg)
        self.stimulus_encoder.fit(dataset_csv=dataset_csv, label_map_csv=label_map_csv, max_samples=config.stimulus_max_samples)

        df = pd.read_csv(dataset_csv)
        if len(df) > config.history_train_samples:
            df = df.sample(n=config.history_train_samples, random_state=config.seed).reset_index(drop=True)
        texts = df["text"].astype(str).tolist()

        simulator = EmotionDynamicsNet(self.dynamics_config)
        histories: List[np.ndarray] = []
        for idx, text in enumerate(texts, start=1):
            encoded = self.stimulus_encoder.encode_text(text)
            run = simulator.clone_fresh().run(encoded["u"], encoded["h"])
            H = run["H"]
            if len(H) > 0:
                histories.append(H)
            if idx % 100 == 0 or idx == len(texts):
                print(f"[Dynamics] generated histories: {idx}/{len(texts)}")

        builder = PrefixDatasetBuilder(
            future_horizon=config.future_horizon,
            min_prefix_len=config.min_prefix_len,
            prefix_stride=config.prefix_stride,
        )
        prefix_dataset = builder.build(histories)
        print(f"[HistoryDataset] sequences={len(histories)} prefixes={len(prefix_dataset)}")

        hist_cfg = HistoryEncoderTrainConfig(
            z_dim=config.z_dim,
            hidden_dim=config.hidden_dim,
            batch_size=config.batch_size,
            lr=config.lr,
            epochs=config.epochs,
            alpha_future=config.alpha_future,
            device=config.device,
        )
        self.history_trainer = HistoryEncoderTrainer(hist_cfg)
        self.history_trainer.fit(prefix_dataset)

        self.metadata = {
            "seed": config.seed,
            "stimulus_model": asdict(self.stimulus_encoder.config),
            "dynamics_config": asdict(self.dynamics_config),
            "history_encoder_config": asdict(hist_cfg),
            "history_dataset_size": len(prefix_dataset),
            "history_sequence_count": len(histories),
            "history_input_dim": int(histories[0].shape[1]) if histories else 0,
            "summary_dim": int(builder.make_summary_target(histories[0]).shape[0]) if histories else 0,
        }
        return self.metadata

    def encode_text(self, text: str) -> Dict[str, Any]:
        if self.stimulus_encoder is None or self.history_trainer is None:
            raise RuntimeError("Pipeline is not fitted.")
        encoded = self.stimulus_encoder.encode_text(text)
        simulator = EmotionDynamicsNet(self.dynamics_config)
        run = simulator.run(encoded["u"], encoded["h"])
        H = run["H"]
        z = self.history_trainer.encode(H)
        return {
            "text": text,
            "score": encoded["score"],
            "appraisal": encoded["appraisal"],
            "u": encoded["u"],
            "h": encoded["h"],
            "H_shape": list(H.shape),
            "H": H,
            "z": z,
            "final_emo": run["final_emo"],
            "logs": run["logs"],
        }

    def save(self, path: Path) -> None:
        if self.stimulus_encoder is None or self.stimulus_encoder.pipeline is None or self.history_trainer is None or self.history_trainer.model is None:
            raise RuntimeError("Nothing to save. Fit the pipeline first.")
        payload = {
            "seed": self.seed,
            "metadata": self.metadata,
            "stimulus_config": asdict(self.stimulus_encoder.config),
            "stimulus_pipeline": self.stimulus_encoder.pipeline,
            "label_map": self.stimulus_encoder.label_map,
            "dynamics_config": asdict(self.dynamics_config),
            "history_train_config": asdict(self.history_trainer.config),
            "history_model_state_dict": self.history_trainer.model.state_dict(),
            "history_input_dim": self.history_trainer.input_dim,
            "history_summary_dim": self.history_trainer.summary_dim,
            "history_train_history": self.history_trainer.train_history,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: Path, map_location: str = "cpu") -> "EmotionZPipeline":
        payload = torch.load(path, map_location=map_location, weights_only=False)
        pipeline = cls(seed=int(payload["seed"]))
        stim_cfg = StimulusModelConfig(**payload["stimulus_config"])
        pipeline.stimulus_encoder = BestRidgeStimulusEncoder(stim_cfg)
        pipeline.stimulus_encoder.pipeline = payload["stimulus_pipeline"]
        pipeline.stimulus_encoder.label_map = payload.get("label_map")
        pipeline.dynamics_config = DynamicsConfig(**payload["dynamics_config"])

        hist_cfg = HistoryEncoderTrainConfig(**payload["history_train_config"])
        trainer = HistoryEncoderTrainer(hist_cfg)
        trainer.input_dim = int(payload["history_input_dim"])
        trainer.summary_dim = int(payload["history_summary_dim"])
        trainer.model = HistoryEncoderModel(
            input_dim=trainer.input_dim,
            hidden_dim=hist_cfg.hidden_dim,
            z_dim=hist_cfg.z_dim,
            summary_dim=trainer.summary_dim,
        )
        trainer.model.load_state_dict(payload["history_model_state_dict"])
        trainer.model.to(torch.device(hist_cfg.device))
        trainer.train_history = payload.get("history_train_history", [])
        pipeline.history_trainer = trainer
        pipeline.metadata = payload.get("metadata", {})
        return pipeline


# =========================
# CLI
# =========================


def write_json(path: Path, data: Dict[str, Any]) -> None:
    def convert(obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    path.write_text(json.dumps(convert(data), ensure_ascii=False, indent=2), encoding="utf-8")


def train_command(args: argparse.Namespace) -> None:
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = PipelineTrainConfig(
        seed=args.seed,
        stimulus_max_samples=args.stimulus_max_samples,
        history_train_samples=args.history_train_samples,
        future_horizon=args.future_horizon,
        min_prefix_len=args.min_prefix_len,
        prefix_stride=args.prefix_stride,
        z_dim=args.z_dim,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        alpha_future=args.alpha_future,
        device=device,
    )
    pipeline = EmotionZPipeline(seed=args.seed)
    meta = pipeline.fit(
        dataset_csv=Path(args.dataset_csv),
        benchmark_csv=Path(args.benchmark_csv) if args.benchmark_csv else None,
        label_map_csv=Path(args.label_map_csv) if args.label_map_csv else None,
        config=cfg,
    )
    pipeline.save(Path(args.model_out))
    print(f"[Saved] {args.model_out}")
    print(json.dumps(meta, ensure_ascii=False, indent=2))


def infer_command(args: argparse.Namespace) -> None:
    pipeline = EmotionZPipeline.load(Path(args.model_path), map_location="cpu")
    result = pipeline.encode_text(args.text)
    H = result.pop("H")
    print(f"score={result['score']:.4f}")
    print(f"H_shape={result['H_shape']}")
    print("u=", [round(v, 4) for v in result["u"]])
    print("h=", [round(v, 4) for v in result["h"]])
    print("z=", [round(float(v), 6) for v in result["z"]])
    print("final_emo=", [round(float(v), 4) for v in result["final_emo"]])
    if args.output_json:
        full = dict(result)
        full["H"] = H
        write_json(Path(args.output_json), full)
        print(f"[Saved JSON] {args.output_json}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Emotion Z Pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    train = sub.add_parser("train")
    train.add_argument("--dataset_csv", type=str, required=True)
    train.add_argument("--benchmark_csv", type=str, default="")
    train.add_argument("--label_map_csv", type=str, default="")
    train.add_argument("--model_out", type=str, required=True)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--stimulus_max_samples", type=int, default=30000)
    train.add_argument("--history_train_samples", type=int, default=512)
    train.add_argument("--future_horizon", type=int, default=3)
    train.add_argument("--min_prefix_len", type=int, default=4)
    train.add_argument("--prefix_stride", type=int, default=2)
    train.add_argument("--z_dim", type=int, default=16)
    train.add_argument("--hidden_dim", type=int, default=64)
    train.add_argument("--batch_size", type=int, default=32)
    train.add_argument("--lr", type=float, default=1e-3)
    train.add_argument("--epochs", type=int, default=5)
    train.add_argument("--alpha_future", type=float, default=0.5)
    train.add_argument("--device", type=str, default="auto")
    train.set_defaults(func=train_command)

    infer = sub.add_parser("infer")
    infer.add_argument("--model_path", type=str, required=True)
    infer.add_argument("--text", type=str, required=True)
    infer.add_argument("--output_json", type=str, default="")
    infer.set_defaults(func=infer_command)
    return ap


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
