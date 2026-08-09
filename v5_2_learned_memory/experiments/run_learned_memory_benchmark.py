from __future__ import annotations

import copy
import csv
import json
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).resolve()
VERSION_ROOT = HERE.parents[1]
REPO_ROOT = HERE.parents[2]
V51_ROOT = REPO_ROOT / "v5_1_semantic_context"
sys.path.insert(0, str(VERSION_ROOT))
sys.path.insert(0, str(V51_ROOT))

from learned_core import LearnedCoreConfig, LearnedLeakyRecurrentCore  # noqa: E402
from semantic_fixture import SemanticArm, SemanticPair, build_semantic_pairs, flatten_pairs  # noqa: E402
from emonet_v5 import DynamicsConfig, EmoNetV5Clean  # noqa: E402


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SEEDS = [7, 13, 21, 42, 100]
TRAIN_EPOCHS = 150
LEARNING_RATE = 2e-3
WEIGHT_DECAY = 1e-5
RIDGE_ALPHA = 3.0
EMA_DECAY = 0.80
OUT_DIR = VERSION_ROOT / "outputs" / "learned_memory_benchmark"


class CachedSentenceEncoder:
    def __init__(self, model_name: str) -> None:
        self.model = SentenceTransformer(model_name)
        dimension = self.model.get_sentence_embedding_dimension()
        if dimension is None:
            raise RuntimeError("embedding dimension unavailable")
        self.dimension = int(dimension)
        self.cache: dict[str, np.ndarray] = {}

    @property
    def output_dim(self) -> int:
        return self.dimension

    def preload(self, texts: Iterable[str]) -> None:
        missing = sorted({str(text) for text in texts if str(text) not in self.cache})
        if not missing:
            return
        vectors = self.model.encode(
            missing,
            batch_size=64,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        for text, vector in zip(missing, vectors, strict=True):
            self.cache[text] = np.asarray(vector, dtype=np.float32).reshape(-1)

    def encode(self, text: str) -> np.ndarray:
        key = str(text)
        if key not in self.cache:
            self.preload([key])
        return self.cache[key].copy()


class RidgeProbe:
    def __init__(self, alpha: float) -> None:
        self.alpha = float(alpha)
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.weights: np.ndarray | None = None
        self.intercept = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeProbe":
        x = np.asarray(x, dtype=np.float64)
        target = np.where(np.asarray(y, dtype=np.int64).reshape(-1) > 0, 1.0, -1.0)
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        self.scale[self.scale < 1e-8] = 1.0
        xs = (x - self.mean) / self.scale
        self.intercept = float(target.mean())
        yc = target - self.intercept
        n, d = xs.shape
        if d > n:
            dual = np.linalg.solve(xs @ xs.T + self.alpha * np.eye(n), yc)
            self.weights = xs.T @ dual
        else:
            self.weights = np.linalg.solve(
                xs.T @ xs + self.alpha * np.eye(d),
                xs.T @ yc,
            )
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None or self.weights is None:
            raise RuntimeError("probe not fit")
        xs = (np.asarray(x, dtype=np.float64) - self.mean) / self.scale
        score = xs @ self.weights + self.intercept
        return (score >= 0.0).astype(np.int64)


def accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y) == np.asarray(pred)))


def all_texts(pairs: list[SemanticPair]) -> set[str]:
    result: set[str] = set()
    for arm in flatten_pairs(pairs):
        result.update(arm.history)
        result.add(arm.current_text)
    return result


def arm_sequence(arm: SemanticArm, encoder: CachedSentenceEncoder) -> np.ndarray:
    texts = list(arm.history) + [arm.current_text]
    return np.stack([encoder.encode(text) for text in texts], axis=0).astype(np.float32)


def sequence_tensor(arms: list[SemanticArm], encoder: CachedSentenceEncoder) -> torch.Tensor:
    return torch.from_numpy(np.stack([arm_sequence(arm, encoder) for arm in arms], axis=0))


def train_core(
    seed: int,
    train_sequences: torch.Tensor,
    config: LearnedCoreConfig,
) -> tuple[LearnedLeakyRecurrentCore, list[dict[str, float]]]:
    # The function deliberately receives no semantic/task labels.
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    model = LearnedLeakyRecurrentCore(train_sequences.shape[-1], config, seed)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(TRAIN_EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, lag_losses = model.delayed_memory_loss(train_sequences)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        model.stabilize_recurrent(max_spectral_norm=0.98)

        value = float(loss.detach().cpu())
        history.append(
            {
                "epoch": float(epoch + 1),
                "loss": value,
                **{f"lag{lag}_loss": val for lag, val in lag_losses.items()},
            }
        )
        if value < best_loss:
            best_loss = value
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, history


def learned_features(
    model: LearnedLeakyRecurrentCore,
    arms: list[SemanticArm],
    encoder: CachedSentenceEncoder,
) -> tuple[dict[tuple[str, int], np.ndarray], dict[tuple[str, int], np.ndarray]]:
    tensor = sequence_tensor(arms, encoder)
    with torch.no_grad():
        real = model.final_event_trace(tensor).cpu().numpy().reshape(len(arms), -1)
        reset = model.reset_final_event_trace(tensor).cpu().numpy().reshape(len(arms), -1)
    real_map = {(arm.pair_id, arm.label): real[i].astype(np.float32) for i, arm in enumerate(arms)}
    reset_map = {(arm.pair_id, arm.label): reset[i].astype(np.float32) for i, arm in enumerate(arms)}
    return real_map, reset_map


def random_features(
    seed: int,
    arms: list[SemanticArm],
    encoder: CachedSentenceEncoder,
) -> dict[tuple[str, int], np.ndarray]:
    model = EmoNetV5Clean(encoder=encoder, config=DynamicsConfig(seed=seed))
    result: dict[tuple[str, int], np.ndarray] = {}
    for arm in arms:
        model.reset_all()
        model.consume_sequence(list(arm.history))
        trace = model.consume_event(arm.current_text)
        result[(arm.pair_id, arm.label)] = trace.states.reshape(-1).astype(np.float32)
    return result


def ema_features(
    arms: list[SemanticArm],
    encoder: CachedSentenceEncoder,
) -> dict[tuple[str, int], np.ndarray]:
    result: dict[tuple[str, int], np.ndarray] = {}
    for arm in arms:
        state = np.zeros(encoder.output_dim, dtype=np.float32)
        for embedding in arm_sequence(arm, encoder):
            state = EMA_DECAY * state + (1.0 - EMA_DECAY) * embedding
        norm = float(np.linalg.norm(state))
        if norm > 0:
            state = state / norm
        result[(arm.pair_id, arm.label)] = state.astype(np.float32)
    return result


def pairs_for_domain(pairs: list[SemanticPair], domain: str) -> list[SemanticPair]:
    return [pair for pair in pairs if pair.domain == domain]


def evaluate_map(
    train_map: dict[tuple[str, int], np.ndarray],
    test_map: dict[tuple[str, int], np.ndarray],
    train_pairs: list[SemanticPair],
    test_pairs: list[SemanticPair],
) -> tuple[float, dict[str, float]]:
    domains = sorted({pair.domain for pair in train_pairs})
    domain_scores: dict[str, float] = {}
    for domain in domains:
        train_arms = flatten_pairs(pairs_for_domain(train_pairs, domain))
        test_arms = flatten_pairs(pairs_for_domain(test_pairs, domain))
        y_train = np.asarray([arm.label for arm in train_arms], dtype=np.int64)
        y_test = np.asarray([arm.label for arm in test_arms], dtype=np.int64)
        x_train = np.stack([train_map[(arm.pair_id, arm.label)] for arm in train_arms])
        x_test = np.stack([test_map[(arm.pair_id, arm.label)] for arm in test_arms])
        probe = RidgeProbe(RIDGE_ALPHA).fit(x_train, y_train)
        domain_scores[domain] = accuracy(y_test, probe.predict(x_test))
    return float(np.mean(list(domain_scores.values()))), domain_scores


def evaluate_learned_controls(
    train_real: dict[tuple[str, int], np.ndarray],
    test_real: dict[tuple[str, int], np.ndarray],
    test_reset: dict[tuple[str, int], np.ndarray],
    train_pairs: list[SemanticPair],
    test_pairs: list[SemanticPair],
) -> tuple[float, float, float, dict[str, dict[str, float]]]:
    domains = sorted({pair.domain for pair in train_pairs})
    per_domain: dict[str, dict[str, float]] = {}
    real_scores: list[float] = []
    reset_scores: list[float] = []
    wrong_scores: list[float] = []

    for domain in domains:
        train_arms = flatten_pairs(pairs_for_domain(train_pairs, domain))
        test_arms = flatten_pairs(pairs_for_domain(test_pairs, domain))
        y_train = np.asarray([arm.label for arm in train_arms], dtype=np.int64)
        y_test = np.asarray([arm.label for arm in test_arms], dtype=np.int64)
        x_train = np.stack([train_real[(arm.pair_id, arm.label)] for arm in train_arms])
        probe = RidgeProbe(RIDGE_ALPHA).fit(x_train, y_train)

        x_real = np.stack([test_real[(arm.pair_id, arm.label)] for arm in test_arms])
        x_reset = np.stack([test_reset[(arm.pair_id, arm.label)] for arm in test_arms])
        x_wrong = np.stack(
            [
                test_real[(arm.pair_id, 0 if arm.label == 1 else 1)]
                for arm in test_arms
            ]
        )
        real_acc = accuracy(y_test, probe.predict(x_real))
        reset_acc = accuracy(y_test, probe.predict(x_reset))
        wrong_acc = accuracy(y_test, probe.predict(x_wrong))
        real_scores.append(real_acc)
        reset_scores.append(reset_acc)
        wrong_scores.append(wrong_acc)
        per_domain[domain] = {
            "real": real_acc,
            "reset": reset_acc,
            "wrong": wrong_acc,
        }

    return (
        float(np.mean(real_scores)),
        float(np.mean(reset_scores)),
        float(np.mean(wrong_scores)),
        per_domain,
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    train_pairs, test_pairs = build_semantic_pairs()
    train_arms = flatten_pairs(train_pairs)
    test_arms = flatten_pairs(test_pairs)
    all_arms = train_arms + test_arms

    encoder = CachedSentenceEncoder(MODEL_NAME)
    encoder.preload(all_texts(train_pairs + test_pairs))
    train_sequences = sequence_tensor(train_arms, encoder)
    test_sequences = sequence_tensor(test_arms, encoder)

    core_config = LearnedCoreConfig()
    seed_rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    domain_rows: list[dict[str, object]] = []

    # EMA has no recurrent seed, so compute once.
    ema_train = ema_features(train_arms, encoder)
    ema_test = ema_features(test_arms, encoder)
    ema_macro, ema_domains = evaluate_map(ema_train, ema_test, train_pairs, test_pairs)

    for seed in SEEDS:
        learned, history = train_core(seed, train_sequences, core_config)
        for row in history:
            training_rows.append({"seed": seed, **row})

        lag3_train_cos = learned.lag_cosine_at_final(train_sequences, lag=3)
        lag3_test_cos = learned.lag_cosine_at_final(test_sequences, lag=3)

        learned_train, _ = learned_features(learned, train_arms, encoder)
        learned_test, learned_reset = learned_features(learned, test_arms, encoder)
        learned_real, learned_reset_acc, learned_wrong, learned_domains = evaluate_learned_controls(
            learned_train,
            learned_test,
            learned_reset,
            train_pairs,
            test_pairs,
        )

        random_train = random_features(seed, train_arms, encoder)
        random_test = random_features(seed, test_arms, encoder)
        random_macro, random_domains = evaluate_map(
            random_train,
            random_test,
            train_pairs,
            test_pairs,
        )

        seed_rows.append(
            {
                "seed": seed,
                "final_train_loss": history[-1]["loss"],
                "lag3_train_cosine": lag3_train_cos,
                "lag3_test_cosine": lag3_test_cos,
                "learned_real_macro": learned_real,
                "learned_reset_macro": learned_reset_acc,
                "learned_wrong_macro": learned_wrong,
                "random_real_macro": random_macro,
                "ema_macro": ema_macro,
            }
        )

        for domain in sorted(learned_domains):
            domain_rows.append(
                {
                    "seed": seed,
                    "domain": domain,
                    "learned_real": learned_domains[domain]["real"],
                    "learned_reset": learned_domains[domain]["reset"],
                    "learned_wrong": learned_domains[domain]["wrong"],
                    "random_real": random_domains[domain],
                    "ema": ema_domains[domain],
                }
            )

    def mean(field: str) -> float:
        return float(np.mean([float(row[field]) for row in seed_rows]))

    learned_mean = mean("learned_real_macro")
    random_mean = mean("random_real_macro")
    reset_mean = mean("learned_reset_macro")
    wrong_mean = mean("learned_wrong_macro")
    lag3_test_mean = mean("lag3_test_cosine")

    acceptance = {
        "lag3_test_cosine_above_0_40": lag3_test_mean >= 0.40,
        "learned_semantic_macro_at_least_0_70": learned_mean >= 0.70,
        "learned_beats_random_by_0_10": learned_mean - random_mean >= 0.10,
        "learned_beats_reset_by_0_15": learned_mean - reset_mean >= 0.15,
        "learned_beats_wrong_by_0_15": learned_mean - wrong_mean >= 0.15,
    }
    acceptance["all_primary_gates"] = all(acceptance.values())

    summary = {
        "version": "v5.2",
        "purpose": "label-free learned recurrent memory development benchmark; not confirmatory evidence",
        "encoder": MODEL_NAME,
        "training": {
            "task_labels_used_by_core": False,
            "emotion_labels_used_by_core": False,
            "objective": "delayed event embedding reconstruction at lags 1/2/3",
            "epochs": TRAIN_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "core_config": asdict(core_config),
        },
        "seeds": SEEDS,
        "mean": {
            "lag3_train_cosine": mean("lag3_train_cosine"),
            "lag3_test_cosine": lag3_test_mean,
            "learned_real_macro": learned_mean,
            "learned_reset_macro": reset_mean,
            "learned_wrong_macro": wrong_mean,
            "v5_0_random_macro": random_mean,
            "ema_embedding_memory_macro": ema_macro,
        },
        "gaps": {
            "learned_minus_random": learned_mean - random_mean,
            "learned_minus_reset": learned_mean - reset_mean,
            "learned_minus_wrong": learned_mean - wrong_mean,
            "learned_minus_ema": learned_mean - ema_macro,
        },
        "acceptance": acceptance,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "per_seed_metrics.csv", seed_rows)
    write_csv(OUT_DIR / "training_curve.csv", training_rows)
    write_csv(OUT_DIR / "per_domain_metrics.csv", domain_rows)
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
