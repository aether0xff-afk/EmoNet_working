from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

HERE = Path(__file__).resolve()
VERSION_ROOT = HERE.parents[1]
REPO_ROOT = HERE.parents[2]
sys.path.insert(0, str(VERSION_ROOT))

from semantic_fixture import SemanticArm, SemanticPair, build_semantic_pairs, flatten_pairs  # noqa: E402
from emonet_v5 import DynamicsConfig, EmoNetV5Clean, NeuralTrace, temporal_shuffle  # noqa: E402


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SEEDS = [7, 13, 21, 42, 100]
RIDGE_ALPHA = 3.0
CONTROL_SEED = 5101
OUT_DIR = VERSION_ROOT / "outputs" / "semantic_context_probe"


class CachedSentenceEncoder:
    """Frozen sentence-transformer adapter implementing the v5.0 TextEncoder protocol."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        dimension = self.model.get_sentence_embedding_dimension()
        if dimension is None:
            raise RuntimeError("sentence transformer did not report an embedding dimension")
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


@dataclass
class RidgeProbe:
    alpha: float
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None
    weights: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeProbe":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64).reshape(-1)
        if x.ndim != 2 or x.shape[0] != y.shape[0]:
            raise ValueError("invalid probe training shapes")
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        self.scale[self.scale < 1e-8] = 1.0
        xs = (x - self.mean) / self.scale
        xa = np.concatenate([xs, np.ones((xs.shape[0], 1), dtype=np.float64)], axis=1)
        target = np.where(y > 0, 1.0, -1.0)
        penalty = np.eye(xa.shape[1], dtype=np.float64) * float(self.alpha)
        penalty[-1, -1] = 0.0
        self.weights = np.linalg.solve(xa.T @ xa + penalty, xa.T @ target)
        return self

    def score(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None or self.weights is None:
            raise RuntimeError("probe must be fit before score")
        x = np.asarray(x, dtype=np.float64)
        xs = (x - self.mean) / self.scale
        xa = np.concatenate([xs, np.ones((xs.shape[0], 1), dtype=np.float64)], axis=1)
        return xa @ self.weights

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.score(x) >= 0.0).astype(np.int64)


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def concat(*arrays: np.ndarray) -> np.ndarray:
    return np.concatenate([np.asarray(a, dtype=np.float32).reshape(-1) for a in arrays]).astype(
        np.float32,
        copy=False,
    )


def collect_texts(pairs: list[SemanticPair]) -> set[str]:
    texts: set[str] = set()
    for arm in flatten_pairs(pairs):
        texts.add(arm.current_text)
        texts.update(arm.history)
        texts.add("\n".join(arm.history))
    return texts


def text_features(arm: SemanticArm, encoder: CachedSentenceEncoder) -> dict[str, np.ndarray]:
    history_vectors = np.stack([encoder.encode(event) for event in arm.history], axis=0)
    return {
        "current": encoder.encode(arm.current_text),
        "last": encoder.encode(arm.history[-1]),
        "history_bag": history_vectors.mean(axis=0).astype(np.float32),
        "full_history": encoder.encode("\n".join(arm.history)),
    }


def run_trace(
    model: EmoNetV5Clean,
    arm: SemanticArm,
    *,
    reset_before_current: bool,
) -> NeuralTrace:
    model.reset_all()
    model.consume_sequence(list(arm.history))
    if reset_before_current:
        model.reset_episode()
    return model.consume_event(arm.current_text)


def pair_key(arm: SemanticArm) -> tuple[str, int]:
    return arm.pair_id, arm.label


def build_seed_rows(
    seed: int,
    encoder: CachedSentenceEncoder,
    train_pairs: list[SemanticPair],
    test_pairs: list[SemanticPair],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    model = EmoNetV5Clean(encoder=encoder, config=DynamicsConfig(seed=seed))
    all_arms = flatten_pairs(train_pairs + test_pairs)

    features: dict[tuple[str, int], dict[str, np.ndarray | NeuralTrace | object]] = {}
    for arm_index, arm in enumerate(all_arms):
        tf = text_features(arm, encoder)
        real_trace = run_trace(model, arm, reset_before_current=False)
        reset_trace = run_trace(model, arm, reset_before_current=True)
        features[pair_key(arm)] = {
            "arm": arm,
            **tf,
            "real_trace": real_trace,
            "real_trace_summary": real_trace.summary_features(),
            "reset_trace_summary": reset_trace.summary_features(),
            "shuffled_trace_summary": temporal_shuffle(
                real_trace,
                CONTROL_SEED + seed * 10000 + arm_index,
            ).summary_features(),
        }

    train_arms = flatten_pairs(train_pairs)
    test_arms = flatten_pairs(test_pairs)
    y_train = np.asarray([arm.label for arm in train_arms], dtype=np.int64)
    y_test = np.asarray([arm.label for arm in test_arms], dtype=np.int64)

    def matrix(arms: list[SemanticArm], field: str) -> np.ndarray:
        return np.stack(
            [np.asarray(features[pair_key(arm)][field], dtype=np.float32) for arm in arms],
            axis=0,
        )

    x_current_train = matrix(train_arms, "current")
    x_current_test = matrix(test_arms, "current")
    x_last_train = matrix(train_arms, "last")
    x_last_test = matrix(test_arms, "last")
    x_bag_train = matrix(train_arms, "history_bag")
    x_bag_test = matrix(test_arms, "history_bag")
    x_full_train = matrix(train_arms, "full_history")
    x_full_test = matrix(test_arms, "full_history")
    x_trace_train = matrix(train_arms, "real_trace_summary")
    x_trace_test = matrix(test_arms, "real_trace_summary")
    x_reset_test = matrix(test_arms, "reset_trace_summary")
    x_shuffle_test = matrix(test_arms, "shuffled_trace_summary")

    wrong_rows: list[np.ndarray] = []
    for arm in test_arms:
        opposite_label = 0 if arm.label == 1 else 1
        wrong_rows.append(
            np.asarray(features[(arm.pair_id, opposite_label)]["real_trace_summary"], dtype=np.float32)
        )
    x_wrong_test = np.stack(wrong_rows, axis=0)

    baseline_specs = {
        "current_text_only": (x_current_train, x_current_test),
        "last_event_only": (x_last_train, x_last_test),
        "history_bag_embedding": (x_bag_train, x_bag_test),
        "full_history_embedding": (x_full_train, x_full_test),
        "trace_only_real": (x_trace_train, x_trace_test),
    }

    metrics: dict[str, float] = {}
    predictions: dict[str, np.ndarray] = {}
    scores: dict[str, np.ndarray] = {}
    for name, (x_train, x_test) in baseline_specs.items():
        probe = RidgeProbe(RIDGE_ALPHA).fit(x_train, y_train)
        predictions[name] = probe.predict(x_test)
        scores[name] = probe.score(x_test)
        metrics[name] = accuracy(y_test, predictions[name])

    x_text_real_train = np.concatenate([x_current_train, x_trace_train], axis=1)
    main_probe = RidgeProbe(RIDGE_ALPHA).fit(x_text_real_train, y_train)
    controlled = {
        "text_plus_real_trace": np.concatenate([x_current_test, x_trace_test], axis=1),
        "text_plus_temporal_shuffle": np.concatenate([x_current_test, x_shuffle_test], axis=1),
        "text_plus_wrong_trace": np.concatenate([x_current_test, x_wrong_test], axis=1),
        "text_plus_reset_trace": np.concatenate([x_current_test, x_reset_test], axis=1),
    }
    for name, x_test in controlled.items():
        predictions[name] = main_probe.predict(x_test)
        scores[name] = main_probe.score(x_test)
        metrics[name] = accuracy(y_test, predictions[name])

    metric_rows = [{"seed": seed, **metrics}]
    prediction_rows: list[dict[str, object]] = []
    for index, arm in enumerate(test_arms):
        row: dict[str, object] = {
            "seed": seed,
            "pair_id": arm.pair_id,
            "domain": arm.domain,
            "label": arm.label,
        }
        for name in predictions:
            row[f"pred_{name}"] = int(predictions[name][index])
            row[f"score_{name}"] = float(scores[name][index])
        prediction_rows.append(row)
    return metric_rows, prediction_rows


def summarize(metric_rows: list[dict[str, object]]) -> dict[str, object]:
    metric_names = [key for key in metric_rows[0].keys() if key != "seed"]
    mean_accuracy = {
        name: float(np.mean([float(row[name]) for row in metric_rows])) for name in metric_names
    }
    std_accuracy = {
        name: float(np.std([float(row[name]) for row in metric_rows])) for name in metric_names
    }
    real = mean_accuracy["text_plus_real_trace"]
    gates = {
        "encoder_full_history_above_0_80": mean_accuracy["full_history_embedding"] >= 0.80,
        "current_text_near_chance": abs(mean_accuracy["current_text_only"] - 0.50) <= 0.06,
        "last_event_near_chance": abs(mean_accuracy["last_event_only"] - 0.50) <= 0.06,
        "real_trace_above_0_65_mean": real >= 0.65,
        "every_seed_real_above_0_55": all(
            float(row["text_plus_real_trace"]) >= 0.55 for row in metric_rows
        ),
        "real_beats_reset_by_0_15": real - mean_accuracy["text_plus_reset_trace"] >= 0.15,
        "real_beats_wrong_by_0_15": real - mean_accuracy["text_plus_wrong_trace"] >= 0.15,
    }
    gates["all_primary_gates"] = all(gates.values())
    return {
        "version": "v5.1",
        "purpose": "held-out natural-language semantic-context memory test; not an affect claim",
        "encoder": MODEL_NAME,
        "recurrent_core": "frozen v5.0 FixedRecurrentDynamics",
        "seeds": SEEDS,
        "train_pairs": 60,
        "test_pairs": 20,
        "train_samples_per_seed": 120,
        "test_samples_per_seed": 40,
        "probe": {"type": "standardized ridge binary", "alpha": RIDGE_ALPHA},
        "mean_accuracy": mean_accuracy,
        "std_accuracy": std_accuracy,
        "gaps": {
            "real_minus_current": real - mean_accuracy["current_text_only"],
            "real_minus_reset": real - mean_accuracy["text_plus_reset_trace"],
            "real_minus_wrong": real - mean_accuracy["text_plus_wrong_trace"],
            "real_minus_shuffle": real - mean_accuracy["text_plus_temporal_shuffle"],
            "full_history_minus_real": mean_accuracy["full_history_embedding"] - real,
        },
        "acceptance": gates,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    train_pairs, test_pairs = build_semantic_pairs()
    encoder = CachedSentenceEncoder(MODEL_NAME)
    encoder.preload(collect_texts(train_pairs + test_pairs))

    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        seed_metrics, seed_predictions = build_seed_rows(
            seed,
            encoder,
            train_pairs,
            test_pairs,
        )
        metric_rows.extend(seed_metrics)
        prediction_rows.extend(seed_predictions)

    summary = summarize(metric_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "per_seed_metrics.csv", metric_rows)
    write_csv(OUT_DIR / "test_predictions.csv", prediction_rows)
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
