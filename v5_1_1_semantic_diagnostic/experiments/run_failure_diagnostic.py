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
V51_ROOT = REPO_ROOT / "v5_1_semantic_context"
sys.path.insert(0, str(V51_ROOT))

from semantic_fixture import SemanticArm, SemanticPair, build_semantic_pairs, flatten_pairs  # noqa: E402
from emonet_v5 import DynamicsConfig, EmoNetV5Clean, NeuralTrace  # noqa: E402


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SEEDS = [7, 13, 21, 42, 100]
RIDGE_ALPHA = 3.0
OUT_DIR = VERSION_ROOT / "outputs" / "failure_diagnostic"


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


@dataclass
class RidgeProbe:
    alpha: float
    mean: np.ndarray | None = None
    scale: np.ndarray | None = None
    weights: np.ndarray | None = None
    intercept: float = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeProbe":
        x = np.asarray(x, dtype=np.float64)
        y_signed = np.where(np.asarray(y, dtype=np.int64).reshape(-1) > 0, 1.0, -1.0)
        self.mean = x.mean(axis=0)
        self.scale = x.std(axis=0)
        self.scale[self.scale < 1e-8] = 1.0
        xs = (x - self.mean) / self.scale
        self.intercept = float(y_signed.mean())
        yc = y_signed - self.intercept
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

    def score(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None or self.weights is None:
            raise RuntimeError("probe not fit")
        xs = (np.asarray(x, dtype=np.float64) - self.mean) / self.scale
        return xs @ self.weights + self.intercept

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.score(x) >= 0.0).astype(np.int64)


def accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y) == np.asarray(pred)))


def history_for_gap(arm: SemanticArm, gap: int) -> tuple[str, ...]:
    if gap == 0:
        return arm.history[:2]
    if gap == 1:
        return arm.history[:3]
    if gap == 2:
        return arm.history
    raise ValueError("gap must be 0, 1, or 2")


def run_trace(model: EmoNetV5Clean, arm: SemanticArm, gap: int, reset: bool) -> NeuralTrace:
    model.reset_all()
    model.consume_sequence(list(history_for_gap(arm, gap)))
    if reset:
        model.reset_episode()
    return model.consume_event(arm.current_text)


def trace_views(trace: NeuralTrace) -> dict[str, np.ndarray]:
    return {
        "final_state": trace.final_state.astype(np.float32),
        "summary_features": trace.summary_features().astype(np.float32),
        "raw_flattened_trace": trace.states.reshape(-1).astype(np.float32),
    }


def normalized_l2(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> float:
    left = np.asarray(a, dtype=np.float32).reshape(-1)
    right = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.sqrt(np.mean(left * left)) + np.sqrt(np.mean(right * right)) + eps)
    return float(np.sqrt(np.mean((left - right) ** 2)) / denom)


def preload_texts(encoder: CachedSentenceEncoder, train: list[SemanticPair], test: list[SemanticPair]) -> None:
    texts: set[str] = set()
    for arm in flatten_pairs(train + test):
        texts.update(arm.history)
        texts.add(arm.current_text)
    encoder.preload(texts)


def baseline_metrics(
    encoder: CachedSentenceEncoder,
    train_pairs: list[SemanticPair],
    test_pairs: list[SemanticPair],
) -> dict[str, float]:
    train_arms = flatten_pairs(train_pairs)
    test_arms = flatten_pairs(test_pairs)
    y_train = np.asarray([arm.label for arm in train_arms], dtype=np.int64)
    y_test = np.asarray([arm.label for arm in test_arms], dtype=np.int64)

    semantic_train = np.stack([encoder.encode(arm.history[1]) for arm in train_arms])
    semantic_test = np.stack([encoder.encode(arm.history[1]) for arm in test_arms])

    bag_train = np.stack(
        [np.stack([encoder.encode(event) for event in arm.history]).mean(axis=0) for arm in train_arms]
    )
    bag_test = np.stack(
        [np.stack([encoder.encode(event) for event in arm.history]).mean(axis=0) for arm in test_arms]
    )

    semantic_probe = RidgeProbe(RIDGE_ALPHA).fit(semantic_train, y_train)
    bag_probe = RidgeProbe(RIDGE_ALPHA).fit(bag_train, y_train)
    return {
        "semantic_event_embedding": accuracy(y_test, semantic_probe.predict(semantic_test)),
        "history_bag_embedding": accuracy(y_test, bag_probe.predict(bag_test)),
    }


def seed_gap_metrics(
    seed: int,
    gap: int,
    encoder: CachedSentenceEncoder,
    train_pairs: list[SemanticPair],
    test_pairs: list[SemanticPair],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    model = EmoNetV5Clean(encoder=encoder, config=DynamicsConfig(seed=seed))
    train_arms = flatten_pairs(train_pairs)
    test_arms = flatten_pairs(test_pairs)
    y_train = np.asarray([arm.label for arm in train_arms], dtype=np.int64)
    y_test = np.asarray([arm.label for arm in test_arms], dtype=np.int64)

    train_real: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    test_real: dict[tuple[str, int], dict[str, np.ndarray]] = {}
    test_reset: dict[tuple[str, int], dict[str, np.ndarray]] = {}

    for arm in train_arms:
        train_real[(arm.pair_id, arm.label)] = trace_views(run_trace(model, arm, gap, reset=False))
    for arm in test_arms:
        key = (arm.pair_id, arm.label)
        test_real[key] = trace_views(run_trace(model, arm, gap, reset=False))
        test_reset[key] = trace_views(run_trace(model, arm, gap, reset=True))

    metric_rows: list[dict[str, object]] = []
    pair_distance_rows: list[dict[str, object]] = []

    for view in ("final_state", "summary_features", "raw_flattened_trace"):
        x_train = np.stack([train_real[(arm.pair_id, arm.label)][view] for arm in train_arms])
        x_real = np.stack([test_real[(arm.pair_id, arm.label)][view] for arm in test_arms])
        x_reset = np.stack([test_reset[(arm.pair_id, arm.label)][view] for arm in test_arms])
        x_wrong = np.stack(
            [
                test_real[(arm.pair_id, 0 if arm.label == 1 else 1)][view]
                for arm in test_arms
            ]
        )

        probe = RidgeProbe(RIDGE_ALPHA).fit(x_train, y_train)
        metric_rows.append(
            {
                "seed": seed,
                "gap": gap,
                "view": view,
                "real_accuracy": accuracy(y_test, probe.predict(x_real)),
                "reset_accuracy": accuracy(y_test, probe.predict(x_reset)),
                "wrong_accuracy": accuracy(y_test, probe.predict(x_wrong)),
            }
        )

        distances: list[float] = []
        for pair in test_pairs:
            usable = test_real[(pair.pair_id, 1)][view]
            blocked = test_real[(pair.pair_id, 0)][view]
            distances.append(normalized_l2(usable, blocked))
        pair_distance_rows.append(
            {
                "seed": seed,
                "gap": gap,
                "view": view,
                "mean_pair_distance": float(np.mean(distances)),
                "min_pair_distance": float(np.min(distances)),
                "max_pair_distance": float(np.max(distances)),
            }
        )

    return metric_rows, pair_distance_rows


def aggregate(rows: list[dict[str, object]], field: str, gap: int, view: str) -> float:
    values = [
        float(row[field])
        for row in rows
        if int(row["gap"]) == gap and str(row["view"]) == view
    ]
    return float(np.mean(values))


def classify_bottleneck(
    baseline: dict[str, float],
    metrics: list[dict[str, object]],
) -> tuple[str, dict[str, float | bool]]:
    input_acc = baseline["semantic_event_embedding"]
    raw0 = aggregate(metrics, "real_accuracy", 0, "raw_flattened_trace")
    raw1 = aggregate(metrics, "real_accuracy", 1, "raw_flattened_trace")
    raw2 = aggregate(metrics, "real_accuracy", 2, "raw_flattened_trace")
    summary2 = aggregate(metrics, "real_accuracy", 2, "summary_features")
    final2 = aggregate(metrics, "real_accuracy", 2, "final_state")

    facts: dict[str, float | bool] = {
        "input_semantic_accuracy": input_acc,
        "raw_gap0_accuracy": raw0,
        "raw_gap1_accuracy": raw1,
        "raw_gap2_accuracy": raw2,
        "summary_gap2_accuracy": summary2,
        "final_gap2_accuracy": final2,
        "gap0_to_gap2_drop": raw0 - raw2,
        "raw_minus_summary_gap2": raw2 - summary2,
        "input_adequate_0_80": input_acc >= 0.80,
    }

    if input_acc < 0.80:
        label = "input_or_fixture_adequacy"
    elif raw0 < 0.65:
        label = "immediate_recurrent_transform_or_linear_readout"
    elif raw0 - raw2 >= 0.10:
        label = "recurrent_memory_decay"
    elif raw2 - summary2 >= 0.10:
        label = "trace_summary_information_loss"
    else:
        label = "mixed_or_unresolved"
    return label, facts


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    train_pairs, test_pairs = build_semantic_pairs()
    encoder = CachedSentenceEncoder(MODEL_NAME)
    preload_texts(encoder, train_pairs, test_pairs)
    baseline = baseline_metrics(encoder, train_pairs, test_pairs)

    metric_rows: list[dict[str, object]] = []
    distance_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        for gap in (0, 1, 2):
            metrics, distances = seed_gap_metrics(
                seed,
                gap,
                encoder,
                train_pairs,
                test_pairs,
            )
            metric_rows.extend(metrics)
            distance_rows.extend(distances)

    bottleneck, facts = classify_bottleneck(baseline, metric_rows)
    summary = {
        "version": "v5.1.1",
        "purpose": "diagnose v5.1 semantic-context failure source",
        "encoder": MODEL_NAME,
        "baseline": baseline,
        "mean_real_accuracy": {
            f"gap{gap}_{view}": aggregate(metric_rows, "real_accuracy", gap, view)
            for gap in (0, 1, 2)
            for view in ("final_state", "summary_features", "raw_flattened_trace")
        },
        "mean_reset_accuracy": {
            f"gap{gap}_{view}": aggregate(metric_rows, "reset_accuracy", gap, view)
            for gap in (0, 1, 2)
            for view in ("final_state", "summary_features", "raw_flattened_trace")
        },
        "mean_wrong_accuracy": {
            f"gap{gap}_{view}": aggregate(metric_rows, "wrong_accuracy", gap, view)
            for gap in (0, 1, 2)
            for view in ("final_state", "summary_features", "raw_flattened_trace")
        },
        "diagnosis": bottleneck,
        "diagnostic_facts": facts,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "probe_metrics.csv", metric_rows)
    write_csv(OUT_DIR / "pair_distances.csv", distance_rows)
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
