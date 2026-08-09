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
OUT_DIR = VERSION_ROOT / "outputs" / "domain_calibration"


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
            self.weights = np.linalg.solve(xs.T @ xs + self.alpha * np.eye(d), xs.T @ yc)
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


def trace_views(trace: NeuralTrace) -> dict[str, np.ndarray]:
    return {
        "final_state": trace.final_state.astype(np.float32),
        "summary_features": trace.summary_features().astype(np.float32),
        "raw_flattened_trace": trace.states.reshape(-1).astype(np.float32),
    }


def run_trace(model: EmoNetV5Clean, arm: SemanticArm, reset: bool) -> NeuralTrace:
    model.reset_all()
    model.consume_sequence(list(arm.history))
    if reset:
        model.reset_episode()
    return model.consume_event(arm.current_text)


def pairs_for_domain(pairs: list[SemanticPair], domain: str) -> list[SemanticPair]:
    return [pair for pair in pairs if pair.domain == domain]


def preload_texts(encoder: CachedSentenceEncoder, pairs: list[SemanticPair]) -> None:
    texts: set[str] = set()
    for arm in flatten_pairs(pairs):
        texts.update(arm.history)
        texts.add(arm.current_text)
    encoder.preload(texts)


def input_domain_metrics(
    encoder: CachedSentenceEncoder,
    train_pairs: list[SemanticPair],
    test_pairs: list[SemanticPair],
) -> list[dict[str, object]]:
    domains = sorted({pair.domain for pair in train_pairs})
    rows: list[dict[str, object]] = []
    for domain in domains:
        train_arms = flatten_pairs(pairs_for_domain(train_pairs, domain))
        test_arms = flatten_pairs(pairs_for_domain(test_pairs, domain))
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
        rows.append(
            {
                "domain": domain,
                "semantic_event_accuracy": accuracy(y_test, semantic_probe.predict(semantic_test)),
                "history_bag_accuracy": accuracy(y_test, bag_probe.predict(bag_test)),
            }
        )
    return rows


def trace_domain_metrics(
    seed: int,
    encoder: CachedSentenceEncoder,
    train_pairs: list[SemanticPair],
    test_pairs: list[SemanticPair],
) -> list[dict[str, object]]:
    domains = sorted({pair.domain for pair in train_pairs})
    model = EmoNetV5Clean(encoder=encoder, config=DynamicsConfig(seed=seed))
    rows: list[dict[str, object]] = []

    for domain in domains:
        train_arms = flatten_pairs(pairs_for_domain(train_pairs, domain))
        test_arms = flatten_pairs(pairs_for_domain(test_pairs, domain))
        y_train = np.asarray([arm.label for arm in train_arms], dtype=np.int64)
        y_test = np.asarray([arm.label for arm in test_arms], dtype=np.int64)

        train_real: dict[tuple[str, int], dict[str, np.ndarray]] = {}
        test_real: dict[tuple[str, int], dict[str, np.ndarray]] = {}
        test_reset: dict[tuple[str, int], dict[str, np.ndarray]] = {}

        for arm in train_arms:
            train_real[(arm.pair_id, arm.label)] = trace_views(run_trace(model, arm, reset=False))
        for arm in test_arms:
            key = (arm.pair_id, arm.label)
            test_real[key] = trace_views(run_trace(model, arm, reset=False))
            test_reset[key] = trace_views(run_trace(model, arm, reset=True))

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
            rows.append(
                {
                    "seed": seed,
                    "domain": domain,
                    "view": view,
                    "real_accuracy": accuracy(y_test, probe.predict(x_real)),
                    "reset_accuracy": accuracy(y_test, probe.predict(x_reset)),
                    "wrong_accuracy": accuracy(y_test, probe.predict(x_wrong)),
                }
            )
    return rows


def mean_field(rows: list[dict[str, object]], field: str, view: str | None = None) -> float:
    values = [
        float(row[field])
        for row in rows
        if view is None or str(row.get("view")) == view
    ]
    return float(np.mean(values))


def summarize(
    input_rows: list[dict[str, object]],
    trace_rows: list[dict[str, object]],
) -> dict[str, object]:
    semantic_macro = mean_field(input_rows, "semantic_event_accuracy")
    bag_macro = mean_field(input_rows, "history_bag_accuracy")
    per_domain_semantic = {
        str(row["domain"]): float(row["semantic_event_accuracy"]) for row in input_rows
    }

    real_macro = {
        view: mean_field(trace_rows, "real_accuracy", view)
        for view in ("final_state", "summary_features", "raw_flattened_trace")
    }
    reset_macro = {
        view: mean_field(trace_rows, "reset_accuracy", view)
        for view in ("final_state", "summary_features", "raw_flattened_trace")
    }
    wrong_macro = {
        view: mean_field(trace_rows, "wrong_accuracy", view)
        for view in ("final_state", "summary_features", "raw_flattened_trace")
    }

    raw_real = real_macro["raw_flattened_trace"]
    gates = {
        "input_semantic_macro_above_0_85": semantic_macro >= 0.85,
        "every_domain_input_at_least_0_75": min(per_domain_semantic.values()) >= 0.75,
        "raw_trace_macro_above_0_65": raw_real >= 0.65,
        "raw_trace_beats_reset_by_0_10": raw_real - reset_macro["raw_flattened_trace"] >= 0.10,
        "raw_trace_beats_wrong_by_0_10": raw_real - wrong_macro["raw_flattened_trace"] >= 0.10,
    }
    gates["calibration_pass"] = all(gates.values())

    return {
        "version": "v5.1.2",
        "purpose": "domain-conditioned protocol calibration; not confirmatory evidence",
        "encoder": MODEL_NAME,
        "input_macro": {
            "semantic_event": semantic_macro,
            "history_bag": bag_macro,
        },
        "input_per_domain": per_domain_semantic,
        "trace_real_macro": real_macro,
        "trace_reset_macro": reset_macro,
        "trace_wrong_macro": wrong_macro,
        "raw_gaps": {
            "real_minus_reset": raw_real - reset_macro["raw_flattened_trace"],
            "real_minus_wrong": raw_real - wrong_macro["raw_flattened_trace"],
            "input_minus_real": semantic_macro - raw_real,
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
    preload_texts(encoder, train_pairs + test_pairs)

    input_rows = input_domain_metrics(encoder, train_pairs, test_pairs)
    trace_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        trace_rows.extend(trace_domain_metrics(seed, encoder, train_pairs, test_pairs))

    summary = summarize(input_rows, trace_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "input_domain_metrics.csv", input_rows)
    write_csv(OUT_DIR / "trace_domain_metrics.csv", trace_rows)
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
