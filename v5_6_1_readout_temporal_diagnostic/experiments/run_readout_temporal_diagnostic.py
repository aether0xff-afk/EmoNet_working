from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve()
VERSION_ROOT = HERE.parents[1]
REPO_ROOT = HERE.parents[2]
V52_EXPERIMENTS = REPO_ROOT / "v5_2_learned_memory" / "experiments"
V54_ROOT = REPO_ROOT / "v5_4_fresh_confirmatory"
V56_ROOT = REPO_ROOT / "v5_6_dual_timescale_state"
sys.path.insert(0, str(VERSION_ROOT))
sys.path.insert(0, str(V52_EXPERIMENTS))
sys.path.insert(0, str(V54_ROOT))
sys.path.insert(0, str(V56_ROOT))

from dual_state import DualTimescaleState, fast_features, slow_features, dual_features  # noqa: E402
from fresh_fixture import FreshPair, build_fresh_pairs, flatten_pairs as flatten_fresh  # noqa: E402
from run_learned_memory_benchmark import MODEL_NAME, RIDGE_ALPHA, SEEDS, CachedSentenceEncoder  # noqa: E402


SLOW_DECAY = 0.80
PROJECTION_DIMS = (8, 16, 32, 64)
PROJECTION_SEED = 5612026
STRUCTURE_PAIR_COUNT = 120
STRUCTURE_TRAIN_PAIRS = 80
OUT_DIR = VERSION_ROOT / "outputs" / "readout_temporal_diagnostic"


class DiagnosticRidge:
    def __init__(self, alpha: float = RIDGE_ALPHA) -> None:
        self.alpha = float(alpha)
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.weights: np.ndarray | None = None
        self.intercept = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "DiagnosticRidge":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64).reshape(-1)
        target = np.where(y > 0, 1.0, -1.0)
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

    def score(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None or self.weights is None:
            raise RuntimeError("probe not fit")
        xs = (np.asarray(x, dtype=np.float64) - self.mean) / self.scale
        return xs @ self.weights + self.intercept

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.score(x) >= 0.0).astype(np.int64)


def accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y) == np.asarray(pred)))


def standardize_scores(train_scores: np.ndarray, test_scores: np.ndarray) -> np.ndarray:
    train_scores = np.asarray(train_scores, dtype=np.float64)
    test_scores = np.asarray(test_scores, dtype=np.float64)
    mean = float(train_scores.mean())
    std = float(train_scores.std())
    if std < 1e-8:
        std = 1.0
    return (test_scores - mean) / std


def deterministic_projection(input_dim: int, output_dim: int, block: str) -> np.ndarray:
    salt = 0 if block == "fast" else 100000
    rng = np.random.default_rng(PROJECTION_SEED + salt + input_dim * 17 + output_dim)
    signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=(input_dim, output_dim))
    return (signs / math.sqrt(output_dim)).astype(np.float32)


def projected_dual(fast: np.ndarray, slow: np.ndarray, dim: int) -> np.ndarray:
    fast = np.asarray(fast, dtype=np.float32).reshape(-1)
    slow = np.asarray(slow, dtype=np.float32).reshape(-1)
    pf = deterministic_projection(fast.size, dim, "fast")
    ps = deterministic_projection(slow.size, dim, "slow")
    fast_proj = fast @ pf
    slow_proj = slow @ ps
    # Per-block L2 normalization occurs after label-free projection.
    fast_norm = float(np.linalg.norm(fast_proj))
    slow_norm = float(np.linalg.norm(slow_proj))
    if fast_norm > 0:
        fast_proj = fast_proj / fast_norm
    if slow_norm > 0:
        slow_proj = slow_proj / slow_norm
    return np.concatenate([fast_proj, slow_proj]).astype(np.float32)


def run_condition(model: DualTimescaleState, history: tuple[str, ...] | list[str], current: str, mode: str):
    model.reset_all()
    model.consume_sequence(list(history))
    if mode == "fast_reset":
        model.reset_fast()
    elif mode == "slow_reset":
        model.reset_slow()
    elif mode == "both_reset":
        model.reset_both()
    elif mode != "real":
        raise ValueError(mode)
    return model.consume_event(current)


def arm_key(arm) -> tuple[str, int]:
    return str(arm.pair_id), int(arm.label)


def pairs_for_domain(pairs: list[FreshPair], domain: str) -> list[FreshPair]:
    return [pair for pair in pairs if pair.domain == domain]


def semantic_diagnostic(
    seed: int,
    encoder: CachedSentenceEncoder,
    train_pairs: list[FreshPair],
    test_pairs: list[FreshPair],
) -> tuple[dict[str, float], list[dict[str, object]]]:
    model = DualTimescaleState(encoder, seed=seed, slow_decay=SLOW_DECAY)
    train_arms = flatten_fresh(train_pairs)
    test_arms = flatten_fresh(test_pairs)
    train_obs = {
        arm_key(arm): run_condition(model, arm.history, arm.current_text, "real")
        for arm in train_arms
    }
    test_obs = {
        arm_key(arm): run_condition(model, arm.history, arm.current_text, "real")
        for arm in test_arms
    }

    rows: list[dict[str, object]] = []
    aggregate: dict[str, list[float]] = {
        "fast_only": [],
        "slow_only": [],
        "raw_concat": [],
        "score_fusion": [],
        **{f"projected_{dim}": [] for dim in PROJECTION_DIMS},
    }

    for domain in sorted({pair.domain for pair in train_pairs}):
        d_train = flatten_fresh(pairs_for_domain(train_pairs, domain))
        d_test = flatten_fresh(pairs_for_domain(test_pairs, domain))
        y_train = np.asarray([arm.label for arm in d_train], dtype=np.int64)
        y_test = np.asarray([arm.label for arm in d_test], dtype=np.int64)

        def stack(arms, feature_fn):
            return np.stack([feature_fn(train_obs[arm_key(a)]) for a in arms])

        def stack_test(arms, feature_fn):
            return np.stack([feature_fn(test_obs[arm_key(a)]) for a in arms])

        x_fast_train = stack(d_train, fast_features)
        x_fast_test = stack_test(d_test, fast_features)
        x_slow_train = stack(d_train, slow_features)
        x_slow_test = stack_test(d_test, slow_features)
        x_raw_train = stack(d_train, dual_features)
        x_raw_test = stack_test(d_test, dual_features)

        fast_probe = DiagnosticRidge().fit(x_fast_train, y_train)
        slow_probe = DiagnosticRidge().fit(x_slow_train, y_train)
        raw_probe = DiagnosticRidge().fit(x_raw_train, y_train)

        fast_acc = accuracy(y_test, fast_probe.predict(x_fast_test))
        slow_acc = accuracy(y_test, slow_probe.predict(x_slow_test))
        raw_acc = accuracy(y_test, raw_probe.predict(x_raw_test))

        fast_train_score = fast_probe.score(x_fast_train)
        slow_train_score = slow_probe.score(x_slow_train)
        fast_test_z = standardize_scores(fast_train_score, fast_probe.score(x_fast_test))
        slow_test_z = standardize_scores(slow_train_score, slow_probe.score(x_slow_test))
        fusion_pred = ((fast_test_z + slow_test_z) / 2.0 >= 0.0).astype(np.int64)
        fusion_acc = accuracy(y_test, fusion_pred)

        result: dict[str, object] = {
            "seed": seed,
            "domain": domain,
            "fast_only": fast_acc,
            "slow_only": slow_acc,
            "raw_concat": raw_acc,
            "score_fusion": fusion_acc,
        }
        aggregate["fast_only"].append(fast_acc)
        aggregate["slow_only"].append(slow_acc)
        aggregate["raw_concat"].append(raw_acc)
        aggregate["score_fusion"].append(fusion_acc)

        for dim in PROJECTION_DIMS:
            x_proj_train = np.stack(
                [
                    projected_dual(
                        fast_features(train_obs[arm_key(a)]),
                        slow_features(train_obs[arm_key(a)]),
                        dim,
                    )
                    for a in d_train
                ]
            )
            x_proj_test = np.stack(
                [
                    projected_dual(
                        fast_features(test_obs[arm_key(a)]),
                        slow_features(test_obs[arm_key(a)]),
                        dim,
                    )
                    for a in d_test
                ]
            )
            projected_acc = accuracy(
                y_test,
                DiagnosticRidge().fit(x_proj_train, y_train).predict(x_proj_test),
            )
            result[f"projected_{dim}"] = projected_acc
            aggregate[f"projected_{dim}"].append(projected_acc)
        rows.append(result)

    return {name: float(np.mean(values)) for name, values in aggregate.items()}, rows


SYLLABLES = (
    "ka", "zu", "mi", "tor", "vel", "shi", "na", "dor", "pel", "rin",
    "qua", "fen", "lum", "bar", "cek", "ivo", "ryn", "sol", "tek", "uma",
)


def token_name(index: int, side: int) -> str:
    # Deterministic pair-specific pseudo-word; no identity repeats across pairs.
    a = SYLLABLES[(index * 7 + side * 3) % len(SYLLABLES)]
    b = SYLLABLES[(index * 11 + side * 5 + 1) % len(SYLLABLES)]
    c = SYLLABLES[(index * 13 + side * 7 + 2) % len(SYLLABLES)]
    return f"{a}{b}{c}-{index:03d}-{side}"


def structural_pair(pair_id: int) -> tuple[list[str], list[str], str, str, str]:
    a = token_name(pair_id, 0)
    b = token_name(pair_id, 1)
    event_a = f"The transient marker {a} appeared."
    event_b = f"The transient marker {b} appeared."
    prefix = f"Structural sequence case {pair_id:03d} begins with a neutral start marker."
    suffix = "The same neutral separator appears after the four transient markers."
    current = "The identical current observation is now presented."

    # Same 2A + 2B multiset and the same final current/suffix.
    class0 = [prefix, event_a, event_b, event_a, event_b, suffix]  # ABAB
    class1 = [prefix, event_a, event_a, event_b, event_b, suffix]  # AABB
    return class0, class1, current, event_a, event_b


def relational_structure_features(sequence: list[str], encoder: CachedSentenceEncoder) -> np.ndarray:
    # Benchmark-validating baseline: pairwise cosine among the four transient events.
    transient = sequence[1:5]
    vectors = np.stack([encoder.encode(text) for text in transient])
    values: list[float] = []
    for i in range(4):
        for j in range(i + 1, 4):
            values.append(float(np.dot(vectors[i], vectors[j])))
    return np.asarray(values, dtype=np.float32)


def temporal_diagnostic(seed: int, encoder: CachedSentenceEncoder) -> dict[str, float]:
    model = DualTimescaleState(encoder, seed=seed, slow_decay=SLOW_DECAY)
    rows: list[dict[str, object]] = []

    for pair_id in range(STRUCTURE_PAIR_COUNT):
        class0, class1, current, _, _ = structural_pair(pair_id)
        split = "train" if pair_id < STRUCTURE_TRAIN_PAIRS else "test"
        for label, sequence in ((0, class0), (1, class1)):
            real = run_condition(model, sequence, current, "real")
            fast_reset = run_condition(model, sequence, current, "fast_reset")
            slow_reset = run_condition(model, sequence, current, "slow_reset")
            rows.append(
                {
                    "pair_id": pair_id,
                    "split": split,
                    "label": label,
                    "fast": fast_features(real),
                    "slow": slow_features(real),
                    "raw_dual": dual_features(real),
                    "fast_reset_dual": dual_features(fast_reset),
                    "slow_reset_dual": dual_features(slow_reset),
                    "relational": relational_structure_features(sequence, encoder),
                    **{
                        f"projected_{dim}": projected_dual(
                            fast_features(real), slow_features(real), dim
                        )
                        for dim in PROJECTION_DIMS
                    },
                }
            )

    train = [row for row in rows if row["split"] == "train"]
    test = [row for row in rows if row["split"] == "test"]
    y_train = np.asarray([row["label"] for row in train], dtype=np.int64)
    y_test = np.asarray([row["label"] for row in test], dtype=np.int64)

    def matrix(source, field):
        return np.stack([np.asarray(row[field], dtype=np.float32) for row in source])

    fields = ["fast", "slow", "raw_dual", "relational", *[f"projected_{d}" for d in PROJECTION_DIMS]]
    result: dict[str, float] = {}
    probes: dict[str, DiagnosticRidge] = {}
    for field in fields:
        probe = DiagnosticRidge().fit(matrix(train, field), y_train)
        probes[field] = probe
        result[field] = accuracy(y_test, probe.predict(matrix(test, field)))

    raw_probe = probes["raw_dual"]
    result["raw_dual_fast_reset"] = accuracy(
        y_test, raw_probe.predict(matrix(test, "fast_reset_dual"))
    )
    result["raw_dual_slow_reset"] = accuracy(
        y_test, raw_probe.predict(matrix(test, "slow_reset_dual"))
    )
    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    train_pairs, test_pairs = build_fresh_pairs()
    encoder = CachedSentenceEncoder(MODEL_NAME)
    texts: set[str] = set()
    for arm in flatten_fresh(train_pairs + test_pairs):
        texts.update(arm.history)
        texts.add(arm.current_text)
    for pair_id in range(STRUCTURE_PAIR_COUNT):
        c0, c1, current, _, _ = structural_pair(pair_id)
        texts.update(c0)
        texts.update(c1)
        texts.add(current)
    encoder.preload(texts)

    seed_rows: list[dict[str, object]] = []
    semantic_domain_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        semantic, domain_rows = semantic_diagnostic(seed, encoder, train_pairs, test_pairs)
        temporal = temporal_diagnostic(seed, encoder)
        seed_rows.append(
            {
                "seed": seed,
                **{f"semantic_{k}": v for k, v in semantic.items()},
                **{f"temporal_{k}": v for k, v in temporal.items()},
            }
        )
        semantic_domain_rows.extend(domain_rows)

    def mean(field: str) -> float:
        return float(np.mean([float(row[field]) for row in seed_rows]))

    semantic_projected = {
        str(dim): mean(f"semantic_projected_{dim}") for dim in PROJECTION_DIMS
    }
    temporal_projected = {
        str(dim): mean(f"temporal_projected_{dim}") for dim in PROJECTION_DIMS
    }
    best_semantic_dim = max(semantic_projected, key=semantic_projected.get)
    best_temporal_dim = max(temporal_projected, key=temporal_projected.get)

    slow_semantic = mean("semantic_slow_only")
    raw_semantic = mean("semantic_raw_concat")
    fusion_semantic = mean("semantic_score_fusion")
    best_balanced_semantic = semantic_projected[best_semantic_dim]

    fast_temporal = mean("temporal_fast")
    slow_temporal = mean("temporal_slow")
    relational_temporal = mean("temporal_relational")
    best_balanced_temporal = temporal_projected[best_temporal_dim]

    if best_balanced_semantic >= slow_semantic - 0.03 or fusion_semantic >= slow_semantic - 0.03:
        semantic_diagnosis = "naive_concat_readout_or_dimensionality_problem"
    elif max(best_balanced_semantic, fusion_semantic) > raw_semantic + 0.05:
        semantic_diagnosis = "partial_readout_swamping_plus_fast_nuisance"
    else:
        semantic_diagnosis = "fast_block_adds_semantic_nuisance_without_complementarity"

    if relational_temporal < 0.90:
        temporal_diagnosis = "benchmark_construction_invalid"
    elif fast_temporal >= slow_temporal + 0.08:
        temporal_diagnosis = "fast_state_has_incremental_identity_invariant_structure"
    elif slow_temporal >= fast_temporal - 0.03:
        temporal_diagnosis = "slow_memory_remains_sufficient_for_tested_structure"
    else:
        temporal_diagnosis = "mixed_or_unresolved"

    summary = {
        "version": "v5.6.1",
        "purpose": "diagnose v5.6 semantic concat and temporal-control failures",
        "state_generators_changed_from_v5_6": False,
        "semantic": {
            "fast_only": mean("semantic_fast_only"),
            "slow_only": slow_semantic,
            "raw_concat": raw_semantic,
            "score_fusion": fusion_semantic,
            "projected_equal_dim": semantic_projected,
            "best_projected_dim": int(best_semantic_dim),
            "best_projected_accuracy": best_balanced_semantic,
            "diagnosis": semantic_diagnosis,
        },
        "temporal_structure": {
            "pair_count": STRUCTURE_PAIR_COUNT,
            "train_pairs": STRUCTURE_TRAIN_PAIRS,
            "test_pairs": STRUCTURE_PAIR_COUNT - STRUCTURE_TRAIN_PAIRS,
            "train_test_token_identities_disjoint": True,
            "fast_only": fast_temporal,
            "slow_only": slow_temporal,
            "raw_dual": mean("temporal_raw_dual"),
            "raw_dual_fast_reset": mean("temporal_raw_dual_fast_reset"),
            "raw_dual_slow_reset": mean("temporal_raw_dual_slow_reset"),
            "relational_structure_baseline": relational_temporal,
            "projected_equal_dim": temporal_projected,
            "best_projected_dim": int(best_temporal_dim),
            "best_projected_accuracy": best_balanced_temporal,
            "diagnosis": temporal_diagnosis,
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "per_seed_metrics.csv", seed_rows)
    write_csv(OUT_DIR / "semantic_per_domain.csv", semantic_domain_rows)
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
