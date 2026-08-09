from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve()
VERSION_ROOT = HERE.parents[1]
REPO_ROOT = HERE.parents[2]
V52_EXPERIMENTS = REPO_ROOT / "v5_2_learned_memory" / "experiments"
V54_ROOT = REPO_ROOT / "v5_4_fresh_confirmatory"
V56_ROOT = REPO_ROOT / "v5_6_dual_timescale_state"
V561_EXPERIMENTS = REPO_ROOT / "v5_6_1_readout_temporal_diagnostic" / "experiments"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(VERSION_ROOT))
sys.path.insert(0, str(V52_EXPERIMENTS))
sys.path.insert(0, str(V54_ROOT))
sys.path.insert(0, str(V56_ROOT))
sys.path.insert(0, str(V561_EXPERIMENTS))

from residual_state import ResidualDrivenState, fast_features as residual_fast_features, slow_features  # noqa: E402
from dual_state import DualTimescaleState, fast_features as raw_fast_features  # noqa: E402
from fresh_fixture import FreshPair, build_fresh_pairs, flatten_pairs as flatten_fresh  # noqa: E402
from run_learned_memory_benchmark import MODEL_NAME, RIDGE_ALPHA, SEEDS, CachedSentenceEncoder  # noqa: E402
from run_readout_temporal_diagnostic import (  # noqa: E402
    DiagnosticRidge,
    STRUCTURE_PAIR_COUNT,
    STRUCTURE_TRAIN_PAIRS,
    accuracy,
    relational_structure_features,
    structural_pair,
)


SLOW_DECAY = 0.80
OUT_DIR = VERSION_ROOT / "outputs" / "residual_fast_benchmark"


def pairs_for_domain(pairs: list[FreshPair], domain: str) -> list[FreshPair]:
    return [pair for pair in pairs if pair.domain == domain]


def arm_key(arm) -> tuple[str, int]:
    return str(arm.pair_id), int(arm.label)


def run_residual_condition(
    model: ResidualDrivenState,
    history: tuple[str, ...] | list[str],
    current: str,
    *,
    reset_fast_before_current: bool,
):
    model.reset_all()
    model.consume_sequence(list(history))
    if reset_fast_before_current:
        model.reset_fast()
    return model.consume_event(current)


def run_raw_condition(
    model: DualTimescaleState,
    history: tuple[str, ...] | list[str],
    current: str,
):
    model.reset_all()
    model.consume_sequence(list(history))
    return model.consume_event(current)


def semantic_seed(
    seed: int,
    encoder: CachedSentenceEncoder,
    train_pairs: list[FreshPair],
    test_pairs: list[FreshPair],
) -> dict[str, float]:
    residual_model = ResidualDrivenState(encoder, seed=seed, slow_decay=SLOW_DECAY)
    raw_model = DualTimescaleState(encoder, seed=seed, slow_decay=SLOW_DECAY)
    train_arms = flatten_fresh(train_pairs)
    test_arms = flatten_fresh(test_pairs)

    train_residual = {
        arm_key(arm): run_residual_condition(
            residual_model, arm.history, arm.current_text, reset_fast_before_current=False
        )
        for arm in train_arms
    }
    test_residual = {
        arm_key(arm): run_residual_condition(
            residual_model, arm.history, arm.current_text, reset_fast_before_current=False
        )
        for arm in test_arms
    }
    train_raw = {
        arm_key(arm): run_raw_condition(raw_model, arm.history, arm.current_text)
        for arm in train_arms
    }
    test_raw = {
        arm_key(arm): run_raw_condition(raw_model, arm.history, arm.current_text)
        for arm in test_arms
    }

    slow_scores: list[float] = []
    residual_fast_scores: list[float] = []
    raw_fast_scores: list[float] = []

    for domain in sorted({pair.domain for pair in train_pairs}):
        d_train = flatten_fresh(pairs_for_domain(train_pairs, domain))
        d_test = flatten_fresh(pairs_for_domain(test_pairs, domain))
        y_train = np.asarray([arm.label for arm in d_train], dtype=np.int64)
        y_test = np.asarray([arm.label for arm in d_test], dtype=np.int64)

        def stack(mapping, arms, feature_fn):
            return np.stack([feature_fn(mapping[arm_key(arm)]) for arm in arms])

        slow_probe = DiagnosticRidge(RIDGE_ALPHA).fit(
            stack(train_residual, d_train, slow_features), y_train
        )
        residual_probe = DiagnosticRidge(RIDGE_ALPHA).fit(
            stack(train_residual, d_train, residual_fast_features), y_train
        )
        raw_probe = DiagnosticRidge(RIDGE_ALPHA).fit(
            stack(train_raw, d_train, raw_fast_features), y_train
        )

        slow_scores.append(
            accuracy(y_test, slow_probe.predict(stack(test_residual, d_test, slow_features)))
        )
        residual_fast_scores.append(
            accuracy(
                y_test,
                residual_probe.predict(
                    stack(test_residual, d_test, residual_fast_features)
                ),
            )
        )
        raw_fast_scores.append(
            accuracy(y_test, raw_probe.predict(stack(test_raw, d_test, raw_fast_features)))
        )

    return {
        "semantic_slow": float(np.mean(slow_scores)),
        "semantic_residual_fast": float(np.mean(residual_fast_scores)),
        "semantic_raw_fast": float(np.mean(raw_fast_scores)),
    }


def direct_residual_change_features(
    encoder: CachedSentenceEncoder,
    history: list[str],
    current: str,
) -> np.ndarray:
    """Simple label-free transient baseline using only residual magnitudes."""

    slow = np.zeros(encoder.output_dim, dtype=np.float32)
    norms: list[float] = []
    delta_norms: list[float] = []
    previous_residual: np.ndarray | None = None

    for text in [*history, current]:
        embedding = encoder.encode(text).astype(np.float32, copy=False)
        residual = (embedding - slow).astype(np.float32, copy=False)
        norms.append(float(np.linalg.norm(residual)))
        if previous_residual is None:
            delta_norms.append(0.0)
        else:
            delta_norms.append(float(np.linalg.norm(residual - previous_residual)))
        previous_residual = residual
        slow = (SLOW_DECAY * slow + (1.0 - SLOW_DECAY) * embedding).astype(
            np.float32, copy=False
        )

    return np.asarray([*norms, *delta_norms], dtype=np.float32)


def temporal_seed(seed: int, encoder: CachedSentenceEncoder) -> dict[str, float]:
    residual_model = ResidualDrivenState(encoder, seed=seed, slow_decay=SLOW_DECAY)
    raw_model = DualTimescaleState(encoder, seed=seed, slow_decay=SLOW_DECAY)
    rows: list[dict[str, object]] = []

    for pair_id in range(STRUCTURE_PAIR_COUNT):
        class0, class1, current, _, _ = structural_pair(pair_id)
        split = "train" if pair_id < STRUCTURE_TRAIN_PAIRS else "test"
        for label, sequence in ((0, class0), (1, class1)):
            residual_real = run_residual_condition(
                residual_model,
                sequence,
                current,
                reset_fast_before_current=False,
            )
            residual_reset = run_residual_condition(
                residual_model,
                sequence,
                current,
                reset_fast_before_current=True,
            )
            raw_real = run_raw_condition(raw_model, sequence, current)
            rows.append(
                {
                    "split": split,
                    "label": label,
                    "residual_fast": residual_fast_features(residual_real),
                    "residual_fast_reset": residual_fast_features(residual_reset),
                    "raw_fast": raw_fast_features(raw_real),
                    "slow": slow_features(residual_real),
                    "direct_residual": direct_residual_change_features(
                        encoder, sequence, current
                    ),
                    "relational": relational_structure_features(sequence, encoder),
                }
            )

    train = [row for row in rows if row["split"] == "train"]
    test = [row for row in rows if row["split"] == "test"]
    y_train = np.asarray([row["label"] for row in train], dtype=np.int64)
    y_test = np.asarray([row["label"] for row in test], dtype=np.int64)

    def matrix(source, field):
        return np.stack([np.asarray(row[field], dtype=np.float32) for row in source])

    fields = ("residual_fast", "raw_fast", "slow", "direct_residual", "relational")
    probes = {
        field: DiagnosticRidge(RIDGE_ALPHA).fit(matrix(train, field), y_train)
        for field in fields
    }
    result = {
        field: accuracy(y_test, probe.predict(matrix(test, field)))
        for field, probe in probes.items()
    }
    result["residual_fast_reset"] = accuracy(
        y_test,
        probes["residual_fast"].predict(matrix(test, "residual_fast_reset")),
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
    for seed in SEEDS:
        semantic = semantic_seed(seed, encoder, train_pairs, test_pairs)
        temporal = temporal_seed(seed, encoder)
        seed_rows.append(
            {
                "seed": seed,
                **semantic,
                **{f"temporal_{key}": value for key, value in temporal.items()},
            }
        )

    def mean(field: str) -> float:
        return float(np.mean([float(row[field]) for row in seed_rows]))

    structural = mean("temporal_residual_fast")
    raw_structural = mean("temporal_raw_fast")
    slow_structural = mean("temporal_slow")
    reset_structural = mean("temporal_residual_fast_reset")
    relational = mean("temporal_relational")
    slow_semantic = mean("semantic_slow")
    direct_residual = mean("temporal_direct_residual")

    gates = {
        "residual_fast_structural_at_least_0_70": structural >= 0.70,
        "residual_fast_beats_raw_fast_by_0_10": structural - raw_structural >= 0.10,
        "residual_fast_beats_slow_by_0_12": structural - slow_structural >= 0.12,
        "fast_reset_reduces_structural_by_0_12": structural - reset_structural >= 0.12,
        "relational_validity_at_least_0_95": relational >= 0.95,
        "slow_semantic_remains_at_least_0_78": slow_semantic >= 0.78,
    }
    gates["all_primary_gates"] = all(gates.values())

    summary = {
        "version": "v5.7",
        "purpose": "residual-driven fast dynamics development benchmark",
        "labels_used_to_construct_state": False,
        "recurrent_topology_changed_from_v5_0": False,
        "slow_memory_changed_from_v5_6": False,
        "mean": {
            "semantic_slow": slow_semantic,
            "semantic_residual_fast": mean("semantic_residual_fast"),
            "semantic_raw_fast": mean("semantic_raw_fast"),
            "structural_residual_fast": structural,
            "structural_raw_fast": raw_structural,
            "structural_slow": slow_structural,
            "structural_residual_fast_reset": reset_structural,
            "direct_residual_change_baseline": direct_residual,
            "relational_structure_baseline": relational,
        },
        "gaps": {
            "residual_fast_minus_raw_fast": structural - raw_structural,
            "residual_fast_minus_slow": structural - slow_structural,
            "residual_fast_minus_reset": structural - reset_structural,
            "residual_fast_minus_direct_residual": structural - direct_residual,
        },
        "acceptance": gates,
        "complexity_check": {
            "recurrent_beats_direct_residual": structural > direct_residual,
            "direct_residual_advantage_if_positive": direct_residual - structural,
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "per_seed_metrics.csv", seed_rows)
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
