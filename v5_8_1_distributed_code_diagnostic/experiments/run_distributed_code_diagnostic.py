from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve()
VERSION_ROOT = HERE.parents[1]
REPO_ROOT = HERE.parents[2]
for path in (
    REPO_ROOT,
    VERSION_ROOT,
    REPO_ROOT / "v5_2_learned_memory" / "experiments",
    REPO_ROOT / "v5_6_1_readout_temporal_diagnostic" / "experiments",
    REPO_ROOT / "v5_7_residual_fast_dynamics",
    REPO_ROOT / "v5_8_adaptive_fast_dynamics",
):
    sys.path.insert(0, str(path))

from adaptive_state import AdaptiveResidualState  # noqa: E402
from temporal_suite import PAIR_COUNT, TASKS, TRAIN_PAIRS, build_case  # noqa: E402
from residual_state import ResidualDrivenState  # noqa: E402
from run_learned_memory_benchmark import MODEL_NAME, RIDGE_ALPHA, SEEDS, CachedSentenceEncoder  # noqa: E402
from run_readout_temporal_diagnostic import DiagnosticRidge, accuracy  # noqa: E402


ADAPTATION_DECAY = 0.995
ADAPTATION_STRENGTH = 0.20
SLOW_DECAY = 0.80
PERMUTATION_SEED = 5812026
OUT_DIR = VERSION_ROOT / "outputs" / "distributed_code_diagnostic"


def run_adaptive(
    model: AdaptiveResidualState,
    history: tuple[str, ...],
    current: str,
    *,
    reset_fast_before_current: bool = False,
):
    model.reset_all()
    model.consume_sequence(history)
    if reset_fast_before_current:
        model.reset_fast()
    return model.consume_event(current)


def run_v57(model: ResidualDrivenState, history: tuple[str, ...], current: str):
    model.reset_all()
    model.consume_sequence(history)
    return model.consume_event(current)


def raw_trace(observation) -> np.ndarray:
    return observation.fast_trace.states.reshape(-1).astype(np.float32, copy=False)


def final_state(observation) -> np.ndarray:
    return observation.fast_trace.states[-1].astype(np.float32, copy=True)


def mean_state(observation) -> np.ndarray:
    return observation.fast_trace.states.mean(axis=0).astype(np.float32)


def adaptation_state(observation) -> np.ndarray:
    return observation.adaptation_state.astype(np.float32, copy=True)


def adaptation_moments(observation) -> np.ndarray:
    a = observation.adaptation_state.astype(np.float32, copy=False)
    return np.asarray(
        [a.mean(), a.std(), np.abs(a).mean(), np.sqrt(np.mean(a * a) + 1e-12)],
        dtype=np.float32,
    )


def permute_trace(observation, permutation: np.ndarray) -> np.ndarray:
    return observation.fast_trace.states[:, permutation].reshape(-1).astype(
        np.float32, copy=False
    )


def matrix(rows: list[dict[str, object]], field: str) -> np.ndarray:
    return np.stack([np.asarray(row[field], dtype=np.float32) for row in rows])


def labels(rows: list[dict[str, object]]) -> np.ndarray:
    return np.asarray([int(row["label"]) for row in rows], dtype=np.int64)


def evaluate_task(task: str, seed: int, encoder: CachedSentenceEncoder) -> dict[str, float]:
    adaptive = AdaptiveResidualState(
        encoder,
        seed=seed,
        adaptation_strength=ADAPTATION_STRENGTH,
        adaptation_decay=ADAPTATION_DECAY,
        slow_decay=SLOW_DECAY,
        use_recurrence=True,
    )
    adaptation_only = AdaptiveResidualState(
        encoder,
        seed=seed,
        adaptation_strength=ADAPTATION_STRENGTH,
        adaptation_decay=ADAPTATION_DECAY,
        slow_decay=SLOW_DECAY,
        use_recurrence=False,
    )
    v57 = ResidualDrivenState(encoder, seed=seed, slow_decay=SLOW_DECAY)
    permutation = np.random.default_rng(PERMUTATION_SEED + seed).permutation(
        adaptive.config.num_neurons
    )

    rows: list[dict[str, object]] = []
    for pair_id in range(PAIR_COUNT):
        case = build_case(task, pair_id)
        for label, sequence, opposite in (
            (0, case.class0, case.class1),
            (1, case.class1, case.class0),
        ):
            real = run_adaptive(adaptive, sequence, case.current)
            reset = run_adaptive(
                adaptive, sequence, case.current, reset_fast_before_current=True
            )
            wrong = run_adaptive(adaptive, opposite, case.current)
            no_recurrence = run_adaptive(adaptation_only, sequence, case.current)
            frozen = run_v57(v57, sequence, case.current)
            rows.append(
                {
                    "pair_id": pair_id,
                    "label": label,
                    "adaptive_raw": raw_trace(real),
                    "adaptive_raw_permuted": permute_trace(real, permutation),
                    "adaptive_reset_raw": raw_trace(reset),
                    "opposite_history_raw": raw_trace(wrong),
                    "v57_raw": raw_trace(frozen),
                    "adaptation_only_raw": raw_trace(no_recurrence),
                    "final_state": final_state(real),
                    "mean_state": mean_state(real),
                    "adaptation_state": adaptation_state(real),
                    "adaptation_moments": adaptation_moments(real),
                }
            )

    train = [row for row in rows if int(row["pair_id"]) < TRAIN_PAIRS]
    test = [row for row in rows if int(row["pair_id"]) >= TRAIN_PAIRS]
    y_train = labels(train)
    y_test = labels(test)

    independent_fields = (
        "adaptive_raw",
        "v57_raw",
        "adaptation_only_raw",
        "final_state",
        "mean_state",
        "adaptation_state",
        "adaptation_moments",
    )
    probes = {
        field: DiagnosticRidge(RIDGE_ALPHA).fit(matrix(train, field), y_train)
        for field in independent_fields
    }
    result = {
        field: accuracy(y_test, probe.predict(matrix(test, field)))
        for field, probe in probes.items()
    }

    adaptive_probe = probes["adaptive_raw"]
    result["adaptive_reset_raw"] = accuracy(
        y_test, adaptive_probe.predict(matrix(test, "adaptive_reset_raw"))
    )
    result["opposite_history_raw"] = accuracy(
        y_test, adaptive_probe.predict(matrix(test, "opposite_history_raw"))
    )
    result["test_only_permutation"] = accuracy(
        y_test, adaptive_probe.predict(matrix(test, "adaptive_raw_permuted"))
    )

    joint_probe = DiagnosticRidge(RIDGE_ALPHA).fit(
        matrix(train, "adaptive_raw_permuted"), y_train
    )
    result["joint_train_test_permutation"] = accuracy(
        y_test, joint_probe.predict(matrix(test, "adaptive_raw_permuted"))
    )
    return result


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    encoder = CachedSentenceEncoder(MODEL_NAME)
    texts: set[str] = set()
    for task in TASKS:
        for pair_id in range(PAIR_COUNT):
            case = build_case(task, pair_id)
            texts.update(case.class0)
            texts.update(case.class1)
            texts.add(case.current)
    encoder.preload(texts)

    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        for task in TASKS:
            rows.append({"seed": seed, "task": task, **evaluate_task(task, seed, encoder)})

    metric_names = [key for key in rows[0] if key not in {"seed", "task"}]
    mean = {
        name: float(np.mean([float(row[name]) for row in rows]))
        for name in metric_names
    }
    per_task_raw = {
        task: float(
            np.mean([float(row["adaptive_raw"]) for row in rows if row["task"] == task])
        )
        for task in TASKS
    }
    per_seed_raw = {
        str(seed): float(
            np.mean([float(row["adaptive_raw"]) for row in rows if int(row["seed"]) == seed])
        )
        for seed in SEEDS
    }

    diagnostic = {
        "adaptive_raw_at_least_0_80": mean["adaptive_raw"] >= 0.80,
        "adaptive_beats_v57_raw_by_0_10": mean["adaptive_raw"] - mean["v57_raw"] >= 0.10,
        "adaptive_beats_adaptation_only_raw_by_0_03": mean["adaptive_raw"] - mean["adaptation_only_raw"] >= 0.03,
        "fast_reset_drop_at_least_0_15": mean["adaptive_raw"] - mean["adaptive_reset_raw"] >= 0.15,
        "opposite_history_at_most_0_35": mean["opposite_history_raw"] <= 0.35,
        "joint_permutation_within_0_01": abs(mean["adaptive_raw"] - mean["joint_train_test_permutation"]) <= 0.01,
        "test_only_permutation_drop_at_least_0_15": mean["adaptive_raw"] - mean["test_only_permutation"] >= 0.15,
    }
    diagnostic["all_diagnostic_conditions"] = all(diagnostic.values())

    summary = {
        "version": "v5.8.1",
        "purpose": "diagnose whether v5.8 adaptation creates a stable distributed neuron-specific temporal code",
        "state_generator_changed_from_v5_8": False,
        "adaptation": {
            "decay": ADAPTATION_DECAY,
            "strength": ADAPTATION_STRENGTH,
        },
        "mean_accuracy": mean,
        "per_task_adaptive_raw": per_task_raw,
        "per_seed_adaptive_raw": per_seed_raw,
        "gaps": {
            "adaptive_minus_v57_raw": mean["adaptive_raw"] - mean["v57_raw"],
            "adaptive_minus_adaptation_only_raw": mean["adaptive_raw"] - mean["adaptation_only_raw"],
            "adaptive_minus_reset_raw": mean["adaptive_raw"] - mean["adaptive_reset_raw"],
            "adaptive_minus_test_only_permutation": mean["adaptive_raw"] - mean["test_only_permutation"],
        },
        "diagnostic_conditions": diagnostic,
        "claim_boundary": "diagnostic on the v5.8 development suite; requires fresh confirmatory patterns before mechanism claim",
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "per_seed_task_metrics.csv", rows)
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
