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
    REPO_ROOT / "v5_7_residual_fast_dynamics",
    REPO_ROOT / "v5_8_adaptive_fast_dynamics",
):
    sys.path.insert(0, str(path))

from emonet_v5 import DynamicsConfig  # noqa: E402
from residual_state import ResidualDrivenState  # noqa: E402
from adaptive_state import AdaptiveResidualState  # noqa: E402
from vector_world import (  # noqa: E402
    PAIR_COUNT,
    RECURRENT_SEEDS,
    TASKS,
    TRAIN_PAIRS,
    WORLD_SEEDS,
    build_case,
    build_vector_world,
    relational_features,
)

SLOW_DECAY = 0.80
ADAPTATION_DECAY = 0.995
ADAPTATION_STRENGTH = 0.20
RIDGE_ALPHA = 1.0
OUT_DIR = VERSION_ROOT / "outputs" / "encoder_free_temporal"


class RidgeProbe:
    def __init__(self, alpha: float = RIDGE_ALPHA) -> None:
        self.alpha = float(alpha)
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.weights: np.ndarray | None = None
        self.intercept = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeProbe":
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
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
            self.weights = np.linalg.solve(xs.T @ xs + self.alpha * np.eye(d), xs.T @ yc)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None or self.weights is None:
            raise RuntimeError("probe not fit")
        xs = (np.asarray(x, dtype=np.float64) - self.mean) / self.scale
        return (xs @ self.weights + self.intercept >= 0.0).astype(np.int64)


def accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y) == np.asarray(pred)))


def raw_trace(observation) -> np.ndarray:
    return observation.fast_trace.states.reshape(-1).astype(np.float32, copy=False)


def population_moments(observation) -> np.ndarray:
    states = observation.fast_trace.states.astype(np.float32, copy=False)
    mean = states.mean(axis=1)
    std = states.std(axis=1)
    mean_abs = np.abs(states).mean(axis=1)
    rms = np.sqrt(np.mean(states * states, axis=1) + 1e-12)
    return np.stack([mean, std, mean_abs, rms], axis=1).reshape(-1).astype(np.float32)


def adaptation_state(observation) -> np.ndarray:
    return observation.adaptation_state.astype(np.float32, copy=True)


def adaptation_moments(observation) -> np.ndarray:
    state = observation.adaptation_state.astype(np.float32, copy=False)
    return np.asarray(
        [state.mean(), state.std(), np.abs(state).mean(), np.sqrt(np.mean(state * state) + 1e-12)],
        dtype=np.float32,
    )


def consume_v57(model: ResidualDrivenState, history: tuple[str, ...], current: str):
    model.reset_all()
    model.consume_sequence(history)
    slow_before = model.slow.state.copy()
    real = model.consume_event(current)
    model.slow.state = slow_before.copy()
    model.reset_fast()
    reset = model.consume_event(current)
    return real, reset


def consume_v58(model: AdaptiveResidualState, history: tuple[str, ...], current: str):
    model.reset_all()
    model.consume_sequence(history)
    slow_before = model.slow.state.copy()
    real = model.consume_event(current)
    model.slow.state = slow_before.copy()
    model.reset_fast()
    reset = model.consume_event(current)
    return real, reset


def consume_real_adaptive(model: AdaptiveResidualState, history: tuple[str, ...], current: str):
    model.reset_all()
    model.consume_sequence(history)
    return model.consume_event(current)


def build_world_seed_rows(world_seed: int, recurrent_seed: int, task: str) -> list[dict[str, object]]:
    encoder = build_vector_world(world_seed)
    config = DynamicsConfig(seed=recurrent_seed)
    v57 = ResidualDrivenState(encoder, seed=recurrent_seed, slow_decay=SLOW_DECAY, dynamics_config=config)
    v58 = AdaptiveResidualState(
        encoder,
        seed=recurrent_seed,
        adaptation_strength=ADAPTATION_STRENGTH,
        adaptation_decay=ADAPTATION_DECAY,
        slow_decay=SLOW_DECAY,
        use_recurrence=True,
        dynamics_config=config,
    )
    adaptation_only = AdaptiveResidualState(
        encoder,
        seed=recurrent_seed,
        adaptation_strength=ADAPTATION_STRENGTH,
        adaptation_decay=ADAPTATION_DECAY,
        slow_decay=SLOW_DECAY,
        use_recurrence=False,
        dynamics_config=config,
    )

    rows: list[dict[str, object]] = []
    for pair_id in range(PAIR_COUNT):
        case = build_case(task, pair_id)
        pair_observations: dict[int, dict[str, np.ndarray]] = {}
        for label, history in ((0, case.class0), (1, case.class1)):
            old_real, old_reset = consume_v57(v57, history, case.current)
            adaptive_real, adaptive_reset = consume_v58(v58, history, case.current)
            no_recurrence = consume_real_adaptive(adaptation_only, history, case.current)
            pair_observations[label] = {
                "v57_raw": raw_trace(old_real),
                "v57_moments": population_moments(old_real),
                "v57_reset": raw_trace(old_reset),
                "v58_raw": raw_trace(adaptive_real),
                "v58_moments": population_moments(adaptive_real),
                "v58_reset": raw_trace(adaptive_reset),
                "adaptation_state": adaptation_state(adaptive_real),
                "adaptation_moments": adaptation_moments(adaptive_real),
                "adaptation_only_raw": raw_trace(no_recurrence),
                "relational": relational_features(history, encoder),
            }

        for label in (0, 1):
            own = pair_observations[label]
            opposite = pair_observations[1 - label]
            rows.append(
                {
                    "world": world_seed,
                    "seed": recurrent_seed,
                    "task": task,
                    "pair_id": pair_id,
                    "label": label,
                    **own,
                    "v57_opposite": opposite["v57_raw"].copy(),
                    "v58_opposite": opposite["v58_raw"].copy(),
                }
            )
    return rows


def matrix(rows: list[dict[str, object]], field: str) -> np.ndarray:
    return np.stack([np.asarray(row[field], dtype=np.float32) for row in rows])


def labels(rows: list[dict[str, object]]) -> np.ndarray:
    return np.asarray([int(row["label"]) for row in rows], dtype=np.int64)


def evaluate_leave_world_out(rows: list[dict[str, object]], recurrent_seed: int, task: str, held_world: int) -> dict[str, float]:
    subset = [row for row in rows if int(row["seed"]) == recurrent_seed and row["task"] == task]
    train = [
        row for row in subset
        if int(row["world"]) != held_world and int(row["pair_id"]) < TRAIN_PAIRS
    ]
    test = [
        row for row in subset
        if int(row["world"]) == held_world and int(row["pair_id"]) >= TRAIN_PAIRS
    ]
    y_train = labels(train)
    y_test = labels(test)

    independent = (
        "v57_raw",
        "v57_moments",
        "v58_raw",
        "v58_moments",
        "adaptation_state",
        "adaptation_moments",
        "adaptation_only_raw",
        "relational",
    )
    probes = {
        field: RidgeProbe().fit(matrix(train, field), y_train)
        for field in independent
    }
    result = {
        field: accuracy(y_test, probe.predict(matrix(test, field)))
        for field, probe in probes.items()
    }
    result["v57_reset"] = accuracy(
        y_test, probes["v57_raw"].predict(matrix(test, "v57_reset"))
    )
    result["v57_opposite"] = accuracy(
        y_test, probes["v57_raw"].predict(matrix(test, "v57_opposite"))
    )
    result["v58_reset"] = accuracy(
        y_test, probes["v58_raw"].predict(matrix(test, "v58_reset"))
    )
    result["v58_opposite"] = accuracy(
        y_test, probes["v58_raw"].predict(matrix(test, "v58_opposite"))
    )
    return result


def evaluate_within_world(rows: list[dict[str, object]], recurrent_seed: int, task: str, world: int, field: str) -> float:
    subset = [
        row for row in rows
        if int(row["seed"]) == recurrent_seed and row["task"] == task and int(row["world"]) == world
    ]
    train = [row for row in subset if int(row["pair_id"]) < TRAIN_PAIRS]
    test = [row for row in subset if int(row["pair_id"]) >= TRAIN_PAIRS]
    probe = RidgeProbe().fit(matrix(train, field), labels(train))
    return accuracy(labels(test), probe.predict(matrix(test, field)))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def gate(mean: dict[str, float], per_task: dict[str, float], prefix: str) -> dict[str, bool]:
    main = mean[f"{prefix}_raw"]
    return {
        "macro_at_least_0_70": main >= 0.70,
        "reset_drop_at_least_0_15": main - mean[f"{prefix}_reset"] >= 0.15,
        "opposite_at_most_0_35": mean[f"{prefix}_opposite"] <= 0.35,
        "at_least_3_of_4_tasks_at_0_65": sum(value >= 0.65 for value in per_task.values()) >= 3,
        "relational_validity_at_least_0_95": mean["relational"] >= 0.95,
    }


def main() -> None:
    all_rows: list[dict[str, object]] = []
    for world in WORLD_SEEDS:
        for seed in RECURRENT_SEEDS:
            for task in TASKS:
                all_rows.extend(build_world_seed_rows(world, seed, task))

    eval_rows: list[dict[str, object]] = []
    for seed in RECURRENT_SEEDS:
        for task in TASKS:
            for held_world in WORLD_SEEDS:
                eval_rows.append(
                    {
                        "seed": seed,
                        "task": task,
                        "held_world": held_world,
                        **evaluate_leave_world_out(all_rows, seed, task, held_world),
                    }
                )

    metric_names = [key for key in eval_rows[0] if key not in {"seed", "task", "held_world"}]
    mean = {
        metric: float(np.mean([float(row[metric]) for row in eval_rows]))
        for metric in metric_names
    }
    per_task_v57 = {
        task: float(np.mean([float(row["v57_raw"]) for row in eval_rows if row["task"] == task]))
        for task in TASKS
    }
    per_task_v58 = {
        task: float(np.mean([float(row["v58_raw"]) for row in eval_rows if row["task"] == task]))
        for task in TASKS
    }

    within_rows: list[dict[str, object]] = []
    for seed in RECURRENT_SEEDS:
        for task in TASKS:
            for world in WORLD_SEEDS:
                within_rows.append(
                    {
                        "seed": seed,
                        "task": task,
                        "world": world,
                        "v57_raw": evaluate_within_world(all_rows, seed, task, world, "v57_raw"),
                        "v58_raw": evaluate_within_world(all_rows, seed, task, world, "v58_raw"),
                    }
                )

    v57_gate = gate(mean, per_task_v57, "v57")
    v57_gate["all"] = all(v57_gate.values())
    v58_gate = gate(mean, per_task_v58, "v58")
    v58_gate["all"] = all(v58_gate.values())

    summary = {
        "version": "v5.9",
        "purpose": "encoder-free mechanistic temporal abstraction benchmark",
        "language_encoder_used": False,
        "input_dimension": 384,
        "vector_worlds": list(WORLD_SEEDS),
        "recurrent_seeds": list(RECURRENT_SEEDS),
        "tasks": list(TASKS),
        "primary_protocol": "leave-one-vector-world-out",
        "mean_accuracy": mean,
        "per_task_v57_raw": per_task_v57,
        "per_task_v58_raw": per_task_v58,
        "mean_within_world": {
            "v57_raw": float(np.mean([row["v57_raw"] for row in within_rows])),
            "v58_raw": float(np.mean([row["v58_raw"] for row in within_rows])),
        },
        "gates": {
            "v57": v57_gate,
            "v58": v58_gate,
            "adaptation_adds_0_03": mean["v58_raw"] - mean["v57_raw"] >= 0.03,
            "recurrence_beats_adaptation_only_by_0_03": mean["v58_raw"] - mean["adaptation_only_raw"] >= 0.03,
        },
        "gaps": {
            "v58_minus_v57": mean["v58_raw"] - mean["v57_raw"],
            "v58_minus_adaptation_only": mean["v58_raw"] - mean["adaptation_only_raw"],
            "v57_leave_world_minus_within_world": mean["v57_raw"] - float(np.mean([row["v57_raw"] for row in within_rows])),
            "v58_leave_world_minus_within_world": mean["v58_raw"] - float(np.mean([row["v58_raw"] for row in within_rows])),
        },
        "claim_boundary": "mechanistic controlled-vector result only; no language or affect claim",
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "leave_world_out_metrics.csv", eval_rows)
    write_csv(OUT_DIR / "within_world_metrics.csv", within_rows)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
