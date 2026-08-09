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
    REPO_ROOT / "v5_9_encoder_free_temporal",
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
from trajectory_features import (  # noqa: E402
    current_raw,
    event_final_state_similarity,
    event_mean_state_similarity,
    event_trace_similarity,
    hashed_full_episode,
)


SLOW_DECAY = 0.80
ADAPTATION_DECAY = 0.995
ADAPTATION_STRENGTH = 0.20
RIDGE_ALPHA = 1.0
OUT_DIR = VERSION_ROOT / "outputs" / "full_trajectory_geometry"


class RidgeProbe:
    def __init__(self, alpha: float = RIDGE_ALPHA) -> None:
        self.alpha = float(alpha)
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.weights: np.ndarray | None = None
        self.intercept = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RidgeProbe":
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
            self.weights = np.linalg.solve(xs.T @ xs + self.alpha * np.eye(d), xs.T @ yc)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.scale is None or self.weights is None:
            raise RuntimeError("probe not fit")
        xs = (np.asarray(x, dtype=np.float64) - self.mean) / self.scale
        return (xs @ self.weights + self.intercept >= 0.0).astype(np.int64)


def accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y) == np.asarray(pred)))


def simulate(model, history: tuple[str, ...], current: str) -> dict[str, np.ndarray]:
    model.reset_all()
    history_observations = model.consume_sequence(history)
    current_observation = model.consume_event(current)
    transient = history_observations[1:5]
    if len(transient) != 4:
        raise AssertionError("transient trajectory extraction failed")
    return {
        "trace_similarity": event_trace_similarity(transient),
        "final_similarity": event_final_state_similarity(transient),
        "mean_similarity": event_mean_state_similarity(transient),
        "episode_hash": hashed_full_episode(history_observations, current_observation),
        "current_raw": current_raw(current_observation),
    }


def build_world_rows(world_seed: int, recurrent_seed: int, task: str) -> list[dict[str, object]]:
    encoder = build_vector_world(world_seed)
    config = DynamicsConfig(seed=recurrent_seed)
    v57 = ResidualDrivenState(
        encoder,
        seed=recurrent_seed,
        slow_decay=SLOW_DECAY,
        dynamics_config=config,
    )
    v58 = AdaptiveResidualState(
        encoder,
        seed=recurrent_seed,
        adaptation_strength=ADAPTATION_STRENGTH,
        adaptation_decay=ADAPTATION_DECAY,
        slow_decay=SLOW_DECAY,
        use_recurrence=True,
        dynamics_config=config,
    )

    rows: list[dict[str, object]] = []
    for pair_id in range(PAIR_COUNT):
        case = build_case(task, pair_id)
        pair: dict[int, dict[str, np.ndarray]] = {}
        for label, history in ((0, case.class0), (1, case.class1)):
            old = simulate(v57, history, case.current)
            adaptive = simulate(v58, history, case.current)
            pair[label] = {
                **{f"v57_{key}": value for key, value in old.items()},
                **{f"v58_{key}": value for key, value in adaptive.items()},
                "input_relational": relational_features(history, encoder),
            }

        for label in (0, 1):
            own = pair[label]
            opposite = pair[1 - label]
            rows.append(
                {
                    "world": world_seed,
                    "seed": recurrent_seed,
                    "task": task,
                    "pair_id": pair_id,
                    "label": label,
                    **own,
                    "v57_trace_similarity_opposite": opposite["v57_trace_similarity"].copy(),
                    "v58_trace_similarity_opposite": opposite["v58_trace_similarity"].copy(),
                }
            )
    return rows


def matrix(rows: list[dict[str, object]], field: str) -> np.ndarray:
    return np.stack([np.asarray(row[field], dtype=np.float32) for row in rows])


def labels(rows: list[dict[str, object]]) -> np.ndarray:
    return np.asarray([int(row["label"]) for row in rows], dtype=np.int64)


def evaluate_leave_world_out(
    rows: list[dict[str, object]],
    recurrent_seed: int,
    task: str,
    held_world: int,
) -> dict[str, float]:
    subset = [
        row for row in rows
        if int(row["seed"]) == recurrent_seed and row["task"] == task
    ]
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

    fields = (
        "v57_trace_similarity",
        "v57_final_similarity",
        "v57_mean_similarity",
        "v57_episode_hash",
        "v57_current_raw",
        "v58_trace_similarity",
        "v58_final_similarity",
        "v58_mean_similarity",
        "v58_episode_hash",
        "v58_current_raw",
        "input_relational",
    )
    probes = {field: RidgeProbe().fit(matrix(train, field), y_train) for field in fields}
    result = {
        field: accuracy(y_test, probe.predict(matrix(test, field)))
        for field, probe in probes.items()
    }
    result["v57_trace_similarity_opposite"] = accuracy(
        y_test,
        probes["v57_trace_similarity"].predict(matrix(test, "v57_trace_similarity_opposite")),
    )
    result["v58_trace_similarity_opposite"] = accuracy(
        y_test,
        probes["v58_trace_similarity"].predict(matrix(test, "v58_trace_similarity_opposite")),
    )
    return result


def evaluate_within_world(
    rows: list[dict[str, object]],
    recurrent_seed: int,
    task: str,
    world: int,
    field: str,
) -> float:
    subset = [
        row for row in rows
        if int(row["seed"]) == recurrent_seed
        and row["task"] == task
        and int(row["world"]) == world
    ]
    train = [row for row in subset if int(row["pair_id"]) < TRAIN_PAIRS]
    test = [row for row in subset if int(row["pair_id"]) >= TRAIN_PAIRS]
    probe = RidgeProbe().fit(matrix(train, field), labels(train))
    return accuracy(labels(test), probe.predict(matrix(test, field)))


def trajectory_gate(
    mean: dict[str, float],
    per_task: dict[str, float],
    within_mean: float,
    prefix: str,
) -> dict[str, bool]:
    main = mean[f"{prefix}_trace_similarity"]
    gates = {
        "macro_at_least_0_85": main >= 0.85,
        "all_4_tasks_at_least_0_80": all(value >= 0.80 for value in per_task.values()),
        "opposite_history_at_most_0_20": mean[f"{prefix}_trace_similarity_opposite"] <= 0.20,
        "leave_world_drop_at_most_0_05": within_mean - main <= 0.05,
        "input_relational_at_least_0_99": mean["input_relational"] >= 0.99,
        "current_only_at_most_0_60": mean[f"{prefix}_current_raw"] <= 0.60,
    }
    gates["all"] = all(gates.values())
    return gates


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    eval_rows: list[dict[str, object]] = []
    within_rows: list[dict[str, object]] = []

    # Process one recurrent seed/task at a time so the large trajectory objects
    # never accumulate across the full benchmark.
    for seed in RECURRENT_SEEDS:
        for task in TASKS:
            rows: list[dict[str, object]] = []
            for world in WORLD_SEEDS:
                rows.extend(build_world_rows(world, seed, task))

            for held_world in WORLD_SEEDS:
                eval_rows.append(
                    {
                        "seed": seed,
                        "task": task,
                        "held_world": held_world,
                        **evaluate_leave_world_out(rows, seed, task, held_world),
                    }
                )

            for world in WORLD_SEEDS:
                within_rows.append(
                    {
                        "seed": seed,
                        "task": task,
                        "world": world,
                        "v57_trace_similarity": evaluate_within_world(
                            rows, seed, task, world, "v57_trace_similarity"
                        ),
                        "v58_trace_similarity": evaluate_within_world(
                            rows, seed, task, world, "v58_trace_similarity"
                        ),
                    }
                )

    metric_names = [key for key in eval_rows[0] if key not in {"seed", "task", "held_world"}]
    mean = {
        metric: float(np.mean([float(row[metric]) for row in eval_rows]))
        for metric in metric_names
    }
    per_task_v57 = {
        task: float(
            np.mean([float(row["v57_trace_similarity"]) for row in eval_rows if row["task"] == task])
        )
        for task in TASKS
    }
    per_task_v58 = {
        task: float(
            np.mean([float(row["v58_trace_similarity"]) for row in eval_rows if row["task"] == task])
        )
        for task in TASKS
    }
    within_v57 = float(np.mean([float(row["v57_trace_similarity"]) for row in within_rows]))
    within_v58 = float(np.mean([float(row["v58_trace_similarity"]) for row in within_rows]))

    gates_v57 = trajectory_gate(mean, per_task_v57, within_v57, "v57")
    gates_v58 = trajectory_gate(mean, per_task_v58, within_v58, "v58")

    summary = {
        "version": "v5.9.1",
        "purpose": "test full event-by-event neural trajectory geometry under frozen encoder-free dynamics",
        "state_generators_changed_from_v5_9": False,
        "language_encoder_used": False,
        "primary_protocol": "leave-one-vector-world-out",
        "vector_worlds": list(WORLD_SEEDS),
        "recurrent_seeds": list(RECURRENT_SEEDS),
        "tasks": list(TASKS),
        "nonprimary_episode_readout": "deterministic 256D signed feature hash of all 14,336 raw episode coordinates",
        "mean_accuracy": mean,
        "per_task_v57_trace_similarity": per_task_v57,
        "per_task_v58_trace_similarity": per_task_v58,
        "mean_within_world_trace_similarity": {
            "v57": within_v57,
            "v58": within_v58,
        },
        "gates": {
            "v57_trace_similarity": gates_v57,
            "v58_trace_similarity": gates_v58,
        },
        "interpretation_guardrail": (
            "A successful trace-similarity result demonstrates identity-invariant preservation/re-expression "
            "of temporal relations in the full trajectory; it does not prove information beyond the raw input relational matrix."
        ),
        "claim_boundary": "encoder-free trajectory diagnostic only; no language or affect claim",
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "leave_world_out_metrics.csv", eval_rows)
    write_csv(OUT_DIR / "within_world_metrics.csv", within_rows)
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
