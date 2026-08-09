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
from emonet_v5.dynamics import FixedRecurrentDynamics  # noqa: E402
from residual_state import ResidualDrivenState  # noqa: E402
from adaptive_state import AdaptiveFastDynamics, AdaptiveResidualState  # noqa: E402
from vector_world import (  # noqa: E402
    PAIR_COUNT,
    RECURRENT_SEEDS,
    TASKS,
    TRAIN_PAIRS,
    WORLD_SEEDS,
    build_case,
    build_vector_world,
)
from attribution_features import geometry_agreement, pairwise_cosines, trace_pairwise_cosines  # noqa: E402


SLOW_DECAY = 0.80
ADAPTATION_DECAY = 0.995
ADAPTATION_STRENGTH = 0.20
RIDGE_ALPHA = 1.0
OUT_DIR = VERSION_ROOT / "outputs" / "trajectory_attribution"


class RidgeProbe:
    def __init__(self, alpha: float = RIDGE_ALPHA) -> None:
        self.alpha = float(alpha)

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
        xs = (np.asarray(x, dtype=np.float64) - self.mean) / self.scale
        return (xs @ self.weights + self.intercept >= 0.0).astype(np.int64)


def accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y) == np.asarray(pred)))


def isolated_fixed_traces(
    vectors: list[np.ndarray], config: DynamicsConfig, input_dim: int
) -> list[np.ndarray]:
    dynamics = FixedRecurrentDynamics(input_dim=input_dim, config=config)
    traces: list[np.ndarray] = []
    for vector in vectors:
        dynamics.reset_state()
        traces.append(dynamics.run_event(np.asarray(vector, dtype=np.float32)))
    return traces


def isolated_adaptive_traces(
    vectors: list[np.ndarray], config: DynamicsConfig, input_dim: int
) -> list[np.ndarray]:
    dynamics = AdaptiveFastDynamics(
        input_dim=input_dim,
        config=config,
        adaptation_strength=ADAPTATION_STRENGTH,
        adaptation_decay=ADAPTATION_DECAY,
        use_recurrence=True,
    )
    traces: list[np.ndarray] = []
    for vector in vectors:
        dynamics.reset_state()
        traces.append(dynamics.run_event(np.asarray(vector, dtype=np.float32)))
    return traces


def sequential_features(model, history: tuple[str, ...], current: str):
    model.reset_all()
    history_obs = model.consume_sequence(history)
    current_obs = model.consume_event(current)
    transient = history_obs[1:5]
    residuals = [obs.residual_input.astype(np.float32, copy=True) for obs in transient]
    traces = [obs.fast_trace.states.astype(np.float32, copy=True) for obs in transient]
    current_raw = current_obs.fast_trace.states.reshape(-1).astype(np.float32, copy=False)
    return residuals, traces, current_raw


def build_world_rows(world: int, seed: int, task: str) -> list[dict[str, object]]:
    encoder = build_vector_world(world)
    config = DynamicsConfig(seed=seed)
    v57 = ResidualDrivenState(
        encoder, seed=seed, slow_decay=SLOW_DECAY, dynamics_config=config
    )
    v58 = AdaptiveResidualState(
        encoder,
        seed=seed,
        adaptation_strength=ADAPTATION_STRENGTH,
        adaptation_decay=ADAPTATION_DECAY,
        slow_decay=SLOW_DECAY,
        dynamics_config=config,
    )

    rows: list[dict[str, object]] = []
    for pair_id in range(PAIR_COUNT):
        case = build_case(task, pair_id)
        pair: dict[int, dict[str, object]] = {}
        for label, history in ((0, case.class0), (1, case.class1)):
            raw_inputs = [encoder.encode(key) for key in history[1:5]]
            residuals57, sequential57, current57 = sequential_features(v57, history, case.current)
            residuals58, sequential58, current58 = sequential_features(v58, history, case.current)
            np.testing.assert_allclose(
                np.stack(residuals57), np.stack(residuals58), atol=1e-6
            )

            drives = [v57.fast.input_weight @ r for r in residuals57]
            isolated_residual57 = isolated_fixed_traces(residuals57, config, encoder.output_dim)
            isolated_raw57 = isolated_fixed_traces(raw_inputs, config, encoder.output_dim)
            isolated_residual58 = isolated_adaptive_traces(residuals58, config, encoder.output_dim)
            isolated_raw58 = isolated_adaptive_traces(raw_inputs, config, encoder.output_dim)

            seq57 = trace_pairwise_cosines(sequential57)
            iso57 = trace_pairwise_cosines(isolated_residual57)
            seq58 = trace_pairwise_cosines(sequential58)
            iso58 = trace_pairwise_cosines(isolated_residual58)
            agree57, dist57 = geometry_agreement(seq57, iso57)
            agree58, dist58 = geometry_agreement(seq58, iso58)

            pair[label] = {
                "input_relational": pairwise_cosines(raw_inputs),
                "residual_relational": pairwise_cosines(residuals57),
                "drive_relational": pairwise_cosines(drives),
                "v57_isolated_residual": iso57,
                "v57_isolated_raw": trace_pairwise_cosines(isolated_raw57),
                "v57_sequential": seq57,
                "v57_delta": (seq57 - iso57).astype(np.float32),
                "v57_current_raw": current57,
                "v58_isolated_residual": iso58,
                "v58_isolated_raw": trace_pairwise_cosines(isolated_raw58),
                "v58_sequential": seq58,
                "v58_delta": (seq58 - iso58).astype(np.float32),
                "v58_current_raw": current58,
                "v57_seq_iso_cosine": agree57,
                "v57_seq_iso_l2": dist57,
                "v58_seq_iso_cosine": agree58,
                "v58_seq_iso_l2": dist58,
                "v57_v58_seq_l2": float(np.linalg.norm(seq57 - seq58)),
            }

        for label in (0, 1):
            own = pair[label]
            opposite = pair[1 - label]
            rows.append(
                {
                    "world": world,
                    "seed": seed,
                    "task": task,
                    "pair_id": pair_id,
                    "label": label,
                    **own,
                    "v57_sequential_opposite": np.asarray(opposite["v57_sequential"]).copy(),
                    "v58_sequential_opposite": np.asarray(opposite["v58_sequential"]).copy(),
                }
            )
    return rows


def matrix(rows: list[dict[str, object]], field: str) -> np.ndarray:
    return np.stack([np.asarray(row[field], dtype=np.float32) for row in rows])


def labels(rows: list[dict[str, object]]) -> np.ndarray:
    return np.asarray([int(row["label"]) for row in rows], dtype=np.int64)


def evaluate(rows: list[dict[str, object]], seed: int, task: str, held_world: int) -> dict[str, float]:
    subset = [row for row in rows if int(row["seed"]) == seed and row["task"] == task]
    train = [
        row for row in subset
        if int(row["world"]) != held_world and int(row["pair_id"]) < TRAIN_PAIRS
    ]
    test = [
        row for row in subset
        if int(row["world"]) == held_world and int(row["pair_id"]) >= TRAIN_PAIRS
    ]
    y_train, y_test = labels(train), labels(test)
    fields = (
        "input_relational",
        "residual_relational",
        "drive_relational",
        "v57_isolated_residual",
        "v57_isolated_raw",
        "v57_sequential",
        "v57_delta",
        "v57_current_raw",
        "v58_isolated_residual",
        "v58_isolated_raw",
        "v58_sequential",
        "v58_delta",
        "v58_current_raw",
    )
    probes = {field: RidgeProbe().fit(matrix(train, field), y_train) for field in fields}
    result = {
        field: accuracy(y_test, probe.predict(matrix(test, field)))
        for field, probe in probes.items()
    }
    result["v57_sequential_opposite"] = accuracy(
        y_test, probes["v57_sequential"].predict(matrix(test, "v57_sequential_opposite"))
    )
    result["v58_sequential_opposite"] = accuracy(
        y_test, probes["v58_sequential"].predict(matrix(test, "v58_sequential_opposite"))
    )
    for scalar in (
        "v57_seq_iso_cosine",
        "v57_seq_iso_l2",
        "v58_seq_iso_cosine",
        "v58_seq_iso_l2",
        "v57_v58_seq_l2",
    ):
        result[scalar] = float(np.mean([float(row[scalar]) for row in test]))
    return result


def copy_like(mean: dict[str, float], prefix: str) -> bool:
    return all(
        (
            mean["input_relational"] >= 0.99,
            mean["residual_relational"] >= 0.95,
            mean["drive_relational"] >= 0.95,
            mean[f"{prefix}_isolated_residual"] >= 0.95,
            mean[f"{prefix}_sequential"] >= 0.95,
            abs(mean[f"{prefix}_sequential"] - mean[f"{prefix}_isolated_residual"]) <= 0.03,
            mean[f"{prefix}_seq_iso_cosine"] >= 0.95,
        )
    )


def recurrent_essential(mean: dict[str, float], prefix: str) -> bool:
    return (
        mean[f"{prefix}_sequential"] >= 0.90
        and mean[f"{prefix}_isolated_residual"] <= 0.75
        and mean[f"{prefix}_sequential"] - mean[f"{prefix}_isolated_residual"] >= 0.15
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    eval_rows: list[dict[str, object]] = []
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
                        **evaluate(rows, seed, task, held_world),
                    }
                )

    metric_names = [key for key in eval_rows[0] if key not in {"seed", "task", "held_world"}]
    mean = {
        metric: float(np.mean([float(row[metric]) for row in eval_rows]))
        for metric in metric_names
    }
    diagnosis = {
        "v57_copy_like_preservation": copy_like(mean, "v57"),
        "v58_copy_like_preservation": copy_like(mean, "v58"),
        "v57_recurrent_essential": recurrent_essential(mean, "v57"),
        "v58_recurrent_essential": recurrent_essential(mean, "v58"),
        "v57_recurrent_modulation_detectable": mean["v57_delta"] >= 0.70 and mean["v57_isolated_residual"] >= 0.90,
        "v58_recurrent_modulation_detectable": mean["v58_delta"] >= 0.70 and mean["v58_isolated_residual"] >= 0.90,
    }
    summary = {
        "version": "v5.9.2",
        "purpose": "attribute v5.9.1 trajectory geometry to input, residual, drive, isolated response, or recurrent carry",
        "state_generators_changed": False,
        "language_encoder_used": False,
        "primary_protocol": "leave-one-vector-world-out",
        "mean": mean,
        "diagnosis": diagnosis,
        "claim_boundary": "mechanistic attribution only; no affect/emotion claim",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "leave_world_out_metrics.csv", eval_rows)
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
