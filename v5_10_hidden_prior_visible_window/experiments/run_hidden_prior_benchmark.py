from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

HERE = Path(__file__).resolve()
VERSION_ROOT = HERE.parents[1]
REPO_ROOT = HERE.parents[2]
for path in (
    REPO_ROOT,
    VERSION_ROOT,
    REPO_ROOT / "v5_7_residual_fast_dynamics",
    REPO_ROOT / "v5_8_adaptive_fast_dynamics",
    REPO_ROOT / "v5_9_2_trajectory_attribution",
):
    sys.path.insert(0, str(path))

from emonet_v5 import DynamicsConfig  # noqa: E402
from residual_state import ResidualDrivenState  # noqa: E402
from adaptive_state import AdaptiveResidualState  # noqa: E402
from attribution_features import trace_pairwise_cosines  # noqa: E402
from batch_dynamics import BatchState, BatchedResidualDynamics  # noqa: E402
from hidden_prior_world import (  # noqa: E402
    DELAYS,
    PAIR_COUNT,
    RECURRENT_SEEDS,
    TASKS,
    TRAIN_PAIRS,
    WORLD_SEEDS,
    build_case,
    build_world,
)


SLOW_DECAY = 0.80
ADAPTATION_DECAY = 0.995
ADAPTATION_STRENGTH = 0.20
RIDGE_ALPHA = 1.0
PRIMARY_TASK = "norm_matched_repeat"
OUT_DIR = VERSION_ROOT / "outputs" / "hidden_prior_visible_window"
PAIR_INDEX = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


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


# ---------------------------------------------------------------------------
# Scalar reference helpers retained for regression/debugging. main() does not
# use them; the formal run uses BatchedResidualDynamics after scalar-equivalence
# tests pass in CI.
# ---------------------------------------------------------------------------

def snapshot(model: Any) -> dict[str, Any]:
    snap: dict[str, Any] = {
        "fast": model.fast.state.copy(),
        "slow": model.slow.state.copy(),
        "event_index": int(model._event_index),
    }
    if hasattr(model.fast, "adaptation"):
        snap["adaptation"] = model.fast.adaptation.copy()
    return snap


def restore(model: Any, snap: dict[str, Any]) -> None:
    model.fast.state = np.asarray(snap["fast"], dtype=np.float32).copy()
    model.slow.state = np.asarray(snap["slow"], dtype=np.float32).copy()
    model._event_index = int(snap["event_index"])
    if "adaptation" in snap:
        model.fast.adaptation = np.asarray(snap["adaptation"], dtype=np.float32).copy()


def prepare_hidden(
    model: Any, hidden: tuple[str, ...], delay_event: str, delay: int
) -> dict[str, Any]:
    model.reset_both()
    model._event_index = 0
    model.consume_sequence(hidden)
    for _ in range(delay):
        model.consume_event(delay_event)
    return snapshot(model)


def visible_features(
    model: Any,
    snap: dict[str, Any],
    visible: tuple[str, ...],
    final_event: str,
    reset: str | None,
) -> dict[str, np.ndarray]:
    restore(model, snap)
    if reset == "fast":
        model.reset_fast()
    elif reset == "slow":
        model.reset_slow()
    elif reset == "both":
        model.reset_both()
    elif reset is not None:
        raise ValueError(reset)

    visible_obs = model.consume_sequence(visible)
    final_obs = model.consume_event(final_event)
    traces = [obs.fast_trace.states.astype(np.float32, copy=False) for obs in visible_obs]
    return {
        "raw": np.concatenate([trace.reshape(-1) for trace in traces]).astype(np.float32),
        "selfsim": trace_pairwise_cosines(traces),
        "final_raw": final_obs.fast_trace.states.reshape(-1).astype(np.float32, copy=False),
    }


# ---------------------------------------------------------------------------
# Batched formal execution.
# ---------------------------------------------------------------------------

def pairwise_batch(vectors: np.ndarray) -> np.ndarray:
    """Six pairwise cosines for [batch,4,features]."""
    x = np.asarray(vectors, dtype=np.float32)
    if x.ndim != 3 or x.shape[1] != 4:
        raise ValueError(f"expected [batch,4,d], got {x.shape}")
    norms = np.linalg.norm(x, axis=2)
    result: list[np.ndarray] = []
    for i, j in PAIR_INDEX:
        numerator = np.sum(x[:, i] * x[:, j], axis=1)
        denominator = norms[:, i] * norms[:, j]
        value = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=np.float32),
            where=denominator > 1e-12,
        )
        result.append(value.astype(np.float32, copy=False))
    return np.stack(result, axis=1).astype(np.float32, copy=False)


def trace_selfsim_batch(traces: np.ndarray) -> np.ndarray:
    """Six cosines for [batch,4,ticks,neurons] visible event traces."""
    traces = np.asarray(traces, dtype=np.float32)
    if traces.ndim != 4 or traces.shape[1] != 4:
        raise ValueError(f"expected [batch,4,ticks,neurons], got {traces.shape}")
    return pairwise_batch(traces.reshape(traces.shape[0], 4, -1))


def take_state(state: BatchState, mask: np.ndarray) -> BatchState:
    mask = np.asarray(mask)
    return BatchState(
        fast=state.fast[mask].copy(),
        slow=state.slow[mask].copy(),
        adaptation=(
            None if state.adaptation is None else state.adaptation[mask].copy()
        ),
    )


def run_visible_batch(
    dynamics: BatchedResidualDynamics,
    initial_state: BatchState,
    visible: np.ndarray,
    final_event: np.ndarray,
) -> dict[str, np.ndarray]:
    state = initial_state.copy()
    event_traces: list[np.ndarray] = []
    for position in range(4):
        state, trace, _ = dynamics.run_event(state, visible[:, position, :])
        event_traces.append(trace)
    visible_traces = np.stack(event_traces, axis=1).astype(np.float32, copy=False)
    state, final_trace, _ = dynamics.run_event(state, final_event)
    return {
        "raw": visible_traces.reshape(visible_traces.shape[0], -1).astype(
            np.float32, copy=False
        ),
        "selfsim": trace_selfsim_batch(visible_traces),
        "final_raw": final_trace.reshape(final_trace.shape[0], -1).astype(
            np.float32, copy=False
        ),
    }


def build_static_inputs(world: int, task: str) -> dict[str, np.ndarray]:
    encoder = build_world(world)
    batch_size = PAIR_COUNT * 2
    pair_ids = np.repeat(np.arange(PAIR_COUNT, dtype=np.int64), 2)
    labels_array = np.tile(np.asarray([0, 1], dtype=np.int64), PAIR_COUNT)
    hidden = np.empty((batch_size, 4, encoder.output_dim), dtype=np.float32)
    visible = np.empty_like(hidden)
    delay_vector = np.empty((batch_size, encoder.output_dim), dtype=np.float32)
    final_vector = np.empty_like(delay_vector)

    cursor = 0
    for pair_id in range(PAIR_COUNT):
        case = build_case(task, pair_id)
        visible_vectors = np.stack([encoder.encode(key) for key in case.visible]).astype(
            np.float32, copy=False
        )
        delay = encoder.encode(case.delay_event)
        final = encoder.encode(case.final_event)
        for label, hidden_keys in ((0, case.hidden0), (1, case.hidden1)):
            hidden[cursor] = np.stack(
                [encoder.encode(key) for key in hidden_keys]
            ).astype(np.float32, copy=False)
            visible[cursor] = visible_vectors
            delay_vector[cursor] = delay
            final_vector[cursor] = final
            if labels_array[cursor] != label:
                raise AssertionError("batched row label order drifted")
            cursor += 1

    return {
        "pair_id": pair_ids,
        "label": labels_array,
        "hidden": hidden,
        "visible": visible,
        "delay": delay_vector,
        "final": final_vector,
        "hidden_relational": pairwise_batch(hidden),
        "visible_relational": pairwise_batch(visible),
    }


def create_batch_dynamics(model_name: str, seed: int) -> BatchedResidualDynamics:
    config = DynamicsConfig(seed=seed)
    if model_name == "v57":
        return BatchedResidualDynamics(
            input_dim=384,
            config=config,
            slow_decay=SLOW_DECAY,
        )
    if model_name == "v58":
        return BatchedResidualDynamics(
            input_dim=384,
            config=config,
            slow_decay=SLOW_DECAY,
            adaptation_strength=ADAPTATION_STRENGTH,
            adaptation_decay=ADAPTATION_DECAY,
        )
    raise ValueError(model_name)


def build_world_features(
    model_name: str,
    seed: int,
    delay: int,
    static: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    dynamics = create_batch_dynamics(model_name, seed)
    batch_size = static["label"].shape[0]
    state = dynamics.zeros(batch_size)

    for position in range(4):
        state, _, _ = dynamics.run_event(state, static["hidden"][:, position, :])
    for _ in range(delay):
        state, _, _ = dynamics.run_event(state, static["delay"])
    previsible = state.copy()

    intact = run_visible_batch(dynamics, previsible, static["visible"], static["final"])
    fast_reset = run_visible_batch(
        dynamics,
        dynamics.reset_fast(previsible),
        static["visible"],
        static["final"],
    )
    slow_reset = run_visible_batch(
        dynamics,
        dynamics.reset_slow(previsible),
        static["visible"],
        static["final"],
    )
    both_reset = run_visible_batch(
        dynamics,
        dynamics.reset_both(previsible),
        static["visible"],
        static["final"],
    )

    features: dict[str, np.ndarray] = {
        "pair_id": static["pair_id"],
        "label": static["label"],
        "visible_relational": static["visible_relational"],
        "hidden_relational": static["hidden_relational"],
        "intact_raw": intact["raw"],
        "intact_selfsim": intact["selfsim"],
        "final_raw": intact["final_raw"],
        "prefast": previsible.fast.astype(np.float32, copy=True),
        "preslow": previsible.slow.astype(np.float32, copy=True),
        "slow_norm": np.linalg.norm(previsible.slow, axis=1, keepdims=True).astype(
            np.float32
        ),
        "fast_reset_raw": fast_reset["raw"],
        "slow_reset_raw": slow_reset["raw"],
        "both_reset_raw": both_reset["raw"],
        "both_reset_selfsim": both_reset["selfsim"],
    }
    if model_name == "v58":
        if previsible.adaptation is None:
            raise AssertionError("adaptive batch is missing adaptation state")
        features["preadaptation"] = previsible.adaptation.astype(
            np.float32, copy=True
        )
    return features


def concatenate_worlds(
    worlds: dict[int, dict[str, np.ndarray]],
    world_ids: list[int],
    field: str,
    *,
    train: bool,
) -> np.ndarray:
    values: list[np.ndarray] = []
    for world in world_ids:
        data = worlds[world]
        mask = (
            data["pair_id"] < TRAIN_PAIRS
            if train
            else data["pair_id"] >= TRAIN_PAIRS
        )
        values.append(np.asarray(data[field])[mask])
    return np.concatenate(values, axis=0)


def evaluate_fold_batch(
    worlds: dict[int, dict[str, np.ndarray]],
    held_world: int,
    model_name: str,
) -> dict[str, float]:
    train_worlds = [world for world in WORLD_SEEDS if world != held_world]
    held = worlds[held_world]
    test_mask = held["pair_id"] >= TRAIN_PAIRS
    test_indices = np.flatnonzero(test_mask)
    opposite_indices = test_indices ^ 1

    y_train = concatenate_worlds(worlds, train_worlds, "label", train=True)
    y_test = held["label"][test_mask]
    independent = [
        "intact_raw",
        "intact_selfsim",
        "final_raw",
        "prefast",
        "preslow",
        "slow_norm",
        "visible_relational",
        "hidden_relational",
    ]
    if model_name == "v58":
        independent.append("preadaptation")

    probes: dict[str, RidgeProbe] = {}
    result: dict[str, float] = {}
    for field in independent:
        x_train = concatenate_worlds(worlds, train_worlds, field, train=True)
        probe = RidgeProbe().fit(x_train, y_train)
        probes[field] = probe
        result[field] = accuracy(y_test, probe.predict(held[field][test_mask]))

    raw_probe = probes["intact_raw"]
    selfsim_probe = probes["intact_selfsim"]
    for control in ("fast_reset_raw", "slow_reset_raw", "both_reset_raw"):
        result[control] = accuracy(
            y_test, raw_probe.predict(held[control][test_mask])
        )
    result["opposite_raw"] = accuracy(
        y_test, raw_probe.predict(held["intact_raw"][opposite_indices])
    )
    result["both_reset_selfsim"] = accuracy(
        y_test, selfsim_probe.predict(held["both_reset_selfsim"][test_mask])
    )
    result["opposite_selfsim"] = accuracy(
        y_test, selfsim_probe.predict(held["intact_selfsim"][opposite_indices])
    )
    return result


def aggregate_model(eval_rows: list[dict[str, object]], model_name: str) -> dict[str, object]:
    subset = [row for row in eval_rows if row["model"] == model_name]
    metric_names = [
        key
        for key in subset[0]
        if key not in {"model", "seed", "task", "delay", "held_world"}
    ]
    mean = {
        metric: float(np.mean([float(row[metric]) for row in subset]))
        for metric in metric_names
    }
    primary = [row for row in subset if row["task"] == PRIMARY_TASK]
    primary_mean = {
        metric: float(np.mean([float(row[metric]) for row in primary]))
        for metric in metric_names
    }
    per_delay_raw = {
        str(delay): float(
            np.mean(
                [
                    float(row["intact_raw"])
                    for row in primary
                    if int(row["delay"]) == delay
                ]
            )
        )
        for delay in DELAYS
    }
    per_delay_selfsim = {
        str(delay): float(
            np.mean(
                [
                    float(row["intact_selfsim"])
                    for row in primary
                    if int(row["delay"]) == delay
                ]
            )
        )
        for delay in DELAYS
    }
    gate = {
        "macro_raw_at_least_0_70": primary_mean["intact_raw"] >= 0.70,
        "at_least_2_of_3_delays_raw_at_0_65": sum(
            value >= 0.65 for value in per_delay_raw.values()
        )
        >= 2,
        "both_reset_raw_at_most_0_55": primary_mean["both_reset_raw"] <= 0.55,
        "opposite_hidden_raw_at_most_0_30": primary_mean["opposite_raw"] <= 0.30,
        "visible_input_relational_at_most_0_55": primary_mean[
            "visible_relational"
        ]
        <= 0.55,
        "hidden_relational_validity_at_least_0_99": primary_mean[
            "hidden_relational"
        ]
        >= 0.99,
        "slow_norm_baseline_at_most_0_55": primary_mean["slow_norm"] <= 0.55,
    }
    gate["all"] = all(gate.values())
    selfsim_gate = {
        "macro_at_least_0_65": primary_mean["intact_selfsim"] >= 0.65,
        "both_reset_at_most_0_55": primary_mean["both_reset_selfsim"] <= 0.55,
        "opposite_hidden_at_most_0_35": primary_mean["opposite_selfsim"]
        <= 0.35,
    }
    selfsim_gate["all"] = all(selfsim_gate.values())
    localization = {
        "slow_path_evidence": primary_mean["fast_reset_raw"] >= 0.65,
        "fast_path_evidence": primary_mean["slow_reset_raw"] >= 0.65,
        "both_paths_independently_useful": primary_mean["fast_reset_raw"] >= 0.65
        and primary_mean["slow_reset_raw"] >= 0.65,
    }
    return {
        "overall_mean": mean,
        "primary_norm_matched_mean": primary_mean,
        "primary_per_delay_raw": per_delay_raw,
        "primary_per_delay_selfsim": per_delay_selfsim,
        "primary_gate": gate,
        "selfsim_gate": selfsim_gate,
        "localization": localization,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    # Static vector inputs are independent of model/recurrent seed/delay and
    # therefore built only once for each world/task.
    static_inputs = {
        (world, task): build_static_inputs(world, task)
        for world in WORLD_SEEDS
        for task in TASKS
    }
    eval_rows: list[dict[str, object]] = []

    for model_name in ("v57", "v58"):
        for seed in RECURRENT_SEEDS:
            for task in TASKS:
                for delay in DELAYS:
                    world_features = {
                        world: build_world_features(
                            model_name,
                            seed,
                            delay,
                            static_inputs[(world, task)],
                        )
                        for world in WORLD_SEEDS
                    }
                    for held_world in WORLD_SEEDS:
                        eval_rows.append(
                            {
                                "model": model_name,
                                "seed": seed,
                                "task": task,
                                "delay": delay,
                                "held_world": held_world,
                                **evaluate_fold_batch(
                                    world_features, held_world, model_name
                                ),
                            }
                        )

    by_model = {
        model_name: aggregate_model(eval_rows, model_name)
        for model_name in ("v57", "v58")
    }
    v57_primary = by_model["v57"]["primary_norm_matched_mean"]["intact_raw"]
    v58_primary = by_model["v58"]["primary_norm_matched_mean"]["intact_raw"]
    summary = {
        "version": "v5.10",
        "purpose": "test whether hidden relational history alters a later identical-input visible neural trajectory",
        "execution": "protocol-equivalent batched frozen dynamics; scalar-equivalence tested before benchmark",
        "language_encoder_used": False,
        "state_generators_changed": False,
        "worlds": list(WORLD_SEEDS),
        "recurrent_seeds": list(RECURRENT_SEEDS),
        "tasks": list(TASKS),
        "delays": list(DELAYS),
        "pair_count": PAIR_COUNT,
        "train_pairs": TRAIN_PAIRS,
        "primary_task": PRIMARY_TASK,
        "by_model": by_model,
        "adaptation_comparison": {
            "v58_minus_v57_primary_raw": float(v58_primary - v57_primary),
            "adaptation_adds_0_03": bool(v58_primary - v57_primary >= 0.03),
        },
        "claim_boundary": "controlled-vector causal state-carry benchmark only; no language or affect claim",
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "leave_world_out_metrics.csv", eval_rows)
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
