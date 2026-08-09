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
from hidden_prior_world import (  # noqa: E402
    DELAYS,
    PAIR_COUNT,
    RECURRENT_SEEDS,
    TASKS,
    TRAIN_PAIRS,
    WORLD_SEEDS,
    build_case,
    build_world,
    pairwise_relational,
)


SLOW_DECAY = 0.80
ADAPTATION_DECAY = 0.995
ADAPTATION_STRENGTH = 0.20
RIDGE_ALPHA = 1.0
PRIMARY_TASK = "norm_matched_repeat"
OUT_DIR = VERSION_ROOT / "outputs" / "hidden_prior_visible_window"


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


def prepare_hidden(model: Any, hidden: tuple[str, ...], delay_event: str, delay: int) -> dict[str, Any]:
    # reset_both avoids rebuilding the fixed seeded topology for every sample.
    model.reset_both()
    model._event_index = 0
    model.consume_sequence(hidden)
    for _ in range(delay):
        model.consume_event(delay_event)
    return snapshot(model)


def visible_features(model: Any, snap: dict[str, Any], visible: tuple[str, ...], final_event: str, reset: str | None) -> dict[str, np.ndarray]:
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


def previsible_features(model: Any, snap: dict[str, Any]) -> dict[str, np.ndarray]:
    result = {
        "prefast": np.asarray(snap["fast"], dtype=np.float32).copy(),
        "preslow": np.asarray(snap["slow"], dtype=np.float32).copy(),
        "slow_norm": np.asarray([np.linalg.norm(snap["slow"])], dtype=np.float32),
    }
    if "adaptation" in snap:
        result["preadaptation"] = np.asarray(snap["adaptation"], dtype=np.float32).copy()
    return result


def create_model(model_name: str, encoder, seed: int):
    config = DynamicsConfig(seed=seed)
    if model_name == "v57":
        return ResidualDrivenState(
            encoder, seed=seed, slow_decay=SLOW_DECAY, dynamics_config=config
        )
    if model_name == "v58":
        return AdaptiveResidualState(
            encoder,
            seed=seed,
            adaptation_strength=ADAPTATION_STRENGTH,
            adaptation_decay=ADAPTATION_DECAY,
            slow_decay=SLOW_DECAY,
            dynamics_config=config,
        )
    raise ValueError(model_name)


def build_rows(model_name: str, seed: int, task: str, delay: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for world in WORLD_SEEDS:
        encoder = build_world(world)
        model = create_model(model_name, encoder, seed)

        for pair_id in range(PAIR_COUNT):
            case = build_case(task, pair_id)
            visible_rel = pairwise_relational(case.visible, encoder)
            hidden_rel = {
                0: pairwise_relational(case.hidden0, encoder),
                1: pairwise_relational(case.hidden1, encoder),
            }
            pair: dict[int, dict[str, object]] = {}

            for label, hidden in ((0, case.hidden0), (1, case.hidden1)):
                snap = prepare_hidden(model, hidden, case.delay_event, delay)
                real = visible_features(model, snap, case.visible, case.final_event, None)
                item: dict[str, object] = {
                    "snap": snap,
                    "real": real,
                    "pre": previsible_features(model, snap),
                    "hidden_rel": hidden_rel[label],
                }
                if pair_id >= TRAIN_PAIRS:
                    item["fast_reset"] = visible_features(
                        model, snap, case.visible, case.final_event, "fast"
                    )
                    item["slow_reset"] = visible_features(
                        model, snap, case.visible, case.final_event, "slow"
                    )
                pair[label] = item

            # Both-reset state has no hidden-label information, so compute once
            # per held-out pair and share it across the two label rows.
            both_reset = None
            if pair_id >= TRAIN_PAIRS:
                both_reset = visible_features(
                    model,
                    pair[0]["snap"],
                    case.visible,
                    case.final_event,
                    "both",
                )

            for label in (0, 1):
                item = pair[label]
                opposite = pair[1 - label]
                row: dict[str, object] = {
                    "world": world,
                    "seed": seed,
                    "task": task,
                    "delay": delay,
                    "pair_id": pair_id,
                    "label": label,
                    "visible_relational": visible_rel.copy(),
                    "hidden_relational": np.asarray(item["hidden_rel"], dtype=np.float32),
                    "intact_raw": np.asarray(item["real"]["raw"], dtype=np.float32),
                    "intact_selfsim": np.asarray(item["real"]["selfsim"], dtype=np.float32),
                    "final_raw": np.asarray(item["real"]["final_raw"], dtype=np.float32),
                    "prefast": np.asarray(item["pre"]["prefast"], dtype=np.float32),
                    "preslow": np.asarray(item["pre"]["preslow"], dtype=np.float32),
                    "slow_norm": np.asarray(item["pre"]["slow_norm"], dtype=np.float32),
                }
                if model_name == "v58":
                    row["preadaptation"] = np.asarray(
                        item["pre"]["preadaptation"], dtype=np.float32
                    )
                if pair_id >= TRAIN_PAIRS:
                    row.update(
                        {
                            "fast_reset_raw": np.asarray(item["fast_reset"]["raw"], dtype=np.float32),
                            "slow_reset_raw": np.asarray(item["slow_reset"]["raw"], dtype=np.float32),
                            "both_reset_raw": np.asarray(both_reset["raw"], dtype=np.float32),
                            "both_reset_selfsim": np.asarray(both_reset["selfsim"], dtype=np.float32),
                            "opposite_raw": np.asarray(opposite["real"]["raw"], dtype=np.float32),
                            "opposite_selfsim": np.asarray(opposite["real"]["selfsim"], dtype=np.float32),
                        }
                    )
                rows.append(row)
    return rows


def matrix(rows: list[dict[str, object]], field: str) -> np.ndarray:
    return np.stack([np.asarray(row[field], dtype=np.float32) for row in rows])


def labels(rows: list[dict[str, object]]) -> np.ndarray:
    return np.asarray([int(row["label"]) for row in rows], dtype=np.int64)


def evaluate_fold(rows: list[dict[str, object]], held_world: int, model_name: str) -> dict[str, float]:
    train = [
        row for row in rows
        if int(row["world"]) != held_world and int(row["pair_id"]) < TRAIN_PAIRS
    ]
    test = [
        row for row in rows
        if int(row["world"]) == held_world and int(row["pair_id"]) >= TRAIN_PAIRS
    ]
    y_train, y_test = labels(train), labels(test)
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
    probes = {field: RidgeProbe().fit(matrix(train, field), y_train) for field in independent}
    result = {
        field: accuracy(y_test, probe.predict(matrix(test, field)))
        for field, probe in probes.items()
    }
    raw_probe = probes["intact_raw"]
    selfsim_probe = probes["intact_selfsim"]
    for control in ("fast_reset_raw", "slow_reset_raw", "both_reset_raw", "opposite_raw"):
        result[control] = accuracy(y_test, raw_probe.predict(matrix(test, control)))
    result["both_reset_selfsim"] = accuracy(
        y_test, selfsim_probe.predict(matrix(test, "both_reset_selfsim"))
    )
    result["opposite_selfsim"] = accuracy(
        y_test, selfsim_probe.predict(matrix(test, "opposite_selfsim"))
    )
    return result


def aggregate_model(eval_rows: list[dict[str, object]], model_name: str) -> dict[str, object]:
    subset = [row for row in eval_rows if row["model"] == model_name]
    metric_names = [key for key in subset[0] if key not in {"model", "seed", "task", "delay", "held_world"}]
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
            np.mean([float(row["intact_raw"]) for row in primary if int(row["delay"]) == delay])
        )
        for delay in DELAYS
    }
    per_delay_selfsim = {
        str(delay): float(
            np.mean([float(row["intact_selfsim"]) for row in primary if int(row["delay"]) == delay])
        )
        for delay in DELAYS
    }
    gate = {
        "macro_raw_at_least_0_70": primary_mean["intact_raw"] >= 0.70,
        "at_least_2_of_3_delays_raw_at_0_65": sum(v >= 0.65 for v in per_delay_raw.values()) >= 2,
        "both_reset_raw_at_most_0_55": primary_mean["both_reset_raw"] <= 0.55,
        "opposite_hidden_raw_at_most_0_30": primary_mean["opposite_raw"] <= 0.30,
        "visible_input_relational_at_most_0_55": primary_mean["visible_relational"] <= 0.55,
        "hidden_relational_validity_at_least_0_99": primary_mean["hidden_relational"] >= 0.99,
        "slow_norm_baseline_at_most_0_55": primary_mean["slow_norm"] <= 0.55,
    }
    gate["all"] = all(gate.values())
    selfsim_gate = {
        "macro_at_least_0_65": primary_mean["intact_selfsim"] >= 0.65,
        "both_reset_at_most_0_55": primary_mean["both_reset_selfsim"] <= 0.55,
        "opposite_hidden_at_most_0_35": primary_mean["opposite_selfsim"] <= 0.35,
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
    eval_rows: list[dict[str, object]] = []

    # Process one model/seed/task/delay block at a time so the 8192D visible
    # trajectories do not accumulate across the entire experiment.
    for model_name in ("v57", "v58"):
        for seed in RECURRENT_SEEDS:
            for task in TASKS:
                for delay in DELAYS:
                    rows = build_rows(model_name, seed, task, delay)
                    for held_world in WORLD_SEEDS:
                        eval_rows.append(
                            {
                                "model": model_name,
                                "seed": seed,
                                "task": task,
                                "delay": delay,
                                "held_world": held_world,
                                **evaluate_fold(rows, held_world, model_name),
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
