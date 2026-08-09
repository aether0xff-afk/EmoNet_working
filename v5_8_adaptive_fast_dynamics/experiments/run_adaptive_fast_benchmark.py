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
    REPO_ROOT / "v5_7_residual_fast_dynamics" / "experiments",
    REPO_ROOT / "v5_7_1_fast_trace_geometry",
):
    sys.path.insert(0, str(path))

from adaptive_state import AdaptiveResidualState  # noqa: E402
from temporal_suite import (  # noqa: E402
    CALIBRATION_TRAIN_PAIRS,
    PAIR_COUNT,
    TASKS,
    TRAIN_PAIRS,
    build_case,
    relational_features,
)
from geometry import population_moments  # noqa: E402
from residual_state import ResidualDrivenState  # noqa: E402
from run_learned_memory_benchmark import MODEL_NAME, RIDGE_ALPHA, SEEDS, CachedSentenceEncoder  # noqa: E402
from run_readout_temporal_diagnostic import DiagnosticRidge, accuracy  # noqa: E402
from run_residual_fast_benchmark import direct_residual_change_features  # noqa: E402


SLOW_DECAY = 0.80
CALIBRATION_SEED = 31415
ADAPTATION_DECAYS = (0.970, 0.985, 0.995)
ADAPTATION_STRENGTHS = (0.20, 0.50, 0.80)
OUT_DIR = VERSION_ROOT / "outputs" / "adaptive_fast_benchmark"


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


def primary_fast_features(observation) -> np.ndarray:
    # Frozen after v5.7.1: this was the strongest label-free trace geometry
    # diagnostic and is invariant to neuron permutation.
    return population_moments(observation.fast_trace)


def raw_fast_features(observation) -> np.ndarray:
    return observation.fast_trace.states.reshape(-1).astype(np.float32, copy=False)


def slow_features(observation) -> np.ndarray:
    return observation.slow_state.astype(np.float32, copy=True)


def task_rows_for_model(
    task: str,
    model: AdaptiveResidualState,
    *,
    pair_stop: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pair_id in range(pair_stop):
        case = build_case(task, pair_id)
        for label, sequence in ((0, case.class0), (1, case.class1)):
            observation = run_adaptive(model, sequence, case.current)
            rows.append(
                {
                    "pair_id": pair_id,
                    "label": label,
                    "primary": primary_fast_features(observation),
                }
            )
    return rows


def matrix(rows: list[dict[str, object]], field: str) -> np.ndarray:
    return np.stack([np.asarray(row[field], dtype=np.float32) for row in rows])


def labels(rows: list[dict[str, object]]) -> np.ndarray:
    return np.asarray([int(row["label"]) for row in rows], dtype=np.int64)


def calibration_score(
    encoder: CachedSentenceEncoder,
    *,
    adaptation_strength: float,
    adaptation_decay: float,
) -> tuple[float, dict[str, float]]:
    per_task: dict[str, float] = {}
    for task in TASKS:
        model = AdaptiveResidualState(
            encoder,
            seed=CALIBRATION_SEED,
            adaptation_strength=adaptation_strength,
            adaptation_decay=adaptation_decay,
            slow_decay=SLOW_DECAY,
            use_recurrence=True,
        )
        rows = task_rows_for_model(task, model, pair_stop=TRAIN_PAIRS)
        train = [row for row in rows if int(row["pair_id"]) < CALIBRATION_TRAIN_PAIRS]
        validation = [
            row
            for row in rows
            if CALIBRATION_TRAIN_PAIRS <= int(row["pair_id"]) < TRAIN_PAIRS
        ]
        probe = DiagnosticRidge(RIDGE_ALPHA).fit(matrix(train, "primary"), labels(train))
        per_task[task] = accuracy(
            labels(validation), probe.predict(matrix(validation, "primary"))
        )
    return float(np.mean(list(per_task.values()))), per_task


def select_adaptation(encoder: CachedSentenceEncoder) -> tuple[dict[str, float], list[dict[str, object]]]:
    candidates: list[dict[str, object]] = []
    best: dict[str, float] | None = None
    best_score = -1.0

    # Loop order is the predeclared deterministic tie-break: smaller decay,
    # then smaller strength wins an exact macro tie.
    for decay in ADAPTATION_DECAYS:
        for strength in ADAPTATION_STRENGTHS:
            macro, per_task = calibration_score(
                encoder,
                adaptation_strength=strength,
                adaptation_decay=decay,
            )
            row: dict[str, object] = {
                "adaptation_decay": decay,
                "adaptation_strength": strength,
                "validation_macro": macro,
                **{f"validation_{task}": per_task[task] for task in TASKS},
            }
            candidates.append(row)
            if macro > best_score + 1e-12:
                best_score = macro
                best = {
                    "adaptation_decay": float(decay),
                    "adaptation_strength": float(strength),
                    "validation_macro": float(macro),
                }

    if best is None:
        raise RuntimeError("no adaptation candidate selected")
    return best, candidates


def evaluate_task(
    task: str,
    *,
    seed: int,
    encoder: CachedSentenceEncoder,
    adaptation_strength: float,
    adaptation_decay: float,
) -> dict[str, float]:
    adaptive = AdaptiveResidualState(
        encoder,
        seed=seed,
        adaptation_strength=adaptation_strength,
        adaptation_decay=adaptation_decay,
        slow_decay=SLOW_DECAY,
        use_recurrence=True,
    )
    adaptation_only = AdaptiveResidualState(
        encoder,
        seed=seed,
        adaptation_strength=adaptation_strength,
        adaptation_decay=adaptation_decay,
        slow_decay=SLOW_DECAY,
        use_recurrence=False,
    )
    v57 = ResidualDrivenState(encoder, seed=seed, slow_decay=SLOW_DECAY)

    rows: list[dict[str, object]] = []
    for pair_id in range(PAIR_COUNT):
        case = build_case(task, pair_id)
        for label, sequence in ((0, case.class0), (1, case.class1)):
            adapted = run_adaptive(adaptive, sequence, case.current)
            reset = run_adaptive(
                adaptive, sequence, case.current, reset_fast_before_current=True
            )
            no_recurrence = run_adaptive(adaptation_only, sequence, case.current)
            frozen = run_v57(v57, sequence, case.current)
            rows.append(
                {
                    "pair_id": pair_id,
                    "label": label,
                    "adaptive": primary_fast_features(adapted),
                    "adaptive_raw": raw_fast_features(adapted),
                    "adaptive_reset": primary_fast_features(reset),
                    "adaptation_only": primary_fast_features(no_recurrence),
                    "v57": primary_fast_features(frozen),
                    "slow": slow_features(adapted),
                    "direct_residual": direct_residual_change_features(
                        encoder, list(sequence), case.current
                    ),
                    "relational": relational_features(sequence, encoder),
                }
            )

    train = [row for row in rows if int(row["pair_id"]) < TRAIN_PAIRS]
    test = [row for row in rows if int(row["pair_id"]) >= TRAIN_PAIRS]
    y_train = labels(train)
    y_test = labels(test)

    fields = (
        "adaptive",
        "adaptive_raw",
        "adaptation_only",
        "v57",
        "slow",
        "direct_residual",
        "relational",
    )
    probes = {
        field: DiagnosticRidge(RIDGE_ALPHA).fit(matrix(train, field), y_train)
        for field in fields
    }
    result = {
        field: accuracy(y_test, probe.predict(matrix(test, field)))
        for field, probe in probes.items()
    }
    result["adaptive_reset"] = accuracy(
        y_test,
        probes["adaptive"].predict(matrix(test, "adaptive_reset")),
    )
    return result


def beta_zero_max_diff(encoder: CachedSentenceEncoder) -> float:
    maximum = 0.0
    case = build_case("alternation", 0)
    for seed in SEEDS:
        ablated = AdaptiveResidualState(
            encoder,
            seed=seed,
            adaptation_strength=0.0,
            adaptation_decay=0.985,
            slow_decay=SLOW_DECAY,
            use_recurrence=True,
        )
        frozen = ResidualDrivenState(encoder, seed=seed, slow_decay=SLOW_DECAY)
        for text in [*case.class0, case.current]:
            a = ablated.consume_event(text)
            b = frozen.consume_event(text)
            maximum = max(
                maximum,
                float(np.max(np.abs(a.fast_trace.states - b.fast_trace.states))),
            )
    return maximum


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

    selected, calibration_rows = select_adaptation(encoder)
    strength = float(selected["adaptation_strength"])
    decay = float(selected["adaptation_decay"])

    task_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        for task in TASKS:
            task_rows.append(
                {
                    "seed": seed,
                    "task": task,
                    **evaluate_task(
                        task,
                        seed=seed,
                        encoder=encoder,
                        adaptation_strength=strength,
                        adaptation_decay=decay,
                    ),
                }
            )

    metric_names = [
        "adaptive",
        "adaptive_raw",
        "adaptive_reset",
        "adaptation_only",
        "v57",
        "slow",
        "direct_residual",
        "relational",
    ]
    mean = {
        metric: float(np.mean([float(row[metric]) for row in task_rows]))
        for metric in metric_names
    }
    per_task_adaptive = {
        task: float(
            np.mean([float(row["adaptive"]) for row in task_rows if row["task"] == task])
        )
        for task in TASKS
    }
    beta0_diff = beta_zero_max_diff(encoder)

    acceptance = {
        "temporal_suite_macro_at_least_0_70": mean["adaptive"] >= 0.70,
        "adaptive_beats_v57_by_0_10": mean["adaptive"] - mean["v57"] >= 0.10,
        "adaptive_beats_slow_by_0_12": mean["adaptive"] - mean["slow"] >= 0.12,
        "fast_reset_reduces_by_0_10": mean["adaptive"] - mean["adaptive_reset"] >= 0.10,
        "at_least_3_of_4_tasks_at_0_65": sum(v >= 0.65 for v in per_task_adaptive.values()) >= 3,
        "relational_validity_macro_at_least_0_95": mean["relational"] >= 0.95,
        "beta_zero_reproduces_v57": beta0_diff <= 1e-6,
    }
    acceptance["all_primary_gates"] = all(acceptance.values())
    complexity = {
        "adaptive_minus_adaptation_only": mean["adaptive"] - mean["adaptation_only"],
        "recurrence_justified_by_0_03": mean["adaptive"] - mean["adaptation_only"] >= 0.03,
    }

    summary = {
        "version": "v5.8",
        "purpose": "development test of activity-dependent adaptation as a label-free fast temporal state mechanism",
        "encoder": MODEL_NAME,
        "calibration_seed": CALIBRATION_SEED,
        "evaluation_seeds": SEEDS,
        "selection_uses_test_identities": False,
        "selected_adaptation": selected,
        "primary_readout": "per-tick population moments from the common-current fast trace, frozen from v5.7.1",
        "tasks": list(TASKS),
        "mean_accuracy": mean,
        "per_task_adaptive_accuracy": per_task_adaptive,
        "gaps": {
            "adaptive_minus_v57": mean["adaptive"] - mean["v57"],
            "adaptive_minus_slow": mean["adaptive"] - mean["slow"],
            "adaptive_minus_reset": mean["adaptive"] - mean["adaptive_reset"],
            "direct_residual_minus_adaptive": mean["direct_residual"] - mean["adaptive"],
        },
        "beta_zero_max_abs_trace_difference": beta0_diff,
        "acceptance": acceptance,
        "complexity": complexity,
        "claim_boundary": "development temporal-structure result only; no affect/emotion claim",
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "calibration_grid.csv", calibration_rows)
    write_csv(OUT_DIR / "per_seed_task_metrics.csv", task_rows)
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
