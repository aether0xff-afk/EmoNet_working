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
):
    sys.path.insert(0, str(path))

from geometry import activation_energy, change_energy, full_geometry, population_moments  # noqa: E402
from residual_state import ResidualDrivenState, fast_features  # noqa: E402
from run_learned_memory_benchmark import MODEL_NAME, RIDGE_ALPHA, SEEDS, CachedSentenceEncoder  # noqa: E402
from run_readout_temporal_diagnostic import (  # noqa: E402
    DiagnosticRidge,
    STRUCTURE_PAIR_COUNT,
    STRUCTURE_TRAIN_PAIRS,
    accuracy,
    relational_structure_features,
    structural_pair,
)
from run_residual_fast_benchmark import (  # noqa: E402
    SLOW_DECAY,
    direct_residual_change_features,
    run_residual_condition,
)

OUT_DIR = VERSION_ROOT / "outputs" / "fast_trace_geometry"


def _matrix(rows: list[dict[str, object]], field: str) -> np.ndarray:
    return np.stack([np.asarray(row[field], dtype=np.float32) for row in rows])


def build_seed_rows(seed: int, encoder: CachedSentenceEncoder) -> list[dict[str, object]]:
    model = ResidualDrivenState(encoder, seed=seed, slow_decay=SLOW_DECAY)
    rows: list[dict[str, object]] = []
    for pair_id in range(STRUCTURE_PAIR_COUNT):
        class0, class1, current, _, _ = structural_pair(pair_id)
        split = "train" if pair_id < STRUCTURE_TRAIN_PAIRS else "test"
        for label, sequence in ((0, class0), (1, class1)):
            observation = run_residual_condition(
                model, sequence, current, reset_fast_before_current=False
            )
            reset_observation = run_residual_condition(
                model, sequence, current, reset_fast_before_current=True
            )
            trace = observation.fast_trace
            reset_trace = reset_observation.fast_trace
            act = activation_energy(trace)
            change = change_energy(trace)
            rows.append(
                {
                    "split": split,
                    "label": label,
                    "raw_coordinates": fast_features(observation),
                    "activation_energy": act,
                    "change_energy": change,
                    "energy_trajectory": np.concatenate([act, change]),
                    "population_moments": population_moments(trace),
                    "full_geometry": full_geometry(trace),
                    "full_geometry_reset": full_geometry(reset_trace),
                    "current_residual_vector": observation.residual_input,
                    "current_residual_norm": np.asarray(
                        [np.linalg.norm(observation.residual_input)], dtype=np.float32
                    ),
                    "full_residual_change": direct_residual_change_features(
                        encoder, sequence, current
                    ),
                    "relational": relational_structure_features(sequence, encoder),
                }
            )
    return rows


def evaluate_seed(seed: int, encoder: CachedSentenceEncoder) -> dict[str, float]:
    rows = build_seed_rows(seed, encoder)
    train = [row for row in rows if row["split"] == "train"]
    test = [row for row in rows if row["split"] == "test"]
    y_train = np.asarray([row["label"] for row in train], dtype=np.int64)
    y_test = np.asarray([row["label"] for row in test], dtype=np.int64)

    fields = (
        "raw_coordinates",
        "activation_energy",
        "change_energy",
        "energy_trajectory",
        "population_moments",
        "full_geometry",
        "current_residual_vector",
        "current_residual_norm",
        "full_residual_change",
        "relational",
    )
    probes = {
        field: DiagnosticRidge(RIDGE_ALPHA).fit(_matrix(train, field), y_train)
        for field in fields
    }
    result = {
        field: accuracy(y_test, probe.predict(_matrix(test, field)))
        for field, probe in probes.items()
    }
    result["full_geometry_reset"] = accuracy(
        y_test,
        probes["full_geometry"].predict(_matrix(test, "full_geometry_reset")),
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
    for pair_id in range(STRUCTURE_PAIR_COUNT):
        class0, class1, current, _, _ = structural_pair(pair_id)
        texts.update(class0)
        texts.update(class1)
        texts.add(current)
    encoder.preload(texts)

    per_seed: list[dict[str, object]] = []
    for seed in SEEDS:
        per_seed.append({"seed": seed, **evaluate_seed(seed, encoder)})

    names = [key for key in per_seed[0] if key != "seed"]
    mean = {
        name: float(np.mean([float(row[name]) for row in per_seed])) for name in names
    }
    geometry_candidates = [
        "activation_energy",
        "change_energy",
        "energy_trajectory",
        "population_moments",
        "full_geometry",
    ]
    best_geometry_name = max(geometry_candidates, key=lambda key: mean[key])
    best_geometry = mean[best_geometry_name]
    raw = mean["raw_coordinates"]
    reset = mean["full_geometry_reset"]
    direct = mean["full_residual_change"]

    if best_geometry >= 0.70 and best_geometry - raw >= 0.10:
        diagnosis = "useful_temporal_signal_survives_in_nonlinear_trace_geometry"
    elif direct >= 0.90 and best_geometry < 0.65:
        diagnosis = "recurrent_trace_fails_to_preserve_or_expose_identity_invariant_change_structure"
    else:
        diagnosis = "mixed_or_unresolved"

    summary = {
        "version": "v5.7.1",
        "purpose": "diagnose whether v5.7 temporal signal survives in fast-trace geometry",
        "state_generator_changed_from_v5_7": False,
        "train_test_token_identities_disjoint": True,
        "mean_accuracy": mean,
        "best_trace_geometry": {
            "name": best_geometry_name,
            "accuracy": best_geometry,
            "advantage_over_raw_coordinates": best_geometry - raw,
            "advantage_over_reset": best_geometry - reset,
        },
        "direct_residual_advantage_over_best_trace_geometry": direct - best_geometry,
        "diagnosis": diagnosis,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "per_seed_metrics.csv", per_seed)
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
