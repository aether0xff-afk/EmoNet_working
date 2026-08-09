from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve()
VERSION_ROOT = HERE.parents[1]
REPO_ROOT = HERE.parents[2]
V51_ROOT = REPO_ROOT / "v5_1_semantic_context"
V52_ROOT = REPO_ROOT / "v5_2_learned_memory"
V52_EXPERIMENTS = V52_ROOT / "experiments"
sys.path.insert(0, str(V51_ROOT))
sys.path.insert(0, str(V52_ROOT))
sys.path.insert(0, str(V52_EXPERIMENTS))

from semantic_fixture import SemanticArm, SemanticPair, build_semantic_pairs, flatten_pairs  # noqa: E402
from learned_core import LearnedCoreConfig  # noqa: E402
from run_learned_memory_benchmark import (  # noqa: E402
    MODEL_NAME,
    SEEDS,
    RIDGE_ALPHA,
    CachedSentenceEncoder,
    RidgeProbe,
    accuracy,
    all_texts,
    ema_features,
    learned_features,
    pairs_for_domain,
    sequence_tensor,
    train_core,
)


OUT_DIR = VERSION_ROOT / "outputs" / "memory_readout_diagnostic"


@torch.no_grad()
def reconstructed_lag3(model, sequences: torch.Tensor) -> np.ndarray:
    states, _ = model.run_sequence(sequences, return_event_traces=False)
    prediction = F.normalize(model.memory_heads[2](states[-1]), dim=-1)
    return prediction.cpu().numpy().astype(np.float32)


def semantic_event_map(
    arms: list[SemanticArm],
    encoder: CachedSentenceEncoder,
) -> dict[tuple[str, int], np.ndarray]:
    return {
        (arm.pair_id, arm.label): encoder.encode(arm.history[1]).astype(np.float32)
        for arm in arms
    }


def array_map(arms: list[SemanticArm], values: np.ndarray) -> dict[tuple[str, int], np.ndarray]:
    return {
        (arm.pair_id, arm.label): np.asarray(values[index], dtype=np.float32)
        for index, arm in enumerate(arms)
    }


def domain_diagnostic(
    domain: str,
    train_pairs: list[SemanticPair],
    test_pairs: list[SemanticPair],
    semantic_train: dict[tuple[str, int], np.ndarray],
    semantic_test: dict[tuple[str, int], np.ndarray],
    recon_train: dict[tuple[str, int], np.ndarray],
    recon_test: dict[tuple[str, int], np.ndarray],
    raw_train: dict[tuple[str, int], np.ndarray],
    raw_test: dict[tuple[str, int], np.ndarray],
    ema_train: dict[tuple[str, int], np.ndarray],
    ema_test: dict[tuple[str, int], np.ndarray],
) -> dict[str, float]:
    train_arms = flatten_pairs(pairs_for_domain(train_pairs, domain))
    test_arms = flatten_pairs(pairs_for_domain(test_pairs, domain))
    y_train = np.asarray([arm.label for arm in train_arms], dtype=np.int64)
    y_test = np.asarray([arm.label for arm in test_arms], dtype=np.int64)

    def matrix(mapping: dict[tuple[str, int], np.ndarray], arms: list[SemanticArm]) -> np.ndarray:
        return np.stack([mapping[(arm.pair_id, arm.label)] for arm in arms])

    x_sem_train = matrix(semantic_train, train_arms)
    x_sem_test = matrix(semantic_test, test_arms)
    x_recon_train = matrix(recon_train, train_arms)
    x_recon_test = matrix(recon_test, test_arms)
    x_raw_train = matrix(raw_train, train_arms)
    x_raw_test = matrix(raw_test, test_arms)
    x_ema_train = matrix(ema_train, train_arms)
    x_ema_test = matrix(ema_test, test_arms)

    semantic_probe = RidgeProbe(RIDGE_ALPHA).fit(x_sem_train, y_train)
    recon_probe = RidgeProbe(RIDGE_ALPHA).fit(x_recon_train, y_train)
    raw_probe = RidgeProbe(RIDGE_ALPHA).fit(x_raw_train, y_train)
    ema_probe = RidgeProbe(RIDGE_ALPHA).fit(x_ema_train, y_train)

    return {
        "semantic_input": accuracy(y_test, semantic_probe.predict(x_sem_test)),
        "semantic_geometry_transfer": accuracy(y_test, semantic_probe.predict(x_recon_test)),
        "reconstruction_native_probe": accuracy(y_test, recon_probe.predict(x_recon_test)),
        "raw_trace_probe": accuracy(y_test, raw_probe.predict(x_raw_test)),
        "ema_probe": accuracy(y_test, ema_probe.predict(x_ema_test)),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    train_pairs, test_pairs = build_semantic_pairs()
    train_arms = flatten_pairs(train_pairs)
    test_arms = flatten_pairs(test_pairs)

    encoder = CachedSentenceEncoder(MODEL_NAME)
    encoder.preload(all_texts(train_pairs + test_pairs))
    train_sequences = sequence_tensor(train_arms, encoder)
    test_sequences = sequence_tensor(test_arms, encoder)

    semantic_train = semantic_event_map(train_arms, encoder)
    semantic_test = semantic_event_map(test_arms, encoder)
    ema_train = ema_features(train_arms, encoder)
    ema_test = ema_features(test_arms, encoder)

    rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []

    for seed in SEEDS:
        model, history = train_core(seed, train_sequences, LearnedCoreConfig())
        train_recon_values = reconstructed_lag3(model, train_sequences)
        test_recon_values = reconstructed_lag3(model, test_sequences)
        recon_train = array_map(train_arms, train_recon_values)
        recon_test = array_map(test_arms, test_recon_values)
        raw_train, _ = learned_features(model, train_arms, encoder)
        raw_test, _ = learned_features(model, test_arms, encoder)

        lag3_train = model.lag_cosine_at_final(train_sequences, lag=3)
        lag3_test = model.lag_cosine_at_final(test_sequences, lag=3)
        seed_rows.append(
            {
                "seed": seed,
                "final_train_loss": history[-1]["loss"],
                "lag3_train_cosine": lag3_train,
                "lag3_test_cosine": lag3_test,
            }
        )

        for domain in sorted({pair.domain for pair in train_pairs}):
            metrics = domain_diagnostic(
                domain,
                train_pairs,
                test_pairs,
                semantic_train,
                semantic_test,
                recon_train,
                recon_test,
                raw_train,
                raw_test,
                ema_train,
                ema_test,
            )
            rows.append({"seed": seed, "domain": domain, **metrics})

    metric_names = [
        "semantic_input",
        "semantic_geometry_transfer",
        "reconstruction_native_probe",
        "raw_trace_probe",
        "ema_probe",
    ]
    macro = {
        name: float(np.mean([float(row[name]) for row in rows]))
        for name in metric_names
    }
    std = {
        name: float(np.std([float(row[name]) for row in rows]))
        for name in metric_names
    }

    native = macro["reconstruction_native_probe"]
    transfer = macro["semantic_geometry_transfer"]
    raw = macro["raw_trace_probe"]

    if native >= 0.70 and raw <= native - 0.10:
        diagnosis = "raw_readout_sample_efficiency_bottleneck"
    elif native < 0.65:
        diagnosis = "cosine_objective_information_bottleneck"
    elif transfer <= native - 0.10:
        diagnosis = "semantic_geometry_distortion"
    else:
        diagnosis = "mixed_or_unresolved"

    summary = {
        "version": "v5.2.1",
        "purpose": "frozen-v5.2 memory readout diagnostic",
        "core_changed_from_v5_2": False,
        "training_protocol_changed_from_v5_2": False,
        "mean_lag3_train_cosine": float(
            np.mean([float(row["lag3_train_cosine"]) for row in seed_rows])
        ),
        "mean_lag3_test_cosine": float(
            np.mean([float(row["lag3_test_cosine"]) for row in seed_rows])
        ),
        "macro_accuracy": macro,
        "std_accuracy": std,
        "diagnosis": diagnosis,
        "diagnostic_gaps": {
            "native_minus_raw": native - raw,
            "native_minus_transfer": native - transfer,
            "input_minus_native": macro["semantic_input"] - native,
            "ema_minus_native": macro["ema_probe"] - native,
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "per_seed_domain_metrics.csv", rows)
    write_csv(OUT_DIR / "per_seed_training_metrics.csv", seed_rows)
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
