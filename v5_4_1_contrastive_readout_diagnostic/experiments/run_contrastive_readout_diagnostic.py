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
V52_ROOT = REPO_ROOT / "v5_2_learned_memory"
V52_EXPERIMENTS = V52_ROOT / "experiments"
V53_EXPERIMENTS = REPO_ROOT / "v5_3_contrastive_memory" / "experiments"
V54_ROOT = REPO_ROOT / "v5_4_fresh_confirmatory"
sys.path.insert(0, str(VERSION_ROOT))
sys.path.insert(0, str(V52_ROOT))
sys.path.insert(0, str(V52_EXPERIMENTS))
sys.path.insert(0, str(V53_EXPERIMENTS))
sys.path.insert(0, str(V54_ROOT))

from fresh_fixture import FreshArm, FreshPair, build_fresh_pairs, flatten_pairs  # noqa: E402
from run_learned_memory_benchmark import (  # noqa: E402
    MODEL_NAME,
    RIDGE_ALPHA,
    SEEDS,
    CachedSentenceEncoder,
    RidgeProbe,
    accuracy,
    ema_features,
    learned_features,
    sequence_tensor,
)
from run_contrastive_memory_benchmark import (  # noqa: E402
    build_event_vocabulary,
    heldout_lag3_retrieval,
    train_contrastive_core,
)


OUT_DIR = VERSION_ROOT / "outputs" / "contrastive_readout_diagnostic"


def pairs_for_domain(pairs: list[FreshPair], domain: str) -> list[FreshPair]:
    return [pair for pair in pairs if pair.domain == domain]


def collect_texts(pairs: list[FreshPair]) -> set[str]:
    texts: set[str] = set()
    for arm in flatten_pairs(pairs):
        texts.update(arm.history)
        texts.add(arm.current_text)
    return texts


@torch.no_grad()
def lag3_head_output(model, sequences: torch.Tensor) -> np.ndarray:
    states, _ = model.run_sequence(sequences, return_event_traces=False)
    prediction = F.normalize(model.memory_heads[2](states[-1]), dim=-1)
    return prediction.cpu().numpy().astype(np.float32)


def array_map(arms: list[FreshArm], values: np.ndarray) -> dict[tuple[str, int], np.ndarray]:
    return {
        (arm.pair_id, arm.label): np.asarray(values[index], dtype=np.float32)
        for index, arm in enumerate(arms)
    }


def semantic_map(
    arms: list[FreshArm],
    encoder: CachedSentenceEncoder,
) -> dict[tuple[str, int], np.ndarray]:
    return {
        (arm.pair_id, arm.label): encoder.encode(arm.history[1]).astype(np.float32)
        for arm in arms
    }


def domain_readout_metrics(
    domain: str,
    train_pairs: list[FreshPair],
    test_pairs: list[FreshPair],
    semantic_train: dict[tuple[str, int], np.ndarray],
    semantic_test: dict[tuple[str, int], np.ndarray],
    head_train: dict[tuple[str, int], np.ndarray],
    head_test: dict[tuple[str, int], np.ndarray],
    raw_train: dict[tuple[str, int], np.ndarray],
    raw_test: dict[tuple[str, int], np.ndarray],
    ema_train: dict[tuple[str, int], np.ndarray],
    ema_test: dict[tuple[str, int], np.ndarray],
) -> dict[str, float]:
    train_arms = flatten_pairs(pairs_for_domain(train_pairs, domain))
    test_arms = flatten_pairs(pairs_for_domain(test_pairs, domain))
    y_train = np.asarray([arm.label for arm in train_arms], dtype=np.int64)
    y_test = np.asarray([arm.label for arm in test_arms], dtype=np.int64)

    def matrix(mapping: dict[tuple[str, int], np.ndarray], arms: list[FreshArm]) -> np.ndarray:
        return np.stack([mapping[(arm.pair_id, arm.label)] for arm in arms])

    x_sem_train = matrix(semantic_train, train_arms)
    x_sem_test = matrix(semantic_test, test_arms)
    x_head_train = matrix(head_train, train_arms)
    x_head_test = matrix(head_test, test_arms)
    x_raw_train = matrix(raw_train, train_arms)
    x_raw_test = matrix(raw_test, test_arms)
    x_ema_train = matrix(ema_train, train_arms)
    x_ema_test = matrix(ema_test, test_arms)

    semantic_probe = RidgeProbe(RIDGE_ALPHA).fit(x_sem_train, y_train)
    head_probe = RidgeProbe(RIDGE_ALPHA).fit(x_head_train, y_train)
    raw_probe = RidgeProbe(RIDGE_ALPHA).fit(x_raw_train, y_train)
    ema_probe = RidgeProbe(RIDGE_ALPHA).fit(x_ema_train, y_train)

    return {
        "semantic_input": accuracy(y_test, semantic_probe.predict(x_sem_test)),
        "semantic_geometry_transfer": accuracy(y_test, semantic_probe.predict(x_head_test)),
        "memory_head_native": accuracy(y_test, head_probe.predict(x_head_test)),
        "raw_trace": accuracy(y_test, raw_probe.predict(x_raw_test)),
        "ema": accuracy(y_test, ema_probe.predict(x_ema_test)),
    }


def semantic_candidate_retrieval(
    model,
    test_arms: list[FreshArm],
    test_sequences: torch.Tensor,
    encoder: CachedSentenceEncoder,
) -> tuple[float, float, float]:
    """Return exact, polarity, and domain accuracy over semantic-event candidates only."""

    candidate_texts = sorted({arm.history[1] for arm in test_arms})
    candidate_index = {text: idx for idx, text in enumerate(candidate_texts)}
    candidates = torch.from_numpy(np.stack([encoder.encode(text) for text in candidate_texts]))
    candidates = F.normalize(candidates.float(), dim=-1)

    text_to_label: dict[str, int] = {}
    text_to_domain: dict[str, str] = {}
    for arm in test_arms:
        text_to_label[arm.history[1]] = arm.label
        text_to_domain[arm.history[1]] = arm.domain

    with torch.no_grad():
        states, _ = model.run_sequence(test_sequences, return_event_traces=False)
        prediction = F.normalize(model.memory_heads[2](states[-1]), dim=-1)
        logits = prediction @ candidates.T
        top = logits.argmax(dim=-1).cpu().numpy()

    exact = 0
    polarity = 0
    domain = 0
    for index, arm in enumerate(test_arms):
        predicted_text = candidate_texts[int(top[index])]
        if predicted_text == arm.history[1]:
            exact += 1
        if text_to_label[predicted_text] == arm.label:
            polarity += 1
        if text_to_domain[predicted_text] == arm.domain:
            domain += 1

    n = len(test_arms)
    return exact / n, polarity / n, domain / n


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    train_pairs, test_pairs = build_fresh_pairs()
    train_arms = flatten_pairs(train_pairs)
    test_arms = flatten_pairs(test_pairs)

    encoder = CachedSentenceEncoder(MODEL_NAME)
    encoder.preload(collect_texts(train_pairs + test_pairs))
    train_sequences = sequence_tensor(train_arms, encoder)
    test_sequences = sequence_tensor(test_arms, encoder)

    _, train_vocab_embeddings, train_event_ids = build_event_vocabulary(train_arms, encoder)
    test_vocab, test_vocab_embeddings, test_event_ids = build_event_vocabulary(test_arms, encoder)

    semantic_train = semantic_map(train_arms, encoder)
    semantic_test = semantic_map(test_arms, encoder)
    ema_train = ema_features(train_arms, encoder)
    ema_test = ema_features(test_arms, encoder)

    domain_rows: list[dict[str, object]] = []
    retrieval_rows: list[dict[str, object]] = []

    for seed in SEEDS:
        model, _ = train_contrastive_core(
            seed,
            train_sequences,
            train_event_ids,
            train_vocab_embeddings,
        )

        train_head = array_map(train_arms, lag3_head_output(model, train_sequences))
        test_head = array_map(test_arms, lag3_head_output(model, test_sequences))
        raw_train, _ = learned_features(model, train_arms, encoder)
        raw_test, _ = learned_features(model, test_arms, encoder)

        for domain in sorted({pair.domain for pair in train_pairs}):
            metrics = domain_readout_metrics(
                domain,
                train_pairs,
                test_pairs,
                semantic_train,
                semantic_test,
                train_head,
                test_head,
                raw_train,
                raw_test,
                ema_train,
                ema_test,
            )
            domain_rows.append({"seed": seed, "domain": domain, **metrics})

        all_vocab_exact = heldout_lag3_retrieval(
            model,
            test_sequences,
            test_event_ids,
            test_vocab_embeddings,
        )
        semantic_exact, polarity_accuracy, domain_accuracy = semantic_candidate_retrieval(
            model,
            test_arms,
            test_sequences,
            encoder,
        )
        retrieval_rows.append(
            {
                "seed": seed,
                "all_event_exact_top1": all_vocab_exact,
                "semantic_candidate_exact_top1": semantic_exact,
                "semantic_candidate_polarity_accuracy": polarity_accuracy,
                "semantic_candidate_domain_accuracy": domain_accuracy,
            }
        )

    readout_names = [
        "semantic_input",
        "semantic_geometry_transfer",
        "memory_head_native",
        "raw_trace",
        "ema",
    ]
    macro = {
        name: float(np.mean([float(row[name]) for row in domain_rows]))
        for name in readout_names
    }
    retrieval_mean = {
        key: float(np.mean([float(row[key]) for row in retrieval_rows]))
        for key in (
            "all_event_exact_top1",
            "semantic_candidate_exact_top1",
            "semantic_candidate_polarity_accuracy",
            "semantic_candidate_domain_accuracy",
        )
    }

    native = macro["memory_head_native"]
    transfer = macro["semantic_geometry_transfer"]
    raw = macro["raw_trace"]
    polarity = retrieval_mean["semantic_candidate_polarity_accuracy"]

    if native >= 0.72 and raw <= native - 0.08:
        diagnosis = "raw_trace_readout_geometry_bottleneck"
    elif native >= 0.70 and transfer <= native - 0.10:
        diagnosis = "semantic_geometry_distortion"
    elif polarity < 0.70 and retrieval_mean["semantic_candidate_exact_top1"] >= 0.30:
        diagnosis = "instance_identity_without_stable_state_abstraction"
    elif native < 0.65:
        diagnosis = "contrastive_head_lacks_robust_semantic_state"
    else:
        diagnosis = "mixed_or_unresolved"

    summary = {
        "version": "v5.4.1",
        "purpose": "diagnose v5.4 exact-retrieval / semantic-state mismatch",
        "core_changed_from_v5_4": False,
        "objective_changed_from_v5_4": False,
        "fixture_changed_from_v5_4": False,
        "macro_accuracy": macro,
        "retrieval": retrieval_mean,
        "diagnosis": diagnosis,
        "diagnostic_gaps": {
            "memory_head_native_minus_raw_trace": native - raw,
            "memory_head_native_minus_geometry_transfer": native - transfer,
            "semantic_input_minus_memory_head_native": macro["semantic_input"] - native,
            "ema_minus_memory_head_native": macro["ema"] - native,
            "polarity_minus_chance": polarity - 0.5,
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "per_seed_domain_readouts.csv", domain_rows)
    write_csv(OUT_DIR / "per_seed_retrieval.csv", retrieval_rows)
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
