from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve()
VERSION_ROOT = HERE.parents[1]
REPO_ROOT = HERE.parents[2]
V52_ROOT = REPO_ROOT / "v5_2_learned_memory"
V52_EXPERIMENTS = V52_ROOT / "experiments"
V53_ROOT = REPO_ROOT / "v5_3_contrastive_memory"
V53_EXPERIMENTS = V53_ROOT / "experiments"
sys.path.insert(0, str(VERSION_ROOT))
sys.path.insert(0, str(V52_ROOT))
sys.path.insert(0, str(V52_EXPERIMENTS))
sys.path.insert(0, str(V53_EXPERIMENTS))

from fresh_fixture import FreshArm, FreshPair, build_fresh_pairs, flatten_pairs  # noqa: E402
from run_learned_memory_benchmark import (  # noqa: E402
    EMA_DECAY,
    LEARNING_RATE,
    MODEL_NAME,
    RIDGE_ALPHA,
    SEEDS,
    TRAIN_EPOCHS,
    WEIGHT_DECAY,
    CachedSentenceEncoder,
    RidgeProbe,
    accuracy,
    ema_features,
    learned_features,
    random_features,
    sequence_tensor,
)
from run_contrastive_memory_benchmark import (  # noqa: E402
    TEMPERATURE,
    build_event_vocabulary,
    heldout_lag3_retrieval,
    train_contrastive_core,
)


OUT_DIR = VERSION_ROOT / "outputs" / "fresh_confirmatory"


def pairs_for_domain(pairs: list[FreshPair], domain: str) -> list[FreshPair]:
    return [pair for pair in pairs if pair.domain == domain]


def collect_texts(pairs: list[FreshPair]) -> set[str]:
    result: set[str] = set()
    for arm in flatten_pairs(pairs):
        result.update(arm.history)
        result.add(arm.current_text)
    return result


def fixture_fingerprint(train_pairs: list[FreshPair], test_pairs: list[FreshPair]) -> str:
    rows: list[dict[str, object]] = []
    for pair in train_pairs + test_pairs:
        for arm in (pair.positive, pair.negative):
            rows.append(
                {
                    "pair_id": arm.pair_id,
                    "split": arm.split,
                    "domain": arm.domain,
                    "label": arm.label,
                    "history": list(arm.history),
                    "current_text": arm.current_text,
                }
            )
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def evaluate_feature_map(
    train_map: dict[tuple[str, int], np.ndarray],
    test_map: dict[tuple[str, int], np.ndarray],
    train_pairs: list[FreshPair],
    test_pairs: list[FreshPair],
) -> tuple[float, dict[str, float]]:
    domain_scores: dict[str, float] = {}
    for domain in sorted({pair.domain for pair in train_pairs}):
        train_arms = flatten_pairs(pairs_for_domain(train_pairs, domain))
        test_arms = flatten_pairs(pairs_for_domain(test_pairs, domain))
        y_train = np.asarray([arm.label for arm in train_arms], dtype=np.int64)
        y_test = np.asarray([arm.label for arm in test_arms], dtype=np.int64)
        x_train = np.stack([train_map[(arm.pair_id, arm.label)] for arm in train_arms])
        x_test = np.stack([test_map[(arm.pair_id, arm.label)] for arm in test_arms])
        probe = RidgeProbe(RIDGE_ALPHA).fit(x_train, y_train)
        domain_scores[domain] = accuracy(y_test, probe.predict(x_test))
    return float(np.mean(list(domain_scores.values()))), domain_scores


def evaluate_contrastive_controls(
    train_real: dict[tuple[str, int], np.ndarray],
    test_real: dict[tuple[str, int], np.ndarray],
    test_reset: dict[tuple[str, int], np.ndarray],
    train_pairs: list[FreshPair],
    test_pairs: list[FreshPair],
) -> tuple[float, float, float, dict[str, dict[str, float]]]:
    per_domain: dict[str, dict[str, float]] = {}
    real_scores: list[float] = []
    reset_scores: list[float] = []
    wrong_scores: list[float] = []

    for domain in sorted({pair.domain for pair in train_pairs}):
        train_arms = flatten_pairs(pairs_for_domain(train_pairs, domain))
        test_arms = flatten_pairs(pairs_for_domain(test_pairs, domain))
        y_train = np.asarray([arm.label for arm in train_arms], dtype=np.int64)
        y_test = np.asarray([arm.label for arm in test_arms], dtype=np.int64)

        x_train = np.stack([train_real[(arm.pair_id, arm.label)] for arm in train_arms])
        probe = RidgeProbe(RIDGE_ALPHA).fit(x_train, y_train)
        x_real = np.stack([test_real[(arm.pair_id, arm.label)] for arm in test_arms])
        x_reset = np.stack([test_reset[(arm.pair_id, arm.label)] for arm in test_arms])
        x_wrong = np.stack(
            [
                test_real[(arm.pair_id, 0 if arm.label == 1 else 1)]
                for arm in test_arms
            ]
        )

        real_acc = accuracy(y_test, probe.predict(x_real))
        reset_acc = accuracy(y_test, probe.predict(x_reset))
        wrong_acc = accuracy(y_test, probe.predict(x_wrong))
        real_scores.append(real_acc)
        reset_scores.append(reset_acc)
        wrong_scores.append(wrong_acc)
        per_domain[domain] = {
            "real": real_acc,
            "reset": reset_acc,
            "wrong": wrong_acc,
        }

    return (
        float(np.mean(real_scores)),
        float(np.mean(reset_scores)),
        float(np.mean(wrong_scores)),
        per_domain,
    )


def semantic_event_input_metrics(
    encoder: CachedSentenceEncoder,
    train_pairs: list[FreshPair],
    test_pairs: list[FreshPair],
) -> tuple[float, dict[str, float]]:
    train_map = {
        (arm.pair_id, arm.label): encoder.encode(arm.history[1]).astype(np.float32)
        for arm in flatten_pairs(train_pairs)
    }
    test_map = {
        (arm.pair_id, arm.label): encoder.encode(arm.history[1]).astype(np.float32)
        for arm in flatten_pairs(test_pairs)
    }
    return evaluate_feature_map(train_map, test_map, train_pairs, test_pairs)


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

    fixture_sha = fixture_fingerprint(train_pairs, test_pairs)
    input_macro, input_domains = semantic_event_input_metrics(encoder, train_pairs, test_pairs)

    _, train_vocab_embeddings, train_event_ids = build_event_vocabulary(train_arms, encoder)
    test_vocab, test_vocab_embeddings, test_event_ids = build_event_vocabulary(test_arms, encoder)

    ema_train = ema_features(train_arms, encoder)
    ema_test = ema_features(test_arms, encoder)
    ema_macro, ema_domains = evaluate_feature_map(ema_train, ema_test, train_pairs, test_pairs)

    seed_rows: list[dict[str, object]] = []
    domain_rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []

    for seed in SEEDS:
        model, history = train_contrastive_core(
            seed,
            train_sequences,
            train_event_ids,
            train_vocab_embeddings,
        )
        for row in history:
            training_rows.append({"seed": seed, **row})

        retrieval = heldout_lag3_retrieval(
            model,
            test_sequences,
            test_event_ids,
            test_vocab_embeddings,
        )
        lag3_cosine = model.lag_cosine_at_final(test_sequences, lag=3)

        learned_train, _ = learned_features(model, train_arms, encoder)
        learned_test, learned_reset = learned_features(model, test_arms, encoder)
        real_macro, reset_macro, wrong_macro, learned_domains = evaluate_contrastive_controls(
            learned_train,
            learned_test,
            learned_reset,
            train_pairs,
            test_pairs,
        )

        random_train = random_features(seed, train_arms, encoder)
        random_test = random_features(seed, test_arms, encoder)
        random_macro, random_domains = evaluate_feature_map(
            random_train,
            random_test,
            train_pairs,
            test_pairs,
        )

        seed_rows.append(
            {
                "seed": seed,
                "final_train_loss": history[-1]["loss"],
                "heldout_lag3_retrieval_top1": retrieval,
                "heldout_lag3_cosine": lag3_cosine,
                "contrastive_real_macro": real_macro,
                "contrastive_reset_macro": reset_macro,
                "contrastive_wrong_macro": wrong_macro,
                "v5_0_random_macro": random_macro,
                "ema_macro": ema_macro,
            }
        )

        for domain in sorted(learned_domains):
            domain_rows.append(
                {
                    "seed": seed,
                    "domain": domain,
                    "contrastive_real": learned_domains[domain]["real"],
                    "contrastive_reset": learned_domains[domain]["reset"],
                    "contrastive_wrong": learned_domains[domain]["wrong"],
                    "v5_0_random": random_domains[domain],
                    "ema": ema_domains[domain],
                    "semantic_input": input_domains[domain],
                }
            )

    def mean(field: str) -> float:
        return float(np.mean([float(row[field]) for row in seed_rows]))

    retrieval = mean("heldout_lag3_retrieval_top1")
    real = mean("contrastive_real_macro")
    reset = mean("contrastive_reset_macro")
    wrong = mean("contrastive_wrong_macro")
    random = mean("v5_0_random_macro")
    seed_pass_count = sum(float(row["contrastive_real_macro"]) >= 0.65 for row in seed_rows)

    acceptance = {
        "heldout_lag3_retrieval_top1_at_least_0_20": retrieval >= 0.20,
        "contrastive_semantic_macro_at_least_0_70": real >= 0.70,
        "contrastive_beats_random_by_0_10": real - random >= 0.10,
        "contrastive_beats_reset_by_0_15": real - reset >= 0.15,
        "contrastive_beats_wrong_by_0_15": real - wrong >= 0.15,
        "at_least_4_of_5_seeds_at_or_above_0_65": seed_pass_count >= 4,
    }
    acceptance["confirmatory_semantic_memory_pass"] = all(acceptance.values())

    protocol = {
        "encoder": MODEL_NAME,
        "seeds": SEEDS,
        "epochs": TRAIN_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "temperature": TEMPERATURE,
        "ridge_alpha": RIDGE_ALPHA,
        "ema_decay": EMA_DECAY,
        "train_pairs": len(train_pairs),
        "test_pairs": len(test_pairs),
        "heldout_event_vocabulary_size": len(test_vocab),
    }

    summary = {
        "version": "v5.4",
        "purpose": "fresh preregistered semantic-memory confirmatory test",
        "fixture_sha256": fixture_sha,
        "github": {
            "sha": os.environ.get("GITHUB_SHA"),
            "run_id": os.environ.get("GITHUB_RUN_ID"),
        },
        "task_labels_used_by_core": False,
        "emotion_labels_used_by_core": False,
        "protocol": protocol,
        "diagnostic_input_semantic_macro": input_macro,
        "mean": {
            "heldout_lag3_retrieval_top1": retrieval,
            "heldout_lag3_cosine": mean("heldout_lag3_cosine"),
            "contrastive_real_macro": real,
            "contrastive_reset_macro": reset,
            "contrastive_wrong_macro": wrong,
            "v5_0_random_macro": random,
            "ema_embedding_memory_macro": ema_macro,
        },
        "gaps": {
            "contrastive_minus_random": real - random,
            "contrastive_minus_reset": real - reset,
            "contrastive_minus_wrong": real - wrong,
            "contrastive_minus_ema": real - ema_macro,
        },
        "seed_pass_count_at_or_above_0_65": seed_pass_count,
        "acceptance": acceptance,
        "complexity_check": {
            "contrastive_reaches_or_beats_ema": real >= ema_macro,
            "ema_advantage_if_positive": ema_macro - real,
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "per_seed_metrics.csv", seed_rows)
    write_csv(OUT_DIR / "per_domain_metrics.csv", domain_rows)
    write_csv(OUT_DIR / "training_curve.csv", training_rows)
    (OUT_DIR / "protocol_manifest.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
