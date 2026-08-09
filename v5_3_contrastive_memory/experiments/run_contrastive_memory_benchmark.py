from __future__ import annotations

import copy
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
from learned_core import LearnedCoreConfig, LearnedLeakyRecurrentCore  # noqa: E402
from run_learned_memory_benchmark import (  # noqa: E402
    LEARNING_RATE,
    MODEL_NAME,
    RIDGE_ALPHA,
    SEEDS,
    TRAIN_EPOCHS,
    WEIGHT_DECAY,
    CachedSentenceEncoder,
    all_texts,
    ema_features,
    evaluate_learned_controls,
    evaluate_map,
    learned_features,
    random_features,
    sequence_tensor,
)


TEMPERATURE = 0.07
V52_COSINE_MACRO = 0.56
OUT_DIR = VERSION_ROOT / "outputs" / "contrastive_memory_benchmark"


def arm_text_sequence(arm: SemanticArm) -> tuple[str, ...]:
    return (*arm.history, arm.current_text)


def build_event_vocabulary(
    arms: list[SemanticArm],
    encoder: CachedSentenceEncoder,
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    vocab = sorted({text for arm in arms for text in arm_text_sequence(arm)})
    index = {text: idx for idx, text in enumerate(vocab)}
    vectors = torch.from_numpy(np.stack([encoder.encode(text) for text in vocab], axis=0))
    ids = torch.tensor(
        [[index[text] for text in arm_text_sequence(arm)] for arm in arms],
        dtype=torch.long,
    )
    return vocab, F.normalize(vectors.float(), dim=-1), ids


def contrastive_delayed_loss(
    model: LearnedLeakyRecurrentCore,
    embeddings: torch.Tensor,
    event_ids: torch.Tensor,
    vocabulary_embeddings: torch.Tensor,
) -> tuple[torch.Tensor, dict[int, float]]:
    states, _ = model.run_sequence(embeddings, return_event_traces=False)
    terms: list[torch.Tensor] = []
    per_lag: dict[int, list[torch.Tensor]] = {
        lag: [] for lag in range(1, model.config.max_lag + 1)
    }
    for event_index, state in enumerate(states):
        for lag in range(1, model.config.max_lag + 1):
            target_index = event_index - lag
            if target_index < 0:
                continue
            prediction = F.normalize(model.memory_heads[lag - 1](state), dim=-1)
            logits = prediction @ vocabulary_embeddings.T / TEMPERATURE
            target = event_ids[:, target_index]
            loss = F.cross_entropy(logits, target)
            terms.append(loss)
            per_lag[lag].append(loss.detach())
    total = torch.stack(terms).mean()
    diagnostics = {
        lag: float(torch.stack(values).mean().cpu()) if values else float("nan")
        for lag, values in per_lag.items()
    }
    return total, diagnostics


def train_contrastive_core(
    seed: int,
    train_sequences: torch.Tensor,
    train_event_ids: torch.Tensor,
    train_vocab_embeddings: torch.Tensor,
) -> tuple[LearnedLeakyRecurrentCore, list[dict[str, float]]]:
    # No task-state or emotion labels enter this function.
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    model = LearnedLeakyRecurrentCore(
        input_dim=train_sequences.shape[-1],
        config=LearnedCoreConfig(),
        seed=seed,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None

    for epoch in range(TRAIN_EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        loss, lag_losses = contrastive_delayed_loss(
            model,
            train_sequences,
            train_event_ids,
            train_vocab_embeddings,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        model.stabilize_recurrent(max_spectral_norm=0.98)

        value = float(loss.detach().cpu())
        history.append(
            {
                "epoch": float(epoch + 1),
                "loss": value,
                **{f"lag{lag}_loss": val for lag, val in lag_losses.items()},
            }
        )
        if value < best_loss:
            best_loss = value
            best_state = copy.deepcopy(model.state_dict())

    if best_state is None:
        raise RuntimeError("contrastive training produced no checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, history


@torch.no_grad()
def heldout_lag3_retrieval(
    model: LearnedLeakyRecurrentCore,
    test_sequences: torch.Tensor,
    test_event_ids: torch.Tensor,
    test_vocab_embeddings: torch.Tensor,
) -> float:
    states, _ = model.run_sequence(test_sequences, return_event_traces=False)
    prediction = F.normalize(model.memory_heads[2](states[-1]), dim=-1)
    logits = prediction @ test_vocab_embeddings.T
    predicted = logits.argmax(dim=-1)
    # final event index is 4; lag 3 target is event index 1 (semantic event)
    target = test_event_ids[:, test_sequences.shape[1] - 1 - 3]
    return float((predicted == target).float().mean().cpu())


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

    train_vocab, train_vocab_embeddings, train_event_ids = build_event_vocabulary(train_arms, encoder)
    test_vocab, test_vocab_embeddings, test_event_ids = build_event_vocabulary(test_arms, encoder)

    ema_train = ema_features(train_arms, encoder)
    ema_test = ema_features(test_arms, encoder)
    ema_macro, ema_domains = evaluate_map(ema_train, ema_test, train_pairs, test_pairs)

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
        real_macro, reset_macro, wrong_macro, learned_domains = evaluate_learned_controls(
            learned_train,
            learned_test,
            learned_reset,
            train_pairs,
            test_pairs,
        )

        random_train = random_features(seed, train_arms, encoder)
        random_test = random_features(seed, test_arms, encoder)
        random_macro, random_domains = evaluate_map(
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
                }
            )

    def mean(field: str) -> float:
        return float(np.mean([float(row[field]) for row in seed_rows]))

    real = mean("contrastive_real_macro")
    reset = mean("contrastive_reset_macro")
    wrong = mean("contrastive_wrong_macro")
    random = mean("v5_0_random_macro")
    retrieval = mean("heldout_lag3_retrieval_top1")

    semantic_gates = {
        "heldout_lag3_retrieval_top1_at_least_0_20": retrieval >= 0.20,
        "contrastive_semantic_macro_at_least_0_70": real >= 0.70,
        "contrastive_beats_random_by_0_10": real - random >= 0.10,
        "contrastive_beats_v5_2_cosine_by_0_10": real - V52_COSINE_MACRO >= 0.10,
        "contrastive_beats_reset_by_0_15": real - reset >= 0.15,
        "contrastive_beats_wrong_by_0_15": real - wrong >= 0.15,
    }
    semantic_gates["semantic_memory_gate"] = all(semantic_gates.values())

    summary = {
        "version": "v5.3",
        "purpose": "contrastive delayed-memory development benchmark; not confirmatory evidence",
        "architecture_changed_from_v5_2": False,
        "objective": "exact delayed event retrieval over unique train-event embedding vocabulary",
        "temperature": TEMPERATURE,
        "task_labels_used_by_core": False,
        "emotion_labels_used_by_core": False,
        "train_event_vocabulary_size": len(train_vocab),
        "heldout_event_vocabulary_size": len(test_vocab),
        "mean": {
            "heldout_lag3_retrieval_top1": retrieval,
            "heldout_lag3_cosine": mean("heldout_lag3_cosine"),
            "contrastive_real_macro": real,
            "contrastive_reset_macro": reset,
            "contrastive_wrong_macro": wrong,
            "v5_0_random_macro": random,
            "v5_2_cosine_historical_macro": V52_COSINE_MACRO,
            "ema_embedding_memory_macro": ema_macro,
        },
        "gaps": {
            "contrastive_minus_random": real - random,
            "contrastive_minus_v5_2_cosine": real - V52_COSINE_MACRO,
            "contrastive_minus_reset": real - reset,
            "contrastive_minus_wrong": real - wrong,
            "contrastive_minus_ema": real - ema_macro,
        },
        "acceptance": semantic_gates,
        "complexity_check": {
            "contrastive_reaches_or_beats_ema": real >= ema_macro,
            "ema_advantage_if_positive": ema_macro - real,
        },
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "per_seed_metrics.csv", seed_rows)
    write_csv(OUT_DIR / "per_domain_metrics.csv", domain_rows)
    write_csv(OUT_DIR / "training_curve.csv", training_rows)
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
