from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

HERE = Path(__file__).resolve()
VERSION_ROOT = HERE.parents[1]
REPO_ROOT = HERE.parents[2]
V52_ROOT = REPO_ROOT / "v5_2_learned_memory"
V52_EXPERIMENTS = V52_ROOT / "experiments"
sys.path.insert(0, str(VERSION_ROOT))
sys.path.insert(0, str(V52_ROOT))
sys.path.insert(0, str(V52_EXPERIMENTS))

from predictive_fixture import PredictiveArm, PredictivePair, build_predictive_pairs, flatten_pairs  # noqa: E402
from learned_core import LearnedCoreConfig, LearnedLeakyRecurrentCore  # noqa: E402
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


TEMPERATURE = 0.07
HISTORICAL_V54_CONTRASTIVE_MACRO = 0.630
OUT_DIR = VERSION_ROOT / "outputs" / "predictive_state_benchmark"


class FuturePredictionHead(nn.Module):
    def __init__(self, hidden_dim: int, output_dim: int, seed: int) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + 5000)
        self.linear = nn.Linear(hidden_dim, output_dim)
        with torch.no_grad():
            self.linear.weight.copy_(
                torch.randn(
                    self.linear.weight.shape,
                    generator=generator,
                    dtype=self.linear.weight.dtype,
                )
                * (0.02 / np.sqrt(max(1, hidden_dim)))
            )
            self.linear.bias.zero_()

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.linear(state), dim=-1)


def collect_texts(pairs: list[PredictivePair]) -> set[str]:
    texts: set[str] = set()
    for arm in flatten_pairs(pairs):
        texts.update(arm.history)
        texts.add(arm.current_text)
        texts.add(arm.future_text)
    return texts


def build_future_vocabulary(
    arms: list[PredictiveArm],
    encoder: CachedSentenceEncoder,
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    vocabulary = sorted({arm.future_text for arm in arms})
    index = {text: idx for idx, text in enumerate(vocabulary)}
    embeddings = torch.from_numpy(
        np.stack([encoder.encode(text) for text in vocabulary], axis=0)
    ).float()
    embeddings = F.normalize(embeddings, dim=-1)
    target_ids = torch.tensor([index[arm.future_text] for arm in arms], dtype=torch.long)
    return vocabulary, embeddings, target_ids


def train_predictive_core(
    seed: int,
    train_sequences: torch.Tensor,
    future_target_ids: torch.Tensor,
    future_vocab_embeddings: torch.Tensor,
) -> tuple[LearnedLeakyRecurrentCore, FuturePredictionHead, list[dict[str, float]]]:
    """Train only from observed event sequence and next-event identity.

    No semantic-state, polarity, usable/blocked, or emotion labels are accepted.
    """

    torch.manual_seed(seed)
    torch.set_num_threads(2)
    config = LearnedCoreConfig()
    core = LearnedLeakyRecurrentCore(
        input_dim=train_sequences.shape[-1],
        config=config,
        seed=seed,
    )
    head = FuturePredictionHead(config.hidden_dim, train_sequences.shape[-1], seed)
    parameters = list(core.parameters()) + list(head.parameters())
    optimizer = torch.optim.AdamW(
        parameters,
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_core: dict[str, torch.Tensor] | None = None
    best_head: dict[str, torch.Tensor] | None = None

    for epoch in range(TRAIN_EPOCHS):
        optimizer.zero_grad(set_to_none=True)
        states, _ = core.run_sequence(train_sequences, return_event_traces=False)
        prediction = head(states[-1])
        logits = prediction @ future_vocab_embeddings.T / TEMPERATURE
        loss = F.cross_entropy(logits, future_target_ids)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
        optimizer.step()
        core.stabilize_recurrent(max_spectral_norm=0.98)

        value = float(loss.detach().cpu())
        with torch.no_grad():
            train_top1 = float((logits.argmax(dim=-1) == future_target_ids).float().mean().cpu())
        history.append(
            {
                "epoch": float(epoch + 1),
                "loss": value,
                "train_future_top1": train_top1,
            }
        )
        if value < best_loss:
            best_loss = value
            best_core = copy.deepcopy(core.state_dict())
            best_head = copy.deepcopy(head.state_dict())

    if best_core is None or best_head is None:
        raise RuntimeError("predictive training produced no checkpoint")
    core.load_state_dict(best_core)
    head.load_state_dict(best_head)
    core.eval()
    head.eval()
    return core, head, history


@torch.no_grad()
def heldout_future_retrieval(
    core: LearnedLeakyRecurrentCore,
    head: FuturePredictionHead,
    test_sequences: torch.Tensor,
    future_target_ids: torch.Tensor,
    future_vocab_embeddings: torch.Tensor,
) -> float:
    states, _ = core.run_sequence(test_sequences, return_event_traces=False)
    prediction = head(states[-1])
    logits = prediction @ future_vocab_embeddings.T
    return float((logits.argmax(dim=-1) == future_target_ids).float().mean().cpu())


def pairs_for_domain(pairs: list[PredictivePair], domain: str) -> list[PredictivePair]:
    return [pair for pair in pairs if pair.domain == domain]


def evaluate_feature_map(
    train_map: dict[tuple[str, int], np.ndarray],
    test_map: dict[tuple[str, int], np.ndarray],
    train_pairs: list[PredictivePair],
    test_pairs: list[PredictivePair],
) -> tuple[float, dict[str, float]]:
    scores: dict[str, float] = {}
    for domain in sorted({pair.domain for pair in train_pairs}):
        train_arms = flatten_pairs(pairs_for_domain(train_pairs, domain))
        test_arms = flatten_pairs(pairs_for_domain(test_pairs, domain))
        y_train = np.asarray([arm.label for arm in train_arms], dtype=np.int64)
        y_test = np.asarray([arm.label for arm in test_arms], dtype=np.int64)
        x_train = np.stack([train_map[(arm.pair_id, arm.label)] for arm in train_arms])
        x_test = np.stack([test_map[(arm.pair_id, arm.label)] for arm in test_arms])
        probe = RidgeProbe(RIDGE_ALPHA).fit(x_train, y_train)
        scores[domain] = accuracy(y_test, probe.predict(x_test))
    return float(np.mean(list(scores.values()))), scores


def evaluate_controls(
    train_real: dict[tuple[str, int], np.ndarray],
    test_real: dict[tuple[str, int], np.ndarray],
    test_reset: dict[tuple[str, int], np.ndarray],
    train_pairs: list[PredictivePair],
    test_pairs: list[PredictivePair],
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
        real = accuracy(y_test, probe.predict(x_real))
        reset = accuracy(y_test, probe.predict(x_reset))
        wrong = accuracy(y_test, probe.predict(x_wrong))
        real_scores.append(real)
        reset_scores.append(reset)
        wrong_scores.append(wrong)
        per_domain[domain] = {"real": real, "reset": reset, "wrong": wrong}

    return (
        float(np.mean(real_scores)),
        float(np.mean(reset_scores)),
        float(np.mean(wrong_scores)),
        per_domain,
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    train_pairs, test_pairs = build_predictive_pairs()
    train_arms = flatten_pairs(train_pairs)
    test_arms = flatten_pairs(test_pairs)

    encoder = CachedSentenceEncoder(MODEL_NAME)
    encoder.preload(collect_texts(train_pairs + test_pairs))
    train_sequences = sequence_tensor(train_arms, encoder)
    test_sequences = sequence_tensor(test_arms, encoder)

    _, train_future_embeddings, train_future_ids = build_future_vocabulary(train_arms, encoder)
    test_future_vocab, test_future_embeddings, test_future_ids = build_future_vocabulary(test_arms, encoder)

    ema_train = ema_features(train_arms, encoder)
    ema_test = ema_features(test_arms, encoder)
    ema_macro, ema_domains = evaluate_feature_map(ema_train, ema_test, train_pairs, test_pairs)

    seed_rows: list[dict[str, object]] = []
    domain_rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []

    for seed in SEEDS:
        core, head, history = train_predictive_core(
            seed,
            train_sequences,
            train_future_ids,
            train_future_embeddings,
        )
        for row in history:
            training_rows.append({"seed": seed, **row})

        future_top1 = heldout_future_retrieval(
            core,
            head,
            test_sequences,
            test_future_ids,
            test_future_embeddings,
        )

        learned_train, _ = learned_features(core, train_arms, encoder)
        learned_test, learned_reset = learned_features(core, test_arms, encoder)
        real_macro, reset_macro, wrong_macro, learned_domains = evaluate_controls(
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
                "heldout_future_top1": future_top1,
                "predictive_real_macro": real_macro,
                "predictive_reset_macro": reset_macro,
                "predictive_wrong_macro": wrong_macro,
                "v5_0_random_macro": random_macro,
                "ema_macro": ema_macro,
            }
        )

        for domain in sorted(learned_domains):
            domain_rows.append(
                {
                    "seed": seed,
                    "domain": domain,
                    "predictive_real": learned_domains[domain]["real"],
                    "predictive_reset": learned_domains[domain]["reset"],
                    "predictive_wrong": learned_domains[domain]["wrong"],
                    "v5_0_random": random_domains[domain],
                    "ema": ema_domains[domain],
                }
            )

    def mean(field: str) -> float:
        return float(np.mean([float(row[field]) for row in seed_rows]))

    retrieval = mean("heldout_future_top1")
    real = mean("predictive_real_macro")
    reset = mean("predictive_reset_macro")
    wrong = mean("predictive_wrong_macro")
    random = mean("v5_0_random_macro")
    seed_pass_count = sum(float(row["predictive_real_macro"]) >= 0.68 for row in seed_rows)

    acceptance = {
        "heldout_future_retrieval_at_least_0_30": retrieval >= 0.30,
        "predictive_semantic_macro_at_least_0_72": real >= 0.72,
        "predictive_beats_random_by_0_10": real - random >= 0.10,
        "predictive_beats_historical_v5_4_by_0_08": real - HISTORICAL_V54_CONTRASTIVE_MACRO >= 0.08,
        "predictive_beats_reset_by_0_15": real - reset >= 0.15,
        "predictive_beats_wrong_by_0_15": real - wrong >= 0.15,
        "at_least_4_of_5_seeds_at_or_above_0_68": seed_pass_count >= 4,
    }
    acceptance["all_primary_gates"] = all(acceptance.values())

    summary = {
        "version": "v5.5",
        "purpose": "label-free future-prediction semantic-state development benchmark",
        "task_labels_used_by_core": False,
        "emotion_labels_used_by_core": False,
        "future_is_hidden_from_evaluated_trace": True,
        "protocol": {
            "encoder": MODEL_NAME,
            "epochs": TRAIN_EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "temperature": TEMPERATURE,
            "ridge_alpha": RIDGE_ALPHA,
            "ema_decay": EMA_DECAY,
            "seeds": SEEDS,
            "heldout_future_vocabulary_size": len(test_future_vocab),
        },
        "mean": {
            "heldout_future_top1": retrieval,
            "predictive_real_macro": real,
            "predictive_reset_macro": reset,
            "predictive_wrong_macro": wrong,
            "v5_0_random_macro": random,
            "historical_v5_4_contrastive_macro": HISTORICAL_V54_CONTRASTIVE_MACRO,
            "ema_embedding_memory_macro": ema_macro,
        },
        "gaps": {
            "predictive_minus_random": real - random,
            "predictive_minus_historical_v5_4": real - HISTORICAL_V54_CONTRASTIVE_MACRO,
            "predictive_minus_reset": real - reset,
            "predictive_minus_wrong": real - wrong,
            "predictive_minus_ema": real - ema_macro,
        },
        "seed_pass_count_at_or_above_0_68": seed_pass_count,
        "acceptance": acceptance,
        "complexity_check": {
            "predictive_reaches_or_beats_ema": real >= ema_macro,
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
