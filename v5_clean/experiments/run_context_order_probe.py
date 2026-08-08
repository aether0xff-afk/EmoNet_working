from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from emonet_v5 import DynamicsConfig, EmoNetV5Clean, HashingTextEncoder
from emonet_v5.probes import RidgeBinaryProbe
from emonet_v5.trace import NeuralTrace, temporal_shuffle


SEEDS = [7, 13, 21, 42, 100]
PAIR_COUNT = 80
TRAIN_PAIRS = 60
PROBE_ALPHA = 1.0
OUTPUT_DIR = Path("outputs/context_order_probe")

FINAL_TEXTS = [
    "지금 상태를 확인한다.",
    "다음 상황을 기다린다.",
    "현재 입력을 처리한다.",
    "이제 같은 신호를 받는다.",
    "마지막 사건이 도착했다.",
    "현재 사건을 관찰한다.",
    "같은 현재 조건에 도달했다.",
    "이 시점의 입력은 동일하다.",
]


def build_pair(pair_id: int) -> dict[str, object]:
    # The two arms contain exactly the same event multiset. Only ALPHA/BETA
    # order changes. Prefix/bridge/suffix and final text are identical within
    # each pair, so text-only, context-bag, and last-history-event baselines
    # cannot identify the label by construction.
    prefix = f"case_{pair_id} 공통 시작 사건"
    bridge = f"case_{pair_id} 공통 중간 사건"
    suffix = f"case_{pair_id} 공통 종료 직전 사건"
    alpha = "고정 표식 alpha 사건"
    beta = "고정 표식 beta 사건"
    context_0 = [prefix, alpha, bridge, beta, suffix]
    context_1 = [prefix, beta, bridge, alpha, suffix]
    return {
        "pair_id": pair_id,
        "final_text": FINAL_TEXTS[pair_id % len(FINAL_TEXTS)],
        "context_0": context_0,
        "context_1": context_1,
        "canonical_context": context_0,
    }


def trace_for(model: EmoNetV5Clean, context: list[str], final_text: str) -> NeuralTrace:
    model.reset_all()
    model.consume_sequence(context)
    return model.consume_event(final_text)


def reset_trace_for(model: EmoNetV5Clean, final_text: str) -> NeuralTrace:
    model.reset_all()
    model.reset_episode()
    return model.consume_event(final_text)


def flat_trace(trace: NeuralTrace) -> np.ndarray:
    return trace.states.astype(np.float32, copy=False).reshape(-1)


def context_bag(encoder: HashingTextEncoder, context: list[str]) -> np.ndarray:
    vectors = np.stack([encoder.encode(text) for text in context], axis=0)
    return vectors.mean(axis=0).astype(np.float32, copy=False)


def build_seed_dataset(seed: int) -> list[dict[str, object]]:
    encoder = HashingTextEncoder(dimension=96)
    model = EmoNetV5Clean(encoder, DynamicsConfig(seed=seed))
    reset_cache: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []

    for pair_id in range(PAIR_COUNT):
        pair = build_pair(pair_id)
        final_text = str(pair["final_text"])
        context_0 = list(pair["context_0"])
        context_1 = list(pair["context_1"])
        canonical = list(pair["canonical_context"])

        real_0 = trace_for(model, context_0, final_text)
        real_1 = trace_for(model, context_1, final_text)
        erased = trace_for(model, canonical, final_text)

        if final_text not in reset_cache:
            reset_cache[final_text] = flat_trace(reset_trace_for(model, final_text))
        reset_vec = reset_cache[final_text]

        common_text = encoder.encode(final_text)
        # Both order arms have exactly the same bag and same last event.
        bag_0 = context_bag(encoder, context_0)
        bag_1 = context_bag(encoder, context_1)
        last_0 = encoder.encode(context_0[-1])
        last_1 = encoder.encode(context_1[-1])

        real_vec_0 = flat_trace(real_0)
        real_vec_1 = flat_trace(real_1)
        shuffled_0 = flat_trace(temporal_shuffle(real_0, seed * 100_000 + pair_id * 2))
        shuffled_1 = flat_trace(temporal_shuffle(real_1, seed * 100_000 + pair_id * 2 + 1))
        erased_vec = flat_trace(erased)

        rows.append(
            {
                "pair_id": pair_id,
                "label": 0,
                "split": "train" if pair_id < TRAIN_PAIRS else "test",
                "text": common_text,
                "bag": bag_0,
                "last": last_0,
                "real": real_vec_0,
                "temporal_shuffle": shuffled_0,
                "wrong": real_vec_1.copy(),
                "reset": reset_vec.copy(),
                "order_erased": erased_vec.copy(),
            }
        )
        rows.append(
            {
                "pair_id": pair_id,
                "label": 1,
                "split": "train" if pair_id < TRAIN_PAIRS else "test",
                "text": common_text.copy(),
                "bag": bag_1,
                "last": last_1,
                "real": real_vec_1,
                "temporal_shuffle": shuffled_1,
                "wrong": real_vec_0.copy(),
                "reset": reset_vec.copy(),
                "order_erased": erased_vec.copy(),
            }
        )
    return rows


def stack(rows: list[dict[str, object]], key: str) -> np.ndarray:
    return np.stack([np.asarray(row[key], dtype=np.float32) for row in rows], axis=0)


def labels(rows: list[dict[str, object]]) -> np.ndarray:
    return np.asarray([int(row["label"]) for row in rows], dtype=np.int64)


def concat_text_trace(rows: list[dict[str, object]], trace_key: str) -> np.ndarray:
    return np.concatenate([stack(rows, "text"), stack(rows, trace_key)], axis=1)


def evaluate_seed(seed: int) -> tuple[dict[str, float], list[dict[str, object]]]:
    rows = build_seed_dataset(seed)
    train = [row for row in rows if row["split"] == "train"]
    test = [row for row in rows if row["split"] == "test"]
    y_train = labels(train)
    y_test = labels(test)

    main_probe = RidgeBinaryProbe(alpha=PROBE_ALPHA).fit(
        concat_text_trace(train, "real"), y_train
    )
    trace_probe = RidgeBinaryProbe(alpha=PROBE_ALPHA).fit(stack(train, "real"), y_train)
    text_probe = RidgeBinaryProbe(alpha=PROBE_ALPHA).fit(stack(train, "text"), y_train)
    bag_probe = RidgeBinaryProbe(alpha=PROBE_ALPHA).fit(stack(train, "bag"), y_train)
    last_probe = RidgeBinaryProbe(alpha=PROBE_ALPHA).fit(stack(train, "last"), y_train)

    metrics = {
        "seed": float(seed),
        "text_only": text_probe.accuracy(stack(test, "text"), y_test),
        "context_bag": bag_probe.accuracy(stack(test, "bag"), y_test),
        "last_history_event": last_probe.accuracy(stack(test, "last"), y_test),
        "trace_only_real": trace_probe.accuracy(stack(test, "real"), y_test),
        "text_plus_real": main_probe.accuracy(concat_text_trace(test, "real"), y_test),
        "text_plus_temporal_shuffle": main_probe.accuracy(
            concat_text_trace(test, "temporal_shuffle"), y_test
        ),
        "text_plus_wrong": main_probe.accuracy(concat_text_trace(test, "wrong"), y_test),
        "text_plus_reset": main_probe.accuracy(concat_text_trace(test, "reset"), y_test),
        "text_plus_order_erased": main_probe.accuracy(
            concat_text_trace(test, "order_erased"), y_test
        ),
    }

    prediction_rows: list[dict[str, object]] = []
    real_pred = main_probe.predict(concat_text_trace(test, "real"))
    wrong_pred = main_probe.predict(concat_text_trace(test, "wrong"))
    reset_pred = main_probe.predict(concat_text_trace(test, "reset"))
    erased_pred = main_probe.predict(concat_text_trace(test, "order_erased"))
    shuffled_pred = main_probe.predict(concat_text_trace(test, "temporal_shuffle"))
    for idx, row in enumerate(test):
        prediction_rows.append(
            {
                "seed": seed,
                "pair_id": int(row["pair_id"]),
                "label": int(row["label"]),
                "real_pred": int(real_pred[idx]),
                "temporal_shuffle_pred": int(shuffled_pred[idx]),
                "wrong_pred": int(wrong_pred[idx]),
                "reset_pred": int(reset_pred[idx]),
                "order_erased_pred": int(erased_pred[idx]),
            }
        )
    return metrics, prediction_rows


def aggregate(per_seed: list[dict[str, float]]) -> dict[str, object]:
    metric_names = [key for key in per_seed[0] if key != "seed"]
    means = {
        name: float(np.mean([row[name] for row in per_seed]))
        for name in metric_names
    }
    stds = {
        name: float(np.std([row[name] for row in per_seed]))
        for name in metric_names
    }
    real = means["text_plus_real"]
    acceptance = {
        "real_above_0_80_mean": real >= 0.80,
        "every_seed_real_above_0_65": all(row["text_plus_real"] >= 0.65 for row in per_seed),
        "text_only_near_chance": means["text_only"] <= 0.60,
        "context_bag_near_chance": means["context_bag"] <= 0.60,
        "last_event_near_chance": means["last_history_event"] <= 0.60,
        "real_beats_wrong_by_0_20": real - means["text_plus_wrong"] >= 0.20,
        "real_beats_reset_by_0_20": real - means["text_plus_reset"] >= 0.20,
        "real_beats_order_erased_by_0_20": real - means["text_plus_order_erased"] >= 0.20,
    }
    acceptance["all_primary_gates"] = all(acceptance.values())
    return {
        "purpose": "controlled temporal-context decodability test; not an affect/semantic claim",
        "task": "classify ALPHA-before-BETA vs BETA-before-ALPHA from the same event multiset and same current text",
        "encoder": "HashingTextEncoder",
        "seeds": SEEDS,
        "pair_count": PAIR_COUNT,
        "train_pairs": TRAIN_PAIRS,
        "test_pairs": PAIR_COUNT - TRAIN_PAIRS,
        "samples_per_seed": PAIR_COUNT * 2,
        "probe": {"type": "ridge_binary", "alpha": PROBE_ALPHA},
        "mean_accuracy": means,
        "std_accuracy": stds,
        "gaps": {
            "real_minus_text_only": real - means["text_only"],
            "real_minus_wrong": real - means["text_plus_wrong"],
            "real_minus_reset": real - means["text_plus_reset"],
            "real_minus_order_erased": real - means["text_plus_order_erased"],
            "real_minus_temporal_shuffle": real - means["text_plus_temporal_shuffle"],
        },
        "acceptance": acceptance,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    per_seed: list[dict[str, float]] = []
    predictions: list[dict[str, object]] = []
    for seed in SEEDS:
        metrics, pred_rows = evaluate_seed(seed)
        per_seed.append(metrics)
        predictions.extend(pred_rows)

    summary = aggregate(per_seed)
    write_csv(OUTPUT_DIR / "per_seed_metrics.csv", per_seed)
    write_csv(OUTPUT_DIR / "test_predictions.csv", predictions)
    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
