from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve()
VERSION_ROOT = HERE.parents[1]
REPO_ROOT = HERE.parents[2]
V51_ROOT = REPO_ROOT / "v5_1_semantic_context"
V52_EXPERIMENTS = REPO_ROOT / "v5_2_learned_memory" / "experiments"
V54_ROOT = REPO_ROOT / "v5_4_fresh_confirmatory"
sys.path.insert(0, str(VERSION_ROOT))
sys.path.insert(0, str(V51_ROOT))
sys.path.insert(0, str(V52_EXPERIMENTS))
sys.path.insert(0, str(V54_ROOT))

from dual_state import DualTimescaleState, dual_features, fast_features, slow_features  # noqa: E402
from fresh_fixture import FreshArm, FreshPair, build_fresh_pairs, flatten_pairs as flatten_fresh  # noqa: E402
from run_learned_memory_benchmark import (  # noqa: E402
    MODEL_NAME,
    RIDGE_ALPHA,
    SEEDS,
    CachedSentenceEncoder,
    RidgeProbe,
    accuracy,
)


SLOW_DECAY = 0.80
ORDER_PAIR_COUNT = 80
ORDER_TRAIN_PAIRS = 60
OUT_DIR = VERSION_ROOT / "outputs" / "dual_timescale_benchmark"

ORDER_CURRENT = [
    "The same current signal is now observed.",
    "I now inspect the identical current condition.",
    "The same present event has arrived.",
    "I have reached the identical current situation.",
]


def key(arm) -> tuple[str, int]:
    return str(arm.pair_id), int(arm.label)


def run_condition(model: DualTimescaleState, history: tuple[str, ...] | list[str], current: str, mode: str):
    model.reset_all()
    model.consume_sequence(list(history))
    if mode == "fast_reset":
        model.reset_fast()
    elif mode == "slow_reset":
        model.reset_slow()
    elif mode == "both_reset":
        model.reset_both()
    elif mode != "real":
        raise ValueError(f"unknown mode: {mode}")
    return model.consume_event(current)


def pairs_for_domain(pairs: list[FreshPair], domain: str) -> list[FreshPair]:
    return [pair for pair in pairs if pair.domain == domain]


def collect_semantic_seed(
    seed: int,
    encoder: CachedSentenceEncoder,
    train_pairs: list[FreshPair],
    test_pairs: list[FreshPair],
) -> tuple[dict[str, float], list[dict[str, object]]]:
    model = DualTimescaleState(encoder, seed=seed, slow_decay=SLOW_DECAY)
    train_arms = flatten_fresh(train_pairs)
    test_arms = flatten_fresh(test_pairs)

    train_real = {
        key(arm): run_condition(model, arm.history, arm.current_text, "real")
        for arm in train_arms
    }
    test_conditions = {
        mode: {
            key(arm): run_condition(model, arm.history, arm.current_text, mode)
            for arm in test_arms
        }
        for mode in ("real", "fast_reset", "slow_reset", "both_reset")
    }

    domain_rows: list[dict[str, object]] = []
    fast_scores: list[float] = []
    slow_scores: list[float] = []
    dual_scores: list[float] = []
    fast_reset_scores: list[float] = []
    slow_reset_scores: list[float] = []
    both_reset_scores: list[float] = []

    for domain in sorted({pair.domain for pair in train_pairs}):
        d_train = flatten_fresh(pairs_for_domain(train_pairs, domain))
        d_test = flatten_fresh(pairs_for_domain(test_pairs, domain))
        y_train = np.asarray([arm.label for arm in d_train], dtype=np.int64)
        y_test = np.asarray([arm.label for arm in d_test], dtype=np.int64)

        def stack(mapping, arms, feature_fn):
            return np.stack([feature_fn(mapping[key(arm)]) for arm in arms])

        fast_probe = RidgeProbe(RIDGE_ALPHA).fit(stack(train_real, d_train, fast_features), y_train)
        slow_probe = RidgeProbe(RIDGE_ALPHA).fit(stack(train_real, d_train, slow_features), y_train)
        dual_probe = RidgeProbe(RIDGE_ALPHA).fit(stack(train_real, d_train, dual_features), y_train)

        fast_acc = accuracy(y_test, fast_probe.predict(stack(test_conditions["real"], d_test, fast_features)))
        slow_acc = accuracy(y_test, slow_probe.predict(stack(test_conditions["real"], d_test, slow_features)))
        dual_acc = accuracy(y_test, dual_probe.predict(stack(test_conditions["real"], d_test, dual_features)))
        fast_reset_acc = accuracy(
            y_test,
            dual_probe.predict(stack(test_conditions["fast_reset"], d_test, dual_features)),
        )
        slow_reset_acc = accuracy(
            y_test,
            dual_probe.predict(stack(test_conditions["slow_reset"], d_test, dual_features)),
        )
        both_reset_acc = accuracy(
            y_test,
            dual_probe.predict(stack(test_conditions["both_reset"], d_test, dual_features)),
        )

        fast_scores.append(fast_acc)
        slow_scores.append(slow_acc)
        dual_scores.append(dual_acc)
        fast_reset_scores.append(fast_reset_acc)
        slow_reset_scores.append(slow_reset_acc)
        both_reset_scores.append(both_reset_acc)
        domain_rows.append(
            {
                "seed": seed,
                "domain": domain,
                "fast_only": fast_acc,
                "slow_only": slow_acc,
                "dual_real": dual_acc,
                "dual_fast_reset": fast_reset_acc,
                "dual_slow_reset": slow_reset_acc,
                "dual_both_reset": both_reset_acc,
            }
        )

    metrics = {
        "semantic_fast_only": float(np.mean(fast_scores)),
        "semantic_slow_only": float(np.mean(slow_scores)),
        "semantic_dual": float(np.mean(dual_scores)),
        "semantic_dual_fast_reset": float(np.mean(fast_reset_scores)),
        "semantic_dual_slow_reset": float(np.mean(slow_reset_scores)),
        "semantic_dual_both_reset": float(np.mean(both_reset_scores)),
    }
    return metrics, domain_rows


def build_order_pair(pair_id: int) -> tuple[list[str], list[str], str]:
    # Same event multiset; only the order of ALPHA/BETA differs.
    prefix = f"Order case {pair_id} begins with the same neutral observation."
    bridge = f"Order case {pair_id} contains the same neutral middle observation."
    suffix = f"Order case {pair_id} ends its history with the same neutral observation."
    alpha = "A distinctive ALPHA event occurs."
    beta = "A distinctive BETA event occurs."
    context_0 = [prefix, alpha, bridge, beta, suffix]
    context_1 = [prefix, beta, bridge, alpha, suffix]
    current = ORDER_CURRENT[pair_id % len(ORDER_CURRENT)]
    return context_0, context_1, current


def collect_order_seed(seed: int, encoder: CachedSentenceEncoder) -> dict[str, float]:
    model = DualTimescaleState(encoder, seed=seed, slow_decay=SLOW_DECAY)
    rows: list[dict[str, object]] = []

    for pair_id in range(ORDER_PAIR_COUNT):
        context_0, context_1, current = build_order_pair(pair_id)
        split = "train" if pair_id < ORDER_TRAIN_PAIRS else "test"
        for label, context in ((0, context_0), (1, context_1)):
            observations = {
                mode: run_condition(model, context, current, mode)
                for mode in ("real", "fast_reset", "slow_reset", "both_reset")
            }
            rows.append(
                {
                    "pair_id": pair_id,
                    "label": label,
                    "split": split,
                    "real_fast": fast_features(observations["real"]),
                    "real_slow": slow_features(observations["real"]),
                    "real_dual": dual_features(observations["real"]),
                    "fast_reset_dual": dual_features(observations["fast_reset"]),
                    "slow_reset_dual": dual_features(observations["slow_reset"]),
                    "both_reset_dual": dual_features(observations["both_reset"]),
                }
            )

    train = [row for row in rows if row["split"] == "train"]
    test = [row for row in rows if row["split"] == "test"]
    y_train = np.asarray([row["label"] for row in train], dtype=np.int64)
    y_test = np.asarray([row["label"] for row in test], dtype=np.int64)

    def stack(which_rows, field):
        return np.stack([np.asarray(row[field], dtype=np.float32) for row in which_rows])

    fast_probe = RidgeProbe(RIDGE_ALPHA).fit(stack(train, "real_fast"), y_train)
    slow_probe = RidgeProbe(RIDGE_ALPHA).fit(stack(train, "real_slow"), y_train)
    dual_probe = RidgeProbe(RIDGE_ALPHA).fit(stack(train, "real_dual"), y_train)

    return {
        "order_fast_only": accuracy(y_test, fast_probe.predict(stack(test, "real_fast"))),
        "order_slow_only": accuracy(y_test, slow_probe.predict(stack(test, "real_slow"))),
        "order_dual": accuracy(y_test, dual_probe.predict(stack(test, "real_dual"))),
        "order_dual_fast_reset": accuracy(
            y_test, dual_probe.predict(stack(test, "fast_reset_dual"))
        ),
        "order_dual_slow_reset": accuracy(
            y_test, dual_probe.predict(stack(test, "slow_reset_dual"))
        ),
        "order_dual_both_reset": accuracy(
            y_test, dual_probe.predict(stack(test, "both_reset_dual"))
        ),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    train_pairs, test_pairs = build_fresh_pairs()
    encoder = CachedSentenceEncoder(MODEL_NAME)
    all_texts: set[str] = set()
    for arm in flatten_fresh(train_pairs + test_pairs):
        all_texts.update(arm.history)
        all_texts.add(arm.current_text)
    for pair_id in range(ORDER_PAIR_COUNT):
        c0, c1, current = build_order_pair(pair_id)
        all_texts.update(c0)
        all_texts.update(c1)
        all_texts.add(current)
    encoder.preload(all_texts)

    seed_rows: list[dict[str, object]] = []
    domain_rows: list[dict[str, object]] = []
    for seed in SEEDS:
        semantic_metrics, semantic_domains = collect_semantic_seed(
            seed, encoder, train_pairs, test_pairs
        )
        order_metrics = collect_order_seed(seed, encoder)
        seed_rows.append({"seed": seed, **semantic_metrics, **order_metrics})
        domain_rows.extend(semantic_domains)

    def mean(field: str) -> float:
        return float(np.mean([float(row[field]) for row in seed_rows]))

    semantic_dual = mean("semantic_dual")
    semantic_fast = mean("semantic_fast_only")
    semantic_slow = mean("semantic_slow_only")
    semantic_slow_reset = mean("semantic_dual_slow_reset")
    order_dual = mean("order_dual")
    order_slow = mean("order_slow_only")
    order_fast_reset = mean("order_dual_fast_reset")

    acceptance = {
        "dual_semantic_macro_at_least_0_78": semantic_dual >= 0.78,
        "dual_semantic_beats_fast_by_0_10": semantic_dual - semantic_fast >= 0.10,
        "slow_reset_reduces_semantic_by_0_10": semantic_dual - semantic_slow_reset >= 0.10,
        "dual_order_at_least_0_80": order_dual >= 0.80,
        "dual_order_beats_slow_by_0_05": order_dual - order_slow >= 0.05,
        "fast_reset_reduces_order_by_0_10": order_dual - order_fast_reset >= 0.10,
        "dual_adds_value_without_semantic_regression": (
            max(semantic_dual - semantic_slow, order_dual - order_slow) >= 0.05
            and semantic_dual >= semantic_slow - 0.02
        ),
    }
    acceptance["all_primary_gates"] = all(acceptance.values())

    summary = {
        "version": "v5.6",
        "purpose": "dual-timescale architecture development benchmark",
        "encoder": MODEL_NAME,
        "fast_state": "frozen v5.0 fixed recurrent trace",
        "slow_state": {"type": "EMA embedding memory", "decay": SLOW_DECAY},
        "mean": {
            field: mean(field)
            for field in seed_rows[0]
            if field != "seed"
        },
        "gaps": {
            "semantic_dual_minus_fast": semantic_dual - semantic_fast,
            "semantic_dual_minus_slow": semantic_dual - semantic_slow,
            "semantic_dual_minus_slow_reset": semantic_dual - semantic_slow_reset,
            "order_dual_minus_fast": order_dual - mean("order_fast_only"),
            "order_dual_minus_slow": order_dual - order_slow,
            "order_dual_minus_fast_reset": order_dual - order_fast_reset,
        },
        "acceptance": acceptance,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "per_seed_metrics.csv", seed_rows)
    write_csv(OUT_DIR / "semantic_per_domain.csv", domain_rows)
    (OUT_DIR / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
