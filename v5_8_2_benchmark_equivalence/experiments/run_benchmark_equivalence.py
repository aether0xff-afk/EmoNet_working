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
    REPO_ROOT / "v5_2_learned_memory" / "experiments",
    REPO_ROOT / "v5_6_1_readout_temporal_diagnostic" / "experiments",
    REPO_ROOT / "v5_7_residual_fast_dynamics",
    REPO_ROOT / "v5_8_adaptive_fast_dynamics",
):
    sys.path.insert(0, str(path))

from adaptive_state import AdaptiveResidualState  # noqa: E402
from residual_state import ResidualDrivenState  # noqa: E402
from temporal_suite import build_case, relational_features  # noqa: E402
from run_learned_memory_benchmark import MODEL_NAME, RIDGE_ALPHA, SEEDS, CachedSentenceEncoder  # noqa: E402
from run_readout_temporal_diagnostic import (  # noqa: E402
    DiagnosticRidge,
    accuracy,
    relational_structure_features,
    structural_pair,
)

PAIR_COUNT = 120
TRAIN_PAIRS = 80
ADAPTATION_DECAY = 0.995
ADAPTATION_STRENGTH = 0.20
SLOW_DECAY = 0.80
OUT_DIR = VERSION_ROOT / "outputs" / "benchmark_equivalence"


def old_case(pair_id: int) -> tuple[tuple[str, ...], tuple[str, ...], str, str, str, str]:
    c0, c1, current, event_a, event_b = structural_pair(pair_id)
    return tuple(c0), tuple(c1), current, event_a, event_b, c0[0]


def new_case(pair_id: int) -> tuple[tuple[str, ...], tuple[str, ...], str, str, str, str]:
    case = build_case("alternation", pair_id)
    event_a = case.class0[1]
    event_b = case.class0[2]
    return case.class0, case.class1, case.current, event_a, event_b, case.class0[0]


def run_v57(model: ResidualDrivenState, history: tuple[str, ...], current: str, reset: bool = False):
    model.reset_all()
    model.consume_sequence(history)
    if reset:
        model.reset_fast()
    return model.consume_event(current)


def run_v58(model: AdaptiveResidualState, history: tuple[str, ...], current: str, reset: bool = False):
    model.reset_all()
    model.consume_sequence(history)
    if reset:
        model.reset_fast()
    return model.consume_event(current)


def raw(observation) -> np.ndarray:
    return observation.fast_trace.states.reshape(-1).astype(np.float32, copy=False)


def normalized_distance(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) + np.linalg.norm(b))
    if denominator <= 1e-12:
        return 0.0
    return float(np.linalg.norm(a - b) / denominator)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(a, b) / denominator)


def build_rows(seed: int, encoder: CachedSentenceEncoder, model_name: str) -> dict[str, list[dict[str, object]]]:
    if model_name == "v57":
        model = ResidualDrivenState(encoder, seed=seed, slow_decay=SLOW_DECAY)
        run = run_v57
    elif model_name == "v58":
        model = AdaptiveResidualState(
            encoder,
            seed=seed,
            adaptation_strength=ADAPTATION_STRENGTH,
            adaptation_decay=ADAPTATION_DECAY,
            slow_decay=SLOW_DECAY,
        )
        run = run_v58
    else:
        raise ValueError(model_name)

    result = {"old": [], "new": []}
    for renderer, factory in (("old", old_case), ("new", new_case)):
        for pair_id in range(PAIR_COUNT):
            c0, c1, current, _, _, _ = factory(pair_id)
            for label, sequence, opposite in ((0, c0, c1), (1, c1, c0)):
                real = run(model, sequence, current, False)
                reset = run(model, sequence, current, True)
                wrong = run(model, opposite, current, False)
                result[renderer].append(
                    {
                        "pair_id": pair_id,
                        "label": label,
                        "real": raw(real),
                        "reset": raw(reset),
                        "opposite": raw(wrong),
                    }
                )
    return result


def matrix(rows: list[dict[str, object]], key: str) -> np.ndarray:
    return np.stack([np.asarray(row[key], dtype=np.float32) for row in rows])


def labels(rows: list[dict[str, object]]) -> np.ndarray:
    return np.asarray([int(row["label"]) for row in rows], dtype=np.int64)


def split(rows: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    train = [row for row in rows if int(row["pair_id"]) < TRAIN_PAIRS]
    test = [row for row in rows if int(row["pair_id"]) >= TRAIN_PAIRS]
    return train, test


def evaluate_model(seed: int, encoder: CachedSentenceEncoder, model_name: str) -> dict[str, float]:
    rows = build_rows(seed, encoder, model_name)
    old_train, old_test = split(rows["old"])
    new_train, new_test = split(rows["new"])
    old_probe = DiagnosticRidge(RIDGE_ALPHA).fit(matrix(old_train, "real"), labels(old_train))
    new_probe = DiagnosticRidge(RIDGE_ALPHA).fit(matrix(new_train, "real"), labels(new_train))

    old_old_pred = old_probe.predict(matrix(old_test, "real"))
    old_new_pred = old_probe.predict(matrix(new_test, "real"))
    new_new_pred = new_probe.predict(matrix(new_test, "real"))
    new_old_pred = new_probe.predict(matrix(old_test, "real"))

    old_labels = labels(old_test)
    new_labels = labels(new_test)
    if not np.array_equal(old_labels, new_labels):
        raise AssertionError("paired OLD/NEW test labels differ")

    return {
        "old_to_old": accuracy(old_labels, old_old_pred),
        "new_to_new": accuracy(new_labels, new_new_pred),
        "old_to_new": accuracy(new_labels, old_new_pred),
        "new_to_old": accuracy(old_labels, new_old_pred),
        "old_probe_pair_prediction_agreement": float(np.mean(old_old_pred == old_new_pred)),
        "new_probe_pair_prediction_agreement": float(np.mean(new_new_pred == new_old_pred)),
        "old_reset": accuracy(old_labels, old_probe.predict(matrix(old_test, "reset"))),
        "new_reset": accuracy(new_labels, new_probe.predict(matrix(new_test, "reset"))),
        "old_opposite": accuracy(old_labels, old_probe.predict(matrix(old_test, "opposite"))),
        "new_opposite": accuracy(new_labels, new_probe.predict(matrix(new_test, "opposite"))),
    }


def renderer_diagnostics(encoder: CachedSentenceEncoder) -> dict[str, float]:
    a_cos: list[float] = []
    b_cos: list[float] = []
    prefix_cos: list[float] = []
    old_rel: list[float] = []
    new_rel: list[float] = []

    for pair_id in range(PAIR_COUNT):
        oc0, oc1, _, oa, ob, oprefix = old_case(pair_id)
        nc0, nc1, _, na, nb, nprefix = new_case(pair_id)
        a_cos.append(cosine(encoder.encode(oa), encoder.encode(na)))
        b_cos.append(cosine(encoder.encode(ob), encoder.encode(nb)))
        prefix_cos.append(cosine(encoder.encode(oprefix), encoder.encode(nprefix)))

        old_x = np.stack([relational_structure_features(list(seq), encoder) for seq in (oc0, oc1)])
        new_x = np.stack([relational_features(seq, encoder) for seq in (nc0, nc1)])
        # Label-free validity proxy: the two structural feature vectors should differ.
        old_rel.append(float(np.linalg.norm(old_x[0] - old_x[1])))
        new_rel.append(float(np.linalg.norm(new_x[0] - new_x[1])))

    return {
        "event_a_old_new_cosine": float(np.mean(a_cos)),
        "event_b_old_new_cosine": float(np.mean(b_cos)),
        "prefix_old_new_cosine": float(np.mean(prefix_cos)),
        "old_relational_class_distance": float(np.mean(old_rel)),
        "new_relational_class_distance": float(np.mean(new_rel)),
    }


def paired_trace_shift(seed: int, encoder: CachedSentenceEncoder, model_name: str) -> float:
    if model_name == "v57":
        model = ResidualDrivenState(encoder, seed=seed, slow_decay=SLOW_DECAY)
        run = run_v57
    else:
        model = AdaptiveResidualState(
            encoder,
            seed=seed,
            adaptation_strength=ADAPTATION_STRENGTH,
            adaptation_decay=ADAPTATION_DECAY,
            slow_decay=SLOW_DECAY,
        )
        run = run_v58
    distances: list[float] = []
    for pair_id in range(TRAIN_PAIRS, PAIR_COUNT):
        old0, old1, current, _, _, _ = old_case(pair_id)
        new0, new1, _, _, _, _ = new_case(pair_id)
        for old_seq, new_seq in ((old0, new0), (old1, new1)):
            old_obs = run(model, old_seq, current, False)
            new_obs = run(model, new_seq, current, False)
            distances.append(normalized_distance(raw(old_obs), raw(new_obs)))
    return float(np.mean(distances))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    encoder = CachedSentenceEncoder(MODEL_NAME)
    texts: set[str] = set()
    for pair_id in range(PAIR_COUNT):
        for factory in (old_case, new_case):
            c0, c1, current, event_a, event_b, prefix = factory(pair_id)
            texts.update(c0)
            texts.update(c1)
            texts.update((current, event_a, event_b, prefix))
    encoder.preload(texts)

    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        for model_name in ("v57", "v58"):
            metrics = evaluate_model(seed, encoder, model_name)
            metrics["paired_trace_normalized_distance"] = paired_trace_shift(
                seed, encoder, model_name
            )
            rows.append({"seed": seed, "model": model_name, **metrics})

    aggregate: dict[str, dict[str, float]] = {}
    for model_name in ("v57", "v58"):
        subset = [row for row in rows if row["model"] == model_name]
        names = [key for key in subset[0] if key not in {"seed", "model"}]
        aggregate[model_name] = {
            key: float(np.mean([float(row[key]) for row in subset])) for key in names
        }
        within = (aggregate[model_name]["old_to_old"] + aggregate[model_name]["new_to_new"]) / 2.0
        cross = (aggregate[model_name]["old_to_new"] + aggregate[model_name]["new_to_old"]) / 2.0
        aggregate[model_name]["within_mean"] = within
        aggregate[model_name]["cross_mean"] = cross
        aggregate[model_name]["cross_drop"] = within - cross

    render = renderer_diagnostics(encoder)
    diagnostic = {
        model: {
            "renderer_robust": values["cross_mean"] >= 0.70 and values["cross_drop"] <= 0.10,
            "renderer_sensitive": values["within_mean"] >= 0.65 and values["cross_drop"] > 0.15,
        }
        for model, values in aggregate.items()
    }

    summary = {
        "version": "v5.8.2",
        "purpose": "benchmark-equivalence and renderer-sensitivity audit",
        "state_generators_changed": False,
        "logic": "ABAB vs AABB for both OLD and NEW renderers",
        "train_pairs": TRAIN_PAIRS,
        "test_pairs": PAIR_COUNT - TRAIN_PAIRS,
        "mean_by_model": aggregate,
        "renderer_embedding_and_relational_diagnostics": render,
        "diagnosis": diagnostic,
        "claim_boundary": "diagnostic only; no retroactive version pass and no affect claim",
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(OUT_DIR / "per_seed_model_metrics.csv", rows)
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
