#!/usr/bin/env python3
"""Mechanistic ablation analysis for v3.1 TRACE formation.

The experiment reuses the *stored* 4D stimulus vectors from the frozen full80
TRACE run, rebuilds the same seed-42 neural graph fresh for every sample, and
asks which internal dynamics are necessary for emotion-related trajectory
geometry.

This is intentionally a mechanism diagnostic, not a new performance-tuned
model. No label is passed into the neural model during a run; labels are used
only after traces are generated to compute group geometry.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import export_neural_activation_traces as exporter  # noqa: E402


LABEL_AXES = [
    "valence",
    "arousal",
    "target",
    "control_state",
    "social_orientation",
    "action_tendency_class",
    "episode_family",
    "appraisal_family",
]
STIM_COLUMNS = ["dopamine", "serotonin", "norepinephrine", "melatonin"]
TEMPORAL_BINS = 4
EPS = 1e-8

DEFAULT_DYNAMICS: dict[str, Any] = {
    "k_threshold_base": 0.64,
    "k_remem_base": 0.82,
    "k_decay": 0.99,
    "refractory_ticks": 1,
    "input_topk": 2,
    "input_signal_clip": 1.50,
    "intrinsic_alignment_gain": 0.24,
    "fatigue_gain": 0.18,
    "fatigue_threshold_gain": 0.10,
    "fatigue_k_leak": 0.05,
    "inhibitory_suppression_gain": 0.18,
    "density_control_start_tick": 0,
    "density_target_high": 1.0,
    "density_soft_k_leak_gain": 0.0,
    "density_hard_cap": 1.0,
    "density_pruned_fatigue_gain": 0.0,
    "ne_thresh_reduce_gain": 0.25,
    "ne_remem_reduce_gain": 0.25,
    "activity_churn_eps": 0.02,
}


@dataclass(frozen=True)
class Condition:
    name: str
    config_overrides: dict[str, float]
    clear_edges: bool = False
    max_ticks: int | None = None


CONDITIONS = [
    Condition("baseline", {}),
    Condition("single_tick", {}, max_ticks=1),
    Condition("no_recurrence", {"dopa_rewire_gain": 0.0, "sero_prune_gain": 0.0}, clear_edges=True),
    Condition("no_alignment", {"intrinsic_alignment_gain": 0.0}),
    Condition(
        "no_memory",
        {"memory_sim_gain": 0.0, "memory_stim_mix": 0.0, "memory_k_mix": 0.0},
    ),
    Condition(
        "no_hysteresis",
        {
            "hysteresis_threshold_gain": 0.0,
            "hysteresis_remem_gain": 0.0,
            "hysteresis_k_bonus": 0.0,
        },
    ),
    Condition("no_inhibition", {"inhibitory_suppression_gain": 0.0}),
    Condition(
        "no_fatigue",
        {"fatigue_gain": 0.0, "fatigue_threshold_gain": 0.0, "fatigue_k_leak": 0.0},
    ),
    Condition(
        "no_modulation",
        {"mela_dropout_gain": 0.0, "ne_thresh_reduce_gain": 0.0, "ne_remem_reduce_gain": 0.0},
    ),
    Condition("no_rewiring", {"dopa_rewire_gain": 0.0, "sero_prune_gain": 0.0}),
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_dynamics(path: Path) -> dict[str, Any]:
    params = dict(DEFAULT_DYNAMICS)
    payload = json.loads(path.read_text(encoding="utf-8"))
    params.update(payload.get("dynamics", {}))
    return params


def make_model_args(seed: int, max_ticks: int, dynamics: dict[str, Any]) -> SimpleNamespace:
    values: dict[str, Any] = {
        "n_neurons": 256,
        "seed": seed,
        "z_encoder_mode": "stat",
        "stim_source": "proxy",
        "max_ticks": max_ticks,
        "min_ticks_before_converged": 6,
        "convergence_patience": 4,
        "progress_every": 0,
        **DEFAULT_DYNAMICS,
    }
    values.update(dynamics)
    return SimpleNamespace(**values)


def stimulus(row: dict[str, str]) -> np.ndarray:
    return np.asarray([float(row[name]) for name in STIM_COLUMNS], dtype=np.float32)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right) + EPS)
    if denom <= EPS:
        return 0.0
    return float(np.dot(left, right) / denom)


def clear_graph(model: Any) -> None:
    for neuron in model.state.neurons:
        neuron.out_neighbors.clear()
        neuron.in_neighbors.clear()


def apply_condition(model: Any, condition: Condition) -> None:
    for key, value in condition.config_overrides.items():
        if not hasattr(model.config, key):
            raise AttributeError(f"EmoNetConfig has no field {key!r}")
        setattr(model.config, key, value)
    if condition.clear_edges:
        clear_graph(model)


def activation_matrix(model: Any) -> np.ndarray:
    log = list(model.state.branch_log)
    matrix = np.zeros((max(1, len(log)), model.config.n_neurons), dtype=np.float32)
    for t, record in enumerate(log):
        for node_id, state in (getattr(record, "node_states", {}) or {}).items():
            matrix[t, int(node_id)] = float(state.K)
    return matrix


def temporal_feature(matrix: np.ndarray) -> np.ndarray:
    # log1p prevents a few very large K values from numerically dominating all
    # other neurons while preserving monotonic activation differences.
    arr = np.log1p(np.maximum(matrix.astype(np.float32, copy=False), 0.0))
    indices = np.array_split(np.arange(arr.shape[0]), TEMPORAL_BINS)
    chunks: list[np.ndarray] = []
    for idx in indices:
        if idx.size == 0:
            chunks.extend([np.zeros(arr.shape[1], dtype=np.float32)] * 2)
            continue
        block = arr[idx]
        chunks.append(block.mean(axis=0))
        chunks.append(block.max(axis=0))
    return np.concatenate(chunks, axis=0).astype(np.float32)


def standardized_distance_matrix(features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    z = (x - mean) / std
    diff = z[:, None, :] - z[None, :, :]
    return np.sqrt(np.mean(diff * diff, axis=2)).astype(np.float32)


def axis_metric(rows: list[dict[str, str]], distance: np.ndarray, axis: str) -> dict[str, Any]:
    labels = [str(row.get(axis, "")).strip().lower() for row in rows]
    intra: list[float] = []
    inter: list[float] = []
    for i in range(len(labels)):
        if not labels[i]:
            continue
        for j in range(i + 1, len(labels)):
            if not labels[j]:
                continue
            if labels[i] == labels[j]:
                intra.append(float(distance[i, j]))
            else:
                inter.append(float(distance[i, j]))
    mean_intra = float(np.mean(intra)) if intra else math.nan
    mean_inter = float(np.mean(inter)) if inter else math.nan
    separation = mean_inter - mean_intra if intra and inter else math.nan
    relative = separation / (0.5 * (mean_inter + mean_intra) + EPS) if intra and inter else math.nan
    return {
        "axis": axis,
        "mean_intra": mean_intra,
        "mean_inter": mean_inter,
        "separation": separation,
        "relative_separation": relative,
        "intra_pairs": len(intra),
        "inter_pairs": len(inter),
    }


def run_sample(
    row: dict[str, str],
    condition: Condition,
    base_dynamics: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, dict[str, Any], Any]:
    max_ticks = condition.max_ticks or 64
    args = make_model_args(seed, max_ticks, base_dynamics)
    model = exporter.build_model(args)
    apply_condition(model, condition)
    stim = stimulus(row)
    outputs = model.forward(stim)
    matrix = activation_matrix(model)
    feature = temporal_feature(matrix)
    raw_log = list(model.state.branch_log)
    dominant = []
    for record in raw_log:
        states = getattr(record, "node_states", {}) or {}
        if states:
            node_id, state = max(states.items(), key=lambda item: float(item[1].K))
            dominant.append((int(node_id), float(state.K)))
        else:
            dominant.append((-1, 0.0))
    summary = {
        "record_id": row.get("record_id", ""),
        "ticks": len(raw_log),
        "termination": str(outputs.get("termination_reason", "")),
        "mean_density": float(np.mean([len(getattr(r, "active_nodes", []) or []) / 256.0 for r in raw_log])) if raw_log else 0.0,
        "dominant_route": [node for node, _ in dominant],
        "dominant_k": [value for _, value in dominant],
    }
    return feature, summary, model


def choose_matched_pair(rows: list[dict[str, str]]) -> tuple[int, int]:
    preferred = [("negative", "positive"), ("negative", "mixed"), ("mixed", "positive")]
    secondary = ["arousal", "target", "control_state", "social_orientation", "action_tendency_class", "appraisal_family"]
    best: tuple[float, int, int] | None = None
    for first_label, second_label in preferred:
        for i, left in enumerate(rows):
            if str(left.get("valence", "")).strip().lower() != first_label:
                continue
            for j, right in enumerate(rows):
                if str(right.get("valence", "")).strip().lower() != second_label:
                    continue
                matches = sum(
                    str(left.get(axis, "")).strip().lower() == str(right.get(axis, "")).strip().lower()
                    and bool(str(left.get(axis, "")).strip())
                    for axis in secondary
                )
                stim_distance = float(np.linalg.norm(stimulus(left) - stimulus(right)))
                score = 10.0 * matches - stim_distance
                if best is None or score > best[0]:
                    best = (score, i, j)
        if best is not None:
            return best[1], best[2]
    raise RuntimeError("could not find rows with different valence labels")


def pair_mechanism_detail(
    rows: list[dict[str, str]],
    base_dynamics: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    left_idx, right_idx = choose_matched_pair(rows)
    left = rows[left_idx]
    right = rows[right_idx]
    baseline = CONDITIONS[0]
    _, left_summary, left_model = run_sample(left, baseline, base_dynamics, seed)
    _, right_summary, right_model = run_sample(right, baseline, base_dynamics, seed)
    left_matrix = activation_matrix(left_model)
    right_matrix = activation_matrix(right_model)
    common_ticks = min(left_matrix.shape[0], right_matrix.shape[0])
    distances: list[dict[str, Any]] = []
    for t in range(common_ticks):
        a = np.log1p(np.maximum(left_matrix[t], 0.0))
        b = np.log1p(np.maximum(right_matrix[t], 0.0))
        cos_dist = 1.0 - cosine(a, b) if np.linalg.norm(a) > EPS and np.linalg.norm(b) > EPS else 0.0
        rms = float(np.sqrt(np.mean((a - b) ** 2)))
        distances.append({"tick": t, "cosine_distance": cos_dist, "rms_logk_difference": rms})

    if distances:
        peak = max(distances, key=lambda item: item["rms_logk_difference"])
        peak_tick = int(peak["tick"])
    else:
        peak_tick = 0

    left_vec = np.log1p(np.maximum(left_matrix[min(peak_tick, left_matrix.shape[0] - 1)], 0.0))
    right_vec = np.log1p(np.maximum(right_matrix[min(peak_tick, right_matrix.shape[0] - 1)], 0.0))
    top_ids = np.argsort(np.abs(left_vec - right_vec))[::-1][:12]
    stim_left = stimulus(left)
    stim_right = stimulus(right)
    neurons: list[dict[str, Any]] = []
    for node_id in top_ids:
        neuron = left_model.state.neurons[int(node_id)]
        parents_left: list[int] = []
        parents_right: list[int] = []
        if peak_tick > 0 and peak_tick - 1 < len(left_model.state.branch_log):
            parents_left = [int(src) for src, dst in left_model.state.branch_log[peak_tick - 1].edges_fired if int(dst) == int(node_id)]
        if peak_tick > 0 and peak_tick - 1 < len(right_model.state.branch_log):
            parents_right = [int(src) for src, dst in right_model.state.branch_log[peak_tick - 1].edges_fired if int(dst) == int(node_id)]
        neurons.append(
            {
                "neuron_id": int(node_id),
                "type": str(neuron.neuron_type),
                "intrinsic_bias": [round(float(x), 5) for x in neuron.intrinsic_bias],
                "alignment_left": cosine(stim_left, neuron.intrinsic_bias),
                "alignment_right": cosine(stim_right, neuron.intrinsic_bias),
                "logK_left": float(left_vec[int(node_id)]),
                "logK_right": float(right_vec[int(node_id)]),
                "abs_difference": float(abs(left_vec[int(node_id)] - right_vec[int(node_id)])),
                "parents_left_previous_tick": parents_left[:24],
                "parents_right_previous_tick": parents_right[:24],
            }
        )

    return {
        "left": {
            "record_id": left.get("record_id", ""),
            "text": left.get("text", ""),
            "labels": {axis: left.get(axis, "") for axis in LABEL_AXES},
            "stimulus": [float(x) for x in stim_left],
            "run": left_summary,
        },
        "right": {
            "record_id": right.get("record_id", ""),
            "text": right.get("text", ""),
            "labels": {axis: right.get(axis, "") for axis in LABEL_AXES},
            "stimulus": [float(x) for x in stim_right],
            "run": right_summary,
        },
        "tick_distances": distances,
        "peak_divergence_tick": peak_tick,
        "top_divergent_neurons": neurons,
    }


def markdown_report(report: dict[str, Any]) -> str:
    valence = report["axis_summary"]["valence"]
    baseline = valence["baseline"]
    input_metric = valence["input"]
    lines = [
        "# TRACE Mechanism Causal-Ablation Report",
        "",
        "## Question",
        "",
        "Which v3.1 neural mechanisms transform the stored emotional stimulus vectors into emotion-related TRACE trajectories?",
        "",
        "## Protocol",
        "",
        f"- samples: `{report['n_samples']}` frozen full80 records",
        f"- seed: `{report['seed']}`",
        "- every sample starts from a freshly rebuilt identical seed-42 graph",
        "- labels are used only after trace generation",
        "- trajectory feature: 4 temporal bins × mean/max of log(1+K) over 256 neurons",
        "- primary statistic: relative separation = (inter - intra) / mean(inter, intra)",
        "",
        "## Valence mechanism ablation",
        "",
        "| condition | relative separation | change vs baseline |",
        "|---|---:|---:|",
        f"| raw 4D stimulus | {input_metric['relative_separation']:+.4f} | n/a |",
    ]
    for name, metric in valence.items():
        if name == "input":
            continue
        delta = metric["relative_separation"] - baseline["relative_separation"]
        lines.append(f"| {name} | {metric['relative_separation']:+.4f} | {delta:+.4f} |")

    lines.extend(["", "## All-axis baseline", "", "| axis | relative separation |", "|---|---:|"])
    for axis in LABEL_AXES:
        lines.append(f"| {axis} | {report['axis_summary'][axis]['baseline']['relative_separation']:+.4f} |")

    pair = report["matched_pair"]
    lines.extend(
        [
            "",
            "## Matched-pair trajectory example",
            "",
            f"- left: `{pair['left']['record_id']}` — {pair['left']['labels']['valence']} / {pair['left']['labels']['arousal']}",
            f"- right: `{pair['right']['record_id']}` — {pair['right']['labels']['valence']} / {pair['right']['labels']['arousal']}",
            f"- peak trajectory divergence tick: `{pair['peak_divergence_tick']}`",
            "",
            "Top neurons at peak divergence:",
            "",
            "| neuron | type | align left | align right | logK left | logK right | |Δ| |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in pair["top_divergent_neurons"][:8]:
        lines.append(
            f"| {item['neuron_id']} | {item['type']} | {item['alignment_left']:.4f} | {item['alignment_right']:.4f} | "
            f"{item['logK_left']:.4f} | {item['logK_right']:.4f} | {item['abs_difference']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "This analysis identifies which neural-dynamics terms are necessary for preserving or amplifying the geometry already present in the stored 4D stimulus vectors. It does not prove that the 4D stimulus itself emerged label-free from language semantics; the historical full80 fallback stimulus path is a known confound.",
            "",
        ]
    )
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_rows(args.summary_csv)
    if args.limit > 0:
        rows = rows[: args.limit]
    base_dynamics = load_dynamics(args.config)

    input_features = np.stack([stimulus(row) for row in rows], axis=0)
    input_distance = standardized_distance_matrix(input_features)

    condition_features: dict[str, np.ndarray] = {}
    run_summaries: dict[str, list[dict[str, Any]]] = {}
    for condition in CONDITIONS:
        features: list[np.ndarray] = []
        summaries: list[dict[str, Any]] = []
        for idx, row in enumerate(rows, start=1):
            feature, summary, _ = run_sample(row, condition, base_dynamics, args.seed)
            features.append(feature)
            summaries.append(summary)
            if args.progress_every > 0 and idx % args.progress_every == 0:
                print(f"[{condition.name}] {idx}/{len(rows)}")
        condition_features[condition.name] = np.stack(features, axis=0)
        run_summaries[condition.name] = summaries

    rng = np.random.default_rng(args.seed + 991)
    permutation = rng.permutation(len(rows))
    shuffled_features: list[np.ndarray] = []
    shuffled_summaries: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        donor = dict(row)
        donor_stim = stimulus(rows[int(permutation[idx])])
        for name, value in zip(STIM_COLUMNS, donor_stim, strict=False):
            donor[name] = str(float(value))
        feature, summary, _ = run_sample(donor, CONDITIONS[0], base_dynamics, args.seed)
        shuffled_features.append(feature)
        shuffled_summaries.append(summary)
    condition_features["shuffled_stimulus"] = np.stack(shuffled_features, axis=0)
    run_summaries["shuffled_stimulus"] = shuffled_summaries

    axis_summary: dict[str, dict[str, Any]] = {}
    for axis in LABEL_AXES:
        axis_result: dict[str, Any] = {"input": axis_metric(rows, input_distance, axis)}
        for name, features in condition_features.items():
            axis_result[name] = axis_metric(rows, standardized_distance_matrix(features), axis)
        axis_summary[axis] = axis_result

    pair = pair_mechanism_detail(rows, base_dynamics, args.seed)

    report = {
        "n_samples": len(rows),
        "seed": args.seed,
        "source_summary": str(args.summary_csv),
        "config": str(args.config),
        "feature_definition": "4 temporal bins x [mean,max] of log1p(K) over 256 neurons",
        "axis_summary": axis_summary,
        "matched_pair": pair,
        "run_summary": {
            name: {
                "mean_ticks": float(np.mean([item["ticks"] for item in summaries])),
                "mean_density": float(np.mean([item["mean_density"] for item in summaries])),
            }
            for name, summaries in run_summaries.items()
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "trace_mechanism_report.json"
    md_path = args.output_dir / "trace_mechanism_report.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md_path.write_text(markdown_report(report), encoding="utf-8")

    flat_rows: list[dict[str, Any]] = []
    for axis, by_condition in axis_summary.items():
        for condition, metric in by_condition.items():
            flat_rows.append({"axis": axis, "condition": condition, **metric})
    write_csv(args.output_dir / "trace_mechanism_metrics.csv", flat_rows)
    print(md_path.read_text(encoding="utf-8"))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("v3.1/outputs/neural_trace_final_candidate_thr064_topk2_clip15_inh018_full80/neural_trace_summary.csv"),
    )
    parser.add_argument("--config", type=Path, default=Path("v3.1/configs/final_dynamics_v1.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("v3.1/outputs/trace_mechanism_v1"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--progress-every", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
