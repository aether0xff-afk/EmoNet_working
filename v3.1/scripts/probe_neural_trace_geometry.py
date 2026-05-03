#!/usr/bin/env python3
"""Probe whether neural activation traces form emotion-relevant geometry."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


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

TEMPORAL_BINS = 4
TRANSITION_BINS = 256


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def norm(value: Any) -> str:
    return " ".join(str(value or "").strip().lower().split())


def to_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "0") or 0)
    except ValueError:
        return 0.0


def _safe_mean_max(matrix: np.ndarray, width: int) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros((width * 2,), dtype=np.float32)
    return np.concatenate([matrix.mean(axis=0), matrix.max(axis=0)]).astype(np.float32)


def _temporal_pool(matrix: np.ndarray, bins: int = TEMPORAL_BINS) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros((0,), dtype=np.float32)
    chunks = np.array_split(matrix, bins, axis=0)
    pooled: list[np.ndarray] = []
    for chunk in chunks:
        if chunk.size == 0:
            pooled.append(np.zeros((matrix.shape[1] * 2,), dtype=np.float32))
        else:
            pooled.append(_safe_mean_max(chunk, matrix.shape[1]))
    return np.concatenate(pooled).astype(np.float32)


def _active_count_stats(active_counts: np.ndarray) -> np.ndarray:
    values = np.asarray(active_counts, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return np.zeros((8,), dtype=np.float32)
    diffs = np.diff(values) if values.size > 1 else np.zeros((1,), dtype=np.float32)
    return np.asarray(
        [
            values.mean(),
            values.std(),
            values.min(),
            values.max(),
            values[-1] - values[0],
            diffs.mean(),
            diffs.std(),
            float(values.argmax()) / max(1, values.size - 1),
        ],
        dtype=np.float32,
    )


def _route_histogram(route_ids: np.ndarray, n_neurons: int) -> np.ndarray:
    route = np.asarray(route_ids, dtype=np.int64).reshape(-1)
    valid = route[(route >= 0) & (route < n_neurons)]
    hist = np.zeros((n_neurons,), dtype=np.float32)
    if valid.size:
        counts = np.bincount(valid, minlength=n_neurons).astype(np.float32)
        hist[: counts.size] = counts[:n_neurons]
        hist /= max(1.0, float(valid.size))
    return hist


def _transition_hash(route_ids: np.ndarray, bins: int = TRANSITION_BINS) -> np.ndarray:
    route = np.asarray(route_ids, dtype=np.int64).reshape(-1)
    route = route[route >= 0]
    hist = np.zeros((bins,), dtype=np.float32)
    if route.size < 2:
        return hist
    for left, right in zip(route[:-1], route[1:]):
        bucket = int((left * 1315423911 + right * 2654435761) % bins)
        hist[bucket] += 1.0
    hist /= max(1.0, float(route.size - 1))
    return hist


def load_trace_features(summary_rows: list[dict[str, str]], trace_dir: Path, feature_kind: str) -> tuple[list[dict[str, str]], np.ndarray]:
    kept_rows: list[dict[str, str]] = []
    features: list[np.ndarray] = []
    for row in summary_rows:
        record_id = str(row.get("record_id", ""))
        npz_path = trace_dir / f"{record_id}.npz"
        if not npz_path.exists():
            continue
        with np.load(npz_path, allow_pickle=False) as payload:
            if feature_kind == "z":
                feature = np.asarray(payload["z"], dtype=np.float32).reshape(-1)
            elif feature_kind == "activation_meanmax":
                activation = np.asarray(payload["activation"], dtype=np.float32)
                width = int(activation.shape[1]) if activation.ndim == 2 else 0
                feature = _safe_mean_max(activation, width)
            elif feature_kind == "activation_temporal":
                activation = np.asarray(payload["activation"], dtype=np.float32)
                feature = _temporal_pool(activation)
            elif feature_kind == "branch_mean":
                branch = np.asarray(payload["branch_tensor"], dtype=np.float32)
                width = int(branch.shape[1]) if branch.ndim == 2 else 0
                feature = _safe_mean_max(branch, width)
            elif feature_kind == "branch_temporal":
                branch = np.asarray(payload["branch_tensor"], dtype=np.float32)
                feature = _temporal_pool(branch)
            elif feature_kind == "route_histogram":
                activation = np.asarray(payload["activation"], dtype=np.float32)
                n_neurons = int(activation.shape[1]) if activation.ndim == 2 else 0
                feature = _route_histogram(payload["dominant_branch_ids"], n_neurons)
            elif feature_kind == "transition_hash":
                feature = _transition_hash(payload["dominant_branch_ids"])
            elif feature_kind == "active_stats":
                feature = _active_count_stats(payload["active_counts"])
            elif feature_kind == "branch_plus_temporal":
                branch = np.asarray(payload["branch_tensor"], dtype=np.float32)
                branch_width = int(branch.shape[1]) if branch.ndim == 2 else 0
                activation = np.asarray(payload["activation"], dtype=np.float32)
                n_neurons = int(activation.shape[1]) if activation.ndim == 2 else 0
                feature = np.concatenate(
                    [
                        _safe_mean_max(branch, branch_width),
                        _temporal_pool(branch),
                        _active_count_stats(payload["active_counts"]),
                        _route_histogram(payload["dominant_branch_ids"], n_neurons),
                        _transition_hash(payload["dominant_branch_ids"]),
                    ]
                ).astype(np.float32)
            else:
                raise ValueError(f"unknown feature kind: {feature_kind}")
        if feature.size == 0 or not np.all(np.isfinite(feature)):
            continue
        kept_rows.append(row)
        features.append(feature)
    if not features:
        return [], np.zeros((0, 0), dtype=np.float32)
    width = max(feature.size for feature in features)
    matrix = np.zeros((len(features), width), dtype=np.float32)
    for idx, feature in enumerate(features):
        matrix[idx, : feature.size] = feature
    return kept_rows, matrix


def standardize(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix
    mean = matrix.mean(axis=0, keepdims=True)
    std = matrix.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    return (matrix - mean) / std


def pairwise_distances(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    normed = standardize(matrix)
    diff = normed[:, None, :] - normed[None, :, :]
    return np.sqrt(np.mean(diff * diff, axis=2)).astype(np.float32)


def majority_baseline(rows: list[dict[str, str]], axis: str) -> float:
    labels = [norm(row.get(axis, "")) for row in rows if norm(row.get(axis, ""))]
    if not labels:
        return 0.0
    return Counter(labels).most_common(1)[0][1] / len(labels)


def nearest_neighbor(rows: list[dict[str, str]], distances: np.ndarray, axes: list[str]) -> dict[str, dict[str, float]]:
    totals = {axis: 0 for axis in axes}
    hits = {axis: 0 for axis in axes}
    if distances.shape[0] < 2:
        return {}
    masked = distances.copy()
    np.fill_diagonal(masked, math.inf)
    nearest = np.argmin(masked, axis=1)
    for idx, neighbor_idx in enumerate(nearest):
        for axis in axes:
            left = norm(rows[idx].get(axis, ""))
            right = norm(rows[int(neighbor_idx)].get(axis, ""))
            if not left or not right:
                continue
            totals[axis] += 1
            if left == right:
                hits[axis] += 1
    result: dict[str, dict[str, float]] = {}
    for axis in axes:
        n = totals[axis]
        consistency = hits[axis] / n if n else 0.0
        baseline = majority_baseline(rows, axis)
        result[axis] = {
            "n": n,
            "nearest_neighbor_consistency": round(consistency, 6),
            "majority_baseline": round(baseline, 6),
            "lift": round(consistency - baseline, 6),
        }
    return result


def group_distances(rows: list[dict[str, str]], distances: np.ndarray, axes: list[str]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for axis in axes:
        intra: list[float] = []
        inter: list[float] = []
        for i in range(len(rows)):
            left = norm(rows[i].get(axis, ""))
            if not left:
                continue
            for j in range(i + 1, len(rows)):
                right = norm(rows[j].get(axis, ""))
                if not right:
                    continue
                if left == right:
                    intra.append(float(distances[i, j]))
                else:
                    inter.append(float(distances[i, j]))
        intra_mean = sum(intra) / len(intra) if intra else 0.0
        inter_mean = sum(inter) / len(inter) if inter else 0.0
        result[axis] = {
            "intra_pairs": len(intra),
            "inter_pairs": len(inter),
            "mean_intra_distance": round(intra_mean, 6),
            "mean_inter_distance": round(inter_mean, 6),
            "separation": round(inter_mean - intra_mean, 6),
        }
    return result


def branch_health(rows: list[dict[str, str]]) -> dict[str, Any]:
    lengths = [to_float(row, "dominant_branch_len") for row in rows]
    densities = [to_float(row, "activation_density") for row in rows]
    if not lengths:
        return {}
    return {
        "n": len(lengths),
        "mean_dominant_branch_len": round(sum(lengths) / len(lengths), 6),
        "len1_count": sum(1 for value in lengths if value <= 1),
        "len1_ratio": round(sum(1 for value in lengths if value <= 1) / len(lengths), 6),
        "mean_activation_density": round(sum(densities) / len(densities), 6),
    }


def value_counts(rows: list[dict[str, str]], axes: list[str]) -> dict[str, dict[str, int]]:
    return {
        axis: dict(Counter(norm(row.get(axis, "")) or "<missing>" for row in rows).most_common())
        for axis in axes
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    summary_rows = read_csv(args.summary_csv)
    if args.limit and args.limit > 0:
        summary_rows = summary_rows[: args.limit]
    rows, matrix = load_trace_features(summary_rows, args.trace_dir, args.feature_kind)
    distances = pairwise_distances(matrix)
    report = {
        "summary_csv": str(args.summary_csv),
        "trace_dir": str(args.trace_dir),
        "feature_kind": args.feature_kind,
        "n": len(rows),
        "feature_dim": int(matrix.shape[1]) if matrix.ndim == 2 else 0,
        "branch_health": branch_health(rows),
        "value_counts": value_counts(rows, LABEL_AXES),
        "nearest_neighbor": nearest_neighbor(rows, distances, LABEL_AXES),
        "group_distances": group_distances(rows, distances, LABEL_AXES),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, default=Path("outputs/neural_trace_probe_v1/neural_trace_summary.csv"))
    parser.add_argument("--trace-dir", type=Path, default=Path("outputs/neural_trace_probe_v1/traces_npz"))
    parser.add_argument(
        "--feature-kind",
        choices=[
            "z",
            "activation_meanmax",
            "activation_temporal",
            "branch_mean",
            "branch_temporal",
            "route_histogram",
            "transition_hash",
            "active_stats",
            "branch_plus_temporal",
        ],
        default="z",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/neural_trace_probe_v1/neural_trace_geometry_z.json"))
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run(args)
    print(json.dumps({
        "n": report["n"],
        "feature_kind": report["feature_kind"],
        "branch_health": report["branch_health"],
        "nearest_neighbor": report["nearest_neighbor"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
