#!/usr/bin/env python3
"""Probe whether trace fields behave like an emotion-state space.

This script intentionally uses only the Python standard library so v3.1 can run
before the heavier app/runtime dependencies are available.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_FIELDS = [
    "episode_label",
    "valence",
    "arousal",
    "target",
    "control_state",
    "social_orientation",
    "preserve",
    "avoid",
    "action_tendency",
]

LABEL_AXES = [
    "episode_label",
    "valence",
    "arousal",
    "target",
    "control_state",
    "social_orientation",
    "action_tendency",
]

NUMERIC_FIELDS = {"valence", "arousal"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def normalize(value: Any) -> str:
    text = "" if value is None else str(value)
    return " ".join(text.strip().lower().split())


def to_float(value: str) -> float | None:
    try:
        if value == "":
            return None
        return float(value)
    except ValueError:
        return None


def tokenize(value: str) -> set[str]:
    value = normalize(value)
    if not value:
        return set()
    for sep in [";", "|", ",", "/", "\\n"]:
        value = value.replace(sep, " ")
    return {part for part in value.split() if part}


def numeric_ranges(rows: list[dict[str, str]], fields: list[str]) -> dict[str, tuple[float, float]]:
    ranges: dict[str, tuple[float, float]] = {}
    for field in fields:
        values = [to_float(normalize(row.get(field, ""))) for row in rows]
        nums = [v for v in values if v is not None]
        if nums:
            ranges[field] = (min(nums), max(nums))
    return ranges


def field_distance(a: str, b: str, field: str, ranges: dict[str, tuple[float, float]]) -> float:
    a = normalize(a)
    b = normalize(b)
    if field in NUMERIC_FIELDS:
        av = to_float(a)
        bv = to_float(b)
        if av is None or bv is None:
            return 1.0 if a != b else 0.0
        low, high = ranges.get(field, (0.0, 1.0))
        span = max(high - low, 1e-9)
        return min(abs(av - bv) / span, 1.0)

    aset = tokenize(a)
    bset = tokenize(b)
    if not aset and not bset:
        return 0.0
    if not aset or not bset:
        return 1.0
    return 1.0 - (len(aset & bset) / len(aset | bset))


def trace_distance(
    left: dict[str, str],
    right: dict[str, str],
    fields: list[str],
    ranges: dict[str, tuple[float, float]],
) -> float:
    distances = [
        field_distance(left.get(field, ""), right.get(field, ""), field, ranges)
        for field in fields
    ]
    return sum(distances) / max(len(distances), 1)


def majority_baseline(rows: list[dict[str, str]], axis: str) -> float:
    labels = [normalize(row.get(axis, "")) for row in rows if normalize(row.get(axis, ""))]
    if not labels:
        return 0.0
    return Counter(labels).most_common(1)[0][1] / len(labels)


def nearest_neighbor_consistency(
    rows: list[dict[str, str]],
    fields: list[str],
    axes: list[str],
) -> dict[str, dict[str, float]]:
    ranges = numeric_ranges(rows, fields)
    totals = {axis: 0 for axis in axes}
    hits = {axis: 0 for axis in axes}

    for i, row in enumerate(rows):
        best_j = None
        best_d = math.inf
        for j, other in enumerate(rows):
            if i == j:
                continue
            d = trace_distance(row, other, fields, ranges)
            if d < best_d:
                best_d = d
                best_j = j
        if best_j is None:
            continue
        neighbor = rows[best_j]
        for axis in axes:
            label = normalize(row.get(axis, ""))
            other_label = normalize(neighbor.get(axis, ""))
            if not label or not other_label:
                continue
            totals[axis] += 1
            if label == other_label:
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


def group_distance_summary(
    rows: list[dict[str, str]],
    fields: list[str],
    axes: list[str],
) -> dict[str, dict[str, float]]:
    ranges = numeric_ranges(rows, fields)
    summary: dict[str, dict[str, float]] = {}

    for axis in axes:
        intra: list[float] = []
        inter: list[float] = []
        for i in range(len(rows)):
            left_label = normalize(rows[i].get(axis, ""))
            if not left_label:
                continue
            for j in range(i + 1, len(rows)):
                right_label = normalize(rows[j].get(axis, ""))
                if not right_label:
                    continue
                d = trace_distance(rows[i], rows[j], fields, ranges)
                if left_label == right_label:
                    intra.append(d)
                else:
                    inter.append(d)
        intra_mean = sum(intra) / len(intra) if intra else 0.0
        inter_mean = sum(inter) / len(inter) if inter else 0.0
        summary[axis] = {
            "intra_pairs": len(intra),
            "inter_pairs": len(inter),
            "mean_intra_distance": round(intra_mean, 6),
            "mean_inter_distance": round(inter_mean, 6),
            "separation": round(inter_mean - intra_mean, 6),
        }
    return summary


def value_counts(rows: list[dict[str, str]], axes: list[str]) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for axis in axes:
        counter = Counter(normalize(row.get(axis, "")) or "<missing>" for row in rows)
        counts[axis] = dict(counter.most_common())
    return counts


def run(input_path: Path, output_path: Path, fields: list[str], axes: list[str]) -> dict[str, Any]:
    rows = read_rows(input_path)
    available_fields = [field for field in fields if field in rows[0]] if rows else []
    available_axes = [axis for axis in axes if axis in rows[0]] if rows else []

    report = {
        "input_path": str(input_path),
        "record_count": len(rows),
        "trace_fields_used": available_fields,
        "label_axes_evaluated": available_axes,
        "value_counts": value_counts(rows, available_axes),
        "nearest_neighbor": nearest_neighbor_consistency(rows, available_fields, available_axes),
        "group_distances": group_distance_summary(rows, available_fields, available_axes),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("../v4/outputs/experiments/superiority_targeted_v1/targeted_records.csv"),
        help="CSV containing trace fields.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/trace_emotion_probe_summary.json"),
        help="Output JSON summary path.",
    )
    parser.add_argument(
        "--fields",
        nargs="*",
        default=DEFAULT_FIELDS,
        help="Trace fields used for distance calculation.",
    )
    parser.add_argument(
        "--axes",
        nargs="*",
        default=LABEL_AXES,
        help="Label axes used for consistency/separation evaluation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run(args.input, args.output, args.fields, args.axes)
    print(json.dumps({
        "record_count": report["record_count"],
        "output": str(args.output),
        "nearest_neighbor": report["nearest_neighbor"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

