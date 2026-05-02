#!/usr/bin/env python3
"""Build trace ablation and perturbation records for causal proof."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


ABLATION_AXES = [
    "target",
    "social_orientation",
    "control_state",
    "action_tendency_class",
]

PERTURBATION_MAP = {
    "target": {
        "other": "self",
        "self": "other",
        "mixed": "other",
        "situation": "self",
    },
    "social_orientation": {
        "defend": "approach",
        "approach": "defend",
        "mixed": "withdraw",
        "withdraw": "approach",
    },
    "control_state": {
        "low": "high",
        "high": "low",
        "mixed": "low",
    },
    "action_tendency_class": {
        "defend": "repair",
        "confront": "withdraw",
        "repair": "confront",
        "seek_support": "defend",
        "plan": "confront",
        "withdraw": "approach",
        "approach": "withdraw",
        "inhibit": "confront",
        "other_action": "defend",
    },
}

ESSENTIAL_COLUMNS = [
    "record_id",
    "text",
    "episode_label",
    "episode_family",
    "appraisal_family",
    "valence",
    "arousal",
    "target",
    "control_state",
    "social_orientation",
    "preserve",
    "avoid",
    "action_tendency",
    "action_tendency_class",
]


def norm(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def choose_base_rows(rows: list[dict[str, str]], max_records: int) -> list[dict[str, str]]:
    """Choose diverse rows with preference for interpretable action/appraisal classes."""

    preferred = [
        row
        for row in rows
        if norm(row.get("action_tendency_class")) not in {"", "other_action"}
        and norm(row.get("appraisal_family")) not in {"", "mixed_appraisal"}
    ]
    fallback = [row for row in rows if row not in preferred]

    selected: list[dict[str, str]] = []
    seen_actions: Counter[str] = Counter()
    seen_appraisals: Counter[str] = Counter()

    for pool in [preferred, fallback]:
        for row in sorted(
            pool,
            key=lambda r: (
                seen_actions[norm(r.get("action_tendency_class"))],
                seen_appraisals[norm(r.get("appraisal_family"))],
                norm(r.get("record_id")),
            ),
        ):
            if len(selected) >= max_records:
                break
            selected.append(row)
            seen_actions[norm(row.get("action_tendency_class"))] += 1
            seen_appraisals[norm(row.get("appraisal_family"))] += 1
        if len(selected) >= max_records:
            break
    return selected


def make_control(row: dict[str, str]) -> dict[str, str]:
    out = {key: row.get(key, "") for key in ESSENTIAL_COLUMNS}
    out.update(
        {
            "causal_condition": "trace_full",
            "manipulation_type": "control",
            "manipulated_axis": "",
            "original_value": "",
            "new_value": "",
            "expected_effect": "Full trace should preserve appraisal, raw affect, and action tendency.",
        }
    )
    return out


def make_ablation(row: dict[str, str], axis: str) -> dict[str, str]:
    out = {key: row.get(key, "") for key in ESSENTIAL_COLUMNS}
    original = out.get(axis, "")
    out[axis] = ""
    out.update(
        {
            "causal_condition": f"ablate_{axis}",
            "manipulation_type": "ablation",
            "manipulated_axis": axis,
            "original_value": original,
            "new_value": "",
            "expected_effect": f"Removing {axis} should reduce fidelity for the matching emotional dimension.",
        }
    )
    return out


def make_perturbation(row: dict[str, str], axis: str) -> dict[str, str] | None:
    out = {key: row.get(key, "") for key in ESSENTIAL_COLUMNS}
    original = norm(out.get(axis))
    new_value = PERTURBATION_MAP.get(axis, {}).get(original)
    if not new_value:
        return None
    out[axis] = new_value
    out.update(
        {
            "causal_condition": f"perturb_{axis}",
            "manipulation_type": "perturbation",
            "manipulated_axis": axis,
            "original_value": original,
            "new_value": new_value,
            "expected_effect": f"Changing {axis} from {original} to {new_value} should shift response direction.",
        }
    )
    return out


def build_rows(source_rows: list[dict[str, str]], max_records: int) -> list[dict[str, str]]:
    selected = choose_base_rows(source_rows, max_records)
    output: list[dict[str, str]] = []

    for row in selected:
        output.append(make_control(row))
        for axis in ABLATION_AXES:
            output.append(make_ablation(row, axis))
        for axis in ABLATION_AXES:
            perturbed = make_perturbation(row, axis)
            if perturbed is not None:
                output.append(perturbed)

    return output


def summarize(rows: list[dict[str, str]], output_path: Path, source_path: Path, max_records: int) -> dict[str, object]:
    base_ids = sorted({row["record_id"] for row in rows})
    return {
        "source_path": str(source_path),
        "output_path": str(output_path),
        "requested_base_records": max_records,
        "base_record_count": len(base_ids),
        "row_count": len(rows),
        "condition_counts": dict(Counter(row["causal_condition"] for row in rows).most_common()),
        "manipulation_type_counts": dict(Counter(row["manipulation_type"] for row in rows).most_common()),
        "axis_counts": dict(Counter(row["manipulated_axis"] or "none" for row in rows).most_common()),
    }


def run(input_path: Path, output_path: Path, summary_path: Path, max_records: int) -> dict[str, object]:
    source_rows = read_csv(input_path)
    rows = build_rows(source_rows, max_records)
    fieldnames = ESSENTIAL_COLUMNS + [
        "causal_condition",
        "manipulation_type",
        "manipulated_axis",
        "original_value",
        "new_value",
        "expected_effect",
    ]
    write_csv(output_path, rows, fieldnames)
    summary = summarize(rows, output_path, input_path, max_records)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/targeted_records_trace_normalized.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/trace_causal_probe_set.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/trace_causal_probe_manifest.json"),
    )
    parser.add_argument("--max-records", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(args.input, args.output, args.summary, args.max_records)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

