from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from emonet.cli import EDGE_STYLE_AXES, NEGATIVE_RAW_AFFECT_AXES, SOFT_BIAS_AXES, STYLE_AXIS_PROFILES


def _as_bool_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        raise ValueError(f"keep column not found: {column}")
    value = df[column]
    if value.dtype == bool:
        return value.fillna(False)
    return value.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def _axis_columns(df: pd.DataFrame, axes: list[str]) -> dict[str, str]:
    columns: dict[str, str] = {}
    for idx, axis in enumerate(axes):
        numbered = f"s_{idx}"
        if numbered in df.columns:
            columns[axis] = numbered
        elif axis in df.columns:
            columns[axis] = axis
    missing = [axis for axis in axes if axis not in columns]
    if missing:
        raise ValueError(f"missing style columns for axes: {', '.join(missing[:8])}")
    return columns


def _mean_axis(df: pd.DataFrame, axis_columns: dict[str, str], axis: str) -> float:
    return float(pd.to_numeric(df[axis_columns[axis]], errors="coerce").mean())


def _mean_group(df: pd.DataFrame, axis_columns: dict[str, str], axes: list[str]) -> float:
    present = [axis for axis in axes if axis in axis_columns]
    if not present or df.empty:
        return 0.0
    values = [_mean_axis(df, axis_columns, axis) for axis in present]
    return float(sum(values) / len(values))


def _top_shifted(df: pd.DataFrame, axis_columns: dict[str, str], limit: int) -> list[dict[str, Any]]:
    rows = []
    for axis in axis_columns:
        mean_value = _mean_axis(df, axis_columns, axis)
        rows.append(
            {
                "axis": axis,
                "mean": round(mean_value, 6),
                "distance_from_neutral": round(abs(mean_value - 0.5), 6),
            }
        )
    return sorted(rows, key=lambda row: row["distance_from_neutral"], reverse=True)[:limit]


def _value_counts(df: pd.DataFrame, column: str) -> dict[str, int]:
    if column not in df.columns:
        return {}
    return {str(key): int(value) for key, value in df[column].fillna("").astype(str).value_counts().to_dict().items()}


def summarize_keep_column(
    df: pd.DataFrame,
    *,
    keep_column: str,
    axes: list[str],
    axis_columns: dict[str, str],
    top_limit: int,
) -> dict[str, Any]:
    mask = _as_bool_series(df, keep_column)
    keep = df.loc[mask].copy()
    summary: dict[str, Any] = {
        "keep_column": keep_column,
        "rows": int(len(keep)),
        "rate": round(float(mask.mean()), 6),
        "soft_bias_mean": round(_mean_group(keep, axis_columns, SOFT_BIAS_AXES), 6),
        "negative_raw_mean": round(_mean_group(keep, axis_columns, NEGATIVE_RAW_AFFECT_AXES), 6),
        "edge_mean": round(_mean_group(keep, axis_columns, EDGE_STYLE_AXES), 6),
        "top_shifted_axes": _top_shifted(keep, axis_columns, top_limit),
        "bucket_counts": _value_counts(keep, "rebalance_bucket"),
        "keep_reason_counts": _value_counts(keep, "keep_reason"),
    }
    summary["focus_axes"] = {
        axis: round(_mean_axis(keep, axis_columns, axis), 6)
        for axis in [
            "softness",
            "calmness",
            "cooperativeness",
            "positivity",
            "warmth",
            "trust",
            "sharpness",
            "tension",
            "hostility",
            "resentment",
            "despair",
            "volatility",
            "fearfulness",
            "shame",
        ]
        if axis in axes
    }
    return summary


def build_markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Style Bias Audit",
        "",
        f"- input_csv: `{payload['input_csv']}`",
        f"- rows: `{payload['rows']}`",
        f"- style_profile: `{payload['style_profile']}`",
        "",
        "## Keep Column Comparison",
        "",
        "| keep column | rows | soft mean | negative raw mean | edge mean | top shifted axes |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for item in payload["keep_summaries"]:
        shifted = ", ".join(f"{row['axis']}={row['mean']:.4f}" for row in item["top_shifted_axes"][:5])
        lines.append(
            "| {keep_column} | {rows} | {soft_bias_mean:.4f} | {negative_raw_mean:.4f} | {edge_mean:.4f} | {shifted} |".format(
                shifted=shifted,
                **item,
            )
        )
    lines.extend(["", "## Focus Axes", ""])
    for item in payload["keep_summaries"]:
        lines.append(f"### {item['keep_column']}")
        for axis, value in item["focus_axes"].items():
            lines.append(f"- {axis}: {value:.4f}")
        if item["bucket_counts"]:
            lines.append(f"- buckets: `{json.dumps(item['bucket_counts'], ensure_ascii=False)}`")
        if item["keep_reason_counts"]:
            lines.append(f"- keep reasons: `{json.dumps(item['keep_reason_counts'], ensure_ascii=False)}`")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def audit_style_bias(
    *,
    input_csv: Path,
    output_json: Path,
    output_md: Path | None,
    style_profile: str,
    keep_columns: list[str],
    top_limit: int,
) -> dict[str, Any]:
    df = pd.read_csv(input_csv)
    if style_profile not in STYLE_AXIS_PROFILES:
        raise ValueError(f"unknown style profile: {style_profile}")
    axes = list(STYLE_AXIS_PROFILES[style_profile])
    axis_columns = _axis_columns(df, axes)
    payload = {
        "input_csv": str(input_csv),
        "rows": int(len(df)),
        "style_profile": style_profile,
        "style_axes": axes,
        "keep_summaries": [
            summarize_keep_column(
                df,
                keep_column=keep_column,
                axes=axes,
                axis_columns=axis_columns,
                top_limit=top_limit,
            )
            for keep_column in keep_columns
            if keep_column in df.columns
        ],
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    if output_md is not None:
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(build_markdown_report(payload), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit style-axis bias for labeled EmoNet CSVs.")
    parser.add_argument("--input-csv", default="outputs/z/out_z_training_learned_extended40_calref_v1.csv")
    parser.add_argument("--output-json", default="outputs/research/style_bias_audit/style_bias_audit.json")
    parser.add_argument("--output-md", default="outputs/research/style_bias_audit/STYLE_BIAS_AUDIT.md")
    parser.add_argument("--style-profile", choices=sorted(STYLE_AXIS_PROFILES), default="extended40")
    parser.add_argument("--keep-columns", nargs="+", default=["keep_sample", "keep_sample_rebalanced"])
    parser.add_argument("--top-limit", type=int, default=10)
    args = parser.parse_args()

    payload = audit_style_bias(
        input_csv=Path(args.input_csv),
        output_json=Path(args.output_json),
        output_md=Path(args.output_md) if args.output_md else None,
        style_profile=args.style_profile,
        keep_columns=list(args.keep_columns),
        top_limit=args.top_limit,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
