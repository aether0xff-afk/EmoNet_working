from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from emonet.cli import EDGE_STYLE_AXES, NEGATIVE_RAW_AFFECT_AXES, SOFT_BIAS_AXES, STYLE_AXIS_PROFILES


BUCKET_CUES: dict[str, list[str]] = {
    "anger_resentment": [
        "화",
        "분노",
        "짜증",
        "열받",
        "억울",
        "원망",
        "배신",
        "질투",
        "미워",
        "싫어",
        "서운",
    ],
    "despair_helplessness": [
        "절망",
        "포기",
        "막막",
        "무기력",
        "우울",
        "죽고",
        "사라지고",
        "끝났",
        "힘들",
        "괴로",
    ],
    "fear_panic": [
        "불안",
        "두려",
        "무서",
        "겁",
        "공포",
        "걱정",
        "떨려",
        "긴장",
        "패닉",
    ],
    "shame_guilt": [
        "창피",
        "부끄",
        "수치",
        "죄책",
        "미안",
        "후회",
        "자책",
        "민망",
    ],
    "relationship_conflict": [
        "친구",
        "가족",
        "엄마",
        "아빠",
        "연인",
        "남친",
        "여친",
        "동료",
        "상사",
        "무시",
        "싸웠",
    ],
}


def _clean(value: object) -> str:
    return " ".join(str(value or "").split())


def _axis_columns(df: pd.DataFrame, axes: list[str]) -> dict[str, str]:
    columns: dict[str, str] = {}
    for idx, axis in enumerate(axes):
        numbered = f"s_{idx}"
        if numbered in df.columns:
            columns[axis] = numbered
        elif axis in df.columns:
            columns[axis] = axis
    return columns


def _bool_mask(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(True, index=df.index)
    return df[column].astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def _cue_count(text: str, cues: list[str]) -> int:
    return sum(1 for cue in cues if cue in text)


def _bucket_for_text(text: str) -> tuple[str, str, int]:
    scores = {bucket: _cue_count(text, cues) for bucket, cues in BUCKET_CUES.items()}
    bucket, score = max(scores.items(), key=lambda item: item[1])
    matched = [cue for cue in BUCKET_CUES[bucket] if cue in text]
    if score <= 0:
        return "low_cue_hardcase", "", 0
    return bucket, ", ".join(matched[:5]), int(score)


def build_style_relabel_set(
    *,
    input_csv: Path,
    output_csv: Path,
    manifest_json: Path,
    style_profile: str,
    keep_column: str,
    target_size: int,
    max_per_bucket: int,
    seed: int,
) -> dict[str, Any]:
    df = pd.read_csv(input_csv)
    if style_profile not in STYLE_AXIS_PROFILES:
        raise ValueError(f"unknown style profile: {style_profile}")
    axes = list(STYLE_AXIS_PROFILES[style_profile])
    axis_columns = _axis_columns(df, axes)
    if "text" not in df.columns:
        raise ValueError("input CSV must contain text column")

    source = df.loc[_bool_mask(df, keep_column)].copy()
    source["record_id"] = source.get("sample_id", source.index.to_series()).astype(str)
    source["text"] = source["text"].map(_clean)

    negative_cols = [axis_columns[axis] for axis in NEGATIVE_RAW_AFFECT_AXES if axis in axis_columns]
    edge_cols = [axis_columns[axis] for axis in EDGE_STYLE_AXES if axis in axis_columns]
    soft_cols = [axis_columns[axis] for axis in SOFT_BIAS_AXES if axis in axis_columns]
    source["current_negative_raw_max"] = source[negative_cols].apply(pd.to_numeric, errors="coerce").max(axis=1)
    source["current_negative_raw_mean"] = source[negative_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    source["current_edge_mean"] = source[edge_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
    source["current_soft_bias_mean"] = source[soft_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)

    bucket_rows = source["text"].map(_bucket_for_text)
    source["relabel_bucket"] = [row[0] for row in bucket_rows]
    source["matched_cues"] = [row[1] for row in bucket_rows]
    source["cue_count"] = [row[2] for row in bucket_rows]
    source["underlabel_score"] = (
        1.2 * source["cue_count"].astype(float)
        + (1.0 - source["current_negative_raw_max"].fillna(0.0))
        + 0.35 * source["current_soft_bias_mean"].fillna(0.0)
        - 0.25 * source["current_edge_mean"].fillna(0.0)
    )

    selected_parts: list[pd.DataFrame] = []
    used_ids: set[str] = set()
    for bucket in BUCKET_CUES:
        candidates = source[(source["relabel_bucket"] == bucket) & ~source["record_id"].isin(used_ids)].copy()
        candidates = candidates.sort_values(["underlabel_score", "current_negative_raw_max"], ascending=[False, True])
        take = candidates.head(max_per_bucket)
        if not take.empty:
            used_ids.update(take["record_id"].astype(str).tolist())
            selected_parts.append(take)

    selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame(columns=source.columns)
    if len(selected) < target_size:
        remaining = source[~source["record_id"].isin(used_ids)].copy()
        remaining = remaining.sort_values(["underlabel_score", "current_negative_raw_max"], ascending=[False, True])
        selected = pd.concat([selected, remaining.head(target_size - len(selected))], ignore_index=True)
    selected = selected.head(target_size).copy()

    output_columns = [
        "record_id",
        "text",
        "relabel_bucket",
        "matched_cues",
        "cue_count",
        "underlabel_score",
        "current_negative_raw_max",
        "current_negative_raw_mean",
        "current_edge_mean",
        "current_soft_bias_mean",
        "keep_reason",
        "rebalance_bucket",
    ]
    for axis in [
        "softness",
        "calmness",
        "cooperativeness",
        "positivity",
        "sharpness",
        "tension",
        "hostility",
        "resentment",
        "despair",
        "volatility",
        "fearfulness",
        "shame",
    ]:
        if axis in axis_columns:
            selected[f"current_{axis}"] = pd.to_numeric(selected[axis_columns[axis]], errors="coerce")
            output_columns.append(f"current_{axis}")

    for column in output_columns:
        if column not in selected.columns:
            selected[column] = ""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    selected[output_columns].to_csv(output_csv, index=False, encoding="utf-8-sig")

    manifest = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "rows": int(len(selected)),
        "target_size": int(target_size),
        "keep_column": keep_column,
        "style_profile": style_profile,
        "bucket_counts": {str(key): int(value) for key, value in selected["relabel_bucket"].value_counts().to_dict().items()},
        "mean_current_negative_raw_max": round(float(selected["current_negative_raw_max"].mean()), 6) if len(selected) else 0.0,
        "mean_current_soft_bias_mean": round(float(selected["current_soft_bias_mean"].mean()), 6) if len(selected) else 0.0,
        "seed": int(seed),
    }
    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Build hard cases for felt-state/style relabeling.")
    parser.add_argument("--input-csv", default="outputs/z/out_z_training_learned_extended40_calref_v1.csv")
    parser.add_argument("--output-csv", default="outputs/research/style_relabel_v1/style_relabel_candidates.csv")
    parser.add_argument("--manifest-json", default="outputs/research/style_relabel_v1/style_relabel_candidates_manifest.json")
    parser.add_argument("--style-profile", choices=sorted(STYLE_AXIS_PROFILES), default="extended40")
    parser.add_argument("--keep-column", default="keep_sample")
    parser.add_argument("--target-size", type=int, default=120)
    parser.add_argument("--max-per-bucket", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = build_style_relabel_set(
        input_csv=Path(args.input_csv),
        output_csv=Path(args.output_csv),
        manifest_json=Path(args.manifest_json),
        style_profile=args.style_profile,
        keep_column=args.keep_column,
        target_size=args.target_size,
        max_per_bucket=args.max_per_bucket,
        seed=args.seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
