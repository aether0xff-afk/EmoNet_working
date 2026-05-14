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

from emonet.cli import NEGATIVE_RAW_AFFECT_AXES, SOFT_BIAS_AXES, STYLE_AXIS_PROFILES


def _axis_columns(df: pd.DataFrame, axes: list[str]) -> dict[str, str]:
    columns: dict[str, str] = {}
    for idx, axis in enumerate(axes):
        numbered = f"s_{idx}"
        if numbered in df.columns:
            columns[axis] = numbered
        elif axis in df.columns:
            columns[axis] = axis
        else:
            raise ValueError(f"missing style axis column: {axis}")
    return columns


def _row_id_series(df: pd.DataFrame) -> pd.Series:
    if "sample_id" in df.columns:
        return df["sample_id"].astype(str)
    if "record_id" in df.columns:
        return df["record_id"].astype(str)
    return df.index.to_series().astype(str)


def _mean_group(df: pd.DataFrame, axis_columns: dict[str, str], axes: list[str], mask: pd.Series) -> float:
    cols = [axis_columns[axis] for axis in axes if axis in axis_columns]
    if not cols or int(mask.sum()) == 0:
        return 0.0
    return float(df.loc[mask, cols].apply(pd.to_numeric, errors="coerce").mean(axis=1).mean())


def apply_style_relabels(
    *,
    base_csv: Path,
    relabel_csv: Path,
    output_csv: Path,
    manifest_json: Path,
    style_profile: str,
    keep_column: str,
) -> dict[str, Any]:
    base = pd.read_csv(base_csv)
    relabels = pd.read_csv(relabel_csv)
    if style_profile not in STYLE_AXIS_PROFILES:
        raise ValueError(f"unknown style profile: {style_profile}")
    axes = list(STYLE_AXIS_PROFILES[style_profile])
    axis_columns = _axis_columns(base, axes)
    if "record_id" not in relabels.columns:
        raise ValueError("relabel CSV must contain record_id")
    relabel_map = {str(row.record_id): row._asdict() for row in relabels.itertuples(index=False)}

    out = base.copy()
    out["style_relabel_applied"] = False
    out["style_relabel_rationale"] = ""
    out["style_relabel_bucket"] = ""
    row_ids = _row_id_series(out)
    applied = 0
    for idx, record_id in row_ids.items():
        relabel = relabel_map.get(str(record_id))
        if not relabel:
            continue
        for axis in axes:
            value = relabel.get(f"calibrated_{axis}")
            if pd.isna(value):
                continue
            out.at[idx, axis_columns[axis]] = float(value)
        out.at[idx, "style_relabel_applied"] = True
        out.at[idx, "style_relabel_rationale"] = str(relabel.get("rationale", ""))
        out.at[idx, "style_relabel_bucket"] = str(relabel.get("relabel_bucket", ""))
        applied += 1

    if keep_column not in out.columns:
        out[keep_column] = out.get("keep_sample", True)
    applied_mask = out["style_relabel_applied"].astype(bool)
    keep_mask = out[keep_column].astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False, encoding="utf-8-sig")

    base_axis_columns = _axis_columns(base, axes)
    base_ids = _row_id_series(base)
    base_applied_mask = base_ids.isin(set(relabel_map))
    manifest = {
        "base_csv": str(base_csv),
        "relabel_csv": str(relabel_csv),
        "output_csv": str(output_csv),
        "style_profile": style_profile,
        "rows": int(len(out)),
        "applied_rows": int(applied),
        "keep_column": keep_column,
        "base_relabel_subset_negative_raw_mean": round(
            _mean_group(base, base_axis_columns, NEGATIVE_RAW_AFFECT_AXES, base_applied_mask),
            6,
        ),
        "output_relabel_subset_negative_raw_mean": round(
            _mean_group(out, axis_columns, NEGATIVE_RAW_AFFECT_AXES, applied_mask),
            6,
        ),
        "base_relabel_subset_soft_bias_mean": round(
            _mean_group(base, base_axis_columns, SOFT_BIAS_AXES, base_applied_mask),
            6,
        ),
        "output_relabel_subset_soft_bias_mean": round(
            _mean_group(out, axis_columns, SOFT_BIAS_AXES, applied_mask),
            6,
        ),
        "output_keep_negative_raw_mean": round(_mean_group(out, axis_columns, NEGATIVE_RAW_AFFECT_AXES, keep_mask), 6),
        "output_keep_soft_bias_mean": round(_mean_group(out, axis_columns, SOFT_BIAS_AXES, keep_mask), 6),
    }
    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    manifest_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply Claude calibrated style relabels to a labeled EmoNet CSV.")
    parser.add_argument("--base-csv", default="outputs/z/out_z_training_learned_extended40_calref_v1.csv")
    parser.add_argument("--relabel-csv", default="outputs/research/style_relabel_v1/style_relabel_claude.csv")
    parser.add_argument("--output-csv", default="outputs/research/style_relabel_v1/out_z_training_learned_extended40_calref_v1_style_relabel_v1.csv")
    parser.add_argument("--manifest-json", default="outputs/research/style_relabel_v1/style_relabel_apply_manifest.json")
    parser.add_argument("--style-profile", choices=sorted(STYLE_AXIS_PROFILES), default="extended40")
    parser.add_argument("--keep-column", default="keep_sample")
    args = parser.parse_args()

    manifest = apply_style_relabels(
        base_csv=Path(args.base_csv),
        relabel_csv=Path(args.relabel_csv),
        output_csv=Path(args.output_csv),
        manifest_json=Path(args.manifest_json),
        style_profile=args.style_profile,
        keep_column=args.keep_column,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
