from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def build_shard_ranges(total_rows: int, num_shards: int) -> list[tuple[int, int]]:
    if total_rows <= 0:
        raise ValueError("input CSV must contain at least one row")
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if num_shards > total_rows:
        raise ValueError("num_shards must be <= number of rows")

    base = total_rows // num_shards
    remainder = total_rows % num_shards
    start = 0
    ranges: list[tuple[int, int]] = []
    for shard_idx in range(num_shards):
        size = base + (1 if shard_idx < remainder else 0)
        end = start + size
        ranges.append((start, end))
        start = end
    return ranges


def shard_filename(prefix: str, shard_index: int, num_shards: int) -> str:
    return f"{prefix}.shard{shard_index + 1:02d}of{num_shards:02d}.csv"


def command_split(args: argparse.Namespace) -> None:
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    prefix = args.prefix or input_csv.stem

    df = pd.read_csv(input_csv)
    ranges = build_shard_ranges(len(df), args.num_shards)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, object]] = []
    for shard_index, (start, end) in enumerate(ranges):
        shard_df = df.iloc[start:end].copy()
        shard_path = output_dir / shard_filename(prefix, shard_index, args.num_shards)
        shard_df.to_csv(shard_path, index=False, encoding="utf-8-sig")
        manifest_rows.append(
            {
                "shard_index": shard_index + 1,
                "num_shards": args.num_shards,
                "start_row": start,
                "end_row_exclusive": end,
                "rows": int(len(shard_df)),
                "path": str(shard_path),
            }
        )

    manifest = {
        "input_csv": str(input_csv),
        "rows": int(len(df)),
        "num_shards": int(args.num_shards),
        "output_dir": str(output_dir),
        "prefix": prefix,
        "shards": manifest_rows,
    }
    manifest_path = output_dir / f"{prefix}.manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def command_merge(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    pattern = args.pattern or "*.csv"
    shard_paths = sorted(path for path in input_dir.glob(pattern) if path.is_file())
    if not shard_paths:
        raise ValueError(f"no shard CSV files matched pattern '{pattern}' in {input_dir}")

    frames: list[pd.DataFrame] = []
    for path in shard_paths:
        frames.append(pd.read_csv(path))

    merged = pd.concat(frames, ignore_index=True)
    if args.dedupe_key:
        if args.dedupe_key not in merged.columns:
            raise ValueError(f"dedupe key '{args.dedupe_key}' not found in merged columns")
        merged = merged.drop_duplicates(subset=[args.dedupe_key], keep="first")
    if args.sort_by:
        if args.sort_by not in merged.columns:
            raise ValueError(f"sort key '{args.sort_by}' not found in merged columns")
        merged = merged.sort_values(by=args.sort_by, kind="stable").reset_index(drop=True)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(
        json.dumps(
            {
                "rows": int(len(merged)),
                "input_dir": str(input_dir),
                "pattern": pattern,
                "matched_files": [str(path) for path in shard_paths],
                "output_csv": str(output_csv),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python scripts/shard_label_csv.py")
    subparsers = parser.add_subparsers(dest="command", required=True)

    split_parser = subparsers.add_parser("split")
    split_parser.add_argument("--input-csv", required=True)
    split_parser.add_argument("--output-dir", required=True)
    split_parser.add_argument("--num-shards", type=int, required=True)
    split_parser.add_argument("--prefix", default=None)
    split_parser.set_defaults(func=command_split)

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--input-dir", required=True)
    merge_parser.add_argument("--pattern", default="*.csv")
    merge_parser.add_argument("--output-csv", required=True)
    merge_parser.add_argument("--sort-by", default="sample_id")
    merge_parser.add_argument("--dedupe-key", default="sample_id")
    merge_parser.set_defaults(func=command_merge)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
