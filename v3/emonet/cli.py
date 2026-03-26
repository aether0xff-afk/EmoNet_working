from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .core import EmoNet, EmoNetConfig, StimEncoderConfig


def build_stim_config(args: argparse.Namespace) -> StimEncoderConfig:
    kwargs = {}
    if args.dataset_csv is not None:
        kwargs["dataset_csv"] = Path(args.dataset_csv)
    if args.benchmark_csv is not None:
        kwargs["benchmark_csv"] = Path(args.benchmark_csv)
    if args.model_cache_path is not None:
        kwargs["model_cache_path"] = Path(args.model_cache_path)
    if args.max_samples is not None:
        kwargs["max_samples"] = args.max_samples
    if args.force_refit:
        kwargs["force_refit"] = True
    return StimEncoderConfig(**kwargs)


def build_model(args: argparse.Namespace) -> EmoNet:
    config = EmoNetConfig(seed=args.seed, z_dim=args.z_dim, z_encoder_mode="stat")
    stim_config = build_stim_config(args)
    return EmoNet(config=config, stim_encoder_config=stim_config)


def command_fit_stim(args: argparse.Namespace) -> None:
    model = build_model(args)
    model.stim_encoder.fit()
    print(json.dumps({"model_cache_path": str(model.stim_encoder.config.model_cache_path)}, ensure_ascii=False, indent=2))


def command_infer(args: argparse.Namespace) -> None:
    model = build_model(args)
    outputs = model.forward(args.text)
    result = {
        "stim_vec": np.asarray(outputs["stim_vec"], dtype=float).tolist(),
        "dominant_branch_len": len(outputs["dominant_branch"]),
        "z": np.asarray(outputs["z"], dtype=float).tolist(),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


def flatten_dialogue_text(content: dict) -> str:
    ordered_keys = ["HS01", "SS01", "HS02", "SS02", "HS03", "SS03"]
    parts = [str(content.get(key, "")).strip() for key in ordered_keys]
    parts = [part for part in parts if part]
    return " [SEP] ".join(parts)


def load_training_json_as_dataframe(input_json: Path) -> pd.DataFrame:
    with input_json.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    rows = []
    for item in data:
        profile = item.get("profile", {})
        emotion = profile.get("emotion", {})
        talk = item.get("talk", {})
        talk_id = talk.get("id", {})
        content = talk.get("content", {})
        rows.append(
            {
                "text": flatten_dialogue_text(content),
                "label": emotion.get("type", ""),
                "persona_id": profile.get("persona-id", ""),
                "talk_id": talk_id.get("talk-id", ""),
                "profile_id": talk_id.get("profile-id", ""),
            }
        )
    return pd.DataFrame(rows)


def load_training_json_records(input_json: Path):
    with input_json.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    for item in data:
        profile = item.get("profile", {})
        emotion = profile.get("emotion", {})
        talk = item.get("talk", {})
        talk_id = talk.get("id", {})
        content = talk.get("content", {})
        yield {
            "text": flatten_dialogue_text(content),
            "label": emotion.get("type", ""),
            "persona_id": profile.get("persona-id", ""),
            "talk_id": talk_id.get("talk-id", ""),
            "profile_id": talk_id.get("profile-id", ""),
        }


def resolve_text_column(df: pd.DataFrame, requested: str | None) -> str:
    if requested and requested in df.columns:
        return requested

    candidates = ["text", "content", "sentence", "utterance", "dialogue"]
    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    available = ", ".join(map(str, df.columns.tolist()))
    raise ValueError(f"text column not found. available columns: {available}")


def export_z_from_dataframe(model: EmoNet, df: pd.DataFrame, text_column: str, output_csv: Path) -> None:
    z_rows = []
    stim_rows = []
    for idx, text in enumerate(df[text_column].astype(str), start=1):
        outputs = model.forward(text)
        z_rows.append(np.asarray(outputs["z"], dtype=np.float32))
        stim_rows.append(np.asarray(outputs["stim_vec"], dtype=np.float32))
        if idx % 100 == 0:
            print(f"processed {idx} rows")

    z_array = np.vstack(z_rows)
    stim_array = np.vstack(stim_rows)
    for dim in range(z_array.shape[1]):
        df[f"z_{dim}"] = z_array[:, dim]
    for dim, name in enumerate(("dopamine", "serotonin", "norepinephrine", "melatonin")):
        df[name] = stim_array[:, dim]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(json.dumps({"rows": int(len(df)), "output_csv": str(output_csv)}, ensure_ascii=False, indent=2))


def build_output_row(source_row: dict, outputs: dict[str, object]) -> dict[str, object]:
    row = dict(source_row)
    z = np.asarray(outputs["z"], dtype=np.float32).reshape(-1)
    stim = np.asarray(outputs["stim_vec"], dtype=np.float32).reshape(-1)
    for dim, value in enumerate(z):
        row[f"z_{dim}"] = float(value)
    for dim, name in enumerate(("dopamine", "serotonin", "norepinephrine", "melatonin")):
        row[name] = float(stim[dim])
    row["dominant_branch_len"] = int(len(outputs["dominant_branch"]))
    return row


def flush_rows(rows: list[dict[str, object]], output_csv: Path, write_header: bool) -> bool:
    if not rows:
        return write_header
    chunk_df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if write_header else "a"
    chunk_df.to_csv(output_csv, mode=mode, index=False, encoding="utf-8-sig", header=write_header)
    return False


def load_existing_ids(output_csv: Path) -> set[str]:
    if not output_csv.exists():
        return set()
    existing = pd.read_csv(output_csv, usecols=["talk_id"]) if output_csv.stat().st_size > 0 else pd.DataFrame()
    if "talk_id" not in existing.columns:
        return set()
    return {str(value) for value in existing["talk_id"].dropna().astype(str)}


def export_z_from_json_stream(
    model: EmoNet,
    input_json: Path,
    output_csv: Path,
    limit: int | None,
    chunk_size: int,
    progress_every: int,
    resume: bool,
) -> None:
    rows_to_write: list[dict[str, object]] = []
    processed = 0
    written = 0
    skipped = 0
    write_header = not output_csv.exists() or not resume
    existing_ids = load_existing_ids(output_csv) if resume else set()
    start_time = time.perf_counter()

    if resume and existing_ids:
        print(f"resume mode: skipping {len(existing_ids)} existing talk_id rows")

    for source_row in load_training_json_records(input_json):
        talk_id = str(source_row.get("talk_id", ""))
        if existing_ids and talk_id and talk_id in existing_ids:
            skipped += 1
            continue

        outputs = model.forward(str(source_row["text"]))
        rows_to_write.append(build_output_row(source_row, outputs))
        processed += 1
        written += 1

        if progress_every > 0 and processed % progress_every == 0:
            elapsed = max(1e-8, time.perf_counter() - start_time)
            print(f"processed {processed} rows ({processed / elapsed:.2f} rows/s)")

        if len(rows_to_write) >= chunk_size:
            write_header = flush_rows(rows_to_write, output_csv, write_header)
            rows_to_write.clear()

        if limit is not None and processed >= limit:
            break

    write_header = flush_rows(rows_to_write, output_csv, write_header)
    elapsed = time.perf_counter() - start_time
    print(
        json.dumps(
            {
                "processed": processed,
                "written": written,
                "skipped": skipped,
                "output_csv": str(output_csv),
                "elapsed_sec": round(elapsed, 3),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def command_export_z(args: argparse.Namespace) -> None:
    model = build_model(args)
    output_csv = Path(args.output_csv)
    text_column = args.text_column

    if bool(args.input_csv) == bool(args.input_json):
        raise ValueError("provide exactly one of --input-csv or --input-json")

    if args.input_json is not None:
        input_json = Path(args.input_json)
        export_z_from_json_stream(
            model=model,
            input_json=input_json,
            output_csv=output_csv,
            limit=args.limit,
            chunk_size=args.chunk_size,
            progress_every=args.progress_every,
            resume=args.resume,
        )
    else:
        input_csv = Path(args.input_csv)
        df = pd.read_csv(input_csv)
        text_column = resolve_text_column(df, text_column)
        if args.limit is not None and args.limit > 0:
            df = df.head(args.limit).copy()
        export_z_from_dataframe(model, df, text_column, output_csv)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m emonet.cli")
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_options(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--dataset-csv", dest="dataset_csv", type=str, default=None)
        subparser.add_argument("--benchmark-csv", dest="benchmark_csv", type=str, default=None)
        subparser.add_argument("--model-cache-path", dest="model_cache_path", type=str, default=None)
        subparser.add_argument("--max-samples", dest="max_samples", type=int, default=None)
        subparser.add_argument("--force-refit", action="store_true")
        subparser.add_argument("--seed", type=int, default=42)
        subparser.add_argument("--z-dim", dest="z_dim", type=int, default=64)

    fit_parser = subparsers.add_parser("fit-stim")
    add_common_options(fit_parser)
    fit_parser.set_defaults(func=command_fit_stim)

    infer_parser = subparsers.add_parser("infer")
    add_common_options(infer_parser)
    infer_parser.add_argument("--text", required=True)
    infer_parser.set_defaults(func=command_infer)

    export_parser = subparsers.add_parser("export-z")
    add_common_options(export_parser)
    export_parser.add_argument("--input-csv", default=None)
    export_parser.add_argument("--input-json", default=None)
    export_parser.add_argument("--text-column", default="text")
    export_parser.add_argument("--output-csv", required=True)
    export_parser.add_argument("--limit", type=int, default=None)
    export_parser.add_argument("--chunk-size", type=int, default=256)
    export_parser.add_argument("--progress-every", type=int, default=100)
    export_parser.add_argument("--resume", action="store_true")
    export_parser.set_defaults(func=command_export_z)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
