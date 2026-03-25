from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def command_export_z(args: argparse.Namespace) -> None:
    model = build_model(args)
    output_csv = Path(args.output_csv)
    text_column = args.text_column

    if bool(args.input_csv) == bool(args.input_json):
        raise ValueError("provide exactly one of --input-csv or --input-json")

    if args.input_json is not None:
        input_json = Path(args.input_json)
        df = load_training_json_as_dataframe(input_json)
        text_column = "text"
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
    export_parser.set_defaults(func=command_export_z)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
