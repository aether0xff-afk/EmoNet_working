from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

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


def build_balanced_subset(
    df: pd.DataFrame,
    target_size: int,
    label_column: str = "label",
    seed: int = 42,
) -> pd.DataFrame:
    if target_size <= 0:
        raise ValueError("target_size must be positive")
    if len(df) <= target_size:
        return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if label_column not in df.columns:
        return df.sample(n=target_size, random_state=seed).reset_index(drop=True)

    rng = np.random.default_rng(seed)
    groups = {label: group.copy() for label, group in df.groupby(label_column, dropna=False)}
    label_keys = sorted(groups.keys(), key=lambda x: str(x))
    base_quota = max(1, target_size // max(1, len(label_keys)))

    selected_indices: list[int] = []
    used_indices: set[int] = set()

    for label in label_keys:
        group = groups[label]
        take = min(len(group), base_quota)
        if take <= 0:
            continue
        chosen = group.sample(n=take, random_state=seed)
        indices = chosen.index.tolist()
        selected_indices.extend(indices)
        used_indices.update(indices)

    remaining = target_size - len(selected_indices)
    if remaining > 0:
        leftovers = df.loc[~df.index.isin(list(used_indices))]
        if len(leftovers) > 0:
            take = min(remaining, len(leftovers))
            chosen = leftovers.sample(n=take, random_state=seed + 1)
            selected_indices.extend(chosen.index.tolist())
            used_indices.update(chosen.index.tolist())

    subset = df.loc[selected_indices].copy()
    if len(subset) > target_size:
        subset = subset.sample(n=target_size, random_state=seed)

    subset = subset.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    subset.insert(0, "sample_id", [f"s_{i:06d}" for i in range(len(subset))])
    return subset


def make_generation_prompt(record: dict[str, object]) -> str:
    text = str(record.get("text", "")).strip()
    z_values = [record[key] for key in sorted(record.keys()) if str(key).startswith("z_")]
    z_lines = "\n".join(f"{key}={float(record[key]):.6f}" for key in sorted(record.keys()) if str(key).startswith("z_"))
    return "\n".join(
        [
            "[TASK]",
            "주어진 대화 입력에 대해 어울리는 응답 스타일 벡터 s 와 예시 응답 1개를 생성하라.",
            "",
            "[INPUT_TEXT]",
            text,
            "",
            "[LATENT_Z]",
            z_lines if z_values else "(none)",
            "",
            "[OUTPUT_FORMAT]",
            "JSON only.",
            '{',
            '  "s": {"verbosity": 0.0, "sentence_length": 0.0, "...": 0.0},',
            '  "response": "string"',
            '}',
            "",
            "[CONSTRAINTS]",
            "- s 는 32축 모두 0~1 범위 실수로 채운다.",
            "- response 는 입력 내용과 정합적이어야 한다.",
            "- 과장하지 말고 자연스러운 한국어로 쓴다.",
            "- z 는 직접 설명하지 말고 내부 상태 힌트로만 사용한다.",
        ]
    )


def make_rating_prompt(record: dict[str, object], response: str) -> str:
    text = str(record.get("text", "")).strip()
    return "\n".join(
        [
            "[TASK]",
            "아래 입력과 응답을 보고 응답의 스타일 벡터 s_hat 를 32축 0~1 값으로 평가하라.",
            "",
            "[INPUT_TEXT]",
            text,
            "",
            "[RESPONSE]",
            response.strip(),
            "",
            "[OUTPUT_FORMAT]",
            "JSON only.",
            '{',
            '  "s_hat": {"verbosity": 0.0, "sentence_length": 0.0, "...": 0.0},',
            '  "notes": "short string"',
            '}',
            "",
            "[CONSTRAINTS]",
            "- 내용 적합성보다 응답의 말투와 표현 특성만 평가한다.",
            "- 각 축은 반드시 0~1 범위로 준다.",
            "- notes 는 한 문장으로 짧게 쓴다.",
        ]
    )


def command_build_llm_subset(args: argparse.Namespace) -> None:
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    prompt_jsonl = Path(args.prompt_jsonl) if args.prompt_jsonl else None
    df = pd.read_csv(input_csv)

    required = {"text", "talk_id"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"missing required columns: {', '.join(missing)}")

    subset = build_balanced_subset(
        df=df,
        target_size=args.target_size,
        label_column=args.label_column,
        seed=args.seed,
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(output_csv, index=False, encoding="utf-8-sig")

    if prompt_jsonl is not None:
        prompt_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with prompt_jsonl.open("w", encoding="utf-8") as handle:
            for row in subset.to_dict(orient="records"):
                payload = {
                    "sample_id": row["sample_id"],
                    "talk_id": row.get("talk_id", ""),
                    "generation_prompt": make_generation_prompt(row),
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    label_counts = subset[args.label_column].value_counts(dropna=False).to_dict() if args.label_column in subset.columns else {}
    print(
        json.dumps(
            {
                "rows": int(len(subset)),
                "output_csv": str(output_csv),
                "prompt_jsonl": str(prompt_jsonl) if prompt_jsonl else None,
                "label_counts": label_counts,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


STYLE_AXIS_NAMES = [
    "verbosity",
    "sentence_length",
    "pace",
    "fragmentation",
    "repetition",
    "rhythmicity",
    "directness",
    "explicitness",
    "specificity",
    "abstraction",
    "certainty",
    "logicality",
    "warmth",
    "distance",
    "politeness",
    "formality",
    "cooperativeness",
    "dominance",
    "calmness",
    "tension",
    "positivity",
    "heaviness",
    "urgency",
    "emotional_openness",
    "softness",
    "sharpness",
    "playfulness",
    "seriousness",
    "metaphoricity",
    "plainness",
    "initiative",
    "reflectiveness",
]


def extract_json_block(text: str) -> dict:
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("no JSON object found in model output")
    candidate = stripped[start : end + 1]
    return json.loads(candidate)


def normalize_style_dict(style_dict: dict, key_name: str) -> dict[str, float]:
    if key_name not in style_dict or not isinstance(style_dict[key_name], dict):
        raise ValueError(f"missing '{key_name}' object in model output")
    result: dict[str, float] = {}
    for axis in STYLE_AXIS_NAMES:
        value = float(style_dict[key_name].get(axis, 0.0))
        result[axis] = float(np.clip(value, 0.0, 1.0))
    return result


def call_openai_compatible_chat(
    base_url: str,
    model_name: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout_sec: int,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "Return JSON only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        body = json.loads(response.read().decode("utf-8"))
    choices = body.get("choices", [])
    if not choices:
        raise ValueError("no choices returned from local model server")
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if not isinstance(content, str):
        raise ValueError("invalid content returned from local model server")
    return content


def compute_consistency(style_a: dict[str, float], style_b: dict[str, float]) -> float:
    values = [abs(style_a[axis] - style_b[axis]) for axis in STYLE_AXIS_NAMES]
    return float(np.mean(values))


def label_subset_with_local_model(
    df: pd.DataFrame,
    output_csv: Path,
    base_url: str,
    model_name: str,
    generation_temperature: float,
    rating_temperature: float,
    max_tokens: int,
    timeout_sec: int,
    progress_every: int,
    limit: int | None,
) -> None:
    rows = []
    total = len(df) if limit is None else min(len(df), limit)
    start_time = time.perf_counter()

    for idx, record in enumerate(df.to_dict(orient="records"), start=1):
        if limit is not None and idx > limit:
            break

        generation_prompt = make_generation_prompt(record)
        generation_raw = call_openai_compatible_chat(
            base_url=base_url,
            model_name=model_name,
            prompt=generation_prompt,
            temperature=generation_temperature,
            max_tokens=max_tokens,
            timeout_sec=timeout_sec,
        )
        generation_obj = extract_json_block(generation_raw)
        style = normalize_style_dict(generation_obj, "s")
        response_text = str(generation_obj.get("response", "")).strip()

        rating_prompt = make_rating_prompt(record, response_text)
        rating_raw = call_openai_compatible_chat(
            base_url=base_url,
            model_name=model_name,
            prompt=rating_prompt,
            temperature=rating_temperature,
            max_tokens=max_tokens,
            timeout_sec=timeout_sec,
        )
        rating_obj = extract_json_block(rating_raw)
        style_hat = normalize_style_dict(rating_obj, "s_hat")
        consistency_l1 = compute_consistency(style, style_hat)

        row = dict(record)
        row["llm_response"] = response_text
        row["generation_raw_json"] = json.dumps(generation_obj, ensure_ascii=False)
        row["rating_raw_json"] = json.dumps(rating_obj, ensure_ascii=False)
        row["consistency_l1"] = consistency_l1
        row["keep_sample"] = bool(consistency_l1 <= 0.12)
        for axis_idx, axis in enumerate(STYLE_AXIS_NAMES):
            row[f"s_{axis_idx}"] = style[axis]
            row[f"s_hat_{axis_idx}"] = style_hat[axis]
        rows.append(row)

        if progress_every > 0 and idx % progress_every == 0:
            elapsed = max(1e-8, time.perf_counter() - start_time)
            print(f"processed {idx}/{total} rows ({idx / elapsed:.2f} rows/s)")

    result_df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    elapsed = time.perf_counter() - start_time
    print(
        json.dumps(
            {
                "rows": int(len(result_df)),
                "kept_rows": int(result_df["keep_sample"].sum()) if len(result_df) else 0,
                "output_csv": str(output_csv),
                "elapsed_sec": round(elapsed, 3),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def command_label_local(args: argparse.Namespace) -> None:
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    df = pd.read_csv(input_csv)
    if "text" not in df.columns:
        raise ValueError("input subset CSV must contain a 'text' column")
    label_subset_with_local_model(
        df=df,
        output_csv=output_csv,
        base_url=args.base_url,
        model_name=args.model_name,
        generation_temperature=args.generation_temperature,
        rating_temperature=args.rating_temperature,
        max_tokens=args.max_tokens,
        timeout_sec=args.timeout_sec,
        progress_every=args.progress_every,
        limit=args.limit,
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(output_csv, index=False, encoding="utf-8-sig")

    if prompt_jsonl is not None:
        prompt_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with prompt_jsonl.open("w", encoding="utf-8") as handle:
            for row in subset.to_dict(orient="records"):
                payload = {
                    "sample_id": row["sample_id"],
                    "talk_id": row.get("talk_id", ""),
                    "generation_prompt": make_generation_prompt(row),
                }
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    label_counts = subset[args.label_column].value_counts(dropna=False).to_dict() if args.label_column in subset.columns else {}
    print(
        json.dumps(
            {
                "rows": int(len(subset)),
                "output_csv": str(output_csv),
                "prompt_jsonl": str(prompt_jsonl) if prompt_jsonl else None,
                "label_counts": label_counts,
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

    subset_parser = subparsers.add_parser("build-llm-subset")
    subset_parser.add_argument("--input-csv", required=True)
    subset_parser.add_argument("--output-csv", required=True)
    subset_parser.add_argument("--prompt-jsonl", default=None)
    subset_parser.add_argument("--target-size", type=int, default=2000)
    subset_parser.add_argument("--label-column", default="label")
    subset_parser.add_argument("--seed", type=int, default=42)
    subset_parser.set_defaults(func=command_build_llm_subset)

    local_parser = subparsers.add_parser("label-local")
    local_parser.add_argument("--input-csv", required=True)
    local_parser.add_argument("--output-csv", required=True)
    local_parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    local_parser.add_argument("--model-name", default="gpt-oss-20b")
    local_parser.add_argument("--generation-temperature", type=float, default=0.7)
    local_parser.add_argument("--rating-temperature", type=float, default=0.1)
    local_parser.add_argument("--max-tokens", type=int, default=1200)
    local_parser.add_argument("--timeout-sec", type=int, default=180)
    local_parser.add_argument("--progress-every", type=int, default=10)
    local_parser.add_argument("--limit", type=int, default=None)
    local_parser.set_defaults(func=command_label_local)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
