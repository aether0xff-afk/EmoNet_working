from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Callable
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

from .core import EmoNet, EmoNetConfig, LinearZtoSDecoder, StimEncoderConfig, ZSDecoderConfig


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
    if args.zs_model_path:
        decoder = LinearZtoSDecoder.load(Path(args.zs_model_path))
        result["s_pred"] = np.asarray(decoder.predict(np.asarray(outputs["z"], dtype=np.float32)), dtype=float).tolist()
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


def resolve_indexed_columns(df: pd.DataFrame, prefix: str, expected_dim: int | None = None) -> list[str]:
    columns = {str(column) for column in df.columns}
    if expected_dim is not None:
        expected = [f"{prefix}{idx}" for idx in range(expected_dim)]
        missing = [column for column in expected if column not in columns]
        if missing:
            raise ValueError(f"missing required columns: {', '.join(missing)}")
        return expected

    indexed: list[tuple[int, str]] = []
    for column in columns:
        if not column.startswith(prefix):
            continue
        suffix = column[len(prefix) :]
        if suffix.isdigit():
            indexed.append((int(suffix), column))
    if not indexed:
        raise ValueError(f"no indexed columns found with prefix '{prefix}'")
    indexed.sort(key=lambda item: item[0])
    return [column for _, column in indexed]


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


def train_zs_decoder_from_dataframe(
    df: pd.DataFrame,
    model_path: Path,
    z_dim: int,
    s_dim: int | None,
    ridge_alpha: float,
    seed: int,
    val_ratio: float,
    use_all_rows: bool,
) -> dict[str, object]:
    original_rows = len(df)
    keep_filtered_rows = 0
    if not use_all_rows and "keep_sample" in df.columns:
        keep_mask = df["keep_sample"].fillna(False).astype(bool)
        keep_filtered_rows = int((~keep_mask).sum())
        df = df.loc[keep_mask].copy()

    z_columns = resolve_indexed_columns(df, "z_", expected_dim=z_dim)
    s_columns = resolve_indexed_columns(df, "s_", expected_dim=s_dim)
    inferred_s_dim = len(s_columns)

    before_dropna = len(df)
    df = df.dropna(subset=z_columns + s_columns).reset_index(drop=True)
    dropped_missing_rows = before_dropna - len(df)
    if len(df) < 2:
        raise ValueError("at least 2 clean labeled rows are required to fit z->s regressor")

    z_matrix = df[z_columns].to_numpy(dtype=np.float32)
    s_matrix = df[s_columns].to_numpy(dtype=np.float32)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(df))

    val_rows = 0
    train_rows = len(df)
    train_mae = None
    val_mae = None
    if 0.0 < val_ratio < 1.0 and len(df) >= 5:
        tentative_val = int(round(len(df) * val_ratio))
        val_rows = min(max(1, tentative_val), len(df) - 2)
        train_rows = len(df) - val_rows
        val_idx = indices[:val_rows]
        train_idx = indices[val_rows:]
        eval_decoder = LinearZtoSDecoder(
            config=ZSDecoderConfig(model_path=model_path, ridge_alpha=ridge_alpha),
            z_dim=z_dim,
            s_dim=inferred_s_dim,
        )
        eval_decoder.fit(z_matrix[train_idx], s_matrix[train_idx])
        train_mae = eval_decoder.mean_absolute_error(z_matrix[train_idx], s_matrix[train_idx])
        val_mae = eval_decoder.mean_absolute_error(z_matrix[val_idx], s_matrix[val_idx])

    decoder = LinearZtoSDecoder(
        config=ZSDecoderConfig(model_path=model_path, ridge_alpha=ridge_alpha),
        z_dim=z_dim,
        s_dim=inferred_s_dim,
    )
    decoder.fit(z_matrix, s_matrix)
    saved_path = decoder.save(model_path)
    return {
        "input_rows": int(original_rows),
        "rows_after_keep_filter": int(original_rows - keep_filtered_rows),
        "rows_used": int(len(df)),
        "keep_filtered_rows": int(keep_filtered_rows),
        "dropped_missing_rows": int(dropped_missing_rows),
        "train_rows": int(train_rows),
        "val_rows": int(val_rows),
        "train_mae": None if train_mae is None else round(float(train_mae), 6),
        "val_mae": None if val_mae is None else round(float(val_mae), 6),
        "z_dim": int(z_dim),
        "s_dim": int(inferred_s_dim),
        "model_path": str(saved_path),
    }


def command_fit_zs_regressor(args: argparse.Namespace) -> None:
    input_csv = Path(args.input_csv)
    model_path = Path(args.model_path)
    df = pd.read_csv(input_csv)
    summary = train_zs_decoder_from_dataframe(
        df=df,
        model_path=model_path,
        z_dim=args.z_dim,
        s_dim=args.s_dim,
        ridge_alpha=args.ridge_alpha,
        seed=args.seed,
        val_ratio=args.val_ratio,
        use_all_rows=args.use_all_rows,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def command_predict_s(args: argparse.Namespace) -> None:
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    df = pd.read_csv(input_csv)
    z_columns = resolve_indexed_columns(df, "z_", expected_dim=args.z_dim)
    decoder = LinearZtoSDecoder.load(Path(args.model_path))
    predictions = decoder.predict(df[z_columns].to_numpy(dtype=np.float32))
    pred_df = pd.DataFrame(
        {f"{args.output_prefix}{axis_idx}": predictions[:, axis_idx] for axis_idx in range(predictions.shape[1])}
    )
    df = pd.concat([df.reset_index(drop=True), pred_df], axis=1)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(
        json.dumps(
            {
                "rows": int(len(df)),
                "output_csv": str(output_csv),
                "model_path": str(args.model_path),
                "output_prefix": args.output_prefix,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def command_generate_response(args: argparse.Namespace) -> None:
    ensure_model_server_ready(args.base_url, args.timeout_sec)
    model = build_model(args)
    decoder = LinearZtoSDecoder.load(Path(args.zs_model_path))
    profile = infer_style_profile(model=model, decoder=decoder, text=args.text)
    response_text, style_prompt = generate_response_from_style(
        base_url=args.base_url,
        model_name=args.model_name,
        input_text=args.text,
        style_dict=profile["style_dict"],
        style_tags=profile["style_tags"],
        style_summary=profile["style_summary"],
        temperature=args.response_temperature,
        max_tokens=args.max_tokens,
        timeout_sec=args.timeout_sec,
        template_path=Path(args.prompt_template) if args.prompt_template else None,
    )
    result = {
        "input_text": args.text,
        "stim_vec": np.asarray(profile["stim_vec"], dtype=float).tolist(),
        "dominant_branch_len": int(profile["dominant_branch_len"]),
        "z": np.asarray(profile["z"], dtype=float).tolist(),
        "s_pred": np.asarray(profile["s_pred"], dtype=float).tolist(),
        "style_tags": list(profile["style_tags"]),
        "style_summary": dict(profile["style_summary"]),
        "style_summary_text": str(profile["style_summary_text"]),
        "style_prompt": style_prompt,
        "llm_response": response_text,
        "decoder_model_path": str(args.zs_model_path),
        "llm_model_name": args.model_name,
        "timestamp_utc": utc_timestamp(),
    }
    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.log_jsonl:
        append_jsonl(Path(args.log_jsonl), [serialize_generation_log(result)])
    print(json.dumps(result, ensure_ascii=False, indent=2))


def command_generate_response_batch(args: argparse.Namespace) -> None:
    ensure_model_server_ready(args.base_url, args.timeout_sec)
    model = build_model(args)
    decoder = LinearZtoSDecoder.load(Path(args.zs_model_path))
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    df = pd.read_csv(input_csv)
    text_column = resolve_text_column(df, args.text_column)
    if args.limit is not None and args.limit > 0:
        df = df.head(args.limit).copy()

    rows: list[dict[str, object]] = []
    jsonl_rows: list[dict[str, object]] = []
    start_time = time.perf_counter()
    for idx, record in enumerate(df.to_dict(orient="records"), start=1):
        text = str(record.get(text_column, "")).strip()
        if not text:
            row = dict(record)
            row["status"] = "error"
            row["error_message"] = f"empty text column '{text_column}'"
            rows.append(row)
            continue

        try:
            profile = infer_style_profile(model=model, decoder=decoder, text=text)
            response_text, style_prompt = generate_response_from_style(
                base_url=args.base_url,
                model_name=args.model_name,
                input_text=text,
                style_dict=profile["style_dict"],
                style_tags=profile["style_tags"],
                style_summary=profile["style_summary"],
                temperature=args.response_temperature,
                max_tokens=args.max_tokens,
                timeout_sec=args.timeout_sec,
                template_path=Path(args.prompt_template) if args.prompt_template else None,
            )
            row = dict(record)
            row["status"] = "ok"
            row["error_message"] = ""
            row["style_tags"] = json.dumps(profile["style_tags"], ensure_ascii=False)
            row["style_summary_text"] = str(profile["style_summary_text"])
            row["style_summary_json"] = json.dumps(profile["style_summary"], ensure_ascii=False)
            row["style_prompt"] = style_prompt
            row["llm_response"] = response_text
            row["decoder_model_path"] = str(args.zs_model_path)
            row["llm_model_name"] = args.model_name
            row["timestamp_utc"] = utc_timestamp()
            for axis_idx, value in enumerate(np.asarray(profile["s_pred"], dtype=np.float32).reshape(-1)):
                row[f"s_pred_{axis_idx}"] = float(value)
            for macro_name, score in dict(profile["style_summary"]).items():
                row[f"macro_{macro_name}"] = float(score)
            rows.append(row)
            jsonl_rows.append(
                serialize_generation_log(
                    {
                    "input_text": text,
                    "talk_id": record.get("talk_id", ""),
                    "stim_vec": np.asarray(profile["stim_vec"], dtype=float).tolist(),
                    "z": np.asarray(profile["z"], dtype=float).tolist(),
                    "s_pred": np.asarray(profile["s_pred"], dtype=float).tolist(),
                    "style_tags": list(profile["style_tags"]),
                    "style_summary": dict(profile["style_summary"]),
                    "style_prompt": style_prompt,
                    "llm_response": response_text,
                    "decoder_model_path": str(args.zs_model_path),
                    "llm_model_name": args.model_name,
                    "timestamp_utc": row["timestamp_utc"],
                    }
                )
            )
        except Exception as exc:
            row = dict(record)
            row["status"] = "error"
            row["error_message"] = str(exc)
            rows.append(row)

        if args.progress_every > 0 and idx % args.progress_every == 0:
            elapsed = max(1e-8, time.perf_counter() - start_time)
            print(f"processed {idx}/{len(df)} rows ({idx / elapsed:.2f} rows/s)")

    result_df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    if args.log_jsonl:
        append_jsonl(Path(args.log_jsonl), jsonl_rows)
    print(
        json.dumps(
            {
                "rows": int(len(result_df)),
                "ok_rows": int((result_df.get("status") == "ok").sum()) if len(result_df) else 0,
                "output_csv": str(output_csv),
                "log_jsonl": args.log_jsonl,
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

STYLE_AXIS_DESCRIPTIONS = {
    "verbosity": "짧고 절제됨 <-> 길고 많이 말함",
    "sentence_length": "짧은 문장 위주 <-> 긴 문장 위주",
    "pace": "느리고 신중함 <-> 빠르고 몰아침",
    "fragmentation": "완결된 문장 <-> 끊긴 조각 문장",
    "repetition": "반복 거의 없음 <-> 표현 반복 많음",
    "rhythmicity": "리듬감 약함 <-> 리듬감 뚜렷함",
    "directness": "에둘러 말함 <-> 직접적으로 말함",
    "explicitness": "암시적 <-> 명시적",
    "specificity": "두루뭉술함 <-> 구체적",
    "abstraction": "구체적/현실적 <-> 추상적/개념적",
    "certainty": "조심스럽고 유보적 <-> 단정적이고 확신함",
    "logicality": "연상적/감각적 <-> 논리적/정리됨",
    "warmth": "차갑고 건조함 <-> 따뜻하고 배려함",
    "distance": "가깝고 친밀함 <-> 거리감 있고 분리됨",
    "politeness": "무뚝뚝함 <-> 공손함",
    "formality": "구어체/일상체 <-> 문어체/격식체",
    "cooperativeness": "비협조적 <-> 협조적",
    "dominance": "유순함 <-> 주도적/통제적",
    "calmness": "동요됨 <-> 차분함",
    "tension": "느슨함 <-> 긴장감 높음",
    "positivity": "부정적 <-> 긍정적",
    "heaviness": "가벼움 <-> 무거움",
    "urgency": "여유로움 <-> 급박함",
    "emotional_openness": "감정 노출 적음 <-> 감정 노출 큼",
    "softness": "딱딱함 <-> 부드러움",
    "sharpness": "둔하고 완만함 <-> 날카롭고 예리함",
    "playfulness": "장난기 없음 <-> 장난기 많음",
    "seriousness": "가벼움 <-> 진지함",
    "metaphoricity": "직설적 표현 <-> 비유적 표현",
    "plainness": "꾸밈 많음 <-> 평이하고 담백함",
    "initiative": "수동적 <-> 먼저 이끔",
    "reflectiveness": "즉흥적 <-> 성찰적",
}

STYLE_SCORE_LEVELS = np.asarray([0.0, 0.25, 0.5, 0.75, 1.0], dtype=np.float32)

STYLE_TAG_LABELS = {
    "verbosity": ("간결함", "장문형"),
    "sentence_length": ("짧은문장", "긴문장"),
    "pace": ("느린호흡", "빠른전개"),
    "fragmentation": ("정돈된문장", "파편적문장"),
    "repetition": ("반복적음", "반복강함"),
    "rhythmicity": ("리듬약함", "리듬강함"),
    "directness": ("완곡함", "직설적"),
    "explicitness": ("암시적", "명시적"),
    "specificity": ("포괄적", "구체적"),
    "abstraction": ("현실적", "추상적"),
    "certainty": ("유보적", "확신형"),
    "logicality": ("감각중심", "논리적"),
    "warmth": ("건조함", "따뜻함"),
    "distance": ("친밀함", "거리감"),
    "politeness": ("무뚝뚝함", "공손함"),
    "formality": ("구어체", "격식체"),
    "cooperativeness": ("단독지향", "협조적"),
    "dominance": ("유순함", "주도적"),
    "calmness": ("동요됨", "차분함"),
    "tension": ("이완됨", "긴장높음"),
    "positivity": ("부정적", "긍정적"),
    "heaviness": ("가벼움", "무게감"),
    "urgency": ("여유있음", "긴급함"),
    "emotional_openness": ("감정절제", "감정개방"),
    "softness": ("단단함", "부드러움"),
    "sharpness": ("완만함", "날카로움"),
    "playfulness": ("진중함", "장난기"),
    "seriousness": ("가벼움", "진지함"),
    "metaphoricity": ("직설표현", "비유표현"),
    "plainness": ("꾸밈있음", "담백함"),
    "initiative": ("수동적", "주도적"),
    "reflectiveness": ("즉흥적", "성찰적"),
}

STYLE_MACRO_AXES = {
    "energy": [("pace", 1.0), ("urgency", 0.9), ("initiative", 0.8), ("verbosity", 0.5)],
    "tension": [("tension", 1.0), ("urgency", 0.8), ("calmness", -0.9), ("heaviness", 0.4)],
    "warmth": [("warmth", 1.0), ("softness", 0.8), ("cooperativeness", 0.7), ("positivity", 0.6)],
    "directness": [("directness", 1.0), ("explicitness", 0.9), ("sharpness", 0.6), ("certainty", 0.5)],
    "formality": [("formality", 1.0), ("politeness", 0.8), ("distance", 0.6), ("plainness", 0.4)],
    "emotional_openness": [("emotional_openness", 1.0), ("reflectiveness", 0.7), ("warmth", 0.5)],
    "seriousness": [("seriousness", 1.0), ("heaviness", 0.8), ("playfulness", -0.8)],
    "structure": [("logicality", 1.0), ("specificity", 0.8), ("fragmentation", -0.7), ("sentence_length", 0.3)],
}

STYLE_MACRO_LABELS = {
    "energy": "에너지",
    "tension": "긴장",
    "warmth": "따뜻함",
    "directness": "직설성",
    "formality": "형식성",
    "emotional_openness": "감정개방성",
    "seriousness": "무게감",
    "structure": "구조화",
}


def resolve_style_axes(style_dim: int | None = None) -> list[str]:
    if style_dim is None:
        return list(STYLE_AXIS_NAMES)
    if style_dim <= 0:
        raise ValueError("style_dim must be positive")
    if style_dim > len(STYLE_AXIS_NAMES):
        raise ValueError(f"style_dim must be <= {len(STYLE_AXIS_NAMES)}")
    return list(STYLE_AXIS_NAMES[:style_dim])


def build_style_blocks(block_size: int, style_axes: list[str]) -> list[list[str]]:
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    return [style_axes[idx : idx + block_size] for idx in range(0, len(style_axes), block_size)]


def format_style_axes(block_axes: list[str], active_axes: list[str]) -> str:
    lines = []
    for axis in block_axes:
        axis_idx = active_axes.index(axis) + 1
        description = STYLE_AXIS_DESCRIPTIONS.get(axis, "")
        suffix = f" ({description})" if description else ""
        lines.append(f"{axis_idx}. {axis}{suffix}")
    return "\n".join(lines)


def format_score_levels() -> str:
    return ", ".join(f"{float(value):.2f}" for value in STYLE_SCORE_LEVELS)


def quantize_style_value(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    idx = int(np.argmin(np.abs(STYLE_SCORE_LEVELS - value)))
    return float(STYLE_SCORE_LEVELS[idx])


def style_vector_to_dict(values: np.ndarray | list[float], axis_names: list[str] | None = None) -> dict[str, float]:
    axes = resolve_style_axes(len(values) if axis_names is None else len(axis_names))
    if axis_names is not None:
        axes = axis_names
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if len(arr) != len(axes):
        raise ValueError(f"style vector length {len(arr)} does not match axis count {len(axes)}")
    return {axis: float(np.clip(arr[idx], 0.0, 1.0)) for idx, axis in enumerate(axes)}


def compute_macro_style_scores(style_dict: dict[str, float]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for macro_name, terms in STYLE_MACRO_AXES.items():
        numerator = 0.0
        denom = 0.0
        for axis_name, weight in terms:
            if axis_name not in style_dict:
                continue
            axis_value = float(style_dict[axis_name])
            value = axis_value if weight >= 0.0 else 1.0 - axis_value
            numerator += abs(weight) * value
            denom += abs(weight)
        scores[macro_name] = float(np.clip(numerator / max(denom, 1e-8), 0.0, 1.0))
    return scores


def describe_macro_level(score: float) -> str:
    if score >= 0.75:
        return "매우 높음"
    if score >= 0.60:
        return "높음"
    if score <= 0.25:
        return "매우 낮음"
    if score <= 0.40:
        return "낮음"
    return "중간"


def build_style_tags(style_dict: dict[str, float], max_tags: int = 8) -> list[str]:
    scored_tags: list[tuple[float, str]] = []
    for axis_name, value in style_dict.items():
        labels = STYLE_TAG_LABELS.get(axis_name)
        if labels is None:
            continue
        intensity = abs(float(value) - 0.5)
        if intensity < 0.18:
            continue
        label = labels[1] if value >= 0.5 else labels[0]
        scored_tags.append((intensity, label))
    scored_tags.sort(key=lambda item: item[0], reverse=True)
    return [label for _, label in scored_tags[:max_tags]]


def build_style_summary(style_dict: dict[str, float]) -> dict[str, float]:
    return compute_macro_style_scores(style_dict)


def summarize_style_summary(style_summary: dict[str, float], top_n: int = 4) -> str:
    ranked = sorted(style_summary.items(), key=lambda item: abs(item[1] - 0.5), reverse=True)
    parts = []
    for macro_name, score in ranked[:top_n]:
        label = STYLE_MACRO_LABELS.get(macro_name, macro_name)
        parts.append(f"{label} {describe_macro_level(score)}")
    return ", ".join(parts)


def format_style_summary_lines(style_summary: dict[str, float]) -> list[str]:
    return [
        f"{STYLE_MACRO_LABELS.get(name, name)}={float(score):.4f} ({describe_macro_level(score)})"
        for name, score in style_summary.items()
    ]


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def request_json_response(
    base_url: str,
    model_name: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout_sec: int,
    max_retries: int,
    validator: Callable[[dict], object] | None = None,
    retry_instruction: str | None = None,
) -> tuple[object, str]:
    last_raw = ""
    last_error = ""
    for attempt in range(max_retries + 1):
        retry_suffix = ""
        if attempt > 0:
            retry_suffix = (
                "\n\n[RETRY_INSTRUCTION]\n"
                + (
                    retry_instruction
                    or "직전 응답은 JSON 형식이 아니었다. 설명 없이 JSON object 하나만 다시 출력하라."
                )
            )
        raw = call_openai_compatible_chat(
            base_url=base_url,
            model_name=model_name,
            prompt=prompt + retry_suffix,
            temperature=temperature if attempt == 0 else 0.0,
            max_tokens=max_tokens,
            timeout_sec=timeout_sec,
        )
        last_raw = raw
        try:
            payload = extract_json_block(raw)
            if not isinstance(payload, dict):
                raise ValueError("model output must be a JSON object")
            if validator is not None:
                return validator(payload), raw
            return payload, raw
        except Exception as exc:
            last_error = str(exc)
            continue
    raise ValueError(f"no JSON object found in model output after retries: {last_error}. raw={last_raw[:500]}")


def normalize_style_dict(style_dict: dict, key_name: str, expected_axes: list[str] | None = None) -> dict[str, float]:
    if key_name not in style_dict or not isinstance(style_dict[key_name], dict):
        raise ValueError(f"missing '{key_name}' object in model output")
    axes = resolve_style_axes() if expected_axes is None else expected_axes
    style_payload = style_dict[key_name]
    missing_axes = [axis for axis in axes if axis not in style_payload]
    extra_axes = sorted(str(axis) for axis in style_payload.keys() if axis not in axes)
    if missing_axes or extra_axes:
        problems = []
        if missing_axes:
            problems.append(f"missing axes: {', '.join(missing_axes)}")
        if extra_axes:
            problems.append(f"unexpected axes: {', '.join(extra_axes)}")
        raise ValueError(f"invalid '{key_name}' keys ({'; '.join(problems)})")
    result: dict[str, float] = {}
    for axis in axes:
        value = style_payload[axis]
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"axis '{axis}' must be numeric") from exc
        result[axis] = quantize_style_value(value)
    return result


def normalize_response_text(payload: dict) -> str:
    response = payload.get("response", "")
    if not isinstance(response, str):
        raise ValueError("'response' must be a string")
    response = response.strip()
    if not response:
        raise ValueError("empty 'response' returned from model output")
    return response


def call_openai_compatible_chat(
    base_url: str,
    model_name: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout_sec: int,
    system_prompt: str = "Return JSON only.",
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
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


def ensure_model_server_ready(base_url: str, timeout_sec: int) -> None:
    models_url = base_url.rstrip("/") + "/models"
    request = urllib.request.Request(models_url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=min(timeout_sec, 10)) as response:
            if response.status >= 500:
                raise ValueError(f"model server returned HTTP {response.status} for {models_url}")
    except urllib.error.HTTPError as exc:
        if exc.code not in (404, 405):
            raise ConnectionError(f"model server check failed: HTTP {exc.code} for {models_url}") from exc
    except urllib.error.URLError as exc:
        raise ConnectionError(f"model server is not reachable at {base_url}: {exc}") from exc


def compute_consistency(style_a: dict[str, float], style_b: dict[str, float], active_axes: list[str]) -> float:
    values = [abs(style_a[axis] - style_b[axis]) for axis in active_axes]
    return float(np.mean(values))


def make_generation_prompt(record: dict[str, object]) -> str:
    text = str(record.get("text", "")).strip()
    z_values = [record[key] for key in sorted(record.keys()) if str(key).startswith("z_")]
    z_lines = "\n".join(f"{key}={float(record[key]):.6f}" for key in sorted(record.keys()) if str(key).startswith("z_"))
    return "\n".join(
        [
            "[TASK]",
            "주어진 대화 입력에 대해 어울리는 응답 1개를 생성하라.",
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
            '  "response": "string"',
            '}',
            "",
            "[CONSTRAINTS]",
            "- response 는 입력 내용과 정합적이어야 한다.",
            "- 과장하지 말고 자연스러운 한국어로 쓴다.",
            "- 응답은 2~4문장으로 쓴다.",
            "- 한 응답 안에서 상충하는 말투를 섞지 말고 하나의 톤을 유지한다.",
            "- 마크다운, bullet, 번호 목록, 따옴표 인용을 쓰지 않는다.",
            "- z 는 직접 설명하지 말고 내부 상태 힌트로만 사용한다.",
            "- 설명 문장 없이 JSON object 하나만 출력한다.",
        ]
    )


def _default_response_generation_template() -> str:
    return "\n".join(
        [
            "[ROLE]",
            "당신은 감정 상태에 맞는 말투와 리듬으로 답하는 한국어 응답 생성기다.",
            "",
            "[USER_INPUT]",
            "{{input_text}}",
            "",
            "[STYLE_TAGS]",
            "{{style_tags}}",
            "",
            "[STYLE_SUMMARY]",
            "{{style_summary_lines}}",
            "",
            "[STYLE_VECTOR]",
            "{{style_vector_lines}}",
            "",
            "[INSTRUCTIONS]",
            "- 사용자 입력의 내용에 직접 답한다.",
            "- STYLE_TAGS와 STYLE_SUMMARY에 맞춰 말투와 표현 밀도를 조절한다.",
            "- 스타일을 설명하지 말고, 그 스타일로 자연스럽게 답한다.",
            "- 한국어 평문으로만 3~6문장 이내로 답한다.",
            "- bullet, markdown, JSON, 코드블록을 쓰지 않는다.",
        ]
    )


def render_template(template: str, variables: dict[str, str]) -> str:
    rendered = template
    for key, value in variables.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", value)
    return rendered


def load_response_generation_template(template_path: Path | None = None) -> str:
    if template_path is not None and template_path.exists():
        return template_path.read_text(encoding="utf-8")
    default_path = Path(__file__).resolve().parents[1] / "prompts" / "response_generation_prompt.md"
    if default_path.exists():
        return default_path.read_text(encoding="utf-8")
    return _default_response_generation_template()


def format_style_vector_lines(style_dict: dict[str, float]) -> str:
    return "\n".join(f"{axis}={float(value):.4f}" for axis, value in style_dict.items())


def build_response_generation_prompt(
    input_text: str,
    style_dict: dict[str, float],
    style_tags: list[str],
    style_summary: dict[str, float],
    template_path: Path | None = None,
) -> str:
    template = load_response_generation_template(template_path)
    return render_template(
        template,
        {
            "input_text": input_text.strip(),
            "style_tags": ", ".join(style_tags) if style_tags else "(none)",
            "style_summary_lines": "\n".join(format_style_summary_lines(style_summary)),
            "style_vector_lines": format_style_vector_lines(style_dict),
        },
    )


def infer_style_profile(
    model: EmoNet,
    decoder: LinearZtoSDecoder,
    text: str,
) -> dict[str, object]:
    outputs = model.forward(text)
    z = np.asarray(outputs["z"], dtype=np.float32).reshape(-1)
    s_pred = np.asarray(decoder.predict(z), dtype=np.float32).reshape(-1)
    style_dict = style_vector_to_dict(s_pred.tolist(), STYLE_AXIS_NAMES[: len(s_pred)])
    style_summary = build_style_summary(style_dict)
    style_tags = build_style_tags(style_dict)
    return {
        "stim_vec": np.asarray(outputs["stim_vec"], dtype=np.float32).reshape(-1),
        "dominant_branch_len": len(outputs["dominant_branch"]),
        "z": z,
        "s_pred": s_pred,
        "style_dict": style_dict,
        "style_tags": style_tags,
        "style_summary": style_summary,
        "style_summary_text": summarize_style_summary(style_summary),
    }


def generate_response_from_style(
    base_url: str,
    model_name: str,
    input_text: str,
    style_dict: dict[str, float],
    style_tags: list[str],
    style_summary: dict[str, float],
    temperature: float,
    max_tokens: int,
    timeout_sec: int,
    template_path: Path | None = None,
) -> tuple[str, str]:
    prompt = build_response_generation_prompt(
        input_text=input_text,
        style_dict=style_dict,
        style_tags=style_tags,
        style_summary=style_summary,
        template_path=template_path,
    )
    response = call_openai_compatible_chat(
        base_url=base_url,
        model_name=model_name,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_sec=timeout_sec,
        system_prompt="Return a plain Korean response only. Do not return JSON.",
    ).strip()
    return response, prompt


def serialize_generation_log(record: dict[str, object]) -> dict[str, object]:
    payload = dict(record)
    for key in ("stim_vec", "z", "s_pred", "style_tags"):
        if key in payload:
            payload[key] = json.dumps(payload[key], ensure_ascii=False)
    if "style_summary" in payload and isinstance(payload["style_summary"], dict):
        payload["style_summary_json"] = json.dumps(payload["style_summary"], ensure_ascii=False)
        del payload["style_summary"]
    return payload


def append_jsonl(output_path: Path, rows: list[dict[str, object]]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_style_block_prompt(
    record: dict[str, object],
    response: str,
    block_axes: list[str],
    active_axes: list[str],
    key_name: str,
) -> str:
    text = str(record.get("text", "")).strip()
    example_lines = "\n".join(f'    "{axis}": 0.0' + ("," if idx < len(block_axes) - 1 else "") for idx, axis in enumerate(block_axes))
    return "\n".join(
        [
            "[TASK]",
            f"아래 입력과 응답을 보고 응답 스타일을 {len(block_axes)}개 축으로 평가하라.",
            "",
            "[INPUT_TEXT]",
            text,
            "",
            "[RESPONSE]",
            response.strip(),
            "",
            "[STYLE_AXES]",
            format_style_axes(block_axes, active_axes),
            "",
            "[SCORING_RULES]",
            f"- 각 축 값은 다음 5개 값 중 하나만 사용한다: {format_score_levels()}",
            "- 0.00 = 왼쪽 성향이 거의 없음, 0.50 = 중간, 1.00 = 오른쪽 성향이 매우 강함",
            "- 응답 표면의 문체만 보고 판단한다. 내용 정답 여부나 화자의 내면 상태는 추정하지 않는다.",
            "- 애매하면 극단값 대신 0.25, 0.50, 0.75 중 하나를 고른다.",
            "",
            "[OUTPUT_FORMAT]",
            "JSON only.",
            "{",
            f'  "{key_name}": {{',
            example_lines,
            "  }",
            "}",
            "",
            "[CONSTRAINTS]",
            "- 반드시 위 STYLE_AXES에 적힌 축 이름만 그대로 사용한다.",
            "- 축 이름을 바꾸거나 dim0 같은 별칭으로 바꾸지 않는다.",
            f"- 각 축은 반드시 다음 값 중 하나로만 준다: {format_score_levels()}",
            "- 설명 없이 JSON object 하나만 출력한다.",
        ]
    )


def run_style_block_pass(
    record: dict[str, object],
    response_text: str,
    block_axes: list[str],
    active_axes: list[str],
    key_name: str,
    base_url: str,
    model_name: str,
    temperature: float,
    max_tokens: int,
    timeout_sec: int,
    max_retries: int,
) -> tuple[dict[str, float], str]:
    prompt = make_style_block_prompt(
        record=record,
        response=response_text,
        block_axes=block_axes,
        active_axes=active_axes,
        key_name=key_name,
    )
    style_values, raw = request_json_response(
        base_url=base_url,
        model_name=model_name,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_sec=timeout_sec,
        max_retries=max_retries,
        validator=lambda payload: normalize_style_dict(payload, key_name, expected_axes=block_axes),
        retry_instruction=(
            "직전 응답의 JSON key 또는 값 형식이 잘못되었다. "
            f"반드시 '{key_name}' object 안에 다음 축만 그대로 넣어라: {', '.join(block_axes)}. "
            "설명 없이 JSON object 하나만 다시 출력하라."
        ),
    )
    if not isinstance(style_values, dict):
        raise ValueError("validated style payload must be a dict")
    return style_values, raw


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
    max_retries: int,
    keep_failures: bool,
    block_size: int,
    style_dim: int,
    keep_threshold: float,
) -> None:
    rows = []
    total = len(df) if limit is None else min(len(df), limit)
    start_time = time.perf_counter()
    active_axes = resolve_style_axes(style_dim)
    style_blocks = build_style_blocks(block_size, active_axes)
    ensure_model_server_ready(base_url, timeout_sec)

    for idx, record in enumerate(df.to_dict(orient="records"), start=1):
        if limit is not None and idx > limit:
            break

        row = dict(record)
        row["status"] = "error"
        row["generation_status"] = "pending"
        row["error_message"] = ""
        for block_idx in range(1, len(style_blocks) + 1):
            row[f"s_block{block_idx}_status"] = "pending"
            row[f"s_hat_block{block_idx}_status"] = "pending"
            row[f"s_block{block_idx}_raw_output"] = ""
            row[f"s_hat_block{block_idx}_raw_output"] = ""
        try:
            generation_prompt = make_generation_prompt(record)
            response_text, generation_raw = request_json_response(
                base_url=base_url,
                model_name=model_name,
                prompt=generation_prompt,
                temperature=generation_temperature,
                max_tokens=max_tokens,
                timeout_sec=timeout_sec,
                max_retries=max_retries,
                validator=normalize_response_text,
                retry_instruction=(
                    "직전 응답의 JSON 형식이 잘못되었거나 response 문자열이 비어 있었다. "
                    "반드시 {'response': '...'} 형식의 JSON object 하나만 다시 출력하라."
                ),
            )
            if not isinstance(response_text, str):
                raise ValueError("validated response must be a string")
            row["generation_status"] = "ok"
            row["llm_response"] = response_text
            row["generation_raw_output"] = generation_raw

            style: dict[str, float] = {}
            style_hat: dict[str, float] = {}

            for block_idx, block_axes in enumerate(style_blocks, start=1):
                block_style, block_raw = run_style_block_pass(
                    record=record,
                    response_text=response_text,
                    block_axes=block_axes,
                    active_axes=active_axes,
                    key_name="s",
                    base_url=base_url,
                    model_name=model_name,
                    temperature=rating_temperature,
                    max_tokens=max_tokens,
                    timeout_sec=timeout_sec,
                    max_retries=max_retries,
                )
                row[f"s_block{block_idx}_status"] = "ok"
                row[f"s_block{block_idx}_raw_output"] = block_raw
                style.update(block_style)

            for block_idx, block_axes in enumerate(style_blocks, start=1):
                block_style_hat, block_raw = run_style_block_pass(
                    record=record,
                    response_text=response_text,
                    block_axes=block_axes,
                    active_axes=active_axes,
                    key_name="s_hat",
                    base_url=base_url,
                    model_name=model_name,
                    temperature=rating_temperature,
                    max_tokens=max_tokens,
                    timeout_sec=timeout_sec,
                    max_retries=max_retries,
                )
                row[f"s_hat_block{block_idx}_status"] = "ok"
                row[f"s_hat_block{block_idx}_raw_output"] = block_raw
                style_hat.update(block_style_hat)

            consistency_l1 = compute_consistency(style, style_hat, active_axes)

            row["status"] = "ok"
            row["consistency_l1"] = consistency_l1
            row["keep_sample"] = bool(consistency_l1 <= keep_threshold)
            row["style_dim"] = len(active_axes)
            for axis_idx, axis in enumerate(active_axes):
                row[f"s_{axis_idx}"] = style[axis]
                row[f"s_hat_{axis_idx}"] = style_hat[axis]
            rows.append(row)
        except Exception as exc:
            if keep_failures:
                row["llm_response"] = row.get("llm_response", "")
                row["generation_raw_output"] = row.get("generation_raw_output", "")
                row["consistency_l1"] = np.nan
                row["keep_sample"] = False
                row["style_dim"] = len(active_axes)
                row["error_message"] = str(exc)
                rows.append(row)
            else:
                raise

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
        max_retries=args.max_retries,
        keep_failures=args.keep_failures,
        block_size=args.block_size,
        style_dim=args.style_dim,
        keep_threshold=args.keep_threshold,
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

    def add_generation_options(subparser: argparse.ArgumentParser) -> None:
        add_common_options(subparser)
        subparser.add_argument("--zs-model-path", required=True)
        subparser.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
        subparser.add_argument("--model-name", default="gpt-oss:20b")
        subparser.add_argument("--response-temperature", type=float, default=0.5)
        subparser.add_argument("--max-tokens", type=int, default=600)
        subparser.add_argument("--timeout-sec", type=int, default=180)
        subparser.add_argument("--prompt-template", default=None)
        subparser.add_argument("--log-jsonl", default=None)

    fit_parser = subparsers.add_parser("fit-stim")
    add_common_options(fit_parser)
    fit_parser.set_defaults(func=command_fit_stim)

    infer_parser = subparsers.add_parser("infer")
    add_common_options(infer_parser)
    infer_parser.add_argument("--text", required=True)
    infer_parser.add_argument("--zs-model-path", default=None)
    infer_parser.set_defaults(func=command_infer)

    generate_parser = subparsers.add_parser("generate-response")
    add_generation_options(generate_parser)
    generate_parser.add_argument("--text", required=True)
    generate_parser.add_argument("--output-json", default=None)
    generate_parser.set_defaults(func=command_generate_response)

    batch_generate_parser = subparsers.add_parser("generate-response-batch")
    add_generation_options(batch_generate_parser)
    batch_generate_parser.add_argument("--input-csv", required=True)
    batch_generate_parser.add_argument("--output-csv", required=True)
    batch_generate_parser.add_argument("--text-column", default="text")
    batch_generate_parser.add_argument("--limit", type=int, default=None)
    batch_generate_parser.add_argument("--progress-every", type=int, default=10)
    batch_generate_parser.set_defaults(func=command_generate_response_batch)

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

    fit_zs_parser = subparsers.add_parser("fit-zs-regressor")
    fit_zs_parser.add_argument("--input-csv", required=True)
    fit_zs_parser.add_argument("--model-path", required=True)
    fit_zs_parser.add_argument("--z-dim", type=int, default=64)
    fit_zs_parser.add_argument("--s-dim", type=int, default=None)
    fit_zs_parser.add_argument("--ridge-alpha", type=float, default=1.0)
    fit_zs_parser.add_argument("--val-ratio", type=float, default=0.1)
    fit_zs_parser.add_argument("--seed", type=int, default=42)
    fit_zs_parser.add_argument("--use-all-rows", action="store_true")
    fit_zs_parser.set_defaults(func=command_fit_zs_regressor)

    predict_s_parser = subparsers.add_parser("predict-s")
    predict_s_parser.add_argument("--input-csv", required=True)
    predict_s_parser.add_argument("--output-csv", required=True)
    predict_s_parser.add_argument("--model-path", required=True)
    predict_s_parser.add_argument("--z-dim", type=int, default=64)
    predict_s_parser.add_argument("--output-prefix", default="s_pred_")
    predict_s_parser.set_defaults(func=command_predict_s)

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
    local_parser.add_argument("--max-retries", type=int, default=2)
    local_parser.add_argument("--block-size", type=int, default=8)
    local_parser.add_argument("--style-dim", type=int, default=32)
    local_parser.add_argument("--keep-threshold", type=float, default=0.12)
    local_parser.add_argument("--keep-failures", action="store_true")
    local_parser.set_defaults(func=command_label_local)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
