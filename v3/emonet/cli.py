from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Callable
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None

from .core import (
    BRANCH_FEATURE_DIM,
    TORCH_AVAILABLE,
    EmoNet,
    EmoNetConfig,
    LinearZtoSDecoder,
    StimEncoderConfig,
    ZSDecoderConfig,
)


DEFAULT_Z_ENCODER_MODEL_PATH = Path(__file__).resolve().parents[1] / "artifacts" / "dominant_branch_encoder.pt"
DEFAULT_STYLE_PROFILE = "core32"
MODEL_OPTIONAL_CONFIG_FIELDS = [
    "max_ticks",
    "min_ticks_before_converged",
    "k_threshold_base",
    "k_remem_base",
    "k_decay",
    "convergence_patience",
    "activity_count_delta_eps",
    "edge_count_delta_eps",
    "activity_churn_eps",
    "refractory_ticks",
    "input_topk",
    "input_signal_clip",
    "memory_decay",
    "memory_stim_mix",
    "memory_k_mix",
    "state_self_stim_mix",
    "state_parent_stim_mix",
    "state_base_stim_mix",
    "state_bias_stim_mix",
    "recent_activity_decay",
    "hysteresis_threshold_gain",
    "hysteresis_remem_gain",
    "hysteresis_k_bonus",
    "intrinsic_alignment_gain",
    "fatigue_decay",
    "fatigue_gain",
    "fatigue_threshold_gain",
    "fatigue_k_leak",
    "fire_output_log_gain",
    "inhibitory_suppression_gain",
    "max_out_degree",
    "min_out_degree",
    "dopa_rewire_gain",
    "sero_prune_gain",
    "mela_dropout_gain",
    "ne_thresh_reduce_gain",
    "ne_remem_reduce_gain",
    "global_recovery_rate",
    "topk_branches",
    "branch_end_window",
    "branch_length_bonus",
]
MODEL_BUILD_ARG_FIELDS = [
    "dataset_csv",
    "benchmark_csv",
    "model_cache_path",
    "max_samples",
    "force_refit",
    "seed",
    "z_dim",
    "z_encoder_mode",
    "z_encoder_path",
    *MODEL_OPTIONAL_CONFIG_FIELDS,
]
_PARALLEL_MODEL: EmoNet | None = None


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(float(seconds))))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def build_progress_line(
    label: str,
    current: int,
    total: int | None,
    start_time: float,
    *,
    unit: str = "rows",
    extra: str | None = None,
) -> str:
    elapsed = max(1e-8, time.perf_counter() - start_time)
    rate = float(current) / elapsed if current > 0 else 0.0
    parts: list[str] = []
    if total is not None and total > 0:
        pct = 100.0 * float(current) / float(total)
        parts.append(f"{label}: {current}/{total} ({pct:.1f}%)")
        if 0 < current < total and rate > 0.0:
            remaining = max(0.0, float(total - current) / rate)
            parts.append(f"eta {format_duration(remaining)}")
    else:
        parts.append(f"{label}: {current}")
    if rate > 0.0:
        parts.append(f"{rate:.2f} {unit}/s")
    parts.append(f"elapsed {format_duration(elapsed)}")
    if extra:
        parts.append(extra)
    return " | ".join(parts)


def maybe_print_progress(
    label: str,
    current: int,
    total: int | None,
    start_time: float,
    *,
    every: int,
    unit: str = "rows",
    extra: str | None = None,
    force: bool = False,
) -> None:
    if every <= 0:
        return
    if not force and current % every != 0:
        return
    print(build_progress_line(label, current, total, start_time, unit=unit, extra=extra))


def resolve_z_encoder_path(raw_path: str | None) -> Path:
    return Path(raw_path) if raw_path else DEFAULT_Z_ENCODER_MODEL_PATH


def resolve_z_encoder_mode(
    requested_mode: str,
    z_encoder_path: Path,
    *,
    allow_missing_checkpoint: bool,
) -> str:
    mode = str(requested_mode or "auto").strip().lower()
    if mode not in {"auto", "stat", "transformer"}:
        raise ValueError("z_encoder_mode must be one of: auto, stat, transformer")
    if mode == "auto":
        if TORCH_AVAILABLE and z_encoder_path.exists():
            return "transformer"
        return "stat"
    if mode == "transformer":
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch is required to use the transformer z encoder")
        if not allow_missing_checkpoint and not z_encoder_path.exists():
            raise FileNotFoundError(f"z encoder checkpoint not found: {z_encoder_path}")
    return mode


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


def resolve_num_workers(raw_value: int | None) -> int:
    if raw_value is None:
        return 1
    if int(raw_value) == 0:
        return max(1, int(os.cpu_count() or 1))
    return max(1, int(raw_value))


def estimate_executor_chunksize(total: int, num_workers: int, preferred: int = 64) -> int:
    if total <= 0:
        return 1
    return max(1, min(int(preferred), max(1, total // max(1, num_workers * 8))))


def build_worker_fallback_plan(requested_workers: int) -> list[int]:
    worker_count = resolve_num_workers(requested_workers)
    if worker_count <= 1:
        return [1]
    plan = [worker_count]
    seen = {worker_count}
    current = worker_count
    while current > 1:
        current = max(1, current // 2)
        if current not in seen:
            plan.append(current)
            seen.add(current)
    if 1 not in seen:
        plan.append(1)
    return plan


def is_parallel_pool_failure(exc: BaseException) -> bool:
    if isinstance(exc, BrokenProcessPool):
        return True
    if isinstance(exc, OSError):
        message = str(exc).lower()
        return "handle is closed" in message or "invalid handle" in message
    return False


def print_worker_fallback(task_name: str, failed_workers: int, next_workers: int, exc: BaseException) -> None:
    print(
        f"{task_name}: worker pool failed with num_workers={failed_workers} "
        f"({exc.__class__.__name__}: {exc}). falling back to num_workers={next_workers} and retrying remaining work.",
        flush=True,
    )


def build_model_payload(args: argparse.Namespace) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for field_name in MODEL_BUILD_ARG_FIELDS:
        payload[field_name] = getattr(args, field_name, None)
    return payload


def prepare_parallel_model_payload(args: argparse.Namespace) -> dict[str, Any]:
    warm_model = build_model(args)
    warm_model.stim_encoder.ensure_fitted()
    payload = build_model_payload(args)
    payload["force_refit"] = False
    return payload


def _init_parallel_model(model_payload: dict[str, Any]) -> None:
    global _PARALLEL_MODEL
    _PARALLEL_MODEL = build_model(argparse.Namespace(**model_payload))


def _require_parallel_model() -> EmoNet:
    if _PARALLEL_MODEL is None:
        raise RuntimeError("parallel worker model is not initialized")
    return _PARALLEL_MODEL


def build_model(
    args: argparse.Namespace,
    *,
    allow_missing_z_encoder_checkpoint: bool = False,
    z_encoder_mode_override: str | None = None,
    load_z_encoder_checkpoint: bool = True,
) -> EmoNet:
    z_encoder_path = resolve_z_encoder_path(getattr(args, "z_encoder_path", None))
    z_encoder_mode = resolve_z_encoder_mode(
        z_encoder_mode_override or getattr(args, "z_encoder_mode", "auto"),
        z_encoder_path,
        allow_missing_checkpoint=allow_missing_z_encoder_checkpoint,
    )
    config_kwargs: dict[str, Any] = {
        "seed": args.seed,
        "z_dim": args.z_dim,
        "z_encoder_mode": z_encoder_mode,
        "z_encoder_path": z_encoder_path,
        "load_z_encoder_checkpoint": load_z_encoder_checkpoint,
    }
    for field_name in MODEL_OPTIONAL_CONFIG_FIELDS:
        field_value = getattr(args, field_name, None)
        if field_value is not None:
            config_kwargs[field_name] = field_value

    config = EmoNetConfig(
        **config_kwargs,
    )
    stim_config = build_stim_config(args)
    return EmoNet(config=config, stim_encoder_config=stim_config)


def to_numpy_array(value: object, dtype: Any | None = None) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):
        array = value.detach().cpu().numpy()
        return array.astype(dtype, copy=False) if dtype is not None else array
    return np.asarray(value, dtype=dtype)


def command_fit_stim(args: argparse.Namespace) -> None:
    model = build_model(args)
    model.stim_encoder.fit()
    print(json.dumps({"model_cache_path": str(model.stim_encoder.config.model_cache_path)}, ensure_ascii=False, indent=2))


def command_infer(args: argparse.Namespace) -> None:
    model = build_model(args)
    outputs = model.forward(args.text)
    result = {
        "stim_vec": to_numpy_array(outputs["stim_vec"], dtype=float).tolist(),
        "dominant_branch_len": len(outputs["dominant_branch"]),
        "z": to_numpy_array(outputs["z"], dtype=float).tolist(),
    }
    if args.zs_model_path:
        decoder = LinearZtoSDecoder.load(Path(args.zs_model_path))
        result["s_pred"] = to_numpy_array(decoder.predict(to_numpy_array(outputs["z"], dtype=np.float32)), dtype=float).tolist()
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


def export_z_from_dataframe(
    model: EmoNet | None,
    df: pd.DataFrame,
    text_column: str,
    output_csv: Path,
    *,
    progress_every: int,
    num_workers: int = 1,
    model_args: argparse.Namespace | None = None,
) -> None:
    records = df.to_dict(orient="records")
    total = len(records)
    start_time = time.perf_counter()
    rows: list[dict[str, object]] = []
    fallback_plan = build_worker_fallback_plan(num_workers)
    worker_count = fallback_plan[0]

    if worker_count <= 1:
        if model is None:
            raise ValueError("model is required when num_workers <= 1")
        for idx, record in enumerate(records, start=1):
            outputs = model.forward(str(record.get(text_column, "")))
            rows.append(build_output_row(record, outputs))
            maybe_print_progress("export-z", idx, total, start_time, every=progress_every)
    else:
        if model_args is None:
            raise ValueError("model_args is required when num_workers > 1")
        payload = prepare_parallel_model_payload(model_args)
        completed = 0
        for plan_idx, current_workers in enumerate(fallback_plan):
            remaining_records = records[completed:]
            if not remaining_records:
                break
            if current_workers <= 1:
                serial_model = model if model is not None else build_model(model_args)
                for record in remaining_records:
                    outputs = serial_model.forward(str(record.get(text_column, "")))
                    rows.append(build_output_row(record, outputs))
                    completed += 1
                    maybe_print_progress("export-z", completed, total, start_time, every=progress_every)
                break

            try:
                chunksize = estimate_executor_chunksize(len(remaining_records), current_workers, preferred=128)
                task_iter = ((record, text_column) for record in remaining_records)
                with ProcessPoolExecutor(
                    max_workers=current_workers,
                    initializer=_init_parallel_model,
                    initargs=(payload,),
                ) as executor:
                    for row in executor.map(_parallel_export_record, task_iter, chunksize=chunksize):
                        rows.append(row)
                        completed += 1
                        maybe_print_progress("export-z", completed, total, start_time, every=progress_every)
                break
            except Exception as exc:
                if not is_parallel_pool_failure(exc) or plan_idx >= len(fallback_plan) - 1:
                    raise
                print_worker_fallback("export-z", current_workers, fallback_plan[plan_idx + 1], exc)

    output_df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    elapsed = time.perf_counter() - start_time
    print(
        json.dumps(
            {
                "rows": int(len(output_df)),
                "output_csv": str(output_csv),
                "elapsed_sec": round(elapsed, 3),
                "num_workers": int(worker_count),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def build_output_row(source_row: dict, outputs: dict[str, object]) -> dict[str, object]:
    row = dict(source_row)
    z = to_numpy_array(outputs["z"], dtype=np.float32).reshape(-1)
    stim = to_numpy_array(outputs["stim_vec"], dtype=np.float32).reshape(-1)
    for dim, value in enumerate(z):
        row[f"z_{dim}"] = float(value)
    for dim, name in enumerate(("dopamine", "serotonin", "norepinephrine", "melatonin")):
        row[name] = float(stim[dim])
    row["dominant_branch_len"] = int(len(outputs["dominant_branch"]))
    return row


def _parallel_export_record(task: tuple[dict[str, object], str]) -> dict[str, object]:
    record, text_column = task
    model = _require_parallel_model()
    outputs = model.forward(str(record.get(text_column, "")))
    return build_output_row(record, outputs)


def _parallel_probe_record(task: tuple[dict[str, object], str]) -> dict[str, object]:
    record, text_column = task
    model = _require_parallel_model()
    outputs = model.forward(str(record.get(text_column, "")))
    row = dict(record)
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


def load_existing_ids(output_csv: Path, column_name: str = "talk_id") -> set[str]:
    if not output_csv.exists():
        return set()
    existing = pd.read_csv(output_csv, usecols=[column_name]) if output_csv.stat().st_size > 0 else pd.DataFrame()
    if column_name not in existing.columns:
        return set()
    return {str(value) for value in existing[column_name].dropna().astype(str)}


def export_z_from_json_stream(
    model: EmoNet | None,
    input_json: Path,
    output_csv: Path,
    limit: int | None,
    chunk_size: int,
    progress_every: int,
    resume: bool,
    *,
    num_workers: int = 1,
    model_args: argparse.Namespace | None = None,
) -> None:
    rows_to_write: list[dict[str, object]] = []
    skipped = 0
    write_header = not output_csv.exists() or not resume
    existing_ids = load_existing_ids(output_csv) if resume else set()
    start_time = time.perf_counter()
    source_rows: list[dict[str, object]] = []
    fallback_plan = build_worker_fallback_plan(num_workers)
    worker_count = fallback_plan[0]

    if resume and existing_ids:
        print(f"resume mode: skipping {len(existing_ids)} existing talk_id rows")

    for source_row in load_training_json_records(input_json):
        talk_id = str(source_row.get("talk_id", ""))
        if existing_ids and talk_id and talk_id in existing_ids:
            skipped += 1
            continue
        source_rows.append(source_row)
        if limit is not None and len(source_rows) >= limit:
            break

    total = len(source_rows)
    processed = 0
    written = 0

    if worker_count <= 1:
        if model is None:
            raise ValueError("model is required when num_workers <= 1")
        for source_row in source_rows:
            outputs = model.forward(str(source_row["text"]))
            rows_to_write.append(build_output_row(source_row, outputs))
            processed += 1
            written += 1
            maybe_print_progress("export-z", processed, total, start_time, every=progress_every)
            if len(rows_to_write) >= chunk_size:
                write_header = flush_rows(rows_to_write, output_csv, write_header)
                rows_to_write.clear()
    else:
        if model_args is None:
            raise ValueError("model_args is required when num_workers > 1")
        payload = prepare_parallel_model_payload(model_args)
        for plan_idx, current_workers in enumerate(fallback_plan):
            remaining_rows = source_rows[processed:]
            if not remaining_rows:
                break
            if current_workers <= 1:
                serial_model = model if model is not None else build_model(model_args)
                for source_row in remaining_rows:
                    outputs = serial_model.forward(str(source_row["text"]))
                    rows_to_write.append(build_output_row(source_row, outputs))
                    processed += 1
                    written += 1
                    maybe_print_progress("export-z", processed, total, start_time, every=progress_every)
                    if len(rows_to_write) >= chunk_size:
                        write_header = flush_rows(rows_to_write, output_csv, write_header)
                        rows_to_write.clear()
                break

            try:
                chunksize = estimate_executor_chunksize(len(remaining_rows), current_workers, preferred=max(16, chunk_size))
                task_iter = ((source_row, "text") for source_row in remaining_rows)
                with ProcessPoolExecutor(
                    max_workers=current_workers,
                    initializer=_init_parallel_model,
                    initargs=(payload,),
                ) as executor:
                    for row in executor.map(_parallel_export_record, task_iter, chunksize=chunksize):
                        rows_to_write.append(row)
                        processed += 1
                        written += 1
                        maybe_print_progress("export-z", processed, total, start_time, every=progress_every)
                        if len(rows_to_write) >= chunk_size:
                            write_header = flush_rows(rows_to_write, output_csv, write_header)
                            rows_to_write.clear()
                break
            except Exception as exc:
                if len(rows_to_write) >= chunk_size:
                    write_header = flush_rows(rows_to_write, output_csv, write_header)
                    rows_to_write.clear()
                if not is_parallel_pool_failure(exc) or plan_idx >= len(fallback_plan) - 1:
                    raise
                print_worker_fallback("export-z", current_workers, fallback_plan[plan_idx + 1], exc)

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
                "num_workers": int(worker_count),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def sample_probe_dataframe(df: pd.DataFrame, sample_size: int | None, sample_mode: str, seed: int) -> pd.DataFrame:
    if sample_size is None or sample_size <= 0 or sample_size >= len(df):
        return df.copy()
    if sample_mode == "head":
        return df.head(sample_size).copy()
    if sample_mode == "random":
        return df.sample(n=sample_size, random_state=seed).reset_index(drop=True)
    raise ValueError("sample_mode must be one of: head, random")


def summarize_branch_lengths(lengths: list[int]) -> dict[str, object]:
    if not lengths:
        return {
            "rows": 0,
            "mean": 0.0,
            "median": 0.0,
            "len1": 0,
            "len1_ratio": 0.0,
            "max": 0,
            "p90": 0,
            "p95": 0,
            "bucket_counts": {},
        }

    arr = np.asarray(lengths, dtype=np.int32)
    bucket_counts = {
        "len1": int(np.sum(arr == 1)),
        "len2_3": int(np.sum((arr >= 2) & (arr <= 3))),
        "len4_7": int(np.sum((arr >= 4) & (arr <= 7))),
        "len8_15": int(np.sum((arr >= 8) & (arr <= 15))),
        "len16_plus": int(np.sum(arr >= 16)),
    }
    return {
        "rows": int(len(arr)),
        "mean": round(float(arr.mean()), 4),
        "median": float(np.median(arr)),
        "len1": int(np.sum(arr == 1)),
        "len1_ratio": round(float(np.mean(arr == 1)), 4),
        "max": int(arr.max()),
        "p90": int(np.quantile(arr, 0.9)),
        "p95": int(np.quantile(arr, 0.95)),
        "bucket_counts": bucket_counts,
    }


def probe_branch_lengths(
    model: EmoNet | None,
    df: pd.DataFrame,
    text_column: str,
    *,
    progress_every: int,
    num_workers: int = 1,
    model_args: argparse.Namespace | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    start_time = time.perf_counter()
    records = df.to_dict(orient="records")
    total = len(records)
    fallback_plan = build_worker_fallback_plan(num_workers)
    worker_count = fallback_plan[0]

    if worker_count <= 1:
        if model is None:
            raise ValueError("model is required when num_workers <= 1")
        for idx, record in enumerate(records, start=1):
            outputs = model.forward(str(record.get(text_column, "")))
            row = dict(record)
            row["dominant_branch_len"] = int(len(outputs["dominant_branch"]))
            rows.append(row)
            maybe_print_progress("probe-branch", idx, total, start_time, every=progress_every)
    else:
        if model_args is None:
            raise ValueError("model_args is required when num_workers > 1")
        payload = prepare_parallel_model_payload(model_args)
        completed = 0
        for plan_idx, current_workers in enumerate(fallback_plan):
            remaining_records = records[completed:]
            if not remaining_records:
                break
            if current_workers <= 1:
                serial_model = model if model is not None else build_model(model_args)
                for record in remaining_records:
                    outputs = serial_model.forward(str(record.get(text_column, "")))
                    row = dict(record)
                    row["dominant_branch_len"] = int(len(outputs["dominant_branch"]))
                    rows.append(row)
                    completed += 1
                    maybe_print_progress("probe-branch", completed, total, start_time, every=progress_every)
                break

            try:
                chunksize = estimate_executor_chunksize(len(remaining_records), current_workers, preferred=64)
                task_iter = ((record, text_column) for record in remaining_records)
                with ProcessPoolExecutor(
                    max_workers=current_workers,
                    initializer=_init_parallel_model,
                    initargs=(payload,),
                ) as executor:
                    for row in executor.map(_parallel_probe_record, task_iter, chunksize=chunksize):
                        rows.append(row)
                        completed += 1
                        maybe_print_progress("probe-branch", completed, total, start_time, every=progress_every)
                break
            except Exception as exc:
                if not is_parallel_pool_failure(exc) or plan_idx >= len(fallback_plan) - 1:
                    raise
                print_worker_fallback("probe-branch", current_workers, fallback_plan[plan_idx + 1], exc)
    return pd.DataFrame(rows)


def command_probe_branch(args: argparse.Namespace) -> None:
    if bool(args.input_csv) == bool(args.input_json):
        raise ValueError("provide exactly one of --input-csv or --input-json")

    if args.input_json is not None:
        df = load_training_json_as_dataframe(Path(args.input_json))
    else:
        df = pd.read_csv(Path(args.input_csv))

    text_column = resolve_text_column(df, args.text_column)
    sampled = sample_probe_dataframe(df, args.sample_size, args.sample_mode, args.seed)
    worker_count = resolve_num_workers(getattr(args, "num_workers", 1))
    model = build_model(args) if worker_count <= 1 else None
    result_df = probe_branch_lengths(
        model=model,
        df=sampled,
        text_column=text_column,
        progress_every=args.progress_every,
        num_workers=worker_count,
        model_args=args,
    )
    summary = summarize_branch_lengths(result_df["dominant_branch_len"].astype(int).tolist())
    payload = {
        "input_rows": int(len(df)),
        "sample_rows": int(len(result_df)),
        "sample_mode": args.sample_mode,
        "seed": int(args.seed),
        "num_workers": int(worker_count),
        **summary,
    }
    if args.output_csv:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
        payload["output_csv"] = str(output_csv)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


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
    keep_column: str = "keep_sample",
) -> dict[str, object]:
    original_rows = len(df)
    keep_filtered_rows = 0
    if not use_all_rows and keep_column in df.columns:
        keep_mask = df[keep_column].fillna(False).astype(bool)
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
        "keep_column": str(keep_column),
    }


def pad_branch_tensor_batch(branch_tensors: list[np.ndarray]) -> tuple["torch.Tensor", "torch.Tensor"]:
    if torch is None:
        raise RuntimeError("torch is required to batch dominant-branch tensors")
    if not branch_tensors:
        raise ValueError("branch_tensors must not be empty")
    max_len = max(int(np.asarray(tensor).shape[0]) for tensor in branch_tensors)
    batch = torch.zeros((len(branch_tensors), max_len, BRANCH_FEATURE_DIM), dtype=torch.float32)
    attention_mask = torch.zeros((len(branch_tensors), max_len), dtype=torch.bool)
    for row_idx, branch_tensor in enumerate(branch_tensors):
        seq = np.asarray(branch_tensor, dtype=np.float32)
        if seq.ndim != 2 or seq.shape[1] != BRANCH_FEATURE_DIM:
            raise ValueError(f"branch tensor must have shape [seq_len, {BRANCH_FEATURE_DIM}], got {seq.shape}")
        seq_len = int(seq.shape[0])
        batch[row_idx, :seq_len] = torch.as_tensor(seq, dtype=torch.float32)
        attention_mask[row_idx, :seq_len] = True
    return batch, attention_mask


def encode_branch_tensors(
    encoder: "nn.Module",
    branch_tensors: list[np.ndarray],
    batch_size: int,
    device: "torch.device",
) -> np.ndarray:
    if torch is None:
        raise RuntimeError("torch is required to encode dominant branches")
    outputs: list[np.ndarray] = []
    encoder.eval()
    with torch.no_grad():
        for start in range(0, len(branch_tensors), batch_size):
            batch_tensors = branch_tensors[start : start + batch_size]
            batch, attention_mask = pad_branch_tensor_batch(batch_tensors)
            z = encoder(batch.to(device), attention_mask.to(device))
            outputs.append(z.detach().cpu().numpy().astype(np.float32, copy=False))
    return np.vstack(outputs) if outputs else np.zeros((0, 0), dtype=np.float32)


def train_transformer_z_encoder_from_dataframe(
    df: pd.DataFrame,
    model: EmoNet,
    text_column: str,
    encoder_model_path: Path,
    zs_model_path: Path,
    z_output_csv: Path | None,
    style_dim: int,
    style_profile: str,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    ridge_alpha: float,
    seed: int,
    val_ratio: float,
    use_all_rows: bool,
    progress_every: int,
    keep_column: str = "keep_sample",
) -> dict[str, object]:
    if torch is None or nn is None or not TORCH_AVAILABLE:
        raise RuntimeError("torch is required to train the transformer z encoder")
    if model.z_encoder is None or model.config.z_encoder_mode != "transformer":
        raise RuntimeError("model must be initialized with transformer z encoder mode")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if epochs <= 0:
        raise ValueError("epochs must be positive")

    original_rows = len(df)
    keep_filtered_rows = 0
    if not use_all_rows and keep_column in df.columns:
        keep_mask = df[keep_column].fillna(False).astype(bool)
        keep_filtered_rows = int((~keep_mask).sum())
        df = df.loc[keep_mask].copy()

    active_axes = resolve_style_axes(style_dim, style_profile=style_profile)
    s_columns = resolve_indexed_columns(df, "s_", expected_dim=len(active_axes))
    before_dropna = len(df)
    df = df.dropna(subset=[text_column] + s_columns).reset_index(drop=True)
    dropped_missing_rows = before_dropna - len(df)
    if len(df) < 2:
        raise ValueError("at least 2 clean labeled rows are required to train the transformer z encoder")

    texts = df[text_column].astype(str).tolist()
    s_matrix = df[s_columns].to_numpy(dtype=np.float32)
    branch_tensors: list[np.ndarray] = []
    feature_start = time.perf_counter()
    for idx, text in enumerate(texts, start=1):
        outputs = model.forward(text)
        branch_tensors.append(np.asarray(outputs["branch_tensor"], dtype=np.float32))
        maybe_print_progress("fit-z-encoder feature prep", idx, len(texts), feature_start, every=progress_every)

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(df))
    val_idx = np.asarray([], dtype=np.int64)
    train_idx = indices
    if 0.0 < val_ratio < 1.0 and len(df) >= 5:
        tentative_val = int(round(len(df) * val_ratio))
        val_rows = min(max(1, tentative_val), len(df) - 2)
        val_idx = indices[:val_rows]
        train_idx = indices[val_rows:]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = model.z_encoder.to(device)
    head = nn.Sequential(
        nn.Linear(model.config.z_dim, 128),
        nn.ReLU(),
        nn.Dropout(model.config.dropout),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, len(active_axes)),
        nn.Sigmoid(),
    ).to(device)
    optimizer = torch.optim.AdamW(
        list(encoder.parameters()) + list(head.parameters()),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    loss_fn = nn.L1Loss()

    best_metric = float("inf")
    best_encoder_state = {key: value.detach().cpu().clone() for key, value in encoder.state_dict().items()}
    best_head_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
    best_train_mae = None
    best_val_mae = None

    epoch_start_time = time.perf_counter()
    total_epochs = max(1, int(epochs))
    for epoch in range(1, total_epochs + 1):
        shuffled = train_idx[rng.permutation(len(train_idx))]
        encoder.train()
        head.train()
        train_loss_sum = 0.0
        train_seen = 0
        for start in range(0, len(shuffled), batch_size):
            batch_idx = shuffled[start : start + batch_size]
            batch_tensors = [branch_tensors[int(idx)] for idx in batch_idx]
            batch, attention_mask = pad_branch_tensor_batch(batch_tensors)
            targets = torch.as_tensor(s_matrix[batch_idx], dtype=torch.float32, device=device)

            optimizer.zero_grad(set_to_none=True)
            pred = head(encoder(batch.to(device), attention_mask.to(device)))
            loss = loss_fn(pred, targets)
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item()) * len(batch_idx)
            train_seen += len(batch_idx)

        train_mae = train_loss_sum / max(train_seen, 1)
        val_mae = train_mae
        if len(val_idx) > 0:
            encoder.eval()
            head.eval()
            val_loss_sum = 0.0
            val_seen = 0
            with torch.no_grad():
                for start in range(0, len(val_idx), batch_size):
                    batch_idx = val_idx[start : start + batch_size]
                    batch_tensors = [branch_tensors[int(idx)] for idx in batch_idx]
                    batch, attention_mask = pad_branch_tensor_batch(batch_tensors)
                    targets = torch.as_tensor(s_matrix[batch_idx], dtype=torch.float32, device=device)
                    pred = head(encoder(batch.to(device), attention_mask.to(device)))
                    batch_loss = loss_fn(pred, targets)
                    val_loss_sum += float(batch_loss.item()) * len(batch_idx)
                    val_seen += len(batch_idx)
            val_mae = val_loss_sum / max(val_seen, 1)

        monitored_metric = val_mae if len(val_idx) > 0 else train_mae
        if monitored_metric <= best_metric:
            best_metric = monitored_metric
            best_train_mae = train_mae
            best_val_mae = None if len(val_idx) == 0 else val_mae
            best_encoder_state = {key: value.detach().cpu().clone() for key, value in encoder.state_dict().items()}
            best_head_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
        maybe_print_progress(
            "fit-z-encoder epoch",
            epoch,
            total_epochs,
            epoch_start_time,
            every=1,
            unit="epochs",
            extra=f"train_mae={train_mae:.4f}, val_mae={val_mae:.4f}, best={best_metric:.4f}",
        )

    encoder.load_state_dict(best_encoder_state)
    head.load_state_dict(best_head_state)
    saved_encoder_path = model.save_z_encoder(encoder_model_path)

    z_matrix = encode_branch_tensors(encoder, branch_tensors, batch_size=max(1, int(batch_size)), device=device)

    decoder_train_mae = None
    decoder_val_mae = None
    if len(val_idx) > 0:
        eval_decoder = LinearZtoSDecoder(
            config=ZSDecoderConfig(model_path=zs_model_path, ridge_alpha=ridge_alpha),
            z_dim=model.config.z_dim,
            s_dim=len(active_axes),
        )
        eval_decoder.fit(z_matrix[train_idx], s_matrix[train_idx])
        decoder_train_mae = eval_decoder.mean_absolute_error(z_matrix[train_idx], s_matrix[train_idx])
        decoder_val_mae = eval_decoder.mean_absolute_error(z_matrix[val_idx], s_matrix[val_idx])

    decoder = LinearZtoSDecoder(
        config=ZSDecoderConfig(model_path=zs_model_path, ridge_alpha=ridge_alpha),
        z_dim=model.config.z_dim,
        s_dim=len(active_axes),
    )
    decoder.fit(z_matrix, s_matrix)
    saved_decoder_path = decoder.save(zs_model_path)

    if z_output_csv is not None:
        export_df = df.copy()
        for dim in range(z_matrix.shape[1]):
            export_df[f"z_{dim}"] = z_matrix[:, dim]
        z_output_csv.parent.mkdir(parents=True, exist_ok=True)
        export_df.to_csv(z_output_csv, index=False, encoding="utf-8-sig")

    return {
        "input_rows": int(original_rows),
        "rows_after_keep_filter": int(original_rows - keep_filtered_rows),
        "rows_used": int(len(df)),
        "keep_filtered_rows": int(keep_filtered_rows),
        "dropped_missing_rows": int(dropped_missing_rows),
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "style_dim": int(len(active_axes)),
        "style_profile": str(style_profile),
        "z_dim": int(model.config.z_dim),
        "encoder_head_train_mae": None if best_train_mae is None else round(float(best_train_mae), 6),
        "encoder_head_val_mae": None if best_val_mae is None else round(float(best_val_mae), 6),
        "decoder_train_mae": None if decoder_train_mae is None else round(float(decoder_train_mae), 6),
        "decoder_val_mae": None if decoder_val_mae is None else round(float(decoder_val_mae), 6),
        "encoder_model_path": str(saved_encoder_path),
        "zs_model_path": str(saved_decoder_path),
        "z_output_csv": None if z_output_csv is None else str(z_output_csv),
        "device": str(device),
        "keep_column": str(keep_column),
    }


def command_fit_z_encoder(args: argparse.Namespace) -> None:
    input_csv = Path(args.input_csv)
    df = pd.read_csv(input_csv)
    text_column = resolve_text_column(df, args.text_column)
    model = build_model(
        args,
        allow_missing_z_encoder_checkpoint=True,
        z_encoder_mode_override="transformer",
        load_z_encoder_checkpoint=bool(args.warm_start_z_encoder),
    )
    summary = train_transformer_z_encoder_from_dataframe(
        df=df,
        model=model,
        text_column=text_column,
        encoder_model_path=resolve_z_encoder_path(args.z_encoder_path),
        zs_model_path=Path(args.zs_model_path),
        z_output_csv=Path(args.z_output_csv) if args.z_output_csv else None,
        style_dim=args.style_dim,
        style_profile=args.style_profile,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        ridge_alpha=args.ridge_alpha,
        seed=args.seed,
        val_ratio=args.val_ratio,
        use_all_rows=args.use_all_rows,
        progress_every=args.progress_every,
        keep_column=args.keep_column,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


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
        keep_column=args.keep_column,
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
    style_profile = getattr(args, "style_profile", DEFAULT_STYLE_PROFILE)
    ensure_model_server_ready(args.base_url, args.timeout_sec)
    model = build_model(args)
    model_config = getattr(model, "config", None)
    decoder = LinearZtoSDecoder.load(Path(args.zs_model_path))
    profile = infer_style_profile(model=model, decoder=decoder, text=args.text, style_profile=style_profile)
    response_text, style_prompt, prompt_sections, response_meta = generate_response_from_profile(
        base_url=args.base_url,
        model_name=args.model_name,
        input_text=args.text,
        profile=profile,
        temperature=args.response_temperature,
        max_tokens=args.max_tokens,
        timeout_sec=args.timeout_sec,
        template_path=Path(args.prompt_template) if args.prompt_template else None,
        max_retries=args.response_max_retries,
        conditioning_mode=args.conditioning_mode,
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
        "expression_cues_text": str(profile["expression_cues_text"]),
        "trace_summary_text": str(profile.get("trace_summary_text", "")),
        "trace_lines": list(profile.get("trace_lines", [])),
        "appraisal_summary_text": str(profile.get("appraisal_summary_text", "")),
        "appraisal_lines": list(profile.get("appraisal_lines", [])),
        "appraisal_target": str(profile.get("appraisal_target", "")),
        "appraisal_tendency": str(profile.get("appraisal_tendency", "")),
        "ticks_run": int(profile.get("ticks_run", 0)),
        "termination_reason": str(profile.get("termination_reason", "")),
        "anti_softening_mode": str(profile["anti_softening_mode"]),
        "anti_softening_rules": list(profile["anti_softening_rules"]),
        "grounding_mode": str(profile["grounding_mode"]),
        "grounding_rules": list(profile["grounding_rules"]),
        "response_retry_count": int(response_meta["retry_count"]),
        "response_validation_errors": list(response_meta["validation_errors"]),
        "conditioning_mode": str(args.conditioning_mode),
        "prompt_sections": prompt_sections,
        "style_prompt": style_prompt,
        "llm_response": response_text,
        "decoder_model_path": str(args.zs_model_path),
        "z_encoder_mode": str(getattr(model_config, "z_encoder_mode", "unknown")),
        "z_encoder_path": str(getattr(model_config, "z_encoder_path", "")),
        "style_profile": str(style_profile),
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
    style_profile = getattr(args, "style_profile", DEFAULT_STYLE_PROFILE)
    ensure_model_server_ready(args.base_url, args.timeout_sec)
    model = build_model(args)
    model_config = getattr(model, "config", None)
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
            profile = infer_style_profile(model=model, decoder=decoder, text=text, style_profile=style_profile)
            response_text, style_prompt, prompt_sections, response_meta = generate_response_from_profile(
                base_url=args.base_url,
                model_name=args.model_name,
                input_text=text,
                profile=profile,
                temperature=args.response_temperature,
                max_tokens=args.max_tokens,
                timeout_sec=args.timeout_sec,
                template_path=Path(args.prompt_template) if args.prompt_template else None,
                max_retries=args.response_max_retries,
                conditioning_mode=args.conditioning_mode,
            )
            row = dict(record)
            row["status"] = "ok"
            row["error_message"] = ""
            row["style_tags"] = json.dumps(profile["style_tags"], ensure_ascii=False)
            row["style_summary_text"] = str(profile["style_summary_text"])
            row["style_summary_json"] = json.dumps(profile["style_summary"], ensure_ascii=False)
            row["expression_cues_text"] = str(profile["expression_cues_text"])
            row["trace_summary_text"] = str(profile.get("trace_summary_text", ""))
            row["trace_lines_json"] = json.dumps(profile.get("trace_lines", []), ensure_ascii=False)
            row["appraisal_summary_text"] = str(profile.get("appraisal_summary_text", ""))
            row["appraisal_lines_json"] = json.dumps(profile.get("appraisal_lines", []), ensure_ascii=False)
            row["appraisal_target"] = str(profile.get("appraisal_target", ""))
            row["appraisal_tendency"] = str(profile.get("appraisal_tendency", ""))
            row["ticks_run"] = int(profile.get("ticks_run", 0))
            row["termination_reason"] = str(profile.get("termination_reason", ""))
            row["anti_softening_mode"] = str(profile["anti_softening_mode"])
            row["anti_softening_rules"] = json.dumps(profile["anti_softening_rules"], ensure_ascii=False)
            row["grounding_mode"] = str(profile["grounding_mode"])
            row["grounding_rules"] = json.dumps(profile["grounding_rules"], ensure_ascii=False)
            row["response_retry_count"] = int(response_meta["retry_count"])
            row["response_validation_errors"] = json.dumps(response_meta["validation_errors"], ensure_ascii=False)
            row["conditioning_mode"] = str(args.conditioning_mode)
            row["prompt_sections"] = prompt_sections
            row["style_prompt"] = style_prompt
            row["llm_response"] = response_text
            row["decoder_model_path"] = str(args.zs_model_path)
            row["z_encoder_mode"] = str(getattr(model_config, "z_encoder_mode", "unknown"))
            row["z_encoder_path"] = str(getattr(model_config, "z_encoder_path", ""))
            row["style_profile"] = str(style_profile)
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
                     "expression_cues_text": str(profile["expression_cues_text"]),
                     "anti_softening_mode": str(profile["anti_softening_mode"]),
                     "anti_softening_rules": list(profile["anti_softening_rules"]),
                     "grounding_mode": str(profile["grounding_mode"]),
                     "grounding_rules": list(profile["grounding_rules"]),
                     "response_retry_count": int(response_meta["retry_count"]),
                     "response_validation_errors": list(response_meta["validation_errors"]),
                     "style_prompt": style_prompt,
                    "llm_response": response_text,
                    "decoder_model_path": str(args.zs_model_path),
                    "z_encoder_mode": str(getattr(model_config, "z_encoder_mode", "unknown")),
                    "z_encoder_path": str(getattr(model_config, "z_encoder_path", "")),
                    "style_profile": str(style_profile),
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

        maybe_print_progress("generate-response-batch", idx, len(df), start_time, every=args.progress_every)

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

RAW_AFFECT_AXIS_NAMES = [
    "hostility",
    "resentment",
    "despair",
    "volatility",
    "fearfulness",
    "shame",
    "relief",
    "trust",
]

NEGATIVE_RAW_AFFECT_AXES = [
    "hostility",
    "resentment",
    "despair",
    "volatility",
    "fearfulness",
    "shame",
]
EDGE_STYLE_AXES = [
    "tension",
    "sharpness",
    "heaviness",
    "urgency",
    "dominance",
]
SOFT_BIAS_AXES = [
    "softness",
    "calmness",
    "cooperativeness",
    "positivity",
    "warmth",
    "trust",
]

STYLE_AXIS_PROFILES = {
    "core32": list(STYLE_AXIS_NAMES),
    "extended40": list(STYLE_AXIS_NAMES) + list(RAW_AFFECT_AXIS_NAMES),
}

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
    "hostility": "적대감 낮음 <-> 적대감 높음",
    "resentment": "원망 적음 <-> 원망 강함",
    "despair": "희망 유지 <-> 절망감 큼",
    "volatility": "정서 변동 적음 <-> 정서 변동 큼",
    "fearfulness": "두려움 적음 <-> 두려움 큼",
    "shame": "수치심 적음 <-> 수치심 큼",
    "relief": "안도감 적음 <-> 안도감 큼",
    "trust": "경계함 <-> 신뢰함",
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
    "hostility": ("비적대적", "적대적"),
    "resentment": ("수용적", "원망강함"),
    "despair": ("희망유지", "절망감"),
    "volatility": ("안정적", "감정요동"),
    "fearfulness": ("담대함", "두려움"),
    "shame": ("자연스러움", "수치심"),
    "relief": ("긴장유지", "안도감"),
    "trust": ("경계함", "신뢰함"),
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
    "raw_negative_affect": [
        ("hostility", 1.0),
        ("resentment", 0.9),
        ("despair", 1.0),
        ("fearfulness", 0.8),
        ("shame", 0.7),
        ("trust", -0.6),
        ("relief", -0.6),
    ],
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
    "raw_negative_affect": "원초적부정정동",
}

ANTI_SOFTENING_TEXT_CUES = (
    "예민",
    "피곤",
    "지쳤",
    "짜증",
    "화나",
    "분노",
    "열받",
    "억울",
    "원망",
    "절망",
    "무기력",
    "불안",
    "초조",
    "두렵",
    "무섭",
    "괴롭",
    "답답",
    "힘들",
    "버겁",
    "상처",
    "서운",
    "지친",
)

RESPONSE_BULLET_PREFIXES = ("- ", "* ", "• ", "1. ", "1) ", "2. ", "2) ")
RESPONSE_HANGING_SUFFIXES = (
    "라면",
    "다면",
    "지만",
    "는데",
    "면서",
    "거나",
    "하며",
    "하고",
    "해서",
    "이며",
    "인데",
    "니까",
    "므로",
    "때문에",
    "같아",
    "같고",
    "같은",
    "처럼",
    "및",
    "또는",
)


def resolve_style_axes(style_dim: int | None = None, style_profile: str = DEFAULT_STYLE_PROFILE) -> list[str]:
    if style_profile not in STYLE_AXIS_PROFILES:
        valid = ", ".join(sorted(STYLE_AXIS_PROFILES))
        raise ValueError(f"unknown style_profile '{style_profile}'. valid profiles: {valid}")
    axes = STYLE_AXIS_PROFILES[style_profile]
    if style_dim is None:
        return list(axes)
    if style_dim <= 0:
        raise ValueError("style_dim must be positive")
    if style_dim > len(axes):
        raise ValueError(f"style_dim must be <= {len(axes)} for style_profile '{style_profile}'")
    return list(axes[:style_dim])


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


def style_vector_to_dict(
    values: np.ndarray | list[float],
    axis_names: list[str] | None = None,
    style_profile: str = DEFAULT_STYLE_PROFILE,
) -> dict[str, float]:
    axes = resolve_style_axes(len(values) if axis_names is None else len(axis_names), style_profile=style_profile)
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
        if denom <= 0.0:
            continue
        scores[macro_name] = float(np.clip(numerator / denom, 0.0, 1.0))
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


def format_style_summary_lines(style_summary: dict[str, float], top_n: int | None = None) -> list[str]:
    items = list(style_summary.items())
    if top_n is not None:
        items = sorted(items, key=lambda item: abs(item[1] - 0.5), reverse=True)[:top_n]
    return [
        f"{STYLE_MACRO_LABELS.get(name, name)}={float(score):.4f} ({describe_macro_level(score)})"
        for name, score in items
    ]


def compute_facial_expression_score(style_dict: dict[str, float]) -> float:
    terms = [
        ("emotional_openness", 1.0),
        ("tension", 0.9),
        ("directness", 0.5),
        ("urgency", 0.4),
        ("warmth", 0.3),
        ("softness", 0.2),
        ("sharpness", 0.2),
    ]
    numerator = 0.0
    denom = 0.0
    for axis_name, weight in terms:
        if axis_name not in style_dict:
            continue
        numerator += abs(weight) * float(style_dict[axis_name])
        denom += abs(weight)
    if denom <= 0.0:
        return 0.5
    return float(np.clip(numerator / denom, 0.0, 1.0))


def describe_facial_expression_mode(style_dict: dict[str, float]) -> str:
    softness = float(style_dict.get("softness", 0.5))
    sharpness = float(style_dict.get("sharpness", 0.5))
    warmth = float(style_dict.get("warmth", 0.5))
    tension = float(style_dict.get("tension", 0.5))
    if sharpness >= max(softness, warmth) + 0.12:
        return "날 선 얼굴 단서"
    if softness >= sharpness + 0.12 or warmth >= 0.68:
        return "부드러운 얼굴 단서"
    if tension >= 0.68:
        return "굳은 얼굴 단서"
    return "절제된 얼굴 단서"


def format_expression_cue_lines(style_dict: dict[str, float]) -> list[str]:
    facial_score = compute_facial_expression_score(style_dict)
    facial_mode = describe_facial_expression_mode(style_dict)
    return [f"표정 변화={facial_score:.4f} ({describe_macro_level(facial_score)}, {facial_mode})"]


def summarize_expression_cues(style_dict: dict[str, float]) -> str:
    return ", ".join(format_expression_cue_lines(style_dict))


def estimate_text_distress_score(text: str) -> float:
    normalized = str(text or "").strip()
    if not normalized:
        return 0.0
    hits = sum(1 for cue in ANTI_SOFTENING_TEXT_CUES if cue in normalized)
    return float(np.clip(hits / 3.0, 0.0, 1.0))


def build_anti_softening_policy(
    input_text: str,
    style_dict: dict[str, float],
    style_summary: dict[str, float],
    stim_vec: np.ndarray | Sequence[float] | None = None,
) -> tuple[str, list[str]]:
    distress_score = estimate_text_distress_score(input_text)
    raw_negative = float(style_summary.get("raw_negative_affect", 0.0))
    tension = float(style_summary.get("tension", 0.5))
    directness = float(style_summary.get("directness", 0.5))
    softness = float(style_dict.get("softness", 0.5))
    warmth = float(style_summary.get("warmth", 0.5))
    positivity = float(style_dict.get("positivity", 0.5))
    softening_pressure = (softness + warmth + positivity) / 3.0

    stim_arr = to_numpy_array(stim_vec, dtype=np.float32).reshape(-1) if stim_vec is not None else np.zeros(4, dtype=np.float32)
    norepinephrine = float(stim_arr[2]) if stim_arr.size >= 3 else 0.0
    melatonin = float(stim_arr[3]) if stim_arr.size >= 4 else 0.0

    strict_trigger = (
        distress_score >= 0.34
        or raw_negative >= 0.30
        or tension >= 0.60
        or directness >= 0.62
        or norepinephrine >= 0.55
    )
    guarded_trigger = strict_trigger or softening_pressure >= 0.72 or melatonin >= 0.50

    if strict_trigger:
        mode = "strict"
        rules = [
            "예민함, 피로감, 분노, 원망, 절망 같은 불편한 정서를 임의로 순화하지 않는다.",
            "사과, 위로, 안심, 응원 문구를 자동으로 덧붙이지 않는다.",
            "지나치게 공손하거나 상담원처럼 다독이는 말투를 피한다.",
            "불편함과 거친 결이 핵심이면 그 톤을 남긴 채 답한다.",
        ]
    elif guarded_trigger:
        mode = "guarded"
        rules = [
            "감정을 필요 이상으로 다독이거나 낙관적으로 정리하지 않는다.",
            "부드러운 위로나 배려 표현은 입력 내용과 스타일 신호가 뒷받침될 때만 사용한다.",
        ]
    else:
        mode = "neutral"
        rules = [
            "입력에 없는 위로나 공손함을 자동으로 덧붙이지 않는다.",
        ]
    return mode, rules


def format_anti_softening_lines(rules: list[str]) -> list[str]:
    return [f"- {rule}" for rule in rules]


def build_grounding_policy(
    input_text: str,
    style_summary: dict[str, float],
    anti_softening_mode: str,
) -> tuple[str, list[str]]:
    distress_score = estimate_text_distress_score(input_text)
    raw_negative = float(style_summary.get("raw_negative_affect", 0.0))
    warmth = float(style_summary.get("warmth", 0.5))
    directness = float(style_summary.get("directness", 0.5))

    needs_strong_grounding = (
        anti_softening_mode in {"strict", "guarded"}
        or distress_score >= 0.34
        or raw_negative >= 0.20
        or warmth >= 0.70
        or directness >= 0.65
    )

    if needs_strong_grounding:
        mode = "grounded"
        rules = [
            "첫 문장에서 사용자의 현재 감정이나 처지를 짧게 짚어 주고 바로 답한다.",
            "사용자를 훈계하거나 판정하지 않는다.",
            "명령조, 협박조, 단정적 예언을 피한다.",
            "사용자 입장에서 확인되지 않은 사실이나 관계를 임의로 단정하지 않는다.",
            "조언이 필요하면 강요형 대신 제안형 표현을 우선한다.",
        ]
    else:
        mode = "light"
        rules = [
            "첫 문장은 입력의 정서와 직접 연결되게 시작한다.",
            "사용자를 평가하거나 결론을 대신 내려주지 않는다.",
        ]
    return mode, rules


def format_grounding_lines(rules: list[str]) -> list[str]:
    return [f"- {rule}" for rule in rules]


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
    api_key: str | None = None,
    response_format: dict[str, Any] | None = None,
    reasoning_effort: str | None = None,
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
            api_key=api_key,
            response_format=response_format,
            reasoning_effort=reasoning_effort,
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


def normalize_style_dict(
    style_dict: dict,
    key_name: str,
    expected_axes: list[str] | None = None,
    style_profile: str = DEFAULT_STYLE_PROFILE,
) -> dict[str, float]:
    if key_name not in style_dict or not isinstance(style_dict[key_name], dict):
        raise ValueError(f"missing '{key_name}' object in model output")
    axes = resolve_style_axes(style_profile=style_profile) if expected_axes is None else expected_axes
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
    return validate_plain_response_text(response)


def clean_plain_response_text(response: str) -> str:
    cleaned = str(response).replace("\r", "\n").strip()
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def split_response_segments(response: str) -> list[str]:
    segments = re.split(r"[.!?…\n]+", response)
    cleaned_segments: list[str] = []
    for segment in segments:
        normalized = segment.strip(" \"'“”‘’()[]{}")
        if normalized:
            cleaned_segments.append(normalized)
    return cleaned_segments


def normalize_segment_key(segment: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]+", "", segment).lower()


def looks_like_incomplete_response(response: str) -> bool:
    tail = response.rstrip(" \"'“”‘’)]}")
    if not tail:
        return False
    if tail[-1] in ",:;/-(":
        return True
    lowered = tail.lower()
    return any(lowered.endswith(suffix) for suffix in RESPONSE_HANGING_SUFFIXES)


def validate_plain_response_text(response: str) -> str:
    normalized = clean_plain_response_text(response)
    if not normalized:
        raise ValueError("empty response returned from model output")

    stripped = normalized.lstrip()
    if stripped.startswith(("```", "{", "[")):
        raise ValueError("response must be plain text, not JSON or markdown")
    if any(stripped.startswith(prefix) for prefix in RESPONSE_BULLET_PREFIXES):
        raise ValueError("response must be plain sentences, not bullet points")
    if len(re.findall(r"[0-9A-Za-z가-힣]", normalized)) < 2:
        raise ValueError("response is too short to be meaningful")

    segments = split_response_segments(normalized)
    if not segments:
        raise ValueError("response does not contain a readable sentence")

    seen_segments: set[str] = set()
    for segment in segments:
        key = normalize_segment_key(segment)
        if len(key) < 4:
            continue
        if key in seen_segments:
            raise ValueError("response repeats the same sentence or clause")
        seen_segments.add(key)

    if len(segments) >= 2:
        last_key = normalize_segment_key(segments[-1])
        prev_key = normalize_segment_key(segments[-2])
        if last_key and last_key == prev_key:
            raise ValueError("response repeats the same ending sentence")

    if looks_like_incomplete_response(normalized):
        raise ValueError("response ends mid-sentence or with a hanging connective")

    return normalized


def request_plain_text_response(
    base_url: str,
    model_name: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout_sec: int,
    max_retries: int,
    validator: Callable[[str], str] | None = None,
    retry_instruction: str | None = None,
    system_prompt: str = "Return a plain Korean response only. Do not return JSON.",
    api_key: str | None = None,
    reasoning_effort: str | None = None,
) -> tuple[str, str, dict[str, object]]:
    last_raw = ""
    validation_errors: list[str] = []
    for attempt in range(max_retries + 1):
        retry_suffix = ""
        if attempt > 0:
            retry_reason = validation_errors[-1] if validation_errors else "직전 응답이 형식 검증을 통과하지 못했다."
            retry_suffix = (
                "\n\n[RETRY_INSTRUCTION]\n"
                + (
                    retry_instruction
                    or "직전 응답은 plain Korean response 규칙을 어겼다. 같은 문장 반복, 미완성 문장, bullet/JSON을 피하고 자연스러운 한국어 평문으로만 다시 출력하라."
                )
                + f"\n- 직전 문제: {retry_reason}"
            )
        raw = call_openai_compatible_chat(
            base_url=base_url,
            model_name=model_name,
            prompt=prompt + retry_suffix,
            temperature=temperature if attempt == 0 else 0.0,
            max_tokens=max_tokens,
            timeout_sec=timeout_sec,
            system_prompt=system_prompt,
            api_key=api_key,
            reasoning_effort=reasoning_effort,
        )
        last_raw = raw
        try:
            validated = validator(raw) if validator is not None else clean_plain_response_text(raw)
            return (
                validated,
                raw,
                {
                    "attempt_count": int(attempt + 1),
                    "retry_count": int(attempt),
                    "validation_errors": list(validation_errors),
                },
            )
        except Exception as exc:
            validation_errors.append(str(exc))
            continue
    last_error = validation_errors[-1] if validation_errors else "unknown validation error"
    raise ValueError(f"invalid plain-text response after retries: {last_error}. raw={last_raw[:500]}")


def call_openai_compatible_chat(
    base_url: str,
    model_name: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout_sec: int,
    system_prompt: str = "Return JSON only.",
    api_key: str | None = None,
    response_format: dict[str, Any] | None = None,
    reasoning_effort: str | None = None,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    is_openai_api = bool(api_key and "api.openai.com" in str(base_url).lower())
    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    }
    if not is_openai_api:
        payload["temperature"] = temperature
    elif temperature not in (0, 0.0, 1, 1.0):
        payload["temperature"] = temperature
    if is_openai_api:
        payload["max_completion_tokens"] = max_tokens
    else:
        payload["max_tokens"] = max_tokens
    if response_format is not None:
        payload["response_format"] = response_format
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_sec) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        error_text = exc.read().decode("utf-8", errors="replace")
        raise ValueError(f"HTTP {exc.code} from {url}: {error_text[:1000]}") from exc
    choices = body.get("choices", [])
    if not choices:
        raise ValueError("no choices returned from model server")
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, list):
        text_parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
        content = "".join(text_parts)
    if not isinstance(content, str):
        raise ValueError("invalid content returned from model server")
    return content


def ensure_model_server_ready(base_url: str, timeout_sec: int, api_key: str | None = None) -> None:
    models_url = base_url.rstrip("/") + "/models"
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(models_url, headers=headers, method="GET")
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


def filter_present_axes(candidate_axes: list[str], active_axes: list[str]) -> list[str]:
    active = set(active_axes)
    return [axis for axis in candidate_axes if axis in active]


def compute_axis_mean(style_dict: dict[str, float], axes: list[str]) -> float:
    if not axes:
        return 0.0
    return float(np.mean([float(style_dict[axis]) for axis in axes]))


def compute_axis_max(style_dict: dict[str, float], axes: list[str]) -> float:
    if not axes:
        return 0.0
    return float(np.max([float(style_dict[axis]) for axis in axes]))


def compute_axis_consistency(style_a: dict[str, float], style_b: dict[str, float], axes: list[str]) -> float:
    if not axes:
        return 0.0
    return compute_consistency(style_a, style_b, axes)


def compute_style_bias_metrics(
    style: dict[str, float],
    style_hat: dict[str, float],
    active_axes: list[str],
) -> dict[str, float | bool]:
    negative_axes = filter_present_axes(NEGATIVE_RAW_AFFECT_AXES, active_axes)
    edge_axes = filter_present_axes(EDGE_STYLE_AXES, active_axes)
    soft_axes = filter_present_axes(SOFT_BIAS_AXES, active_axes)
    metrics: dict[str, float | bool] = {
        "negative_raw_mean": compute_axis_mean(style, negative_axes),
        "negative_raw_max": compute_axis_max(style, negative_axes),
        "edge_mean": compute_axis_mean(style, edge_axes),
        "edge_max": compute_axis_max(style, edge_axes),
        "soft_bias_mean": compute_axis_mean(style, soft_axes),
        "consistency_negative_raw_l1": compute_axis_consistency(style, style_hat, negative_axes),
        "consistency_edge_l1": compute_axis_consistency(style, style_hat, edge_axes),
    }
    return metrics


def classify_style_balance_bucket(
    metrics: dict[str, float | bool],
    *,
    raw_affect_min: float,
    edge_spike_min: float,
    soft_bias_floor: float,
) -> str:
    negative_raw_max = float(metrics.get("negative_raw_max", 0.0))
    edge_max = float(metrics.get("edge_max", 0.0))
    soft_bias_mean = float(metrics.get("soft_bias_mean", 0.0))
    if negative_raw_max >= raw_affect_min:
        return "rare_raw"
    if edge_max >= edge_spike_min:
        return "edgy"
    if soft_bias_mean >= soft_bias_floor and negative_raw_max < raw_affect_min and edge_max < edge_spike_min:
        return "soft_safe"
    return "mixed"


def compute_style_rebalance_score(metrics: dict[str, float | bool]) -> float:
    negative_raw_mean = float(metrics.get("negative_raw_mean", 0.0))
    negative_raw_max = float(metrics.get("negative_raw_max", 0.0))
    edge_mean = float(metrics.get("edge_mean", 0.0))
    edge_max = float(metrics.get("edge_max", 0.0))
    soft_bias_mean = float(metrics.get("soft_bias_mean", 0.0))
    return (
        1.8 * negative_raw_max
        + 0.9 * negative_raw_mean
        + 1.0 * edge_mean
        + 0.6 * edge_max
        - 1.15 * soft_bias_mean
    )


def decide_axis_aware_keep(
    consistency_l1: float,
    metrics: dict[str, float | bool],
    *,
    keep_threshold: float,
    rare_keep_threshold: float,
    raw_affect_min: float,
    edge_spike_min: float,
    soft_bias_floor: float,
    raw_affect_keep_threshold: float,
    edge_keep_threshold: float,
) -> tuple[bool, str, str]:
    bucket = classify_style_balance_bucket(
        metrics,
        raw_affect_min=raw_affect_min,
        edge_spike_min=edge_spike_min,
        soft_bias_floor=soft_bias_floor,
    )
    negative_raw_l1 = float(metrics.get("consistency_negative_raw_l1", 0.0))
    edge_l1 = float(metrics.get("consistency_edge_l1", 0.0))
    rare_candidate = bucket in {"rare_raw", "edgy"}
    oversoft_candidate = bucket == "soft_safe"
    base_keep = consistency_l1 <= keep_threshold and not oversoft_candidate
    rescue_keep = (
        rare_candidate
        and consistency_l1 <= rare_keep_threshold
        and negative_raw_l1 <= raw_affect_keep_threshold
        and edge_l1 <= edge_keep_threshold
    )
    if rescue_keep:
        return True, "rare_affect_rescue", bucket
    if base_keep:
        return True, "consistent_nonsoft", bucket
    if oversoft_candidate and consistency_l1 <= keep_threshold:
        return False, "oversoft_trim", bucket
    if consistency_l1 > rare_keep_threshold:
        return False, "consistency_too_high", bucket
    return False, "axis_keep_rejected", bucket


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
            "- 감정 표현 수단으로 어휘, 문장 리듬과 함께 필요하면 짧은 표정 변화 단서를 활용할 수 있다.",
            "- 마크다운, bullet, 번호 목록, 따옴표 인용을 쓰지 않는다.",
            "- z 는 직접 설명하지 말고 내부 상태 힌트로만 사용한다.",
            "- 설명 문장 없이 JSON object 하나만 출력한다.",
        ]
    )


def _default_response_generation_template() -> str:
    return "\n".join(
        [
            "[ROLE]",
            "당신은 감정 상태에 맞는 말투와 밀도로 답하는 한국어 응답 생성기다.",
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
            "[ANTI_SOFTENING_RULES]",
            "{{anti_softening_lines}}",
            "",
            "[INSTRUCTIONS]",
            "- 사용자 입력의 내용에 직접 답한다.",
            "- STYLE_TAGS와 STYLE_SUMMARY만 참고해 말투, 거리감, 표현 밀도를 조절한다.",
            "- ANTI_SOFTENING_RULES가 있으면 반드시 지킨다.",
            "- 스타일을 설명하지 말고, 그 스타일로 자연스럽게 답한다.",
            "- 한국어 평문으로만 2~5문장 이내로 답한다.",
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


TRACE_AXIS_LABELS = (
    "접근/밀어붙임",
    "안정/완충",
    "긴장/날카로움",
    "피로/둔화",
)
APPRAISAL_LABELS = {
    "goal_blockage": "목표 차단",
    "social_exclusion": "배제감",
    "injustice": "억울함/불공정",
    "control_loss": "통제 상실",
    "exhaustion": "소진",
    "threat": "위협감",
}
BLOCKAGE_HINTS = {
    "막혔",
    "안돼",
    "안 되",
    "실패",
    "꼬였",
    "답답",
    "막막",
    "손에 안 잡",
    "지연",
    "blocked",
    "fail",
    "stuck",
}
EXCLUSION_HINTS = {
    "제외",
    "빼고",
    "무시",
    "소외",
    "왕따",
    "혼자",
    "ignored",
    "left out",
    "excluded",
}
INJUSTICE_HINTS = {
    "억울",
    "부당",
    "불공평",
    "불공정",
    "차별",
    "편파",
    "unfair",
    "unjust",
}
CONTROL_LOSS_HINTS = {
    "어쩔 수 없",
    "통제",
    "불확실",
    "마음대로 안",
    "손에 안 잡",
    "답이 안",
    "불안",
    "uncertain",
    "out of control",
}
TARGET_OTHER_HINTS = {
    "상사",
    "대표",
    "팀장",
    "회사",
    "그 사람",
    "걔",
    "they",
    "boss",
    "manager",
}
TARGET_SELF_HINTS = {
    "내가",
    "나는",
    "저는",
    "나만",
    "내 탓",
    "i ",
    "me ",
    "myself",
}
TARGET_SITUATION_HINTS = {
    "업무",
    "일",
    "상황",
    "결과",
    "심사",
    "야근",
    "deadline",
    "result",
    "work",
}


def describe_trace_axis_level(value: float) -> str:
    if value >= 0.75:
        return "매우 높음"
    if value >= 0.60:
        return "높음"
    if value >= 0.40:
        return "중간"
    if value >= 0.20:
        return "낮음"
    return "매우 낮음"


def summarize_trace_stim_signature(stim_vec: Sequence[float] | np.ndarray, top_n: int = 2) -> str:
    stim = np.asarray(stim_vec, dtype=np.float32).reshape(-1)
    if stim.size == 0:
        return "정서 축 정보 없음"
    top_indices = np.argsort(-stim)[: max(1, min(top_n, stim.size))]
    parts = [
        f"{TRACE_AXIS_LABELS[idx]} {describe_trace_axis_level(float(stim[idx]))}"
        for idx in top_indices
    ]
    return ", ".join(parts)


def hint_fraction(text: str, hints: set[str], cap: float = 1.0) -> float:
    hits = sum(1 for token in hints if token in text)
    return min(hits / 3.0, cap)


def _segment_branch(branch: Sequence[Any], start: int, end: int) -> Sequence[Any]:
    if not branch:
        return []
    start = max(0, min(start, len(branch)))
    end = max(start + 1, min(end, len(branch)))
    return branch[start:end]


def build_trace_profile(
    *,
    pruned_branch_log: Sequence[Any],
    dominant_branch: Sequence[Any],
    n_neurons: int,
    termination_reason: str,
    ticks_run: int,
) -> dict[str, object]:
    active_records = [record for record in pruned_branch_log if getattr(record, "active_nodes", None)]
    active_counts = np.asarray([len(record.active_nodes) for record in pruned_branch_log], dtype=np.float32)
    edge_counts = np.asarray([len(record.edges_fired) for record in pruned_branch_log], dtype=np.float32)
    first_active_tick = int(active_records[0].tick) if active_records else -1
    last_active_tick = int(active_records[-1].tick) if active_records else -1
    active_window_ticks = int(len(active_records))
    mean_active_nodes = float(active_counts.mean()) if active_counts.size else 0.0
    max_active_nodes = int(active_counts.max()) if active_counts.size else 0
    mean_edges_fired = float(edge_counts.mean()) if edge_counts.size else 0.0
    max_edges_fired = int(edge_counts.max()) if edge_counts.size else 0
    branch_len = int(len(dominant_branch))

    thirds = max(1, branch_len // 3)
    early = _segment_branch(dominant_branch, 0, thirds)
    middle = _segment_branch(dominant_branch, max(0, branch_len // 3), max(1, (2 * branch_len) // 3))
    late = _segment_branch(dominant_branch, max(0, (2 * branch_len) // 3), branch_len)

    def summarize_phase(name: str, steps: Sequence[Any]) -> str:
        if not steps:
            return f"{name}: 유효한 branch 단계가 거의 없음"
        phase_stim = np.mean(
            [
                np.asarray(getattr(step, "stim_vec", np.zeros(4, dtype=np.float32)), dtype=np.float32).reshape(-1)[:4]
                for step in steps
            ],
            axis=0,
        )
        phase_k = float(np.mean([float(getattr(step, "K", 0.0)) for step in steps]))
        tick_start = int(getattr(steps[0], "tick", 0))
        tick_end = int(getattr(steps[-1], "tick", tick_start))
        return (
            f"{name}: tick {tick_start}-{tick_end}, "
            f"K 평균 {phase_k:.2f}, "
            f"{summarize_trace_stim_signature(phase_stim)}"
        )

    phase_lines = [
        summarize_phase("초기", early),
        summarize_phase("중기", middle),
        summarize_phase("후기", late),
    ]
    active_ratio = mean_active_nodes / max(1, n_neurons)
    if active_ratio >= 0.70:
        density_text = "활성 노드 밀도가 매우 높아 포화에 가까움"
    elif active_ratio >= 0.40:
        density_text = "활성 노드 밀도가 중간 이상으로 넓게 퍼짐"
    elif active_ratio > 0.0:
        density_text = "활성 노드 밀도가 비교적 좁아 선택적으로 움직임"
    else:
        density_text = "유의미한 활성 노드가 거의 없음"

    if termination_reason == "max_ticks":
        termination_text = "상한 tick에 닿을 때까지 감정 궤적이 지속됨"
    elif termination_reason == "stable_convergence":
        termination_text = "후반부 변화량이 안정되어 수렴 종료됨"
    elif termination_reason == "delta_k":
        termination_text = "활성 변화량이 작아져 조기 종료됨"
    else:
        termination_text = "종료 사유가 명확하지 않음"

    trace_lines = [
        f"전체 tick {ticks_run}, dominant branch 길이 {branch_len}, 종료={termination_reason}",
        (
            f"첫 활성 tick {first_active_tick}, 마지막 활성 tick {last_active_tick}, "
            f"활성 구간 {active_window_ticks} tick"
            if first_active_tick >= 0
            else "유의미한 활성 tick이 거의 없었음"
        ),
        (
            f"평균 활성 노드 {mean_active_nodes:.1f}/{n_neurons}, 최대 {max_active_nodes}, "
            f"평균 firing edge {mean_edges_fired:.1f}, 최대 {max_edges_fired}"
        ),
        density_text,
        termination_text,
        *phase_lines,
    ]
    trace_summary_text = " / ".join(trace_lines[:5])
    return {
        "trace_lines": trace_lines,
        "trace_summary_text": trace_summary_text,
        "first_active_tick": first_active_tick,
        "last_active_tick": last_active_tick,
        "active_window_ticks": active_window_ticks,
        "mean_active_nodes": mean_active_nodes,
        "max_active_nodes": max_active_nodes,
        "mean_edges_fired": mean_edges_fired,
        "max_edges_fired": max_edges_fired,
        "ticks_run": int(ticks_run),
        "termination_reason": str(termination_reason),
        "dominant_branch_len": branch_len,
    }


def describe_appraisal_level(value: float) -> str:
    if value >= 0.75:
        return "매우 높음"
    if value >= 0.55:
        return "높음"
    if value >= 0.35:
        return "중간"
    if value >= 0.15:
        return "낮음"
    return "매우 낮음"


def build_appraisal_profile(
    *,
    input_text: str,
    stim_vec: Sequence[float] | np.ndarray,
    trace_profile: dict[str, object],
    style_summary: dict[str, float],
) -> dict[str, object]:
    text = str(input_text).lower()
    stim = np.asarray(stim_vec, dtype=np.float32).reshape(-1)
    dopamine = float(stim[0]) if stim.size > 0 else 0.0
    serotonin = float(stim[1]) if stim.size > 1 else 0.0
    norepinephrine = float(stim[2]) if stim.size > 2 else 0.0
    melatonin = float(stim[3]) if stim.size > 3 else 0.0
    raw_negative = float(style_summary.get("raw_negative_affect", 0.0))
    active_window_ticks = int(trace_profile.get("active_window_ticks", 0))
    ticks_run = max(1, int(trace_profile.get("ticks_run", 1)))
    active_window_ratio = float(active_window_ticks) / float(ticks_run)

    goal_blockage = max(
        hint_fraction(text, BLOCKAGE_HINTS),
        min(1.0, 0.45 * norepinephrine + 0.20 * raw_negative + 0.20 * (1.0 - serotonin)),
    )
    social_exclusion = max(
        hint_fraction(text, EXCLUSION_HINTS),
        min(1.0, 0.70 * hint_fraction(text, EXCLUSION_HINTS) + 0.20 * raw_negative),
    )
    injustice = max(
        hint_fraction(text, INJUSTICE_HINTS),
        min(1.0, 0.50 * social_exclusion + 0.35 * raw_negative + 0.15 * norepinephrine),
    )
    control_loss = max(
        hint_fraction(text, CONTROL_LOSS_HINTS),
        min(1.0, 0.40 * norepinephrine + 0.25 * (1.0 - dopamine) + 0.20 * (1.0 - serotonin)),
    )
    exhaustion = max(
        hint_fraction(text, FATIGUE_HINTS),
        min(1.0, 0.65 * melatonin + 0.15 * active_window_ratio),
    )
    threat = max(
        hint_fraction(text, THREAT_HINTS.union(ALERT_HINTS)),
        min(1.0, 0.60 * norepinephrine + 0.15 * control_loss + 0.10 * raw_negative),
    )

    target_scores = {
        "self": hint_fraction(text, TARGET_SELF_HINTS) + 0.20 * exhaustion,
        "other": hint_fraction(text, TARGET_OTHER_HINTS) + 0.40 * social_exclusion + 0.20 * injustice,
        "situation": hint_fraction(text, TARGET_SITUATION_HINTS) + 0.30 * goal_blockage + 0.20 * control_loss,
    }
    sorted_targets = sorted(target_scores.items(), key=lambda item: item[1], reverse=True)
    target_label = sorted_targets[0][0]
    if len(sorted_targets) >= 2 and abs(sorted_targets[0][1] - sorted_targets[1][1]) < 0.12:
        target_label = "mixed"

    tendencies = {
        "대치/표출": 0.55 * injustice + 0.45 * social_exclusion + 0.20 * raw_negative,
        "방어/경계": 0.60 * threat + 0.35 * control_loss,
        "회복/후퇴": 0.70 * exhaustion + 0.20 * (1.0 - dopamine),
        "정리/수습": 0.50 * goal_blockage + 0.30 * control_loss + 0.20 * serotonin,
    }
    dominant_tendency = max(tendencies.items(), key=lambda item: item[1])[0]

    appraisal_scores = {
        "goal_blockage": float(min(goal_blockage, 1.0)),
        "social_exclusion": float(min(social_exclusion, 1.0)),
        "injustice": float(min(injustice, 1.0)),
        "control_loss": float(min(control_loss, 1.0)),
        "exhaustion": float(min(exhaustion, 1.0)),
        "threat": float(min(threat, 1.0)),
    }
    sorted_axes = sorted(appraisal_scores.items(), key=lambda item: item[1], reverse=True)
    top_axes = sorted_axes[:3]
    top_axis_text = ", ".join(
        f"{APPRAISAL_LABELS[key]} {describe_appraisal_level(value)}" for key, value in top_axes
    )
    target_text_map = {
        "self": "감정의 주된 방향은 자기 상태와 자기 해석 쪽이다.",
        "other": "감정의 주된 방향은 타인이나 관계 쪽이다.",
        "situation": "감정의 주된 방향은 상황이나 업무 맥락 쪽이다.",
        "mixed": "감정의 주된 방향이 자기, 타인, 상황에 걸쳐 섞여 있다.",
    }
    appraisal_lines = [
        f"핵심 appraisal: {top_axis_text}",
        target_text_map[target_label],
        f"현재 행동 성향은 '{dominant_tendency}' 쪽이 상대적으로 우세하다.",
        f"stimulus 요약: {summarize_trace_stim_signature(stim)}",
        f"trace 요약: {str(trace_profile.get('trace_summary_text', ''))}",
    ]
    appraisal_summary_text = " / ".join(appraisal_lines[:3])
    return {
        "appraisal_scores": appraisal_scores,
        "appraisal_lines": appraisal_lines,
        "appraisal_summary_text": appraisal_summary_text,
        "appraisal_target": target_label,
        "appraisal_tendency": dominant_tendency,
    }


def build_response_generation_prompt(
    input_text: str,
    style_dict: dict[str, float],
    style_tags: list[str],
    style_summary: dict[str, float],
    anti_softening_rules: list[str] | None = None,
    grounding_rules: list[str] | None = None,
    template_path: Path | None = None,
) -> str:
    template = load_response_generation_template(template_path)
    condensed_tags = style_tags[:4]
    condensed_summary = format_style_summary_lines(style_summary, top_n=3)
    return render_template(
        template,
        {
            "input_text": input_text.strip(),
            "style_tags": ", ".join(condensed_tags) if condensed_tags else "(none)",
              "style_summary_lines": "\n".join(condensed_summary) if condensed_summary else "(none)",
              "anti_softening_lines": "\n".join(format_anti_softening_lines(anti_softening_rules or []))
              if anti_softening_rules
              else "- 입력에 없는 위로나 공손함을 자동으로 덧붙이지 않는다.",
              "grounding_lines": "\n".join(format_grounding_lines(grounding_rules or []))
              if grounding_rules
              else "- 첫 문장은 입력의 정서와 직접 연결되게 시작한다.",
              "expression_cue_lines": "\n".join(format_expression_cue_lines(style_dict)),
              "style_vector_lines": format_style_vector_lines(style_dict),
          },
      )


def build_trace_generation_prompt(
    input_text: str,
    trace_lines: Sequence[str],
    anti_softening_rules: list[str] | None = None,
    grounding_rules: list[str] | None = None,
) -> str:
    trace_block = "\n".join(f"- {line}" for line in trace_lines) if trace_lines else "- 유효한 trace 정보 없음"
    anti_block = (
        "\n".join(format_anti_softening_lines(anti_softening_rules or []))
        if anti_softening_rules
        else "- 입력에 없는 위로나 공손함을 자동으로 덧붙이지 않는다."
    )
    grounding_block = (
        "\n".join(format_grounding_lines(grounding_rules or []))
        if grounding_rules
        else "- 첫 문장은 입력의 정서와 직접 연결되게 시작한다."
    )
    return "\n".join(
        [
            "[ROLE]",
            "당신은 내부 감정 궤적을 읽고 그 결을 유지한 채 한국어로 답하는 응답 생성기다.",
            "",
            "[USER_INPUT]",
            input_text.strip(),
            "",
            "[RAW_TRACE]",
            trace_block,
            "",
            "[ANTI_SOFTENING_RULES]",
            anti_block,
            "",
            "[GROUNDING_RULES]",
            grounding_block,
            "",
            "[INSTRUCTIONS]",
            "- 사용자 입력의 내용에 직접 답한다.",
            "- RAW_TRACE를 보고 감정이 어떻게 시작되고 유지되고 수렴했는지 반영한다.",
            "- RAW_TRACE의 숫자나 섹션 이름을 그대로 언급하지 않는다.",
            "- 감정의 거친 결, 짜증, 예민함, 피로, 소진감이 핵심이면 그 결을 남긴다.",
            "- 불필요하게 달래거나 과잉 해석하지 않는다.",
            "- 한국어 평문으로만 2~5문장 이내로 답한다.",
            "- 같은 문장이나 핵심 구절을 반복하지 않는다.",
            "- 문장을 중간에 끊거나 조건절로 끝내지 않는다. 마지막 문장은 완결된 문장으로 끝낸다.",
            "- bullet, markdown, JSON, 코드블록을 쓰지 않는다.",
        ]
    )


def build_appraisal_generation_prompt(
    input_text: str,
    appraisal_lines: Sequence[str],
    anti_softening_rules: list[str] | None = None,
    grounding_rules: list[str] | None = None,
) -> str:
    appraisal_block = "\n".join(f"- {line}" for line in appraisal_lines) if appraisal_lines else "- appraisal 정보 없음"
    anti_block = (
        "\n".join(format_anti_softening_lines(anti_softening_rules or []))
        if anti_softening_rules
        else "- 입력에 없는 위로나 공손함을 자동으로 덧붙이지 않는다."
    )
    grounding_block = (
        "\n".join(format_grounding_lines(grounding_rules or []))
        if grounding_rules
        else "- 첫 문장은 입력의 정서와 직접 연결되게 시작한다."
    )
    return "\n".join(
        [
            "[ROLE]",
            "당신은 사용자의 감정을 appraisal 관점에서 읽고 그 결을 유지한 채 한국어로 답하는 응답 생성기다.",
            "",
            "[USER_INPUT]",
            input_text.strip(),
            "",
            "[APPRAISAL_TRACE]",
            appraisal_block,
            "",
            "[ANTI_SOFTENING_RULES]",
            anti_block,
            "",
            "[GROUNDING_RULES]",
            grounding_block,
            "",
            "[INSTRUCTIONS]",
            "- 사용자 입력의 내용에 직접 답한다.",
            "- APPRAISAL_TRACE를 보고 왜 이런 감정이 생겼는지, 감정이 어디를 향하는지, 어떤 행동 성향이 우세한지 반영한다.",
            "- APPRAISAL_TRACE의 숫자나 섹션 이름을 그대로 언급하지 않는다.",
            "- 감정의 핵심이 짜증, 배제감, 억울함, 소진, 위협감이라면 그 결을 남긴다.",
            "- 불필요하게 달래거나 도덕적 훈계를 하지 않는다.",
            "- 한국어 평문으로만 2~5문장 이내로 답한다.",
            "- 같은 문장이나 핵심 구절을 반복하지 않는다.",
            "- 문장을 중간에 끊거나 조건절로 끝내지 않는다. 마지막 문장은 완결된 문장으로 끝낸다.",
            "- bullet, markdown, JSON, 코드블록을 쓰지 않는다.",
        ]
    )


def build_hybrid_trace_generation_prompt(
    input_text: str,
    style_dict: dict[str, float],
    style_tags: list[str],
    style_summary: dict[str, float],
    trace_lines: Sequence[str],
    anti_softening_rules: list[str] | None = None,
    grounding_rules: list[str] | None = None,
) -> str:
    condensed_tags = ", ".join(style_tags[:4]) if style_tags else "(none)"
    summary_lines = "\n".join(format_style_summary_lines(style_summary, top_n=3))
    trace_block = "\n".join(f"- {line}" for line in trace_lines) if trace_lines else "- 유효한 trace 정보 없음"
    anti_block = (
        "\n".join(format_anti_softening_lines(anti_softening_rules or []))
        if anti_softening_rules
        else "- 입력에 없는 위로나 공손함을 자동으로 덧붙이지 않는다."
    )
    grounding_block = (
        "\n".join(format_grounding_lines(grounding_rules or []))
        if grounding_rules
        else "- 첫 문장은 입력의 정서와 직접 연결되게 시작한다."
    )
    return "\n".join(
        [
            "[ROLE]",
            "당신은 감정 궤적과 스타일 요약을 함께 참고해 한국어로 답하는 응답 생성기다.",
            "",
            "[USER_INPUT]",
            input_text.strip(),
            "",
            "[RAW_TRACE]",
            trace_block,
            "",
            "[STYLE_TAGS]",
            condensed_tags,
            "",
            "[STYLE_SUMMARY]",
            summary_lines if summary_lines else "(none)",
            "",
            "[ANTI_SOFTENING_RULES]",
            anti_block,
            "",
            "[GROUNDING_RULES]",
            grounding_block,
            "",
            "[INSTRUCTIONS]",
            "- 사용자 입력의 내용에 직접 답한다.",
            "- RAW_TRACE를 우선 참고해 감정의 흐름과 결을 잡는다.",
            "- STYLE_TAGS와 STYLE_SUMMARY는 말투 밀도와 거리감을 미세 조정하는 데만 쓴다.",
            "- RAW_TRACE와 STYLE 정보가 충돌하면 감정 결을 더 우선한다.",
            "- 숫자나 섹션 이름을 그대로 언급하지 않는다.",
            "- 한국어 평문으로만 2~5문장 이내로 답한다.",
            "- 같은 문장이나 핵심 구절을 반복하지 않는다.",
            "- 문장을 중간에 끊거나 조건절로 끝내지 않는다. 마지막 문장은 완결된 문장으로 끝낸다.",
            "- bullet, markdown, JSON, 코드블록을 쓰지 않는다.",
        ]
    )


def build_conditioned_generation_prompt(
    *,
    input_text: str,
    profile: dict[str, object],
    conditioning_mode: str,
    template_path: Path | None = None,
) -> tuple[str, str]:
    if conditioning_mode == "style":
        return (
            build_response_generation_prompt(
                input_text=input_text,
                style_dict=dict(profile["style_dict"]),
                style_tags=list(profile["style_tags"]),
                style_summary=dict(profile["style_summary"]),
                anti_softening_rules=list(profile.get("anti_softening_rules", [])),
                grounding_rules=list(profile.get("grounding_rules", [])),
                template_path=template_path,
            ),
            "style_tags,style_summary,anti_softening_rules,grounding_rules",
        )
    if conditioning_mode == "raw_trace":
        return (
            build_trace_generation_prompt(
                input_text=input_text,
                trace_lines=list(profile.get("trace_lines", [])),
                anti_softening_rules=list(profile.get("anti_softening_rules", [])),
                grounding_rules=list(profile.get("grounding_rules", [])),
            ),
            "raw_trace,anti_softening_rules,grounding_rules",
        )
    if conditioning_mode == "appraisal_trace":
        return (
            build_appraisal_generation_prompt(
                input_text=input_text,
                appraisal_lines=list(profile.get("appraisal_lines", [])),
                anti_softening_rules=list(profile.get("anti_softening_rules", [])),
                grounding_rules=list(profile.get("grounding_rules", [])),
            ),
            "appraisal_trace,anti_softening_rules,grounding_rules",
        )
    if conditioning_mode == "hybrid_trace":
        return (
            build_hybrid_trace_generation_prompt(
                input_text=input_text,
                style_dict=dict(profile["style_dict"]),
                style_tags=list(profile["style_tags"]),
                style_summary=dict(profile["style_summary"]),
                trace_lines=list(profile.get("trace_lines", [])),
                anti_softening_rules=list(profile.get("anti_softening_rules", [])),
                grounding_rules=list(profile.get("grounding_rules", [])),
            ),
            "raw_trace,style_tags,style_summary,anti_softening_rules,grounding_rules",
        )
    raise ValueError(f"unsupported conditioning_mode: {conditioning_mode}")


def infer_style_profile(
    model: EmoNet,
    decoder: LinearZtoSDecoder,
    text: str,
    style_profile: str = DEFAULT_STYLE_PROFILE,
) -> dict[str, object]:
    outputs = model.forward(text)
    z = to_numpy_array(outputs["z"], dtype=np.float32).reshape(-1)
    s_pred = to_numpy_array(decoder.predict(z), dtype=np.float32).reshape(-1)
    style_axes = resolve_style_axes(len(s_pred), style_profile=style_profile)
    style_dict = style_vector_to_dict(s_pred.tolist(), style_axes, style_profile=style_profile)
    style_summary = build_style_summary(style_dict)
    style_tags = build_style_tags(style_dict)
    stim_vec = to_numpy_array(outputs["stim_vec"], dtype=np.float32).reshape(-1)
    anti_softening_mode, anti_softening_rules = build_anti_softening_policy(
        input_text=text,
        style_dict=style_dict,
        style_summary=style_summary,
        stim_vec=stim_vec,
    )
    grounding_mode, grounding_rules = build_grounding_policy(
        input_text=text,
        style_summary=style_summary,
        anti_softening_mode=anti_softening_mode,
    )
    trace_profile = build_trace_profile(
        pruned_branch_log=list(outputs.get("pruned_branch_log", [])),
        dominant_branch=list(outputs.get("dominant_branch", [])),
        n_neurons=int(getattr(model.config, "n_neurons", 256)),
        termination_reason=str(outputs.get("termination_reason", "unknown")),
        ticks_run=int(outputs.get("ticks_run", 0)),
    )
    appraisal_profile = build_appraisal_profile(
        input_text=text,
        stim_vec=stim_vec,
        trace_profile=trace_profile,
        style_summary=style_summary,
    )
    return {
        "stim_vec": stim_vec,
        "dominant_branch_len": len(outputs["dominant_branch"]),
        "z": z,
        "s_pred": s_pred,
        "style_dict": style_dict,
        "style_tags": style_tags,
        "style_summary": style_summary,
        "style_summary_text": summarize_style_summary(style_summary),
        "expression_cues_text": summarize_expression_cues(style_dict),
        "anti_softening_mode": anti_softening_mode,
        "anti_softening_rules": anti_softening_rules,
        "grounding_mode": grounding_mode,
        "grounding_rules": grounding_rules,
        "trace_lines": list(trace_profile["trace_lines"]),
        "trace_summary_text": str(trace_profile["trace_summary_text"]),
        "trace_profile": trace_profile,
        "ticks_run": int(trace_profile["ticks_run"]),
        "termination_reason": str(trace_profile["termination_reason"]),
        "appraisal_scores": dict(appraisal_profile["appraisal_scores"]),
        "appraisal_lines": list(appraisal_profile["appraisal_lines"]),
        "appraisal_summary_text": str(appraisal_profile["appraisal_summary_text"]),
        "appraisal_target": str(appraisal_profile["appraisal_target"]),
        "appraisal_tendency": str(appraisal_profile["appraisal_tendency"]),
    }


def generate_response_from_style(
    base_url: str,
    model_name: str,
    input_text: str,
    style_dict: dict[str, float],
    style_tags: list[str],
    style_summary: dict[str, float],
    anti_softening_rules: list[str] | None,
    grounding_rules: list[str] | None,
    temperature: float,
    max_tokens: int,
    timeout_sec: int,
    template_path: Path | None = None,
    max_retries: int = 2,
) -> tuple[str, str, dict[str, object]]:
    prompt = build_response_generation_prompt(
        input_text=input_text,
        style_dict=style_dict,
        style_tags=style_tags,
        style_summary=style_summary,
        anti_softening_rules=anti_softening_rules,
        grounding_rules=grounding_rules,
        template_path=template_path,
    )
    response, _raw_output, response_meta = request_plain_text_response(
        base_url=base_url,
        model_name=model_name,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_sec=timeout_sec,
        max_retries=max_retries,
        validator=validate_plain_response_text,
        retry_instruction=(
            "직전 응답은 반복, 미완성 문장, bullet/JSON, 혹은 부자연스러운 출력 때문에 거부되었다. "
            "같은 문장이나 핵심 구절을 반복하지 말고, 마지막 문장은 완결된 한국어 평문으로 끝내라."
        ),
        system_prompt="Return a plain Korean response only. Do not return JSON.",
    )
    return response, prompt, response_meta


def generate_response_from_profile(
    *,
    base_url: str,
    model_name: str,
    input_text: str,
    profile: dict[str, object],
    temperature: float,
    max_tokens: int,
    timeout_sec: int,
    template_path: Path | None = None,
    max_retries: int = 2,
    conditioning_mode: str = "style",
) -> tuple[str, str, str, dict[str, object]]:
    prompt, prompt_sections = build_conditioned_generation_prompt(
        input_text=input_text,
        profile=profile,
        conditioning_mode=conditioning_mode,
        template_path=template_path,
    )
    response, _raw_output, response_meta = request_plain_text_response(
        base_url=base_url,
        model_name=model_name,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_sec=timeout_sec,
        max_retries=max_retries,
        validator=validate_plain_response_text,
        retry_instruction=(
            "직전 응답은 반복, 미완성 문장, bullet/JSON, 혹은 부자연스러운 출력 때문에 거부되었다. "
            "같은 문장이나 핵심 구절을 반복하지 말고, 마지막 문장은 완결된 한국어 평문으로 끝내라."
        ),
        system_prompt="Return a plain Korean response only. Do not return JSON.",
    )
    return response, prompt, prompt_sections, response_meta


def serialize_generation_log(record: dict[str, object]) -> dict[str, object]:
    payload = dict(record)
    for key in (
        "stim_vec",
        "z",
        "s_pred",
        "style_tags",
        "anti_softening_rules",
        "grounding_rules",
        "trace_lines",
        "appraisal_lines",
    ):
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


def append_csv_rows(
    output_path: Path,
    rows: list[dict[str, object]],
    columns: list[str] | None = None,
) -> None:
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chunk_df = pd.DataFrame(rows, columns=columns)
    write_header = not output_path.exists() or output_path.stat().st_size == 0
    mode = "w" if write_header else "a"
    chunk_df.to_csv(output_path, mode=mode, index=False, encoding="utf-8-sig", header=write_header)


def load_csv_columns(output_csv: Path) -> list[str]:
    if not output_csv.exists() or output_csv.stat().st_size == 0:
        return []
    return pd.read_csv(output_csv, nrows=0).columns.tolist()


def categorize_validation_error(exc: Exception) -> str:
    message = str(exc).lower()
    if isinstance(exc, ConnectionError):
        return "llm_server_unreachable"
    if isinstance(exc, FileNotFoundError):
        return "artifact_or_input_missing"
    if isinstance(exc, PermissionError):
        return "filesystem_permission_error"
    if isinstance(exc, urllib.error.HTTPError):
        return "llm_http_error"
    if isinstance(exc, urllib.error.URLError):
        return "network_error"
    if isinstance(exc, RuntimeError):
        if "scikit-learn" in message or "joblib" in message:
            return "dependency_missing"
        if "not fitted" in message:
            return "model_not_ready"
        return "runtime_error"
    if isinstance(exc, ValueError):
        if "missing required columns" in message or "column not found" in message or "text column not found" in message:
            return "input_schema_error"
        if "json" in message or "response" in message:
            return "model_output_invalid"
        if "shape" in message or "z_dim" in message or "s_dim" in message:
            return "shape_mismatch"
        return "validation_error"
    return exc.__class__.__name__.lower()


def build_validation_stage_result(
    stage_id: str,
    title: str,
    status: str,
    criteria: list[str],
    observed: dict[str, Any] | None = None,
    failure_category: str | None = None,
    error_message: str | None = None,
) -> dict[str, object]:
    return {
        "stage_id": stage_id,
        "title": title,
        "status": status,
        "criteria": list(criteria),
        "observed": {} if observed is None else observed,
        "failure_category": failure_category,
        "error_message": error_message,
    }


def build_e2e_validation_row(report: dict[str, object]) -> dict[str, object]:
    result = dict(report.get("result", {}))
    stage_status = dict(report.get("stage_status", {}))
    failure = dict(report.get("failure", {}))
    return {
        "timestamp_utc": report.get("timestamp_utc", ""),
        "input_text": report.get("input_text", ""),
        "overall_status": report.get("overall_status", ""),
        "stage1_status": stage_status.get("text_to_z", ""),
        "stage2_status": stage_status.get("z_to_s_pred", ""),
        "stage3_status": stage_status.get("s_pred_text_to_llm_response", ""),
        "stage4_status": stage_status.get("artifact_logging", ""),
        "failure_stage": failure.get("stage_id", ""),
        "failure_category": failure.get("category", ""),
        "failure_message": failure.get("message", ""),
        "dominant_branch_len": result.get("dominant_branch_len", ""),
        "stim_dim": len(result.get("stim_vec", [])) if isinstance(result.get("stim_vec"), list) else 0,
        "z_dim": len(result.get("z", [])) if isinstance(result.get("z"), list) else 0,
        "s_pred_dim": len(result.get("s_pred", [])) if isinstance(result.get("s_pred"), list) else 0,
        "llm_response": result.get("llm_response", ""),
        "response_retry_count": result.get("response_retry_count", ""),
        "response_validation_errors_json": json.dumps(result.get("response_validation_errors", []), ensure_ascii=False),
        "style_summary_text": result.get("style_summary_text", ""),
        "expression_cues_text": result.get("expression_cues_text", ""),
        "anti_softening_mode": result.get("anti_softening_mode", ""),
        "anti_softening_rules_json": json.dumps(result.get("anti_softening_rules", []), ensure_ascii=False),
        "grounding_mode": result.get("grounding_mode", ""),
        "grounding_rules_json": json.dumps(result.get("grounding_rules", []), ensure_ascii=False),
        "stim_vec_json": json.dumps(result.get("stim_vec", []), ensure_ascii=False),
        "z_json": json.dumps(result.get("z", []), ensure_ascii=False),
        "s_pred_json": json.dumps(result.get("s_pred", []), ensure_ascii=False),
        "style_tags_json": json.dumps(result.get("style_tags", []), ensure_ascii=False),
        "style_summary_json": json.dumps(result.get("style_summary", {}), ensure_ascii=False),
        "report_json_path": str(report.get("report_json_path", "")),
    }


def first_failed_stage(stages: list[dict[str, object]]) -> dict[str, object] | None:
    for stage in stages:
        if stage.get("status") == "failed":
            return stage
    return None


def command_e2e_check(args: argparse.Namespace) -> None:
    style_profile = getattr(args, "style_profile", DEFAULT_STYLE_PROFILE)
    report_json = Path(args.report_json)
    output_csv = Path(args.output_csv)
    log_jsonl = Path(args.log_jsonl)
    timestamp = utc_timestamp()
    report: dict[str, object] = {
        "timestamp_utc": timestamp,
        "input_text": args.text,
        "decoder_model_path": str(args.zs_model_path),
        "llm_base_url": args.base_url,
        "llm_model_name": args.model_name,
        "report_json_path": str(report_json),
        "output_csv_path": str(output_csv),
        "log_jsonl_path": str(log_jsonl),
        "overall_status": "failed",
        "stage_status": {},
        "failure": {},
        "stages": [],
        "result": {
            "input_text": args.text,
            "decoder_model_path": str(args.zs_model_path),
            "llm_model_name": args.model_name,
        },
    }

    stage_map: dict[str, dict[str, object]] = {}

    def record_stage(stage: dict[str, object]) -> None:
        stage_id = str(stage["stage_id"])
        stage_map[stage_id] = stage

    text_to_z_criteria = [
        "stim_vec must be length 4 with finite values in [0, 1]",
        "dominant_branch_len must be at least 1",
        f"z must be length {args.z_dim} with finite values",
    ]
    try:
        model = build_model(args)
        outputs = model.forward(args.text)
        stim_vec = to_numpy_array(outputs["stim_vec"], dtype=np.float32).reshape(-1)
        z = to_numpy_array(outputs["z"], dtype=np.float32).reshape(-1)
        dominant_branch_len = int(len(outputs["dominant_branch"]))
        failures = []
        if stim_vec.shape != (4,):
            failures.append(f"stim_vec shape must be (4,), got {tuple(stim_vec.shape)}")
        if not np.all(np.isfinite(stim_vec)):
            failures.append("stim_vec contains non-finite values")
        if np.any((stim_vec < 0.0) | (stim_vec > 1.0)):
            failures.append("stim_vec contains values outside [0, 1]")
        if dominant_branch_len < 1:
            failures.append("dominant_branch_len must be >= 1")
        if z.shape != (args.z_dim,):
            failures.append(f"z shape must be ({args.z_dim},), got {tuple(z.shape)}")
        if not np.all(np.isfinite(z)):
            failures.append("z contains non-finite values")
        if failures:
            raise ValueError("; ".join(failures))
        report["result"]["stim_vec"] = stim_vec.astype(float).tolist()
        report["result"]["dominant_branch_len"] = dominant_branch_len
        report["result"]["z"] = z.astype(float).tolist()
        record_stage(
            build_validation_stage_result(
                stage_id="text_to_z",
                title="text -> stim_vec -> EmoNet -> z",
                status="passed",
                criteria=text_to_z_criteria,
                observed={
                    "stim_dim": int(stim_vec.shape[0]),
                    "stim_min": round(float(stim_vec.min()), 6),
                    "stim_max": round(float(stim_vec.max()), 6),
                    "dominant_branch_len": dominant_branch_len,
                    "z_dim": int(z.shape[0]),
                    "z_abs_max": round(float(np.abs(z).max()), 6),
                },
            )
        )
    except Exception as exc:
        record_stage(
            build_validation_stage_result(
                stage_id="text_to_z",
                title="text -> stim_vec -> EmoNet -> z",
                status="failed",
                criteria=text_to_z_criteria,
                failure_category=categorize_validation_error(exc),
                error_message=str(exc),
            )
        )

    z_to_s_criteria = [
        "decoder artifact must load successfully",
        "s_pred must be finite and each value must be in [0, 1]",
        "s_pred must contain at least one style axis",
    ]
    if stage_map["text_to_z"]["status"] == "passed":
        try:
            decoder = LinearZtoSDecoder.load(Path(args.zs_model_path))
            z_arr = np.asarray(report["result"]["z"], dtype=np.float32)
            s_pred = np.asarray(decoder.predict(z_arr), dtype=np.float32).reshape(-1)
            failures = []
            if s_pred.size <= 0:
                failures.append("s_pred must contain at least one value")
            if not np.all(np.isfinite(s_pred)):
                failures.append("s_pred contains non-finite values")
            if np.any((s_pred < 0.0) | (s_pred > 1.0)):
                failures.append("s_pred contains values outside [0, 1]")
            if failures:
                raise ValueError("; ".join(failures))
            style_axes = resolve_style_axes(len(s_pred), style_profile=style_profile)
            style_dict = style_vector_to_dict(s_pred.tolist(), style_axes, style_profile=style_profile)
            style_summary = build_style_summary(style_dict)
            style_tags = build_style_tags(style_dict)
            anti_softening_mode, anti_softening_rules = build_anti_softening_policy(
                input_text=args.text,
                style_dict=style_dict,
                style_summary=style_summary,
                stim_vec=report["result"].get("stim_vec", []),
            )
            report["result"]["s_pred"] = s_pred.astype(float).tolist()
            report["result"]["style_tags"] = list(style_tags)
            report["result"]["style_summary"] = dict(style_summary)
            report["result"]["style_summary_text"] = summarize_style_summary(style_summary)
            report["result"]["expression_cues_text"] = summarize_expression_cues(style_dict)
            report["result"]["anti_softening_mode"] = anti_softening_mode
            report["result"]["anti_softening_rules"] = list(anti_softening_rules)
            grounding_mode, grounding_rules = build_grounding_policy(
                input_text=args.text,
                style_summary=style_summary,
                anti_softening_mode=anti_softening_mode,
            )
            report["result"]["grounding_mode"] = grounding_mode
            report["result"]["grounding_rules"] = list(grounding_rules)
            report["result"]["style_profile"] = str(style_profile)
            record_stage(
                build_validation_stage_result(
                    stage_id="z_to_s_pred",
                    title="z -> s_pred",
                    status="passed",
                    criteria=z_to_s_criteria,
                    observed={
                        "s_pred_dim": int(s_pred.shape[0]),
                        "s_pred_min": round(float(s_pred.min()), 6),
                        "s_pred_max": round(float(s_pred.max()), 6),
                    },
                )
            )
        except Exception as exc:
            record_stage(
                build_validation_stage_result(
                    stage_id="z_to_s_pred",
                    title="z -> s_pred",
                    status="failed",
                    criteria=z_to_s_criteria,
                    failure_category=categorize_validation_error(exc),
                    error_message=str(exc),
                )
            )
    else:
        record_stage(
            build_validation_stage_result(
                stage_id="z_to_s_pred",
                title="z -> s_pred",
                status="skipped",
                criteria=z_to_s_criteria,
                error_message="skipped because text_to_z failed",
            )
        )

    llm_criteria = [
        "LLM server must be reachable",
        "llm_response must be a non-empty plain-text string that passes repetition/completion validation",
        "style_prompt, style_summary, and s_pred must all be available together",
    ]
    if stage_map["z_to_s_pred"]["status"] == "passed":
        try:
            style_dict = style_vector_to_dict(
                report["result"]["s_pred"],
                resolve_style_axes(len(report["result"]["s_pred"]), style_profile=style_profile),
                style_profile=style_profile,
            )
            style_summary = dict(report["result"]["style_summary"])
            style_tags = list(report["result"]["style_tags"])
            ensure_model_server_ready(args.base_url, args.timeout_sec)
            response_text, style_prompt, response_meta = generate_response_from_style(
                base_url=args.base_url,
                model_name=args.model_name,
                input_text=args.text,
                style_dict=style_dict,
                style_tags=style_tags,
                style_summary=style_summary,
                anti_softening_rules=report["result"].get("anti_softening_rules", []),
                grounding_rules=report["result"].get("grounding_rules", []),
                temperature=args.response_temperature,
                max_tokens=args.max_tokens,
                timeout_sec=args.timeout_sec,
                template_path=Path(args.prompt_template) if args.prompt_template else None,
                max_retries=args.response_max_retries,
            )
            if not isinstance(response_text, str) or not response_text.strip():
                raise ValueError("llm_response must be a non-empty string")
            report["result"]["style_prompt"] = style_prompt
            report["result"]["llm_response"] = response_text.strip()
            report["result"]["response_retry_count"] = int(response_meta["retry_count"])
            report["result"]["response_validation_errors"] = list(response_meta["validation_errors"])
            record_stage(
                build_validation_stage_result(
                    stage_id="s_pred_text_to_llm_response",
                    title="s_pred + input_text -> LLM response",
                    status="passed",
                    criteria=llm_criteria,
                    observed={
                        "response_length": len(report["result"]["llm_response"]),
                        "response_retry_count": int(response_meta["retry_count"]),
                        "response_validation_rejections": len(response_meta["validation_errors"]),
                        "style_tag_count": len(style_tags),
                        "style_summary_keys": sorted(style_summary.keys()),
                    },
                )
            )
        except Exception as exc:
            record_stage(
                build_validation_stage_result(
                    stage_id="s_pred_text_to_llm_response",
                    title="s_pred + input_text -> LLM response",
                    status="failed",
                    criteria=llm_criteria,
                    failure_category=categorize_validation_error(exc),
                    error_message=str(exc),
                )
            )
    else:
        record_stage(
            build_validation_stage_result(
                stage_id="s_pred_text_to_llm_response",
                title="s_pred + input_text -> LLM response",
                status="skipped",
                criteria=llm_criteria,
                error_message="skipped because z_to_s_pred failed",
            )
        )

    report["stages"] = [
        stage_map["text_to_z"],
        stage_map["z_to_s_pred"],
        stage_map["s_pred_text_to_llm_response"],
    ]
    failed_stage = first_failed_stage(report["stages"])
    report["failure"] = (
        {}
        if failed_stage is None
        else {
            "stage_id": failed_stage["stage_id"],
            "category": failed_stage.get("failure_category"),
            "message": failed_stage.get("error_message"),
        }
    )

    artifact_criteria = [
        "report JSON must be written",
        "validation CSV row must be appended",
        "validation JSONL log row must be appended",
    ]
    try:
        stage_map["artifact_logging"] = build_validation_stage_result(
            stage_id="artifact_logging",
            title="로그/산출물 저장",
            status="passed",
            criteria=artifact_criteria,
            observed={
                "report_json": str(report_json),
                "output_csv": str(output_csv),
                "log_jsonl": str(log_jsonl),
            },
        )
        report["stages"].append(stage_map["artifact_logging"])
        report["stage_status"] = {str(stage["stage_id"]): str(stage["status"]) for stage in report["stages"]}
        report["overall_status"] = "passed" if all(stage["status"] == "passed" for stage in report["stages"]) else "failed"
        report["failure"] = (
            {}
            if first_failed_stage(report["stages"]) is None
            else {
                "stage_id": first_failed_stage(report["stages"])["stage_id"],
                "category": first_failed_stage(report["stages"]).get("failure_category"),
                "message": first_failed_stage(report["stages"]).get("error_message"),
            }
        )
        validation_row = build_e2e_validation_row(report)
        append_csv_rows(output_csv, [validation_row])
        append_jsonl(log_jsonl, [validation_row])
        report_json.parent.mkdir(parents=True, exist_ok=True)
        report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        failed_artifact_stage = build_validation_stage_result(
            stage_id="artifact_logging",
            title="로그/산출물 저장",
            status="failed",
            criteria=artifact_criteria,
            observed={
                "report_json": str(report_json),
                "output_csv": str(output_csv),
                "log_jsonl": str(log_jsonl),
            },
            failure_category=categorize_validation_error(exc),
            error_message=str(exc),
        )
        if len(report["stages"]) == 3:
            report["stages"].append(failed_artifact_stage)
        else:
            report["stages"][-1] = failed_artifact_stage
        report["stage_status"] = {str(stage["stage_id"]): str(stage["status"]) for stage in report["stages"]}
        report["overall_status"] = "failed"
        report["failure"] = {
            "stage_id": failed_artifact_stage["stage_id"],
            "category": failed_artifact_stage.get("failure_category"),
            "message": failed_artifact_stage.get("error_message"),
        }
        try:
            report_json.parent.mkdir(parents=True, exist_ok=True)
            report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass

    print(json.dumps(report, ensure_ascii=False, indent=2))


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
            "- 응답에 표정 변화 같은 명시적 비언어 단서가 있으면 그것도 표현 특성으로 반영한다.",
            "- 응답이 건조하거나 날카롭거나 체념적이거나 적대적이면, 사회적으로 바람직한 톤으로 순화하지 말고 그대로 반영한다.",
            "- hostility, resentment, despair, volatility, fearfulness, shame 축은 응답 표면에 그런 정서가 실제로 드러나면 0이 아닌 값도 적극적으로 사용한다.",
            "- blunt, cold, tense, urgent, hopeless, accusatory한 표현이면 calmness/softness/cooperativeness/positivity를 자동으로 높게 주지 않는다.",
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


def build_label_output_columns(input_columns: list[str], block_count: int, style_dim: int) -> list[str]:
    columns = list(input_columns)
    columns.extend(
        [
            "status",
            "generation_status",
            "llm_response",
            "generation_raw_output",
            "consistency_l1",
            "consistency_negative_raw_l1",
            "consistency_edge_l1",
            "negative_raw_mean",
            "negative_raw_max",
            "edge_mean",
            "edge_max",
            "soft_bias_mean",
            "rare_affect_candidate",
            "oversoft_candidate",
            "rebalance_bucket",
            "selection_score",
            "keep_reason",
            "keep_sample",
            "style_dim",
            "style_profile",
            "error_message",
        ]
    )
    for block_idx in range(1, block_count + 1):
        columns.extend(
            [
                f"s_block{block_idx}_status",
                f"s_block{block_idx}_raw_output",
                f"s_hat_block{block_idx}_status",
                f"s_hat_block{block_idx}_raw_output",
            ]
        )
    columns.extend([f"s_{axis_idx}" for axis_idx in range(style_dim)])
    columns.extend([f"s_hat_{axis_idx}" for axis_idx in range(style_dim)])
    return columns


def initialize_label_output_row(
    record: dict[str, object],
    block_count: int,
    style_dim: int,
    style_profile: str,
) -> dict[str, object]:
    row = dict(record)
    row["status"] = "error"
    row["generation_status"] = "pending"
    row["llm_response"] = ""
    row["generation_raw_output"] = ""
    row["consistency_l1"] = np.nan
    row["consistency_negative_raw_l1"] = np.nan
    row["consistency_edge_l1"] = np.nan
    row["negative_raw_mean"] = np.nan
    row["negative_raw_max"] = np.nan
    row["edge_mean"] = np.nan
    row["edge_max"] = np.nan
    row["soft_bias_mean"] = np.nan
    row["rare_affect_candidate"] = False
    row["oversoft_candidate"] = False
    row["rebalance_bucket"] = ""
    row["selection_score"] = np.nan
    row["keep_reason"] = ""
    row["keep_sample"] = False
    row["style_dim"] = style_dim
    row["style_profile"] = style_profile
    row["error_message"] = ""
    for block_idx in range(1, block_count + 1):
        row[f"s_block{block_idx}_status"] = "pending"
        row[f"s_block{block_idx}_raw_output"] = ""
        row[f"s_hat_block{block_idx}_status"] = "pending"
        row[f"s_hat_block{block_idx}_raw_output"] = ""
    for axis_idx in range(style_dim):
        row[f"s_{axis_idx}"] = np.nan
        row[f"s_hat_{axis_idx}"] = np.nan
    return row


def build_style_dict_from_row(row: dict[str, object], active_axes: list[str], prefix: str) -> dict[str, float]:
    style: dict[str, float] = {}
    for axis_idx, axis in enumerate(active_axes):
        value = row.get(f"{prefix}{axis_idx}", np.nan)
        if pd.isna(value):
            raise ValueError(f"missing style value for {prefix}{axis_idx}")
        style[axis] = float(value)
    return style


def analyze_style_balance_row(
    row: dict[str, object],
    active_axes: list[str],
    *,
    keep_threshold: float,
    rare_keep_threshold: float,
    raw_affect_min: float,
    edge_spike_min: float,
    soft_bias_floor: float,
    raw_affect_keep_threshold: float,
    edge_keep_threshold: float,
) -> dict[str, object]:
    style = build_style_dict_from_row(row, active_axes, "s_")
    style_hat = build_style_dict_from_row(row, active_axes, "s_hat_")
    consistency_l1 = row.get("consistency_l1", np.nan)
    if pd.isna(consistency_l1):
        consistency_l1 = compute_consistency(style, style_hat, active_axes)
    consistency_l1 = float(consistency_l1)
    metrics = compute_style_bias_metrics(style, style_hat, active_axes)
    keep_sample, keep_reason, bucket = decide_axis_aware_keep(
        consistency_l1=consistency_l1,
        metrics=metrics,
        keep_threshold=keep_threshold,
        rare_keep_threshold=rare_keep_threshold,
        raw_affect_min=raw_affect_min,
        edge_spike_min=edge_spike_min,
        soft_bias_floor=soft_bias_floor,
        raw_affect_keep_threshold=raw_affect_keep_threshold,
        edge_keep_threshold=edge_keep_threshold,
    )
    rare_affect_candidate = bucket in {"rare_raw", "edgy"}
    oversoft_candidate = bucket == "soft_safe"
    selection_score = compute_style_rebalance_score(metrics)
    return {
        "consistency_l1": consistency_l1,
        "consistency_negative_raw_l1": float(metrics["consistency_negative_raw_l1"]),
        "consistency_edge_l1": float(metrics["consistency_edge_l1"]),
        "negative_raw_mean": float(metrics["negative_raw_mean"]),
        "negative_raw_max": float(metrics["negative_raw_max"]),
        "edge_mean": float(metrics["edge_mean"]),
        "edge_max": float(metrics["edge_max"]),
        "soft_bias_mean": float(metrics["soft_bias_mean"]),
        "rare_affect_candidate": bool(rare_affect_candidate),
        "oversoft_candidate": bool(oversoft_candidate),
        "rebalance_bucket": str(bucket),
        "selection_score": float(selection_score),
        "keep_reason": str(keep_reason),
        "keep_sample": bool(keep_sample),
    }


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
    rare_keep_threshold: float,
    raw_affect_min: float,
    edge_spike_min: float,
    soft_bias_floor: float,
    raw_affect_keep_threshold: float,
    edge_keep_threshold: float,
    flush_every: int,
    resume: bool,
    style_profile: str = DEFAULT_STYLE_PROFILE,
) -> None:
    start_time = time.perf_counter()
    active_axes = resolve_style_axes(style_dim, style_profile=style_profile)
    style_blocks = build_style_blocks(block_size, active_axes)
    block_count = len(style_blocks)
    output_columns = build_label_output_columns(df.columns.tolist(), block_count, len(active_axes))
    resume_key = "sample_id" if "sample_id" in df.columns else "talk_id" if "talk_id" in df.columns else None
    existing_ids = load_existing_ids(output_csv, resume_key) if resume and resume_key else set()
    existing_rows = len(existing_ids)
    existing_kept = 0
    if resume and output_csv.exists():
        existing_columns = load_csv_columns(output_csv)
        if existing_columns and existing_columns != output_columns:
            raise ValueError(
                "existing output CSV schema does not match current labeling configuration; "
                "start a fresh output file or use matching options"
            )
        if output_csv.stat().st_size > 0:
            existing_df = pd.read_csv(output_csv, usecols=lambda name: name in {resume_key, "keep_sample"} if resume_key else name == "keep_sample")
            if "keep_sample" in existing_df.columns:
                existing_kept = int(existing_df["keep_sample"].fillna(False).astype(bool).sum())
    elif output_csv.exists():
        output_csv.unlink()

    remaining_rows = max(0, len(df) - len(existing_ids)) if existing_ids else len(df)
    total = remaining_rows if limit is None else min(remaining_rows, limit)
    pending_rows: list[dict[str, object]] = []
    processed = 0
    written = 0
    skipped = 0
    kept_this_run = 0
    ensure_model_server_ready(base_url, timeout_sec)

    for record in df.to_dict(orient="records"):
        if limit is not None and processed >= limit:
            break

        if existing_ids and resume_key:
            resume_value = str(record.get(resume_key, ""))
            if resume_value and resume_value in existing_ids:
                skipped += 1
                continue

        row = initialize_label_output_row(record, block_count, len(active_axes), style_profile)
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
            balance = analyze_style_balance_row(
                {
                    **row,
                    **{f"s_{axis_idx}": style[axis] for axis_idx, axis in enumerate(active_axes)},
                    **{f"s_hat_{axis_idx}": style_hat[axis] for axis_idx, axis in enumerate(active_axes)},
                    "consistency_l1": consistency_l1,
                },
                active_axes=active_axes,
                keep_threshold=keep_threshold,
                rare_keep_threshold=rare_keep_threshold,
                raw_affect_min=raw_affect_min,
                edge_spike_min=edge_spike_min,
                soft_bias_floor=soft_bias_floor,
                raw_affect_keep_threshold=raw_affect_keep_threshold,
                edge_keep_threshold=edge_keep_threshold,
            )

            row["status"] = "ok"
            row["style_dim"] = len(active_axes)
            row["style_profile"] = style_profile
            for axis_idx, axis in enumerate(active_axes):
                row[f"s_{axis_idx}"] = style[axis]
                row[f"s_hat_{axis_idx}"] = style_hat[axis]
            row.update(balance)
            pending_rows.append(row)
            kept_this_run += int(row["keep_sample"])
        except Exception as exc:
            if keep_failures:
                row["consistency_l1"] = np.nan
                row["keep_sample"] = False
                row["style_dim"] = len(active_axes)
                row["style_profile"] = style_profile
                row["error_message"] = str(exc)
                pending_rows.append(row)
            else:
                raise

        processed += 1
        written += 1

        if flush_every > 0 and len(pending_rows) >= flush_every:
            append_csv_rows(output_csv, pending_rows, columns=output_columns)
            pending_rows.clear()

        maybe_print_progress("label-local", processed, total, start_time, every=progress_every)

    append_csv_rows(output_csv, pending_rows, columns=output_columns)
    elapsed = time.perf_counter() - start_time
    print(
        json.dumps(
            {
                "rows": int(existing_rows + written),
                "session_rows": int(written),
                "skipped_rows": int(skipped),
                "kept_rows": int(existing_kept + kept_this_run),
                "output_csv": str(output_csv),
                "elapsed_sec": round(elapsed, 3),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def command_label_local(args: argparse.Namespace) -> None:
    style_profile = getattr(args, "style_profile", DEFAULT_STYLE_PROFILE)
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
        rare_keep_threshold=args.rare_keep_threshold,
        raw_affect_min=args.raw_affect_min,
        edge_spike_min=args.edge_spike_min,
        soft_bias_floor=args.soft_bias_floor,
        raw_affect_keep_threshold=args.raw_affect_keep_threshold,
        edge_keep_threshold=args.edge_keep_threshold,
        flush_every=args.flush_every,
        resume=args.resume,
        style_profile=style_profile,
    )


def rebalance_labeled_style_dataframe(
    df: pd.DataFrame,
    *,
    style_dim: int,
    style_profile: str,
    keep_threshold: float,
    rare_keep_threshold: float,
    raw_affect_min: float,
    edge_spike_min: float,
    soft_bias_floor: float,
    raw_affect_keep_threshold: float,
    edge_keep_threshold: float,
    soft_cap_ratio: float,
    target_size: int | None,
    base_keep_column: str,
    keep_column: str,
    progress_every: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if soft_cap_ratio < 0.0 or soft_cap_ratio > 1.0:
        raise ValueError("soft_cap_ratio must be between 0.0 and 1.0")
    active_axes = resolve_style_axes(style_dim, style_profile=style_profile)
    start_time = time.perf_counter()
    records = df.to_dict(orient="records")
    analyzed_rows: list[dict[str, object]] = []
    for idx, record in enumerate(records, start=1):
        row = dict(record)
        status = str(record.get("status", "")).strip().lower()
        if status == "ok":
            try:
                row.update(
                    analyze_style_balance_row(
                        record,
                        active_axes=active_axes,
                        keep_threshold=keep_threshold,
                        rare_keep_threshold=rare_keep_threshold,
                        raw_affect_min=raw_affect_min,
                        edge_spike_min=edge_spike_min,
                        soft_bias_floor=soft_bias_floor,
                        raw_affect_keep_threshold=raw_affect_keep_threshold,
                        edge_keep_threshold=edge_keep_threshold,
                    )
                )
            except Exception as exc:
                row["keep_reason"] = f"analysis_error:{exc}"
                row["keep_sample"] = False
                row["rare_affect_candidate"] = False
                row["oversoft_candidate"] = False
                row["rebalance_bucket"] = ""
                row["selection_score"] = np.nan
        analyzed_rows.append(row)
        maybe_print_progress("rebalance-labeled", idx, len(records), start_time, every=progress_every)
    maybe_print_progress("rebalance-labeled", len(records), len(records), start_time, every=progress_every, force=True)

    analyzed = pd.DataFrame(analyzed_rows)
    if keep_column not in analyzed.columns:
        analyzed[keep_column] = False
    rare_series = (
        analyzed["rare_affect_candidate"]
        if "rare_affect_candidate" in analyzed.columns
        else pd.Series(False, index=analyzed.index, dtype="boolean")
    )
    oversoft_series = (
        analyzed["oversoft_candidate"]
        if "oversoft_candidate" in analyzed.columns
        else pd.Series(False, index=analyzed.index, dtype="boolean")
    )
    analyzed["rare_affect_candidate"] = pd.Series(rare_series, index=analyzed.index).astype("boolean").fillna(False).astype(bool)
    analyzed["oversoft_candidate"] = pd.Series(oversoft_series, index=analyzed.index).astype("boolean").fillna(False).astype(bool)
    analyzed["status"] = analyzed.get("status", "").astype(str)
    analyzed["consistency_l1"] = pd.to_numeric(analyzed.get("consistency_l1"), errors="coerce")
    analyzed["consistency_negative_raw_l1"] = pd.to_numeric(
        analyzed.get("consistency_negative_raw_l1"),
        errors="coerce",
    ).fillna(0.0)
    analyzed["consistency_edge_l1"] = pd.to_numeric(analyzed.get("consistency_edge_l1"), errors="coerce").fillna(0.0)
    analyzed["selection_score"] = pd.to_numeric(analyzed.get("selection_score"), errors="coerce").fillna(-999.0)

    eligible_mask = (analyzed["status"].str.lower() == "ok") & (
        (analyzed["consistency_l1"] <= keep_threshold)
        | (
            analyzed["rare_affect_candidate"]
            & (analyzed["consistency_l1"] <= rare_keep_threshold)
            & (analyzed["consistency_negative_raw_l1"] <= raw_affect_keep_threshold)
            & (analyzed["consistency_edge_l1"] <= edge_keep_threshold)
        )
    )
    analyzed["axis_keep_candidate"] = eligible_mask

    if target_size is None or int(target_size) <= 0:
        if base_keep_column in analyzed.columns:
            target_size = int(analyzed[base_keep_column].fillna(False).astype(bool).sum())
        else:
            target_size = int(eligible_mask.sum())
    target_size = min(int(target_size), int(eligible_mask.sum()))
    rng = np.random.default_rng(seed)
    bucket_priority = {"rare_raw": 3, "edgy": 2, "mixed": 1, "soft_safe": 0}
    eligible = analyzed.loc[eligible_mask].copy()
    eligible["bucket_priority"] = eligible["rebalance_bucket"].map(bucket_priority).fillna(-1).astype(int)
    eligible["selection_tiebreak"] = rng.random(len(eligible))
    eligible = eligible.sort_values(
        by=["bucket_priority", "selection_score", "consistency_l1", "selection_tiebreak"],
        ascending=[False, False, True, True],
    )
    nonsoft = eligible.loc[~eligible["oversoft_candidate"]]
    soft = eligible.loc[eligible["oversoft_candidate"]]
    selected_indices = nonsoft.index.tolist()[:target_size]
    remaining = max(0, target_size - len(selected_indices))
    soft_cap = min(len(soft), max(0, int(round(target_size * soft_cap_ratio))))
    if remaining > 0:
        selected_indices.extend(soft.index.tolist()[: min(remaining, soft_cap)])
    remaining = max(0, target_size - len(selected_indices))
    if remaining > 0:
        leftover_soft = [idx for idx in soft.index.tolist() if idx not in set(selected_indices)]
        selected_indices.extend(leftover_soft[:remaining])
    selected_set = set(selected_indices)
    analyzed[keep_column] = analyzed.index.to_series().isin(selected_set)

    before_soft_mean = float(
        pd.to_numeric(analyzed.loc[eligible_mask, "soft_bias_mean"], errors="coerce").fillna(0.0).mean()
    ) if int(eligible_mask.sum()) > 0 else 0.0
    after_soft_mean = float(
        pd.to_numeric(analyzed.loc[analyzed[keep_column], "soft_bias_mean"], errors="coerce").fillna(0.0).mean()
    ) if int(analyzed[keep_column].sum()) > 0 else 0.0
    before_neg_mean = float(
        pd.to_numeric(analyzed.loc[eligible_mask, "negative_raw_mean"], errors="coerce").fillna(0.0).mean()
    ) if int(eligible_mask.sum()) > 0 else 0.0
    after_neg_mean = float(
        pd.to_numeric(analyzed.loc[analyzed[keep_column], "negative_raw_mean"], errors="coerce").fillna(0.0).mean()
    ) if int(analyzed[keep_column].sum()) > 0 else 0.0
    before_edge_mean = float(
        pd.to_numeric(analyzed.loc[eligible_mask, "edge_mean"], errors="coerce").fillna(0.0).mean()
    ) if int(eligible_mask.sum()) > 0 else 0.0
    after_edge_mean = float(
        pd.to_numeric(analyzed.loc[analyzed[keep_column], "edge_mean"], errors="coerce").fillna(0.0).mean()
    ) if int(analyzed[keep_column].sum()) > 0 else 0.0
    summary = {
        "input_rows": int(len(analyzed)),
        "ok_rows": int((analyzed["status"].str.lower() == "ok").sum()),
        "eligible_rows": int(eligible_mask.sum()),
        "target_size": int(target_size),
        "selected_rows": int(analyzed[keep_column].sum()),
        "selected_rare_rows": int((analyzed[keep_column] & analyzed["rare_affect_candidate"]).sum()),
        "selected_oversoft_rows": int((analyzed[keep_column] & analyzed["oversoft_candidate"]).sum()),
        "keep_column": str(keep_column),
        "base_keep_column": str(base_keep_column),
        "soft_cap_ratio": round(float(soft_cap_ratio), 4),
        "before_soft_bias_mean": round(before_soft_mean, 6),
        "after_soft_bias_mean": round(after_soft_mean, 6),
        "before_negative_raw_mean": round(before_neg_mean, 6),
        "after_negative_raw_mean": round(after_neg_mean, 6),
        "before_edge_mean": round(before_edge_mean, 6),
        "after_edge_mean": round(after_edge_mean, 6),
        "bucket_counts_selected": {
            key: int(value)
            for key, value in analyzed.loc[analyzed[keep_column], "rebalance_bucket"].value_counts().to_dict().items()
        },
    }
    return analyzed, summary


def command_rebalance_labeled(args: argparse.Namespace) -> None:
    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    df = pd.read_csv(input_csv)
    analyzed, summary = rebalance_labeled_style_dataframe(
        df=df,
        style_dim=args.style_dim,
        style_profile=args.style_profile,
        keep_threshold=args.keep_threshold,
        rare_keep_threshold=args.rare_keep_threshold,
        raw_affect_min=args.raw_affect_min,
        edge_spike_min=args.edge_spike_min,
        soft_bias_floor=args.soft_bias_floor,
        raw_affect_keep_threshold=args.raw_affect_keep_threshold,
        edge_keep_threshold=args.edge_keep_threshold,
        soft_cap_ratio=args.soft_cap_ratio,
        target_size=args.target_size,
        base_keep_column=args.base_keep_column,
        keep_column=args.keep_column,
        progress_every=args.progress_every,
        seed=args.seed,
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    analyzed.to_csv(output_csv, index=False, encoding="utf-8-sig")
    payload = dict(summary)
    payload["output_csv"] = str(output_csv)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def command_export_z(args: argparse.Namespace) -> None:
    output_csv = Path(args.output_csv)
    text_column = args.text_column
    worker_count = resolve_num_workers(getattr(args, "num_workers", 1))
    model = build_model(args) if worker_count <= 1 else None

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
            num_workers=worker_count,
            model_args=args,
        )
    else:
        input_csv = Path(args.input_csv)
        df = pd.read_csv(input_csv)
        text_column = resolve_text_column(df, text_column)
        if args.limit is not None and args.limit > 0:
            df = df.head(args.limit).copy()
        export_z_from_dataframe(
            model,
            df,
            text_column,
            output_csv,
            progress_every=args.progress_every,
            num_workers=worker_count,
            model_args=args,
        )


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
        subparser.add_argument("--z-encoder-mode", choices=["auto", "stat", "transformer"], default="auto")
        subparser.add_argument("--z-encoder-path", dest="z_encoder_path", type=str, default=str(DEFAULT_Z_ENCODER_MODEL_PATH))
        subparser.add_argument("--max-ticks", dest="max_ticks", type=int, default=None)
        subparser.add_argument("--min-ticks-before-converged", dest="min_ticks_before_converged", type=int, default=None)
        subparser.add_argument("--convergence-patience", dest="convergence_patience", type=int, default=None)
        subparser.add_argument("--activity-count-delta-eps", dest="activity_count_delta_eps", type=float, default=None)
        subparser.add_argument("--edge-count-delta-eps", dest="edge_count_delta_eps", type=float, default=None)
        subparser.add_argument("--activity-churn-eps", dest="activity_churn_eps", type=float, default=None)
        subparser.add_argument("--k-threshold-base", dest="k_threshold_base", type=float, default=None)
        subparser.add_argument("--k-remem-base", dest="k_remem_base", type=float, default=None)
        subparser.add_argument("--k-decay", dest="k_decay", type=float, default=None)
        subparser.add_argument("--refractory-ticks", dest="refractory_ticks", type=int, default=None)
        subparser.add_argument("--input-topk", dest="input_topk", type=int, default=None)
        subparser.add_argument("--input-signal-clip", dest="input_signal_clip", type=float, default=None)
        subparser.add_argument("--memory-decay", dest="memory_decay", type=float, default=None)
        subparser.add_argument("--memory-stim-mix", dest="memory_stim_mix", type=float, default=None)
        subparser.add_argument("--memory-k-mix", dest="memory_k_mix", type=float, default=None)
        subparser.add_argument("--state-self-stim-mix", dest="state_self_stim_mix", type=float, default=None)
        subparser.add_argument("--state-parent-stim-mix", dest="state_parent_stim_mix", type=float, default=None)
        subparser.add_argument("--state-base-stim-mix", dest="state_base_stim_mix", type=float, default=None)
        subparser.add_argument("--state-bias-stim-mix", dest="state_bias_stim_mix", type=float, default=None)
        subparser.add_argument("--recent-activity-decay", dest="recent_activity_decay", type=float, default=None)
        subparser.add_argument("--hysteresis-threshold-gain", dest="hysteresis_threshold_gain", type=float, default=None)
        subparser.add_argument("--hysteresis-remem-gain", dest="hysteresis_remem_gain", type=float, default=None)
        subparser.add_argument("--hysteresis-k-bonus", dest="hysteresis_k_bonus", type=float, default=None)
        subparser.add_argument("--intrinsic-alignment-gain", dest="intrinsic_alignment_gain", type=float, default=None)
        subparser.add_argument("--fatigue-decay", dest="fatigue_decay", type=float, default=None)
        subparser.add_argument("--fatigue-gain", dest="fatigue_gain", type=float, default=None)
        subparser.add_argument("--fatigue-threshold-gain", dest="fatigue_threshold_gain", type=float, default=None)
        subparser.add_argument("--fatigue-k-leak", dest="fatigue_k_leak", type=float, default=None)
        subparser.add_argument("--fire-output-log-gain", dest="fire_output_log_gain", type=float, default=None)
        subparser.add_argument("--inhibitory-suppression-gain", dest="inhibitory_suppression_gain", type=float, default=None)
        subparser.add_argument("--max-out-degree", dest="max_out_degree", type=int, default=None)
        subparser.add_argument("--min-out-degree", dest="min_out_degree", type=int, default=None)
        subparser.add_argument("--dopa-rewire-gain", dest="dopa_rewire_gain", type=float, default=None)
        subparser.add_argument("--sero-prune-gain", dest="sero_prune_gain", type=float, default=None)
        subparser.add_argument("--mela-dropout-gain", dest="mela_dropout_gain", type=float, default=None)
        subparser.add_argument("--ne-thresh-reduce-gain", dest="ne_thresh_reduce_gain", type=float, default=None)
        subparser.add_argument("--ne-remem-reduce-gain", dest="ne_remem_reduce_gain", type=float, default=None)
        subparser.add_argument("--global-recovery-rate", dest="global_recovery_rate", type=float, default=None)
        subparser.add_argument("--topk-branches", dest="topk_branches", type=int, default=None)
        subparser.add_argument("--branch-end-window", dest="branch_end_window", type=int, default=None)
        subparser.add_argument("--branch-length-bonus", dest="branch_length_bonus", type=float, default=None)

    def add_generation_options(subparser: argparse.ArgumentParser, log_jsonl_default: str | None = None) -> None:
        add_common_options(subparser)
        subparser.add_argument("--zs-model-path", required=True)
        subparser.add_argument("--style-profile", choices=sorted(STYLE_AXIS_PROFILES), default=DEFAULT_STYLE_PROFILE)
        subparser.add_argument(
            "--conditioning-mode",
            choices=["style", "raw_trace", "appraisal_trace", "hybrid_trace"],
            default="style",
        )
        subparser.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
        subparser.add_argument("--model-name", default="gpt-oss:20b")
        subparser.add_argument("--response-temperature", type=float, default=0.5)
        subparser.add_argument("--response-max-retries", type=int, default=2)
        subparser.add_argument("--max-tokens", type=int, default=600)
        subparser.add_argument("--timeout-sec", type=int, default=180)
        subparser.add_argument("--prompt-template", default=None)
        subparser.add_argument("--log-jsonl", default=log_jsonl_default)

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

    e2e_parser = subparsers.add_parser("e2e-check")
    add_generation_options(e2e_parser, log_jsonl_default=str(Path("outputs") / "validation" / "e2e_check_runs.jsonl"))
    e2e_parser.add_argument("--text", required=True)
    e2e_parser.add_argument("--report-json", default=str(Path("outputs") / "validation" / "e2e_check_report.json"))
    e2e_parser.add_argument("--output-csv", default=str(Path("outputs") / "validation" / "e2e_check_runs.csv"))
    e2e_parser.set_defaults(func=command_e2e_check)

    batch_generate_parser = subparsers.add_parser("generate-response-batch")
    add_generation_options(batch_generate_parser)
    batch_generate_parser.add_argument("--input-csv", required=True)
    batch_generate_parser.add_argument("--output-csv", required=True)
    batch_generate_parser.add_argument("--text-column", default="text")
    batch_generate_parser.add_argument("--limit", type=int, default=None)
    batch_generate_parser.add_argument("--progress-every", type=int, default=10)
    batch_generate_parser.set_defaults(func=command_generate_response_batch)

    probe_parser = subparsers.add_parser("probe-branch")
    add_common_options(probe_parser)
    probe_parser.add_argument("--input-csv", default=None)
    probe_parser.add_argument("--input-json", default=None)
    probe_parser.add_argument("--text-column", default="text")
    probe_parser.add_argument("--sample-size", type=int, default=200)
    probe_parser.add_argument("--sample-mode", choices=["head", "random"], default="random")
    probe_parser.add_argument("--progress-every", type=int, default=20)
    probe_parser.add_argument("--num-workers", type=int, default=1, help="0 uses all logical CPU cores")
    probe_parser.add_argument("--output-csv", default=None)
    probe_parser.set_defaults(func=command_probe_branch)

    export_parser = subparsers.add_parser("export-z")
    add_common_options(export_parser)
    export_parser.add_argument("--input-csv", default=None)
    export_parser.add_argument("--input-json", default=None)
    export_parser.add_argument("--text-column", default="text")
    export_parser.add_argument("--output-csv", required=True)
    export_parser.add_argument("--limit", type=int, default=None)
    export_parser.add_argument("--chunk-size", type=int, default=256)
    export_parser.add_argument("--progress-every", type=int, default=100)
    export_parser.add_argument("--num-workers", type=int, default=1, help="0 uses all logical CPU cores")
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
    fit_zs_parser.add_argument("--keep-column", default="keep_sample")
    fit_zs_parser.set_defaults(func=command_fit_zs_regressor)

    fit_z_encoder_parser = subparsers.add_parser("fit-z-encoder")
    add_common_options(fit_z_encoder_parser)
    fit_z_encoder_parser.add_argument("--input-csv", required=True)
    fit_z_encoder_parser.add_argument("--text-column", default="text")
    fit_z_encoder_parser.add_argument("--zs-model-path", required=True)
    fit_z_encoder_parser.add_argument("--z-output-csv", default=None)
    fit_z_encoder_parser.add_argument("--style-dim", type=int, default=32)
    fit_z_encoder_parser.add_argument("--style-profile", choices=sorted(STYLE_AXIS_PROFILES), default=DEFAULT_STYLE_PROFILE)
    fit_z_encoder_parser.add_argument("--epochs", type=int, default=12)
    fit_z_encoder_parser.add_argument("--batch-size", type=int, default=32)
    fit_z_encoder_parser.add_argument("--learning-rate", type=float, default=5e-4)
    fit_z_encoder_parser.add_argument("--weight-decay", type=float, default=1e-4)
    fit_z_encoder_parser.add_argument("--ridge-alpha", type=float, default=1.0)
    fit_z_encoder_parser.add_argument("--val-ratio", type=float, default=0.1)
    fit_z_encoder_parser.add_argument("--progress-every", type=int, default=100)
    fit_z_encoder_parser.add_argument("--use-all-rows", action="store_true")
    fit_z_encoder_parser.add_argument("--warm-start-z-encoder", action="store_true")
    fit_z_encoder_parser.add_argument("--keep-column", default="keep_sample")
    fit_z_encoder_parser.set_defaults(func=command_fit_z_encoder)

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
    local_parser.add_argument("--style-profile", choices=sorted(STYLE_AXIS_PROFILES), default=DEFAULT_STYLE_PROFILE)
    local_parser.add_argument("--keep-threshold", type=float, default=0.12)
    local_parser.add_argument("--rare-keep-threshold", type=float, default=0.18)
    local_parser.add_argument("--raw-affect-min", type=float, default=0.25)
    local_parser.add_argument("--edge-spike-min", type=float, default=0.75)
    local_parser.add_argument("--soft-bias-floor", type=float, default=0.90)
    local_parser.add_argument("--raw-affect-keep-threshold", type=float, default=0.25)
    local_parser.add_argument("--edge-keep-threshold", type=float, default=0.20)
    local_parser.add_argument("--keep-failures", action="store_true")
    local_parser.add_argument("--flush-every", type=int, default=10)
    local_parser.add_argument("--resume", action="store_true")
    local_parser.set_defaults(func=command_label_local)

    rebalance_parser = subparsers.add_parser("rebalance-labeled")
    rebalance_parser.add_argument("--input-csv", required=True)
    rebalance_parser.add_argument("--output-csv", required=True)
    rebalance_parser.add_argument("--style-dim", type=int, default=40)
    rebalance_parser.add_argument("--style-profile", choices=sorted(STYLE_AXIS_PROFILES), default=DEFAULT_STYLE_PROFILE)
    rebalance_parser.add_argument("--keep-threshold", type=float, default=0.12)
    rebalance_parser.add_argument("--rare-keep-threshold", type=float, default=0.18)
    rebalance_parser.add_argument("--raw-affect-min", type=float, default=0.25)
    rebalance_parser.add_argument("--edge-spike-min", type=float, default=0.75)
    rebalance_parser.add_argument("--soft-bias-floor", type=float, default=0.90)
    rebalance_parser.add_argument("--raw-affect-keep-threshold", type=float, default=0.25)
    rebalance_parser.add_argument("--edge-keep-threshold", type=float, default=0.20)
    rebalance_parser.add_argument("--soft-cap-ratio", type=float, default=0.25)
    rebalance_parser.add_argument("--target-size", type=int, default=None)
    rebalance_parser.add_argument("--base-keep-column", default="keep_sample")
    rebalance_parser.add_argument("--keep-column", default="keep_sample_rebalanced")
    rebalance_parser.add_argument("--progress-every", type=int, default=100)
    rebalance_parser.add_argument("--seed", type=int, default=42)
    rebalance_parser.set_defaults(func=command_rebalance_labeled)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
