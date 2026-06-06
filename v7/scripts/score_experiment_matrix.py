from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from emonet.cli import (
    append_csv_rows,
    ensure_model_server_ready,
    maybe_print_progress,
    request_json_response,
)


SCORE_KEYS = [
    "content_fit",
    "emotional_appropriateness",
    "style_match",
    "naturalness",
    "overall_quality",
]

SCORE_LABELS = {
    "content_fit": "content_fit",
    "emotional_appropriateness": "emotion_fit",
    "style_match": "style_match",
    "naturalness": "naturalness",
    "overall_quality": "overall",
}


def normalize_scores(payload: dict[str, object]) -> dict[str, int]:
    scores = payload.get("scores", payload)
    if not isinstance(scores, dict):
        raise ValueError("scores object is required")

    normalized: dict[str, int] = {}
    for key in SCORE_KEYS:
        if key not in scores:
            raise ValueError(f"missing score key: {key}")
        value = scores[key]
        if isinstance(value, bool):
            raise ValueError(f"score must be numeric: {key}")
        number = int(round(float(value)))
        if number < 1 or number > 5:
            raise ValueError(f"score must be in [1, 5]: {key}={value}")
        normalized[key] = number
    return normalized


def build_judge_prompt(row: dict[str, object]) -> str:
    text = str(row.get("text", "")).strip()
    response = str(row.get("llm_response", "")).strip()
    condition = str(row.get("condition", "")).strip()
    target_summary = str(row.get("style_summary_text", "")).strip()
    target_tags = str(row.get("style_tags_json", "[]")).strip()

    return "\n".join(
        [
            "[ROLE]",
            "?뱀떊? ?쒓뎅??媛먯젙 ?묐떟 ?덉쭏??梨꾩젏?섎뒗 ?ъ궗?먮떎.",
            "",
            "[INPUT_TEXT]",
            text,
            "",
            "[MODEL_CONDITION]",
            condition,
            "",
            "[TARGET_STYLE_TAGS]",
            target_tags,
            "",
            "[TARGET_STYLE_SUMMARY]",
            target_summary,
            "",
            "[MODEL_RESPONSE]",
            response,
            "",
            "[SCORING_RULE]",
            "- 媛???ぉ??1??留ㅼ슦 ?섏겏)遺??5??留ㅼ슦 醫뗭쓬)源뚯? ?뺤닔濡??됯??쒕떎.",
            "- content_fit: ?낅젰 ?댁슜??吏곸젒?곸쑝濡?留욌뒗媛",
            "- emotional_appropriateness: ?낅젰 媛먯젙 ?곹깭??留욌뒗媛",
            "- style_match: 紐⑺몴 ?ㅽ????붿빟怨??쒓렇???쇰쭏??留욌뒗媛",
            "- naturalness: ?쒓뎅???묐떟???먯뿰?ㅻ윭?닿?",
            "- overall_quality: ?꾩껜?곸쑝濡??ㅻ뱷???덈뒗媛",
            "",
            "[OUTPUT_FORMAT]",
            "JSON only.",
            "{",
            '  "scores": {',
            '    "content_fit": 1,',
            '    "emotional_appropriateness": 1,',
            '    "style_match": 1,',
            '    "naturalness": 1,',
            '    "overall_quality": 1',
            "  }",
            "}",
        ]
    )


def resolve_api_key(api_key_env: str | None) -> str | None:
    if not api_key_env:
        return None
    value = os.environ.get(str(api_key_env), "").strip()
    if not value:
        raise ValueError(f"environment variable '{api_key_env}' is not set or empty")
    return value


def should_use_json_mode(base_url: str, api_key: str | None) -> bool:
    return bool(api_key and "api.openai.com" in str(base_url).lower())


def resolve_reasoning_effort(base_url: str, api_key: str | None, requested: str | None) -> str | None:
    if not requested:
        return None
    if not (api_key and "api.openai.com" in str(base_url).lower()):
        raise ValueError("--reasoning-effort is only supported for api.openai.com requests with an API key")
    return str(requested).strip()


def request_score_payload(
    row: dict[str, object],
    *,
    base_url: str,
    model_name: str,
    timeout_sec: int,
    max_tokens: int,
    temperature: float,
    max_retries: int,
    api_key: str | None = None,
    reasoning_effort: str | None = None,
) -> tuple[dict[str, int], str, str]:
    json_prompt = build_judge_prompt(row)
    payload, raw = request_json_response(
        base_url=base_url,
        model_name=model_name,
        prompt=json_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_sec=timeout_sec,
        max_retries=max_retries,
        validator=normalize_scores,
        retry_instruction="Previous output was not a valid JSON scores object. Return exactly one JSON object with scores for the five metrics.",
        api_key=api_key,
        response_format={"type": "json_object"} if should_use_json_mode(base_url, api_key) else None,
        reasoning_effort=reasoning_effort,
    )
    return payload, raw, "json"


def load_existing_keys(output_csv: Path) -> set[tuple[str, str]]:
    if not output_csv.exists() or output_csv.stat().st_size == 0:
        return set()
    df = pd.read_csv(output_csv)
    if "status" in df.columns:
        df = df[df["status"].fillna("") == "ok"].copy()
    if df.empty:
        return set()
    return {
        (str(record_id), str(condition))
        for record_id, condition in zip(df["record_id"].astype(str), df["condition"].astype(str), strict=True)
    }


def score_matrix(
    input_csv: Path,
    output_csv: Path,
    base_url: str,
    model_name: str,
    timeout_sec: int,
    max_tokens: int,
    temperature: float,
    max_retries: int,
    progress_every: int,
    flush_every: int,
    keep_failures: bool,
    resume: bool,
    limit: int | None,
    api_key: str | None,
    reasoning_effort: str | None,
) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    ok_df = df[df["status"].fillna("") == "ok"].copy()
    if limit is not None and limit > 0:
        ok_df = ok_df.head(limit).copy()
    if ok_df.empty:
        raise ValueError("no successful matrix rows found to score")

    existing_keys = load_existing_keys(output_csv) if resume else set()
    output_rows: list[dict[str, object]] = []
    output_columns = [
        "record_id",
        "condition",
        "condition_group",
        "status",
        "error_message",
        "text",
        "llm_response",
        "response_length",
        "response_retry_count",
        "judge_parse_mode",
        "judge_raw_output",
        *SCORE_KEYS,
    ]
    start_time = time.perf_counter()

    for idx, row in enumerate(ok_df.to_dict(orient="records"), start=1):
        record_id = str(row.get("record_id", row.get("sample_id", f"row_{idx:06d}")))
        condition = str(row.get("condition", ""))
        key = (record_id, condition)
        if key in existing_keys:
            continue
        retry_value = pd.to_numeric(row.get("response_retry_count", 0), errors="coerce")

        scored = {
            "record_id": record_id,
            "condition": condition,
            "condition_group": str(row.get("condition_group", "")),
            "status": "error",
            "error_message": "",
            "text": str(row.get("text", "")),
            "llm_response": str(row.get("llm_response", "")),
            "response_length": int(len(str(row.get("llm_response", "")))),
            "response_retry_count": 0 if pd.isna(retry_value) else int(retry_value),
            "judge_parse_mode": "",
            "judge_raw_output": "",
        }
        for key_name in SCORE_KEYS:
            scored[key_name] = pd.NA

        try:
            payload, raw, parse_mode = request_score_payload(
                row,
                base_url=base_url,
                model_name=model_name,
                timeout_sec=timeout_sec,
                max_tokens=max_tokens,
                temperature=temperature,
                max_retries=max_retries,
                api_key=api_key,
                reasoning_effort=reasoning_effort,
            )
            scored["status"] = "ok"
            scored["judge_parse_mode"] = parse_mode
            scored["judge_raw_output"] = raw
            for key_name, value in payload.items():
                scored[key_name] = int(value)
            output_rows.append(scored)
        except Exception as exc:
            scored["error_message"] = str(exc)
            if keep_failures:
                output_rows.append(scored)
            else:
                raise

        if flush_every > 0 and len(output_rows) >= flush_every:
            append_csv_rows(output_csv, output_rows, columns=output_columns)
            output_rows.clear()

        maybe_print_progress("score-experiment-matrix", idx, len(ok_df), start_time, every=progress_every)

    append_csv_rows(output_csv, output_rows, columns=output_columns)
    scored_df = pd.read_csv(output_csv)
    return scored_df


def summarize_scores(scored_df: pd.DataFrame) -> pd.DataFrame:
    if "status" in scored_df.columns:
        scored_df = scored_df[scored_df["status"].fillna("") == "ok"].copy()
    if scored_df.empty:
        return pd.DataFrame(
            columns=[
                "condition",
                "condition_group",
                "rows",
                "mean_response_length",
                "mean_response_retry_count",
                *[f"mean_{key}" for key in SCORE_KEYS],
                "mean_total",
            ]
        )
    rows: list[dict[str, object]] = []
    for condition, group in scored_df.groupby("condition", dropna=False):
        row = {
            "condition": str(condition),
            "condition_group": str(group["condition_group"].iloc[0]) if "condition_group" in group.columns else "",
            "rows": int(len(group)),
            "mean_response_length": round(float(group["response_length"].mean()), 3),
            "mean_response_retry_count": round(float(pd.to_numeric(group.get("response_retry_count", 0), errors="coerce").fillna(0).mean()), 3),
        }
        for key in SCORE_KEYS:
            row[f"mean_{key}"] = round(float(group[key].mean()), 4)
        row["mean_total"] = round(float(group[[key for key in SCORE_KEYS]].mean(axis=1).mean()), 4)
        rows.append(row)
    summary = pd.DataFrame(rows)
    if not summary.empty:
        summary = summary.sort_values(["mean_total", "condition"], ascending=[False, True]).reset_index(drop=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Score paper_matrix responses with an LLM judge.")
    parser.add_argument(
        "--input-csv",
        default=str(PROJECT_ROOT / "outputs" / "experiments" / "paper_matrix.csv"),
    )
    parser.add_argument(
        "--output-csv",
        default=str(PROJECT_ROOT / "outputs" / "experiments" / "paper_matrix_scored.csv"),
    )
    parser.add_argument(
        "--summary-csv",
        default=str(PROJECT_ROOT / "outputs" / "paper" / "requested_tables" / "baseline_generation_table.csv"),
    )
    parser.add_argument(
        "--summary-json",
        default=str(PROJECT_ROOT / "outputs" / "paper" / "requested_tables" / "baseline_generation_table.json"),
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--model-name", default="gpt-oss:120b-cloud")
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--flush-every", type=int, default=10)
    parser.add_argument("--keep-failures", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--reasoning-effort", default=None)
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    summary_csv = Path(args.summary_csv)
    summary_json = Path(args.summary_json)

    api_key = resolve_api_key(args.api_key_env)
    reasoning_effort = resolve_reasoning_effort(args.base_url, api_key, args.reasoning_effort)
    ensure_model_server_ready(args.base_url, args.timeout_sec, api_key=api_key)
    scored_df = score_matrix(
        input_csv=input_csv,
        output_csv=output_csv,
        base_url=args.base_url,
        model_name=args.model_name,
        timeout_sec=args.timeout_sec,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_retries=args.max_retries,
        progress_every=args.progress_every,
        flush_every=args.flush_every,
        keep_failures=args.keep_failures,
        resume=args.resume,
        limit=args.limit,
        api_key=api_key,
        reasoning_effort=reasoning_effort,
    )
    summary_df = summarize_scores(scored_df)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    ok_rows = 0
    error_rows = 0
    if "status" in scored_df.columns:
        status_series = scored_df["status"].fillna("").astype(str)
        ok_rows = int((status_series == "ok").sum())
        error_rows = int((status_series != "ok").sum())

    payload = {
        "rows": int(len(scored_df)),
        "ok_rows": ok_rows,
        "error_rows": error_rows,
        "conditions": summary_df.to_dict(orient="records"),
        "scored_csv": str(output_csv),
        "summary_csv": str(summary_csv),
    }
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
