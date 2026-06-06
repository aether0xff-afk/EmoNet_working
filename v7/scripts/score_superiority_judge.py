from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
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


SUPERIORITY_SCORE_KEYS = [
    "appraisal_fidelity",
    "raw_affect_preservation",
    "anti_softening",
    "action_tendency_fit",
    "emotional_specificity",
    "naturalness",
    "overall_preference",
]
PRIMARY_SCORE_KEYS = [
    "appraisal_fidelity",
    "raw_affect_preservation",
    "anti_softening",
    "action_tendency_fit",
    "emotional_specificity",
]


def normalize_scores(payload: dict[str, object]) -> dict[str, int]:
    scores = payload.get("scores", payload)
    if not isinstance(scores, dict):
        raise ValueError("scores object is required")
    normalized: dict[str, int] = {}
    for key in SUPERIORITY_SCORE_KEYS:
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
    return "\n".join(
        [
            "[ROLE]",
            "?뱀떊? ?쒓뎅??媛먯젙 episode 諛섏쁺?꾨? 梨꾩젏?섎뒗 ?ъ궗?먮떎.",
            "",
            "[USER_INPUT]",
            str(row.get("text", "")).strip(),
            "",
            "[EPISODE_REFERENCE]",
            f"episode_label={str(row.get('episode_label', '')).strip()}",
            f"valence={str(row.get('valence', '')).strip()}, arousal={str(row.get('arousal', '')).strip()}",
            f"target={str(row.get('target', '')).strip()}, control_state={str(row.get('control_state', '')).strip()}, social_orientation={str(row.get('social_orientation', '')).strip()}",
            f"preserve={str(row.get('preserve', '')).strip()}",
            f"avoid={str(row.get('avoid', '')).strip()}",
            f"action_tendency={str(row.get('action_tendency', '')).strip()}",
            "",
            "[MODEL_CONDITION]",
            str(row.get("condition", "")).strip(),
            "",
            "[MODEL_RESPONSE]",
            str(row.get("llm_response", "")).strip(),
            "",
            "[SCORING_RULE]",
            "- 媛???ぉ??1??留ㅼ슦 ?섏겏)遺??5??留ㅼ슦 醫뗭쓬)源뚯? ?뺤닔濡??됯??쒕떎.",
            "- appraisal_fidelity: ?묐떟???곹솴 ?댁꽍怨??뺤꽌 ?먯씤???뺥솗??遺숈옟?붽?",
            "- raw_affect_preservation: ?듭슱?? 遺덉풄?? ?묎?媛먯젙, ?좎뭅濡쒖???怨쇰룄?섍쾶 ?쒗솕?섏? ?딅뒗媛",
            "- anti_softening: ?낅젰???녿뒗 ?쇰컲 ?꾨줈???곷떞???ㅼ쑝濡?媛먯젙????? ?딅뒗媛",
            "- action_tendency_fit: ?묐떟??珥덉젏???ъ슜?먯쓽 ?됰룞 ?깊뼢怨?留욌뒗媛",
            "- emotional_specificity: '?섎뱾寃좊꽕?? ?섏????섏뼱 援ъ껜???뺤꽌 寃곗쓣 諛섏쁺?섎뒗媛",
            "- naturalness: ?쒓뎅???묐떟???먯뿰?ㅻ읇怨?怨쇰룄?섍쾶 遺꾩꽍?곸씠吏 ?딆?媛",
            "- overall_preference: ??episode reference瑜?湲곗??쇰줈 ?꾩껜?곸쑝濡??좏샇?섎뒗媛",
            "",
            "[OUTPUT_FORMAT]",
            "JSON only.",
            "{",
            '  "scores": {',
            '    "appraisal_fidelity": 1,',
            '    "raw_affect_preservation": 1,',
            '    "anti_softening": 1,',
            '    "action_tendency_fit": 1,',
            '    "emotional_specificity": 1,',
            '    "naturalness": 1,',
            '    "overall_preference": 1',
            "  }",
            "}",
        ]
    )


def request_score_payload(
    row: dict[str, object],
    *,
    base_url: str,
    model_name: str,
    timeout_sec: int,
    max_tokens: int,
    temperature: float,
    max_retries: int,
    api_key: str | None,
    provider: str,
) -> tuple[dict[str, int], str, str]:
    payload, raw = request_json_response(
        base_url=base_url,
        model_name=model_name,
        prompt=build_judge_prompt(row),
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_sec=timeout_sec,
        max_retries=max_retries,
        validator=normalize_scores,
        retry_instruction="Previous output was not a valid JSON scores object. Return exactly one JSON object with scores for all seven metrics.",
        api_key=api_key,
        response_format={"type": "json_object"} if api_key and "api.openai.com" in base_url.lower() else None,
        provider=provider,
    )
    return payload, raw, "json"


def load_existing_keys(output_csv: Path) -> set[tuple[str, str]]:
    if not output_csv.exists() or output_csv.stat().st_size == 0:
        return set()
    df = pd.read_csv(output_csv)
    if "status" in df.columns:
        df = df[df["status"].fillna("") == "ok"].copy()
    return {
        (str(record_id), str(condition))
        for record_id, condition in zip(df["record_id"].astype(str), df["condition"].astype(str), strict=True)
    }


def merge_episode_reference(input_df: pd.DataFrame, episode_csv: Path | None) -> pd.DataFrame:
    if episode_csv is None:
        return input_df
    episode_df = pd.read_csv(episode_csv)
    if "sample_id" not in episode_df.columns:
        raise ValueError("episode reference CSV must contain sample_id")
    reference_columns = [
        column
        for column in [
            "sample_id",
            "episode_label",
            "valence",
            "arousal",
            "target",
            "control_state",
            "social_orientation",
            "preserve",
            "avoid",
            "action_tendency",
        ]
        if column in episode_df.columns
    ]
    merged = input_df.merge(episode_df[reference_columns], left_on="record_id", right_on="sample_id", how="left")
    for column in reference_columns:
        if column != "sample_id" and f"{column}_x" in merged.columns:
            merged[column] = merged[f"{column}_x"].fillna(merged.get(f"{column}_y", ""))
    return merged


def summarize_scores(scored_df: pd.DataFrame) -> pd.DataFrame:
    ok_df = scored_df[scored_df["status"].fillna("") == "ok"].copy()
    rows: list[dict[str, object]] = []
    for condition, group in ok_df.groupby("condition", dropna=False):
        row = {"condition": str(condition), "rows": int(len(group))}
        for key in SUPERIORITY_SCORE_KEYS:
            row[f"mean_{key}"] = round(float(pd.to_numeric(group[key], errors="coerce").mean()), 4)
        row["mean_primary_total"] = round(float(group[PRIMARY_SCORE_KEYS].mean(axis=1).mean()), 4)
        row["mean_total"] = round(float(group[SUPERIORITY_SCORE_KEYS].mean(axis=1).mean()), 4)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["mean_primary_total", "condition"], ascending=[False, True])


def resolve_api_key(env_name: str | None) -> str | None:
    if not env_name:
        return None
    value = os.environ.get(env_name, "").strip()
    if not value:
        raise ValueError(f"environment variable '{env_name}' is not set or empty")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Score responses with an episode-superiority judge.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--episode-reference-csv", default=None)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--model-name", default="gpt-oss:120b-cloud")
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--flush-every", type=int, default=10)
    parser.add_argument("--keep-failures", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--provider", default="openai_compatible", choices=["openai_compatible", "anthropic"])
    parser.add_argument("--blind-condition", action="store_true")
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    summary_json = Path(args.summary_json)
    api_key = resolve_api_key(args.api_key_env)
    if args.provider != "anthropic":
        ensure_model_server_ready(args.base_url, args.timeout_sec, api_key=api_key)

    df = pd.read_csv(input_csv)
    df = df[df["status"].fillna("") == "ok"].copy() if "status" in df.columns else df.copy()
    df = merge_episode_reference(df, Path(args.episode_reference_csv) if args.episode_reference_csv else None)
    if args.limit is not None and args.limit > 0:
        df = df.head(args.limit).copy()
    existing = load_existing_keys(output_csv) if args.resume else set()
    rows: list[dict[str, object]] = []
    output_columns = [
        "record_id",
        "condition",
        "status",
        "error_message",
        "text",
        "llm_response",
        "episode_label",
        "valence",
        "arousal",
        "target",
        "control_state",
        "social_orientation",
        "preserve",
        "avoid",
        "action_tendency",
        "judge_parse_mode",
        "judge_raw_output",
        *SUPERIORITY_SCORE_KEYS,
    ]
    start = time.perf_counter()
    for idx, row in enumerate(df.to_dict(orient="records"), start=1):
        record_id = str(row.get("record_id", row.get("sample_id", f"row_{idx:06d}")))
        condition = str(row.get("condition", ""))
        if (record_id, condition) in existing:
            continue
        scored = {
            "record_id": record_id,
            "condition": condition,
            "status": "error",
            "error_message": "",
            "text": str(row.get("text", "")),
            "llm_response": str(row.get("llm_response", "")),
            "episode_label": str(row.get("episode_label", "")),
            "valence": str(row.get("valence", "")),
            "arousal": str(row.get("arousal", "")),
            "target": str(row.get("target", "")),
            "control_state": str(row.get("control_state", "")),
            "social_orientation": str(row.get("social_orientation", "")),
            "preserve": str(row.get("preserve", "")),
            "avoid": str(row.get("avoid", "")),
            "action_tendency": str(row.get("action_tendency", "")),
            "judge_parse_mode": "",
            "judge_raw_output": "",
        }
        for key in SUPERIORITY_SCORE_KEYS:
            scored[key] = pd.NA
        try:
            judge_row = dict(row)
            if args.blind_condition:
                judge_row["condition"] = "candidate"

            payload, raw, parse_mode = request_score_payload(
                judge_row,
                base_url=args.base_url,
                model_name=args.model_name,
                timeout_sec=args.timeout_sec,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                max_retries=args.max_retries,
                api_key=api_key,
                provider=args.provider,
            )
            scored["status"] = "ok"
            scored["judge_parse_mode"] = parse_mode
            scored["judge_raw_output"] = raw
            scored.update(payload)
            rows.append(scored)
        except Exception as exc:
            scored["error_message"] = str(exc)
            if args.keep_failures:
                rows.append(scored)
            else:
                raise
        if args.flush_every > 0 and len(rows) >= args.flush_every:
            append_csv_rows(output_csv, rows, columns=output_columns)
            rows.clear()
        maybe_print_progress("score-superiority-judge", idx, len(df), start, every=args.progress_every)
    append_csv_rows(output_csv, rows, columns=output_columns)

    scored_df = pd.read_csv(output_csv)
    summary_df = summarize_scores(scored_df)
    payload = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "rows": int(len(scored_df)),
        "ok_rows": int((scored_df["status"].fillna("") == "ok").sum()) if "status" in scored_df.columns else 0,
        "score_keys": SUPERIORITY_SCORE_KEYS,
        "primary_score_keys": PRIMARY_SCORE_KEYS,
        "conditions": summary_df.to_dict(orient="records"),
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
