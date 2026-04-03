from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from emonet.cli import append_csv_rows, ensure_model_server_ready, request_json_response


SCORE_KEYS = [
    "content_fit",
    "emotional_appropriateness",
    "style_match",
    "naturalness",
    "overall_quality",
]


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
    target_summary = str(row.get("style_summary_json", "{}")).strip()
    target_tags = str(row.get("style_tags_json", "[]")).strip()

    return "\n".join(
        [
            "[ROLE]",
            "당신은 한국어 감정 응답 품질을 채점하는 심사자다.",
            "",
            "[INPUT_TEXT]",
            text,
            "",
            "[MODEL_CONDITION]",
            condition,
            "",
            "[TARGET_STYLE_TAGS_JSON]",
            target_tags,
            "",
            "[TARGET_STYLE_SUMMARY_JSON]",
            target_summary,
            "",
            "[MODEL_RESPONSE]",
            response,
            "",
            "[SCORING_RULE]",
            "- 각 항목을 1점(매우 나쁨)부터 5점(매우 좋음)까지 정수로 평가한다.",
            "- content_fit: 입력 내용에 직접적으로 맞는가",
            "- emotional_appropriateness: 입력 감정 상태에 맞는가",
            "- style_match: 목표 스타일 요약과 태그에 얼마나 맞는가",
            "- naturalness: 한국어 응답이 자연스러운가",
            "- overall_quality: 전체적으로 설득력 있는가",
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


def load_existing_keys(output_csv: Path) -> set[tuple[str, str]]:
    if not output_csv.exists() or output_csv.stat().st_size == 0:
        return set()
    df = pd.read_csv(output_csv, usecols=["record_id", "condition"])
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
) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    ok_df = df[df["status"].fillna("") == "ok"].copy()
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
        "judge_raw_output",
        *SCORE_KEYS,
    ]

    for idx, row in enumerate(ok_df.to_dict(orient="records"), start=1):
        record_id = str(row.get("record_id", row.get("sample_id", f"row_{idx:06d}")))
        condition = str(row.get("condition", ""))
        key = (record_id, condition)
        if key in existing_keys:
            continue

        scored = {
            "record_id": record_id,
            "condition": condition,
            "condition_group": str(row.get("condition_group", "")),
            "status": "error",
            "error_message": "",
            "text": str(row.get("text", "")),
            "llm_response": str(row.get("llm_response", "")),
            "response_length": int(len(str(row.get("llm_response", "")))),
            "judge_raw_output": "",
        }
        for key_name in SCORE_KEYS:
            scored[key_name] = pd.NA

        prompt = build_judge_prompt(row)
        try:
            payload, raw = request_json_response(
                base_url=base_url,
                model_name=model_name,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_sec=timeout_sec,
                max_retries=max_retries,
                validator=normalize_scores,
                retry_instruction=(
                    "직전 응답의 JSON 형식 또는 점수 범위가 잘못되었다. "
                    "반드시 scores object 안에 다섯 항목을 1~5 정수로 다시 출력하라."
                ),
            )
            scored["status"] = "ok"
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

        if progress_every > 0 and idx % progress_every == 0:
            print(f"scored {idx}/{len(ok_df)} rows")

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
    parser.add_argument("--model-name", default="gpt-oss:20b")
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--max-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--flush-every", type=int, default=10)
    parser.add_argument("--keep-failures", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    summary_csv = Path(args.summary_csv)
    summary_json = Path(args.summary_json)

    ensure_model_server_ready(args.base_url, args.timeout_sec)
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
    )
    summary_df = summarize_scores(scored_df)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    payload = {
        "rows": int(len(scored_df)),
        "conditions": summary_df.to_dict(orient="records"),
        "scored_csv": str(output_csv),
        "summary_csv": str(summary_csv),
    }
    summary_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
