from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from emonet.cli import (
    DEFAULT_STYLE_PROFILE,
    append_csv_rows,
    append_jsonl,
    build_model as build_emonet_model,
    build_response_generation_prompt,
    ensure_model_server_ready,
    format_expression_cue_lines,
    format_style_summary_lines,
    format_style_vector_lines,
    infer_style_profile,
    maybe_print_progress,
    request_plain_text_response,
    resolve_text_column,
    serialize_generation_log,
    utc_timestamp,
    validate_plain_response_text,
)
from emonet.core import EmoNet, LinearZtoSDecoder


CONDITION_SPECS: dict[str, dict[str, object]] = {
    "direct": {
        "group": "baseline",
        "description": "Input text only direct prompting.",
        "needs_profile": False,
    },
    "stim_only": {
        "group": "baseline",
        "description": "Input text plus 4D stim vector only.",
        "needs_profile": True,
    },
    "emonet_full": {
        "group": "emonet",
        "description": "Full EmoNet prompt with tags, summary, cues, and vector.",
        "needs_profile": True,
    },
    "emonet_no_summary": {
        "group": "ablation",
        "description": "Remove STYLE_SUMMARY from the full prompt.",
        "needs_profile": True,
    },
    "emonet_no_tags": {
        "group": "ablation",
        "description": "Remove STYLE_TAGS from the full prompt.",
        "needs_profile": True,
    },
    "emonet_no_expression": {
        "group": "ablation",
        "description": "Remove EXPRESSION_CUES from the full prompt.",
        "needs_profile": True,
    },
    "emonet_vector_only": {
        "group": "ablation",
        "description": "Use raw style vector only.",
        "needs_profile": True,
    },
    "emonet_macro_only": {
        "group": "ablation",
        "description": "Use STYLE_TAGS and STYLE_SUMMARY without the raw vector.",
        "needs_profile": True,
    },
}

DEFAULT_CONDITIONS = [
    "direct",
    "stim_only",
    "emonet_full",
    "emonet_no_summary",
    "emonet_no_tags",
    "emonet_vector_only",
]

OUTPUT_COLUMNS = [
    "record_id",
    "talk_id",
    "sample_id",
    "text",
    "condition",
    "condition_group",
    "condition_description",
    "status",
    "error_message",
    "prompt",
    "llm_response",
    "response_length",
    "style_sections",
    "dominant_branch_len",
    "style_summary_text",
    "expression_cues_text",
    "anti_softening_mode",
    "anti_softening_rules_json",
    "response_retry_count",
    "response_validation_errors_json",
    "style_tags_json",
    "style_summary_json",
    "stim_vec_json",
    "s_pred_json",
    "decoder_model_path",
    "llm_model_name",
    "response_temperature",
    "timestamp_utc",
]


def parse_conditions(raw: str | None) -> list[str]:
    if not raw:
        return list(DEFAULT_CONDITIONS)

    tokens = [token.strip() for token in raw.replace(";", ",").split(",")]
    conditions = [token for token in tokens if token]
    invalid = [name for name in conditions if name not in CONDITION_SPECS]
    if invalid:
        valid = ", ".join(sorted(CONDITION_SPECS))
        raise ValueError(f"invalid conditions: {', '.join(invalid)}. valid conditions: {valid}")
    if not conditions:
        raise ValueError("at least one condition is required")
    return conditions


def build_model(args: argparse.Namespace) -> EmoNet:
    return build_emonet_model(args)


def build_direct_prompt(input_text: str) -> str:
    return "\n".join(
        [
            "[ROLE]",
            "당신은 한국어 감정 응답 생성기다.",
            "",
            "[USER_INPUT]",
            input_text.strip(),
            "",
            "[INSTRUCTIONS]",
            "- 사용자 입력의 내용에 직접 답한다.",
            "- 말투를 설명하지 말고 자연스럽게 응답한다.",
            "- 한국어 평문으로만 3~6문장 이내로 답한다.",
            "- 같은 문장이나 핵심 구절을 반복하지 않는다.",
            "- 문장을 중간에 끊지 말고 마지막 문장은 완결된 문장으로 끝낸다.",
            "- bullet, markdown, JSON, 코드블록을 쓰지 않는다.",
        ]
    )


def describe_stim_level(value: float) -> str:
    if value >= 0.75:
        return "매우 높음"
    if value >= 0.60:
        return "높음"
    if value >= 0.40:
        return "중간"
    if value >= 0.25:
        return "낮음"
    return "매우 낮음"


def build_stim_only_prompt(input_text: str, stim_vec: list[float]) -> str:
    stim_names = ["dopamine", "serotonin", "norepinephrine", "melatonin"]
    stim_lines = [
        f"{name}={value:.4f} ({describe_stim_level(value)})"
        for name, value in zip(stim_names, stim_vec, strict=True)
    ]
    return "\n".join(
        [
            "[ROLE]",
            "당신은 입력 문장과 정서 자극 힌트를 참고해 한국어로 답하는 응답 생성기다.",
            "",
            "[USER_INPUT]",
            input_text.strip(),
            "",
            "[AFFECT_STIMULUS]",
            *stim_lines,
            "",
            "[INSTRUCTIONS]",
            "- 사용자 입력의 내용에 직접 답한다.",
            "- AFFECT_STIMULUS를 참고해 전반적인 정서 분위기만 조절한다.",
            "- 숫자를 그대로 언급하지 않는다.",
            "- 한국어 평문으로만 3~6문장 이내로 답한다.",
            "- 같은 문장이나 핵심 구절을 반복하지 않는다.",
            "- 문장을 중간에 끊지 말고 마지막 문장은 완결된 문장으로 끝낸다.",
            "- bullet, markdown, JSON, 코드블록을 쓰지 않는다.",
        ]
    )


def build_variant_prompt(
    input_text: str,
    style_dict: dict[str, float],
    style_tags: list[str],
    style_summary: dict[str, float],
    anti_softening_rules: list[str],
    *,
    include_tags: bool,
    include_summary: bool,
    include_expression: bool,
    include_vector: bool,
) -> tuple[str, str]:
    sections: list[str] = []
    lines = [
        "[ROLE]",
        "당신은 감정 상태에 맞는 말투와 리듬으로 답하는 한국어 응답 생성기다.",
        "",
        "[USER_INPUT]",
        input_text.strip(),
        "",
    ]

    if include_tags:
        sections.append("style_tags")
        lines.extend(["[STYLE_TAGS]", ", ".join(style_tags) if style_tags else "(none)", ""])
    if include_summary:
        sections.append("style_summary")
        lines.extend(["[STYLE_SUMMARY]", *format_style_summary_lines(style_summary), ""])
    if include_expression:
        sections.append("expression_cues")
        lines.extend(["[EXPRESSION_CUES]", *format_expression_cue_lines(style_dict), ""])
    if include_vector:
        sections.append("style_vector")
        lines.extend(["[STYLE_VECTOR]", format_style_vector_lines(style_dict), ""])
    if anti_softening_rules:
        sections.append("anti_softening_rules")
        lines.extend(["[ANTI_SOFTENING_RULES]", *[f"- {rule}" for rule in anti_softening_rules], ""])

    instruction_parts = []
    if include_vector:
        instruction_parts.append("STYLE_VECTOR")
    if include_tags:
        instruction_parts.append("STYLE_TAGS")
    if include_summary:
        instruction_parts.append("STYLE_SUMMARY")
    if include_expression:
        instruction_parts.append("EXPRESSION_CUES")

    lines.extend(
        [
            "[INSTRUCTIONS]",
            "- 사용자 입력의 내용에 직접 답한다.",
            (
                f"- {', '.join(instruction_parts)}를 참고해 말투와 표현 밀도를 조절한다."
                if instruction_parts
                else "- 별도의 스타일 힌트 없이 자연스럽게 응답한다."
            ),
            "- 스타일을 설명하지 말고, 그 스타일로 자연스럽게 답한다.",
            "- 한국어 평문으로만 3~6문장 이내로 답한다.",
            "- 같은 문장이나 핵심 구절을 반복하지 않는다.",
            "- 문장을 중간에 끊지 말고 마지막 문장은 완결된 문장으로 끝낸다.",
            "- bullet, markdown, JSON, 코드블록을 쓰지 않는다.",
        ]
    )
    return "\n".join(lines), ",".join(sections) if sections else "none"


def build_condition_prompt(condition: str, input_text: str, profile: dict[str, object] | None) -> tuple[str, str]:
    if condition == "direct":
        return build_direct_prompt(input_text), "input_only"
    if profile is None:
        raise ValueError(f"profile is required for condition '{condition}'")

    stim_vec = [float(value) for value in profile["stim_vec"]]
    style_dict = dict(profile["style_dict"])
    style_tags = list(profile["style_tags"])
    style_summary = dict(profile["style_summary"])
    anti_softening_rules = list(profile.get("anti_softening_rules", []))

    if condition == "stim_only":
        return build_stim_only_prompt(input_text, stim_vec), "stim_vec"
    if condition == "emonet_full":
        return (
            build_response_generation_prompt(
                input_text=input_text,
                style_dict=style_dict,
                style_tags=style_tags,
                style_summary=style_summary,
                anti_softening_rules=anti_softening_rules,
                template_path=None,
            ),
            "style_tags,style_summary,anti_softening_rules",
        )
    if condition == "emonet_no_summary":
        return build_variant_prompt(
            input_text=input_text,
            style_dict=style_dict,
            style_tags=style_tags,
            style_summary=style_summary,
            anti_softening_rules=anti_softening_rules,
            include_tags=True,
            include_summary=False,
            include_expression=True,
            include_vector=True,
        )
    if condition == "emonet_no_tags":
        return build_variant_prompt(
            input_text=input_text,
            style_dict=style_dict,
            style_tags=style_tags,
            style_summary=style_summary,
            anti_softening_rules=anti_softening_rules,
            include_tags=False,
            include_summary=True,
            include_expression=True,
            include_vector=True,
        )
    if condition == "emonet_no_expression":
        return build_variant_prompt(
            input_text=input_text,
            style_dict=style_dict,
            style_tags=style_tags,
            style_summary=style_summary,
            anti_softening_rules=anti_softening_rules,
            include_tags=True,
            include_summary=True,
            include_expression=False,
            include_vector=True,
        )
    if condition == "emonet_vector_only":
        return build_variant_prompt(
            input_text=input_text,
            style_dict=style_dict,
            style_tags=style_tags,
            style_summary=style_summary,
            anti_softening_rules=anti_softening_rules,
            include_tags=False,
            include_summary=False,
            include_expression=False,
            include_vector=True,
        )
    if condition == "emonet_macro_only":
        return build_variant_prompt(
            input_text=input_text,
            style_dict=style_dict,
            style_tags=style_tags,
            style_summary=style_summary,
            anti_softening_rules=anti_softening_rules,
            include_tags=True,
            include_summary=True,
            include_expression=True,
            include_vector=False,
        )
    raise ValueError(f"unsupported condition: {condition}")


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


def summarize_output(path: Path) -> dict[str, object]:
    df = pd.read_csv(path)
    by_condition: dict[str, object] = {}
    for condition, group in df.groupby("condition", dropna=False):
        ok_mask = group["status"].fillna("") == "ok"
        response_lengths = pd.to_numeric(group.loc[ok_mask, "response_length"], errors="coerce").dropna()
        by_condition[str(condition)] = {
            "rows": int(len(group)),
            "ok_rows": int(ok_mask.sum()),
            "error_rows": int((~ok_mask).sum()),
            "mean_response_length": round(float(response_lengths.mean()), 3) if len(response_lengths) else None,
        }
    return {
        "path": str(path),
        "rows": int(len(df)),
        "conditions": by_condition,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline and ablation response-generation experiments.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--summary-json", default=None)
    parser.add_argument("--log-jsonl", default=None)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--record-id-column", default="sample_id")
    parser.add_argument("--talk-id-column", default="talk_id")
    parser.add_argument("--conditions", default=",".join(DEFAULT_CONDITIONS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--sample-mode", choices=["head", "random"], default="random")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--flush-every", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--zs-model-path", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--model-name", default="gpt-oss:20b")
    parser.add_argument("--response-temperature", type=float, default=0.5)
    parser.add_argument("--response-max-retries", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=600)
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--dataset-csv", dest="dataset_csv", type=str, default=None)
    parser.add_argument("--benchmark-csv", dest="benchmark_csv", type=str, default=None)
    parser.add_argument("--model-cache-path", dest="model_cache_path", type=str, default=None)
    parser.add_argument("--max-samples", dest="max_samples", type=int, default=None)
    parser.add_argument("--force-refit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--z-dim", dest="z_dim", type=int, default=64)
    parser.add_argument("--z-encoder-mode", choices=["auto", "stat", "transformer"], default="auto")
    parser.add_argument("--z-encoder-path", default=None)
    parser.add_argument("--style-profile", choices=["core32", "extended40"], default=DEFAULT_STYLE_PROFILE)
    args = parser.parse_args()

    conditions = parse_conditions(args.conditions)
    output_csv = Path(args.output_csv)
    log_jsonl = Path(args.log_jsonl) if args.log_jsonl else None
    summary_json = Path(args.summary_json) if args.summary_json else None

    ensure_model_server_ready(args.base_url, args.timeout_sec)
    model = build_model(args)
    decoder = LinearZtoSDecoder.load(Path(args.zs_model_path))

    input_df = pd.read_csv(Path(args.input_csv))
    text_column = resolve_text_column(input_df, args.text_column)
    if args.sample_size is not None and args.sample_size > 0 and len(input_df) > args.sample_size:
        if args.sample_mode == "random":
            input_df = input_df.sample(n=args.sample_size, random_state=args.sample_seed).reset_index(drop=True)
        else:
            input_df = input_df.head(args.sample_size).copy()
    if args.limit is not None and args.limit > 0:
        input_df = input_df.head(args.limit).copy()

    existing_keys = load_existing_keys(output_csv) if args.resume else set()
    rows_buffer: list[dict[str, object]] = []
    jsonl_buffer: list[dict[str, object]] = []
    start_time = time.perf_counter()

    for idx, record in enumerate(input_df.to_dict(orient="records"), start=1):
        text = str(record.get(text_column, "")).strip()
        if not text:
            continue

        fallback_id = f"row_{idx:06d}"
        record_id = str(record.get(args.record_id_column) or fallback_id)
        talk_id = str(record.get(args.talk_id_column, "") or "")
        sample_id = str(record.get("sample_id", "") or "")

        profile: dict[str, object] | None = None
        profile_error: Exception | None = None
        if any(bool(CONDITION_SPECS[name]["needs_profile"]) for name in conditions):
            try:
                profile = infer_style_profile(model=model, decoder=decoder, text=text, style_profile=args.style_profile)
            except Exception as exc:  # pragma: no cover
                profile_error = exc

        for condition in conditions:
            key = (record_id, condition)
            if key in existing_keys:
                continue

            spec = CONDITION_SPECS[condition]
            row = {
                "record_id": record_id,
                "talk_id": talk_id,
                "sample_id": sample_id,
                "text": text,
                "condition": condition,
                "condition_group": str(spec["group"]),
                "condition_description": str(spec["description"]),
                "status": "error",
                "error_message": "",
                "prompt": "",
                "llm_response": "",
                "response_length": None,
                "style_sections": "",
                "dominant_branch_len": None,
                "style_summary_text": "",
                "expression_cues_text": "",
                "anti_softening_mode": "",
                "anti_softening_rules_json": "[]",
                "response_retry_count": 0,
                "response_validation_errors_json": "[]",
                "style_tags_json": "[]",
                "style_summary_json": "{}",
                "stim_vec_json": "[]",
                "s_pred_json": "[]",
                "decoder_model_path": str(args.zs_model_path),
                "llm_model_name": args.model_name,
                "response_temperature": args.response_temperature,
                "timestamp_utc": utc_timestamp(),
            }

            if bool(spec["needs_profile"]) and profile is None:
                row["error_message"] = str(profile_error or "style profile could not be inferred")
                rows_buffer.append(row)
                continue

            try:
                prompt, style_sections = build_condition_prompt(condition, text, profile)
                response_text, _raw_output, response_meta = request_plain_text_response(
                    base_url=args.base_url,
                    model_name=args.model_name,
                    prompt=prompt,
                    temperature=args.response_temperature,
                    max_tokens=args.max_tokens,
                    timeout_sec=args.timeout_sec,
                    max_retries=args.response_max_retries,
                    validator=validate_plain_response_text,
                    retry_instruction=(
                        "직전 응답은 반복, 미완성 문장, bullet/JSON, 혹은 부자연스러운 출력 때문에 거부되었다. "
                        "같은 문장이나 핵심 구절을 반복하지 말고 마지막 문장은 완결된 한국어 평문으로 끝내라."
                    ),
                    system_prompt="Return a plain Korean response only. Do not return JSON.",
                )

                row["status"] = "ok"
                row["prompt"] = prompt
                row["llm_response"] = response_text
                row["response_length"] = len(response_text)
                row["style_sections"] = style_sections
                row["response_retry_count"] = int(response_meta["retry_count"])
                row["response_validation_errors_json"] = json.dumps(response_meta["validation_errors"], ensure_ascii=False)
                if profile is not None:
                    row["dominant_branch_len"] = int(profile["dominant_branch_len"])
                    row["style_summary_text"] = str(profile["style_summary_text"])
                    row["expression_cues_text"] = str(profile["expression_cues_text"])
                    row["anti_softening_mode"] = str(profile.get("anti_softening_mode", ""))
                    row["anti_softening_rules_json"] = json.dumps(profile.get("anti_softening_rules", []), ensure_ascii=False)
                    row["style_tags_json"] = json.dumps(profile["style_tags"], ensure_ascii=False)
                    row["style_summary_json"] = json.dumps(profile["style_summary"], ensure_ascii=False)
                    row["stim_vec_json"] = json.dumps([float(value) for value in profile["stim_vec"]], ensure_ascii=False)
                    row["s_pred_json"] = json.dumps([float(value) for value in profile["s_pred"]], ensure_ascii=False)

                if profile is not None:
                    jsonl_payload = serialize_generation_log(
                        {
                            "record_id": record_id,
                            "talk_id": talk_id,
                            "condition": condition,
                            "input_text": text,
                            "stim_vec": [float(value) for value in profile["stim_vec"]],
                            "z": [float(value) for value in profile["z"]],
                            "s_pred": [float(value) for value in profile["s_pred"]],
                            "style_tags": list(profile["style_tags"]),
                            "style_summary": dict(profile["style_summary"]),
                            "style_summary_text": str(profile["style_summary_text"]),
                            "expression_cues_text": str(profile["expression_cues_text"]),
                            "anti_softening_mode": str(profile.get("anti_softening_mode", "")),
                            "anti_softening_rules": list(profile.get("anti_softening_rules", [])),
                            "response_retry_count": int(response_meta["retry_count"]),
                            "response_validation_errors": list(response_meta["validation_errors"]),
                            "style_prompt": prompt,
                            "llm_response": response_text,
                            "llm_model_name": args.model_name,
                            "timestamp_utc": row["timestamp_utc"],
                        }
                    )
                else:
                    jsonl_payload = {
                        "record_id": record_id,
                        "talk_id": talk_id,
                        "condition": condition,
                        "input_text": text,
                        "style_prompt": prompt,
                        "llm_response": response_text,
                        "response_retry_count": int(response_meta["retry_count"]),
                        "response_validation_errors": list(response_meta["validation_errors"]),
                        "llm_model_name": args.model_name,
                        "timestamp_utc": row["timestamp_utc"],
                    }
                jsonl_buffer.append(jsonl_payload)
            except Exception as exc:
                row["error_message"] = str(exc)

            rows_buffer.append(row)

        if args.flush_every > 0 and len(rows_buffer) >= args.flush_every:
            append_csv_rows(output_csv, rows_buffer, columns=OUTPUT_COLUMNS)
            rows_buffer.clear()
            if log_jsonl is not None and jsonl_buffer:
                append_jsonl(log_jsonl, jsonl_buffer)
                jsonl_buffer.clear()

        maybe_print_progress("experiment-matrix", idx, len(input_df), start_time, every=args.progress_every)

    if rows_buffer:
        append_csv_rows(output_csv, rows_buffer, columns=OUTPUT_COLUMNS)
    if log_jsonl is not None and jsonl_buffer:
        append_jsonl(log_jsonl, jsonl_buffer)

    summary = summarize_output(output_csv)
    if summary_json is not None:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
