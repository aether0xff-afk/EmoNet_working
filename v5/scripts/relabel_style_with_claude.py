from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import time
import urllib.error
import urllib.request
from typing import Any

import pandas as pd


RAW_AFFECT_AXES = [
    "hostility",
    "resentment",
    "despair",
    "volatility",
    "fearfulness",
    "shame",
    "relief",
    "trust",
]

STYLE_AXES = [
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
    *RAW_AFFECT_AXES,
]


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise
        payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("Claude response must be a JSON object")
    return payload


def _clamp_score(value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = 0.0
    return max(0.0, min(1.0, number))


def _normalize_axis_map(payload: dict[str, Any], key: str, axes: list[str]) -> dict[str, float]:
    raw = payload.get(key, {})
    if not isinstance(raw, dict):
        raw = {}
    return {axis: _clamp_score(raw.get(axis, 0.0)) for axis in axes}


def build_prompt(row: dict[str, Any]) -> str:
    current = {
        axis: row.get(f"current_{axis}", "")
        for axis in [
            "softness",
            "calmness",
            "cooperativeness",
            "positivity",
            "sharpness",
            "tension",
            "hostility",
            "resentment",
            "despair",
            "volatility",
            "fearfulness",
            "shame",
        ]
    }
    return "\n".join(
        [
            "[TASK]",
            "Re-label the Korean dialogue for EmoNet without defaulting to safe, soothing, or counselor-like surface tone.",
            "",
            "[CORE_RULE]",
            "Separate felt_state from response_style. Felt_state describes what emotion remains in the situation, even if a polite response would hide it.",
            "",
            "[INPUT_TEXT]",
            str(row.get("text", "")).strip(),
            "",
            "[CURRENT_WEAK_LABEL_HINT]",
            json.dumps(current, ensure_ascii=False),
            "",
            "[LABELING_GUIDANCE]",
            "- Use 0.00, 0.25, 0.50, 0.75, or 1.00.",
            "- Do not set hostility, resentment, despair, fearfulness, shame, or volatility to 0 merely because the ideal response should be polite.",
            "- If the user says they are angry, betrayed, resentful, ashamed, afraid, helpless, or exhausted, preserve that in felt_state.",
            "- response_style should describe a natural answer surface, but it must not erase felt_state.",
            "- calibrated_s is the final 40-axis target used for training; keep raw affect axes closer to felt_state than to a softened response.",
            "",
            "[OUTPUT_JSON_SCHEMA]",
            json.dumps(
                {
                    "felt_state": {axis: 0.0 for axis in RAW_AFFECT_AXES},
                    "response_style": {axis: 0.0 for axis in STYLE_AXES[:32]},
                    "calibrated_s": {axis: 0.0 for axis in STYLE_AXES},
                    "rationale": "short Korean explanation",
                },
                ensure_ascii=False,
            ),
            "",
            "Return JSON only.",
        ]
    )


def request_claude(
    *,
    api_key: str,
    model: str,
    prompt: str,
    max_tokens: int,
    temperature: float,
    timeout_sec: int,
) -> str:
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}],
    }
    request = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "content-type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        payload = json.loads(response.read().decode("utf-8"))
    chunks = payload.get("content", [])
    return "".join(chunk.get("text", "") for chunk in chunks if isinstance(chunk, dict))


def append_row(path: Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fieldnames})


def relabel_rows(
    *,
    input_csv: Path,
    output_csv: Path,
    raw_jsonl: Path,
    api_key: str,
    model: str,
    limit: int | None,
    resume: bool,
    max_retries: int,
    sleep_sec: float,
    max_tokens: int,
    temperature: float,
    timeout_sec: int,
) -> None:
    df = pd.read_csv(input_csv)
    done: set[str] = set()
    if resume and output_csv.exists():
        previous = pd.read_csv(output_csv)
        if "record_id" in previous.columns:
            done = set(previous["record_id"].astype(str))
    rows = df.to_dict(orient="records")
    if limit is not None:
        rows = rows[:limit]

    fieldnames = [
        "record_id",
        "relabel_bucket",
        "matched_cues",
        "rationale",
        *[f"felt_{axis}" for axis in RAW_AFFECT_AXES],
        *[f"calibrated_{axis}" for axis in STYLE_AXES],
    ]
    raw_jsonl.parent.mkdir(parents=True, exist_ok=True)
    for idx, row in enumerate(rows, start=1):
        record_id = str(row.get("record_id", "") or idx)
        if record_id in done:
            continue
        prompt = build_prompt(row)
        last_error = ""
        for attempt in range(max_retries + 1):
            try:
                raw_text = request_claude(
                    api_key=api_key,
                    model=model,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature if attempt == 0 else 0.0,
                    timeout_sec=timeout_sec,
                )
                payload = _extract_json(raw_text)
                felt = _normalize_axis_map(payload, "felt_state", RAW_AFFECT_AXES)
                calibrated = _normalize_axis_map(payload, "calibrated_s", STYLE_AXES)
                out = {
                    "record_id": record_id,
                    "relabel_bucket": row.get("relabel_bucket", ""),
                    "matched_cues": row.get("matched_cues", ""),
                    "rationale": str(payload.get("rationale", "")),
                }
                out.update({f"felt_{axis}": felt[axis] for axis in RAW_AFFECT_AXES})
                out.update({f"calibrated_{axis}": calibrated[axis] for axis in STYLE_AXES})
                append_row(output_csv, out, fieldnames)
                with raw_jsonl.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"record_id": record_id, "raw": payload}, ensure_ascii=False) + "\n")
                print(json.dumps({"record_id": record_id, "status": "ok", "index": idx}, ensure_ascii=False))
                break
            except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, ValueError) as exc:
                last_error = str(exc)
                if attempt >= max_retries:
                    print(json.dumps({"record_id": record_id, "status": "failed", "error": last_error}, ensure_ascii=False))
                else:
                    time.sleep(sleep_sec)
        if sleep_sec > 0:
            time.sleep(sleep_sec)


def main() -> None:
    parser = argparse.ArgumentParser(description="Relabel style hard cases with Claude using an ephemeral API key.")
    parser.add_argument("--input-csv", default="outputs/research/style_relabel_v1/style_relabel_candidates.csv")
    parser.add_argument("--output-csv", default="outputs/research/style_relabel_v1/style_relabel_claude.csv")
    parser.add_argument("--raw-jsonl", default="outputs/research/style_relabel_v1/style_relabel_claude_raw.jsonl")
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--sleep-sec", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--timeout-sec", type=int, default=120)
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY is required and is not stored by this script.")
    relabel_rows(
        input_csv=Path(args.input_csv),
        output_csv=Path(args.output_csv),
        raw_jsonl=Path(args.raw_jsonl),
        api_key=api_key,
        model=args.model,
        limit=args.limit,
        resume=args.resume,
        max_retries=args.max_retries,
        sleep_sec=args.sleep_sec,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        timeout_sec=args.timeout_sec,
    )


if __name__ == "__main__":
    main()
