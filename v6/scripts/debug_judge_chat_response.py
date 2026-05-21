from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import urllib.request

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from emonet.cli import ensure_model_server_ready
from scripts.score_experiment_matrix import (
    build_judge_prompt,
)


def send_raw_chat_completion(
    *,
    base_url: str,
    model_name: str,
    prompt: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout_sec: int,
    api_key: str | None,
) -> dict[str, object]:
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
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(
        url,
        data=data,
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_sec) as response:
        body_bytes = response.read()
        status_code = int(response.status)
        headers = dict(response.headers.items())
    body_text = body_bytes.decode("utf-8", errors="replace")
    try:
        body_json = json.loads(body_text)
    except Exception:
        body_json = None
    return {
        "url": url,
        "request_payload": payload,
        "http_status": status_code,
        "headers": headers,
        "response_text": body_text,
        "response_json": body_json,
    }


def summarize_response(result: dict[str, object]) -> dict[str, object]:
    body_json = result.get("response_json")
    summary: dict[str, object] = {
        "http_status": result.get("http_status"),
        "response_text_preview": str(result.get("response_text", ""))[:500],
        "has_json_body": isinstance(body_json, dict),
    }
    if not isinstance(body_json, dict):
        return summary

    choices = body_json.get("choices", [])
    summary["num_choices"] = len(choices) if isinstance(choices, list) else 0
    if not choices or not isinstance(choices, list):
        return summary

    choice0 = choices[0] if isinstance(choices[0], dict) else {}
    message = choice0.get("message", {})
    summary["finish_reason"] = choice0.get("finish_reason")
    summary["choice_keys"] = sorted(choice0.keys()) if isinstance(choice0, dict) else []
    summary["message_keys"] = sorted(message.keys()) if isinstance(message, dict) else []
    content = message.get("content", "") if isinstance(message, dict) else ""
    summary["content_type"] = type(content).__name__
    if isinstance(content, str):
        summary["content_len"] = len(content)
        summary["content_preview"] = content[:500]
    else:
        summary["content_preview"] = str(content)[:500]
    return summary


def load_target_row(input_csv: Path, record_id: str | None, condition: str | None) -> dict[str, object]:
    df = pd.read_csv(input_csv)
    ok_df = df[df["status"].fillna("") == "ok"].copy()
    if ok_df.empty:
        raise ValueError("no successful matrix rows found")
    if record_id is not None:
        ok_df = ok_df[ok_df["record_id"].astype(str) == str(record_id)].copy()
    if condition is not None:
        ok_df = ok_df[ok_df["condition"].astype(str) == str(condition)].copy()
    if ok_df.empty:
        raise ValueError("no matching successful matrix row found")
    return dict(ok_df.iloc[0].to_dict())


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump raw chat completion bodies for judge prompts.")
    parser.add_argument(
        "--input-csv",
        default=str(PROJECT_ROOT / "outputs" / "experiments" / "paper_matrix_current_structfix_stylefix_v2.csv"),
    )
    parser.add_argument("--record-id", default=None)
    parser.add_argument("--condition", default=None)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--model-name", default="gpt-oss:120b-cloud")
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "experiments" / "judge_debug"),
    )
    args = parser.parse_args()

    api_key = None
    if args.api_key_env:
        api_key = os.environ.get(str(args.api_key_env), "").strip()
        if not api_key:
            raise ValueError(f"environment variable '{args.api_key_env}' is not set or empty")

    ensure_model_server_ready(args.base_url, args.timeout_sec, api_key=api_key)
    row = load_target_row(Path(args.input_csv), args.record_id, args.condition)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    requests = [
        {
            "name": "judge_json",
            "prompt": build_judge_prompt(row),
            "system_prompt": "Return JSON only.",
            "temperature": 0.0,
            "max_tokens": 300,
        },
        {
            "name": "control_json",
            "prompt": "Return exactly this JSON object and nothing else:\n{\"scores\":{\"content_fit\":4,\"emotional_appropriateness\":4,\"style_match\":4,\"naturalness\":4,\"overall_quality\":4}}",
            "system_prompt": "Return JSON only.",
            "temperature": 0.0,
            "max_tokens": 64,
        },
    ]

    summary_rows: list[dict[str, object]] = []
    for spec in requests:
        result = send_raw_chat_completion(
            base_url=args.base_url,
            model_name=args.model_name,
            prompt=str(spec["prompt"]),
            system_prompt=str(spec["system_prompt"]),
            temperature=float(spec["temperature"]),
            max_tokens=int(spec["max_tokens"]),
            timeout_sec=args.timeout_sec,
            api_key=api_key,
        )
        summary = summarize_response(result)
        summary["name"] = spec["name"]
        summary["prompt_chars"] = len(str(spec["prompt"]))
        summary["system_prompt"] = str(spec["system_prompt"])
        summary_rows.append(summary)

        request_path = output_dir / f"{spec['name']}.request.json"
        response_path = output_dir / f"{spec['name']}.response.json"
        request_path.write_text(
            json.dumps(
                {
                    "name": spec["name"],
                    "prompt": spec["prompt"],
                    "system_prompt": spec["system_prompt"],
                    "temperature": spec["temperature"],
                    "max_tokens": spec["max_tokens"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        response_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    summary_path = output_dir / "summary.json"
    payload = {
        "record_id": row.get("record_id"),
        "condition": row.get("condition"),
        "output_dir": str(output_dir),
        "summaries": summary_rows,
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
