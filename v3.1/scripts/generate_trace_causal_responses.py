#!/usr/bin/env python3
"""Generate responses for trace causal proof rows."""

from __future__ import annotations

import argparse
import csv
import json
import time
import urllib.request
from pathlib import Path


OUTPUT_COLUMNS = [
    "record_id",
    "causal_condition",
    "manipulation_type",
    "manipulated_axis",
    "original_value",
    "new_value",
    "status",
    "error_message",
    "text",
    "llm_response",
    "response_length",
    "prompt",
    "episode_label",
    "episode_family",
    "appraisal_family",
    "valence",
    "arousal",
    "target",
    "control_state",
    "social_orientation",
    "preserve",
    "avoid",
    "action_tendency",
    "action_tendency_class",
    "expected_effect",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def compact(value: str | None, limit: int = 260) -> str:
    text = " ".join(str(value or "").strip().split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "..."


def build_prompt(row: dict[str, str]) -> str:
    return "\n".join(
        [
            "[ROLE]",
            "You generate a short Korean emotional response. Use the trace as the user's internal emotion state, not as labels to mention.",
            "",
            "[USER_INPUT]",
            str(row.get("text", "")).strip(),
            "",
            "[TRACE_AS_EMOTION_STATE]",
            f"episode_family={row.get('episode_family', '')}",
            f"appraisal_family={row.get('appraisal_family', '')}",
            f"valence={row.get('valence', '')}, arousal={row.get('arousal', '')}",
            f"target={row.get('target', '')}, control_state={row.get('control_state', '')}, social_orientation={row.get('social_orientation', '')}",
            f"preserve={compact(row.get('preserve', ''), 180)}",
            f"avoid={compact(row.get('avoid', ''), 180)}",
            f"action_tendency_class={row.get('action_tendency_class', '')}",
            f"action_tendency={compact(row.get('action_tendency', ''), 220)}",
            "",
            "[CAUSAL_MANIPULATION]",
            f"condition={row.get('causal_condition', '')}",
            f"type={row.get('manipulation_type', '')}",
            f"axis={row.get('manipulated_axis', '')}",
            f"original_value={row.get('original_value', '')}",
            f"new_value={row.get('new_value', '')}",
            f"expected_effect={row.get('expected_effect', '')}",
            "",
            "[RESPONSE_RULES]",
            "- Do not expose trace field names, categories, or analysis terms.",
            "- The first sentence should directly touch the user's emotional cause.",
            "- Preserve rough negative affect when the trace calls for it; do not over-soften.",
            "- Keep the response 2 to 4 Korean sentences.",
            "- Avoid generic comfort if it conflicts with the trace.",
        ]
    )


def call_chat(
    *,
    base_url: str,
    model_name: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout_sec: int,
) -> str:
    body = {
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content": "Return only a natural Korean response. Do not include JSON, headings, or analysis.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return str(payload["choices"][0]["message"].get("content", "")).strip()


def run(args: argparse.Namespace) -> dict[str, object]:
    rows = read_csv(args.input)
    if args.base_records and args.base_records > 0:
        selected_ids: list[str] = []
        for row in rows:
            record_id = str(row.get("record_id", ""))
            if record_id not in selected_ids:
                selected_ids.append(record_id)
            if len(selected_ids) >= args.base_records:
                break
        rows = [row for row in rows if str(row.get("record_id", "")) in set(selected_ids)]
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    output_rows: list[dict[str, str]] = []
    start = time.perf_counter()
    for idx, row in enumerate(rows, start=1):
        out = {column: str(row.get(column, "")) for column in OUTPUT_COLUMNS}
        out["status"] = "error"
        out["error_message"] = ""
        out["llm_response"] = ""
        out["response_length"] = "0"
        prompt = build_prompt(row)
        out["prompt"] = prompt
        try:
            response = ""
            last_error: Exception | None = None
            for attempt in range(args.max_retries + 1):
                try:
                    response = call_chat(
                        base_url=args.base_url,
                        model_name=args.model_name,
                        prompt=prompt,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        timeout_sec=args.timeout_sec,
                    )
                    if response:
                        break
                    raise ValueError("empty response")
                except Exception as exc:
                    last_error = exc
                    if attempt >= args.max_retries:
                        raise
                    time.sleep(0.6 * (attempt + 1))
            if not response:
                raise ValueError(str(last_error or "empty response"))
            out["status"] = "ok"
            out["llm_response"] = response
            out["response_length"] = str(len(response))
        except Exception as exc:
            out["error_message"] = str(exc)
        output_rows.append(out)
        if args.progress_every > 0 and idx % args.progress_every == 0:
            print(f"generate-trace-causal: {idx}/{len(rows)} elapsed={time.perf_counter() - start:.1f}s")

    write_csv(args.output, output_rows)
    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "rows": len(output_rows),
        "ok_rows": sum(1 for row in output_rows if row["status"] == "ok"),
        "error_rows": sum(1 for row in output_rows if row["status"] != "ok"),
        "base_record_count": len({row["record_id"] for row in output_rows}),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/trace_causal_probe_set.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/trace_causal_responses_dry3.csv"))
    parser.add_argument("--summary", type=Path, default=Path("outputs/trace_causal_responses_dry3_summary.json"))
    parser.add_argument("--base-records", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--model-name", default="gpt-oss:20b")
    parser.add_argument("--temperature", type=float, default=0.45)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--timeout-sec", type=int, default=240)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
