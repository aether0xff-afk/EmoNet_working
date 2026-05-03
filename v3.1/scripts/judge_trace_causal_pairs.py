#!/usr/bin/env python3
"""Judge trace-causal effects with cheap pairwise A/B decisions.

This script replaces the earlier multi-metric causal judge. It asks one small
question per pair and expects a tiny JSON object, which keeps cost and format
failure low.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


AXIS_LABEL = {
    "target": "target/blame direction",
    "social_orientation": "social orientation",
    "control_state": "control or agency state",
    "action_tendency_class": "action tendency",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compact(value: object, limit: int) -> str:
    text = " ".join(str(value or "").strip().split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "..."


def extract_json(raw: str) -> dict[str, Any]:
    text = str(raw or "").strip()
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        payload = json.loads(match.group(0))
    if not isinstance(payload, dict):
        raise ValueError("judge output must be a JSON object")
    return payload


def normalize_winner(value: object) -> str:
    winner = str(value or "").strip().upper()
    if winner in {"A", "B", "TIE"}:
        return winner
    if winner in {"EQUAL", "SAME", "DRAW"}:
        return "TIE"
    raise ValueError(f"invalid winner: {value}")


def normalize_confidence(value: object) -> int:
    try:
        confidence = int(round(float(value)))
    except Exception:
        confidence = 1
    return max(1, min(3, confidence))


def normalize_payload(raw: str) -> dict[str, str]:
    payload = extract_json(raw)
    winner = normalize_winner(payload.get("winner"))
    confidence = normalize_confidence(payload.get("confidence"))
    rationale = compact(payload.get("rationale", ""), 180)
    return {
        "winner": winner,
        "confidence": str(confidence),
        "rationale": rationale,
    }


def build_pairs(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_record: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "ok" and row.get("llm_response"):
            by_record[row.get("record_id", "")].append(row)

    pairs: list[dict[str, str]] = []
    for record_id, group in sorted(by_record.items()):
        full_rows = [row for row in group if row.get("causal_condition") == "trace_full"]
        if not full_rows:
            continue
        full = full_rows[0]
        for row in group:
            if row is full:
                continue
            axis = row.get("manipulated_axis", "")
            if axis not in AXIS_LABEL:
                continue
            manipulation_type = row.get("manipulation_type", "")
            if manipulation_type == "ablation":
                question_type = "ablation_preservation"
                expected_winner = "A"
                question = (
                    f"Which response better preserves the original {AXIS_LABEL[axis]} "
                    f"({row.get('original_value', '')}) while remaining natural?"
                )
            elif manipulation_type == "perturbation":
                question_type = "perturbation_shift"
                expected_winner = "B"
                question = (
                    f"Which response better shifts toward the perturbed {AXIS_LABEL[axis]} "
                    f"({row.get('new_value', '')}) while remaining natural?"
                )
            else:
                continue
            pair = {
                "pair_id": f"{record_id}::{row.get('causal_condition', '')}",
                "record_id": record_id,
                "question_type": question_type,
                "manipulation_type": manipulation_type,
                "manipulated_axis": axis,
                "original_value": row.get("original_value", ""),
                "new_value": row.get("new_value", ""),
                "expected_winner": expected_winner,
                "question": question,
                "text": row.get("text", ""),
                "episode_label": row.get("episode_label", ""),
                "appraisal_family": row.get("appraisal_family", ""),
                "preserve": row.get("preserve", ""),
                "avoid": row.get("avoid", ""),
                "action_tendency": row.get("action_tendency", ""),
                "response_a_condition": "trace_full",
                "response_a": full.get("llm_response", ""),
                "response_b_condition": row.get("causal_condition", ""),
                "response_b": row.get("llm_response", ""),
            }
            pairs.append(pair)
    return pairs


def build_prompt(pair: dict[str, str]) -> str:
    return "\n".join(
        [
            "You are judging a trace-as-emotion causal experiment.",
            "Pick the better Korean response for the exact question.",
            "Return JSON only: {\"winner\":\"A|B|tie\",\"confidence\":1-3,\"rationale\":\"short\"}",
            "",
            "[QUESTION]",
            pair["question"],
            "",
            "[USER_INPUT]",
            compact(pair.get("text", ""), 360),
            "",
            "[TRACE_CONTEXT]",
            f"episode_label={compact(pair.get('episode_label', ''), 120)}",
            f"appraisal_family={pair.get('appraisal_family', '')}",
            f"preserve={compact(pair.get('preserve', ''), 180)}",
            f"avoid={compact(pair.get('avoid', ''), 160)}",
            f"action_tendency={compact(pair.get('action_tendency', ''), 220)}",
            f"axis={pair.get('manipulated_axis', '')}",
            f"original_value={pair.get('original_value', '')}",
            f"new_value={pair.get('new_value', '')}",
            "",
            "[RESPONSE_A]",
            compact(pair.get("response_a", ""), 420),
            "",
            "[RESPONSE_B]",
            compact(pair.get("response_b", ""), 420),
        ]
    )


def estimate_tokens(text: str) -> int:
    # Conservative mixed Korean/English approximation for budget guarding.
    return max(1, int(len(text) / 2.2) + 1)


def estimate_cost_usd(pairs: list[dict[str, str]], max_output_tokens: int) -> dict[str, Any]:
    input_tokens = sum(estimate_tokens(build_prompt(pair)) for pair in pairs)
    output_tokens = len(pairs) * max_output_tokens
    return {
        "pairs": len(pairs),
        "estimated_input_tokens": input_tokens,
        "max_output_tokens": output_tokens,
        "openai_gpt54_mini_cost_usd": round(input_tokens / 1_000_000 * 0.75 + output_tokens / 1_000_000 * 4.50, 6),
        "note": "Uses GPT-5.4 mini price assumptions: $0.75/M input, $4.50/M output.",
    }


def openai_chat(args: argparse.Namespace, prompt: str) -> str:
    api_key = os.environ.get(args.openai_api_key_env, "")
    if not api_key:
        raise RuntimeError(f"missing environment variable: {args.openai_api_key_env}")
    body = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": "Return one compact JSON object only."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_completion_tokens": args.max_output_tokens,
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=args.timeout_sec) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return str(payload["choices"][0]["message"].get("content", "")).strip()


def anthropic_chat(args: argparse.Namespace, prompt: str) -> str:
    api_key = os.environ.get(args.anthropic_api_key_env, "")
    if not api_key:
        raise RuntimeError(f"missing environment variable: {args.anthropic_api_key_env}")
    body = {
        "model": args.model,
        "max_tokens": args.max_output_tokens,
        "temperature": 0,
        "system": "Return one compact JSON object only.",
        "messages": [{"role": "user", "content": prompt}],
    }
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=args.timeout_sec) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    parts = payload.get("content", [])
    return "".join(str(part.get("text", "")) for part in parts if part.get("type") == "text").strip()


def call_judge(args: argparse.Namespace, prompt: str) -> tuple[dict[str, str], str]:
    last_error: Exception | None = None
    for attempt in range(args.max_retries + 1):
        try:
            if args.provider == "openai":
                raw = openai_chat(args, prompt)
            elif args.provider == "anthropic":
                raw = anthropic_chat(args, prompt)
            else:
                raise ValueError(f"unknown provider: {args.provider}")
            return normalize_payload(raw), raw
        except Exception as exc:
            last_error = exc
            if attempt < args.max_retries:
                time.sleep(0.8 * (attempt + 1))
    assert last_error is not None
    raise last_error


def judge_pairs(args: argparse.Namespace) -> list[dict[str, str]]:
    rows = read_csv(args.input)
    pairs = build_pairs(rows)
    if args.limit and args.limit > 0:
        pairs = pairs[: args.limit]
    if args.estimate_only:
        print(json.dumps(estimate_cost_usd(pairs, args.max_output_tokens), ensure_ascii=False, indent=2))
        return []
    if args.export_pairs_only:
        return [
            {
                **pair,
                "judge_status": "not_run",
                "judge_error": "",
                "judge_winner": "",
                "judge_confidence": "",
                "judge_rationale": "",
                "judge_raw": "",
                "success": "",
            }
            for pair in pairs
        ]

    output_rows: list[dict[str, str]] = []
    for idx, pair in enumerate(pairs, start=1):
        out = dict(pair)
        out.update(
            {
                "judge_status": "error",
                "judge_error": "",
                "judge_winner": "",
                "judge_confidence": "",
                "judge_rationale": "",
                "judge_raw": "",
                "success": "",
            }
        )
        try:
            normalized, raw = call_judge(args, build_prompt(pair))
            winner = normalized["winner"]
            out["judge_winner"] = winner
            out["judge_confidence"] = normalized["confidence"]
            out["judge_rationale"] = normalized["rationale"]
            out["judge_raw"] = raw
            out["judge_status"] = "ok"
            out["success"] = str(winner == pair["expected_winner"]).lower()
        except Exception as exc:
            out["judge_error"] = str(exc)
        output_rows.append(out)
        if args.progress_every > 0 and idx % args.progress_every == 0:
            print(f"judge-trace-pairs: {idx}/{len(pairs)}")
    return output_rows


def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    ok = [row for row in rows if row.get("judge_status") == "ok"]
    pending = [row for row in rows if row.get("judge_status") == "not_run"]
    success_rows = [row for row in ok if row.get("judge_winner") != "TIE"]
    by_type: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_axis: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in ok:
        by_type[row.get("question_type", "")].append(row)
        by_axis[row.get("manipulated_axis", "")].append(row)

    def block(items: list[dict[str, str]]) -> dict[str, Any]:
        non_tie = [row for row in items if row.get("judge_winner") != "TIE"]
        return {
            "n": len(items),
            "non_tie_n": len(non_tie),
            "winner_counts": dict(Counter(row.get("judge_winner", "") for row in items).most_common()),
            "success_count": sum(1 for row in items if row.get("success") == "true"),
            "success_rate": round(
                sum(1 for row in items if row.get("success") == "true") / len(items),
                6,
            )
            if items
            else 0.0,
            "non_tie_success_rate": round(
                sum(1 for row in non_tie if row.get("success") == "true") / len(non_tie),
                6,
            )
            if non_tie
            else 0.0,
        }

    return {
        "rows": len(rows),
        "ok_rows": len(ok),
        "pending_rows": len(pending),
        "error_rows": len(rows) - len(ok) - len(pending),
        "overall": block(ok),
        "overall_non_tie_success_rate": round(
            sum(1 for row in success_rows if row.get("success") == "true") / len(success_rows),
            6,
        )
        if success_rows
        else 0.0,
        "by_question_type": {key: block(value) for key, value in sorted(by_type.items())},
        "by_axis": {key: block(value) for key, value in sorted(by_axis.items())},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/trace_causal_responses_dry3.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/trace_causal_pair_judgments.csv"))
    parser.add_argument("--summary", type=Path, default=Path("outputs/trace_causal_pair_judgments_summary.json"))
    parser.add_argument("--provider", choices=["openai", "anthropic"], default="openai")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=90)
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--export-pairs-only", action="store_true")
    parser.add_argument("--timeout-sec", type=int, default=90)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--openai-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--anthropic-api-key-env", default="ANTHROPIC_API_KEY")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = judge_pairs(args)
    if args.estimate_only:
        return
    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = []
    write_csv(args.output, rows, fieldnames)
    summary = summarize(rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
