#!/usr/bin/env python3
"""Blind axis-only judge for trace-causal response pairs.

This judge is deliberately narrower than the earlier pairwise judge. It asks
whether a response expresses a target appraisal/action axis, and explicitly
forbids judging helpfulness, warmth, fluency, or general answer quality.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


AXIS_KO = {
    "target": "감정이 향하는 대상",
    "social_orientation": "사회적 방향",
    "control_state": "통제감 상태",
    "action_tendency_class": "행동 경향",
}

AXIS_GUIDE = {
    "target": "감정이 자기 자신, 타인, 상황, 관계 중 어디를 향하는지 본다.",
    "social_orientation": "도움 요청, 접근, 방어, 철수, 거리두기 같은 대인 방향을 본다.",
    "control_state": "무력감, 낮은 통제감, 계획 가능성, 주도감 같은 단서를 본다.",
    "action_tendency_class": "회피, 도움 요청, 문제 해결, 방어, 접근 같은 행동 준비 방향을 본다.",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compact(value: object, limit: int) -> str:
    text = " ".join(str(value or "").strip().split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "..."


def stable_swap(pair_id: str) -> bool:
    digest = hashlib.sha256(pair_id.encode("utf-8")).hexdigest()
    return int(digest[:2], 16) % 2 == 1


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
    if winner in {"EQUAL", "SAME", "DRAW", "NONE"}:
        return "TIE"
    raise ValueError(f"invalid winner: {value}")


def normalize_payload(raw: str) -> dict[str, str]:
    payload = extract_json(raw)
    return {
        "winner": normalize_winner(payload.get("winner")),
        "axis_value_a": compact(payload.get("axis_value_a", ""), 80),
        "axis_value_b": compact(payload.get("axis_value_b", ""), 80),
        "evidence_a": compact(payload.get("evidence_a", ""), 120),
        "evidence_b": compact(payload.get("evidence_b", ""), 120),
        "rationale": compact(payload.get("rationale", ""), 180),
    }


def rows_by_record(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "ok" and row.get("llm_response"):
            grouped[row.get("record_id", "")].append(row)
    return grouped


def make_pair(
    *,
    pair_id: str,
    record_id: str,
    question_type: str,
    axis: str,
    target_value: str,
    expected_condition: str,
    text: str,
    response_left_condition: str,
    response_left: str,
    response_right_condition: str,
    response_right: str,
) -> dict[str, str]:
    response_a_condition = response_left_condition
    response_a = response_left
    response_b_condition = response_right_condition
    response_b = response_right
    if stable_swap(pair_id):
        response_a_condition, response_b_condition = response_b_condition, response_a_condition
        response_a, response_b = response_b, response_a
    if question_type == "null_same_response":
        expected_winner = "TIE"
    elif response_a_condition == expected_condition:
        expected_winner = "A"
    elif response_b_condition == expected_condition:
        expected_winner = "B"
    else:
        raise ValueError(f"expected condition {expected_condition!r} is not in pair {pair_id!r}")
    return {
        "pair_id": pair_id,
        "record_id": record_id,
        "question_type": question_type,
        "manipulated_axis": axis,
        "axis_ko": AXIS_KO.get(axis, axis),
        "axis_guide": AXIS_GUIDE.get(axis, ""),
        "target_axis_value": target_value,
        "expected_condition": expected_condition,
        "expected_winner": expected_winner,
        "text": text,
        "response_a_condition_hidden": response_a_condition,
        "response_b_condition_hidden": response_b_condition,
        "response_a": response_a,
        "response_b": response_b,
    }


def build_pairs(rows: list[dict[str, str]], include_null: bool = True) -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []
    for record_id, group in sorted(rows_by_record(rows).items()):
        full_rows = [row for row in group if row.get("causal_condition") == "trace_full"]
        if not full_rows:
            continue
        full = full_rows[0]
        text = full.get("text", "")
        by_condition = {row.get("causal_condition", ""): row for row in group}
        for row in group:
            condition = row.get("causal_condition", "")
            if condition == "trace_full":
                continue
            axis = row.get("manipulated_axis", "")
            if axis not in AXIS_KO:
                continue
            manipulation = row.get("manipulation_type", "")
            if manipulation == "ablation":
                target_value = row.get("original_value", "")
                pairs.append(
                    make_pair(
                        pair_id=f"{record_id}::{condition}::axis_only_original",
                        record_id=record_id,
                        question_type="ablation_axis_original",
                        axis=axis,
                        target_value=target_value,
                        expected_condition="trace_full",
                        text=text,
                        response_left_condition="trace_full",
                        response_left=full.get("llm_response", ""),
                        response_right_condition=condition,
                        response_right=row.get("llm_response", ""),
                    )
                )
            elif manipulation == "perturbation":
                target_value = row.get("new_value", "")
                pairs.append(
                    make_pair(
                        pair_id=f"{record_id}::{condition}::axis_only_perturbed",
                        record_id=record_id,
                        question_type="perturbation_axis_shift",
                        axis=axis,
                        target_value=target_value,
                        expected_condition=condition,
                        text=text,
                        response_left_condition="trace_full",
                        response_left=full.get("llm_response", ""),
                        response_right_condition=condition,
                        response_right=row.get("llm_response", ""),
                    )
                )
        if include_null:
            for axis in AXIS_KO:
                source = by_condition.get("trace_full", full)
                pairs.append(
                    make_pair(
                        pair_id=f"{record_id}::trace_full::{axis}::null_same",
                        record_id=record_id,
                        question_type="null_same_response",
                        axis=axis,
                        target_value=source.get(axis, ""),
                        expected_condition="tie",
                        text=text,
                        response_left_condition="trace_full_copy_1",
                        response_left=source.get("llm_response", ""),
                        response_right_condition="trace_full_copy_2",
                        response_right=source.get("llm_response", ""),
                    )
                )
    return pairs


def build_prompt(pair: dict[str, str]) -> str:
    return "\n".join(
        [
            "You are an axis-only evaluator for a trace-as-emotion experiment.",
            "Do NOT judge helpfulness, warmth, politeness, fluency, empathy, or overall answer quality.",
            "Ignore which answer sounds better. Only judge the specified emotion/appraisal/action axis.",
            "If both responses express the target axis equally, choose tie.",
            "Return JSON only with keys: winner, axis_value_a, axis_value_b, evidence_a, evidence_b, rationale.",
            "Do not put quotation marks inside JSON string values. Paraphrase evidence instead of directly quoting.",
            "Keep evidence_a, evidence_b, and rationale short.",
            "",
            "[AXIS_TO_JUDGE]",
            f"axis={pair['axis_ko']} ({pair['manipulated_axis']})",
            f"axis_guide={pair['axis_guide']}",
            f"target_axis_value={pair['target_axis_value']}",
            "",
            "[TASK]",
            "Which response more strongly expresses the target_axis_value on this axis only?",
            "Use short evidence phrases from the responses. If evidence is absent or equal, choose tie.",
            "",
            "[USER_INPUT_CONTEXT]",
            compact(pair.get("text", ""), 360),
            "",
            "[RESPONSE_A]",
            compact(pair.get("response_a", ""), 520),
            "",
            "[RESPONSE_B]",
            compact(pair.get("response_b", ""), 520),
        ]
    )


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
            raw = anthropic_chat(args, prompt)
            return normalize_payload(raw), raw
        except Exception as exc:
            last_error = exc
            if attempt < args.max_retries:
                time.sleep(0.8 * (attempt + 1))
    assert last_error is not None
    raise last_error


def judge(args: argparse.Namespace) -> list[dict[str, str]]:
    pairs = build_pairs(read_csv(args.input), include_null=not args.no_null)
    if args.limit and args.limit > 0:
        pairs = pairs[: args.limit]
    if args.export_pairs_only:
        return [{**pair, **empty_judge("not_run")} for pair in pairs]
    output_rows: list[dict[str, str]] = []
    for idx, pair in enumerate(pairs, start=1):
        out = {**pair, **empty_judge("error")}
        try:
            normalized, raw = call_judge(args, build_prompt(pair))
            winner = normalized["winner"]
            out.update(
                {
                    "judge_status": "ok",
                    "judge_error": "",
                    "judge_winner": winner,
                    "judge_axis_value_a": normalized["axis_value_a"],
                    "judge_axis_value_b": normalized["axis_value_b"],
                    "judge_evidence_a": normalized["evidence_a"],
                    "judge_evidence_b": normalized["evidence_b"],
                    "judge_rationale": normalized["rationale"],
                    "judge_raw": raw,
                    "success": str(winner == pair["expected_winner"]).lower(),
                }
            )
        except Exception as exc:
            out["judge_error"] = str(exc)
        output_rows.append(out)
        if args.progress_every > 0 and idx % args.progress_every == 0:
            print(f"judge-axis-blind: {idx}/{len(pairs)}")
    return output_rows


def empty_judge(status: str) -> dict[str, str]:
    return {
        "judge_status": status,
        "judge_error": "",
        "judge_winner": "",
        "judge_axis_value_a": "",
        "judge_axis_value_b": "",
        "judge_evidence_a": "",
        "judge_evidence_b": "",
        "judge_rationale": "",
        "judge_raw": "",
        "success": "",
    }


def summarize(rows: list[dict[str, str]]) -> dict[str, Any]:
    ok = [row for row in rows if row.get("judge_status") == "ok"]
    by_type: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_axis: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in ok:
        by_type[row.get("question_type", "")].append(row)
        by_axis[row.get("manipulated_axis", "")].append(row)

    def block(items: list[dict[str, str]]) -> dict[str, Any]:
        return {
            "n": len(items),
            "winner_counts": dict(Counter(row.get("judge_winner", "") for row in items).most_common()),
            "success_count": sum(1 for row in items if row.get("success") == "true"),
            "success_rate": round(sum(1 for row in items if row.get("success") == "true") / len(items), 6)
            if items
            else 0.0,
            "tie_rate": round(sum(1 for row in items if row.get("judge_winner") == "TIE") / len(items), 6)
            if items
            else 0.0,
        }

    return {
        "rows": len(rows),
        "ok_rows": len(ok),
        "error_rows": len(rows) - len(ok),
        "overall": block(ok),
        "by_question_type": {key: block(value) for key, value in sorted(by_type.items())},
        "by_axis": {key: block(value) for key, value in sorted(by_axis.items())},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/trace_causal_responses_dry3.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/trace_axis_blind_judgments.csv"))
    parser.add_argument("--summary", type=Path, default=Path("outputs/trace_axis_blind_judgments_summary.json"))
    parser.add_argument("--model", default="claude-haiku-4-5-20251001")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=500)
    parser.add_argument("--timeout-sec", type=int, default=90)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--anthropic-api-key-env", default="ANTHROPIC_API_KEY")
    parser.add_argument("--export-pairs-only", action="store_true")
    parser.add_argument("--no-null", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = judge(args)
    write_csv(args.output, rows)
    summary = summarize(rows)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
