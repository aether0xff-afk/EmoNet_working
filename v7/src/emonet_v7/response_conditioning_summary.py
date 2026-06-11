"""Summaries for neutral trace-report response-conditioning runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .response_conditioning import (
    CLAIM_BOUNDARY,
    RESPONSE_CONDITIONS,
    ResponseConditioningRow,
)


OVERCLAIM_PATTERNS = {
    "subjective_emotion_claim": [
        "감정을 느낀",
        "감정을 느낀다",
        "감정을 가진",
        "감정을 가졌다",
        "feels emotion",
        "has emotions",
        "subjective emotion",
    ],
    "ground_truth_emotion_label_claim": [
        "정답 감정",
        "ground-truth emotion",
        "true emotion",
    ],
}


def detect_overclaim_flags(text: str) -> list[str]:
    lowered = text.lower()
    flags = []
    for flag, patterns in OVERCLAIM_PATTERNS.items():
        if any(pattern.lower() in lowered for pattern in patterns):
            flags.append(flag)
    return flags


def summarize_response_conditioning(rows: list[ResponseConditioningRow]) -> dict[str, Any]:
    grouped = _group_by_case(rows)
    condition_counts = {
        condition: sum(1 for row in rows if row.condition == condition)
        for condition in RESPONSE_CONDITIONS
    }
    neutral_changed = 0
    masked_changed = 0
    shuffled_changed = 0
    overclaim_flags: dict[str, list[str]] = {}
    for case_id, by_condition in grouped.items():
        direct = by_condition.get("direct_response")
        neutral = by_condition.get("neutral_report")
        masked = by_condition.get("masked_report")
        shuffled = by_condition.get("shuffled_report")
        if direct and neutral and direct.response != neutral.response:
            neutral_changed += 1
        if direct and masked and direct.response != masked.response:
            masked_changed += 1
        if neutral and shuffled and neutral.response != shuffled.response:
            shuffled_changed += 1
        for condition, row in by_condition.items():
            flags = detect_overclaim_flags(row.response)
            if flags:
                overclaim_flags[f"{case_id}:{condition}"] = flags

    case_count = len(grouped)
    influence = _classify_influence(
        case_count=case_count,
        neutral_changed=neutral_changed,
        masked_changed=masked_changed,
        shuffled_changed=shuffled_changed,
    )
    return {
        "claim_boundary": CLAIM_BOUNDARY,
        "case_count": case_count,
        "condition_counts": condition_counts,
        "response_delta": {
            "neutral_vs_direct_changed": neutral_changed,
            "masked_vs_direct_changed": masked_changed,
            "shuffled_vs_neutral_changed": shuffled_changed,
        },
        "overclaim_flags": overclaim_flags,
        "decision": {
            "trace_conditioned_response_influence": influence,
            "emotion_ground_truth_used": False,
            "interpretation": (
                "Report-conditioned response changes are response-surface evidence only. "
                "They do not validate subjective emotion or emotion labels."
            ),
        },
    }


def load_response_conditioning_rows(path: str | Path) -> list[ResponseConditioningRow]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            rows.append(ResponseConditioningRow(**payload))
    return rows


def write_response_conditioning_summary(
    *,
    input_jsonl: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    rows = load_response_conditioning_rows(input_jsonl)
    summary = summarize_response_conditioning(rows)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output / "decision_report.json").write_text(
        json.dumps(
            {
                "claim_boundary": summary["claim_boundary"],
                "decision": summary["decision"],
                "response_delta": summary["response_delta"],
                "overclaim_flags": summary["overclaim_flags"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def _group_by_case(rows: list[ResponseConditioningRow]) -> dict[str, dict[str, ResponseConditioningRow]]:
    grouped: dict[str, dict[str, ResponseConditioningRow]] = {}
    for row in rows:
        grouped.setdefault(row.case_id, {})[row.condition] = row
    return grouped


def _classify_influence(
    *,
    case_count: int,
    neutral_changed: int,
    masked_changed: int,
    shuffled_changed: int,
) -> str:
    if case_count == 0 or neutral_changed == 0:
        return "not_observed"
    if neutral_changed == case_count and masked_changed == 0 and shuffled_changed > 0:
        return "observed"
    return "mixed"
