#!/usr/bin/env python3
"""Normalize free-form trace axes into canonical emotion-state categories."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


ACTION_RULES: list[tuple[str, list[str]]] = [
    (
        "confront",
        [
            "confront",
            "argue",
            "protest",
            "challenge",
            "attack",
            "blame",
            "따지",
            "항의",
            "반박",
            "공격",
            "비난",
            "문제제기",
            "맞서",
            "책임",
            "해명",
        ],
    ),
    (
        "defend",
        [
            "defend",
            "protect",
            "boundary",
            "guard",
            "refuse",
            "방어",
            "보호",
            "경계",
            "거절",
            "차단",
            "선을",
            "거리",
            "자기보호",
        ],
    ),
    (
        "withdraw",
        [
            "withdraw",
            "avoid",
            "retreat",
            "leave",
            "stop",
            "distance",
            "회피",
            "물러",
            "철회",
            "중단",
            "피하",
            "떠나",
            "숨",
            "끊",
        ],
    ),
    (
        "repair",
        [
            "repair",
            "apolog",
            "restore",
            "reconcile",
            "thank",
            "사과",
            "미안",
            "복구",
            "화해",
            "회복",
            "감사",
            "보답",
        ],
    ),
    (
        "seek_support",
        [
            "support",
            "help",
            "consult",
            "share",
            "도움",
            "지지",
            "상담",
            "공유",
            "의지",
            "요청",
            "말하",
        ],
    ),
    (
        "plan",
        [
            "plan",
            "prepare",
            "check",
            "organize",
            "verify",
            "계획",
            "준비",
            "확인",
            "점검",
            "자료",
            "정리",
            "대비",
            "대응",
            "검토",
        ],
    ),
    (
        "inhibit",
        [
            "inhibit",
            "suppress",
            "hold",
            "wait",
            "pause",
            "참",
            "억제",
            "보류",
            "멈추",
            "조절",
            "견디",
            "버티",
        ],
    ),
    (
        "approach",
        [
            "approach",
            "contact",
            "meet",
            "ask",
            "다가",
            "접근",
            "연락",
            "만나",
            "묻",
            "다가가",
        ],
    ),
]


EPISODE_RULES: list[tuple[str, list[str]]] = [
    ("other_blame_boundary", ["공세", "경계", "분노", "모욕", "침해", "배신", "권위", "반박", "방어"]),
    ("self_blame_guilt", ["죄책", "미안", "사과", "자기비난", "후회", "부끄", "수치"]),
    ("threat_anxiety", ["불안", "위협", "긴장", "걱정", "예상", "위기"]),
    ("loss_sadness", ["상실", "실패", "무력", "고립", "외로", "슬픔"]),
    ("repair_gratitude", ["감사", "보답", "회복", "관계복구", "좋은"]),
    ("planning_control", ["계획", "준비", "확인", "대비", "통제", "관리"]),
]


def normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    return " ".join(str(value).strip().lower().split())


def classify_by_rules(text: str, rules: list[tuple[str, list[str]]], fallback: str) -> tuple[str, dict[str, int]]:
    text = normalize_text(text)
    scores: dict[str, int] = {}
    for label, keywords in rules:
        score = 0
        for keyword in keywords:
            if keyword.lower() in text:
                score += 1
        scores[label] = score
    best_label, best_score = max(scores.items(), key=lambda item: item[1])
    if best_score <= 0:
        return fallback, scores
    return best_label, scores


def derive_appraisal_family(row: dict[str, str]) -> str:
    target = normalize_text(row.get("target"))
    control = normalize_text(row.get("control_state"))
    social = normalize_text(row.get("social_orientation"))
    valence = normalize_text(row.get("valence"))

    if target == "other" and social in {"defend", "mixed"}:
        return "other_directed_defense"
    if target == "self":
        return "self_directed_evaluation"
    if control == "low" and valence == "negative":
        return "low_control_distress"
    if social == "approach":
        return "approach_or_repair"
    if social == "withdraw":
        return "withdrawal_or_protection"
    if target == "situation":
        return "situation_focused_coping"
    if target == "mixed":
        return "mixed_appraisal"
    return "unspecified_appraisal"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run(input_path: Path, output_path: Path, summary_path: Path) -> dict[str, object]:
    rows = read_csv(input_path)
    normalized_rows: list[dict[str, str]] = []

    for row in rows:
        new_row = dict(row)
        action_class, action_scores = classify_by_rules(
            row.get("action_tendency", ""),
            ACTION_RULES,
            "other_action",
        )
        episode_family, episode_scores = classify_by_rules(
            row.get("episode_label", ""),
            EPISODE_RULES,
            "other_episode",
        )
        new_row["action_tendency_class"] = action_class
        new_row["episode_family"] = episode_family
        new_row["appraisal_family"] = derive_appraisal_family(row)
        new_row["trace_emotion_signature"] = "|".join(
            [
                normalize_text(row.get("valence")),
                normalize_text(row.get("arousal")),
                normalize_text(row.get("target")),
                normalize_text(row.get("control_state")),
                normalize_text(row.get("social_orientation")),
                action_class,
            ]
        )
        new_row["action_rule_scores_json"] = json.dumps(action_scores, ensure_ascii=False, sort_keys=True)
        new_row["episode_rule_scores_json"] = json.dumps(episode_scores, ensure_ascii=False, sort_keys=True)
        normalized_rows.append(new_row)

    fieldnames = list(rows[0].keys()) if rows else []
    for field in [
        "action_tendency_class",
        "episode_family",
        "appraisal_family",
        "trace_emotion_signature",
        "action_rule_scores_json",
        "episode_rule_scores_json",
    ]:
        if field not in fieldnames:
            fieldnames.append(field)

    write_csv(output_path, normalized_rows, fieldnames)

    summary = {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "record_count": len(normalized_rows),
        "action_tendency_class_counts": dict(Counter(row["action_tendency_class"] for row in normalized_rows).most_common()),
        "episode_family_counts": dict(Counter(row["episode_family"] for row in normalized_rows).most_common()),
        "appraisal_family_counts": dict(Counter(row["appraisal_family"] for row in normalized_rows).most_common()),
        "trace_emotion_signature_unique": len({row["trace_emotion_signature"] for row in normalized_rows}),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("../v4/outputs/experiments/superiority_targeted_v1/targeted_records.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/targeted_records_trace_normalized.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("outputs/trace_axis_normalization_summary.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run(args.input, args.output, args.summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

