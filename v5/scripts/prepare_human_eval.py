from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import pandas as pd


def parse_conditions(raw: str | None, discovered: list[str]) -> list[str]:
    if not raw:
        return discovered
    tokens = [token.strip() for token in raw.replace(";", ",").split(",")]
    conditions = [token for token in tokens if token]
    if not conditions:
        raise ValueError("at least one condition is required")
    return conditions


def build_instruction_markdown(candidate_labels: list[str]) -> str:
    label_text = ", ".join(candidate_labels)
    return "\n".join(
        [
            "# Human Evaluation Instructions",
            "",
            "각 행은 하나의 사용자 입력과 여러 후보 응답으로 구성된다.",
            f"후보 열은 {label_text} 순서로 제시되며, 어떤 모델 조건인지 숨겨져 있다.",
            "",
            "권장 평가 항목:",
            "",
            "- content_fit: 입력 내용에 직접적으로 맞는가",
            "- emotional_appropriateness: 입력 감정 상태에 맞는가",
            "- style_match: 더 설득력 있는 말투를 보이는가",
            "- naturalness: 한국어 응답이 자연스러운가",
            "",
            "실제 평가 시에는 각 항목별 점수 또는 최고 후보를 별도 시트에 기록한다.",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare blinded CSVs for human evaluation.")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--answer-key-json", required=True)
    parser.add_argument("--instructions-md", default=None)
    parser.add_argument("--conditions", default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--record-id-column", default="record_id")
    parser.add_argument("--text-column", default="text")
    args = parser.parse_args()

    input_csv = Path(args.input_csv)
    output_csv = Path(args.output_csv)
    answer_key_json = Path(args.answer_key_json)
    instructions_md = Path(args.instructions_md) if args.instructions_md else None

    df = pd.read_csv(input_csv)
    ok_df = df[df["status"].fillna("") == "ok"].copy()
    if ok_df.empty:
        raise ValueError("no successful rows found in input csv")

    discovered_conditions = sorted(ok_df["condition"].dropna().astype(str).unique().tolist())
    conditions = parse_conditions(args.conditions, discovered_conditions)
    ok_df = ok_df[ok_df["condition"].astype(str).isin(conditions)].copy()

    rng = random.Random(args.seed)
    grouped_rows: list[dict[str, object]] = []
    answer_key: dict[str, object] = {"seed": args.seed, "conditions": conditions, "rows": []}

    candidate_labels = [f"candidate_{chr(ord('a') + idx)}" for idx in range(len(conditions))]
    for group_idx, (record_id, group) in enumerate(ok_df.groupby(args.record_id_column, dropna=False), start=1):
        seen_conditions = set(group["condition"].astype(str).tolist())
        if any(condition not in seen_conditions for condition in conditions):
            continue

        text = str(group.iloc[0][args.text_column])
        row_candidates = []
        for condition in conditions:
            match = group[group["condition"].astype(str) == condition].iloc[0]
            row_candidates.append(
                {
                    "condition": condition,
                    "response": str(match["llm_response"]),
                    "condition_group": str(match.get("condition_group", "")),
                }
            )
        rng.shuffle(row_candidates)

        eval_id = f"eval_{group_idx:05d}"
        output_row = {
            "eval_id": eval_id,
            "record_id": str(record_id),
            "text": text,
        }
        row_key = {"eval_id": eval_id, "record_id": str(record_id), "candidates": []}
        for label, candidate in zip(candidate_labels, row_candidates, strict=True):
            output_row[label] = candidate["response"]
            row_key["candidates"].append(
                {
                    "label": label,
                    "condition": candidate["condition"],
                    "condition_group": candidate["condition_group"],
                }
            )
        output_row.update(
            {
                "winner": "",
                "confidence": "",
                "reason": "",
                "appraisal_fidelity": "",
                "raw_affect_preservation": "",
                "anti_softening": "",
                "action_tendency_fit": "",
                "emotional_specificity": "",
                "naturalness": "",
            }
        )

        grouped_rows.append(output_row)
        answer_key["rows"].append(row_key)

    if args.sample_size is not None and args.sample_size > 0 and len(grouped_rows) > args.sample_size:
        selected_indices = list(range(len(grouped_rows)))
        rng.shuffle(selected_indices)
        keep_indices = set(selected_indices[: args.sample_size])
        grouped_rows = [row for idx, row in enumerate(grouped_rows) if idx in keep_indices]
        keep_eval_ids = {row["eval_id"] for row in grouped_rows}
        answer_key["rows"] = [row for row in answer_key["rows"] if row["eval_id"] in keep_eval_ids]

    if not grouped_rows:
        raise ValueError("no complete record groups found for the requested conditions")

    output_df = pd.DataFrame(grouped_rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_csv, index=False, encoding="utf-8-sig")

    answer_key_json.parent.mkdir(parents=True, exist_ok=True)
    answer_key_json.write_text(json.dumps(answer_key, ensure_ascii=False, indent=2), encoding="utf-8")

    if instructions_md is not None:
        instructions_md.parent.mkdir(parents=True, exist_ok=True)
        instructions_md.write_text(build_instruction_markdown(candidate_labels), encoding="utf-8")

    print(
        json.dumps(
            {
                "rows": int(len(output_df)),
                "conditions": conditions,
                "output_csv": str(output_csv),
                "answer_key_json": str(answer_key_json),
                "instructions_md": str(instructions_md) if instructions_md is not None else "",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
