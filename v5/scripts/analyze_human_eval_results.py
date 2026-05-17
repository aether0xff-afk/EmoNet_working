from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


def two_sided_sign_test_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n <= 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def load_answer_map(path: Path) -> dict[str, dict[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    answer_map: dict[str, dict[str, str]] = {}
    for row in payload["rows"]:
        answer_map[str(row["eval_id"])] = {
            str(candidate["label"]): str(candidate["condition"])
            for candidate in row["candidates"]
        }
    return answer_map


def normalize_winner(raw: object) -> str:
    value = str(raw).strip().lower()
    aliases = {
        "a": "candidate_a",
        "b": "candidate_b",
        "candidate a": "candidate_a",
        "candidate b": "candidate_b",
        "candidate_a": "candidate_a",
        "candidate_b": "candidate_b",
        "tie": "tie",
        "draw": "tie",
        "same": "tie",
        "": "",
        "nan": "",
    }
    return aliases.get(value, value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze blinded human A/B evaluation results.")
    parser.add_argument("--results-csv", required=True, help="Filled human-eval CSV with a winner column.")
    parser.add_argument("--answer-key-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--target-condition", default="episode_trace_v3")
    parser.add_argument("--winner-column", default="winner")
    args = parser.parse_args()

    results_csv = Path(args.results_csv)
    answer_key_json = Path(args.answer_key_json)
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv) if args.output_csv else None

    df = pd.read_csv(results_csv)
    if args.winner_column not in df.columns:
        raise ValueError(f"missing winner column: {args.winner_column}")

    answer_map = load_answer_map(answer_key_json)
    analyzed_rows: list[dict[str, object]] = []
    wins = ties = losses = invalid = 0

    for _, row in df.iterrows():
        eval_id = str(row["eval_id"])
        label_to_condition = answer_map.get(eval_id)
        winner = normalize_winner(row[args.winner_column])
        outcome = "invalid"
        winning_condition = ""

        if label_to_condition and winner == "tie":
            outcome = "tie"
            ties += 1
        elif label_to_condition and winner in label_to_condition:
            winning_condition = label_to_condition[winner]
            if winning_condition == args.target_condition:
                outcome = "win"
                wins += 1
            else:
                outcome = "loss"
                losses += 1
        else:
            invalid += 1

        analyzed_rows.append(
            {
                "eval_id": eval_id,
                "record_id": row.get("record_id", ""),
                "winner": winner,
                "winning_condition": winning_condition,
                "target_condition": args.target_condition,
                "outcome": outcome,
            }
        )

    total_valid = wins + ties + losses
    non_tie = wins + losses
    summary = {
        "results_csv": str(results_csv),
        "answer_key_json": str(answer_key_json),
        "target_condition": args.target_condition,
        "total_rows": int(len(df)),
        "valid_rows": int(total_valid),
        "invalid_rows": int(invalid),
        "wins": int(wins),
        "ties": int(ties),
        "losses": int(losses),
        "win_rate": float(wins / total_valid) if total_valid else None,
        "non_tie_win_rate": float(wins / non_tie) if non_tie else None,
        "sign_test_p": two_sided_sign_test_p(wins, losses),
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(analyzed_rows).to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
