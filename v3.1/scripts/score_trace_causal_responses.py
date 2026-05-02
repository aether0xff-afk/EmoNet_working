#!/usr/bin/env python3
"""Score trace causal proof responses with a causal judge."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
import urllib.request
from collections import defaultdict
from pathlib import Path


SCORE_KEYS = [
    "appraisal_fidelity",
    "target_direction_fit",
    "social_orientation_fit",
    "control_state_fit",
    "action_tendency_fit",
    "raw_affect_preservation",
    "naturalness",
    "manipulation_success",
]

MATCHING_METRIC = {
    "target": "target_direction_fit",
    "social_orientation": "social_orientation_fit",
    "control_state": "control_state_fit",
    "action_tendency_class": "action_tendency_fit",
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


def compact(value: str | None, limit: int = 320) -> str:
    text = " ".join(str(value or "").strip().split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "..."


def build_judge_prompt(row: dict[str, str]) -> str:
    return "\n".join(
        [
            "[ROLE]",
            "You are a causal judge for a trace-as-emotion experiment. Score whether the response follows the manipulated trace state.",
            "",
            "[USER_INPUT]",
            compact(row.get("text", ""), 360),
            "",
            "[TRACE_STATE_TO_EVALUATE]",
            f"condition={row.get('causal_condition', '')}",
            f"manipulation_type={row.get('manipulation_type', '')}",
            f"manipulated_axis={row.get('manipulated_axis', '')}",
            f"original_value={row.get('original_value', '')}",
            f"new_value={row.get('new_value', '')}",
            f"episode_family={row.get('episode_family', '')}",
            f"appraisal_family={row.get('appraisal_family', '')}",
            f"valence={row.get('valence', '')}, arousal={row.get('arousal', '')}",
            f"target={row.get('target', '')}, control_state={row.get('control_state', '')}, social_orientation={row.get('social_orientation', '')}",
            f"action_tendency_class={row.get('action_tendency_class', '')}",
            f"preserve={compact(row.get('preserve', ''), 220)}",
            f"avoid={compact(row.get('avoid', ''), 220)}",
            "",
            "[MODEL_RESPONSE]",
            compact(row.get("llm_response", ""), 520),
            "",
            "[SCORING]",
            "Use integer scores from 1 to 5.",
            "appraisal_fidelity: response fits the evaluated appraisal family and emotional cause.",
            "target_direction_fit: response fits the evaluated target direction, such as self/other/situation/mixed.",
            "social_orientation_fit: response fits defend/approach/withdraw/mixed orientation.",
            "control_state_fit: response fits the evaluated agency/control state.",
            "action_tendency_fit: response fits the evaluated action tendency class.",
            "raw_affect_preservation: response preserves rough affect when needed and does not over-soften.",
            "naturalness: response is fluent Korean and not analysis-like.",
            "manipulation_success: for control, fits the full trace; for ablation, shows expected weakening; for perturbation, shifts toward new_value.",
            "",
            "[OUTPUT]",
            "Return JSON only:",
            json.dumps({"scores": {key: 1 for key in SCORE_KEYS}}, ensure_ascii=False, indent=2),
        ]
    )


def build_compact_judge_prompt(row: dict[str, str]) -> str:
    return "\n".join(
        [
            "Score this trace-causal response with eight integers from 1 to 5.",
            "Return only comma-separated numbers, no words.",
            "Order:",
            ",".join(SCORE_KEYS),
            f"user_input={compact(row.get('text', ''), 220)}",
            f"condition={row.get('causal_condition', '')}",
            f"type={row.get('manipulation_type', '')}",
            f"axis={row.get('manipulated_axis', '')}",
            f"original={row.get('original_value', '')}",
            f"new={row.get('new_value', '')}",
            f"trace=target:{row.get('target', '')}; control:{row.get('control_state', '')}; social:{row.get('social_orientation', '')}; action:{row.get('action_tendency_class', '')}; appraisal:{row.get('appraisal_family', '')}",
            f"response={compact(row.get('llm_response', ''), 320)}",
            "For manipulation_success: control means fits full trace; ablation means expected weakening appears; perturbation means response shifts toward new value.",
            "Example output: 4,4,3,4,4,4,5,3",
        ]
    )


def normalize_scores(payload: object) -> dict[str, int]:
    if isinstance(payload, dict) and "scores" in payload:
        payload = payload["scores"]
    if not isinstance(payload, dict):
        raise ValueError("scores object required")
    scores: dict[str, int] = {}
    for key in SCORE_KEYS:
        if key not in payload:
            raise ValueError(f"missing score: {key}")
        value = int(round(float(payload[key])))
        if value < 1 or value > 5:
            raise ValueError(f"score out of range: {key}={value}")
        scores[key] = value
    return scores


def extract_json(text: str) -> object:
    raw = str(text or "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def call_json(
    *,
    base_url: str,
    model_name: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout_sec: int,
) -> tuple[dict[str, int], str]:
    body = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "Return JSON only."},
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
    raw = str(payload["choices"][0]["message"].get("content", "")).strip()
    return normalize_scores(extract_json(raw)), raw


def parse_compact_scores(raw: str) -> dict[str, int]:
    numbers = [int(token) for token in re.findall(r"(?<!\d)([1-5])(?!\d)", str(raw or ""))]
    if len(numbers) < len(SCORE_KEYS):
        raise ValueError("compact response did not contain eight scores")
    return {key: value for key, value in zip(SCORE_KEYS, numbers[: len(SCORE_KEYS)], strict=True)}


def call_compact(
    *,
    base_url: str,
    model_name: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    timeout_sec: int,
) -> tuple[dict[str, int], str]:
    body = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": "You must answer with exactly eight comma-separated integers from 1 to 5. No explanation.\n\n"
                + prompt,
            },
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
    raw = str(payload["choices"][0]["message"].get("content", "")).strip()
    return parse_compact_scores(raw), raw


def score_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    rows = [row for row in read_csv(args.input) if row.get("status") == "ok"]
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    scored: list[dict[str, str]] = []
    start = time.perf_counter()
    for idx, row in enumerate(rows, start=1):
        out = dict(row)
        out["judge_status"] = "error"
        out["judge_error"] = ""
        out["judge_raw"] = ""
        out["judge_mode"] = ""
        try:
            scores: dict[str, int] | None = None
            raw = ""
            mode = "json"
            for attempt in range(args.max_retries + 1):
                try:
                    scores, raw = call_json(
                        base_url=args.base_url,
                        model_name=args.model_name,
                        prompt=build_judge_prompt(row),
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        timeout_sec=args.timeout_sec,
                    )
                    break
                except Exception:
                    if attempt >= args.max_retries:
                        scores, raw = call_compact(
                            base_url=args.base_url,
                            model_name=args.model_name,
                            prompt=build_compact_judge_prompt(row),
                            temperature=0.0,
                            max_tokens=300,
                            timeout_sec=args.timeout_sec,
                        )
                        mode = "compact"
                        break
                    time.sleep(0.8 * (attempt + 1))
            assert scores is not None
            for key, value in scores.items():
                out[key] = str(value)
            out["judge_raw"] = raw
            out["judge_mode"] = mode
            out["judge_status"] = "ok"
        except Exception as exc:
            out["judge_error"] = str(exc)
            for key in SCORE_KEYS:
                out[key] = ""
        scored.append(out)
        if args.progress_every > 0 and idx % args.progress_every == 0:
            print(f"score-trace-causal: {idx}/{len(rows)} elapsed={time.perf_counter() - start:.1f}s")
    return scored


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(scored: list[dict[str, str]]) -> dict[str, object]:
    ok = [row for row in scored if row.get("judge_status") == "ok"]
    by_condition: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in ok:
        by_condition[row.get("causal_condition", "")].append(row)

    condition_means = {}
    for condition, rows in sorted(by_condition.items()):
        condition_means[condition] = {
            key: round(mean([float(row[key]) for row in rows if row.get(key)]), 6)
            for key in SCORE_KEYS
        }
        condition_means[condition]["n"] = len(rows)

    full_by_record = {
        row["record_id"]: row
        for row in ok
        if row.get("causal_condition") == "trace_full"
    }
    ablation_deltas = []
    perturbation_scores = []
    for row in ok:
        record_id = row.get("record_id", "")
        full = full_by_record.get(record_id)
        axis = row.get("manipulated_axis", "")
        metric = MATCHING_METRIC.get(axis)
        if not full or not metric:
            continue
        if row.get("manipulation_type") == "ablation":
            ablation_deltas.append(
                {
                    "record_id": record_id,
                    "axis": axis,
                    "metric": metric,
                    "delta_full_minus_ablation": float(full[metric]) - float(row[metric]),
                }
            )
        elif row.get("manipulation_type") == "perturbation":
            perturbation_scores.append(
                {
                    "record_id": record_id,
                    "axis": axis,
                    "metric": metric,
                    "manipulation_success": float(row["manipulation_success"]),
                    "metric_score": float(row[metric]),
                }
            )

    by_axis_delta: dict[str, list[float]] = defaultdict(list)
    for item in ablation_deltas:
        by_axis_delta[item["axis"]].append(float(item["delta_full_minus_ablation"]))

    by_axis_perturb: dict[str, list[float]] = defaultdict(list)
    for item in perturbation_scores:
        by_axis_perturb[item["axis"]].append(float(item["manipulation_success"]))

    return {
        "rows": len(scored),
        "ok_rows": len(ok),
        "error_rows": len(scored) - len(ok),
        "condition_means": condition_means,
        "ablation_delta_full_minus_ablation_by_axis": {
            axis: {
                "n": len(values),
                "mean_delta": round(mean(values), 6),
                "positive_count": sum(1 for value in values if value > 0),
            }
            for axis, values in sorted(by_axis_delta.items())
        },
        "perturbation_success_by_axis": {
            axis: {
                "n": len(values),
                "mean_manipulation_success": round(mean(values), 6),
                "success_ge_4_count": sum(1 for value in values if value >= 4),
            }
            for axis, values in sorted(by_axis_perturb.items())
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/trace_causal_responses_dry3.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/trace_causal_responses_dry3_scored.csv"))
    parser.add_argument("--summary", type=Path, default=Path("outputs/trace_causal_responses_dry3_scored_summary.json"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--model-name", default="gpt-oss:20b")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=700)
    parser.add_argument("--timeout-sec", type=int, default=240)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scored = score_rows(args)
    if scored:
        fieldnames = list(scored[0].keys())
    else:
        fieldnames = []
    for extra in ["judge_status", "judge_error", "judge_raw", "judge_mode", *SCORE_KEYS]:
        if extra not in fieldnames:
            fieldnames.append(extra)
    write_csv(args.output, scored, fieldnames)
    summary = summarize(scored)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
