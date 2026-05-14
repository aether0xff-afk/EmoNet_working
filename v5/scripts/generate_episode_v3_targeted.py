from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import urllib.request
import os

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from emonet.episode_conditioning import build_episode_v3_generation_prompt, load_episode_payload
from emonet.llm_api import request_plain_text_response


OUTPUT_COLUMNS = [
    "record_id",
    "condition",
    "condition_group",
    "status",
    "error_message",
    "text",
    "llm_response",
    "response_length",
    "prompt",
    "episode_label",
    "valence",
    "arousal",
    "target",
    "control_state",
    "social_orientation",
    "preserve",
    "avoid",
    "action_tendency",
]


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
            {"role": "system", "content": "Return a plain Korean response only. Do not return JSON. Keep reasoning brief."},
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


def load_existing_baselines(path: Path, conditions: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    df = pd.read_csv(path)
    if "status" in df.columns:
        df = df[df["status"].fillna("") == "ok"].copy()
    df = df[df["condition"].astype(str).isin(conditions)].copy()
    for column in OUTPUT_COLUMNS:
        if column not in df.columns:
            df[column] = ""
    df["condition_group"] = df["condition_group"].fillna("baseline")
    return df[OUTPUT_COLUMNS]


def build_reference_map(targeted: pd.DataFrame) -> dict[str, dict[str, object]]:
    return {str(row["record_id"]): row for row in targeted.to_dict(orient="records")}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate targeted episode_trace_v3 responses without fitting EmoNet.")
    parser.add_argument("--targeted-csv", default="outputs/experiments/superiority_targeted_v1/targeted_records.csv")
    parser.add_argument("--baseline-matrix-csv", default="outputs/experiments/paper_matrix_current_episode_v2_scored.csv")
    parser.add_argument("--episode-dir", default="outputs/research/trajectory_batch_matrix120_v1_gpt54")
    parser.add_argument("--output-csv", default="outputs/experiments/superiority_targeted_v1/targeted_matrix.csv")
    parser.add_argument("--summary-json", default="outputs/experiments/superiority_targeted_v1/targeted_matrix_summary.json")
    parser.add_argument("--baseline-conditions", default="stim_only,episode_trace")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--provider", default="openai_compatible", choices=["openai_compatible", "anthropic"])
    parser.add_argument("--model-name", default="gpt-oss:20b")
    parser.add_argument("--api-key-env", default=None)
    parser.add_argument("--temperature", type=float, default=0.45)
    parser.add_argument("--max-tokens", type=int, default=1600)
    parser.add_argument("--timeout-sec", type=int, default=240)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=1)
    args = parser.parse_args()
    api_key = os.environ.get(args.api_key_env, "").strip() if args.api_key_env else None

    targeted = pd.read_csv(args.targeted_csv)
    if args.limit is not None and args.limit > 0:
        targeted = targeted.head(args.limit).copy()
    target_ids = set(targeted["record_id"].astype(str))
    reference_by_id = build_reference_map(targeted)

    baseline_conditions = [token.strip() for token in args.baseline_conditions.split(",") if token.strip()]
    baseline = load_existing_baselines(Path(args.baseline_matrix_csv), baseline_conditions)
    baseline = baseline[baseline["record_id"].astype(str).isin(target_ids)].copy()
    for idx, row in baseline.iterrows():
        ref = reference_by_id.get(str(row["record_id"]), {})
        for column in ["episode_label", "valence", "arousal", "target", "control_state", "social_orientation", "preserve", "avoid", "action_tendency"]:
            baseline.at[idx, column] = ref.get(column, baseline.at[idx, column] if column in baseline.columns else "")

    rows = baseline.to_dict(orient="records")
    start = time.perf_counter()
    for idx, record in enumerate(targeted.to_dict(orient="records"), start=1):
        record_id = str(record["record_id"])
        row = {
            "record_id": record_id,
            "condition": "episode_trace_v3",
            "condition_group": "episode",
            "status": "error",
            "error_message": "",
            "text": str(record.get("text", "")),
            "llm_response": "",
            "response_length": 0,
            "prompt": "",
            "episode_label": str(record.get("episode_label", "")),
            "valence": str(record.get("valence", "")),
            "arousal": str(record.get("arousal", "")),
            "target": str(record.get("target", "")),
            "control_state": str(record.get("control_state", "")),
            "social_orientation": str(record.get("social_orientation", "")),
            "preserve": str(record.get("preserve", "")),
            "avoid": str(record.get("avoid", "")),
            "action_tendency": str(record.get("action_tendency", "")),
        }
        try:
            episode_payload = load_episode_payload(Path(args.episode_dir) / record_id / "episode_interpretation.json")
            prompt = build_episode_v3_generation_prompt(input_text=row["text"], episode_payload=episode_payload)
            response, _raw, _meta = request_plain_text_response(
                provider=args.provider,
                base_url=args.base_url,
                model_name=args.model_name,
                prompt=prompt,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                timeout_sec=args.timeout_sec,
                max_retries=args.max_retries,
                system_prompt="Return a plain Korean response only. Do not return JSON. Keep reasoning brief.",
                api_key=api_key,
            )
            if not response:
                raise ValueError("empty LLM response")
            row["status"] = "ok"
            row["prompt"] = prompt
            row["llm_response"] = response
            row["response_length"] = len(response)
        except Exception as exc:
            row["error_message"] = str(exc)
        rows.append(row)
        if args.progress_every > 0 and idx % args.progress_every == 0:
            elapsed = time.perf_counter() - start
            print(f"generate-episode-v3-targeted: {idx}/{len(targeted)} elapsed={elapsed:.1f}s")

    output = pd.DataFrame(rows)
    output = output[OUTPUT_COLUMNS]
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False, encoding="utf-8-sig")

    summary = {
        "output_csv": str(output_path),
        "rows": int(len(output)),
        "conditions": {
            str(condition): {
                "rows": int(len(group)),
                "ok_rows": int((group["status"].fillna("") == "ok").sum()),
                "error_rows": int((group["status"].fillna("") != "ok").sum()),
            }
            for condition, group in output.groupby("condition", dropna=False)
        },
    }
    summary_path = Path(args.summary_json)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
