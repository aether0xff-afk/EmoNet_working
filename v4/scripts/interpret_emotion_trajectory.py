from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from emonet.cli import ensure_model_server_ready, maybe_print_progress, request_json_response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Single-sample trace dir or batch trace dir")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sample-ids", default=None, help="Optional comma-separated sample ids to restrict interpretation")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--model-name", default="gpt-5.4")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--reasoning-effort", default="low")
    parser.add_argument("--progress-every", type=int, default=1)
    return parser.parse_args()


def resolve_api_key(api_key_env: str | None) -> str | None:
    if not api_key_env:
        return None
    value = os.environ.get(str(api_key_env), "").strip()
    if not value:
        raise ValueError(f"environment variable '{api_key_env}' is not set or empty")
    return value


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def detect_sample_dirs(input_dir: Path, requested_ids: list[str] | None = None) -> list[Path]:
    if (input_dir / "emotion_trace_summary.json").exists():
        return [input_dir]

    requested = set(requested_ids or [])
    ordered_dirs: list[Path] = []
    sample_summary_path = input_dir / "sample_summary.csv"
    if sample_summary_path.exists():
        sample_df = pd.read_csv(sample_summary_path)
        if "record_id" in sample_df.columns:
            for record_id in sample_df["record_id"].astype(str).tolist():
                if requested and record_id not in requested:
                    continue
                candidate = input_dir / record_id
                if (candidate / "emotion_trace_summary.json").exists():
                    ordered_dirs.append(candidate)

    known = {path.resolve() for path in ordered_dirs}
    for child in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        if requested and child.name not in requested:
            continue
        if (child / "emotion_trace_summary.json").exists() and child.resolve() not in known:
            ordered_dirs.append(child)
    return ordered_dirs


def sample_id_from_summary(summary: dict[str, Any], sample_dir: Path) -> str:
    input_meta = summary.get("input_meta", {})
    if isinstance(input_meta, dict) and str(input_meta.get("sample_id", "")).strip():
        return str(input_meta["sample_id"]).strip()
    return sample_dir.name


def describe_phase_row(row: dict[str, Any]) -> str:
    return (
        f"- {row['phase']} | tick {int(row['start_tick'])}-{int(row['end_tick'])} | "
        f"duration={int(row['duration'])} | active_nodes={float(row['mean_active_nodes']):.2f} | "
        f"edges={float(row['mean_edges_fired']):.2f} | dominant_signal={row['dominant_signal']} | "
        f"signal_conflict={float(row['signal_conflict']):.4f}"
    )


def select_key_ticks(tick_df: pd.DataFrame, trajectory_summary: dict[str, Any]) -> pd.DataFrame:
    if tick_df.empty:
        return tick_df.copy()
    tick_candidates = {
        0,
        int(trajectory_summary.get("peak_alarm_tick", 0)),
        int(trajectory_summary.get("peak_fatigue_tick", 0)),
        int(trajectory_summary.get("peak_conflict_tick", 0)),
        int(tick_df["tick"].max()),
    }
    if "phase" in tick_df.columns:
        for _, row in tick_df.groupby("phase", sort=False).head(1).iterrows():
            tick_candidates.add(int(row["tick"]))
    selected = tick_df[tick_df["tick"].isin(sorted(tick_candidates))].copy()
    return selected.sort_values("tick").drop_duplicates(subset=["tick"]).reset_index(drop=True)


def build_causal_transcript(
    *,
    summary: dict[str, Any],
    trajectory_summary: dict[str, Any],
    phase_df: pd.DataFrame,
    tick_df: pd.DataFrame,
    top_nodes_df: pd.DataFrame,
) -> str:
    sample_id = sample_id_from_summary(summary, Path("."))
    input_text = str(summary.get("input_text", "")).strip()
    key_ticks = select_key_ticks(tick_df, trajectory_summary)
    phase_lines = [describe_phase_row(row) for row in phase_df.to_dict(orient="records")]

    top_node_lines: list[str] = []
    for row in top_nodes_df.head(8).to_dict(orient="records"):
        top_node_lines.append(
            f"- node={int(row['node_id'])} | type={row['neuron_type']} | bias={row['bias_label']} | "
            f"activity_ticks={int(row['activity_ticks'])} | k_mean={float(row['k_mean']):.4f} | "
            f"stim=(drive={float(row['stim_drive']):.4f}, brake={float(row['stim_brake']):.4f}, "
            f"alarm={float(row['stim_alarm']):.4f}, fatigue={float(row['stim_fatigue']):.4f})"
        )

    tick_lines: list[str] = []
    for row in key_ticks.to_dict(orient="records"):
        tick_lines.append(
            f"- tick={int(row['tick'])} | active_nodes={int(row['active_nodes'])} | edges_fired={int(row['edges_fired'])} | "
            f"dominant_signal={row['dominant_signal']} | combined=(drive={float(row['combined_drive']):.4f}, "
            f"brake={float(row['combined_brake']):.4f}, alarm={float(row['combined_alarm']):.4f}, "
            f"fatigue={float(row['combined_fatigue']):.4f}) | mix=(inh={float(row['inhibitory_ratio']):.4f}, "
            f"exc={float(row['excitatory_ratio']):.4f}, mod={float(row['modulatory_ratio']):.4f})"
        )

    lines = [
        "[SAMPLE]",
        f"id={sample_id}",
        "",
        "[INPUT_TEXT]",
        input_text,
        "",
        "[GLOBAL_TRACE]",
        f"- ticks_run={int(summary['ticks_run'])}",
        f"- termination_reason={summary['termination_reason']}",
        f"- dominant_branch_len={int(summary['dominant_branch_len'])}",
        f"- active_tick_count={int(summary.get('active_tick_count', 0))}",
        f"- persistence_ratio={float(summary['persistence_ratio']):.4f}",
        f"- saturation_ratio={float(summary['saturation_ratio']):.4f}",
        f"- dominant_global_signal={summary['dominant_global_signal']}",
        f"- signal_means=(drive={float(summary['drive']):.4f}, brake={float(summary['brake']):.4f}, alarm={float(summary['alarm']):.4f}, fatigue={float(summary['fatigue']):.4f})",
        f"- node_mix=(inh={float(summary['inhibitory_ratio']):.4f}, exc={float(summary['excitatory_ratio']):.4f}, mod={float(summary['modulatory_ratio']):.4f})",
        "",
        "[TRAJECTORY]",
        f"- trajectory_pattern={trajectory_summary['trajectory_pattern']}",
        f"- phase_count={int(trajectory_summary['phase_count'])}",
        f"- phase_sequence={' -> '.join(str(item) for item in trajectory_summary['phase_sequence'])}",
        f"- peak_alarm_tick={int(trajectory_summary['peak_alarm_tick'])}",
        f"- peak_fatigue_tick={int(trajectory_summary['peak_fatigue_tick'])}",
        f"- peak_conflict_tick={int(trajectory_summary['peak_conflict_tick'])}",
        "",
        "[PHASE_SEGMENTS]",
        *(phase_lines or ["- no phases"]),
        "",
        "[KEY_TICKS]",
        *(tick_lines or ["- no key ticks"]),
        "",
        "[TOP_ACTIVE_NODES]",
        *(top_node_lines or ["- no top nodes"]),
        "",
        "[IMPORTANT]",
        "- 감정은 입력 자극 자체가 아니라, 이 자극이 네트워크 안에서 어떻게 점화되고 지속되고 충돌하고 수렴하는지의 episode로 해석한다.",
        "- 단일 high-arousal 상태를 자동으로 분노로 단정하지 않는다.",
        "- 접근 신호가 크고 alarm이 지배적이지 않다면 기대/호감/들뜸 같은 positive arousal 가능성을 검토한다.",
        "- 활성화가 거의 없으면 강한 감정을 꾸며내지 말고 미점화, 보류, anticipatory state 가능성을 명시한다.",
    ]
    return "\n".join(lines)


def build_episode_prompt(transcript: str) -> str:
    return "\n".join(
        [
            "[ROLE]",
            "당신은 감정을 자극 라벨이 아니라 정보 전달 episode로 해석하는 분석가다.",
            "",
            "[TASK]",
            "아래 trajectory transcript를 읽고, 입력 사건이 네트워크 내부에서 어떤 감정 episode로 처리되었는지 한국어 JSON 하나로 해석하라.",
            "",
            "[RULES]",
            "- 입력 문장의 표면 감정 표현보다 trace evidence를 우선한다.",
            "- 감정은 stimulus, appraisal, trajectory, action tendency를 함께 본 episode로 해석한다.",
            "- evidence에는 반드시 transcript에서 직접 근거가 되는 문장 2개 이상을 적는다.",
            "- soft한 상담 답변을 만들지 말고, 내부 처리 상태를 솔직하게 진단한다.",
            "",
            "[OUTPUT_JSON_SCHEMA]",
            "{",
            '  "episode_label": "짧은 한국어 라벨",',
            '  "stimulus_reading": "무슨 일이 자극이었는지 1-2문장",',
            '  "appraisal": {',
            '    "primary_appraisal": "핵심 해석",',
            '    "secondary_appraisal": "보조 해석",',
            '    "target": "self|other|situation|mixed",',
            '    "control_state": "low|mixed|high",',
            '    "social_orientation": "approach|defend|withdraw|mixed"',
            "  },",
            '  "trajectory": {',
            '    "overall_pattern": "trajectory pattern 해석",',
            '    "ignition": "점화 구간 해석",',
            '    "persistence": "지속/증폭 구간 해석",',
            '    "resolution": "마지막 수렴 상태 해석"',
            "  },",
            '  "action_tendency": "지금 이 상태가 어떤 행동 성향으로 가는지",',
            '  "rawness": {',
            '    "valence": "negative|positive|mixed|flat",',
            '    "arousal": "low|medium|high",',
            '    "softened_output_risk": "low|medium|high",',
            '    "should_preserve_harshness": true',
            "  },",
            '  "response_guidance": {',
            '    "preserve": "출력에서 반드시 남겨야 할 결",',
            '    "avoid": "순화하거나 왜곡하면 안 되는 것",',
            '    "tone_hint": "응답 표면 톤 힌트"',
            "  },",
            '  "evidence": ["근거1", "근거2"],',
            '  "confidence": 0.0',
            "}",
            "",
            "[TRANSCRIPT]",
            transcript,
        ]
    )


def normalize_nonempty_string(value: object, field_name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"missing non-empty string for '{field_name}'")
    return text


def normalize_enum(value: object, field_name: str, allowed: set[str]) -> str:
    text = normalize_nonempty_string(value, field_name)
    if text not in allowed:
        raise ValueError(f"invalid value for '{field_name}': {text}")
    return text


def normalize_episode_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("payload must be a JSON object")

    appraisal = payload.get("appraisal")
    trajectory = payload.get("trajectory")
    rawness = payload.get("rawness")
    guidance = payload.get("response_guidance")
    evidence = payload.get("evidence")

    if not isinstance(appraisal, dict):
        raise ValueError("appraisal object is required")
    if not isinstance(trajectory, dict):
        raise ValueError("trajectory object is required")
    if not isinstance(rawness, dict):
        raise ValueError("rawness object is required")
    if not isinstance(guidance, dict):
        raise ValueError("response_guidance object is required")
    if not isinstance(evidence, list) or len(evidence) < 2:
        raise ValueError("evidence must contain at least two items")

    normalized = {
        "episode_label": normalize_nonempty_string(payload.get("episode_label"), "episode_label"),
        "stimulus_reading": normalize_nonempty_string(payload.get("stimulus_reading"), "stimulus_reading"),
        "appraisal": {
            "primary_appraisal": normalize_nonempty_string(appraisal.get("primary_appraisal"), "appraisal.primary_appraisal"),
            "secondary_appraisal": normalize_nonempty_string(appraisal.get("secondary_appraisal"), "appraisal.secondary_appraisal"),
            "target": normalize_enum(appraisal.get("target"), "appraisal.target", {"self", "other", "situation", "mixed"}),
            "control_state": normalize_enum(appraisal.get("control_state"), "appraisal.control_state", {"low", "mixed", "high"}),
            "social_orientation": normalize_enum(
                appraisal.get("social_orientation"),
                "appraisal.social_orientation",
                {"approach", "defend", "withdraw", "mixed"},
            ),
        },
        "trajectory": {
            "overall_pattern": normalize_nonempty_string(trajectory.get("overall_pattern"), "trajectory.overall_pattern"),
            "ignition": normalize_nonempty_string(trajectory.get("ignition"), "trajectory.ignition"),
            "persistence": normalize_nonempty_string(trajectory.get("persistence"), "trajectory.persistence"),
            "resolution": normalize_nonempty_string(trajectory.get("resolution"), "trajectory.resolution"),
        },
        "action_tendency": normalize_nonempty_string(payload.get("action_tendency"), "action_tendency"),
        "rawness": {
            "valence": normalize_enum(rawness.get("valence"), "rawness.valence", {"negative", "positive", "mixed", "flat"}),
            "arousal": normalize_enum(rawness.get("arousal"), "rawness.arousal", {"low", "medium", "high"}),
            "softened_output_risk": normalize_enum(
                rawness.get("softened_output_risk"),
                "rawness.softened_output_risk",
                {"low", "medium", "high"},
            ),
            "should_preserve_harshness": bool(rawness.get("should_preserve_harshness")),
        },
        "response_guidance": {
            "preserve": normalize_nonempty_string(guidance.get("preserve"), "response_guidance.preserve"),
            "avoid": normalize_nonempty_string(guidance.get("avoid"), "response_guidance.avoid"),
            "tone_hint": normalize_nonempty_string(guidance.get("tone_hint"), "response_guidance.tone_hint"),
        },
        "evidence": [normalize_nonempty_string(item, "evidence") for item in evidence[:6]],
        "confidence": float(payload.get("confidence")),
    }
    if not 0.0 <= normalized["confidence"] <= 1.0:
        raise ValueError("confidence must be in [0,1]")
    return normalized


def load_sample_artifacts(sample_dir: Path) -> dict[str, Any]:
    summary = load_json(sample_dir / "emotion_trace_summary.json")
    trajectory_summary = load_json(sample_dir / "emotion_trajectory_summary.json")
    phase_df = pd.read_csv(sample_dir / "trajectory_phases.csv")
    tick_df = pd.read_csv(sample_dir / "trajectory_ticks.csv")
    top_nodes_df = pd.read_csv(sample_dir / "top_nodes.csv")
    return {
        "summary": summary,
        "trajectory_summary": trajectory_summary,
        "phase_df": phase_df,
        "tick_df": tick_df,
        "top_nodes_df": top_nodes_df,
    }


def flatten_episode_row(*, sample_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    appraisal = payload["appraisal"]
    trajectory = payload["trajectory"]
    rawness = payload["rawness"]
    guidance = payload["response_guidance"]
    row = {
        "sample_id": sample_id,
        "episode_label": payload["episode_label"],
        "stimulus_reading": payload["stimulus_reading"],
        "primary_appraisal": appraisal["primary_appraisal"],
        "secondary_appraisal": appraisal["secondary_appraisal"],
        "target": appraisal["target"],
        "control_state": appraisal["control_state"],
        "social_orientation": appraisal["social_orientation"],
        "overall_pattern": trajectory["overall_pattern"],
        "ignition": trajectory["ignition"],
        "persistence": trajectory["persistence"],
        "resolution": trajectory["resolution"],
        "action_tendency": payload["action_tendency"],
        "valence": rawness["valence"],
        "arousal": rawness["arousal"],
        "softened_output_risk": rawness["softened_output_risk"],
        "should_preserve_harshness": rawness["should_preserve_harshness"],
        "preserve": guidance["preserve"],
        "avoid": guidance["avoid"],
        "tone_hint": guidance["tone_hint"],
        "confidence": payload["confidence"],
        "evidence_json": json.dumps(payload["evidence"], ensure_ascii=False),
    }
    return row


def write_report(output_dir: Path, episode_df: pd.DataFrame) -> None:
    report_path = output_dir / "EPISODE_INTERPRETATION_REPORT.md"
    lines = [
        "# Emotion Episode Interpretation Report",
        "",
        f"- samples: {int(len(episode_df))}",
        "",
        "## Episode Summary",
        "",
    ]
    for row in episode_df.to_dict(orient="records"):
        lines.append(
            f"- {row['sample_id']}: {row['episode_label']} | "
            f"{row['primary_appraisal']} | {row['action_tendency']} | "
            f"valence={row['valence']} arousal={row['arousal']} "
            f"softened_risk={row['softened_output_risk']} confidence={float(row['confidence']):.2f}"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_ids = [item.strip() for item in str(args.sample_ids or "").split(",") if item.strip()] or None
    sample_dirs = detect_sample_dirs(input_dir, requested_ids)
    if args.limit is not None:
        sample_dirs = sample_dirs[: max(0, int(args.limit))]
    if not sample_dirs:
        raise ValueError("no sample trace directories found")

    api_key = resolve_api_key(args.api_key_env)
    ensure_model_server_ready(args.base_url, args.timeout_sec, api_key=api_key)

    rows: list[dict[str, Any]] = []
    start_time = time.perf_counter()
    jsonl_path = output_dir / "episode_payloads.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
        for idx, sample_dir in enumerate(sample_dirs, start=1):
            artifacts = load_sample_artifacts(sample_dir)
            summary = artifacts["summary"]
            trajectory_summary = artifacts["trajectory_summary"]
            phase_df = artifacts["phase_df"]
            tick_df = artifacts["tick_df"]
            top_nodes_df = artifacts["top_nodes_df"]
            sample_id = sample_id_from_summary(summary, sample_dir)

            transcript = build_causal_transcript(
                summary=summary,
                trajectory_summary=trajectory_summary,
                phase_df=phase_df,
                tick_df=tick_df,
                top_nodes_df=top_nodes_df,
            )
            prompt = build_episode_prompt(transcript)
            payload, raw = request_json_response(
                base_url=args.base_url,
                model_name=args.model_name,
                prompt=prompt,
                temperature=0.0,
                max_tokens=args.max_tokens,
                timeout_sec=args.timeout_sec,
                max_retries=args.max_retries,
                validator=normalize_episode_payload,
                retry_instruction=(
                    "직전 응답은 schema를 지키지 못했다. JSON object 하나만 다시 출력하고, "
                    "required keys를 모두 채워라."
                ),
                api_key=api_key,
                response_format={"type": "json_object"},
                reasoning_effort=args.reasoning_effort,
            )

            sample_out = output_dir / sample_id
            sample_out.mkdir(parents=True, exist_ok=True)
            (sample_out / "episode_transcript.txt").write_text(transcript + "\n", encoding="utf-8")
            (sample_out / "episode_prompt.txt").write_text(prompt + "\n", encoding="utf-8")
            (sample_out / "episode_interpretation.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (sample_out / "episode_raw_output.txt").write_text(str(raw), encoding="utf-8")

            jsonl_record = {"sample_id": sample_id, **payload}
            jsonl_file.write(json.dumps(jsonl_record, ensure_ascii=False) + "\n")
            rows.append(flatten_episode_row(sample_id=sample_id, payload=payload))

            maybe_print_progress(
                "trajectory-interpret",
                idx,
                len(sample_dirs),
                start_time,
                every=max(1, int(args.progress_every)),
                unit="samples",
                force=idx == len(sample_dirs),
            )

    episode_df = pd.DataFrame(rows)
    episode_df.to_csv(output_dir / "episode_summary.csv", index=False, encoding="utf-8-sig")
    write_report(output_dir, episode_df)

    payload = {
        "output_dir": str(output_dir),
        "samples": int(len(episode_df)),
        "episode_summary_csv": str(output_dir / "episode_summary.csv"),
        "episode_payloads_jsonl": str(jsonl_path),
        "report_path": str(output_dir / "EPISODE_INTERPRETATION_REPORT.md"),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
