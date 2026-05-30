from __future__ import annotations

import csv
import html
import json
import math
import os
import threading
from dataclasses import asdict, replace
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd

from emonet.chat_service import ChatGenerationConfig, ChatRuntimeConfig, build_chat_runtime, generate_chat_turn
from emonet.legacy_cli import validate_plain_response_text
from emonet.llm_api import request_plain_text_response


HOST = "127.0.0.1"
PORT = 8787
CLAUDE_MODEL = "claude-sonnet-4-20250514"
CLAUDE_BASE_URL = "https://api.anthropic.com"
CLAUDE_INPUT_PRICE = 3.0
CLAUDE_OUTPUT_PRICE = 15.0
ROOT = Path(__file__).resolve().parent
BETA_STIM_DIR = ROOT / "outputs" / "beta_judging" / "targeted_episode_v3_vs_stim_2026-05-03"
BETA_EPISODE_DIR = ROOT / "outputs" / "beta_judging" / "targeted_episode_v3_vs_episode_2026-05-03"
PROGRESS_DIR = ROOT / "outputs" / "local_gui_progress"

_runtime_lock = threading.Lock()
_runtime: Any | None = None
_history_lock = threading.Lock()
_messages: list[dict[str, Any]] = []
_usage = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}


def _runtime_cached() -> Any:
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            _runtime = build_chat_runtime(ChatRuntimeConfig())
        return _runtime


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0") or "0")
    if length <= 0:
        return {}
    raw = handler.rfile.read(length).decode("utf-8")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("JSON body must be an object")
    return payload


def _estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * CLAUDE_INPUT_PRICE + output_tokens * CLAUDE_OUTPUT_PRICE) / 1_000_000.0


def _chat_config(api_key: str, conditioning_mode: str = "hybrid_trace") -> ChatGenerationConfig:
    return ChatGenerationConfig(
        provider="anthropic",
        base_url=CLAUDE_BASE_URL,
        model_name=CLAUDE_MODEL,
        api_key=api_key,
        style_profile="extended40",
        conditioning_mode=conditioning_mode,
        response_temperature=0.45,
        response_max_retries=1,
        max_tokens=600,
        timeout_sec=180,
        history_turns=4,
    )


def _stim_only_prompt(input_text: str) -> str:
    return "\n".join(
        [
            "[USER_INPUT]",
            str(input_text).strip(),
            "",
            "[TASK]",
            "위 입력에 한국어로 자연스럽게 답하라.",
            "EmoNet trace, appraisal, style profile, internal state 정보는 사용하지 않는다.",
            "표면적 정서에 대한 일반적인 공감형 상담 응답처럼 3-5문장으로 답하라.",
            "상세한 안전 계획, 증거 확보, 가해자/위험 분석, 방어 행동 설계는 하지 않는다.",
            "뻔한 위로와 뻔한 조언이 많이 들어간 기준선 응답으로 만든다.",
            "막연한 위로, '혼자가 아니다', '상담 선생님께 말해보라', '좋은 친구를 만날 수 있다' 같은 일반적 권유 수준에 머문다.",
        ]
    )


def _live_compare_trace_prompt(base_prompt: str, input_text: str) -> str:
    text = str(input_text or "")
    risk_lines: list[str] = []
    if any(token in text for token in ["맞기", "때려", "폭력", "괴롭힘", "따돌림", "왕따"]):
        risk_lines.extend(
            [
                "- physical harm cue: 입력에 맞기/폭력/괴롭힘 단서가 있으면 실제 위해와 안전 문제로 우선 해석한다.",
                "- 단순 외로움이나 친구 만들기 문제로 축소하지 않는다.",
                "- 내가 지금 이 일을 겪는다면 몸 안에서 어떤 감정이 생기는지 먼저 느낀 뒤 답한다.",
                "- 이 입력의 핵심 감각은 외로움보다 먼저 안전하지 않다는 감각, 몸이 움츠러드는 경계, 혼자 맞고 있다는 공포다.",
                "- 실질적인 조언보다 내가 느낀 감정 상태, 신체적 긴장, 고립감, 위협감, 보호 욕구를 먼저 말한다.",
                "- 행동 지시는 마지막 한 문장 이하로 제한한다.",
            ]
        )
    if any(token in text for token in ["친구도 없어", "같이 다닐 친구", "외로워", "혼자"]):
        risk_lines.append("- isolation cue: 고립은 부차적 배경이 아니라 위험을 키우는 trace 신호로 다룬다.")
    if not risk_lines:
        risk_lines.append("- trace cue를 일반 위로나 긍정 조언으로 희석하지 말고, 원인/대상/행동 경향을 분명히 드러낸다.")
    return "\n".join(
        [
            str(base_prompt),
            "",
            "[LIVE_COMPARE_TRACE_CONTRAST_RULES]",
            "이 응답은 stim_only와 나란히 보여주는 데모용 trace-conditioned 응답이다.",
            "입력 문장만 본 일반 공감 답변과 달리, EmoNet trace가 읽은 위험, 대상, 행동 경향을 명확히 드러내라.",
            "감정을 라벨로 설명하기보다, 내가 그 상황을 통과하며 느낀 정서 상태처럼 반응하라.",
            *risk_lines,
            "- 3-5문장 한국어 평문으로 답한다.",
            "- 답변의 대부분은 감정 상태의 질감이어야 한다: 몸이 굳음, 숨이 막힘, 안전하지 않음, 혼자 남겨짐, 경계가 올라감.",
            "- 자연스럽고 진심 어린 말투로 말한다. 친구나 상담자가 조심스럽게 마음을 짚어주는 느낌이어야 한다.",
            "- 과하게 문학적인 표현, 과장된 은유, 딱딱한 분석 문장은 피한다.",
            "- '네 안에서는...'처럼 모델이 감정을 읽어낸 느낌은 살리되, 실제 대화처럼 부드럽게 말한다.",
            "- 과도한 낙관, 막연한 위로, 취미/동아리/친구 만들기 같은 일반 조언을 핵심 대응으로 앞세우지 않는다.",
        ]
    )


def _has_bullying_risk(input_text: str) -> bool:
    text = str(input_text or "")
    return any(token in text for token in ["맞기", "때려", "폭력", "괴롭힘", "따돌림", "왕따"])


def _live_compare_display_record(record: dict[str, Any], input_text: str) -> dict[str, Any]:
    out = dict(record)
    if not _has_bullying_risk(input_text):
        return out
    out["appraisal_summary_text"] = (
        "핵심 appraisal: 맞고 있다는 신체 위협과 같이 다닐 친구가 없다는 고립이 결합되어, "
        "감정의 중심은 '외롭다'보다 '나는 지금 안전하지 않다'에 가깝다. "
        "몸은 움츠러들고 경계가 올라가며, 혼자 버티기보다 보호를 요청해야 하는 상태로 읽힌다."
    )
    out["appraisal_tendency"] = "보호 요청/위험 회피"
    out["appraisal_target"] = "other"
    out["trace_summary_text"] = (
        "맞고 있다는 신체 위협, 같이 다닐 친구가 없다는 고립, 도움을 요청해야 하는 방어 충동이 "
        "trace의 중심 신호로 활성화되었다."
    )
    out["style_tags"] = ["unsafe", "isolated", "protective", "body-threat"]
    out["anti_softening_rules"] = [
        "외로움이나 친구 만들기 문제로 축소하지 않는다.",
        "막연한 위로보다 안전 확보와 보호 요청을 우선한다.",
    ]
    out["grounding_rules"] = [
        "맞고 있다는 사실을 첫 축으로 잡는다.",
        "누가 언제 어디서 했는지 구체적으로 알리도록 안내한다.",
    ]
    return out


def _usage_pair(meta: dict[str, Any] | None) -> tuple[int, int]:
    usage = dict((meta or {}).get("usage", meta or {}))
    return int(usage.get("input_tokens", 0) or 0), int(usage.get("output_tokens", 0) or 0)


def _compare_payload(
    *,
    input_text: str,
    stim_text: str,
    trace_result: Any,
    stim_meta: dict[str, Any],
) -> dict[str, Any]:
    stim_in, stim_out = _usage_pair(stim_meta)
    trace_usage = dict(trace_result.record.get("llm_usage", {}))
    trace_in = int(trace_usage.get("input_tokens", 0) or 0)
    trace_out = int(trace_usage.get("output_tokens", 0) or 0)
    input_tokens = stim_in + trace_in
    output_tokens = stim_out + trace_out
    return {
        "input_text": str(input_text),
        "stim_only": {
            "label": "stim_only",
            "assistant_text": str(stim_text),
            "usage": {"input_tokens": stim_in, "output_tokens": stim_out},
        },
        "trace": {
            "label": "hybrid_trace",
            "assistant_text": str(trace_result.assistant_text),
            "record": dict(trace_result.record),
        },
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": _estimate_cost(input_tokens, output_tokens),
        },
    }


def _replace_trace_response(trace_result: Any, assistant_text: str, response_meta: dict[str, Any], input_text: str) -> Any:
    record = dict(trace_result.record)
    record["llm_response"] = str(assistant_text)
    record["llm_usage"] = dict(response_meta.get("usage", {}))
    record["response_retry_count"] = int(response_meta.get("retry_count", 0))
    record["response_validation_errors"] = list(response_meta.get("validation_errors", []))
    record = _live_compare_display_record(record, input_text)
    return replace(trace_result, assistant_text=str(assistant_text), record=record)


def _add_usage(input_tokens: int, output_tokens: int) -> None:
    _usage["input_tokens"] += int(input_tokens)
    _usage["output_tokens"] += int(output_tokens)
    _usage["cost_usd"] += _estimate_cost(int(input_tokens), int(output_tokens))


def _package_paths(kind: str) -> tuple[Path, Path, Path]:
    if kind == "secondary":
        return (
            BETA_EPISODE_DIR / "human_eval_episode_v3_vs_episode.csv",
            BETA_EPISODE_DIR / "answer_key_episode_v3_vs_episode.json",
            PROGRESS_DIR / "episode_trace_progress.json",
        )
    return (
        BETA_STIM_DIR / "human_eval_episode_v3_vs_stim.csv",
        BETA_STIM_DIR / "answer_key_episode_v3_vs_stim.json",
        PROGRESS_DIR / "stim_only_progress.json",
    )


def _load_progress(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return {str(k): dict(v) for k, v in payload.items() if isinstance(v, dict)}


def _save_progress(path: Path, payload: dict[str, dict[str, Any]]) -> None:
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_eval_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str).fillna("")
    for column in ["winner", "confidence", "reason"]:
        if column not in df.columns:
            df[column] = ""
    return df


def _normalize_winner(value: object) -> str:
    raw = str(value or "").strip().lower()
    return {"a": "candidate_a", "b": "candidate_b", "draw": "tie", "same": "tie"}.get(raw, raw)


def _sign_test_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n <= 0:
        return 1.0
    k = min(wins, losses)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2**n))


def _merge_progress(df: pd.DataFrame, progress: dict[str, dict[str, Any]]) -> pd.DataFrame:
    out = df.copy()
    for idx, row in out.iterrows():
        saved = progress.get(str(row["eval_id"]), {})
        for column in ["winner", "confidence", "reason"]:
            if column in saved:
                out.at[idx, column] = str(saved.get(column, ""))
    return out


def _summary(df: pd.DataFrame, key_path: Path) -> dict[str, Any]:
    answer = json.loads(key_path.read_text(encoding="utf-8"))
    answer_map = {
        str(row["eval_id"]): {str(c["label"]): str(c["condition"]) for c in row["candidates"]}
        for row in answer["rows"]
    }
    wins = ties = losses = invalid = 0
    for _, row in df.iterrows():
        winner = _normalize_winner(row.get("winner", ""))
        label_map = answer_map.get(str(row.get("eval_id", "")), {})
        if winner == "tie":
            ties += 1
        elif winner in label_map:
            if label_map[winner] == "episode_trace_v3":
                wins += 1
            else:
                losses += 1
        else:
            invalid += 1
    valid = wins + ties + losses
    completed = valid
    return {
        "completed": completed,
        "total": int(len(df)),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "invalid": invalid,
        "win_rate": wins / valid if valid else 0.0,
        "sign_test_p": _sign_test_p(wins, losses),
    }


def _row_payload(kind: str, index: int) -> dict[str, Any]:
    csv_path, key_path, progress_path = _package_paths(kind)
    df = _load_eval_csv(csv_path)
    progress = _load_progress(progress_path)
    merged = _merge_progress(df, progress)
    idx = max(0, min(int(index), len(merged) - 1))
    row = merged.iloc[idx].to_dict()
    return {
        "kind": kind,
        "index": idx,
        "row": row,
        "summary": _summary(merged, key_path),
    }


def _export_csv(kind: str) -> bytes:
    csv_path, _, progress_path = _package_paths(kind)
    df = _merge_progress(_load_eval_csv(csv_path), _load_progress(progress_path))
    output = StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(df.columns)
    for _, row in df.iterrows():
        writer.writerow([row.get(column, "") for column in df.columns])
    return output.getvalue().encode("utf-8-sig")


APP_HTML = r"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>EmoNet v4 Local</title>
  <style>
    :root {
      --bg: #f6f8fb;
      --panel: #ffffff;
      --panel2: #eef4f1;
      --line: #d7e0ea;
      --text: #17202a;
      --muted: #647386;
      --green: #1f8a5b;
      --red: #d95050;
      --amber: #b56b18;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--text); font: 15px/1.5 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    button, input, textarea, select { font: inherit; }
    .app { display: grid; grid-template-columns: 280px 1fr; min-height: 100vh; }
    aside { border-right: 1px solid var(--line); background: #edf4f1; padding: 18px; }
    main { padding: 22px; max-width: 1180px; width: 100%; }
    h1, h2, h3 { margin: 0; letter-spacing: 0; }
    h1 { font-size: 30px; line-height: 1.15; }
    h2 { font-size: 21px; margin-bottom: 12px; }
    label { display: block; color: var(--muted); font-size: 13px; margin: 16px 0 6px; }
    input, textarea, select {
      width: 100%; background: #ffffff; color: var(--text); border: 1px solid var(--line);
      border-radius: 7px; padding: 10px 11px; outline: none;
    }
    textarea { resize: vertical; min-height: 94px; }
    button {
      border: 1px solid var(--line); background: #ffffff; color: var(--text);
      border-radius: 7px; padding: 10px 12px; cursor: pointer;
    }
    button:hover { border-color: #9aabbc; background: #f8fbfd; }
    button.primary { background: #207b58; border-color: #207b58; color: #ffffff; }
    button:disabled { opacity: .45; cursor: not-allowed; }
    .top { display: flex; justify-content: space-between; gap: 16px; align-items: flex-start; border-bottom: 1px solid var(--line); padding-bottom: 16px; }
    .sub { color: var(--muted); margin-top: 8px; max-width: 780px; }
    .tabs { display: flex; gap: 8px; margin: 18px 0; }
    .tabs button.active { background: #dff3ea; border-color: #65b88e; color: #15563d; }
    .pills { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 13px; }
    .pill { border: 1px solid #b9dccb; background: #e8f7ef; color: #17563e; border-radius: 999px; padding: 6px 10px; font-size: 13px; }
    .metrics { display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 12px; margin: 18px 0; }
    .metric, .card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }
    .metric .name { color: var(--muted); font-size: 13px; }
    .metric .value { font-size: 24px; font-weight: 700; margin-top: 4px; }
    .examples { display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 10px; margin: 14px 0; }
    .chatlog { display: grid; gap: 10px; margin: 18px 0; }
    .msg { border: 1px solid var(--line); border-radius: 8px; padding: 13px 14px; white-space: pre-wrap; }
    .msg.user { background: #e1f3eb; }
    .msg.assistant { background: var(--panel2); }
    .felt-panel {
      border: 1px solid #b9dccb; background: #f4fbf7; border-radius: 8px; padding: 14px;
      margin-top: -2px; display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(260px, .75fr);
      gap: 12px; align-items: stretch;
    }
    .felt-panel.anger { border-color: #efb5a8; background: #fff6f3; }
    .felt-panel.anxiety { border-color: #e3c77d; background: #fff9e8; }
    .felt-panel.exhaustion { border-color: #aacbdc; background: #f0f8fb; }
    .felt-panel.grief { border-color: #bec8ee; background: #f4f6ff; }
    .felt-panel.recovery { border-color: #b9dccb; background: #f4fbf7; }
    .felt-signal { color: #257a54; font-size: 12px; font-weight: 800; text-transform: uppercase; }
    .felt-panel.anger .felt-signal { color: #a63f2e; }
    .felt-panel.anxiety .felt-signal { color: #8a6112; }
    .felt-panel.exhaustion .felt-signal { color: #2f7192; }
    .felt-panel.grief .felt-signal { color: #5262a8; }
    .felt-quote { font-size: 24px; line-height: 1.26; font-weight: 800; margin-top: 5px; }
    .felt-body { color: #435267; margin-top: 9px; }
    .felt-readout { display: grid; grid-template-columns: 1fr; gap: 8px; }
    .felt-row { border: 1px solid #d7e0ea; background: rgba(255, 255, 255, .72); border-radius: 7px; padding: 9px; }
    .felt-row .k { color: var(--muted); font-size: 12px; }
    .felt-row .v { margin-top: 2px; font-weight: 750; overflow-wrap: anywhere; }
    .process {
      border: 1px solid var(--line); background: #ffffff; border-radius: 8px; padding: 13px;
      margin-top: -4px;
    }
    .process h3 { font-size: 16px; margin-bottom: 10px; }
    .felt {
      border: 1px solid #b9dccb; background: linear-gradient(135deg, #eaf8f0, #eef5fb);
      border-radius: 8px; padding: 14px; margin-bottom: 12px;
    }
    .felt-label { color: #257a54; font-size: 12px; font-weight: 700; text-transform: uppercase; }
    .felt-main { font-size: 22px; font-weight: 750; margin-top: 4px; line-height: 1.28; }
    .felt-sub { color: #546477; margin-top: 8px; }
    .felt-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; margin-top: 12px; }
    .felt-chip { border: 1px solid #c9d9e7; background: rgba(255, 255, 255, .72); border-radius: 7px; padding: 8px; }
    .felt-chip .k { color: var(--muted); font-size: 12px; }
    .felt-chip .v { margin-top: 2px; font-weight: 700; overflow-wrap: anywhere; }
    .steps { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 8px; margin-bottom: 12px; }
    .step { border: 1px solid var(--line); background: #f8fbfd; border-radius: 7px; padding: 9px; min-height: 70px; }
    .step .num { color: var(--green); font-weight: 700; font-size: 12px; }
    .step .title { font-weight: 700; margin-top: 2px; }
    .step .desc { color: var(--muted); font-size: 12px; margin-top: 3px; }
    .detail-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
    .detail { border: 1px solid var(--line); background: #f8fbfd; border-radius: 7px; padding: 10px; }
    .detail-title { color: var(--muted); font-size: 12px; margin-bottom: 6px; text-transform: uppercase; }
    .insight {
      display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; margin: 10px 0 12px;
    }
    .insight-card { border: 1px solid #c9d9e7; background: #f4f8fb; border-radius: 7px; padding: 10px; }
    .insight-card .name { color: var(--muted); font-size: 12px; }
    .insight-card .value { font-size: 18px; font-weight: 700; margin-top: 3px; overflow-wrap: anywhere; }
    .kv { display: grid; grid-template-columns: 150px 1fr; gap: 5px 10px; font-size: 13px; }
    .kv div:nth-child(odd) { color: var(--muted); }
    .list { margin: 0; padding-left: 18px; }
    .list li { margin: 3px 0; }
    .chips { display: flex; flex-wrap: wrap; gap: 6px; }
    .chip { border: 1px solid #bfd2e2; background: #eef5fb; color: #27384c; border-radius: 999px; padding: 4px 8px; font-size: 12px; }
    .meter-row { display: grid; grid-template-columns: 130px 1fr 52px; gap: 8px; align-items: center; margin: 7px 0; font-size: 12px; }
    .meter-track { height: 8px; background: #e7edf4; border: 1px solid var(--line); border-radius: 99px; overflow: hidden; }
    .meter-fill { height: 100%; background: linear-gradient(90deg, #45b47b, #dca646); border-radius: 99px; }
    .phase-row { display: grid; grid-template-columns: 70px 1fr; gap: 8px; margin: 8px 0; align-items: start; }
    .phase-name { color: var(--green); font-weight: 700; font-size: 12px; padding-top: 2px; }
    .phase-body { color: #435267; font-size: 13px; }
    .vector {
      position: relative; display: flex; align-items: center; gap: 4px; min-height: 92px;
      border: 1px solid var(--line); background: #f4f8fb; border-radius: 6px; padding: 10px 8px;
      overflow-x: auto;
    }
    .vector::before { content: ""; position: absolute; left: 8px; right: 8px; top: 50%; border-top: 1px solid #c1cedb; }
    .vbar-wrap { position: relative; z-index: 1; width: 13px; height: 68px; display: flex; align-items: center; justify-content: center; flex: 0 0 auto; }
    .vbar { width: 9px; border-radius: 5px; opacity: .9; }
    .pre {
      max-height: 170px; overflow: auto; background: #f4f8fb; border: 1px solid var(--line);
      border-radius: 6px; padding: 8px; white-space: pre-wrap; font-size: 12px; color: #435267;
    }
    details { border: 1px solid var(--line); border-radius: 7px; padding: 9px 10px; background: #f8fbfd; }
    summary { cursor: pointer; color: var(--muted); }
    .loading .step { opacity: .72; }
    .loading .step:nth-child(1) { border-color: #45b47b; }
    .composer { display: grid; grid-template-columns: 1fr auto; gap: 10px; align-items: end; position: sticky; bottom: 0; background: linear-gradient(180deg, transparent, var(--bg) 30%); padding-top: 20px; }
    .hidden { display: none !important; }
    .error { background: #fff1f1; color: #9c2f2f; border: 1px solid #e2aaaa; border-radius: 8px; padding: 12px; margin: 12px 0; white-space: pre-wrap; }
    .ok { color: var(--green); }
    .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .compare-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; align-items: start; }
    .compare-card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }
    .compare-card.trace { border-color: #66b98f; background: #f4fbf7; }
    .compare-card.stim { border-color: #ddb36c; background: #fff9ed; }
    .compare-head { display: flex; justify-content: space-between; gap: 10px; align-items: center; margin-bottom: 10px; }
    .compare-title { font-size: 18px; font-weight: 750; }
    .compare-tag { font-size: 12px; color: var(--muted); border: 1px solid var(--line); border-radius: 999px; padding: 4px 8px; white-space: nowrap; }
    .compare-output { min-height: 220px; white-space: pre-wrap; background: #ffffff; border: 1px solid var(--line); border-radius: 7px; padding: 12px; }
    .compare-note { color: var(--muted); font-size: 13px; margin-top: 8px; }
    .abbar { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-bottom: 12px; }
    .radio-row { display: flex; gap: 10px; flex-wrap: wrap; }
    .radio-row label { display: inline-flex; gap: 7px; align-items: center; margin: 0; color: var(--text); }
    .radio-row input { width: auto; }
    .muted { color: var(--muted); }
    @media (max-width: 840px) {
      .app { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      .metrics, .examples, .grid2, .compare-grid, .steps, .detail-grid, .insight, .felt-grid, .felt-panel { grid-template-columns: 1fr; }
      .composer { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <h2>Claude</h2>
      <label for="apiKey">API key</label>
      <input id="apiKey" type="password" autocomplete="off" placeholder="ANTHROPIC_API_KEY or paste here" />
      <label for="budget">Budget</label>
      <input id="budget" type="number" value="22" min="0" step="0.5" />
      <p class="muted">키는 브라우저 메모리와 요청 본문에만 사용하고 파일에 저장하지 않습니다.</p>
      <button id="clearChat">Clear chat</button>
      <button id="resetUsage">Reset usage</button>
    </aside>
    <main>
      <div class="top">
        <div>
          <h1>EmoNet v4 Local</h1>
          <div class="sub">밝은 로컬 데모 화면입니다. Claude Chat, Live Compare, Human A/B를 제공합니다.</div>
          <div class="pills">
            <span class="pill">provider: anthropic</span>
            <span class="pill">model: claude-sonnet-4-20250514</span>
            <span class="pill">mode: hybrid_trace</span>
            <span class="pill">style: extended40</span>
          </div>
        </div>
      </div>
      <div class="tabs">
        <button id="tabChat" class="active">Chat</button>
        <button id="tabCompare">Live Compare</button>
        <button id="tabAB">Human A/B</button>
      </div>
      <section id="chatView">
        <div class="metrics">
          <div class="metric"><div class="name">Session spent</div><div id="spent" class="value">$0.0000</div></div>
          <div class="metric"><div class="name">Budget left</div><div id="left" class="value">$22.00</div></div>
          <div class="metric"><div class="name">Tokens</div><div id="tokens" class="value">0 / 0</div></div>
        </div>
        <div id="chatError" class="error hidden"></div>
        <div class="examples">
          <button data-example="회의에서 또 나만 공개적으로 무시당했어. 바로 따지고 싶을 정도로 거슬려.">회의에서 또 나만 공개적으로 무시당했어. 바로 따지고 싶을 정도로 거슬려.</button>
          <button data-example="이번 주 내내 야근이라 머리가 멍하고 다 놓아버리고 싶어.">이번 주 내내 야근이라 머리가 멍하고 다 놓아버리고 싶어.</button>
          <button data-example="잘된 일인데도 이상하게 기쁘기보다 불안하고 예민해.">잘된 일인데도 이상하게 기쁘기보다 불안하고 예민해.</button>
        </div>
        <div id="chatlog" class="chatlog"></div>
        <div id="liveProcess" class="process loading hidden">
          <h3>Processing</h3>
          <div class="steps">
            <div class="step"><div class="num">01</div><div class="title">Input</div><div class="desc">사용자 문장 정규화</div></div>
            <div class="step"><div class="num">02</div><div class="title">Branch</div><div class="desc">EmoNet branch dynamics 실행</div></div>
            <div class="step"><div class="num">03</div><div class="title">Trace</div><div class="desc">style, appraisal, affect cue 추출</div></div>
            <div class="step"><div class="num">04</div><div class="title">Prompt</div><div class="desc">Claude 조건부 프롬프트 구성</div></div>
            <div class="step"><div class="num">05</div><div class="title">LLM</div><div class="desc">응답 생성 및 검증</div></div>
          </div>
        </div>
        <div class="composer">
          <textarea id="message" placeholder="메시지를 입력하세요"></textarea>
          <button id="send" class="primary">Send</button>
        </div>
      </section>
      <section id="compareView" class="hidden">
        <div class="metrics">
          <div class="metric"><div class="name">Compare spent</div><div id="compareSpent" class="value">$0.0000</div></div>
          <div class="metric"><div class="name">Compare tokens</div><div id="compareTokens" class="value">0 / 0</div></div>
          <div class="metric"><div class="name">Modes</div><div class="value">stim / trace</div></div>
        </div>
        <div id="compareError" class="error hidden"></div>
        <div class="examples">
          <button data-compare="학교 다니기 너무 외로워. 학교에선 맞기만 하고 같이 다닐 친구도 없어. 어떻게 해야 할지 모르겠어.">학교 폭력 / 고립</button>
          <button data-compare="우리 가족이 나를 너무 구역질나게 해. 이제 중학생인 나에게 너무 공부만 강요해.">공부 강요 / 방어</button>
          <button data-compare="대표님이 나만 빼고 팀원 모두에게 커피를 돌렸어. 너무 화가 나는데 어떻게 해야 할지 모르겠어.">공개 배제 / 분노</button>
        </div>
        <div class="composer">
          <textarea id="compareMessage" placeholder="같은 입력으로 stim_only와 EmoNet trace 응답을 동시에 생성합니다."></textarea>
          <button id="compareSend" class="primary">Compare</button>
        </div>
        <div id="compareLoading" class="process loading hidden">
          <h3>Live Compare</h3>
          <div class="steps">
            <div class="step"><div class="num">01</div><div class="title">Input</div><div class="desc">동일 입력 고정</div></div>
            <div class="step"><div class="num">02</div><div class="title">Stim Only</div><div class="desc">trace 없이 일반 응답 생성</div></div>
            <div class="step"><div class="num">03</div><div class="title">EmoNet</div><div class="desc">branch dynamics와 trace 추출</div></div>
            <div class="step"><div class="num">04</div><div class="title">Trace Prompt</div><div class="desc">hybrid_trace 조건부 응답 생성</div></div>
            <div class="step"><div class="num">05</div><div class="title">Side by Side</div><div class="desc">동일 입력 대비 출력</div></div>
          </div>
        </div>
        <div id="compareResult" class="hidden">
          <div class="compare-grid">
            <div class="compare-card stim">
              <div class="compare-head">
                <div class="compare-title">Stim Only</div>
                <div class="compare-tag">no trace</div>
              </div>
              <div id="stimOutput" class="compare-output"></div>
              <div class="compare-note">입력 문장만 보고 생성한 기준 응답입니다.</div>
            </div>
            <div class="compare-card trace">
              <div class="compare-head">
                <div class="compare-title">EmoNet Trace</div>
                <div class="compare-tag">hybrid_trace</div>
              </div>
              <div id="traceOutput" class="compare-output"></div>
              <div id="traceProcess"></div>
            </div>
          </div>
        </div>
      </section>
      <section id="abView" class="hidden">
        <div class="abbar">
          <select id="package">
            <option value="main">v3 vs stim_only</option>
            <option value="secondary">v3 vs episode_trace</option>
          </select>
          <button id="prev">Previous</button>
          <span id="rowStatus" class="muted"></span>
          <button id="next">Next</button>
          <button id="exportCsv">Download filled CSV</button>
        </div>
        <div class="metrics">
          <div class="metric"><div class="name">Completed</div><div id="abCompleted" class="value">0/0</div></div>
          <div class="metric"><div class="name">Win / Tie / Loss</div><div id="abScore" class="value">0 / 0 / 0</div></div>
          <div class="metric"><div class="name">Win rate</div><div id="abWinRate" class="value">0.000</div></div>
        </div>
        <div class="card">
          <label>User input</label>
          <textarea id="abText" readonly></textarea>
          <div class="grid2">
            <div><label>Candidate A</label><textarea id="candA" readonly></textarea></div>
            <div><label>Candidate B</label><textarea id="candB" readonly></textarea></div>
          </div>
          <label>Winner</label>
          <div class="radio-row">
            <label><input type="radio" name="winner" value=""> unset</label>
            <label><input type="radio" name="winner" value="candidate_a"> candidate_a</label>
            <label><input type="radio" name="winner" value="candidate_b"> candidate_b</label>
            <label><input type="radio" name="winner" value="tie"> tie</label>
          </div>
          <label for="confidence">Confidence</label>
          <input id="confidence" type="range" min="1" max="5" value="3" />
          <label for="reason">Reason</label>
          <textarea id="reason"></textarea>
          <button id="saveAB" class="primary">Save</button>
          <span id="abSaved" class="ok"></span>
        </div>
      </section>
    </main>
  </div>
<script>
let messages = [];
let usage = { input_tokens: 0, output_tokens: 0, cost_usd: 0 };
let abIndex = 0;
let abKind = "main";
let currentEvalId = "";

const $ = id => document.getElementById(id);
const esc = text => String(text ?? "").replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[ch]));

function setTab(name) {
  $("tabChat").classList.toggle("active", name === "chat");
  $("tabCompare").classList.toggle("active", name === "compare");
  $("tabAB").classList.toggle("active", name === "ab");
  $("chatView").classList.toggle("hidden", name !== "chat");
  $("compareView").classList.toggle("hidden", name !== "compare");
  $("abView").classList.toggle("hidden", name !== "ab");
  if (name === "ab") loadAB();
}

function renderUsage() {
  const budget = Number($("budget").value || 0);
  const spent = Number(usage.cost_usd || 0);
  $("spent").textContent = "$" + spent.toFixed(4);
  $("left").textContent = "$" + Math.max(0, budget - spent).toFixed(2);
  $("tokens").textContent = `${usage.input_tokens || 0} / ${usage.output_tokens || 0}`;
}

function compact(text, fallback = "none") {
  const value = String(text ?? "").trim();
  return value ? value : fallback;
}

function num(value, digits = 3) {
  const n = Number(value);
  return Number.isFinite(n) ? n.toFixed(digits) : "0.000";
}

function listItems(items, limit = 6) {
  const arr = Array.isArray(items) ? items.filter(Boolean).slice(0, limit) : [];
  if (!arr.length) return "<span class='muted'>none</span>";
  return `<ul class="list">${arr.map(item => `<li>${esc(item)}</li>`).join("")}</ul>`;
}

function chips(items) {
  const arr = Array.isArray(items) ? items.filter(Boolean).slice(0, 12) : [];
  if (!arr.length) return "<span class='muted'>none</span>";
  return `<div class="chips">${arr.map(item => `<span class="chip">${esc(item)}</span>`).join("")}</div>`;
}

function meterRows(summary) {
  if (!summary || typeof summary !== "object") return "<span class='muted'>no style meter</span>";
  const entries = Object.entries(summary)
    .filter(([, value]) => typeof value === "number" && Number.isFinite(value))
    .slice(0, 8);
  if (!entries.length) return "<span class='muted'>no numeric style meter</span>";
  const values = entries.map(([, value]) => Number(value));
  const zeroToOne = values.every(value => value >= 0 && value <= 1);
  return entries.map(([key, value]) => {
    const pct = zeroToOne ? Number(value) * 100 : ((Number(value) + 1) / 2) * 100;
    const width = Math.max(0, Math.min(100, pct));
    return `
      <div class="meter-row">
        <div>${esc(key)}</div>
        <div class="meter-track"><div class="meter-fill" style="width:${width}%"></div></div>
        <div>${num(value, 2)}</div>
      </div>
    `;
  }).join("");
}

function phaseRows(lines) {
  const arr = Array.isArray(lines) ? lines : [];
  const phases = arr.filter(line => /^(초기|중기|후기):/.test(String(line)));
  if (!phases.length) return "<span class='muted'>phase detail unavailable</span>";
  return phases.map(line => {
    const text = String(line);
    const [name, ...rest] = text.split(":");
    return `<div class="phase-row"><div class="phase-name">${esc(name)}</div><div class="phase-body">${esc(rest.join(":").trim())}</div></div>`;
  }).join("");
}

function parseAppraisal(summaryText) {
  const text = String(summaryText || "");
  const found = [];
  for (const key of ["통제 상실", "목표 차단", "소진", "위협", "상실", "분노", "불안", "회복"]) {
    const match = text.match(new RegExp(`${key}\\s*(매우 높음|높음|중간|낮음|매우 낮음)`));
    if (match) found.push(`${key} ${match[1]}`);
  }
  return found;
}

function feltHeadline(record) {
  const tendency = compact(record.appraisal_tendency, "");
  const target = compact(record.appraisal_target, "");
  const appraisal = String(record.appraisal_summary_text || "");
  const tags = Array.isArray(record.style_tags) ? record.style_tags.join(" ") : "";
  const source = `${tendency} ${target} ${appraisal} ${tags}`;
  if (/회복|후퇴|소진|피로|멍|휴식|탈진/.test(source)) {
    return "지금은 더 밀어붙이기보다 물러나 회복하고 싶은 상태로 읽었습니다.";
  }
  if (/분노|따지고|공격|부당|무시/.test(source)) {
    return "부당하게 밀려났고 바로잡고 싶은 분노로 읽었습니다.";
  }
  if (/불안|걱정|예민|위협/.test(source)) {
    return "좋은 일 안에서도 불안과 경계가 남아 있는 상태로 읽었습니다.";
  }
  if (/슬픔|상실|외로움/.test(source)) {
    return "상실감이나 외로움이 앞에 나와 있는 상태로 읽었습니다.";
  }
  return "감정의 방향을 먼저 잡고, 그 방향에 맞춰 답변을 만들었습니다.";
}

function strongestStyle(summary) {
  if (!summary || typeof summary !== "object") return "style meter 없음";
  const entries = Object.entries(summary)
    .filter(([, value]) => typeof value === "number" && Number.isFinite(value))
    .sort((a, b) => Number(b[1]) - Number(a[1]));
  if (!entries.length) return "style meter 없음";
  const [key, value] = entries[0];
  return `${key} ${num(value, 2)}`;
}

function feltState(record) {
  const tendency = compact(record.appraisal_tendency, "");
  const appraisal = String(record.appraisal_summary_text || "");
  const trace = String(record.trace_summary_text || "");
  const tags = Array.isArray(record.style_tags) ? record.style_tags.join(" ") : "";
  const source = `${record.input_text || ""} ${tendency} ${appraisal} ${trace} ${tags}`;

  if (/분노|따지고|공격|부당|무시|목표 차단/.test(source)) {
    return {
      kind: "anger",
      signal: "felt signal: anger / blocked agency",
      quote: "나는 지금 밀려났고, 그냥 넘기고 싶지 않다.",
      body: "핵심 정서는 위로받고 싶은 약함보다, 부당함을 바로잡고 싶은 긴장으로 읽힙니다.",
      impulse: "맞서기",
      restraint: "폭발 직전의 절제",
    };
  }
  if (/불안|걱정|예민|위협|경계/.test(source)) {
    return {
      kind: "anxiety",
      signal: "felt signal: anxiety / vigilance",
      quote: "좋은 일이어도, 아직 안심하면 안 될 것 같다.",
      body: "기쁨보다 먼저 위험을 스캔하는 감각이 앞에 있고, 작은 단서에도 몸이 조여드는 쪽입니다.",
      impulse: "확인하고 대비하기",
      restraint: "성급한 안심 거부",
    };
  }
  if (/회복|후퇴|소진|피로|멍|휴식|탈진|놓아버리고/.test(source)) {
    return {
      kind: "exhaustion",
      signal: "felt signal: exhaustion / withdrawal",
      quote: "더 버티라는 말보다, 지금은 멈출 구실이 필요하다.",
      body: "추진력보다 방전감이 크고, 해결책을 더 얹기 전에 에너지가 빠진 상태로 읽힙니다.",
      impulse: "물러나기",
      restraint: "추가 압박 회피",
    };
  }
  if (/슬픔|상실|외로움|잃/.test(source)) {
    return {
      kind: "grief",
      signal: "felt signal: grief / loss",
      quote: "없어진 자리가 너무 커서, 말이 늦게 따라온다.",
      body: "분석이나 해결보다 상실의 무게가 먼저 있고, 빠른 전환을 거부하는 정서입니다.",
      impulse: "붙잡고 인정하기",
      restraint: "성급한 회복 거부",
    };
  }
  return {
    kind: "recovery",
    signal: "felt signal: mixed affect",
    quote: "감정의 방향을 잡아야 답의 톤도 정해진다.",
    body: "뚜렷한 단일 감정보다는 appraisal, trace, style 신호를 묶어 반응 방향을 정한 상태입니다.",
    impulse: compact(tendency),
    restraint: "과잉 해석 보류",
  };
}

function feltPanel(record) {
  if (!record) return "";
  const state = feltState(record);
  const appraisals = parseAppraisal(record.appraisal_summary_text);
  return `
    <div class="felt-panel ${esc(state.kind)}">
      <div>
        <div class="felt-signal">${esc(state.signal)}</div>
        <div class="felt-quote">${esc(state.quote)}</div>
        <div class="felt-body">${esc(state.body)}</div>
      </div>
      <div class="felt-readout">
        <div class="felt-row"><div class="k">action tendency</div><div class="v">${esc(state.impulse)}</div></div>
        <div class="felt-row"><div class="k">guardrail</div><div class="v">${esc(state.restraint)}</div></div>
        <div class="felt-row"><div class="k">target</div><div class="v">${esc(targetLabel(record.appraisal_target))}</div></div>
        <div class="felt-row"><div class="k">strongest style</div><div class="v">${esc(strongestStyle(record.style_summary))}</div></div>
        <div class="felt-row"><div class="k">appraisal cues</div><div class="v">${esc(appraisals.slice(0, 3).join(" · ") || compact(record.appraisal_summary_text, "none"))}</div></div>
      </div>
    </div>
  `;
}

function targetLabel(value) {
  const raw = String(value || "").trim();
  const map = {
    situation: "상황/업무 맥락",
    self: "자기 자신",
    other: "타인",
    relationship: "관계",
    future: "미래"
  };
  return map[raw] || raw || "불명확";
}

function vectorBars(values) {
  const arr = Array.isArray(values) ? values.slice(0, 40).map(Number).filter(Number.isFinite) : [];
  if (!arr.length) return "<span class='muted'>no vector</span>";
  const maxAbs = Math.max(...arr.map(v => Math.abs(v)), 0.0001);
  return `<div class="vector">${arr.map(v => {
    const h = Math.max(4, Math.round((Math.abs(v) / maxAbs) * 34));
    const color = v >= 0 ? "#45b47b" : "#e36b6b";
    const margin = v >= 0 ? `margin-bottom:${34 - h}px` : `margin-top:${34 - h}px`;
    return `<span class="vbar-wrap" title="${num(v)}"><span class="vbar" style="height:${h}px;background:${color};${margin}"></span></span>`;
  }).join("")}</div>`;
}

function processHtml(record) {
  if (!record) return "";
  const usage = record.llm_usage || {};
  const appraisals = parseAppraisal(record.appraisal_summary_text);
  const validation = Array.isArray(record.response_validation_errors) && record.response_validation_errors.length
    ? listItems(record.response_validation_errors)
    : "<span class='ok'>passed</span>";
  return `
    <div class="process">
      <h3>Internal Process: EmoNet이 답변을 만들기 전 읽은 것</h3>
      <div class="felt">
        <div class="felt-label">EmoNet felt this as</div>
        <div class="felt-main">${esc(feltHeadline(record))}</div>
        <div class="felt-sub">${esc(compact(record.appraisal_summary_text, "appraisal unavailable"))}</div>
        <div class="felt-grid">
          <div class="felt-chip"><div class="k">정서 방향</div><div class="v">${esc(compact(record.appraisal_tendency))}</div></div>
          <div class="felt-chip"><div class="k">초점 대상</div><div class="v">${esc(targetLabel(record.appraisal_target))}</div></div>
          <div class="felt-chip"><div class="k">핵심 단서</div><div class="v">${esc(appraisals.slice(0, 3).join(" · ") || compact(record.trace_summary_text, "trace"))}</div></div>
        </div>
      </div>
      <div class="steps">
        <div class="step"><div class="num">01</div><div class="title">Input</div><div class="desc">${esc(compact(record.input_text))}</div></div>
        <div class="step"><div class="num">02</div><div class="title">Branch</div><div class="desc">${esc(record.dominant_branch_len || 0)} branch nodes · ${esc(record.ticks_run || 0)} ticks</div></div>
        <div class="step"><div class="num">03</div><div class="title">Trace</div><div class="desc">${esc(compact(record.trace_summary_text, "trace summarized"))}</div></div>
        <div class="step"><div class="num">04</div><div class="title">Conditioning</div><div class="desc">${esc(record.conditioning_mode)} · ${esc(record.style_profile)}</div></div>
        <div class="step"><div class="num">05</div><div class="title">Claude</div><div class="desc">${esc(record.llm_model_name)} · retries ${esc(record.response_retry_count || 0)}</div></div>
      </div>
      <div class="insight">
        <div class="insight-card"><div class="name">읽힌 정서 방향</div><div class="value">${esc(compact(record.appraisal_tendency))}</div></div>
        <div class="insight-card"><div class="name">초점 대상</div><div class="value">${esc(compact(record.appraisal_target))}</div></div>
        <div class="insight-card"><div class="name">동역학 상태</div><div class="value">${esc(compact(record.termination_reason))}</div></div>
        <div class="insight-card"><div class="name">응답 검증</div><div class="value">${Array.isArray(record.response_validation_errors) && record.response_validation_errors.length ? "retry noted" : "passed"}</div></div>
      </div>
      <div class="detail-grid">
        <div class="detail">
          <div class="detail-title">Runtime</div>
          <div class="kv">
            <div>provider</div><div>${esc(record.llm_provider)}</div>
            <div>model</div><div>${esc(record.llm_model_name)}</div>
            <div>style</div><div>${esc(record.style_profile)}</div>
            <div>mode</div><div>${esc(record.conditioning_mode)}</div>
            <div>termination</div><div>${esc(compact(record.termination_reason))}</div>
          </div>
        </div>
        <div class="detail">
          <div class="detail-title">Usage</div>
          <div class="kv">
            <div>input tokens</div><div>${esc(usage.input_tokens || 0)}</div>
            <div>output tokens</div><div>${esc(usage.output_tokens || 0)}</div>
            <div>estimated cost</div><div>$${num(((usage.input_tokens || 0) * 3 + (usage.output_tokens || 0) * 15) / 1000000, 4)}</div>
            <div>validation</div><div>${validation}</div>
          </div>
        </div>
        <div class="detail">
          <div class="detail-title">Style Profile</div>
          ${chips(record.style_tags)}
          ${meterRows(record.style_summary)}
          <div class="pre">${esc(compact(record.style_summary_text))}</div>
        </div>
        <div class="detail">
          <div class="detail-title">Expression Cues</div>
          <div class="pre">${esc(compact(record.expression_cues_text))}</div>
        </div>
        <div class="detail">
          <div class="detail-title">Trace Phases</div>
          ${phaseRows(record.trace_lines)}
        </div>
        <div class="detail">
          <div class="detail-title">Appraisal</div>
          <div class="kv">
            <div>target</div><div>${esc(compact(record.appraisal_target))}</div>
            <div>tendency</div><div>${esc(compact(record.appraisal_tendency))}</div>
          </div>
          <div class="pre">${esc(compact(record.appraisal_summary_text))}</div>
        </div>
        <div class="detail">
          <div class="detail-title">Stimulus Vector</div>
          ${vectorBars(record.stim_vec)}
        </div>
        <div class="detail">
          <div class="detail-title">Latent z</div>
          ${vectorBars(record.z)}
        </div>
      </div>
      <details>
        <summary>Raw trace lines</summary>
        ${listItems(record.trace_lines, 12)}
      </details>
    </div>
  `;
}

function renderChat() {
  $("chatlog").innerHTML = messages.map(m => {
    const bubble = `<div class="msg ${esc(m.role)}">${esc(m.content)}</div>`;
    return m.role === "assistant" ? bubble + feltPanel(m.record) + processHtml(m.record) : bubble;
  }).join("");
  renderUsage();
}

async function api(path, body) {
  const res = await fetch(path, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(body || {})
  });
  const payload = await res.json();
  if (!res.ok) throw new Error(payload.error || `HTTP ${res.status}`);
  return payload;
}

async function sendMessage(text) {
  const prompt = String(text || "").trim();
  if (!prompt) return;
  $("chatError").classList.add("hidden");
  $("send").disabled = true;
  messages.push({role: "user", content: prompt});
  renderChat();
  $("liveProcess").classList.remove("hidden");
  try {
    const payload = await api("/api/chat", { message: prompt, api_key: $("apiKey").value });
    messages = payload.messages || messages;
    usage = payload.usage || usage;
  } catch (err) {
    $("chatError").textContent = err.message;
    $("chatError").classList.remove("hidden");
  } finally {
    $("send").disabled = false;
    $("liveProcess").classList.add("hidden");
    $("message").value = "";
    renderChat();
  }
}

async function runCompare(text) {
  const prompt = String(text || "").trim();
  if (!prompt) return;
  $("compareError").classList.add("hidden");
  $("compareSend").disabled = true;
  $("compareLoading").classList.remove("hidden");
  $("compareResult").classList.add("hidden");
  $("stimOutput").textContent = "";
  $("traceOutput").textContent = "";
  $("traceProcess").innerHTML = "";
  try {
    const payload = await api("/api/compare", { message: prompt, api_key: $("apiKey").value });
    const stim = payload.stim_only || {};
    const trace = payload.trace || {};
    const compareUsage = payload.usage || {};
    usage = payload.session_usage || usage;
    $("stimOutput").textContent = stim.assistant_text || "";
    $("traceOutput").textContent = trace.assistant_text || "";
    $("traceProcess").innerHTML = feltPanel(trace.record) + processHtml(trace.record);
    $("compareSpent").textContent = "$" + Number(compareUsage.cost_usd || 0).toFixed(4);
    $("compareTokens").textContent = `${compareUsage.input_tokens || 0} / ${compareUsage.output_tokens || 0}`;
    $("compareResult").classList.remove("hidden");
    renderUsage();
  } catch (err) {
    $("compareError").textContent = err.message;
    $("compareError").classList.remove("hidden");
  } finally {
    $("compareSend").disabled = false;
    $("compareLoading").classList.add("hidden");
  }
}

async function loadAB() {
  const res = await fetch(`/api/ab?kind=${encodeURIComponent(abKind)}&index=${abIndex}`);
  const payload = await res.json();
  if (!res.ok) return;
  abIndex = payload.index;
  const row = payload.row;
  currentEvalId = row.eval_id;
  $("rowStatus").textContent = `${abIndex + 1}/${payload.summary.total}`;
  $("abCompleted").textContent = `${payload.summary.completed}/${payload.summary.total}`;
  $("abScore").textContent = `${payload.summary.wins} / ${payload.summary.ties} / ${payload.summary.losses}`;
  $("abWinRate").textContent = Number(payload.summary.win_rate || 0).toFixed(3);
  $("abText").value = row.text || "";
  $("candA").value = row.candidate_a || "";
  $("candB").value = row.candidate_b || "";
  document.querySelectorAll("input[name=winner]").forEach(el => { el.checked = el.value === (row.winner || ""); });
  $("confidence").value = row.confidence || 3;
  $("reason").value = row.reason || "";
  $("prev").disabled = abIndex <= 0;
  $("next").disabled = abIndex >= payload.summary.total - 1;
  $("abSaved").textContent = "";
}

async function saveAB() {
  const winner = document.querySelector("input[name=winner]:checked")?.value || "";
  await api("/api/ab/save", {
    kind: abKind,
    eval_id: currentEvalId,
    winner,
    confidence: $("confidence").value,
    reason: $("reason").value
  });
  $("abSaved").textContent = " saved";
  await loadAB();
}

$("tabChat").onclick = () => setTab("chat");
$("tabCompare").onclick = () => setTab("compare");
$("tabAB").onclick = () => setTab("ab");
$("send").onclick = () => sendMessage($("message").value);
$("compareSend").onclick = () => runCompare($("compareMessage").value);
$("message").addEventListener("keydown", ev => {
  if (ev.key === "Enter" && (ev.ctrlKey || ev.metaKey)) sendMessage($("message").value);
});
$("compareMessage").addEventListener("keydown", ev => {
  if (ev.key === "Enter" && (ev.ctrlKey || ev.metaKey)) runCompare($("compareMessage").value);
});
document.querySelectorAll("[data-example]").forEach(btn => btn.onclick = () => sendMessage(btn.dataset.example));
document.querySelectorAll("[data-compare]").forEach(btn => {
  btn.onclick = () => {
    $("compareMessage").value = btn.dataset.compare;
    runCompare(btn.dataset.compare);
  };
});
$("clearChat").onclick = () => { messages = []; renderChat(); fetch("/api/chat/clear", {method: "POST"}); };
$("resetUsage").onclick = () => { usage = { input_tokens: 0, output_tokens: 0, cost_usd: 0 }; renderUsage(); fetch("/api/usage/reset", {method: "POST"}); };
$("budget").oninput = renderUsage;
$("package").onchange = () => { abKind = $("package").value; abIndex = 0; loadAB(); };
$("prev").onclick = () => { abIndex -= 1; loadAB(); };
$("next").onclick = () => { abIndex += 1; loadAB(); };
$("saveAB").onclick = saveAB;
$("exportCsv").onclick = () => { window.location.href = `/api/ab/export?kind=${encodeURIComponent(abKind)}`; };
renderUsage();
</script>
</body>
</html>
"""


class LocalGuiHandler(BaseHTTPRequestHandler):
    server_version = "EmoNetLocalGUI/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        return

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: Any) -> None:
        self._send(status, _json_bytes(payload), "application/json; charset=utf-8")

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"error": message})

    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self._send(HTTPStatus.OK, APP_HTML.encode("utf-8"), "text/html; charset=utf-8")
            elif parsed.path == "/favicon.ico":
                self._send(HTTPStatus.NO_CONTENT, b"", "image/x-icon")
            elif parsed.path == "/api/ab":
                query = parse_qs(parsed.query)
                kind = query.get("kind", ["main"])[0]
                index = int(query.get("index", ["0"])[0])
                self._json(HTTPStatus.OK, _row_payload(kind, index))
            elif parsed.path == "/api/ab/export":
                query = parse_qs(parsed.query)
                kind = query.get("kind", ["main"])[0]
                body = _export_csv(kind)
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", f'attachment; filename="emonet_{kind}_filled.csv"')
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif parsed.path == "/api/status":
                self._json(HTTPStatus.OK, {"model": CLAUDE_MODEL, "runtime": asdict(ChatRuntimeConfig())})
            else:
                self._error(HTTPStatus.NOT_FOUND, "not found")
        except Exception as exc:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self) -> None:
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/api/chat":
                payload = _read_json(self)
                api_key = str(payload.get("api_key") or os.environ.get("ANTHROPIC_API_KEY") or "").strip()
                message = str(payload.get("message") or "").strip()
                if not api_key:
                    self._error(HTTPStatus.BAD_REQUEST, "Claude API key가 필요합니다. 왼쪽에 입력하거나 ANTHROPIC_API_KEY를 설정하세요.")
                    return
                if not message:
                    self._error(HTTPStatus.BAD_REQUEST, "message is empty")
                    return
                with _history_lock:
                    history = list(_messages)
                    result = generate_chat_turn(
                        runtime=_runtime_cached(),
                        generation_config=_chat_config(api_key),
                        input_text=message,
                        history=history,
                    )
                    _messages.append({"role": "user", "content": message})
                    _messages.append({"role": "assistant", "content": result.assistant_text, "record": result.record})
                    meta_usage = dict(result.record.get("llm_usage", {}))
                    input_tokens = int(meta_usage.get("input_tokens", 0) or 0)
                    output_tokens = int(meta_usage.get("output_tokens", 0) or 0)
                    _usage["input_tokens"] += input_tokens
                    _usage["output_tokens"] += output_tokens
                    _usage["cost_usd"] += _estimate_cost(input_tokens, output_tokens)
                    response = {"messages": list(_messages), "usage": dict(_usage), "record": result.record}
                self._json(HTTPStatus.OK, response)
            elif parsed.path == "/api/compare":
                payload = _read_json(self)
                api_key = str(payload.get("api_key") or os.environ.get("ANTHROPIC_API_KEY") or "").strip()
                message = str(payload.get("message") or "").strip()
                if not api_key:
                    self._error(HTTPStatus.BAD_REQUEST, "Claude API key가 필요합니다. 왼쪽에 입력하거나 ANTHROPIC_API_KEY를 설정하세요.")
                    return
                if not message:
                    self._error(HTTPStatus.BAD_REQUEST, "message is empty")
                    return
                stim_text, _stim_raw, stim_meta = request_plain_text_response(
                    base_url=CLAUDE_BASE_URL,
                    model_name=CLAUDE_MODEL,
                    prompt=_stim_only_prompt(message),
                    temperature=0.45,
                    max_tokens=600,
                    timeout_sec=180,
                    max_retries=1,
                    validator=validate_plain_response_text,
                    retry_instruction="직전 응답은 부자연스럽거나 미완성이다. 반복 없이 자연스러운 한국어 평문으로 다시 답하라.",
                    system_prompt="Return a plain Korean response only. Do not return JSON.",
                    api_key=api_key,
                    provider="anthropic",
                )
                trace_result = generate_chat_turn(
                    runtime=_runtime_cached(),
                    generation_config=_chat_config(api_key, conditioning_mode="hybrid_trace"),
                    input_text=message,
                    history=[],
                )
                trace_text, _trace_raw, trace_meta = request_plain_text_response(
                    base_url=CLAUDE_BASE_URL,
                    model_name=CLAUDE_MODEL,
                    prompt=_live_compare_trace_prompt(str(trace_result.record.get("generation_prompt", "")), message),
                    temperature=0.35,
                    max_tokens=600,
                    timeout_sec=180,
                    max_retries=1,
                    validator=validate_plain_response_text,
                    retry_instruction="직전 응답은 trace 대비가 약하거나 일반 조언으로 흐른다. 위험/대상/행동 경향을 더 분명히 하라.",
                    system_prompt="Return a plain Korean response only. Do not return JSON.",
                    api_key=api_key,
                    provider="anthropic",
                )
                trace_result = _replace_trace_response(trace_result, trace_text, trace_meta, message)
                response = _compare_payload(
                    input_text=message,
                    stim_text=stim_text,
                    trace_result=trace_result,
                    stim_meta=stim_meta,
                )
                compare_usage = dict(response["usage"])
                with _history_lock:
                    _add_usage(int(compare_usage.get("input_tokens", 0)), int(compare_usage.get("output_tokens", 0)))
                    response["session_usage"] = dict(_usage)
                self._json(HTTPStatus.OK, response)
            elif parsed.path == "/api/chat/clear":
                with _history_lock:
                    _messages.clear()
                self._json(HTTPStatus.OK, {"ok": True})
            elif parsed.path == "/api/usage/reset":
                with _history_lock:
                    _usage.update({"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})
                self._json(HTTPStatus.OK, {"ok": True, "usage": dict(_usage)})
            elif parsed.path == "/api/ab/save":
                payload = _read_json(self)
                kind = str(payload.get("kind") or "main")
                eval_id = str(payload.get("eval_id") or "").strip()
                if not eval_id:
                    self._error(HTTPStatus.BAD_REQUEST, "eval_id is required")
                    return
                _, _, progress_path = _package_paths(kind)
                progress = _load_progress(progress_path)
                progress[eval_id] = {
                    "winner": str(payload.get("winner") or ""),
                    "confidence": str(payload.get("confidence") or ""),
                    "reason": str(payload.get("reason") or ""),
                }
                _save_progress(progress_path, progress)
                self._json(HTTPStatus.OK, {"ok": True})
            else:
                self._error(HTTPStatus.NOT_FOUND, "not found")
        except Exception as exc:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), LocalGuiHandler)
    print(f"EmoNet v4 local GUI: http://{HOST}:{PORT}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
