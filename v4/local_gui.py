from __future__ import annotations

import csv
import html
import json
import math
import os
import threading
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd

from emonet.chat_service import ChatGenerationConfig, ChatRuntimeConfig, build_chat_runtime, generate_chat_turn


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


def _chat_config(api_key: str) -> ChatGenerationConfig:
    return ChatGenerationConfig(
        provider="anthropic",
        base_url=CLAUDE_BASE_URL,
        model_name=CLAUDE_MODEL,
        api_key=api_key,
        style_profile="extended40",
        conditioning_mode="hybrid_trace",
        response_temperature=0.45,
        response_max_retries=1,
        max_tokens=600,
        timeout_sec=180,
        history_turns=4,
    )


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
      --bg: #0f1318;
      --panel: #171d24;
      --panel2: #202833;
      --line: #303b49;
      --text: #eef3f8;
      --muted: #aab6c4;
      --green: #6fd08a;
      --red: #ff7171;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--text); font: 15px/1.5 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    button, input, textarea, select { font: inherit; }
    .app { display: grid; grid-template-columns: 280px 1fr; min-height: 100vh; }
    aside { border-right: 1px solid var(--line); background: var(--panel); padding: 18px; }
    main { padding: 22px; max-width: 1180px; width: 100%; }
    h1, h2, h3 { margin: 0; letter-spacing: 0; }
    h1 { font-size: 30px; line-height: 1.15; }
    h2 { font-size: 21px; margin-bottom: 12px; }
    label { display: block; color: var(--muted); font-size: 13px; margin: 16px 0 6px; }
    input, textarea, select {
      width: 100%; background: #0f141a; color: var(--text); border: 1px solid var(--line);
      border-radius: 7px; padding: 10px 11px; outline: none;
    }
    textarea { resize: vertical; min-height: 94px; }
    button {
      border: 1px solid var(--line); background: var(--panel2); color: var(--text);
      border-radius: 7px; padding: 10px 12px; cursor: pointer;
    }
    button:hover { border-color: #536274; }
    button.primary { background: #1c5a39; border-color: #2b7b50; }
    button:disabled { opacity: .45; cursor: not-allowed; }
    .top { display: flex; justify-content: space-between; gap: 16px; align-items: flex-start; border-bottom: 1px solid var(--line); padding-bottom: 16px; }
    .sub { color: var(--muted); margin-top: 8px; max-width: 780px; }
    .tabs { display: flex; gap: 8px; margin: 18px 0; }
    .tabs button.active { background: #123421; border-color: #2b7b50; color: #dcf8e3; }
    .pills { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 13px; }
    .pill { border: 1px solid #285238; background: #10251a; color: #d7f7dd; border-radius: 999px; padding: 6px 10px; font-size: 13px; }
    .metrics { display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 12px; margin: 18px 0; }
    .metric, .card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }
    .metric .name { color: var(--muted); font-size: 13px; }
    .metric .value { font-size: 24px; font-weight: 700; margin-top: 4px; }
    .examples { display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 10px; margin: 14px 0; }
    .chatlog { display: grid; gap: 10px; margin: 18px 0; }
    .msg { border: 1px solid var(--line); border-radius: 8px; padding: 13px 14px; white-space: pre-wrap; }
    .msg.user { background: #1d4a3b; }
    .msg.assistant { background: var(--panel2); }
    .felt-panel {
      border: 1px solid #375f4b; background: #111a20; border-radius: 8px; padding: 14px;
      margin-top: -2px; display: grid; grid-template-columns: minmax(0, 1.25fr) minmax(260px, .75fr);
      gap: 12px; align-items: stretch;
    }
    .felt-panel.anger { border-color: #7f4d44; background: #201715; }
    .felt-panel.anxiety { border-color: #75613a; background: #1f1c14; }
    .felt-panel.exhaustion { border-color: #49606b; background: #141b20; }
    .felt-panel.grief { border-color: #4f5b7b; background: #151823; }
    .felt-panel.recovery { border-color: #41664d; background: #121d17; }
    .felt-signal { color: #9bd9ad; font-size: 12px; font-weight: 800; text-transform: uppercase; }
    .felt-panel.anger .felt-signal { color: #ffb29e; }
    .felt-panel.anxiety .felt-signal { color: #f1cf80; }
    .felt-panel.exhaustion .felt-signal { color: #9bc7dc; }
    .felt-panel.grief .felt-signal { color: #aebeff; }
    .felt-quote { font-size: 24px; line-height: 1.26; font-weight: 800; margin-top: 5px; }
    .felt-body { color: #cdd9e4; margin-top: 9px; }
    .felt-readout { display: grid; grid-template-columns: 1fr; gap: 8px; }
    .felt-row { border: 1px solid rgba(170, 182, 196, .23); background: rgba(8, 12, 17, .38); border-radius: 7px; padding: 9px; }
    .felt-row .k { color: var(--muted); font-size: 12px; }
    .felt-row .v { margin-top: 2px; font-weight: 750; overflow-wrap: anywhere; }
    .process {
      border: 1px solid #314253; background: #121922; border-radius: 8px; padding: 13px;
      margin-top: -4px;
    }
    .process h3 { font-size: 16px; margin-bottom: 10px; }
    .felt {
      border: 1px solid #315d43; background: linear-gradient(135deg, #12281d, #182330);
      border-radius: 8px; padding: 14px; margin-bottom: 12px;
    }
    .felt-label { color: #9bd9ad; font-size: 12px; font-weight: 700; text-transform: uppercase; }
    .felt-main { font-size: 22px; font-weight: 750; margin-top: 4px; line-height: 1.28; }
    .felt-sub { color: #c4d4df; margin-top: 8px; }
    .felt-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; margin-top: 12px; }
    .felt-chip { border: 1px solid #355167; background: rgba(10, 15, 21, .45); border-radius: 7px; padding: 8px; }
    .felt-chip .k { color: var(--muted); font-size: 12px; }
    .felt-chip .v { margin-top: 2px; font-weight: 700; overflow-wrap: anywhere; }
    .steps { display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 8px; margin-bottom: 12px; }
    .step { border: 1px solid var(--line); background: #0f151c; border-radius: 7px; padding: 9px; min-height: 70px; }
    .step .num { color: var(--green); font-weight: 700; font-size: 12px; }
    .step .title { font-weight: 700; margin-top: 2px; }
    .step .desc { color: var(--muted); font-size: 12px; margin-top: 3px; }
    .detail-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 10px; }
    .detail { border: 1px solid var(--line); background: #0f151c; border-radius: 7px; padding: 10px; }
    .detail-title { color: var(--muted); font-size: 12px; margin-bottom: 6px; text-transform: uppercase; }
    .insight {
      display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; margin: 10px 0 12px;
    }
    .insight-card { border: 1px solid #314253; background: #15202b; border-radius: 7px; padding: 10px; }
    .insight-card .name { color: var(--muted); font-size: 12px; }
    .insight-card .value { font-size: 18px; font-weight: 700; margin-top: 3px; overflow-wrap: anywhere; }
    .kv { display: grid; grid-template-columns: 150px 1fr; gap: 5px 10px; font-size: 13px; }
    .kv div:nth-child(odd) { color: var(--muted); }
    .list { margin: 0; padding-left: 18px; }
    .list li { margin: 3px 0; }
    .chips { display: flex; flex-wrap: wrap; gap: 6px; }
    .chip { border: 1px solid #3a5064; background: #182230; color: #d9e6f2; border-radius: 999px; padding: 4px 8px; font-size: 12px; }
    .meter-row { display: grid; grid-template-columns: 130px 1fr 52px; gap: 8px; align-items: center; margin: 7px 0; font-size: 12px; }
    .meter-track { height: 8px; background: #0a0f15; border: 1px solid var(--line); border-radius: 99px; overflow: hidden; }
    .meter-fill { height: 100%; background: linear-gradient(90deg, #6fd08a, #e2c36f); border-radius: 99px; }
    .phase-row { display: grid; grid-template-columns: 70px 1fr; gap: 8px; margin: 8px 0; align-items: start; }
    .phase-name { color: var(--green); font-weight: 700; font-size: 12px; padding-top: 2px; }
    .phase-body { color: #d8e2ec; font-size: 13px; }
    .vector {
      position: relative; display: flex; align-items: center; gap: 4px; min-height: 92px;
      border: 1px solid var(--line); background: #0b1016; border-radius: 6px; padding: 10px 8px;
      overflow-x: auto;
    }
    .vector::before { content: ""; position: absolute; left: 8px; right: 8px; top: 50%; border-top: 1px solid #344253; }
    .vbar-wrap { position: relative; z-index: 1; width: 13px; height: 68px; display: flex; align-items: center; justify-content: center; flex: 0 0 auto; }
    .vbar { width: 9px; border-radius: 5px; opacity: .9; }
    .pre {
      max-height: 170px; overflow: auto; background: #0b1016; border: 1px solid var(--line);
      border-radius: 6px; padding: 8px; white-space: pre-wrap; font-size: 12px; color: #d8e2ec;
    }
    details { border: 1px solid var(--line); border-radius: 7px; padding: 9px 10px; background: #0f151c; }
    summary { cursor: pointer; color: var(--muted); }
    .loading .step { opacity: .72; }
    .loading .step:nth-child(1) { border-color: #6fd08a; }
    .composer { display: grid; grid-template-columns: 1fr auto; gap: 10px; align-items: end; position: sticky; bottom: 0; background: linear-gradient(180deg, transparent, var(--bg) 30%); padding-top: 20px; }
    .hidden { display: none !important; }
    .error { background: #3a1f24; color: #ffb8b8; border: 1px solid #75404a; border-radius: 8px; padding: 12px; margin: 12px 0; white-space: pre-wrap; }
    .ok { color: var(--green); }
    .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
    .abbar { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-bottom: 12px; }
    .radio-row { display: flex; gap: 10px; flex-wrap: wrap; }
    .radio-row label { display: inline-flex; gap: 7px; align-items: center; margin: 0; color: var(--text); }
    .radio-row input { width: auto; }
    .muted { color: var(--muted); }
    @media (max-width: 840px) {
      .app { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      .metrics, .examples, .grid2, .steps, .detail-grid, .insight, .felt-grid, .felt-panel { grid-template-columns: 1fr; }
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
          <div class="sub">Streamlit을 제거한 안정판입니다. Claude Chat과 Human A/B만 제공합니다.</div>
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
  $("tabAB").classList.toggle("active", name === "ab");
  $("chatView").classList.toggle("hidden", name !== "chat");
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
    const color = v >= 0 ? "#6fd08a" : "#ff8a8a";
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
$("tabAB").onclick = () => setTab("ab");
$("send").onclick = () => sendMessage($("message").value);
$("message").addEventListener("keydown", ev => {
  if (ev.key === "Enter" && (ev.ctrlKey || ev.metaKey)) sendMessage($("message").value);
});
document.querySelectorAll("[data-example]").forEach(btn => btn.onclick = () => sendMessage(btn.dataset.example));
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
