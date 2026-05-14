from __future__ import annotations

import html
import json
import os
import threading
import traceback
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from emonet.character import CharacterSessionState, load_character_card
from emonet.chat_service import ChatGenerationConfig, ChatRuntimeConfig, build_chat_runtime, generate_chat_turn
from emonet.legacy_cli import validate_plain_response_text
from emonet.llm_api import request_plain_text_response


HOST = "127.0.0.1"
PORT = 8788
CLAUDE_MODEL = os.environ.get("EMONET_CLAUDE_MODEL", "claude-haiku-4-5-20251001")
CLAUDE_BASE_URL = os.environ.get("EMONET_CLAUDE_BASE_URL", "https://api.anthropic.com")
CLAUDE_INPUT_PRICE = float(os.environ.get("EMONET_CLAUDE_INPUT_PRICE", "1.0"))
CLAUDE_OUTPUT_PRICE = float(os.environ.get("EMONET_CLAUDE_OUTPUT_PRICE", "5.0"))

_runtime_lock = threading.Lock()
_runtime: Any | None = None
_history_lock = threading.Lock()
_messages: list[dict[str, Any]] = []
_character_session = CharacterSessionState()
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
    payload = json.loads(handler.rfile.read(length).decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JSON body must be an object")
    return payload


def _estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * CLAUDE_INPUT_PRICE + output_tokens * CLAUDE_OUTPUT_PRICE) / 1_000_000.0


def _chat_config(api_key: str, *, affect_input_mode: str = "encoder") -> ChatGenerationConfig:
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
        history_turns=6,
        affect_input_mode=affect_input_mode,
    )


def _append_usage_from_record(record: dict[str, Any]) -> None:
    meta_usage = dict(record.get("llm_usage", {}))
    input_tokens = int(meta_usage.get("input_tokens", 0) or 0)
    output_tokens = int(meta_usage.get("output_tokens", 0) or 0)
    _usage["input_tokens"] += input_tokens
    _usage["output_tokens"] += output_tokens
    _usage["cost_usd"] += _estimate_cost(input_tokens, output_tokens)


def _recent_dialogue_for_ai(max_messages: int = 12) -> str:
    lines: list[str] = []
    for message in _messages[-max_messages:]:
        role = str(message.get("role", "")).upper()
        content = " ".join(str(message.get("content", "")).split())
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines) if lines else "(no dialogue yet)"


def _generate_ai_user_message(*, api_key: str, scenario: str, turn_index: int) -> tuple[str, dict[str, Any]]:
    card = load_character_card()
    prompt = "\n".join(
        [
            "[ROLE]",
            "너는 Ruca와 대화하는 테스트용 상대 AI다. 사용자의 역할을 맡아 한국어로 한 번만 말한다.",
            "",
            "[TEST_GOAL]",
            "Ruca가 자기 내부 감정 trace를 대화로 잘 변환하는지 테스트한다. 과장된 지문이나 분석 설명을 하지 말고, 자연스러운 대화 발화만 만든다.",
            "",
            "[SCENARIO]",
            scenario.strip() or "관계가 흔들리는 밤, 떠남과 붙잡음이 섞인 대화.",
            "",
            "[RUCA_PROFILE]",
            f"name: {card.name}",
            f"persona: {card.persona}",
            "",
            "[RECENT_DIALOGUE]",
            _recent_dialogue_for_ai(),
            "",
            "[TURN]",
            str(turn_index),
            "",
            "[INSTRUCTIONS]",
            "- USER 역할의 한 발화만 출력한다.",
            "- 1~3문장 이내.",
            "- JSON, bullet, markdown, [ACTION]을 쓰지 않는다.",
            "- 테스트를 위해 매 턴 조금씩 감정 압력을 변화시킨다.",
            "- 같은 말을 반복하지 않는다.",
        ]
    )
    text, _raw, meta = request_plain_text_response(
        base_url=CLAUDE_BASE_URL,
        model_name=CLAUDE_MODEL,
        prompt=prompt,
        temperature=0.72,
        max_tokens=220,
        timeout_sec=120,
        max_retries=1,
        validator=validate_plain_response_text,
        retry_instruction="USER 역할의 자연스러운 한국어 발화만 다시 출력하라.",
        system_prompt="Return one plain Korean user utterance only.",
        api_key=api_key,
        provider="anthropic",
    )
    usage = dict(meta.get("usage", {}))
    return text, usage


def _state_payload() -> dict[str, Any]:
    card = load_character_card()
    runtime_config = asdict(ChatRuntimeConfig())
    return {
        "character": card.to_record(),
        "session": _character_session.to_record(),
        "messages": list(_messages),
        "usage": dict(_usage),
        "api": {
            "provider": "anthropic",
            "base_url": CLAUDE_BASE_URL,
            "model": CLAUDE_MODEL,
            "input_price_per_mtok": CLAUDE_INPUT_PRICE,
            "output_price_per_mtok": CLAUDE_OUTPUT_PRICE,
        },
        "runtime": {key: str(value) for key, value in runtime_config.items()},
    }


APP_HTML = r"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>EmoNet v5 Character Chat</title>
  <style>
    :root {
      --bg: #101315;
      --panel: #171b1f;
      --panel2: #20262b;
      --line: #313a42;
      --text: #eef2f4;
      --muted: #aeb8c0;
      --accent: #75c59a;
      --warn: #f1c36d;
      --bad: #ff8989;
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--text); font: 15px/1.5 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    button, input, textarea { font: inherit; }
    .app { display: grid; grid-template-columns: 330px minmax(0, 1fr); min-height: 100vh; }
    aside { border-right: 1px solid var(--line); background: var(--panel); padding: 18px; overflow: auto; }
    main { padding: 22px; max-width: 1120px; width: 100%; }
    h1, h2, h3 { margin: 0; letter-spacing: 0; }
    h1 { font-size: 28px; line-height: 1.15; }
    h2 { font-size: 18px; margin: 20px 0 8px; }
    h3 { font-size: 15px; margin-bottom: 8px; }
    label { display: block; color: var(--muted); font-size: 13px; margin: 14px 0 6px; }
    input, textarea {
      width: 100%; background: #0e1215; color: var(--text); border: 1px solid var(--line);
      border-radius: 7px; padding: 10px 11px; outline: none;
    }
    textarea { resize: vertical; min-height: 92px; }
    button {
      border: 1px solid var(--line); background: var(--panel2); color: var(--text);
      border-radius: 7px; padding: 10px 12px; cursor: pointer;
    }
    button.primary { background: #1f6040; border-color: #33845a; }
    button:disabled { opacity: .45; cursor: not-allowed; }
    .checkrow { display: flex; align-items: center; gap: 9px; margin: 12px 0; color: var(--muted); font-size: 13px; }
    .checkrow input { width: auto; }
    .sub { color: var(--muted); margin-top: 8px; max-width: 780px; }
    .card { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; margin: 12px 0; }
    .character-name { font-size: 26px; font-weight: 800; }
    .small { color: var(--muted); font-size: 13px; }
    .list { margin: 0; padding-left: 18px; }
    .list li { margin: 4px 0; }
    .metrics { display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 12px; margin: 18px 0; }
    .metric { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }
    .metric .name { color: var(--muted); font-size: 13px; }
    .metric .value { font-size: 22px; font-weight: 750; margin-top: 4px; }
    .chatlog { display: grid; gap: 10px; margin: 18px 0; }
    .msg { border: 1px solid var(--line); border-radius: 8px; padding: 13px 14px; white-space: pre-wrap; }
    .msg.user { background: #223b32; }
    .msg.assistant { background: var(--panel2); }
    .trace { border: 1px solid #385544; background: #121a17; border-radius: 8px; padding: 12px; margin-top: -4px; }
    .trace-title { color: var(--accent); font-size: 12px; font-weight: 800; text-transform: uppercase; }
    .trace-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; margin-top: 9px; }
    .trace-cell { border: 1px solid rgba(174,184,192,.25); border-radius: 7px; padding: 8px; background: rgba(5,8,10,.3); }
    .trace-cell .k { color: var(--muted); font-size: 12px; }
    .trace-cell .v { font-weight: 720; overflow-wrap: anywhere; }
    .process-panel { border: 1px solid #40515f; background: #12181d; border-radius: 8px; padding: 14px; margin: 14px 0; }
    .process-header { display: flex; align-items: baseline; justify-content: space-between; gap: 10px; margin-bottom: 10px; }
    .process-title { font-size: 17px; font-weight: 800; }
    .process-steps { display: grid; gap: 10px; }
    .process-step { border: 1px solid rgba(174,184,192,.22); background: rgba(5,8,10,.25); border-radius: 7px; padding: 10px; }
    .process-step .step-name { color: var(--accent); font-size: 12px; font-weight: 800; text-transform: uppercase; }
    .process-step pre { margin: 8px 0 0; white-space: pre-wrap; overflow-wrap: anywhere; color: var(--text); font-size: 12px; line-height: 1.45; }
    .emotion { border: 1px solid #5d5140; background: #1a1712; border-radius: 8px; padding: 12px; margin-top: -4px; }
    .emotion-head { display: flex; justify-content: space-between; gap: 10px; align-items: baseline; }
    .emotion-label { font-size: 20px; font-weight: 800; }
    .emotion-intensity { color: var(--warn); font-weight: 750; }
    .bar { height: 8px; border-radius: 99px; background: #0d1012; border: 1px solid var(--line); overflow: hidden; margin-top: 8px; }
    .bar-fill { height: 100%; width: 0%; background: linear-gradient(90deg, #75c59a, #f1c36d, #ff8989); }
    .composer { display: grid; grid-template-columns: 1fr auto; gap: 10px; align-items: end; position: sticky; bottom: 0; background: linear-gradient(180deg, transparent, var(--bg) 30%); padding-top: 20px; }
    .error { background: #3a1f24; color: #ffb8b8; border: 1px solid #75404a; border-radius: 8px; padding: 12px; margin: 12px 0; white-space: pre-wrap; }
    .hidden { display: none !important; }
    @media (max-width: 860px) {
      .app { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      .metrics, .trace-grid, .composer { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <h2>Claude API</h2>
      <label for="apiKey">API key</label>
      <input id="apiKey" type="password" autocomplete="off" placeholder="ANTHROPIC_API_KEY or paste here" />
      <label for="budget">Budget</label>
      <input id="budget" type="number" value="22" min="0" step="0.5" />
      <p class="small">model: """ + CLAUDE_MODEL + r"""</p>
      <label class="checkrow"><input id="llmPerception" type="checkbox" checked /> Raw signal input</label>
      <p class="small">켜면 Haiku가 감정 이름 없이 raw 신호만 만들고, EmoNet이 감정 상태를 판단합니다.</p>
      <label class="checkrow"><input id="showInternals" type="checkbox" /> Show internals</label>
      <button id="clearChat">Clear session</button>
      <button id="resetMemory">Reset memory</button>
      <button id="resetUsage">Reset usage</button>

      <section class="card">
        <div class="small">Character</div>
        <div id="characterName" class="character-name">...</div>
        <p id="persona"></p>
        <h3>Speech</h3>
        <p id="speech" class="small"></p>
      </section>

      <section class="card">
        <h3>Relationship</h3>
        <p id="relationship" class="small"></p>
        <h3>Scene</h3>
        <p id="scene" class="small"></p>
      </section>

      <section class="card">
        <h3>Memory</h3>
        <ul id="memory" class="list small"></ul>
      </section>
    </aside>
    <main>
      <h1>EmoNet v5 Character Chat</h1>
      <div class="sub">v5는 EmoNet trace를 캐릭터의 내부 정서 상태로 취급하고, 캐릭터 카드와 세션 기억을 함께 사용합니다.</div>

      <div class="metrics">
        <div class="metric"><div class="name">Session spent</div><div id="spent" class="value">$0.0000</div></div>
        <div class="metric"><div class="name">Budget left</div><div id="left" class="value">$22.00</div></div>
        <div class="metric"><div class="name">Tokens</div><div id="tokens" class="value">0 / 0</div></div>
      </div>
      <div id="processPanel" class="process-panel hidden"></div>

      <div id="chatError" class="error hidden"></div>
      <div id="chatlog" class="chatlog"></div>
      <section class="card">
        <h2>AI Dialogue Test</h2>
        <div class="sub">상대 AI가 USER 역할 발화를 만들고, Ruca가 기존 EmoNet 경로로 답합니다.</div>
        <label for="aiScenario">Scenario</label>
        <textarea id="aiScenario">관계가 흔들리는 밤. 상대는 Ruca에게 곧 떠나야 한다고 말하고, Ruca가 자기 내부 감정으로 반응하는지 테스트한다.</textarea>
        <label for="aiTurns">Turns</label>
        <input id="aiTurns" type="number" min="1" max="12" value="4" />
        <button id="runAi" class="primary">Run AI dialogue</button>
      </section>
      <div class="composer">
        <textarea id="message" placeholder="메시지를 입력하세요"></textarea>
        <button id="send" class="primary">Send</button>
      </div>
    </main>
  </div>
<script>
let messages = [];
let usage = { input_tokens: 0, output_tokens: 0, cost_usd: 0 };
let character = {};
let session = {};

const $ = id => document.getElementById(id);
const esc = value => String(value ?? "").replace(/[&<>"']/g, ch => ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", "\"":"&quot;", "'":"&#39;" }[ch]));
const compact = (value, fallback = "") => {
  const text = String(value || "").replace(/\s+/g, " ").trim();
  return text || fallback;
};

function renderState() {
  $("characterName").textContent = character.name || "...";
  $("persona").textContent = character.persona || "";
  $("speech").textContent = character.speech_style || "";
  $("relationship").textContent = session.relationship_state || character.relationship_defaults || "";
  $("scene").textContent = session.scene_state || character.world_state || "";
  const memory = Array.isArray(session.user_memory) ? session.user_memory : [];
  $("memory").innerHTML = memory.length ? memory.map(item => `<li>${esc(item)}</li>`).join("") : "<li>아직 기억 없음</li>";
  renderUsage();
  renderChat();
}

function renderUsage() {
  const spent = Number(usage.cost_usd || 0);
  const budget = Number($("budget").value || 0);
  $("spent").textContent = `$${spent.toFixed(4)}`;
  $("left").textContent = `$${Math.max(0, budget - spent).toFixed(2)}`;
  $("tokens").textContent = `${usage.input_tokens || 0} / ${usage.output_tokens || 0}`;
}

function traceHtml(record) {
  if (!record) return "";
  const state = record.emotion_state || {};
  const saturation = Math.max(0, Math.min(1, Number(state.saturation_ratio || 0)));
  return `
    <div class="emotion">
      <div class="emotion-head">
        <div class="emotion-label">${esc(compact(state.label, "감정 상태 없음"))}</div>
        <div class="emotion-intensity">강도 ${esc(compact(state.intensity, "unknown"))}</div>
      </div>
      <div class="small">${esc(compact(state.summary, ""))}</div>
      <div class="bar" title="saturation ${Math.round(saturation * 100)}%"><div class="bar-fill" style="width:${Math.round(saturation * 100)}%"></div></div>
    </div>
    <div class="trace">
      <div class="trace-title">Internal state used for character response</div>
      <div class="trace-grid">
        <div class="trace-cell"><div class="k">felt direction</div><div class="v">${esc(compact(record.appraisal_tendency, "unknown"))}</div></div>
        <div class="trace-cell"><div class="k">target</div><div class="v">${esc(compact(record.appraisal_target, "unknown"))}</div></div>
        <div class="trace-cell"><div class="k">input mode</div><div class="v">${esc(compact(record.affect_input_mode, "encoder"))}</div></div>
        <div class="trace-cell"><div class="k">trace summary</div><div class="v">${esc(compact(record.trace_summary_text, "none"))}</div></div>
        <div class="trace-cell"><div class="k">raw signal</div><div class="v">${esc(compact(JSON.stringify((record.agent_perception || {}).raw_signal || {}), "none"))}</div></div>
        <div class="trace-cell"><div class="k">saturation</div><div class="v">${esc(Math.round(saturation * 100))}%</div></div>
      </div>
    </div>
  `;
}

function latestAssistantRecord() {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "assistant" && messages[i].record) return messages[i].record;
  }
  return null;
}

function formatJson(value) {
  try { return JSON.stringify(value ?? {}, null, 2); }
  catch (_err) { return String(value ?? ""); }
}

function processStep(name, value) {
  return `<div class="process-step"><div class="step-name">${esc(name)}</div><pre>${esc(value)}</pre></div>`;
}

function renderProcessPanel() {
  const panel = $("processPanel");
  if (!$("showInternals").checked) {
    panel.classList.add("hidden");
    panel.innerHTML = "";
    return;
  }
  const record = latestAssistantRecord();
  if (!record) {
    panel.classList.remove("hidden");
    panel.innerHTML = `
      <div class="process-header">
        <div class="process-title">Internal Processing</div>
        <div class="small">No assistant turn yet</div>
      </div>`;
    return;
  }
  const rawSignal = (record.agent_perception || {}).raw_signal || {};
  const emotion = record.emotion_state || {};
  const felt = record.agent_felt_state || {};
  const steps = [
    processStep("1. input", record.input_text || ""),
    processStep("2. raw signal", formatJson({
      mode: record.affect_input_mode || "encoder",
      signal: rawSignal,
      confidence: (record.agent_perception || {}).confidence,
      mapped_stim_vec: record.affect_input_stim_vec || record.stim_vec || []
    })),
    processStep("3. EmoNet trace", (record.trace_lines || []).join("\n")),
    processStep("4. felt state", formatJson({
      emotion_state: emotion,
      agent_felt_state: felt,
      appraisal_tendency: record.appraisal_tendency,
      appraisal_target: record.appraisal_target
    })),
    processStep("5. response validation", formatJson({
      retry_count: record.response_retry_count || 0,
      validation_errors: record.response_validation_errors || []
    }))
  ].join("");
  panel.classList.remove("hidden");
  panel.innerHTML = `
    <div class="process-header">
      <div class="process-title">Internal Processing</div>
      <div class="small">${esc(record.character_name || "character")} · ${esc(record.llm_model_name || "")}</div>
    </div>
    <div class="process-steps">${steps}</div>`;
}

function renderChat() {
  $("chatlog").innerHTML = messages.map(m => {
    const bubble = `<div class="msg ${esc(m.role)}">${esc(m.content)}</div>`;
    const internals = $("showInternals").checked ? traceHtml(m.record) : "";
    return m.role === "assistant" ? bubble + internals : bubble;
  }).join("");
  renderProcessPanel();
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

async function loadState() {
  const res = await fetch("/api/status");
  const payload = await res.json();
  character = payload.character || {};
  session = payload.session || {};
  messages = payload.messages || [];
  usage = payload.usage || usage;
  renderState();
}

async function sendMessage(text) {
  const prompt = String(text || "").trim();
  if (!prompt) return;
  $("chatError").classList.add("hidden");
  $("send").disabled = true;
  messages.push({role: "user", content: prompt});
  renderChat();
  try {
    const payload = await api("/api/chat", {
      message: prompt,
      api_key: $("apiKey").value,
      affect_input_mode: $("llmPerception").checked ? "llm_raw_signal" : "encoder"
    });
    messages = payload.messages || messages;
    usage = payload.usage || usage;
    session = payload.session || session;
    character = payload.character || character;
  } catch (err) {
    $("chatError").textContent = err.message;
    $("chatError").classList.remove("hidden");
  } finally {
    $("send").disabled = false;
    $("message").value = "";
    renderState();
  }
}

async function runAiDialogue() {
  $("chatError").classList.add("hidden");
  $("runAi").disabled = true;
  $("send").disabled = true;
  try {
    const payload = await api("/api/ai-dialogue/run", {
      api_key: $("apiKey").value,
      scenario: $("aiScenario").value,
      turns: Number($("aiTurns").value || 4),
      affect_input_mode: $("llmPerception").checked ? "llm_raw_signal" : "encoder"
    });
    messages = payload.messages || messages;
    usage = payload.usage || usage;
    session = payload.session || session;
    character = payload.character || character;
  } catch (err) {
    $("chatError").textContent = err.message;
    $("chatError").classList.remove("hidden");
  } finally {
    $("runAi").disabled = false;
    $("send").disabled = false;
    renderState();
  }
}

$("send").onclick = () => sendMessage($("message").value);
$("runAi").onclick = () => runAiDialogue();
$("message").addEventListener("keydown", ev => {
  if (ev.key === "Enter" && (ev.ctrlKey || ev.metaKey)) sendMessage($("message").value);
});
$("clearChat").onclick = async () => { await fetch("/api/chat/clear", {method: "POST"}); await loadState(); };
$("resetMemory").onclick = async () => { await fetch("/api/memory/reset", {method: "POST"}); await loadState(); };
$("resetUsage").onclick = async () => { await fetch("/api/usage/reset", {method: "POST"}); await loadState(); };
$("budget").oninput = renderUsage;
$("showInternals").onchange = renderChat;
loadState();
</script>
</body>
</html>
"""


class LocalGuiHandler(BaseHTTPRequestHandler):
    server_version = "EmoNetV5LocalGUI/1.0"

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
            elif parsed.path == "/api/status":
                self._json(HTTPStatus.OK, _state_payload())
            else:
                self._error(HTTPStatus.NOT_FOUND, "not found")
        except Exception as exc:
            traceback.print_exc()
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self) -> None:
        global _character_session
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/api/chat":
                payload = _read_json(self)
                api_key = str(payload.get("api_key") or os.environ.get("ANTHROPIC_API_KEY") or "").strip()
                message = str(payload.get("message") or "").strip()
                affect_input_mode = str(payload.get("affect_input_mode") or "encoder").strip()
                if not api_key:
                    self._error(HTTPStatus.BAD_REQUEST, "Claude API key가 필요합니다. 왼쪽에 입력하거나 ANTHROPIC_API_KEY를 설정하세요.")
                    return
                if not message:
                    self._error(HTTPStatus.BAD_REQUEST, "message is empty")
                    return
                with _history_lock:
                    card = load_character_card()
                    result = generate_chat_turn(
                        runtime=_runtime_cached(),
                        generation_config=_chat_config(api_key, affect_input_mode=affect_input_mode),
                        input_text=message,
                        history=list(_messages),
                        character_card=card,
                        character_session=_character_session,
                    )
                    _character_session = result.character_session
                    _messages.append({"role": "user", "content": message})
                    _messages.append({"role": "assistant", "content": result.assistant_text, "record": result.record})
                    _append_usage_from_record(result.record)
                    response = _state_payload()
                self._json(HTTPStatus.OK, response)
            elif parsed.path == "/api/ai-dialogue/run":
                payload = _read_json(self)
                api_key = str(payload.get("api_key") or os.environ.get("ANTHROPIC_API_KEY") or "").strip()
                scenario = str(payload.get("scenario") or "").strip()
                turns = max(1, min(12, int(payload.get("turns") or 4)))
                affect_input_mode = str(payload.get("affect_input_mode") or "encoder").strip()
                if not api_key:
                    self._error(HTTPStatus.BAD_REQUEST, "Claude API key가 필요합니다. 왼쪽에 입력하거나 ANTHROPIC_API_KEY를 설정하세요.")
                    return
                with _history_lock:
                    card = load_character_card()
                    for turn_index in range(1, turns + 1):
                        user_message, user_usage = _generate_ai_user_message(
                            api_key=api_key,
                            scenario=scenario,
                            turn_index=turn_index,
                        )
                        _usage["input_tokens"] += int(user_usage.get("input_tokens", 0) or 0)
                        _usage["output_tokens"] += int(user_usage.get("output_tokens", 0) or 0)
                        _usage["cost_usd"] += _estimate_cost(
                            int(user_usage.get("input_tokens", 0) or 0),
                            int(user_usage.get("output_tokens", 0) or 0),
                        )
                        result = generate_chat_turn(
                            runtime=_runtime_cached(),
                            generation_config=_chat_config(api_key, affect_input_mode=affect_input_mode),
                            input_text=user_message,
                            history=list(_messages),
                            character_card=card,
                            character_session=_character_session,
                        )
                        _character_session = result.character_session
                        _messages.append({"role": "user", "content": user_message, "source": "ai_user"})
                        _messages.append({"role": "assistant", "content": result.assistant_text, "record": result.record})
                        _append_usage_from_record(result.record)
                    response = _state_payload()
                self._json(HTTPStatus.OK, response)
            elif parsed.path == "/api/chat/clear":
                with _history_lock:
                    _messages.clear()
                    _character_session = CharacterSessionState()
                self._json(HTTPStatus.OK, _state_payload())
            elif parsed.path == "/api/memory/reset":
                with _history_lock:
                    _character_session = CharacterSessionState()
                self._json(HTTPStatus.OK, _state_payload())
            elif parsed.path == "/api/usage/reset":
                with _history_lock:
                    _usage.update({"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0})
                self._json(HTTPStatus.OK, _state_payload())
            else:
                self._error(HTTPStatus.NOT_FOUND, "not found")
        except Exception as exc:
            traceback.print_exc()
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), LocalGuiHandler)
    print(f"EmoNet v5 character GUI: http://{HOST}:{PORT}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
