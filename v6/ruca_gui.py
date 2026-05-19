from __future__ import annotations

import json
import os
import sys
import traceback
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


HOST = "127.0.0.1"
PORT = int(os.environ.get("RUCA_GUI_PORT", "8790"))
ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
V5_ROOT = REPO_ROOT / "v5"
if str(V5_ROOT) not in sys.path:
    sys.path.insert(0, str(V5_ROOT))

from emonet.character import CharacterSessionState  # noqa: E402
from emonet.chat_service import (  # noqa: E402
    ChatGenerationConfig,
    ChatRuntimeConfig,
    EmoNetChatRuntime,
    build_chat_runtime,
    generate_chat_turn,
)


STATE_DIR = ROOT / "outputs" / "gui"
SESSION_PATH = STATE_DIR / "ruca_v5_character_session.json"
HISTORY_PATH = STATE_DIR / "ruca_v5_history.json"
LOG_PATH = STATE_DIR / "ruca_gui.log.jsonl"
ARTIFACT_ROOT = Path(os.environ.get("EMONET_ARTIFACT_ROOT", str(ROOT / "artifacts")))

_runtime: EmoNetChatRuntime | None = None


def _runtime_instance() -> EmoNetChatRuntime:
    global _runtime
    if _runtime is None:
        _runtime = build_chat_runtime(
            ChatRuntimeConfig(
                model_cache_path=ARTIFACT_ROOT / "ridge_stim_encoder.joblib",
                z_encoder_path=ARTIFACT_ROOT / "dominant_branch_encoder_extended40_calref_v1.pt",
                zs_model_path=ARTIFACT_ROOT / "z_to_s_decoder_extended40_calref_v1.npz",
            )
        )
    return _runtime


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def _read_json(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0") or "0")
    if length <= 0:
        return {}
    payload = json.loads(handler.rfile.read(length).decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("JSON body must be an object")
    return payload


def _append_log(event: str, payload: dict[str, Any]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    record = {"created_at": _now_iso(), "event": event, **payload}
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read_log_tail(limit: int = 80) -> list[dict[str, Any]]:
    if not LOG_PATH.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in LOG_PATH.read_text(encoding="utf-8").splitlines()[-max(1, int(limit)) :]:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _load_character_session() -> CharacterSessionState:
    if not SESSION_PATH.exists():
        return CharacterSessionState()
    try:
        payload = json.loads(SESSION_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return CharacterSessionState()
    return CharacterSessionState.from_mapping(payload)


def _save_character_session(session: CharacterSessionState) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    SESSION_PATH.write_text(json.dumps(session.to_record(), ensure_ascii=False, indent=2), encoding="utf-8")


def _load_history() -> list[dict[str, Any]]:
    if not HISTORY_PATH.exists():
        return []
    try:
        payload = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    history: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "").strip()
        content = str(item.get("content") or "").strip()
        if role not in {"user", "assistant"} or not content:
            continue
        message = {"role": role, "content": content}
        if isinstance(item.get("record"), dict):
            message["record"] = item["record"]
        history.append(message)
    return history[-80:]


def _save_history(history: list[dict[str, Any]]) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    HISTORY_PATH.write_text(json.dumps(history[-80:], ensure_ascii=False, indent=2), encoding="utf-8")


def _generation_config(payload: dict[str, Any]) -> ChatGenerationConfig:
    return ChatGenerationConfig(
        provider=str(payload.get("provider") or "openai_compatible").strip(),
        base_url=str(payload.get("base_url") or "http://127.0.0.1:11434/v1").strip(),
        model_name=str(payload.get("model_name") or "qwen3:14b").strip(),
        api_key=str(payload.get("api_key") or "").strip() or None,
        response_temperature=float(payload.get("temperature") or 0.45),
        max_tokens=int(payload.get("max_tokens") or 900),
        timeout_sec=int(payload.get("timeout_sec") or 180),
        reasoning_effort=str(payload.get("reasoning_effort") or "").strip() or None,
        style_profile=str(payload.get("style_profile") or "extended40").strip(),
        conditioning_mode=str(payload.get("conditioning_mode") or "hybrid_trace").strip(),
        affect_input_mode=str(payload.get("affect_input_mode") or "encoder").strip(),
        raw_signal_policy=str(payload.get("raw_signal_policy") or "event_annotated").strip(),
    )


def _messages_from_history() -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for item in _load_history():
        message = {"role": item["role"], "content": item["content"]}
        if isinstance(item.get("record"), dict):
            message["summary"] = item["record"]
        messages.append(message)
    return messages


def _status_payload() -> dict[str, Any]:
    session = _load_character_session()
    return {
        "engine": "v5/emonet.chat_service.generate_chat_turn",
        "artifact_root": str(ARTIFACT_ROOT),
        "session": session.to_record(),
        "messages": _messages_from_history(),
        "memory": [dict(item) for item in session.emotion_memory],
        "logs": _read_log_tail(),
        "paths": {
            "session": str(SESSION_PATH),
            "history": str(HISTORY_PATH),
            "log": str(LOG_PATH),
        },
        "default_config": {
            "provider": "openai_compatible",
            "base_url": "http://127.0.0.1:11434/v1",
            "model_name": "qwen3:14b",
            "max_tokens": 900,
            "temperature": 0.45,
            "api_key_env": "",
            "style_profile": "extended40",
            "conditioning_mode": "hybrid_trace",
            "affect_input_mode": "encoder",
        },
        "env": {
            "openai_key": bool(os.environ.get("OPENAI_API_KEY", "").strip()),
            "anthropic_key": bool(os.environ.get("ANTHROPIC_API_KEY", "").strip()),
        },
    }


APP_HTML = r"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Ruca/Rookie v6 GUI</title>
  <style>
    :root {
      --bg:#101214; --surface:#171b1f; --surface2:#20262b; --field:#0b0f12;
      --line:#313b44; --line2:#45525d; --text:#eef3f6; --muted:#aab6bf;
      --accent:#7bc7a4; --accent2:#d7b46a; --user:#1f3b34; --bad:#ffaaaa;
    }
    * { box-sizing:border-box; }
    html { color-scheme:dark; }
    body { margin:0; background:var(--bg); color:var(--text); font:15px/1.5 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
    button,input,select,textarea { font:inherit; }
    .app { display:grid; grid-template-columns:360px minmax(0,1fr); min-height:100vh; }
    aside { border-right:1px solid var(--line); background:var(--surface); padding:18px; overflow:auto; }
    main { padding:22px; max-width:1180px; width:100%; }
    h1 { margin:0; font-size:28px; line-height:1.15; letter-spacing:0; }
    h2 { margin:18px 0 8px; font-size:17px; letter-spacing:0; }
    label { display:block; color:var(--muted); font-size:13px; margin:12px 0 6px; }
    input,select,textarea { width:100%; background:var(--field); color:var(--text); border:1px solid var(--line); border-radius:7px; padding:10px 11px; outline:none; }
    input:focus,select:focus,textarea:focus { border-color:var(--accent); box-shadow:0 0 0 3px rgba(123,199,164,.14); }
    textarea { min-height:112px; resize:vertical; }
    button { border:1px solid var(--line); background:var(--surface2); color:var(--text); border-radius:7px; padding:10px 12px; cursor:pointer; }
    button:hover:not(:disabled) { border-color:var(--line2); filter:brightness(1.05); }
    button.primary { background:#1f6a47; border-color:#3a9465; min-width:92px; }
    button:disabled { opacity:.55; cursor:wait; }
    .checkrow { display:flex; gap:9px; align-items:center; color:var(--muted); font-size:13px; margin:12px 0; }
    .checkrow input { width:auto; }
    .sub,.small { color:var(--muted); }
    .small { font-size:13px; white-space:pre-wrap; overflow-wrap:anywhere; }
    .card,.metric,.msg,.debug { border:1px solid var(--line); border-radius:8px; }
    .card { background:rgba(255,255,255,.018); padding:14px; margin:12px 0; }
    .metrics { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; margin:18px 0; }
    .metric { background:var(--surface); padding:13px; }
    .metric .name { color:var(--muted); font-size:12px; }
    .metric .value { font-size:18px; font-weight:760; margin-top:4px; overflow-wrap:anywhere; }
    .chatlog { display:grid; gap:12px; margin:18px 0; max-height:calc(100vh - 260px); overflow:auto; padding-right:4px; }
    .msg { padding:13px 14px; white-space:pre-wrap; max-width:78ch; }
    .msg.user { justify-self:end; background:var(--user); }
    .msg.assistant { justify-self:start; background:var(--surface2); }
    .debug { background:#12181d; padding:12px; margin-top:-6px; }
    .debug-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:9px; }
    .debug-cell { border:1px solid rgba(174,184,192,.22); border-radius:7px; padding:9px; background:rgba(5,8,10,.25); }
    .debug-cell .k { color:var(--muted); font-size:12px; }
    pre { white-space:pre-wrap; overflow-wrap:anywhere; margin:0; font-size:12px; line-height:1.45; }
    .composer { display:grid; grid-template-columns:1fr auto; gap:10px; align-items:end; position:sticky; bottom:0; padding-top:18px; background:linear-gradient(180deg,transparent,var(--bg) 30%); }
    .error { background:#3a1f24; color:var(--bad); border:1px solid #75404a; border-radius:8px; padding:12px; margin:12px 0; white-space:pre-wrap; }
    .row { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
    .pill { display:inline-flex; align-items:center; gap:6px; color:#111; background:var(--accent); border-radius:999px; padding:4px 9px; font-size:12px; font-weight:700; margin-bottom:10px; }
    .hidden { display:none !important; }
    @media (max-width:880px) { .app,.metrics,.debug-grid,.composer,.row { grid-template-columns:1fr; } aside { border-right:0; border-bottom:1px solid var(--line); } .msg { max-width:100%; } }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <div class="pill">v6 GUI</div>
      <h2>LLM 설정</h2>
      <label class="checkrow"><input id="useLlm" type="checkbox" checked disabled /> v5 EmoNet trace + v6 Ruca GUI</label>
      <label for="provider">Provider</label>
      <select id="provider"><option value="openai_compatible">OpenAI compatible</option><option value="anthropic">Anthropic</option></select>
      <label for="baseUrl">Base URL</label><input id="baseUrl" value="http://127.0.0.1:11434/v1" />
      <label for="modelName">Model</label><input id="modelName" value="qwen3:14b" />
      <label for="apiKey">API key</label><input id="apiKey" type="password" autocomplete="off" placeholder="optional" />
      <div class="row">
        <div><label for="maxTokens">Max tokens</label><input id="maxTokens" type="number" min="128" step="64" value="900" /></div>
        <div><label for="temperature">Temperature</label><input id="temperature" type="number" min="0" max="2" step="0.05" value="0.45" /></div>
      </div>
      <label for="conditioningMode">Conditioning</label>
      <select id="conditioningMode"><option value="hybrid_trace">hybrid_trace</option><option value="raw_trace">raw_trace</option><option value="appraisal_trace">appraisal_trace</option><option value="style">style</option></select>
      <label for="affectInputMode">Affect input</label>
      <select id="affectInputMode"><option value="encoder">encoder</option><option value="llm_raw_signal">llm_raw_signal</option><option value="llm_perception">llm_perception</option></select>
      <label class="checkrow"><input id="showDebug" type="checkbox" /> v5 record 표시</label>
      <label class="checkrow"><input id="showLogs" type="checkbox" /> 로그 표시</label>
      <button id="resetState">세션 초기화</button>
      <section class="card"><h2>상태</h2><div class="small" id="envState">불러오는 중...</div><div class="small" id="paths"></div></section>
      <section class="card hidden" id="logPanel"><h2>로그</h2><pre id="logs"></pre></section>
    </aside>
    <main>
      <h1>Ruca/Rookie v6</h1>
      <div class="sub">v5 EmoNet trace를 기반으로 Ruca 캐릭터 세션을 실행하는 로컬 GUI입니다.</div>
      <div class="metrics">
        <div class="metric"><div class="name">engine</div><div id="engine" class="value">v6 GUI</div></div>
        <div class="metric"><div class="name">model</div><div id="model" class="value">qwen3:14b</div></div>
        <div class="metric"><div class="name">conditioning</div><div id="conditioning" class="value">hybrid_trace</div></div>
        <div class="metric"><div class="name">branch len</div><div id="branchLen" class="value">-</div></div>
      </div>
      <div id="error" class="error hidden"></div>
      <div id="chatlog" class="chatlog"></div>
      <div class="composer"><textarea id="message" placeholder="바로 말해줘. Enter로 보내고, Shift+Enter로 줄바꿈할 수 있어."></textarea><button id="send" class="primary">보내기</button></div>
    </main>
  </div>
<script>
let messages = [];
let latestDebug = null;
let status = {};
const $ = id => document.getElementById(id);
const esc = value => String(value ?? "").replace(/[&<>"']/g, ch => ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;" }[ch]));
const compact = (value, fallback = "-") => value === null || value === undefined || value === "" ? fallback : String(value);

function currentSettings() {
    return {
    provider: $("provider").value, base_url: $("baseUrl").value, model_name: $("modelName").value,
    max_tokens: $("maxTokens").value, temperature: $("temperature").value, conditioning_mode: $("conditioningMode").value,
    style_profile: "extended40",
    affect_input_mode: $("affectInputMode").value, show_debug: $("showDebug").checked, show_logs: $("showLogs").checked
  };
}
function saveSettings() { localStorage.setItem("ruca_v6_gui_settings", JSON.stringify(currentSettings())); }
function loadSettings() {
  let saved = {};
  try {
    saved = JSON.parse(localStorage.getItem("ruca_v6_gui_settings") || localStorage.getItem("ruca_v5_gui_settings") || "{}");
  } catch (_err) {}
  const defaults = status.default_config || {};
  $("provider").value = saved.provider || defaults.provider || "openai_compatible";
  $("baseUrl").value = saved.base_url || defaults.base_url || "http://127.0.0.1:11434/v1";
  $("modelName").value = saved.model_name || defaults.model_name || "qwen3:14b";
  $("maxTokens").value = saved.max_tokens || String(defaults.max_tokens || 900);
  $("temperature").value = saved.temperature || "0.45";
  $("conditioningMode").value = saved.conditioning_mode || defaults.conditioning_mode || "hybrid_trace";
  $("affectInputMode").value = saved.affect_input_mode || defaults.affect_input_mode || "encoder";
  $("showDebug").checked = saved.show_debug ?? false;
  $("showLogs").checked = saved.show_logs ?? false;
}
function debugHtml(record) {
  if (!$("showDebug").checked || !record) return "";
  const trace = { stim_vec: record.affect_input_stim_vec, z: record.z, s_pred: record.s_pred, trace_profile: record.trace_profile, trace_summary: record.trace_summary_text, trace_lines: record.trace_lines };
  const appraisal = { summary: record.appraisal_summary_text, target: record.appraisal_target, tendency: record.appraisal_tendency, lines: record.appraisal_lines };
  const llm = { model: record.llm_model_name, base_url: record.llm_base_url, usage: record.response_usage, retry_count: record.response_retry_count };
  return `<div class="debug">
    <div class="debug-grid">
      <div class="debug-cell"><div class="k">emotion</div><pre>${esc(JSON.stringify(record.emotion_state || {}, null, 2))}</pre></div>
      <div class="debug-cell"><div class="k">felt / drive</div><pre>${esc(JSON.stringify({ felt: record.agent_felt_state || record.felt_self, drive: record.drive }, null, 2))}</pre></div>
      <div class="debug-cell"><div class="k">LLM</div><pre>${esc(JSON.stringify(llm, null, 2))}</pre></div>
    </div>
    <h2>EmoNet Trace</h2><pre>${esc(JSON.stringify(trace, null, 2))}</pre>
    <h2>Appraisal</h2><pre>${esc(JSON.stringify(appraisal, null, 2))}</pre>
    <h2>Memory</h2><pre>${esc(JSON.stringify(record.emotion_memory || record.character_memory || [], null, 2))}</pre>
  </div>`;
}
function render() {
  $("engine").textContent = "v6 GUI / v5 trace";
  $("model").textContent = compact((latestDebug || {}).llm_model_name, $("modelName").value);
  $("conditioning").textContent = compact((latestDebug || {}).conditioning_mode, $("conditioningMode").value);
  $("branchLen").textContent = compact((latestDebug || {}).dominant_branch_len);
  $("envState").textContent = `engine: ${status.engine || ""}\nartifact_root: ${status.artifact_root || ""}\nOPENAI_API_KEY=${status.env?.openai_key ? "set" : "missing"} / ANTHROPIC_API_KEY=${status.env?.anthropic_key ? "set" : "missing"}`;
  $("paths").textContent = `session: ${status.paths?.session || ""}\nhistory: ${status.paths?.history || ""}\nlog: ${status.paths?.log || ""}`;
  $("logPanel").classList.toggle("hidden", !$("showLogs").checked);
  $("logs").textContent = (status.logs || []).map(item => JSON.stringify(item)).join("\n");
  const visibleMessages = messages.length ? messages : [{ role:"assistant", content:"좋아. 바로 말해줘.", starter:true }];
  $("chatlog").innerHTML = visibleMessages.map(message => {
    const bubble = `<div class="msg ${esc(message.role)}">${esc(message.content)}</div>`;
    if (message.starter) return bubble;
    const summary = message.summary && $("showDebug").checked ? debugHtml(message.summary) : "";
    return bubble + (message.debug ? debugHtml(message.debug) : summary);
  }).join("");
  $("chatlog").scrollTop = $("chatlog").scrollHeight;
}
async function api(path, body) {
  const res = await fetch(path, { method:"POST", headers:{ "Content-Type":"application/json" }, body:JSON.stringify(body || {}) });
  const payload = await res.json();
  if (!res.ok) throw new Error(payload.error || `HTTP ${res.status}`);
  return payload;
}
async function loadStatus() {
  const res = await fetch("/api/status");
  status = await res.json();
  loadSettings();
  messages = status.messages || [];
  latestDebug = null;
  render();
}
async function sendMessage() {
  const text = $("message").value.trim();
  if (!text) return;
  $("error").classList.add("hidden");
  $("send").disabled = true;
  $("send").textContent = "전송 중...";
  messages.push({ role:"user", content:text });
  render();
  saveSettings();
  try {
    const payload = await api("/api/chat", {
      message:text, provider:$("provider").value, base_url:$("baseUrl").value, model_name:$("modelName").value,
      api_key:$("apiKey").value, max_tokens:$("maxTokens").value, temperature:$("temperature").value, conditioning_mode:$("conditioningMode").value,
      affect_input_mode:$("affectInputMode").value, style_profile:"extended40"
    });
    latestDebug = payload.record;
    messages.push({ role:"assistant", content:payload.assistant_text, debug:payload.record });
    status = payload.status;
    $("message").value = "";
  } catch (err) {
    $("error").textContent = err.message;
    $("error").classList.remove("hidden");
  } finally {
    $("send").disabled = false;
    $("send").textContent = "보내기";
    render();
  }
}
$("provider").onchange = () => {
  if ($("provider").value === "anthropic") {
    $("baseUrl").value = "https://api.anthropic.com";
    $("modelName").value = "claude-haiku-4-5-20251001";
  } else {
    $("baseUrl").value = "http://127.0.0.1:11434/v1";
    $("modelName").value = "qwen3:14b";
  }
  saveSettings();
};
$("send").onclick = sendMessage;
$("message").addEventListener("keydown", ev => {
  if (ev.key === "Enter" && !ev.shiftKey) {
    ev.preventDefault();
    sendMessage();
  }
});
for (const id of ["baseUrl", "modelName", "maxTokens", "temperature", "conditioningMode", "affectInputMode", "showDebug", "showLogs"]) {
  $(id).onchange = () => { saveSettings(); render(); };
}
$("resetState").onclick = async () => { await api("/api/reset", {}); messages = []; latestDebug = null; await loadStatus(); };
loadStatus().then(() => $("message").focus());
</script>
</body>
</html>
"""


class RucaGuiHandler(BaseHTTPRequestHandler):
    server_version = "RucaV6Gui/1.0"

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
            elif parsed.path == "/api/status":
                self._json(HTTPStatus.OK, _status_payload())
            elif parsed.path == "/favicon.ico":
                self._send(HTTPStatus.NO_CONTENT, b"", "image/x-icon")
            else:
                self._error(HTTPStatus.NOT_FOUND, "not found")
        except Exception as exc:
            traceback.print_exc()
            _append_log("error", {"error": str(exc), "traceback": traceback.format_exc()})
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self) -> None:
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/api/chat":
                payload = _read_json(self)
                text = str(payload.get("message") or "").strip()
                if not text:
                    self._error(HTTPStatus.BAD_REQUEST, "message is empty")
                    return
                history = _load_history()
                session = _load_character_session()
                config = _generation_config(payload)
                result = generate_chat_turn(
                    runtime=_runtime_instance(),
                    generation_config=config,
                    input_text=text,
                    history=history,
                    character_session=session,
                )
                next_history = history + [
                    {"role": "user", "content": text},
                    {"role": "assistant", "content": result.assistant_text, "record": result.record},
                ]
                _save_history(next_history)
                _save_character_session(result.character_session)
                _append_log(
                    "v5_chat_turn",
                    {
                        "user_text": text,
                        "assistant_text": result.assistant_text,
                        "model": result.record.get("llm_model_name"),
                        "conditioning_mode": result.record.get("conditioning_mode"),
                        "affect_input_mode": result.record.get("affect_input_mode"),
                        "dominant_branch_len": result.record.get("dominant_branch_len"),
                        "trace_summary_text": result.record.get("trace_summary_text"),
                    },
                )
                self._json(
                    HTTPStatus.OK,
                    {
                        "assistant_text": result.assistant_text,
                        "record": result.record,
                        "status": _status_payload(),
                    },
                )
            elif parsed.path == "/api/reset":
                for path in (SESSION_PATH, HISTORY_PATH, LOG_PATH):
                    if path.exists():
                        path.unlink()
                _append_log("reset", {"engine": "v6_gui"})
                self._json(HTTPStatus.OK, _status_payload())
            else:
                self._error(HTTPStatus.NOT_FOUND, "not found")
        except Exception as exc:
            traceback.print_exc()
            _append_log("error", {"error": str(exc), "traceback": traceback.format_exc()})
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))


def main() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((HOST, PORT), RucaGuiHandler)
    print(f"Ruca/Rookie v6 GUI: http://{HOST}:{PORT}/")
    server.serve_forever()


if __name__ == "__main__":
    main()
