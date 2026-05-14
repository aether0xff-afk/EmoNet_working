from __future__ import annotations

import html
import os
from typing import Any

import streamlit as st

from emonet.chat_service import ChatGenerationConfig, ChatRuntimeConfig, build_chat_runtime, generate_chat_turn


st.set_page_config(page_title="EmoNet", layout="wide", initial_sidebar_state="expanded")

MESSAGES_KEY = "emonet_messages"
ERROR_KEY = "emonet_error"
USAGE_KEY = "emonet_usage"

CLAUDE_MODEL = "claude-haiku-4-5-20251001"
CLAUDE_BASE_URL = "https://api.anthropic.com"
CLAUDE_INPUT_PRICE = 1.0
CLAUDE_OUTPUT_PRICE = 5.0

SIGNAL_LABELS = (
    ("dopamine", "Dopamine"),
    ("serotonin", "Serotonin"),
    ("norepinephrine", "Norepinephrine"),
    ("melatonin", "Melatonin"),
)


def _init_state() -> None:
    if MESSAGES_KEY not in st.session_state:
        st.session_state[MESSAGES_KEY] = []
    if ERROR_KEY not in st.session_state:
        st.session_state[ERROR_KEY] = ""
    if USAGE_KEY not in st.session_state:
        st.session_state[USAGE_KEY] = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}


def _inject_css() -> None:
    st.markdown(
        """
<style>
    :root {
        --bg: #0f1318;
        --panel: #151b22;
        --panel-soft: #19212a;
        --panel-muted: #11171d;
        --border: rgba(154, 166, 184, 0.16);
        --border-strong: rgba(154, 166, 184, 0.24);
        --text: #edf2f7;
        --muted: #aab6c5;
        --faint: #7f8b9a;
        --accent: #78b88a;
        --accent-soft: rgba(120, 184, 138, 0.13);
        --user: #1f3f35;
    }

    .stApp {
        background: var(--bg);
        color: var(--text);
    }

    .block-container {
        max-width: 1040px;
        padding-top: 1.25rem;
        padding-bottom: 6rem;
    }

    section[data-testid="stSidebar"] {
        background: var(--panel);
        border-right: 1px solid var(--border);
    }

    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] label {
        color: var(--muted);
    }

    .app-header {
        border-bottom: 1px solid var(--border);
        padding-bottom: 1rem;
        margin-bottom: 1rem;
    }

    .app-title-row {
        align-items: flex-end;
        display: flex;
        gap: 0.8rem;
        justify-content: space-between;
        flex-wrap: wrap;
    }

    .app-title {
        font-size: 2rem;
        font-weight: 650;
        letter-spacing: 0;
        line-height: 1.1;
        margin: 0;
    }

    .app-subtitle {
        color: var(--muted);
        font-size: 0.98rem;
        line-height: 1.55;
        margin: 0.42rem 0 0 0;
        max-width: 720px;
    }

    .status-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin-top: 0.85rem;
    }

    .status-pill {
        background: var(--panel-soft);
        border: 1px solid var(--border);
        border-radius: 8px;
        color: #d8e1ea;
        font-size: 0.82rem;
        line-height: 1.2;
        padding: 0.38rem 0.6rem;
        white-space: normal;
    }

    .usage-wrap {
        margin: 0.25rem 0 1.1rem 0;
    }

    div[data-testid="stMetric"] {
        background: var(--panel-muted);
        border: 1px solid var(--border);
        border-radius: 8px;
        min-height: 86px;
        padding: 0.85rem 0.95rem;
    }

    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] {
        color: var(--muted);
    }

    div[data-testid="stMetricValue"] {
        color: var(--text);
        font-size: 1.28rem;
        letter-spacing: 0;
    }

    .bubble {
        border: 1px solid var(--border);
        border-radius: 8px;
        line-height: 1.68;
        margin-bottom: 0.45rem;
        padding: 0.9rem 1rem;
        overflow-wrap: anywhere;
        word-break: keep-all;
    }

    .bubble.user {
        background: var(--user);
        color: #f5fbf7;
    }

    .bubble.assistant {
        background: var(--panel-soft);
        color: var(--text);
    }

    .signal-grid {
        display: grid;
        gap: 0.55rem;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        margin: 0.35rem 0 0.8rem 0;
    }

    .signal-row {
        background: var(--panel-muted);
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 0.65rem 0.7rem;
    }

    .signal-head {
        align-items: center;
        display: flex;
        justify-content: space-between;
        gap: 0.65rem;
        margin-bottom: 0.45rem;
    }

    .signal-name {
        color: #d9e2eb;
        font-size: 0.86rem;
        font-weight: 580;
    }

    .signal-value {
        color: var(--muted);
        font-size: 0.82rem;
        font-variant-numeric: tabular-nums;
    }

    .signal-track {
        background: rgba(154, 166, 184, 0.14);
        border-radius: 999px;
        height: 6px;
        overflow: hidden;
    }

    .signal-fill {
        background: var(--accent);
        border-radius: 999px;
        height: 6px;
    }

    .detail-line {
        color: var(--muted);
        font-size: 0.9rem;
        line-height: 1.55;
        margin: 0.18rem 0;
        overflow-wrap: anywhere;
    }

    .empty-state {
        border: 1px solid var(--border);
        border-radius: 8px;
        color: var(--muted);
        margin: 1rem 0;
        padding: 0.85rem 0.95rem;
    }

    .stButton > button {
        background: var(--panel-soft);
        border: 1px solid var(--border);
        border-radius: 8px;
        color: var(--text);
        min-height: 3.15rem;
        white-space: normal;
    }

    .stButton > button:hover {
        background: #1d2731;
        border-color: var(--border-strong);
        color: var(--text);
    }

    div[data-testid="stExpander"] {
        background: transparent;
        border: 1px solid var(--border);
        border-radius: 8px;
        margin-bottom: 0.8rem;
    }

    div[data-testid="stExpander"] summary {
        color: #d9e2eb;
    }

    div[data-testid="stChatInput"] textarea {
        background: var(--panel);
        border-color: var(--border);
        color: var(--text);
    }

    @media (max-width: 720px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .app-title {
            font-size: 1.7rem;
        }

        .signal-grid {
            grid-template-columns: 1fr;
        }
    }
</style>
        """,
        unsafe_allow_html=True,
    )


def _api_key_from_input(raw: str) -> str:
    return str(raw or "").strip() or os.environ.get("ANTHROPIC_API_KEY", "").strip()


def _estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (input_tokens * CLAUDE_INPUT_PRICE + output_tokens * CLAUDE_OUTPUT_PRICE) / 1_000_000.0


def _usage_metrics(budget_usd: float) -> None:
    usage = st.session_state[USAGE_KEY]
    spent = float(usage.get("cost_usd", 0.0) or 0.0)
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    st.markdown("<div class='usage-wrap'>", unsafe_allow_html=True)
    cols = st.columns(3)
    cols[0].metric("Session spent", f"${spent:.4f}")
    cols[1].metric("Budget left", f"${max(0.0, budget_usd - spent):.2f}")
    cols[2].metric("Tokens", f"{input_tokens} in / {output_tokens} out")
    st.markdown("</div>", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading EmoNet runtime...")
def _load_runtime() -> Any:
    return build_chat_runtime(ChatRuntimeConfig())


def _generation_config(api_key: str) -> ChatGenerationConfig:
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


def _render_header() -> None:
    st.markdown(
        """
<header class="app-header">
  <div class="app-title-row">
    <h1 class="app-title">EmoNet</h1>
  </div>
  <p class="app-subtitle">감정의 결을 낮추지 않고 읽어, 지금 상황에 맞는 한국어 응답을 만듭니다.</p>
  <div class="status-row">
    <span class="status-pill">Claude Haiku 4.5</span>
    <span class="status-pill">hybrid_trace</span>
    <span class="status-pill">extended40</span>
  </div>
</header>
        """,
        unsafe_allow_html=True,
    )


def _safe_html_text(content: object) -> str:
    return html.escape(str(content or "")).replace("\n", "<br>")


def _render_message(role: str, content: str, record: dict[str, Any] | None = None) -> None:
    css_role = "user" if role == "user" else "assistant"
    with st.chat_message(role):
        st.markdown(f"<div class='bubble {css_role}'>{_safe_html_text(content)}</div>", unsafe_allow_html=True)
        if css_role == "assistant" and record:
            _render_signal_summary(record)


def _clamped_signal(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _render_signal_grid(stim_vec: list[float]) -> None:
    rows: list[str] = []
    for idx, (_key, label) in enumerate(SIGNAL_LABELS):
        value = _clamped_signal(stim_vec[idx] if idx < len(stim_vec) else 0.0)
        rows.append(
            f"""
<div class="signal-row">
  <div class="signal-head">
    <span class="signal-name">{html.escape(label)}</span>
    <span class="signal-value">{value:.2f}</span>
  </div>
  <div class="signal-track"><div class="signal-fill" style="width: {value * 100:.1f}%"></div></div>
</div>
            """
        )
    st.markdown(f"<div class='signal-grid'>{''.join(rows)}</div>", unsafe_allow_html=True)


def _render_signal_summary(record: dict[str, Any]) -> None:
    with st.expander("Signal summary", expanded=False):
        stim_vec = list(record.get("stim_vec", []) or [])
        _render_signal_grid(stim_vec)

        style_tags = [str(tag) for tag in record.get("style_tags", []) if str(tag).strip()]
        style_text = ", ".join(style_tags[:8]) if style_tags else "No style tags"
        branch_len = int(record.get("dominant_branch_len", 0) or 0)
        termination = str(record.get("termination_reason", "") or "unknown")
        retry_count = int(record.get("response_retry_count", 0) or 0)

        st.markdown(
            "\n".join(
                [
                    f"<p class='detail-line'><b>Style</b>: {html.escape(style_text)}</p>",
                    f"<p class='detail-line'><b>Dominant branch</b>: {branch_len}</p>",
                    f"<p class='detail-line'><b>Termination</b>: {html.escape(termination)}</p>",
                    f"<p class='detail-line'><b>Retries</b>: {retry_count}</p>",
                ]
            ),
            unsafe_allow_html=True,
        )


def _generate_reply(prompt: str, api_key: str) -> None:
    if not api_key:
        raise ValueError("Claude API key가 필요합니다. 왼쪽 sidebar에 입력하거나 ANTHROPIC_API_KEY 환경변수를 설정하세요.")

    history = [
        {"role": str(message.get("role", "")), "content": str(message.get("content", ""))}
        for message in st.session_state[MESSAGES_KEY]
    ]
    st.session_state[MESSAGES_KEY].append({"role": "user", "content": prompt})

    runtime = _load_runtime()
    result = generate_chat_turn(
        runtime=runtime,
        generation_config=_generation_config(api_key),
        input_text=prompt,
        history=history,
    )
    st.session_state[MESSAGES_KEY].append(
        {"role": "assistant", "content": result.assistant_text, "record": result.record}
    )

    usage = dict(result.record.get("llm_usage", {}))
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    total = dict(st.session_state[USAGE_KEY])
    total["input_tokens"] = int(total.get("input_tokens", 0) or 0) + input_tokens
    total["output_tokens"] = int(total.get("output_tokens", 0) or 0) + output_tokens
    total["cost_usd"] = float(total.get("cost_usd", 0.0) or 0.0) + _estimate_cost(input_tokens, output_tokens)
    st.session_state[USAGE_KEY] = total


def _submit_prompt(prompt: str, api_key: str) -> None:
    try:
        st.session_state[ERROR_KEY] = ""
        _generate_reply(prompt.strip(), api_key)
    except Exception as exc:
        st.session_state[ERROR_KEY] = str(exc)


def _render_examples(api_key: str) -> None:
    examples = [
        "회의에서 또 나만 공개적으로 무시당했어. 바로 따지고 싶을 정도로 거슬려.",
        "이번 주 내내 야근이라 머리가 멍하고 다 놓아버리고 싶어.",
        "잘된 일인데도 이상하게 기쁘기보다 불안하고 예민해.",
    ]
    cols = st.columns(3)
    for idx, example in enumerate(examples):
        if cols[idx].button(example, use_container_width=True):
            _submit_prompt(example, api_key)
            st.rerun()


def _render_chat(api_key: str, budget_usd: float) -> None:
    _render_header()
    _usage_metrics(budget_usd)

    if st.session_state[ERROR_KEY]:
        st.error(st.session_state[ERROR_KEY])

    if not st.session_state[MESSAGES_KEY]:
        st.markdown(
            "<div class='empty-state'>아래 예시를 누르거나, 지금 감정이 생긴 상황을 직접 적어보세요.</div>",
            unsafe_allow_html=True,
        )
        _render_examples(api_key)

    for message in st.session_state[MESSAGES_KEY]:
        record = message.get("record") if isinstance(message.get("record"), dict) else None
        _render_message(str(message.get("role", "")), str(message.get("content", "")), record)

    prompt = st.chat_input("지금 감정이 생긴 상황을 적어보세요")
    if prompt:
        _submit_prompt(prompt, api_key)
        st.rerun()


def _render_sidebar() -> tuple[str, float]:
    with st.sidebar:
        st.header("Claude")
        api_key = _api_key_from_input(st.text_input("API key", type="password"))
        budget = st.number_input("Budget", min_value=0.0, max_value=1000.0, value=22.0, step=0.5)
        st.caption("API key는 파일에 저장하지 않습니다.")

        st.divider()
        if st.button("Clear chat", use_container_width=True):
            st.session_state[MESSAGES_KEY] = []
            st.session_state[ERROR_KEY] = ""
            st.rerun()
        if st.button("Reset usage", use_container_width=True):
            st.session_state[USAGE_KEY] = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
            st.rerun()
    return api_key, float(budget)


def main() -> None:
    _init_state()
    _inject_css()
    api_key, budget = _render_sidebar()
    _render_chat(api_key, budget)


if __name__ == "__main__":
    main()
