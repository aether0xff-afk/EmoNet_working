from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ruca_engine import LLMConfig, RucaPipeline, TurnResult
from runtime import ThoughtCouncil


PROVIDER_DEFAULTS = {
    "lm_studio": {
        "base_url": "http://100.115.40.97:1234/v1",
        "model_name": "gemma-4-26b-a4b-it-qat",
        "api_key_env": "",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta",
        "model_name": "gemini-2.5-flash",
        "api_key_env": "GEMINI_API_KEY",
    },
    "openai_compatible": {
        "base_url": "https://api.openai.com/v1",
        "model_name": "gpt-5.4",
        "api_key_env": "OPENAI_API_KEY",
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com",
        "model_name": "claude-sonnet-4-5",
        "api_key_env": "ANTHROPIC_API_KEY",
    },
}


def build_llm_config(
    *,
    provider: str,
    api_key: str,
    model_name: str,
    base_url: str,
    temperature: float = 0.7,
    max_tokens: int = 900,
    timeout_sec: int = 60,
) -> LLMConfig:
    clean_provider = str(provider or "gemini").strip()
    defaults = PROVIDER_DEFAULTS.get(clean_provider, PROVIDER_DEFAULTS["gemini"])
    client_provider = "openai_compatible" if clean_provider == "lm_studio" else clean_provider
    return LLMConfig(
        provider=client_provider,
        base_url=str(base_url or defaults["base_url"]).strip(),
        model_name=str(model_name or defaults["model_name"]).strip(),
        api_key=str(api_key or "").strip() or None,
        api_key_env=str(defaults["api_key_env"]),
        temperature=float(temperature),
        max_tokens=int(max_tokens),
        timeout_sec=int(timeout_sec),
    )


def make_pipeline(*, use_llm: bool, llm_config: LLMConfig) -> RucaPipeline:
    return RucaPipeline(use_emonet=True, use_llm=use_llm, llm_config=llm_config, response_timing_mode="neural")


def compact_trace(result: TurnResult) -> dict[str, Any]:
    trace = dict(result.debug_record.get("emonet_trace") or {})
    profile = dict(trace.get("trace_profile") or {})
    memory = dict(trace.get("neuron_memory") or {})
    return {
        "source": trace.get("source"),
        "event_kind": trace.get("event_kind"),
        "tick": profile.get("tick_index"),
        "stim_dim": len(trace.get("stim_vec") or []),
        "dominant_cluster": profile.get("dominant_cluster_id"),
        "mean_active_nodes": profile.get("mean_active_nodes"),
        "stored_memory_count": memory.get("stored_memory_count", 0),
        "summary": trace.get("trace_summary_text", ""),
    }


def run_gui_event(
    pipeline: RucaPipeline,
    messages: list[dict[str, Any]],
    *,
    event_type: str,
    text: str = "",
    elapsed_minutes: float = 0.0,
    thought_council: ThoughtCouncil | None = None,
) -> dict[str, Any]:
    clean_text = ""
    if event_type == "user_message":
        clean_text = str(text or "").strip()
        if not clean_text:
            raise ValueError("user_message requires text")
        messages.append({"role": "user", "content": clean_text})
        result = pipeline.run_turn(clean_text)
    else:
        result = pipeline.run_event(event_type=event_type, elapsed_minutes=elapsed_minutes)

    thought_lines = _tick_thought_council(
        thought_council,
        event_type=event_type,
        text=clean_text if event_type == "user_message" else text,
        elapsed_minutes=elapsed_minutes,
    )
    trace = compact_trace(result)
    if result.assistant_text:
        messages.append({"role": "assistant", "content": result.assistant_text, "trace": trace})
    return {
        "result": result,
        "trace": trace,
        "thought_lines": thought_lines,
        "pending_text": clean_text if event_type == "user_message" and not result.assistant_text else "",
    }


def run_background_tick(
    pipeline: RucaPipeline,
    messages: list[dict[str, Any]],
    *,
    elapsed_minutes: float = 1.0 / 60.0,
    pending_text: str = "",
    pending_seconds: float = 0.0,
    thought_council: ThoughtCouncil | None = None,
) -> dict[str, Any]:
    if pending_text and pending_seconds >= 2.0:
        result = pipeline.run_event(event_type="delayed_speech", text=pending_text, elapsed_minutes=pending_seconds / 60.0)
        thought_lines = _tick_thought_council(
            thought_council,
            event_type="delayed_speech",
            text=pending_text,
            elapsed_minutes=pending_seconds / 60.0,
        )
        trace = compact_trace(result)
        if result.assistant_text:
            messages.append({"role": "assistant", "content": result.assistant_text, "trace": trace})
        return {"result": result, "trace": trace, "thought_lines": thought_lines, "released_pending": bool(result.assistant_text)}

    result = pipeline.run_event(event_type="silence_tick", elapsed_minutes=elapsed_minutes)
    thought_lines = _tick_thought_council(
        thought_council,
        event_type="silence_tick",
        elapsed_minutes=elapsed_minutes,
    )
    trace = compact_trace(result)
    if result.assistant_text:
        messages.append({"role": "assistant", "content": result.assistant_text, "trace": trace})
    return {"result": result, "trace": trace, "thought_lines": thought_lines, "released_pending": False}


def _tick_thought_council(
    thought_council: ThoughtCouncil | None,
    *,
    event_type: str,
    text: str = "",
    elapsed_minutes: float = 0.0,
) -> list[dict[str, Any]]:
    if thought_council is None:
        return []
    lines = thought_council.tick(
        event_kind=str(event_type or "silence_tick"),
        text=str(text or ""),
        elapsed_seconds=max(0.0, float(elapsed_minutes)) * 60.0,
    )
    return [line.to_record() for line in lines]


def _ensure_streamlit_state(st: Any) -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_trace" not in st.session_state:
        st.session_state.last_trace = None
    if "pipeline" not in st.session_state:
        config = build_llm_config(provider="gemini", api_key="", model_name="", base_url="")
        st.session_state.pipeline = make_pipeline(use_llm=False, llm_config=config)
        st.session_state.pipeline_signature = ("gemini", "", "", False)
    if "auto_tick_count" not in st.session_state:
        st.session_state.auto_tick_count = 0
    if "pending_speech_text" not in st.session_state:
        st.session_state.pending_speech_text = st.session_state.get("pending_reply_text", "")
    if "pending_speech_seconds" not in st.session_state:
        st.session_state.pending_speech_seconds = st.session_state.get("pending_reply_seconds", 0.0)
    if "thought_council" not in st.session_state:
        st.session_state.thought_council = ThoughtCouncil()


def _reset_pipeline(st: Any, *, provider: str, api_key: str, model_name: str, base_url: str, use_llm: bool) -> None:
    config = build_llm_config(provider=provider, api_key=api_key, model_name=model_name, base_url=base_url)
    st.session_state.pipeline = make_pipeline(use_llm=use_llm, llm_config=config)
    st.session_state.pipeline_signature = (provider, model_name, base_url, use_llm)
    st.session_state.messages = []
    st.session_state.last_trace = None
    st.session_state.pending_speech_text = ""
    st.session_state.pending_speech_seconds = 0.0
    st.session_state.thought_council = ThoughtCouncil()


def _render_app() -> None:
    import streamlit as st

    st.set_page_config(page_title="EmoNet v7", page_icon="E7", layout="wide")
    _ensure_streamlit_state(st)

    st.title("EmoNet v7")
    st.caption("Always-on neural runtime: 8D stim vector, cluster trace, per-neuron memory, trace episode speech.")

    with st.sidebar:
        st.header("Runtime")
        provider = st.selectbox("Provider", ["lm_studio", "gemini", "openai_compatible", "anthropic"], index=0)
        defaults = PROVIDER_DEFAULTS[provider]
        use_llm = st.toggle("LLM speech", value=False)
        st.session_state.auto_tick_enabled = st.toggle("Auto tick every second", value=st.session_state.get("auto_tick_enabled", True))
        api_key = st.text_input(
            "API key",
            value="",
            type="password",
            placeholder=f"Optional. Falls back to {defaults['api_key_env']}",
        )
        model_name = st.text_input("Model", value=str(defaults["model_name"]))
        base_url = st.text_input("Base URL", value=str(defaults["base_url"]))

        if st.button("Restart EmoNet", use_container_width=True):
            _reset_pipeline(st, provider=provider, api_key=api_key, model_name=model_name, base_url=base_url, use_llm=use_llm)
            st.rerun()

        st.divider()
        st.subheader("Clock")
        st.metric("Auto ticks", st.session_state.auto_tick_count)
        with st.expander("Manual environmental ticks", expanded=False):
            tick_col1, tick_col2 = st.columns(2)
            if tick_col1.button("Typing", use_container_width=True):
                payload = run_gui_event(
                    st.session_state.pipeline,
                    st.session_state.messages,
                    event_type="typing",
                    elapsed_minutes=0.1,
                    thought_council=st.session_state.thought_council,
                )
                st.session_state.last_trace = payload["trace"]
                st.rerun()
            if tick_col2.button("Processing", use_container_width=True):
                payload = run_gui_event(
                    st.session_state.pipeline,
                    st.session_state.messages,
                    event_type="processing",
                    elapsed_minutes=0.1,
                    thought_council=st.session_state.thought_council,
                )
                st.session_state.last_trace = payload["trace"]
                st.rerun()
            if st.button("Answering Tick", use_container_width=True):
                payload = run_gui_event(
                    st.session_state.pipeline,
                    st.session_state.messages,
                    event_type="answering",
                    elapsed_minutes=0.1,
                    thought_council=st.session_state.thought_council,
                )
                st.session_state.last_trace = payload["trace"]
                st.rerun()
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.last_trace = None
            st.session_state.pending_speech_text = ""
            st.session_state.pending_speech_seconds = 0.0
            st.session_state.thought_council = ThoughtCouncil()
            st.rerun()

        st.divider()
        key_state = "set" if api_key or os.environ.get(defaults["api_key_env"]) else "missing"
        st.metric("API key", key_state)
        if st.session_state.last_trace:
            st.json(st.session_state.last_trace, expanded=False)

    chat_col, thought_col, trace_col = st.columns([0.50, 0.25, 0.25], gap="large")
    with chat_col:
        for message in st.session_state.messages:
            role = "assistant" if message["role"] == "system" else message["role"]
            with st.chat_message(role):
                st.write(message["content"])
                if message.get("role") == "system":
                    st.caption("internal-only stimulation")

        prompt = st.chat_input("메시지를 입력하세요")
        if prompt:
            signature = (provider, model_name, base_url, use_llm)
            if st.session_state.get("pipeline_signature") != signature:
                _reset_pipeline(st, provider=provider, api_key=api_key, model_name=model_name, base_url=base_url, use_llm=use_llm)
            try:
                payload = run_gui_event(
                    st.session_state.pipeline,
                    st.session_state.messages,
                    event_type="user_message",
                    text=prompt,
                    thought_council=st.session_state.thought_council,
                )
                st.session_state.last_trace = payload["trace"]
                if payload.get("pending_text"):
                    st.session_state.pending_speech_text = payload["pending_text"]
                    st.session_state.pending_speech_seconds = 0.0
                else:
                    st.session_state.pending_speech_text = ""
                    st.session_state.pending_speech_seconds = 0.0
            except Exception as exc:
                st.error(str(exc))
            st.rerun()

    with thought_col:
        _render_thought_panel(st)

    with trace_col:
        _render_live_trace_panel(st)


def _render_thought_panel(st: Any) -> None:
    st.subheader("Thoughts")
    council = st.session_state.thought_council
    snapshots = council.snapshot_records()
    active_count = sum(1 for item in snapshots if float(item.get("mean_activity") or 0.0) > 0.10)
    st.metric("Models", len(snapshots))
    st.metric("Active", active_count)
    with st.expander("Emotion models", expanded=False):
        for item in snapshots:
            st.caption(
                f"{item['name']} · cluster={item['dominant_cluster_id']} · "
                f"activity={float(item.get('mean_activity') or 0.0):.2f}"
            )
    records = council.to_records(limit=12)
    if not records:
        st.info("아직 머릿속 대화가 시작되지 않았습니다.")
        return
    for record in reversed(records):
        st.markdown(f"**{record['speaker_name']}**")
        st.write(record["text"])
        st.caption(f"tick={record['tick_index']} · intensity={float(record['intensity']):.2f}")


def _render_live_trace_panel(st: Any) -> None:
    @st.fragment(run_every="1s")
    def live_trace() -> None:
        st.subheader("Trace")
        if st.session_state.get("auto_tick_enabled", True):
            if st.session_state.pending_speech_text:
                st.session_state.pending_speech_seconds += 1.0
            payload = run_background_tick(
                st.session_state.pipeline,
                st.session_state.messages,
                pending_text=st.session_state.pending_speech_text,
                pending_seconds=st.session_state.pending_speech_seconds,
                thought_council=st.session_state.thought_council,
            )
            st.session_state.last_trace = payload["trace"]
            st.session_state.auto_tick_count += 1
            if payload.get("released_pending"):
                st.session_state.pending_speech_text = ""
                st.session_state.pending_speech_seconds = 0.0
                st.rerun()

        trace = st.session_state.last_trace
        if trace:
            m1, m2, m3 = st.columns(3)
            m1.metric("Tick", trace["tick"])
            m2.metric("Stim", trace["stim_dim"])
            m3.metric("Cluster", trace["dominant_cluster"])
            st.metric("Neuron Memories", trace["stored_memory_count"])
            st.write(trace["summary"])
            if st.session_state.pending_speech_text:
                st.caption(f"아직 말이 올라오지 않음: {st.session_state.pending_speech_seconds:.0f}s")
            st.caption("The neural clock is flowing once per second while auto tick is enabled.")
        else:
            st.info("The neural clock starts when the page is open.")

    live_trace()


if __name__ == "__main__":
    _render_app()
