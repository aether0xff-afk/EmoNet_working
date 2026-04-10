from __future__ import annotations

import html
import json
from pathlib import Path

import streamlit as st

from emonet.chat_service import (
    CONDITIONING_MODES,
    ChatGenerationConfig,
    ChatRuntimeConfig,
    available_style_profiles,
    build_chat_runtime,
    generate_chat_turn,
    parse_episode_payload_text,
)


st.set_page_config(
    page_title="EmoNet Chat",
    layout="wide",
    initial_sidebar_state="expanded",
)

SETTINGS_KEY = "emonet_chat_settings"
MESSAGES_KEY = "emonet_chat_messages"
ERROR_KEY = "emonet_chat_error"
SHOW_DIAGNOSTICS_KEY = "emonet_chat_show_diagnostics"
SHOW_PROMPT_KEY = "emonet_chat_show_prompt"


def _default_settings() -> dict[str, object]:
    runtime = ChatRuntimeConfig()
    generation = ChatGenerationConfig()
    return {
        "dataset_csv": str(runtime.dataset_csv),
        "benchmark_csv": str(runtime.benchmark_csv),
        "model_cache_path": str(runtime.model_cache_path),
        "z_encoder_path": str(runtime.z_encoder_path),
        "zs_model_path": str(runtime.zs_model_path),
        "seed": int(runtime.seed),
        "z_dim": int(runtime.z_dim),
        "z_encoder_mode": str(runtime.z_encoder_mode),
        "prompt_template": str(generation.prompt_template),
        "base_url": str(generation.base_url),
        "model_name": str(generation.model_name),
        "api_key": "",
        "style_profile": str(generation.style_profile),
        "conditioning_mode": str(generation.conditioning_mode),
        "response_temperature": float(generation.response_temperature),
        "response_max_retries": int(generation.response_max_retries),
        "max_tokens": int(generation.max_tokens),
        "timeout_sec": int(generation.timeout_sec),
        "reasoning_effort": "",
        "history_turns": int(generation.history_turns),
    }


def _init_session_state() -> None:
    if SETTINGS_KEY not in st.session_state:
        st.session_state[SETTINGS_KEY] = _default_settings()
    if MESSAGES_KEY not in st.session_state:
        st.session_state[MESSAGES_KEY] = []
    if ERROR_KEY not in st.session_state:
        st.session_state[ERROR_KEY] = ""
    if SHOW_DIAGNOSTICS_KEY not in st.session_state:
        st.session_state[SHOW_DIAGNOSTICS_KEY] = True
    if SHOW_PROMPT_KEY not in st.session_state:
        st.session_state[SHOW_PROMPT_KEY] = False


@st.cache_resource(show_spinner="EmoNet 자산을 불러오는 중입니다...")
def _load_runtime(
    dataset_csv: str,
    benchmark_csv: str,
    model_cache_path: str,
    z_encoder_path: str,
    zs_model_path: str,
    seed: int,
    z_dim: int,
    z_encoder_mode: str,
):
    return build_chat_runtime(
        ChatRuntimeConfig(
            dataset_csv=Path(dataset_csv),
            benchmark_csv=Path(benchmark_csv),
            model_cache_path=Path(model_cache_path),
            z_encoder_path=Path(z_encoder_path),
            zs_model_path=Path(zs_model_path),
            seed=int(seed),
            z_dim=int(z_dim),
            z_encoder_mode=str(z_encoder_mode),
        )
    )


def _inject_css() -> None:
    st.markdown(
        """
<style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(43, 138, 90, 0.14), transparent 34%),
            radial-gradient(circle at top right, rgba(190, 151, 92, 0.12), transparent 28%),
            linear-gradient(180deg, #0c1117 0%, #121923 55%, #171f2a 100%);
        color: #edf2f7;
    }
    section[data-testid="stSidebar"] {
        background:
            linear-gradient(180deg, rgba(11, 17, 24, 0.96) 0%, rgba(15, 24, 34, 0.96) 100%);
        border-right: 1px solid rgba(151, 165, 184, 0.16);
    }
    .app-shell {
        max-width: 980px;
        margin: 0 auto 1.25rem auto;
    }
    .hero-card {
        padding: 1.2rem 1.3rem 1rem 1.3rem;
        border-radius: 24px;
        border: 1px solid rgba(164, 178, 198, 0.14);
        background: linear-gradient(135deg, rgba(13, 19, 27, 0.94), rgba(22, 30, 41, 0.9));
        box-shadow: 0 24px 60px rgba(0, 0, 0, 0.24);
        margin-bottom: 1rem;
    }
    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        line-height: 1.1;
        margin-bottom: 0.35rem;
        letter-spacing: -0.02em;
    }
    .hero-subtitle {
        color: #b7c3d3;
        font-size: 0.98rem;
        line-height: 1.55;
        margin-bottom: 0.85rem;
    }
    .hero-badges {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
    }
    .hero-badge {
        border-radius: 999px;
        padding: 0.38rem 0.72rem;
        border: 1px solid rgba(123, 211, 137, 0.18);
        background: rgba(39, 84, 61, 0.24);
        color: #d5f5dc;
        font-size: 0.82rem;
    }
    .chat-bubble {
        padding: 0.95rem 1rem;
        border-radius: 22px;
        border: 1px solid rgba(167, 180, 196, 0.12);
        line-height: 1.72;
        box-shadow: 0 16px 40px rgba(0, 0, 0, 0.14);
    }
    .chat-bubble.user {
        background: linear-gradient(135deg, rgba(31, 66, 54, 0.95), rgba(37, 95, 71, 0.88));
        color: #f2f9f4;
    }
    .chat-bubble.assistant {
        background: linear-gradient(135deg, rgba(24, 31, 40, 0.96), rgba(35, 45, 58, 0.92));
        color: #edf2f7;
    }
    .suggestion-card {
        border-radius: 18px;
        border: 1px solid rgba(167, 180, 196, 0.14);
        background: rgba(17, 24, 33, 0.84);
        padding: 0.9rem 1rem;
        min-height: 132px;
    }
    .suggestion-title {
        font-size: 0.94rem;
        color: #f3f6fa;
        margin-bottom: 0.4rem;
        font-weight: 600;
    }
    .suggestion-body {
        font-size: 0.88rem;
        color: #aeb8c6;
        line-height: 1.55;
    }
    .sidebar-note {
        color: #9eabba;
        font-size: 0.84rem;
        line-height: 1.55;
    }
</style>
        """,
        unsafe_allow_html=True,
    )


def _render_header(settings: dict[str, object]) -> None:
    st.markdown(
        (
            "<div class='app-shell'>"
            "<div class='hero-card'>"
            "<div class='hero-title'>EmoNet Chat</div>"
            "<div class='hero-subtitle'>"
            "입력 문장을 EmoNet branch dynamics로 먼저 읽고, 그 결과를 OpenAI-compatible 모델에 condition하여 "
            "대화형 답변을 만드는 Streamlit 인터페이스다."
            "</div>"
            "<div class='hero-badges'>"
            f"<span class='hero-badge'>mode: {html.escape(str(settings['conditioning_mode']))}</span>"
            f"<span class='hero-badge'>style: {html.escape(str(settings['style_profile']))}</span>"
            f"<span class='hero-badge'>llm: {html.escape(str(settings['model_name']))}</span>"
            "</div>"
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def _render_suggestions() -> str | None:
    examples = [
        ("공개적 무시", "회의에서 또 나만 공개적으로 무시당했어. 바로 따지고 싶을 정도로 거슬려."),
        ("소진과 피로", "이번 주 내내 야근이라 머리가 멍하고 다 놓아버리고 싶어."),
        ("양가감정", "잘된 일인데도 이상하게 기쁘기보다 불안하고 예민해."),
    ]
    prompt = None
    st.markdown("<div class='app-shell'>", unsafe_allow_html=True)
    cols = st.columns(len(examples))
    for idx, (title, example) in enumerate(examples):
        with cols[idx]:
            st.markdown(
                (
                    "<div class='suggestion-card'>"
                    f"<div class='suggestion-title'>{html.escape(title)}</div>"
                    f"<div class='suggestion-body'>{html.escape(example)}</div>"
                    "</div>"
                ),
                unsafe_allow_html=True,
            )
            if st.button("이 예시로 시작", key=f"example_{idx}", use_container_width=True):
                prompt = example
    st.markdown("</div>", unsafe_allow_html=True)
    return prompt


def _render_message(role: str, content: str) -> None:
    css_class = "user" if role == "user" else "assistant"
    with st.chat_message(role):
        st.markdown(
            f"<div class='chat-bubble {css_class}'>{html.escape(content).replace(chr(10), '<br>')}</div>",
            unsafe_allow_html=True,
        )


def _render_assistant_details(message: dict[str, object], show_prompt: bool) -> None:
    record = message.get("record")
    if not isinstance(record, dict):
        return
    with st.expander("감정 진단", expanded=False):
        top_cols = st.columns(3)
        top_cols[0].metric("Dominant Branch", int(record.get("dominant_branch_len", 0)))
        top_cols[1].metric("Ticks", int(record.get("ticks_run", 0)))
        top_cols[2].metric("Retries", int(record.get("response_retry_count", 0)))
        if record.get("style_tags"):
            st.caption("Style tags")
            st.write(", ".join(str(tag) for tag in record["style_tags"]))
        if record.get("style_summary_text"):
            st.caption("Style summary")
            st.write(str(record["style_summary_text"]))
        if record.get("expression_cues_text"):
            st.caption("Expression cues")
            st.write(str(record["expression_cues_text"]))
        if record.get("trace_summary_text"):
            st.caption("Trace summary")
            st.write(str(record["trace_summary_text"]))
        if record.get("appraisal_summary_text"):
            st.caption("Appraisal summary")
            st.write(str(record["appraisal_summary_text"]))
        if record.get("episode_summary_text"):
            st.caption("Episode summary")
            st.write(str(record["episode_summary_text"]))
        details = {
            "stim_vec": record.get("stim_vec", []),
            "termination_reason": record.get("termination_reason", ""),
            "anti_softening_rules": record.get("anti_softening_rules", []),
            "grounding_rules": record.get("grounding_rules", []),
            "response_validation_errors": record.get("response_validation_errors", []),
        }
        st.json(details, expanded=False)
    if show_prompt:
        with st.expander("생성 프롬프트", expanded=False):
            st.code(str(record.get("generation_prompt", "")), language="markdown")


def _build_runtime_config(settings: dict[str, object]) -> ChatRuntimeConfig:
    return ChatRuntimeConfig(
        dataset_csv=Path(str(settings["dataset_csv"])),
        benchmark_csv=Path(str(settings["benchmark_csv"])),
        model_cache_path=Path(str(settings["model_cache_path"])),
        z_encoder_path=Path(str(settings["z_encoder_path"])),
        zs_model_path=Path(str(settings["zs_model_path"])),
        seed=int(settings["seed"]),
        z_dim=int(settings["z_dim"]),
        z_encoder_mode=str(settings["z_encoder_mode"]),
    )


def _build_generation_config(settings: dict[str, object]) -> ChatGenerationConfig:
    reasoning_effort = str(settings.get("reasoning_effort", "") or "").strip() or None
    api_key = str(settings.get("api_key", "") or "").strip() or None
    return ChatGenerationConfig(
        base_url=str(settings["base_url"]),
        model_name=str(settings["model_name"]),
        api_key=api_key,
        prompt_template=Path(str(settings["prompt_template"])),
        style_profile=str(settings["style_profile"]),
        conditioning_mode=str(settings["conditioning_mode"]),
        response_temperature=float(settings["response_temperature"]),
        response_max_retries=int(settings["response_max_retries"]),
        max_tokens=int(settings["max_tokens"]),
        timeout_sec=int(settings["timeout_sec"]),
        reasoning_effort=reasoning_effort,
        history_turns=int(settings["history_turns"]),
    )


def _transcript_json(settings: dict[str, object], messages: list[dict[str, object]]) -> str:
    payload = {"settings": settings, "messages": messages}
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _render_sidebar(settings: dict[str, object]) -> tuple[dict[str, object], dict[str, object] | None]:
    active_settings = dict(settings)
    episode_payload = None
    with st.sidebar:
        st.markdown("## App Settings")
        st.markdown(
            "<div class='sidebar-note'>"
            "LLM endpoint, conditioning mode, artifact 경로를 여기서 고정한 뒤 대화를 시작한다."
            "</div>",
            unsafe_allow_html=True,
        )
        with st.form("settings_form"):
            base_url = st.text_input("Base URL", value=str(active_settings["base_url"]))
            model_name = st.text_input("Model Name", value=str(active_settings["model_name"]))
            api_key = st.text_input("API Key", value=str(active_settings["api_key"]), type="password")
            conditioning_mode = st.selectbox(
                "Conditioning Mode",
                options=list(CONDITIONING_MODES),
                index=list(CONDITIONING_MODES).index(str(active_settings["conditioning_mode"])),
            )
            style_profiles = list(available_style_profiles())
            style_profile = st.selectbox(
                "Style Profile",
                options=style_profiles,
                index=style_profiles.index(str(active_settings["style_profile"])),
            )
            history_turns = st.slider("Recent Turns", min_value=0, max_value=8, value=int(active_settings["history_turns"]))
            response_temperature = st.slider(
                "Response Temperature",
                min_value=0.0,
                max_value=1.0,
                value=float(active_settings["response_temperature"]),
                step=0.05,
            )
            response_max_retries = st.number_input(
                "Max Retries",
                min_value=0,
                max_value=6,
                value=int(active_settings["response_max_retries"]),
                step=1,
            )
            max_tokens = st.number_input(
                "Max Tokens",
                min_value=64,
                max_value=4096,
                value=int(active_settings["max_tokens"]),
                step=64,
            )
            timeout_sec = st.number_input(
                "Timeout (sec)",
                min_value=10,
                max_value=600,
                value=int(active_settings["timeout_sec"]),
                step=10,
            )
            reasoning_effort = st.text_input("Reasoning Effort", value=str(active_settings["reasoning_effort"]))
            with st.expander("Runtime Paths", expanded=False):
                dataset_csv = st.text_input("Dataset CSV", value=str(active_settings["dataset_csv"]))
                benchmark_csv = st.text_input("Benchmark CSV", value=str(active_settings["benchmark_csv"]))
                model_cache_path = st.text_input("Stim Cache Path", value=str(active_settings["model_cache_path"]))
                z_encoder_path = st.text_input("Z Encoder Path", value=str(active_settings["z_encoder_path"]))
                zs_model_path = st.text_input("ZS Decoder Path", value=str(active_settings["zs_model_path"]))
                prompt_template = st.text_input("Prompt Template", value=str(active_settings["prompt_template"]))
            with st.expander("Model Parameters", expanded=False):
                seed = st.number_input("Seed", min_value=0, max_value=999999, value=int(active_settings["seed"]), step=1)
                z_dim = st.number_input("Z Dimension", min_value=8, max_value=256, value=int(active_settings["z_dim"]), step=8)
                z_encoder_mode = st.selectbox(
                    "Z Encoder Mode",
                    options=["auto", "stat", "transformer"],
                    index=["auto", "stat", "transformer"].index(str(active_settings["z_encoder_mode"])),
                )
            submitted = st.form_submit_button("설정 적용", use_container_width=True)
        if submitted:
            active_settings = {
                "base_url": base_url,
                "model_name": model_name,
                "api_key": api_key,
                "conditioning_mode": conditioning_mode,
                "style_profile": style_profile,
                "history_turns": int(history_turns),
                "response_temperature": float(response_temperature),
                "response_max_retries": int(response_max_retries),
                "max_tokens": int(max_tokens),
                "timeout_sec": int(timeout_sec),
                "reasoning_effort": reasoning_effort,
                "dataset_csv": dataset_csv,
                "benchmark_csv": benchmark_csv,
                "model_cache_path": model_cache_path,
                "z_encoder_path": z_encoder_path,
                "zs_model_path": zs_model_path,
                "prompt_template": prompt_template,
                "seed": int(seed),
                "z_dim": int(z_dim),
                "z_encoder_mode": z_encoder_mode,
            }
            st.session_state[SETTINGS_KEY] = active_settings
            st.session_state[ERROR_KEY] = ""
            st.rerun()
        episode_file = st.file_uploader("Episode JSON", type=["json"])
        if episode_file is not None:
            try:
                episode_payload = parse_episode_payload_text(episode_file.getvalue().decode("utf-8"))
                st.success("episode payload loaded")
            except Exception as exc:
                st.error(f"episode payload error: {exc}")
        st.toggle("내부 진단 보기", key=SHOW_DIAGNOSTICS_KEY)
        st.toggle("생성 프롬프트 보기", key=SHOW_PROMPT_KEY)
        if st.button("대화 초기화", use_container_width=True):
            st.session_state[MESSAGES_KEY] = []
            st.session_state[ERROR_KEY] = ""
            st.rerun()
        st.download_button(
            "대화 JSON 내려받기",
            data=_transcript_json(active_settings, st.session_state[MESSAGES_KEY]),
            file_name="emonet_chat_transcript.json",
            mime="application/json",
            use_container_width=True,
        )
    return active_settings, episode_payload


def _process_prompt(prompt: str, settings: dict[str, object], episode_payload: dict[str, object] | None) -> None:
    history = [
        {"role": str(message.get("role", "")), "content": str(message.get("content", ""))}
        for message in st.session_state[MESSAGES_KEY]
    ]
    st.session_state[MESSAGES_KEY].append({"role": "user", "content": prompt})
    runtime_config = _build_runtime_config(settings)
    generation_config = _build_generation_config(settings)
    with st.spinner("응답을 생성하는 중입니다..."):
        runtime = _load_runtime(
            dataset_csv=str(runtime_config.dataset_csv),
            benchmark_csv=str(runtime_config.benchmark_csv),
            model_cache_path=str(runtime_config.model_cache_path),
            z_encoder_path=str(runtime_config.z_encoder_path),
            zs_model_path=str(runtime_config.zs_model_path),
            seed=int(runtime_config.seed),
            z_dim=int(runtime_config.z_dim),
            z_encoder_mode=str(runtime_config.z_encoder_mode),
        )
        result = generate_chat_turn(
            runtime=runtime,
            generation_config=generation_config,
            input_text=prompt,
            history=history,
            episode_payload=episode_payload,
        )
    st.session_state[MESSAGES_KEY].append(
        {
            "role": "assistant",
            "content": result.assistant_text,
            "record": result.record,
        }
    )


def main() -> None:
    _init_session_state()
    _inject_css()
    settings, episode_payload = _render_sidebar(st.session_state[SETTINGS_KEY])
    _render_header(settings)
    if st.session_state[ERROR_KEY]:
        st.error(st.session_state[ERROR_KEY])

    starter_prompt = None
    if not st.session_state[MESSAGES_KEY]:
        starter_prompt = _render_suggestions()

    st.markdown("<div class='app-shell'>", unsafe_allow_html=True)
    for message in st.session_state[MESSAGES_KEY]:
        _render_message(str(message.get("role", "")), str(message.get("content", "")))
        if str(message.get("role", "")) == "assistant" and bool(st.session_state[SHOW_DIAGNOSTICS_KEY]):
            _render_assistant_details(message, bool(st.session_state[SHOW_PROMPT_KEY]))
    st.markdown("</div>", unsafe_allow_html=True)

    prompt = st.chat_input("메시지를 입력하세요")
    active_prompt = starter_prompt or prompt
    if not active_prompt:
        return
    try:
        st.session_state[ERROR_KEY] = ""
        _process_prompt(str(active_prompt).strip(), settings, episode_payload)
    except Exception as exc:
        st.session_state[ERROR_KEY] = str(exc)
    st.rerun()


if __name__ == "__main__":
    main()
