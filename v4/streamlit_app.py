from __future__ import annotations

import html
import json
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from emonet.chat_service import ChatGenerationConfig, ChatRuntimeConfig, build_chat_runtime, generate_chat_turn


st.set_page_config(page_title="EmoNet v4", layout="wide", initial_sidebar_state="expanded")

MESSAGES_KEY = "emonet_messages"
ERROR_KEY = "emonet_error"
USAGE_KEY = "emonet_usage"
AB_RESULTS_KEY = "emonet_ab_results"

CLAUDE_MODEL = "claude-sonnet-4-20250514"
CLAUDE_BASE_URL = "https://api.anthropic.com"
CLAUDE_INPUT_PRICE = 3.0
CLAUDE_OUTPUT_PRICE = 15.0
BETA_STIM_DIR = Path("outputs/beta_judging/targeted_episode_v3_vs_stim_2026-05-03")
BETA_EPISODE_DIR = Path("outputs/beta_judging/targeted_episode_v3_vs_episode_2026-05-03")


def _init_state() -> None:
    if MESSAGES_KEY not in st.session_state:
        st.session_state[MESSAGES_KEY] = []
    if ERROR_KEY not in st.session_state:
        st.session_state[ERROR_KEY] = ""
    if USAGE_KEY not in st.session_state:
        st.session_state[USAGE_KEY] = {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
    if AB_RESULTS_KEY not in st.session_state:
        st.session_state[AB_RESULTS_KEY] = {}


def _inject_css() -> None:
    st.markdown(
        """
<style>
    .stApp { background: #101418; color: #edf2f7; }
    section[data-testid="stSidebar"] { background: #151b22; border-right: 1px solid rgba(160, 174, 192, 0.16); }
    .block-container { padding-top: 1.4rem; }
    .hero {
        border-bottom: 1px solid rgba(160, 174, 192, 0.18);
        padding-bottom: 1rem;
        margin-bottom: 1rem;
    }
    .hero h1 { font-size: 2rem; line-height: 1.15; margin: 0 0 0.35rem 0; letter-spacing: 0; }
    .hero p { color: #b7c3d3; font-size: 1rem; line-height: 1.55; margin: 0; }
    .pill-row { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 0.8rem; }
    .pill {
        border: 1px solid rgba(102, 179, 122, 0.24);
        background: #10291f;
        color: #d8f7df;
        border-radius: 8px;
        padding: 0.38rem 0.68rem;
        font-size: 0.86rem;
    }
    .bubble {
        border: 1px solid rgba(160, 174, 192, 0.14);
        border-radius: 8px;
        padding: 0.9rem 1rem;
        line-height: 1.7;
    }
    .bubble.user { background: #1f4b3d; color: #f4fbf6; }
    .bubble.assistant { background: #202934; color: #edf2f7; }
    .compact-note { color: #aeb8c6; font-size: 0.9rem; line-height: 1.5; }
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
    cols = st.columns(3)
    cols[0].metric("Session spent", f"${spent:.4f}")
    cols[1].metric("Budget left", f"${max(0.0, budget_usd - spent):.2f}")
    cols[2].metric("Tokens", f"{usage.get('input_tokens', 0)} in / {usage.get('output_tokens', 0)} out")


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
<div class="hero">
  <h1>EmoNet v4</h1>
  <p>Claude API로 답변을 생성하는 안정판입니다. 불안정한 연구 대시보드와 고급 런타임 설정은 제거했습니다.</p>
  <div class="pill-row">
    <span class="pill">provider: anthropic</span>
    <span class="pill">model: claude-sonnet-4-20250514</span>
    <span class="pill">mode: hybrid_trace</span>
    <span class="pill">style: extended40</span>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )


def _render_message(role: str, content: str) -> None:
    css_role = "user" if role == "user" else "assistant"
    with st.chat_message(role):
        safe = html.escape(str(content)).replace("\n", "<br>")
        st.markdown(f"<div class='bubble {css_role}'>{safe}</div>", unsafe_allow_html=True)


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


def _render_chat(api_key: str, budget_usd: float) -> None:
    _render_header()
    _usage_metrics(budget_usd)
    if st.session_state[ERROR_KEY]:
        st.error(st.session_state[ERROR_KEY])

    examples = [
        "회의에서 또 나만 공개적으로 무시당했어. 바로 따지고 싶을 정도로 거슬려.",
        "이번 주 내내 야근이라 머리가 멍하고 다 놓아버리고 싶어.",
        "잘된 일인데도 이상하게 기쁘기보다 불안하고 예민해.",
    ]
    if not st.session_state[MESSAGES_KEY]:
        cols = st.columns(3)
        for idx, example in enumerate(examples):
            if cols[idx].button(example, use_container_width=True):
                try:
                    st.session_state[ERROR_KEY] = ""
                    _generate_reply(example, api_key)
                except Exception as exc:
                    st.session_state[ERROR_KEY] = str(exc)
                st.rerun()

    for message in st.session_state[MESSAGES_KEY]:
        _render_message(str(message.get("role", "")), str(message.get("content", "")))

    prompt = st.chat_input("메시지를 입력하세요")
    if prompt:
        try:
            st.session_state[ERROR_KEY] = ""
            _generate_reply(prompt.strip(), api_key)
        except Exception as exc:
            st.session_state[ERROR_KEY] = str(exc)
        st.rerun()


def _package_paths(kind: str) -> tuple[Path, Path]:
    if kind == "main":
        return (
            BETA_STIM_DIR / "human_eval_episode_v3_vs_stim.csv",
            BETA_STIM_DIR / "answer_key_episode_v3_vs_stim.json",
        )
    return (
        BETA_EPISODE_DIR / "human_eval_episode_v3_vs_episode.csv",
        BETA_EPISODE_DIR / "answer_key_episode_v3_vs_episode.json",
    )


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


def _analyze_results(df: pd.DataFrame, key_path: Path, target_condition: str = "episode_trace_v3") -> dict[str, Any]:
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
            if label_map[winner] == target_condition:
                wins += 1
            else:
                losses += 1
        else:
            invalid += 1
    valid = wins + ties + losses
    non_tie = wins + losses
    return {
        "valid_rows": valid,
        "invalid_rows": invalid,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "win_rate": wins / valid if valid else 0.0,
        "non_tie_win_rate": wins / non_tie if non_tie else 0.0,
        "sign_test_p": _sign_test_p(wins, losses),
    }


def _render_ab() -> None:
    st.header("Human A/B")
    st.markdown("<p class='compact-note'>평가용 CSV를 한 행씩 채우고, 완료 후 filled CSV를 받는 최소 기능만 남겼습니다.</p>", unsafe_allow_html=True)
    kind = st.radio("Package", ["main", "secondary"], format_func=lambda v: "v3 vs stim_only" if v == "main" else "v3 vs episode_trace", horizontal=True)
    csv_path, key_path = _package_paths(kind)
    df = _load_eval_csv(csv_path)
    result_key = f"{kind}:{csv_path.name}"
    saved = dict(st.session_state[AB_RESULTS_KEY].get(result_key, {}))

    row_number = st.number_input("Row", min_value=1, max_value=len(df), value=1, step=1)
    row = df.iloc[int(row_number) - 1].to_dict()
    eval_id = str(row["eval_id"])
    current = dict(saved.get(eval_id, {}))

    st.caption(f"{int(row_number)}/{len(df)} · completed {sum(1 for x in saved.values() if x.get('winner'))}/{len(df)}")
    st.text_area("User input", row["text"], height=120, disabled=True)
    left, right = st.columns(2)
    left.text_area("Candidate A", row["candidate_a"], height=220, disabled=True)
    right.text_area("Candidate B", row["candidate_b"], height=220, disabled=True)

    winner_options = ["", "candidate_a", "candidate_b", "tie"]
    winner = st.radio(
        "Winner",
        winner_options,
        index=winner_options.index(current.get("winner", "")) if current.get("winner", "") in winner_options else 0,
        horizontal=True,
    )
    confidence = st.slider("Confidence", 1, 5, int(current.get("confidence", 3) or 3))
    reason = st.text_area("Reason", str(current.get("reason", "")), height=80)

    c1, c2, c3 = st.columns(3)
    if c1.button("Save", use_container_width=True):
        saved[eval_id] = {"winner": winner, "confidence": confidence, "reason": reason}
        all_results = dict(st.session_state[AB_RESULTS_KEY])
        all_results[result_key] = saved
        st.session_state[AB_RESULTS_KEY] = all_results
        st.rerun()
    if c2.button("Previous", use_container_width=True, disabled=int(row_number) <= 1):
        st.session_state["Row"] = int(row_number) - 1
        st.rerun()
    if c3.button("Next", use_container_width=True, disabled=int(row_number) >= len(df)):
        st.session_state["Row"] = int(row_number) + 1
        st.rerun()

    export_df = df.copy()
    for idx, export_row in export_df.iterrows():
        row_saved = saved.get(str(export_row["eval_id"]), {})
        for column in ["winner", "confidence", "reason"]:
            if row_saved:
                export_df.at[idx, column] = str(row_saved.get(column, ""))

    summary = _analyze_results(export_df, key_path)
    s1, s2, s3 = st.columns(3)
    s1.metric("Win / Tie / Loss", f"{summary['wins']} / {summary['ties']} / {summary['losses']}")
    s2.metric("Win rate", f"{summary['win_rate']:.3f}")
    s3.metric("Sign-test p", f"{summary['sign_test_p']:.4f}")
    st.download_button(
        "Download filled CSV",
        data=export_df.to_csv(index=False).encode("utf-8-sig"),
        file_name=csv_path.stem + "_filled.csv",
        mime="text/csv",
        use_container_width=True,
    )


def _render_sidebar() -> tuple[str, float]:
    with st.sidebar:
        st.header("Claude")
        api_key = _api_key_from_input(st.text_input("API key", type="password"))
        budget = st.number_input("Budget", min_value=0.0, max_value=1000.0, value=22.0, step=0.5)
        st.caption("API key는 파일에 저장하지 않습니다.")
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
    view = st.radio("View", ["Chat", "Human A/B"], horizontal=True, label_visibility="collapsed")
    if view == "Chat":
        _render_chat(api_key, budget)
    else:
        _render_ab()


if __name__ == "__main__":
    main()
