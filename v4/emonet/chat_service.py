from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from .core import LinearZtoSDecoder
from .episode_conditioning import augment_profile_with_episode
from .legacy_cli import (
    DEFAULT_STYLE_PROFILE,
    MODEL_OPTIONAL_CONFIG_FIELDS,
    STYLE_AXIS_PROFILES,
    build_conditioned_generation_prompt,
    build_model,
    ensure_model_server_ready,
    infer_style_profile,
    validate_plain_response_text,
)
from .llm_api import request_plain_text_response
from .paths import default_benchmark_csv, default_stim_dataset_csv, project_root


CONDITIONING_MODES = (
    "style",
    "raw_trace",
    "appraisal_trace",
    "hybrid_trace",
    "episode_trace",
    "hybrid_episode",
)
DEFAULT_MODEL_CACHE_PATH = project_root() / "artifacts" / "ridge_stim_encoder.joblib"
DEFAULT_PROMPT_TEMPLATE_PATH = project_root() / "prompts" / "response_generation_prompt.md"
DEFAULT_REQUEST_SYSTEM_PROMPT = "Return a plain Korean response only. Do not return JSON."
DEFAULT_RESPONSE_RETRY_INSTRUCTION = (
    "직전 응답은 반복, 미완성 문장, bullet/JSON, 혹은 부자연스러운 출력 때문에 거부되었다. "
    "같은 문장이나 핵심 구절을 반복하지 말고, 마지막 문장은 완결된 한국어 평문으로 끝내라."
)


def _first_existing_path(candidates: Sequence[Path], label: str) -> Path:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    joined = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"{label} not found. looked for: {joined}")


def resolve_default_z_encoder_path() -> Path:
    return _first_existing_path(
        [
            project_root() / "artifacts" / "dominant_branch_encoder_extended40_calref_v1.pt",
            project_root() / "artifacts" / "dominant_branch_encoder.pt",
        ],
        "z encoder checkpoint",
    )


def resolve_default_zs_model_path() -> Path:
    return _first_existing_path(
        [
            project_root() / "artifacts" / "z_to_s_decoder_extended40_calref_v1.npz",
            project_root() / "artifacts" / "z_to_s_decoder.npz",
        ],
        "z-to-s decoder checkpoint",
    )


def resolve_default_prompt_template_path() -> Path:
    return _first_existing_path([DEFAULT_PROMPT_TEMPLATE_PATH], "response generation prompt template")


def available_style_profiles() -> tuple[str, ...]:
    return tuple(sorted(STYLE_AXIS_PROFILES))


@dataclass(frozen=True)
class ChatRuntimeConfig:
    dataset_csv: Path = field(default_factory=default_stim_dataset_csv)
    benchmark_csv: Path = field(default_factory=default_benchmark_csv)
    model_cache_path: Path = field(default_factory=lambda: DEFAULT_MODEL_CACHE_PATH)
    z_encoder_path: Path = field(default_factory=resolve_default_z_encoder_path)
    zs_model_path: Path = field(default_factory=resolve_default_zs_model_path)
    seed: int = 42
    z_dim: int = 64
    z_encoder_mode: str = "auto"


@dataclass(frozen=True)
class ChatGenerationConfig:
    base_url: str = "http://127.0.0.1:11434/v1"
    model_name: str = "gpt-oss:20b"
    api_key: str | None = None
    prompt_template: Path = field(default_factory=resolve_default_prompt_template_path)
    style_profile: str = DEFAULT_STYLE_PROFILE
    conditioning_mode: str = "hybrid_trace"
    response_temperature: float = 0.45
    response_max_retries: int = 2
    max_tokens: int = 600
    timeout_sec: int = 180
    reasoning_effort: str | None = None
    history_turns: int = 4


@dataclass
class EmoNetChatRuntime:
    config: ChatRuntimeConfig
    model: Any
    decoder: LinearZtoSDecoder


@dataclass(frozen=True)
class ChatTurnResult:
    assistant_text: str
    record: dict[str, Any]


def _build_model_args(config: ChatRuntimeConfig) -> SimpleNamespace:
    payload: dict[str, Any] = {field_name: None for field_name in MODEL_OPTIONAL_CONFIG_FIELDS}
    payload.update(
        {
            "dataset_csv": str(config.dataset_csv),
            "benchmark_csv": str(config.benchmark_csv),
            "model_cache_path": str(config.model_cache_path),
            "max_samples": None,
            "force_refit": False,
            "seed": int(config.seed),
            "z_dim": int(config.z_dim),
            "z_encoder_mode": str(config.z_encoder_mode),
            "z_encoder_path": str(config.z_encoder_path),
        }
    )
    return SimpleNamespace(**payload)


def build_chat_runtime(config: ChatRuntimeConfig | None = None) -> EmoNetChatRuntime:
    active_config = config or ChatRuntimeConfig()
    if not active_config.zs_model_path.exists():
        raise FileNotFoundError(f"decoder checkpoint not found: {active_config.zs_model_path}")
    model = build_model(_build_model_args(active_config))
    model.stim_encoder.ensure_fitted()
    decoder = LinearZtoSDecoder.load(active_config.zs_model_path)
    return EmoNetChatRuntime(config=active_config, model=model, decoder=decoder)


def parse_episode_payload_text(payload_text: str) -> dict[str, Any]:
    payload = json.loads(str(payload_text or "").strip())
    if not isinstance(payload, dict):
        raise ValueError("episode payload must be a JSON object")
    return payload


def _compact_text(value: object, limit: int = 220) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def build_recent_dialogue_block(history: Sequence[Mapping[str, Any]] | None, max_turns: int) -> str:
    if not history or max_turns <= 0:
        return ""
    normalized: list[str] = []
    recent_messages = list(history)[-max(0, int(max_turns) * 2) :]
    for message in recent_messages:
        role = str(message.get("role", "")).strip().lower()
        content = _compact_text(message.get("content", ""))
        if not content:
            continue
        if role == "user":
            normalized.append(f"- USER: {content}")
        elif role == "assistant":
            normalized.append(f"- ASSISTANT: {content}")
    return "\n".join(normalized)


def inject_chat_history(prompt: str, history: Sequence[Mapping[str, Any]] | None, max_turns: int) -> str:
    dialogue_block = build_recent_dialogue_block(history, max_turns)
    if not dialogue_block:
        return prompt
    return "\n".join(
        [
            "[RECENT_DIALOGUE]",
            dialogue_block,
            "",
            prompt,
            "",
            "[CHAT_CONTINUITY_RULES]",
            "- RECENT_DIALOGUE는 맥락 유지를 위한 참고 정보다.",
            "- 가장 최근 USER_INPUT에 직접 답한다.",
            "- 직전 ASSISTANT 문장을 기계적으로 반복하지 않는다.",
            "- 앞선 대화와 충돌하지 않되, 감정 결은 최신 USER_INPUT을 우선한다.",
        ]
    )


def _float_list(value: object) -> list[float]:
    return np.asarray(value, dtype=float).tolist()


def _string_list(value: object) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return []


def _serialize_profile(
    *,
    input_text: str,
    assistant_text: str,
    profile: Mapping[str, Any],
    prompt: str,
    prompt_sections: str,
    response_meta: Mapping[str, Any],
    runtime_config: ChatRuntimeConfig,
    generation_config: ChatGenerationConfig,
    chat_history_excerpt: str,
) -> dict[str, Any]:
    return {
        "input_text": str(input_text),
        "llm_response": str(assistant_text),
        "stim_vec": _float_list(profile.get("stim_vec", [])),
        "dominant_branch_len": int(profile.get("dominant_branch_len", 0)),
        "z": _float_list(profile.get("z", [])),
        "s_pred": _float_list(profile.get("s_pred", [])),
        "style_tags": _string_list(profile.get("style_tags", [])),
        "style_summary": dict(profile.get("style_summary", {})),
        "style_summary_text": str(profile.get("style_summary_text", "")),
        "expression_cues_text": str(profile.get("expression_cues_text", "")),
        "trace_summary_text": str(profile.get("trace_summary_text", "")),
        "trace_lines": _string_list(profile.get("trace_lines", [])),
        "appraisal_summary_text": str(profile.get("appraisal_summary_text", "")),
        "appraisal_lines": _string_list(profile.get("appraisal_lines", [])),
        "appraisal_target": str(profile.get("appraisal_target", "")),
        "appraisal_tendency": str(profile.get("appraisal_tendency", "")),
        "episode_label": str(profile.get("episode_label", "")),
        "episode_summary_text": str(profile.get("episode_summary_text", "")),
        "episode_lite_text": str(profile.get("episode_lite_text", "")),
        "episode_lines": _string_list(profile.get("episode_lines", [])),
        "episode_lite_lines": _string_list(profile.get("episode_lite_lines", [])),
        "anti_softening_mode": str(profile.get("anti_softening_mode", "")),
        "anti_softening_rules": _string_list(profile.get("anti_softening_rules", [])),
        "grounding_mode": str(profile.get("grounding_mode", "")),
        "grounding_rules": _string_list(profile.get("grounding_rules", [])),
        "ticks_run": int(profile.get("ticks_run", 0)),
        "termination_reason": str(profile.get("termination_reason", "")),
        "conditioning_mode": str(generation_config.conditioning_mode),
        "style_profile": str(generation_config.style_profile),
        "response_retry_count": int(response_meta.get("retry_count", 0)),
        "response_validation_errors": _string_list(response_meta.get("validation_errors", [])),
        "prompt_sections": str(prompt_sections),
        "generation_prompt": str(prompt),
        "chat_history_excerpt": str(chat_history_excerpt),
        "llm_base_url": str(generation_config.base_url),
        "llm_model_name": str(generation_config.model_name),
        "z_encoder_mode": str(runtime_config.z_encoder_mode),
        "z_encoder_path": str(runtime_config.z_encoder_path),
        "decoder_model_path": str(runtime_config.zs_model_path),
    }


def generate_chat_turn(
    *,
    runtime: EmoNetChatRuntime,
    generation_config: ChatGenerationConfig,
    input_text: str,
    history: Sequence[Mapping[str, Any]] | None = None,
    episode_payload: Mapping[str, Any] | None = None,
) -> ChatTurnResult:
    user_text = str(input_text or "").strip()
    if not user_text:
        raise ValueError("input text is empty")
    if generation_config.conditioning_mode not in CONDITIONING_MODES:
        raise ValueError(f"unsupported conditioning_mode: {generation_config.conditioning_mode}")
    if generation_config.style_profile not in STYLE_AXIS_PROFILES:
        valid = ", ".join(available_style_profiles())
        raise ValueError(f"unknown style_profile '{generation_config.style_profile}'. valid profiles: {valid}")

    ensure_model_server_ready(
        generation_config.base_url,
        generation_config.timeout_sec,
        api_key=generation_config.api_key,
    )
    profile = infer_style_profile(
        model=runtime.model,
        decoder=runtime.decoder,
        text=user_text,
        style_profile=generation_config.style_profile,
    )
    if generation_config.conditioning_mode in {"episode_trace", "hybrid_episode"}:
        if not isinstance(episode_payload, Mapping):
            raise ValueError("episode payload is required for episode-based conditioning")
        profile = augment_profile_with_episode(profile, dict(episode_payload))

    base_prompt, prompt_sections = build_conditioned_generation_prompt(
        input_text=user_text,
        profile=profile,
        conditioning_mode=generation_config.conditioning_mode,
        template_path=generation_config.prompt_template,
    )
    prompt = inject_chat_history(base_prompt, history, generation_config.history_turns)
    response_text, _raw_output, response_meta = request_plain_text_response(
        base_url=generation_config.base_url,
        model_name=generation_config.model_name,
        prompt=prompt,
        temperature=generation_config.response_temperature,
        max_tokens=generation_config.max_tokens,
        timeout_sec=generation_config.timeout_sec,
        max_retries=generation_config.response_max_retries,
        validator=validate_plain_response_text,
        retry_instruction=DEFAULT_RESPONSE_RETRY_INSTRUCTION,
        system_prompt=DEFAULT_REQUEST_SYSTEM_PROMPT,
        api_key=generation_config.api_key,
        reasoning_effort=generation_config.reasoning_effort,
    )
    chat_history_excerpt = build_recent_dialogue_block(history, generation_config.history_turns)
    record = _serialize_profile(
        input_text=user_text,
        assistant_text=response_text,
        profile=profile,
        prompt=prompt,
        prompt_sections=prompt_sections,
        response_meta=response_meta,
        runtime_config=runtime.config,
        generation_config=generation_config,
        chat_history_excerpt=chat_history_excerpt,
    )
    return ChatTurnResult(assistant_text=response_text, record=record)
