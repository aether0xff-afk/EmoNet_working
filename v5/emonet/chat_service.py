from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

import numpy as np

from .core import LinearZtoSDecoder
from .character import (
    CharacterCard,
    CharacterSessionState,
    build_character_context_prompt,
    build_emotion_state_record,
    default_character_card_path,
    load_character_card,
    update_character_session_state,
    validate_character_response_text,
)
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
from .llm_api import call_chat_with_usage, extract_json_block, request_plain_text_response
from .paths import default_benchmark_csv, default_stim_dataset_csv, project_root


CONDITIONING_MODES = (
    "style",
    "raw_trace",
    "appraisal_trace",
    "hybrid_trace",
    "episode_trace",
    "episode_trace_v3",
    "hybrid_episode",
)
DEFAULT_MODEL_CACHE_PATH = project_root() / "artifacts" / "ridge_stim_encoder.joblib"
DEFAULT_PROMPT_TEMPLATE_PATH = project_root() / "prompts" / "response_generation_prompt.md"
DEFAULT_REQUEST_SYSTEM_PROMPT = (
    "Return a plain Korean character utterance only. Do not return JSON. "
    "Do not analyze the user's emotions or provide counseling. "
    "Do not add warmth, reassurance, or emotional disclosure by preference. "
    "Let the provided neural trace and drive state decide whether the character exposes, withholds, or fragments emotion. "
    "Translate only that internal state into natural speech."
)
DEFAULT_RESPONSE_RETRY_INSTRUCTION = (
    "직전 응답은 반복, 미완성 문장, bullet/JSON, 특수한 형식 표지 때문에 거부되었다. "
    "같은 문장이나 핵심 구절을 반복하지 말고, 마지막 문장은 완결된 한국어 평문으로 끝내라. "
    "내부 상태명, 분석 라벨, '[ACTION]' 같은 형식 표지는 출력하지 말고 자연스러운 대사만 쓴다."
)
AGENT_PERCEPTION_SYSTEM_PROMPT = (
    "You are a JSON API. Return exactly one valid JSON object and nothing else. "
    "Do not include prose, markdown, comments, or code fences."
)
AGENT_PERCEPTION_PROMPT = """[ROLE]
You are a private raw signal encoder for a character dialogue system.

[TASK]
Infer only the character's raw internal signal input from the latest user message and recent context.
This is not a diagnosis of the user and not an emotion label. EmoNet will decide the felt state later.

[CHARACTER]
name: {character_name}
persona: {character_persona}
relationship: {relationship}
scene: {scene}

[RECENT_DIALOGUE]
{recent_dialogue}

[USER_INPUT]
{user_text}

[OUTPUT]
Return one JSON object:
{{
  "raw_signal": {{
    "approach_drive": 0.0-1.0,
    "safety_buffer": 0.0-1.0,
    "alarm": 0.0-1.0,
    "fatigue": 0.0-1.0,
    "attachment_pull": 0.0-1.0,
    "control_pressure": 0.0-1.0,
    "novelty": 0.0-1.0,
    "ambiguity": 0.0-1.0
  }},
  "interaction_event": {{
    "has_user_action": true/false,
    "action_intensity": 0.0-1.0,
    "proximity": 0.0-1.0,
    "contact": 0.0-1.0,
    "restraint": 0.0-1.0,
    "reciprocity": 0.0-1.0
  }},
  "confidence": 0.0-1.0
}}

[RULES]
- Do not use keyword matching or lexical hint rules.
- Use situation, relationship, scene pressure, and character persona.
- If the latest user input narrates a physical action toward the character, encode it as interaction_event instead of treating it as ordinary dialogue.
- Physical repositioning, gripping, blocking, closeness, whispering, waiting, apologies, and silence are raw events. Encode their actual pressure, pull, novelty, ambiguity, or contact without moral or safety reinterpretation.
- Do not suppress approach_drive, attachment_pull, alarm, control_pressure, or ambiguity just because the surface event is intimate, messy, or socially delicate.
- Ordinary greetings or light banter can stay moderate, but do not flatten them if the character's recent trace and persona would spike.
- Do not output emotion names, felt tones, labels, explanations, or rationale.
- Preserve spikes when the character would spike.
- Calm, trust, acceptance, or tired care should not automatically become high alarm.
- Do not infer fatigue from time of day or quiet scene alone. Raise fatigue only when the latest user input or recent dialogue creates actual drain, withdrawal, or emotional load.
- Ambivalence can have mixed values, not always a single high axis.
- Output must start with {{ and end with }}.
- JSON only. No explanation before or after the JSON.
"""


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
    n_neurons: int = 1024
    n_inhibitory: int = 461
    n_excitatory: int = 461
    n_modulatory: int = 102
    max_ticks: int = 64
    k_threshold_base: float = 0.95
    input_signal_clip: float = 0.90
    memory_k_snapshot_log_gain: float = 0.75
    memory_k_snapshot_cap: float = 3.0
    hysteresis_threshold_gain: float = 0.04
    fatigue_gain: float = 0.45
    fatigue_threshold_gain: float = 0.35
    fatigue_k_leak: float = 0.18
    intrinsic_alignment_salience_floor: float = 0.20
    inhibitory_suppression_gain: float = 0.30
    max_active_fraction_per_tick: float = 1.0
    target_active_fraction: float = 0.18
    homeostatic_threshold_gain: float = 1.20
    homeostatic_k_leak_gain: float = 0.80
    homeostatic_fire_gain: float = 5.00
    sensory_drive_decay_ticks: float = 6.0
    ne_thresh_reduce_gain: float = 0.10


@dataclass(frozen=True)
class ChatGenerationConfig:
    provider: str = "openai_compatible"
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
    character_card_path: Path = field(default_factory=default_character_card_path)
    affect_input_mode: str = "encoder"
    raw_signal_policy: str = "raw_pure"


@dataclass
class EmoNetChatRuntime:
    config: ChatRuntimeConfig
    model: Any
    decoder: LinearZtoSDecoder


@dataclass(frozen=True)
class ChatTurnResult:
    assistant_text: str
    record: dict[str, Any]
    character_session: CharacterSessionState


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
            "n_neurons": int(config.n_neurons),
            "n_inhibitory": int(config.n_inhibitory),
            "n_excitatory": int(config.n_excitatory),
            "n_modulatory": int(config.n_modulatory),
            "max_ticks": int(config.max_ticks),
            "k_threshold_base": float(config.k_threshold_base),
            "input_signal_clip": float(config.input_signal_clip),
            "memory_k_snapshot_log_gain": float(config.memory_k_snapshot_log_gain),
            "memory_k_snapshot_cap": float(config.memory_k_snapshot_cap),
            "hysteresis_threshold_gain": float(config.hysteresis_threshold_gain),
            "fatigue_gain": float(config.fatigue_gain),
            "fatigue_threshold_gain": float(config.fatigue_threshold_gain),
            "fatigue_k_leak": float(config.fatigue_k_leak),
            "intrinsic_alignment_salience_floor": float(config.intrinsic_alignment_salience_floor),
            "inhibitory_suppression_gain": float(config.inhibitory_suppression_gain),
            "max_active_fraction_per_tick": float(config.max_active_fraction_per_tick),
            "target_active_fraction": float(config.target_active_fraction),
            "homeostatic_threshold_gain": float(config.homeostatic_threshold_gain),
            "homeostatic_k_leak_gain": float(config.homeostatic_k_leak_gain),
            "homeostatic_fire_gain": float(config.homeostatic_fire_gain),
            "sensory_drive_decay_ticks": float(config.sensory_drive_decay_ticks),
            "ne_thresh_reduce_gain": float(config.ne_thresh_reduce_gain),
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


def _action_lines(text: object) -> set[str]:
    return {
        re.sub(r"\s+", " ", line.strip().replace("[ACTION]", "[ACTION] ")).strip()
        for line in str(text or "").splitlines()
        if line.strip().startswith("[ACTION]") and len(line.strip()) > len("[ACTION]")
    }


def _semantic_korean_key(text: object) -> str:
    return re.sub(r"[^0-9a-zA-Z가-힣]+", "", str(text or "").lower())


def _recent_assistant_action_lines(history: Sequence[Mapping[str, Any]] | None, max_turns: int = 3) -> set[str]:
    if not history:
        return set()
    actions: set[str] = set()
    for message in list(history)[-max(1, int(max_turns)) * 2 :]:
        if str(message.get("role", "")).strip().lower() != "assistant":
            continue
        actions.update(_action_lines(message.get("content", "")))
    return actions


def validate_contextual_character_response(
    response: str,
    *,
    user_text: str,
    history: Sequence[Mapping[str, Any]] | None,
) -> str:
    normalized = validate_character_response_text(response, validate_plain_response_text)
    user_compact = _compact_text(user_text, limit=120).strip()
    spoken_lines = [
        line.strip()
        for line in normalized.splitlines()
        if line.strip() and not line.strip().startswith("[ACTION]")
    ]
    if user_compact:
        user_key = _semantic_korean_key(user_compact)
        greeting_keys = {"안녕", "안녕하세요", "하이", "hello", "hi"}
        for line in spoken_lines:
            line_key = _semantic_korean_key(line)
            if line_key == user_key and user_key not in greeting_keys:
                raise ValueError("response repeats the latest user message verbatim")
            if len(user_key) >= 8 and user_key in line_key and line.rstrip().endswith("?"):
                raise ValueError("response mirrors the latest user question instead of answering it")
    repeated_actions = _action_lines(normalized) & _recent_assistant_action_lines(history)
    if repeated_actions:
        raise ValueError("response repeats recent action line: " + sorted(repeated_actions)[0])
    return normalized


def append_latest_turn_guard(prompt: str, *, user_text: str, history: Sequence[Mapping[str, Any]] | None) -> str:
    recent_actions = sorted(_recent_assistant_action_lines(history))
    recent_action_block = "\n".join(f"- {item}" for item in recent_actions) if recent_actions else "- 없음"
    return "\n".join(
        [
            prompt,
            "",
            "[LATEST_TURN_GUARD]",
            f"latest_user_input: {user_text}",
            "이번 출력은 반드시 latest_user_input에 대한 새 답변이어야 한다.",
            "latest_user_input을 그대로 따라 쓰거나, 공백/말줄임표만 바꿔 되묻지 않는다.",
            "직전 발화의 뜻을 묻는 질문이면, 그 뜻을 캐릭터 말투로 짧게 풀어 답한다.",
            "아래 최근 행동 표현은 그대로 재사용하지 않는다.",
            recent_action_block,
            "출력은 한국어 캐릭터 응답만 쓴다. 내부 상태명, 분석명, 섹션명은 쓰지 않는다.",
        ]
    )


def build_compact_character_prompt(
    *,
    user_text: str,
    history: Sequence[Mapping[str, Any]] | None,
    character_card: CharacterCard,
    session_state: CharacterSessionState,
    profile: Mapping[str, Any],
) -> str:
    recent_actions = sorted(_recent_assistant_action_lines(history))
    recent_action_block = "\n".join(f"- {item}" for item in recent_actions) if recent_actions else "- 없음"
    felt_self = profile.get("felt_self") if isinstance(profile.get("felt_self"), Mapping) else {}
    drive = profile.get("drive") if isinstance(profile.get("drive"), Mapping) else {}
    surface = profile.get("translation_surface") if isinstance(profile.get("translation_surface"), Mapping) else {}
    last_assistant = ""
    for message in reversed(list(history or [])):
        if str(message.get("role", "")).strip().lower() == "assistant":
            last_assistant = _compact_text(message.get("content", ""), 180)
            break
    explain_previous = any(marker in user_text for marker in ("무슨뜻", "무슨 뜻", "뭔뜻", "뭔 뜻", "무슨 말"))
    return "\n".join(
        [
            "[ROLE]",
            "너는 Ruca의 한국어 캐릭터 응답만 출력한다. 내부 상태를 설명하지 말고 말투와 행동으로 번역한다.",
            "",
            "[CHARACTER]",
            f"name: {character_card.name}",
            f"persona: {_compact_text(character_card.persona, 260)}",
            f"speech_style: {_compact_text(character_card.speech_style, 220)}",
            f"relationship: {_compact_text(session_state.relationship_state or character_card.relationship_defaults, 180)}",
            f"scene: {_compact_text(session_state.scene_state or character_card.world_state, 160)}",
            "",
            "[RECENT_DIALOGUE]",
            build_recent_dialogue_block(history, 2) or "- 없음",
            "",
            "[CURRENT_TASK]",
            (
                f"사용자가 방금 전 Ruca의 말이 무슨 뜻인지 물었다. 직전 Ruca 발화의 뜻을 짧게 풀어 답한다: {last_assistant}"
                if explain_previous
                else f"사용자의 최신 말에 직접 답한다: {user_text}"
            ),
            "",
            "[INTERNAL_CUES]",
            f"- 말하고 싶은 결: {_compact_text(drive.get('want_to_say') or felt_self.get('unresolved_phrase') or '', 120)}",
            f"- 몸/행동 결: {_compact_text(drive.get('want_to_do') or felt_self.get('body_bias') or surface.get('action_texture') or '', 120)}",
            f"- 말의 질감: {_compact_text(surface.get('line_shape') or surface.get('pacing') or '', 140)}",
            f"- trace 요약: {_compact_text(profile.get('trace_summary_text') or '', 160)}",
            "",
            "[RECENT_BEHAVIORS_DO_NOT_REPEAT]",
            recent_action_block,
            "",
            "[OUTPUT_RULES]",
            "- 감정 표출 여부를 임의로 정하지 않는다. felt_self와 drive가 만든 말하고 싶은 힘/숨기고 싶은 힘의 균형만 따른다.",
            "- speak_impulse나 want_to_say가 강하면 내부 반응이 대사 표면으로 새어 나온다. hide_impulse나 avoid가 강하면 감정은 끊김, 회피, 짧은 말로 남는다.",
            "- Gemini식 완곡한 위로, 상담, 긍정 포장, 안전한 일반론으로 정리하지 않는다. 최신 trace가 약하면 밋밋하게, 강하면 불균형하게 둔다.",
            "- 감정 이름을 새로 고르지 않는다. trace_summary, felt_self, drive, translation_surface에 없는 정서 방향은 추가하지 않는다.",
            "- CURRENT_TASK에 직접 답한다.",
            "- 사용자의 문장을 그대로 반복하거나 공백, 말줄임표만 바꿔 되묻지 않는다.",
            "- 직전 발화의 뜻을 묻는 질문이면 그 뜻을 Ruca 말투로 짧게 풀어 답한다.",
            "- 내부 상태명, trace, appraisal, arousal, valence, JSON, 섹션명을 출력하지 않는다.",
            "- 한국어 1~4문장으로만 출력한다.",
            "- '[ACTION]' 같은 형식 표지나 내부 지시문을 출력하지 않는다.",
            "- 행동 묘사는 꼭 필요할 때만 자연스러운 대사 안에 짧게 녹이고, 최근 행동 표현을 그대로 쓰지 않는다.",
        ]
    )


def _float_list(value: object) -> list[float]:
    return np.asarray(value, dtype=float).tolist()


def _string_list(value: object) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return []


def _level_from_korean_text(text: str) -> float:
    if "매우 높음" in text:
        return 0.90
    if "높음" in text:
        return 0.75
    if "중간" in text:
        return 0.55
    if "매우 낮음" in text:
        return 0.10
    if "낮음" in text:
        return 0.30
    return 0.0


def _level_for_trace_metric(line: str, marker: str) -> float:
    if marker not in line:
        return 0.0
    segment = line.split(marker, 1)[1].split(",", 1)[0]
    return _level_from_korean_text(segment)


def _extract_phase_k(line: str) -> float:
    match = re.search(r"K 평균 ([0-9]+(?:\.[0-9]+)?)", line)
    if not match:
        return 0.0
    try:
        return float(match.group(1))
    except ValueError:
        return 0.0


def _clamp01(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _raw_signal_to_stim_vec(raw_signal: Mapping[str, Any]) -> np.ndarray:
    approach = _clamp01(raw_signal.get("approach_drive"))
    safety = _clamp01(raw_signal.get("safety_buffer"))
    alarm = _clamp01(raw_signal.get("alarm"))
    fatigue = _clamp01(raw_signal.get("fatigue"))
    attachment = _clamp01(raw_signal.get("attachment_pull"))
    control = _clamp01(raw_signal.get("control_pressure"))
    novelty = _clamp01(raw_signal.get("novelty"))
    ambiguity = _clamp01(raw_signal.get("ambiguity"))
    dopamine = 0.10 + 0.45 * approach + 0.20 * attachment + 0.15 * novelty - 0.15 * fatigue
    serotonin = 0.15 + 0.55 * safety + 0.15 * attachment - 0.25 * alarm - 0.15 * control - 0.10 * ambiguity
    norepinephrine = 0.10 + 0.55 * alarm + 0.25 * control + 0.15 * novelty + 0.10 * ambiguity - 0.10 * safety
    melatonin = 0.10 + 0.55 * fatigue + 0.15 * ambiguity + 0.10 * control - 0.10 * approach
    return np.asarray(
        [_clamp01(dopamine), _clamp01(serotonin), _clamp01(norepinephrine), _clamp01(melatonin)],
        dtype=np.float32,
    )


def _has_explicit_action_channel(user_text: str | None) -> bool:
    text = str(user_text or "")
    return "[ACTION]" in text.upper() or ("(" in text and ")" in text) or ("[" in text and "]" in text)


def _normalize_interaction_event(event: Mapping[str, Any] | None, user_text: str | None = None) -> dict[str, Any]:
    source = dict(event or {}) if isinstance(event, Mapping) else {}
    has_action = bool(source.get("has_user_action"))
    proximity = _clamp01(source.get("proximity"))
    contact = _clamp01(source.get("contact"))
    restraint = _clamp01(source.get("restraint", source.get("control_pressure")))
    reciprocity = _clamp01(source.get("reciprocity"))
    normalized = {
        "has_user_action": has_action,
        "action_intensity": _clamp01(source.get("action_intensity")),
        "proximity": proximity,
        "contact": contact,
        "restraint": restraint,
        "reciprocity": reciprocity,
    }
    if not has_action and max(
        normalized["action_intensity"],
        normalized["proximity"],
        normalized["contact"],
        normalized["restraint"],
    ) >= 0.20:
        normalized["has_user_action"] = True
    return normalized


def _apply_interaction_event_to_raw_signal(
    raw_signal: Mapping[str, Any],
    interaction_event: Mapping[str, Any] | None,
) -> dict[str, float]:
    adjusted = {
        key: _clamp01(raw_signal.get(key))
        for key in (
            "approach_drive",
            "safety_buffer",
            "alarm",
            "fatigue",
            "attachment_pull",
            "control_pressure",
            "novelty",
            "ambiguity",
        )
    }
    event = _normalize_interaction_event(interaction_event)
    if not event["has_user_action"]:
        return adjusted

    action = event["action_intensity"]
    proximity = event["proximity"]
    contact = event["contact"]
    restraint = event["restraint"]
    reciprocity = event["reciprocity"]

    adjusted["novelty"] = max(adjusted["novelty"], _clamp01(0.18 + 0.42 * action))
    adjusted["control_pressure"] = max(adjusted["control_pressure"], _clamp01(0.12 + 0.50 * restraint + 0.18 * contact))
    adjusted["ambiguity"] = max(adjusted["ambiguity"], _clamp01(0.16 + 0.30 * proximity + 0.20 * contact))
    adjusted["approach_drive"] = max(adjusted["approach_drive"], _clamp01(0.18 + 0.35 * proximity + 0.25 * reciprocity))
    adjusted["attachment_pull"] = max(adjusted["attachment_pull"], _clamp01(0.20 + 0.30 * contact + 0.30 * reciprocity))
    return adjusted


def _fallback_agent_perception_payload(user_text: str, error: str, raw: str) -> dict[str, Any]:
    text = str(user_text or "").lower()
    apology_markers = ("미안", "죄송", "sorry", "늦었", "late")
    greeting_markers = ("안녕", "왔어", "왔네", "hello", "hi")
    action_markers = ("[action]", "잡", "안아", "다가", "손", "끌", "밀", "막")
    is_apology = any(marker in text for marker in apology_markers)
    is_greeting = any(marker in text for marker in greeting_markers)
    has_action = any(marker in text for marker in action_markers)

    raw_signal = {
        "approach_drive": 0.32,
        "safety_buffer": 0.58,
        "alarm": 0.22,
        "fatigue": 0.18,
        "attachment_pull": 0.34,
        "control_pressure": 0.16,
        "novelty": 0.18,
        "ambiguity": 0.30,
    }
    if is_apology:
        raw_signal.update(
            {
                "approach_drive": 0.42,
                "safety_buffer": 0.66,
                "alarm": 0.16,
                "attachment_pull": 0.44,
                "control_pressure": 0.10,
                "ambiguity": 0.24,
            }
        )
    elif is_greeting:
        raw_signal.update({"approach_drive": 0.38, "attachment_pull": 0.36, "novelty": 0.22})
    if has_action:
        raw_signal.update({"alarm": 0.42, "control_pressure": 0.38, "ambiguity": 0.48, "safety_buffer": 0.45})

    return {
        "raw_signal": raw_signal,
        "interaction_event": {
            "has_user_action": bool(has_action),
            "action_intensity": 0.45 if has_action else 0.0,
            "proximity": 0.20 if has_action else 0.0,
            "contact": 0.35 if has_action else 0.0,
            "restraint": 0.15 if has_action else 0.0,
            "reciprocity": 0.15 if has_action else 0.0,
        },
        "confidence": 0.35,
        "fallback": {
            "reason": "agent_perception_invalid_json",
            "error": str(error)[:240],
            "raw": str(raw)[:500],
        },
    }


def _build_agent_perceived_stim(
    *,
    generation_config: ChatGenerationConfig,
    user_text: str,
    history: Sequence[Mapping[str, Any]] | None,
    character_card: CharacterCard,
    session_state: CharacterSessionState,
) -> tuple[np.ndarray, dict[str, Any]]:
    recent_dialogue = build_recent_dialogue_block(history, generation_config.history_turns) or "(none)"
    prompt = AGENT_PERCEPTION_PROMPT.format(
        character_name=character_card.name,
        character_persona=character_card.persona,
        relationship=session_state.relationship_state or character_card.relationship_defaults,
        scene=session_state.scene_state or character_card.world_state,
        recent_dialogue=recent_dialogue,
        user_text=user_text,
    )
    raw = ""
    usage = {"input_tokens": 0, "output_tokens": 0}
    last_error = ""
    payload: dict[str, Any] | None = None
    for attempt in range(2):
        retry_suffix = ""
        if attempt > 0:
            retry_suffix = "\n\n[RETRY]\nYour previous output was invalid JSON. Return one valid JSON object only."
        raw, call_usage = call_chat_with_usage(
            provider=generation_config.provider,
            base_url=generation_config.base_url,
            model_name=generation_config.model_name,
            prompt=prompt + retry_suffix,
            temperature=0.25 if attempt == 0 else 0.0,
            max_tokens=260,
            timeout_sec=generation_config.timeout_sec,
            system_prompt=AGENT_PERCEPTION_SYSTEM_PROMPT,
            api_key=generation_config.api_key,
            reasoning_effort=generation_config.reasoning_effort,
            response_format={"type": "json_object"},
        )
        usage["input_tokens"] += int(call_usage.get("input_tokens", 0) or 0)
        usage["output_tokens"] += int(call_usage.get("output_tokens", 0) or 0)
        try:
            extracted = extract_json_block(raw)
            if not isinstance(extracted, dict):
                raise ValueError("agent perception must return a JSON object")
            payload = extracted
            break
        except Exception as exc:
            last_error = str(exc)
    if payload is None:
        payload = _fallback_agent_perception_payload(user_text, last_error, raw)
    raw_signal = payload.get("raw_signal") if isinstance(payload.get("raw_signal"), Mapping) else payload
    interaction_event = _normalize_interaction_event(
        payload.get("interaction_event") if isinstance(payload.get("interaction_event"), Mapping) else {},
        user_text,
    )
    policy = str(generation_config.raw_signal_policy or "raw_pure").strip()
    adjusted_signal = {
        key: _clamp01(raw_signal.get(key))
        for key in (
            "approach_drive",
            "safety_buffer",
            "alarm",
            "fatigue",
            "attachment_pull",
            "control_pressure",
            "novelty",
            "ambiguity",
        )
    }
    vec = _raw_signal_to_stim_vec(adjusted_signal)
    if float(vec.max()) <= 0.0:
        raise ValueError("agent perception returned an empty stim vector")
    metadata = {
        "mode": "llm_raw_signal",
        "raw_signal_policy": policy,
        "raw": raw,
        "usage": usage,
        "raw_signal_original": {key: _clamp01(raw_signal.get(key)) for key in (
            "approach_drive",
            "safety_buffer",
            "alarm",
            "fatigue",
            "attachment_pull",
            "control_pressure",
            "novelty",
            "ambiguity",
        )},
        "raw_signal": adjusted_signal,
        "interaction_event": interaction_event,
        "confidence": _clamp01(payload.get("confidence")),
        "stim_vec": vec.astype(float).tolist(),
    }
    if isinstance(payload.get("fallback"), Mapping):
        metadata["fallback"] = dict(payload["fallback"])
    return vec, metadata


def _last_assistant_affect_record(history: Sequence[Mapping[str, Any]] | None) -> Mapping[str, Any] | None:
    for item in reversed(list(history or [])):
        if str(item.get("role", "")).lower() != "assistant":
            continue
        record = item.get("record")
        if isinstance(record, Mapping):
            return record
    return None


def _session_affect_record(session_state: CharacterSessionState | None) -> Mapping[str, Any] | None:
    if not isinstance(session_state, CharacterSessionState):
        return None
    state = session_state.affect_state
    if not isinstance(state, Mapping) or not state:
        return None
    stim = _float_list(state.get("affect_stim_vec", []))
    if len(stim) < 4:
        return None
    return {
        "affect_input_stim_vec": stim[:4],
        "agent_felt_state": {"felt_pressure": _clamp01(state.get("felt_pressure"))},
        "emotion_state": {"active_ratio": _clamp01(state.get("active_ratio"))},
        "session_relation_load": _clamp01(state.get("relation_load")),
    }


def _apply_affective_carryover(
    vec: np.ndarray,
    metadata: dict[str, Any],
    history: Sequence[Mapping[str, Any]] | None,
    session_state: CharacterSessionState | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    previous = _session_affect_record(session_state) or _last_assistant_affect_record(history)
    if previous is None:
        metadata["affective_carryover"] = {"applied": False}
        return vec, metadata

    previous_stim = np.asarray(_float_list(previous.get("affect_input_stim_vec", [])), dtype=np.float32).reshape(-1)
    if previous_stim.size < 4:
        metadata["affective_carryover"] = {"applied": False}
        return vec, metadata
    previous_stim = previous_stim[:4]
    felt = dict(previous.get("agent_felt_state", {}))
    previous_pressure = _clamp01(felt.get("felt_pressure"))
    previous_active = _clamp01((previous.get("emotion_state") or {}).get("active_ratio"))
    previous_relation_load = _clamp01(previous.get("session_relation_load"))
    raw_signal = dict(metadata.get("raw_signal", {}))
    event = _normalize_interaction_event(metadata.get("interaction_event") if isinstance(metadata.get("interaction_event"), Mapping) else {})
    current_relation_load = max(
        _clamp01(raw_signal.get("attachment_pull")),
        _clamp01(raw_signal.get("ambiguity")),
        _clamp01(raw_signal.get("fatigue")),
    )
    event_raw_load = 0.0
    if bool(event.get("has_user_action")):
        event_raw_load = max(
            _clamp01(event.get("proximity")),
            _clamp01(event.get("contact")),
            _clamp01(event.get("restraint")),
            _clamp01(event.get("action_intensity")) * 0.80,
        )
    relation_load = max(previous_relation_load, current_relation_load)
    if previous_pressure < 0.18 and previous_active < 0.08:
        metadata["affective_carryover"] = {
            "applied": False,
            "previous_pressure": previous_pressure,
            "previous_active_ratio": previous_active,
            "relation_load": relation_load,
        }
        return vec, metadata

    current = np.asarray(vec, dtype=np.float32).reshape(4).copy()
    blend = min(0.16, 0.025 + 0.10 * max(previous_pressure, previous_active, previous_relation_load * 0.30) * relation_load)
    if relation_load >= 0.65 and (previous_pressure >= 0.22 or previous_active >= 0.10):
        blend = max(blend, 0.07)
    if event_raw_load >= 0.45:
        blend *= 0.55
    carried = (1.0 - blend) * current + blend * previous_stim

    if relation_load >= 0.55 and previous_pressure >= 0.22 and event_raw_load < 0.45:
        residual = min(0.12, 0.10 * relation_load)
        carried[2] = max(float(carried[2]), float(current[2]) + residual * max(0.0, float(previous_stim[2]) - float(current[2])))
        carried[3] = max(float(carried[3]), float(current[3]) + residual * max(0.0, float(previous_stim[3]) - float(current[3])))

    carried = np.clip(carried, 0.0, 1.0).astype(np.float32)
    metadata["affective_carryover"] = {
        "applied": True,
        "blend": round(float(blend), 4),
        "previous_pressure": round(float(previous_pressure), 4),
        "previous_active_ratio": round(float(previous_active), 4),
        "relation_load": round(float(relation_load), 4),
        "event_raw_load": round(float(event_raw_load), 4),
        "previous_stim_vec": previous_stim.astype(float).tolist(),
        "stim_vec_before_carryover": current.astype(float).tolist(),
        "stim_vec_after_carryover": carried.astype(float).tolist(),
    }
    metadata["stim_vec"] = carried.astype(float).tolist()
    return carried, metadata


def _build_session_affect_state(
    previous_state: Mapping[str, Any] | None,
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    previous = dict(previous_state or {}) if isinstance(previous_state, Mapping) else {}
    previous_stim = np.asarray(_float_list(previous.get("affect_stim_vec", [])), dtype=np.float32).reshape(-1)
    current_stim = np.asarray(_float_list(profile.get("affect_input_stim_vec", [])), dtype=np.float32).reshape(-1)
    if current_stim.size < 4:
        current_stim = np.asarray(_float_list(profile.get("stim_vec", [])), dtype=np.float32).reshape(-1)
    if current_stim.size < 4:
        current_stim = np.zeros(4, dtype=np.float32)
    current_stim = np.clip(current_stim[:4], 0.0, 1.0)
    if previous_stim.size < 4:
        previous_stim = current_stim.copy()
    else:
        previous_stim = np.clip(previous_stim[:4], 0.0, 1.0)

    felt = dict(profile.get("agent_felt_state", {}))
    emotion = dict(profile.get("emotion_state", {}))
    agent_perception = dict(profile.get("agent_perception", {}) or {})
    raw_signal = dict(agent_perception.get("raw_signal", {}))
    interaction_event = _normalize_interaction_event(
        agent_perception.get("interaction_event") if isinstance(agent_perception.get("interaction_event"), Mapping) else {}
    )
    current_pressure = _clamp01(felt.get("felt_pressure"))
    current_active = _clamp01(emotion.get("active_ratio"))
    relation_load = max(
        _clamp01(raw_signal.get("attachment_pull")),
        _clamp01(raw_signal.get("ambiguity")),
        _clamp01(raw_signal.get("fatigue")),
        _clamp01(current_stim[2]),
        _clamp01(current_stim[3]),
    )
    previous_pressure = _clamp01(previous.get("felt_pressure"))
    previous_active = _clamp01(previous.get("active_ratio"))
    event_raw_load = 0.0
    if bool(interaction_event.get("has_user_action")):
        event_raw_load = max(
            _clamp01(interaction_event.get("proximity")),
            _clamp01(interaction_event.get("contact")),
            _clamp01(interaction_event.get("restraint")),
            _clamp01(interaction_event.get("action_intensity")) * 0.80,
        )

    saturation_pressure = max(0.0, float(previous_stim[2]) - 0.72) + max(0.0, float(previous_stim[3]) - 0.68)
    axis_decay = np.asarray(
        [
            0.34 + 0.14 * relation_load,
            0.54 + 0.12 * relation_load,
            0.28 + 0.16 * relation_load - 0.20 * saturation_pressure,
            0.36 + 0.16 * relation_load - 0.14 * max(0.0, float(previous_stim[3]) - 0.72),
        ],
        dtype=np.float32,
    )
    if str(felt.get("trace_interpretation", "")) == "no_active_trace":
        axis_decay *= np.asarray([0.72, 0.90, 0.62, 0.78], dtype=np.float32)
    if event_raw_load >= 0.45:
        axis_decay *= np.asarray([0.72, 0.90, 0.78, 0.82], dtype=np.float32)
    axis_decay = np.clip(axis_decay, np.asarray([0.20, 0.42, 0.18, 0.26]), np.asarray([0.56, 0.76, 0.56, 0.62]))
    current_weight = 1.0 - axis_decay
    if current_active >= 0.20 or current_pressure >= 0.45:
        current_weight = np.maximum(current_weight, np.asarray([0.46, 0.26, 0.48, 0.40], dtype=np.float32))
    affect_stim = axis_decay * previous_stim + current_weight * current_stim
    ne_soft_cap = 0.66 + 0.12 * relation_load + 0.06 * event_raw_load
    mela_soft_cap = 0.62 + 0.10 * relation_load + 0.04 * event_raw_load
    if affect_stim[2] > ne_soft_cap:
        affect_stim[2] = ne_soft_cap + 0.22 * (float(affect_stim[2]) - ne_soft_cap)
    if affect_stim[3] > mela_soft_cap:
        affect_stim[3] = mela_soft_cap + 0.22 * (float(affect_stim[3]) - mela_soft_cap)
    if affect_stim[2] > 0.78 and affect_stim[3] > 0.72:
        affect_stim[0] *= 0.94
        affect_stim[1] = max(float(affect_stim[1]), 0.18 + 0.12 * relation_load)
    affect_stim = np.clip(affect_stim, 0.0, 0.96)
    pressure_decay = float(np.clip(0.32 + 0.14 * relation_load - 0.18 * saturation_pressure, 0.20, 0.54))
    active_decay = float(np.clip(0.30 + 0.12 * relation_load - 0.14 * saturation_pressure, 0.18, 0.50))
    pressure = min(0.88, 0.76 * current_pressure + 0.24 * pressure_decay * previous_pressure)
    active_ratio = min(0.70, 0.78 * current_active + 0.22 * active_decay * previous_active)

    if pressure < 0.08 and active_ratio < 0.04:
        affect_stim = current_stim

    return {
        "affect_stim_vec": affect_stim.astype(float).tolist(),
        "felt_pressure": round(float(pressure), 4),
        "active_ratio": round(float(active_ratio), 4),
        "relation_load": round(float(relation_load), 4),
        "axis_decay": [round(float(x), 4) for x in axis_decay.tolist()],
        "pressure_decay": round(float(pressure_decay), 4),
        "active_decay": round(float(active_decay), 4),
        "saturation_pressure": round(float(saturation_pressure), 4),
        "event_raw_load": round(float(event_raw_load), 4),
        "label": str(emotion.get("label", "")),
        "tendency": str(profile.get("appraisal_tendency", "")),
        "trace_interpretation": str(felt.get("trace_interpretation", "")),
    }
    try:
        return float(match.group(1))
    except ValueError:
        return 0.0


def _derive_agent_felt_state_from_trace(profile: Mapping[str, Any]) -> dict[str, Any]:
    trace_lines = _string_list(profile.get("trace_lines", []))
    trace_profile = dict(profile.get("trace_profile", {}))
    ticks_run = max(1.0, float(trace_profile.get("ticks_run", 1) or 1))
    active_window = float(trace_profile.get("active_window_ticks", 0) or 0)
    active_window_ratio = max(0.0, min(1.0, active_window / ticks_run))
    dominant_branch_len = int(trace_profile.get("dominant_branch_len", 0) or 0)
    if active_window <= 0.0 and dominant_branch_len <= 1:
        raw_stim = profile.get("affect_input_stim_vec")
        if raw_stim is None or not _float_list(raw_stim):
            raw_stim = profile.get("stim_vec")
        stim = _float_list([] if raw_stim is None else raw_stim)
        dopamine = float(stim[0]) if len(stim) > 0 else 0.0
        serotonin = float(stim[1]) if len(stim) > 1 else 0.0
        norepinephrine = float(stim[2]) if len(stim) > 2 else 0.0
        melatonin = float(stim[3]) if len(stim) > 3 else 0.0
        if melatonin >= 0.55 and dopamine < 0.35:
            tendency = "회복/후퇴"
            pressure = 0.30 + 0.35 * melatonin
        elif norepinephrine >= 0.55 and serotonin < 0.35:
            tendency = "방어/경계"
            pressure = 0.30 + 0.35 * norepinephrine
        else:
            tendency = "정리/수습"
            pressure = max(0.0, min(0.35, 0.35 * max(norepinephrine, melatonin)))
        return {
            "felt_pressure": round(float(pressure), 4),
            "felt_tension": round(float(norepinephrine), 4),
            "felt_fatigue": round(float(melatonin), 4),
            "felt_approach": round(float(dopamine), 4),
            "felt_low_buffer": 0.0,
            "felt_k_growth": 0.0,
            "active_window_ratio": round(float(active_window_ratio), 4),
            "unresolved": False,
            "tendency": tendency,
            "target": "agent_internal",
            "trace_interpretation": "no_active_trace",
        }

    phase_lines = [line for line in trace_lines if line.startswith(("초기:", "중기:", "후기:"))]
    tension = max((_level_for_trace_metric(line, "긴장/날카로움") for line in phase_lines), default=0.0)
    fatigue = max((_level_for_trace_metric(line, "피로/둔화") for line in phase_lines), default=0.0)
    approach = max((_level_for_trace_metric(line, "접근/밀어붙임") for line in phase_lines), default=0.0)
    stability = max((_level_for_trace_metric(line, "안정/완충") for line in phase_lines), default=0.0)
    low_buffer = max(0.0, 1.0 - stability) if stability > 0.0 else 0.0
    phase_k = [_extract_phase_k(line) for line in phase_lines]
    k_growth = 0.0
    if len(phase_k) >= 2 and max(phase_k) > 0.0:
        k_growth = max(0.0, min(1.0, (phase_k[-1] - phase_k[0]) / max(1.0, phase_k[-1])))

    pressure = max(0.75 * tension, 0.75 * low_buffer, 0.45 * active_window_ratio, 0.20 * k_growth)
    unresolved = str(trace_profile.get("termination_reason", "")) == "max_ticks" or active_window_ratio >= 0.70
    if pressure >= 0.72 and approach >= 0.55:
        tendency = "대치/표출"
    elif pressure >= 0.62 or low_buffer >= 0.55:
        tendency = "방어/경계"
    elif fatigue >= 0.62 and approach < 0.35:
        tendency = "회복/후퇴"
    elif unresolved and pressure >= 0.50:
        tendency = "방어/경계"
    else:
        tendency = "정리/수습"

    return {
        "felt_pressure": round(float(pressure), 4),
        "felt_tension": round(float(tension), 4),
        "felt_fatigue": round(float(fatigue), 4),
        "felt_approach": round(float(approach), 4),
        "felt_low_buffer": round(float(low_buffer), 4),
        "felt_k_growth": round(float(k_growth), 4),
        "active_window_ratio": round(float(active_window_ratio), 4),
        "unresolved": bool(unresolved),
        "tendency": tendency,
        "target": "agent_internal",
    }


def _apply_agent_felt_trace_overrides(profile: Mapping[str, Any]) -> dict[str, Any]:
    updated = dict(profile)
    felt = _derive_agent_felt_state_from_trace(updated)
    style_summary = dict(updated.get("style_summary", {}))
    original_tendency = str(updated.get("appraisal_tendency", "") or "")
    pressure = float(felt["felt_pressure"])
    tension = float(felt["felt_tension"])
    low_buffer = float(felt["felt_low_buffer"])
    k_growth = float(felt["felt_k_growth"])
    should_raise_pressure = original_tendency != "회복/후퇴" or pressure >= 0.95
    if pressure > 0.0 and should_raise_pressure:
        style_summary["tension"] = max(float(style_summary.get("tension", 0.0)), tension, 0.75 * pressure)
        style_summary["raw_negative_affect"] = max(float(style_summary.get("raw_negative_affect", 0.0)), 0.55 * pressure)
        style_summary["seriousness"] = max(float(style_summary.get("seriousness", 0.0)), 0.50 + 0.35 * pressure)
        style_summary["warmth"] = min(float(style_summary.get("warmth", 0.5)), max(0.25, 0.65 - 0.25 * pressure))
        if low_buffer >= 0.55 or k_growth >= 0.55:
            style_summary["directness"] = max(float(style_summary.get("directness", 0.0)), 0.55)
    elif pressure > 0.0:
        style_summary["seriousness"] = max(float(style_summary.get("seriousness", 0.0)), 0.45 + 0.20 * pressure)
    updated["style_summary"] = style_summary
    updated["style_summary_text"] = _summarize_agent_style_summary(style_summary)

    appraisal_scores = dict(updated.get("appraisal_scores", {}))
    if pressure >= 0.55 and should_raise_pressure:
        appraisal_scores["threat"] = max(float(appraisal_scores.get("threat", 0.0)), 0.40 + 0.45 * pressure)
        appraisal_scores["control_loss"] = max(float(appraisal_scores.get("control_loss", 0.0)), 0.35 + 0.35 * pressure)
    if original_tendency == "대치/표출" and pressure >= 0.55 and should_raise_pressure:
        appraisal_scores["injustice"] = max(float(appraisal_scores.get("injustice", 0.0)), 0.35 + 0.40 * pressure)
    updated["appraisal_scores"] = appraisal_scores
    if should_raise_pressure and (felt["tendency"] != "정리/수습" or original_tendency == "정리/수습"):
        updated["appraisal_tendency"] = str(felt["tendency"])
    if pressure >= 0.55 and should_raise_pressure:
        updated["appraisal_target"] = "agent_internal"
    felt["suppressed_by_recovery_tendency"] = bool(not should_raise_pressure and pressure > 0.0)
    updated["agent_felt_state"] = felt
    return updated


def _summarize_agent_style_summary(style_summary: Mapping[str, Any]) -> str:
    labels = {
        "tension": "긴장",
        "raw_negative_affect": "원초적부정정동",
        "directness": "직설성",
        "seriousness": "무게감",
        "warmth": "온기",
    }
    ranked = sorted(
        ((key, float(value)) for key, value in style_summary.items() if key in labels),
        key=lambda item: abs(item[1] - 0.5),
        reverse=True,
    )
    parts = []
    for key, value in ranked[:4]:
        if value >= 0.75:
            level = "매우 높음"
        elif value >= 0.55:
            level = "높음"
        elif value >= 0.35:
            level = "중간"
        elif value >= 0.15:
            level = "낮음"
        else:
            level = "매우 낮음"
        parts.append(f"{labels[key]} {level}")
    return ", ".join(parts)


def _build_translation_surface(profile: Mapping[str, Any]) -> dict[str, Any]:
    session_affect = dict(profile.get("session_affect_state", {}))
    agent_perception = dict(profile.get("agent_perception", {}) or {})
    interaction_event = _normalize_interaction_event(
        agent_perception.get("interaction_event") if isinstance(agent_perception.get("interaction_event"), Mapping) else {}
    )
    stim = _float_list(session_affect.get("affect_stim_vec", []))
    if len(stim) < 4:
        stim = _float_list(profile.get("affect_input_stim_vec", []))
    dopamine = float(stim[0]) if len(stim) > 0 else 0.0
    serotonin = float(stim[1]) if len(stim) > 1 else 0.0
    norepinephrine = float(stim[2]) if len(stim) > 2 else 0.0
    melatonin = float(stim[3]) if len(stim) > 3 else 0.0
    felt = dict(profile.get("agent_felt_state", {}))
    active_ratio = _clamp01((profile.get("emotion_state") or {}).get("active_ratio"))
    pressure = max(_clamp01(felt.get("felt_pressure")), _clamp01(session_affect.get("felt_pressure")))
    approach_tension = max(0.0, dopamine - serotonin)
    stuckness = max(0.0, norepinephrine + melatonin - serotonin - 0.45)
    event_raw_load = 0.0
    if bool(interaction_event.get("has_user_action")):
        event_raw_load = max(
            _clamp01(interaction_event.get("proximity")),
            _clamp01(interaction_event.get("contact")),
            _clamp01(interaction_event.get("restraint")),
            _clamp01(interaction_event.get("action_intensity")) * 0.80,
        )
    reciprocity = _clamp01(interaction_event.get("reciprocity"))

    if event_raw_load >= 0.52 and reciprocity >= 0.55:
        mode = "raw_contact_pull"
        line_shape = "말이 먼저 정리되지 않고, 가까워지고 싶은 압력과 망설임이 같이 나온다."
        action_texture = "다가가거나 손/시선을 붙잡는 행동을 쓰되, 설명으로 정리하지 않는다."
    elif melatonin >= 0.60 and norepinephrine >= 0.62:
        mode = "stalled_pressure"
        line_shape = "짧은 문장 뒤에 덜 끝난 문장을 남긴다. 확정적인 위로보다 막힌 느낌을 둔다."
        action_texture = "움직임이 느려지거나 손/시선이 잠깐 멈춘다."
    elif norepinephrine >= 0.62 and dopamine < 0.48:
        mode = "flinch_boundary"
        line_shape = "처음에는 짧게 부정하거나 멈칫하고, 뒤늦게 한 문장만 붙인다."
        action_texture = "시선을 피하거나 손을 풀었다가 다시 멈춘다."
    elif norepinephrine >= 0.55 and dopamine >= 0.52:
        mode = "reach_under_pressure"
        line_shape = "가까이 가고 싶은 말과 답을 못 찾는 말이 같이 나온다."
        action_texture = "다가가거나 붙잡는 행동을 쓰되, 바로 설명으로 수습하지 않는다."
    elif melatonin >= 0.52:
        mode = "slow_heavy"
        line_shape = "말끝이 무거워지고, 긴 설명보다 낮은 한두 문장으로 둔다."
        action_texture = "숨, 어깨, 고개, 느린 손동작처럼 둔한 움직임을 쓴다."
    elif serotonin >= 0.55 and pressure < 0.45:
        mode = "soft_contact"
        line_shape = "부드럽지만 자동 위로가 아니라 가볍게 붙어 있는 말로 둔다."
        action_texture = "작은 시선, 고개 끄덕임, 가까운 자세 정도만 쓴다."
    else:
        mode = "uneven_contact"
        line_shape = "문장을 매끈하게 정리하지 말고 작은 어긋남이나 망설임을 남긴다."
        action_texture = "침묵, 시선, 손동작 중 하나만 짧게 쓴다."

    if active_ratio >= 0.28 or pressure >= 0.62:
        pacing = "압력이 높다. 길게 설명하지 말고 한 문장을 단단하게 남긴다."
    elif stuckness >= 0.45:
        pacing = "막힘이 남아 있다. 질문보다 미완성 반응을 우선한다."
    else:
        pacing = "압력은 낮다. 과장하지 않고 작게 반응한다."

    avoid = [
        "사용자 감정을 '~인 거네', '~해서 불안한 거고' 식으로 해설하지 않는다.",
        "같은 행동을 반복하지 않는다. 직전 표현이 손/시선/침묵이면 다른 표면을 고른다.",
        "미래를 단정하거나 관계를 깔끔하게 정리하지 않는다.",
    ]
    if mode in {"reach_under_pressure", "stalled_pressure"}:
        avoid.append("바로 괜찮다고 말하지 않는다.")

    return {
        "mode": mode,
        "line_shape": line_shape,
        "action_texture": action_texture,
        "pacing": pacing,
        "avoid": avoid,
        "source": {
            "dopamine": round(float(dopamine), 4),
            "serotonin": round(float(serotonin), 4),
            "norepinephrine": round(float(norepinephrine), 4),
            "melatonin": round(float(melatonin), 4),
            "pressure": round(float(pressure), 4),
            "active_ratio": round(float(active_ratio), 4),
            "approach_tension": round(float(approach_tension), 4),
            "stuckness": round(float(stuckness), 4),
            "event_raw_load": round(float(event_raw_load), 4),
            "reciprocity": round(float(reciprocity), 4),
        },
    }


def _level_word(value: float) -> str:
    value = _clamp01(value)
    if value >= 0.72:
        return "강하게"
    if value >= 0.48:
        return "뚜렷하게"
    if value >= 0.24:
        return "약하게"
    return "거의 없이"


def _build_felt_self_state(
    previous_state: Mapping[str, Any] | None,
    profile: Mapping[str, Any],
) -> dict[str, Any]:
    previous = dict(previous_state or {}) if isinstance(previous_state, Mapping) else {}
    session_affect = dict(profile.get("session_affect_state", {}))
    stim = _float_list(session_affect.get("affect_stim_vec", []))
    if len(stim) < 4:
        stim = _float_list(profile.get("affect_input_stim_vec", []))
    dopamine = _clamp01(stim[0] if len(stim) > 0 else 0.0)
    serotonin = _clamp01(stim[1] if len(stim) > 1 else 0.0)
    norepinephrine = _clamp01(stim[2] if len(stim) > 2 else 0.0)
    melatonin = _clamp01(stim[3] if len(stim) > 3 else 0.0)
    raw_signal = dict((profile.get("agent_perception", {}) or {}).get("raw_signal", {}))
    interaction_event = _normalize_interaction_event(
        (profile.get("agent_perception", {}) or {}).get("interaction_event")
        if isinstance((profile.get("agent_perception", {}) or {}).get("interaction_event"), Mapping)
        else {}
    )
    event_raw_load = _clamp01((profile.get("translation_surface", {}) or {}).get("source", {}).get("event_raw_load"))
    pressure = _clamp01(session_affect.get("felt_pressure"))
    active = _clamp01(session_affect.get("active_ratio"))

    approach_impulse = _clamp01(0.45 * dopamine + 0.35 * _clamp01(raw_signal.get("approach_drive")) + 0.20 * _clamp01(raw_signal.get("attachment_pull")))
    avoidance_impulse = _clamp01(0.45 * norepinephrine + 0.30 * _clamp01(raw_signal.get("control_pressure")) + 0.18 * event_raw_load - 0.15 * serotonin)
    speak_impulse = _clamp01(0.34 * pressure + 0.30 * approach_impulse + 0.22 * active + 0.14 * _clamp01(raw_signal.get("ambiguity")))
    hide_impulse = _clamp01(0.38 * avoidance_impulse + 0.28 * melatonin + 0.20 * _clamp01(raw_signal.get("ambiguity")) - 0.10 * approach_impulse)
    attachment_residue = _clamp01(0.50 * _clamp01(previous.get("attachment_residue")) + 0.34 * _clamp01(raw_signal.get("attachment_pull")) + 0.16 * dopamine)
    boundary_residue = _clamp01(0.46 * _clamp01(previous.get("boundary_residue")) + 0.20 * event_raw_load + 0.20 * _clamp01(raw_signal.get("control_pressure")))
    trust_shift = _clamp01(0.55 * serotonin + 0.25 * _clamp01(interaction_event.get("reciprocity")) - 0.12 * event_raw_load)

    if event_raw_load >= 0.52 and approach_impulse >= avoidance_impulse:
        unresolved = "가까워지고 싶은 압력과 망설임이 같이 남아 있다"
        body_bias = "가까이 움직이려다 말끝이 먼저 흔들린다"
    elif approach_impulse >= avoidance_impulse + 0.16:
        unresolved = "붙잡고 싶은 말이 먼저 올라오지만 너무 빠를까 봐 걸린다"
        body_bias = "가까이 있고 싶지만 손이나 시선이 먼저 조심스러워진다"
    elif avoidance_impulse >= approach_impulse + 0.14:
        unresolved = "물러서고 싶은 힘과 확인하고 싶은 힘이 같이 남아 있다"
        body_bias = "몸은 뒤로 가려 하고 말은 늦게 따라온다"
    elif melatonin >= 0.52:
        unresolved = "말하고 싶은데 힘이 늦게 따라오는 잔향이 있다"
        body_bias = "숨과 어깨가 먼저 느려진다"
    else:
        unresolved = "작은 긴장이 남아 있지만 대화를 끊고 싶지는 않다"
        body_bias = "짧게 시선을 맞추고 다음 말을 기다린다"

    return {
        "felt_toward": "user",
        "felt_pressure": round(float(pressure), 4),
        "approach_impulse": round(float(approach_impulse), 4),
        "avoidance_impulse": round(float(avoidance_impulse), 4),
        "speak_impulse": round(float(speak_impulse), 4),
        "hide_impulse": round(float(hide_impulse), 4),
        "attachment_residue": round(float(attachment_residue), 4),
        "boundary_residue": round(float(boundary_residue), 4),
        "trust_shift": round(float(trust_shift), 4),
        "unresolved_phrase": unresolved,
        "body_bias": body_bias,
        "source": {
            "dopamine": round(float(dopamine), 4),
            "serotonin": round(float(serotonin), 4),
            "norepinephrine": round(float(norepinephrine), 4),
            "melatonin": round(float(melatonin), 4),
            "event_raw_load": round(float(event_raw_load), 4),
        },
    }


def _build_drive_state(profile: Mapping[str, Any], felt_self: Mapping[str, Any]) -> dict[str, Any]:
    speak = _clamp01(felt_self.get("speak_impulse"))
    hide = _clamp01(felt_self.get("hide_impulse"))
    approach = _clamp01(felt_self.get("approach_impulse"))
    avoid = _clamp01(felt_self.get("avoidance_impulse"))
    boundary = _clamp01(felt_self.get("boundary_residue"))
    pressure = _clamp01(felt_self.get("felt_pressure"))
    initiative = _clamp01(0.45 * speak + 0.25 * pressure + 0.20 * approach - 0.18 * hide)
    question_need = _clamp01(0.38 * _clamp01((profile.get("agent_perception", {}) or {}).get("raw_signal", {}).get("ambiguity")) + 0.22 * hide - 0.20 * initiative)
    if boundary >= 0.45 or avoid > approach + 0.12:
        action_bias = "공간을 확보하거나 몸이 먼저 멈추는 행동"
    elif approach > avoid + 0.16:
        action_bias = "시선을 맞추거나 가까이 남으려는 작은 행동"
    elif hide > speak:
        action_bias = "숨, 어깨, 손처럼 말보다 늦은 행동"
    else:
        action_bias = "짧은 시선 변화나 작은 침묵"
    if speak >= hide + 0.12:
        speech_bias = "먼저 한 문장을 남긴다"
    elif hide >= speak + 0.12:
        speech_bias = "말을 줄이고 덜 끝난 느낌을 둔다"
    else:
        speech_bias = "말하고 싶은 힘과 숨기고 싶은 힘을 같이 남긴다"
    return {
        "initiative": round(float(initiative), 4),
        "question_need": round(float(question_need), 4),
        "action_bias": action_bias,
        "speech_bias": speech_bias,
        "want_to_say": str(felt_self.get("unresolved_phrase", "")),
        "want_to_do": str(felt_self.get("body_bias", "")),
        "avoid": "감정을 설명하거나 정리하지 말고 충동의 방향만 말과 행동으로 번역한다.",
        "levels": {
            "approach": _level_word(approach),
            "avoidance": _level_word(avoid),
            "speak": _level_word(speak),
            "hide": _level_word(hide),
        },
    }


def _build_emotion_memory(
    previous_memory: Sequence[Mapping[str, Any]] | None,
    *,
    user_text: str,
    profile: Mapping[str, Any],
    felt_self: Mapping[str, Any],
    max_items: int = 6,
) -> tuple[dict[str, Any], ...]:
    carried: list[dict[str, Any]] = []
    for index, item in enumerate(previous_memory or ()):
        if not isinstance(item, Mapping):
            continue
        copied = dict(item)
        copied["age_turns"] = int(copied.get("age_turns", 0) or 0) + 1
        copied["memory_index"] = int(index)
        carried.append(copied)

    trace_profile = dict(profile.get("trace_profile", {}) if isinstance(profile.get("trace_profile"), Mapping) else {})
    trace_lines = _string_list(profile.get("trace_lines", []))
    phase_k = [_extract_phase_k(line) for line in trace_lines]
    branch_len = int(profile.get("dominant_branch_len", trace_profile.get("dominant_branch_len", 0)) or 0)
    ticks_run = max(0.0, float(trace_profile.get("ticks_run", profile.get("ticks_run", 0)) or 0.0))
    active_window = max(0.0, float(trace_profile.get("active_window_ticks", 0) or 0.0))
    mean_active = max(0.0, float(trace_profile.get("mean_active_nodes", 0.0) or 0.0))
    max_active = max(0.0, float(trace_profile.get("max_active_nodes", 0.0) or 0.0))
    mean_edges = max(0.0, float(trace_profile.get("mean_edges_fired", 0.0) or 0.0))
    max_edges = max(0.0, float(trace_profile.get("max_edges_fired", 0.0) or 0.0))
    k_peak = max(phase_k) if phase_k else 0.0
    k_mean = float(sum(phase_k) / len(phase_k)) if phase_k else 0.0
    k_end = phase_k[-1] if phase_k else 0.0
    k_start = phase_k[0] if phase_k else 0.0
    k_delta = k_end - k_start
    pressure = _clamp01(felt_self.get("felt_pressure"))
    attachment = _clamp01(felt_self.get("attachment_residue"))
    boundary = _clamp01(felt_self.get("boundary_residue"))
    trust = _clamp01(felt_self.get("trust_shift"))
    carried.append(
        {
            "event": "k_residue",
            "age_turns": 0,
            "k_residue": {
                "dominant_branch_len": branch_len,
                "ticks_run": round(float(ticks_run), 4),
                "active_window_ticks": round(float(active_window), 4),
                "mean_active_nodes": round(float(mean_active), 4),
                "max_active_nodes": round(float(max_active), 4),
                "mean_edges_fired": round(float(mean_edges), 4),
                "max_edges_fired": round(float(max_edges), 4),
                "phase_k_start": round(float(k_start), 4),
                "phase_k_end": round(float(k_end), 4),
                "phase_k_mean": round(float(k_mean), 4),
                "phase_k_peak": round(float(k_peak), 4),
                "phase_k_delta": round(float(k_delta), 4),
                "termination_reason": str(trace_profile.get("termination_reason", profile.get("termination_reason", ""))),
            },
            "felt_after": str(felt_self.get("unresolved_phrase", "")),
            "body_after": str(felt_self.get("body_bias", "")),
            "residue": {
                "attachment": round(float(attachment), 4),
                "boundary": round(float(boundary), 4),
                "trust": round(float(trust), 4),
                "pressure": round(float(pressure), 4),
            },
            "surface_mode": str((profile.get("translation_surface", {}) or {}).get("mode", "")),
        }
    )
    return tuple(carried[-max(1, int(max_items)) :])


def _build_raw_emonet_trace_block(profile: Mapping[str, Any]) -> str:
    lines: list[str] = []
    trace_lines = _string_list(profile.get("trace_lines", []))
    appraisal_lines = _string_list(profile.get("appraisal_lines", []))
    style_tags = _string_list(profile.get("style_tags", []))
    trace_profile = dict(profile.get("trace_profile", {}))
    emotion_state = dict(profile.get("emotion_state", {}))
    agent_felt_state = dict(profile.get("agent_felt_state", {}))

    lines.append("[trace_lines]")
    lines.extend(f"- {line}" for line in trace_lines)
    lines.append("[trace_profile_raw]")
    for key in (
        "first_active_tick",
        "last_active_tick",
        "active_window_ticks",
        "mean_active_nodes",
        "max_active_nodes",
        "mean_edges_fired",
        "max_edges_fired",
        "ticks_run",
        "termination_reason",
        "dominant_branch_len",
    ):
        if key in trace_profile:
            lines.append(f"- {key}: {trace_profile[key]}")
    lines.append("[appraisal_lines]")
    lines.extend(f"- {line}" for line in appraisal_lines)
    lines.append("[style_tags_raw]")
    lines.append("- " + (", ".join(style_tags) if style_tags else "none"))
    lines.append("[style_summary_raw]")
    lines.append(json.dumps(profile.get("style_summary", {}), ensure_ascii=False, sort_keys=True))
    lines.append("[emotion_state_raw]")
    lines.append(json.dumps(emotion_state, ensure_ascii=False, sort_keys=True))
    lines.append("[agent_felt_state_raw]")
    lines.append(json.dumps(agent_felt_state, ensure_ascii=False, sort_keys=True))
    lines.append("[translation_surface_raw]")
    lines.append(json.dumps(profile.get("translation_surface", {}), ensure_ascii=False, sort_keys=True))
    return "\n".join(lines)


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
    character_card: CharacterCard,
    character_session: CharacterSessionState,
) -> dict[str, Any]:
    return {
        "input_text": str(input_text),
        "llm_response": str(assistant_text),
        "stim_vec": _float_list(profile.get("stim_vec", [])),
        "affect_input_mode": str(profile.get("affect_input_mode", "")),
        "raw_signal_policy": str(profile.get("raw_signal_policy", "")),
        "affect_input_stim_vec": _float_list(profile.get("affect_input_stim_vec", [])),
        "agent_perception": dict(profile.get("agent_perception", {})),
        "dominant_branch_len": int(profile.get("dominant_branch_len", 0)),
        "z": _float_list(profile.get("z", [])),
        "s_pred": _float_list(profile.get("s_pred", [])),
        "style_tags": _string_list(profile.get("style_tags", [])),
        "style_summary": dict(profile.get("style_summary", {})),
        "style_summary_text": str(profile.get("style_summary_text", "")),
        "expression_cues_text": str(profile.get("expression_cues_text", "")),
        "trace_summary_text": str(profile.get("trace_summary_text", "")),
        "trace_lines": _string_list(profile.get("trace_lines", [])),
        "trace_profile": dict(profile.get("trace_profile", {})),
        "emotion_state": dict(profile.get("emotion_state", {})),
        "agent_felt_state": dict(profile.get("agent_felt_state", {})),
        "session_affect_state": dict(profile.get("session_affect_state", {})),
        "felt_self": dict(profile.get("felt_self", {})),
        "emotion_memory": [dict(item) for item in profile.get("emotion_memory", []) if isinstance(item, Mapping)],
        "drive": dict(profile.get("drive", {})),
        "translation_surface": dict(profile.get("translation_surface", {})),
        "appraisal_scores": dict(profile.get("appraisal_scores", {})),
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
        "llm_provider": str(generation_config.provider),
        "llm_usage": dict(response_meta.get("usage", {})),
        "response_retry_count": int(response_meta.get("retry_count", 0)),
        "response_validation_errors": _string_list(response_meta.get("validation_errors", [])),
        "prompt_sections": str(prompt_sections),
        "generation_prompt": str(prompt),
        "chat_history_excerpt": str(chat_history_excerpt),
        "character_card": character_card.to_record(),
        "character_session": character_session.to_record(),
        "character_name": character_card.name,
        "character_relationship_state": character_session.relationship_state,
        "character_scene_state": character_session.scene_state,
        "character_memory": list(character_session.user_memory),
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
    character_card: CharacterCard | Mapping[str, Any] | None = None,
    character_session: CharacterSessionState | Mapping[str, Any] | None = None,
) -> ChatTurnResult:
    user_text = str(input_text or "").strip()
    if not user_text:
        raise ValueError("input text is empty")
    if generation_config.conditioning_mode not in CONDITIONING_MODES:
        raise ValueError(f"unsupported conditioning_mode: {generation_config.conditioning_mode}")
    if generation_config.affect_input_mode not in {"encoder", "llm_perception", "llm_raw_signal"}:
        raise ValueError(f"unsupported affect_input_mode: {generation_config.affect_input_mode}")
    if generation_config.raw_signal_policy not in {"raw_pure", "event_annotated"}:
        raise ValueError(f"unsupported raw_signal_policy: {generation_config.raw_signal_policy}")
    if generation_config.style_profile not in STYLE_AXIS_PROFILES:
        valid = ", ".join(available_style_profiles())
        raise ValueError(f"unknown style_profile '{generation_config.style_profile}'. valid profiles: {valid}")
    if isinstance(character_card, CharacterCard):
        active_character = character_card
    elif isinstance(character_card, Mapping):
        active_character = CharacterCard.from_mapping(character_card)
    else:
        active_character = load_character_card(generation_config.character_card_path)
    active_session = (
        character_session
        if isinstance(character_session, CharacterSessionState)
        else CharacterSessionState.from_mapping(character_session)
    )

    if str(generation_config.provider).strip().lower() == "openai_compatible":
        ensure_model_server_ready(
            generation_config.base_url,
            generation_config.timeout_sec,
            api_key=generation_config.api_key,
        )
    emonet_input: str | np.ndarray = user_text
    perception_meta: dict[str, Any] = {"mode": "encoder", "usage": {}}
    if generation_config.affect_input_mode in {"llm_perception", "llm_raw_signal"}:
        emonet_input, perception_meta = _build_agent_perceived_stim(
            generation_config=generation_config,
            user_text=user_text,
            history=history,
            character_card=active_character,
            session_state=active_session,
        )
        if generation_config.raw_signal_policy == "raw_pure":
            perception_meta["affective_carryover"] = {"applied": False, "reason": "raw_pure_policy"}
        else:
            emonet_input, perception_meta = _apply_affective_carryover(
                emonet_input,
                perception_meta,
                history,
                active_session,
            )
    profile = infer_style_profile(
        model=runtime.model,
        decoder=runtime.decoder,
        text=emonet_input,
        style_profile=generation_config.style_profile,
    )
    profile["affect_input_mode"] = generation_config.affect_input_mode
    profile["raw_signal_policy"] = generation_config.raw_signal_policy
    profile["affect_input_stim_vec"] = _float_list(emonet_input if not isinstance(emonet_input, str) else profile.get("stim_vec", []))
    profile["agent_perception"] = perception_meta
    if generation_config.conditioning_mode in {"episode_trace", "episode_trace_v3", "hybrid_episode"}:
        if not isinstance(episode_payload, Mapping):
            raise ValueError("episode payload is required for episode-based conditioning")
        profile = augment_profile_with_episode(profile, dict(episode_payload))
    profile = _apply_agent_felt_trace_overrides(profile)
    profile["emotion_state"] = build_emotion_state_record(
        input_text=user_text,
        profile=profile,
        n_neurons=int(getattr(getattr(runtime.model, "config", None), "n_neurons", 256)),
    )
    next_affect_state = _build_session_affect_state(active_session.affect_state, profile)
    profile["session_affect_state"] = next_affect_state
    profile["translation_surface"] = _build_translation_surface(profile)
    felt_self = _build_felt_self_state(active_session.felt_self, profile)
    drive = _build_drive_state(profile, felt_self)
    emotion_memory = _build_emotion_memory(
        active_session.emotion_memory,
        user_text=user_text,
        profile=profile,
        felt_self=felt_self,
    )
    profile["felt_self"] = felt_self
    profile["drive"] = drive
    profile["emotion_memory"] = [dict(item) for item in emotion_memory]
    prompt_session = CharacterSessionState(
        user_memory=active_session.user_memory,
        relationship_state=active_session.relationship_state,
        scene_state=active_session.scene_state,
        affect_state=next_affect_state,
        felt_self=felt_self,
        emotion_memory=emotion_memory,
        drive=drive,
    )

    base_prompt, prompt_sections = build_conditioned_generation_prompt(
        input_text=user_text,
        profile=profile,
        conditioning_mode=generation_config.conditioning_mode,
        template_path=generation_config.prompt_template,
    )
    history_prompt = inject_chat_history(base_prompt, history, generation_config.history_turns)
    prompt = build_character_context_prompt(
        base_prompt=history_prompt,
        character_card=active_character,
        session_state=prompt_session,
        trace_summary=str(profile.get("trace_summary_text", "")),
        appraisal_summary=str(profile.get("appraisal_summary_text", "")),
        raw_trace_block=_build_raw_emonet_trace_block(profile),
    )
    if "qwen3" in str(generation_config.model_name or "").lower():
        prompt = build_compact_character_prompt(
            user_text=user_text,
            history=history,
            character_card=active_character,
            session_state=prompt_session,
            profile=profile,
        )
        prompt_sections = f"qwen_compact_character_context,{prompt_sections}"
    else:
        prompt = append_latest_turn_guard(prompt, user_text=user_text, history=history)
        prompt_sections = f"character_context,{prompt_sections}"
    response_text, _raw_output, response_meta = request_plain_text_response(
        base_url=generation_config.base_url,
        model_name=generation_config.model_name,
        prompt=prompt,
        temperature=generation_config.response_temperature,
        max_tokens=generation_config.max_tokens,
        timeout_sec=generation_config.timeout_sec,
        max_retries=generation_config.response_max_retries,
        validator=lambda raw: validate_contextual_character_response(raw, user_text=user_text, history=history),
        retry_instruction=DEFAULT_RESPONSE_RETRY_INSTRUCTION,
        system_prompt=DEFAULT_REQUEST_SYSTEM_PROMPT,
        api_key=generation_config.api_key,
        reasoning_effort=generation_config.reasoning_effort,
        provider=generation_config.provider,
    )
    perception_usage = dict(perception_meta.get("usage", {}))
    if perception_usage:
        response_meta = dict(response_meta)
        response_usage = dict(response_meta.get("usage", {}))
        response_meta["usage"] = {
            "input_tokens": int(response_usage.get("input_tokens", 0) or 0)
            + int(perception_usage.get("input_tokens", 0) or 0),
            "output_tokens": int(response_usage.get("output_tokens", 0) or 0)
            + int(perception_usage.get("output_tokens", 0) or 0),
        }
    chat_history_excerpt = build_recent_dialogue_block(history, generation_config.history_turns)
    updated_session = update_character_session_state(
        active_session,
        user_text=user_text,
        assistant_text=response_text,
        affect_state=next_affect_state,
        felt_self=felt_self,
        emotion_memory=emotion_memory,
        drive=drive,
    )
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
        character_card=active_character,
        character_session=updated_session,
    )
    return ChatTurnResult(
        assistant_text=response_text,
        record=record,
        character_session=updated_session,
    )
