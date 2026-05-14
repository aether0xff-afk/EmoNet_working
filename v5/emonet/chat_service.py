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
DEFAULT_REQUEST_SYSTEM_PROMPT = "Return a plain Korean response only. Do not return JSON."
DEFAULT_RESPONSE_RETRY_INSTRUCTION = (
    "직전 응답은 반복, 미완성 문장, bullet/JSON, 혹은 부자연스러운 출력 때문에 거부되었다. "
    "같은 문장이나 핵심 구절을 반복하지 말고, 마지막 문장은 완결된 한국어 평문으로 끝내라."
)
AGENT_PERCEPTION_SYSTEM_PROMPT = "Return JSON only. No markdown."
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
  "confidence": 0.0-1.0
}}

[RULES]
- Do not use keyword matching or lexical hint rules.
- Use situation, relationship, scene pressure, and character persona.
- Do not output emotion names, felt tones, labels, explanations, or rationale.
- Keep signals moderate unless the character would truly spike.
- Calm, trust, acceptance, or tired care should not automatically become high alarm.
- Do not infer fatigue from time of day or quiet scene alone. Raise fatigue only when the latest user input or recent dialogue creates actual drain, withdrawal, or emotional load.
- Ambivalence can have mixed values, not always a single high axis.
- JSON only.
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
        raise ValueError(f"agent perception returned invalid JSON: {last_error}. raw={raw[:300]}")
    raw_signal = payload.get("raw_signal") if isinstance(payload.get("raw_signal"), Mapping) else payload
    vec = _raw_signal_to_stim_vec(raw_signal)
    if float(vec.max()) <= 0.0:
        raise ValueError("agent perception returned an empty stim vector")
    metadata = {
        "mode": "llm_raw_signal",
        "raw": raw,
        "usage": usage,
        "raw_signal": {key: _clamp01(raw_signal.get(key)) for key in (
            "approach_drive",
            "safety_buffer",
            "alarm",
            "fatigue",
            "attachment_pull",
            "control_pressure",
            "novelty",
            "ambiguity",
        )},
        "confidence": _clamp01(payload.get("confidence")),
        "stim_vec": vec.astype(float).tolist(),
    }
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
    if previous_pressure < 0.45 and previous_active < 0.20:
        metadata["affective_carryover"] = {"applied": False, "previous_pressure": previous_pressure}
        return vec, metadata

    current = np.asarray(vec, dtype=np.float32).reshape(4).copy()
    raw_signal = dict(metadata.get("raw_signal", {}))
    attachment = _clamp01(raw_signal.get("attachment_pull"))
    ambiguity = _clamp01(raw_signal.get("ambiguity"))
    fatigue = _clamp01(raw_signal.get("fatigue"))
    relation_load = max(attachment, ambiguity, fatigue)
    blend = min(0.28, 0.08 + 0.18 * max(previous_pressure, previous_active) * relation_load)
    carried = (1.0 - blend) * current + blend * previous_stim

    if relation_load >= 0.45 and previous_pressure >= 0.50:
        carried[2] = max(float(carried[2]), min(0.65, 0.60 * float(previous_stim[2]) + 0.18))
        carried[3] = max(float(carried[3]), min(0.60, 0.65 * float(current[3]) + 0.12 * previous_pressure))
        carried[1] = min(float(carried[1]), max(0.0, float(current[1]) - 0.05 * previous_pressure))

    carried = np.clip(carried, 0.0, 1.0).astype(np.float32)
    metadata["affective_carryover"] = {
        "applied": True,
        "blend": round(float(blend), 4),
        "previous_pressure": round(float(previous_pressure), 4),
        "previous_active_ratio": round(float(previous_active), 4),
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
    raw_signal = dict((profile.get("agent_perception", {}) or {}).get("raw_signal", {}))
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

    decay = 0.52 + 0.30 * relation_load
    if str(felt.get("trace_interpretation", "")) == "no_active_trace":
        decay *= 0.82
    decay = float(np.clip(decay, 0.35, 0.86))
    current_weight = 1.0 - decay
    if current_active >= 0.20 or current_pressure >= 0.45:
        current_weight = max(current_weight, 0.38)
    affect_stim = np.clip(decay * previous_stim + current_weight * current_stim, 0.0, 1.0)
    pressure = max(current_pressure, decay * previous_pressure)
    active_ratio = max(current_active, decay * previous_active)

    if pressure < 0.08 and active_ratio < 0.04:
        affect_stim = current_stim

    return {
        "affect_stim_vec": affect_stim.astype(float).tolist(),
        "felt_pressure": round(float(pressure), 4),
        "active_ratio": round(float(active_ratio), 4),
        "relation_load": round(float(relation_load), 4),
        "decay": round(float(decay), 4),
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

    if str(generation_config.provider).strip().lower() != "anthropic":
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
        session_state=active_session,
        trace_summary=str(profile.get("trace_summary_text", "")),
        appraisal_summary=str(profile.get("appraisal_summary_text", "")),
        raw_trace_block=_build_raw_emonet_trace_block(profile),
    )
    prompt_sections = f"character_context,{prompt_sections}"
    response_text, _raw_output, response_meta = request_plain_text_response(
        base_url=generation_config.base_url,
        model_name=generation_config.model_name,
        prompt=prompt,
        temperature=generation_config.response_temperature,
        max_tokens=generation_config.max_tokens,
        timeout_sec=generation_config.timeout_sec,
        max_retries=generation_config.response_max_retries,
        validator=lambda raw: validate_character_response_text(raw, validate_plain_response_text),
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
