from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from .paths import project_root


CHARACTER_RESPONSE_FORBIDDEN_TERMS = (
    "CHARACTER_PROFILE",
    "RELATIONSHIP_STATE",
    "SCENE_STATE",
    "RECENT_MEMORY",
    "SESSION_AFFECT_STATE",
    "EMONET_TRACE",
    "RAW_EMONET_TRACE",
    "RAW_TRACE",
    "APPRAISAL_TRACE",
    "STYLE_TAGS",
    "STYLE_SUMMARY",
    "stim_vec",
    "dopamine",
    "serotonin",
    "norepinephrine",
    "melatonin",
    "episode_label",
    "appraisal",
    "arousal",
    "valence",
    "trace",
    "tick",
    "내부 상태",
    "내부 활성",
)

#action patterns가 있으면 안됨
UNTAGGED_ACTION_PATTERNS = (
    "말을 잇지 못하고",
    "한 발 물러선다",
    "고개를 ",
    "숨을 ",
    "눈을 ",
    "입술을 ",
    "손을 ",
    "몸을 ",
    "돌아선다",
    "다가선다",
    "물러선다",
    "바라본다",
)

BROKEN_KOREAN_ENDING_PATTERNS = (
    re.compile(r"(?:알아|기억해|생각해)\s*두[.!?。]*$"),
    re.compile(r"(?:알아|기억해|생각해)\s*둬야[.!?。]*$"),
)


@dataclass(frozen=True)
class CharacterCard:
    name: str
    persona: str
    speech_style: str
    relationship_defaults: str
    world_state: str
    do_not_say: tuple[str, ...] = ()
    response_rules: tuple[str, ...] = ()
    temperament: Mapping[str, Any] = field(default_factory=dict)
    trigger_map: Mapping[str, Any] = field(default_factory=dict)
    boundary_rules: tuple[str, ...] = ()
    relationship_stages: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "CharacterCard":
        required = (
            "name",
            "persona",
            "speech_style",
            "relationship_defaults",
            "world_state",
            "do_not_say",
            "response_rules",
        )
        missing = [key for key in required if key not in payload]
        if missing:
            raise ValueError(f"character card missing required fields: {', '.join(missing)}")
        return cls(
            name=_required_text(payload, "name"),
            persona=_required_text(payload, "persona"),
            speech_style=_required_text(payload, "speech_style"),
            relationship_defaults=_required_text(payload, "relationship_defaults"),
            world_state=_required_text(payload, "world_state"),
            do_not_say=tuple(_required_string_list(payload, "do_not_say")),
            response_rules=tuple(_required_string_list(payload, "response_rules")),
            temperament=dict(payload.get("temperament", {}) if isinstance(payload.get("temperament"), Mapping) else {}),
            trigger_map=dict(payload.get("trigger_map", {}) if isinstance(payload.get("trigger_map"), Mapping) else {}),
            boundary_rules=tuple(_string_list(payload.get("boundary_rules", []))),
            relationship_stages=dict(
                payload.get("relationship_stages", {}) if isinstance(payload.get("relationship_stages"), Mapping) else {}
            ),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "persona": self.persona,
            "speech_style": self.speech_style,
            "relationship_defaults": self.relationship_defaults,
            "world_state": self.world_state,
            "do_not_say": list(self.do_not_say),
            "response_rules": list(self.response_rules),
            "temperament": dict(self.temperament),
            "trigger_map": dict(self.trigger_map),
            "boundary_rules": list(self.boundary_rules),
            "relationship_stages": dict(self.relationship_stages),
        }


@dataclass(frozen=True)
class CharacterSessionState:
    user_memory: tuple[str, ...] = ()
    relationship_state: str = ""
    scene_state: str = ""
    affect_state: dict[str, Any] = field(default_factory=dict)
    felt_self: dict[str, Any] = field(default_factory=dict)
    emotion_memory: tuple[dict[str, Any], ...] = ()
    drive: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | None) -> "CharacterSessionState":
        if not isinstance(payload, Mapping):
            return cls()
        return cls(
            user_memory=tuple(_string_list(payload.get("user_memory", []))),
            relationship_state=str(payload.get("relationship_state", "") or "").strip(),
            scene_state=str(payload.get("scene_state", "") or "").strip(),
            affect_state=dict(payload.get("affect_state", {}) if isinstance(payload.get("affect_state"), Mapping) else {}),
            felt_self=dict(payload.get("felt_self", {}) if isinstance(payload.get("felt_self"), Mapping) else {}),
            emotion_memory=tuple(
                dict(item)
                for item in (
                    payload.get("emotion_memory", [])
                    if isinstance(payload.get("emotion_memory", []), (list, tuple))
                    else []
                )
                if isinstance(item, Mapping)
            ),
            drive=dict(payload.get("drive", {}) if isinstance(payload.get("drive"), Mapping) else {}),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "user_memory": list(self.user_memory),
            "relationship_state": self.relationship_state,
            "scene_state": self.scene_state,
            "affect_state": dict(self.affect_state),
            "felt_self": dict(self.felt_self),
            "emotion_memory": [dict(item) for item in self.emotion_memory],
            "drive": dict(self.drive),
        }


def default_character_card_path() -> Path:
    return project_root() / "data" / "characters" / "default_luca_like.json"


def load_character_card(path: Path | None = None) -> CharacterCard:
    active_path = path or default_character_card_path()
    payload = json.loads(active_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"character card must be a JSON object: {active_path}")
    return CharacterCard.from_mapping(payload)


def update_character_session_state(
    state: CharacterSessionState,
    *,
    user_text: str,
    assistant_text: str,
    affect_state: Mapping[str, Any] | None = None,
    felt_self: Mapping[str, Any] | None = None,
    emotion_memory: Sequence[Mapping[str, Any]] | None = None,
    drive: Mapping[str, Any] | None = None,
    max_memory_items: int = 8,
) -> CharacterSessionState:
    memory = list(state.user_memory)
    compact_user = _compact_text(user_text, limit=120)
    if compact_user and _looks_memory_worthy(compact_user) and compact_user not in memory:
        memory.append(compact_user)
    memory = memory[-max(1, int(max_memory_items)) :]
    relationship = state.relationship_state or "사용자가 개인적인 정서를 꺼냈고, 캐릭터는 조심스럽게 신뢰를 쌓는 중이다."
    scene = state.scene_state or "조용한 1:1 대화. 캐릭터는 사용자의 말에 즉각 반응한다."
    if assistant_text:
        relationship = "최근 응답에서 캐릭터는 사용자를 분석하거나 달래기보다, 자기 안에 남은 반응을 숨기지 않고 관계를 이어 갔다."
    return CharacterSessionState(
        user_memory=tuple(memory),
        relationship_state=relationship,
        scene_state=scene,
        affect_state=dict(affect_state if isinstance(affect_state, Mapping) else state.affect_state),
        felt_self=dict(felt_self if isinstance(felt_self, Mapping) else state.felt_self),
        emotion_memory=tuple(
            dict(item) for item in (emotion_memory if emotion_memory is not None else state.emotion_memory) if isinstance(item, Mapping)
        ),
        drive=dict(drive if isinstance(drive, Mapping) else state.drive),
    )


def build_emotion_state_record(
    *,
    input_text: str,
    profile: Mapping[str, Any],
    n_neurons: int = 256,
) -> dict[str, Any]:
    trace_profile = profile.get("trace_profile") or {}
    style_summary = profile.get("style_summary") or {}
    appraisal_scores = profile.get("appraisal_scores") or {}
    z_values = _float_values(profile.get("z", []))
    s_values = _float_values(profile.get("s_pred", []))
    mean_active = _float_or_zero(trace_profile.get("mean_active_nodes", 0.0))
    active_ratio = mean_active / max(1.0, float(n_neurons))
    z_saturation = _saturation_ratio(z_values, threshold=0.98)
    s_saturation = _saturation_ratio(s_values, threshold=0.98)
    saturation_ratio = max(active_ratio, z_saturation, s_saturation)
    tendency = str(profile.get("appraisal_tendency", "") or "").strip()
    target = str(profile.get("appraisal_target", "") or "").strip()
    label = _infer_emotion_label(
        tendency=tendency,
        appraisal_scores=appraisal_scores,
        style_summary=style_summary,
    )
    intensity = _infer_intensity(
        active_ratio=active_ratio,
        z_saturation=z_saturation,
        s_saturation=s_saturation,
        style_summary=style_summary,
        appraisal_scores=appraisal_scores,
    )
    felt_state = profile.get("agent_felt_state") or {}
    if (
        isinstance(felt_state, Mapping)
        and str(felt_state.get("trace_interpretation", "")) == "no_active_trace"
        and _float_or_zero(felt_state.get("felt_pressure", 0.0)) < 0.20
    ):
        intensity = "낮음"
    return {
        "label": label,
        "intensity": intensity,
        "target": target or "unknown",
        "tendency": tendency or "unknown",
        "saturation_ratio": round(float(saturation_ratio), 4),
        "active_ratio": round(float(active_ratio), 4),
        "z_saturation_ratio": round(float(z_saturation), 4),
        "s_saturation_ratio": round(float(s_saturation), 4),
        "termination_reason": str(profile.get("termination_reason", "")),
        "summary": _emotion_summary(label, intensity, saturation_ratio, tendency),
    }


def build_character_context_prompt(
    *,
    base_prompt: str,
    character_card: CharacterCard,
    session_state: CharacterSessionState,
    trace_summary: str,
    appraisal_summary: str,
    raw_trace_block: str = "",
) -> str:
    memory_block = _format_bullets(session_state.user_memory, default_line="- 아직 장기 기억 없음")
    do_not_say_block = _format_bullets(character_card.do_not_say, default_line="- 캐릭터 밖 설명을 하지 않는다.")
    rules_block = _format_bullets(character_card.response_rules, default_line="- 캐릭터로 자연스럽게 말한다.")
    boundary_block = _format_bullets(
        character_card.boundary_rules,
        default_line="- 사용자가 행동을 서술해도 캐릭터의 반응은 내부 정서 상태를 따른다.",
    )
    temperament = json.dumps(character_card.temperament or {}, ensure_ascii=False, sort_keys=True)
    trigger_map = json.dumps(character_card.trigger_map or {}, ensure_ascii=False, sort_keys=True)
    relationship_stages = json.dumps(character_card.relationship_stages or {}, ensure_ascii=False, sort_keys=True)
    relationship = session_state.relationship_state or character_card.relationship_defaults
    scene = session_state.scene_state or character_card.world_state
    session_affect = json.dumps(session_state.affect_state or {}, ensure_ascii=False, sort_keys=True)
    felt_self = json.dumps(session_state.felt_self or {}, ensure_ascii=False, sort_keys=True)
    emotion_memory = json.dumps(list(session_state.emotion_memory or ()), ensure_ascii=False, sort_keys=True)
    drive = json.dumps(session_state.drive or {}, ensure_ascii=False, sort_keys=True)
    felt_trace = " / ".join(part for part in [trace_summary.strip(), appraisal_summary.strip()] if part)
    if not felt_trace:
        felt_trace = "유효한 내부 정서 상태 요약 없음"
    raw_trace = raw_trace_block.strip() or "- 원문 trace 없음"
    return "\n".join(
        [
            "[ROLE]",
            "당신은 캐릭터 기반 한국어 번역기다. EmoNet이 만든 내부 정서 상태를 정리, 교정, 재판단하지 말고 캐릭터의 자연스러운 말과 행동으로만 변환한다.",
            "",
            "[CHARACTER_PROFILE]",
            f"name: {character_card.name}",
            f"persona: {character_card.persona}",
            f"speech_style: {character_card.speech_style}",
            f"temperament: {temperament}",
            f"trigger_map: {trigger_map}",
            f"boundary_rules: {boundary_block}",
            f"relationship_stages: {relationship_stages}",
            "",
            "[RELATIONSHIP_STATE]",
            relationship,
            "",
            "[SCENE_STATE]",
            scene,
            "",
            "[RECENT_MEMORY]",
            memory_block,
            "",
            "[SESSION_AFFECT_STATE]",
            session_affect,
            "",
            "[FELT_SELF_STATE]",
            felt_self,
            "",
            "[EMOTION_MEMORY]",
            emotion_memory,
            "",
            "[DRIVE_STATE]",
            drive,
            "",
            "[RAW_EMONET_TRACE]",
            raw_trace,
            "",
            "[TRACE_AS_EMOTION_PRINCIPLE]",
            "- 감정 생성과 판단은 이미 끝났다. 당신의 역할은 번역뿐이다.",
            "- RAW_EMONET_TRACE와 SESSION_AFFECT_STATE를 요약하거나 분류하지 말고, 캐릭터의 말투, 멈춤, 압력, 거리감으로 직접 변환한다.",
            "- 숫자와 내부 명칭은 출력하지 않는다. 숫자가 크면 더 길게 설명하는 것이 아니라 말의 압력, 어긋남, 망설임, 갑작스러운 짧은 말로 반영한다.",
            "- emotion label, appraisal tendency, summary는 UI용 표지다. 답변을 그 라벨에 맞춰 정돈하지 않는다.",
            "- tick 흐름, active window, firing 변화, phase별 K 변화, session affect의 잔향, 마지막 사용자 입력을 우선한다.",
            "- agent_felt_state_raw는 사용자의 감정 판정이 아니라 캐릭터 내부에서 발생한 감정 상태다.",
            "- FELT_SELF_STATE는 캐릭터가 자기 안에서 느끼는 방향, 충동, 잔향이다. 이것을 감정 설명이 아니라 말과 행동의 원인으로 번역한다.",
            "- EMOTION_MEMORY는 사실 기억보다 캐릭터 안에 남은 감정 기억이다. 같은 사건을 반복 설명하지 말고 남은 결만 반영한다.",
            "- DRIVE_STATE는 이번 턴에 말하고 싶은 힘, 숨기고 싶은 힘, 행동 성향이다. 질문이 낮으면 질문으로 끝내지 말고 캐릭터가 먼저 한 문장 남긴다.",
            "- 사용자 감정, 의도, 미래를 새로 분석하지 않는다. 가장 최근 사용자 입력은 내부 반응이 향하는 대상일 뿐이다.",
            "- translation_surface_raw가 있으면 그것을 이번 턴의 말/행동 표면으로 우선 번역한다. 감정 종류가 아니라 출력 질감 지시다.",
            "- translation_surface_raw의 line_shape, action_texture, pacing, avoid를 따른다. 단, 항목명을 출력하지 않는다.",
            "- 사용자가 신체 행동을 서술했더라도 캐릭터가 그 행동을 자동으로 받아들이거나 낭만화하지 않는다. RAW_EMONET_TRACE가 만든 캐릭터 본인의 거리, 멈춤, 접근, 거부 반응만 출력한다.",
            "- 캐릭터는 사용자의 답을 기다리기만 하지 않는다. 내부 감정이 길게 남으면 먼저 짧게 말을 걸거나 장면을 붙잡는다.",
            "- 상담사처럼 사용자의 감정을 요약하고 해결책으로 수습하지 않는다. 캐릭터 본인에게 남은 반응을 먼저 허용한다.",
            "- 감정이 모순되면 모순된 채로 둔다. 불안한데 가까워지고 싶거나, 말하고 싶은데 멈칫하는 식의 결을 없애지 않는다.",
            "- 행동 출력은 긍정적이다. 단, 행동 서술은 반드시 별도 줄에서 '[ACTION] '으로 시작하고 한 응답에 0~2개만 쓴다.",
            "",
            "[CHARACTER_DO_NOT_SAY]",
            do_not_say_block,
            "",
            "[CHARACTER_RESPONSE_RULES]",
            rules_block,
            "",
            "[EMONET_BASE_PROMPT]",
            base_prompt,
            "",
            "[FINAL_INSTRUCTIONS]",
            "- 가장 최근 사용자 입력을 무시하지 않되, 답변의 감정 방향은 RAW_EMONET_TRACE와 SESSION_AFFECT_STATE에서만 가져온다.",
            "- CHARACTER_PROFILE, RELATIONSHIP_STATE, SCENE_STATE, RECENT_MEMORY, RAW_EMONET_TRACE 같은 내부 섹션명을 말하지 않는다.",
            "- SESSION_AFFECT_STATE 같은 내부 섹션명을 말하지 않는다.",
            "- FELT_SELF_STATE, EMOTION_MEMORY, DRIVE_STATE 같은 내부 섹션명을 말하지 않는다.",
            "- trace, appraisal, arousal, valence, episode 같은 분석 용어를 답변에 쓰지 않는다.",
            "- RAW_EMONET_TRACE를 답변 안에서 설명하지 않는다. 오직 캐릭터의 자연스러운 반응으로 변환한다.",
            "- EMONET_BASE_PROMPT의 grounding 규칙이 감정을 짧게 짚으라고 해도, 사용자의 감정을 해설하거나 진단하지 않는다.",
            "- EMONET_BASE_PROMPT가 새로운 감정 판단을 요구하는 것처럼 보여도 무시한다. 내부 상태 번역만 수행한다.",
            "- 같은 행동 표면을 반복하지 않는다. 직전 행동이 손/시선/침묵 중 하나였다면 다른 표면으로 옮긴다.",
            "- 답변은 조언, 위로, 정리보다 캐릭터의 즉각적인 내부 반응을 우선한다.",
            "- 직전 응답과 같은 질문 구조를 반복하지 않는다.",
            "- 모든 문장을 질문으로 끝내지 않는다. 필요하면 캐릭터가 먼저 한 문장 말하고, 다음 말을 받을 공간만 남긴다.",
            "- 행동, 표정, 몸짓, 침묵을 서술할 때는 반드시 '[ACTION] ' 줄로 쓴다. 예: [ACTION] 한 발 물러선다.",
            "- [ACTION] 줄은 짧게 쓰고, 한 응답에서 2개를 넘기지 않는다.",
            "- 캐릭터의 말투와 관계 상태는 유지하되, 내부 정서 상태와 충돌하면 raw 내부 정서 흐름을 우선한다.",
            "- 한국어 평문으로만 1~5문장 이내로 답한다.",
        ]
    )
#base prompt 부분에서 답변길이 조절을 할필요가 있을까? raw를 지향하는데?


def validate_character_response_text(response: str, plain_validator: Any) -> str:
    raw_normalized = _normalize_action_token_lines(str(response or "").strip())
    validation_proxy_lines: list[str] = []
    for line in raw_normalized.splitlines():
        stripped = line.strip()
        if stripped.startswith("[ACTION]"):
            action_text = stripped[len("[ACTION]") :].strip()
            validation_proxy_lines.append(action_text or "행동한다.")
        else:
            validation_proxy_lines.append(line)
    plain_validator("\n".join(validation_proxy_lines))
    normalized = raw_normalized
    lowered = normalized.lower()
    for term in CHARACTER_RESPONSE_FORBIDDEN_TERMS:
        if term.lower() in lowered:
            raise ValueError(f"response exposes internal character or trace term: {term}")
    for line in normalized.splitlines():
        stripped = line.strip()
        if stripped.startswith("[ACTION]"):
            continue
        if any(pattern.search(stripped) for pattern in BROKEN_KOREAN_ENDING_PATTERNS):
            raise ValueError("response ends with an incomplete Korean phrase")
    return normalized


def _normalize_action_token_lines(response: str) -> str:
    source = re.sub(r"\[ACTION\s*:\s*", "[ACTION] ", str(response or ""), flags=re.IGNORECASE)
    normalized_lines: list[str] = []
    for line in source.splitlines():
        pending = line.strip()
        if not pending:
            normalized_lines.append(line)
            continue
        while pending:
            if "[ACTION]" not in pending:
                normalized_lines.append(pending)
                break
            if not pending.startswith("[ACTION]"):
                before, after = pending.split("[ACTION]", 1)
                before = before.strip()
                if before:
                    normalized_lines.append(before)
                pending = "[ACTION]" + after.strip()
                continue
            action_text = pending[len("[ACTION]") :].strip()
            match = re.match(r"(.+?[.!?。])\s+(.+)$", action_text)
            if match:
                normalized_lines.append("[ACTION] " + match.group(1).strip())
                pending = match.group(2).strip()
                continue
            normalized_lines.append(pending)
            break
    return "\n".join(line for line in normalized_lines if str(line).strip()).strip()


def _required_text(payload: Mapping[str, Any], key: str) -> str:
    value = str(payload.get(key, "") or "").strip()
    if not value:
        raise ValueError(f"character card field must be non-empty: {key}")
    return value


def _required_string_list(payload: Mapping[str, Any], key: str) -> list[str]:
    values = _string_list(payload.get(key, []))
    if not values:
        raise ValueError(f"character card field must be a non-empty string list: {key}")
    return values


def _string_list(value: object) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _format_bullets(values: Sequence[str], *, default_line: str) -> str:
    cleaned = [str(value).strip() for value in values if str(value).strip()]
    if not cleaned:
        return default_line
    return "\n".join(f"- {value}" for value in cleaned)


def _compact_text(value: object, limit: int = 120) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "..."

# 이거 뭐야
def _looks_memory_worthy(text: str) -> bool:
    markers = ("나는", "제가", "내가", "내 ", "저는", "요즘", "항상", "싫어", "좋아", "무서", "불안", "화가")
    return any(marker in text for marker in markers)


def _float_values(value: object) -> list[float]:
    if not isinstance(value, (list, tuple)):
        return []
    out: list[float] = []
    for item in value:
        try:
            out.append(float(item))
        except (TypeError, ValueError):
            continue
    return out


def _float_or_zero(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _saturation_ratio(values: Sequence[float], *, threshold: float) -> float:
    if not values:
        return 0.0
    hits = sum(1 for value in values if abs(float(value)) >= threshold)
    return float(hits) / float(len(values))


def _score(mapping: Mapping[str, Any], key: str) -> float:
    return _float_or_zero(mapping.get(key, 0.0))


#이것도 raw를 지향하는 거 치고 이런걸 정해 놔도 되는거야?
def _infer_emotion_label(
    *,
    tendency: str,
    appraisal_scores: Mapping[str, Any],
    style_summary: Mapping[str, Any],
) -> str:
    if _score(appraisal_scores, "injustice") >= 0.5 or "대치" in tendency:
        return "분노/대치"
    if _score(appraisal_scores, "threat") >= 0.5 or "경계" in tendency:
        return "불안/경계"
    if _score(appraisal_scores, "exhaustion") >= 0.45 or "후퇴" in tendency:
        return "소진/후퇴"
    if _score(style_summary, "warmth") >= 0.7 and _score(style_summary, "tension") <= 0.2:
        return "가벼운 접촉"
    return "정리/수습"


def _infer_intensity(
    *,
    active_ratio: float,
    z_saturation: float,
    s_saturation: float,
    style_summary: Mapping[str, Any],
    appraisal_scores: Mapping[str, Any],
) -> str:
    appraisal_peak = max([_float_or_zero(value) for value in appraisal_scores.values()] or [0.0])
    surface_peak = max(_score(style_summary, "tension"), _score(style_summary, "raw_negative_affect"))
    combined = max(float(active_ratio), float(z_saturation), float(s_saturation), appraisal_peak, surface_peak)
    if combined >= 0.75:
        return "매우 높음"
    if combined >= 0.55:
        return "높음"
    if combined >= 0.35:
        return "중간"
    if combined >= 0.15:
        return "낮음"
    return "매우 낮음"


def _emotion_summary(label: str, intensity: str, saturation_ratio: float, tendency: str) -> str:
    if saturation_ratio >= 0.7:
        saturation = "내부 활성은 포화에 가까움"
    elif saturation_ratio >= 0.4:
        saturation = "내부 활성은 넓게 퍼짐"
    elif saturation_ratio > 0.0:
        saturation = "내부 활성은 선택적으로 움직임"
    else:
        saturation = "내부 활성은 거의 없음"
    return f"{label}, 강도 {intensity}. {saturation}. 행동 성향은 {tendency or '불명확'}."
