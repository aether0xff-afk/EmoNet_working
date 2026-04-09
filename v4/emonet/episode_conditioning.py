from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def _format_rule_block(rules: Sequence[str], default_line: str) -> str:
    cleaned = [str(rule).strip() for rule in rules if str(rule).strip()]
    if not cleaned:
        return default_line
    return "\n".join(f"- {line}" for line in cleaned)


def _format_style_summary(style_summary: Mapping[str, object]) -> str:
    items: list[tuple[str, float]] = []
    for key, value in style_summary.items():
        try:
            items.append((str(key), float(value)))
        except (TypeError, ValueError):
            continue
    if not items:
        return "(none)"
    items.sort(key=lambda item: abs(item[1]), reverse=True)
    return "\n".join(f"- {key}={value:.4f}" for key, value in items[:4])


def load_episode_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"episode payload must be a JSON object: {path}")
    return payload


def resolve_episode_payload_path(
    *,
    episode_dir: Path,
    record: Mapping[str, object],
    explicit_record_id_column: str | None = None,
) -> Path:
    candidate_ids: list[str] = []
    if explicit_record_id_column:
        value = str(record.get(explicit_record_id_column, "") or "").strip()
        if value:
            candidate_ids.append(value)
    for key in ("sample_id", "record_id", "talk_id"):
        value = str(record.get(key, "") or "").strip()
        if value and value not in candidate_ids:
            candidate_ids.append(value)
    for candidate_id in candidate_ids:
        candidate_path = episode_dir / candidate_id / "episode_interpretation.json"
        if candidate_path.exists():
            return candidate_path
    hint = ", ".join(candidate_ids) if candidate_ids else "(no record id candidates)"
    raise FileNotFoundError(f"episode_interpretation.json not found under {episode_dir} for record ids: {hint}")


def build_episode_summary_text(payload: Mapping[str, Any]) -> str:
    appraisal = payload.get("appraisal") or {}
    parts = [
        str(payload.get("episode_label", "")).strip(),
        str(appraisal.get("primary_appraisal", "")).strip(),
        str(payload.get("action_tendency", "")).strip(),
    ]
    return " | ".join(part for part in parts if part)


def build_episode_lines(payload: Mapping[str, Any]) -> list[str]:
    appraisal = payload.get("appraisal") or {}
    trajectory = payload.get("trajectory") or {}
    rawness = payload.get("rawness") or {}
    guidance = payload.get("response_guidance") or {}
    evidence = payload.get("evidence") or []
    evidence_lines = [str(item).strip() for item in evidence if str(item).strip()][:4]
    return [
        f"episode_label: {str(payload.get('episode_label', '')).strip()}",
        f"stimulus_reading: {str(payload.get('stimulus_reading', '')).strip()}",
        f"primary_appraisal: {str(appraisal.get('primary_appraisal', '')).strip()}",
        f"secondary_appraisal: {str(appraisal.get('secondary_appraisal', '')).strip()}",
        "appraisal_state: "
        f"target={str(appraisal.get('target', '')).strip()}, "
        f"control={str(appraisal.get('control_state', '')).strip()}, "
        f"social_orientation={str(appraisal.get('social_orientation', '')).strip()}",
        f"trajectory_overall: {str(trajectory.get('overall_pattern', '')).strip()}",
        f"trajectory_ignition: {str(trajectory.get('ignition', '')).strip()}",
        f"trajectory_persistence: {str(trajectory.get('persistence', '')).strip()}",
        f"trajectory_resolution: {str(trajectory.get('resolution', '')).strip()}",
        f"action_tendency: {str(payload.get('action_tendency', '')).strip()}",
        "rawness: "
        f"valence={str(rawness.get('valence', '')).strip()}, "
        f"arousal={str(rawness.get('arousal', '')).strip()}, "
        f"softened_output_risk={str(rawness.get('softened_output_risk', '')).strip()}, "
        f"preserve_harshness={bool(rawness.get('should_preserve_harshness', False))}",
        f"response_preserve: {str(guidance.get('preserve', '')).strip()}",
        f"response_avoid: {str(guidance.get('avoid', '')).strip()}",
        f"response_tone_hint: {str(guidance.get('tone_hint', '')).strip()}",
        *[f"evidence: {line}" for line in evidence_lines],
    ]


def augment_profile_with_episode(
    profile: Mapping[str, Any],
    episode_payload: Mapping[str, Any],
    *,
    episode_source_path: str | None = None,
) -> dict[str, Any]:
    enriched = dict(profile)
    payload_dict = dict(episode_payload)
    enriched["episode_payload"] = payload_dict
    enriched["episode_label"] = str(payload_dict.get("episode_label", "")).strip()
    enriched["episode_summary_text"] = build_episode_summary_text(payload_dict)
    enriched["episode_lines"] = build_episode_lines(payload_dict)
    enriched["episode_source_path"] = str(episode_source_path or "")
    return enriched


def build_episode_generation_prompt(
    *,
    input_text: str,
    episode_payload: Mapping[str, Any],
    anti_softening_rules: Sequence[str] | None = None,
    grounding_rules: Sequence[str] | None = None,
) -> str:
    episode_block = "\n".join(f"- {line}" for line in build_episode_lines(episode_payload))
    anti_block = _format_rule_block(
        anti_softening_rules or [],
        "- 입력에 없는 위로나 공손함을 자동으로 덧붙이지 않는다.",
    )
    grounding_block = _format_rule_block(
        grounding_rules or [],
        "- 첫 문장은 입력의 정서와 직접 연결되게 시작한다.",
    )
    return "\n".join(
        [
            "[ROLE]",
            "당신은 내부 감정 episode 해석을 읽고 그 결을 유지한 채 한국어로 답하는 응답 생성기다.",
            "",
            "[USER_INPUT]",
            input_text.strip(),
            "",
            "[EPISODE_TRACE]",
            episode_block if episode_block else "- episode 정보 없음",
            "",
            "[ANTI_SOFTENING_RULES]",
            anti_block,
            "",
            "[GROUNDING_RULES]",
            grounding_block,
            "",
            "[INSTRUCTIONS]",
            "- 사용자 입력의 내용에 직접 답한다.",
            "- EPISODE_TRACE를 보고 자극, appraisal, trajectory, action tendency를 반영한다.",
            "- EPISODE_TRACE의 섹션 이름, field 이름, JSON 구조를 그대로 언급하지 않는다.",
            "- response_preserve와 tone_hint를 우선 반영한다.",
            "- response_avoid에 적힌 순화나 왜곡을 하지 않는다.",
            "- softened_output_risk가 높고 preserve_harshness가 true면 감정의 날카로운 결을 유지한다.",
            "- 한국어 평문으로만 2~5문장 이내로 답한다.",
            "- 같은 문장이나 핵심 구절을 반복하지 않는다.",
            "- 문장을 중간에 끊거나 조건절로 끝내지 않는다. 마지막 문장은 완결된 문장으로 끝낸다.",
            "- bullet, markdown, JSON, 코드블록을 쓰지 않는다.",
        ]
    )


def build_hybrid_episode_generation_prompt(
    *,
    input_text: str,
    style_tags: Sequence[str],
    style_summary: Mapping[str, object],
    episode_payload: Mapping[str, Any],
    anti_softening_rules: Sequence[str] | None = None,
    grounding_rules: Sequence[str] | None = None,
) -> str:
    episode_block = "\n".join(f"- {line}" for line in build_episode_lines(episode_payload))
    anti_block = _format_rule_block(
        anti_softening_rules or [],
        "- 입력에 없는 위로나 공손함을 자동으로 덧붙이지 않는다.",
    )
    grounding_block = _format_rule_block(
        grounding_rules or [],
        "- 첫 문장은 입력의 정서와 직접 연결되게 시작한다.",
    )
    return "\n".join(
        [
            "[ROLE]",
            "당신은 감정 episode 해석과 스타일 요약을 함께 참고해 한국어로 답하는 응답 생성기다.",
            "",
            "[USER_INPUT]",
            input_text.strip(),
            "",
            "[EPISODE_TRACE]",
            episode_block if episode_block else "- episode 정보 없음",
            "",
            "[STYLE_TAGS]",
            ", ".join(str(tag).strip() for tag in style_tags if str(tag).strip()) or "(none)",
            "",
            "[STYLE_SUMMARY]",
            _format_style_summary(style_summary),
            "",
            "[ANTI_SOFTENING_RULES]",
            anti_block,
            "",
            "[GROUNDING_RULES]",
            grounding_block,
            "",
            "[INSTRUCTIONS]",
            "- 사용자 입력의 내용에 직접 답한다.",
            "- EPISODE_TRACE를 우선 참고해 감정의 인과와 행동 성향을 잡는다.",
            "- STYLE_TAGS와 STYLE_SUMMARY는 말투 밀도와 거리감만 미세 조정하는 데 쓴다.",
            "- STYLE 정보와 EPISODE_TRACE가 충돌하면 EPISODE_TRACE를 우선한다.",
            "- 섹션 이름이나 JSON field 이름을 그대로 언급하지 않는다.",
            "- 한국어 평문으로만 2~5문장 이내로 답한다.",
            "- 같은 문장이나 핵심 구절을 반복하지 않는다.",
            "- 문장을 중간에 끊거나 조건절로 끝내지 않는다. 마지막 문장은 완결된 문장으로 끝낸다.",
            "- bullet, markdown, JSON, 코드블록을 쓰지 않는다.",
        ]
    )
