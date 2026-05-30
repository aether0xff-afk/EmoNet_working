from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .models import EmotionState, clamp


ROOT = Path(__file__).resolve().parents[2]
V5_ROOT = ROOT / "v5"
V6_ROOT = ROOT / "v6"


@dataclass(frozen=True)
class EmoNetTraceResult:
    emotion_state: EmotionState
    profile: dict[str, Any]
    source: str = "emonet_v5_runtime_v6_artifacts"

    def to_record(self) -> dict[str, Any]:
        trace_profile = _required_mapping(self.profile, "trace_profile")
        return {
            "source": self.source,
            "emotion_state": self.emotion_state.to_record(),
            "stim_vec": _required_float_list(self.profile, "stim_vec"),
            "affect_input_stim_vec": _required_float_list(self.profile, "affect_input_stim_vec"),
            "dominant_branch_len": _required_int(self.profile, "dominant_branch_len"),
            "trace_summary_text": _required_str(self.profile, "trace_summary_text"),
            "trace_lines": _required_str_list(self.profile, "trace_lines"),
            "trace_profile": trace_profile,
            "style_tags": _required_str_list(self.profile, "style_tags"),
            "style_summary": _required_mapping(self.profile, "style_summary"),
            "z_dim": len(_required_float_list(self.profile, "z")),
            "s_pred_dim": len(_required_float_list(self.profile, "s_pred")),
        }


def infer_emonet_trace(text: str) -> EmoNetTraceResult:
    runtime, infer_style_profile = _runtime_and_infer()
    profile = infer_style_profile(
        model=runtime.model,
        decoder=runtime.decoder,
        text=str(text or ""),
        style_profile="extended40",
    )
    profile = dict(profile)
    emotion_state = _profile_to_emotion_state(profile)
    return EmoNetTraceResult(emotion_state=emotion_state, profile=profile)


@lru_cache(maxsize=1)
def _runtime_and_infer() -> tuple[Any, Any]:
    if str(V5_ROOT) not in sys.path:
        sys.path.insert(0, str(V5_ROOT))
    from emonet.chat_service import ChatRuntimeConfig, build_chat_runtime
    from emonet.legacy_cli import infer_style_profile

    config = ChatRuntimeConfig(
        model_cache_path=V6_ROOT / "artifacts" / "ridge_stim_encoder.joblib",
        z_encoder_path=V6_ROOT / "artifacts" / "dominant_branch_encoder_extended40_calref_v1.pt",
        zs_model_path=V6_ROOT / "artifacts" / "z_to_s_decoder_extended40_calref_v1.npz",
    )
    return build_chat_runtime(config), infer_style_profile


def _profile_to_emotion_state(profile: dict[str, Any]) -> EmotionState:
    stim = _required_stim_vec(profile, "stim_vec")
    dopamine, serotonin, norepinephrine, melatonin = stim[:4]
    trace_profile = _required_mapping(profile, "trace_profile")
    branch_len = _required_float(profile, "dominant_branch_len")
    ticks_run = _required_float(trace_profile, "ticks_run")
    if ticks_run <= 0:
        raise ValueError("trace_profile.ticks_run must be positive")
    active_window = _required_float(trace_profile, "active_window_ticks")
    active_ratio = clamp(active_window / ticks_run, 0.0, 1.0)
    mean_active = _required_float(trace_profile, "mean_active_nodes")
    activity_pressure = clamp(mean_active / 128.0, 0.0, 1.0)

    valence = clamp((dopamine + serotonin) * 0.55 - (norepinephrine + melatonin) * 0.45)
    arousal = clamp(0.15 + norepinephrine * 0.55 + dopamine * 0.25 + active_ratio * 0.20, 0.0, 1.0)
    affinity = clamp(serotonin * 0.52 + dopamine * 0.28 - norepinephrine * 0.20, 0.0, 1.0)
    stability = clamp(serotonin * 0.65 + melatonin * 0.20 - norepinephrine * 0.35 - active_ratio * 0.15, 0.0, 1.0)
    protective = clamp(norepinephrine * 0.55 + activity_pressure * 0.25 + active_ratio * 0.20, 0.0, 1.0)
    curiosity = clamp(dopamine * 0.45 + active_ratio * 0.20 + min(branch_len / 12.0, 1.0) * 0.20, 0.0, 1.0)
    return EmotionState(
        valence=valence,
        arousal=arousal,
        affinity=affinity,
        stability=stability,
        protective_tension=protective,
        curiosity=curiosity,
    )


def _required_float_list(mapping: dict[str, Any], field_name: str) -> list[float]:
    if field_name not in mapping:
        raise ValueError(f"EmoNet profile missing required field: {field_name}")
    value = mapping[field_name]
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"EmoNet profile field {field_name} must be a numeric sequence") from exc
    if not result:
        raise ValueError(f"EmoNet profile field {field_name} must not be empty")
    return result


def _required_stim_vec(mapping: dict[str, Any], field_name: str) -> list[float]:
    values = _required_float_list(mapping, field_name)
    if len(values) != 4:
        raise ValueError(f"EmoNet profile field {field_name} must contain exactly 4 values")
    return [clamp(item, 0.0, 1.0) for item in values]


def _required_mapping(mapping: dict[str, Any], field_name: str) -> dict[str, Any]:
    if field_name not in mapping or not isinstance(mapping[field_name], dict):
        raise ValueError(f"EmoNet profile field {field_name} must be an object")
    return dict(mapping[field_name])


def _required_float(mapping: dict[str, Any], field_name: str) -> float:
    if field_name not in mapping:
        raise ValueError(f"EmoNet profile missing required field: {field_name}")
    try:
        return float(mapping[field_name])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"EmoNet profile field {field_name} must be numeric") from exc


def _required_int(mapping: dict[str, Any], field_name: str) -> int:
    value = _required_float(mapping, field_name)
    if int(value) != value:
        raise ValueError(f"EmoNet profile field {field_name} must be an integer")
    return int(value)


def _required_str(mapping: dict[str, Any], field_name: str) -> str:
    if field_name not in mapping or not isinstance(mapping[field_name], str):
        raise ValueError(f"EmoNet profile field {field_name} must be a string")
    return mapping[field_name]


def _required_str_list(mapping: dict[str, Any], field_name: str) -> list[str]:
    if field_name not in mapping:
        raise ValueError(f"EmoNet profile missing required field: {field_name}")
    value = mapping[field_name]
    if not isinstance(value, list):
        raise ValueError(f"EmoNet profile field {field_name} must be a list")
    return [str(item) for item in value]
