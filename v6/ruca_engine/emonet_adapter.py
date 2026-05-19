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
        trace_profile = dict(self.profile.get("trace_profile", {}))
        return {
            "source": self.source,
            "emotion_state": self.emotion_state.to_record(),
            "stim_vec": _float_list(self.profile.get("stim_vec", [])),
            "affect_input_stim_vec": _float_list(self.profile.get("affect_input_stim_vec", [])),
            "dominant_branch_len": int(self.profile.get("dominant_branch_len", 0) or 0),
            "trace_summary_text": str(self.profile.get("trace_summary_text", "")),
            "trace_lines": [str(item) for item in self.profile.get("trace_lines", [])],
            "trace_profile": trace_profile,
            "style_tags": [str(item) for item in self.profile.get("style_tags", [])],
            "style_summary": dict(self.profile.get("style_summary", {})),
            "z_dim": len(_float_list(self.profile.get("z", []))),
            "s_pred_dim": len(_float_list(self.profile.get("s_pred", []))),
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
    stim = _pad4(_float_list(profile.get("stim_vec", [])))
    dopamine, serotonin, norepinephrine, melatonin = stim[:4]
    trace_profile = dict(profile.get("trace_profile", {}))
    branch_len = float(profile.get("dominant_branch_len", trace_profile.get("dominant_branch_len", 0)) or 0)
    ticks_run = max(1.0, float(trace_profile.get("ticks_run", 1) or 1))
    active_window = float(trace_profile.get("active_window_ticks", 0) or 0)
    active_ratio = clamp(active_window / ticks_run, 0.0, 1.0)
    mean_active = float(trace_profile.get("mean_active_nodes", 0.0) or 0.0)
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


def _float_list(value: Any) -> list[float]:
    try:
        return [float(item) for item in value]
    except Exception:
        return []


def _pad4(values: list[float]) -> list[float]:
    padded = list(values[:4])
    while len(padded) < 4:
        padded.append(0.0)
    return [clamp(item, 0.0, 1.0) for item in padded]
