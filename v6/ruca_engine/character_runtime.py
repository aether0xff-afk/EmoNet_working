from __future__ import annotations

from .context import TurnContext
from .emotion import InputSignals
from .models import CharacterProfile, ResponseDecision
from .trait_state import CharacterTraitState


def select_visible_speaker(
    *,
    profiles: dict[str, CharacterProfile],
    user_text: str,
    signals: InputSignals,
    context: TurnContext,
    response_decision: ResponseDecision,
    trait_state: CharacterTraitState,
) -> CharacterProfile | None:
    if response_decision.action != "send_message":
        return None

    lowered = user_text.lower()
    if _needs_analysis(lowered, signals, context, trait_state):
        return profiles["ricky"]
    if _needs_action(lowered, signals, trait_state):
        return profiles["rocky"]
    return profiles["ruca"]


def _needs_analysis(text: str, signals: InputSignals, context: TurnContext, trait_state: CharacterTraitState) -> bool:
    ricky = trait_state.characters.get("ricky", {})
    keywords = ("분석", "구조", "정리", "원인", "비교")
    return any(keyword in text for keyword in keywords) or (
        context.event_type == "question" and signals.curiosity >= 0.35 and ricky.get("analysis", 0.0) >= 0.55
    )


def _needs_action(text: str, signals: InputSignals, trait_state: CharacterTraitState) -> bool:
    rocky = trait_state.characters.get("rocky", {})
    keywords = ("바로", "실행", "멈추지", "빨리", "긴급")
    return signals.action_pressure >= 0.60 or (any(keyword in text for keyword in keywords) and rocky.get("initiative", 0.0) >= 0.55)
