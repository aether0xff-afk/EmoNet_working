from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .character_runtime import select_visible_speaker
from .composer import compose_response
from .context import TurnContext, analyze_turn_context
from .emotion import update_emotion_for_event
from .emonet_adapter import EmoNetTraceResult, infer_emonet_trace
from .event_scheduler import RucaEvent, make_event, schedule_event
from .inner_voice import generate_inner_voices
from .llm_client import LLMConfig, LLMResponse, generate_llm_response
from .memory import MemoryStore
from .models import EmotionState, InnerVoiceCandidate, MemoryItem, ResponseDecision, SpontaneousReactionDecision
from .plot_manager import update_plot_state
from .prompt_builder import build_response_prompt
from .profiles import load_character_profiles
from .relationship_graph import update_relationship_graph
from .response_gate import decide_response_action
from .session import RucaSessionState, SessionStore
from .spontaneous import decide_spontaneous_reaction
from .trait_state import update_trait_state


@dataclass(frozen=True)
class TurnResult:
    assistant_text: str
    previous_emotion_state: EmotionState
    emotion_state: EmotionState
    turn_context: TurnContext
    retrieved_memories: tuple[MemoryItem, ...]
    inner_voices: tuple[InnerVoiceCandidate, ...]
    spontaneous_reaction: SpontaneousReactionDecision
    response_decision: ResponseDecision
    response_prompt: str
    emonet_trace: EmoNetTraceResult | None
    emonet_error: str | None
    llm_response: LLMResponse | None
    llm_error: str | None
    visible_speaker: Any
    session_state: RucaSessionState
    saved_memory: MemoryItem | None
    event: RucaEvent
    debug_record: dict[str, Any]


class RucaPipeline:
    def __init__(
        self,
        *,
        profiles_path: Path | None = None,
        memory_store: MemoryStore | None = None,
        session_store: SessionStore | None = None,
        session_state: RucaSessionState | None = None,
        llm_config: LLMConfig | None = None,
        use_llm: bool = False,
        use_emonet: bool = False,
        fallback_to_rule_emotion: bool = False,
        fallback_to_rule_composer: bool = True,
    ) -> None:
        self.profiles = load_character_profiles(profiles_path)
        self.memory_store = memory_store or MemoryStore.from_items()
        self.session_store = session_store
        self.session_state = session_state or (session_store.load() if session_store else RucaSessionState())
        self.llm_config = llm_config
        self.use_llm = bool(use_llm)
        self.use_emonet = bool(use_emonet)
        self.fallback_to_rule_emotion = bool(fallback_to_rule_emotion)
        self.fallback_to_rule_composer = bool(fallback_to_rule_composer)

    def run_turn(
        self,
        user_text: str,
        previous_emotion: EmotionState | Mapping[str, Any] | None = None,
        *,
        elapsed_minutes: float = 0.0,
        force_silence: bool = False,
    ) -> TurnResult:
        event = schedule_event(user_text, elapsed_minutes=elapsed_minutes, force_silence=force_silence)
        reference_text = event.user_text or _last_user_text(self.session_state)
        return self._run_event(event=event, reference_text=reference_text, previous_emotion=previous_emotion)

    def run_event(
        self,
        *,
        event_type: str,
        elapsed_minutes: float = 0.0,
        text: str = "",
        previous_emotion: EmotionState | Mapping[str, Any] | None = None,
    ) -> TurnResult:
        if event_type == "user_message":
            return self.run_turn(text, previous_emotion=previous_emotion)
        event = make_event(event_type=event_type, text=text, elapsed_minutes=elapsed_minutes)
        reference_text = text.strip() or _last_user_text(self.session_state)
        return self._run_event(event=event, reference_text=reference_text, previous_emotion=previous_emotion)

    def _run_event(
        self,
        *,
        event: RucaEvent,
        reference_text: str,
        previous_emotion: EmotionState | Mapping[str, Any] | None,
    ) -> TurnResult:
        if previous_emotion is None:
            prev_state = self.session_state.emotion_state
        else:
            prev_state = previous_emotion if isinstance(previous_emotion, EmotionState) else EmotionState.from_mapping(previous_emotion)

        emotion_event_type = "no_reply" if event.event_type in {"no_reply", "silence_tick", "long_silence"} else event.event_type
        rule_next_state, signals = update_emotion_for_event(
            prev_state,
            event_type=emotion_event_type,
            text=reference_text,
            elapsed_minutes=event.elapsed_minutes,
        )
        next_state = rule_next_state
        emonet_trace: EmoNetTraceResult | None = None
        emonet_error: str | None = None
        if self.use_emonet:
            try:
                emonet_trace = infer_emonet_trace(_text_for_emonet(event, reference_text))
                next_state = emonet_trace.emotion_state
            except Exception as exc:
                emonet_error = str(exc)
                if not self.fallback_to_rule_emotion:
                    raise RuntimeError(f"EmoNet trace requested but failed: {emonet_error}") from exc

        memories = self.memory_store.retrieve(reference_text)
        context = analyze_turn_context(
            user_text=reference_text,
            scheduled_event_type=event.event_type,
            rookie=self.profiles["rookie"],
            signals=signals,
            memories=memories,
            event_type=event.event_type,
            elapsed_minutes=event.elapsed_minutes,
        )
        voices = generate_inner_voices(
            profiles=self.profiles,
            user_text=reference_text,
            emotion_state=next_state,
            memories=memories,
            signals=signals,
            context=context,
            event=event,
        )
        spontaneous = decide_spontaneous_reaction(
            emotion_state=next_state,
            signals=signals,
            memories=memories,
            event_type=event.event_type,
            elapsed_minutes=event.elapsed_minutes,
        )
        response_decision = decide_response_action(
            event_type=event.event_type,
            emotion_state=next_state,
            spontaneous=spontaneous,
            elapsed_minutes=event.elapsed_minutes,
        )
        trait_state = update_trait_state(
            self.session_state.trait_state,
            self.profiles,
            signals,
            event_type=emotion_event_type,
        )
        plot_state = update_plot_state(
            self.session_state.plot_state,
            event_type=emotion_event_type,
            user_text=reference_text,
            signals=signals,
            context=context,
            elapsed_minutes=event.elapsed_minutes,
        )
        relationship_graph = update_relationship_graph(
            self.session_state.relationship_graph,
            signals=signals,
            emotion_state=next_state,
            event_type=emotion_event_type,
        )
        visible_speaker = select_visible_speaker(
            profiles=self.profiles,
            user_text=reference_text,
            signals=signals,
            context=context,
            response_decision=response_decision,
            trait_state=trait_state,
        )
        response_prompt = build_response_prompt(
            user_text=reference_text,
            event=event,
            ruca=self.profiles["ruca"],
            emotion_state=next_state,
            turn_context=context,
            memories=memories,
            voices=voices,
            spontaneous=spontaneous,
            emonet_trace=emonet_trace,
            trait_state=trait_state,
            plot_state=plot_state,
            relationship_graph=relationship_graph,
            visible_speaker=visible_speaker,
        )

        assistant_text = ""
        llm_response: LLMResponse | None = None
        llm_error: str | None = None
        expression_mode = "internal_only"
        if response_decision.action == "send_message":
            if self.fallback_to_rule_composer:
                assistant_text = compose_response(
                    user_text=reference_text,
                    emotion_state=next_state,
                    voices=voices,
                    spontaneous=spontaneous,
                    response_decision=response_decision,
                    visible_speaker=visible_speaker,
                )
                expression_mode = "rule"
            if self.use_llm:
                try:
                    llm_response = generate_llm_response(response_prompt, self.llm_config or LLMConfig())
                    assistant_text = llm_response.text
                    expression_mode = "llm"
                except Exception as exc:
                    llm_error = str(exc)
                    raise RuntimeError(f"LLM expression layer failed: {llm_error}") from exc
            elif not self.fallback_to_rule_composer:
                raise RuntimeError("Ruca was scheduled to speak, but neither an LLM nor the development rule composer was enabled.")

        saved_memory = self.memory_store.observe_turn(
            user_text=event.user_text,
            assistant_text=assistant_text,
            emotion_state=next_state,
            signals=signals,
            event_type=emotion_event_type,
        )
        visible_speaker_id = visible_speaker.character_id if visible_speaker else None
        next_session = self.session_state.next_turn(
            user_text=event.user_text,
            assistant_text=assistant_text,
            emotion_state=next_state,
            trait_state=trait_state,
            plot_state=plot_state,
            relationship_graph=relationship_graph,
            debug_summary={
                "event_type": event.event_type,
                "visible_speaker": visible_speaker_id,
                "spontaneous_reaction": spontaneous.to_record(),
                "response_decision": response_decision.to_record(),
            },
        )
        self.session_state = next_session
        if self.session_store:
            self.session_store.save(next_session)

        debug_record = {
            "turn_index": next_session.turn_index,
            "event": {
                **event.to_record(),
                "source_text": event.user_text,
                "reference_text": reference_text,
            },
            "session": next_session.to_record(),
            "input_signals": signals.to_record(),
            "previous_emotion_state": prev_state.to_record(),
            "rule_emotion_state": rule_next_state.to_record(),
            "emotion_state": next_state.to_record(),
            "trait_state": trait_state.to_record(),
            "plot_state": plot_state.to_record(),
            "relationship_graph": relationship_graph.to_record(),
            "emotion_source": "emonet" if emonet_trace else "rule_dev_fallback",
            "emonet_trace": emonet_trace.to_record() if emonet_trace else None,
            "emonet_error": emonet_error,
            "turn_context": context.to_record(),
            "retrieved_memories": [item.to_record() for item in memories],
            "inner_voices": [voice.to_record() for voice in voices],
            "spontaneous_reaction": spontaneous.to_record(),
            "response_decision": response_decision.to_record(),
            "response_prompt": response_prompt,
            "expression_mode": expression_mode,
            "llm_config": (self.llm_config or LLMConfig()).to_record() if self.use_llm else None,
            "llm_response": llm_response.to_record() if llm_response else None,
            "llm_error": llm_error,
            "saved_memory": saved_memory.to_record() if saved_memory else None,
            "visible_speaker": visible_speaker.to_record() if visible_speaker else None,
            "visible_character": (visible_speaker or self.profiles["ruca"]).to_record(),
        }
        return TurnResult(
            assistant_text=assistant_text,
            previous_emotion_state=prev_state,
            emotion_state=next_state,
            turn_context=context,
            retrieved_memories=memories,
            inner_voices=voices,
            spontaneous_reaction=spontaneous,
            response_decision=response_decision,
            response_prompt=response_prompt,
            emonet_trace=emonet_trace,
            emonet_error=emonet_error,
            llm_response=llm_response,
            llm_error=llm_error,
            visible_speaker=visible_speaker,
            session_state=next_session,
            saved_memory=saved_memory,
            event=event,
            debug_record=debug_record,
        )


def run_turn(
    user_text: str,
    *,
    previous_emotion: EmotionState | Mapping[str, Any] | None = None,
    memory_path: Path | None = None,
    session_path: Path | None = None,
    profiles_path: Path | None = None,
    use_llm: bool = False,
    llm_config: LLMConfig | None = None,
    use_emonet: bool = False,
    fallback_to_rule_emotion: bool = False,
    fallback_to_rule_composer: bool = True,
    elapsed_minutes: float = 0.0,
    force_silence: bool = False,
) -> TurnResult:
    pipeline = RucaPipeline(
        profiles_path=profiles_path,
        memory_store=MemoryStore(memory_path) if memory_path else None,
        session_store=SessionStore(session_path) if session_path else None,
        llm_config=llm_config,
        use_llm=use_llm,
        use_emonet=use_emonet,
        fallback_to_rule_emotion=fallback_to_rule_emotion,
        fallback_to_rule_composer=fallback_to_rule_composer,
    )
    return pipeline.run_turn(
        user_text,
        previous_emotion=previous_emotion,
        elapsed_minutes=elapsed_minutes,
        force_silence=force_silence,
    )


def _last_user_text(session_state: RucaSessionState) -> str:
    for item in reversed(session_state.recent_history):
        text = str(item.get("user_text", "") or "").strip()
        if text:
            return text
    return ""


def _text_for_emonet(event: RucaEvent, reference_text: str) -> str:
    if event.event_type == "user_message":
        return reference_text
    return f"[{event.event_type} after {event.elapsed_minutes:.1f} minutes without a new reply] {reference_text}".strip()
