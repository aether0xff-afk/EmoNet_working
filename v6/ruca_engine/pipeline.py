from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .composer import compose_response
from .context import TurnContext, analyze_turn_context
from .emotion import update_emotion_state
from .inner_voice import generate_inner_voices
from .llm_client import LLMConfig, LLMResponse, generate_llm_response
from .memory import MemoryStore
from .models import EmotionState, InnerVoiceCandidate, MemoryItem, SpontaneousReactionDecision
from .prompt_builder import build_response_prompt
from .profiles import load_character_profiles
from .session import RucaSessionState, SessionStore
from .spontaneous import decide_spontaneous_reaction


@dataclass(frozen=True)
class TurnResult:
    assistant_text: str
    previous_emotion_state: EmotionState
    emotion_state: EmotionState
    turn_context: TurnContext
    retrieved_memories: tuple[MemoryItem, ...]
    inner_voices: tuple[InnerVoiceCandidate, ...]
    spontaneous_reaction: SpontaneousReactionDecision
    response_prompt: str
    llm_response: LLMResponse | None
    llm_error: str | None
    session_state: RucaSessionState
    saved_memory: MemoryItem | None
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
        fallback_to_rule_composer: bool = True,
    ) -> None:
        self.profiles = load_character_profiles(profiles_path)
        self.memory_store = memory_store or MemoryStore.from_items()
        self.session_store = session_store
        self.session_state = session_state or (session_store.load() if session_store else RucaSessionState())
        self.llm_config = llm_config
        self.use_llm = bool(use_llm)
        self.fallback_to_rule_composer = bool(fallback_to_rule_composer)

    def run_turn(self, user_text: str, previous_emotion: EmotionState | Mapping[str, Any] | None = None) -> TurnResult:
        if previous_emotion is None:
            prev_state = self.session_state.emotion_state
        else:
            prev_state = previous_emotion if isinstance(previous_emotion, EmotionState) else EmotionState.from_mapping(previous_emotion)
        next_state, signals = update_emotion_state(prev_state, user_text)
        memories = self.memory_store.retrieve(user_text)
        context = analyze_turn_context(
            user_text=user_text,
            rookie=self.profiles["rookie"],
            signals=signals,
            memories=memories,
        )
        voices = generate_inner_voices(
            profiles=self.profiles,
            user_text=user_text,
            emotion_state=next_state,
            memories=memories,
            signals=signals,
            context=context,
        )
        spontaneous = decide_spontaneous_reaction(emotion_state=next_state, signals=signals, memories=memories)
        response_prompt = build_response_prompt(
            user_text=user_text,
            ruca=self.profiles["ruca"],
            emotion_state=next_state,
            turn_context=context,
            memories=memories,
            voices=voices,
            spontaneous=spontaneous,
        )
        rule_assistant_text = compose_response(
            user_text=user_text,
            emotion_state=next_state,
            voices=voices,
            spontaneous=spontaneous,
        )
        assistant_text = rule_assistant_text
        llm_response: LLMResponse | None = None
        llm_error: str | None = None
        composer_mode = "rule"
        if self.use_llm:
            try:
                llm_response = generate_llm_response(response_prompt, self.llm_config or LLMConfig())
                assistant_text = llm_response.text
                composer_mode = "llm"
            except Exception as exc:
                llm_error = str(exc)
                composer_mode = "llm_fallback"
                if not self.fallback_to_rule_composer:
                    raise
        saved_memory = self.memory_store.observe_turn(
            user_text=user_text,
            assistant_text=assistant_text,
            emotion_state=next_state,
            signals=signals,
        )
        next_session = self.session_state.next_turn(
            user_text=user_text,
            assistant_text=assistant_text,
            emotion_state=next_state,
            debug_summary={
                "event_type": context.event_type,
                "spontaneous_reaction": spontaneous.to_record(),
            },
        )
        self.session_state = next_session
        if self.session_store:
            self.session_store.save(next_session)
        debug_record = {
            "turn_index": next_session.turn_index,
            "session": next_session.to_record(),
            "input_signals": signals.to_record(),
            "previous_emotion_state": prev_state.to_record(),
            "emotion_state": next_state.to_record(),
            "turn_context": context.to_record(),
            "retrieved_memories": [item.to_record() for item in memories],
            "inner_voices": [voice.to_record() for voice in voices],
            "spontaneous_reaction": spontaneous.to_record(),
            "response_prompt": response_prompt,
            "composer_mode": composer_mode,
            "rule_assistant_text": rule_assistant_text,
            "llm_config": (self.llm_config or LLMConfig()).to_record() if self.use_llm else None,
            "llm_response": llm_response.to_record() if llm_response else None,
            "llm_error": llm_error,
            "saved_memory": saved_memory.to_record() if saved_memory else None,
            "visible_character": self.profiles["ruca"].to_record(),
        }
        return TurnResult(
            assistant_text=assistant_text,
            previous_emotion_state=prev_state,
            emotion_state=next_state,
            turn_context=context,
            retrieved_memories=memories,
            inner_voices=voices,
            spontaneous_reaction=spontaneous,
            response_prompt=response_prompt,
            llm_response=llm_response,
            llm_error=llm_error,
            session_state=next_session,
            saved_memory=saved_memory,
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
    fallback_to_rule_composer: bool = True,
) -> TurnResult:
    pipeline = RucaPipeline(
        profiles_path=profiles_path,
        memory_store=MemoryStore(memory_path) if memory_path else None,
        session_store=SessionStore(session_path) if session_path else None,
        llm_config=llm_config,
        use_llm=use_llm,
        fallback_to_rule_composer=fallback_to_rule_composer,
    )
    return pipeline.run_turn(user_text, previous_emotion=previous_emotion)
