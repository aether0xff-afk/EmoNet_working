import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ruca_engine import RucaPipeline, SessionStore, load_character_profiles, run_turn
from ruca_engine.emotion import update_emotion_state
from ruca_engine.llm_client import LLMConfig, LLMResponse
from ruca_engine.memory import MemoryStore
from ruca_engine.models import EmotionState, MemoryItem


class RucaEngineTests(unittest.TestCase):
    def test_default_profiles_load_required_characters(self) -> None:
        profiles = load_character_profiles()
        self.assertEqual(set(profiles), {"ruca", "rookie", "ricky", "rocky"})
        self.assertEqual(profiles["ruca"].visibility, "external")
        self.assertEqual(profiles["ricky"].visibility, "internal")
        self.assertTrue(profiles["rocky"].traits["initiative"] > 0.0)

    def test_emotion_state_reacts_to_alarm_input(self) -> None:
        previous = EmotionState()
        next_state, signals = update_emotion_state(previous, "나 지금 너무 불안하고 무서워")
        self.assertGreater(signals.alarm, 0.5)
        self.assertGreater(next_state.protective_tension, previous.protective_tension)
        self.assertLess(next_state.stability, previous.stability)

    def test_pipeline_generates_inner_voices_and_ruca_response(self) -> None:
        pipeline = RucaPipeline(memory_store=MemoryStore.from_items())
        result = pipeline.run_turn("실제로 구현하려면 어떻게 해야 할지 알려줘")
        self.assertIn("Ruca", result.debug_record["visible_character"]["name"])
        self.assertEqual(len(result.inner_voices), 3)
        self.assertEqual({voice.source_character for voice in result.inner_voices}, {"Ruca", "Ricky", "Rocky"})
        self.assertTrue(result.assistant_text)
        self.assertIn("spontaneous_reaction", result.debug_record)
        self.assertEqual(result.turn_context.event_type, "implementation_request")
        self.assertIn("Rookie", result.turn_context.rookie_question)
        self.assertTrue(any("Rookie 관점" in voice.content for voice in result.inner_voices))

    def test_pipeline_carries_emotion_state_between_turns_by_default(self) -> None:
        pipeline = RucaPipeline(memory_store=MemoryStore.from_items())
        first = pipeline.run_turn("나 지금 너무 불안하고 무서워")
        second = pipeline.run_turn("이제 조금 정리해줘")
        self.assertEqual(second.previous_emotion_state, first.emotion_state)
        self.assertGreater(second.emotion_state.protective_tension, EmotionState().protective_tension)
        self.assertEqual(second.debug_record["turn_index"], 2)
        self.assertEqual(second.session_state.turn_index, 2)

    def test_session_store_persists_turn_state_across_pipeline_instances(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            session_path = Path(tmpdir) / "session.json"
            first = run_turn("나 지금 너무 불안하고 무서워", session_path=session_path)
            second = run_turn("이제 조금 정리해줘", session_path=session_path)
            stored = SessionStore(session_path).load()

            self.assertEqual(first.session_state.turn_index, 1)
            self.assertEqual(second.previous_emotion_state, first.emotion_state)
            self.assertEqual(second.session_state.turn_index, 2)
            self.assertEqual(stored.turn_index, 2)
            self.assertEqual(len(stored.recent_history), 2)

    def test_response_prompt_is_built_for_later_llm_composer(self) -> None:
        result = run_turn("실제로 구현하려면 어떻게 해야 할지 알려줘")
        self.assertIn("[ROLE]", result.response_prompt)
        self.assertIn("[INNER_VOICES]", result.response_prompt)
        self.assertIn("Rookie", result.response_prompt)
        self.assertIn("Return only Ruca", result.response_prompt)

    def test_spontaneous_reaction_records_reason_for_alarm(self) -> None:
        result = run_turn("나 지금 너무 불안해서 무서워!")
        self.assertTrue(result.spontaneous_reaction.should_react)
        self.assertEqual(result.spontaneous_reaction.reaction_type, "check_in")
        self.assertTrue(result.spontaneous_reaction.reason)

    def test_llm_composer_can_replace_rule_response(self) -> None:
        with patch(
            "ruca_engine.pipeline.generate_llm_response",
            return_value=LLMResponse(
                text="응, Ruca가 지금 흐름을 받아서 한 문장으로 이어갈게.",
                raw_text="응, Ruca가 지금 흐름을 받아서 한 문장으로 이어갈게.",
                usage={"input_tokens": 10, "output_tokens": 8},
            ),
        ) as llm_mock:
            result = run_turn(
                "Ruca 말투로 답해줘",
                use_llm=True,
                llm_config=LLMConfig(model_name="test-model", api_key="test-key"),
            )

        self.assertEqual(result.assistant_text, "응, Ruca가 지금 흐름을 받아서 한 문장으로 이어갈게.")
        self.assertEqual(result.debug_record["composer_mode"], "llm")
        self.assertEqual(result.llm_response.usage["input_tokens"], 10)
        self.assertIn("[INNER_VOICES]", llm_mock.call_args.args[0])

    def test_llm_composer_falls_back_to_rule_response(self) -> None:
        with patch("ruca_engine.pipeline.generate_llm_response", side_effect=RuntimeError("network down")):
            result = run_turn(
                "이 설계에 맞춰서 구현 계속해줘",
                use_llm=True,
                llm_config=LLMConfig(model_name="test-model", api_key="test-key"),
            )

        self.assertTrue(result.assistant_text)
        self.assertEqual(result.debug_record["composer_mode"], "llm_fallback")
        self.assertIn("network down", result.llm_error)

    def test_memory_store_persists_json_when_path_is_given(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = Path(tmpdir) / "memory.json"
            result = run_turn("고마워, 이건 진짜 기억해 줬으면 해", memory_path=memory_path)
            self.assertIsNotNone(result.saved_memory)
            self.assertTrue(memory_path.exists())
            store = MemoryStore(memory_path)
            self.assertGreaterEqual(len(store.all_items()), 1)

    def test_retrieved_memory_can_change_debug_context(self) -> None:
        stored = MemoryItem(
            memory_id="mem-test",
            memory_type="relationship",
            summary="사용자가 구현을 끝까지 맡기겠다고 말함",
            source_event="끝까지 구현해줘",
            emotion_snapshot=EmotionState(protective_tension=0.7).to_record(),
            importance=0.9,
        )
        pipeline = RucaPipeline(memory_store=MemoryStore.from_items([stored]))
        result = pipeline.run_turn("구현 계속해줘")
        self.assertGreaterEqual(len(result.retrieved_memories), 1)
        self.assertIn("구현", result.retrieved_memories[0].summary)
        self.assertIn("이전 관계 기억 반영", result.turn_context.unresolved_need)

    def test_memory_retrieve_updates_last_accessed_on_store(self) -> None:
        stored = MemoryItem(
            memory_id="mem-0007",
            memory_type="relationship",
            summary="사용자가 Ruca에게 구현 지속을 맡김",
            source_event="계속 구현해줘",
            emotion_snapshot=EmotionState(protective_tension=0.6).to_record(),
            importance=0.9,
            last_accessed_at="2000-01-01T00:00:00+00:00",
        )
        store = MemoryStore.from_items([stored])
        retrieved = store.retrieve("구현 계속")
        self.assertEqual(len(retrieved), 1)
        self.assertNotEqual(store.all_items()[0].last_accessed_at, "2000-01-01T00:00:00+00:00")

    def test_memory_ids_do_not_reuse_after_short_term_trim(self) -> None:
        store = MemoryStore.from_items(max_short_term=1)
        for index in range(3):
            store.observe_turn(
                user_text=f"일반 구현 요청 {index}",
                assistant_text="응.",
                emotion_state=EmotionState(),
                signals=update_emotion_state(EmotionState(), "구현해줘")[1],
            )
        ids = [item.memory_id for item in store.all_items()]
        self.assertEqual(ids, ["mem-0003"])


if __name__ == "__main__":
    unittest.main()
