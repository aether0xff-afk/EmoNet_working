
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
from ruca_engine.event_scheduler import schedule_event
from ruca_engine.llm_client import LLMConfig, LLMResponse
from ruca_engine.memory import MemoryStore
from ruca_engine.models import EmotionState, MemoryItem


class RucaEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.llm_config = LLMConfig(model_name="test-model", api_key="test-key")
        self.llm_patcher = patch(
            "ruca_engine.pipeline.generate_llm_response",
            return_value=LLMResponse(text="Ruca response", raw_text="Ruca response", usage={"input_tokens": 1, "output_tokens": 1}),
        )
        self.llm_patcher.start()

    def tearDown(self) -> None:
        self.llm_patcher.stop()

    def test_default_profiles_load_required_characters(self) -> None:
        profiles = load_character_profiles()
        self.assertEqual(set(profiles), {"ruca", "rookie", "ricky", "rocky"})
        self.assertEqual(profiles["ruca"].visibility, "external")
        self.assertEqual(profiles["ricky"].visibility, "internal")
        self.assertTrue(profiles["rocky"].traits["initiative"] > 0.0)

    def test_emotion_state_reacts_to_alarm_input(self) -> None:
        previous = EmotionState()
        next_state, signals = update_emotion_state(previous, "I am scared and anxious right now")
        self.assertGreater(signals.alarm, 0.5)
        self.assertGreater(next_state.protective_tension, previous.protective_tension)
        self.assertLess(next_state.stability, previous.stability)

    def test_pipeline_generates_inner_voices_and_ruca_response(self) -> None:
        pipeline = RucaPipeline(memory_store=MemoryStore.from_items(), use_llm=True, llm_config=self.llm_config)
        result = pipeline.run_turn("Please build this for real")
        self.assertIn("Ruca", result.debug_record["visible_character"]["name"])
        self.assertEqual(len(result.inner_voices), 3)
        self.assertEqual({voice.source_character for voice in result.inner_voices}, {"Ruca", "Ricky", "Rocky"})
        self.assertTrue(result.assistant_text)
        self.assertIn("spontaneous_reaction", result.debug_record)
        self.assertEqual(result.turn_context.event_type, "implementation_request")
        self.assertIn("Rookie", result.turn_context.rookie_question)
        self.assertTrue(any("Rookie" in voice.content for voice in result.inner_voices))

    def test_pipeline_carries_emotion_state_between_turns_by_default(self) -> None:
        pipeline = RucaPipeline(memory_store=MemoryStore.from_items(), use_llm=True, llm_config=self.llm_config)
        first = pipeline.run_turn("I am scared and anxious right now")
        second = pipeline.run_turn("?댁젣 議곌툑 ?뺣━?댁쨾")
        self.assertEqual(second.previous_emotion_state, first.emotion_state)
        self.assertGreater(second.emotion_state.protective_tension, EmotionState().protective_tension)
        self.assertEqual(second.debug_record["turn_index"], 2)
        self.assertEqual(second.session_state.turn_index, 2)

    def test_silence_tick_updates_session_without_speaking(self) -> None:
        pipeline = RucaPipeline(memory_store=MemoryStore.from_items())
        result = pipeline.run_turn("", elapsed_minutes=10, force_silence=True)

        self.assertEqual(result.event.event_type, "silence_tick")
        self.assertFalse(result.event.should_speak)
        self.assertEqual(result.assistant_text, "")
        self.assertEqual(result.spontaneous_reaction.reaction_type, "internal_only")
        self.assertEqual(result.session_state.recent_history[-1]["spoke"], False)

    def test_long_silence_can_trigger_quiet_check_in(self) -> None:
        previous = EmotionState(protective_tension=0.5)
        result = run_turn("", previous_emotion=previous, use_llm=True, llm_config=self.llm_config, elapsed_minutes=60)

        self.assertEqual(result.event.event_type, "long_silence")
        self.assertTrue(result.event.should_speak)
        self.assertEqual(result.spontaneous_reaction.reaction_type, "quiet_check_in")
        self.assertTrue(result.assistant_text)

    def test_event_scheduler_keeps_user_message_as_speak_event(self) -> None:
        event = schedule_event("hello", elapsed_minutes=120)

        self.assertEqual(event.event_type, "user_message")
        self.assertEqual(event.user_text, "hello")
        self.assertTrue(event.should_speak)

    def test_session_store_persists_turn_state_across_pipeline_instances(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            session_path = Path(tmpdir) / "session.json"
            first = run_turn("I am scared and anxious right now", session_path=session_path, use_llm=True, llm_config=self.llm_config)
            second = run_turn("?댁젣 議곌툑 ?뺣━?댁쨾", session_path=session_path, use_llm=True, llm_config=self.llm_config)
            stored = SessionStore(session_path).load()

            self.assertEqual(first.session_state.turn_index, 1)
            self.assertEqual(second.previous_emotion_state, first.emotion_state)
            self.assertEqual(second.session_state.turn_index, 2)
            self.assertEqual(stored.turn_index, 2)
            self.assertEqual(len(stored.recent_history), 2)

    def test_response_prompt_is_built_for_later_llm_composer(self) -> None:
        result = run_turn("Please help me implement this for real", use_llm=True, llm_config=self.llm_config)
        self.assertIn("[ROLE]", result.response_prompt)
        self.assertIn("[INNER_VOICES]", result.response_prompt)
        self.assertIn("Rookie", result.response_prompt)
        self.assertIn("Return only Ruca", result.response_prompt)

    def test_spontaneous_reaction_records_reason_for_alarm(self) -> None:
        result = run_turn("I am anxious and scared", use_llm=True, llm_config=self.llm_config)
        self.assertTrue(result.spontaneous_reaction.should_react)
        self.assertEqual(result.spontaneous_reaction.reaction_type, "check_in")
        self.assertTrue(result.spontaneous_reaction.reason)

    def test_llm_composer_can_replace_rule_response(self) -> None:
        with patch(
            "ruca_engine.pipeline.generate_llm_response",
            return_value=LLMResponse(
                text="?? Ruca媛 吏湲??먮쫫??諛쏆븘????臾몄옣?쇰줈 ?댁뼱媛덇쾶.",
                raw_text="?? Ruca媛 吏湲??먮쫫??諛쏆븘????臾몄옣?쇰줈 ?댁뼱媛덇쾶.",
                usage={"input_tokens": 10, "output_tokens": 8},
            ),
        ) as llm_mock:
            result = run_turn(
                "Ruca tone please",
                use_llm=True,
                llm_config=LLMConfig(model_name="test-model", api_key="test-key"),
            )

        self.assertEqual(result.assistant_text, "?? Ruca媛 吏湲??먮쫫??諛쏆븘????臾몄옣?쇰줈 ?댁뼱媛덇쾶.")
        self.assertEqual(result.debug_record["expression_mode"], "llm")
        self.assertEqual(result.llm_response.usage["input_tokens"], 10)
        self.assertIn("[INNER_VOICES]", llm_mock.call_args.args[0])

    def test_llm_failure_raises_instead_of_falling_back(self) -> None:
        with patch("ruca_engine.pipeline.generate_llm_response", side_effect=RuntimeError("network down")):
            with self.assertRaisesRegex(RuntimeError, "LLM expression layer failed"):
                run_turn("???ㅺ퀎??留욎떠??援ы쁽 怨꾩냽?댁쨾", use_llm=True, llm_config=self.llm_config)

    def test_memory_store_persists_json_when_path_is_given(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = Path(tmpdir) / "memory.json"
            result = run_turn("Please remember this", memory_path=memory_path, use_llm=True, llm_config=self.llm_config)
            self.assertIsNotNone(result.saved_memory)
            self.assertTrue(memory_path.exists())
            store = MemoryStore(memory_path)
            self.assertGreaterEqual(len(store.all_items()), 1)

    def test_retrieved_memory_can_change_debug_context(self) -> None:
        stored = MemoryItem(
            memory_id="mem-test",
            memory_type="relationship",
            summary="?ъ슜?먭? 援ы쁽???앷퉴吏 留↔린寃좊떎怨?留먰븿",
            source_event="?앷퉴吏 援ы쁽?댁쨾",
            emotion_snapshot=EmotionState(protective_tension=0.7).to_record(),
            importance=0.9,
        )
        pipeline = RucaPipeline(memory_store=MemoryStore.from_items([stored]), use_llm=True, llm_config=self.llm_config)
        result = pipeline.run_turn("援ы쁽 怨꾩냽?댁쨾")
        self.assertGreaterEqual(len(result.retrieved_memories), 1)
        self.assertIn("援ы쁽", result.retrieved_memories[0].summary)
        self.assertGreater(result.turn_context.memory_pressure, 0.0)

    def test_memory_retrieve_updates_last_accessed_on_store(self) -> None:
        stored = MemoryItem(
            memory_id="mem-0007",
            memory_type="relationship",
            summary="?ъ슜?먭? Ruca?먭쾶 援ы쁽 吏?띿쓣 留↔?",
            source_event="怨꾩냽 援ы쁽?댁쨾",
            emotion_snapshot=EmotionState(protective_tension=0.6).to_record(),
            importance=0.9,
            last_accessed_at="2000-01-01T00:00:00+00:00",
        )
        store = MemoryStore.from_items([stored])
        retrieved = store.retrieve("援ы쁽 怨꾩냽")
        self.assertEqual(len(retrieved), 1)
        self.assertNotEqual(store.all_items()[0].last_accessed_at, "2000-01-01T00:00:00+00:00")

    def test_memory_ids_do_not_reuse_after_short_term_trim(self) -> None:
        store = MemoryStore.from_items(max_short_term=1)
        for index in range(3):
            store.observe_turn(
                user_text=f"?쇰컲 援ы쁽 ?붿껌 {index}",
                assistant_text="??",
                emotion_state=EmotionState(),
                signals=update_emotion_state(EmotionState(), "援ы쁽?댁쨾")[1],
            )
        ids = [item.memory_id for item in store.all_items()]
        self.assertEqual(ids, ["mem-0003"])


if __name__ == "__main__":
    unittest.main()
