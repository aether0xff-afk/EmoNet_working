
import sys
import subprocess
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

    def test_no_reply_event_ticks_state_without_user_text(self) -> None:
        pipeline = RucaPipeline(memory_store=MemoryStore.from_items())
        first = pipeline.run_turn("고마워, 조금 있다가 다시 올게")
        second = pipeline.run_event(event_type="no_reply", elapsed_minutes=45)

        self.assertEqual(second.turn_context.event_type, "no_reply")
        self.assertEqual(second.debug_record["event"]["event_type"], "no_reply")
        self.assertGreater(second.emotion_state.arousal, first.emotion_state.arousal)
        self.assertGreaterEqual(second.emotion_state.protective_tension, first.emotion_state.protective_tension)
        self.assertEqual(second.response_decision.action, "update_internal_only")
        self.assertEqual(second.assistant_text, "")

    def test_no_reply_gate_can_send_spontaneous_message_after_long_silence(self) -> None:
        pipeline = RucaPipeline(memory_store=MemoryStore.from_items())
        pipeline.run_turn("오늘은 진짜 고마웠어")
        result = pipeline.run_event(event_type="no_reply", elapsed_minutes=180)

        self.assertEqual(result.response_decision.action, "send_message")
        self.assertTrue(result.spontaneous_reaction.should_react)
        self.assertTrue(result.assistant_text)
        self.assertIn("no_reply", result.debug_record["event"]["event_type"])

    def test_internal_only_no_reply_does_not_call_llm_composer(self) -> None:
        pipeline = RucaPipeline(memory_store=MemoryStore.from_items(), use_llm=True)
        pipeline.run_turn("고마워, 조금 있다가 다시 올게")
        with patch("ruca_engine.pipeline.generate_llm_response") as llm_mock:
            result = pipeline.run_event(event_type="no_reply", elapsed_minutes=45)

        self.assertEqual(result.response_decision.action, "update_internal_only")
        self.assertEqual(result.assistant_text, "")
        llm_mock.assert_not_called()

    def test_saved_memory_keeps_ruca_interpretation_and_deltas(self) -> None:
        result = run_turn("고마워, 이건 진짜 기억해 줬으면 해")

        self.assertIsNotNone(result.saved_memory)
        self.assertIn("ruca_interpretation", result.saved_memory.to_record())
        self.assertIn("emotion_delta", result.saved_memory.to_record())
        self.assertIn("relationship_effect", result.saved_memory.to_record())

    def test_trait_state_updates_and_persists(self) -> None:
        from ruca_engine.trait_state import CharacterTraitState

        profiles = load_character_profiles()
        initial = CharacterTraitState.from_profiles(profiles)
        pipeline = RucaPipeline(memory_store=MemoryStore.from_items())
        warm = pipeline.run_turn("고마워, 오늘 네가 있어서 조금 안심됐어")
        alarm = pipeline.run_turn("나 지금 너무 불안하고 무서워")

        self.assertGreater(warm.session_state.trait_state.characters["ruca"]["warmth"], initial.characters["ruca"]["warmth"])
        self.assertGreater(alarm.session_state.trait_state.characters["rocky"]["protectiveness"], warm.session_state.trait_state.characters["rocky"]["protectiveness"])
        self.assertIn("trait_state", alarm.debug_record)

        with tempfile.TemporaryDirectory() as tmpdir:
            session_path = Path(tmpdir) / "session.json"
            first = run_turn("고마워, 오늘 네가 있어서 조금 안심됐어", session_path=session_path)
            stored = SessionStore(session_path).load()

        self.assertEqual(stored.trait_state, first.session_state.trait_state)

    def test_rookie_plot_state_tracks_threads_and_pressure(self) -> None:
        pipeline = RucaPipeline(memory_store=MemoryStore.from_items())
        first = pipeline.run_turn("v6 설계 전체를 구현해줘")
        second = pipeline.run_event(event_type="no_reply", elapsed_minutes=90)

        self.assertIn("plot_state", first.debug_record)
        self.assertGreaterEqual(len(first.session_state.plot_state.unresolved_threads), 1)
        self.assertIn("implementation", first.session_state.plot_state.unresolved_threads[0]["thread_type"])
        self.assertGreater(second.session_state.plot_state.scene_pressure, first.session_state.plot_state.scene_pressure)
        self.assertEqual(second.response_decision.action, "update_internal_only")

    def test_no_reply_records_event_without_replaying_user_text(self) -> None:
        pipeline = RucaPipeline(memory_store=MemoryStore.from_items())
        pipeline.run_turn("고마워, 조금 있다가 다시 올게")
        result = pipeline.run_event(event_type="no_reply", elapsed_minutes=180)

        self.assertEqual(result.debug_record["event"]["source_text"], "")
        self.assertIn("reference_text", result.debug_record["event"])
        self.assertEqual(result.session_state.recent_history[-1]["user_text"], "")
        self.assertEqual(result.saved_memory.source_event, "")

    def test_silence_plot_thread_does_not_evict_existing_work_thread(self) -> None:
        pipeline = RucaPipeline(memory_store=MemoryStore.from_items())
        pipeline.run_turn("v6 설계 전체를 구현해줘")
        for _ in range(9):
            pipeline.run_event(event_type="no_reply", elapsed_minutes=20)

        thread_types = [thread["thread_type"] for thread in pipeline.session_state.plot_state.unresolved_threads]
        self.assertIn("implementation", thread_types)
        self.assertEqual(thread_types.count("silence_followup"), 1)

    def test_relationship_graph_accumulates_edges(self) -> None:
        pipeline = RucaPipeline(memory_store=MemoryStore.from_items())
        warm = pipeline.run_turn("고마워, 오늘 네가 있어서 안심됐어")
        alarm = pipeline.run_turn("나 지금 너무 불안하고 무서워")
        silence = pipeline.run_event(event_type="no_reply", elapsed_minutes=120)

        self.assertIn("relationship_graph", silence.debug_record)
        graph = silence.session_state.relationship_graph
        user_ruca = graph.edge("user", "ruca")
        ruca_rocky = graph.edge("ruca", "rocky")
        self.assertGreater(user_ruca.metrics["trust"], 0.0)
        self.assertGreater(user_ruca.metrics["need_for_reassurance"], warm.session_state.relationship_graph.edge("user", "ruca").metrics["need_for_reassurance"])
        self.assertGreater(ruca_rocky.metrics["protective_tension"], alarm.session_state.relationship_graph.edge("ruca", "rocky").metrics["protective_tension"])

    def test_character_runtime_selects_visible_speaker(self) -> None:
        pipeline = RucaPipeline(memory_store=MemoryStore.from_items())
        default = pipeline.run_turn("오늘은 그냥 조금 이야기하고 싶어")
        analysis = pipeline.run_turn("이 상황을 분석하고 구조적으로 정리해줘")
        action = pipeline.run_turn("멈추지 말고 바로 실행해줘")
        quiet = pipeline.run_event(event_type="no_reply", elapsed_minutes=20)

        self.assertEqual(default.visible_speaker.character_id, "ruca")
        self.assertEqual(analysis.visible_speaker.character_id, "ricky")
        self.assertEqual(action.visible_speaker.character_id, "rocky")
        self.assertIsNone(quiet.visible_speaker)
        self.assertEqual(quiet.assistant_text, "")
        self.assertIn("visible_speaker", analysis.debug_record)

    def test_cli_can_run_no_reply_event(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "ruca_engine.cli",
                "--event-type",
                "no_reply",
                "--elapsed-minutes",
                "45",
                "--debug",
            ],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn('"event_type": "no_reply"', completed.stdout)
        self.assertIn('"response_decision"', completed.stdout)

if __name__ == "__main__":
    unittest.main()
