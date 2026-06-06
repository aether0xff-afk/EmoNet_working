import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from streamlit_gui import build_llm_config, compact_trace, make_pipeline, run_background_tick, run_gui_event
from ruca_engine import RucaPipeline
from runtime import ThoughtCouncil


class StreamlitGuiTests(unittest.TestCase):
    def test_build_llm_config_defaults_gemini_to_env(self) -> None:
        config = build_llm_config(provider="gemini", api_key="", model_name="", base_url="")

        self.assertEqual(config.provider, "gemini")
        self.assertEqual(config.api_key_env, "GEMINI_API_KEY")
        self.assertEqual(config.model_name, "gemini-2.5-flash")
        self.assertEqual(config.base_url, "https://generativelanguage.googleapis.com/v1beta")

    def test_build_llm_config_maps_lm_studio_to_openai_compatible(self) -> None:
        config = build_llm_config(provider="lm_studio", api_key="", model_name="", base_url="")

        self.assertEqual(config.provider, "openai_compatible")
        self.assertEqual(config.api_key_env, "")
        self.assertEqual(config.model_name, "gemma-4-26b-a4b-it-qat")
        self.assertEqual(config.base_url, "http://100.115.40.97:1234/v1")

    def test_run_gui_event_keeps_pipeline_alive_across_typing_tick(self) -> None:
        pipeline = RucaPipeline(use_emonet=True, use_llm=False)
        messages: list[dict] = []

        first = run_gui_event(pipeline, messages, event_type="user_message", text="첫 메시지")
        typing = run_gui_event(pipeline, messages, event_type="typing", elapsed_minutes=0.1)

        self.assertEqual(first["trace"]["tick"], 0)
        self.assertEqual(typing["trace"]["tick"], 1)
        self.assertEqual(typing["trace"]["event_kind"], "typing")
        self.assertEqual(messages[0]["role"], "user")
        self.assertEqual(messages[1]["role"], "assistant")
        self.assertEqual(len(messages), 2)

    def test_gui_pipeline_defaults_to_neural_response_timing(self) -> None:
        config = build_llm_config(provider="gemini", api_key="", model_name="", base_url="")
        pipeline = make_pipeline(use_llm=False, llm_config=config)
        messages: list[dict] = []

        payload = run_gui_event(pipeline, messages, event_type="user_message", text="짧게 확인")

        self.assertEqual(payload["result"].response_decision.action, "update_internal_only")
        self.assertEqual(payload["pending_text"], "짧게 확인")
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[-1]["role"], "user")

    def test_compact_trace_uses_cluster_and_memory_fields(self) -> None:
        pipeline = RucaPipeline(use_emonet=True, use_llm=False)
        result = pipeline.run_turn("trace 확인")

        trace = compact_trace(result)

        self.assertEqual(trace["stim_dim"], 8)
        self.assertIn("dominant_cluster", trace)
        self.assertIn("stored_memory_count", trace)

    def test_background_tick_updates_trace_without_chat_spam(self) -> None:
        pipeline = make_pipeline(use_llm=False, llm_config=build_llm_config(provider="gemini", api_key="", model_name="", base_url=""))
        messages: list[dict] = []

        first = run_background_tick(pipeline, messages)
        second = run_background_tick(pipeline, messages)

        self.assertEqual(first["trace"]["event_kind"], "idle")
        self.assertEqual(second["trace"]["tick"], first["trace"]["tick"] + 1)
        self.assertEqual(messages, [])

    def test_background_tick_releases_pending_speech_after_delay(self) -> None:
        pipeline = make_pipeline(use_llm=False, llm_config=build_llm_config(provider="gemini", api_key="", model_name="", base_url=""))
        messages: list[dict] = [{"role": "user", "content": "조금 늦게 답해도 돼"}]

        payload = run_background_tick(pipeline, messages, pending_text="조금 늦게 답해도 돼", pending_seconds=2.0)

        self.assertTrue(payload["released_pending"])
        self.assertEqual(payload["trace"]["event_kind"], "delayed_speech")
        self.assertEqual(messages[-1]["role"], "assistant")

    def test_gui_event_updates_thought_council(self) -> None:
        pipeline = make_pipeline(use_llm=False, llm_config=build_llm_config(provider="gemini", api_key="", model_name="", base_url=""))
        council = ThoughtCouncil()
        messages: list[dict] = []

        payload = run_gui_event(
            pipeline,
            messages,
            event_type="user_message",
            text="머릿속 생각들을 보여줘",
            thought_council=council,
        )

        self.assertGreaterEqual(len(payload["thought_lines"]), 2)
        self.assertEqual(len(council.snapshot_records()), 4)

    def test_background_tick_updates_thought_council(self) -> None:
        pipeline = make_pipeline(use_llm=False, llm_config=build_llm_config(provider="gemini", api_key="", model_name="", base_url=""))
        council = ThoughtCouncil()
        messages: list[dict] = []

        payload = run_background_tick(pipeline, messages, thought_council=council)

        self.assertGreaterEqual(len(payload["thought_lines"]), 1)
        self.assertEqual(council.snapshot_records()[0]["tick_index"], 0)


if __name__ == "__main__":
    unittest.main()
