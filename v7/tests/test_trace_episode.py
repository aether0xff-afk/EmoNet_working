import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ruca_engine.llm_client import LLMConfig, LLMResponse
from ruca_engine.models import EmotionState
from ruca_engine.pipeline import RucaPipeline
from ruca_engine.trace_episode import build_episode_prompt, build_response_prompt_from_episode


class TraceEpisodeTests(unittest.TestCase):
    def make_trace(self) -> SimpleNamespace:
        record = {
            "source": "v7_trace_runtime",
            "emotion_state": EmotionState().to_record(),
            "stim_vec": [0.1, 0.2, 0.3, 0.4],
            "dominant_branch_len": 99,
            "trace_summary_text": "legacy branch summary must not drive v7",
            "trace_lines": ["tick=0 active=3 edges=2", "tick=1 active=2 edges=1"],
            "trace_profile": {
                "ticks_run": 2,
                "active_window_ticks": 2,
                "mean_active_nodes": 2.5,
                "edge_count_total": 3,
            },
        }
        return SimpleNamespace(to_record=lambda: record, emotion_state=EmotionState())

    def test_episode_prompt_uses_trace_packet_not_branch_fields(self) -> None:
        prompt = build_episode_prompt(
            user_text="지금 너무 복잡해",
            event_type="user_message",
            elapsed_minutes=0.0,
            trace_record=self.make_trace().to_record(),
        )

        self.assertIn("[TRACE_PACKET]", prompt)
        self.assertIn("tick=0 active=3 edges=2", prompt)
        self.assertIn("mean_active_nodes", prompt)
        self.assertNotIn("dominant_branch", prompt)
        self.assertNotIn("branch_len", prompt)

    def test_response_prompt_uses_episode_and_hides_trace(self) -> None:
        prompt = build_response_prompt_from_episode(
            user_text="지금 너무 복잡해",
            event_type="user_message",
            elapsed_minutes=0.0,
            episode_text="사용자의 말에 반응해 내부 상태가 빠르게 흔들린다.",
        )

        self.assertIn("[EPISODE]", prompt)
        self.assertIn("사용자의 말에 반응해 내부 상태가 빠르게 흔들린다.", prompt)
        self.assertNotIn("[TRACE_PACKET]", prompt)
        self.assertNotIn("dominant_branch", prompt)

    def test_emonet_llm_pipeline_builds_episode_before_response(self) -> None:
        calls: list[str] = []

        def fake_llm(prompt: str, config: LLMConfig) -> LLMResponse:
            calls.append(prompt)
            if "[TRACE_PACKET]" in prompt:
                return LLMResponse(text="trace episode", raw_text="trace episode", usage={"input_tokens": 1, "output_tokens": 1})
            return LLMResponse(text="final response", raw_text="final response", usage={"input_tokens": 2, "output_tokens": 2})

        pipeline = RucaPipeline(use_emonet=True, use_llm=True, llm_config=LLMConfig(model_name="test", api_key="key"))
        with patch("ruca_engine.pipeline.infer_emonet_trace", return_value=self.make_trace()):
            with patch("ruca_engine.pipeline.generate_llm_response", side_effect=fake_llm):
                result = pipeline.run_turn("지금 너무 복잡해")

        self.assertEqual(result.assistant_text, "final response")
        self.assertEqual(result.debug_record["trace_episode"], "trace episode")
        self.assertEqual(len(calls), 2)
        self.assertIn("[TRACE_PACKET]", calls[0])
        self.assertIn("[EPISODE]", calls[1])
        self.assertNotIn("dominant_branch", calls[0])
        self.assertNotIn("dominant_branch", calls[1])


if __name__ == "__main__":
    unittest.main()
