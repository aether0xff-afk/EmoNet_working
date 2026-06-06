import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ruca_engine.pipeline import RucaPipeline


class AlwaysOnPipelineTests(unittest.TestCase):
    def test_pipeline_uses_always_on_trace_runtime(self) -> None:
        pipeline = RucaPipeline(use_emonet=True)

        result = pipeline.run_turn("계속 켜져 있는 신경망으로 연결해")
        trace = result.debug_record["emonet_trace"]

        self.assertEqual(trace["source"], "emonet_v7_always_on_runtime")
        self.assertEqual(len(trace["stim_vec"]), 8)
        self.assertIn("cluster_profile", trace)
        self.assertIn("neuron_memory", trace)
        self.assertNotIn("z_dim", trace)
        self.assertNotIn("s_pred_dim", trace)

    def test_typing_event_ticks_the_same_runtime_without_reset(self) -> None:
        pipeline = RucaPipeline(use_emonet=True)

        first = pipeline.run_turn("처음 자극")
        second = pipeline.run_event(event_type="typing", elapsed_minutes=0.1)

        first_trace = first.debug_record["emonet_trace"]
        second_trace = second.debug_record["emonet_trace"]

        self.assertEqual(second_trace["event_kind"], "typing")
        self.assertGreater(second_trace["trace_profile"]["tick_index"], first_trace["trace_profile"]["tick_index"])
        self.assertNotEqual(second_trace["stim_vec"], [0.0] * 8)

    def test_neural_timing_can_hold_user_message_without_reply(self) -> None:
        pipeline = RucaPipeline(use_emonet=True, response_timing_mode="neural")

        result = pipeline.run_turn("짧게 확인")

        self.assertEqual(result.response_decision.action, "update_internal_only")
        self.assertEqual(result.assistant_text, "")
        self.assertEqual(result.debug_record["response_timing_mode"], "neural")


if __name__ == "__main__":
    unittest.main()
