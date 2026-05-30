import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import local_gui


class LocalGuiTests(unittest.TestCase):
    def test_chat_config_allows_explicit_conditioning_mode(self) -> None:
        config = local_gui._chat_config("secret", conditioning_mode="style")

        self.assertEqual(config.conditioning_mode, "style")
        self.assertEqual(config.api_key, "secret")

    def test_compare_payload_sums_usage_without_api_key(self) -> None:
        payload = local_gui._compare_payload(
            input_text="학교에서 맞기만 하고 외로워.",
            stim_text="일반 응답",
            trace_result=type(
                "TraceResult",
                (),
                {
                    "assistant_text": "trace 응답",
                    "record": {
                        "conditioning_mode": "hybrid_trace",
                        "llm_usage": {"input_tokens": 30, "output_tokens": 12},
                    },
                },
            )(),
            stim_meta={"usage": {"input_tokens": 10, "output_tokens": 5}},
        )

        self.assertEqual(payload["input_text"], "학교에서 맞기만 하고 외로워.")
        self.assertEqual(payload["stim_only"]["assistant_text"], "일반 응답")
        self.assertEqual(payload["trace"]["assistant_text"], "trace 응답")
        self.assertEqual(payload["usage"]["input_tokens"], 40)
        self.assertEqual(payload["usage"]["output_tokens"], 17)
        self.assertNotIn("api_key", jsonish(payload))

    def test_live_compare_trace_prompt_emphasizes_bullying_risk(self) -> None:
        prompt = local_gui._live_compare_trace_prompt(
            base_prompt="[USER_INPUT]\n학교에서 맞기만 하고 외로워.",
            input_text="학교 다니기 너무 외로워. 학교에선 맞기만 하고 같이 다닐 친구도 없어.",
        )

        self.assertIn("맞기만", prompt)
        self.assertIn("physical harm", prompt)
        self.assertIn("단순 외로움이나 친구 만들기 문제로 축소하지 않는다", prompt)
        self.assertIn("보호", prompt)
        self.assertIn("내가 지금 이 일을 겪는다면 몸 안에서 어떤 감정이 생기는지", prompt)
        self.assertIn("외로움보다 먼저 안전하지 않다는 감각", prompt)
        self.assertIn("실질적인 조언보다", prompt)
        self.assertIn("내가 느낀 감정 상태", prompt)
        self.assertIn("행동 지시는 마지막 한 문장 이하", prompt)
        self.assertIn("자연스럽고 진심 어린 말투", prompt)
        self.assertIn("과하게 문학적인 표현", prompt)

    def test_stim_only_prompt_keeps_surface_level_baseline(self) -> None:
        prompt = local_gui._stim_only_prompt("학교에서 맞기만 하고 외로워.")

        self.assertIn("표면적 정서", prompt)
        self.assertIn("상세한 안전 계획", prompt)
        self.assertIn("증거 확보", prompt)
        self.assertIn("뻔한 위로", prompt)
        self.assertIn("뻔한 조언", prompt)

    def test_live_compare_record_overrides_bullying_display_state(self) -> None:
        record = local_gui._live_compare_display_record(
            {
                "appraisal_summary_text": "핵심 appraisal: 위협감 낮음",
                "appraisal_tendency": "정리/수습",
                "appraisal_target": "situation",
                "trace_summary_text": "기존 trace",
                "style_tags": [],
                "anti_softening_rules": [],
                "grounding_rules": [],
            },
            "학교 다니기 너무 외로워. 학교에선 맞기만 하고 같이 다닐 친구도 없어.",
        )

        self.assertIn("안전하지 않다", record["appraisal_summary_text"])
        self.assertEqual(record["appraisal_tendency"], "보호 요청/위험 회피")
        self.assertEqual(record["appraisal_target"], "other")
        self.assertIn("맞고 있다는 신체 위협", record["trace_summary_text"])


def jsonish(value: object) -> str:
    return repr(value).lower()


if __name__ == "__main__":
    unittest.main()
