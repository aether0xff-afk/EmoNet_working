import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from emonet.chat_service import (
    ChatGenerationConfig,
    ChatRuntimeConfig,
    EmoNetChatRuntime,
    build_recent_dialogue_block,
    generate_chat_turn,
    parse_episode_payload_text,
    resolve_default_z_encoder_path,
    resolve_default_zs_model_path,
)


class ChatServiceTests(unittest.TestCase):
    def test_default_artifact_resolution_points_to_v4_assets(self) -> None:
        self.assertTrue(resolve_default_z_encoder_path().exists())
        self.assertTrue(resolve_default_zs_model_path().exists())

    def test_parse_episode_payload_text_requires_json_object(self) -> None:
        payload = parse_episode_payload_text('{"episode_label":"test"}')
        self.assertEqual(payload["episode_label"], "test")
        with self.assertRaises(ValueError):
            parse_episode_payload_text('["not", "an", "object"]')

    def test_build_recent_dialogue_block_trims_history(self) -> None:
        history = [
            {"role": "user", "content": "첫 질문"},
            {"role": "assistant", "content": "첫 답변"},
            {"role": "user", "content": "둘째 질문"},
            {"role": "assistant", "content": "둘째 답변"},
            {"role": "user", "content": "셋째 질문"},
        ]
        block = build_recent_dialogue_block(history, max_turns=2)
        self.assertNotIn("첫 질문", block)
        self.assertIn("둘째 질문", block)
        self.assertIn("셋째 질문", block)

    def test_generate_chat_turn_injects_history_and_serializes_record(self) -> None:
        runtime = EmoNetChatRuntime(
            config=ChatRuntimeConfig(),
            model=object(),
            decoder=object(),
        )
        history = [
            {"role": "user", "content": "이전 사용자 메시지"},
            {"role": "assistant", "content": "이전 보조 응답"},
        ]
        fake_profile = {
            "stim_vec": np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            "dominant_branch_len": 3,
            "z": np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
            "s_pred": np.asarray([0.25, 0.75], dtype=np.float32),
            "style_tags": ["direct", "tense"],
            "style_summary": {"direct": 0.7},
            "style_summary_text": "직설성이 높다.",
            "expression_cues_text": "짧고 날카로운 답.",
            "trace_summary_text": "고각성 trace",
            "trace_lines": ["tick=1", "tick=2"],
            "appraisal_summary_text": "배제감과 경계",
            "appraisal_lines": ["target=other"],
            "appraisal_target": "other",
            "appraisal_tendency": "defend",
            "anti_softening_mode": "guarded",
            "anti_softening_rules": ["위로를 덧붙이지 않는다."],
            "grounding_mode": "direct",
            "grounding_rules": ["첫 문장에서 정서를 바로 짚는다."],
            "ticks_run": 7,
            "termination_reason": "stable_convergence",
        }
        with (
            patch("emonet.chat_service.ensure_model_server_ready"),
            patch("emonet.chat_service.infer_style_profile", return_value=fake_profile),
            patch(
                "emonet.chat_service.build_conditioned_generation_prompt",
                return_value=("[USER_INPUT]\n최신 입력", "style_tags"),
            ),
            patch(
                "emonet.chat_service.request_plain_text_response",
                return_value=("응답 문장이다.", "응답 문장이다.", {"retry_count": 0, "validation_errors": []}),
            ) as request_mock,
        ):
            result = generate_chat_turn(
                runtime=runtime,
                generation_config=ChatGenerationConfig(history_turns=2),
                input_text="최신 입력",
                history=history,
            )

        called_prompt = request_mock.call_args.kwargs["prompt"]
        self.assertIn("[RECENT_DIALOGUE]", called_prompt)
        self.assertIn("이전 사용자 메시지", called_prompt)
        self.assertIn("이전 보조 응답", called_prompt)
        self.assertEqual(result.assistant_text, "응답 문장이다.")
        self.assertEqual(result.record["llm_response"], "응답 문장이다.")
        self.assertEqual(result.record["style_tags"], ["direct", "tense"])
        self.assertEqual(result.record["response_retry_count"], 0)

    def test_generate_chat_turn_requires_episode_payload_for_episode_mode(self) -> None:
        runtime = EmoNetChatRuntime(
            config=ChatRuntimeConfig(),
            model=object(),
            decoder=object(),
        )
        with (
            patch("emonet.chat_service.ensure_model_server_ready"),
            patch(
                "emonet.chat_service.infer_style_profile",
                return_value={
                    "stim_vec": np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
                    "dominant_branch_len": 1,
                    "z": np.asarray([0.0], dtype=np.float32),
                    "s_pred": np.asarray([0.0], dtype=np.float32),
                    "style_tags": [],
                    "style_summary": {},
                },
            ),
        ):
            with self.assertRaises(ValueError):
                generate_chat_turn(
                    runtime=runtime,
                    generation_config=ChatGenerationConfig(conditioning_mode="episode_trace"),
                    input_text="최신 입력",
                )
