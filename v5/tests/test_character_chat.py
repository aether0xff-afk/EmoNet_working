import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from emonet.character import (
    CharacterCard,
    CharacterSessionState,
    build_character_context_prompt,
    load_character_card,
    validate_character_response_text,
)
from emonet.chat_service import ChatGenerationConfig, ChatRuntimeConfig, EmoNetChatRuntime, generate_chat_turn
from emonet.legacy_cli import validate_plain_response_text


class CharacterChatTests(unittest.TestCase):
    def test_default_character_card_loads_required_fields(self) -> None:
        card = load_character_card()
        self.assertTrue(card.name)
        self.assertTrue(card.persona)
        self.assertTrue(card.speech_style)
        self.assertTrue(card.relationship_defaults)
        self.assertTrue(card.world_state)
        self.assertGreater(len(card.do_not_say), 0)
        self.assertGreater(len(card.response_rules), 0)

    def test_character_context_prompt_includes_v31_trace_as_emotion_principle(self) -> None:
        card = CharacterCard(
            name="테스트",
            persona="차분한 탐색자",
            speech_style="짧게 말한다",
            relationship_defaults="조심스러운 동료",
            world_state="밤의 연구소",
            do_not_say=("분석 용어를 말하지 않는다",),
            response_rules=("마지막 말에 답한다",),
        )
        state = CharacterSessionState(
            user_memory=("사용자는 공개적 무시에 예민하다.",),
            relationship_state="신뢰를 쌓는 중",
            scene_state="비가 오는 복도",
            affect_state={"felt_pressure": 0.6, "label": "불안/경계"},
        )
        prompt = build_character_context_prompt(
            base_prompt="[USER_INPUT]\n최신 입력",
            character_card=card,
            session_state=state,
            trace_summary="고각성 trace",
            appraisal_summary="타인 방향 방어 성향",
            raw_trace_block="[trace_lines]\n- tick=1\n- tick=2",
        )
        self.assertIn("[CHARACTER_PROFILE]", prompt)
        self.assertIn("[RELATIONSHIP_STATE]", prompt)
        self.assertIn("[SCENE_STATE]", prompt)
        self.assertIn("[RECENT_MEMORY]", prompt)
        self.assertIn("[RAW_EMONET_TRACE]", prompt)
        self.assertIn("[SESSION_AFFECT_STATE]", prompt)
        self.assertIn("불안/경계", prompt)
        self.assertIn("요약하거나 분류하지 말고", prompt)
        self.assertIn("tick=1", prompt)
        self.assertIn("사용자는 공개적 무시에 예민하다.", prompt)

    def test_character_response_validator_rejects_internal_terms(self) -> None:
        def plain(value: str) -> str:
            return value.strip()

        self.assertEqual(validate_character_response_text("그건 그냥 넘길 일이 아니야.", plain), "그건 그냥 넘길 일이 아니야.")
        self.assertEqual(
            validate_character_response_text("[ACTION] 한 발 물러선다.\n...그래. 가.", plain),
            "[ACTION] 한 발 물러선다.\n...그래. 가.",
        )
        with self.assertRaisesRegex(ValueError, "internal"):
            validate_character_response_text("trace를 보면 화가 큽니다.", plain)
        with self.assertRaisesRegex(ValueError, "internal"):
            validate_character_response_text("현재 내부 상태를 보면 긴장이 높습니다.", plain)
        with self.assertRaisesRegex(ValueError, "markdown"):
            validate_character_response_text("좋아.\n\n---\n\n다시 말할게.", plain)
        with self.assertRaisesRegex(ValueError, "ACTION|action"):
            validate_character_response_text("말을 잇지 못하고 한 발 물러선다.", plain)
        with self.assertRaisesRegex(ValueError, "ACTION|action"):
            validate_character_response_text("그래. [ACTION] 고개를 든다. 다시 말할게.", plain)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            validate_character_response_text("너 때문은 아니야. 그건 알아 두.", plain)

        self.assertEqual(
            validate_character_response_text("[ACTION] 창문을 다시 본다.\n...그래. 가.", validate_plain_response_text),
            "[ACTION] 창문을 다시 본다.\n...그래. 가.",
        )

    def test_generate_chat_turn_injects_character_memory_and_trace(self) -> None:
        runtime = EmoNetChatRuntime(
            config=ChatRuntimeConfig(),
            model=object(),
            decoder=object(),
        )
        card = CharacterCard(
            name="Ruca",
            persona="조심스러운 탐색자",
            speech_style="짧고 직접적",
            relationship_defaults="아직 조심스러운 동료",
            world_state="폐연구소 내부",
            do_not_say=("내부 분석 용어를 말하지 않는다.",),
            response_rules=("감정을 덮지 않는다.",),
        )
        state = CharacterSessionState(
            user_memory=("사용자는 공개적으로 무시당하는 일을 싫어한다.",),
            relationship_state="신뢰를 쌓는 중",
            scene_state="비가 오는 밤",
        )
        fake_profile = {
            "stim_vec": np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            "dominant_branch_len": 3,
            "z": np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
            "s_pred": np.asarray([0.25, 0.75], dtype=np.float32),
            "style_dict": {"direct": 0.7},
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
                return_value=("[USER_INPUT]\n최신 입력\n[RAW_TRACE]\n고각성 trace", "raw_trace"),
            ),
            patch(
                "emonet.chat_service.request_plain_text_response",
                return_value=("그건 그냥 넘길 일이 아니야.", "그건 그냥 넘길 일이 아니야.", {"retry_count": 0, "validation_errors": []}),
            ) as request_mock,
        ):
            result = generate_chat_turn(
                runtime=runtime,
                generation_config=ChatGenerationConfig(history_turns=2),
                input_text="최신 입력",
                character_card=card,
                character_session=state,
            )

        called_prompt = request_mock.call_args.kwargs["prompt"]
        self.assertIn("[CHARACTER_PROFILE]", called_prompt)
        self.assertIn("[RECENT_MEMORY]", called_prompt)
        self.assertIn("[RAW_EMONET_TRACE]", called_prompt)
        self.assertIn("[trace_lines]", called_prompt)
        self.assertIn("tick=1", called_prompt)
        self.assertIn("사용자는 공개적으로 무시당하는 일을 싫어한다.", called_prompt)
        self.assertEqual(result.record["character_name"], "Ruca")
        self.assertEqual(result.record["prompt_sections"], "character_context,raw_trace")
        self.assertIn("affect_state", result.character_session.to_record())


if __name__ == "__main__":
    unittest.main()
