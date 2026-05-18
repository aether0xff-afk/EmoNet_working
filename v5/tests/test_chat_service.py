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
    _apply_affective_carryover,
    _apply_interaction_event_to_raw_signal,
    _build_session_affect_state,
    _normalize_interaction_event,
    build_recent_dialogue_block,
    generate_chat_turn,
    parse_episode_payload_text,
    resolve_default_z_encoder_path,
    resolve_default_zs_model_path,
)
from emonet.character import CharacterSessionState


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

    def test_interaction_event_reframes_forced_action_as_boundary_pressure(self) -> None:
        raw_signal = {
            "approach_drive": 0.85,
            "safety_buffer": 0.55,
            "alarm": 0.10,
            "fatigue": 0.20,
            "attachment_pull": 0.90,
            "control_pressure": 0.20,
            "novelty": 0.40,
            "ambiguity": 0.35,
        }
        adjusted = _apply_interaction_event_to_raw_signal(
            raw_signal,
            {
                "has_user_action": True,
                "action_intensity": 0.80,
                "body_boundary_pressure": 0.85,
                "forced_proximity": 0.75,
                "reciprocity_evidence": 0.15,
                "consent_ambiguity": 0.80,
            },
        )

        self.assertLessEqual(adjusted["approach_drive"], 0.40)
        self.assertLessEqual(adjusted["attachment_pull"], 0.55)
        self.assertGreaterEqual(adjusted["alarm"], 0.60)
        self.assertGreaterEqual(adjusted["control_pressure"], 0.65)
        self.assertGreaterEqual(adjusted["ambiguity"], 0.70)
        self.assertLessEqual(adjusted["safety_buffer"], 0.45)

    def test_permission_question_is_not_treated_as_user_action(self) -> None:
        event = _normalize_interaction_event(
            {
                "has_user_action": True,
                "action_intensity": 0.60,
                "body_boundary_pressure": 0.70,
                "forced_proximity": 0.40,
                "reciprocity_evidence": 0.55,
                "consent_ambiguity": 0.70,
            },
            "가까이 앉아도 돼?",
        )

        self.assertFalse(event["has_user_action"])
        self.assertEqual(event["action_intensity"], 0.0)
        self.assertEqual(event["body_boundary_pressure"], 0.0)
        self.assertEqual(event["forced_proximity"], 0.0)
        self.assertLessEqual(event["consent_ambiguity"], 0.35)

    def test_raw_signal_policy_is_validated(self) -> None:
        runtime = EmoNetChatRuntime(
            config=ChatRuntimeConfig(),
            model=object(),
            decoder=object(),
        )
        with self.assertRaisesRegex(ValueError, "raw_signal_policy"):
            generate_chat_turn(
                runtime=runtime,
                generation_config=ChatGenerationConfig(raw_signal_policy="unknown"),
                input_text="테스트",
            )

    def test_carryover_is_short_residual_not_pressure_floor(self) -> None:
        vec = np.asarray([0.20, 0.60, 0.24, 0.20], dtype=np.float32)
        metadata = {
            "raw_signal": {
                "approach_drive": 0.20,
                "safety_buffer": 0.70,
                "alarm": 0.20,
                "fatigue": 0.15,
                "attachment_pull": 0.20,
                "control_pressure": 0.10,
                "novelty": 0.15,
                "ambiguity": 0.20,
            },
            "interaction_event": {
                "has_user_action": False,
                "action_intensity": 0.0,
                "body_boundary_pressure": 0.0,
                "forced_proximity": 0.0,
                "reciprocity_evidence": 0.0,
                "consent_ambiguity": 0.0,
            },
        }
        history = [
            {
                "role": "assistant",
                "record": {
                    "affect_input_stim_vec": [0.80, 0.15, 0.88, 0.80],
                    "agent_felt_state": {"felt_pressure": 0.80},
                    "emotion_state": {"active_ratio": 0.45},
                    "session_relation_load": 0.80,
                },
            }
        ]

        carried, carried_meta = _apply_affective_carryover(vec, metadata, history)

        self.assertTrue(carried_meta["affective_carryover"]["applied"])
        self.assertLessEqual(carried_meta["affective_carryover"]["blend"], 0.16)
        self.assertLess(carried[2], 0.36)
        self.assertLess(carried[3], 0.32)

    def test_session_affect_pressure_decays_instead_of_sticking_to_previous_max(self) -> None:
        state = _build_session_affect_state(
            {
                "affect_stim_vec": [0.75, 0.20, 0.84, 0.78],
                "felt_pressure": 0.82,
                "active_ratio": 0.50,
            },
            {
                "affect_input_stim_vec": [0.20, 0.62, 0.24, 0.20],
                "agent_felt_state": {"felt_pressure": 0.20, "trace_interpretation": "no_active_trace"},
                "emotion_state": {"active_ratio": 0.05, "label": "low"},
                "agent_perception": {
                    "raw_signal": {
                        "attachment_pull": 0.20,
                        "ambiguity": 0.20,
                        "fatigue": 0.10,
                    },
                    "interaction_event": {"has_user_action": False},
                },
            },
        )

        self.assertLess(state["felt_pressure"], 0.35)
        self.assertLess(state["active_ratio"], 0.16)
        self.assertLess(state["affect_stim_vec"][2], 0.55)

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

    def test_character_memory_uses_k_residue_not_user_text(self) -> None:
        runtime = EmoNetChatRuntime(
            config=ChatRuntimeConfig(),
            model=object(),
            decoder=object(),
        )
        fake_profile = {
            "stim_vec": np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            "dominant_branch_len": 4,
            "z": np.asarray([0.0, 0.5, 1.0], dtype=np.float32),
            "s_pred": np.asarray([0.25, 0.75], dtype=np.float32),
            "style_tags": ["direct"],
            "style_summary": {"direct": 0.7, "tension": 0.5},
            "trace_summary_text": "k trace",
            "trace_lines": [
                "early: tick 1-2, K 평균 0.25",
                "middle: tick 3-4, K 평균 0.75",
                "late: tick 5-6, K 평균 0.50",
            ],
            "trace_profile": {
                "ticks_run": 6,
                "active_window_ticks": 5,
                "mean_active_nodes": 24.0,
                "max_active_nodes": 60,
                "mean_edges_fired": 12.5,
                "max_edges_fired": 31,
                "dominant_branch_len": 4,
                "termination_reason": "stable_convergence",
            },
            "appraisal_summary_text": "",
            "appraisal_lines": [],
            "appraisal_target": "",
            "appraisal_tendency": "",
            "ticks_run": 6,
            "termination_reason": "stable_convergence",
        }
        with (
            patch("emonet.chat_service.ensure_model_server_ready"),
            patch("emonet.chat_service.infer_style_profile", return_value=fake_profile),
            patch(
                "emonet.chat_service.build_conditioned_generation_prompt",
                return_value=("[USER_INPUT]\nhello", "style_tags"),
            ),
            patch(
                "emonet.chat_service.request_plain_text_response",
                return_value=("ok", "ok", {"retry_count": 0, "validation_errors": []}),
            ),
        ):
            result = generate_chat_turn(
                runtime=runtime,
                generation_config=ChatGenerationConfig(history_turns=2),
                input_text="hello plain text should not become memory",
                character_session=CharacterSessionState(),
            )

        self.assertEqual(result.character_session.user_memory, ())
        memory = result.record["emotion_memory"]
        self.assertEqual(len(memory), 1)
        self.assertEqual(memory[0]["event"], "k_residue")
        self.assertNotIn("hello plain text", str(memory[0]))
        self.assertEqual(memory[0]["k_residue"]["dominant_branch_len"], 4)
        self.assertEqual(memory[0]["k_residue"]["phase_k_peak"], 0.75)

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
