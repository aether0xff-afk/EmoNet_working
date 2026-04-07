import csv
import io
import contextlib
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd

from emonet import (
    BranchExtractor,
    BranchPath,
    BranchStep,
    EmoNet,
    EmoNetConfig,
    LinearZtoSDecoder,
    NodeStepState,
    StimEncoderConfig,
    TickRecord,
    TORCH_AVAILABLE,
)
if TORCH_AVAILABLE:
    import torch
from emonet.cli import (
    STYLE_AXIS_NAMES,
    build_anti_softening_policy,
    build_balanced_subset,
    build_response_generation_prompt,
    command_e2e_check,
    command_fit_z_encoder,
    command_probe_branch,
    command_predict_s,
    command_generate_response,
    command_generate_response_batch,
    command_build_llm_subset,
    build_style_summary,
    build_style_tags,
    export_z_from_json_stream,
    extract_json_block,
    label_subset_with_local_model,
    normalize_style_dict,
    request_json_response,
    resolve_style_axes,
    summarize_expression_cues,
    train_zs_decoder_from_dataframe,
)


def write_csv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


class EmoNetSmokeTests(unittest.TestCase):
    class FakeGenerativeModel:
        def forward(self, text: str):
            z = np.linspace(0.0, 1.0, 64, dtype=np.float32)
            return {
                "stim_vec": np.asarray([0.2, 0.4, 0.6, 0.8], dtype=np.float32),
                "dominant_branch": [object(), object()],
                "z": z,
            }

    class FakeDecoder:
        def predict(self, z):
            arr = np.asarray(z, dtype=np.float32)
            if arr.ndim == 1:
                return np.linspace(0.1, 0.9, 32, dtype=np.float32)
            return np.tile(np.linspace(0.1, 0.9, 32, dtype=np.float32), (arr.shape[0], 1))

    def make_stim_encoder_config(self, temp_dir: Path) -> StimEncoderConfig:
        dataset_csv = temp_dir / "dataset_for_regression.csv"
        benchmark_csv = temp_dir / "benchmark_results.csv"
        model_cache_path = temp_dir / "stim_encoder.joblib"

        write_csv(
            dataset_csv,
            [
                ["text", "label", "y", "talk_id", "persona_id"],
                ["urgent critical alert now", "E10", "0.05", "t1", "p1"],
                ["i feel calm and safe with support", "E20", "0.90", "t2", "p2"],
                ["too tired and burned out i need rest", "E30", "0.20", "t3", "p3"],
                ["we made progress and i can handle this", "E40", "0.85", "t4", "p4"],
                ["this risk is scary and stressful", "E50", "0.10", "t5", "p5"],
                ["quiet stable day and peaceful mood", "E60", "0.95", "t6", "p6"],
            ],
        )
        write_csv(
            benchmark_csv,
            [
                ["vector", "model", "status", "MAE(mean)", "RMSE(mean)"],
                ["char_tfidf", "Ridge", "ok", "0.1", "0.2"],
            ],
        )
        return StimEncoderConfig(
            dataset_csv=dataset_csv,
            benchmark_csv=benchmark_csv,
            model_cache_path=model_cache_path,
            force_refit=True,
        )

    def test_simulation_produces_branch_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            stim_config = self.make_stim_encoder_config(Path(temp_dir_name))
            model = EmoNet(EmoNetConfig(seed=7), stim_encoder_config=stim_config)
            base_stim = model.run_until_converged("urgent critical alert now")

            self.assertEqual(base_stim.shape, (4,))
            self.assertGreaterEqual(len(model.state.branch_log), 1)
            self.assertTrue(np.all(base_stim >= 0.0))
            self.assertTrue(np.all(base_stim <= 1.0))

            pruned = model.prune_to_survivors()
            branches = model.extract_topk_branches()
            dominant = model.build_dominant_branch()
            tensor = model.dominant_branch_to_tensor(dominant)
            z = model.encode_z(tensor)

            self.assertIsInstance(pruned, list)
            self.assertIsInstance(branches, list)
            self.assertGreaterEqual(len(dominant), 1)
            self.assertEqual(tuple(tensor.shape), (len(dominant), 6))
            self.assertEqual(tuple(z.shape), (64,))

    def test_forward_returns_z_without_torch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            stim_config = self.make_stim_encoder_config(Path(temp_dir_name))
            model = EmoNet(EmoNetConfig(seed=11), stim_encoder_config=stim_config)
            outputs = model.forward("quiet stable day and peaceful mood")

            self.assertIn("stim_vec", outputs)
            self.assertIn("dominant_branch", outputs)
            self.assertIn("branch_tensor", outputs)
            self.assertIn("z", outputs)
            self.assertNotIn("s", outputs)
            self.assertEqual(tuple(outputs["stim_vec"].shape), (4,))
            self.assertEqual(tuple(outputs["branch_tensor"].shape)[1], 6)
            self.assertEqual(tuple(outputs["z"].shape), (64,))

    def test_export_z_from_json_stream_writes_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            stim_config = self.make_stim_encoder_config(temp_dir)
            model = EmoNet(EmoNetConfig(seed=13), stim_encoder_config=stim_config)

            input_json = temp_dir / "dialogs.json"
            output_csv = temp_dir / "out_z.csv"
            dialogs = [
                {
                    "profile": {"persona-id": "p1", "emotion": {"type": "E10"}},
                    "talk": {
                        "id": {"talk-id": "t1", "profile-id": "p1"},
                        "content": {"HS01": "urgent alert", "SS01": "calm down", "HS02": "", "SS02": "", "HS03": "", "SS03": ""},
                    },
                },
                {
                    "profile": {"persona-id": "p2", "emotion": {"type": "E20"}},
                    "talk": {
                        "id": {"talk-id": "t2", "profile-id": "p2"},
                        "content": {"HS01": "i need rest", "SS01": "take a break", "HS02": "", "SS02": "", "HS03": "", "SS03": ""},
                    },
                },
            ]
            input_json.write_text(json.dumps(dialogs, ensure_ascii=False), encoding="utf-8")

            export_z_from_json_stream(
                model=model,
                input_json=input_json,
                output_csv=output_csv,
                limit=None,
                chunk_size=1,
                progress_every=1,
                resume=False,
            )

            df = pd.read_csv(output_csv)
            self.assertEqual(len(df), 2)
            self.assertIn("text", df.columns)
            self.assertIn("talk_id", df.columns)
            self.assertIn("z_0", df.columns)
            self.assertIn("z_63", df.columns)
            self.assertIn("dopamine", df.columns)
            self.assertIn("dominant_branch_len", df.columns)

    def test_aggregate_signal_bundle_uses_topk_sum_and_weighted_stim(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            stim_config = self.make_stim_encoder_config(Path(temp_dir_name))
            model = EmoNet(
                EmoNetConfig(seed=17, input_topk=2, input_signal_clip=1.2),
                stim_encoder_config=stim_config,
            )
            strength, stim_vec = model._aggregate_signal_bundle(
                [
                    (0.9, np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)),
                    (0.7, np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32)),
                    (0.2, np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float32)),
                ]
            )

            self.assertAlmostEqual(strength, 1.2, places=6)
            np.testing.assert_allclose(
                stim_vec,
                np.asarray([0.5625, 0.4375, 0.0, 0.0], dtype=np.float32),
                atol=1e-6,
            )

    def test_compose_neuron_stimulus_carries_self_parent_base_and_bias(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            stim_config = self.make_stim_encoder_config(Path(temp_dir_name))
            model = EmoNet(
                EmoNetConfig(
                    seed=19,
                    state_self_stim_mix=0.50,
                    state_parent_stim_mix=0.25,
                    state_base_stim_mix=0.15,
                    state_bias_stim_mix=0.10,
                ),
                stim_encoder_config=stim_config,
            )
            neuron = model.state.neurons[0]
            neuron.stim_vec = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            neuron.intrinsic_bias = np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            composed = model._compose_neuron_stimulus(
                neuron,
                base_stim_vec=np.asarray([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
                parent_stim_vec=np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
            )

            np.testing.assert_allclose(
                composed,
                np.asarray([0.5, 0.25, 0.15, 0.1], dtype=np.float32),
                atol=1e-6,
            )

    def test_compute_local_activation_params_apply_hysteresis(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            stim_config = self.make_stim_encoder_config(Path(temp_dir_name))
            model = EmoNet(
                EmoNetConfig(
                    seed=23,
                    hysteresis_threshold_gain=0.10,
                    hysteresis_remem_gain=0.05,
                ),
                stim_encoder_config=stim_config,
            )
            neuron = model.state.neurons[0]
            neuron.recent_activity = 1.5
            threshold, remem = model._compute_local_activation_params(neuron, 0.72, 0.95)

            self.assertAlmostEqual(threshold, 0.57, places=6)
            self.assertAlmostEqual(remem, 0.875, places=6)

    def test_command_probe_branch_reports_stats(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            input_csv = temp_dir / "probe_input.csv"
            output_csv = temp_dir / "probe_output.csv"
            pd.DataFrame(
                [
                    {"text": "len1", "talk_id": "t1"},
                    {"text": "len3", "talk_id": "t2"},
                    {"text": "len5", "talk_id": "t3"},
                ]
            ).to_csv(input_csv, index=False, encoding="utf-8-sig")

            class ProbeModel:
                def forward(self, text: str):
                    branch_len = int(str(text).replace("len", ""))
                    return {
                        "dominant_branch": [object()] * branch_len,
                    }

            class Args:
                pass

            args = Args()
            args.dataset_csv = None
            args.benchmark_csv = None
            args.model_cache_path = None
            args.max_samples = None
            args.force_refit = False
            args.seed = 7
            args.z_dim = 64
            args.z_encoder_mode = "stat"
            args.z_encoder_path = None
            args.max_ticks = None
            args.min_ticks_before_converged = None
            args.k_threshold_base = None
            args.k_remem_base = None
            args.k_decay = None
            args.refractory_ticks = None
            args.input_topk = None
            args.input_signal_clip = None
            args.memory_decay = None
            args.memory_stim_mix = None
            args.memory_k_mix = None
            args.state_self_stim_mix = None
            args.state_parent_stim_mix = None
            args.state_base_stim_mix = None
            args.state_bias_stim_mix = None
            args.recent_activity_decay = None
            args.hysteresis_threshold_gain = None
            args.hysteresis_remem_gain = None
            args.hysteresis_k_bonus = None
            args.max_out_degree = None
            args.min_out_degree = None
            args.dopa_rewire_gain = None
            args.sero_prune_gain = None
            args.mela_dropout_gain = None
            args.ne_thresh_reduce_gain = None
            args.ne_remem_reduce_gain = None
            args.global_recovery_rate = None
            args.topk_branches = None
            args.branch_end_window = None
            args.branch_length_bonus = None
            args.input_csv = str(input_csv)
            args.input_json = None
            args.text_column = "text"
            args.sample_size = 2
            args.sample_mode = "head"
            args.progress_every = 0
            args.output_csv = str(output_csv)

            stdout = io.StringIO()
            with patch("emonet.cli.build_model", return_value=ProbeModel()), contextlib.redirect_stdout(stdout):
                command_probe_branch(args)

            payload = json.loads(stdout.getvalue())
            self.assertEqual(payload["input_rows"], 3)
            self.assertEqual(payload["sample_rows"], 2)
            self.assertEqual(payload["mean"], 2.0)
            self.assertEqual(payload["len1"], 1)
            self.assertEqual(payload["max"], 3)
            self.assertTrue(output_csv.exists())
            saved = pd.read_csv(output_csv)
            self.assertEqual(saved["dominant_branch_len"].tolist(), [1, 3])

    def test_run_until_converged_respects_min_ticks_before_stopping(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            stim_config = self.make_stim_encoder_config(Path(temp_dir_name))
            model = EmoNet(
                EmoNetConfig(
                    seed=17,
                    delta_k_eps=1e9,
                    min_ticks_before_converged=4,
                ),
                stim_encoder_config=stim_config,
            )
            model.run_until_converged("urgent critical alert now")
            self.assertGreaterEqual(model.state.tick, 4)

    def test_build_balanced_subset_and_prompt_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            input_csv = temp_dir / "out_z_training.csv"
            output_csv = temp_dir / "subset.csv"
            prompt_jsonl = temp_dir / "subset_prompts.jsonl"

            df = pd.DataFrame(
                [
                    {"text": f"text {i}", "talk_id": f"t{i}", "label": "E10" if i < 4 else "E20", **{f"z_{j}": float(i + j) for j in range(64)}}
                    for i in range(8)
                ]
            )
            df.to_csv(input_csv, index=False, encoding="utf-8-sig")

            subset = build_balanced_subset(df, target_size=4, label_column="label", seed=7)
            self.assertEqual(len(subset), 4)
            self.assertIn("sample_id", subset.columns)
            self.assertEqual(set(subset["label"].unique()), {"E10", "E20"})

            class Args:
                pass

            args = Args()
            args.input_csv = str(input_csv)
            args.output_csv = str(output_csv)
            args.prompt_jsonl = str(prompt_jsonl)
            args.target_size = 4
            args.label_column = "label"
            args.seed = 7
            command_build_llm_subset(args)

            saved = pd.read_csv(output_csv)
            self.assertEqual(len(saved), 4)
            lines = prompt_jsonl.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 4)
            payload = json.loads(lines[0])
            self.assertIn("generation_prompt", payload)

    def test_json_extraction_and_style_normalization(self) -> None:
        style_payload = {axis: 0.5 for axis in STYLE_AXIS_NAMES}
        style_payload["verbosity"] = 1.2
        style_payload["sentence_length"] = -0.2
        payload = extract_json_block(f"prefix\n{json.dumps({'s': style_payload})}\nsuffix")
        style = normalize_style_dict(payload, "s")
        self.assertEqual(style["verbosity"], 1.0)
        self.assertEqual(style["sentence_length"], 0.0)
        self.assertEqual(len(style), 32)
        with self.assertRaises(ValueError):
            normalize_style_dict({"s": {"verbosity": 0.3}}, "s", expected_axes=["verbosity", "sentence_length"])

    def test_build_response_generation_prompt_uses_condensed_controls(self) -> None:
        style = {axis: 0.5 for axis in STYLE_AXIS_NAMES}
        style["tension"] = 1.0
        style["warmth"] = 0.0
        prompt = build_response_generation_prompt(
            input_text="예시 입력",
            style_dict=style,
            style_tags=["긴장높음", "건조함", "직설적", "무게감", "여분태그"],
            style_summary=build_style_summary(style),
        )
        self.assertIn("[STYLE_TAGS]", prompt)
        self.assertIn("[STYLE_SUMMARY]", prompt)
        self.assertIn("[ANTI_SOFTENING_RULES]", prompt)
        self.assertNotIn("[EXPRESSION_CUES]", prompt)
        self.assertNotIn("[STYLE_VECTOR]", prompt)
        self.assertNotIn("여분태그", prompt)

    def test_build_anti_softening_policy_turns_strict_for_distress_text(self) -> None:
        style = {axis: 0.5 for axis in STYLE_AXIS_NAMES}
        style["softness"] = 0.95
        style["positivity"] = 0.95
        style["tension"] = 0.1
        style_summary = build_style_summary(style)
        mode, rules = build_anti_softening_policy(
            input_text="지금 너무 예민하고 피곤해.",
            style_dict=style,
            style_summary=style_summary,
            stim_vec=np.asarray([0.2, 0.3, 0.6, 0.7], dtype=np.float32),
        )
        self.assertEqual(mode, "strict")
        self.assertGreaterEqual(len(rules), 3)

    def test_resolve_extended_style_axes(self) -> None:
        axes = resolve_style_axes(40, style_profile="extended40")
        self.assertEqual(len(axes), 40)
        self.assertIn("hostility", axes)
        self.assertIn("trust", axes)

    def test_build_dominant_branch_uses_single_best_path(self) -> None:
        extractor = BranchExtractor()
        best_path = BranchPath(
            score=3.0,
            steps=[
                BranchStep(tick=0, node_id=1, K=1.2, stim_vec=np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)),
                BranchStep(tick=1, node_id=3, K=1.8, stim_vec=np.asarray([0.4, 0.3, 0.2, 0.1], dtype=np.float32)),
            ],
        )
        weaker_path = BranchPath(
            score=2.0,
            steps=[
                BranchStep(tick=0, node_id=2, K=0.9, stim_vec=np.asarray([0.9, 0.1, 0.1, 0.1], dtype=np.float32)),
            ],
        )
        dominant = extractor.build_dominant_branch(
            topk_paths=[best_path, weaker_path],
            fallback_stim_vec=np.zeros(4, dtype=np.float32),
            branch_log=[],
            topk=2,
        )
        self.assertEqual(len(dominant), 2)
        self.assertAlmostEqual(float(dominant[0].K), 1.2)
        self.assertTrue(np.allclose(dominant[0].stim_vec, best_path.steps[0].stim_vec))
        self.assertAlmostEqual(float(dominant[1].K), 1.8)

    def test_extract_topk_branches_prefers_persistent_path_over_late_spike(self) -> None:
        extractor = BranchExtractor()
        branch_log = [
            TickRecord(
                tick=0,
                active_nodes=[1],
                node_states={1: NodeStepState(K=1.0, stim_vec=np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32))},
                edges_fired=[(1, 2)],
            ),
            TickRecord(
                tick=1,
                active_nodes=[2],
                node_states={2: NodeStepState(K=1.0, stim_vec=np.asarray([0.2, 0.2, 0.3, 0.4], dtype=np.float32))},
                edges_fired=[(2, 3)],
            ),
            TickRecord(
                tick=2,
                active_nodes=[3],
                node_states={3: NodeStepState(K=1.0, stim_vec=np.asarray([0.3, 0.2, 0.3, 0.4], dtype=np.float32))},
                edges_fired=[],
            ),
            TickRecord(
                tick=3,
                active_nodes=[9],
                node_states={9: NodeStepState(K=2.2, stim_vec=np.asarray([0.9, 0.1, 0.1, 0.1], dtype=np.float32))},
                edges_fired=[],
            ),
        ]

        topk = extractor.extract_topk_branches_with_strategy(
            branch_log,
            topk=2,
            end_window=2,
            length_bonus=0.5,
        )

        self.assertEqual(len(topk), 2)
        self.assertEqual([step.node_id for step in topk[0].steps], [1, 2, 3])
        self.assertGreater(topk[0].score, topk[1].score)

    def test_label_subset_with_local_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            output_csv = temp_dir / "labeled.csv"
            active_axes = STYLE_AXIS_NAMES[:16]
            df = pd.DataFrame(
                [
                    {
                        "sample_id": "s_000000",
                        "text": "example text",
                        "talk_id": "t1",
                        "label": "E10",
                        **{f"z_{j}": float(j) / 64.0 for j in range(64)},
                    }
                ]
            )

            generation_response = json.dumps({"response": "예시 응답"}, ensure_ascii=False)
            block_responses = []
            for block_idx in range(0, len(active_axes), 8):
                block_axes = active_axes[block_idx : block_idx + 8]
                block_responses.append(json.dumps({"s": {axis: 0.5 for axis in block_axes}}, ensure_ascii=False))
            for block_idx in range(0, len(active_axes), 8):
                block_axes = active_axes[block_idx : block_idx + 8]
                block_responses.append(json.dumps({"s_hat": {axis: 0.55 for axis in block_axes}}, ensure_ascii=False))

            with patch("emonet.cli.ensure_model_server_ready"), patch(
                "emonet.cli.call_openai_compatible_chat", side_effect=[generation_response, *block_responses]
            ):
                label_subset_with_local_model(
                    df=df,
                    output_csv=output_csv,
                    base_url="http://127.0.0.1:8000/v1",
                    model_name="gpt-oss-20b",
                    generation_temperature=0.7,
                    rating_temperature=0.1,
                    max_tokens=1200,
                    timeout_sec=30,
                    progress_every=1,
                    limit=None,
                    max_retries=1,
                    keep_failures=True,
                    block_size=8,
                    style_dim=16,
                    keep_threshold=0.12,
                    flush_every=1,
                    resume=False,
                )

            saved = pd.read_csv(output_csv)
            self.assertEqual(len(saved), 1)
            self.assertIn("llm_response", saved.columns)
            self.assertIn("generation_status", saved.columns)
            self.assertIn("s_block1_status", saved.columns)
            self.assertIn("s_hat_block2_status", saved.columns)
            self.assertIn("s_0", saved.columns)
            self.assertIn("s_hat_15", saved.columns)
            self.assertIn("consistency_l1", saved.columns)
            self.assertEqual(saved.loc[0, "status"], "ok")
            self.assertEqual(saved.loc[0, "generation_status"], "ok")
            self.assertEqual(saved.loc[0, "s_block1_status"], "ok")
            self.assertEqual(saved.loc[0, "s_hat_block2_status"], "ok")
            self.assertEqual(saved.loc[0, "style_dim"], 16)
            self.assertEqual(saved.loc[0, "style_profile"], "core32")
            self.assertNotIn("s_16", saved.columns)

    def test_label_subset_with_local_model_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            output_csv = temp_dir / "labeled_resume.csv"
            active_axes = STYLE_AXIS_NAMES[:8]
            df = pd.DataFrame(
                [
                    {
                        "sample_id": "s_000000",
                        "text": "example text 0",
                        "talk_id": "t1",
                        "label": "E10",
                        **{f"z_{j}": float(j) / 64.0 for j in range(64)},
                    },
                    {
                        "sample_id": "s_000001",
                        "text": "example text 1",
                        "talk_id": "t2",
                        "label": "E20",
                        **{f"z_{j}": float(j + 1) / 64.0 for j in range(64)},
                    },
                ]
            )

            row1_responses = [
                json.dumps({"response": "첫 번째 응답"}, ensure_ascii=False),
                json.dumps({"s": {axis: 0.4 for axis in active_axes}}, ensure_ascii=False),
                json.dumps({"s_hat": {axis: 0.45 for axis in active_axes}}, ensure_ascii=False),
            ]
            row2_responses = [
                json.dumps({"response": "두 번째 응답"}, ensure_ascii=False),
                json.dumps({"s": {axis: 0.6 for axis in active_axes}}, ensure_ascii=False),
                json.dumps({"s_hat": {axis: 0.55 for axis in active_axes}}, ensure_ascii=False),
            ]

            with patch("emonet.cli.ensure_model_server_ready"), patch(
                "emonet.cli.call_openai_compatible_chat", side_effect=row1_responses
            ):
                label_subset_with_local_model(
                    df=df,
                    output_csv=output_csv,
                    base_url="http://127.0.0.1:8000/v1",
                    model_name="gpt-oss-20b",
                    generation_temperature=0.7,
                    rating_temperature=0.1,
                    max_tokens=1200,
                    timeout_sec=30,
                    progress_every=1,
                    limit=1,
                    max_retries=1,
                    keep_failures=True,
                    block_size=8,
                    style_dim=8,
                    keep_threshold=0.12,
                    flush_every=1,
                    resume=False,
                )

            with patch("emonet.cli.ensure_model_server_ready"), patch(
                "emonet.cli.call_openai_compatible_chat", side_effect=row2_responses
            ):
                label_subset_with_local_model(
                    df=df,
                    output_csv=output_csv,
                    base_url="http://127.0.0.1:8000/v1",
                    model_name="gpt-oss-20b",
                    generation_temperature=0.7,
                    rating_temperature=0.1,
                    max_tokens=1200,
                    timeout_sec=30,
                    progress_every=1,
                    limit=None,
                    max_retries=1,
                    keep_failures=True,
                    block_size=8,
                    style_dim=8,
                    keep_threshold=0.12,
                    flush_every=1,
                    resume=True,
                )

            saved = pd.read_csv(output_csv)
            self.assertEqual(len(saved), 2)
            self.assertEqual(saved["sample_id"].tolist(), ["s_000000", "s_000001"])
            self.assertEqual(saved["llm_response"].tolist(), ["첫 번째 응답", "두 번째 응답"])

    def test_request_json_response_retries_on_schema_validation(self) -> None:
        with patch(
            "emonet.cli.call_openai_compatible_chat",
            side_effect=[
                "{\"s\": {\"dim0\": 0.2}}",
                "{\"s\": {\"verbosity\": 0.2, \"sentence_length\": 0.3}}",
            ],
        ):
            payload, raw = request_json_response(
                base_url="http://127.0.0.1:8000/v1",
                model_name="gpt-oss-20b",
                prompt="test",
                temperature=0.1,
                max_tokens=100,
                timeout_sec=30,
                max_retries=1,
                validator=lambda body: normalize_style_dict(
                    body,
                    "s",
                    expected_axes=["verbosity", "sentence_length"],
                ),
                retry_instruction="축 이름을 그대로 다시 출력하라.",
            )
        self.assertEqual(payload["verbosity"], 0.25)
        self.assertEqual(payload["sentence_length"], 0.25)
        self.assertEqual(raw, "{\"s\": {\"verbosity\": 0.2, \"sentence_length\": 0.3}}")

    def test_request_json_response_retries_once(self) -> None:
        with patch("emonet.cli.call_openai_compatible_chat", side_effect=["not json", "{\"ok\": true}"]):
            payload, raw = request_json_response(
                base_url="http://127.0.0.1:8000/v1",
                model_name="gpt-oss-20b",
                prompt="test",
                temperature=0.7,
                max_tokens=100,
                timeout_sec=30,
                max_retries=1,
            )
        self.assertTrue(payload["ok"])
        self.assertEqual(raw, "{\"ok\": true}")

    def test_build_style_tags_and_summary(self) -> None:
        style = {axis: 0.5 for axis in STYLE_AXIS_NAMES}
        style["tension"] = 1.0
        style["warmth"] = 0.0
        style["directness"] = 0.75
        tags = build_style_tags(style, max_tags=5)
        summary = build_style_summary(style)
        expression_cues = summarize_expression_cues(style)
        self.assertIn("긴장높음", tags)
        self.assertIn("건조함", tags)
        self.assertIn("tension", summary)
        self.assertIn("warmth", summary)
        self.assertIn("표정 변화=", expression_cues)

    def test_command_generate_response(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            output_json = temp_dir / "response.json"
            log_jsonl = temp_dir / "response_log.jsonl"

            class Args:
                pass

            args = Args()
            args.dataset_csv = None
            args.benchmark_csv = None
            args.model_cache_path = None
            args.max_samples = None
            args.force_refit = False
            args.seed = 42
            args.z_dim = 64
            args.zs_model_path = str(temp_dir / "decoder.npz")
            args.base_url = "http://127.0.0.1:11434/v1"
            args.model_name = "gpt-oss:20b"
            args.response_temperature = 0.5
            args.max_tokens = 300
            args.timeout_sec = 30
            args.prompt_template = None
            args.log_jsonl = str(log_jsonl)
            args.text = "지금 너무 예민하고 피곤해."
            args.output_json = str(output_json)

            with patch("emonet.cli.ensure_model_server_ready"), patch(
                "emonet.cli.build_model", return_value=self.FakeGenerativeModel()
            ), patch("emonet.cli.LinearZtoSDecoder.load", return_value=self.FakeDecoder()), patch(
                "emonet.cli.call_openai_compatible_chat", return_value="조금 쉬어가면서 우선순위를 다시 정리해 보세요."
            ):
                command_generate_response(args)

            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["llm_response"], "조금 쉬어가면서 우선순위를 다시 정리해 보세요.")
            self.assertIn("style_tags", payload)
            self.assertIn("style_summary", payload)
            self.assertIn("expression_cues_text", payload)
            self.assertIn("anti_softening_mode", payload)
            self.assertIn("anti_softening_rules", payload)
            self.assertTrue(log_jsonl.exists())

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is required for grad tensor response test")
    def test_command_generate_response_accepts_grad_tensor_z(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            output_json = temp_dir / "response_grad.json"

            class GradTensorModel:
                def forward(self, text: str):
                    z = torch.linspace(0.0, 1.0, 64, dtype=torch.float32, requires_grad=True)
                    return {
                        "stim_vec": torch.tensor([0.2, 0.4, 0.6, 0.8], dtype=torch.float32),
                        "dominant_branch": [object(), object(), object()],
                        "z": z,
                    }

            class Args:
                pass

            args = Args()
            args.dataset_csv = None
            args.benchmark_csv = None
            args.model_cache_path = None
            args.max_samples = None
            args.force_refit = False
            args.seed = 42
            args.z_dim = 64
            args.zs_model_path = str(temp_dir / "decoder.npz")
            args.base_url = "http://127.0.0.1:11434/v1"
            args.model_name = "gpt-oss:20b"
            args.response_temperature = 0.5
            args.max_tokens = 300
            args.timeout_sec = 30
            args.prompt_template = None
            args.log_jsonl = None
            args.text = "지금 너무 예민하고 피곤해."
            args.output_json = str(output_json)

            with patch("emonet.cli.ensure_model_server_ready"), patch(
                "emonet.cli.build_model", return_value=GradTensorModel()
            ), patch("emonet.cli.LinearZtoSDecoder.load", return_value=self.FakeDecoder()), patch(
                "emonet.cli.call_openai_compatible_chat", return_value="조금만 정리하고 쉬자."
            ):
                command_generate_response(args)

            payload = json.loads(output_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["dominant_branch_len"], 3)
            self.assertEqual(len(payload["z"]), 64)

    @unittest.skipUnless(TORCH_AVAILABLE, "torch is required for transformer z-encoder training")
    def test_command_fit_z_encoder(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            stim_config = self.make_stim_encoder_config(temp_dir)
            input_csv = temp_dir / "labeled_for_z_encoder.csv"
            z_encoder_path = temp_dir / "dominant_branch_encoder.pt"
            zs_model_path = temp_dir / "z_to_s_decoder.npz"
            z_output_csv = temp_dir / "learned_z.csv"

            rows = []
            texts = [
                "urgent alert and i feel angry",
                "i am tired and need some rest",
                "this feels stable and calm",
                "i am nervous but trying to hold on",
                "i feel resentful and exhausted",
                "there is still some relief now",
            ]
            for idx, text in enumerate(texts):
                row = {"text": text, "talk_id": f"t{idx}", "keep_sample": True}
                for axis_idx in range(8):
                    row[f"s_{axis_idx}"] = float(((idx + axis_idx) % 5) / 4.0)
                rows.append(row)
            pd.DataFrame(rows).to_csv(input_csv, index=False, encoding="utf-8-sig")

            class Args:
                pass

            args = Args()
            args.input_csv = str(input_csv)
            args.text_column = "text"
            args.dataset_csv = str(stim_config.dataset_csv)
            args.benchmark_csv = str(stim_config.benchmark_csv)
            args.model_cache_path = str(stim_config.model_cache_path)
            args.max_samples = None
            args.force_refit = True
            args.seed = 42
            args.z_dim = 16
            args.z_encoder_mode = "auto"
            args.z_encoder_path = str(z_encoder_path)
            args.zs_model_path = str(zs_model_path)
            args.z_output_csv = str(z_output_csv)
            args.style_dim = 8
            args.style_profile = "core32"
            args.epochs = 1
            args.batch_size = 2
            args.learning_rate = 1e-3
            args.weight_decay = 0.0
            args.ridge_alpha = 1.0
            args.val_ratio = 0.34
            args.progress_every = 100
            args.use_all_rows = True
            args.warm_start_z_encoder = False

            command_fit_z_encoder(args)

            self.assertTrue(z_encoder_path.exists())
            self.assertTrue(zs_model_path.exists())
            saved = pd.read_csv(z_output_csv)
            self.assertIn("z_0", saved.columns)
            self.assertIn("z_15", saved.columns)
            decoder = LinearZtoSDecoder.load(zs_model_path)
            z_columns = [f"z_{idx}" for idx in range(16)]
            pred = decoder.predict(saved.loc[0, z_columns].to_numpy(dtype=np.float32))
            self.assertEqual(tuple(pred.shape), (8,))

    def test_command_generate_response_batch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            input_csv = temp_dir / "batch_input.csv"
            output_csv = temp_dir / "batch_output.csv"
            log_jsonl = temp_dir / "batch_log.jsonl"
            pd.DataFrame(
                [
                    {"talk_id": "t1", "text": "너무 불안해."},
                    {"talk_id": "t2", "text": "조금 기쁘기도 해."},
                ]
            ).to_csv(input_csv, index=False, encoding="utf-8-sig")

            class Args:
                pass

            args = Args()
            args.dataset_csv = None
            args.benchmark_csv = None
            args.model_cache_path = None
            args.max_samples = None
            args.force_refit = False
            args.seed = 42
            args.z_dim = 64
            args.zs_model_path = str(temp_dir / "decoder.npz")
            args.base_url = "http://127.0.0.1:11434/v1"
            args.model_name = "gpt-oss:20b"
            args.response_temperature = 0.5
            args.max_tokens = 300
            args.timeout_sec = 30
            args.prompt_template = None
            args.log_jsonl = str(log_jsonl)
            args.input_csv = str(input_csv)
            args.output_csv = str(output_csv)
            args.text_column = "text"
            args.limit = None
            args.progress_every = 1

            with patch("emonet.cli.ensure_model_server_ready"), patch(
                "emonet.cli.build_model", return_value=self.FakeGenerativeModel()
            ), patch("emonet.cli.LinearZtoSDecoder.load", return_value=self.FakeDecoder()), patch(
                "emonet.cli.call_openai_compatible_chat", side_effect=["응답 하나", "응답 둘"]
            ):
                command_generate_response_batch(args)

            saved = pd.read_csv(output_csv)
            self.assertEqual(len(saved), 2)
            self.assertIn("style_tags", saved.columns)
            self.assertIn("style_summary_text", saved.columns)
            self.assertIn("expression_cues_text", saved.columns)
            self.assertIn("llm_response", saved.columns)
            self.assertIn("macro_tension", saved.columns)
            self.assertIn("s_pred_31", saved.columns)
            self.assertTrue(log_jsonl.exists())

    def test_command_e2e_check_success(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            report_json = temp_dir / "e2e_report.json"
            output_csv = temp_dir / "e2e_runs.csv"
            log_jsonl = temp_dir / "e2e_runs.jsonl"

            class Args:
                pass

            args = Args()
            args.dataset_csv = None
            args.benchmark_csv = None
            args.model_cache_path = None
            args.max_samples = None
            args.force_refit = False
            args.seed = 42
            args.z_dim = 64
            args.zs_model_path = str(temp_dir / "decoder.npz")
            args.base_url = "http://127.0.0.1:11434/v1"
            args.model_name = "gpt-oss:20b"
            args.response_temperature = 0.5
            args.max_tokens = 300
            args.timeout_sec = 30
            args.prompt_template = None
            args.log_jsonl = str(log_jsonl)
            args.text = "지금 너무 예민하고 피곤해."
            args.report_json = str(report_json)
            args.output_csv = str(output_csv)

            with patch("emonet.cli.ensure_model_server_ready"), patch(
                "emonet.cli.build_model", return_value=self.FakeGenerativeModel()
            ), patch("emonet.cli.LinearZtoSDecoder.load", return_value=self.FakeDecoder()), patch(
                "emonet.cli.call_openai_compatible_chat", return_value="조금 쉬면서 호흡을 가다듬어 보세요."
            ):
                command_e2e_check(args)

            report = json.loads(report_json.read_text(encoding="utf-8"))
            self.assertEqual(report["overall_status"], "passed")
            self.assertEqual(report["stage_status"]["text_to_z"], "passed")
            self.assertEqual(report["stage_status"]["z_to_s_pred"], "passed")
            self.assertEqual(report["stage_status"]["s_pred_text_to_llm_response"], "passed")
            self.assertEqual(report["stage_status"]["artifact_logging"], "passed")
            self.assertEqual(report["result"]["llm_response"], "조금 쉬면서 호흡을 가다듬어 보세요.")
            self.assertIn("expression_cues_text", report["result"])

            saved = pd.read_csv(output_csv)
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved.loc[0, "overall_status"], "passed")
            self.assertEqual(saved.loc[0, "stage4_status"], "passed")
            self.assertIn("expression_cues_text", saved.columns)
            self.assertTrue(log_jsonl.exists())
            self.assertEqual(len(log_jsonl.read_text(encoding="utf-8").strip().splitlines()), 1)

    def test_command_e2e_check_records_llm_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            report_json = temp_dir / "e2e_report.json"
            output_csv = temp_dir / "e2e_runs.csv"
            log_jsonl = temp_dir / "e2e_runs.jsonl"

            class Args:
                pass

            args = Args()
            args.dataset_csv = None
            args.benchmark_csv = None
            args.model_cache_path = None
            args.max_samples = None
            args.force_refit = False
            args.seed = 42
            args.z_dim = 64
            args.zs_model_path = str(temp_dir / "decoder.npz")
            args.base_url = "http://127.0.0.1:11434/v1"
            args.model_name = "gpt-oss:20b"
            args.response_temperature = 0.5
            args.max_tokens = 300
            args.timeout_sec = 30
            args.prompt_template = None
            args.log_jsonl = str(log_jsonl)
            args.text = "지금 너무 예민하고 피곤해."
            args.report_json = str(report_json)
            args.output_csv = str(output_csv)

            with patch("emonet.cli.build_model", return_value=self.FakeGenerativeModel()), patch(
                "emonet.cli.LinearZtoSDecoder.load", return_value=self.FakeDecoder()
            ), patch(
                "emonet.cli.ensure_model_server_ready",
                side_effect=ConnectionError("model server is not reachable at http://127.0.0.1:11434/v1"),
            ):
                command_e2e_check(args)

            report = json.loads(report_json.read_text(encoding="utf-8"))
            self.assertEqual(report["overall_status"], "failed")
            self.assertEqual(report["stage_status"]["text_to_z"], "passed")
            self.assertEqual(report["stage_status"]["z_to_s_pred"], "passed")
            self.assertEqual(report["stage_status"]["s_pred_text_to_llm_response"], "failed")
            self.assertEqual(report["stage_status"]["artifact_logging"], "passed")
            self.assertEqual(report["failure"]["stage_id"], "s_pred_text_to_llm_response")
            self.assertEqual(report["failure"]["category"], "llm_server_unreachable")

            saved = pd.read_csv(output_csv)
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved.loc[0, "overall_status"], "failed")
            self.assertEqual(saved.loc[0, "failure_category"], "llm_server_unreachable")
            self.assertTrue(log_jsonl.exists())

    def test_train_and_predict_zs_regressor(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            model_path = temp_dir / "z_to_s_decoder.npz"
            input_csv = temp_dir / "zs_train.csv"
            predict_input_csv = temp_dir / "zs_predict.csv"
            predict_output_csv = temp_dir / "zs_predict_out.csv"

            rng = np.random.default_rng(7)
            rows = []
            for idx in range(24):
                z = rng.random(64, dtype=np.float32)
                s = []
                for axis_idx in range(16):
                    signal = 0.65 * float(z[axis_idx]) + 0.20 * float(z[(axis_idx + 7) % 64]) + 0.05
                    s.append(float(np.clip(signal, 0.0, 1.0)))
                row = {
                    "sample_id": f"s_{idx:06d}",
                    "text": f"synthetic text {idx}",
                    "talk_id": f"t{idx}",
                    "keep_sample": idx % 5 != 0,
                }
                row.update({f"z_{j}": float(z[j]) for j in range(64)})
                row.update({f"s_{j}": float(s[j]) for j in range(16)})
                rows.append(row)

            train_df = pd.DataFrame(rows)
            train_df.to_csv(input_csv, index=False, encoding="utf-8-sig")

            summary = train_zs_decoder_from_dataframe(
                df=train_df,
                model_path=model_path,
                z_dim=64,
                s_dim=16,
                ridge_alpha=1.0,
                seed=7,
                val_ratio=0.2,
                use_all_rows=False,
            )
            self.assertTrue(model_path.exists())
            self.assertEqual(summary["input_rows"], 24)
            self.assertGreater(summary["rows_used"], 10)
            self.assertIsNotNone(summary["val_mae"])

            decoder = LinearZtoSDecoder.load(model_path)
            pred = decoder.predict(train_df.loc[0, [f"z_{j}" for j in range(64)]].to_numpy(dtype=np.float32))
            self.assertEqual(pred.shape, (16,))

            train_df[[f"z_{j}" for j in range(64)]].head(3).to_csv(predict_input_csv, index=False, encoding="utf-8-sig")

            class Args:
                pass

            args = Args()
            args.input_csv = str(predict_input_csv)
            args.output_csv = str(predict_output_csv)
            args.model_path = str(model_path)
            args.z_dim = 64
            args.output_prefix = "s_pred_"
            command_predict_s(args)

            predicted_df = pd.read_csv(predict_output_csv)
            self.assertEqual(len(predicted_df), 3)
            self.assertIn("s_pred_0", predicted_df.columns)
            self.assertIn("s_pred_15", predicted_df.columns)


if __name__ == "__main__":
    unittest.main()
