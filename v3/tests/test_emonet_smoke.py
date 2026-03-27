import csv
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd

from emonet import EmoNet, EmoNetConfig, LinearZtoSDecoder, StimEncoderConfig
from emonet.cli import (
    STYLE_AXIS_NAMES,
    build_balanced_subset,
    command_e2e_check,
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
            self.assertNotIn("s_16", saved.columns)

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
        self.assertIn("긴장높음", tags)
        self.assertIn("건조함", tags)
        self.assertIn("tension", summary)
        self.assertIn("warmth", summary)

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
            self.assertTrue(log_jsonl.exists())

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

            saved = pd.read_csv(output_csv)
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved.loc[0, "overall_status"], "passed")
            self.assertEqual(saved.loc[0, "stage4_status"], "passed")
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
