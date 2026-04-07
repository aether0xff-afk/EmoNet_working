import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_module(module_name: str, relative_path: str):
    file_path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class ExperimentToolTests(unittest.TestCase):
    def test_branch_param_sweep_ofat_outputs_ranked_summary(self) -> None:
        module = load_module("branch_param_sweep_ofat_module", "scripts/branch_param_sweep.py")
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            input_csv = temp_dir / "probe_input.csv"
            dataset_csv = temp_dir / "dataset_for_regression.csv"
            benchmark_csv = temp_dir / "benchmark_results.csv"
            model_cache = temp_dir / "stim_encoder.joblib"
            output_dir = temp_dir / "branch_sweep_ofat"

            pd.DataFrame(
                [
                    {"text": "요즘 너무 예민하고 피곤하다."},
                    {"text": "일이 많아서 버겁고 짜증난다."},
                    {"text": "무시당한 것 같아 화가 난다."},
                ]
            ).to_csv(input_csv, index=False, encoding="utf-8-sig")

            pd.DataFrame(
                [
                    {"text": "urgent critical alert now", "label": "E10", "y": 0.05, "talk_id": "t1", "persona_id": "p1"},
                    {"text": "i feel calm and safe with support", "label": "E20", "y": 0.90, "talk_id": "t2", "persona_id": "p2"},
                    {"text": "too tired and burned out i need rest", "label": "E30", "y": 0.20, "talk_id": "t3", "persona_id": "p3"},
                    {"text": "we made progress and i can handle this", "label": "E40", "y": 0.85, "talk_id": "t4", "persona_id": "p4"},
                    {"text": "this risk is scary and stressful", "label": "E50", "y": 0.10, "talk_id": "t5", "persona_id": "p5"},
                    {"text": "quiet stable day and peaceful mood", "label": "E60", "y": 0.95, "talk_id": "t6", "persona_id": "p6"},
                ]
            ).to_csv(dataset_csv, index=False, encoding="utf-8-sig")
            pd.DataFrame(
                [{"vector": "char_tfidf", "model": "Ridge", "status": "ok", "MAE(mean)": 0.1, "RMSE(mean)": 0.2}]
            ).to_csv(benchmark_csv, index=False, encoding="utf-8-sig")

            import sys

            original_argv = sys.argv[:]
            try:
                sys.argv = [
                    "branch_param_sweep.py",
                    "--input-csv",
                    str(input_csv),
                    "--dataset-csv",
                    str(dataset_csv),
                    "--benchmark-csv",
                    str(benchmark_csv),
                    "--model-cache-path",
                    str(model_cache),
                    "--force-refit",
                    "--sample-size",
                    "3",
                    "--sample-mode",
                    "head",
                    "--progress-every",
                    "0",
                    "--ofat-params",
                    "k_threshold_base,memory_k_mix",
                    "--fixed",
                    "branch_length_bonus=0.35",
                    "--output-dir",
                    str(output_dir),
                ]
                module.main()
            finally:
                sys.argv = original_argv

            summary = pd.read_csv(output_dir / "summary.csv")
            self.assertIn("score", summary.columns)
            self.assertIn("baseline", set(summary["config_name"].tolist()))
            self.assertTrue(any(name.startswith("k_threshold_base:down=") for name in summary["config_name"].tolist()))
            self.assertTrue(any(name.startswith("memory_k_mix:up=") for name in summary["config_name"].tolist()))

    def test_branch_param_sweep_outputs_csv_and_svg(self) -> None:
        module = load_module("branch_param_sweep_module", "scripts/branch_param_sweep.py")
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            input_csv = temp_dir / "probe_input.csv"
            dataset_csv = temp_dir / "dataset_for_regression.csv"
            benchmark_csv = temp_dir / "benchmark_results.csv"
            model_cache = temp_dir / "stim_encoder.joblib"
            output_dir = temp_dir / "branch_sweep"

            pd.DataFrame(
                [
                    {"text": "일이 너무 많아서 버겁다."},
                    {"text": "요즘 너무 예민하고 피곤하다."},
                    {"text": "화가 나는데 참고 있다."},
                ]
            ).to_csv(input_csv, index=False, encoding="utf-8-sig")

            pd.DataFrame(
                [
                    {"text": "urgent critical alert now", "label": "E10", "y": 0.05, "talk_id": "t1", "persona_id": "p1"},
                    {"text": "i feel calm and safe with support", "label": "E20", "y": 0.90, "talk_id": "t2", "persona_id": "p2"},
                    {"text": "too tired and burned out i need rest", "label": "E30", "y": 0.20, "talk_id": "t3", "persona_id": "p3"},
                    {"text": "we made progress and i can handle this", "label": "E40", "y": 0.85, "talk_id": "t4", "persona_id": "p4"},
                    {"text": "this risk is scary and stressful", "label": "E50", "y": 0.10, "talk_id": "t5", "persona_id": "p5"},
                    {"text": "quiet stable day and peaceful mood", "label": "E60", "y": 0.95, "talk_id": "t6", "persona_id": "p6"},
                ]
            ).to_csv(dataset_csv, index=False, encoding="utf-8-sig")

            pd.DataFrame(
                [{"vector": "char_tfidf", "model": "Ridge", "status": "ok", "MAE(mean)": 0.1, "RMSE(mean)": 0.2}]
            ).to_csv(benchmark_csv, index=False, encoding="utf-8-sig")

            import sys

            original_argv = sys.argv[:]
            try:
                sys.argv = [
                    "branch_param_sweep.py",
                    "--input-csv",
                    str(input_csv),
                    "--dataset-csv",
                    str(dataset_csv),
                    "--benchmark-csv",
                    str(benchmark_csv),
                    "--model-cache-path",
                    str(model_cache),
                    "--force-refit",
                    "--sample-size",
                    "3",
                    "--sample-mode",
                    "head",
                    "--progress-every",
                    "0",
                    "--sweep-param",
                    "k_threshold_base",
                    "--values",
                    "0.68,0.72",
                    "--fixed",
                    "branch_length_bonus=0.35",
                    "--output-dir",
                    str(output_dir),
                ]
                module.main()
            finally:
                sys.argv = original_argv

            summary_csv = output_dir / "summary.csv"
            details_csv = output_dir / "details.csv"
            spec_json = output_dir / "specs.json"
            mean_svg = output_dir / "branch_sweep_mean.svg"
            len1_svg = output_dir / "branch_sweep_len1_ratio.svg"
            bucket_svg = output_dir / "branch_sweep_bucket_ratio.svg"

            self.assertTrue(summary_csv.exists())
            self.assertTrue(details_csv.exists())
            self.assertTrue(spec_json.exists())
            self.assertTrue(mean_svg.exists())
            self.assertTrue(len1_svg.exists())
            self.assertTrue(bucket_svg.exists())

            summary = pd.read_csv(summary_csv)
            details = pd.read_csv(details_csv)
            self.assertEqual(len(summary), 2)
            self.assertEqual(set(summary["config_name"].tolist()), {"k_threshold_base=0.68", "k_threshold_base=0.72"})
            self.assertEqual(len(details), 6)
            self.assertIn("dominant_branch_len", details.columns)

    def test_shard_and_merge_label_csv(self) -> None:
        module = load_module("shard_label_csv_module", "scripts/shard_label_csv.py")
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            input_csv = temp_dir / "subset.csv"
            shard_dir = temp_dir / "shards"
            merged_csv = temp_dir / "merged.csv"

            df = pd.DataFrame(
                [
                    {"sample_id": "s_000003", "text": "입력 3", "label": "E10"},
                    {"sample_id": "s_000001", "text": "입력 1", "label": "E20"},
                    {"sample_id": "s_000004", "text": "입력 4", "label": "E30"},
                    {"sample_id": "s_000002", "text": "입력 2", "label": "E40"},
                    {"sample_id": "s_000005", "text": "입력 5", "label": "E50"},
                ]
            )
            df.to_csv(input_csv, index=False, encoding="utf-8-sig")

            import sys

            original_argv = sys.argv[:]
            try:
                sys.argv = [
                    "shard_label_csv.py",
                    "split",
                    "--input-csv",
                    str(input_csv),
                    "--output-dir",
                    str(shard_dir),
                    "--num-shards",
                    "3",
                    "--prefix",
                    "extended40_subset",
                ]
                module.main()

                sys.argv = [
                    "shard_label_csv.py",
                    "merge",
                    "--input-dir",
                    str(shard_dir),
                    "--pattern",
                    "extended40_subset.shard*.csv",
                    "--output-csv",
                    str(merged_csv),
                ]
                module.main()
            finally:
                sys.argv = original_argv

            shard_files = sorted(shard_dir.glob("extended40_subset.shard*.csv"))
            self.assertEqual(len(shard_files), 3)

            merged = pd.read_csv(merged_csv)
            self.assertEqual(len(merged), 5)
            self.assertEqual(
                merged["sample_id"].tolist(),
                ["s_000001", "s_000002", "s_000003", "s_000004", "s_000005"],
            )

    def test_condition_prompt_variants_hide_requested_sections(self) -> None:
        module = load_module("experiment_matrix_module", "scripts/experiment_matrix.py")
        profile = {
            "stim_vec": [0.1, 0.2, 0.3, 0.4],
            "style_dict": {"warmth": 0.8, "softness": 0.7, "directness": 0.4},
            "style_tags": ["부드러움", "따뜻함"],
            "style_summary": {"warmth": 0.7, "tension": 0.2},
        }

        prompt, sections = module.build_condition_prompt("emonet_no_summary", "예시 입력", profile)
        self.assertIn("[STYLE_TAGS]", prompt)
        self.assertIn("[STYLE_VECTOR]", prompt)
        self.assertNotIn("[STYLE_SUMMARY]", prompt)
        self.assertEqual(sections, "style_tags,expression_cues,style_vector")

        prompt, sections = module.build_condition_prompt("emonet_vector_only", "예시 입력", profile)
        self.assertIn("[STYLE_VECTOR]", prompt)
        self.assertNotIn("[STYLE_TAGS]", prompt)
        self.assertNotIn("[STYLE_SUMMARY]", prompt)
        self.assertNotIn("[EXPRESSION_CUES]", prompt)
        self.assertEqual(sections, "style_vector")

    def test_prepare_human_eval_blinds_condition_order(self) -> None:
        module = load_module("prepare_human_eval_module", "scripts/prepare_human_eval.py")
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            input_csv = temp_dir / "matrix.csv"
            output_csv = temp_dir / "eval.csv"
            answer_key_json = temp_dir / "answer_key.json"
            instructions_md = temp_dir / "instructions.md"

            df = pd.DataFrame(
                [
                    {
                        "record_id": "r1",
                        "text": "입력 1",
                        "condition": "direct",
                        "condition_group": "baseline",
                        "status": "ok",
                        "llm_response": "응답 A",
                    },
                    {
                        "record_id": "r1",
                        "text": "입력 1",
                        "condition": "emonet_full",
                        "condition_group": "emonet",
                        "status": "ok",
                        "llm_response": "응답 B",
                    },
                    {
                        "record_id": "r2",
                        "text": "입력 2",
                        "condition": "direct",
                        "condition_group": "baseline",
                        "status": "ok",
                        "llm_response": "응답 C",
                    },
                    {
                        "record_id": "r2",
                        "text": "입력 2",
                        "condition": "emonet_full",
                        "condition_group": "emonet",
                        "status": "ok",
                        "llm_response": "응답 D",
                    },
                ]
            )
            df.to_csv(input_csv, index=False, encoding="utf-8-sig")

            import sys

            original_argv = sys.argv[:]
            try:
                sys.argv = [
                    "prepare_human_eval.py",
                    "--input-csv",
                    str(input_csv),
                    "--output-csv",
                    str(output_csv),
                    "--answer-key-json",
                    str(answer_key_json),
                    "--instructions-md",
                    str(instructions_md),
                    "--conditions",
                    "direct,emonet_full",
                    "--seed",
                    "7",
                ]
                module.main()
            finally:
                sys.argv = original_argv

            eval_df = pd.read_csv(output_csv)
            self.assertEqual(len(eval_df), 2)
            self.assertIn("candidate_a", eval_df.columns)
            self.assertIn("candidate_b", eval_df.columns)

            answer_key = json.loads(answer_key_json.read_text(encoding="utf-8"))
            self.assertEqual(answer_key["conditions"], ["direct", "emonet_full"])
            self.assertEqual(len(answer_key["rows"]), 2)
            self.assertTrue(instructions_md.exists())


if __name__ == "__main__":
    unittest.main()
