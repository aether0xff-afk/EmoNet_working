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
    def test_analyze_branch_traces_outputs_report_and_figures(self) -> None:
        module = load_module("analyze_branch_traces_module", "scripts/analyze_branch_traces.py")
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            summary_csv = temp_dir / "summary.csv"
            details_csv = temp_dir / "details.csv"
            tick_csv = temp_dir / "tick_details.csv"
            output_dir = temp_dir / "trace_report"

            pd.DataFrame(
                [
                    {
                        "config_name": "baseline",
                        "balanced_score": 30.0,
                        "constraint_penalty": 1.2,
                        "constraint_failures": "hit_max_ticks_ratio>0.8",
                        "len1_ratio": 0.0,
                        "hit_max_ticks_ratio": 1.0,
                        "mean_first_active_tick": 20.0,
                        "late_ignition_ratio_ge_15": 1.0,
                        "mean_branch_len": 30.0,
                    },
                    {
                        "config_name": "candidate_a",
                        "balanced_score": 45.0,
                        "constraint_penalty": 0.7,
                        "constraint_failures": "mean_first_active_tick>20.0",
                        "len1_ratio": 0.1,
                        "hit_max_ticks_ratio": 0.7,
                        "mean_first_active_tick": 18.0,
                        "late_ignition_ratio_ge_15": 0.6,
                        "mean_branch_len": 24.0,
                    },
                    {
                        "config_name": "candidate_b",
                        "balanced_score": 44.0,
                        "constraint_penalty": 0.5,
                        "constraint_failures": "",
                        "len1_ratio": 0.2,
                        "hit_max_ticks_ratio": 0.5,
                        "mean_first_active_tick": 8.0,
                        "late_ignition_ratio_ge_15": 0.2,
                        "mean_branch_len": 18.0,
                    },
                ]
            ).to_csv(summary_csv, index=False, encoding="utf-8-sig")

            pd.DataFrame(
                [
                    {"sample_index": 1, "text": "sample one", "dominant_branch_len": 30, "first_active_tick": 20, "last_active_tick": 31, "active_window_ticks": 12, "mean_active_nodes": 40.0, "mean_edges_fired": 100.0, "max_active_nodes": 60, "max_edges_fired": 120, "config_name": "baseline"},
                    {"sample_index": 2, "text": "sample two", "dominant_branch_len": 10, "first_active_tick": 25, "last_active_tick": 31, "active_window_ticks": 7, "mean_active_nodes": 35.0, "mean_edges_fired": 80.0, "max_active_nodes": 50, "max_edges_fired": 95, "config_name": "baseline"},
                    {"sample_index": 1, "text": "sample one", "dominant_branch_len": 24, "first_active_tick": 18, "last_active_tick": 31, "active_window_ticks": 14, "mean_active_nodes": 32.0, "mean_edges_fired": 70.0, "max_active_nodes": 48, "max_edges_fired": 88, "config_name": "candidate_a"},
                    {"sample_index": 2, "text": "sample two", "dominant_branch_len": 16, "first_active_tick": 10, "last_active_tick": 25, "active_window_ticks": 16, "mean_active_nodes": 28.0, "mean_edges_fired": 60.0, "max_active_nodes": 42, "max_edges_fired": 76, "config_name": "candidate_a"},
                    {"sample_index": 1, "text": "sample one", "dominant_branch_len": 18, "first_active_tick": 8, "last_active_tick": 22, "active_window_ticks": 15, "mean_active_nodes": 22.0, "mean_edges_fired": 40.0, "max_active_nodes": 35, "max_edges_fired": 55, "config_name": "candidate_b"},
                    {"sample_index": 2, "text": "sample two", "dominant_branch_len": 14, "first_active_tick": 6, "last_active_tick": 18, "active_window_ticks": 13, "mean_active_nodes": 20.0, "mean_edges_fired": 36.0, "max_active_nodes": 30, "max_edges_fired": 46, "config_name": "candidate_b"},
                ]
            ).to_csv(details_csv, index=False, encoding="utf-8-sig")

            tick_rows = []
            for config_name, sample_index, active_values, edge_values in [
                ("baseline", 1, [0, 0, 0, 10, 20, 30], [0, 0, 0, 20, 40, 60]),
                ("baseline", 2, [0, 0, 0, 0, 10, 12], [0, 0, 0, 0, 16, 20]),
                ("candidate_a", 1, [0, 0, 8, 12, 20, 24], [0, 0, 10, 18, 30, 36]),
                ("candidate_a", 2, [0, 6, 10, 14, 18, 22], [0, 8, 12, 18, 24, 28]),
                ("candidate_b", 1, [4, 8, 12, 14, 12, 10], [6, 10, 14, 16, 14, 12]),
                ("candidate_b", 2, [3, 6, 9, 12, 10, 8], [4, 8, 12, 14, 12, 10]),
            ]:
                for tick, (active_nodes, edges_fired) in enumerate(zip(active_values, edge_values, strict=True)):
                    tick_rows.append(
                        {
                            "sample_index": sample_index,
                            "tick": tick,
                            "active_nodes": active_nodes,
                            "edges_fired": edges_fired,
                            "has_activity": int(active_nodes > 0),
                            "config_name": config_name,
                        }
                    )
            pd.DataFrame(tick_rows).to_csv(tick_csv, index=False, encoding="utf-8-sig")

            import sys

            original_argv = sys.argv[:]
            try:
                sys.argv = [
                    "analyze_branch_traces.py",
                    "--summary-csv",
                    str(summary_csv),
                    "--details-csv",
                    str(details_csv),
                    "--tick-csv",
                    str(tick_csv),
                    "--output-dir",
                    str(output_dir),
                    "--top-k-candidates",
                    "2",
                    "--top-k-samples",
                    "3",
                    "--progress-every",
                    "0",
                ]
                module.main()
            finally:
                sys.argv = original_argv

            self.assertTrue((output_dir / "config_comparison.csv").exists())
            self.assertTrue((output_dir / "representative_samples.csv").exists())
            self.assertTrue((output_dir / "pairwise_deltas.csv").exists())
            self.assertTrue((output_dir / "config_mean_active_nodes.svg").exists())
            self.assertTrue((output_dir / "TRACE_ANALYSIS_REPORT.md").exists())

            comparison = pd.read_csv(output_dir / "config_comparison.csv")
            self.assertIn("p50_activity_tick", comparison.columns)
            self.assertIn("baseline", set(comparison["config_name"].tolist()))

    def test_optimize_branch_dynamics_outputs_ranked_artifacts(self) -> None:
        module = load_module("optimize_branch_dynamics_module", "scripts/optimize_branch_dynamics.py")
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            input_csv = temp_dir / "probe_input.csv"
            dataset_csv = temp_dir / "dataset_for_regression.csv"
            benchmark_csv = temp_dir / "benchmark_results.csv"
            model_cache = temp_dir / "stim_encoder.joblib"
            output_dir = temp_dir / "branch_optimize"

            pd.DataFrame(
                [
                    {"text": "요즘 너무 예민하고 피곤하다."},
                    {"text": "일이 너무 많아서 버겁다."},
                    {"text": "무시당한 기분이라 화가 난다."},
                ]
            ).to_csv(input_csv, index=False, encoding="utf-8-sig")

            pd.DataFrame(
                [
                    {"text": "urgent critical alert now", "label": "E10", "y": 0.05, "talk_id": "t1", "persona_id": "p1"},
                    {"text": "i feel calm and safe with support", "label": "E20", "y": 0.90, "talk_id": "t2", "persona_id": "p2"},
                    {"text": "too tired and burned out i need rest", "label": "E30", "y": 0.20, "talk_id": "t3", "persona_id": "p3"},
                    {"text": "this risk is scary and stressful", "label": "E40", "y": 0.10, "talk_id": "t4", "persona_id": "p4"},
                ]
            ).to_csv(dataset_csv, index=False, encoding="utf-8-sig")

            pd.DataFrame(
                [{"vector": "char_tfidf", "model": "Ridge", "status": "ok", "MAE(mean)": 0.1, "RMSE(mean)": 0.2}]
            ).to_csv(benchmark_csv, index=False, encoding="utf-8-sig")

            import sys

            original_argv = sys.argv[:]
            try:
                sys.argv = [
                    "optimize_branch_dynamics.py",
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
                    "1",
                    "--sample-mode",
                    "head",
                    "--progress-every",
                    "0",
                    "--search-mode",
                    "random",
                    "--budget",
                    "1",
                    "--include-baseline",
                    "--fixed",
                    "max_ticks=8",
                    "--space",
                    "k_threshold_base=0.72,0.95",
                    "--output-dir",
                    str(output_dir),
                ]
                module.main()
            finally:
                sys.argv = original_argv

            summary = pd.read_csv(output_dir / "summary.csv")
            details = pd.read_csv(output_dir / "details.csv")
            tick_details = pd.read_csv(output_dir / "tick_details.csv")

            self.assertIn("balanced_score", summary.columns)
            self.assertIn("is_pareto_front", summary.columns)
            self.assertTrue((output_dir / "best_config.json").exists())
            self.assertTrue((output_dir / "optimizer_balanced_score.svg").exists())
            self.assertTrue((output_dir / "optimizer_len1_vs_hitmax.svg").exists())
            self.assertTrue((output_dir / "BRANCH_OPTIMIZATION_REPORT.md").exists())
            self.assertGreaterEqual(len(summary), 2)
            self.assertFalse(details.empty)
            self.assertFalse(tick_details.empty)

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
                    "max_ticks=40",
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
                    "max_ticks=40",
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
            "anti_softening_rules": ["같은 문장을 반복하지 않는다."],
            "grounding_rules": ["첫 문장에서 감정을 짧게 짚고 바로 답한다."],
        }

        prompt, sections = module.build_condition_prompt("emonet_no_summary", "예시 입력", profile)
        self.assertIn("[STYLE_TAGS]", prompt)
        self.assertIn("[STYLE_VECTOR]", prompt)
        self.assertIn("[ANTI_SOFTENING_RULES]", prompt)
        self.assertIn("[GROUNDING_RULES]", prompt)
        self.assertNotIn("[STYLE_SUMMARY]", prompt)
        self.assertEqual(sections, "style_tags,expression_cues,style_vector,anti_softening_rules,grounding_rules")

        prompt, sections = module.build_condition_prompt("emonet_vector_only", "예시 입력", profile)
        self.assertIn("[STYLE_VECTOR]", prompt)
        self.assertIn("[ANTI_SOFTENING_RULES]", prompt)
        self.assertIn("[GROUNDING_RULES]", prompt)
        self.assertNotIn("[STYLE_TAGS]", prompt)
        self.assertNotIn("[STYLE_SUMMARY]", prompt)
        self.assertNotIn("[EXPRESSION_CUES]", prompt)
        self.assertEqual(sections, "style_vector,anti_softening_rules,grounding_rules")

    def test_experiment_matrix_records_response_retry_metadata(self) -> None:
        module = load_module("experiment_matrix_retry_module", "scripts/experiment_matrix.py")
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            input_csv = temp_dir / "matrix_input.csv"
            output_csv = temp_dir / "matrix_output.csv"
            summary_json = temp_dir / "matrix_summary.json"
            log_jsonl = temp_dir / "matrix_log.jsonl"

            pd.DataFrame([{"sample_id": "s1", "talk_id": "t1", "text": "지금 너무 예민하고 피곤해."}]).to_csv(
                input_csv, index=False, encoding="utf-8-sig"
            )

            profile = {
                "stim_vec": [0.2, 0.4, 0.6, 0.8],
                "dominant_branch_len": 5,
                "z": [0.1] * 64,
                "s_pred": [0.2] * 32,
                "style_dict": {"softness": 0.4, "directness": 0.6},
                "style_tags": ["건조함", "직설적"],
                "style_summary": {"warmth": 0.3, "tension": 0.7},
                "style_summary_text": "긴장 높음, 따뜻함 낮음",
                "expression_cues_text": "표정 변화=0.6",
                "anti_softening_mode": "strict",
                "anti_softening_rules": ["같은 문장을 반복하지 않는다."],
                "grounding_mode": "grounded",
                "grounding_rules": ["첫 문장에서 감정을 짧게 짚고 바로 답한다."],
            }

            import sys

            original_argv = sys.argv[:]
            try:
                sys.argv = [
                    "experiment_matrix.py",
                    "--input-csv",
                    str(input_csv),
                    "--output-csv",
                    str(output_csv),
                    "--summary-json",
                    str(summary_json),
                    "--log-jsonl",
                    str(log_jsonl),
                    "--conditions",
                    "direct,emonet_full",
                    "--progress-every",
                    "0",
                    "--flush-every",
                    "1",
                    "--zs-model-path",
                    str(temp_dir / "decoder.npz"),
                    "--response-max-retries",
                    "2",
                ]
                with unittest.mock.patch.object(module, "ensure_model_server_ready"), unittest.mock.patch.object(
                    module, "build_model", return_value=object()
                ), unittest.mock.patch.object(module.LinearZtoSDecoder, "load", return_value=object()), unittest.mock.patch.object(
                    module, "infer_style_profile", return_value=profile
                ), unittest.mock.patch.object(
                    module,
                    "request_plain_text_response",
                    side_effect=[
                        ("직접 응답이다.", "직접 응답이다.", {"retry_count": 0, "validation_errors": []}),
                        (
                            "지금은 건드리지 말고 잠깐 쉬어.",
                            "지금은 건드리지 말고 잠깐 쉬어.",
                            {"retry_count": 2, "validation_errors": ["repeat", "hanging"]},
                        ),
                    ],
                ):
                    module.main()
            finally:
                sys.argv = original_argv

            saved = pd.read_csv(output_csv)
            self.assertEqual(len(saved), 2)
            self.assertIn("response_retry_count", saved.columns)
            self.assertIn("response_validation_errors_json", saved.columns)
            self.assertIn("grounding_mode", saved.columns)
            self.assertIn("grounding_rules_json", saved.columns)
            emonet_row = saved[saved["condition"] == "emonet_full"].iloc[0]
            self.assertEqual(int(emonet_row["response_retry_count"]), 2)
            self.assertIn("hanging", str(emonet_row["response_validation_errors_json"]))
            self.assertEqual(str(emonet_row["grounding_mode"]), "grounded")

    def test_score_experiment_matrix_summary_includes_retry_mean(self) -> None:
        module = load_module("score_experiment_matrix_module", "scripts/score_experiment_matrix.py")
        scored = pd.DataFrame(
            [
                {
                    "condition": "direct",
                    "condition_group": "baseline",
                    "status": "ok",
                    "response_length": 10,
                    "response_retry_count": 0,
                    "content_fit": 4,
                    "emotional_appropriateness": 4,
                    "style_match": 3,
                    "naturalness": 4,
                    "overall_quality": 4,
                },
                {
                    "condition": "direct",
                    "condition_group": "baseline",
                    "status": "ok",
                    "response_length": 12,
                    "response_retry_count": 2,
                    "content_fit": 5,
                    "emotional_appropriateness": 4,
                    "style_match": 4,
                    "naturalness": 5,
                    "overall_quality": 4,
                },
            ]
        )
        summary = module.summarize_scores(scored)
        self.assertIn("mean_response_retry_count", summary.columns)
        self.assertAlmostEqual(float(summary.loc[0, "mean_response_retry_count"]), 1.0)

    def test_score_experiment_matrix_compact_fallback(self) -> None:
        module = load_module("score_experiment_matrix_compact_module", "scripts/score_experiment_matrix.py")
        row = {
            "text": "지금 너무 예민하고 피곤해.",
            "llm_response": "지금은 건드리지 말고 잠깐 쉬어.",
            "condition": "emonet_full",
            "style_summary_text": "긴장 높음, 따뜻함 낮음",
            "style_tags_json": "[\"건조함\", \"직설적\"]",
        }
        with unittest.mock.patch.object(
            module, "request_json_response", side_effect=ValueError("json failed")
        ), unittest.mock.patch.object(
            module,
            "request_plain_text_response",
            return_value=("4,4,3,4,4", "4,4,3,4,4", {"attempt_count": 2, "retry_count": 1, "validation_errors": ["empty"]}),
        ):
            payload, raw, parse_mode = module.request_score_payload(
                row,
                base_url="http://127.0.0.1:11434/v1",
                model_name="gpt-oss:20b",
                timeout_sec=30,
                max_tokens=300,
                temperature=0.0,
                max_retries=2,
            )
        self.assertEqual(parse_mode, "compact")
        self.assertEqual(raw, "4,4,3,4,4")
        self.assertEqual(
            payload,
            {
                "content_fit": 4,
                "emotional_appropriateness": 4,
                "style_match": 3,
                "naturalness": 4,
                "overall_quality": 4,
                },
            )

    def test_score_experiment_matrix_compact_failure_surfaces_both_errors(self) -> None:
        module = load_module("score_experiment_matrix_compact_error_module", "scripts/score_experiment_matrix.py")
        row = {
            "text": "입력",
            "llm_response": "응답",
            "condition": "direct",
        }
        with unittest.mock.patch.object(
            module, "request_json_response", side_effect=ValueError("json failed")
        ), unittest.mock.patch.object(
            module, "request_plain_text_response", side_effect=ValueError("compact failed")
        ):
            with self.assertRaisesRegex(
                ValueError,
                "json_error=json failed; compact_error=compact failed; minimal_error=compact failed",
            ):
                module.request_score_payload(
                    row,
                    base_url="http://127.0.0.1:11434/v1",
                    model_name="gpt-oss:20b",
                    timeout_sec=30,
                    max_tokens=300,
                    temperature=0.0,
                    max_retries=2,
                )

    def test_experiment_matrix_resume_skips_only_ok_rows(self) -> None:
        module = load_module("experiment_matrix_resume_module", "scripts/experiment_matrix.py")
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            output_csv = temp_dir / "matrix_output.csv"
            pd.DataFrame(
                [
                    {"record_id": "r1", "condition": "direct", "status": "ok"},
                    {"record_id": "r2", "condition": "direct", "status": "error"},
                ]
            ).to_csv(output_csv, index=False, encoding="utf-8-sig")

            keys = module.load_existing_keys(output_csv)
            self.assertEqual(keys, {("r1", "direct")})

    def test_score_experiment_matrix_resume_skips_only_ok_rows(self) -> None:
        module = load_module("score_experiment_matrix_resume_module", "scripts/score_experiment_matrix.py")
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            output_csv = temp_dir / "scored_output.csv"
            pd.DataFrame(
                [
                    {"record_id": "r1", "condition": "direct", "status": "ok"},
                    {"record_id": "r2", "condition": "direct", "status": "error"},
                ]
            ).to_csv(output_csv, index=False, encoding="utf-8-sig")

            keys = module.load_existing_keys(output_csv)
            self.assertEqual(keys, {("r1", "direct")})

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
