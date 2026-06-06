import importlib.util
import json
import tempfile
import unittest
import unittest.mock
from pathlib import Path
from unittest.mock import patch

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
    def test_interpret_emotion_trajectory_outputs_episode_artifacts(self) -> None:
        module = load_module("interpret_emotion_trajectory_module", "scripts/interpret_emotion_trajectory.py")
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            input_dir = temp_dir / "trajectory_batch"
            sample_dir = input_dir / "s_1"
            output_dir = temp_dir / "episode_interp"
            sample_dir.mkdir(parents=True, exist_ok=True)

            (sample_dir / "emotion_trace_summary.json").write_text(
                json.dumps(
                    {
                        "input_text": "??쒓? ?섎쭔 怨듦컻?곸쑝濡?臾댁떆?댁꽌 ?붽? ?쒕떎.",
                        "input_meta": {"sample_id": "s_1"},
                        "ticks_run": 12,
                        "termination_reason": "stable_convergence",
                        "dominant_branch_len": 10,
                        "active_tick_count": 10,
                        "persistence_ratio": 0.83,
                        "saturation_ratio": 0.72,
                        "dominant_global_signal": "怨듭꽭??湲댁옣",
                        "drive": 0.22,
                        "brake": 0.18,
                        "alarm": 0.61,
                        "fatigue": 0.14,
                        "inhibitory_ratio": 0.42,
                        "excitatory_ratio": 0.48,
                        "modulatory_ratio": 0.10,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            (sample_dir / "emotion_trajectory_summary.json").write_text(
                json.dumps(
                    {
                        "input_text": "??쒓? ?섎쭔 怨듦컻?곸쑝濡?臾댁떆?댁꽌 ?붽? ?쒕떎.",
                        "input_meta": {"sample_id": "s_1"},
                        "trajectory_pattern": "high_arousal_persistence",
                        "phase_count": 3,
                        "phase_sequence": ["dormant", "ignition", "persistence"],
                        "peak_alarm_tick": 9,
                        "peak_fatigue_tick": 3,
                        "peak_conflict_tick": 0,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            pd.DataFrame(
                [
                    {"phase_index": 0, "phase": "dormant", "start_tick": 0, "end_tick": 1, "duration": 2, "mean_active_nodes": 0.0, "mean_edges_fired": 0.0, "dominant_signal": "異붾룞/?묎렐", "signal_conflict": 1.0},
                    {"phase_index": 1, "phase": "ignition", "start_tick": 2, "end_tick": 4, "duration": 3, "mean_active_nodes": 120.0, "mean_edges_fired": 580.0, "dominant_signal": "寃쎄퀎/?좎뭅濡쒖?", "signal_conflict": 0.94},
                    {"phase_index": 2, "phase": "persistence", "start_tick": 5, "end_tick": 11, "duration": 7, "mean_active_nodes": 210.0, "mean_edges_fired": 940.0, "dominant_signal": "寃쎄퀎/?좎뭅濡쒖?", "signal_conflict": 0.72},
                ]
            ).to_csv(sample_dir / "trajectory_phases.csv", index=False, encoding="utf-8-sig")

            pd.DataFrame(
                [
                    {"tick": 0, "active_nodes": 0, "edges_fired": 0, "dominant_signal": "嫄곗쓽 臾댁깋", "combined_drive": 0.0, "combined_brake": 0.0, "combined_alarm": 0.0, "combined_fatigue": 0.0, "inhibitory_ratio": 0.0, "excitatory_ratio": 0.0, "modulatory_ratio": 0.0, "phase": "dormant"},
                    {"tick": 2, "active_nodes": 88, "edges_fired": 430, "dominant_signal": "怨듭꽭??湲댁옣", "combined_drive": 0.26, "combined_brake": 0.16, "combined_alarm": 0.52, "combined_fatigue": 0.12, "inhibitory_ratio": 0.10, "excitatory_ratio": 0.75, "modulatory_ratio": 0.15, "phase": "ignition"},
                    {"tick": 9, "active_nodes": 230, "edges_fired": 1012, "dominant_signal": "怨듭꽭??湲댁옣", "combined_drive": 0.27, "combined_brake": 0.20, "combined_alarm": 0.64, "combined_fatigue": 0.18, "inhibitory_ratio": 0.43, "excitatory_ratio": 0.47, "modulatory_ratio": 0.10, "phase": "persistence"},
                    {"tick": 11, "active_nodes": 214, "edges_fired": 922, "dominant_signal": "怨듭꽭??湲댁옣", "combined_drive": 0.24, "combined_brake": 0.18, "combined_alarm": 0.58, "combined_fatigue": 0.16, "inhibitory_ratio": 0.42, "excitatory_ratio": 0.48, "modulatory_ratio": 0.10, "phase": "persistence"},
                ]
            ).to_csv(sample_dir / "trajectory_ticks.csv", index=False, encoding="utf-8-sig")

            pd.DataFrame(
                [
                    {"node_id": 145, "neuron_type": "excitatory", "bias_label": "怨듭꽭??湲댁옣", "activity_ticks": 10, "k_mean": 420.0, "stim_drive": 0.03, "stim_brake": 0.01, "stim_alarm": 0.92, "stim_fatigue": 0.02},
                    {"node_id": 201, "neuron_type": "inhibitory", "bias_label": "?꾩땐/?듭젣", "activity_ticks": 10, "k_mean": 380.0, "stim_drive": 0.18, "stim_brake": 0.25, "stim_alarm": 0.39, "stim_fatigue": 0.21},
                ]
            ).to_csv(sample_dir / "top_nodes.csv", index=False, encoding="utf-8-sig")

            fake_payload = {
                "episode_label": "諛곗젣 ?먭레???섑빐 ?좎??섎뒗 諛⑹뼱??遺꾨끂",
                "stimulus_reading": "怨듦컻??臾댁떆瑜??ы쉶??諛곗젣? 遺덇났???ш굔?쇰줈 泥섎━??episode??",
                "appraisal": {
                    "primary_appraisal": "遺덇났?뺢낵 諛곗젣",
                    "secondary_appraisal": "?듭젣 怨ㅻ?",
                    "target": "other",
                    "control_state": "low",
                    "social_orientation": "defend",
                },
                "trajectory": {
                    "overall_pattern": "怨좉컖??寃쎄퀎媛 ?먰솕 ??湲멸쾶 ?좎??쒕떎.",
                    "ignition": "珥덇린 ?먰솕??alarm ?곗꽭??怨듦꺽??湲댁옣?먯꽌 ?쒖옉?쒕떎.",
                    "persistence": "以묐컲 ?댄썑?먮룄 寃쎄퀎? 諛⑹뼱媛 ?믪? ?섏??쇰줈 吏?띾맂??",
                    "resolution": "紐낇솗???댁냼 ?놁씠 諛⑹뼱??湲댁옣?쇰줈 ?섎졃?쒕떎.",
                },
                "action_tendency": "利됯컖 諛섍꺽?섍굅???뺣㈃ ??묓븯怨??띠?留?諛⑺뼢? ?꾩쭅 ?뺣━?섏? ?딆? ?곹깭",
                "rawness": {
                    "valence": "negative",
                    "arousal": "high",
                    "softened_output_risk": "high",
                    "should_preserve_harshness": True,
                },
                "response_guidance": {
                    "preserve": "?듭슱?④낵 ?좎뭅濡쒖슫 寃쎄퀎",
                    "avoid": "do not soften into generic counseling",
                    "tone_hint": "direct but not overheated",
                },
                "evidence": [
                    "phase_sequence媛 dormant -> ignition -> persistence濡??댁뼱吏怨?persistence媛 湲몃떎.",
                    "dominant_global_signal??怨듭꽭??湲댁옣?닿퀬 alarm ?됯퇏??0.61濡??믩떎.",
                ],
                "confidence": 0.86,
            }

            import sys

            original_argv = sys.argv[:]
            try:
                sys.argv = [
                    "interpret_emotion_trajectory.py",
                    "--input-dir",
                    str(input_dir),
                    "--output-dir",
                    str(output_dir),
                    "--progress-every",
                    "0",
                ]
                with unittest.mock.patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}, clear=False), unittest.mock.patch.object(
                    module, "ensure_model_server_ready"
                ), unittest.mock.patch.object(
                    module,
                    "request_json_response",
                    return_value=(fake_payload, json.dumps(fake_payload, ensure_ascii=False)),
                ):
                    module.main()
            finally:
                sys.argv = original_argv

            summary_df = pd.read_csv(output_dir / "episode_summary.csv")
            self.assertEqual(len(summary_df), 1)
            self.assertTrue((output_dir / "episode_payloads.jsonl").exists())
            self.assertTrue((output_dir / "EPISODE_INTERPRETATION_REPORT.md").exists())
            self.assertTrue((output_dir / "s_1" / "episode_transcript.txt").exists())
            self.assertTrue((output_dir / "s_1" / "episode_interpretation.json").exists())
            self.assertEqual(summary_df.loc[0, "episode_label"], fake_payload["episode_label"])
            self.assertEqual(summary_df.loc[0, "target"], "other")

    def test_analyze_emotion_trajectory_batch_outputs_aggregate_artifacts(self) -> None:
        module = load_module("analyze_emotion_trajectory_batch_module", "scripts/analyze_emotion_trajectory_batch.py")
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            dataset_csv = temp_dir / "dataset_for_regression.csv"
            benchmark_csv = temp_dir / "benchmark_results.csv"
            model_cache = temp_dir / "stim_encoder.joblib"
            input_csv = temp_dir / "samples.csv"
            output_dir = temp_dir / "trajectory_batch"

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

            pd.DataFrame(
                [
                    {"sample_id": "s_1", "text": "吏湲??덈Т ?덈??섍퀬 ?쇨낀?섎떎."},
                    {"sample_id": "s_2", "text": "???쇱㎏ ?쇨렐?대씪 ?뺣쭚 吏移쒕떎."},
                ]
            ).to_csv(input_csv, index=False, encoding="utf-8-sig")

            import sys

            original_argv = sys.argv[:]
            try:
                sys.argv = [
                    "analyze_emotion_trajectory_batch.py",
                    "--input-csv",
                    str(input_csv),
                    "--record-id-column",
                    "sample_id",
                    "--record-ids",
                    "s_1,s_2",
                    "--dataset-csv",
                    str(dataset_csv),
                    "--benchmark-csv",
                    str(benchmark_csv),
                    "--model-cache-path",
                    str(model_cache),
                    "--force-refit",
                    "--max-ticks",
                    "8",
                    "--progress-every",
                    "0",
                    "--save-per-sample",
                    "--output-dir",
                    str(output_dir),
                ]
                module.main()
            finally:
                sys.argv = original_argv

            sample_summary = pd.read_csv(output_dir / "sample_summary.csv")
            phase_summary = pd.read_csv(output_dir / "phase_summary.csv")

            self.assertEqual(len(sample_summary), 2)
            self.assertTrue((output_dir / "figures" / "top_emotion_scores.svg").exists())
            self.assertTrue((output_dir / "figures" / "alarm_fatigue_pressure.svg").exists())
            self.assertTrue((output_dir / "figures" / "trajectory_pattern_counts.svg").exists())
            self.assertTrue((output_dir / "BATCH_TRAJECTORY_REPORT.md").exists())
            self.assertIn("trajectory_pattern", sample_summary.columns)
            self.assertIn("top_emotion", sample_summary.columns)
            self.assertIn("phase", phase_summary.columns)
            self.assertTrue((output_dir / "s_1" / "emotion_trajectory_summary.json").exists())
            self.assertTrue((output_dir / "s_2" / "raw_trace.json").exists())
            self.assertTrue((output_dir / "selected_record_ids.txt").exists())
            self.assertTrue((output_dir / "selected_records.csv").exists())

    def test_analyze_emotion_trajectory_batch_supports_random_sampling(self) -> None:
        module = load_module("analyze_emotion_trajectory_batch_sampling_module", "scripts/analyze_emotion_trajectory_batch.py")
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            dataset_csv = temp_dir / "dataset_for_regression.csv"
            benchmark_csv = temp_dir / "benchmark_results.csv"
            model_cache = temp_dir / "stim_encoder.joblib"
            input_csv = temp_dir / "samples.csv"
            output_dir = temp_dir / "trajectory_batch_random"

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

            pd.DataFrame(
                [
                    {"sample_id": "s_1", "text": "吏湲??덈Т ?덈??섍퀬 ?쇨낀?섎떎."},
                    {"sample_id": "s_2", "text": "???쇱㎏ ?쇨렐?대씪 ?뺣쭚 吏移쒕떎."},
                    {"sample_id": "s_3", "text": "?댁씪 諛쒗몴??遺덉븞?섎떎."},
                ]
            ).to_csv(input_csv, index=False, encoding="utf-8-sig")

            import sys

            original_argv = sys.argv[:]
            try:
                sys.argv = [
                    "analyze_emotion_trajectory_batch.py",
                    "--input-csv",
                    str(input_csv),
                    "--record-id-column",
                    "sample_id",
                    "--sample-size",
                    "2",
                    "--sample-mode",
                    "random",
                    "--sample-seed",
                    "7",
                    "--dataset-csv",
                    str(dataset_csv),
                    "--benchmark-csv",
                    str(benchmark_csv),
                    "--model-cache-path",
                    str(model_cache),
                    "--force-refit",
                    "--max-ticks",
                    "8",
                    "--progress-every",
                    "0",
                    "--output-dir",
                    str(output_dir),
                ]
                module.main()
            finally:
                sys.argv = original_argv

            selected_ids = [line.strip() for line in (output_dir / "selected_record_ids.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
            self.assertEqual(len(selected_ids), 2)
            sample_summary = pd.read_csv(output_dir / "sample_summary.csv")
            self.assertEqual(len(sample_summary), 2)

    def test_inspect_emotion_trace_outputs_raw_trace_artifacts(self) -> None:
        module = load_module("inspect_emotion_trace_module", "scripts/inspect_emotion_trace.py")
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            dataset_csv = temp_dir / "dataset_for_regression.csv"
            benchmark_csv = temp_dir / "benchmark_results.csv"
            model_cache = temp_dir / "stim_encoder.joblib"
            output_dir = temp_dir / "emotion_trace"

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
                    "inspect_emotion_trace.py",
                    "--text",
                    "吏湲??덈Т ?덈??섍퀬 ?쇨낀?섎떎.",
                    "--dataset-csv",
                    str(dataset_csv),
                    "--benchmark-csv",
                    str(benchmark_csv),
                    "--model-cache-path",
                    str(model_cache),
                    "--force-refit",
                    "--max-ticks",
                    "8",
                    "--output-dir",
                    str(output_dir),
                ]
                module.main()
            finally:
                sys.argv = original_argv

            summary = json.loads((output_dir / "emotion_trace_summary.json").read_text(encoding="utf-8"))
            candidates = pd.read_csv(output_dir / "emotion_candidates.csv")
            tick_summary = pd.read_csv(output_dir / "tick_summary.csv")
            trajectory_summary = json.loads((output_dir / "emotion_trajectory_summary.json").read_text(encoding="utf-8"))
            trajectory_ticks = pd.read_csv(output_dir / "trajectory_ticks.csv")
            trajectory_phases = pd.read_csv(output_dir / "trajectory_phases.csv")

            self.assertTrue((output_dir / "raw_trace.json").exists())
            self.assertTrue((output_dir / "node_catalog.csv").exists())
            self.assertTrue((output_dir / "node_trace.csv").exists())
            self.assertTrue((output_dir / "top_nodes.csv").exists())
            self.assertTrue((output_dir / "trajectory_ticks.csv").exists())
            self.assertTrue((output_dir / "trajectory_phases.csv").exists())
            self.assertTrue((output_dir / "emotion_trajectory_summary.json").exists())
            self.assertTrue((output_dir / "figures" / "tick_activity.svg").exists())
            self.assertTrue((output_dir / "figures" / "raw_signal_curves.svg").exists())
            self.assertTrue((output_dir / "figures" / "emotion_candidates.svg").exists())
            self.assertTrue((output_dir / "figures" / "trajectory_activity.svg").exists())
            self.assertTrue((output_dir / "figures" / "trajectory_signals.svg").exists())
            self.assertTrue((output_dir / "figures" / "trajectory_emotions.svg").exists())
            self.assertIn("top_emotions", summary)
            self.assertIn("dominant_global_signal", summary)
            self.assertIn("trajectory_pattern", trajectory_summary)
            self.assertIn("emotion", candidates.columns)
            self.assertIn("score", candidates.columns)
            self.assertIn("tick", tick_summary.columns)
            self.assertIn("combined_alarm", tick_summary.columns)
            self.assertIn("phase", trajectory_ticks.columns)
            self.assertIn("top_emotion", trajectory_ticks.columns)
            self.assertIn("phase", trajectory_phases.columns)

    def test_inspect_emotion_trace_serializes_csv_row_metadata(self) -> None:
        module = load_module("inspect_emotion_trace_module", "scripts/inspect_emotion_trace.py")
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            dataset_csv = temp_dir / "dataset_for_regression.csv"
            benchmark_csv = temp_dir / "benchmark_results.csv"
            model_cache = temp_dir / "stim_encoder.joblib"
            input_csv = temp_dir / "samples.csv"
            output_dir = temp_dir / "emotion_trace_csv"

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

            pd.DataFrame(
                [
                    {"sample_id": "s_1", "text": "吏湲??덈Т ?덈??섍퀬 ?쇨낀?섎떎.", "talk_id": 123, "priority": 2},
                ]
            ).to_csv(input_csv, index=False, encoding="utf-8-sig")

            import sys

            original_argv = sys.argv[:]
            try:
                sys.argv = [
                    "inspect_emotion_trace.py",
                    "--input-csv",
                    str(input_csv),
                    "--record-id-column",
                    "sample_id",
                    "--record-id",
                    "s_1",
                    "--dataset-csv",
                    str(dataset_csv),
                    "--benchmark-csv",
                    str(benchmark_csv),
                    "--model-cache-path",
                    str(model_cache),
                    "--force-refit",
                    "--max-ticks",
                    "8",
                    "--output-dir",
                    str(output_dir),
                ]
                module.main()
            finally:
                sys.argv = original_argv

            payload = json.loads((output_dir / "raw_trace.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["input_meta"]["sample_id"], "s_1")
            self.assertEqual(payload["input_meta"]["talk_id"], 123)

    def test_calibrate_reference_config_outputs_evidence_and_recommendation(self) -> None:
        module = load_module("calibrate_reference_config_module", "scripts/calibrate_reference_config.py")
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            input_csv = temp_dir / "probe_input.csv"
            dataset_csv = temp_dir / "dataset_for_regression.csv"
            benchmark_csv = temp_dir / "benchmark_results.csv"
            model_cache = temp_dir / "stim_encoder.joblib"
            output_dir = temp_dir / "branch_calibration"

            pd.DataFrame(
                [
                    {"text": "?붿쬁 ?덈Т ?덈??섍퀬 ?쇨낀?섎떎."},
                    {"text": "?쇱씠 ?덈Т 留롮븘??踰꾧쾪??"},
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
                    "calibrate_reference_config.py",
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
                    "--calibrate-params",
                    "k_threshold_base,intrinsic_alignment_gain",
                    "--space",
                    "k_threshold_base=0.72,0.90",
                    "--space",
                    "intrinsic_alignment_gain=0.20,0.24",
                    "--fixed",
                    "max_ticks=8",
                    "--output-dir",
                    str(output_dir),
                ]
                module.main()
            finally:
                sys.argv = original_argv

            evidence = pd.read_csv(output_dir / "parameter_evidence.csv")
            recommendations = pd.read_csv(output_dir / "parameter_recommendations.csv")

            self.assertTrue((output_dir / "center_config.json").exists())
            self.assertTrue((output_dir / "calibrated_reference_config.json").exists())
            self.assertTrue((output_dir / "combined_validation.json").exists())
            self.assertTrue((output_dir / "CALIBRATION_REPORT.md").exists())
            self.assertTrue((output_dir / "figures" / "k_threshold_base_calibration.svg").exists())
            self.assertIn("no_activity_ratio", evidence.columns)
            self.assertIn("is_recommended", evidence.columns)
            self.assertEqual(set(recommendations["parameter_name"].tolist()), {"intrinsic_alignment_gain", "k_threshold_base"})

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
                    {"text": "?붿쬁 ?덈Т ?덈??섍퀬 ?쇨낀?섎떎."},
                    {"text": "?쇱씠 ?덈Т 留롮븘??踰꾧쾪??"},
                    {"text": "臾댁떆?뱁븳 湲곕텇?대씪 ?붽? ?쒕떎."},
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
                    {"text": "?붿쬁 ?덈Т ?덈??섍퀬 ?쇨낀?섎떎."},
                    {"text": "?쇱씠 留롮븘??踰꾧쾪怨?吏쒖쬆?쒕떎."},
                    {"text": "臾댁떆?뱁븳 寃?媛숈븘 ?붽? ?쒕떎."},
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
                    {"text": "?쇱씠 ?덈Т 留롮븘??踰꾧쾪??"},
                    {"text": "?붿쬁 ?덈Т ?덈??섍퀬 ?쇨낀?섎떎."},
                    {"text": "?붽? ?섎뒗??李멸퀬 ?덈떎."},
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
                    {"sample_id": "s_000003", "text": "?낅젰 3", "label": "E10"},
                    {"sample_id": "s_000001", "text": "?낅젰 1", "label": "E20"},
                    {"sample_id": "s_000004", "text": "?낅젰 4", "label": "E30"},
                    {"sample_id": "s_000002", "text": "?낅젰 2", "label": "E40"},
                    {"sample_id": "s_000005", "text": "?낅젰 5", "label": "E50"},
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
            "style_tags": ["warm", "direct"],
            "style_summary": {"warmth": 0.7, "tension": 0.2},
            "anti_softening_rules": ["媛숈? 臾몄옣??諛섎났?섏? ?딅뒗??"],
            "grounding_rules": ["泥?臾몄옣?먯꽌 媛먯젙??吏㏐쾶 吏싰퀬 諛붾줈 ?듯븳??"],
        }

        prompt, sections = module.build_condition_prompt("emonet_no_summary", "?덉떆 ?낅젰", profile)
        self.assertIn("[STYLE_TAGS]", prompt)
        self.assertIn("[STYLE_VECTOR]", prompt)
        self.assertIn("[ANTI_SOFTENING_RULES]", prompt)
        self.assertIn("[GROUNDING_RULES]", prompt)
        self.assertNotIn("[STYLE_SUMMARY]", prompt)
        self.assertEqual(sections, "style_tags,expression_cues,style_vector,anti_softening_rules,grounding_rules")

        prompt, sections = module.build_condition_prompt("emonet_vector_only", "?덉떆 ?낅젰", profile)
        self.assertIn("[STYLE_VECTOR]", prompt)
        self.assertIn("[ANTI_SOFTENING_RULES]", prompt)
        self.assertIn("[GROUNDING_RULES]", prompt)
        self.assertNotIn("[STYLE_TAGS]", prompt)
        self.assertNotIn("[STYLE_SUMMARY]", prompt)
        self.assertNotIn("[EXPRESSION_CUES]", prompt)
        self.assertEqual(sections, "style_vector,anti_softening_rules,grounding_rules")

    def test_condition_prompt_episode_variants_use_episode_trace(self) -> None:
        module = load_module("experiment_matrix_episode_module", "scripts/experiment_matrix.py")
        profile = {
            "stim_vec": [0.1, 0.2, 0.3, 0.4],
            "style_dict": {"warmth": 0.8, "softness": 0.7, "directness": 0.4},
            "style_tags": ["warm", "direct"],
            "style_summary": {"warmth": 0.7, "tension": 0.2},
            "anti_softening_rules": ["媛숈? 臾몄옣??諛섎났?섏? ?딅뒗??"],
            "grounding_rules": ["泥?臾몄옣?먯꽌 媛먯젙??吏㏐쾶 吏싰퀬 諛붾줈 ?듯븳??"],
        }
        episode_payload = {
            "episode_label": "怨듭꽭??寃쎄퀎-遺꾨끂 ?쇰?",
            "stimulus_reading": "怨듦컻??諛곗젣 ?ш굔",
            "appraisal": {
                "primary_appraisal": "exclusion and humiliation",
                "secondary_appraisal": "low control",
                "target": "other",
                "control_state": "mixed",
                "social_orientation": "defend",
            },
            "trajectory": {
                "overall_pattern": "persistent defense",
                "ignition": "quick ignition",
                "persistence": "long persistence",
                "resolution": "unresolved",
            },
            "action_tendency": "?곗?嫄곕굹 ??쓽?섍퀬 ?띠쓬",
            "rawness": {
                "valence": "negative",
                "arousal": "high",
                "softened_output_risk": "high",
                "should_preserve_harshness": True,
            },
            "response_guidance": {
                "preserve": "諛곗젣媛먭낵 ?좎뭅濡쒖?",
                "avoid": "媛踰쇱슫 ?쒖슫?⑥쑝濡?異뺤냼",
                "tone_hint": "cold and diagnostic",
            },
            "evidence": ["e1", "e2"],
            "confidence": 0.91,
        }
        profile = module.augment_profile_with_episode(profile, episode_payload, episode_source_path="episodes/s_1.json")

        prompt, sections = module.build_condition_prompt("episode_trace", "?덉떆 ?낅젰", profile)
        self.assertIn("[EPISODE_TRACE]", prompt)
        self.assertIn("怨듭꽭??寃쎄퀎-遺꾨끂 ?쇰?", prompt)
        self.assertIn("surface_keep: 諛곗젣媛먭낵 ?좎뭅濡쒖?", prompt)
        self.assertNotIn("appraisal_state:", prompt)
        self.assertNotIn("trajectory_overall:", prompt)
        self.assertNotIn("[STYLE_TAGS]", prompt)
        self.assertEqual(sections, "episode_trace,anti_softening_rules,grounding_rules")

        prompt, sections = module.build_condition_prompt("hybrid_episode", "?덉떆 ?낅젰", profile)
        self.assertIn("[EPISODE_TRACE]", prompt)
        self.assertIn("[STYLE_TAGS]", prompt)
        self.assertIn("[STYLE_SUMMARY]", prompt)
        self.assertIn("surface_keep: 諛곗젣媛먭낵 ?좎뭅濡쒖?", prompt)
        self.assertNotIn("trajectory_persistence:", prompt)
        self.assertEqual(sections, "episode_trace,style_tags,style_summary,anti_softening_rules,grounding_rules")

        prompt, sections = module.build_condition_prompt("episode_trace_v3", "?덉떆 ?낅젰", profile)
        self.assertIn("[INTERNAL_RESPONSE_PRIORITIES]", prompt)
        self.assertIn("keep_in_surface: 諛곗젣媛먭낵 ?좎뭅濡쒖?", prompt)
        self.assertIn("avoid_in_surface: 媛踰쇱슫 ?쒖슫?⑥쑝濡?異뺤냼", prompt)
        self.assertNotIn("episode_label:", prompt)
        self.assertNotIn("appraisal_state:", prompt)
        self.assertNotIn("trajectory_persistence:", prompt)
        self.assertEqual(sections, "episode_trace_v3,anti_softening_rules,grounding_rules")

        with self.assertRaisesRegex(ValueError, "episode_trace_v3 conditioning requires episode_payload"):
            module.build_condition_prompt("episode_trace_v3", "?덉떆 ?낅젰", {key: value for key, value in profile.items() if key != "episode_payload"})

    def test_experiment_matrix_records_response_retry_metadata(self) -> None:
        module = load_module("experiment_matrix_retry_module", "scripts/experiment_matrix.py")
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            input_csv = temp_dir / "matrix_input.csv"
            output_csv = temp_dir / "matrix_output.csv"
            summary_json = temp_dir / "matrix_summary.json"
            log_jsonl = temp_dir / "matrix_log.jsonl"

            pd.DataFrame([{"sample_id": "s1", "talk_id": "t1", "text": "吏湲??덈Т ?덈??섍퀬 ?쇨낀??"}]).to_csv(
                input_csv, index=False, encoding="utf-8-sig"
            )

            profile = {
                "stim_vec": [0.2, 0.4, 0.6, 0.8],
                "dominant_branch_len": 5,
                "z": [0.1] * 64,
                "s_pred": [0.2] * 32,
                "style_dict": {"softness": 0.4, "directness": 0.6},
                "style_tags": ["dry", "direct"],
                "style_summary": {"warmth": 0.3, "tension": 0.7},
                "style_summary_text": "湲댁옣 ?믪쓬, ?곕쑜????쓬",
                "expression_cues_text": "?쒖젙 蹂??0.6",
                "anti_softening_mode": "strict",
                "anti_softening_rules": ["媛숈? 臾몄옣??諛섎났?섏? ?딅뒗??"],
                "grounding_mode": "grounded",
                "grounding_rules": ["泥?臾몄옣?먯꽌 媛먯젙??吏㏐쾶 吏싰퀬 諛붾줈 ?듯븳??"],
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
                        ("吏곸젒 ?묐떟?대떎.", "吏곸젒 ?묐떟?대떎.", {"retry_count": 0, "validation_errors": []}),
                        (
                            "吏湲덉? 嫄대뱶由ъ? 留먭퀬 ?좉퉸 ?ъ뼱.",
                            "吏湲덉? 嫄대뱶由ъ? 留먭퀬 ?좉퉸 ?ъ뼱.",
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

    def test_experiment_matrix_episode_condition_loads_episode_payload(self) -> None:
        module = load_module("experiment_matrix_episode_run_module", "scripts/experiment_matrix.py")
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            input_csv = temp_dir / "matrix_input.csv"
            output_csv = temp_dir / "matrix_output.csv"
            summary_json = temp_dir / "matrix_summary.json"
            episode_dir = temp_dir / "episodes"
            episode_sample_dir = episode_dir / "s1"
            episode_sample_dir.mkdir(parents=True, exist_ok=True)

            pd.DataFrame([{"sample_id": "s1", "talk_id": "t1", "text": "??쒓? ?섎쭔 怨듦컻?곸쑝濡?臾댁떆?댁꽌 ?붽? ?쒕떎."}]).to_csv(
                input_csv, index=False, encoding="utf-8-sig"
            )
            (episode_sample_dir / "episode_interpretation.json").write_text(
                json.dumps(
                    {
                        "episode_label": "諛곗젣 ?먭레???섑빐 ?좎??섎뒗 諛⑹뼱??遺꾨끂",
                        "stimulus_reading": "怨듦컻??諛곗젣 ?ш굔",
                        "appraisal": {
                            "primary_appraisal": "遺덇났?뺢낵 諛곗젣",
                            "secondary_appraisal": "low control",
                            "target": "other",
                            "control_state": "low",
                            "social_orientation": "defend",
                        },
                        "trajectory": {
                            "overall_pattern": "怨좉컖??寃쎄퀎媛 ?먰솕 ??湲멸쾶 ?좎??쒕떎.",
                            "ignition": "珥덇린 ?먰솕??鍮좊Ⅴ??",
                            "persistence": "以묐컲 ?댄썑?먮룄 寃쎄퀎媛 吏?띾맂??",
                            "resolution": "諛⑹뼱??湲댁옣?쇰줈 ?섎졃?쒕떎.",
                        },
                        "action_tendency": "?뺣㈃ ??묓븯怨??띠쓬",
                        "rawness": {
                            "valence": "negative",
                            "arousal": "high",
                            "softened_output_risk": "high",
                            "should_preserve_harshness": True,
                        },
                        "response_guidance": {
                            "preserve": "?듭슱?④낵 ?좎뭅濡쒖슫 寃쎄퀎",
                            "avoid": "媛踰쇱슫 ?쒖슫?⑥쑝濡?異뺤냼",
                            "tone_hint": "direct and cold",
                        },
                        "evidence": ["e1", "e2"],
                        "confidence": 0.9,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            profile = {
                "stim_vec": [0.2, 0.4, 0.6, 0.8],
                "dominant_branch_len": 5,
                "z": [0.1] * 64,
                "s_pred": [0.2] * 32,
                "style_dict": {"softness": 0.4, "directness": 0.6},
                "style_tags": ["dry", "direct"],
                "style_summary": {"warmth": 0.3, "tension": 0.7},
                "style_summary_text": "湲댁옣 ?믪쓬, ?곕쑜????쓬",
                "expression_cues_text": "?쒖젙 蹂??0.6",
                "anti_softening_mode": "strict",
                "anti_softening_rules": ["媛숈? 臾몄옣??諛섎났?섏? ?딅뒗??"],
                "grounding_mode": "grounded",
                "grounding_rules": ["泥?臾몄옣?먯꽌 媛먯젙??吏㏐쾶 吏싰퀬 諛붾줈 ?듯븳??"],
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
                    "--conditions",
                    "episode_trace",
                    "--episode-dir",
                    str(episode_dir),
                    "--zs-model-path",
                    str(temp_dir / "decoder.npz"),
                    "--progress-every",
                    "0",
                    "--flush-every",
                    "1",
                ]
                with unittest.mock.patch.object(module, "ensure_model_server_ready"), unittest.mock.patch.object(
                    module, "build_model", return_value=object()
                ), unittest.mock.patch.object(module.LinearZtoSDecoder, "load", return_value=object()), unittest.mock.patch.object(
                    module, "infer_style_profile", return_value=profile
                ), unittest.mock.patch.object(
                    module,
                    "request_plain_text_response",
                    return_value=("洹몃옒, 洹멸굔 ??볤퀬 臾댁떆?뱁븳 ?먮굦?대씪 ???대컺吏.", "raw", {"retry_count": 0, "validation_errors": []}),
                ):
                    module.main()
            finally:
                sys.argv = original_argv

            saved = pd.read_csv(output_csv)
            self.assertEqual(len(saved), 1)
            self.assertEqual(saved.loc[0, "condition"], "episode_trace")
            self.assertEqual(saved.loc[0, "episode_label"], "諛곗젣 ?먭레???섑빐 ?좎??섎뒗 諛⑹뼱??遺꾨끂")
            self.assertIn("?듭슱?④낵 ?좎뭅濡쒖슫 寃쎄퀎", str(saved.loc[0, "prompt"]))
            self.assertTrue(summary_json.exists())

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

    def test_score_experiment_matrix_json_failure_is_not_rewritten_as_compact_success(self) -> None:
        module = load_module("score_experiment_matrix_json_error_module", "scripts/score_experiment_matrix.py")
        row = {
            "text": "input",
            "llm_response": "response",
            "condition": "direct",
        }
        with patch.object(module, "request_json_response", side_effect=ValueError("json failed")):
            with self.assertRaisesRegex(ValueError, "json failed"):
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
                        "text": "?낅젰 1",
                        "condition": "direct",
                        "condition_group": "baseline",
                        "status": "ok",
                        "llm_response": "?묐떟 A",
                    },
                    {
                        "record_id": "r1",
                        "text": "?낅젰 1",
                        "condition": "emonet_full",
                        "condition_group": "emonet",
                        "status": "ok",
                        "llm_response": "?묐떟 B",
                    },
                    {
                        "record_id": "r2",
                        "text": "?낅젰 2",
                        "condition": "direct",
                        "condition_group": "baseline",
                        "status": "ok",
                        "llm_response": "?묐떟 C",
                    },
                    {
                        "record_id": "r2",
                        "text": "?낅젰 2",
                        "condition": "emonet_full",
                        "condition_group": "emonet",
                        "status": "ok",
                        "llm_response": "?묐떟 D",
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

    def test_build_targeted_superiority_set_outputs_required_columns(self) -> None:
        module = load_module("build_targeted_superiority_set_module", "scripts/build_targeted_superiority_set.py")
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            episode_csv = temp_dir / "episode_summary.csv"
            scored_csv = temp_dir / "scored.csv"
            paired_csv = temp_dir / "paired.csv"
            output_csv = temp_dir / "targeted.csv"
            manifest_json = temp_dir / "manifest.json"

            rows = []
            for idx in range(10):
                rows.append(
                    {
                        "sample_id": f"s_{idx}",
                        "episode_label": "二꾩콉媛?湲곕컲 留뚰쉶" if idx == 5 else "諛곗젣 ?꾪삊",
                        "valence": "negative",
                        "arousal": "high",
                        "target": "other" if idx < 4 else "mixed",
                        "control_state": "low",
                        "social_orientation": "mixed" if idx in {4, 5, 6} else "defend",
                        "preserve": "?좎뭅濡쒖?",
                        "avoid": "?쇰컲 ?꾨줈",
                        "action_tendency": "?ш낵? 留뚰쉶" if idx == 5 else "諛⑹뼱",
                    }
                )
            pd.DataFrame(rows).to_csv(episode_csv, index=False, encoding="utf-8-sig")
            pd.DataFrame([{"record_id": f"s_{idx}", "text": f"?낅젰 {idx}"} for idx in range(10)]).to_csv(
                scored_csv, index=False, encoding="utf-8-sig"
            )
            pd.DataFrame(
                [
                    {"condition": "episode_trace", "record_id": "s_9", "delta_mean_total": -2.0},
                    {"condition": "episode_trace", "record_id": "s_8", "delta_mean_total": -1.5},
                ]
            ).to_csv(paired_csv, index=False, encoding="utf-8-sig")

            manifest = module.build_targeted_set(
                episode_summary_csv=episode_csv,
                scored_csv=scored_csv,
                paired_examples_csv=paired_csv,
                output_csv=output_csv,
                manifest_json=manifest_json,
                target_size=8,
                seed=7,
            )

            self.assertTrue(output_csv.exists())
            self.assertTrue(manifest_json.exists())
            saved = pd.read_csv(output_csv)
            self.assertGreaterEqual(len(saved), 8)
            self.assertIn("record_id", saved.columns)
            self.assertIn("selection_bucket", saved.columns)
            self.assertIn("preserve", saved.columns)
            self.assertEqual(manifest["rows"], len(saved))

    def test_superiority_judge_normalization_and_prompt(self) -> None:
        module = load_module("score_superiority_judge_module", "scripts/score_superiority_judge.py")
        row = {
            "text": "??쒓? ?섎쭔 鍮쇨퀬 而ㅽ뵾瑜??뚮젮???붽? ??",
            "condition": "episode_trace_v3",
            "llm_response": "洹멸굔 洹몃깷 ?쒖슫???뺣룄媛 ?꾨땲????볤퀬 諛곗젣?뱁븳 ?먮굦?대씪 ?붽? ??留뚰빐.",
            "episode_label": "諛곗젣 ?먭레???섑빐 ?좎??섎뒗 諛⑹뼱??遺꾨끂",
            "valence": "negative",
            "arousal": "high",
            "target": "other",
            "control_state": "low",
            "social_orientation": "defend",
            "preserve": "?듭슱?④낵 ?좎뭅濡쒖슫 寃쎄퀎",
            "avoid": "媛踰쇱슫 ?쒖슫?⑥쑝濡?異뺤냼",
            "action_tendency": "臾몄젣?쒓린 異⑸룞",
        }
        prompt = module.build_judge_prompt(row)
        self.assertIn("appraisal_fidelity", prompt)
        self.assertIn("raw_affect_preservation", prompt)
        self.assertIn("anti_softening", prompt)
        payload = module.normalize_scores(
            {
                "scores": {
                    "appraisal_fidelity": 4,
                    "raw_affect_preservation": 5,
                    "anti_softening": 4,
                    "action_tendency_fit": 3,
                    "emotional_specificity": 5,
                    "naturalness": 4,
                    "overall_preference": 4,
                }
            }
        )
        self.assertEqual(payload["raw_affect_preservation"], 5)

    def test_paired_superiority_accepts_custom_score_keys(self) -> None:
        module = load_module("paired_superiority_custom_keys_module", "scripts/analyze_paired_superiority.py")
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            scored_csv = temp_dir / "superiority_scored.csv"
            output_dir = temp_dir / "paired"
            pd.DataFrame(
                [
                    {"record_id": "r1", "condition": "stim_only", "appraisal_fidelity": 2, "raw_affect_preservation": 2, "anti_softening": 2, "action_tendency_fit": 2, "emotional_specificity": 2},
                    {"record_id": "r1", "condition": "episode_trace_v3", "appraisal_fidelity": 4, "raw_affect_preservation": 4, "anti_softening": 4, "action_tendency_fit": 4, "emotional_specificity": 4},
                    {"record_id": "r2", "condition": "stim_only", "appraisal_fidelity": 3, "raw_affect_preservation": 3, "anti_softening": 3, "action_tendency_fit": 3, "emotional_specificity": 3},
                    {"record_id": "r2", "condition": "episode_trace_v3", "appraisal_fidelity": 2, "raw_affect_preservation": 2, "anti_softening": 2, "action_tendency_fit": 2, "emotional_specificity": 2},
                ]
            ).to_csv(scored_csv, index=False, encoding="utf-8-sig")

            import sys

            original_argv = sys.argv[:]
            try:
                sys.argv = [
                    "analyze_paired_superiority.py",
                    "--scored-csv",
                    str(scored_csv),
                    "--episode-summary-csv",
                    "",
                    "--baseline",
                    "stim_only",
                    "--conditions",
                    "episode_trace_v3",
                    "--output-dir",
                    str(output_dir),
                    "--bootstrap",
                    "100",
                    "--score-keys",
                    "appraisal_fidelity,raw_affect_preservation,anti_softening,action_tendency_fit,emotional_specificity",
                ]
                module.main()
            finally:
                sys.argv = original_argv

            overall = pd.read_csv(output_dir / "paired_overall.csv")
            mean_row = overall[overall["metric"] == "mean_total"].iloc[0]
            self.assertEqual(int(mean_row["paired_n"]), 2)
            self.assertAlmostEqual(float(mean_row["delta_mean"]), 0.5)


if __name__ == "__main__":
    unittest.main()
