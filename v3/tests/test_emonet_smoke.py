import csv
import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
import pandas as pd

from emonet import EmoNet, EmoNetConfig, StimEncoderConfig
from emonet.cli import (
    build_balanced_subset,
    command_build_llm_subset,
    export_z_from_json_stream,
    extract_json_block,
    label_subset_with_local_model,
    normalize_style_dict,
)


def write_csv(path: Path, rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


class EmoNetSmokeTests(unittest.TestCase):
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
        payload = extract_json_block("prefix\n{\"s\": {\"verbosity\": 1.2, \"sentence_length\": -0.2}}\nsuffix")
        style = normalize_style_dict(payload, "s")
        self.assertEqual(style["verbosity"], 1.0)
        self.assertEqual(style["sentence_length"], 0.0)
        self.assertEqual(len(style), 32)

    def test_label_subset_with_local_model(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            output_csv = temp_dir / "labeled.csv"
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

            generation_response = json.dumps(
                {
                    "s": {axis: 0.5 for axis in [
                        "verbosity", "sentence_length", "pace", "fragmentation", "repetition", "rhythmicity",
                        "directness", "explicitness", "specificity", "abstraction", "certainty", "logicality",
                        "warmth", "distance", "politeness", "formality", "cooperativeness", "dominance",
                        "calmness", "tension", "positivity", "heaviness", "urgency", "emotional_openness",
                        "softness", "sharpness", "playfulness", "seriousness", "metaphoricity", "plainness",
                        "initiative", "reflectiveness"
                    ]},
                    "response": "예시 응답",
                },
                ensure_ascii=False,
            )
            rating_response = json.dumps(
                {
                    "s_hat": {axis: 0.55 for axis in [
                        "verbosity", "sentence_length", "pace", "fragmentation", "repetition", "rhythmicity",
                        "directness", "explicitness", "specificity", "abstraction", "certainty", "logicality",
                        "warmth", "distance", "politeness", "formality", "cooperativeness", "dominance",
                        "calmness", "tension", "positivity", "heaviness", "urgency", "emotional_openness",
                        "softness", "sharpness", "playfulness", "seriousness", "metaphoricity", "plainness",
                        "initiative", "reflectiveness"
                    ]},
                    "notes": "ok",
                },
                ensure_ascii=False,
            )

            with patch("emonet.cli.call_openai_compatible_chat", side_effect=[generation_response, rating_response]):
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
                )

            saved = pd.read_csv(output_csv)
            self.assertEqual(len(saved), 1)
            self.assertIn("llm_response", saved.columns)
            self.assertIn("s_0", saved.columns)
            self.assertIn("s_hat_31", saved.columns)
            self.assertIn("consistency_l1", saved.columns)


if __name__ == "__main__":
    unittest.main()
