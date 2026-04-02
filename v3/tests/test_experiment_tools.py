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
