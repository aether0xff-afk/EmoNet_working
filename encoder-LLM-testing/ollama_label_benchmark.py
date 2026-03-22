#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Ollama Label Classification Benchmark
=====================================

목적
- text -> label 분류를 여러 Ollama 모델로 비교
- CSV 입력: 최소 text, label 컬럼 필요
- 결과:
  - raw_predictions.csv
  - summary_metrics.csv
  - stability_metrics.csv
  - confusion_matrix_*.csv
  - 그래프 PNG

지원 기본 모델
- gemini-3-flash-preview
- deepseek-v3.2:cloud
- gpt-oss:20b
- llama3.3

실행 예시 (PowerShell)
python ollama_label_benchmark.py `
  --input_csv dataset_stimulus.csv `
  --output_dir ollama_benchmark_outputs `
  --runs_per_model 1 `
  --start_idx 0 `
  --max_samples 30 `
  --label_list E01,E02,E03,E04,E05,E06,E07,E08,E09,E10,E11,E12,E13,E14,E15,E16,E17,E18
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests


MODEL_TO_LABEL = {
    "gemini_3_flash_preview": "Gemini 3 Flash Preview",
    "deepseek_v32": "DeepSeek V3.2",
    "gpt_oss_20b": "GPT-OSS 20B",
    "llama_33": "Llama 3.3",
}

MODEL_TO_OLLAMA_NAME = {
    "gemini_3_flash_preview": "gemini-3-flash-preview",
    "deepseek_v32": "deepseek-v3.2:cloud",
    "gpt_oss_20b": "gpt-oss:20b",
    "llama_33": "llama3.3",
}


@dataclass
class ExperimentConfig:
    input_csv: str
    output_dir: str
    label_list: List[str]

    timeout: int = 300
    sleep_between_calls: float = 1.0
    max_retries: int = 3
    runs_per_model: int = 1
    temperature: float = 0.0
    random_seed: int = 42

    start_idx: int = 0
    max_samples: int = 30

    ollama_base_url: str = "http://localhost:11434"
    max_predict_tokens: int = 64

    gemini_3_flash_preview_model: str = "gemini-3-flash-preview"
    deepseek_v32_model: str = "deepseek-v3.2:cloud"
    gpt_oss_20b_model: str = "gpt-oss:20b"
    llama_33_model: str = "llama3.3"

    evaluate_gemini_3_flash_preview: bool = True
    evaluate_deepseek_v32: bool = True
    evaluate_gpt_oss_20b: bool = True
    evaluate_llama_33: bool = True


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def maybe_sleep(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def sanitize_error_message(msg: str) -> str:
    if not msg:
        return msg
    msg = re.sub(r"(https?://[^\s]*)(key=)([^&\s]+)", r"\1\2***API_KEY***", msg)
    return msg


def extract_json_block(text: str) -> Dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise ValueError("빈 응답이라 JSON 파싱이 불가능하다.")

    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidates = fenced[:]

    brace_matches = re.findall(r"(\{.*?\})", text, flags=re.DOTALL)
    candidates.extend(brace_matches)

    last_error = None
    for cand in candidates:
        try:
            obj = json.loads(cand)
            if isinstance(obj, dict):
                return obj
        except Exception as e:
            last_error = e

    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception as e:
        last_error = e

    raise ValueError(f"JSON 파싱 실패: {last_error}\n원문:\n{text[:1000]}")


def normalize_label(raw_obj: Dict[str, Any], label_list: List[str]) -> str:
    pred = str(raw_obj.get("label", "")).strip()
    if pred not in label_list:
        raise ValueError(f"허용되지 않은 label 예측: {pred}")
    return pred


def parse_label_from_text(raw_text: str, label_list: List[str]) -> str:
    raw_text = (raw_text or "").strip()
    if not raw_text:
        raise ValueError("응답 텍스트가 비어 있다.")

    try:
        raw_obj = extract_json_block(raw_text)
        return normalize_label(raw_obj, label_list)
    except Exception:
        pass

    m = re.search(r"\bE\d{2}\b", raw_text)
    if m:
        pred = m.group(0).strip()
        if pred in label_list:
            return pred

    raise ValueError(f"허용되지 않은 label 예측: {raw_text[:300]}")


def accuracy_score_manual(y_true: List[str], y_pred: List[str]) -> float:
    if not y_true:
        return 0.0
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    return correct / len(y_true)


def confusion_matrix_manual(y_true: List[str], y_pred: List[str], labels: List[str]) -> np.ndarray:
    idx = {label: i for i, label in enumerate(labels)}
    mat = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in idx and p in idx:
            mat[idx[t], idx[p]] += 1
    return mat


def precision_recall_f1_from_confmat(confmat: np.ndarray) -> Tuple[float, float, float]:
    precisions = []
    recalls = []
    f1s = []

    for i in range(confmat.shape[0]):
        tp = confmat[i, i]
        fp = confmat[:, i].sum() - tp
        fn = confmat[i, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return float(np.mean(precisions)), float(np.mean(recalls)), float(np.mean(f1s))


def build_classification_prompt(text: str, label_list: List[str]) -> str:
    label_text = ", ".join(label_list)

    prompt = f"""
You are a careful emotion label classifier.

Task:
Read the full Korean dialogue text and predict exactly one label from the allowed label set.

Allowed labels:
{label_text}

Rules:
1. Output ONLY one JSON object.
2. The JSON must have exactly one key: "label"
3. The value must be exactly one label from the allowed label set.
4. Do not output explanation, markdown, comments, or extra text.

Output example:
{{"label": "{label_list[0]}"}}

Input text:
{text}
""".strip()

    return prompt


class BaseModelClient:
    def predict(self, text: str, label_list: List[str], run_idx: int = 0) -> Tuple[str, str]:
        raise NotImplementedError


class OllamaClient(BaseModelClient):
    def __init__(
        self,
        model: str,
        base_url: str,
        timeout: int,
        temperature: float,
        max_predict_tokens: int,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.max_predict_tokens = max_predict_tokens
        self.url = f"{self.base_url}/api/generate"

    def predict(self, text: str, label_list: List[str], run_idx: int = 0) -> Tuple[str, str]:
        prompt = build_classification_prompt(text, label_list)

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_predict_tokens,
            },
        }

        resp = requests.post(self.url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()

        raw_text = str(data.get("response", "")).strip()
        if not raw_text:
            raw_text = json.dumps(data, ensure_ascii=False, indent=2)

        pred = parse_label_from_text(raw_text, label_list)
        return pred, raw_text


class OllamaLabelBenchmark:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        ensure_dir(cfg.output_dir)
        np.random.seed(cfg.random_seed)
        self.clients: Dict[str, BaseModelClient] = {}
        self._init_clients()

    def _init_clients(self) -> None:
        if self.cfg.evaluate_gemini_3_flash_preview:
            self.clients["gemini_3_flash_preview"] = OllamaClient(
                model=self.cfg.gemini_3_flash_preview_model,
                base_url=self.cfg.ollama_base_url,
                timeout=self.cfg.timeout,
                temperature=self.cfg.temperature,
                max_predict_tokens=self.cfg.max_predict_tokens,
            )

        if self.cfg.evaluate_deepseek_v32:
            self.clients["deepseek_v32"] = OllamaClient(
                model=self.cfg.deepseek_v32_model,
                base_url=self.cfg.ollama_base_url,
                timeout=self.cfg.timeout,
                temperature=self.cfg.temperature,
                max_predict_tokens=self.cfg.max_predict_tokens,
            )

        if self.cfg.evaluate_gpt_oss_20b:
            self.clients["gpt_oss_20b"] = OllamaClient(
                model=self.cfg.gpt_oss_20b_model,
                base_url=self.cfg.ollama_base_url,
                timeout=self.cfg.timeout,
                temperature=self.cfg.temperature,
                max_predict_tokens=self.cfg.max_predict_tokens,
            )

        if self.cfg.evaluate_llama_33:
            self.clients["llama_33"] = OllamaClient(
                model=self.cfg.llama_33_model,
                base_url=self.cfg.ollama_base_url,
                timeout=self.cfg.timeout,
                temperature=self.cfg.temperature,
                max_predict_tokens=self.cfg.max_predict_tokens,
            )

        if not self.clients:
            raise ValueError("활성화된 모델이 하나도 없다.")

    def _load_dataset(self) -> pd.DataFrame:
        df = pd.read_csv(self.cfg.input_csv)
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError("입력 CSV에는 최소한 text, label 컬럼이 있어야 한다.")

        total_len = len(df)
        start_idx = max(0, self.cfg.start_idx)

        if start_idx >= total_len:
            raise ValueError(f"start_idx({start_idx})가 데이터 길이({total_len})보다 크거나 같다.")

        if self.cfg.max_samples is not None and self.cfg.max_samples > 0:
            end_idx = min(start_idx + self.cfg.max_samples, total_len)
            df = df.iloc[start_idx:end_idx].copy()
        else:
            df = df.iloc[start_idx:].copy()

        df = df.reset_index(drop=False).rename(columns={"index": "original_sample_id"})

        print(
            f"[INFO] 데이터셋 슬라이스: original rows {start_idx} ~ "
            f"{start_idx + len(df) - 1} / total {total_len}"
        )
        print(f"[INFO] 이번 실행 처리 건수: {len(df)}")
        return df

    def _call_with_retry(self, model_key: str, text: str, run_idx: int) -> Tuple[str, str]:
        client = self.clients[model_key]
        last_error = None

        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                pred, raw = client.predict(text, self.cfg.label_list, run_idx=run_idx)
                maybe_sleep(self.cfg.sleep_between_calls)
                return pred, raw

            except requests.HTTPError as e:
                last_error = e
                status_code = None
                try:
                    status_code = e.response.status_code
                except Exception:
                    pass

                err_msg = sanitize_error_message(str(e))
                print(f"[WARN] model={model_key} run={run_idx} attempt={attempt} 실패: {err_msg}")

                if attempt < self.cfg.max_retries:
                    wait_sec = self.cfg.sleep_between_calls * attempt
                    if status_code == 429:
                        wait_sec = max(wait_sec, 10.0 * attempt)
                    print(f"[INFO] retry 대기: {wait_sec:.1f}초")
                    maybe_sleep(wait_sec)

            except Exception as e:
                last_error = e
                err_msg = sanitize_error_message(str(e))
                print(f"[WARN] model={model_key} run={run_idx} attempt={attempt} 실패: {err_msg}")
                if attempt < self.cfg.max_retries:
                    maybe_sleep(self.cfg.sleep_between_calls * attempt)

        raise RuntimeError(f"{model_key} 최종 실패: {sanitize_error_message(str(last_error))}")

    def run(self) -> None:
        df = self._load_dataset()
        raw_records: List[Dict[str, Any]] = []

        for model_key in self.clients.keys():
            print(f"[INFO] ===== {model_key} 시작 =====")
            for _, row in df.iterrows():
                sample_id = int(row["original_sample_id"])
                text = str(row["text"])
                gt_label = str(row["label"]).strip()

                for run_idx in range(self.cfg.runs_per_model):
                    try:
                        pred_label, raw_text = self._call_with_retry(model_key, text, run_idx)
                        raw_records.append({
                            "model_key": model_key,
                            "model_label": MODEL_TO_LABEL[model_key],
                            "ollama_model_name": self._get_ollama_model_name(model_key),
                            "sample_id": sample_id,
                            "run_idx": run_idx,
                            "text": text,
                            "gt_label": gt_label,
                            "pred_label": pred_label,
                            "correct": int(gt_label == pred_label),
                            "raw_response": raw_text,
                            "error": "",
                        })
                    except Exception as e:
                        raw_records.append({
                            "model_key": model_key,
                            "model_label": MODEL_TO_LABEL[model_key],
                            "ollama_model_name": self._get_ollama_model_name(model_key),
                            "sample_id": sample_id,
                            "run_idx": run_idx,
                            "text": text,
                            "gt_label": gt_label,
                            "pred_label": "",
                            "correct": 0,
                            "raw_response": "",
                            "error": sanitize_error_message(str(e)),
                        })
                        print(f"[ERROR] model={model_key} sample={sample_id} run={run_idx}")
                        traceback.print_exc()

        raw_df = pd.DataFrame(raw_records)
        raw_df.to_csv(
            os.path.join(self.cfg.output_dir, "raw_predictions.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        valid_df = raw_df[raw_df["pred_label"].astype(str) != ""].copy()
        if len(valid_df) == 0:
            raise RuntimeError("유효한 예측 결과가 없다.")

        summary_df = self._compute_summary_metrics(valid_df)
        summary_df.to_csv(
            os.path.join(self.cfg.output_dir, "summary_metrics.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        stability_df = self._compute_stability_metrics(valid_df)
        stability_df.to_csv(
            os.path.join(self.cfg.output_dir, "stability_metrics.csv"),
            index=False,
            encoding="utf-8-sig",
        )

        for model_key, g in valid_df.groupby("model_key"):
            confmat = confusion_matrix_manual(
                y_true=g["gt_label"].tolist(),
                y_pred=g["pred_label"].tolist(),
                labels=self.cfg.label_list,
            )
            confmat_df = pd.DataFrame(confmat, index=self.cfg.label_list, columns=self.cfg.label_list)
            confmat_df.to_csv(
                os.path.join(self.cfg.output_dir, f"confusion_matrix_{model_key}.csv"),
                encoding="utf-8-sig",
            )

        self._make_all_plots(valid_df, summary_df, stability_df)
        print(f"[DONE] 결과 저장 완료: {self.cfg.output_dir}")

    def _get_ollama_model_name(self, model_key: str) -> str:
        mapping = {
            "gemini_3_flash_preview": self.cfg.gemini_3_flash_preview_model,
            "deepseek_v32": self.cfg.deepseek_v32_model,
            "gpt_oss_20b": self.cfg.gpt_oss_20b_model,
            "llama_33": self.cfg.llama_33_model,
        }
        return mapping[model_key]

    def _compute_summary_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for model_key, g in df.groupby("model_key"):
            y_true = g["gt_label"].tolist()
            y_pred = g["pred_label"].tolist()

            acc = accuracy_score_manual(y_true, y_pred)
            confmat = confusion_matrix_manual(y_true, y_pred, self.cfg.label_list)
            macro_precision, macro_recall, macro_f1 = precision_recall_f1_from_confmat(confmat)

            rows.append({
                "model_key": model_key,
                "model_label": MODEL_TO_LABEL[model_key],
                "ollama_model_name": self._get_ollama_model_name(model_key),
                "samples": len(g),
                "accuracy": acc,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_f1": macro_f1,
            })

        return pd.DataFrame(rows).sort_values(by="accuracy", ascending=False)

    def _compute_stability_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for model_key, g_model in df.groupby("model_key"):
            stable_scores = []
            for _, g_sample in g_model.groupby("sample_id"):
                preds = g_sample["pred_label"].tolist()
                if len(preds) < 2:
                    continue
                majority = max(set(preds), key=preds.count)
                consistency = preds.count(majority) / len(preds)
                stable_scores.append(consistency)

            rows.append({
                "model_key": model_key,
                "model_label": MODEL_TO_LABEL[model_key],
                "ollama_model_name": self._get_ollama_model_name(model_key),
                "mean_consistency": float(np.mean(stable_scores)) if stable_scores else 0.0,
                "n_samples_used": len(stable_scores),
            })

        return pd.DataFrame(rows).sort_values(by="mean_consistency", ascending=False)

    def _make_all_plots(
        self,
        raw_df: pd.DataFrame,
        summary_df: pd.DataFrame,
        stability_df: pd.DataFrame,
    ) -> None:
        self._plot_summary(summary_df)
        self._plot_stability(stability_df)

        for model_key, g in raw_df.groupby("model_key"):
            self._plot_confusion_matrix(model_key, g)

    def _plot_summary(self, summary_df: pd.DataFrame) -> None:
        fig = plt.figure(figsize=(9, 5))
        plt.bar(summary_df["model_label"], summary_df["accuracy"])
        plt.title("Accuracy by Model")
        plt.xlabel("Model")
        plt.ylabel("Accuracy")
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.output_dir, "plot_accuracy.png"), dpi=200)
        plt.close(fig)

        fig = plt.figure(figsize=(9, 5))
        plt.bar(summary_df["model_label"], summary_df["macro_f1"])
        plt.title("Macro F1 by Model")
        plt.xlabel("Model")
        plt.ylabel("Macro F1")
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.output_dir, "plot_macro_f1.png"), dpi=200)
        plt.close(fig)

    def _plot_stability(self, stability_df: pd.DataFrame) -> None:
        fig = plt.figure(figsize=(9, 5))
        plt.bar(stability_df["model_label"], stability_df["mean_consistency"])
        plt.title("Repeated-run Consistency by Model")
        plt.xlabel("Model")
        plt.ylabel("Consistency")
        plt.xticks(rotation=15, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.output_dir, "plot_consistency.png"), dpi=200)
        plt.close(fig)

    def _plot_confusion_matrix(self, model_key: str, g: pd.DataFrame) -> None:
        confmat = confusion_matrix_manual(
            y_true=g["gt_label"].tolist(),
            y_pred=g["pred_label"].tolist(),
            labels=self.cfg.label_list,
        )

        fig = plt.figure(figsize=(10, 8))
        plt.imshow(confmat)
        plt.colorbar()
        plt.xticks(range(len(self.cfg.label_list)), self.cfg.label_list, rotation=45, ha="right")
        plt.yticks(range(len(self.cfg.label_list)), self.cfg.label_list)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"Confusion Matrix - {MODEL_TO_LABEL[model_key]}")
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.cfg.output_dir, f"plot_confusion_matrix_{model_key}.png"),
            dpi=200,
        )
        plt.close(fig)


def parse_args() -> ExperimentConfig:
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--label_list",
        type=str,
        required=True,
        help="쉼표로 구분한 label 목록. 예: E01,E02,E03",
    )

    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--sleep_between_calls", type=float, default=1.0)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--runs_per_model", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--random_seed", type=int, default=42)

    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=30)

    parser.add_argument("--ollama_base_url", type=str, default="http://localhost:11434")
    parser.add_argument("--max_predict_tokens", type=int, default=64)

    parser.add_argument("--gemini_3_flash_preview_model", type=str, default="gemini-3-flash-preview")
    parser.add_argument("--deepseek_v32_model", type=str, default="deepseek-v3.2:cloud")
    parser.add_argument("--gpt_oss_20b_model", type=str, default="gpt-oss:20b")
    parser.add_argument("--llama_33_model", type=str, default="llama3.3")

    parser.add_argument("--disable_gemini_3_flash_preview", action="store_true")
    parser.add_argument("--disable_deepseek_v32", action="store_true")
    parser.add_argument("--disable_gpt_oss_20b", action="store_true")
    parser.add_argument("--disable_llama_33", action="store_true")

    args = parser.parse_args()

    label_list = [x.strip() for x in args.label_list.split(",") if x.strip()]
    if not label_list:
        raise ValueError("label_list가 비어 있다.")

    return ExperimentConfig(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        label_list=label_list,
        timeout=args.timeout,
        sleep_between_calls=args.sleep_between_calls,
        max_retries=args.max_retries,
        runs_per_model=args.runs_per_model,
        temperature=args.temperature,
        random_seed=args.random_seed,
        start_idx=args.start_idx,
        max_samples=args.max_samples,
        ollama_base_url=args.ollama_base_url,
        max_predict_tokens=args.max_predict_tokens,
        gemini_3_flash_preview_model=args.gemini_3_flash_preview_model,
        deepseek_v32_model=args.deepseek_v32_model,
        gpt_oss_20b_model=args.gpt_oss_20b_model,
        llama_33_model=args.llama_33_model,
        evaluate_gemini_3_flash_preview=not args.disable_gemini_3_flash_preview,
        evaluate_deepseek_v32=not args.disable_deepseek_v32,
        evaluate_gpt_oss_20b=not args.disable_gpt_oss_20b,
        evaluate_llama_33=not args.disable_llama_33,
    )


def main() -> None:
    cfg = parse_args()
    runner = OllamaLabelBenchmark(cfg)
    runner.run()


if __name__ == "__main__":
    main()