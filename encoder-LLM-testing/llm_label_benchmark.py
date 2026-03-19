#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLM Label Classification Benchmark
=================================

목적
- text -> label 분류를 OpenAI / Gemini / Anthropic 모델로 비교
- CSV 입력: 최소 text, label 컬럼 필요
- 결과:
  - raw_predictions.csv
  - summary_metrics.csv
  - stability_metrics.csv
  - confusion_matrix_*.csv
  - 그래프 PNG

실행 예시 (PowerShell)
python llm_label_benchmark.py `
  --input_csv dataset_labels.csv `
  --output_dir benchmark_outputs `
  --runs_per_model 3 `
  --start_idx 0 `
  --max_samples 100 `
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


PROVIDER_TO_LABEL = {
    "openai": "ChatGPT",
    "gemini": "Gemini",
    "anthropic": "Claude Sonnet",
}


@dataclass
class ExperimentConfig:
    input_csv: str
    output_dir: str
    label_list: List[str]

    timeout: int = 120
    sleep_between_calls: float = 1.0
    max_retries: int = 3
    runs_per_model: int = 3
    temperature: float = 0.0
    random_seed: int = 42

    start_idx: int = 0
    max_samples: int = 100

    openai_model: str = "gpt-5.4-mini"
    gemini_model: str = "gemini-2.5-flash"
    anthropic_model: str = "claude-sonnet-4-5-20250929"

    evaluate_openai: bool = True
    evaluate_gemini: bool = True
    evaluate_anthropic: bool = True


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def maybe_sleep(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


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
        pred = normalize_label(raw_obj, label_list)
        return pred
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


class BaseProviderClient:
    def predict(self, text: str, label_list: List[str], run_idx: int = 0) -> Tuple[str, str]:
        raise NotImplementedError


class OpenAIClient(BaseProviderClient):
    def __init__(self, api_key: str, model: str, timeout: int, temperature: float):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.url = "https://api.openai.com/v1/responses"

    @staticmethod
    def _extract_output_text_from_response(data: Dict[str, Any]) -> str:
        output_text = data.get("output_text", "")
        if isinstance(output_text, str) and output_text.strip():
            return output_text.strip()

        collected: List[str] = []
        output_items = data.get("output", [])

        if isinstance(output_items, list):
            for item in output_items:
                if not isinstance(item, dict):
                    continue

                content_list = item.get("content", [])
                if not isinstance(content_list, list):
                    continue

                for content in content_list:
                    if not isinstance(content, dict):
                        continue

                    content_type = content.get("type", "")
                    if content_type in {"output_text", "text"}:
                        text_value = content.get("text", "")
                        if isinstance(text_value, str) and text_value.strip():
                            collected.append(text_value.strip())

        joined = "\n".join(collected).strip()
        if joined:
            return joined

        return ""

    def predict(self, text: str, label_list: List[str], run_idx: int = 0) -> Tuple[str, str]:
        prompt = build_classification_prompt(text, label_list)

        payload = {
            "model": self.model,
            "input": prompt,
            "temperature": self.temperature,
            "max_output_tokens": 32,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "label_prediction",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "label": {
                                "type": "string",
                                "enum": label_list
                            }
                        },
                        "required": ["label"],
                        "additionalProperties": False,
                    },
                }
            },
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        resp = requests.post(self.url, headers=headers, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()

        raw_text = self._extract_output_text_from_response(data)

        if not raw_text:
            raw_text = json.dumps(data, ensure_ascii=False, indent=2)

        pred = parse_label_from_text(raw_text, label_list)
        return pred, raw_text


class AnthropicClient(BaseProviderClient):
    def __init__(self, api_key: str, model: str, timeout: int, temperature: float):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.url = "https://api.anthropic.com/v1/messages"

    def predict(self, text: str, label_list: List[str], run_idx: int = 0) -> Tuple[str, str]:
        prompt = build_classification_prompt(text, label_list)
        payload = {
            "model": self.model,
            "max_tokens": 200,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        resp = requests.post(self.url, headers=headers, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()

        content = data.get("content", [])
        raw_text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                raw_text_parts.append(item.get("text", ""))

        raw_text = "\n".join(raw_text_parts).strip()
        pred = parse_label_from_text(raw_text, label_list)
        return pred, raw_text


class GeminiClient(BaseProviderClient):
    def __init__(self, api_key: str, model: str, timeout: int, temperature: float):
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.temperature = temperature

    def predict(self, text: str, label_list: List[str], run_idx: int = 0) -> Tuple[str, str]:
        prompt = build_classification_prompt(text, label_list)
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent?key={self.api_key}"
        )
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "responseMimeType": "application/json",
            },
        }

        headers = {"Content-Type": "application/json"}
        resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()

        raw_text = ""
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            raw_text = "\n".join(
                part.get("text", "") for part in parts if isinstance(part, dict)
            ).strip()

        pred = parse_label_from_text(raw_text, label_list)
        return pred, raw_text


class LLMLabelBenchmark:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        ensure_dir(cfg.output_dir)
        np.random.seed(cfg.random_seed)
        self.clients: Dict[str, BaseProviderClient] = {}
        self._init_clients()

    def _init_clients(self) -> None:
        if self.cfg.evaluate_openai:
            key = os.getenv("OPENAI_API_KEY", "").strip()
            if not key:
                raise EnvironmentError("OPENAI_API_KEY가 설정되지 않았다.")
            self.clients["openai"] = OpenAIClient(
                key,
                self.cfg.openai_model,
                self.cfg.timeout,
                self.cfg.temperature,
            )

        if self.cfg.evaluate_gemini:
            key = os.getenv("GEMINI_API_KEY", "").strip()
            if not key:
                raise EnvironmentError("GEMINI_API_KEY가 설정되지 않았다.")
            self.clients["gemini"] = GeminiClient(
                key,
                self.cfg.gemini_model,
                self.cfg.timeout,
                self.cfg.temperature,
            )

        if self.cfg.evaluate_anthropic:
            key = os.getenv("ANTHROPIC_API_KEY", "").strip()
            if not key:
                raise EnvironmentError("ANTHROPIC_API_KEY가 설정되지 않았다.")
            self.clients["anthropic"] = AnthropicClient(
                key,
                self.cfg.anthropic_model,
                self.cfg.timeout,
                self.cfg.temperature,
            )

    def _load_dataset(self) -> pd.DataFrame:
        df = pd.read_csv(self.cfg.input_csv)
        if "text" not in df.columns or "label" not in df.columns:
            raise ValueError("입력 CSV에는 최소한 text, label 컬럼이 있어야 한다.")

        total_len = len(df)
        start_idx = max(0, self.cfg.start_idx)

        if start_idx >= total_len:
            raise ValueError(
                f"start_idx({start_idx})가 데이터 길이({total_len})보다 크거나 같다."
            )

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

    def _call_with_retry(self, provider: str, text: str, run_idx: int) -> Tuple[str, str]:
        client = self.clients[provider]
        last_error = None

        for attempt in range(1, self.cfg.max_retries + 1):
            try:
                pred, raw = client.predict(text, self.cfg.label_list, run_idx=run_idx)
                maybe_sleep(self.cfg.sleep_between_calls)
                return pred, raw
            except Exception as e:
                last_error = e
                print(f"[WARN] provider={provider} run={run_idx} attempt={attempt} 실패: {e}")
                if attempt < self.cfg.max_retries:
                    maybe_sleep(self.cfg.sleep_between_calls * attempt)

        raise RuntimeError(f"{provider} 최종 실패: {last_error}")

    def run(self) -> None:
        df = self._load_dataset()
        raw_records: List[Dict[str, Any]] = []

        for provider in self.clients.keys():
            print(f"[INFO] ===== {provider} 시작 =====")
            for _, row in df.iterrows():
                sample_id = int(row["original_sample_id"])
                text = str(row["text"])
                gt_label = str(row["label"]).strip()

                for run_idx in range(self.cfg.runs_per_model):
                    try:
                        pred_label, raw_text = self._call_with_retry(provider, text, run_idx)
                        raw_records.append({
                            "provider": provider,
                            "provider_label": PROVIDER_TO_LABEL[provider],
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
                            "provider": provider,
                            "provider_label": PROVIDER_TO_LABEL[provider],
                            "sample_id": sample_id,
                            "run_idx": run_idx,
                            "text": text,
                            "gt_label": gt_label,
                            "pred_label": "",
                            "correct": 0,
                            "raw_response": "",
                            "error": str(e),
                        })
                        print(f"[ERROR] provider={provider} sample={sample_id} run={run_idx}")
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

        for provider, g in valid_df.groupby("provider"):
            confmat = confusion_matrix_manual(
                y_true=g["gt_label"].tolist(),
                y_pred=g["pred_label"].tolist(),
                labels=self.cfg.label_list,
            )
            confmat_df = pd.DataFrame(confmat, index=self.cfg.label_list, columns=self.cfg.label_list)
            confmat_df.to_csv(
                os.path.join(self.cfg.output_dir, f"confusion_matrix_{provider}.csv"),
                encoding="utf-8-sig",
            )

        self._make_all_plots(valid_df, summary_df, stability_df)
        print(f"[DONE] 결과 저장 완료: {self.cfg.output_dir}")

    def _compute_summary_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for provider, g in df.groupby("provider"):
            y_true = g["gt_label"].tolist()
            y_pred = g["pred_label"].tolist()

            acc = accuracy_score_manual(y_true, y_pred)
            confmat = confusion_matrix_manual(y_true, y_pred, self.cfg.label_list)
            macro_precision, macro_recall, macro_f1 = precision_recall_f1_from_confmat(confmat)

            rows.append({
                "provider": provider,
                "provider_label": PROVIDER_TO_LABEL[provider],
                "samples": len(g),
                "accuracy": acc,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "macro_f1": macro_f1,
            })

        return pd.DataFrame(rows).sort_values(by="accuracy", ascending=False)

    def _compute_stability_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for provider, g_provider in df.groupby("provider"):
            stable_scores = []
            for _, g_sample in g_provider.groupby("sample_id"):
                preds = g_sample["pred_label"].tolist()
                if len(preds) < 2:
                    continue
                majority = max(set(preds), key=preds.count)
                consistency = preds.count(majority) / len(preds)
                stable_scores.append(consistency)

            rows.append({
                "provider": provider,
                "provider_label": PROVIDER_TO_LABEL[provider],
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

        for provider, g in raw_df.groupby("provider"):
            self._plot_confusion_matrix(provider, g)

    def _plot_summary(self, summary_df: pd.DataFrame) -> None:
        fig = plt.figure(figsize=(8, 5))
        plt.bar(summary_df["provider_label"], summary_df["accuracy"])
        plt.title("Accuracy by Model")
        plt.xlabel("Model")
        plt.ylabel("Accuracy")
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.output_dir, "plot_accuracy.png"), dpi=200)
        plt.close(fig)

        fig = plt.figure(figsize=(8, 5))
        plt.bar(summary_df["provider_label"], summary_df["macro_f1"])
        plt.title("Macro F1 by Model")
        plt.xlabel("Model")
        plt.ylabel("Macro F1")
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.output_dir, "plot_macro_f1.png"), dpi=200)
        plt.close(fig)

    def _plot_stability(self, stability_df: pd.DataFrame) -> None:
        fig = plt.figure(figsize=(8, 5))
        plt.bar(stability_df["provider_label"], stability_df["mean_consistency"])
        plt.title("Repeated-run Consistency by Model")
        plt.xlabel("Model")
        plt.ylabel("Consistency")
        plt.tight_layout()
        plt.savefig(os.path.join(self.cfg.output_dir, "plot_consistency.png"), dpi=200)
        plt.close(fig)

    def _plot_confusion_matrix(self, provider: str, g: pd.DataFrame) -> None:
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
        plt.title(f"Confusion Matrix - {PROVIDER_TO_LABEL[provider]}")
        plt.tight_layout()
        plt.savefig(
            os.path.join(self.cfg.output_dir, f"plot_confusion_matrix_{provider}.png"),
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

    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--sleep_between_calls", type=float, default=1.0)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--runs_per_model", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--random_seed", type=int, default=42)

    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=100)

    parser.add_argument("--openai_model", type=str, default="gpt-5.4-mini")
    parser.add_argument("--gemini_model", type=str, default="gemini-2.5-flash")
    parser.add_argument("--anthropic_model", type=str, default="claude-sonnet-4-5-20250929")

    parser.add_argument("--disable_openai", action="store_true")
    parser.add_argument("--disable_gemini", action="store_true")
    parser.add_argument("--disable_anthropic", action="store_true")

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
        openai_model=args.openai_model,
        gemini_model=args.gemini_model,
        anthropic_model=args.anthropic_model,
        evaluate_openai=not args.disable_openai,
        evaluate_gemini=not args.disable_gemini,
        evaluate_anthropic=not args.disable_anthropic,
    )


def main() -> None:
    cfg = parse_args()
    runner = LLMLabelBenchmark(cfg)
    runner.run()


if __name__ == "__main__":
    main()