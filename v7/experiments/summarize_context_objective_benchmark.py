"""Summarize context-objective benchmark outputs into a compact decision report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REQUIRED_MODELS = {
    "snn_next_only",
    "snn_context_contrastive",
    "context_free_mlp",
    "gru_context_contrastive",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="runs/context_objective_benchmark_lmstudio")
    parser.add_argument("--output")
    return parser.parse_args()


def mean_for(frame: pd.DataFrame, model_type: str, column: str) -> float:
    selected = frame.loc[frame["model_type"] == model_type, column]
    if selected.empty:
        raise ValueError(f"missing model rows for {model_type}")
    return float(selected.mean())


def positive_rate(frame: pd.DataFrame, model_type: str, column: str) -> float:
    selected = frame.loc[frame["model_type"] == model_type, column]
    if selected.empty:
        raise ValueError(f"missing model rows for {model_type}")
    return float((selected > 0).mean())


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    csv_path = input_dir / "by_seed_model.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"benchmark result not found: {csv_path}")
    frame = pd.read_csv(csv_path)
    present_models = set(frame["model_type"].unique())
    missing = REQUIRED_MODELS - present_models
    if missing:
        raise ValueError(f"missing benchmark models: {sorted(missing)}")

    snn_next_margin = mean_for(frame, "snn_next_only", "real_context_margin")
    snn_contrast_margin = mean_for(frame, "snn_context_contrastive", "real_context_margin")
    gru_margin = mean_for(frame, "gru_context_contrastive", "real_context_margin")
    mlp_margin = mean_for(frame, "context_free_mlp", "real_context_margin")
    snn_history_gap = mean_for(frame, "snn_context_contrastive", "real_minus_shuffled_context_margin")
    gru_history_gap = mean_for(frame, "gru_context_contrastive", "real_minus_shuffled_context_margin")

    report = {
        "input": str(input_dir),
        "seed_count": int(frame["seed"].nunique()),
        "means": {
            "snn_next_only_real_context_margin": snn_next_margin,
            "snn_context_contrastive_real_context_margin": snn_contrast_margin,
            "context_free_mlp_real_context_margin": mlp_margin,
            "gru_context_contrastive_real_context_margin": gru_margin,
            "snn_context_contrastive_real_minus_shuffled_margin": snn_history_gap,
            "gru_context_contrastive_real_minus_shuffled_margin": gru_history_gap,
        },
        "positive_rates": {
            "snn_context_contrastive_real_margin": positive_rate(frame, "snn_context_contrastive", "real_context_margin"),
            "snn_context_contrastive_real_minus_shuffled_margin": positive_rate(frame, "snn_context_contrastive", "real_minus_shuffled_context_margin"),
            "gru_context_contrastive_real_minus_shuffled_margin": positive_rate(frame, "gru_context_contrastive", "real_minus_shuffled_context_margin"),
        },
        "comparisons": {
            "snn_contrastive_minus_next_only_margin": snn_contrast_margin - snn_next_margin,
            "snn_contrastive_minus_gru_margin": snn_contrast_margin - gru_margin,
            "snn_history_gap_minus_gru_history_gap": snn_history_gap - gru_history_gap,
        },
        "checks": {
            "contrastive_snn_beats_next_only_margin": snn_contrast_margin > snn_next_margin,
            "contrastive_snn_uses_correct_history": snn_history_gap > 0,
            "context_free_mlp_is_near_zero": abs(mlp_margin) < 1e-6,
            "snn_has_margin_advantage_over_gru": snn_contrast_margin > gru_margin,
        },
        "interpretation_boundary": (
            "This report evaluates context-sensitive prediction and correct-history reliance. "
            "It does not establish emotional semantics or biological fidelity."
        ),
    }
    output_path = Path(args.output) if args.output else input_dir / "decision_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
