"""Summarize semantic representation ablation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="runs/trace_semantic_representation_ablation_lmstudio")
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    csv_path = input_dir / "by_seed_representation.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"ablation result not found: {csv_path}")
    frame = pd.read_csv(csv_path)
    required = {"seed", "model_type", "representation_mode", "real_targeted_mae", "shuffled_history_mae_degradation", "reset_history_mae_degradation", "real_direction_accuracy", "real_pair_order_accuracy"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing ablation columns: {missing}")

    grouped = (
        frame.groupby(["model_type", "representation_mode"], as_index=False)
        .agg(
            real_targeted_mae=("real_targeted_mae", "mean"),
            shuffled_history_mae_degradation=("shuffled_history_mae_degradation", "mean"),
            reset_history_mae_degradation=("reset_history_mae_degradation", "mean"),
            real_direction_accuracy=("real_direction_accuracy", "mean"),
            real_pair_order_accuracy=("real_pair_order_accuracy", "mean"),
            seed_count=("seed", "nunique"),
        )
        .sort_values(["real_targeted_mae", "model_type", "representation_mode"])
    )
    rows = grouped.to_dict(orient="records")

    snn = grouped.loc[grouped["model_type"] == "snn_context_contrastive"].sort_values("real_targeted_mae")
    if snn.empty:
        raise ValueError("missing snn_context_contrastive rows")
    best_snn = snn.iloc[0].to_dict()
    raw_snn = snn.loc[snn["representation_mode"] == "raw_pool"]
    raw_snn_record = raw_snn.iloc[0].to_dict() if not raw_snn.empty else None

    gru = grouped.loc[grouped["model_type"] == "gru_context_contrastive"].sort_values("real_targeted_mae")
    best_gru = gru.iloc[0].to_dict() if not gru.empty else None
    text = grouped.loc[grouped["model_type"] == "context_free_mlp"].sort_values("real_targeted_mae")
    text_record = text.iloc[0].to_dict() if not text.empty else None

    best_snn_mae = float(best_snn["real_targeted_mae"])
    raw_snn_mae = float(raw_snn_record["real_targeted_mae"]) if raw_snn_record else None
    best_gru_mae = float(best_gru["real_targeted_mae"]) if best_gru else None
    text_mae = float(text_record["real_targeted_mae"]) if text_record else None

    report = {
        "input": str(input_dir),
        "representations_ranked_by_real_targeted_mae": rows,
        "best_snn_context_contrastive_representation": best_snn,
        "raw_snn_context_contrastive_representation": raw_snn_record,
        "best_gru_context_contrastive_representation": best_gru,
        "context_free_current_text_representation": text_record,
        "diagnostic_checks": {
            "some_snn_representation_beats_raw_pool": raw_snn_mae is not None and best_snn_mae < raw_snn_mae,
            "best_snn_representation_beats_current_text": text_mae is not None and best_snn_mae < text_mae,
            "best_snn_representation_is_not_worse_than_gru": best_gru_mae is not None and best_snn_mae <= best_gru_mae,
            "best_snn_representation_uses_history": float(best_snn["shuffled_history_mae_degradation"]) > 0.0 and float(best_snn["reset_history_mae_degradation"]) > 0.0,
            "best_snn_direction_accuracy_is_above_chance": float(best_snn["real_direction_accuracy"]) > 0.5,
            "best_snn_pair_order_accuracy_is_above_chance": float(best_snn["real_pair_order_accuracy"]) > 0.5,
        },
        "interpretation_guide": {
            "pooling_bottleneck": "If another SNN representation clearly beats raw_pool, semantic information may exist but the original pooling is lossy.",
            "objective_or_capacity_issue": "If every SNN representation remains weak while GRU performs better, the current SNN objective, dynamics, or capacity likely needs revision before cluster experiments.",
            "history_delta_signal": "If a history_delta representation improves results, separating persistent-history contribution from the current event is useful for later semantic reports."
        }
    }
    output_path = Path(args.output) if args.output else input_dir / "decision_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
