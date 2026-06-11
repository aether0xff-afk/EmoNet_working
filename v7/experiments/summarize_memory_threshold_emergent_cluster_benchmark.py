"""Summarize adjacency-based emergent-cluster diagnostic outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


MIN_POSITIVE_RATE = 0.8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="runs/memory_threshold_emergent_cluster_best_lmstudio")
    parser.add_argument("--output")
    return parser.parse_args()


def mean(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        raise ValueError(f"missing benchmark column: {column}")
    return float(frame[column].mean())


def positive_rate(frame: pd.DataFrame, column: str) -> float:
    if column not in frame.columns:
        raise ValueError(f"missing benchmark column: {column}")
    return float((frame[column] > 0).mean())


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input)
    csv_path = input_dir / "by_seed_cluster.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"cluster benchmark result not found: {csv_path}")
    frame = pd.read_csv(csv_path)

    trained_modularity = mean(frame, "trained_modularity")
    initial_modularity = mean(frame, "initial_modularity")
    null_modularity = mean(frame, "null_modularity_mean")
    trained_minus_initial = mean(frame, "trained_minus_initial_modularity")
    trained_minus_null = mean(frame, "trained_minus_null_modularity")
    coherence_gap = mean(frame, "response_coherence_gap")
    null_coherence_gap = mean(frame, "null_response_coherence_gap_mean")
    trained_minus_null_coherence = mean(frame, "trained_minus_null_response_coherence_gap")
    dominant_axis_corr = mean(frame, "community_dominant_axis_mean_abs_correlation")
    neuron_axis_corr = mean(frame, "neuron_axis_mean_abs_correlation")
    cluster_count = mean(frame, "selected_cluster_count")

    checks = {
        "trained_weighted_adjacency_is_more_modular_than_initialization_on_average": trained_minus_initial > 0.0,
        "trained_weighted_adjacency_is_more_modular_than_weight_shuffled_null_on_average": trained_minus_null > 0.0,
        "trained_weighted_adjacency_beats_weight_shuffled_null_for_most_seeds": positive_rate(frame, "trained_minus_null_modularity") >= MIN_POSITIVE_RATE,
        "within_community_memory_responses_are_more_coherent_than_between_community_responses": coherence_gap > 0.0,
        "within_community_response_coherence_beats_label_permutation_null_on_average": trained_minus_null_coherence > 0.0,
        "within_community_response_coherence_beats_null_for_most_seeds": positive_rate(frame, "trained_minus_null_response_coherence_gap") >= MIN_POSITIVE_RATE,
        "more_than_one_community_is_discovered": cluster_count > 1.0,
    }
    structural_evidence = all(
        checks[key]
        for key in (
            "trained_weighted_adjacency_is_more_modular_than_weight_shuffled_null_on_average",
            "trained_weighted_adjacency_beats_weight_shuffled_null_for_most_seeds",
            "more_than_one_community_is_discovered",
        )
    )
    functional_evidence = all(
        checks[key]
        for key in (
            "within_community_memory_responses_are_more_coherent_than_between_community_responses",
            "within_community_response_coherence_beats_label_permutation_null_on_average",
            "within_community_response_coherence_beats_null_for_most_seeds",
        )
    )

    if structural_evidence and functional_evidence:
        verdict = "adjacency_community_evidence_detected"
    elif structural_evidence:
        verdict = "structural_community_evidence_only"
    else:
        verdict = "community_evidence_not_established"

    report = {
        "input": str(input_dir),
        "seed_count": int(frame["seed"].nunique()),
        "stage_verdict": verdict,
        "means": {
            "selected_cluster_count": cluster_count,
            "trained_modularity": trained_modularity,
            "initial_modularity": initial_modularity,
            "null_modularity": null_modularity,
            "trained_minus_initial_modularity": trained_minus_initial,
            "trained_minus_null_modularity": trained_minus_null,
            "within_community_response_correlation": mean(frame, "within_community_response_correlation"),
            "between_community_response_correlation": mean(frame, "between_community_response_correlation"),
            "response_coherence_gap": coherence_gap,
            "null_response_coherence_gap": null_coherence_gap,
            "trained_minus_null_response_coherence_gap": trained_minus_null_coherence,
            "community_dominant_axis_mean_abs_correlation": dominant_axis_corr,
            "neuron_axis_mean_abs_correlation": neuron_axis_corr,
        },
        "positive_rates": {
            "trained_minus_initial_modularity": positive_rate(frame, "trained_minus_initial_modularity"),
            "trained_minus_null_modularity": positive_rate(frame, "trained_minus_null_modularity"),
            "trained_minus_null_response_coherence_gap": positive_rate(frame, "trained_minus_null_response_coherence_gap"),
        },
        "checks": checks,
        "interpretation": {
            "structural_evidence": structural_evidence,
            "functional_evidence": functional_evidence,
            "dominant_axis_summary": (
                "Community-axis correlations are descriptive only. Semantic labels were not used for training or community discovery."
            ),
            "fixed_mask_caveat": (
                "The sparse recurrent mask is fixed at initialization. A positive result means trained weights organize useful communities on that substrate, not that topology itself has fully self-organized."
            ),
        },
        "interpretation_boundary": (
            "This diagnostic tests weighted-adjacency communities and within-community memory-response coherence under a controlled fixture. "
            "Because the sparse mask is fixed at initialization, it does not establish fully self-organized topology, stable neuron roles, emotional ground truth, or biological fidelity."
        ),
    }
    output_path = Path(args.output) if args.output else input_dir / "decision_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
