from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(".")
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

TRACE_SUMMARY_JSON = ROOT / "outputs" / "experiments" / "eval_axis_split_v1" / "trace_sensitive_full58_claude_summary.json"
PAIRED_OVERALL_CSV = ROOT / "outputs" / "experiments" / "eval_axis_split_v1" / "paired_trace_sensitive_full58_claude" / "paired_overall.csv"
PAIRED_EXAMPLES_CSV = ROOT / "outputs" / "experiments" / "eval_axis_split_v1" / "paired_trace_sensitive_full58_claude" / "paired_examples.csv"
EPISODE_SUMMARY_CSV = ROOT / "outputs" / "research" / "trajectory_batch_matrix120_v1_gpt54" / "episode_summary.csv"


def save_current(name: str) -> None:
    plt.tight_layout()
    plt.savefig(OUT / f"{name}.svg", format="svg")
    plt.savefig(OUT / f"{name}.png", dpi=220)
    plt.close()


def fig_trace_sensitive_scores() -> None:
    payload = json.loads(TRACE_SUMMARY_JSON.read_text(encoding="utf-8"))
    rows = payload["conditions"]
    df = pd.DataFrame(rows).set_index("condition")

    metrics = [
        "mean_appraisal_fidelity",
        "mean_raw_affect_preservation",
        "mean_anti_softening",
        "mean_action_tendency_fit",
        "mean_emotional_specificity",
        "mean_naturalness",
        "mean_overall_preference",
    ]

    labels = [
        "appraisal",
        "raw affect",
        "anti-softening",
        "action tendency",
        "specificity",
        "naturalness",
        "preference",
    ]

    plot_df = df.loc[["stim_only", "episode_trace_v3"], metrics].T
    plot_df.index = labels

    ax = plot_df.plot(kind="bar", figsize=(11, 5), rot=35)
    ax.set_title("Trace-sensitive evaluation scores")
    ax.set_ylabel("Mean score (1-5)")
    ax.set_xlabel("Metric")
    ax.set_ylim(0, 5)
    ax.legend(title="Condition")
    save_current("fig_trace_sensitive_scores_full58")


def fig_trace_sensitive_delta() -> None:
    df = pd.read_csv(PAIRED_OVERALL_CSV)
    df = df[df["condition"].astype(str) == "episode_trace_v3"].copy()
    df = df[df["metric"].astype(str) != "mean_total"].copy()

    labels = {
        "appraisal_fidelity": "appraisal",
        "raw_affect_preservation": "raw affect",
        "anti_softening": "anti-softening",
        "action_tendency_fit": "action tendency",
        "emotional_specificity": "specificity",
        "naturalness": "naturalness",
        "overall_preference": "preference",
    }

    df["label"] = df["metric"].map(labels).fillna(df["metric"])
    yerr_low = df["delta_mean"] - df["bootstrap_ci_low"]
    yerr_high = df["bootstrap_ci_high"] - df["delta_mean"]

    plt.figure(figsize=(11, 5))
    plt.bar(df["label"], df["delta_mean"], yerr=[yerr_low, yerr_high], capsize=4)
    plt.axhline(0, linewidth=1)
    plt.title("Paired delta against stim_only")
    plt.ylabel("Mean delta: episode_trace_v3 - stim_only")
    plt.xlabel("Metric")
    plt.xticks(rotation=35, ha="right")
    save_current("fig_trace_sensitive_metric_delta_full58")


def fig_paired_delta_distribution() -> None:
    df = pd.read_csv(PAIRED_EXAMPLES_CSV)
    df = df[df["condition"].astype(str) == "episode_trace_v3"].copy()
    values = pd.to_numeric(df["delta_mean_total"], errors="coerce").dropna()

    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=12)
    plt.axvline(values.mean(), linewidth=2)
    plt.title("Distribution of paired mean-total deltas")
    plt.xlabel("episode_trace_v3 - stim_only")
    plt.ylabel("Count")
    save_current("fig_paired_delta_distribution_full58")


def fig_win_tie_loss() -> None:
    df = pd.read_csv(PAIRED_OVERALL_CSV)
    row = df[(df["condition"].astype(str) == "episode_trace_v3") & (df["metric"].astype(str) == "mean_total")].iloc[0]

    labels = ["wins", "ties", "losses"]
    values = [int(row["wins"]), int(row["ties"]), int(row["losses"])]

    plt.figure(figsize=(6, 5))
    plt.bar(labels, values)
    plt.title("Paired win / tie / loss against stim_only")
    plt.ylabel("Number of paired samples")
    plt.xlabel("Outcome")
    save_current("fig_win_tie_loss_full58")


def fig_episode_valence_arousal_distribution() -> None:
    df = pd.read_csv(EPISODE_SUMMARY_CSV)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    valence_order = ["negative", "mixed", "positive"]
    arousal_order = ["high", "medium", "low"]

    valence_counts = df["valence"].astype(str).value_counts().reindex(valence_order).fillna(0)
    arousal_counts = df["arousal"].astype(str).value_counts().reindex(arousal_order).fillna(0)

    axes[0].bar(valence_counts.index, valence_counts.values)
    axes[0].set_title("Valence distribution")
    axes[0].set_ylabel("Count")

    axes[1].bar(arousal_counts.index, arousal_counts.values)
    axes[1].set_title("Arousal distribution")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    plt.savefig(OUT / "fig_episode_valence_arousal_distribution.svg", format="svg")
    plt.savefig(OUT / "fig_episode_valence_arousal_distribution.png", dpi=220)
    plt.close()


def fig_episode_confidence_distribution() -> None:
    df = pd.read_csv(EPISODE_SUMMARY_CSV)
    if "confidence" not in df.columns:
        print("skip confidence figure: confidence column not found")
        return

    values = pd.to_numeric(df["confidence"], errors="coerce").dropna()
    plt.figure(figsize=(8, 5))
    plt.hist(values, bins=12)
    plt.axvline(values.mean(), linewidth=2)
    plt.title("Episode interpretation confidence distribution")
    plt.xlabel("Confidence")
    plt.ylabel("Count")
    save_current("fig_episode_confidence_distribution")


def main() -> None:
    fig_trace_sensitive_scores()
    fig_trace_sensitive_delta()
    fig_paired_delta_distribution()
    fig_win_tie_loss()
    fig_episode_valence_arousal_distribution()
    fig_episode_confidence_distribution()

    print(f"saved figures to: {OUT.resolve()}")


if __name__ == "__main__":
    main()
