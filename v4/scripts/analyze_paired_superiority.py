from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


SCORE_KEYS = [
    "content_fit",
    "emotional_appropriateness",
    "style_match",
    "naturalness",
    "overall_quality",
]


def parse_condition_list(raw: str) -> list[str]:
    conditions = [token.strip() for token in str(raw or "").replace(";", ",").split(",") if token.strip()]
    if not conditions:
        raise ValueError("at least one comparison condition is required")
    return conditions


def parse_score_keys(raw: str | None) -> list[str]:
    if not raw:
        return list(SCORE_KEYS)
    keys = [token.strip() for token in str(raw).replace(";", ",").split(",") if token.strip()]
    if not keys:
        raise ValueError("at least one score key is required")
    return keys


def bootstrap_mean_ci(values: np.ndarray, seed: int, n_bootstrap: int) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(int(n_bootstrap), values.size), replace=True).mean(axis=1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return float(low), float(high)


def sign_test_two_sided(wins: int, losses: int) -> float:
    n = int(wins + losses)
    if n <= 0:
        return float("nan")
    k = min(int(wins), int(losses))
    # Exact binomial two-sided p-value under p=0.5.
    prob = 0.0
    for i in range(k + 1):
        prob += float(math.comb(n, i)) * (0.5**n)
    return float(min(1.0, 2.0 * prob))


def add_mean_total(df: pd.DataFrame, score_keys: list[str]) -> pd.DataFrame:
    scored = df.copy()
    missing = [key for key in score_keys if key not in scored.columns]
    if missing:
        raise ValueError(f"score columns not found: {', '.join(missing)}")
    for key in score_keys:
        scored[key] = pd.to_numeric(scored[key], errors="coerce")
    scored["mean_total"] = scored[score_keys].mean(axis=1)
    return scored


def summarize_pair(
    base: pd.DataFrame,
    other: pd.DataFrame,
    *,
    baseline: str,
    condition: str,
    metric: str,
    seed: int,
    n_bootstrap: int,
) -> dict[str, object]:
    joined = base[[metric]].join(other[[metric]], how="inner", lsuffix=f"_{baseline}", rsuffix=f"_{condition}")
    joined = joined.dropna()
    delta = joined[f"{metric}_{condition}"] - joined[f"{metric}_{baseline}"]
    values = delta.to_numpy(dtype=float)
    wins = int((values > 0.0).sum())
    ties = int((values == 0.0).sum())
    losses = int((values < 0.0).sum())
    ci_low, ci_high = bootstrap_mean_ci(values, seed=seed, n_bootstrap=n_bootstrap)
    return {
        "condition": condition,
        "baseline": baseline,
        "metric": metric,
        "paired_n": int(values.size),
        "delta_mean": round(float(values.mean()), 6) if values.size else None,
        "delta_median": round(float(np.median(values)), 6) if values.size else None,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "win_rate": round(float(wins / values.size), 6) if values.size else None,
        "non_tie_win_rate": round(float(wins / (wins + losses)), 6) if wins + losses else None,
        "bootstrap_ci_low": round(ci_low, 6) if values.size else None,
        "bootstrap_ci_high": round(ci_high, 6) if values.size else None,
        "sign_test_p": round(sign_test_two_sided(wins, losses), 8) if wins + losses else None,
    }


def build_pair_rows(
    scored_df: pd.DataFrame,
    *,
    baseline: str,
    condition: str,
    record_id_column: str,
    score_keys: list[str],
) -> pd.DataFrame:
    optional_columns = [
        column
        for column in [
            "status",
            "text",
            "llm_response",
            "episode_label",
            "valence",
            "arousal",
            "target",
            "control_state",
            "social_orientation",
            "action_tendency",
        ]
        if column in scored_df.columns
    ]
    metric_columns = score_keys + ["mean_total", *optional_columns]
    base = scored_df[scored_df["condition"].astype(str) == baseline][[record_id_column, *metric_columns]].set_index(
        record_id_column
    )
    other = scored_df[scored_df["condition"].astype(str) == condition][[record_id_column, *metric_columns]].set_index(
        record_id_column
    )
    joined = base.join(other, how="inner", lsuffix=f"_{baseline}", rsuffix=f"_{condition}")
    joined = joined.dropna(subset=[f"mean_total_{baseline}", f"mean_total_{condition}"]).copy()
    joined["delta_mean_total"] = joined[f"mean_total_{condition}"] - joined[f"mean_total_{baseline}"]
    joined = joined.reset_index().rename(columns={record_id_column: "record_id"})
    for column in [
        "text",
        "llm_response",
        "episode_label",
        "valence",
        "arousal",
        "target",
        "control_state",
        "social_orientation",
        "action_tendency",
    ]:
        condition_column = f"{column}_{condition}"
        baseline_column = f"{column}_{baseline}"
        if condition_column in joined.columns:
            joined[column] = joined[condition_column]
        elif baseline_column in joined.columns:
            joined[column] = joined[baseline_column]
    joined.insert(1, "condition", condition)
    joined.insert(2, "baseline", baseline)
    return joined


def load_episode_summary(path: Path | None) -> pd.DataFrame | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"episode summary not found: {path}")
    return pd.read_csv(path)


def merge_episode(pair_rows: pd.DataFrame, episode_df: pd.DataFrame | None) -> pd.DataFrame:
    if episode_df is None or episode_df.empty:
        return pair_rows
    if "sample_id" not in episode_df.columns:
        raise ValueError("episode summary must contain sample_id")
    keep_columns = [
        column
        for column in [
            "sample_id",
            "episode_label",
            "valence",
            "arousal",
            "target",
            "control_state",
            "social_orientation",
            "action_tendency",
            "confidence",
        ]
        if column in episode_df.columns
    ]
    merged = pair_rows.merge(episode_df[keep_columns], left_on="record_id", right_on="sample_id", how="left")
    for column in keep_columns:
        if column == "sample_id":
            continue
        left = f"{column}_x"
        right = f"{column}_y"
        if left in merged.columns:
            merged[column] = merged[left].fillna(merged[right] if right in merged.columns else "")
            merged = merged.drop(columns=[left])
        if right in merged.columns:
            merged = merged.drop(columns=[right])
    return merged


def summarize_subsets(pair_rows: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_column in group_columns:
        if group_column not in pair_rows.columns:
            continue
        for (condition, baseline, group_value), group in pair_rows.groupby(
            ["condition", "baseline", group_column],
            dropna=False,
        ):
            values = pd.to_numeric(group["delta_mean_total"], errors="coerce").dropna().to_numpy(dtype=float)
            if values.size == 0:
                continue
            wins = int((values > 0.0).sum())
            ties = int((values == 0.0).sum())
            losses = int((values < 0.0).sum())
            rows.append(
                {
                    "condition": str(condition),
                    "baseline": str(baseline),
                    "subset_axis": group_column,
                    "subset_value": "(missing)" if pd.isna(group_value) else str(group_value),
                    "paired_n": int(values.size),
                    "delta_mean": round(float(values.mean()), 6),
                    "delta_median": round(float(np.median(values)), 6),
                    "wins": wins,
                    "ties": ties,
                    "losses": losses,
                    "win_rate": round(float(wins / values.size), 6),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["condition", "subset_axis", "delta_mean"], ascending=[True, True, False])


def format_markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "(no rows)"
    shown = df.copy()
    for column in columns:
        if column not in shown.columns:
            shown[column] = ""
    shown = shown[columns].copy()
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in shown.to_dict(orient="records"):
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    *,
    baseline: str,
    overall: pd.DataFrame,
    subsets: pd.DataFrame,
    examples: pd.DataFrame,
    output_dir: Path,
) -> None:
    main_metric = overall[overall["metric"] == "mean_total"].copy()
    top_examples = examples.sort_values("delta_mean_total", ascending=False).head(8)
    bottom_examples = examples.sort_values("delta_mean_total", ascending=True).head(8)
    report = "\n".join(
        [
            "# Paired Superiority Analysis",
            "",
            f"Baseline condition: `{baseline}`",
            "",
            "## Mean Total Comparisons",
            "",
            format_markdown_table(
                main_metric,
                [
                    "condition",
                    "paired_n",
                    "delta_mean",
                    "delta_median",
                    "wins",
                    "ties",
                    "losses",
                    "win_rate",
                    "bootstrap_ci_low",
                    "bootstrap_ci_high",
                    "sign_test_p",
                ],
            ),
            "",
            "## Metric-Level Comparisons",
            "",
            format_markdown_table(
                overall,
                [
                    "condition",
                    "metric",
                    "paired_n",
                    "delta_mean",
                    "wins",
                    "ties",
                    "losses",
                    "win_rate",
                    "bootstrap_ci_low",
                    "bootstrap_ci_high",
                ],
            ),
            "",
            "## Episode Subsets",
            "",
            format_markdown_table(
                subsets,
                ["condition", "subset_axis", "subset_value", "paired_n", "delta_mean", "wins", "ties", "losses", "win_rate"],
            ),
            "",
            "## Largest Wins",
            "",
            format_markdown_table(
                top_examples,
                ["condition", "record_id", "delta_mean_total", "episode_label", "valence", "arousal"],
            ),
            "",
            "## Largest Losses",
            "",
            format_markdown_table(
                bottom_examples,
                ["condition", "record_id", "delta_mean_total", "episode_label", "valence", "arousal"],
            ),
            "",
            "## Artifacts",
            "",
            f"- overall CSV: `{output_dir / 'paired_overall.csv'}`",
            f"- subset CSV: `{output_dir / 'paired_subsets.csv'}`",
            f"- examples CSV: `{output_dir / 'paired_examples.csv'}`",
            f"- summary JSON: `{output_dir / 'paired_summary.json'}`",
            "",
        ]
    )
    path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze paired superiority against a baseline condition.")
    parser.add_argument(
        "--scored-csv",
        default=str(PROJECT_ROOT / "outputs" / "experiments" / "paper_matrix_current_episode_v2_scored.csv"),
    )
    parser.add_argument(
        "--episode-summary-csv",
        default=str(
            PROJECT_ROOT
            / "outputs"
            / "research"
            / "trajectory_batch_matrix120_v1_gpt54"
            / "episode_summary.csv"
        ),
    )
    parser.add_argument("--baseline", default="stim_only")
    parser.add_argument(
        "--conditions",
        default="episode_trace,raw_trace,emonet_full,hybrid_episode,direct",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "experiments" / "paired_superiority_episode_v2"),
    )
    parser.add_argument("--record-id-column", default="record_id")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score-keys", default=",".join(SCORE_KEYS))
    args = parser.parse_args()

    scored_csv = Path(args.scored_csv)
    output_dir = Path(args.output_dir)
    episode_summary_csv = Path(args.episode_summary_csv) if args.episode_summary_csv else None
    conditions = parse_condition_list(args.conditions)
    score_keys = parse_score_keys(args.score_keys)

    scored_df = add_mean_total(pd.read_csv(scored_csv), score_keys)
    episode_df = load_episode_summary(episode_summary_csv)

    baseline_df = scored_df[scored_df["condition"].astype(str) == args.baseline].set_index(args.record_id_column)
    if baseline_df.empty:
        raise ValueError(f"baseline condition not found: {args.baseline}")

    summary_rows: list[dict[str, object]] = []
    pair_frames: list[pd.DataFrame] = []
    metrics = ["mean_total", *score_keys]
    for condition in conditions:
        condition_df = scored_df[scored_df["condition"].astype(str) == condition].set_index(args.record_id_column)
        if condition_df.empty:
            raise ValueError(f"condition not found: {condition}")
        for metric in metrics:
            summary_rows.append(
                summarize_pair(
                    baseline_df,
                    condition_df,
                    baseline=args.baseline,
                    condition=condition,
                    metric=metric,
                    seed=args.seed,
                    n_bootstrap=args.bootstrap,
                )
            )
        pair_frames.append(
            merge_episode(
                build_pair_rows(
                    scored_df,
                    baseline=args.baseline,
                    condition=condition,
                    record_id_column=args.record_id_column,
                    score_keys=score_keys,
                ),
                episode_df,
            )
        )

    overall_df = pd.DataFrame(summary_rows)
    examples_df = pd.concat(pair_frames, ignore_index=True) if pair_frames else pd.DataFrame()
    subsets_df = summarize_subsets(examples_df, ["valence", "arousal", "target", "control_state", "social_orientation"])

    output_dir.mkdir(parents=True, exist_ok=True)
    overall_path = output_dir / "paired_overall.csv"
    subsets_path = output_dir / "paired_subsets.csv"
    examples_path = output_dir / "paired_examples.csv"
    report_path = output_dir / "PAIRED_SUPERIORITY_REPORT.md"
    summary_path = output_dir / "paired_summary.json"

    overall_df.to_csv(overall_path, index=False, encoding="utf-8-sig")
    subsets_df.to_csv(subsets_path, index=False, encoding="utf-8-sig")
    examples_df.to_csv(examples_path, index=False, encoding="utf-8-sig")
    write_report(
        report_path,
        baseline=args.baseline,
        overall=overall_df,
        subsets=subsets_df,
        examples=examples_df,
        output_dir=output_dir,
    )

    payload = {
        "scored_csv": str(scored_csv),
        "episode_summary_csv": str(episode_summary_csv) if episode_summary_csv else "",
        "baseline": args.baseline,
        "conditions": conditions,
        "score_keys": score_keys,
        "overall_csv": str(overall_path),
        "subsets_csv": str(subsets_path),
        "examples_csv": str(examples_path),
        "report_md": str(report_path),
        "main_results": overall_df[overall_df["metric"] == "mean_total"].to_dict(orient="records"),
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
