from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from emonet.cli import resolve_style_axes
from emonet.core import LinearZtoSDecoder, ZSDecoderConfig
from scripts.generate_paper_svgs import (
    bar_chart_horizontal,
    bar_chart_vertical,
    grouped_bar_chart,
    histogram_chart,
)


OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "paper" / "refresh_2026-04-07_structfix_v3"
FIG_DIR = OUTPUT_ROOT / "figures"
TABLE_DIR = OUTPUT_ROOT / "tables"

LABELED_CSV = PROJECT_ROOT / "outputs" / "llm" / "llm_subset_labeled_4000_extended40.csv"
LEARNED_Z_CSV = PROJECT_ROOT / "outputs" / "z" / "out_z_training_learned_extended40_structfix_reuse.csv"
OLD_Z_CSV = PROJECT_ROOT / "outputs" / "z" / "out_z_training_extended40.csv"
NEW_BRANCH_CSV = PROJECT_ROOT / "outputs" / "z" / "out_z_training_extended40_structfix.csv"
BENCHMARK_CSV = PROJECT_ROOT.parent / "encoder-ML testing" / "out_benchmark" / "benchmark_results_20260305_180830.csv"
CURRENT_GENERATION_TABLE_CSV = TABLE_DIR / "baseline_generation_table_current.csv"

SEEDS = [7, 13, 21, 42, 84]


def round_float(value: float, digits: int = 6) -> float:
    return round(float(value), digits)


def evaluate_decoder_features(x: np.ndarray, s: np.ndarray, val_rows: int) -> dict[str, float]:
    runs: list[tuple[float, float]] = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(x))
        val_idx = indices[:val_rows]
        train_idx = indices[val_rows:]
        decoder = LinearZtoSDecoder(
            config=ZSDecoderConfig(model_path=PROJECT_ROOT / "artifacts" / "tmp_refresh_decoder.npz", ridge_alpha=1.0),
            z_dim=x.shape[1],
            s_dim=s.shape[1],
        )
        decoder.fit(x[train_idx], s[train_idx])
        pred = decoder.predict(x[val_idx])
        mae = float(np.mean(np.abs(pred - s[val_idx])))
        baseline = np.broadcast_to(s[train_idx].mean(axis=0, dtype=np.float32), s[val_idx].shape)
        baseline_mae = float(np.mean(np.abs(baseline - s[val_idx])))
        runs.append((mae, baseline_mae))
    return {
        "decoder_mae_mean": round_float(np.mean([mae for mae, _ in runs])),
        "baseline_mae_mean": round_float(np.mean([baseline for _, baseline in runs])),
        "mean_gain": round_float(np.mean([baseline - mae for mae, baseline in runs])),
    }


def evaluate_text_baseline(texts: list[str], s: np.ndarray, val_rows: int) -> dict[str, float]:
    runs: list[tuple[float, float]] = []
    text_arr = np.asarray(texts, dtype=object)
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(text_arr))
        val_idx = indices[:val_rows]
        train_idx = indices[val_rows:]
        vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=2)
        train_sparse = vectorizer.fit_transform(text_arr[train_idx].tolist())
        n_features = int(train_sparse.shape[1])
        if n_features <= 1:
            train_x = train_sparse.toarray().astype(np.float32)
            val_x = vectorizer.transform(text_arr[val_idx].tolist()).toarray().astype(np.float32)
        else:
            n_components = min(128, max(1, n_features - 1))
            svd = TruncatedSVD(n_components=n_components, random_state=seed)
            train_x = svd.fit_transform(train_sparse).astype(np.float32)
            val_x = svd.transform(vectorizer.transform(text_arr[val_idx].tolist())).astype(np.float32)

        decoder = LinearZtoSDecoder(
            config=ZSDecoderConfig(model_path=PROJECT_ROOT / "artifacts" / "tmp_refresh_text_decoder.npz", ridge_alpha=1.0),
            z_dim=train_x.shape[1],
            s_dim=s.shape[1],
        )
        decoder.fit(train_x, s[train_idx])
        pred = decoder.predict(val_x)
        mae = float(np.mean(np.abs(pred - s[val_idx])))
        baseline = np.broadcast_to(s[train_idx].mean(axis=0, dtype=np.float32), s[val_idx].shape)
        baseline_mae = float(np.mean(np.abs(baseline - s[val_idx])))
        runs.append((mae, baseline_mae))
    return {
        "decoder_mae_mean": round_float(np.mean([mae for mae, _ in runs])),
        "baseline_mae_mean": round_float(np.mean([baseline for _, baseline in runs])),
        "mean_gain": round_float(np.mean([baseline - mae for mae, baseline in runs])),
    }


def summarize_branch_lengths(values: pd.Series) -> dict[str, float | int]:
    return {
        "rows": int(len(values)),
        "mean": round_float(values.mean(), 4),
        "len1": int((values == 1).sum()),
        "len1_ratio": round_float((values == 1).mean(), 4),
        "p50": int(values.quantile(0.50)),
        "p75": int(values.quantile(0.75)),
        "p90": int(values.quantile(0.90)),
        "p95": int(values.quantile(0.95)),
        "p99": int(values.quantile(0.99)),
        "max": int(values.max()),
    }


def build_encoder_chart() -> None:
    df = pd.read_csv(BENCHMARK_CSV)
    df = df[df["status"] == "ok"].copy()
    df["MAE(mean)"] = df["MAE(mean)"].astype(float)
    df = df.sort_values("MAE(mean)").head(6)
    labels = [f"{row['vector']} + {row['model']}" for _, row in df.iterrows()]
    bar_chart_horizontal(
        path=FIG_DIR / "encoder_benchmark_top6.svg",
        title="Encoder Benchmark",
        subtitle="30,000-sample regression benchmark sorted by MAE",
        labels=labels,
        values=df["MAE(mean)"].astype(float).tolist(),
        colors=["#2563eb", "#3b82f6", "#60a5fa", "#93c5fd", "#1d4ed8", "#1e40af"],
        x_label="mean MAE",
        note="Lower is better. The best setting remained char_tfidf + Ridge.",
        value_format="{:.4f}",
        height=520,
    )


def build_branch_figures(old_df: pd.Series, new_df: pd.Series) -> dict[str, Any]:
    before = summarize_branch_lengths(old_df)
    after = summarize_branch_lengths(new_df)

    counts = new_df.value_counts().sort_index()
    labels = [str(int(idx)) for idx in counts.index.tolist()]
    total = float(counts.sum())
    values = [(count / total) * 100.0 for count in counts.tolist()]
    bar_chart_vertical(
        path=FIG_DIR / "dominant_branch_length_distribution_structfix.svg",
        title="Dominant Branch Length Distribution",
        subtitle="Structfix export, 51,628 samples",
        labels=labels,
        values=values,
        colors=["#2563eb" if label == "1" else "#93c5fd" for label in labels],
        y_label="sample percentage (%)",
        note="Length-1 collapse is largely removed in the structfix export.",
        value_format="{:.2f}",
        width=1100,
        height=560,
    )

    group_labels = ["before", "after"]
    series_labels = ["len1_ratio", "mean", "p95"]
    grouped_bar_chart(
        path=FIG_DIR / "dominant_branch_before_after.svg",
        title="Dominant Branch Before vs After Structfix",
        subtitle="Legacy export compared with current structfix export",
        group_labels=group_labels,
        series_labels=series_labels,
        values=[
            [float(before["len1_ratio"]), float(before["mean"]), float(before["p95"])],
            [float(after["len1_ratio"]), float(after["mean"]), float(after["p95"])],
        ],
        colors=["#dc2626", "#2563eb", "#14b8a6"],
        y_label="value",
        note="len1_ratio should drop while mean/p95 rise.",
    )

    return {"before": before, "after": after}


def build_consistency_and_bias_figures(df: pd.DataFrame) -> dict[str, Any]:
    keep = df[df["keep_sample"].fillna(False).astype(bool)].reset_index(drop=True)
    histogram_chart(
        path=FIG_DIR / "style_consistency_histogram_extended40.svg",
        title="Style Label Consistency",
        subtitle="extended40 labeled subset, all parsed rows",
        values=pd.to_numeric(df["consistency_l1"], errors="coerce").dropna().tolist(),
        bins=16,
        x_label="consistency L1",
        note="4,000 labeled rows; keep rows average lower consistency error.",
    )

    axes = resolve_style_axes(40, style_profile="extended40")
    mean_map = {axis: float(keep[f"s_{idx}"].mean()) for idx, axis in enumerate(axes)}
    selected_axes = [
        "softness",
        "calmness",
        "cooperativeness",
        "positivity",
        "warmth",
        "hostility",
        "resentment",
        "despair",
        "volatility",
        "fearfulness",
        "shame",
        "relief",
        "trust",
    ]
    bar_chart_horizontal(
        path=FIG_DIR / "style_bias_axes_extended40.svg",
        title="Style Axis Bias (extended40)",
        subtitle="Mean axis values for kept rows",
        labels=selected_axes,
        values=[mean_map[axis] for axis in selected_axes],
        colors=[
            "#14b8a6" if mean_map[axis] > 0.75 else "#ef4444" if mean_map[axis] < 0.10 else "#3b82f6"
            for axis in selected_axes
        ],
        x_label="mean axis value",
        note="Softness/calmness/cooperativeness remain high, while raw negative affect stays near zero.",
        value_format="{:.3f}",
        height=620,
    )

    return {
        "rows_ok": int(len(df[df["status"] == "ok"])) if "status" in df.columns else int(len(df)),
        "rows_keep": int(len(keep)),
        "consistency_mean_ok": round_float(pd.to_numeric(df["consistency_l1"], errors="coerce").dropna().mean(), 4),
        "consistency_mean_keep": round_float(pd.to_numeric(keep["consistency_l1"], errors="coerce").dropna().mean(), 4),
        "selected_axis_means": {axis: round_float(mean_map[axis], 4) for axis in selected_axes},
        "top_high": [
            {"axis": axis, "mean": round_float(value, 4)}
            for axis, value in sorted(mean_map.items(), key=lambda item: item[1], reverse=True)[:8]
        ],
        "top_low": [
            {"axis": axis, "mean": round_float(value, 4)}
            for axis, value in sorted(mean_map.items(), key=lambda item: item[1])[:8]
        ],
    }


def build_predictor_table_and_chart(labeled_df: pd.DataFrame, learned_df: pd.DataFrame) -> dict[str, Any]:
    keep = labeled_df[labeled_df["keep_sample"].fillna(False).astype(bool)].copy().reset_index(drop=True)
    learned_z = learned_df[["sample_id"] + [f"z_{idx}" for idx in range(64)]].copy()
    merged = keep.merge(learned_z, on="sample_id", suffixes=("_old", "_new"))
    s = merged[[f"s_{idx}" for idx in range(40)]].to_numpy(dtype=np.float32)
    val_rows = max(32, int(round(len(merged) * 0.1)))
    old_z = merged[[f"z_{idx}_old" for idx in range(64)]].to_numpy(dtype=np.float32)
    new_z = merged[[f"z_{idx}_new" for idx in range(64)]].to_numpy(dtype=np.float32)
    stim = merged[["dopamine", "serotonin", "norepinephrine", "melatonin"]].to_numpy(dtype=np.float32)

    mean_metrics = evaluate_text_baseline(merged["text"].astype(str).tolist(), s, val_rows)
    baseline_mae = mean_metrics["baseline_mae_mean"]
    stim_metrics = evaluate_decoder_features(stim, s, val_rows)
    text_metrics = mean_metrics
    legacy_metrics = evaluate_decoder_features(old_z, s, val_rows)
    structfix_metrics = evaluate_decoder_features(new_z, s, val_rows)

    rows = [
        {
            "name": "mean_baseline",
            "decoder_mae_mean": baseline_mae,
            "baseline_mae_mean": baseline_mae,
            "mean_gain": 0.0,
        },
        {"name": "stim_only_ridge", **stim_metrics},
        {"name": "text_tfidf_ridge", **text_metrics},
        {"name": "legacy_z64_ridge", **legacy_metrics},
        {"name": "structfix_learned_z64_ridge", **structfix_metrics},
    ]
    predictor_df = pd.DataFrame(rows)
    predictor_df.to_csv(TABLE_DIR / "baseline_predictor_table_current.csv", index=False, encoding="utf-8-sig")
    predictor_payload = {
        row["name"]: {
            "decoder_mae_mean": row["decoder_mae_mean"],
            "baseline_mae_mean": row["baseline_mae_mean"],
            "mean_gain": row["mean_gain"],
        }
        for row in rows
    }
    (TABLE_DIR / "baseline_predictor_table_current.json").write_text(
        json.dumps(predictor_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    labels = ["mean baseline", "stim-only", "text tfidf", "legacy z64", "structfix z64"]
    values = [float(row["decoder_mae_mean"]) for row in rows]
    bar_chart_vertical(
        path=FIG_DIR / "predictor_mae_comparison_current.svg",
        title="Predictor MAE Comparison (extended40 current)",
        subtitle="Validation MAE on 2,832 kept rows",
        labels=labels,
        values=values,
        colors=["#9ca3af", "#14b8a6", "#60a5fa", "#2563eb", "#7c3aed"],
        y_label="validation MAE",
        note="Lower is better. Current learned z does not yet beat the text baseline.",
        reference=float(baseline_mae),
        reference_label="mean baseline",
        value_format="{:.4f}",
    )

    return {
        "rows_used": int(len(merged)),
        "val_rows": int(val_rows),
        "mean_baseline": baseline_mae,
        "stim_only": stim_metrics,
        "text_tfidf": text_metrics,
        "legacy_z64": legacy_metrics,
        "structfix_learned_z64": structfix_metrics,
    }


def build_generation_table_and_chart() -> dict[str, Any] | None:
    if not CURRENT_GENERATION_TABLE_CSV.exists():
        return None

    df = pd.read_csv(CURRENT_GENERATION_TABLE_CSV)
    required_columns = {
        "condition",
        "mean_content_fit",
        "mean_emotional_appropriateness",
        "mean_style_match",
        "mean_naturalness",
        "mean_overall_quality",
        "mean_total",
    }
    if not required_columns.issubset(set(df.columns)):
        missing = sorted(required_columns.difference(set(df.columns)))
        raise ValueError(f"current generation table is missing columns: {', '.join(missing)}")

    metric_columns = [
        "mean_content_fit",
        "mean_emotional_appropriateness",
        "mean_style_match",
        "mean_naturalness",
        "mean_overall_quality",
    ]
    series_labels = ["content_fit", "emotion_fit", "style_match", "naturalness", "overall"]
    label_map = {
        "direct": "direct",
        "stim_only": "stim_only",
        "emonet_full": "EmoNet full",
        "emonet_no_summary": "w/o summary",
        "emonet_no_tags": "w/o tags",
        "emonet_no_expression": "w/o expression",
        "emonet_vector_only": "vector only",
        "emonet_macro_only": "macro only",
    }
    group_labels = [label_map.get(str(name), str(name)) for name in df["condition"].tolist()]
    values = [[float(row[column]) for column in metric_columns] for _, row in df.iterrows()]
    grouped_bar_chart(
        path=FIG_DIR / "baseline_generation_scores_current.svg",
        title="Generation Quality Comparison (current)",
        subtitle="LLM-judge 5-point scores across current response-generation conditions",
        group_labels=group_labels,
        series_labels=series_labels,
        values=values,
        colors=["#2563eb", "#14b8a6", "#f59e0b", "#8b5cf6", "#ef4444"],
        y_label="score",
        note="Higher is better. This chart reflects the current structfix/stylefix generation stack.",
    )
    top_row = df.sort_values(["mean_total", "condition"], ascending=[False, True]).iloc[0]
    return {
        "rows": int(len(df)),
        "best_condition": str(top_row["condition"]),
        "best_mean_total": round_float(float(top_row["mean_total"]), 4),
        "conditions": df.to_dict(orient="records"),
    }


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    labeled_df = pd.read_csv(LABELED_CSV)
    learned_df = pd.read_csv(LEARNED_Z_CSV)
    old_branch_series = pd.read_csv(OLD_Z_CSV, usecols=["dominant_branch_len"])["dominant_branch_len"]
    new_branch_series = pd.read_csv(NEW_BRANCH_CSV, usecols=["dominant_branch_len"])["dominant_branch_len"]

    build_encoder_chart()
    branch_summary = build_branch_figures(old_branch_series, new_branch_series)
    style_summary = build_consistency_and_bias_figures(labeled_df)
    predictor_summary = build_predictor_table_and_chart(labeled_df, learned_df)
    generation_summary = build_generation_table_and_chart()

    summary_payload = {
        "output_root": str(OUTPUT_ROOT),
        "figures": sorted(str(path) for path in FIG_DIR.glob("*.svg")),
        "branch_summary": branch_summary,
        "style_summary": style_summary,
        "predictor_summary": predictor_summary,
        "generation_summary": generation_summary,
        "notes": [
            "Structfix solves branch-length collapse at the distribution level but does not yet solve style-target bias.",
            "Current learned z remains competitive with the mean baseline but does not outperform the text baseline.",
        ],
    }
    if generation_summary is None:
        summary_payload["notes"].insert(
            0,
            "Current refresh excludes generation baseline re-scoring because no current scored experiment matrix was present locally.",
        )
    else:
        summary_payload["notes"].insert(
            0,
            f"Current generation baseline table is included; best condition is {generation_summary['best_condition']}.",
        )
    (TABLE_DIR / "paper_refresh_summary.json").write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
