from __future__ import annotations

import json
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "outputs" / "paper" / "figures"

STYLE_AXES = (
    "verbosity",
    "sentence_length",
    "pace",
    "fragmentation",
    "repetition",
    "rhythmicity",
    "directness",
    "explicitness",
    "specificity",
    "abstraction",
    "certainty",
    "logicality",
    "warmth",
    "distance",
    "politeness",
    "formality",
    "cooperativeness",
    "dominance",
    "calmness",
    "tension",
    "positivity",
    "heaviness",
    "urgency",
    "emotional_openness",
    "softness",
    "sharpness",
    "playfulness",
    "seriousness",
    "metaphoricity",
    "plainness",
    "initiative",
    "reflectiveness",
)


def write_svg(path: Path, width: int, height: int, body: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<style>',
        "text { font-family: 'Malgun Gothic', 'Segoe UI', sans-serif; fill: #1f2937; }",
        ".title { font-size: 24px; font-weight: 700; }",
        ".subtitle { font-size: 13px; fill: #6b7280; }",
        ".axis { font-size: 12px; fill: #4b5563; }",
        ".label { font-size: 12px; }",
        ".small { font-size: 11px; fill: #4b5563; }",
        ".value { font-size: 11px; font-weight: 600; }",
        ".grid { stroke: #e5e7eb; stroke-width: 1; }",
        ".axis-line { stroke: #9ca3af; stroke-width: 1.2; }",
        ".note { font-size: 12px; fill: #374151; }",
        "</style>",
        '<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>',
        *body,
        "</svg>",
    ]
    path.write_text("\n".join(svg), encoding="utf-8")


def nice_condition_label(name: str) -> str:
    mapping = {
        "direct": "direct",
        "stim_only": "stim_only",
        "emonet_full": "EmoNet full",
        "full": "full",
        "without_inhibitory": "w/o inhib.",
        "without_excitatory": "w/o excit.",
        "without_modulatory": "w/o modul.",
        "without_memory": "w/o memory",
        "without_rewiring": "w/o rewire",
        "z_dim_32": "z=32",
        "z_dim_64": "z=64",
        "z_dim_128": "z=128",
        "mean_baseline": "mean baseline",
        "stim_only_ridge": "stim-only",
        "text_tfidf_ridge": "text tfidf",
        "emonet_z64_ridge": "EmoNet z64",
    }
    return mapping.get(name, name)


def draw_axes_frame(body: list[str], left: int, top: int, right: int, bottom: int) -> None:
    body.append(f'<line class="axis-line" x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}"/>')
    body.append(f'<line class="axis-line" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>')


def add_title(body: list[str], title: str, subtitle: str) -> None:
    body.append(f'<text class="title" x="40" y="42">{escape(title)}</text>')
    body.append(f'<text class="subtitle" x="40" y="66">{escape(subtitle)}</text>')


def bar_chart_vertical(
    *,
    path: Path,
    title: str,
    subtitle: str,
    labels: list[str],
    values: list[float],
    colors: list[str],
    y_label: str,
    note: str | None = None,
    reference: float | None = None,
    reference_label: str | None = None,
    value_format: str = "{:.3f}",
    width: int = 980,
    height: int = 560,
) -> None:
    body: list[str] = []
    add_title(body, title, subtitle)
    left, top, right, bottom = 90, 110, width - 40, height - 95
    plot_w = right - left
    plot_h = bottom - top
    max_value = max(values + ([reference] if reference is not None else []))
    max_value *= 1.15
    if max_value <= 0:
        max_value = 1.0

    for step in range(6):
        y = top + plot_h * step / 5
        value = max_value * (1 - step / 5)
        body.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}"/>')
        body.append(f'<text class="axis" x="{left - 8}" y="{y + 4:.2f}" text-anchor="end">{value:.3f}</text>')

    draw_axes_frame(body, left, top, right, bottom)

    if reference is not None:
        ry = bottom - (reference / max_value) * plot_h
        body.append(f'<line x1="{left}" y1="{ry:.2f}" x2="{right}" y2="{ry:.2f}" stroke="#dc2626" stroke-width="2" stroke-dasharray="7 5"/>')
        if reference_label:
            body.append(f'<text class="small" x="{right - 4}" y="{ry - 8:.2f}" text-anchor="end">{escape(reference_label)}</text>')

    n = len(values)
    gap = 18
    slot = plot_w / max(1, n)
    bar_w = min(56, slot - gap)
    for idx, (label, value, color) in enumerate(zip(labels, values, colors, strict=True)):
        x = left + slot * idx + (slot - bar_w) / 2
        h = (value / max_value) * plot_h
        y = bottom - h
        body.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" rx="6" fill="{color}"/>')
        body.append(f'<text class="value" x="{x + bar_w/2:.2f}" y="{y - 8:.2f}" text-anchor="middle">{value_format.format(value)}</text>')
        body.append(f'<text class="label" x="{x + bar_w/2:.2f}" y="{bottom + 20:.2f}" text-anchor="middle">{escape(label)}</text>')

    body.append(f'<text class="axis" x="{left - 58}" y="{top - 16}" text-anchor="start">{escape(y_label)}</text>')
    if note:
        body.append(f'<text class="note" x="40" y="{height - 24}">{escape(note)}</text>')
    write_svg(path, width, height, body)


def bar_chart_horizontal(
    *,
    path: Path,
    title: str,
    subtitle: str,
    labels: list[str],
    values: list[float],
    colors: list[str],
    x_label: str,
    note: str | None = None,
    value_format: str = "{:.3f}",
    width: int = 980,
    height: int = 620,
) -> None:
    body: list[str] = []
    add_title(body, title, subtitle)
    left, top, right, bottom = 180, 110, width - 50, height - 55
    plot_w = right - left
    plot_h = bottom - top
    max_value = max(values) * 1.15
    if max_value <= 0:
        max_value = 1.0

    for step in range(6):
        x = left + plot_w * step / 5
        value = max_value * step / 5
        body.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{bottom}"/>')
        body.append(f'<text class="axis" x="{x:.2f}" y="{bottom + 22}" text-anchor="middle">{value:.2f}</text>')

    draw_axes_frame(body, left, top, right, bottom)
    slot = plot_h / max(1, len(values))
    bar_h = min(28, slot - 12)
    for idx, (label, value, color) in enumerate(zip(labels, values, colors, strict=True)):
        y = top + slot * idx + (slot - bar_h) / 2
        w = (value / max_value) * plot_w
        body.append(f'<rect x="{left}" y="{y:.2f}" width="{w:.2f}" height="{bar_h:.2f}" rx="6" fill="{color}"/>')
        body.append(f'<text class="label" x="{left - 10}" y="{y + bar_h/2 + 4:.2f}" text-anchor="end">{escape(label)}</text>')
        body.append(f'<text class="value" x="{left + w + 8:.2f}" y="{y + bar_h/2 + 4:.2f}" text-anchor="start">{value_format.format(value)}</text>')

    body.append(f'<text class="axis" x="{(left + right)/2:.2f}" y="{height - 18}" text-anchor="middle">{escape(x_label)}</text>')
    if note:
        body.append(f'<text class="note" x="40" y="{height - 18}">{escape(note)}</text>')
    write_svg(path, width, height, body)


def grouped_bar_chart(
    *,
    path: Path,
    title: str,
    subtitle: str,
    group_labels: list[str],
    series_labels: list[str],
    values: list[list[float]],
    colors: list[str],
    y_label: str,
    note: str | None = None,
    width: int = 1080,
    height: int = 620,
) -> None:
    body: list[str] = []
    add_title(body, title, subtitle)
    left, top, right, bottom = 90, 120, width - 50, height - 105
    plot_w = right - left
    plot_h = bottom - top
    flat = [item for row in values for item in row]
    max_value = max(flat) * 1.15
    if max_value <= 0:
        max_value = 1.0

    for step in range(6):
        y = top + plot_h * step / 5
        value = max_value * (1 - step / 5)
        body.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}"/>')
        body.append(f'<text class="axis" x="{left - 8}" y="{y + 4:.2f}" text-anchor="end">{value:.1f}</text>')

    draw_axes_frame(body, left, top, right, bottom)
    group_slot = plot_w / max(1, len(group_labels))
    inner_gap = 10
    series_count = len(series_labels)
    total_bar_w = group_slot - 30
    bar_w = max(12, min(42, (total_bar_w - inner_gap * (series_count - 1)) / max(1, series_count)))

    for g_idx, group in enumerate(group_labels):
        start_x = left + group_slot * g_idx + (group_slot - (bar_w * series_count + inner_gap * (series_count - 1))) / 2
        for s_idx, series in enumerate(series_labels):
            value = values[g_idx][s_idx]
            x = start_x + s_idx * (bar_w + inner_gap)
            h = (value / max_value) * plot_h
            y = bottom - h
            color = colors[s_idx % len(colors)]
            body.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{h:.2f}" rx="5" fill="{color}"/>')
            body.append(f'<text class="small" x="{x + bar_w/2:.2f}" y="{y - 6:.2f}" text-anchor="middle">{value:.2f}</text>')
        body.append(f'<text class="label" x="{left + group_slot * g_idx + group_slot/2:.2f}" y="{bottom + 22:.2f}" text-anchor="middle">{escape(group)}</text>')

    legend_x = left
    legend_y = height - 58
    for idx, label in enumerate(series_labels):
        x = legend_x + idx * 190
        body.append(f'<rect x="{x}" y="{legend_y - 11}" width="18" height="18" rx="3" fill="{colors[idx % len(colors)]}"/>')
        body.append(f'<text class="small" x="{x + 26}" y="{legend_y + 2}">{escape(label)}</text>')

    body.append(f'<text class="axis" x="{left - 58}" y="{top - 16}" text-anchor="start">{escape(y_label)}</text>')
    if note:
        body.append(f'<text class="note" x="40" y="{height - 18}">{escape(note)}</text>')
    write_svg(path, width, height, body)


def histogram_chart(
    *,
    path: Path,
    title: str,
    subtitle: str,
    values: list[float],
    bins: int,
    x_label: str,
    note: str | None = None,
    width: int = 980,
    height: int = 560,
) -> None:
    body: list[str] = []
    add_title(body, title, subtitle)
    left, top, right, bottom = 90, 110, width - 40, height - 90
    plot_w = right - left
    plot_h = bottom - top
    min_v = min(values)
    max_v = max(values)
    if max_v <= min_v:
        max_v = min_v + 1e-6
    step = (max_v - min_v) / bins
    counts = [0] * bins
    for value in values:
        idx = int((value - min_v) / step) if step > 0 else 0
        idx = min(bins - 1, max(0, idx))
        counts[idx] += 1
    max_count = max(counts) * 1.15
    if max_count <= 0:
        max_count = 1.0

    for grid_idx in range(6):
        y = top + plot_h * grid_idx / 5
        count = max_count * (1 - grid_idx / 5)
        body.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{right}" y2="{y:.2f}"/>')
        body.append(f'<text class="axis" x="{left - 8}" y="{y + 4:.2f}" text-anchor="end">{count:.0f}</text>')

    draw_axes_frame(body, left, top, right, bottom)
    bar_w = plot_w / bins
    for idx, count in enumerate(counts):
        x = left + idx * bar_w + 1
        h = (count / max_count) * plot_h
        y = bottom - h
        body.append(f'<rect x="{x:.2f}" y="{y:.2f}" width="{max(1, bar_w - 2):.2f}" height="{h:.2f}" fill="#2563eb" rx="3"/>')
    for tick_idx in range(bins + 1):
        value = min_v + (max_v - min_v) * tick_idx / bins
        x = left + plot_w * tick_idx / bins
        body.append(f'<text class="axis" x="{x:.2f}" y="{bottom + 20}" text-anchor="middle">{value:.2f}</text>')

    body.append(f'<text class="axis" x="{(left + right)/2:.2f}" y="{height - 22}" text-anchor="middle">{escape(x_label)}</text>')
    if note:
        body.append(f'<text class="note" x="40" y="{height - 22}">{escape(note)}</text>')
    write_svg(path, width, height, body)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def make_encoder_chart() -> None:
    df = pd.read_csv(ROOT.parent / "encoder-ML testing" / "out_benchmark" / "benchmark_results_20260305_180830.csv")
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
        note="Lower is better. The best setting was char_tfidf + Ridge.",
        value_format="{:.4f}",
        height=520,
    )


def make_baseline_generation_chart() -> None:
    df = pd.read_csv(ROOT / "outputs" / "paper" / "requested_tables" / "baseline_generation_table.csv")
    metrics = [
        "mean_content_fit",
        "mean_emotional_appropriateness",
        "mean_style_match",
        "mean_naturalness",
        "mean_overall_quality",
    ]
    series_labels = ["content_fit", "emotion_fit", "style_match", "naturalness", "overall"]
    group_labels = [nice_condition_label(name) for name in df["condition"].tolist()]
    values = [[float(row[m]) for m in metrics] for _, row in df.iterrows()]
    grouped_bar_chart(
        path=FIG_DIR / "baseline_generation_scores.svg",
        title="Generation Quality Comparison",
        subtitle="LLM-judge 5-point scores across response-generation conditions",
        group_labels=group_labels,
        series_labels=series_labels,
        values=values,
        colors=["#2563eb", "#14b8a6", "#f59e0b", "#8b5cf6", "#ef4444"],
        y_label="score",
        note="direct and stim_only outperformed the current EmoNet full prompt.",
    )


def make_predictor_chart() -> None:
    payload = load_json(ROOT / "outputs" / "paper" / "requested_tables" / "baseline_predictor_table.json")
    order = ["mean_baseline", "stim_only_ridge", "text_tfidf_ridge", "emonet_z64_ridge"]
    labels = [nice_condition_label(key) for key in order]
    values = [float(payload[key]["decoder_mae_mean"]) if key != "mean_baseline" else float(payload[key]["decoder_mae_mean"]) for key in order]
    baseline = float(payload["mean_baseline"]["decoder_mae_mean"])
    bar_chart_vertical(
        path=FIG_DIR / "predictor_mae_comparison.svg",
        title="z-to-s Predictor Comparison",
        subtitle="Validation MAE across baseline predictors and EmoNet z64",
        labels=labels,
        values=values,
        colors=["#9ca3af", "#14b8a6", "#60a5fa", "#2563eb"],
        y_label="validation MAE",
        note="Lower is better. None of the current predictors beat the mean baseline.",
        reference=baseline,
        reference_label="mean baseline",
        value_format="{:.4f}",
    )


def make_neuron_ablation_chart() -> None:
    df = pd.read_csv(ROOT / "outputs" / "paper" / "requested_tables" / "neuron_function_ablation_table.csv")
    labels = [nice_condition_label(name) for name in df["name"].tolist()]
    values = df["decoder_mae_mean"].astype(float).tolist()
    baseline = float(df["baseline_mae_mean"].astype(float).iloc[0])
    bar_chart_vertical(
        path=FIG_DIR / "neuron_ablation_mae.svg",
        title="Neuron Function Ablation",
        subtitle="Validation MAE after removing each dynamics component",
        labels=labels,
        values=values,
        colors=["#1d4ed8", "#3b82f6", "#60a5fa", "#93c5fd", "#0f766e", "#14b8a6"],
        y_label="validation MAE",
        note="Lower is better. Several removals performed similarly to the full model.",
        reference=baseline,
        reference_label="mean baseline",
        value_format="{:.4f}",
    )


def make_zdim_chart() -> None:
    df = pd.read_csv(ROOT / "outputs" / "paper" / "requested_tables" / "z_size_ablation_table.csv")
    labels = [nice_condition_label(name) for name in df["name"].tolist()]
    values = df["decoder_mae_mean"].astype(float).tolist()
    baseline = float(df["baseline_mae_mean"].astype(float).iloc[0])
    bar_chart_vertical(
        path=FIG_DIR / "zdim_ablation_mae.svg",
        title="Latent Dimension Ablation",
        subtitle="Validation MAE for z sizes 32, 64, and 128",
        labels=labels,
        values=values,
        colors=["#14b8a6", "#2563eb", "#7c3aed"],
        y_label="validation MAE",
        note="Lower is better. z=32 was the least harmful among the tested sizes.",
        reference=baseline,
        reference_label="mean baseline",
        value_format="{:.4f}",
    )


def make_consistency_chart() -> None:
    df = pd.read_csv(ROOT / "outputs" / "llm" / "llm_subset_labeled_200_ollama.csv")
    values = pd.to_numeric(df["consistency_l1"], errors="coerce").dropna().tolist()
    histogram_chart(
        path=FIG_DIR / "style_consistency_histogram.svg",
        title="Style Label Consistency",
        subtitle="Distribution of mean absolute difference between s and s_hat",
        values=values,
        bins=12,
        x_label="consistency L1",
        note="200 labeled samples, keep threshold = 0.12",
    )


def make_style_bias_chart() -> None:
    payload = load_json(ROOT / "outputs" / "paper" / "paper_metrics_snapshot.json")
    axes = payload["style_bias"]["interesting_axes"]
    labels = list(axes.keys())
    values = [float(axes[key]) for key in labels]
    colors = ["#ef4444" if v < 0.2 else "#14b8a6" if v > 0.75 else "#3b82f6" for v in values]
    bar_chart_horizontal(
        path=FIG_DIR / "style_bias_axes.svg",
        title="Style Axis Bias",
        subtitle="Mean values on selected style axes for kept samples",
        labels=labels,
        values=values,
        colors=colors,
        x_label="mean axis value",
        note="The current style space is heavily biased toward calm, soft, and cooperative responses.",
        value_format="{:.3f}",
        height=580,
    )


def make_branch_distribution_chart() -> None:
    df = pd.read_csv(ROOT / "outputs" / "z" / "out_z_training.csv", usecols=["dominant_branch_len"])
    counts = df["dominant_branch_len"].value_counts().sort_index()
    labels = [str(int(idx)) for idx in counts.index.tolist()]
    total = float(counts.sum())
    values = [(count / total) * 100.0 for count in counts.tolist()]
    bar_chart_vertical(
        path=FIG_DIR / "dominant_branch_length_distribution.svg",
        title="Dominant Branch Length Distribution",
        subtitle="Percentage of samples by dominant-branch length",
        labels=labels,
        values=values,
        colors=["#2563eb" if label == "1" else "#93c5fd" for label in labels],
        y_label="sample percentage (%)",
        note="Length-1 branches dominate the current pipeline.",
        value_format="{:.2f}",
        width=1100,
        height=560,
    )


def main() -> None:
    make_encoder_chart()
    make_baseline_generation_chart()
    make_predictor_chart()
    make_neuron_ablation_chart()
    make_zdim_chart()
    make_consistency_chart()
    make_style_bias_chart()
    make_branch_distribution_chart()
    print(json.dumps({"figure_dir": str(FIG_DIR), "count": len(list(FIG_DIR.glob("*.svg")))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
