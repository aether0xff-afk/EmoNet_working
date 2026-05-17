from __future__ import annotations

import json
from pathlib import Path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "poster_assets"
OUT.mkdir(parents=True, exist_ok=True)

FONT_PATHS = [
    Path(r"C:\Windows\Fonts\malgunbd.ttf"),
    Path(r"C:\Windows\Fonts\NotoSansKR-VF.ttf"),
    Path(r"C:\Windows\Fonts\malgun.ttf"),
]
FONT_PATH = next(p for p in FONT_PATHS if p.exists())
fm.fontManager.addfont(str(FONT_PATH))
FONT_NAME = fm.FontProperties(fname=str(FONT_PATH)).get_name()
plt.rcParams["font.family"] = FONT_NAME
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.titleweight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["axes.linewidth"] = 2.4

BLUE = "#0B55A0"
DARK = "#111827"
GRAY = "#6B7280"
LIGHT = "#F8FAFC"
PALETTE = ["#2563EB", "#16A34A", "#F97316", "#9333EA", "#DC2626", "#0891B2"]


def save(fig, name: str) -> None:
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def box(ax, x, y, w, h, text, fc="#EFF6FF", ec=BLUE, size=15, weight="bold", color=DARK):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.025,rounding_size=0.04",
        linewidth=3.2,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size, weight=weight, color=color)
    return patch


def arrow(ax, x1, y1, x2, y2, color=GRAY, lw=2.2):
    arr = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=22, linewidth=max(lw, 3.2), color=color)
    ax.add_patch(arr)


def title(ax, text, subtitle=None):
    ax.text(0.02, 0.94, text, transform=ax.transAxes, ha="left", va="top", fontsize=24, weight="bold", color=DARK)
    if subtitle:
        ax.text(0.02, 0.86, subtitle, transform=ax.transAxes, ha="left", va="top", fontsize=15, weight="bold", color=GRAY)


def fig_v1_pipeline():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.8)
    ax.axis("off")
    title(ax, "v1 결과: emotion_z_pipeline", "텍스트를 감정 벡터 z로 바꾸는 초기 흐름")

    labels = [
        ("입력", "문장"),
        ("인코더", "자극값"),
        ("계산망", "감정반응"),
        ("기록", "시간변화"),
        ("요약", "GRU"),
        ("출력", "z"),
    ]
    xs = [0.45, 2.45, 4.45, 6.45, 8.45, 10.45]
    for i, (head, sub) in enumerate(labels):
        box(ax, xs[i], 2.8, 1.35, 1.1, f"{head}\n{sub}", fc=["#EFF6FF", "#ECFDF5", "#FFF7ED", "#F5F3FF", "#FEF2F2", "#E0F2FE"][i], size=16)
        if i < len(labels) - 1:
            arrow(ax, xs[i] + 1.35, 3.35, xs[i + 1], 3.35)

    ax.text(0.75, 1.35, "의미", fontsize=19, weight="bold", color=BLUE)
    ax.text(0.75, 0.95, "감정을 숫자로 바꾸는 첫 실험", fontsize=17, weight="bold", color=DARK)
    ax.text(6.0, 1.35, "한계", fontsize=19, weight="bold", color="#DC2626")
    ax.text(6.0, 0.95, "시간 변화 설명이 부족함", fontsize=17, weight="bold", color=DARK)
    save(fig, "v1_emotion_z_pipeline_flow")


def fig_v2_modules():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.8)
    ax.axis("off")
    title(ax, "v2 결과: 모듈형 EmoNet", "기능을 나누어 연구용 골격을 만든 단계")

    modules = [
        (0.55, 3.35, "입력\n인코더", "#EFF6FF"),
        (3.15, 3.35, "감정\n계산", "#ECFDF5"),
        (5.75, 3.35, "경로\n추적", "#FFF7ED"),
        (8.35, 3.35, "흐름\n요약", "#F5F3FF"),
        (0.55, 1.65, "군집\n분석", "#E0F2FE"),
        (3.15, 1.65, "연결\n조정", "#FEF2F2"),
        (5.75, 1.65, "스타일\n변환", "#F0FDFA"),
        (8.35, 1.65, "응답\n제약", "#FAF5FF"),
    ]
    for x, y, text, fc in modules:
        box(ax, x, y, 1.75, 0.95, text, fc=fc, size=17)
    for a, b in [(0, 1), (1, 2), (2, 3), (4, 5), (5, 6), (6, 7)]:
        x1, y1, _, _ = modules[a]
        x2, y2, _, _ = modules[b]
        arrow(ax, x1 + 1.75, y1 + 0.48, x2, y2 + 0.48)

    ax.text(0.55, 0.82, "핵심 변화", fontsize=19, weight="bold", color=BLUE)
    ax.text(2.05, 0.82, "입력, 감정 계산, 경로, 기록, 스타일을 분리", fontsize=17, weight="bold", color=DARK)
    save(fig, "v2_module_architecture")


def fig_v2_pytorch_flow():
    fig, ax = plt.subplots(figsize=(14, 7.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6.4)
    ax.axis("off")
    title(ax, "v2 결과: 반복 계산, z, s", "감정 흐름을 만들고 응답 스타일로 바꾸는 구조")

    box(ax, 0.35, 3.7, 1.35, 0.9, "입력\n문장", fc="#EFF6FF", size=16)
    box(ax, 2.1, 3.7, 1.35, 0.9, "제어값\nh_t", fc="#ECFDF5", size=16)
    box(ax, 4.05, 3.2, 2.1, 1.95, "반복 계산\n\n활성 activity\n기억 memory\n군집 cluster\nbranch 경로", fc="#FFF7ED", size=13)
    box(ax, 6.95, 3.7, 1.55, 0.9, "history\n+ branch", fc="#F5F3FF", size=14)
    box(ax, 9.05, 4.45, 1.35, 0.85, "z\n감정요약", fc="#E0F2FE", size=15)
    box(ax, 9.05, 3.2, 1.35, 0.85, "s\n스타일", fc="#F0FDFA", size=15)
    box(ax, 10.75, 3.7, 1.05, 0.9, "응답\n제약", fc="#FAF5FF", size=14)

    arrow(ax, 1.7, 4.15, 2.1, 4.15)
    arrow(ax, 3.45, 4.15, 4.05, 4.15)
    arrow(ax, 6.15, 4.15, 6.95, 4.15)
    arrow(ax, 8.5, 4.15, 9.05, 4.86)
    arrow(ax, 9.72, 4.45, 9.72, 4.05, "#16A34A")
    arrow(ax, 10.4, 3.62, 10.75, 4.15)

    ax.text(0.55, 1.85, "z", fontsize=22, weight="bold", color=BLUE)
    ax.text(0.95, 1.9, "시간별 내부 상태와 대표 경로를 압축한 감정 흐름 요약 벡터", fontsize=15, weight="bold", color=DARK)
    ax.text(0.55, 1.1, "s", fontsize=22, weight="bold", color="#16A34A")
    ax.text(0.95, 1.15, "z를 말투, 강도, 부드러움, 구체성 같은 응답 스타일로 변환", fontsize=15, weight="bold", color=DARK)
    save(fig, "v2_pytorch_model_flow")


def fig_v3_experiment_results():
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [1.25, 0.9]})
    fig.suptitle("v3 결과: 실험 비교", fontsize=24, weight="bold", x=0.02, ha="left", y=0.97)
    fig.text(0.02, 0.89, "여러 조건의 생성 결과를 점수로 비교", fontsize=15, weight="bold", color=GRAY)

    path = ROOT / "v3" / "outputs" / "experiments" / "paper_matrix_current_calref_v1_gpt54_scored.csv"
    df = pd.read_csv(path)
    metrics = ["content_fit", "emotional_appropriateness", "style_match", "naturalness", "overall_quality"]
    clean = df[df["status"].eq("ok")].dropna(subset=metrics)
    means = clean.groupby("condition")[metrics].mean().sort_values("overall_quality", ascending=False).head(5)

    ax = axes[0]
    means[metrics].plot(kind="bar", ax=ax, color=PALETTE[:5], width=0.78)
    ax.set_ylim(0, 5)
    ax.set_ylabel("평균 점수", fontsize=15, weight="bold")
    ax.set_xlabel("")
    ax.set_title("조건별 평균 점수", fontsize=18, weight="bold")
    ax.legend(["내용", "감정", "스타일", "자연스러움", "전체"], fontsize=12, ncols=2)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", rotation=20, labelsize=13, width=2)
    ax.tick_params(axis="y", labelsize=13, width=2)

    ax = axes[1]
    status_counts = df["status"].value_counts()
    ax.pie(status_counts.values, labels=status_counts.index, autopct="%1.0f%%", colors=["#16A34A", "#F97316", "#DC2626"], startangle=90)
    ax.set_title("실험 처리 상태", fontsize=18, weight="bold")
    ax.text(0.0, -1.35, f"총 {len(df)}개 응답 비교", ha="center", fontsize=15, weight="bold", color=DARK)
    fig.tight_layout(rect=[0, 0, 1, 0.83])
    save(fig, "v3_experiment_results_summary")


def fig_v31_trace_core():
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [1.12, 1]})
    fig.suptitle("v3.1 결과: trace = 감정 상태", fontsize=24, weight="bold", x=0.02, ha="left", y=0.97)
    fig.text(0.02, 0.89, "핵심 주장: trace는 감정 상태 자체의 기록", fontsize=15, weight="bold", color=GRAY)

    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    box(ax, 0.25, 3.25, 1.55, 0.9, "입력\n상황", fc="#EFF6FF", size=16)
    box(ax, 2.55, 3.25, 1.55, 0.9, "trace\n기록", fc="#ECFDF5", size=16)
    box(ax, 4.85, 3.25, 1.55, 0.9, "감정\n상태", fc="#FFF7ED", size=16)
    box(ax, 7.15, 3.25, 1.55, 0.9, "반응\n제약", fc="#F5F3FF", size=16)
    for x in [1.9, 4.1, 6.3]:
        arrow(ax, x, 3.7, x + 0.6, 3.7)
    ax.text(0.6, 2.1, "이전 관점", fontsize=18, weight="bold", color="#DC2626")
    ax.text(0.6, 1.68, "감정 = 라벨", fontsize=16, weight="bold")
    ax.text(5.1, 2.1, "v3.1 관점", fontsize=18, weight="bold", color=BLUE)
    ax.text(5.1, 1.68, "감정 = 변화 기록", fontsize=16, weight="bold")

    ax = axes[1]
    sweep = pd.read_csv(ROOT / "v3.1" / "outputs" / "neural_trace_dynamics_sweep_v1" / "dynamics_sweep_summary.csv")
    top = sweep.sort_values("mean_branch_len", ascending=False).head(6)
    ax.barh(top["config"], top["mean_branch_len"], color="#2563EB")
    ax.set_xlabel("평균 branch 길이", fontsize=15, weight="bold")
    ax.set_title("trace 지속성", fontsize=18, weight="bold")
    ax.grid(axis="x", alpha=0.25)
    ax.invert_yaxis()
    ax.tick_params(axis="both", labelsize=13, width=2)

    fig.tight_layout(rect=[0, 0, 1, 0.83])
    save(fig, "v31_trace_as_emotion_core")


def fig_v4_evaluation():
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [1.25, 0.9]})
    fig.suptitle("v4 결과: 평가", fontsize=24, weight="bold", x=0.02, ha="left", y=0.97)
    fig.text(0.02, 0.89, "episode_trace_v3와 stim_only 비교", fontsize=15, weight="bold", color=GRAY)

    paired = pd.read_csv(ROOT / "v4" / "outputs" / "experiments" / "superiority_targeted_v1" / "paired_vs_stim" / "paired_overall.csv")
    selected = paired[
        paired["condition"].eq("episode_trace_v3")
        & paired["baseline"].eq("stim_only")
        & paired["metric"].isin(["mean_total", "appraisal_fidelity", "raw_affect_preservation", "anti_softening", "action_tendency_fit", "emotional_specificity"])
    ].copy()
    labels = {
        "mean_total": "전체",
        "appraisal_fidelity": "상황 이해",
        "raw_affect_preservation": "감정 보존",
        "anti_softening": "과잉 완화 억제",
        "action_tendency_fit": "행동 경향",
        "emotional_specificity": "감정 구체성",
    }
    selected["label"] = selected["metric"].map(labels)

    ax = axes[0]
    x = np.arange(len(selected))
    ax.bar(x, selected["delta_mean"], color="#2563EB")
    ax.errorbar(
        x,
        selected["delta_mean"],
        yerr=[selected["delta_mean"] - selected["bootstrap_ci_low"], selected["bootstrap_ci_high"] - selected["delta_mean"]],
        fmt="none",
        ecolor=DARK,
        capsize=4,
        linewidth=1.4,
    )
    ax.axhline(0, color=DARK, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(selected["label"], rotation=18)
    ax.set_ylabel("평균 점수 차이", fontsize=15, weight="bold")
    ax.set_title("점수 차이", fontsize=18, weight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="x", labelsize=13, width=2)
    ax.tick_params(axis="y", labelsize=13, width=2)

    ax = axes[1]
    row = paired[paired["condition"].eq("episode_trace_v3") & paired["metric"].eq("mean_total")].iloc[0]
    counts = [row["wins"], row["ties"], row["losses"]]
    ax.bar(["Win", "Tie", "Loss"], counts, color=["#16A34A", "#9CA3AF", "#DC2626"])
    ax.set_title("전체 점수 승/무/패", fontsize=18, weight="bold")
    ax.set_ylabel("paired count", fontsize=15, weight="bold")
    for i, v in enumerate(counts):
        ax.text(i, v + 1, str(int(v)), ha="center", fontsize=18, weight="bold")
    ax.text(1, max(counts) * 0.55, f"win rate\n{row['win_rate']:.1%}", ha="center", va="center", fontsize=20, weight="bold", color=BLUE)
    ax.grid(axis="y", alpha=0.25)
    ax.tick_params(axis="both", labelsize=14, width=2)

    fig.tight_layout(rect=[0, 0, 1, 0.83])
    save(fig, "v4_evaluation_results")


def main():
    fig_v1_pipeline()
    fig_v2_modules()
    fig_v2_pytorch_flow()
    fig_v3_experiment_results()
    fig_v31_trace_core()
    fig_v4_evaluation()
    print(OUT)
    for p in sorted(OUT.glob("*.png")):
        print(p.name)


if __name__ == "__main__":
    main()
