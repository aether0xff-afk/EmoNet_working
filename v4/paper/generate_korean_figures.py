from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "figures_full_ko"
OUT.mkdir(parents=True, exist_ok=True)

REGULAR = r"C:\Windows\Fonts\malgun.ttf"
BOLD = r"C:\Windows\Fonts\malgunbd.ttf"
FP = font_manager.FontProperties(fname=REGULAR)
FP_BOLD = font_manager.FontProperties(fname=BOLD)
font_manager.fontManager.addfont(REGULAR)
font_manager.fontManager.addfont(BOLD)

mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["font.family"] = FP.get_name()
mpl.rcParams["figure.dpi"] = 150


def set_korean_axes(ax):
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(FP)
    ax.xaxis.label.set_fontproperties(FP)
    ax.yaxis.label.set_fontproperties(FP)
    ax.title.set_fontproperties(FP_BOLD)
    leg = ax.get_legend()
    if leg:
        for text in leg.get_texts():
            text.set_fontproperties(FP)


def save_version_timeline():
    fig, ax = plt.subplots(figsize=(11, 3.8))
    versions = ["v1", "v2", "v3", "v3.1", "v4"]
    xs = np.arange(len(versions))
    labels = [
        "감정 벡터 z\n초기 동역학/GUI",
        "모듈형 MVP\ntrait, branch, z→s",
        "지배 경로 추출\n붕괴 보정/논문 지표",
        "trace-as-emotion\n표현+인과 검증",
        "episode 응답\ntargeted 우월성 평가",
    ]
    ax.plot(xs, [0] * len(xs), color="#2f5d62", lw=3)
    ax.scatter(
        xs,
        [0] * len(xs),
        s=520,
        color=["#4f7cac", "#6a994e", "#bc6c25", "#9d4edd", "#d62828"],
        zorder=3,
    )
    for x, v, lab in zip(xs, versions, labels):
        ax.text(x, 0.18, v, ha="center", va="bottom", fontsize=15, fontproperties=FP_BOLD)
        ax.text(x, -0.18, lab, ha="center", va="top", fontsize=10, fontproperties=FP)
    ax.set_ylim(-0.75, 0.65)
    ax.set_xlim(-0.5, len(xs) - 0.5)
    ax.axis("off")
    ax.set_title("EmoNet 연구 버전의 개념 발전", fontsize=16, pad=12, fontproperties=FP_BOLD)
    fig.tight_layout()
    fig.savefig(OUT / "version_timeline.png", bbox_inches="tight")
    plt.close(fig)


def save_branch_recovery():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(["이전", "보정 후"], [1.0539, 70.4684], color=["#b56576", "#4f7cac"])
    axes[0].set_title("지배 흐름 평균 길이", fontproperties=FP_BOLD)
    axes[0].set_ylabel("길이", fontproperties=FP)
    for i, y in enumerate([1.0539, 70.4684]):
        axes[0].text(i, y + 2, f"{y:.2f}", ha="center", fontproperties=FP)

    axes[1].bar(["이전", "보정 후"], [0.9734, 0.0948], color=["#b56576", "#4f7cac"])
    axes[1].set_title("한 칸짜리 흐름 비율", fontproperties=FP_BOLD)
    axes[1].set_ylim(0, 1.05)
    for i, y in enumerate([0.9734, 0.0948]):
        axes[1].text(i, y + 0.03, f"{y:.4f}", ha="center", fontproperties=FP)
    for ax in axes:
        set_korean_axes(ax)
    fig.suptitle("v3 브랜치 붕괴 완화: 51,628개 export 기준", fontsize=15, fontproperties=FP_BOLD)
    fig.tight_layout()
    fig.savefig(OUT / "branch_collapse_recovery.png", bbox_inches="tight")
    plt.close(fig)


def save_style_bias():
    axes_names = [
        "부드러움",
        "차분함",
        "협조성",
        "긍정성",
        "따뜻함",
        "적대성",
        "원망",
        "절망",
        "불안정",
        "두려움",
        "수치심",
    ]
    vals = [0.9276, 0.9132, 0.9202, 0.9051, 0.7596, 0.0003, 0.0003, 0.0044, 0.0022, 0.0100, 0.0017]
    colors = ["#588157"] * 5 + ["#bc4749"] * 6
    fig, ax = plt.subplots(figsize=(10, 4.8))
    y = np.arange(len(axes_names))
    ax.barh(y, vals, color=colors)
    ax.set_yticks(y, axes_names, fontproperties=FP)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("평균 축 값", fontproperties=FP)
    ax.set_title("v3 스타일 목표 편향: 온건한 축은 높고 날것의 감정축은 낮음", fontproperties=FP_BOLD)
    for yi, v in zip(y, vals):
        ax.text(v + 0.015, yi, f"{v:.4f}", va="center", fontsize=9, fontproperties=FP)
    set_korean_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT / "style_bias_korean.png", bbox_inches="tight")
    plt.close(fig)


def save_v31_adaptive():
    metrics = [
        "감정 흐름 길이\n(50.475/80)",
        "활성 밀도\n(0.709)",
        "구조 분리도\n(0.239)",
        "균형 보정 신호\n(0.136)",
    ]
    vals = [50.475 / 80, 0.709412, 0.238547, 0.136426]
    fig, ax = plt.subplots(figsize=(9.2, 4.5))
    ax.bar(metrics, vals, color=["#277da1", "#43aa8b", "#f9c74f", "#f9844a"])
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("정규화 값", fontproperties=FP)
    ax.set_title("v3.1 후반부 적응형 활성 조절의 핵심 결과", fontproperties=FP_BOLD)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.025, f"{v:.3f}", ha="center", fontproperties=FP)
    set_korean_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT / "v31_adaptive_metrics.png", bbox_inches="tight")
    plt.close(fig)


def save_v31_causal():
    labels = ["전체", "강한 정보\n중립화", "방향 교란", "동일 응답\n무효 비교"]
    vals = [0.916667, 0.975, 0.775, 1.0]
    fig, ax = plt.subplots(figsize=(8.8, 4.5))
    ax.bar(labels, vals, color=["#577590", "#43aa8b", "#f8961e", "#90be6d"])
    ax.axhline(0.75, color="#444", ls="--", lw=1, label="주요 통과 기준 0.75")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("성공률 또는 tie 비율", fontproperties=FP)
    ax.set_title("v3.1 축-전용 눈가림 인과 확인 실험", fontproperties=FP_BOLD)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.03, f"{v:.3f}", ha="center", fontproperties=FP)
    ax.legend(prop=FP, frameon=False, loc="lower right")
    set_korean_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT / "v31_causal_results.png", bbox_inches="tight")
    plt.close(fig)


def save_v4_targeted():
    conditions = ["자극만", "에피소드\n추적", "에피소드\n추적 v3"]
    primary = [1.5923, 2.9896, 3.43]
    mean_total = [1.9542, 3.1521, 3.55]
    natural = [4.141, 4.3117, 4.4]
    x = np.arange(len(conditions))
    w = 0.25
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(x - w, primary, w, label="주 지표 평균", color="#4f7cac")
    ax.bar(x, mean_total, w, label="전체 평균", color="#f9844a")
    ax.bar(x + w, natural, w, label="자연스러움", color="#43aa8b")
    ax.set_xticks(x, conditions, fontproperties=FP)
    ax.set_ylim(0, 5)
    ax.set_ylabel("점수", fontproperties=FP)
    ax.set_title("v4 targeted episode-sensitive 입력에서의 조건별 점수", fontproperties=FP_BOLD)
    ax.legend(prop=FP, frameon=False)
    set_korean_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT / "v4_targeted_scores.png", bbox_inches="tight")
    plt.close(fig)


def save_v4_paired():
    labels = ["v3 vs 자극만\n승/무/패", "기존 vs 자극만\n승/무/패", "v3 vs 기존\n승/무/패"]
    wins = [70, 69, 41]
    ties = [3, 6, 6]
    losses = [5, 2, 30]
    fig, ax = plt.subplots(figsize=(9.5, 4.7))
    y = np.arange(len(labels))
    ax.barh(y, wins, color="#43aa8b", label="승")
    ax.barh(y, ties, left=wins, color="#f9c74f", label="무")
    ax.barh(y, losses, left=np.array(wins) + np.array(ties), color="#bc4749", label="패")
    ax.set_yticks(y, labels, fontproperties=FP)
    ax.invert_yaxis()
    ax.set_xlabel("쌍 비교 개수", fontproperties=FP)
    ax.set_title("v4 쌍대 우월성 평가", fontproperties=FP_BOLD)
    ax.legend(prop=FP, frameon=False)
    set_korean_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT / "v4_paired_win_tie_loss.png", bbox_inches="tight")
    plt.close(fig)


def save_general_vs_targeted_after():
    labels = ["일반 episode v2\nepisode_trace vs 자극만", "targeted v4\nepisode_trace_v3 vs 자극만"]
    delta = [-0.081416, 1.830769]
    ci_low = [-0.226549, 1.566667]
    ci_high = [0.056637, 2.089744]
    x = np.arange(len(labels))
    yerr = np.array([[d - lo for d, lo in zip(delta, ci_low)], [hi - d for d, hi in zip(delta, ci_high)]])
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    colors = ["#bc4749" if v < 0 else "#43aa8b" for v in delta]
    ax.bar(x, delta, color=colors, width=0.58)
    ax.errorbar(x, delta, yerr=yerr, fmt="none", color="#222", capsize=6, lw=1.3)
    ax.axhline(0, color="#333", lw=1)
    ax.set_xticks(x, labels, fontproperties=FP)
    ax.set_ylabel("자극만 대비 평균 차이", fontproperties=FP)
    ax.set_title("이전 일반 조건과 현재 targeted 조건의 결과 차이", fontproperties=FP_BOLD)
    for i, v in enumerate(delta):
        ax.text(i, v + (0.12 if v >= 0 else -0.18), f"{v:+.3f}", ha="center", fontproperties=FP)
    set_korean_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT / "general_vs_targeted_after.png", bbox_inches="tight")
    plt.close(fig)


def save_targeted_metric_deltas():
    metrics = ["평가 충실도", "날것의 정서 보존", "과도한 온건화 방지", "행동 경향 적합도", "감정 특이성"]
    stim = [1.5513, 1.4359, 1.6282, 1.9744, 1.3718]
    current = [3.5, 2.95, 3.7875, 3.7125, 3.2]
    delta = [c - s for c, s in zip(current, stim)]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    y = np.arange(len(metrics))
    ax.barh(y, delta, color="#277da1")
    ax.set_yticks(y, metrics, fontproperties=FP)
    ax.invert_yaxis()
    ax.set_xlabel("episode_trace_v3 - 자극만", fontproperties=FP)
    ax.set_title("현재 targeted 조건에서 온건화 편향이 완화된 지점", fontproperties=FP_BOLD)
    for yi, v in zip(y, delta):
        ax.text(v + 0.04, yi, f"+{v:.3f}", va="center", fontproperties=FP)
    set_korean_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT / "targeted_metric_deltas_after.png", bbox_inches="tight")
    plt.close(fig)


def save_bias_status_summary():
    labels = ["v3 스타일 목표\n긍정성", "v3 스타일 목표\n부드러움", "v4 targeted\nanti-softening", "v4 targeted\nraw affect 보존"]
    vals = [0.9051, 0.9276, 3.7875 / 5.0, 2.95 / 5.0]
    colors = ["#588157", "#588157", "#277da1", "#277da1"]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.bar(labels, vals, color=colors)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("0--1 환산 값", fontproperties=FP)
    ax.set_title("전역 스타일 편향은 남아 있고, targeted 응답에서는 일부 완화됨", fontproperties=FP_BOLD)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.025, f"{v:.3f}", ha="center", fontproperties=FP)
    set_korean_axes(ax)
    fig.tight_layout()
    fig.savefig(OUT / "bias_status_summary.png", bbox_inches="tight")
    plt.close(fig)


def draw_box(ax, xy, wh, text, fc="#f8f9fa", ec="#333", fontsize=10, bold=False):
    x, y = xy
    w, h = wh
    box = mpl.patches.FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.2,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontproperties=FP_BOLD if bold else FP,
        linespacing=1.25,
    )


def draw_arrow(ax, start, end, color="#333"):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="->", lw=1.6, color=color, shrinkA=4, shrinkB=4),
    )


def save_v1_pipeline():
    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis("off")
    nodes = [
        (0.25, "입력 문장", "#e9ecef"),
        (1.85, "감정/자극\n벡터", "#d8f3dc"),
        (3.55, "Emotion\nDynamicsNet", "#b7e4c7"),
        (5.45, "시간별\n히스토리", "#caf0f8"),
        (7.15, "GRU history\nencoder", "#ade8f4"),
        (9.0, "잠재 상태\nz", "#ffd6a5"),
        (10.55, "GUI 관찰\n상태 변화 확인", "#ffc8dd"),
    ]
    for x, label, color in nodes:
        draw_box(ax, (x, 1.05), (1.25, 0.85), label, fc=color, fontsize=9, bold=("GUI" in label))
    for i in range(len(nodes) - 1):
        draw_arrow(ax, (nodes[i][0] + 1.25, 1.48), (nodes[i + 1][0], 1.48))
    ax.text(
        6,
        2.45,
        "v1: 감정을 출력 라벨이 아니라 내부 동역학을 거친 잠재 상태로 보내는 첫 실행 파이프라인",
        ha="center",
        fontsize=14,
        fontproperties=FP_BOLD,
    )
    fig.tight_layout()
    fig.savefig(OUT / "v1_pipeline_gui.png", bbox_inches="tight")
    plt.close(fig)


def save_v2_module_structure():
    fig, ax = plt.subplots(figsize=(12, 5.3))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    modules = [
        (0.4, 4.1, "encoders\n텍스트→4차원\n제어 벡터", "#d8f3dc"),
        (2.7, 4.1, "dynamics\n뉴런망 전파", "#b7e4c7"),
        (5.0, 4.1, "branching\npath 기록", "#caf0f8"),
        (7.3, 4.1, "history\nencoder\n전역/지배 경로", "#ade8f4"),
        (9.7, 4.1, "tone\nregressor\nz→s", "#ffd6a5"),
        (9.7, 1.75, "prompt\ngenerator\nLLM 조건화", "#ffc8dd"),
        (7.3, 1.75, "style\nscorer\n응답 평가", "#ffafcc"),
        (2.7, 1.75, "clustering\nrewiring\n구조 조정", "#e9ecef"),
        (0.4, 1.75, "trait EMA\nmemory\n이전 상태 흔적", "#f1faee"),
    ]
    for x, y, label, color in modules:
        draw_box(ax, (x, y), (1.65, 0.95), label, fc=color, fontsize=9)
    arrows = [
        ((2.05, 4.58), (2.7, 4.58)),
        ((4.35, 4.58), (5.0, 4.58)),
        ((6.65, 4.58), (7.3, 4.58)),
        ((8.95, 4.58), (9.7, 4.58)),
        ((10.52, 4.1), (10.52, 2.7)),
        ((9.7, 2.22), (8.95, 2.22)),
        ((3.52, 2.7), (3.52, 4.1)),
        ((2.05, 2.22), (2.7, 2.22)),
        ((1.22, 2.7), (1.22, 4.1)),
    ]
    for s, e in arrows:
        draw_arrow(ax, s, e)
    ax.text(
        6,
        5.55,
        "v2: 하나의 스크립트에서 분해 가능한 모듈형 감정 시스템으로",
        ha="center",
        fontsize=14,
        fontproperties=FP_BOLD,
    )
    fig.tight_layout()
    fig.savefig(OUT / "v2_module_structure.png", bbox_inches="tight")
    plt.close(fig)


def save_v2_neuron_types():
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    draw_box(ax, (0.6, 2.05), (1.55, 0.9), "정서 자극\n입력", fc="#e9ecef", bold=True)
    draw_box(ax, (3.1, 3.45), (2.0, 0.9), "흥분성 뉴런\n감정 신호 확산", fc="#ffddd2")
    draw_box(ax, (3.1, 2.05), (2.0, 0.9), "억제성 뉴런\n과활성 억제", fc="#d0f4de")
    draw_box(ax, (3.1, 0.65), (2.0, 0.9), "조절성 뉴런\n민감도/방향 조정", fc="#cddafd")
    draw_box(ax, (6.4, 2.05), (2.3, 0.9), "균형 잡힌\n내부 활성 흐름", fc="#ffd6a5", bold=True)
    for y in [3.9, 2.5, 1.1]:
        draw_arrow(ax, (2.15, 2.5), (3.1, y))
        draw_arrow(ax, (5.1, y), (6.4, 2.5))
    ax.text(
        5,
        4.65,
        "v2 뉴런 타입: 확산, 억제, 조절을 분리해 감정 흐름을 구성",
        ha="center",
        fontsize=14,
        fontproperties=FP_BOLD,
    )
    fig.tight_layout()
    fig.savefig(OUT / "v2_neuron_types.png", bbox_inches="tight")
    plt.close(fig)


def save_emotion_definition_shift():
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4)
    ax.axis("off")
    stages = [
        (0.6, "v1/v2\n결과 벡터가 감정", "입력 벡터→동역학→z\n변형된 잠재 상태를\n감정으로 해석", "#d8f3dc"),
        (4.0, "v3\n대표 경로가 중요", "dominant branch\nz→s 연결\ncollapse 발견", "#caf0f8"),
        (7.4, "v3.1\ntrace 자체가 감정", "초기 벡터는 자극\n신경 활성 추적이\n감정 상태 표현 후보", "#ffd6a5"),
    ]
    for x, title, body, color in stages:
        draw_box(ax, (x, 1.15), (2.6, 1.55), f"{title}\n\n{body}", fc=color, fontsize=10, bold=False)
    draw_arrow(ax, (3.2, 1.93), (4.0, 1.93))
    draw_arrow(ax, (6.6, 1.93), (7.4, 1.93))
    ax.text(
        5.5,
        3.5,
        "감정 정의의 변화: 벡터 결과에서 내부 활성 과정으로",
        ha="center",
        fontsize=14,
        fontproperties=FP_BOLD,
    )
    fig.tight_layout()
    fig.savefig(OUT / "emotion_definition_shift.png", bbox_inches="tight")
    plt.close(fig)


def save_full_research_pipeline():
    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 5.2)
    ax.axis("off")
    top = [
        (0.4, "텍스트 입력\n사건/정서 단서", "#e9ecef"),
        (2.2, "자극화\n4차원/평가 단서", "#d8f3dc"),
        (4.0, "신경 동역학\n흥분/억제/조절", "#b7e4c7"),
        (5.9, "trace/branch\n활성 경로 기록", "#caf0f8"),
        (7.8, "z와 episode\n압축/해석", "#ade8f4"),
        (9.8, "응답 조건화\ns/prompt", "#ffd6a5"),
        (11.2, "평가\n지표/쌍비교", "#ffc8dd"),
    ]
    for x, label, color in top:
        draw_box(ax, (x, 3.05), (1.25, 0.9), label, fc=color, fontsize=8.8)
    for i in range(len(top) - 1):
        draw_arrow(ax, (top[i][0] + 1.25, 3.5), (top[i + 1][0], 3.5))
    bottom = [
        (1.15, "v1\n실행 파이프라인/GUI"),
        (3.1, "v2\n모듈화/branch 기록"),
        (5.05, "v3\n붕괴 발견/보정"),
        (7.0, "v3.1\ntrace 표현 검증"),
        (9.05, "v4\nepisode 응답 평가"),
    ]
    for x, label in bottom:
        draw_box(ax, (x, 1.1), (1.45, 0.85), label, fc="#f8f9fa", fontsize=9)
    for x, _ in bottom:
        draw_arrow(ax, (x + 0.72, 1.95), (x + 0.72, 3.05), color="#555")
    ax.text(6.25, 4.65, "EmoNet 전체 연구 파이프라인과 버전별 역할", ha="center", fontsize=15, fontproperties=FP_BOLD)
    fig.tight_layout()
    fig.savefig(OUT / "full_research_pipeline.png", bbox_inches="tight")
    plt.close(fig)


def save_experiment_before_after_map():
    fig, ax = plt.subplots(figsize=(12.5, 6.0))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 6)
    ax.axis("off")
    rows = [
        ("v1", "감정을 라벨로만 보지 않음", "내부 동역학+z+GUI", "내부 상태화 가능성"),
        ("v2", "원인 분리 어려움", "모듈화+뉴런 타입+branch 기록", "진단 가능한 구조"),
        ("v3", "branch collapse", "reference config 보정", "흐름 길이 회복"),
        ("v3.1", "trace가 감정인지 불명확", "표현 지표+인과 조작", "trace-as-emotion 근거"),
        ("v4", "일반 조건 우월성 실패", "targeted episode 평가", "좁고 방어 가능한 claim"),
    ]
    headers = ["버전", "이전 문제/질문", "행동", "이후 의미"]
    xs = [0.35, 1.55, 5.0, 8.55]
    ws = [0.85, 3.05, 3.1, 3.45]
    for x, w, h in zip(xs, ws, headers):
        draw_box(ax, (x, 5.0), (w, 0.55), h, fc="#343a40", ec="#343a40", fontsize=9, bold=True)
        ax.text(x + w / 2, 5.275, h, color="white", ha="center", va="center", fontsize=9, fontproperties=FP_BOLD)
    for r, row in enumerate(rows):
        y = 4.15 - r * 0.78
        for x, w, text in zip(xs, ws, row):
            draw_box(ax, (x, y), (w, 0.56), text, fc="#f8f9fa" if r % 2 == 0 else "#edf6f9", fontsize=8.5, bold=(x == xs[0]))
    ax.text(6.25, 5.78, "모든 실험의 이전과 이후: 문제, 행동, 의미", ha="center", fontsize=15, fontproperties=FP_BOLD)
    fig.tight_layout()
    fig.savefig(OUT / "experiment_before_after_map.png", bbox_inches="tight")
    plt.close(fig)


def save_metric_taxonomy():
    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.8)
    ax.axis("off")
    groups = [
        (0.55, "동역학 지표", ["흐름 길이", "한 칸짜리 비율", "활성 밀도", "늦은 점화"], "#d8f3dc"),
        (3.35, "표현 지표", ["구조 분리도", "균형 보정 신호", "축별 정보", "trace 안정성"], "#caf0f8"),
        (6.15, "인과 지표", ["방향 교란", "정보 중립화", "축 전용 judge", "성공률"], "#ffd6a5"),
        (8.95, "생성 지표", ["평가 충실도", "원초 정서", "온건화 방지", "행동 경향"], "#ffc8dd"),
    ]
    for x, title, items, color in groups:
        draw_box(ax, (x, 3.95), (2.25, 0.62), title, fc=color, fontsize=10, bold=True)
        for i, item in enumerate(items):
            draw_box(ax, (x, 3.15 - i * 0.62), (2.25, 0.46), item, fc="#f8f9fa", fontsize=8.8)
    ax.text(6, 5.25, "EmoNet 지표 체계: trace가 생겼는지, 의미가 있는지, 응답에 도움이 되는지", ha="center", fontsize=14, fontproperties=FP_BOLD)
    fig.tight_layout()
    fig.savefig(OUT / "metric_taxonomy.png", bbox_inches="tight")
    plt.close(fig)


def save_causal_validation_workflow():
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4.8)
    ax.axis("off")
    nodes = [
        (0.5, "원본 trace\n감정축 정보", "#caf0f8"),
        (2.4, "방향 교란\n특정 축 흔들기", "#ffd6a5"),
        (4.4, "강한 정보\n중립화", "#ffddd2"),
        (6.35, "축 전용\n눈가림 judge", "#d8f3dc"),
        (8.45, "성공/실패\n판정", "#e9ecef"),
        (10.25, "trace가\n조작 가능한가", "#ffc8dd"),
    ]
    for x, label, color in nodes:
        draw_box(ax, (x, 2.1), (1.35, 0.9), label, fc=color, fontsize=9)
    for i in range(len(nodes) - 1):
        draw_arrow(ax, (nodes[i][0] + 1.35, 2.55), (nodes[i + 1][0], 2.55))
    draw_box(ax, (2.4, 0.75), (2.2, 0.62), "축 정보가 없다면\njudge가 구분하기 어려움", fc="#f8f9fa", fontsize=8.5)
    draw_box(ax, (6.2, 0.75), (2.4, 0.62), "구분 가능하면\ntrace 내부 정보의 증거", fc="#f8f9fa", fontsize=8.5)
    draw_arrow(ax, (3.5, 1.37), (6.8, 2.1), color="#666")
    ax.text(6, 4.25, "v3.1 인과 검증 흐름: 관찰 지표에서 조작 가능한 표현으로", ha="center", fontsize=14, fontproperties=FP_BOLD)
    fig.tight_layout()
    fig.savefig(OUT / "causal_validation_workflow.png", bbox_inches="tight")
    plt.close(fig)


def save_episode_conditioning_workflow():
    fig, ax = plt.subplots(figsize=(12, 5.0))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    nodes = [
        (0.5, "trace\n활성 흐름", "#caf0f8"),
        (2.3, "episode\n해석", "#ade8f4"),
        (4.1, "평가 구조\n원인/통제감", "#d8f3dc"),
        (5.95, "원초 정서\n분노/불안 등", "#ffd6a5"),
        (7.8, "행동 경향\n항의/회피/보류", "#ffddd2"),
        (9.65, "LLM 응답\n조건화", "#ffc8dd"),
    ]
    for x, label, color in nodes:
        draw_box(ax, (x, 2.55), (1.35, 0.9), label, fc=color, fontsize=8.8)
    for i in range(len(nodes) - 1):
        draw_arrow(ax, (nodes[i][0] + 1.35, 3.0), (nodes[i + 1][0], 3.0))
    draw_box(ax, (2.6, 0.85), (3.0, 0.75), "자극만 쓰면 놓치기 쉬운\n사건별 의미를 보강", fc="#f8f9fa", fontsize=9)
    draw_box(ax, (6.2, 0.85), (3.2, 0.75), "targeted 조건에서\n에피소드 충실도 평가", fc="#f8f9fa", fontsize=9)
    draw_arrow(ax, (4.1, 1.6), (5.0, 2.55), color="#666")
    draw_arrow(ax, (7.8, 1.6), (8.4, 2.55), color="#666")
    ax.text(6, 4.35, "v4 episode conditioning: trace를 응답 가능한 사건 구조로 번역", ha="center", fontsize=14, fontproperties=FP_BOLD)
    fig.tight_layout()
    fig.savefig(OUT / "episode_conditioning_workflow.png", bbox_inches="tight")
    plt.close(fig)


def save_claim_boundary_map():
    fig, ax = plt.subplots(figsize=(11.5, 5.3))
    ax.set_xlim(0, 11.5)
    ax.set_ylim(0, 5.3)
    ax.axis("off")
    draw_box(ax, (0.7, 3.0), (4.5, 1.15), "현재 주장 가능한 것\ntrace는 감정축 정보를 일부 담고\n조작 반응을 보이며 targeted 조건에서 유용함", fc="#d8f3dc", fontsize=10, bold=True)
    draw_box(ax, (6.2, 3.0), (4.6, 1.15), "아직 주장하면 안 되는 것\n인간처럼 감정을 느낀다\n모든 감정 응답에서 항상 우월하다", fc="#ffddd2", fontsize=10, bold=True)
    draw_box(ax, (0.7, 1.25), (4.5, 0.85), "근거\nv3.1 표현/인과 지표\nv4 targeted paired superiority", fc="#f8f9fa", fontsize=9)
    draw_box(ax, (6.2, 1.25), (4.6, 0.85), "필요한 추가 증거\n사람 평가, 다중 judge,\n대규모 ablation, 장기 안정성", fc="#f8f9fa", fontsize=9)
    ax.axvline(5.75, ymin=0.18, ymax=0.86, color="#444", lw=1.4, ls="--")
    ax.text(5.75, 4.65, "claim 경계", ha="center", fontsize=14, fontproperties=FP_BOLD)
    fig.tight_layout()
    fig.savefig(OUT / "claim_boundary_map.png", bbox_inches="tight")
    plt.close(fig)


def save_module_evidence_map():
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    ax.set_xlim(0, 12.5)
    ax.set_ylim(0, 5.8)
    ax.axis("off")
    modules = [
        (0.5, 4.2, "자극 인코더", "입력 단서\n구조화"),
        (2.6, 4.2, "동역학", "흐름 길이\n밀도"),
        (4.7, 4.2, "branch/trace", "collapse\n분리도"),
        (6.8, 4.2, "z/episode", "평가 구조\n행동 경향"),
        (8.9, 4.2, "응답 조건화", "anti-softening\nraw affect"),
        (10.7, 4.2, "평가", "paired 승률\n신뢰구간"),
    ]
    for x, y, m, e in modules:
        draw_box(ax, (x, y), (1.45, 0.62), m, fc="#caf0f8", fontsize=9, bold=True)
        draw_box(ax, (x, y - 1.05), (1.45, 0.75), e, fc="#f8f9fa", fontsize=8.5)
        draw_arrow(ax, (x + 0.72, y), (x + 0.72, y - 0.3), color="#555")
    for i in range(len(modules) - 1):
        draw_arrow(ax, (modules[i][0] + 1.45, 4.51), (modules[i + 1][0], 4.51))
    ax.text(6.25, 5.35, "모듈과 증거의 연결: 각 행동에는 대응 지표가 있어야 한다", ha="center", fontsize=14, fontproperties=FP_BOLD)
    fig.tight_layout()
    fig.savefig(OUT / "module_evidence_map.png", bbox_inches="tight")
    plt.close(fig)


def main():
    save_v1_pipeline()
    save_v2_module_structure()
    save_v2_neuron_types()
    save_emotion_definition_shift()
    save_full_research_pipeline()
    save_experiment_before_after_map()
    save_metric_taxonomy()
    save_causal_validation_workflow()
    save_episode_conditioning_workflow()
    save_claim_boundary_map()
    save_module_evidence_map()
    save_version_timeline()
    save_branch_recovery()
    save_style_bias()
    save_v31_adaptive()
    save_v31_causal()
    save_v4_targeted()
    save_v4_paired()
    save_general_vs_targeted_after()
    save_targeted_metric_deltas()
    save_bias_status_summary()
    print(f"Generated Korean figures in {OUT}")


if __name__ == "__main__":
    main()
