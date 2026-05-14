from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(".")
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

episode_path = ROOT / "outputs" / "research" / "trajectory_batch_matrix120_v1_gpt54" / "s_000555" / "episode_interpretation.json"
payload = json.loads(episode_path.read_text(encoding="utf-8"))

label = payload.get("episode_label", "")
confidence = payload.get("confidence", "")

phase_segments = [
    ("dormant", 0, 2),
    ("ignition", 2, 5),
    ("persistence", 5, 55),
]

fig, ax = plt.subplots(figsize=(11, 3.8))

for phase, start, end in phase_segments:
    ax.barh([0], [end - start], left=[start], height=0.35)
    ax.text((start + end) / 2, 0, phase, ha="center", va="center", fontsize=11)

ax.axvline(2, linestyle="--", linewidth=1)
ax.axvline(5, linestyle="--", linewidth=1)
ax.axvline(54, linestyle="--", linewidth=1)

ax.text(2, 0.35, "tick 2\nignition", ha="center", va="bottom", fontsize=9)
ax.text(5, 0.35, "tick 5\npersistence begins", ha="center", va="bottom", fontsize=9)
ax.text(54, 0.35, "peak alarm\ntick 54", ha="center", va="bottom", fontsize=9)

ax.set_xlim(0, 56)
ax.set_ylim(-0.6, 0.8)
ax.set_yticks([])
ax.set_xlabel("Tick")
ax.set_title("s_000555 EmoNet internal trace phase timeline")

summary = (
    f"episode label: {label}\n"
    f"confidence: {confidence} | persistence_ratio: 0.9636\n"
    f"dominant signal: 공세적 긴장 | pattern: high_arousal_persistence"
)

fig.text(0.02, 0.02, summary, fontsize=9, va="bottom")

plt.tight_layout(rect=[0, 0.16, 1, 1])
plt.savefig(OUT / "fig_s000555_trace_phase_timeline.svg", format="svg")
plt.savefig(OUT / "fig_s000555_trace_phase_timeline.png", dpi=220)
plt.close()

print("saved:", OUT / "fig_s000555_trace_phase_timeline.svg")
print("saved:", OUT / "fig_s000555_trace_phase_timeline.png")
