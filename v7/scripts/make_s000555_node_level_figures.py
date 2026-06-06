from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(".")
SRC = ROOT / "outputs" / "research" / "trajectory_batch_matrix120_v1" / "s_000555"
OUT = ROOT / "paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

raw_path = SRC / "raw_trace.json"
node_trace_path = SRC / "node_trace.csv"
node_catalog_path = SRC / "node_catalog.csv"
trajectory_phases_path = SRC / "trajectory_phases.csv"

raw = json.loads(raw_path.read_text(encoding="utf-8"))
node_trace = pd.read_csv(node_trace_path)
node_catalog = pd.read_csv(node_catalog_path)
phase_df = pd.read_csv(trajectory_phases_path)

# -----------------------------
# Helpers
# -----------------------------

def savefig(name: str) -> None:
    plt.tight_layout()
    plt.savefig(OUT / f"{name}.svg", format="svg")
    plt.savefig(OUT / f"{name}.png", dpi=240)
    plt.close()


def phase_color(phase: str) -> str:
    return {
        "dormant": "#dddddd",
        "ignition": "#fee08b",
        "escalation": "#f46d43",
        "persistence": "#d73027",
        "fatigue_shift": "#74add1",
        "decay": "#abd9e9",
    }.get(str(phase), "#eeeeee")


def add_phase_spans(ax) -> None:
    for _, row in phase_df.iterrows():
        ax.axvspan(
            float(row["start_tick"]),
            float(row["end_tick"]),
            color=phase_color(str(row["phase"])),
            alpha=0.12,
            linewidth=0,
        )


# -----------------------------
# 1. Dominant node path
# -----------------------------
# If a raw dominant_branch exists, use it. Otherwise reconstruct a tick-dominant path:
# for each tick, choose the active node with the highest K value.

dominant_branch = raw.get("dominant_branch")
path_rows = []

if dominant_branch:
    # Original model-provided dominant branch, if available.
    for idx, item in enumerate(dominant_branch):
        if isinstance(item, dict):
            node_id = item.get("node_id", item.get("id", item.get("node")))
            tick = item.get("tick", idx)
            k = item.get("K", item.get("k", None))
        else:
            node_id = item
            tick = idx
            k = None
        path_rows.append({"path_index": idx, "tick": tick, "node_id": int(node_id), "K": k, "source": "dominant_branch"})
else:
    # Reconstructed path from node_trace.
    for tick, group in node_trace.groupby("tick"):
        if group.empty:
            continue
        row = group.sort_values(["K", "node_id"], ascending=[False, True]).iloc[0]
        path_rows.append(
            {
                "path_index": len(path_rows),
                "tick": int(tick),
                "node_id": int(row["node_id"]),
                "K": float(row["K"]),
                "source": "reconstructed_tick_topK",
            }
        )

path_df = pd.DataFrame(path_rows)

# Compress consecutive same nodes for readability.
compressed = []
prev_node = None
for _, row in path_df.iterrows():
    node = int(row["node_id"])
    if node != prev_node:
        compressed.append(row.to_dict())
        prev_node = node
compressed_df = pd.DataFrame(compressed)
compressed_df.to_csv(SRC / "dominant_node_path_reconstructed.csv", index=False, encoding="utf-8-sig")

plt.figure(figsize=(12, 4.8))
ax = plt.gca()
add_phase_spans(ax)

plt.plot(path_df["tick"], path_df["node_id"], marker="o", linewidth=1.8, markersize=4)
plt.xlabel("Tick")
plt.ylabel("Dominant node id")
title_source = "model dominant_branch" if dominant_branch else "reconstructed by top-K active node per tick"
plt.title(f"s_000555 dominant node path ({title_source})")

# Label only compressed path points to reduce clutter.
for _, row in compressed_df.iterrows():
    if int(row["path_index"]) % 2 == 0 or len(compressed_df) <= 12:
        plt.text(float(row["tick"]), float(row["node_id"]), str(int(row["node_id"])), fontsize=8, ha="center", va="bottom")

savefig("fig_s000555_dominant_node_path")


# -----------------------------
# 2. Tick × cluster heatmap
# -----------------------------
candidate_cluster_cols = [
    "cluster",
    "cluster_id",
    "community",
    "community_id",
    "module",
    "module_id",
]
cluster_col = None
for col in candidate_cluster_cols:
    if col in node_trace.columns:
        cluster_col = col
        break

if cluster_col is None:
    merge_cols = [c for c in candidate_cluster_cols if c in node_catalog.columns]
    if merge_cols:
        cluster_col = merge_cols[0]
        node_trace = node_trace.merge(node_catalog[["node_id", cluster_col]], on="node_id", how="left")
    else:
        raise ValueError("node trace or catalog must contain an explicit cluster/community/module column")

# Activation value: sum K per tick per cluster-like group.
heat = (
    node_trace.assign(cluster_value=node_trace[cluster_col].astype(str))
    .pivot_table(index="cluster_value", columns="tick", values="K", aggfunc="sum", fill_value=0.0)
)

# Sort by total activation.
heat = heat.loc[heat.sum(axis=1).sort_values(ascending=False).index]

plt.figure(figsize=(12, max(4.8, 0.35 * len(heat.index))))
ax = plt.gca()
img = ax.imshow(heat.values, aspect="auto", interpolation="nearest")
plt.colorbar(img, ax=ax, label="sum of node K")

tick_cols = list(heat.columns)
step = max(1, len(tick_cols) // 10)
ax.set_xticks(range(0, len(tick_cols), step))
ax.set_xticklabels([str(tick_cols[i]) for i in range(0, len(tick_cols), step)])

ax.set_yticks(range(len(heat.index)))
ax.set_yticklabels(heat.index)

ax.set_xlabel("Tick")
ax.set_ylabel(cluster_col)
ax.set_title(f"s_000555 tick × {cluster_col} activation heatmap")

savefig("fig_s000555_tick_cluster_heatmap")


# -----------------------------
# 3. Top active node summary
# -----------------------------
agg = (
    node_trace.groupby(["node_id", "neuron_type", "bias_label"], as_index=False)
    .agg(
        activity_ticks=("tick", "nunique"),
        first_tick=("tick", "min"),
        last_tick=("tick", "max"),
        k_sum=("K", "sum"),
        k_mean=("K", "mean"),
        stim_drive=("stim_drive", "mean"),
        stim_brake=("stim_brake", "mean"),
        stim_alarm=("stim_alarm", "mean"),
        stim_fatigue=("stim_fatigue", "mean"),
    )
    .sort_values(["k_sum", "activity_ticks"], ascending=[False, False])
    .reset_index(drop=True)
)

top = agg.head(15).copy()
top.to_csv(SRC / "top_active_nodes_for_paper.csv", index=False, encoding="utf-8-sig")

plt.figure(figsize=(10, 5))
labels = [f"n{int(r.node_id)}\n{r.bias_label}" for r in top.itertuples()]
plt.bar(labels, top["k_sum"])
plt.xticks(rotation=45, ha="right")
plt.ylabel("Total K across ticks")
plt.xlabel("Node")
plt.title("s_000555 top active nodes by accumulated K")
savefig("fig_s000555_top_active_nodes")


# -----------------------------
# 4. Markdown table for paper appendix
# -----------------------------
md_path = SRC / "TOP_ACTIVE_NODES_FOR_PAPER.md"
with md_path.open("w", encoding="utf-8") as f:
    f.write("# s_000555 Top Active Nodes\n\n")
    f.write("This table summarizes the most active nodes in the s_000555 EmoNet trace inspection.\n\n")
    f.write(top.to_markdown(index=False))

print("Saved figures:")
print(" -", OUT / "fig_s000555_dominant_node_path.svg")
print(" -", OUT / "fig_s000555_tick_cluster_heatmap.svg")
print(" -", OUT / "fig_s000555_top_active_nodes.svg")
print("Saved tables:")
print(" -", SRC / "dominant_node_path_reconstructed.csv")
print(" -", SRC / "top_active_nodes_for_paper.csv")
print(" -", SRC / "TOP_ACTIVE_NODES_FOR_PAPER.md")
print("cluster column used:", cluster_col)
print("dominant path source:", "model dominant_branch" if dominant_branch else "reconstructed_tick_topK")
