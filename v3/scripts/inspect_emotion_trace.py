from __future__ import annotations

import argparse
import json
from dataclasses import fields
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from emonet.cli import MODEL_OPTIONAL_CONFIG_FIELDS, build_model, resolve_text_column
from emonet.core import EmoNetConfig


STIM_KEYS = ("drive", "brake", "alarm", "fatigue")
STIM_LABELS = {
    "drive": "추동/접근",
    "brake": "완충/억제",
    "alarm": "경계/날카로움",
    "fatigue": "피로/둔화",
}
STIM_PLOT_LABELS = {
    "drive": "drive",
    "brake": "brake",
    "alarm": "alarm",
    "fatigue": "fatigue",
}
TYPE_KEYS = ("inhibitory", "excitatory", "modulatory")
EMOTION_PLOT_LABELS = {
    "예민함/신경과민": "irritability",
    "짜증/분노압": "anger_pressure",
    "소진/탈진": "exhaustion",
    "방어적 경계": "guardedness",
    "무기력/철수": "withdrawal",
}


def clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    return value


def describe_level(value: float) -> str:
    if value >= 0.75:
        return "매우 높음"
    if value >= 0.55:
        return "높음"
    if value >= 0.35:
        return "중간"
    if value >= 0.15:
        return "낮음"
    return "매우 낮음"


def normalize_weights(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    total = float(arr.sum())
    if total <= 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return arr / total


def safe_mean(values: pd.Series) -> float:
    if values.empty:
        return 0.0
    return float(values.mean())


def dominant_stim_label(values: np.ndarray) -> str:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size != len(STIM_KEYS) or float(arr.max()) <= 0.0:
        return "거의 무색"
    order = np.argsort(arr)[::-1]
    top = STIM_KEYS[int(order[0])]
    second = STIM_KEYS[int(order[1])] if arr.size > 1 else top
    if top == "alarm" and second == "fatigue":
        return "피로성 경계"
    if top == "alarm" and second == "drive":
        return "공세적 긴장"
    if top == "fatigue" and second == "brake":
        return "수축/둔화"
    return STIM_LABELS[top]


def build_emotion_candidates(summary: dict[str, float]) -> pd.DataFrame:
    drive = float(summary.get("drive", 0.0))
    brake = float(summary.get("brake", 0.0))
    alarm = float(summary.get("alarm", 0.0))
    fatigue = float(summary.get("fatigue", 0.0))
    persistence = float(summary.get("persistence_ratio", 0.0))
    saturation = float(summary.get("saturation_ratio", 0.0))
    inhibitory = float(summary.get("inhibitory_ratio", 0.0))
    excitatory = float(summary.get("excitatory_ratio", 0.0))

    rows = [
        {
            "emotion": "예민함/신경과민",
            "score": clamp01(0.55 * alarm + 0.20 * drive + 0.15 * persistence + 0.10 * saturation - 0.10 * brake),
            "explanation": "경계 신호와 지속 활성의 결합",
        },
        {
            "emotion": "짜증/분노압",
            "score": clamp01(0.40 * alarm + 0.30 * drive + 0.15 * excitatory + 0.15 * (1.0 - brake)),
            "explanation": "경계 + 추동 + 낮은 완충",
        },
        {
            "emotion": "소진/탈진",
            "score": clamp01(0.55 * fatigue + 0.15 * alarm + 0.15 * (1.0 - drive) + 0.15 * persistence),
            "explanation": "피로 부하와 잔류 활성의 결합",
        },
        {
            "emotion": "방어적 경계",
            "score": clamp01(0.45 * alarm + 0.25 * brake + 0.15 * inhibitory + 0.15 * persistence),
            "explanation": "경계 신호 위에 억제/보호 성향이 얹힌 상태",
        },
        {
            "emotion": "무기력/철수",
            "score": clamp01(0.45 * fatigue + 0.25 * brake + 0.20 * (1.0 - drive) + 0.10 * (1.0 - excitatory)),
            "explanation": "피로와 접근 동기 저하의 결합",
        },
    ]
    frame = pd.DataFrame(rows).sort_values(["score", "emotion"], ascending=[False, True]).reset_index(drop=True)
    frame["level"] = frame["score"].map(describe_level)
    return frame


def add_model_build_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-csv", type=str, default=None)
    parser.add_argument("--benchmark-csv", type=str, default=None)
    parser.add_argument("--model-cache-path", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--force-refit", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--z-dim", type=int, default=64)
    parser.add_argument("--z-encoder-mode", choices=["auto", "stat", "transformer"], default="auto")
    parser.add_argument("--z-encoder-path", type=str, default=str(PROJECT_ROOT / "artifacts" / "dominant_branch_encoder.pt"))

    config_defaults = EmoNetConfig()
    for field_name in MODEL_OPTIONAL_CONFIG_FIELDS:
        default_value = getattr(config_defaults, field_name)
        arg_type = int if isinstance(default_value, int) and not isinstance(default_value, bool) else float
        parser.add_argument(f"--{field_name.replace('_', '-')}", dest=field_name, type=arg_type, default=None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--text", default=None)
    source_group.add_argument("--input-csv", default=None)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--record-id-column", default=None)
    parser.add_argument("--record-id", default=None)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    add_model_build_args(parser)
    return parser.parse_args()


def resolve_input_record(args: argparse.Namespace) -> tuple[str, dict[str, Any]]:
    if args.text is not None:
        return str(args.text).strip(), {"source": "direct_text"}

    input_csv = Path(args.input_csv)
    df = pd.read_csv(input_csv)
    text_column = resolve_text_column(df, args.text_column)
    selected_row: pd.Series
    if args.record_id is not None:
        if not args.record_id_column:
            raise ValueError("--record-id-column is required when --record-id is used")
        if args.record_id_column not in df.columns:
            raise ValueError(f"record id column not found: {args.record_id_column}")
        matches = df[df[args.record_id_column].astype(str) == str(args.record_id)]
        if matches.empty:
            raise ValueError(f"record id not found: {args.record_id}")
        selected_row = matches.iloc[0]
    else:
        if args.row_index < 0 or args.row_index >= len(df):
            raise ValueError(f"row_index out of range: {args.row_index}")
        selected_row = df.iloc[int(args.row_index)]

    text = str(selected_row.get(text_column, "")).strip()
    if not text:
        raise ValueError(f"selected row has empty text column '{text_column}'")
    meta = {key: selected_row[key] for key in selected_row.index.tolist()}
    meta["source"] = str(input_csv)
    meta["text_column"] = text_column
    return text, meta


def build_node_trace(model: Any, branch_log: list[Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    neurons = list(model.state.neurons)
    node_catalog_rows: list[dict[str, Any]] = []
    for neuron in neurons:
        bias = np.asarray(neuron.intrinsic_bias, dtype=np.float32).reshape(-1)
        row: dict[str, Any] = {
            "node_id": int(neuron.neuron_id),
            "neuron_type": str(neuron.neuron_type),
            "out_degree": int(len(neuron.out_neighbors)),
            "in_degree": int(len(neuron.in_neighbors)),
            "bias_label": dominant_stim_label(bias),
        }
        for idx, key in enumerate(STIM_KEYS):
            row[f"bias_{key}"] = float(bias[idx]) if idx < bias.size else 0.0
        node_catalog_rows.append(row)
    node_catalog = pd.DataFrame(node_catalog_rows).sort_values("node_id").reset_index(drop=True)
    node_meta = node_catalog.set_index("node_id").to_dict(orient="index")

    node_rows: list[dict[str, Any]] = []
    tick_rows: list[dict[str, Any]] = []
    for record in branch_log:
        tick_active = list(getattr(record, "active_nodes", []))
        tick_edges = list(getattr(record, "edges_fired", []))
        k_values: list[float] = []
        stim_rows: list[np.ndarray] = []
        bias_rows: list[np.ndarray] = []
        type_counts = {key: 0 for key in TYPE_KEYS}

        for node_id in tick_active:
            state = record.node_states[node_id]
            meta = node_meta[int(node_id)]
            stim_vec = np.asarray(state.stim_vec, dtype=np.float32).reshape(-1)
            bias_vec = np.asarray([meta[f"bias_{key}"] for key in STIM_KEYS], dtype=np.float32)
            k_value = float(state.K)
            k_values.append(k_value)
            stim_rows.append(stim_vec)
            bias_rows.append(bias_vec)
            neuron_type = str(meta["neuron_type"])
            if neuron_type in type_counts:
                type_counts[neuron_type] += 1

            row = {
                "tick": int(record.tick),
                "node_id": int(node_id),
                "neuron_type": neuron_type,
                "K": k_value,
                "out_degree": int(meta["out_degree"]),
                "in_degree": int(meta["in_degree"]),
                "bias_label": str(meta["bias_label"]),
            }
            for idx, key in enumerate(STIM_KEYS):
                row[f"stim_{key}"] = float(stim_vec[idx]) if idx < stim_vec.size else 0.0
                row[f"bias_{key}"] = float(bias_vec[idx]) if idx < bias_vec.size else 0.0
            node_rows.append(row)

        if k_values:
            weights = normalize_weights(np.asarray(k_values, dtype=np.float32))
            stim_matrix = np.vstack(stim_rows).astype(np.float32)
            bias_matrix = np.vstack(bias_rows).astype(np.float32)
            weighted_stim = (weights[:, None] * stim_matrix).sum(axis=0)
            weighted_bias = (weights[:, None] * bias_matrix).sum(axis=0)
            weighted_combined = 0.5 * (weighted_stim + weighted_bias)
            dominant_label = dominant_stim_label(weighted_combined)
            mean_k = float(np.mean(k_values))
            max_k = float(np.max(k_values))
        else:
            weighted_stim = np.zeros(len(STIM_KEYS), dtype=np.float32)
            weighted_bias = np.zeros(len(STIM_KEYS), dtype=np.float32)
            weighted_combined = np.zeros(len(STIM_KEYS), dtype=np.float32)
            dominant_label = "거의 무색"
            mean_k = 0.0
            max_k = 0.0

        tick_row: dict[str, Any] = {
            "tick": int(record.tick),
            "active_nodes": int(len(tick_active)),
            "edges_fired": int(len(tick_edges)),
            "mean_K": mean_k,
            "max_K": max_k,
            "dominant_signal": dominant_label,
        }
        for key in TYPE_KEYS:
            tick_row[f"{key}_nodes"] = int(type_counts[key])
        total_active = max(1, len(tick_active))
        for key in TYPE_KEYS:
            tick_row[f"{key}_ratio"] = float(type_counts[key]) / float(total_active) if tick_active else 0.0
        for idx, key in enumerate(STIM_KEYS):
            tick_row[f"stim_{key}"] = float(weighted_stim[idx])
            tick_row[f"bias_{key}"] = float(weighted_bias[idx])
            tick_row[f"combined_{key}"] = float(weighted_combined[idx])
        tick_rows.append(tick_row)

    node_trace = pd.DataFrame(node_rows)
    tick_summary = pd.DataFrame(tick_rows)
    return node_catalog, node_trace, tick_summary


def summarize_global_evidence(
    node_catalog: pd.DataFrame,
    node_trace: pd.DataFrame,
    tick_summary: pd.DataFrame,
    outputs: dict[str, Any],
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    ticks_run = int(outputs.get("ticks_run", 0))
    dominant_branch_len = int(len(outputs.get("dominant_branch", [])))
    termination_reason = str(outputs.get("termination_reason", "unknown"))
    active_tick_count = int((tick_summary["active_nodes"] > 0).sum()) if not tick_summary.empty else 0
    persistence_ratio = float(active_tick_count) / float(max(1, ticks_run))
    n_neurons = max(1, len(node_catalog))
    saturation_ratio = safe_mean(tick_summary["active_nodes"]) / float(n_neurons) if not tick_summary.empty else 0.0

    if node_trace.empty:
        combined = {key: 0.0 for key in STIM_KEYS}
        type_ratios = {f"{key}_ratio": 0.0 for key in TYPE_KEYS}
        top_nodes = pd.DataFrame(columns=["node_id", "neuron_type", "bias_label", "activity_ticks", "k_sum", "k_mean", *[f"stim_{key}" for key in STIM_KEYS]])
    else:
        weights = normalize_weights(node_trace["K"].astype(np.float32).to_numpy())
        for idx, key in enumerate(STIM_KEYS):
            node_trace[f"weighted_{key}"] = weights * node_trace[f"stim_{key}"].astype(np.float32)
            node_trace[f"weighted_bias_{key}"] = weights * node_trace[f"bias_{key}"].astype(np.float32)
        combined = {
            key: float(
                0.5 * node_trace[f"weighted_{key}"].sum()
                + 0.5 * node_trace[f"weighted_bias_{key}"].sum()
            )
            for key in STIM_KEYS
        }
        type_counts = node_trace["neuron_type"].value_counts(normalize=True)
        type_ratios = {f"{key}_ratio": float(type_counts.get(key, 0.0)) for key in TYPE_KEYS}
        top_nodes = (
            node_trace.groupby(["node_id", "neuron_type", "bias_label"], as_index=False)
            .agg(
                activity_ticks=("tick", "count"),
                k_sum=("K", "sum"),
                k_mean=("K", "mean"),
                stim_drive=("stim_drive", "mean"),
                stim_brake=("stim_brake", "mean"),
                stim_alarm=("stim_alarm", "mean"),
                stim_fatigue=("stim_fatigue", "mean"),
            )
            .sort_values(["k_sum", "activity_ticks"], ascending=[False, False])
            .head(20)
            .reset_index(drop=True)
        )

    summary = {
        "ticks_run": ticks_run,
        "termination_reason": termination_reason,
        "dominant_branch_len": dominant_branch_len,
        "active_tick_count": active_tick_count,
        "persistence_ratio": persistence_ratio,
        "mean_active_nodes": safe_mean(tick_summary["active_nodes"]) if not tick_summary.empty else 0.0,
        "max_active_nodes": int(tick_summary["active_nodes"].max()) if not tick_summary.empty else 0,
        "mean_edges_fired": safe_mean(tick_summary["edges_fired"]) if not tick_summary.empty else 0.0,
        "max_edges_fired": int(tick_summary["edges_fired"].max()) if not tick_summary.empty else 0,
        "saturation_ratio": saturation_ratio,
        **combined,
        **type_ratios,
        "dominant_global_signal": dominant_stim_label(np.asarray([combined[key] for key in STIM_KEYS], dtype=np.float32)),
    }

    candidates = build_emotion_candidates(summary)
    return summary, candidates, top_nodes


def build_raw_trace_json(
    text: str,
    input_meta: dict[str, Any],
    branch_log: list[Any],
    node_catalog: pd.DataFrame,
) -> dict[str, Any]:
    node_meta = node_catalog.set_index("node_id").to_dict(orient="index")
    ticks: list[dict[str, Any]] = []
    for record in branch_log:
        tick_payload = {
            "tick": int(record.tick),
            "active_nodes": [int(node_id) for node_id in record.active_nodes],
            "edges_fired": [[int(src), int(dst)] for src, dst in record.edges_fired],
            "node_states": [],
        }
        for node_id in record.active_nodes:
            meta = node_meta[int(node_id)]
            state = record.node_states[node_id]
            tick_payload["node_states"].append(
                {
                    "node_id": int(node_id),
                    "neuron_type": str(meta["neuron_type"]),
                    "K": float(state.K),
                    "stim_vec": [float(value) for value in np.asarray(state.stim_vec, dtype=np.float32).reshape(-1)],
                    "intrinsic_bias": [float(meta[f"bias_{key}"]) for key in STIM_KEYS],
                    "bias_label": str(meta["bias_label"]),
                }
            )
        ticks.append(tick_payload)
    return {
        "input_text": text,
        "input_meta": input_meta,
        "ticks": ticks,
    }


def save_figures(output_dir: Path, tick_summary: pd.DataFrame, candidates: pd.DataFrame) -> None:
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if not tick_summary.empty:
        fig, ax1 = plt.subplots(figsize=(8, 4.5))
        ax1.plot(tick_summary["tick"], tick_summary["active_nodes"], label="active_nodes", color="#d95f02", linewidth=2)
        ax1.set_xlabel("tick")
        ax1.set_ylabel("active nodes")
        ax2 = ax1.twinx()
        ax2.plot(tick_summary["tick"], tick_summary["edges_fired"], label="edges_fired", color="#1b9e77", linewidth=2)
        ax2.set_ylabel("edges fired")
        ax1.set_title("Raw Trace Activity")
        fig.tight_layout()
        fig.savefig(figures_dir / "tick_activity.svg", format="svg")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        for key, color in zip(STIM_KEYS, ["#4c78a8", "#72b7b2", "#e45756", "#f58518"], strict=False):
            ax.plot(
                tick_summary["tick"],
                tick_summary[f"combined_{key}"],
                label=STIM_PLOT_LABELS[key],
                color=color,
                linewidth=2,
            )
        ax.set_xlabel("tick")
        ax.set_ylabel("weighted signal")
        ax.set_title("Raw Affect Signal Curves")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(figures_dir / "raw_signal_curves.svg", format="svg")
        plt.close(fig)

    if not candidates.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        plot_labels = [EMOTION_PLOT_LABELS.get(str(label), str(label)) for label in candidates["emotion"].tolist()]
        ax.barh(plot_labels, candidates["score"], color="#5f0f40")
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("score")
        ax.set_title("Raw Emotion Candidates")
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(figures_dir / "emotion_candidates.svg", format="svg")
        plt.close(fig)


def write_report(
    output_dir: Path,
    *,
    text: str,
    summary: dict[str, Any],
    candidates: pd.DataFrame,
    top_nodes: pd.DataFrame,
) -> None:
    report_path = output_dir / "EMOTION_TRACE_REPORT.md"
    top_candidate_lines = []
    for row in candidates.head(3).to_dict(orient="records"):
        top_candidate_lines.append(f"- {row['emotion']}: {float(row['score']):.4f} ({row['level']})")

    node_lines = []
    for row in top_nodes.head(10).to_dict(orient="records"):
        node_lines.append(
            f"- node {int(row['node_id'])} [{row['neuron_type']}] {row['bias_label']} | "
            f"activity_ticks={int(row['activity_ticks'])}, k_sum={float(row['k_sum']):.2f}, k_mean={float(row['k_mean']):.2f}"
        )

    lines = [
        "# Emotion Trace Report",
        "",
        "## Input",
        "",
        text,
        "",
        "## Raw Trace Summary",
        "",
        f"- ticks_run: {int(summary['ticks_run'])}",
        f"- termination_reason: {summary['termination_reason']}",
        f"- dominant_branch_len: {int(summary['dominant_branch_len'])}",
        f"- persistence_ratio: {float(summary['persistence_ratio']):.4f}",
        f"- saturation_ratio: {float(summary['saturation_ratio']):.4f}",
        f"- dominant_global_signal: {summary['dominant_global_signal']}",
        "",
        "## Raw Signal Means",
        "",
        *[
            f"- {STIM_LABELS[key]}: {float(summary[key]):.4f} ({describe_level(float(summary[key]))})"
            for key in STIM_KEYS
        ],
        "",
        "## Candidate Emotions",
        "",
        *top_candidate_lines,
        "",
        "## Top Active Nodes",
        "",
    ]
    lines.extend(node_lines if node_lines else ["- no active nodes"])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    text, input_meta = resolve_input_record(args)
    model = build_model(args)
    outputs = model.forward(text)
    branch_log = list(model.state.branch_log)

    node_catalog, node_trace, tick_summary = build_node_trace(model, branch_log)
    summary, candidates, top_nodes = summarize_global_evidence(node_catalog, node_trace, tick_summary, outputs)

    raw_trace_payload = build_raw_trace_json(text=text, input_meta=input_meta, branch_log=branch_log, node_catalog=node_catalog)
    raw_trace_payload["summary"] = summary
    raw_trace_payload["top_emotions"] = candidates.head(5).to_dict(orient="records")

    node_catalog.to_csv(output_dir / "node_catalog.csv", index=False, encoding="utf-8-sig")
    node_trace.to_csv(output_dir / "node_trace.csv", index=False, encoding="utf-8-sig")
    tick_summary.to_csv(output_dir / "tick_summary.csv", index=False, encoding="utf-8-sig")
    candidates.to_csv(output_dir / "emotion_candidates.csv", index=False, encoding="utf-8-sig")
    top_nodes.to_csv(output_dir / "top_nodes.csv", index=False, encoding="utf-8-sig")
    (output_dir / "raw_trace.json").write_text(
        json.dumps(to_jsonable(raw_trace_payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "emotion_trace_summary.json").write_text(
        json.dumps(
            to_jsonable(
                {
                    "input_text": text,
                    "input_meta": input_meta,
                    **summary,
                    "top_emotions": candidates.head(5).to_dict(orient="records"),
                }
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    save_figures(output_dir, tick_summary, candidates)
    write_report(output_dir, text=text, summary=summary, candidates=candidates, top_nodes=top_nodes)

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "ticks_run": int(summary["ticks_run"]),
                "termination_reason": str(summary["termination_reason"]),
                "dominant_branch_len": int(summary["dominant_branch_len"]),
                "top_emotion": candidates.iloc[0]["emotion"] if not candidates.empty else None,
                "top_emotion_score": float(candidates.iloc[0]["score"]) if not candidates.empty else None,
                "summary_path": str(output_dir / "emotion_trace_summary.json"),
                "raw_trace_path": str(output_dir / "raw_trace.json"),
                "report_path": str(output_dir / "EMOTION_TRACE_REPORT.md"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
