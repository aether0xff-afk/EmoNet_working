from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from emonet.cli import build_model, maybe_print_progress, resolve_text_column
import inspect_emotion_trace as INSPECT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--record-id-column", default="sample_id")
    parser.add_argument("--record-ids", default=None, help="Optional comma-separated ids")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--sample-mode", choices=["head", "random"], default="random")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--save-per-sample", action="store_true")
    INSPECT.add_model_build_args(parser)
    return parser.parse_args()


def sanitize_name(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value))
    return text.strip("._") or "sample"


def build_sample_summary_row(
    *,
    record_id: str,
    text: str,
    summary: dict[str, Any],
    trajectory_summary: dict[str, Any],
    candidates: pd.DataFrame,
) -> dict[str, Any]:
    top_candidates = candidates.head(3).to_dict(orient="records")
    row = {
        "record_id": record_id,
        "text": text,
        "ticks_run": int(summary["ticks_run"]),
        "termination_reason": str(summary["termination_reason"]),
        "dominant_branch_len": int(summary["dominant_branch_len"]),
        "persistence_ratio": float(summary["persistence_ratio"]),
        "saturation_ratio": float(summary["saturation_ratio"]),
        "dominant_global_signal": str(summary["dominant_global_signal"]),
        "top_emotion": str(top_candidates[0]["emotion"]) if top_candidates else "",
        "top_emotion_score": float(top_candidates[0]["score"]) if top_candidates else 0.0,
        "top_emotion_level": str(top_candidates[0]["level"]) if top_candidates else "",
        "second_emotion": str(top_candidates[1]["emotion"]) if len(top_candidates) > 1 else "",
        "second_emotion_score": float(top_candidates[1]["score"]) if len(top_candidates) > 1 else 0.0,
        "third_emotion": str(top_candidates[2]["emotion"]) if len(top_candidates) > 2 else "",
        "third_emotion_score": float(top_candidates[2]["score"]) if len(top_candidates) > 2 else 0.0,
        "trajectory_pattern": str(trajectory_summary["trajectory_pattern"]),
        "phase_count": int(trajectory_summary["phase_count"]),
        "phase_sequence": " -> ".join(trajectory_summary["phase_sequence"]),
        "peak_alarm_tick": int(trajectory_summary["peak_alarm_tick"]),
        "peak_fatigue_tick": int(trajectory_summary["peak_fatigue_tick"]),
        "peak_conflict_tick": int(trajectory_summary["peak_conflict_tick"]),
        "drive": float(summary["drive"]),
        "brake": float(summary["brake"]),
        "alarm": float(summary["alarm"]),
        "fatigue": float(summary["fatigue"]),
        "inhibitory_ratio": float(summary["inhibitory_ratio"]),
        "excitatory_ratio": float(summary["excitatory_ratio"]),
        "modulatory_ratio": float(summary["modulatory_ratio"]),
    }
    return row


def save_aggregate_figures(output_dir: Path, sample_df: pd.DataFrame) -> list[str]:
    figure_paths: list[str] = []
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    if sample_df.empty:
        return figure_paths

    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.45 * len(sample_df))))
    ordered = sample_df.sort_values("top_emotion_score", ascending=True)
    ax.barh(ordered["record_id"], ordered["top_emotion_score"], color="#5f0f40")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("top emotion score")
    ax.set_title("Top Emotion Score by Sample")
    fig.tight_layout()
    path = figures_dir / "top_emotion_scores.svg"
    fig.savefig(path, format="svg")
    plt.close(fig)
    figure_paths.append(str(path))

    fig, ax = plt.subplots(figsize=(9, max(4.5, 0.45 * len(sample_df))))
    ordered = sample_df.sort_values("record_id")
    ax.barh(ordered["record_id"], ordered["alarm"], color="#e45756", label="alarm")
    ax.barh(ordered["record_id"], ordered["fatigue"], color="#f58518", left=ordered["alarm"], label="fatigue")
    ax.set_xlim(0.0, 2.0)
    ax.set_xlabel("signal sum")
    ax.set_title("Alarm/Fatigue Pressure by Sample")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = figures_dir / "alarm_fatigue_pressure.svg"
    fig.savefig(path, format="svg")
    plt.close(fig)
    figure_paths.append(str(path))

    pattern_counts = sample_df["trajectory_pattern"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(pattern_counts.index.astype(str), pattern_counts.to_numpy(), color="#1b9e77")
    ax.set_ylabel("samples")
    ax.set_title("Trajectory Pattern Counts")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    path = figures_dir / "trajectory_pattern_counts.svg"
    fig.savefig(path, format="svg")
    plt.close(fig)
    figure_paths.append(str(path))

    return figure_paths


def write_aggregate_report(output_dir: Path, sample_df: pd.DataFrame, phase_df: pd.DataFrame) -> None:
    report_path = output_dir / "BATCH_TRAJECTORY_REPORT.md"
    lines = [
        "# Batch Emotion Trajectory Report",
        "",
        f"- samples: {int(len(sample_df))}",
        f"- unique_top_emotions: {int(sample_df['top_emotion'].nunique()) if not sample_df.empty else 0}",
        "",
        "## Top Emotion By Sample",
        "",
    ]
    for row in sample_df.sort_values(["top_emotion_score", "record_id"], ascending=[False, True]).to_dict(orient="records"):
        lines.append(
            f"- {row['record_id']}: {row['top_emotion']} ({float(row['top_emotion_score']):.4f}), "
            f"pattern={row['trajectory_pattern']}, signal={row['dominant_global_signal']}"
        )
    lines.extend(["", "## Phase Counts", ""])
    if phase_df.empty:
        lines.append("- no phase rows")
    else:
        counts = phase_df["phase"].value_counts().sort_index()
        for phase, count in counts.items():
            lines.append(f"- {phase}: {int(count)}")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def select_records(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    requested_ids = [item.strip() for item in str(args.record_ids or "").split(",") if item.strip()]
    if requested_ids:
        selected = df[df[args.record_id_column].astype(str).isin(requested_ids)].copy()
        if selected.empty:
            raise ValueError("no matching records found")
        selected["_requested_order"] = selected[args.record_id_column].astype(str).map(
            {rid: idx for idx, rid in enumerate(requested_ids)}
        )
        return selected.sort_values("_requested_order").drop(columns=["_requested_order"])

    selected = df.copy()
    if args.sample_size is not None and args.sample_size > 0 and len(selected) > int(args.sample_size):
        if args.sample_mode == "random":
            selected = selected.sample(n=int(args.sample_size), random_state=int(args.sample_seed)).reset_index(drop=True)
        else:
            selected = selected.head(int(args.sample_size)).copy()
    if selected.empty:
        raise ValueError("no records selected")
    return selected.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    if args.record_id_column not in df.columns:
        raise ValueError(f"record id column not found: {args.record_id_column}")
    text_column = resolve_text_column(df, args.text_column)
    selected = select_records(df, args)
    selected_ids = selected[args.record_id_column].astype(str).tolist()
    (output_dir / "selected_record_ids.txt").write_text("\n".join(selected_ids) + "\n", encoding="utf-8")
    selected[[args.record_id_column, text_column]].to_csv(
        output_dir / "selected_records.csv",
        index=False,
        encoding="utf-8-sig",
    )

    model = build_model(args)
    sample_rows: list[dict[str, Any]] = []
    phase_rows: list[dict[str, Any]] = []
    start_time = time.perf_counter()

    for idx, row in enumerate(selected.to_dict(orient="records"), start=1):
        record_id = str(row[args.record_id_column])
        text = str(row.get(text_column, "")).strip()
        outputs = model.forward(text)
        branch_log = list(model.state.branch_log)
        node_catalog, node_trace, tick_summary = INSPECT.build_node_trace(model, branch_log)
        summary, candidates, top_nodes = INSPECT.summarize_global_evidence(node_catalog, node_trace, tick_summary, outputs)
        trajectory_ticks = INSPECT.build_tick_emotion_frame(tick_summary)
        trajectory_phases = INSPECT.summarize_trajectory_phases(trajectory_ticks)
        trajectory_summary = INSPECT.build_trajectory_summary(summary, trajectory_phases, trajectory_ticks)

        sample_rows.append(
            build_sample_summary_row(
                record_id=record_id,
                text=text,
                summary=summary,
                trajectory_summary=trajectory_summary,
                candidates=candidates,
            )
        )

        for phase_row in trajectory_phases.to_dict(orient="records"):
            phase_rows.append({"record_id": record_id, **phase_row})

        if args.save_per_sample:
            sample_dir = output_dir / sanitize_name(record_id)
            sample_dir.mkdir(parents=True, exist_ok=True)
            raw_trace_payload = INSPECT.build_raw_trace_json(text=text, input_meta=row, branch_log=branch_log, node_catalog=node_catalog)
            raw_trace_payload["summary"] = summary
            raw_trace_payload["top_emotions"] = candidates.head(5).to_dict(orient="records")
            raw_trace_payload["trajectory_summary"] = trajectory_summary
            raw_trace_payload["trajectory_phases"] = trajectory_phases.to_dict(orient="records")

            node_catalog.to_csv(sample_dir / "node_catalog.csv", index=False, encoding="utf-8-sig")
            node_trace.to_csv(sample_dir / "node_trace.csv", index=False, encoding="utf-8-sig")
            tick_summary.to_csv(sample_dir / "tick_summary.csv", index=False, encoding="utf-8-sig")
            candidates.to_csv(sample_dir / "emotion_candidates.csv", index=False, encoding="utf-8-sig")
            top_nodes.to_csv(sample_dir / "top_nodes.csv", index=False, encoding="utf-8-sig")
            trajectory_ticks.to_csv(sample_dir / "trajectory_ticks.csv", index=False, encoding="utf-8-sig")
            trajectory_phases.to_csv(sample_dir / "trajectory_phases.csv", index=False, encoding="utf-8-sig")
            (sample_dir / "raw_trace.json").write_text(
                json.dumps(INSPECT.to_jsonable(raw_trace_payload), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (sample_dir / "emotion_trace_summary.json").write_text(
                json.dumps(
                    INSPECT.to_jsonable(
                        {
                            "input_text": text,
                            "input_meta": row,
                            **summary,
                            "top_emotions": candidates.head(5).to_dict(orient="records"),
                        }
                    ),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            (sample_dir / "emotion_trajectory_summary.json").write_text(
                json.dumps(
                    INSPECT.to_jsonable(
                        {
                            "input_text": text,
                            "input_meta": row,
                            **trajectory_summary,
                            "phase_segments": trajectory_phases.to_dict(orient="records"),
                        }
                    ),
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            INSPECT.save_figures(sample_dir, tick_summary, candidates)
            INSPECT.save_trajectory_figures(sample_dir, trajectory_ticks, trajectory_phases)
            INSPECT.write_report(sample_dir, text=text, summary=summary, candidates=candidates, top_nodes=top_nodes)
            INSPECT.write_trajectory_report(
                sample_dir,
                text=text,
                summary=summary,
                trajectory_summary=trajectory_summary,
                phase_df=trajectory_phases,
            )

        maybe_print_progress(
            "trajectory-batch",
            idx,
            len(selected),
            start_time,
            every=max(1, int(args.progress_every)),
            unit="samples",
            force=idx == len(selected),
        )

    sample_df = pd.DataFrame(sample_rows)
    phase_df = pd.DataFrame(phase_rows)
    sample_df.to_csv(output_dir / "sample_summary.csv", index=False, encoding="utf-8-sig")
    phase_df.to_csv(output_dir / "phase_summary.csv", index=False, encoding="utf-8-sig")
    figure_paths = save_aggregate_figures(output_dir, sample_df)
    write_aggregate_report(output_dir, sample_df, phase_df)

    payload = {
        "output_dir": str(output_dir),
        "samples": int(len(sample_df)),
        "selected_record_ids_txt": str(output_dir / "selected_record_ids.txt"),
        "selected_records_csv": str(output_dir / "selected_records.csv"),
        "sample_summary_csv": str(output_dir / "sample_summary.csv"),
        "phase_summary_csv": str(output_dir / "phase_summary.csv"),
        "figure_paths": figure_paths,
        "report_path": str(output_dir / "BATCH_TRAJECTORY_REPORT.md"),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
