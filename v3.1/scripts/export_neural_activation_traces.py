#!/usr/bin/env python3
"""Export neural activation traces from the v3 EmoNet network.

This is the v3.1 core object of study:

    stimulus vector -> network dynamics -> activation trace

The structured appraisal fields are kept only as external labels for probing.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
V3_ROOT = REPO_ROOT / "v3"
if str(V3_ROOT) not in sys.path:
    sys.path.insert(0, str(V3_ROOT))

from emonet.core import EmoNet, EmoNetConfig, SKLEARN_AVAILABLE, StimEncoder, StimEncoderConfig  # noqa: E402


LABEL_COLUMNS = [
    "record_id",
    "text",
    "valence",
    "arousal",
    "target",
    "control_state",
    "social_orientation",
    "action_tendency_class",
    "episode_family",
    "appraisal_family",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def build_model(args: argparse.Namespace) -> EmoNet:
    n_modulatory = max(1, int(round(args.n_neurons * 0.10)))
    n_inhibitory = int(round(args.n_neurons * 0.45))
    n_excitatory = int(args.n_neurons - n_inhibitory - n_modulatory)
    config = EmoNetConfig(
        n_neurons=args.n_neurons,
        n_inhibitory=n_inhibitory,
        n_excitatory=n_excitatory,
        n_modulatory=n_modulatory,
        seed=args.seed,
        z_encoder_mode=args.z_encoder_mode,
        load_z_encoder_checkpoint=False,
        max_ticks=args.max_ticks,
        min_ticks_before_converged=args.min_ticks_before_converged,
        convergence_patience=args.convergence_patience,
        k_threshold_base=args.k_threshold_base,
        k_remem_base=args.k_remem_base,
        k_decay=args.k_decay,
        refractory_ticks=args.refractory_ticks,
        input_topk=args.input_topk,
        input_signal_clip=args.input_signal_clip,
        intrinsic_alignment_gain=args.intrinsic_alignment_gain,
        fatigue_gain=args.fatigue_gain,
        fatigue_threshold_gain=args.fatigue_threshold_gain,
        fatigue_k_leak=args.fatigue_k_leak,
        inhibitory_suppression_gain=args.inhibitory_suppression_gain,
        density_control_start_tick=args.density_control_start_tick,
        density_target_high=args.density_target_high,
        density_soft_k_leak_gain=args.density_soft_k_leak_gain,
        density_hard_cap=args.density_hard_cap,
        density_pruned_fatigue_gain=args.density_pruned_fatigue_gain,
        ne_thresh_reduce_gain=args.ne_thresh_reduce_gain,
        ne_remem_reduce_gain=args.ne_remem_reduce_gain,
        activity_churn_eps=args.activity_churn_eps,
    )
    stim_config = StimEncoderConfig(force_refit=False)
    model = EmoNet(config=config, stim_encoder_config=stim_config)
    return model


def base_score_from_labels(row: dict[str, str]) -> float:
    valence = str(row.get("valence", "")).strip().lower()
    arousal = str(row.get("arousal", "")).strip().lower()
    score = {"positive": 0.78, "mixed": 0.48, "negative": 0.22}.get(valence, 0.45)
    if arousal == "high":
        score -= 0.06
    elif arousal == "low":
        score += 0.06
    return float(np.clip(score, 0.0, 1.0))


def proxy_stim_vec(row: dict[str, str]) -> np.ndarray:
    text = str(row.get("text", ""))
    score = base_score_from_labels(row)
    stim = StimEncoder._build_proxy_targets([text], np.asarray([score], dtype=np.float32))[0]

    # Label nudges keep this as a stimulus vector fallback, not an appraisal trace.
    # The network still produces the activation trajectory.
    target = str(row.get("target", "")).strip().lower()
    social = str(row.get("social_orientation", "")).strip().lower()
    action = str(row.get("action_tendency_class", "")).strip().lower()
    if target == "other" or social == "defend" or action in {"confront", "defend"}:
        stim[2] += 0.10
    if action in {"withdraw", "inhibit"}:
        stim[3] += 0.08
    if action in {"approach", "repair", "seek_support"}:
        stim[1] += 0.08
    if action in {"plan"}:
        stim[0] += 0.05
    return np.clip(stim, 0.0, 1.0).astype(np.float32)


def model_input_for_row(row: dict[str, str], args: argparse.Namespace) -> str | np.ndarray:
    if args.stim_source == "text":
        return str(row.get("text", ""))
    if args.stim_source == "proxy":
        return proxy_stim_vec(row)
    if SKLEARN_AVAILABLE:
        return str(row.get("text", ""))
    return proxy_stim_vec(row)


def tick_activation_matrix(outputs: dict[str, Any], n_neurons: int) -> np.ndarray:
    branch_log = list(outputs.get("pruned_branch_log") or outputs.get("branch_log") or [])
    if not branch_log:
        branch_log = list(outputs.get("pruned_branch_log") or [])
    ticks = max(len(branch_log), 1)
    matrix = np.zeros((ticks, n_neurons), dtype=np.float32)
    for row_idx, tick_record in enumerate(branch_log):
        node_states = getattr(tick_record, "node_states", {}) or {}
        for node_id, node_state in node_states.items():
            if 0 <= int(node_id) < n_neurons:
                matrix[row_idx, int(node_id)] = float(getattr(node_state, "K", 0.0))
    return matrix


def active_count_series(outputs: dict[str, Any]) -> list[int]:
    branch_log = list(outputs.get("pruned_branch_log") or [])
    return [len(getattr(tick, "active_nodes", []) or []) for tick in branch_log]


def dominant_branch_ids(outputs: dict[str, Any]) -> list[int]:
    branch = list(outputs.get("dominant_branch") or [])
    return [int(getattr(step, "node_id", -1)) for step in branch]


def summarize_trace(
    *,
    row: dict[str, str],
    outputs: dict[str, Any],
    activation: np.ndarray,
    branch_tensor: np.ndarray,
    z: np.ndarray,
) -> dict[str, Any]:
    active_counts = active_count_series(outputs)
    nonzero = activation[activation > 0]
    dominant_ids = dominant_branch_ids(outputs)
    summary: dict[str, Any] = {column: row.get(column, "") for column in LABEL_COLUMNS}
    summary.update(
        {
            "ticks_run": int(outputs.get("ticks_run", activation.shape[0])),
            "termination_reason": str(outputs.get("termination_reason", "")),
            "activation_ticks": int(activation.shape[0]),
            "activation_neurons": int(activation.shape[1]),
            "activation_nonzero": int(np.count_nonzero(activation)),
            "activation_density": float(np.count_nonzero(activation) / max(activation.size, 1)),
            "activation_k_mean": float(nonzero.mean()) if nonzero.size else 0.0,
            "activation_k_max": float(activation.max()) if activation.size else 0.0,
            "active_count_mean": float(np.mean(active_counts)) if active_counts else 0.0,
            "active_count_max": int(max(active_counts)) if active_counts else 0,
            "dominant_branch_len": int(len(dominant_ids)),
            "dominant_unique_nodes": int(len(set(dominant_ids))),
            "branch_tensor_len": int(branch_tensor.shape[0]) if branch_tensor.ndim >= 1 else 0,
            "branch_tensor_dim": int(branch_tensor.shape[1]) if branch_tensor.ndim == 2 else 0,
            "z_dim": int(z.size),
            "z_l2": float(np.linalg.norm(z.reshape(-1))) if z.size else 0.0,
            "dominant_branch_ids": " ".join(str(node_id) for node_id in dominant_ids[:64]),
        }
    )
    stim_vec = to_numpy(outputs.get("stim_vec", np.zeros(4, dtype=np.float32))).reshape(-1)
    for idx, name in enumerate(["dopamine", "serotonin", "norepinephrine", "melatonin"]):
        summary[name] = float(stim_vec[idx]) if idx < stim_vec.size else 0.0
    return summary


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_csv(args.input)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    model = build_model(args)
    trace_dir = args.output_dir / "traces_npz"
    trace_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for idx, row in enumerate(rows, start=1):
        record_id = str(row.get("record_id") or f"row_{idx:06d}")
        text = str(row.get("text", ""))
        try:
            outputs = model.forward(model_input_for_row(row, args))
            activation = tick_activation_matrix(outputs, args.n_neurons)
            branch_tensor = to_numpy(outputs.get("branch_tensor", np.zeros((0, 0), dtype=np.float32)))
            z = to_numpy(outputs.get("z", np.zeros((0,), dtype=np.float32))).reshape(-1)
            stim_vec = to_numpy(outputs.get("stim_vec", np.zeros(4, dtype=np.float32))).reshape(-1)

            np.savez_compressed(
                trace_dir / f"{record_id}.npz",
                activation=activation,
                branch_tensor=branch_tensor,
                z=z,
                stim_vec=stim_vec,
                dominant_branch_ids=np.asarray(dominant_branch_ids(outputs), dtype=np.int32),
                active_counts=np.asarray(active_count_series(outputs), dtype=np.int32),
            )
            summary_rows.append(
                summarize_trace(
                    row=row,
                    outputs=outputs,
                    activation=activation,
                    branch_tensor=branch_tensor,
                    z=z,
                )
            )
        except Exception as exc:
            errors.append({"record_id": record_id, "error": str(exc)})
        if args.progress_every > 0 and idx % args.progress_every == 0:
            print(f"export-neural-traces: {idx}/{len(rows)}")

    summary_csv = args.output_dir / "neural_trace_summary.csv"
    fieldnames = list(summary_rows[0].keys()) if summary_rows else LABEL_COLUMNS
    write_csv(summary_csv, summary_rows, fieldnames)

    manifest = {
        "input": str(args.input),
        "output_dir": str(args.output_dir),
        "trace_dir": str(trace_dir),
        "summary_csv": str(summary_csv),
        "requested_rows": len(rows),
        "ok_rows": len(summary_rows),
        "error_rows": len(errors),
        "n_neurons": args.n_neurons,
        "seed": args.seed,
        "z_encoder_mode": args.z_encoder_mode,
        "stim_source": args.stim_source,
        "sklearn_available": bool(SKLEARN_AVAILABLE),
        "dynamics": {
            "k_threshold_base": args.k_threshold_base,
            "k_remem_base": args.k_remem_base,
            "k_decay": args.k_decay,
            "refractory_ticks": args.refractory_ticks,
            "input_topk": args.input_topk,
            "input_signal_clip": args.input_signal_clip,
            "intrinsic_alignment_gain": args.intrinsic_alignment_gain,
            "fatigue_gain": args.fatigue_gain,
            "fatigue_threshold_gain": args.fatigue_threshold_gain,
            "fatigue_k_leak": args.fatigue_k_leak,
            "inhibitory_suppression_gain": args.inhibitory_suppression_gain,
            "density_control_start_tick": args.density_control_start_tick,
            "density_target_high": args.density_target_high,
            "density_soft_k_leak_gain": args.density_soft_k_leak_gain,
            "density_hard_cap": args.density_hard_cap,
            "density_pruned_fatigue_gain": args.density_pruned_fatigue_gain,
            "ne_thresh_reduce_gain": args.ne_thresh_reduce_gain,
            "ne_remem_reduce_gain": args.ne_remem_reduce_gain,
            "activity_churn_eps": args.activity_churn_eps,
        },
        "errors": errors[:20],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "neural_trace_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("outputs/targeted_records_trace_normalized.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/neural_trace_probe_v1"))
    parser.add_argument("--limit", type=int, default=24)
    parser.add_argument("--n-neurons", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--z-encoder-mode", choices=["stat", "transformer"], default="stat")
    parser.add_argument("--stim-source", choices=["auto", "text", "proxy"], default="auto")
    parser.add_argument("--max-ticks", type=int, default=64)
    parser.add_argument("--min-ticks-before-converged", type=int, default=6)
    parser.add_argument("--convergence-patience", type=int, default=4)
    parser.add_argument("--k-threshold-base", type=float, default=0.72)
    parser.add_argument("--k-remem-base", type=float, default=0.95)
    parser.add_argument("--k-decay", type=float, default=0.99)
    parser.add_argument("--refractory-ticks", type=int, default=1)
    parser.add_argument("--input-topk", type=int, default=2)
    parser.add_argument("--input-signal-clip", type=float, default=1.50)
    parser.add_argument("--intrinsic-alignment-gain", type=float, default=0.24)
    parser.add_argument("--fatigue-gain", type=float, default=0.30)
    parser.add_argument("--fatigue-threshold-gain", type=float, default=0.18)
    parser.add_argument("--fatigue-k-leak", type=float, default=0.08)
    parser.add_argument("--inhibitory-suppression-gain", type=float, default=0.18)
    parser.add_argument("--density-control-start-tick", type=int, default=0)
    parser.add_argument("--density-target-high", type=float, default=1.0)
    parser.add_argument("--density-soft-k-leak-gain", type=float, default=0.0)
    parser.add_argument("--density-hard-cap", type=float, default=1.0)
    parser.add_argument("--density-pruned-fatigue-gain", type=float, default=0.0)
    parser.add_argument("--ne-thresh-reduce-gain", type=float, default=0.25)
    parser.add_argument("--ne-remem-reduce-gain", type=float, default=0.25)
    parser.add_argument("--activity-churn-eps", type=float, default=0.02)
    parser.add_argument("--progress-every", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = run(args)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
