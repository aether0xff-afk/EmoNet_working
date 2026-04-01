from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from emonet.cli import STYLE_AXIS_NAMES
from emonet.core import LinearZtoSDecoder, ZSDecoderConfig


def summarize_csv(path: Path) -> dict[str, object]:
    df = pd.read_csv(path)
    summary: dict[str, object] = {
        "path": str(path),
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
    }
    if "label" in df.columns:
        summary["unique_labels"] = int(df["label"].nunique(dropna=False))
    if "keep_sample" in df.columns:
        keep = df["keep_sample"].fillna(False).astype(bool)
        summary["kept_rows"] = int(keep.sum())
        summary["keep_rate"] = round(float(keep.mean()), 6)
    if "consistency_l1" in df.columns:
        vals = pd.to_numeric(df["consistency_l1"], errors="coerce").dropna()
        if len(vals):
            summary["consistency_mean"] = round(float(vals.mean()), 6)
            summary["consistency_median"] = round(float(vals.median()), 6)
    return summary


def evaluate_decoder(df: pd.DataFrame, seeds: list[int], val_rows: int = 19) -> dict[str, object]:
    keep = df[df["keep_sample"].fillna(False).astype(bool)].reset_index(drop=True)
    z_cols = [f"z_{i}" for i in range(64)]
    s_cols = [f"s_{i}" for i in range(32)]
    z = keep[z_cols].to_numpy(dtype=np.float32)
    s = keep[s_cols].to_numpy(dtype=np.float32)

    runs: list[dict[str, object]] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(keep))
        val_idx = indices[:val_rows]
        train_idx = indices[val_rows:]

        decoder = LinearZtoSDecoder(
            config=ZSDecoderConfig(model_path=Path("artifacts/tmp_eval_decoder.npz"), ridge_alpha=1.0),
            z_dim=64,
            s_dim=32,
        )
        decoder.fit(z[train_idx], s[train_idx])
        pred = decoder.predict(z[val_idx])
        mae = float(np.mean(np.abs(pred - s[val_idx])))

        mean_baseline = np.broadcast_to(s[train_idx].mean(axis=0, dtype=np.float32), s[val_idx].shape)
        baseline_mae = float(np.mean(np.abs(mean_baseline - s[val_idx])))
        runs.append(
            {
                "seed": int(seed),
                "decoder_mae": round(mae, 6),
                "mean_baseline_mae": round(baseline_mae, 6),
                "gain": round(baseline_mae - mae, 6),
            }
        )

    return {
        "rows_used": int(len(keep)),
        "val_rows": int(val_rows),
        "runs": runs,
        "decoder_mae_mean": round(float(np.mean([row["decoder_mae"] for row in runs])), 6),
        "baseline_mae_mean": round(float(np.mean([row["mean_baseline_mae"] for row in runs])), 6),
        "mean_gain": round(float(np.mean([row["gain"] for row in runs])), 6),
    }


def summarize_style_bias(df: pd.DataFrame) -> dict[str, object]:
    keep = df[df["keep_sample"].fillna(False).astype(bool)].reset_index(drop=True)
    means = {}
    for idx, axis in enumerate(STYLE_AXIS_NAMES):
        means[axis] = float(keep[f"s_{idx}"].mean())

    sorted_axes = sorted(means.items(), key=lambda item: abs(item[1] - 0.5), reverse=True)
    return {
        "rows_used": int(len(keep)),
        "interesting_axes": {
            key: round(means[key], 4)
            for key in [
                "warmth",
                "politeness",
                "cooperativeness",
                "calmness",
                "softness",
                "sharpness",
                "tension",
                "dominance",
                "positivity",
                "seriousness",
            ]
        },
        "top_shifted_axes": [
            {"axis": axis, "mean": round(float(value), 4)}
            for axis, value in sorted_axes[:10]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--output-json", default="outputs/paper/paper_metrics_snapshot.json")
    args = parser.parse_args()

    root = Path(args.root)
    output_json = Path(args.output_json)

    files = {
        "z_train": root / "outputs" / "z" / "out_z_training.csv",
        "llm_subset": root / "outputs" / "llm" / "llm_subset.csv",
        "labeled_50_ollama": root / "outputs" / "llm" / "llm_subset_labeled_50_ollama.csv",
        "labeled_200_ollama": root / "outputs" / "llm" / "llm_subset_labeled_200_ollama.csv",
    }

    metrics: dict[str, object] = {"summaries": {}}
    for name, path in files.items():
        metrics["summaries"][name] = summarize_csv(path)

    labeled_200 = pd.read_csv(files["labeled_200_ollama"])
    metrics["decoder_eval"] = evaluate_decoder(labeled_200, seeds=[7, 13, 21, 42, 84], val_rows=19)
    metrics["style_bias"] = summarize_style_bias(labeled_200)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
