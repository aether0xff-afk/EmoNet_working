from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from emonet.core import EmoNet, EmoNetConfig, LinearZtoSDecoder, ZSDecoderConfig


DEFAULT_SEEDS = [7, 13, 21, 42, 84]


def parse_seeds(raw: str) -> list[int]:
    values = [token.strip() for token in raw.split(",")]
    seeds = [int(token) for token in values if token]
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def load_keep_rows(path: Path, sample_limit: int | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "keep_sample" not in df.columns:
        raise ValueError(f"'keep_sample' column not found in {path}")
    keep = df[df["keep_sample"].fillna(False).astype(bool)].reset_index(drop=True)
    if sample_limit is not None and sample_limit > 0:
        keep = keep.head(sample_limit).reset_index(drop=True)
    if keep.empty:
        raise ValueError("no keep_sample rows available")
    return keep


def evaluate_decoder_features(
    x: np.ndarray,
    s: np.ndarray,
    seeds: list[int],
    val_rows: int,
    ridge_alpha: float,
) -> dict[str, object]:
    if len(x) <= val_rows + 1:
        raise ValueError(f"need more rows than val_rows+1, got {len(x)} rows and val_rows={val_rows}")

    runs: list[dict[str, object]] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(x))
        val_idx = indices[:val_rows]
        train_idx = indices[val_rows:]

        decoder = LinearZtoSDecoder(
            config=ZSDecoderConfig(model_path=Path("artifacts/tmp_eval_decoder.npz"), ridge_alpha=ridge_alpha),
            z_dim=x.shape[1],
            s_dim=s.shape[1],
        )
        decoder.fit(x[train_idx], s[train_idx])
        pred = decoder.predict(x[val_idx])
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
        "rows_used": int(len(x)),
        "val_rows": int(val_rows),
        "decoder_mae_mean": round(float(np.mean([row["decoder_mae"] for row in runs])), 6),
        "baseline_mae_mean": round(float(np.mean([row["mean_baseline_mae"] for row in runs])), 6),
        "mean_gain": round(float(np.mean([row["gain"] for row in runs])), 6),
        "runs": runs,
    }


def evaluate_mean_baseline(s: np.ndarray, seeds: list[int], val_rows: int) -> dict[str, object]:
    if len(s) <= val_rows + 1:
        raise ValueError(f"need more rows than val_rows+1, got {len(s)} rows and val_rows={val_rows}")
    runs: list[dict[str, object]] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        indices = rng.permutation(len(s))
        val_idx = indices[:val_rows]
        train_idx = indices[val_rows:]
        mean_baseline = np.broadcast_to(s[train_idx].mean(axis=0, dtype=np.float32), s[val_idx].shape)
        baseline_mae = float(np.mean(np.abs(mean_baseline - s[val_idx])))
        runs.append(
            {
                "seed": int(seed),
                "decoder_mae": round(baseline_mae, 6),
            }
        )
    return {
        "rows_used": int(len(s)),
        "val_rows": int(val_rows),
        "decoder_mae_mean": round(float(np.mean([row["decoder_mae"] for row in runs])), 6),
        "runs": runs,
    }


def evaluate_text_baseline(
    texts: list[str],
    s: np.ndarray,
    seeds: list[int],
    val_rows: int,
    ridge_alpha: float,
) -> dict[str, object]:
    runs: list[dict[str, object]] = []
    text_arr = np.asarray(texts, dtype=object)
    for seed in seeds:
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
            config=ZSDecoderConfig(model_path=Path("artifacts/tmp_text_decoder.npz"), ridge_alpha=ridge_alpha),
            z_dim=train_x.shape[1],
            s_dim=s.shape[1],
        )
        decoder.fit(train_x, s[train_idx])
        pred = decoder.predict(val_x)
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
        "rows_used": int(len(texts)),
        "val_rows": int(val_rows),
        "decoder_mae_mean": round(float(np.mean([row["decoder_mae"] for row in runs])), 6),
        "baseline_mae_mean": round(float(np.mean([row["mean_baseline_mae"] for row in runs])), 6),
        "mean_gain": round(float(np.mean([row["gain"] for row in runs])), 6),
        "runs": runs,
    }


def collect_forward_features(
    texts: list[str],
    config: EmoNetConfig,
    progress_every: int,
) -> dict[str, object]:
    model = EmoNet(config=config)
    z_rows: list[np.ndarray] = []
    dominant_branch_lens: list[int] = []
    pruned_ticks: list[int] = []
    fired_edges: list[int] = []
    active_nodes: list[int] = []

    for idx, text in enumerate(texts, start=1):
        outputs = model.forward(text)
        z_rows.append(np.asarray(outputs["z"], dtype=np.float32).reshape(-1))
        dominant_branch_lens.append(int(len(outputs["dominant_branch"])))
        pruned_ticks.append(int(len(outputs["pruned_branch_log"])))
        fired_edges.append(int(sum(len(record.edges_fired) for record in outputs["pruned_branch_log"])))
        active_nodes.append(int(sum(len(record.active_nodes) for record in outputs["pruned_branch_log"])))
        if progress_every > 0 and idx % progress_every == 0:
            print(json.dumps({"progress": idx, "z_dim": config.z_dim}, ensure_ascii=False))

    return {
        "z": np.vstack(z_rows).astype(np.float32),
        "dominant_branch_len_mean": round(float(np.mean(dominant_branch_lens)), 6),
        "dominant_branch_len_std": round(float(np.std(dominant_branch_lens)), 6),
        "pruned_ticks_mean": round(float(np.mean(pruned_ticks)), 6),
        "fired_edges_mean": round(float(np.mean(fired_edges)), 6),
        "active_nodes_mean": round(float(np.mean(active_nodes)), 6),
    }


def make_row(
    family: str,
    name: str,
    config_text: str,
    metrics: dict[str, object],
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    row = {
        "experiment_family": family,
        "name": name,
        "config": config_text,
        "rows_used": metrics["rows_used"],
        "val_rows": metrics["val_rows"],
        "decoder_mae_mean": metrics["decoder_mae_mean"],
    }
    if "baseline_mae_mean" in metrics:
        row["baseline_mae_mean"] = metrics["baseline_mae_mean"]
        row["mean_gain"] = metrics["mean_gain"]
    if extra:
        row.update(extra)
    return row


def build_baseline_predictor_table(
    keep: pd.DataFrame,
    texts: list[str],
    s: np.ndarray,
    seeds: list[int],
    val_rows: int,
    ridge_alpha: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    details: dict[str, object] = {}

    mean_metrics = evaluate_mean_baseline(s, seeds=seeds, val_rows=val_rows)
    rows.append(
        make_row(
            family="baseline_predictor",
            name="mean_baseline",
            config_text="predict train-set mean style vector",
            metrics=mean_metrics,
            extra={"feature_dim": 0},
        )
    )
    details["mean_baseline"] = mean_metrics

    stim_x = keep[["dopamine", "serotonin", "norepinephrine", "melatonin"]].to_numpy(dtype=np.float32)
    stim_metrics = evaluate_decoder_features(stim_x, s, seeds=seeds, val_rows=val_rows, ridge_alpha=ridge_alpha)
    rows.append(
        make_row(
            family="baseline_predictor",
            name="stim_only_ridge",
            config_text="4D stim_vec -> s ridge",
            metrics=stim_metrics,
            extra={"feature_dim": 4},
        )
    )
    details["stim_only_ridge"] = stim_metrics

    text_metrics = evaluate_text_baseline(texts, s, seeds=seeds, val_rows=val_rows, ridge_alpha=ridge_alpha)
    rows.append(
        make_row(
            family="baseline_predictor",
            name="text_tfidf_ridge",
            config_text="char TF-IDF + SVD + ridge",
            metrics=text_metrics,
            extra={"feature_dim": 128},
        )
    )
    details["text_tfidf_ridge"] = text_metrics

    z_cols = [f"z_{idx}" for idx in range(64)]
    z_x = keep[z_cols].to_numpy(dtype=np.float32)
    emonet_metrics = evaluate_decoder_features(z_x, s, seeds=seeds, val_rows=val_rows, ridge_alpha=ridge_alpha)
    rows.append(
        make_row(
            family="baseline_predictor",
            name="emonet_z64_ridge",
            config_text="published 64D z -> s ridge",
            metrics=emonet_metrics,
            extra={"feature_dim": 64},
        )
    )
    details["emonet_z64_ridge"] = emonet_metrics

    return pd.DataFrame(rows), details


def build_z_size_table(
    texts: list[str],
    s: np.ndarray,
    seeds: list[int],
    val_rows: int,
    ridge_alpha: float,
    progress_every: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    rows: list[dict[str, object]] = []
    details: dict[str, object] = {}
    for z_dim in (32, 64, 128):
        config = EmoNetConfig(seed=42, z_dim=z_dim)
        features = collect_forward_features(texts, config=config, progress_every=progress_every)
        metrics = evaluate_decoder_features(
            features["z"],
            s,
            seeds=seeds,
            val_rows=val_rows,
            ridge_alpha=ridge_alpha,
        )
        rows.append(
            make_row(
                family="z_size",
                name=f"z_dim_{z_dim}",
                config_text=f"full model with z_dim={z_dim}",
                metrics=metrics,
                extra={
                    "z_dim": z_dim,
                    "dominant_branch_len_mean": features["dominant_branch_len_mean"],
                    "dominant_branch_len_std": features["dominant_branch_len_std"],
                    "pruned_ticks_mean": features["pruned_ticks_mean"],
                    "fired_edges_mean": features["fired_edges_mean"],
                    "active_nodes_mean": features["active_nodes_mean"],
                },
            )
        )
        details[f"z_dim_{z_dim}"] = {"metrics": metrics, "features": features}
    return pd.DataFrame(rows), details


def build_neuron_ablation_table(
    texts: list[str],
    s: np.ndarray,
    seeds: list[int],
    val_rows: int,
    ridge_alpha: float,
    progress_every: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    base = EmoNetConfig(seed=42, z_dim=64)
    configs = [
        ("full", base, "full 115/115/26 configuration"),
        (
            "without_inhibitory",
            replace(base, n_inhibitory=0, n_excitatory=230, n_modulatory=26),
            "remove inhibitory neurons, keep total N=256",
        ),
        (
            "without_excitatory",
            replace(base, n_inhibitory=230, n_excitatory=0, n_modulatory=26),
            "remove excitatory neurons, keep total N=256",
        ),
        (
            "without_modulatory",
            replace(base, n_inhibitory=128, n_excitatory=128, n_modulatory=0),
            "remove modulatory neurons, rebalance inhibitory/excitatory",
        ),
        (
            "without_memory",
            replace(
                base,
                max_memory_per_neuron=0,
                memory_decay=0.0,
                memory_delete_threshold=1.0,
                memory_sim_gain=0.0,
                memory_stim_mix=0.0,
                memory_k_mix=0.0,
            ),
            "disable memory accumulation and reuse",
        ),
        (
            "without_rewiring",
            replace(base, dopa_rewire_gain=0.0, sero_prune_gain=0.0),
            "disable prune/add rewiring dynamics",
        ),
    ]

    rows: list[dict[str, object]] = []
    details: dict[str, object] = {}
    for name, config, note in configs:
        features = collect_forward_features(texts, config=config, progress_every=progress_every)
        metrics = evaluate_decoder_features(
            features["z"],
            s,
            seeds=seeds,
            val_rows=val_rows,
            ridge_alpha=ridge_alpha,
        )
        rows.append(
            make_row(
                family="neuron_ablation",
                name=name,
                config_text=note,
                metrics=metrics,
                extra={
                    "z_dim": config.z_dim,
                    "dominant_branch_len_mean": features["dominant_branch_len_mean"],
                    "dominant_branch_len_std": features["dominant_branch_len_std"],
                    "pruned_ticks_mean": features["pruned_ticks_mean"],
                    "fired_edges_mean": features["fired_edges_mean"],
                    "active_nodes_mean": features["active_nodes_mean"],
                },
            )
        )
        details[name] = {"metrics": metrics, "features": features, "config": config.__dict__}
    return pd.DataFrame(rows), details


def write_outputs(output_dir: Path, prefix: str, table: pd.DataFrame, details: dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_dir / f"{prefix}.csv", index=False, encoding="utf-8-sig")
    (output_dir / f"{prefix}.json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build offline paper tables for ablations and z-size.")
    parser.add_argument(
        "--labeled-csv",
        default=str(PROJECT_ROOT / "outputs" / "llm" / "llm_subset_labeled_200_ollama.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "outputs" / "paper" / "requested_tables"),
    )
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--seeds", default="7,13,21,42,84")
    parser.add_argument("--val-rows", type=int, default=19)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    labeled_csv = Path(args.labeled_csv)
    output_dir = Path(args.output_dir)
    seeds = parse_seeds(args.seeds)

    keep = load_keep_rows(labeled_csv, sample_limit=args.sample_limit)
    if len(keep) <= 2:
        raise ValueError("need at least 3 keep_sample rows to build paper tables")
    effective_val_rows = min(args.val_rows, max(1, len(keep) - 2))
    texts = keep["text"].astype(str).tolist()
    s_cols = [f"s_{idx}" for idx in range(32)]
    missing_s = [column for column in s_cols if column not in keep.columns]
    if missing_s:
        raise ValueError(f"missing required style columns: {', '.join(missing_s)}")
    s = keep[s_cols].to_numpy(dtype=np.float32)

    baseline_table, baseline_details = build_baseline_predictor_table(
        keep=keep,
        texts=texts,
        s=s,
        seeds=seeds,
        val_rows=effective_val_rows,
        ridge_alpha=args.ridge_alpha,
    )
    write_outputs(output_dir, "baseline_predictor_table", baseline_table, baseline_details)

    z_size_table, z_size_details = build_z_size_table(
        texts=texts,
        s=s,
        seeds=seeds,
        val_rows=effective_val_rows,
        ridge_alpha=args.ridge_alpha,
        progress_every=args.progress_every,
    )
    write_outputs(output_dir, "z_size_ablation_table", z_size_table, z_size_details)

    neuron_table, neuron_details = build_neuron_ablation_table(
        texts=texts,
        s=s,
        seeds=seeds,
        val_rows=effective_val_rows,
        ridge_alpha=args.ridge_alpha,
        progress_every=args.progress_every,
    )
    write_outputs(output_dir, "neuron_function_ablation_table", neuron_table, neuron_details)

    summary = {
        "rows_used": int(len(keep)),
        "val_rows": int(effective_val_rows),
        "output_dir": str(output_dir),
        "files": [
            str(output_dir / "baseline_predictor_table.csv"),
            str(output_dir / "baseline_predictor_table.json"),
            str(output_dir / "z_size_ablation_table.csv"),
            str(output_dir / "z_size_ablation_table.json"),
            str(output_dir / "neuron_function_ablation_table.csv"),
            str(output_dir / "neuron_function_ablation_table.json"),
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
