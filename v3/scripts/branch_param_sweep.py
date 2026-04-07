from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from emonet.cli import (
    DEFAULT_Z_ENCODER_MODEL_PATH,
    build_model,
    load_training_json_as_dataframe,
    probe_branch_lengths,
    resolve_text_column,
    sample_probe_dataframe,
    summarize_branch_lengths,
)
from emonet.core import EmoNetConfig


SWEEPABLE_FIELDS: dict[str, type] = {
    "max_ticks": int,
    "min_ticks_before_converged": int,
    "k_threshold_base": float,
    "k_remem_base": float,
    "k_decay": float,
    "refractory_ticks": int,
    "memory_decay": float,
    "memory_stim_mix": float,
    "memory_k_mix": float,
    "max_out_degree": int,
    "min_out_degree": int,
    "dopa_rewire_gain": float,
    "sero_prune_gain": float,
    "mela_dropout_gain": float,
    "ne_thresh_reduce_gain": float,
    "ne_remem_reduce_gain": float,
    "global_recovery_rate": float,
    "topk_branches": int,
    "branch_end_window": int,
    "branch_length_bonus": float,
}

DEFAULT_OFAT_STEPS: dict[str, Any] = {
    "max_ticks": 4,
    "min_ticks_before_converged": 2,
    "k_threshold_base": 0.04,
    "k_remem_base": 0.05,
    "k_decay": 0.002,
    "refractory_ticks": 1,
    "memory_decay": 0.003,
    "memory_stim_mix": 0.05,
    "memory_k_mix": 0.10,
    "max_out_degree": 2,
    "min_out_degree": 1,
    "dopa_rewire_gain": 0.20,
    "sero_prune_gain": 0.02,
    "mela_dropout_gain": 0.02,
    "ne_thresh_reduce_gain": 0.05,
    "ne_remem_reduce_gain": 0.05,
    "global_recovery_rate": 0.02,
    "topk_branches": 1,
    "branch_end_window": 2,
    "branch_length_bonus": 0.10,
}


def load_svg_helpers():
    helper_path = PROJECT_ROOT / "scripts" / "generate_paper_svgs.py"
    spec = importlib.util.spec_from_file_location("generate_paper_svgs_module", helper_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def parse_assignment(raw: str) -> tuple[str, Any]:
    if "=" not in raw:
        raise ValueError(f"expected key=value assignment, got: {raw}")
    key, value = raw.split("=", 1)
    key = key.strip()
    value = value.strip()
    if key not in SWEEPABLE_FIELDS:
        valid = ", ".join(sorted(SWEEPABLE_FIELDS))
        raise ValueError(f"unknown parameter '{key}'. valid fields: {valid}")
    caster = SWEEPABLE_FIELDS[key]
    return key, caster(value)


def parse_value_list(raw: str, caster: type) -> list[Any]:
    values = [token.strip() for token in str(raw).split(",") if token.strip()]
    if not values:
        raise ValueError("at least one sweep value is required")
    return [caster(value) for value in values]


def format_value_label(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def load_input_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    if bool(args.input_csv) == bool(args.input_json):
        raise ValueError("provide exactly one of --input-csv or --input-json")
    if args.input_json:
        return load_training_json_as_dataframe(Path(args.input_json))
    return pd.read_csv(Path(args.input_csv))


def get_default_param_values() -> dict[str, Any]:
    config = EmoNetConfig()
    return {field_name: getattr(config, field_name) for field_name in SWEEPABLE_FIELDS}


def build_model_namespace(args: argparse.Namespace, overrides: dict[str, Any]) -> argparse.Namespace:
    payload = {
        "dataset_csv": args.dataset_csv,
        "benchmark_csv": args.benchmark_csv,
        "model_cache_path": args.model_cache_path,
        "max_samples": args.max_samples,
        "force_refit": args.force_refit,
        "seed": args.model_seed,
        "z_dim": args.z_dim,
        "z_encoder_mode": "stat",
        "z_encoder_path": str(DEFAULT_Z_ENCODER_MODEL_PATH),
    }
    for field_name in SWEEPABLE_FIELDS:
        payload[field_name] = None
    payload.update(overrides)
    return argparse.Namespace(**payload)


def parse_assignment_map(raw_values: list[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for raw in raw_values:
        key, value = parse_assignment(raw)
        parsed[key] = value
    return parsed


def bounded_step_value(base_value: Any, step_value: Any, direction: int, caster: type) -> Any:
    candidate = base_value + direction * step_value
    if caster is int:
        return max(1, int(round(candidate)))
    if caster is float:
        return float(max(0.0, candidate))
    return caster(candidate)


def build_ofat_specs(args: argparse.Namespace, fixed: dict[str, Any]) -> list[dict[str, Any]]:
    defaults = get_default_param_values()
    center = dict(defaults)
    center.update(fixed)
    step_overrides = parse_assignment_map(args.step)

    raw_params = [token.strip() for token in str(args.ofat_params or "").split(",") if token.strip()]
    if not raw_params:
        raise ValueError("ofat_params must not be empty")
    if len(raw_params) == 1 and raw_params[0].lower() == "all":
        params = list(SWEEPABLE_FIELDS.keys())
    else:
        params = raw_params

    invalid = [name for name in params if name not in SWEEPABLE_FIELDS]
    if invalid:
        valid = ", ".join(sorted(SWEEPABLE_FIELDS))
        raise ValueError(f"unknown ofat params: {', '.join(invalid)}. valid fields: {valid}")

    specs: list[dict[str, Any]] = []
    if not args.skip_baseline:
        specs.append({"name": "baseline", "params": dict(center)})

    for param_name in params:
        caster = SWEEPABLE_FIELDS[param_name]
        base_value = center[param_name]
        step_value = step_overrides.get(param_name, DEFAULT_OFAT_STEPS.get(param_name))
        if step_value is None:
            raise ValueError(f"no default step configured for '{param_name}', use --step {param_name}=...")
        down_value = bounded_step_value(base_value, step_value, -1, caster)
        up_value = bounded_step_value(base_value, step_value, 1, caster)

        for direction_name, candidate in (("down", down_value), ("base", base_value), ("up", up_value)):
            params_payload = dict(center)
            params_payload[param_name] = candidate
            specs.append(
                {
                    "name": f"{param_name}:{direction_name}={format_value_label(candidate)}",
                    "params": params_payload,
                }
            )
    return specs


def build_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    fixed = parse_assignment_map(args.fixed)

    if args.ofat_params:
        return build_ofat_specs(args, fixed)

    if args.config_json:
        config_payload = json.loads(Path(args.config_json).read_text(encoding="utf-8"))
        if not isinstance(config_payload, list):
            raise ValueError("config_json must contain a JSON array")
        specs: list[dict[str, Any]] = []
        for item in config_payload:
            if not isinstance(item, dict) or "name" not in item:
                raise ValueError("each config_json item must be an object with a 'name'")
            params = dict(fixed)
            params.update(dict(item.get("params", {})))
            specs.append({"name": str(item["name"]), "params": params})
        return specs

    if not args.sweep_param or not args.values:
        raise ValueError("provide either --config-json or both --sweep-param and --values")
    if args.sweep_param not in SWEEPABLE_FIELDS:
        valid = ", ".join(sorted(SWEEPABLE_FIELDS))
        raise ValueError(f"unknown sweep_param '{args.sweep_param}'. valid fields: {valid}")

    caster = SWEEPABLE_FIELDS[args.sweep_param]
    values = parse_value_list(args.values, caster)
    specs = []
    for value in values:
        params = dict(fixed)
        params[args.sweep_param] = value
        specs.append({"name": f"{args.sweep_param}={format_value_label(value)}", "params": params})
    return specs


def summarize_records(config_name: str, params: dict[str, Any], result_df: pd.DataFrame) -> dict[str, Any]:
    stats = summarize_branch_lengths(result_df["dominant_branch_len"].astype(int).tolist())
    row: dict[str, Any] = {
        "config_name": config_name,
        "params_json": json.dumps(params, ensure_ascii=False, sort_keys=True),
        "rows": int(stats["rows"]),
        "mean": float(stats["mean"]),
        "median": float(stats["median"]),
        "len1": int(stats["len1"]),
        "len1_ratio": float(stats["len1_ratio"]),
        "max": int(stats["max"]),
        "p90": int(stats["p90"]),
        "p95": int(stats["p95"]),
    }
    bucket_counts = dict(stats["bucket_counts"])
    total = max(1, int(stats["rows"]))
    for bucket_name, bucket_value in bucket_counts.items():
        row[f"{bucket_name}_count"] = int(bucket_value)
        row[f"{bucket_name}_ratio"] = round(float(bucket_value) / float(total), 4)
    return row


def score_summary_row(row: pd.Series) -> float:
    len1_ratio = float(row["len1_ratio"])
    mean = float(row["mean"])
    p95 = float(row["p95"])
    len16_ratio = float(row.get("len16_plus_ratio", 0.0))
    len8_ratio = float(row.get("len8_15_ratio", 0.0))
    # Lower len1 is primary. Mean/p95 and longer-tail ratios are secondary.
    return round(
        (1.0 - len1_ratio) * 100.0
        + mean * 2.0
        + p95 * 0.5
        + len8_ratio * 5.0
        + len16_ratio * 10.0,
        4,
    )


def render_figures(summary_df: pd.DataFrame, output_dir: Path, title_suffix: str) -> list[Path]:
    helpers = load_svg_helpers()
    figure_paths: list[Path] = []

    labels = summary_df["config_name"].astype(str).tolist()
    mean_values = summary_df["mean"].astype(float).tolist()
    len1_values = summary_df["len1_ratio"].astype(float).tolist()
    p95_values = summary_df["p95"].astype(float).tolist()
    colors = ["#0f766e", "#1d4ed8", "#ea580c", "#b45309", "#7c3aed", "#be123c", "#4d7c0f"][: len(labels)]
    while len(colors) < len(labels):
        colors.append("#334155")

    mean_path = output_dir / "branch_sweep_mean.svg"
    helpers.bar_chart_vertical(
        path=mean_path,
        title="Branch Sweep: Mean Dominant Branch Length",
        subtitle=title_suffix,
        labels=labels,
        values=mean_values,
        colors=colors,
        y_label="mean branch length",
        note="higher is better if len1_ratio also decreases",
        value_format="{:.2f}",
    )
    figure_paths.append(mean_path)

    len1_path = output_dir / "branch_sweep_len1_ratio.svg"
    helpers.bar_chart_vertical(
        path=len1_path,
        title="Branch Sweep: Length-1 Ratio",
        subtitle=title_suffix,
        labels=labels,
        values=len1_values,
        colors=colors,
        y_label="ratio of dominant_branch_len = 1",
        note="lower is better",
        value_format="{:.3f}",
    )
    figure_paths.append(len1_path)

    p95_path = output_dir / "branch_sweep_p95.svg"
    helpers.bar_chart_vertical(
        path=p95_path,
        title="Branch Sweep: P95 Branch Length",
        subtitle=title_suffix,
        labels=labels,
        values=p95_values,
        colors=colors,
        y_label="p95 branch length",
        note="checks whether the tail grows beyond isolated spikes",
        value_format="{:.0f}",
    )
    figure_paths.append(p95_path)

    bucket_series = ["len1_ratio", "len2_3_ratio", "len4_7_ratio", "len8_15_ratio", "len16_plus_ratio"]
    bucket_values = [
        [float(summary_df.iloc[row_idx][series_name]) for series_name in bucket_series]
        for row_idx in range(len(summary_df))
    ]
    bucket_colors = ["#dc2626", "#f59e0b", "#65a30d", "#0891b2", "#2563eb"]
    bucket_path = output_dir / "branch_sweep_bucket_ratio.svg"
    helpers.grouped_bar_chart(
        path=bucket_path,
        title="Branch Sweep: Length Bucket Ratios",
        subtitle=title_suffix,
        group_labels=labels,
        series_labels=["len1", "2-3", "4-7", "8-15", "16+"],
        values=bucket_values,
        colors=bucket_colors,
        y_label="ratio",
        note="distribution view for paper figures",
    )
    figure_paths.append(bucket_path)

    return figure_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", default=None)
    parser.add_argument("--input-json", default=None)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--sample-mode", choices=["head", "random"], default="random")
    parser.add_argument("--seed", type=int, default=42, help="seed for sample selection")
    parser.add_argument("--model-seed", type=int, default=42, help="seed for model graph initialization")
    parser.add_argument("--progress-every", type=int, default=20)
    parser.add_argument("--dataset-csv", default=None)
    parser.add_argument("--benchmark-csv", default=None)
    parser.add_argument("--model-cache-path", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--force-refit", action="store_true")
    parser.add_argument("--z-dim", type=int, default=64)
    parser.add_argument("--sweep-param", default=None)
    parser.add_argument("--values", default=None)
    parser.add_argument("--fixed", action="append", default=[])
    parser.add_argument("--ofat-params", default=None, help="comma-separated param names or 'all' for down/base/up sweeps")
    parser.add_argument("--step", action="append", default=[], help="override OFAT step as key=value")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--config-json", default=None)
    parser.add_argument("--output-dir", default=str(Path("outputs") / "branch_sweep" / "latest"))
    args = parser.parse_args()

    input_df = load_input_dataframe(args)
    text_column = resolve_text_column(input_df, args.text_column)
    sampled_df = sample_probe_dataframe(input_df, args.sample_size, args.sample_mode, args.seed)
    specs = build_specs(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sampled_df.to_csv(output_dir / "sampled_inputs.csv", index=False, encoding="utf-8-sig")

    summary_rows: list[dict[str, Any]] = []
    detail_frames: list[pd.DataFrame] = []
    for spec in specs:
        config_name = str(spec["name"])
        params = dict(spec["params"])
        model_args = build_model_namespace(args, params)
        model = build_model(model_args)
        result_df = probe_branch_lengths(
            model=model,
            df=sampled_df,
            text_column=text_column,
            progress_every=args.progress_every,
        )
        result_df = result_df.copy()
        result_df["config_name"] = config_name
        result_df["params_json"] = json.dumps(params, ensure_ascii=False, sort_keys=True)
        detail_frames.append(result_df)
        summary_rows.append(summarize_records(config_name, params, result_df))

    summary_df = pd.DataFrame(summary_rows)
    summary_df["score"] = summary_df.apply(score_summary_row, axis=1)
    summary_df = summary_df.sort_values(by=["score", "len1_ratio", "mean"], ascending=[False, True, False]).reset_index(drop=True)
    detail_df = pd.concat(detail_frames, ignore_index=True)

    summary_csv = output_dir / "summary.csv"
    details_csv = output_dir / "details.csv"
    spec_json = output_dir / "specs.json"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    detail_df.to_csv(details_csv, index=False, encoding="utf-8-sig")
    spec_json.write_text(json.dumps(specs, ensure_ascii=False, indent=2), encoding="utf-8")

    title_suffix = f"sample_rows={len(sampled_df)}, sample_mode={args.sample_mode}, sample_seed={args.seed}, model_seed={args.model_seed}"
    figure_paths = render_figures(summary_df, output_dir, title_suffix)

    payload = {
        "input_rows": int(len(input_df)),
        "sample_rows": int(len(sampled_df)),
        "summary_csv": str(summary_csv),
        "details_csv": str(details_csv),
        "spec_json": str(spec_json),
        "figure_paths": [str(path) for path in figure_paths],
        "best_config": summary_df.iloc[0].to_dict() if not summary_df.empty else {},
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
