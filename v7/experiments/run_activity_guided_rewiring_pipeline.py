"""Run rewiring stability search and cluster diagnostics in one command.

The pipeline:
1. searches for a semantic-preserving rewiring region,
2. summarizes and selects the best eligible rewiring config,
3. evaluates adjacency-community evidence on its saved checkpoints,
4. summarizes the final community report.

For remote LM Studio usage, set ``EMONET_LMSTUDIO_BASE_URL`` once on the
external machine. A command-line ``--base-url`` still overrides it.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = PROJECT_ROOT / "experiments"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/activity_guided_rewiring_pipeline_lmstudio")
    parser.add_argument("--baseline", default="runs/memory_threshold_parameter_sweep_lmstudio")
    parser.add_argument("--fixture", default="fixtures/semantic_alignment_episodes.yaml")
    parser.add_argument("--encoder", choices=["hash", "lmstudio"], default="lmstudio")
    parser.add_argument("--base-url")
    parser.add_argument("--embedding-model", default="text-embedding-nomic-embed-text-v1.5")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 13, 21, 42, 100])
    parser.add_argument("--null-permutations", type=int, default=64)
    return parser.parse_args()


def run(command: list[str]) -> None:
    """Execute one child command and fail immediately on error."""

    print("\n$ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def common_remote_args(args: argparse.Namespace) -> list[str]:
    common = [
        "--fixture",
        args.fixture,
        "--encoder",
        args.encoder,
        "--embedding-model",
        args.embedding_model,
    ]
    if args.base_url:
        common.extend(["--base-url", args.base_url])
    return common


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"expected report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    sweep_output = output / "stability_sweep"
    cluster_output = output / "rewired_cluster"
    output.mkdir(parents=True, exist_ok=True)

    seed_args = ["--seeds", *[str(seed) for seed in args.seeds]]
    run(
        [
            sys.executable,
            str(EXPERIMENTS / "run_activity_guided_rewiring_stability_sweep.py"),
            "--output",
            str(sweep_output),
            "--epochs",
            str(args.epochs),
            *seed_args,
            *common_remote_args(args),
        ]
    )
    run(
        [
            sys.executable,
            str(EXPERIMENTS / "summarize_activity_guided_rewiring_stability_sweep.py"),
            "--input",
            str(sweep_output),
            "--baseline",
            args.baseline,
        ]
    )
    sweep_report = read_json(sweep_output / "decision_report.json")
    best = sweep_report.get("best_semantic_preserving_rewiring_config")
    if not best:
        final_report = {
            "stage_verdict": "semantic_preserving_rewiring_region_not_found",
            "stability_sweep_report": sweep_report,
            "next_step": "Revise or further weaken the rewiring rule before running a cluster diagnostic.",
        }
        (output / "pipeline_report.json").write_text(json.dumps(final_report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(final_report, ensure_ascii=False, indent=2))
        return

    config_key = str(best["config_key"])
    run(
        [
            sys.executable,
            str(EXPERIMENTS / "run_activity_guided_rewiring_emergent_cluster_benchmark.py"),
            "--input",
            str(sweep_output),
            "--output",
            str(cluster_output),
            "--config-key",
            config_key,
            "--null-permutations",
            str(args.null_permutations),
            *seed_args,
            *common_remote_args(args),
        ]
    )
    run(
        [
            sys.executable,
            str(EXPERIMENTS / "summarize_activity_guided_rewiring_emergent_cluster_benchmark.py"),
            "--input",
            str(cluster_output),
        ]
    )
    cluster_report = read_json(cluster_output / "decision_report.json")
    final_report = {
        "stage_verdict": cluster_report.get("stage_verdict"),
        "best_semantic_preserving_rewiring_config": best,
        "stability_sweep_report": sweep_report,
        "rewired_cluster_report": cluster_report,
        "interpretation_boundary": (
            "This pipeline identifies a controlled semantic-preserving rewiring regime and evaluates adjacency-community evidence. "
            "It does not establish final rewiring rules, emotional ground truth, stable neuron roles, or biological fidelity."
        ),
    }
    (output / "pipeline_report.json").write_text(json.dumps(final_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(final_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
