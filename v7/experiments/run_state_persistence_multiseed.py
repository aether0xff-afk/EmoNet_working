"""Repeat the persistent-state baseline comparison across multiple seeds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emonet_v7.run_logger import RunLogger  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", default="fixtures/semantic_training_episodes.yaml")
    parser.add_argument("--output", default="runs/state_persistence_multiseed")
    parser.add_argument("--encoder", choices=["hash", "lmstudio"], default="hash")
    parser.add_argument("--base-url")
    parser.add_argument("--embedding-model", default="text-embedding-nomic-embed-text-v1.5")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-neurons", type=int, default=128)
    parser.add_argument("--event-ticks", type=int, default=16)
    parser.add_argument("--stimulation-ticks", type=int, default=6)
    parser.add_argument("--device", default="cpu", help="Torch device: cpu, cuda, cuda:0, or auto")
    parser.add_argument("--no-cuda-fallback", action="store_true", help="Fail instead of falling back to CPU when CUDA is unavailable")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 13, 21, 42, 100])
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def run_seed(args: argparse.Namespace, logger: RunLogger, seed: int) -> dict:
    seed_output = Path(args.output) / f"seed_{seed}"
    command = [
        sys.executable,
        "experiments/run_state_persistence_ablation.py",
        "--fixture",
        args.fixture,
        "--output",
        str(seed_output),
        "--encoder",
        args.encoder,
        "--embedding-model",
        args.embedding_model,
        "--epochs",
        str(args.epochs),
        "--learning-rate",
        str(args.learning_rate),
        "--num-neurons",
        str(args.num_neurons),
        "--event-ticks",
        str(args.event_ticks),
        "--stimulation-ticks",
        str(args.stimulation_ticks),
        "--device",
        args.device,
        "--seed",
        str(seed),
    ]
    if args.base_url:
        command.extend(["--base-url", args.base_url])
    if args.no_cuda_fallback:
        command.append("--no-cuda-fallback")
    if args.quiet:
        command.append("--quiet")
    logger.log("seed.start", "Seed baseline 비교를 시작한다.", seed=seed, command=command)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    comparison_path = seed_output / "comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    logger.log("seed.done", "Seed baseline 비교를 마쳤다.", seed=seed, comparison=comparison)
    return {
        "seed": seed,
        "persistent_best_validation_total": comparison["persistent"]["best_validation_total"],
        "reset_best_validation_total": comparison["reset_each_transition"]["best_validation_total"],
        "delta_reset_minus_persistent": comparison["best_validation_delta_reset_minus_persistent"],
        "persistent_is_better": comparison["persistent_is_better"],
    }


def main() -> None:
    args = parse_args()
    if args.encoder == "lmstudio" and not args.base_url:
        raise ValueError("--base-url is required when --encoder lmstudio is used")
    if not args.seeds:
        raise ValueError("--seeds must not be empty")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(output_dir=output, verbose=not args.quiet)
    logger.section("multi-seed state persistence ablation")
    logger.log("config", "Multi-seed baseline 설정을 불러왔다.", **vars(args))

    rows = [run_seed(args, logger, seed) for seed in args.seeds]
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "by_seed.csv", index=False, encoding="utf-8-sig")
    summary = {
        "encoder": args.encoder,
        "requested_device": args.device,
        "seeds": args.seeds,
        "seed_count": len(args.seeds),
        "persistent_win_count": int(frame["persistent_is_better"].sum()),
        "persistent_win_rate": float(frame["persistent_is_better"].mean()),
        "delta_reset_minus_persistent_mean": float(frame["delta_reset_minus_persistent"].mean()),
        "delta_reset_minus_persistent_std": float(frame["delta_reset_minus_persistent"].std(ddof=1)) if len(frame) > 1 else 0.0,
        "note": (
            "Positive mean delta and repeated persistent wins suggest predictive value from preserved state "
            "on this starter fixture. This is not evidence of emotional semantics."
        ),
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.log("multiseed.done", "Multi-seed baseline 비교를 마쳤다.", summary=summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
