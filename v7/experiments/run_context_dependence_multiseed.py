"""Repeat ambiguity-controlled context dependence comparisons across seeds."""

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
    parser.add_argument("--fixture", default="fixtures/context_dependence_episodes.yaml")
    parser.add_argument("--output", default="runs/context_dependence_multiseed")
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
        "experiments/run_context_dependence_ablation.py",
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
    logger.log("seed.start", "Seed별 context dependence 비교를 시작한다.", seed=seed, command=command)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    comparison = json.loads((seed_output / "comparison.json").read_text(encoding="utf-8"))
    row = {
        "seed": seed,
        "persistent_minus_reset_trained_context_margin": comparison["persistent_minus_reset_trained_context_margin"],
        "persistent_minus_reset_trained_prediction_distance": comparison["persistent_minus_reset_trained_prediction_distance"],
        "persistent_minus_reset_trained_latent_distance": comparison["persistent_minus_reset_trained_latent_distance"],
    }
    logger.log("seed.done", "Seed별 context dependence 비교를 마쳤다.", **row)
    return row


def main() -> None:
    args = parse_args()
    if args.encoder == "lmstudio" and not args.base_url:
        raise ValueError("--base-url is required when --encoder lmstudio is used")
    if not args.seeds:
        raise ValueError("--seeds must not be empty")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(output_dir=output, verbose=not args.quiet)
    logger.section("multi-seed context dependence ablation")
    logger.log("config", "Multi-seed context dependence 설정을 불러왔다.", **vars(args))

    rows = [run_seed(args, logger, seed) for seed in args.seeds]
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "by_seed.csv", index=False, encoding="utf-8-sig")
    margin = frame["persistent_minus_reset_trained_context_margin"]
    prediction = frame["persistent_minus_reset_trained_prediction_distance"]
    latent = frame["persistent_minus_reset_trained_latent_distance"]
    summary = {
        "encoder": args.encoder,
        "requested_device": args.device,
        "seeds": args.seeds,
        "seed_count": len(args.seeds),
        "positive_context_margin_count": int((margin > 0).sum()),
        "positive_context_margin_rate": float((margin > 0).mean()),
        "context_margin_mean": float(margin.mean()),
        "context_margin_std": float(margin.std(ddof=1)) if len(frame) > 1 else 0.0,
        "prediction_distance_mean": float(prediction.mean()),
        "prediction_distance_std": float(prediction.std(ddof=1)) if len(frame) > 1 else 0.0,
        "latent_distance_mean": float(latent.mean()),
        "latent_distance_std": float(latent.std(ddof=1)) if len(frame) > 1 else 0.0,
        "note": (
            "Positive context-margin differences across seeds suggest that preserved state contributes to "
            "context-sensitive prediction on the controlled fixture. This is not evidence of emotional semantics."
        ),
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.log("multiseed.done", "Multi-seed context dependence 비교를 마쳤다.", summary=summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
