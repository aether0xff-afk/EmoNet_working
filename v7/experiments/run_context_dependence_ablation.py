"""Train persistent/reset models and compare context-sensitive predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emonet_v7.run_logger import RunLogger  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", default="fixtures/context_dependence_episodes.yaml")
    parser.add_argument("--output", default="runs/context_dependence_ablation")
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def run_command(command: list[str], logger: RunLogger, event: str) -> None:
    logger.log(event, "하위 실험을 실행한다.", command=command)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    if args.encoder == "lmstudio" and not args.base_url:
        raise ValueError("--base-url is required when --encoder lmstudio is used")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(output_dir=output, verbose=not args.quiet)
    logger.section("context dependence ablation")
    logger.log("config", "Context dependence baseline 설정을 불러왔다.", **vars(args))

    training_output = output / "training"
    train_command = [
        sys.executable,
        "experiments/run_state_persistence_ablation.py",
        "--fixture",
        args.fixture,
        "--output",
        str(training_output),
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
        str(args.seed),
    ]
    if args.base_url:
        train_command.extend(["--base-url", args.base_url])
    if args.no_cuda_fallback:
        train_command.append("--no-cuda-fallback")
    if args.quiet:
        train_command.append("--quiet")
    run_command(train_command, logger, "training.start")

    evaluation_summaries: dict[str, dict] = {}
    for policy in ("persistent", "reset_each_transition"):
        evaluation_output = output / f"evaluation_{policy}"
        checkpoint = training_output / policy / "best_checkpoint.pt"
        command = [
            sys.executable,
            "experiments/evaluate_context_dependence.py",
            "--checkpoint",
            str(checkpoint),
            "--fixture",
            args.fixture,
            "--output",
            str(evaluation_output),
            "--encoder",
            args.encoder,
            "--embedding-model",
            args.embedding_model,
            "--device",
            args.device,
        ]
        if args.base_url:
            command.extend(["--base-url", args.base_url])
        if args.no_cuda_fallback:
            command.append("--no-cuda-fallback")
        if args.quiet:
            command.append("--quiet")
        run_command(command, logger, f"evaluation.{policy}.start")
        summary = json.loads((evaluation_output / "summary.json").read_text(encoding="utf-8"))
        evaluation_summaries[policy] = summary
        logger.log(f"evaluation.{policy}.done", "정책별 context 평가를 마쳤다.", summary=summary)

    persistent = evaluation_summaries["persistent"]
    reset = evaluation_summaries["reset_each_transition"]
    comparison = {
        "seed": args.seed,
        "persistent": persistent,
        "reset_each_transition": reset,
        "persistent_minus_reset_trained_context_margin": (
            persistent["trained_context_margin_mean"] - reset["trained_context_margin_mean"]
        ),
        "persistent_minus_reset_trained_prediction_distance": (
            persistent["trained_prediction_distance_mean"] - reset["trained_prediction_distance_mean"]
        ),
        "persistent_minus_reset_trained_latent_distance": (
            persistent["trained_latent_distance_mean"] - reset["trained_latent_distance_mean"]
        ),
        "note": (
            "Positive persistent-minus-reset context margin suggests that preserved state helps distinguish "
            "identical current text under different prior contexts. Repeat across seeds before making a claim."
        ),
    }
    (output / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.log("comparison.done", "Context dependence baseline 비교를 마쳤다.", comparison=comparison)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
