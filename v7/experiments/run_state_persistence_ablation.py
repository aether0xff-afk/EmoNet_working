"""Compare persistent and per-transition-reset semantic dynamics training."""

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
    parser.add_argument("--fixture", default="fixtures/semantic_training_episodes.yaml")
    parser.add_argument("--output", default="runs/state_persistence_ablation")
    parser.add_argument("--encoder", choices=["hash", "lmstudio"], default="hash")
    parser.add_argument("--base-url")
    parser.add_argument("--embedding-model", default="text-embedding-nomic-embed-text-v1.5")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-neurons", type=int, default=128)
    parser.add_argument("--event-ticks", type=int, default=16)
    parser.add_argument("--stimulation-ticks", type=int, default=6)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def run_policy(args: argparse.Namespace, logger: RunLogger, policy: str) -> dict:
    output = Path(args.output) / policy
    command = [
        sys.executable,
        "experiments/train_semantic_dynamics.py",
        "--fixture",
        args.fixture,
        "--output",
        str(output),
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
        "--state-policy",
        policy,
        "--device",
        args.device,
        "--seed",
        str(args.seed),
    ]
    if args.base_url:
        command.extend(["--base-url", args.base_url])
    if args.quiet:
        command.append("--quiet")
    logger.log("policy.start", "상태 정책 학습을 시작한다.", policy=policy, command=command)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    logger.log("policy.done", "상태 정책 학습을 마쳤다.", policy=policy, summary=summary)
    return summary


def main() -> None:
    args = parse_args()
    if args.encoder == "lmstudio" and not args.base_url:
        raise ValueError("--base-url is required when --encoder lmstudio is used")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(output_dir=output, verbose=not args.quiet)
    logger.section("state persistence ablation")
    logger.log("config", "Baseline 비교 설정을 불러왔다.", **vars(args))

    persistent = run_policy(args, logger, "persistent")
    reset = run_policy(args, logger, "reset_each_transition")
    comparison = {
        "persistent": persistent,
        "reset_each_transition": reset,
        "best_validation_delta_reset_minus_persistent": (
            reset["best_validation_total"] - persistent["best_validation_total"]
        ),
        "persistent_is_better": (
            persistent["best_validation_total"] < reset["best_validation_total"]
        ),
        "note": (
            "Positive delta favors persistent state. Starter curriculum only; "
            "repeat across seeds before making a structural claim."
        ),
    }
    (output / "comparison.json").write_text(
        json.dumps(comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.log("comparison.done", "상태 유지 baseline 비교를 마쳤다.", comparison=comparison)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
