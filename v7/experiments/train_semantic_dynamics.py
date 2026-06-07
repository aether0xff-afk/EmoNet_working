"""Train persistent episode dynamics with a next-event prediction objective."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emonet_v7.adaptive_rsnn import AdaptiveSparseRSNN  # noqa: E402
from emonet_v7.embedding_cache import CachedTextEncoder  # noqa: E402
from emonet_v7.episode_dataset import Episode, iter_transitions, load_episodes, select_split  # noqa: E402
from emonet_v7.event_encoder import EventEncoder  # noqa: E402
from emonet_v7.lmstudio_client import LMStudioClient  # noqa: E402
from emonet_v7.run_logger import RunLogger  # noqa: E402
from emonet_v7.self_supervised import NextEventPredictor, compute_objective  # noqa: E402
from emonet_v7.text_encoder import DeterministicHashTextEncoder, LMStudioEmbeddingTextEncoder  # noqa: E402
from emonet_v7.trace_encoder import TraceEncoder  # noqa: E402
from emonet_v7.training_window import run_differentiable_window  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", default="fixtures/semantic_training_episodes.yaml")
    parser.add_argument("--output", default="runs/semantic_dynamics_training")
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


def build_text_encoder(args: argparse.Namespace, output: Path):
    if args.encoder == "hash":
        encoder = DeterministicHashTextEncoder(output_dim=128)
    else:
        if not args.base_url:
            raise ValueError("--base-url is required when --encoder lmstudio is used")
        client = LMStudioClient(base_url=args.base_url, model=args.embedding_model)
        encoder = LMStudioEmbeddingTextEncoder(client, args.embedding_model)
    return CachedTextEncoder(encoder, output / "embedding_cache.json")


def run_episode(
    *,
    episode: Episode,
    text_encoder,
    event_encoder: EventEncoder,
    snn: AdaptiveSparseRSNN,
    trace_encoder: TraceEncoder,
    predictor: NextEventPredictor,
    event_ticks: int,
    stimulation_ticks: int,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Run one persistent episode and return its mean objective."""

    state = snn.initial_state(batch_size=1, device=device)
    losses = []
    next_event_values = []
    firing_values = []
    inactive_values = []
    stability_values = []
    for transition in iter_transitions(episode):
        current_embedding = text_encoder.encode([transition.current.text]).to(device)
        target_embedding = text_encoder.encode([transition.target.text]).to(device)
        current = event_encoder(current_embedding, [transition.current])
        state, window = run_differentiable_window(
            snn=snn,
            event_current=current,
            state=state,
            event_ticks=event_ticks,
            stimulation_ticks=stimulation_ticks,
        )
        latent = trace_encoder(window.spike, window.membrane, window.adaptation)
        predicted = predictor(latent)
        objective = compute_objective(
            predicted_embedding=predicted,
            target_embedding=target_embedding,
            window=window,
        )
        losses.append(objective.total)
        next_event_values.append(objective.next_event)
        firing_values.append(objective.firing_rate)
        inactive_values.append(objective.inactive_neuron)
        stability_values.append(objective.stability)
    mean_total = torch.stack(losses).mean()
    metrics = {
        "total": float(mean_total.detach()),
        "next_event": float(torch.stack(next_event_values).mean().detach()),
        "firing_rate": float(torch.stack(firing_values).mean().detach()),
        "inactive_neuron": float(torch.stack(inactive_values).mean().detach()),
        "stability": float(torch.stack(stability_values).mean().detach()),
    }
    return mean_total, metrics


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = ["total", "next_event", "firing_rate", "inactive_neuron", "stability"]
    return {key: sum(row[key] for row in rows) / max(1, len(rows)) for key in keys}


def evaluate(
    *,
    episodes: list[Episode],
    text_encoder,
    event_encoder: EventEncoder,
    snn: AdaptiveSparseRSNN,
    trace_encoder: TraceEncoder,
    predictor: NextEventPredictor,
    event_ticks: int,
    stimulation_ticks: int,
    device: torch.device,
) -> dict[str, float]:
    event_encoder.eval()
    snn.eval()
    trace_encoder.eval()
    predictor.eval()
    rows = []
    with torch.no_grad():
        for episode in episodes:
            _, metrics = run_episode(
                episode=episode,
                text_encoder=text_encoder,
                event_encoder=event_encoder,
                snn=snn,
                trace_encoder=trace_encoder,
                predictor=predictor,
                event_ticks=event_ticks,
                stimulation_ticks=stimulation_ticks,
                device=device,
            )
            rows.append(metrics)
    return aggregate(rows)


def main() -> None:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if not 0 < args.stimulation_ticks <= args.event_ticks:
        raise ValueError("stimulation ticks must be positive and not exceed event ticks")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(output_dir=output, verbose=not args.quiet)
    logger.section("semantic dynamics training")
    logger.log("config", "학습 설정을 불러왔다.", **vars(args))

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    episodes = load_episodes(args.fixture)
    train_episodes = select_split(episodes, "train")
    validation_episodes = select_split(episodes, "validation")
    if not train_episodes or not validation_episodes:
        raise ValueError("fixture must contain both train and validation episodes")
    logger.log(
        "dataset.ready",
        "Episode 데이터셋을 불러왔다.",
        train_episodes=len(train_episodes),
        validation_episodes=len(validation_episodes),
    )

    logger.log("embedding.init", "텍스트 encoder와 캐시를 초기화한다.", encoder=args.encoder)
    text_encoder = build_text_encoder(args, output)
    logger.log("embedding.ready", "Embedding encoder가 준비됐다.", output_dim=text_encoder.output_dim)

    event_encoder = EventEncoder(text_embedding_dim=text_encoder.output_dim, num_neurons=args.num_neurons).to(device)
    snn = AdaptiveSparseRSNN(
        num_neurons=args.num_neurons,
        recurrent_density=0.10,
        seed=args.seed,
        recurrent_weight_std=0.30,
        input_weight_std=0.15,
    ).to(device)
    trace_encoder = TraceEncoder(num_neurons=args.num_neurons, hidden_dim=64, output_dim=64).to(device)
    predictor = NextEventPredictor(latent_dim=64, hidden_dim=128, embedding_dim=text_encoder.output_dim).to(device)
    parameters = list(event_encoder.parameters()) + list(snn.parameters()) + list(trace_encoder.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=1e-4)
    logger.log("model.ready", "학습 가능한 모듈을 초기화했다.", parameter_count=sum(parameter.numel() for parameter in parameters))

    history: list[dict[str, float | int]] = []
    best_validation = float("inf")
    best_epoch = -1
    for epoch in range(args.epochs):
        event_encoder.train()
        snn.train()
        trace_encoder.train()
        predictor.train()
        epoch_rows = []
        for episode in train_episodes:
            optimizer.zero_grad()
            total, metrics = run_episode(
                episode=episode,
                text_encoder=text_encoder,
                event_encoder=event_encoder,
                snn=snn,
                trace_encoder=trace_encoder,
                predictor=predictor,
                event_ticks=args.event_ticks,
                stimulation_ticks=args.stimulation_ticks,
                device=device,
            )
            total.backward()
            nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            optimizer.step()
            epoch_rows.append(metrics)
        train_metrics = aggregate(epoch_rows)
        validation_metrics = evaluate(
            episodes=validation_episodes,
            text_encoder=text_encoder,
            event_encoder=event_encoder,
            snn=snn,
            trace_encoder=trace_encoder,
            predictor=predictor,
            event_ticks=args.event_ticks,
            stimulation_ticks=args.stimulation_ticks,
            device=device,
        )
        row = {
            "epoch": epoch,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(row)
        logger.log("epoch.done", "Epoch 학습과 검증을 마쳤다.", **row)
        if validation_metrics["total"] < best_validation:
            best_validation = validation_metrics["total"]
            best_epoch = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "args": vars(args),
                    "event_encoder": event_encoder.state_dict(),
                    "snn": snn.state_dict(),
                    "trace_encoder": trace_encoder.state_dict(),
                    "predictor": predictor.state_dict(),
                    "validation_total": best_validation,
                },
                output / "best_checkpoint.pt",
            )
            logger.log("checkpoint.saved", "Validation 기준 최고 checkpoint를 저장했다.", epoch=epoch, validation_total=best_validation)

    frame = pd.DataFrame(history)
    frame.to_csv(output / "history.csv", index=False, encoding="utf-8-sig")
    summary = {
        "encoder": args.encoder,
        "seed": args.seed,
        "epochs": args.epochs,
        "train_episode_count": len(train_episodes),
        "validation_episode_count": len(validation_episodes),
        "initial_train_total": float(frame["train_total"].iloc[0]),
        "final_train_total": float(frame["train_total"].iloc[-1]),
        "initial_validation_total": float(frame["validation_total"].iloc[0]),
        "final_validation_total": float(frame["validation_total"].iloc[-1]),
        "best_validation_total": best_validation,
        "best_epoch": best_epoch,
        "note": "Starter episode curriculum. Trainability and validation behavior only; not evidence of emotional semantics.",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.log(
        "training.done",
        "Semantic dynamics 학습 실행을 마쳤다.",
        summary=summary,
        files=["run_log.jsonl", "embedding_cache.json", "history.csv", "summary.json", "best_checkpoint.pt"],
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
