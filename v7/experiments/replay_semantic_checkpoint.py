"""Replay a semantic-dynamics checkpoint and compare it with the initial model."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys

import pandas as pd
import torch
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emonet_v7.adaptive_rsnn import AdaptiveSparseRSNN  # noqa: E402
from emonet_v7.device import resolve_device  # noqa: E402
from emonet_v7.embedding_cache import CachedTextEncoder  # noqa: E402
from emonet_v7.episode_dataset import Episode, iter_transitions, load_episodes, select_split  # noqa: E402
from emonet_v7.event_encoder import EventEncoder  # noqa: E402
from emonet_v7.lmstudio_client import LMStudioClient  # noqa: E402
from emonet_v7.run_logger import RunLogger  # noqa: E402
from emonet_v7.state_bridge import build_neutral_state_report  # noqa: E402
from emonet_v7.text_encoder import DeterministicHashTextEncoder, LMStudioEmbeddingTextEncoder  # noqa: E402
from emonet_v7.trace_encoder import TraceEncoder, traces_to_sequences  # noqa: E402
from emonet_v7.self_supervised import NextEventPredictor  # noqa: E402


@dataclass
class ModelBundle:
    event_encoder: EventEncoder
    snn: AdaptiveSparseRSNN
    trace_encoder: TraceEncoder
    predictor: NextEventPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fixture", default="fixtures/semantic_training_episodes.yaml")
    parser.add_argument("--split", choices=["train", "validation"], default="validation")
    parser.add_argument("--output", default="runs/semantic_checkpoint_replay")
    parser.add_argument("--encoder", choices=["hash", "lmstudio"], default="hash")
    parser.add_argument("--base-url")
    parser.add_argument("--embedding-model", default="text-embedding-nomic-embed-text-v1.5")
    parser.add_argument("--device", default="cpu", help="Torch device: cpu, cuda, cuda:0, or auto")
    parser.add_argument("--no-cuda-fallback", action="store_true", help="Fail instead of falling back to CPU when CUDA is unavailable")
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


def build_bundle(*, text_dim: int, num_neurons: int, seed: int, device: torch.device) -> ModelBundle:
    torch.manual_seed(seed)
    event_encoder = EventEncoder(text_embedding_dim=text_dim, num_neurons=num_neurons).to(device)
    snn = AdaptiveSparseRSNN(
        num_neurons=num_neurons,
        recurrent_density=0.10,
        seed=seed,
        recurrent_weight_std=0.30,
        input_weight_std=0.15,
    ).to(device)
    trace_encoder = TraceEncoder(num_neurons=num_neurons, hidden_dim=64, output_dim=64).to(device)
    predictor = NextEventPredictor(latent_dim=64, hidden_dim=128, embedding_dim=text_dim).to(device)
    return ModelBundle(event_encoder=event_encoder, snn=snn, trace_encoder=trace_encoder, predictor=predictor)


def load_trained_bundle(*, checkpoint: dict, text_dim: int, device: torch.device) -> ModelBundle:
    saved_args = checkpoint["args"]
    bundle = build_bundle(
        text_dim=text_dim,
        num_neurons=int(saved_args["num_neurons"]),
        seed=int(saved_args["seed"]),
        device=device,
    )
    bundle.event_encoder.load_state_dict(checkpoint["event_encoder"])
    bundle.snn.load_state_dict(checkpoint["snn"])
    bundle.trace_encoder.load_state_dict(checkpoint["trace_encoder"])
    bundle.predictor.load_state_dict(checkpoint["predictor"])
    return bundle


def replay_episode(
    *,
    episode: Episode,
    condition: str,
    text_encoder,
    bundle: ModelBundle,
    event_ticks: int,
    stimulation_ticks: int,
    state_policy: str,
    device: torch.device,
    logger: RunLogger,
) -> list[dict]:
    state = bundle.snn.initial_state(batch_size=1, device=device)
    rows: list[dict] = []
    for transition in iter_transitions(episode):
        if state_policy == "reset_each_transition":
            state = bundle.snn.initial_state(batch_size=1, device=device)
        current_embedding = text_encoder.encode([transition.current.text]).to(device)
        target_embedding = text_encoder.encode([transition.target.text]).to(device)
        current = bundle.event_encoder(current_embedding, [transition.current])
        state, traces = bundle.snn.run_window(
            event_current=current,
            state=state,
            event_ticks=event_ticks,
            stimulation_ticks=stimulation_ticks,
        )
        sequences = traces_to_sequences(traces)
        latent = bundle.trace_encoder(*(sequence.to(device) for sequence in sequences))
        predicted = bundle.predictor(latent)
        similarity = float(F.cosine_similarity(predicted, target_embedding, dim=-1).mean().detach().cpu())
        report = build_neutral_state_report(
            traces=traces,
            latent_z=latent,
            stimulation_ticks=stimulation_ticks,
        )
        row = {
            "condition": condition,
            "episode_id": episode.episode_id,
            "step_index": transition.step_index,
            "current_text": transition.current.text,
            "target_text": transition.target.text,
            "target_cosine_similarity": similarity,
            "active_ratio": report["active_ratio"],
            "trace_persistence": report["trace_persistence"],
            "peak_spike_count": report["peak_spike_count"],
            "final_spike_count": report["final_spike_count"],
        }
        rows.append(row)
        logger.log("replay.transition", "Checkpoint replay transition을 측정했다.", **row)
    return rows


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(output_dir=output, verbose=not args.quiet)
    logger.section("semantic checkpoint replay")
    logger.log("config", "Replay 설정을 불러왔다.", **vars(args))

    device, used_device_fallback = resolve_device(
        args.device,
        allow_cuda_fallback=not args.no_cuda_fallback,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = checkpoint["args"]
    text_encoder = build_text_encoder(args, output)
    episodes = select_split(load_episodes(args.fixture), args.split)
    if not episodes:
        raise ValueError(f"fixture does not contain split: {args.split}")
    logger.log(
        "checkpoint.loaded",
        "Checkpoint와 replay dataset을 불러왔다.",
        checkpoint_epoch=checkpoint["epoch"],
        saved_validation_total=checkpoint["validation_total"],
        episode_count=len(episodes),
        requested_device=args.device,
        resolved_device=str(device),
        used_device_fallback=used_device_fallback,
    )

    trained = load_trained_bundle(checkpoint=checkpoint, text_dim=text_encoder.output_dim, device=device)
    initial = build_bundle(
        text_dim=text_encoder.output_dim,
        num_neurons=int(saved_args["num_neurons"]),
        seed=int(saved_args["seed"]),
        device=device,
    )
    event_ticks = int(saved_args["event_ticks"])
    stimulation_ticks = int(saved_args["stimulation_ticks"])
    state_policy = str(saved_args.get("state_policy", "persistent"))

    rows: list[dict] = []
    for condition, bundle in (("initial", initial), ("trained", trained)):
        logger.section(f"condition={condition}")
        bundle.event_encoder.eval()
        bundle.snn.eval()
        bundle.trace_encoder.eval()
        bundle.predictor.eval()
        with torch.no_grad():
            for episode in episodes:
                rows.extend(
                    replay_episode(
                        episode=episode,
                        condition=condition,
                        text_encoder=text_encoder,
                        bundle=bundle,
                        event_ticks=event_ticks,
                        stimulation_ticks=stimulation_ticks,
                        state_policy=state_policy,
                        device=device,
                        logger=logger,
                    )
                )

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "by_transition.csv", index=False, encoding="utf-8-sig")
    condition_summary = frame.groupby("condition")[[
        "target_cosine_similarity",
        "active_ratio",
        "trace_persistence",
        "peak_spike_count",
        "final_spike_count",
    ]].mean()
    condition_summary.to_csv(output / "summary_by_condition.csv", encoding="utf-8-sig")
    initial_similarity = float(condition_summary.loc["initial", "target_cosine_similarity"])
    trained_similarity = float(condition_summary.loc["trained", "target_cosine_similarity"])
    summary = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "state_policy": state_policy,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "requested_device": args.device,
        "resolved_device": str(device),
        "used_device_fallback": used_device_fallback,
        "initial_target_cosine_similarity": initial_similarity,
        "trained_target_cosine_similarity": trained_similarity,
        "target_cosine_similarity_delta": trained_similarity - initial_similarity,
        "note": "Replay comparison only. Improved next-event similarity does not establish emotional semantics.",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.log(
        "replay.done",
        "Checkpoint replay 비교를 마쳤다.",
        summary=summary,
        files=["run_log.jsonl", "embedding_cache.json", "by_transition.csv", "summary_by_condition.csv", "summary.json"],
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
