"""Measure whether prior context changes predictions for identical current text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import torch
from torch.nn import functional as F
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emonet_v7.device import resolve_device  # noqa: E402
from emonet_v7.embedding_cache import CachedTextEncoder  # noqa: E402
from emonet_v7.episode_dataset import Episode, iter_transitions, load_episodes, select_split  # noqa: E402
from emonet_v7.lmstudio_client import LMStudioClient  # noqa: E402
from emonet_v7.run_logger import RunLogger  # noqa: E402
from emonet_v7.semantic_bundle import ModelBundle, build_bundle, load_trained_bundle  # noqa: E402
from emonet_v7.selectivity import cosine_distance  # noqa: E402
from emonet_v7.text_encoder import DeterministicHashTextEncoder, LMStudioEmbeddingTextEncoder  # noqa: E402
from emonet_v7.trace_encoder import traces_to_sequences  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fixture", default="fixtures/context_dependence_episodes.yaml")
    parser.add_argument("--split", choices=["train", "validation"], default="validation")
    parser.add_argument("--output", default="runs/context_dependence_evaluation")
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


def load_contrast_pairs(path: str | Path, *, split: str) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    pairs = data.get("contrast_pairs") if isinstance(data, dict) else None
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("fixture must contain contrast_pairs")
    selected = [pair for pair in pairs if str(pair.get("split", "validation")) == split]
    if not selected:
        raise ValueError(f"fixture does not contain contrast_pairs for split: {split}")
    return selected


def collect_outputs(
    *,
    episodes: list[Episode],
    text_encoder,
    bundle: ModelBundle,
    event_ticks: int,
    stimulation_ticks: int,
    state_policy: str,
    device: torch.device,
) -> dict[tuple[str, int], dict]:
    outputs: dict[tuple[str, int], dict] = {}
    bundle.eval()
    with torch.no_grad():
        for episode in episodes:
            state = bundle.snn.initial_state(batch_size=1, device=device)
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
                outputs[(episode.episode_id, transition.step_index)] = {
                    "predicted": predicted.detach().cpu(),
                    "target": target_embedding.detach().cpu(),
                    "latent": latent.detach().cpu(),
                    "current_text": transition.current.text,
                    "target_text": transition.target.text,
                }
    return outputs


def similarity(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.cosine_similarity(left, right, dim=-1).mean())


def measure_pairs(*, condition: str, pairs: list[dict], outputs: dict[tuple[str, int], dict]) -> list[dict]:
    rows: list[dict] = []
    for pair in pairs:
        step_index = int(pair["step_index"])
        left = outputs[(str(pair["left"]), step_index)]
        right = outputs[(str(pair["right"]), step_index)]
        left_correct = similarity(left["predicted"], left["target"])
        left_cross = similarity(left["predicted"], right["target"])
        right_correct = similarity(right["predicted"], right["target"])
        right_cross = similarity(right["predicted"], left["target"])
        rows.append(
            {
                "condition": condition,
                "relation": pair["relation"],
                "left_episode": pair["left"],
                "right_episode": pair["right"],
                "step_index": step_index,
                "current_text_matches": left["current_text"] == right["current_text"],
                "prediction_cosine_distance": cosine_distance(left["predicted"], right["predicted"]),
                "latent_cosine_distance": cosine_distance(left["latent"], right["latent"]),
                "left_correct_similarity": left_correct,
                "left_cross_similarity": left_cross,
                "right_correct_similarity": right_correct,
                "right_cross_similarity": right_cross,
                "context_margin": ((left_correct - left_cross) + (right_correct - right_cross)) / 2.0,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(output_dir=output, verbose=not args.quiet)
    logger.section("context dependence evaluation")
    logger.log("config", "Context dependence 평가 설정을 불러왔다.", **vars(args))

    device, used_device_fallback = resolve_device(
        args.device,
        allow_cuda_fallback=not args.no_cuda_fallback,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = checkpoint["args"]
    text_encoder = build_text_encoder(args, output)
    episodes = select_split(load_episodes(args.fixture), args.split)
    pairs = load_contrast_pairs(args.fixture, split=args.split)
    if not episodes:
        raise ValueError(f"fixture does not contain split: {args.split}")

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
    logger.log(
        "checkpoint.loaded",
        "Checkpoint와 context fixture를 불러왔다.",
        checkpoint_epoch=checkpoint["epoch"],
        state_policy=state_policy,
        episode_count=len(episodes),
        contrast_pair_count=len(pairs),
        requested_device=args.device,
        resolved_device=str(device),
        used_device_fallback=used_device_fallback,
    )

    rows: list[dict] = []
    for condition, bundle in (("initial", initial), ("trained", trained)):
        condition_outputs = collect_outputs(
            episodes=episodes,
            text_encoder=text_encoder,
            bundle=bundle,
            event_ticks=event_ticks,
            stimulation_ticks=stimulation_ticks,
            state_policy=state_policy,
            device=device,
        )
        condition_rows = measure_pairs(condition=condition, pairs=pairs, outputs=condition_outputs)
        rows.extend(condition_rows)
        for row in condition_rows:
            logger.log("context_pair.measured", "동일 현재 문장의 맥락 차이를 측정했다.", **row)

    frame = pd.DataFrame(rows)
    frame.to_csv(output / "by_pair.csv", index=False, encoding="utf-8-sig")
    metric_columns = ["prediction_cosine_distance", "latent_cosine_distance", "context_margin"]
    grouped = frame.groupby("condition")[metric_columns].mean()
    grouped.to_csv(output / "summary_by_condition.csv", encoding="utf-8-sig")
    summary = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "state_policy": state_policy,
        "requested_device": args.device,
        "resolved_device": str(device),
        "used_device_fallback": used_device_fallback,
        "initial_prediction_distance_mean": float(grouped.loc["initial", "prediction_cosine_distance"]),
        "trained_prediction_distance_mean": float(grouped.loc["trained", "prediction_cosine_distance"]),
        "initial_latent_distance_mean": float(grouped.loc["initial", "latent_cosine_distance"]),
        "trained_latent_distance_mean": float(grouped.loc["trained", "latent_cosine_distance"]),
        "initial_context_margin_mean": float(grouped.loc["initial", "context_margin"]),
        "trained_context_margin_mean": float(grouped.loc["trained", "context_margin"]),
        "note": "Positive trained context margin suggests context-sensitive prediction on this fixture. This is not evidence of emotional semantics.",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.log("context_evaluation.done", "Context dependence 평가를 마쳤다.", summary=summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
