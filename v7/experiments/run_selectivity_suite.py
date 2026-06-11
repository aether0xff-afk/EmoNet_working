"""Run fixture-based multi-seed text-event plumbing experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emonet_v7.adaptive_rsnn import AdaptiveSparseRSNN  # noqa: E402
from emonet_v7.event_encoder import EventEncoder  # noqa: E402
from emonet_v7.schemas import Event  # noqa: E402
from emonet_v7.selectivity import cosine_distance, encode_event_trace  # noqa: E402
from emonet_v7.text_encoder import (  # noqa: E402
    DeterministicHashTextEncoder,
    SentenceTransformerTextEncoder,
)
from emonet_v7.trace_encoder import TraceEncoder  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", default="fixtures/selectivity_sentences.yaml")
    parser.add_argument("--output", default="runs/selectivity_suite")
    parser.add_argument("--encoder", choices=["hash", "sentence-transformer"], default="hash")
    parser.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 13, 21, 42, 100])
    return parser.parse_args()


def load_fixture(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or "sentences" not in data or "pairs" not in data:
        raise ValueError("fixture must contain sentences and pairs")
    return data


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    fixture = load_fixture(args.fixture)
    if args.encoder == "hash":
        text_encoder = DeterministicHashTextEncoder(output_dim=384)
    else:
        text_encoder = SentenceTransformerTextEncoder(args.model, args.device)

    sentences = {item["id"]: item["text"] for item in fixture["sentences"]}
    rows: list[dict] = []
    for seed in args.seeds:
        torch.manual_seed(seed)
        event_encoder = EventEncoder(text_embedding_dim=text_encoder.output_dim, num_neurons=128).to(device)
        snn = AdaptiveSparseRSNN(
            num_neurons=128,
            recurrent_density=0.10,
            seed=seed,
            recurrent_weight_std=0.70,
            input_weight_std=0.10,
        ).to(device)
        trace_encoder = TraceEncoder(num_neurons=128).to(device)
        results = {}
        for sentence_id, text in sentences.items():
            results[sentence_id] = encode_event_trace(
                event=Event(sentence_id, "user_message", text, "human"),
                text_encoder=text_encoder,
                event_encoder=event_encoder,
                snn=snn,
                trace_encoder=trace_encoder,
                event_ticks=32,
                stimulation_ticks=8,
                device=args.device,
            )
        for pair in fixture["pairs"]:
            left = results[pair["left"]]
            right = results[pair["right"]]
            rows.append(
                {
                    "seed": seed,
                    "relation": pair["relation"],
                    "left": pair["left"],
                    "right": pair["right"],
                    "text_embedding_cosine_distance": cosine_distance(left.text_embedding, right.text_embedding),
                    "input_current_cosine_distance": cosine_distance(left.current, right.current),
                    "trace_latent_cosine_distance": cosine_distance(left.latent_z, right.latent_z),
                    "left_spike_count": left.spike_count,
                    "right_spike_count": right.spike_count,
                }
            )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "pairwise_by_seed.csv", index=False, encoding="utf-8-sig")
    metric_columns = [
        "text_embedding_cosine_distance",
        "input_current_cosine_distance",
        "trace_latent_cosine_distance",
        "left_spike_count",
        "right_spike_count",
    ]
    summary = frame.groupby("relation")[metric_columns].agg(["mean", "std", "min", "max"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.reset_index().to_csv(output / "summary_by_relation.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "encoder": args.encoder,
        "model": args.model if args.encoder == "sentence-transformer" else None,
        "seeds": args.seeds,
        "fixture": args.fixture,
        "row_count": len(frame),
        "note": "Plumbing experiment only. Random EventEncoder and TraceEncoder weights are reinitialized per seed.",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(summary.to_string())


if __name__ == "__main__":
    main()
