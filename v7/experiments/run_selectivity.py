"""Run the Milestone 2 sentence-to-trace selectivity experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import torch

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
    parser.add_argument("--output", default="runs/selectivity_seed42")
    parser.add_argument(
        "--encoder",
        choices=["hash", "sentence-transformer"],
        default="hash",
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    labeled_texts = [
        ("A", "친구가 답장을 하지 않았다."),
        ("A_repeat", "친구가 답장을 하지 않았다."),
        ("B", "친구가 바빠서 답장을 늦게 했다."),
        ("C", "친구가 일부러 나를 무시했다."),
    ]
    if args.encoder == "hash":
        text_encoder = DeterministicHashTextEncoder(output_dim=384)
    else:
        text_encoder = SentenceTransformerTextEncoder(args.model, args.device)

    num_neurons = 128
    event_encoder = EventEncoder(
        text_embedding_dim=text_encoder.output_dim,
        num_neurons=num_neurons,
    )
    snn = AdaptiveSparseRSNN(
        num_neurons=num_neurons,
        recurrent_density=0.10,
        seed=args.seed,
        recurrent_weight_std=0.70,
        input_weight_std=0.10,
    )
    trace_encoder = TraceEncoder(num_neurons=num_neurons)

    results = []
    for label, text in labeled_texts:
        event = Event(label, "user_message", text, "human")
        result = encode_event_trace(
            event=event,
            text_encoder=text_encoder,
            event_encoder=event_encoder,
            snn=snn,
            trace_encoder=trace_encoder,
            event_ticks=32,
            stimulation_ticks=8,
            device=args.device,
        )
        results.append((label, result))

    rows = []
    for index, (left_label, left) in enumerate(results):
        for right_label, right in results[index + 1 :]:
            rows.append(
                {
                    "left": left_label,
                    "right": right_label,
                    "text_embedding_cosine_distance": cosine_distance(
                        left.text_embedding, right.text_embedding
                    ),
                    "trace_latent_cosine_distance": cosine_distance(
                        left.latent_z, right.latent_z
                    ),
                    "left_spike_count": left.spike_count,
                    "right_spike_count": right.spike_count,
                }
            )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "pairwise_distances.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    metrics = {
        "encoder": args.encoder,
        "model": args.model if args.encoder == "sentence-transformer" else None,
        "seed": args.seed,
        "pairs": rows,
        "note": (
            "Hash mode is an offline wiring smoke test, not a semantic experiment."
            if args.encoder == "hash"
            else "Sentence-transformer mode is the semantic selectivity experiment."
        ),
    }
    (output / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
