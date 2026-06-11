"""Run Milestone 3 offline internal-thought feedback ablations."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys

import pandas as pd
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emonet_v7.adaptive_rsnn import AdaptiveSparseRSNN, SNNState  # noqa: E402
from emonet_v7.event_encoder import EventEncoder  # noqa: E402
from emonet_v7.schemas import Event  # noqa: E402
from emonet_v7.selectivity import cosine_distance  # noqa: E402
from emonet_v7.text_encoder import (  # noqa: E402
    DeterministicHashTextEncoder,
    SentenceTransformerTextEncoder,
)
from emonet_v7.trace_encoder import TraceEncoder, traces_to_sequences  # noqa: E402


ABLATIONS = {
    "text_only": dict(include_event_kind=False, include_speaker=False, include_elapsed_time=False),
    "text_plus_kind": dict(include_event_kind=True, include_speaker=False, include_elapsed_time=False),
    "text_plus_kind_plus_speaker": dict(include_event_kind=True, include_speaker=True, include_elapsed_time=False),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", default="fixtures/internal_thoughts.yaml")
    parser.add_argument("--output", default="runs/internal_thought_ablation")
    parser.add_argument("--encoder", choices=["hash", "sentence-transformer"], default="hash")
    parser.add_argument("--model", default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 13, 21, 42, 100])
    return parser.parse_args()


def load_fixture(path: str) -> dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or "external_event" not in data or "thoughts" not in data:
        raise ValueError("fixture must contain external_event and thoughts")
    return data


def clone_state(state: SNNState) -> SNNState:
    return replace(
        state,
        membrane=state.membrane.clone(),
        spike=state.spike.clone(),
        adaptation=state.adaptation.clone(),
        threshold=state.threshold.clone(),
    )


def run_event(
    *,
    event: Event,
    text_encoder,
    event_encoder: EventEncoder,
    snn: AdaptiveSparseRSNN,
    trace_encoder: TraceEncoder,
    state: SNNState,
    device: str,
):
    embedding = text_encoder.encode([event.text]).to(device)
    current = event_encoder(embedding, [event])
    state, traces = snn.run_window(
        event_current=current,
        state=state,
        event_ticks=32,
        stimulation_ticks=8,
    )
    sequences = traces_to_sequences(traces)
    latent_z = trace_encoder(*(sequence.to(device) for sequence in sequences))
    return state, embedding.detach().cpu(), current.detach().cpu(), latent_z.detach().cpu(), sum(int(trace.spike.sum()) for trace in traces)


def main() -> None:
    args = parse_args()
    fixture = load_fixture(args.fixture)
    device = torch.device(args.device)
    if args.encoder == "hash":
        text_encoder = DeterministicHashTextEncoder(output_dim=384)
    else:
        text_encoder = SentenceTransformerTextEncoder(args.model, args.device)

    external = fixture["external_event"]
    pairwise_rows: list[dict] = []
    condition_rows: list[dict] = []
    for seed in args.seeds:
        for ablation_name, ablation_kwargs in ABLATIONS.items():
            torch.manual_seed(seed)
            event_encoder = EventEncoder(
                text_embedding_dim=text_encoder.output_dim,
                num_neurons=128,
                **ablation_kwargs,
            ).to(device)
            snn = AdaptiveSparseRSNN(
                num_neurons=128,
                recurrent_density=0.10,
                seed=seed,
                recurrent_weight_std=0.70,
                input_weight_std=0.10,
            ).to(device)
            trace_encoder = TraceEncoder(num_neurons=128).to(device)
            initial_state = snn.initial_state(batch_size=1, device=device)
            base_state, _, _, base_z, base_spikes = run_event(
                event=Event(external["id"], "user_message", external["text"], "human"),
                text_encoder=text_encoder,
                event_encoder=event_encoder,
                snn=snn,
                trace_encoder=trace_encoder,
                state=initial_state,
                device=args.device,
            )
            thought_results = {}
            for thought in fixture["thoughts"]:
                state, embedding, current, latent_z, spikes = run_event(
                    event=Event(thought["id"], "internal_thought", thought["text"], "module_0"),
                    text_encoder=text_encoder,
                    event_encoder=event_encoder,
                    snn=snn,
                    trace_encoder=trace_encoder,
                    state=clone_state(base_state),
                    device=args.device,
                )
                thought_results[thought["id"]] = (embedding, current, latent_z, spikes)
                condition_rows.append(
                    {
                        "seed": seed,
                        "ablation": ablation_name,
                        "condition": thought["id"],
                        "base_to_thought_trace_distance": cosine_distance(base_z, latent_z),
                        "base_spike_count": base_spikes,
                        "thought_spike_count": spikes,
                    }
                )
            for pair in fixture.get("pairs", []):
                left = thought_results[pair["left"]]
                right = thought_results[pair["right"]]
                pairwise_rows.append(
                    {
                        "seed": seed,
                        "ablation": ablation_name,
                        "relation": pair["relation"],
                        "left": pair["left"],
                        "right": pair["right"],
                        "text_embedding_cosine_distance": cosine_distance(left[0], right[0]),
                        "input_current_cosine_distance": cosine_distance(left[1], right[1]),
                        "trace_latent_cosine_distance": cosine_distance(left[2], right[2]),
                        "left_spike_count": left[3],
                        "right_spike_count": right[3],
                    }
                )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    pairwise = pd.DataFrame(pairwise_rows)
    conditions = pd.DataFrame(condition_rows)
    pairwise.to_csv(output / "pairwise_by_seed.csv", index=False, encoding="utf-8-sig")
    conditions.to_csv(output / "conditions_by_seed.csv", index=False, encoding="utf-8-sig")
    pairwise_summary = pairwise.groupby(["ablation", "relation"])[[
        "text_embedding_cosine_distance",
        "input_current_cosine_distance",
        "trace_latent_cosine_distance",
        "left_spike_count",
        "right_spike_count",
    ]].agg(["mean", "std", "min", "max"])
    pairwise_summary.columns = ["_".join(column) for column in pairwise_summary.columns]
    pairwise_summary.reset_index().to_csv(output / "pairwise_summary.csv", index=False, encoding="utf-8-sig")
    condition_summary = conditions.groupby(["ablation", "condition"])[[
        "base_to_thought_trace_distance",
        "base_spike_count",
        "thought_spike_count",
    ]].agg(["mean", "std", "min", "max"])
    condition_summary.columns = ["_".join(column) for column in condition_summary.columns]
    condition_summary.reset_index().to_csv(output / "condition_summary.csv", index=False, encoding="utf-8-sig")
    metadata = {
        "encoder": args.encoder,
        "model": args.model if args.encoder == "sentence-transformer" else None,
        "seeds": args.seeds,
        "fixture": args.fixture,
        "note": "Offline injected-thought plumbing ablation. Thoughts are fixture text, not LLM-generated text. Modules remain untrained.",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pairwise_summary.to_string())
    print(condition_summary.to_string())


if __name__ == "__main__":
    main()
