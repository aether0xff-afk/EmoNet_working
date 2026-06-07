"""Run one LM Studio-generated internal-thought feedback experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emonet_v7.adaptive_rsnn import AdaptiveSparseRSNN  # noqa: E402
from emonet_v7.event_encoder import EventEncoder  # noqa: E402
from emonet_v7.lmstudio_client import LMStudioClient  # noqa: E402
from emonet_v7.schemas import Event  # noqa: E402
from emonet_v7.selectivity import cosine_distance  # noqa: E402
from emonet_v7.state_bridge import build_neutral_state_report  # noqa: E402
from emonet_v7.text_encoder import SentenceTransformerTextEncoder  # noqa: E402
from emonet_v7.thought_module import ThoughtModule  # noqa: E402
from emonet_v7.trace_encoder import TraceEncoder, traces_to_sequences  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--chat-model", required=True)
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    parser.add_argument("--user-text", default="친구가 답장을 하지 않았다.")
    parser.add_argument("--output", default="runs/lmstudio_thought_feedback")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run_event(*, event, text_encoder, event_encoder, snn, trace_encoder, state, device):
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
    return state, traces, latent_z.detach().cpu()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    text_encoder = SentenceTransformerTextEncoder(args.embedding_model, args.device)
    event_encoder = EventEncoder(text_embedding_dim=text_encoder.output_dim, num_neurons=128).to(device)
    snn = AdaptiveSparseRSNN(
        num_neurons=128,
        recurrent_density=0.10,
        seed=args.seed,
        recurrent_weight_std=0.70,
        input_weight_std=0.10,
    ).to(device)
    trace_encoder = TraceEncoder(num_neurons=128).to(device)
    client = LMStudioClient(base_url=args.base_url, model=args.chat_model)
    thought_module = ThoughtModule(client)

    initial_state = snn.initial_state(batch_size=1, device=device)
    user_event = Event("user_0", "user_message", args.user_text, "human")
    state_after_user, user_traces, user_z = run_event(
        event=user_event,
        text_encoder=text_encoder,
        event_encoder=event_encoder,
        snn=snn,
        trace_encoder=trace_encoder,
        state=initial_state,
        device=args.device,
    )
    state_report = build_neutral_state_report(
        traces=user_traces,
        latent_z=user_z,
        stimulation_ticks=8,
    )
    internal_thought = thought_module.generate_internal_thought(
        user_text=args.user_text,
        state_report=state_report,
    )
    thought_event = Event("thought_0", "internal_thought", internal_thought, "module_0")
    _, thought_traces, thought_z = run_event(
        event=thought_event,
        text_encoder=text_encoder,
        event_encoder=event_encoder,
        snn=snn,
        trace_encoder=trace_encoder,
        state=state_after_user,
        device=args.device,
    )
    thought_report = build_neutral_state_report(
        traces=thought_traces,
        latent_z=thought_z,
        stimulation_ticks=8,
    )
    result = {
        "seed": args.seed,
        "user_text": args.user_text,
        "internal_thought": internal_thought,
        "state_after_user": state_report,
        "state_after_thought": thought_report,
        "user_to_thought_trace_distance": cosine_distance(user_z, thought_z),
        "note": "Single generated-thought plumbing run. This is not evidence of emotional semantics.",
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
