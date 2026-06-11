"""Run a tiny offline next-event prediction training smoke test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emonet_v7.adaptive_rsnn import AdaptiveSparseRSNN  # noqa: E402
from emonet_v7.event_encoder import EventEncoder  # noqa: E402
from emonet_v7.schemas import Event  # noqa: E402
from emonet_v7.self_supervised import NextEventPredictor, compute_objective  # noqa: E402
from emonet_v7.text_encoder import DeterministicHashTextEncoder  # noqa: E402
from emonet_v7.trace_encoder import TraceEncoder  # noqa: E402
from emonet_v7.training_window import run_differentiable_window  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/self_supervised_smoke")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    text_encoder = DeterministicHashTextEncoder(output_dim=64)
    event_encoder = EventEncoder(text_embedding_dim=64, num_neurons=32)
    snn = AdaptiveSparseRSNN(
        num_neurons=32,
        recurrent_density=0.15,
        seed=args.seed,
        recurrent_weight_std=0.30,
        input_weight_std=0.15,
    )
    trace_encoder = TraceEncoder(num_neurons=32, hidden_dim=32, output_dim=16)
    predictor = NextEventPredictor(latent_dim=16, hidden_dim=32, embedding_dim=64)
    parameters = list(event_encoder.parameters()) + list(snn.parameters()) + list(trace_encoder.parameters()) + list(predictor.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=3e-3, weight_decay=1e-4)

    pairs = [
        ("친구가 답장을 하지 않았다.", "바빠서 늦게 답한 것일 수도 있다."),
        ("비가 내릴 것 같다.", "우산을 챙기는 편이 좋겠다."),
        ("시험 범위가 넓다.", "학습 계획을 나누어 세워야겠다."),
    ]
    history: list[dict[str, float | int]] = []
    for step in range(args.steps):
        optimizer.zero_grad()
        total = torch.tensor(0.0)
        next_event = torch.tensor(0.0)
        firing_rate = torch.tensor(0.0)
        inactive_neuron = torch.tensor(0.0)
        stability = torch.tensor(0.0)
        for pair_index, (current_text, next_text) in enumerate(pairs):
            current_embedding = text_encoder.encode([current_text])
            target_embedding = text_encoder.encode([next_text])
            event = Event(f"step_{step}_pair_{pair_index}", "user_message", current_text, "human")
            current = event_encoder(current_embedding, [event])
            state = snn.initial_state(batch_size=1, device="cpu")
            _, window = run_differentiable_window(
                snn=snn,
                event_current=current,
                state=state,
                event_ticks=12,
                stimulation_ticks=4,
            )
            latent = trace_encoder(window.spike, window.membrane, window.adaptation)
            predicted = predictor(latent)
            losses = compute_objective(
                predicted_embedding=predicted,
                target_embedding=target_embedding,
                window=window,
            )
            total = total + losses.total
            next_event = next_event + losses.next_event
            firing_rate = firing_rate + losses.firing_rate
            inactive_neuron = inactive_neuron + losses.inactive_neuron
            stability = stability + losses.stability
        scale = float(len(pairs))
        total = total / scale
        total.backward()
        nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
        optimizer.step()
        history.append(
            {
                "step": step,
                "total": float(total.detach()),
                "next_event": float((next_event / scale).detach()),
                "firing_rate": float((firing_rate / scale).detach()),
                "inactive_neuron": float((inactive_neuron / scale).detach()),
                "stability": float((stability / scale).detach()),
            }
        )

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "seed": args.seed,
        "steps": args.steps,
        "initial_total": history[0]["total"],
        "final_total": history[-1]["total"],
        "history": history,
        "note": "Hash-encoder optimizer smoke test only. This does not demonstrate learned emotional meaning.",
    }
    (output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
