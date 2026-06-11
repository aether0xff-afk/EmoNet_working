"""Diagnose neuron-level accumulation and separate memory-threshold behavior.

This is a structural smoke test, not a semantic benchmark. It checks whether:

1. a weak one-off event leaves little persistent memory,
2. repeated weak events accumulate and consolidate more strongly,
3. a strong one-off event consolidates immediately,
4. firing and memory consolidation thresholds are managed separately.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emonet_v7.memory_threshold_rsnn import NeuronMemoryThresholdRSNN  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="runs/neuron_memory_threshold_diagnostic")
    parser.add_argument("--num-neurons", type=int, default=32)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--event-ticks", type=int, default=6)
    parser.add_argument("--stimulation-ticks", type=int, default=3)
    parser.add_argument("--weak-current", type=float, default=0.25)
    parser.add_argument("--strong-current", type=float, default=1.20)
    parser.add_argument("--repeat-count", type=int, default=8)
    parser.add_argument("--idle-events", type=int, default=4)
    parser.add_argument("--firing-threshold", type=float, default=1.0)
    parser.add_argument("--memory-threshold", type=float, default=0.60)
    parser.add_argument("--accumulation-decay", type=float, default=0.85)
    parser.add_argument("--memory-decay", type=float, default=0.98)
    parser.add_argument("--memory-gate-sharpness", type=float, default=20.0)
    return parser.parse_args()


def tensor_stats(tensor: torch.Tensor) -> dict[str, float]:
    detached = tensor.detach().cpu()
    return {
        "mean_abs": float(detached.abs().mean()),
        "max_abs": float(detached.abs().max()),
        "mean": float(detached.mean()),
    }


def run_case(
    *,
    model: NeuronMemoryThresholdRSNN,
    event_currents: list[torch.Tensor],
    event_ticks: int,
    stimulation_ticks: int,
) -> dict[str, Any]:
    state = model.initial_state(batch_size=1, device="cpu")
    events: list[dict[str, Any]] = []
    for index, current in enumerate(event_currents):
        state, traces, consolidation = model.run_window(
            event_current=current,
            state=state,
            event_ticks=event_ticks,
            stimulation_ticks=stimulation_ticks,
        )
        firing_rate = float(torch.stack([trace.spike for trace in traces], dim=1).mean())
        events.append(
            {
                "event_index": index,
                "input_mean_abs": float(current.abs().mean()),
                "firing_rate": firing_rate,
                "accumulation_before_reset": tensor_stats(consolidation.accumulation_before_reset),
                "accumulation_after_reset": tensor_stats(consolidation.accumulation_after_reset),
                "memory_gate": tensor_stats(consolidation.memory_gate),
                "memory_strength": tensor_stats(state.memory_strength),
            }
        )
    return {
        "events": events,
        "final": {
            "accumulation": tensor_stats(state.accumulation),
            "memory_strength": tensor_stats(state.memory_strength),
            "membrane": tensor_stats(state.membrane),
            "adaptation": tensor_stats(state.adaptation),
        },
    }


def main() -> None:
    args = parse_args()
    if args.num_neurons <= 0:
        raise ValueError("--num-neurons must be positive")
    if args.repeat_count <= 1:
        raise ValueError("--repeat-count must be greater than one")
    if args.idle_events < 0:
        raise ValueError("--idle-events must be non-negative")
    if not 0 <= args.stimulation_ticks <= args.event_ticks:
        raise ValueError("--stimulation-ticks must remain between zero and --event-ticks")

    model = NeuronMemoryThresholdRSNN(
        num_neurons=args.num_neurons,
        recurrent_density=0.10,
        seed=args.seed,
        threshold_base=args.firing_threshold,
        accumulation_decay=args.accumulation_decay,
        memory_threshold=args.memory_threshold,
        memory_gate_sharpness=args.memory_gate_sharpness,
        memory_decay=args.memory_decay,
        memory_feedback_strength=0.0,
    )
    weak = torch.full((1, args.num_neurons), args.weak_current)
    strong = torch.full((1, args.num_neurons), args.strong_current)
    idle = torch.zeros((1, args.num_neurons))

    weak_single = run_case(
        model=model,
        event_currents=[weak] + [idle] * args.idle_events,
        event_ticks=args.event_ticks,
        stimulation_ticks=args.stimulation_ticks,
    )
    weak_repeated = run_case(
        model=model,
        event_currents=[weak] * args.repeat_count,
        event_ticks=args.event_ticks,
        stimulation_ticks=args.stimulation_ticks,
    )
    strong_single = run_case(
        model=model,
        event_currents=[strong],
        event_ticks=args.event_ticks,
        stimulation_ticks=args.stimulation_ticks,
    )

    weak_single_first = weak_single["events"][0]
    weak_single_final = weak_single["final"]["memory_strength"]["mean_abs"]
    weak_repeat_final = weak_repeated["final"]["memory_strength"]["mean_abs"]
    weak_repeat_gate_peak = max(event["memory_gate"]["mean_abs"] for event in weak_repeated["events"])
    strong_first = strong_single["events"][0]
    strong_memory = strong_single["final"]["memory_strength"]["mean_abs"]

    checks = {
        "firing_and_memory_thresholds_are_separate": abs(args.firing_threshold - args.memory_threshold) > 1e-9,
        "weak_single_stays_mostly_unconsolidated": weak_single_first["memory_gate"]["mean_abs"] < 0.10,
        "weak_repetition_increases_consolidation_gate": weak_repeat_gate_peak > weak_single_first["memory_gate"]["mean_abs"] + 0.10,
        "weak_repetition_builds_more_memory_than_weak_single": weak_repeat_final > weak_single_final + 0.05,
        "strong_single_consolidates_immediately": strong_first["memory_gate"]["mean_abs"] > 0.50,
        "strong_single_builds_more_memory_than_weak_single": strong_memory > weak_single_final + 0.05,
    }
    report = {
        "config": vars(args),
        "cases": {
            "weak_single_then_idle": weak_single,
            "weak_repeated": weak_repeated,
            "strong_single": strong_single,
        },
        "summary": {
            "weak_single_first_gate_mean_abs": weak_single_first["memory_gate"]["mean_abs"],
            "weak_single_final_memory_mean_abs": weak_single_final,
            "weak_repeated_peak_gate_mean_abs": weak_repeat_gate_peak,
            "weak_repeated_final_memory_mean_abs": weak_repeat_final,
            "strong_single_first_gate_mean_abs": strong_first["memory_gate"]["mean_abs"],
            "strong_single_final_memory_mean_abs": strong_memory,
        },
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "interpretation_boundary": (
            "This smoke test validates selective neuron-local accumulation and consolidation mechanics only. "
            "It does not establish semantic memory, emotional meaning, or biological fidelity."
        ),
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "diagnostic_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
