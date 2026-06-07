"""Run the Milestone 1 zero-input decay experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emonet_v7.adaptive_rsnn import AdaptiveSparseRSNN  # noqa: E402
from emonet_v7.config import load_yaml  # noqa: E402
from emonet_v7.metrics import tick_rows  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="runs/decay_seed42")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    seed = int(config["seed"])
    torch.manual_seed(seed)
    snn_cfg = config["snn"]
    exp_cfg = config["experiment"]
    device = torch.device(args.device)

    model_kwargs = {
        key: value
        for key, value in snn_cfg.items()
        if key not in {"event_ticks", "stimulation_ticks"}
    }
    model = AdaptiveSparseRSNN(seed=seed, **model_kwargs).to(device)
    state = model.initial_state(batch_size=int(exp_cfg["batch_size"]), device=device)
    event_current = torch.zeros(int(exp_cfg["batch_size"]), int(snn_cfg["num_neurons"]), device=device)
    event_current[:, : max(1, int(snn_cfg["num_neurons"]) // 8)] = float(exp_cfg["input_scale"])

    state, traces = model.run_window(
        event_current=event_current,
        state=state,
        event_ticks=int(snn_cfg["event_ticks"]),
        stimulation_ticks=int(snn_cfg["stimulation_ticks"]),
    )
    idle_ticks = int(exp_cfg["idle_ticks"])
    idle_current = torch.zeros_like(event_current)
    for _ in range(idle_ticks):
        previous_spike = state.spike
        state = model.step(idle_current, state)
        traces.append(
            type(traces[0])(
                tick=len(traces),
                membrane=state.membrane.detach().cpu(),
                spike=state.spike.detach().cpu(),
                adaptation=state.adaptation.detach().cpu(),
                threshold=state.threshold.detach().cpu(),
                active_edges=(
                    previous_spike.unsqueeze(-1)
                    * state.spike.unsqueeze(-2)
                    * model.recurrent_mask
                ).detach().cpu(),
            )
        )

    output = Path(args.output)
    plots = output / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    rows = tick_rows(traces)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "tick_summary.csv", index=False)

    metrics = {
        "seed": seed,
        "num_ticks": len(frame),
        "peak_active_ratio": float(frame["active_ratio"].max()),
        "final_active_ratio": float(frame["active_ratio"].iloc[-1]),
        "membrane_abs_max": float(frame["membrane_abs_max"].max()),
        "contains_nan": bool(frame.isna().any().any()),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    plt.figure()
    plt.plot(frame["tick"], frame["active_ratio"])
    plt.xlabel("tick")
    plt.ylabel("active ratio")
    plt.tight_layout()
    plt.savefig(plots / "active_ratio.png", dpi=160)
    plt.close()

    plt.figure()
    plt.plot(frame["tick"], frame["membrane_abs_max"])
    plt.xlabel("tick")
    plt.ylabel("absolute membrane maximum")
    plt.tight_layout()
    plt.savefig(plots / "membrane_abs_max.png", dpi=160)
    plt.close()

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
