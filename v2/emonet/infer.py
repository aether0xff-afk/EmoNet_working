from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .config import STYLE_NAMES, default_config
from .model import EmotionArchitecture


def main() -> None:
    parser = argparse.ArgumentParser(description="Run EmoNet inference")
    parser.add_argument("--text", type=str, required=True, help="Input text")
    parser.add_argument("--latent_dim", type=int, default=64, choices=[32, 64, 128])
    parser.add_argument("--out", type=str, default="", help="Optional JSON output path")
    args = parser.parse_args()

    cfg = default_config()
    model = EmotionArchitecture(cfg)
    out = model.infer(args.text, latent_dim=args.latent_dim)
    payload = {
        "h_t": out.h_t.tolist(),
        "z_dim": int(out.z.numel()),
        "z": out.z.tolist(),
        "styles": {name: float(out.s[i].item()) for i, name in enumerate(STYLE_NAMES)},
        "prompt": out.prompt,
        "dominant_branch": None
        if out.dominant_branch is None
        else {
            "neuron_path": out.dominant_branch.neuron_path,
            "cluster_path": out.dominant_branch.cluster_path,
            "edge_weights": out.dominant_branch.edge_weights,
            "score": out.dominant_branch.score,
            "persistence": out.dominant_branch.persistence,
        },
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
    else:
        print(text)


if __name__ == "__main__":
    main()
