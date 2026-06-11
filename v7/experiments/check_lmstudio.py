"""Check LM Studio connectivity and list exposed local models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emonet_v7.lmstudio_client import LMStudioClient  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", default="placeholder")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = LMStudioClient(base_url=args.base_url, model=args.model)
    models = client.list_models()
    result = {
        "base_url": client.base_url,
        "reachable": True,
        "models": models,
        "model_count": len(models),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
