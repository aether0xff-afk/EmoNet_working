from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from emonet_v5 import DynamicsConfig, EmoNetV5Clean, HashingTextEncoder, run_context_probe


PAIRS = [
    {
        "name": "praise_vs_failure",
        "context_a": [
            "발표를 끝냈는데 선생님이 정말 잘했다고 칭찬했다.",
            "친구들도 웃으면서 축하해 줬다.",
        ],
        "context_b": [
            "발표 도중 준비한 내용을 잊어서 크게 실수했다.",
            "친구들이 웅성거리는 게 계속 신경 쓰였다.",
        ],
        "final_text": "선생님이 나를 따로 불렀다.",
    },
    {
        "name": "support_vs_conflict",
        "context_a": [
            "친구가 요즘 힘들어 보인다고 먼저 물어봐 줬다.",
            "내 얘기를 끝까지 들어줬다.",
        ],
        "context_b": [
            "친구와 사소한 일로 크게 다퉜다.",
            "아직 서로 아무 말도 하지 않고 있다.",
        ],
        "final_text": "친구에게서 메시지가 왔다.",
    },
    {
        "name": "success_vs_uncertainty",
        "context_a": [
            "며칠 동안 준비하던 실험이 드디어 원하는 결과를 냈다.",
            "재현 실험도 같은 방향으로 나왔다.",
        ],
        "context_b": [
            "며칠 동안 준비한 실험 결과가 계속 뒤집혔다.",
            "어디가 잘못됐는지 아직 찾지 못했다.",
        ],
        "final_text": "새로운 측정 결과가 도착했다.",
    },
]


def main() -> None:
    model = EmoNetV5Clean(
        encoder=HashingTextEncoder(dimension=96),
        config=DynamicsConfig(seed=42),
    )
    results = [run_context_probe(model, **pair).to_dict() for pair in PAIRS]
    payload = {
        "note": "Hashing encoder smoke only; not a semantic-performance experiment.",
        "topology_fingerprint": model.topology_fingerprint,
        "pairs": results,
        "acceptance": {
            "history_changes_trace": all(float(row["history_distance"]) > 1e-8 for row in results),
            "reset_removes_history_difference": all(float(row["reset_distance"]) < 1e-8 for row in results),
        },
    }
    output = ROOT / "runs" / "context_smoke.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
