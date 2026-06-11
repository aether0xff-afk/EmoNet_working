"""Run a deterministic two-module thought runtime smoke test."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emonet_v7.thought_module import (  # noqa: E402
    ThoughtModule,
    ThoughtModuleState,
    TwoModuleThoughtRuntime,
)


class ScriptedChatClient:
    """Small local client for repeatable runtime smoke tests."""

    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)

    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.7) -> str:
        if not self.responses:
            raise RuntimeError("scripted chat client has no response left")
        return json.dumps(self.responses.pop(0), ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--user-text", default="친구가 답장을 하지 않았다.")
    parser.add_argument("--output", default="runs/two_module_thought_runtime_smoke")
    parser.add_argument("--max-rounds", type=int, default=2)
    return parser.parse_args()


def state_report_provider(state: ThoughtModuleState, round_index: int) -> dict:
    return {
        "module_id": state.module_id,
        "round_index": round_index,
        "active_ratio": 0.08 if state.module_id == "module_planner" else 0.11,
        "trace_persistence": 0.31 if state.module_id == "module_planner" else 0.42,
        "peak_spike_count": 3 if state.module_id == "module_planner" else 4,
        "final_spike_count": 0,
        "latent_signature": [round_index, state.participation_budget_remaining],
    }


def build_runtime() -> TwoModuleThoughtRuntime:
    return TwoModuleThoughtRuntime(
        modules={
            "module_planner": ThoughtModule(
                ScriptedChatClient(
                    [
                        {
                            "internal_thought": "가능한 설명을 나누어 보고 바로 단정하지 않는 답이 필요하다.",
                            "module_message": "상황적 가능성을 먼저 제시하자.",
                            "candidate_output": "바빠서 못 봤을 수도 있으니 조금 기다려 보자.",
                            "termination_vote": "answer_ready",
                        }
                    ]
                ),
                module_id="module_planner",
            ),
            "module_skeptic": ThoughtModule(
                ScriptedChatClient(
                    [
                        {
                            "internal_thought": "정보가 부족하므로 관계 단절로 결론내리면 안 된다.",
                            "module_message": "확신 표현은 낮추고 확인 질문을 넣자.",
                            "candidate_output": "단정하지 말고, 필요하면 짧게 확인해 보는 편이 낫다.",
                            "termination_vote": "answer_ready",
                        }
                    ]
                ),
                module_id="module_skeptic",
            ),
        },
        module_states={
            "module_planner": ThoughtModuleState(
                module_id="module_planner",
                role_hint="checks sequence, constraints, and next action",
                participation_budget_remaining=2,
            ),
            "module_skeptic": ThoughtModuleState(
                module_id="module_skeptic",
                role_hint="checks uncertainty, missing evidence, and risk",
                participation_budget_remaining=2,
            ),
        },
        state_report_provider=state_report_provider,
        temperature=0.0,
    )


def main() -> None:
    args = parse_args()
    runtime = build_runtime()
    result = runtime.run(user_text=args.user_text, max_rounds=args.max_rounds)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    result_payload = asdict(result)
    (output / "result.json").write_text(
        json.dumps(result_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output / "messages.jsonl").open("w", encoding="utf-8") as handle:
        for message in result.messages:
            handle.write(json.dumps(asdict(message), ensure_ascii=False) + "\n")
    print(json.dumps(result_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
