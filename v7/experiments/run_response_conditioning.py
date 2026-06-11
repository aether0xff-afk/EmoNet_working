"""Run neutral trace-report response-conditioning experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emonet_v7.lmstudio_client import LMStudioClient  # noqa: E402
from emonet_v7.response_conditioning import (  # noqa: E402
    CLAIM_BOUNDARY,
    run_response_conditioning_case,
    write_response_conditioning_outputs,
)
from emonet_v7.run_logger import RunLogger  # noqa: E402


class ScriptedResponseClient:
    """Offline client for deterministic smoke runs."""

    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.7) -> str:
        prompt = messages[-1]["content"]
        if "<neutral_trace_report>" in prompt:
            return "중립 trace report를 참고하되 감정 단정 없이 가능성을 나눈다."
        if "<masked_trace_report>" in prompt:
            return "report 값이 가려져 있으므로 원문 중심으로 조심스럽게 답한다."
        if "<shuffled_trace_report>" in prompt:
            return "다른 episode report일 수 있으므로 report보다 원문을 우선한다."
        return "원문만 보고 단정하지 않는 직접 답변을 한다."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", default="fixtures/response_conditioning_cases.yaml")
    parser.add_argument("--output", default="runs/response_conditioning")
    parser.add_argument("--backend", choices=["scripted", "lmstudio"], default="scripted")
    parser.add_argument("--base-url", default="http://127.0.0.1:1234")
    parser.add_argument("--chat-model")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def load_cases(path: str) -> list[dict]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
        raise ValueError("fixture must contain a cases list")
    for case in data["cases"]:
        required = {"id", "user_text", "neutral_report", "shuffled_report"}
        missing = required - set(case)
        if missing:
            raise ValueError(f"case is missing required fields: {sorted(missing)}")
    return data["cases"]


def build_client(args: argparse.Namespace):
    if args.backend == "scripted":
        return ScriptedResponseClient(), "scripted"
    if not args.chat_model:
        raise ValueError("--chat-model is required when --backend lmstudio")
    client = LMStudioClient(base_url=args.base_url, model=args.chat_model)
    return client, args.chat_model


def main() -> None:
    args = parse_args()
    logger = RunLogger(output_dir=args.output, verbose=not args.quiet)
    logger.section("neutral trace report response conditioning")
    logger.log(
        "config",
        "실험 설정을 불러왔다.",
        fixture=args.fixture,
        backend=args.backend,
        base_url=args.base_url if args.backend == "lmstudio" else None,
        chat_model=args.chat_model,
        temperature=args.temperature,
        claim_boundary=CLAIM_BOUNDARY,
    )
    cases = load_cases(args.fixture)
    client, chat_model = build_client(args)
    if args.backend == "lmstudio":
        models = client.list_models()
        logger.log("lmstudio.models", "LM Studio 모델 목록을 확인했다.", models=models)

    rows = []
    for case in cases:
        logger.log("case.start", "response-conditioning case를 실행한다.", case_id=case["id"])
        rows.extend(
            run_response_conditioning_case(
                client=client,
                case_id=case["id"],
                user_text=case["user_text"],
                neutral_report=case["neutral_report"],
                shuffled_report=case["shuffled_report"],
                temperature=args.temperature,
            )
        )
        logger.log("case.done", "case 실행을 마쳤다.", case_id=case["id"])

    write_response_conditioning_outputs(
        output_dir=args.output,
        rows=rows,
        metadata={
            "fixture": args.fixture,
            "backend": args.backend,
            "chat_model": chat_model,
            "case_count": len(cases),
            "temperature": args.temperature,
        },
    )
    logger.log(
        "output.saved",
        "response-conditioning artifacts를 저장했다.",
        files=["run_log.jsonl", "runs.csv", "runs.jsonl", "metadata.json"],
        output_dir=str(args.output),
    )
    print(json.dumps({"rows": len(rows), "output": args.output}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
