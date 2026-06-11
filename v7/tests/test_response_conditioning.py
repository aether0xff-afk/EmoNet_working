from __future__ import annotations

import json

from emonet_v7.response_conditioning import (
    CLAIM_BOUNDARY,
    RESPONSE_CONDITIONS,
    mask_report,
    run_response_conditioning_case,
    write_response_conditioning_outputs,
)


class InspectingChatClient:
    def __init__(self) -> None:
        self.messages = []

    def chat(self, messages, *, temperature: float = 0.7) -> str:
        self.messages.append(messages)
        prompt = messages[-1]["content"]
        if "<neutral_trace_report>" in prompt:
            return "중립 trace report를 참고하되 감정 단정은 피한다."
        if "<masked_trace_report>" in prompt:
            return "가려진 report라서 원문 중심으로 답한다."
        if "<shuffled_trace_report>" in prompt:
            return "다른 episode report일 수 있어 조심스럽게 답한다."
        return "원문만 보고 직접 답한다."


def test_mask_report_preserves_shape_without_values() -> None:
    report = {
        "active_ratio": 0.12,
        "trace_persistence": 0.34,
        "latent_signature": [0.1, -0.2],
        "note": "neutral",
    }

    masked = mask_report(report)

    assert masked == {
        "active_ratio": "[MASKED]",
        "trace_persistence": "[MASKED]",
        "latent_signature": ["[MASKED]", "[MASKED]"],
        "note": "[MASKED]",
    }


def test_run_response_conditioning_case_generates_all_conditions() -> None:
    client = InspectingChatClient()
    report = {"active_ratio": 0.12, "trace_persistence": 0.34}
    shuffled_report = {"active_ratio": 0.91, "trace_persistence": 0.02}

    rows = run_response_conditioning_case(
        client=client,
        case_id="case_0",
        user_text="친구가 답장을 하지 않았다.",
        neutral_report=report,
        shuffled_report=shuffled_report,
        temperature=0.0,
    )

    assert [row.condition for row in rows] == RESPONSE_CONDITIONS
    assert [row.response for row in rows] == [
        "원문만 보고 직접 답한다.",
        "중립 trace report를 참고하되 감정 단정은 피한다.",
        "가려진 report라서 원문 중심으로 답한다.",
        "다른 episode report일 수 있어 조심스럽게 답한다.",
    ]
    assert CLAIM_BOUNDARY in client.messages[0][0]["content"]
    assert "<neutral_trace_report>" in client.messages[1][1]["content"]
    assert "<masked_trace_report>" in client.messages[2][1]["content"]
    assert "<shuffled_trace_report>" in client.messages[3][1]["content"]


def test_write_response_conditioning_outputs_records_boundary(tmp_path) -> None:
    client = InspectingChatClient()
    rows = run_response_conditioning_case(
        client=client,
        case_id="case_0",
        user_text="친구가 답장을 하지 않았다.",
        neutral_report={"active_ratio": 0.12},
        shuffled_report={"active_ratio": 0.91},
    )

    write_response_conditioning_outputs(
        output_dir=tmp_path,
        rows=rows,
        metadata={
            "fixture": "fixture.yaml",
            "chat_model": "fake",
        },
    )

    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    jsonl_rows = [
        json.loads(line)
        for line in (tmp_path / "runs.jsonl").read_text(encoding="utf-8").splitlines()
    ]

    assert (tmp_path / "runs.csv").exists()
    assert metadata["claim_boundary"] == CLAIM_BOUNDARY
    assert metadata["conditions"] == RESPONSE_CONDITIONS
    assert len(jsonl_rows) == 4
