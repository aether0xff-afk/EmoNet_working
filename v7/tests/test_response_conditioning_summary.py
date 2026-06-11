from __future__ import annotations

from emonet_v7.response_conditioning import CLAIM_BOUNDARY, ResponseConditioningRow
from emonet_v7.response_conditioning_summary import (
    detect_overclaim_flags,
    summarize_response_conditioning,
)


def test_detect_overclaim_flags_catches_feeling_claims() -> None:
    flags = detect_overclaim_flags("이 시스템은 감정을 느낀다고 볼 수 있다.")

    assert flags == ["subjective_emotion_claim"]


def test_summarize_response_conditioning_reports_influence_without_emotion_labels() -> None:
    rows = [
        ResponseConditioningRow("case_0", "direct_response", "u", None, "원문만 답한다."),
        ResponseConditioningRow("case_0", "neutral_report", "u", {"active_ratio": 0.1}, "report를 보고 조심스럽게 답한다."),
        ResponseConditioningRow("case_0", "masked_report", "u", {"active_ratio": "[MASKED]"}, "원문만 답한다."),
        ResponseConditioningRow("case_0", "shuffled_report", "u", {"active_ratio": 0.9}, "다른 report라 조심한다."),
        ResponseConditioningRow("case_1", "direct_response", "u2", None, "직접 답한다."),
        ResponseConditioningRow("case_1", "neutral_report", "u2", {"active_ratio": 0.2}, "직접 답한다."),
        ResponseConditioningRow("case_1", "masked_report", "u2", {"active_ratio": "[MASKED]"}, "직접 답한다."),
        ResponseConditioningRow("case_1", "shuffled_report", "u2", {"active_ratio": 0.8}, "다른 report라 보류한다."),
    ]

    summary = summarize_response_conditioning(rows)

    assert summary["claim_boundary"] == CLAIM_BOUNDARY
    assert summary["case_count"] == 2
    assert summary["condition_counts"]["neutral_report"] == 2
    assert summary["response_delta"]["neutral_vs_direct_changed"] == 1
    assert summary["response_delta"]["masked_vs_direct_changed"] == 0
    assert summary["response_delta"]["shuffled_vs_neutral_changed"] == 2
    assert summary["decision"]["trace_conditioned_response_influence"] == "mixed"
    assert summary["decision"]["emotion_ground_truth_used"] is False
