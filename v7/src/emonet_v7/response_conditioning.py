"""Response-conditioning protocol for neutral trace reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from typing import Any, Protocol


class ChatClient(Protocol):
    def chat(self, messages: list[dict[str, str]], *, temperature: float = 0.7) -> str:
        """Return a local model response."""


RESPONSE_CONDITIONS = [
    "direct_response",
    "neutral_report",
    "masked_report",
    "shuffled_report",
]

CLAIM_BOUNDARY = (
    "Internal State is not yet an Emotion-Related State; do not claim subjective "
    "emotion, human-like feeling, or ground-truth emotion labels."
)


@dataclass(frozen=True)
class ResponseConditioningRow:
    case_id: str
    condition: str
    user_text: str
    report_payload: dict[str, Any] | None
    response: str


def mask_report(report: dict[str, Any]) -> dict[str, Any]:
    """Preserve report structure while removing values."""

    return {key: _mask_value(value) for key, value in report.items()}


def build_response_messages(
    *,
    user_text: str,
    condition: str,
    neutral_report: dict[str, Any],
    shuffled_report: dict[str, Any],
) -> tuple[list[dict[str, str]], dict[str, Any] | None]:
    """Build one response prompt for a report-conditioning condition."""

    if condition not in RESPONSE_CONDITIONS:
        raise ValueError(f"unknown response condition: {condition}")

    system = (
        "너는 사용자에게 답하는 assistant다. 원문에 충실하게 답하라. "
        "제공되는 trace report는 중립적인 내부 동역학 요약이며 감정 정답 라벨이 아니다. "
        f"{CLAIM_BOUNDARY}"
    )
    report_payload: dict[str, Any] | None = None
    report_block = ""
    if condition == "neutral_report":
        report_payload = neutral_report
        report_block = _format_report_block("neutral_trace_report", neutral_report)
    elif condition == "masked_report":
        report_payload = mask_report(neutral_report)
        report_block = _format_report_block("masked_trace_report", report_payload)
    elif condition == "shuffled_report":
        report_payload = shuffled_report
        report_block = _format_report_block("shuffled_trace_report", shuffled_report)

    user = f"<user_event>\n{user_text}\n</user_event>"
    if report_block:
        user = f"{user}\n\n{report_block}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ], report_payload


def run_response_conditioning_case(
    *,
    client: ChatClient,
    case_id: str,
    user_text: str,
    neutral_report: dict[str, Any],
    shuffled_report: dict[str, Any],
    temperature: float = 0.0,
) -> list[ResponseConditioningRow]:
    rows: list[ResponseConditioningRow] = []
    for condition in RESPONSE_CONDITIONS:
        messages, report_payload = build_response_messages(
            user_text=user_text,
            condition=condition,
            neutral_report=neutral_report,
            shuffled_report=shuffled_report,
        )
        response = " ".join(client.chat(messages, temperature=temperature).strip().splitlines()).strip()
        if not response:
            raise RuntimeError(f"empty response for condition {condition}")
        rows.append(
            ResponseConditioningRow(
                case_id=case_id,
                condition=condition,
                user_text=user_text,
                report_payload=report_payload,
                response=response,
            )
        )
    return rows


def write_response_conditioning_outputs(
    *,
    output_dir: str | Path,
    rows: list[ResponseConditioningRow],
    metadata: dict[str, Any],
) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    row_dicts = [asdict(row) for row in rows]
    fieldnames = ["case_id", "condition", "user_text", "report_payload", "response"]
    with (output / "runs.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in row_dicts:
            serialized = dict(row)
            serialized["report_payload"] = json.dumps(
                serialized["report_payload"],
                ensure_ascii=False,
                sort_keys=True,
            )
            writer.writerow(serialized)
    with (output / "runs.jsonl").open("w", encoding="utf-8") as handle:
        for row in row_dicts:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    full_metadata = {
        **metadata,
        "conditions": RESPONSE_CONDITIONS,
        "claim_boundary": CLAIM_BOUNDARY,
        "note": (
            "Response-conditioning artifact. Report influence is not evidence of "
            "subjective emotion or validated emotion labels."
        ),
    }
    (output / "metadata.json").write_text(
        json.dumps(full_metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _mask_value(value: Any) -> Any:
    if isinstance(value, list):
        return [_mask_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _mask_value(item) for key, item in value.items()}
    return "[MASKED]"


def _format_report_block(tag: str, report: dict[str, Any]) -> str:
    return f"<{tag}>\n{json.dumps(report, ensure_ascii=False, indent=2)}\n</{tag}>"
