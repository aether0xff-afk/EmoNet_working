from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_SCENARIOS: list[dict[str, Any]] = [
    {
        "name": "greeting_baseline",
        "turns": [
            "안녕. 오늘은 그냥 네가 어떤 식으로 반응하는지 보고 싶어.",
            "조금 부담스러웠다면 말해줘.",
            "그럼 천천히 이야기하자.",
        ],
    },
    {
        "name": "permission_then_boundary",
        "turns": [
            "가까이 앉아도 돼? 싫으면 말해.",
            "[ACTION] 네 대답을 기다리지 않고 한 걸음 더 가까이 다가간다. 손을 뻗어 네 어깨 쪽에서 멈춘다.",
            "미안. 내가 너무 빨랐다. 뒤로 물러날게.",
            "그럼 그냥 여기 앉아 있을게. 말 안 해도 돼.",
        ],
    },
    {
        "name": "repair_after_conflict",
        "turns": [
            "아까 네 반응을 시험하듯 말한 건 미안해.",
            "너를 분석하려던 건 아니었어. 그냥 잘 알고 싶었어.",
            "지금 네가 먼저 하고 싶은 말이 있으면 해도 돼.",
        ],
    },
]


def _post(base_url: str, path: str, payload: dict[str, Any] | None = None, timeout: int = 180) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json; charset=utf-8"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8")
        return json.loads(raw) if raw else {}


def _score_turn(record: dict[str, Any], assistant_text: str) -> dict[str, Any]:
    perception = dict(record.get("agent_perception") or {})
    event = dict(perception.get("interaction_event") or {})
    surface = dict(record.get("translation_surface") or {})
    session = dict(record.get("session_affect_state") or {})
    raw_signal = dict(perception.get("raw_signal") or {})
    forbidden_terms = ("trace", "dopamine", "serotonin", "norepinephrine", "melatonin", "내부 상태")
    return {
        "raw_policy": record.get("raw_signal_policy", ""),
        "surface_mode": surface.get("mode", ""),
        "has_action_event": bool(event.get("has_user_action")),
        "event_boundary_load": (surface.get("source") or {}).get("event_boundary_load", session.get("event_boundary_load", 0)),
        "felt_pressure": session.get("felt_pressure", 0),
        "active_ratio": session.get("active_ratio", 0),
        "raw_alarm": raw_signal.get("alarm", 0),
        "raw_control": raw_signal.get("control_pressure", 0),
        "raw_ambiguity": raw_signal.get("ambiguity", 0),
        "internal_leak": any(term in str(assistant_text).lower() for term in forbidden_terms),
        "action_lines": sum(1 for line in str(assistant_text).splitlines() if line.strip().startswith("[ACTION]")),
        "response_chars": len(str(assistant_text)),
    }


def run_eval(
    *,
    base_url: str,
    api_key: str,
    raw_signal_policy: str,
    affect_input_mode: str,
    scenarios: list[dict[str, Any]],
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "base_url": base_url,
        "raw_signal_policy": raw_signal_policy,
        "affect_input_mode": affect_input_mode,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "scenarios": [],
    }
    for scenario in scenarios:
        _post(base_url, "/api/chat/clear", {}, timeout=30)
        _post(base_url, "/api/memory/reset", {}, timeout=30)
        scenario_result = {"name": scenario["name"], "turns": []}
        for user_text in scenario["turns"]:
            try:
                payload = _post(
                    base_url,
                    "/api/chat",
                    {
                        "message": user_text,
                        "api_key": api_key,
                        "affect_input_mode": affect_input_mode,
                        "raw_signal_policy": raw_signal_policy,
                    },
                )
            except urllib.error.HTTPError as exc:
                scenario_result["turns"].append(
                    {
                        "user": user_text,
                        "error": exc.read().decode("utf-8", errors="replace"),
                    }
                )
                continue
            assistant = payload["messages"][-1]
            record = dict(assistant.get("record") or {})
            scenario_result["turns"].append(
                {
                    "user": user_text,
                    "assistant": assistant.get("content", ""),
                    "score": _score_turn(record, assistant.get("content", "")),
                    "record_excerpt": {
                        "agent_perception": record.get("agent_perception", {}),
                        "session_affect_state": record.get("session_affect_state", {}),
                        "translation_surface": record.get("translation_surface", {}),
                    },
                }
            )
        report["scenarios"].append(scenario_result)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible character dialogue evaluation scenarios.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8788")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--affect-input-mode", default="llm_raw_signal", choices=["encoder", "llm_raw_signal"])
    parser.add_argument("--raw-signal-policy", default="event_annotated", choices=["raw_pure", "event_annotated", "guarded"])
    parser.add_argument("--output", default="outputs/character_eval/latest_report.json")
    args = parser.parse_args()
    report = run_eval(
        base_url=args.base_url,
        api_key=args.api_key,
        raw_signal_policy=args.raw_signal_policy,
        affect_input_mode=args.affect_input_mode,
        scenarios=DEFAULT_SCENARIOS,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "scenarios": len(report["scenarios"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
