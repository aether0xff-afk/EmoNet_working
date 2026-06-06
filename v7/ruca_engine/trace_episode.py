from __future__ import annotations

import json
from typing import Any, Mapping


def build_episode_prompt(
    *,
    user_text: str,
    event_type: str,
    elapsed_minutes: float,
    trace_record: Mapping[str, Any],
) -> str:
    packet = _trace_packet_without_branch_fields(trace_record)
    packet_json = json.dumps(packet, ensure_ascii=False, indent=2, sort_keys=True)
    return f"""[ROLE]
Convert the neural trace packet into one compact episode description.
Use the trace as observation, not as a named branch or hand-authored emotion label.

[INPUT_EVENT]
event_type: {event_type}
elapsed_minutes: {elapsed_minutes:.1f}
user_text: {user_text}

[TRACE_PACKET]
{packet_json}

[OUTPUT]
Return only a concise Korean episode description in plain text.
Do not expose JSON, field names, hidden reasoning, or branch terminology.
"""


def build_response_prompt_from_episode(
    *,
    user_text: str,
    event_type: str,
    elapsed_minutes: float,
    episode_text: str,
) -> str:
    return f"""[ROLE]
Write the next observable user-facing reply from the episode.

[INPUT_EVENT]
event_type: {event_type}
elapsed_minutes: {elapsed_minutes:.1f}
user_text: {user_text}

[EPISODE]
{episode_text}

[OUTPUT]
Return only the final Korean reply.
Do not mention traces, episodes, JSON, hidden state, or internal modules.
"""


def _trace_packet_without_branch_fields(trace_record: Mapping[str, Any]) -> dict[str, Any]:
    packet: dict[str, Any] = {}
    for key, value in trace_record.items():
        lowered = str(key).lower()
        if "branch" in lowered or key in {"style_tags", "style_summary", "z_dim", "s_pred_dim"}:
            continue
        packet[str(key)] = value
    return packet
