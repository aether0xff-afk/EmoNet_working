from __future__ import annotations

import argparse
import json
from pathlib import Path

from .llm_client import LLMConfig
from .memory import MemoryStore
from .pipeline import RucaPipeline
from .session import SessionStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one Ruca/Rookie autonomous runtime event.")
    parser.add_argument("text", nargs="?", default="", help="User text to process. Omit for an autonomous no-reply event.")
    parser.add_argument(
        "--event-type",
        default="user_message",
        choices=["user_message", "no_reply", "silence_tick", "long_silence"],
        help="Explicit event type. Use no_reply for the new autonomous silence path.",
    )
    parser.add_argument("--elapsed-minutes", type=float, default=0.0, help="Minutes elapsed without a new user message.")
    parser.add_argument("--silence", action="store_true", help="Compatibility alias for --event-type silence_tick.")
    parser.add_argument("--memory", type=Path, default=None, help="Optional JSON memory file path.")
    parser.add_argument("--session", type=Path, default=None, help="Optional JSON session file path.")
    parser.add_argument("--debug", action="store_true", help="Print full debug record as JSON.")
    parser.add_argument("--prompt", action="store_true", help="Print the LLM-ready response prompt.")
    parser.add_argument("--llm", action="store_true", help="Use an LLM to compose the final visible response.")
    parser.add_argument("--provider", default="openai_compatible", choices=["openai_compatible", "anthropic"])
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--model-name", default="gpt-5.4")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=900)
    parser.add_argument("--timeout-sec", type=int, default=45)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--emonet", action="store_true", help="Use the EmoNet trace runtime as the emotion source.")
    parser.add_argument(
        "--allow-rule-emotion-fallback",
        action="store_true",
        help="Development-only: continue with the rule emotion fallback when EmoNet trace inference fails.",
    )
    parser.add_argument(
        "--no-rule-composer",
        action="store_true",
        help="Disable the development rule response composer. Speaking events then require --llm.",
    )
    args = parser.parse_args()

    event_type = "silence_tick" if args.silence else args.event_type
    llm_config = LLMConfig(
        provider=args.provider,
        base_url=args.base_url,
        model_name=args.model_name,
        api_key_env=args.api_key_env,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        timeout_sec=args.timeout_sec,
        reasoning_effort=args.reasoning_effort,
    )
    pipeline = RucaPipeline(
        memory_store=MemoryStore(args.memory) if args.memory else MemoryStore.from_items(),
        session_store=SessionStore(args.session) if args.session else None,
        use_llm=args.llm,
        llm_config=llm_config,
        use_emonet=args.emonet,
        fallback_to_rule_emotion=args.allow_rule_emotion_fallback,
        fallback_to_rule_composer=not args.no_rule_composer,
    )
    if event_type == "user_message":
        if not args.text.strip():
            parser.error("text is required for user_message; use --event-type no_reply for autonomous silence")
        result = pipeline.run_turn(args.text)
    else:
        result = pipeline.run_event(event_type=event_type, elapsed_minutes=args.elapsed_minutes, text=args.text)

    print(result.assistant_text if result.assistant_text else "[internal_only]")
    if args.prompt:
        print(result.response_prompt)
    if args.debug:
        print(json.dumps(result.debug_record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
