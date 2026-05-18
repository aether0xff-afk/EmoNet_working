from __future__ import annotations

import argparse
import json
from pathlib import Path

from .llm_client import LLMConfig
from .pipeline import run_turn


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one Ruca/Rookie MVP turn.")
    parser.add_argument("text", help="User text to process.")
    parser.add_argument("--memory", type=Path, default=None, help="Optional JSON memory file path.")
    parser.add_argument("--session", type=Path, default=None, help="Optional JSON session file path.")
    parser.add_argument("--debug", action="store_true", help="Print full debug record as JSON.")
    parser.add_argument("--prompt", action="store_true", help="Print the LLM-ready response prompt.")
    parser.add_argument("--llm", action="store_true", help="Use an LLM to compose the final Ruca response.")
    parser.add_argument("--provider", default="openai_compatible", choices=["openai_compatible", "anthropic"])
    parser.add_argument("--base-url", default="https://api.openai.com/v1")
    parser.add_argument("--model-name", default="gpt-5.4")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=320)
    parser.add_argument("--timeout-sec", type=int, default=45)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--no-fallback", action="store_true", help="Raise LLM errors instead of falling back to rule composer.")
    args = parser.parse_args()

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
    result = run_turn(
        args.text,
        memory_path=args.memory,
        session_path=args.session,
        use_llm=args.llm,
        llm_config=llm_config,
        fallback_to_rule_composer=not args.no_fallback,
    )
    print(result.assistant_text)
    if args.prompt:
        print(result.response_prompt)
    if args.debug:
        print(json.dumps(result.debug_record, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
